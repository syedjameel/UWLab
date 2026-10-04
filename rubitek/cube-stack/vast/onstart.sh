#!/usr/bin/env bash
# vast.ai onstart script — runs under the CONTAINER'S INIT, not an ssh session.
#
# That distinction is the whole design. This box family reaps detached processes (setsid, nohup,
# tmux alike) when the ssh connection that spawned them closes — recorded in the meaningful-latents
# LOG.md as "INFRA note: vast.ai box REAPS detached procs (setsid/nohup/tmux) on ssh-close -> long
# runs must be held". A multi-hour training run therefore cannot be launched from ssh at all. This
# script sets the machine up and then watches a job directory; ssh is only ever used to drop a job
# file in and to read logs, so no ssh session needs to stay alive.
set -uo pipefail

WS=/workspace
LOG=$WS/provision.log
mkdir -p "$WS/jobs" "$WS/done" "$WS/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== provision start $(date -Is) ==="

# FAIL FAST ON AN OLD HOST DRIVER. Isaac Sim 5.1 pulls torch built against CUDA 12.6, and a host on
# the 555 branch supports only up to 12.5: every CUDA call then dies with
#   Error 803: system has unsupported display driver / cuda driver combination
# That surfaces fifteen minutes into provisioning, at the first training step, after the box has
# already downloaded ~10 GB and been paid for. Check it in the first second instead.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRV_MAJOR=${DRV%%.*}
echo "host driver $DRV"
if [ "${DRV_MAJOR:-0}" -lt 570 ] 2>/dev/null; then
  echo "=== ABORT: driver $DRV is too old for the CUDA 12.6 torch build (need >= 570) ==="
  echo "DRIVER_TOO_OLD $DRV" > "$WS/PROVISION_FAILED"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# libglu1-mesa is the one the guide calls out by name: without it Isaac fails with
# "HydraEngine rtx failed". The rest are ordinary build/runtime deps.
apt-get install -y -qq --no-install-recommends \
  git git-lfs curl ca-certificates build-essential cmake ninja-build \
  libglu1-mesa libgl1 libxrender1 libxext6 libsm6 rsync

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

VENV=$WS/venv
uv venv --python 3.11 "$VENV"
PY="$VENV/bin/python"

echo "=== isaacsim 5.1.0 ==="
uv pip install --python "$PY" "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
echo "=== torch ==="
uv pip install --python "$PY" torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128

echo "=== IsaacLab (upstream main, editable) ==="
git clone --depth 1 --branch main https://github.com/isaac-sim/IsaacLab.git "$WS/IsaacLab"
for d in isaaclab isaaclab_assets isaaclab_tasks 'isaaclab_rl[all]'; do
  uv pip install --python "$PY" -e "$WS/IsaacLab/source/${d%%[*}" --extra-index-url https://pypi.nvidia.com
done
# gSDE lives ONLY in the UW-Lab fork. Upstream rsl-rl-lib has no noise_std_type="gsde" and the run
# dies at construction with a ValueError, which cost a full provisioning cycle the first time.
# --no-deps is required: resolving this package's deps drags torch from 2.7.0+cu128 to 2.13.0+cu130,
# which Isaac 5.1 does not load. Torch is re-pinned immediately after, and both facts are printed as
# positive proof -- an install that silently no-ops here does not surface until a training run has
# already spent its startup minutes and then dies on the ValueError.
#
# Install with uv, never `python -m pip`: this venv comes from `uv venv`, which does not put pip
# inside it, so a pip invocation is a no-op that leaves the wrong rsl_rl in place.
uv pip install --python "$PY" --no-deps --force-reinstall \
  "rsl-rl-lib @ git+https://github.com/UW-Lab/rsl_rl.git@main"
# --no-deps means the fork's own runtime dependencies are not pulled in, and rsl_rl imports both of
# these at module import time: rsl_rl.utils.utils does `import git`, rsl_rl.storage uses tensordict.
# On the first two boxes they happened to already be present from the rsl-rl-lib==3.1.2 install this
# replaces; on a box where the fork is the only rsl_rl ever installed, training dies at import with
# ModuleNotFoundError before a single environment is built.
uv pip install --python "$PY" GitPython tensordict onnx
uv pip install --python "$PY" torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$PY" -c "import torch, inspect, rsl_rl.modules.actor_critic as m; print('torch', torch.__version__, '| gsde present:', 'gsde' in inspect.getsource(m))"


echo "=== waiting for the repo to be rsynced to $WS/UWLab ==="
# The working tree is pushed from the dev box rather than cloned: the cube-stacking changes live on
# a local branch and are deliberately not published to anyone's fork.
while [ ! -f "$WS/UWLab/.repo_ready" ]; do sleep 10; done
echo "repo present"

for d in "$WS"/UWLab/source/*/; do
  [ -f "$d/setup.py" ] && uv pip install --python "$PY" -e "$d"
done

touch "$WS/READY"
echo "=== provision complete $(date -Is) ==="

# ---------------------------------------------------------------------------------------
# Job loop. Anything dropped into $WS/jobs/*.sh runs here, owned by init, and survives every
# ssh disconnect. Each job's stdout goes to $WS/logs/<name>.log and the script is moved to
# $WS/done/ so it cannot run twice.
echo "=== job loop armed ==="
while true; do
  for job in "$WS"/jobs/*.sh; do
    [ -e "$job" ] || continue
    name=$(basename "$job" .sh)
    echo "[$(date -Is)] running job: $name"
    mv "$job" "$WS/done/$name.sh"
    bash "$WS/done/$name.sh" > "$WS/logs/$name.log" 2>&1
    echo "[$(date -Is)] job $name finished, wrapper exit $? (not trusted -- grade on artefacts)"
  done
  sleep 15
done
