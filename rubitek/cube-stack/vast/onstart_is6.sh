#!/usr/bin/env bash
# vast.ai onstart for the ISAAC SIM 6 / IsaacLab 3 cube-stack env, run under the CONTAINER'S INIT.
#
# Not an ssh session: this box family reaps detached processes (setsid/nohup/tmux) when the ssh
# connection that spawned them closes, so a multi-hour job cannot be launched over ssh. This script
# provisions and then watches $WS/jobs for work; ssh is only used to drop a job file in and read logs.
#
# Differs from the older onstart.sh (Isaac Sim 5.1 + upstream IsaacLab main): this stack is Isaac Sim
# 6.0.1 + the LOCAL _isaaclab3 worktree, which is pushed rather than cloned because it carries
# local fixes, and torch is 2.11.0+cu130 (needs a 580-branch driver).
set -uo pipefail
WS=/workspace
LOG=$WS/provision.log
mkdir -p "$WS/jobs" "$WS/done" "$WS/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== provision start $(date -Is) ==="

# SSH KEY. Vast team accounts reject account-level ssh keys ("SSH keys can only be created in
# personal context"), so the key is injected here instead and reached over the --direct endpoint.
mkdir -p /root/.ssh && chmod 700 /root/.ssh
grep -qxF '__SSH_PUBKEY__' /root/.ssh/authorized_keys 2>/dev/null || echo '__SSH_PUBKEY__' >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "host driver $DRV"
if [ "${DRV%%.*}" -lt 580 ] 2>/dev/null; then
  echo "=== ABORT: driver $DRV cannot run the cu130 torch build (need >= 580) ==="
  echo "DRIVER_TOO_OLD $DRV" > "$WS/PROVISION_FAILED"; exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# libglu1-mesa is required by the RTX renderer ("HydraEngine rtx failed" without it); the bank QA
# step renders, so this box is provisioned for rendering, not only for training.
apt-get install -y -qq --no-install-recommends \
  git git-lfs curl ca-certificates build-essential cmake ninja-build \
  libglu1-mesa libgl1 libxrender1 libxext6 libsm6 rsync

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
VENV=$WS/venv
uv venv --python 3.12 "$VENV"
PY="$VENV/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES

echo "=== isaacsim 6.0.1.0 ==="
# --index-strategy unsafe-best-match is REQUIRED: isaacsim-core 6.0.1.0 depends on
# mujoco-usd-converter==0.2.0, which exists on pypi.org but not on pypi.nvidia.com. With uv's
# default first-index-wins strategy the resolve fails with "No solution found" and provisioning
# silently continues without Isaac Sim installed at all.
uv pip install --python "$PY" --index-strategy unsafe-best-match \
  "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com

echo "=== waiting for pushed trees (UWLab + _isaaclab3 + compat) ==="
while [ ! -f "$WS/UWLab/.repo_ready" ] || [ ! -f "$WS/_isaaclab3/.repo_ready" ]; do sleep 10; done
echo "trees present"

echo "=== IsaacLab 3 editable installs (local worktree) ==="
for d in isaaclab isaaclab_physx isaaclab_newton isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_ppisp; do
  [ -d "$WS/_isaaclab3/source/$d" ] && uv pip install --python "$PY" -e "$WS/_isaaclab3/source/$d" \
    --extra-index-url https://pypi.nvidia.com
done

echo "=== UWLab source packages (editable) ==="
for d in "$WS"/UWLab/source/*/; do
  [ -f "$d/setup.py" ] && uv pip install --python "$PY" -e "$d"
done

# gSDE lives ONLY in the UW-Lab fork; upstream rsl-rl-lib raises
# "Unknown standard deviation type: gsde" at runner construction. --no-deps, because resolving the
# fork's deps moves torch off the cu130 build; its import-time deps are installed explicitly below.
echo "=== rsl_rl (UW-Lab fork, gSDE) ==="
uv pip install --python "$PY" --no-deps --force-reinstall \
  "rsl-rl-lib @ git+https://github.com/UW-Lab/rsl_rl.git@main"
uv pip install --python "$PY" GitPython tensordict onnx tensorboard

echo "=== re-pin torch 2.11.0+cu130 (editable installs can pull a different build) ==="
uv pip install --python "$PY" torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
"$PY" - <<'PYEOF'
import inspect, torch
import rsl_rl.modules.actor_critic as m
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| gsde:", "gsde" in inspect.getsource(m))
PYEOF

touch "$WS/READY"
echo "=== provision complete $(date -Is) ==="

# Job loop: anything dropped into $WS/jobs/*.sh runs here, owned by init, surviving ssh disconnects.
echo "=== job loop armed ==="
while true; do
  for job in "$WS"/jobs/*.sh; do
    [ -e "$job" ] || continue
    name=$(basename "$job" .sh)
    echo "[$(date -Is)] running job: $name"
    mv "$job" "$WS/done/$name.sh"
    bash "$WS/done/$name.sh" > "$WS/logs/$name.log" 2>&1
    echo "[$(date -Is)] job $name finished, wrapper exit $? (grade on artefacts, not on this)"
  done
  sleep 15
done
