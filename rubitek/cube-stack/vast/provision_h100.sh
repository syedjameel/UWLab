#!/usr/bin/env bash
# Provision the Isaac Sim 6.0.1 / IsaacLab 3 cube-stack stack on the h100-server (a100-learner).
#
# Differences from the vast.ai onstart_is6.sh this is derived from:
#   * No apt: every runtime lib the RTX renderer needs (libGLU, libGL, libXrender, libXext, libSM,
#     libX11) is already present, and there is no passwordless sudo. Nothing is installed system-wide.
#   * WS lives under $HOME, not /workspace, so the whole footprint is removable with one rm -rf.
#   * uv is fetched as a release binary into ~/.local/bin rather than via curl|sh.
#   * Processes survive ssh here (a real host, not a reaping container), so the job loop runs under
#     tmux instead of the container init. The loop is kept anyway: it serialises jobs onto one GPU
#     and gives every job a log without me having to hold a connection open.
#
# Removal:  rm -rf ~/rubitek ~/.local/bin/uv ~/.local/bin/uvx ~/.cache/uv
set -uo pipefail
WS=${WS:-$HOME/rubitek}
LOG=$WS/provision.log
mkdir -p "$WS/jobs" "$WS/done" "$WS/logs" "$HOME/.local/bin"
exec > >(tee -a "$LOG") 2>&1
echo "=== provision start $(date -Is) on $(hostname) ==="

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "host driver $DRV"
if [ "${DRV%%.*}" -lt 580 ] 2>/dev/null; then
  echo "=== ABORT: driver $DRV cannot run the cu130 torch build (need >= 580) ==="
  echo "DRIVER_TOO_OLD $DRV" > "$WS/PROVISION_FAILED"; exit 1
fi

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "=== uv (release binary -> ~/.local/bin) ==="
  curl -Ls https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz \
    | tar xz -C "$HOME/.local/bin" --strip-components=1 || { echo "UV_FETCH_FAILED" > "$WS/PROVISION_FAILED"; exit 1; }
fi
uv --version

VENV=$WS/venv
PY="$VENV/bin/python"
[ -x "$PY" ] || uv venv --python 3.12 "$VENV"
export OMNI_KIT_ACCEPT_EULA=YES

if ! "$PY" -c "import isaacsim" 2>/dev/null; then
  echo "=== isaacsim 6.0.1.0 ==="
  # --index-strategy unsafe-best-match is REQUIRED: isaacsim-core 6.0.1.0 depends on
  # mujoco-usd-converter==0.2.0, which exists on pypi.org but not on pypi.nvidia.com. With uv's
  # default first-index-wins strategy the resolve fails with "No solution found" and provisioning
  # silently continues without Isaac Sim installed at all.
  uv pip install --python "$PY" --index-strategy unsafe-best-match \
    "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com \
    || { echo "ISAACSIM_INSTALL_FAILED" > "$WS/PROVISION_FAILED"; exit 1; }
fi

# THREE HOURS, not one. The tailnet link to this box runs at ~12 KB/s, so 32 MB of trees takes over
# an hour to arrive; the original 60-minute wait expired with 23.3 of 27.2 MB already on disk, wrote
# TREES_NEVER_ARRIVED and killed the session -- which also meant the job loop never armed, so the
# pre-staged curriculum would have sat untouched all night. Waiting longer costs nothing: the loop
# breaks the moment the trees land.
echo "=== waiting for pushed trees (UWLab + _isaaclab3) ==="
for _ in $(seq 1 1080); do
  [ -f "$WS/UWLab/.repo_ready" ] && [ -f "$WS/_isaaclab3/.repo_ready" ] && break
  sleep 10
done
[ -f "$WS/UWLab/.repo_ready" ] || { echo "TREES_NEVER_ARRIVED" > "$WS/PROVISION_FAILED"; exit 1; }
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

# Job loop: anything dropped into $WS/jobs/*.sh runs here. Serialises work onto the one free GPU and
# gives each job its own log, so no ssh connection has to stay open for hours.
echo "=== job loop armed ==="
while true; do
  for job in "$WS"/jobs/*.sh; do
    [ -e "$job" ] || continue
    name=$(basename "$job" .sh)
    echo "[$(date -Is)] running job: $name"
    mv "$job" "$WS/done/$name.sh"
    WS="$WS" bash "$WS/done/$name.sh" > "$WS/logs/$name.log" 2>&1
    echo "[$(date -Is)] job $name finished, wrapper exit $? (grade on artefacts, not on this)"
  done
  sleep 15
done
