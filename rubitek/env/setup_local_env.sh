#!/usr/bin/env bash
# Local Isaac Sim 5.1 + UWLab environment for the cube-stacking work.
# Mirrors UR10E_OMNIRESET_GUIDE.md section 1a, but with a uv-managed venv instead of conda
# (no conda on this box; system python is 3.14, Isaac Sim 5.1 needs 3.11).
set -euo pipefail

UV=/home/mattephi/.local/share/mise/installs/uv/latest/uv-x86_64-unknown-linux-musl/uv
VENV=/home/mattephi/Work/Rubitek/uwlab-env/.venv
UWLAB=/home/mattephi/Work/Rubitek/UWLab
LOG=/home/mattephi/Work/Rubitek/uwlab-env/setup.log

exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -Is) starting ==="

step() { echo; echo "### [$(date +%H:%M:%S)] $*"; df -h /home | tail -1; }

step "1. venv (python 3.11)"
[ -d "$VENV" ] || "$UV" venv --python 3.11 "$VENV"
PY="$VENV/bin/python"
PIP="$UV pip install --python $PY"

step "2. isaacsim 5.1.0"
$PIP "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

step "3. torch 2.7.0 cu128"
$PIP torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

step "4. clone upstream IsaacLab"
mkdir -p "$UWLAB/_isaaclab"
[ -d "$UWLAB/_isaaclab/IsaacLab/.git" ] || \
  git clone --depth 1 --branch main https://github.com/isaac-sim/IsaacLab.git "$UWLAB/_isaaclab/IsaacLab"

step "5. IsaacLab editable installs"
for pkg in isaaclab isaaclab_assets isaaclab_tasks 'isaaclab_rl[all]'; do
  d="${pkg%%[*}"
  $PIP -e "$UWLAB/_isaaclab/IsaacLab/source/$d" --extra-index-url https://pypi.nvidia.com
done

step "6. UWLab source packages (editable)"
for d in "$UWLAB"/source/*/; do
  [ -f "$d/setup.py" ] && $PIP -e "$d"
done

step "7. rsl_rl (UW-Lab fork 3.1.2)"
$PIP "rsl-rl-lib==3.1.2" || $PIP "git+https://github.com/UW-Lab/rsl_rl.git"

step "8. done"
"$PY" -c "import isaacsim, torch; print('isaacsim ok; torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "=== $(date -Is) finished ==="
