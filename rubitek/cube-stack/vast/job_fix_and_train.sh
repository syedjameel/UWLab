#!/usr/bin/env bash
# Install the UW-Lab rsl_rl fork, then train the oriented variant.
#
# WHY: the task's agent cfg sets noise_std_type="gsde" (the paper's §3.3 exploration decision).
# Upstream rsl-rl-lib on PyPI does not implement it and raises
#   ValueError: Unknown standard deviation type: gsde. Should be 'scalar' or 'log'
# uwlab_rl/setup.py pins the right source: rsl-rl-lib @ git+https://github.com/UW-Lab/rsl_rl.git@main
#
# --no-deps IS REQUIRED. Installing it with deps drags torch to 2.13.0+cu130, which Isaac Sim 5.1
# does not expect; observed locally. The fork only changes rsl_rl itself.
set -uo pipefail
WS=/workspace
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
cd "$WS/UWLab" || exit 1
PY=$WS/venv/bin/python
export PATH="$HOME/.local/bin:$PATH"

echo "=== installing UW-Lab rsl_rl fork (no deps) ==="
uv pip install --python "$PY" --no-deps --force-reinstall \
  "rsl-rl-lib @ git+https://github.com/UW-Lab/rsl_rl.git@main" 2>&1 | tail -3
"$PY" - <<'PYEOF'
import inspect, torch
from rsl_rl.modules import actor_critic as ac
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gsde supported:", "gsde" in inspect.getsource(ac.ActorCritic.__init__))
PYEOF

ITERS=${ITERS:-2000}
ENVS=${ENVS:-2048}
SEED=${SEED:-42}
echo
echo "=== training ORIENTED: $ITERS iters, $ENVS envs, seed $SEED ==="
"$PY" scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-UR10eDelto-CubeStack-RelCartesianOSC-State-v0 \
  --num_envs "$ENVS" --max_iterations "$ITERS" --seed "$SEED" --headless \
  env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 \
  env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e_delto/OmniReset 2>&1 \
  | tr '\r' '\n' | grep -vE '^\[[0-9]' | tail -600

echo "=== oriented train wrapper exited $? (not trusted) ==="
find logs/rsl_rl -name 'model_*.pt' -printf '%s %p\n' 2>/dev/null | sort -rn | head -3
