#!/usr/bin/env bash
# THE ORIENTED VARIANT: the paper's cube stacking, where a chosen face must end up pointing up.
#
# HOW IT DIFFERS FROM THE POSITION-ONLY RUN. One threshold and one reward clause, nothing else:
#   success requires euler_xy < 0.05 rad AS WELL AS position < 0.005 m
#   dense_success_reward keeps its orientation summand (and the 1/2 that goes with it, per Eq. 5)
# Same robot, same banks, same physics, same optimiser settings. It is a strictly harder test on
# identical dynamics, not a different task.
#
# Note euler_xy drops YAW upstream (ProgressContext sums |wrap_to_pi(e_x)| + |wrap_to_pi(e_y)|), so
# "oriented" here means roll and pitch are constrained while the cube may be spun freely about the
# vertical -- exactly the paper's "with a desired orientation" as this codebase implements it.
#
# STAGED, NOT FLAT. Near-goal first, then the full four-family mixture resuming from it. The
# position-only run measured what a flat mixture costs at this compute budget: near-goal-only
# reached its ceiling in 60 iterations while the flat mixture was still near zero at 400. Orientation
# only makes that worse, and the branch's own table-leg measurements put the cost of adding
# orientation at 4.5x on median precision (4.8 mm -> 26.4 mm). Spending stage 1 where the signal is
# is the only way this variant produces anything within the budget.
set -uo pipefail
WS=/workspace
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
cd "$WS/UWLab" || exit 1

UV="$HOME/.local/bin/uv"
PY="$WS/venv/bin/python"
if ! "$PY" -c "import inspect, rsl_rl.modules.actor_critic as m, sys; sys.exit(0 if 'gsde' in inspect.getsource(m) else 1)"; then
  echo "=== repairing rsl_rl: installing the UW-Lab gSDE fork ==="
  "$UV" pip install --python "$PY" --no-deps --force-reinstall \
    "rsl-rl-lib @ git+https://github.com/UW-Lab/rsl_rl.git@main" || exit 1
  "$UV" pip install --python "$PY" torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128 || exit 1
fi
"$PY" -c "import torch, inspect, rsl_rl.modules.actor_critic as m; print('torch', torch.__version__, '| gsde present:', 'gsde' in inspect.getsource(m))" || exit 1

ENVS=${ENVS:-2048}
SEED=${SEED:-42}
ENTROPY=${ENTROPY:-0.0005}
RUNS=logs/rsl_rl/ur10e_delto_omnireset_agent
newest_ck () { find "$RUNS" -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-; }

echo "=== ORIENTED stage 1: near-goal only, ${ITERS_A:-400} iters ==="
"$PY" scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-UR10eDelto-CubeStack-RelCartesianOSC-State-v0 \
  --num_envs "$ENVS" --max_iterations "${ITERS_A:-400}" --seed "$SEED" --headless \
  --run_name oriented_neargoal \
  env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e_delto/OmniReset \
  'env.events.reset_from_reset_states.params.reset_types=[ObjectPartiallyAssembledEEGrasped]' \
  'env.events.reset_from_reset_states.params.probs=[1.0]' \
  agent.algorithm.entropy_coef="$ENTROPY" \
  agent.save_interval=50 2>&1 | tr '\r' '\n' | grep -vE '^\[[0-9]'

CK=$(newest_ck)
echo "=== ORIENTED stage 2: flat four-family mixture, resuming from $CK ==="
"$PY" scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-UR10eDelto-CubeStack-RelCartesianOSC-State-v0 \
  --num_envs "$ENVS" --max_iterations "${ITERS_B:-900}" --seed "$SEED" --headless \
  --resume_path "$CK" --run_name oriented_mixture \
  env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e_delto/OmniReset \
  agent.algorithm.entropy_coef="$ENTROPY" \
  agent.save_interval=50 2>&1 | tr '\r' '\n' | grep -vE '^\[[0-9]'

echo "=== oriented done ==="
find "$RUNS" -name 'model_*.pt' -newermt '-1 day' -printf '%s %p\n' | sort -rn | head -8
