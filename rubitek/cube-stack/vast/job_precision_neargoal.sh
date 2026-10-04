#!/usr/bin/env bash
# Does the added short-range shaping term break the 20% near-goal plateau?
#
# One variable against a baseline that is already measured twice (seeds 42 and 7 both rose to ~20%
# by iteration 60 and then held flat for 240 more, median end-of-episode error ~16 mm vs a 5 mm
# tolerance). The variable is `dense_success_reward_fine`, an exp(-d/0.02) goal-distance term added
# alongside the shipped exp(-d/1.0) one -- see _apply_precision_shaping in ur10e_delto_cubestack_cfg
# for the arithmetic. Everything else, including entropy_coef=0.0005, matches the baseline runs.
#
# The plateau appeared by iteration 60, so this answers within roughly half an hour of GPU time. If
# the curve does not leave 20% by iteration 150 the hypothesis is wrong and the full-mixture run on
# the other box should be stopped rather than left to finish.
set -uo pipefail
WS=/workspace
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
cd "$WS/UWLab" || exit 1

# This box provisioned from the onstart script as it was BEFORE the gSDE fix was folded into it, so
# it carries upstream rsl-rl-lib, which has no noise_std_type="gsde" and dies at construction with
#   ValueError: Unknown standard deviation type: gsde. Should be 'scalar' or 'log'
#
# Two things the first attempt at this repair got wrong, both of which failed silently:
#   - it probed `rsl_rl.modules` (the package __init__, which never mentions gsde) instead of
#     `rsl_rl.modules.actor_critic`, where the check actually lives;
#   - it installed with `python -m pip`, but this venv was created by `uv venv`, which does not put
#     pip inside it. The install was a no-op and training died on the same ValueError as before.
# Hence: probe the real module, install with uv, and fail loudly if either step fails.
#
# --no-deps because resolving this package's deps drags torch 2.7.0+cu128 up to 2.13.0+cu130, which
# Isaac 5.1 will not load; torch is re-pinned immediately after and printed as positive proof.
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

ITERS=${ITERS:-1500}
ENVS=${ENVS:-2048}
SEED=${SEED:-42}
ENTROPY=${ENTROPY:-0.0005}

echo "=== precision shaping / near-goal / entropy=$ENTROPY envs=$ENVS seed=$SEED ==="
"$WS/venv/bin/python" scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-UR10eDelto-CubeStackNearGoal-RelCartesianOSC-State-v0 \
  --num_envs "$ENVS" --max_iterations "$ITERS" --seed "$SEED" --headless \
  env.scene.insertive_object=cube34 \
  env.scene.receptive_object=cube34 \
  env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e_delto/OmniReset \
  agent.algorithm.entropy_coef="$ENTROPY" \
  agent.save_interval=50 2>&1 | tr '\r' '\n' | grep -vE '^\[[0-9]'
echo "=== precision near-goal exited ==="
find logs/rsl_rl -name 'model_*.pt' -newermt '-1 day' -printf '%s %p\n' | sort -rn | head -5
