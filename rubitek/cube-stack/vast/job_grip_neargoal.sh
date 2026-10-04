#!/usr/bin/env bash
# Does GRIP shaping break the 16 mm near-goal plateau, when goal-distance shaping did not?
#
# WHAT IS ALREADY SETTLED. Two independent seeds plateau at ~20% end-of-episode success with median
# end-of-episode position error ~17 mm against a 5 mm tolerance. `dense_success_reward_fine`
# (exp(-d/0.02) added alongside the shipped exp(-d/1.0)) was the arithmetic's answer: it raises the
# value of closing 16 mm -> 5 mm from +1.1e-4 to +3.3e-2 per step, ~30x. Measured against both
# baselines over matched windows it changed NOTHING:
#
#     run                  it 0-19   20-39   40-58    median err
#     baseline seed 42      0.106    0.145   0.170     17.4 mm
#     baseline seed 7       0.084    0.118   0.168     23.9 mm
#     + dense fine          0.100    0.143   0.170     17.0 mm
#
# So the plateau is not a missing gradient in the goal-distance term. The policy is not failing to
# know where to put the cube; it is failing to put it there.
#
# THE HYPOTHESIS THIS TESTS. In the near-goal family the cube starts in the hand, but nothing in the
# shipped reward ever paid for HOLDING it -- there was no contact term at all, so the grasp the
# policy inherits from the reset is whatever the bank recorded and is free to degrade. A cube held
# loosely slips under the acceleration needed to move it 13 mm, which would cap achievable precision
# regardless of how sharp the goal-distance gradient is.
#
# The full reward chain is now in the config: fingertip_object_distance (dense, pulls the tips onto
# the cube), any_finger_contact, grasp_contact (opposed tips), object_lift. If a firmer grip is what
# the plateau is made of, this moves it. If the curve tracks the same ~20% and ~17 mm, the plateau is
# something else -- most likely the OSC's own positioning precision under load -- and that is worth
# knowing too, because it would mean no reward change can fix it.
#
# Trained from scratch, not resumed: the reward changed, so a resumed critic would spend the early
# iterations relearning a scale rather than testing the hypothesis, and the comparison is against
# from-scratch baselines.
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

ITERS=${ITERS:-800}
ENVS=${ENVS:-2048}
SEED=${SEED:-42}
ENTROPY=${ENTROPY:-0.0005}

echo "=== grip shaping / near-goal / envs=$ENVS seed=$SEED ==="
"$PY" scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-UR10eDelto-CubeStackNearGoal-RelCartesianOSC-State-v0 \
  --num_envs "$ENVS" --max_iterations "$ITERS" --seed "$SEED" --headless \
  --run_name grip_neargoal \
  env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e_delto/OmniReset \
  agent.algorithm.entropy_coef="$ENTROPY" \
  agent.save_interval=50 2>&1 | tr '\r' '\n' | grep -vE '^\[[0-9]'
echo "=== grip near-goal exited ==="
find logs/rsl_rl -name 'model_*.pt' -newermt '-1 day' -printf '%s %p\n' | sort -rn | head -5
