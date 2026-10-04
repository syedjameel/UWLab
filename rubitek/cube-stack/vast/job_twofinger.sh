#!/usr/bin/env bash
# TWO-FINGER HAND, staged backwards curriculum. The variant the brief asked for at the start.
#
# WHY REDUCE THE HAND. Grasping the cube from the Reaching family requires ten of the twenty finger
# joints to move together, all in the same direction, five of them for nine to thirteen consecutive
# full-scale action steps. With independent Gaussian exploration at gSDE std 1.0 that coincidence has
# probability ~1e-6 per episode, and the stored grasp bank shows it is not a search over grasp
# families at all: all 1056 entries are the same scripted-close posture replayed from different
# approach directions, with a maximum pairwise spread of 15.2 degrees SUMMED over all twenty joints.
# Cube handling needs an opposed pair, not five fingers.
#
# Dropping to two driven fingers takes the action space from 26 to 14 and turns that ten-joint
# coincidence into a four-joint one. The undriven fingers are NOT forced to a tucked posture -- they
# simply leave the action term and hold whatever pose the reset gave them, which is what keeps every
# stored reset bank valid instead of requiring hours of re-recording.
#
# STAGED, because a flat mixture spends three quarters of its samples on families the policy cannot
# yet reach. Each stage adds the next family backwards along the task and resumes the previous
# stage's weights. Stage A is trained here rather than seeded, because the 26-dim checkpoints do not
# load into a 14-dim actor.
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

ENVS=${ENVS:-2048}
SEED=${SEED:-42}
ENTROPY=${ENTROPY:-0.0005}
TASK=${TASK:-OmniReset-UR10eDelto-CubeStackTwoFinger-RelCartesianOSC-State-v0}
RUNS=logs/rsl_rl/ur10e_delto_omnireset_agent

C4=ObjectPartiallyAssembledEEGrasped
C3=ObjectAnywhereEEGrasped
C2=ObjectRestingEEGrasped
C1=ObjectAnywhereEEAnywhere

newest_ck () { find "$RUNS" -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-; }

stage () {  # stage <name> <iters> <types> <probs> <resume-ck>
  local name=$1 iters=$2 types=$3 probs=$4 ck=$5
  echo
  echo "=== STAGE $name : $iters iters, families $types ==="
  local resume=()
  [ -n "$ck" ] && [ -f "$ck" ] && { resume=(--resume_path "$ck"); echo "=== resuming from $ck ==="; }
  "$PY" scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$TASK" --num_envs "$ENVS" --max_iterations "$iters" --seed "$SEED" --headless \
    "${resume[@]}" --run_name "twofinger_$name" \
    env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e_delto/OmniReset \
    "env.events.reset_from_reset_states.params.reset_types=$types" \
    "env.events.reset_from_reset_states.params.probs=$probs" \
    agent.algorithm.entropy_coef="$ENTROPY" \
    agent.save_interval=50 2>&1 | tr '\r' '\n' | grep -vE '^\[[0-9]'
  echo "=== stage $name done, newest: $(newest_ck) ==="
}

# STAGE LENGTHS ARE SET FROM THE MEASURED CURVES, NOT ROUND NUMBERS. Near-goal reaches its ceiling
# fast: both five-finger baselines are at ~0.20 by iteration 60 and flat thereafter, and this
# two-finger run was already at 0.171 by iteration 45. Spending 350 iterations there buys nothing
# and costs three hours. The budget belongs at the far end, where the task is actually unsolved --
# stage D is the only one that contains Reaching, and Reaching is the whole question.
#
# At the measured ~33 s/iteration this reaches stage D in about 6 hours instead of 13.
stage A "${ITERS_A:-140}" "[$C4]"                 "[1.0]"                 ""      ; CK=$(newest_ck)
stage B "${ITERS_B:-280}" "[$C4,$C3]"             "[0.5,0.5]"             "$CK"   ; CK=$(newest_ck)
stage C "${ITERS_C:-280}" "[$C4,$C3,$C2]"         "[0.34,0.33,0.33]"      "$CK"   ; CK=$(newest_ck)
stage D "${ITERS_D:-900}" "[$C4,$C3,$C2,$C1]"     "[0.25,0.25,0.25,0.25]" "$CK"

echo "=== two-finger curriculum done ==="
find "$RUNS" -name 'model_*.pt' -newermt '-1 day' -printf '%s %p\n' | sort -rn | head -8
