#!/usr/bin/env bash
# TWO-FINGER HAND + GRIP BIAS, staged backwards curriculum.
#
# WHAT CHANGED AND WHY. Every "EEGrasped" reset family was hollow. `grip_probe.py` resets into a
# family, applies the null action for six seconds, and reads the object back:
#
#   ObjectPartiallyAssembledEEGrasped   cube 55.8 -> 54.8 mm   nearest fingertip 35.3 mm
#   ObjectRestingEEGrasped              cube 21.5 -> 21.5 mm   nearest fingertip 40.5 mm
#   ObjectAnywhereEEGrasped             cube 102.3 -> 34.2 mm  nearest fingertip 30.6 mm
#
# The first two only look stable because their cube is resting on the base cube or on the table.
# The third is the only family whose cube starts in the air, and it falls out of the hand. No
# fingertip in any family carries more than 0.08 N against a 0.29 N cube weight.
#
# The cause is the action term, not the banks. The hand runs RelativeJointPositionAction with
# use_zero_offset=True, so target = measured + scale*action, and the first apply_actions() after
# every reset overwrites the joint_position_target the bank restored. At action 0 the PD error is
# zero and so is the grip force -- the recorded squeeze is discarded one control step into every
# episode. That is why Metrics/task_1_success_rate (ObjectAnywhereEEGrasped) sat at 0.0000 through
# 200 iterations while near-goal trained to 0.20: near-goal never needed a grasp.
#
# The Grip task adds a constant closing offset on the flexion joints, which restores exactly the
# condition the recorder captured -- a set point held a fixed angle ahead of the measurement. The
# reset banks are unchanged and become valid instead of hollow.
#
# Combined with the two-finger hand, which leads every five-finger run at every window from
# iteration 16 on with an 8-dimensional hand instead of 20.
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
TASK=${TASK:-OmniReset-UR10eDelto-CubeStackTwoFingerGrip-RelCartesianOSC-State-v0}
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
    "${resume[@]}" --run_name "grip_$name" \
    env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e_delto/OmniReset \
    "env.events.reset_from_reset_states.params.reset_types=$types" \
    "env.events.reset_from_reset_states.params.probs=$probs" \
    agent.algorithm.entropy_coef="$ENTROPY" \
    agent.save_interval=50 2>&1 | tr '\r' '\n' | grep -vE '^\[[0-9]'
  echo "=== stage $name done, newest: $(newest_ck) ==="
}

# STAGE B IS THE ONE THAT MATTERS NOW. C3 is the family the grip fix unblocks: the cube starts in
# the hand 209 mm from the goal, there is no release or zero-velocity requirement in the success
# predicate, and a policy that simply holds and transports scores. Under the old action term the
# cube was on the table long before it could get there. Budget accordingly -- A is a short warm
# start on the family that already trains, and D is where Reaching finally enters.
# START_CK, when set, is an EXPLICIT path -- never "the newest checkpoint on the box". Defaulting
# it to newest once silently resumed stage B from a previous run's untrained model_0 while printing
# "resuming from...", which is indistinguishable from a real resume in the log. An unset START_CK
# means stage A trains from scratch, which is correct for a fresh box.
CK0=${START_CK:-}
if [ -n "$CK0" ] && [ ! -f "$CK0" ]; then
  echo "=== START_CK=$CK0 does not exist; refusing to silently train from scratch ==="
  exit 1
fi
if [ -n "$CK0" ]; then
  echo "=== stage A skipped: seeding from $CK0 ==="
  CK=$CK0
else
  stage A "${ITERS_A:-120}" "[$C4]"               "[1.0]"                 ""      ; CK=$(newest_ck)
fi
stage B "${ITERS_B:-400}" "[$C4,$C3]"             "[0.4,0.6]"             "$CK"   ; CK=$(newest_ck)
stage C "${ITERS_C:-300}" "[$C4,$C3,$C2]"         "[0.3,0.4,0.3]"         "$CK"   ; CK=$(newest_ck)
stage D "${ITERS_D:-900}" "[$C4,$C3,$C2,$C1]"     "[0.25,0.25,0.25,0.25]" "$CK"

echo "=== grip curriculum done ==="
find "$RUNS" -name 'model_*.pt' -newermt '-1 day' -printf '%s %p\n' | sort -rn | head -8
