#!/usr/bin/env bash
# Resume a trained checkpoint on the C4+C3 MIXTURE, with the published objective (bonus weight 0).
#
# WHY. The hold/carry bonus failed its own pre-committed test: at iteration 400 the 50-iteration
# rolling mean of end_of_episode_pos_align_error was 0.1869 m against a 0.170 m bar set before the
# data, while the bonus term itself doubled (0.0315 -> 0.0649) and mean reward went 2.30 -> 3.77.
# It was inflating its own payout far faster than it moved the task, and the task metric stayed ~37x
# away from the 5 mm success gate. So it is switched off here: UWLAB_HOLD_BONUS_WEIGHT=0.
#
# AND WHY THE MIXTURE. The A/B deliberately trained C3-ONLY so the comparison would be clean. But C3
# (ObjectAnywhereEEGrasped) starts the cube 210 mm from the goal and the policy never once seats it,
# so a C3-only curriculum never shows the agent what seating looks like -- the clean experiment may
# have removed the very behaviour being asked for. The only run that ever got near the goal
# (146 mm) was carry_hold, which trained the 50/50 C4+C3 mixture: C4
# (ObjectPartiallyAssembledEEGrasped) starts the cube already seated, so it supplies the near-goal
# release-and-withdraw experience C3 cannot.
#
# Resuming rather than restarting keeps the grasp, which IS solved: 64/64 environments closing the
# two-finger gate at 10-12 N, against 0.0000 N for every policy before tonight. That took the control
# fixes (hand action scale 0.02 rad/step, entropy_coef 0.001) and should not be paid for twice.
#
# Usage:  ./resume_mixed.sh <checkpoint.pt> <run_name> [iters] [envs] [weight]
set -uo pipefail
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
CS=/home/mattephi/Work/Rubitek/cube-stack
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv-is6/bin/python
cd "$REPO" || exit 1

CK=${1:?usage: resume_mixed.sh <checkpoint.pt> <run_name> [iters] [envs] [weight]}
NAME=${2:?usage: resume_mixed.sh <checkpoint.pt> <run_name> [iters] [envs] [weight]}
ITERS=${3:-600}
N=${4:-512}
WEIGHT=${5:-0}
SEED=42
TASK=OmniReset-UR10eDelto-CubeStackTwoFingerGripCarry-RelCartesianOSC-State-v0
C3=ObjectAnywhereEEGrasped
C4=ObjectPartiallyAssembledEEGrasped
[ -f "$CK" ] || { echo "no such checkpoint: $CK"; exit 1; }

export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="/home/mattephi/Work/Rubitek/uwlab-env/compat:$REPO/source/uwlab_assets:$REPO/source/uwlab_tasks:$REPO/source/uwlab:$REPO/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-0.15
export UWLAB_LATCHED_HAND=1 UWLAB_TUCK_OTHERS=0 UWLAB_GRIP_BIAS_RAD=0.05
export UWLAB_HAND_EFFORT_SCALE=${UWLAB_HAND_EFFORT_SCALE:-20}
export UWLAB_HAND_ACTION_SCALE=0.02          # control fix, not a reward change
export UWLAB_HOLD_BONUS_WEIGHT="$WEIGHT"     # 0 = OmniReset's published objective

echo "RESUME name=$NAME from=$(basename "$CK") families=[C4,C3] 50/50 weight=$WEIGHT iters=$ITERS envs=$N $(date -Is)"
"$PY" scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$TASK" \
  --num_envs "$N" --max_iterations "$ITERS" --seed "$SEED" --headless \
  --run_name "$NAME" --resume_path "$CK" \
  "env.events.reset_from_reset_states.params.reset_types=[$C4,$C3]" \
  "env.events.reset_from_reset_states.params.probs=[0.5,0.5]" \
  agent.algorithm.entropy_coef=0.001 agent.save_interval=25 \
  2>&1 | tr '\r' '\n' | grep -vE '^\[[0-9]' > "$CS/logs/$NAME.log"
RC=$?
echo "RESUME_DONE $NAME rc=$RC $(date -Is)"
# `grep -c` exits 1 on a zero count; never let it set the script's status (that silently broke an
# earlier `A && B` chain, so the second arm never ran).
echo "  'bonus DISABLED' lines: $(grep -acE 'hold.carry bonus DISABLED' "$CS/logs/$NAME.log" || true)"
exit "$RC"
