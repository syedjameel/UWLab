#!/usr/bin/env bash
# A/B the hold/carry bonus against the BASELINE OmniReset objective, matched in every other respect.
#
# WHY THIS FILE EXISTS. The first arm of this experiment (weight 0) was launched as an inline shell
# command and the exact invocation was then unrecoverable -- no script, no fish history, and the
# training log does not echo argv. That made the arms impossible to verify as matched, which is
# fatal: the PREVIOUS cross-run comparison I attempted had to be retracted because the two runs
# silently differed in reset mixture (C3-only vs a 50/50 C4+C3 mix) and env count (512 vs 1024).
# Episode-reward means across different task distributions are not comparable. So both arms now come
# from this one file, and the only thing that may differ between them is UWLAB_HOLD_BONUS_WEIGHT.
#
# WHAT IS BEING TESTED. With the two control fixes in place (hand action scale 0.02 rad/step,
# entropy_coef 0.001), the baseline objective DOES produce a real grasp -- measured on C3 from the
# scene's own contact sensor: 64/64 envs close the two-finger gate at t=0 at a median 10.4 N. But it
# does not carry: the hand releases at a median step 12, the cube falls to the table in 56/64 envs
# (z 170 -> 21 mm), and cube-to-goal ends where it started (237.3 -> 237.9 mm), 0/64 success.
# The hold/carry bonus targets exactly that release. This run asks whether it actually fixes it, or
# whether it just buys the two failure modes it has already produced (clamp-and-wander at flat
# weight; carry-then-hover with the untapered form).
#
# Usage:  ./ab_hold_bonus.sh <weight> <run_name> [iters] [envs]
#   ./ab_hold_bonus.sh 0.5 ab_bonus05        # the arm to compare against ablation_nobonus
#   ./ab_hold_bonus.sh 0   ab_nobonus_rerun  # reproduces the weight-0 arm from scratch
set -uo pipefail
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
CS=/home/mattephi/Work/Rubitek/cube-stack
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv-is6/bin/python
cd "$REPO" || exit 1

WEIGHT=${1:?usage: ab_hold_bonus.sh <weight> <run_name> [iters] [envs]}
NAME=${2:?usage: ab_hold_bonus.sh <weight> <run_name> [iters] [envs]}
ITERS=${3:-200}
N=${4:-512}
SEED=42
TASK=OmniReset-UR10eDelto-CubeStackTwoFingerGripCarry-RelCartesianOSC-State-v0
C3=ObjectAnywhereEEGrasped

export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="/home/mattephi/Work/Rubitek/uwlab-env/compat:$REPO/source/uwlab_assets:$REPO/source/uwlab_tasks:$REPO/source/uwlab:$REPO/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-0.15
export UWLAB_LATCHED_HAND=1 UWLAB_TUCK_OTHERS=0 UWLAB_GRIP_BIAS_RAD=0.05
# The endurance probe selected x20; the acceptance harness uses x20. Both arms must match it.
export UWLAB_HAND_EFFORT_SCALE=${UWLAB_HAND_EFFORT_SCALE:-20}
# The two control fixes -- NOT reward changes, and applied identically to both arms.
export UWLAB_HAND_ACTION_SCALE=0.02
# The single variable under test.
export UWLAB_HOLD_BONUS_WEIGHT="$WEIGHT"

echo "AB weight=$WEIGHT name=$NAME iters=$ITERS envs=$N seed=$SEED effort=x$UWLAB_HAND_EFFORT_SCALE $(date -Is)"
"$PY" scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$TASK" \
  --num_envs "$N" --max_iterations "$ITERS" --seed "$SEED" --headless \
  --run_name "$NAME" \
  "env.events.reset_from_reset_states.params.reset_types=[$C3]" \
  "env.events.reset_from_reset_states.params.probs=[1.0]" \
  agent.algorithm.entropy_coef=0.001 agent.save_interval=25 \
  2>&1 | tr '\r' '\n' | grep -vE '^\[[0-9]' > "$CS/logs/$NAME.log"
RC=$?
echo "AB_DONE $NAME rc=$RC $(date -Is)"
# `grep -c` exits 1 when the count is zero, which is the CORRECT result for a weight>0 arm. That
# exit status used to become the script's, so chaining two arms with `&&` silently short-circuited
# after the first one and the second never ran. Report the count without letting it set $?.
echo "  'bonus DISABLED' lines: $(grep -acE 'hold.carry bonus DISABLED' "$CS/logs/$NAME.log" || true)"
echo "  Episode_Reward/hold_bonus lines: $(grep -acE 'Episode_Reward/hold_bonus' "$CS/logs/$NAME.log" || true)"
exit "$RC"
