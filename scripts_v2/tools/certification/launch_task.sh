#!/usr/bin/env bash
# Launch ONE dexlift run for ANY task id, with the plant specified as four independent axes, and
# refuse to keep running if the plant that actually booted is not the one asked for.
#
#   TASK=<gym id> COLLIDERS=hullfix3 SELFCOLL=on HAND=ref ARM=ours [WARM=<ckpt>] [ITERS=6000] \
#   [WANDB=0] [OVERRIDES="a=b c=d"] launch_task.sh <gpu> <name> <num_envs> <minibatch>
#
# This is launch_plant.sh generalized over the task id. Everything about the assertion block is
# unchanged and is the point of the script: there are sixteen reachable plants plus now several task
# ids, and a wrong combination is invisible -- it trains, it logs, it produces a curve.
#
# MINIBATCH MUST SCALE WITH num_envs. updates/epoch = 5 * N * 36 / minibatch; the certified run does
# 20, so minibatch = 9 * N. N=2048 -> 18432, N=4096 -> 36864, N=8192 -> 73728.
set -uo pipefail

GPU=${1:?usage: launch_task.sh <gpu> <name> <num_envs> <minibatch>}
NAME=${2:?}; NENV=${3:?}; MB=${4:?}
TASK=${TASK:?set TASK=<gym id>}
COLLIDERS=${COLLIDERS:?set COLLIDERS=hullfix3|hullfix|hull}
SELFCOLL=${SELFCOLL:?set SELFCOLL=on|off}
HAND=${HAND:?set HAND=ref|ours}
ARM=${ARM:?set ARM=ref|ours}
ITERS=${ITERS:-6000}
WANDB=${WANDB:-1}
OVERRIDES=${OVERRIDES:-}

EXPECT=$(( 5 * NENV * 36 / MB ))
[ "$EXPECT" -eq 20 ] || { echo "REFUSING: minibatch $MB gives $EXPECT updates/epoch, not 20"; exit 1; }
[ $(( NENV * 36 % MB )) -eq 0 ] || { echo "REFUSING: minibatch $MB does not divide batch $(( NENV * 36 ))"; exit 1; }

# This box is SHARED. Never launch on a GPU another user is on.
USED=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
[ "${USED:-999999}" -lt 2000 ] || { echo "REFUSING: GPU $GPU has ${USED} MiB in use"; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export DEXLIFT_REF_RESET=1

# DEXLIFT_HULLFIX / DEXLIFT_CONVEXHULL / DEXLIFT_NO_SELFCOLL removed -- read by no python
# anywhere (bead dr-76w.18). COLLIDERS/SELFCOLL still select WANT_* and are still asserted.
case "$COLLIDERS" in
  hullfix)  WANT_COLL=hullfix ;;
  hullfix2) WANT_COLL=hullfix2 ;;
  hullfix3) WANT_COLL=hullfix3 ;;
  hull)     WANT_COLL=convexHull ;;
  *) echo "REFUSING: COLLIDERS=$COLLIDERS"; exit 1 ;;
esac
case "$SELFCOLL" in
  on)  WANT_SC=ON ;;
  off) WANT_SC=OFF ;;
  *) echo "REFUSING: SELFCOLL=$SELFCOLL"; exit 1 ;;
esac
export DEXLIFT_REF_ACTUATORS=1
case "$HAND" in ref) export DEXLIFT_REF_HAND_ACT=1; WANT_HAND=reference ;;
                ours) export DEXLIFT_REF_HAND_ACT=0; WANT_HAND=identified ;;
                *) echo "REFUSING: HAND=$HAND"; exit 1 ;; esac
case "$ARM" in ref) export DEXLIFT_REF_ARM_ACT=1; WANT_ARM=reference ;;
               ours) export DEXLIFT_REF_ARM_ACT=0; WANT_ARM=identified ;;
               *) echo "REFUSING: ARM=$ARM"; exit 1 ;; esac

WANT="colliders=$WANT_COLL self_collisions=$WANT_SC hand_act=$WANT_HAND arm_act=$WANT_ARM"

# OPTIONAL ORIENTATION STAGING. Narrows the object's reset orientation AND the goal's together, so
# the DEMANDED rotation is bounded. It redefines the task, so it is asserted below like the plant is:
# a run that was meant to be staged and silently was not would look like a hard task rather than a
# missing variable.
TILT=${TILT:-}
[ -n "$TILT" ] && export DEXLIFT_POSE_TILT="$TILT"

cd "$HOME/UWLab_ur5edelto" || exit 1
export PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl"

LOG="$HOME/train_${NAME}.log"; : > "$LOG"
echo "task=$TASK gpu=$GPU envs=$NENV minibatch=$MB updates/epoch=$EXPECT iters=$ITERS" | tee -a "$LOG"
echo "want: $WANT" | tee -a "$LOG"
[ -n "$OVERRIDES" ] && echo "overrides: $OVERRIDES" | tee -a "$LOG"

# OPTIONAL WARM START. rl_games RESTORES THE EPOCH COUNTER, so --max_iterations is ABSOLUTE, not
# additional. It does NOT restore the ADR curriculum, so every warm start re-climbs from difficulty
# 0 and looks like a regression for ~50-200 epochs. That dip is the curriculum, not the checkpoint.
ARGS=()
WARM=${WARM:-}
if [ -n "$WARM" ]; then
  [ -f "$WARM" ] || { echo "REFUSING: WARM=$WARM does not exist"; exit 1; }
  echo "warm start from: $WARM" | tee -a "$LOG"
  ARGS+=(--checkpoint "$WARM")
fi
if [ "$WANDB" = 1 ]; then
  ARGS+=(--track --wandb-project-name dexr-tessolo-lift --wandb-name "$NAME"
         --wandb-entity i_domrachev-interactive-robotic-systems-lab-kaist)
fi
# Word-split OVERRIDES deliberately: hydra key=value pairs, none of which may contain spaces.
# shellcheck disable=SC2206
[ -n "$OVERRIDES" ] && ARGS+=($OVERRIDES)

setsid nohup timeout -s KILL 172800 "$HOME/UWLab/env_uwlab/bin/python" -u \
  scripts/reinforcement_learning/rl_games/train.py \
  --task "$TASK" \
  "${ARGS[@]}" \
  --num_envs "$NENV" --seed 42 --max_iterations "$ITERS" --headless \
  agent.params.config.minibatch_size="$MB" \
  >> "$LOG" 2>&1 < /dev/null &

PID=$!
echo "LAUNCHED pid $PID"

# KILL THE WHOLE PROCESS GROUP, NOT THE WRAPPER. The job is launched under setsid, so $PID is the
# `timeout` leader and the Isaac python is its child in that new session. `kill -9 $PID` reaps only
# the wrapper and leaves Isaac holding ~17 GB of VRAM with ppid 1 -- which then makes the NEXT launch
# refuse the GPU it just freed. Measured: a killed run left pid 3324676 alive on 16907 MiB.
kill_run() {
  kill -9 -"$PID" 2>/dev/null || kill -9 "$PID" 2>/dev/null
  sleep 5
}

DEADLINE=$(( SECONDS + 2400 )); STATUS=TIMEOUT
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  sleep 20
  grep -aqE "OutOfMemoryError|out of memory|Traceback|Error executing job" "$LOG" && { STATUS=CRASHED; break; }
  grep -aqE "epoch: [0-9]+/" "$LOG" && { STATUS=TRAINING; break; }
  kill -0 "$PID" 2>/dev/null || { STATUS=EXITED; break; }
done

echo "LAUNCH_STATUS=$STATUS"
if [ "$STATUS" != "TRAINING" ]; then
  grep -anE "out of memory|OutOfMemoryError|Traceback|Error executing job|ValueError|KeyError" "$LOG" | tail -8
  tail -25 "$LOG"
  exit 1
fi

GOT=$(grep -aoE "\[dexlift\] PLANT .*" "$LOG" | head -1)
echo "got:  $GOT"
# MATCH WHOLE SPACE-DELIMITED TOKENS, not substrings: a substring test passes "colliders=hullfix"
# against a booted "colliders=hullfix3", which is the exact silent-wrong-experiment this prevents.
for kv in $WANT; do
  case " $GOT " in
    *" $kv "*) ;;
    *) echo "PLANT MISMATCH: expected token '$kv', booted plant is: $GOT -- KILLING"
       kill_run; exit 1 ;;
  esac
done
echo "PLANT_CONFIRMED"

# Assert the staging separately from the plant: it comes from a different line and means a different
# thing (a narrower TASK, not a different ROBOT).
if [ -n "$TILT" ]; then
  # COMPARE THE FULL FORMATTED NUMBER, not a substring, and do not exclude '.' from the capture.
  # The first version of this check used [^.]* and truncated "+-0.3000 rad" to "+-0", then killed a
  # correctly-staged run for a mismatch it had manufactured itself. A substring test would also pass
  # "+-0.35" for a requested "+-0.3", so the printf-matched whole token is the check.
  TGOT=$(grep -aoE "\[dexlift\] POSE_TILT staged: .*" "$LOG" | head -1)
  WANT_TILT=$(printf "+-%.4f rad" "$TILT")
  case "$TGOT" in
    *"$WANT_TILT"*) echo "TILT_CONFIRMED: $WANT_TILT" ;;
    *) echo "TILT MISMATCH: asked for '$WANT_TILT', log says: ${TGOT:-<no POSE_TILT line at all>} -- KILLING"
       kill_run; exit 1 ;;
  esac
else
  grep -aq "POSE_TILT staged" "$LOG" && {
    echo "TILT MISMATCH: no TILT requested but the run staged one -- KILLING"; kill_run; exit 1; }
fi

grep -aoE "minibatch_size[^,}]*" "$LOG" | head -2
grep -aoE "epoch: [0-9]+/[0-9]+" "$LOG" | tail -1
grep -aoE "fps total: [0-9]+" "$LOG" | tail -1
nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader
