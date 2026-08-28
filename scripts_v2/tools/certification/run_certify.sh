#!/usr/bin/env bash
# Run certify_pose.py in the plant the policy was TRAINED in, and assert that plant before believing
# the numbers. Same four axes as launch_task.sh, same reason: probing a policy under a different arm
# or a different collider set measures a different robot.
#
#   TASK=<id> COLLIDERS=hullfix3 SELFCOLL=on HAND=ref ARM=ours [GPU=3] [EPISODES=256] [POS_TOL=0.01] \
#       [ADR_DIFFICULTY=max] run_certify.sh <tag> <checkpoint>
#
# ADR_DIFFICULTY is forwarded to certify_pose.py's own --adr_difficulty (choices: max/configured;
# default max). Certification protocols that need to be comparable across runs should still set this
# explicitly rather than lean on the default -- see cert_g3z4_finetune.sh's own comment on why a
# silent dependency on a default is the wrong shape for a number a 6-hour finetune decision reads.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=${GPU:-3}
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export DEXLIFT_REF_RESET=1 DEXLIFT_REF_ACTUATORS=1
# THE EVALUATOR MUST REPRODUCE THE TRAINING TASK, INCLUDING ITS DROP HEIGHT. _apply_pose_tilt_stage
# clamps the reset drop to 0.05 m by DEFAULT whenever a tilt is staged, so certifying a policy that
# trained with the unclamped 0.4 m drop under a bare TILT= would score it on a task it never saw.
# Pass DROP_Z explicitly to match the run being certified.
[ -n "${TILT:-}" ] && export DEXLIFT_POSE_TILT="$TILT"
[ -n "${DROP_Z:-}" ] && export DEXLIFT_DROP_Z="$DROP_Z"
unset DEXLIFT_HULLFIX DEXLIFT_CONVEXHULL DEXLIFT_NO_SELFCOLL

case "${COLLIDERS:?set COLLIDERS}" in
  hullfix)  export DEXLIFT_HULLFIX=1; WANT_COLL=hullfix ;;
  hullfix2) export DEXLIFT_HULLFIX=2; WANT_COLL=hullfix2 ;;
  hullfix3) export DEXLIFT_HULLFIX=3; WANT_COLL=hullfix3 ;;
  hull)     export DEXLIFT_CONVEXHULL=1; WANT_COLL=convexHull ;;
  *) echo "REFUSING: COLLIDERS=$COLLIDERS"; exit 1 ;;
esac
WANT_SC=ON; [ "${SELFCOLL:?set SELFCOLL}" = off ] && { export DEXLIFT_NO_SELFCOLL=1; WANT_SC=OFF; }
case "${HAND:-ref}" in ref) export DEXLIFT_REF_HAND_ACT=1; WANT_HAND=reference ;;
                       ours) export DEXLIFT_REF_HAND_ACT=0; WANT_HAND=identified ;;
                       *) echo "REFUSING: HAND=$HAND"; exit 1 ;; esac
case "${ARM:-ref}"  in ref) export DEXLIFT_REF_ARM_ACT=1; WANT_ARM=reference ;;
                       ours) export DEXLIFT_REF_ARM_ACT=0; WANT_ARM=identified ;;
                       *) echo "REFUSING: ARM=$ARM"; exit 1 ;; esac

TAG=${1:?usage: run_certify.sh <tag> <checkpoint>}
CKPT=${2:?}
[ -f "$CKPT" ] || { echo "REFUSING: checkpoint $CKPT does not exist"; exit 1; }

# BOX LAYOUT IS NOW OVERRIDABLE, DEFAULTS UNCHANGED. Every default below reproduces exactly what
# this script did before, so the existing cert drivers keep working untouched. The overrides exist
# because the three paths this script used to hard-code (the repo at $HOME/UWLab_ur5edelto, the
# interpreter at $HOME/UWLab/env_uwlab/bin/python, the entry point at $HOME/certify_pose.py) are
# all ABSENT on at least one box this project actually runs on -- verified by listing them, not
# assumed. The failure that caused was not loud: `cd ... || exit 1` returns 1, a caller that pipes
# this script's output to tee without checking the status then proceeds to its next run, and the
# driver prints its own ALL_DONE marker having certified nothing at all.
REPO_DIR=${REPO_DIR:-$HOME/UWLab_ur5edelto}
PYTHON_BIN=${PYTHON_BIN:-$HOME/UWLab/env_uwlab/bin/python}
CERTIFY_PY=${CERTIFY_PY:-$HOME/certify_pose.py}
OUT_DIR=${OUT_DIR:-$HOME}

cd "$REPO_DIR" || { echo "REFUSING: REPO_DIR=$REPO_DIR does not exist"; exit 1; }
[ -x "$PYTHON_BIN" ] || { echo "REFUSING: PYTHON_BIN=$PYTHON_BIN is not an executable"; exit 1; }
[ -f "$CERTIFY_PY" ] || { echo "REFUSING: CERTIFY_PY=$CERTIFY_PY does not exist"; exit 1; }
mkdir -p "$OUT_DIR" || { echo "REFUSING: cannot create OUT_DIR=$OUT_DIR"; exit 1; }
export PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl"

LOG="$OUT_DIR/cert_$TAG.log"; OUT="$OUT_DIR/cert_$TAG.json"
timeout -s KILL 7200 "$PYTHON_BIN" -u "$CERTIFY_PY" \
    --task "${TASK:?set TASK}" --checkpoint "$CKPT" \
    --num_envs "${NUM_ENVS:-256}" --episodes "${EPISODES:-256}" \
    --pos_tol "${POS_TOL:-0.01}" --adr_difficulty "${ADR_DIFFICULTY:-max}" \
    --out "$OUT" --headless > "$LOG" 2>&1
echo "rc=$?"

WANT="colliders=$WANT_COLL self_collisions=$WANT_SC hand_act=$WANT_HAND arm_act=$WANT_ARM"
GOT=$(grep -aoE "\[dexlift\] PLANT .*" "$LOG" | head -1)
for kv in $WANT; do
  case " $GOT " in
    *" $kv "*) ;;
    *) echo "PLANT MISMATCH: wanted '$kv', got: $GOT"; exit 1 ;;
  esac
done
echo "plant: $GOT"
grep -aE "RESULT|Traceback|Error|inference tensor" "$LOG" | tail -12
