#!/usr/bin/env bash
# DISTANCE LADDER for Stable-Grasp. Resumes an existing run rather than restarting it.
#
# THE PROBLEM THIS ADDRESSES. Stable-Grasp has read 0.0000 on every box, and the reason is not the
# reward: a completed carry is worth roughly 15x the cost of moving in accumulated return. It is
# exploration. Measured off the banks:
#
#     Near-Goal      n=125   min 0 mm   median  12 mm   max  25 mm
#     Stable-Grasp   n=122   min 15 mm  median 215 mm   max 392 mm   (14 states under 100 mm)
#
# The curriculum jumps from a 12 mm nudge to a 215 mm carry with nothing between, and the far tail
# is past what the arm can traverse in the step budget at all -- those episodes cannot terminate
# successfully however good the policy is. A policy that never once reaches the goal never observes
# that reaching it is valuable, so standing still stays locally optimal. Both tracked boxes show arm
# action magnitude DECAYING through training, which is what that looks like from outside.
#
# THE LADDER. Stage B1 draws Stable-Grasp only from states already within 148 mm -- close enough
# that exploration can stumble into the goal and bootstrap the value -- then B2 opens the full bank.
# Selected purely by dataset_dir; the filtered root holds the same filenames and every other family
# unchanged, so no code, task id or reward moves.
#
# AND A DEFECT UNDER THE GAP. Success compares the insertive BOTTOM face to the receptive TOP face,
# both offsets rotated by their own body. A cube held inverted has its bottom face pointing up, so
# zeroing the predicate would require its centre INSIDE the base cube -- unreachable at that
# attitude, and the NoOrient task pays nothing for rotating out of it. Measured on this bank:
#
#     Stable-Grasp   45 of 122 states (37%) need the cube below the base's top face
#     Near-Goal      13 of 125 states (10%)
#
# So roughly a third of every Stable-Grasp batch was gradient saying the family is unsolvable. Every
# root below is filtered on SEATABILITY as well as distance; `seatall` keeps the full 30-389 mm
# spread and drops only the states that cannot be seated at all. That is a data fix, not an easier
# task -- and evaluation still reports against the unfiltered bank so the number stays comparable.
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

stage () {  # stage <name> <iters> <types> <probs> <resume-ck> [dataset-root]
  local name=$1 iters=$2 types=$3 probs=$4 ck=$5
  local DATA_DIR=${6:-./Datasets_ur10e_delto/OmniReset}
  echo "=== dataset root: $DATA_DIR ==="
  echo
  echo "=== STAGE $name : $iters iters, families $types ==="
  local resume=()
  [ -n "$ck" ] && [ -f "$ck" ] && { resume=(--resume_path "$ck"); echo "=== resuming from $ck ==="; }
  "$PY" scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$TASK" --num_envs "$ENVS" --max_iterations "$iters" --seed "$SEED" --headless \
    "${resume[@]}" --run_name "grip_$name" \
    env.events.reset_from_reset_states.params.dataset_dir="${DATA_DIR:-./Datasets_ur10e_delto/OmniReset}" \
    "env.actions.arm.motion_stiffness=[1000.0,1000.0,1000.0,3.0,3.0,3.0]" \
    "env.actions.arm.motion_damping_ratio=[1.0,1.0,1.0,0.1,0.1,0.1]" \
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

# Unlike job_stiff.sh, this job REQUIRES a resume. It continues a run already trained on the stiff
# arm rather than changing the plant, so the checkpoint is a genuine head start -- starting from
# scratch would discard the near-goal competence the ladder is meant to build on.
if [ -z "$CK0" ]; then
  echo "=== START_CK is required: this job continues an existing stiff-arm run ==="
  exit 1
fi
if [ ! -f "$CK0" ]; then
  echo "=== START_CK=$CK0 does not exist ==="
  exit 1
fi

NEAR=${NEAR_DIR:-./Datasets_ur10e_delto/OmniReset_seat150}
MID=${MID_DIR:-./Datasets_ur10e_delto/OmniReset_seat250}
FULL=${FULL_DIR:-./Datasets_ur10e_delto/OmniReset_seatall}

echo "=== ladder resuming from $CK0 ==="
CK=$CK0
stage B1 "${ITERS_B1:-200}" "[$C4,$C3]"         "[0.4,0.6]"             "$CK" "$NEAR" ; CK=$(newest_ck)
stage B2 "${ITERS_B2:-200}" "[$C4,$C3]"         "[0.4,0.6]"             "$CK" "$MID"  ; CK=$(newest_ck)
stage B3 "${ITERS_B3:-200}" "[$C4,$C3]"         "[0.4,0.6]"             "$CK" "$FULL" ; CK=$(newest_ck)
stage C  "${ITERS_C:-300}"  "[$C4,$C3,$C2]"     "[0.3,0.4,0.3]"         "$CK" "$FULL" ; CK=$(newest_ck)
stage D  "${ITERS_D:-900}"  "[$C4,$C3,$C2,$C1]" "[0.25,0.25,0.25,0.25]" "$CK" "$FULL"

echo "=== distance ladder done ==="
find "$RUNS" -name 'model_*.pt' -newermt '-1 day' -printf '%s %p\n' | sort -rn | head -8
