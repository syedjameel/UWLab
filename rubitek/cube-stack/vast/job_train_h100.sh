#!/usr/bin/env bash
# DELTO two-finger pinch cube-stack curriculum, on the h100-server.
#
# Derived from vast/job_train_grip.sh. Differences:
#   * WS is $HOME/rubitek, not /workspace -- this is a shared machine, not a container.
#   * CUDA_VISIBLE_DEVICES=1. GPU 0 carries another user's work and is not mine to fill.
#   * UWLAB_HAND_EFFORT_SCALE / UWLAB_TUCK_OTHERS come from the endurance probe, not from a guess.
#
# The four environment defects this run depends on, each measured on the C3 bank with a null action:
#
#   UWLAB_LATCHED_HAND -- RelativeJointPositionAction re-adds its offset to the LIVE MEASUREMENT on
#   every one of the 12 physics substeps, so a "0.05 rad bias" was a ramp: fingers travelled
#   +0.453 rad past the reset posture, crushed through the 34 mm cube and ejected it. Latching the
#   target instead takes in_hand at step 5 from 0.461 to 0.898.
#
#   UWLAB_OSC_GRAVITY_COMP / _SIGN -- the OSC computes tau = J^T f and nothing else while the arm
#   joints carry stiffness 0.0 / damping 0.0, so a zero action commanded ZERO torque and the arm
#   fell 114 mm in 12 s, riding any grasp into the table. The sign is measured: +1 made it 8x worse.
#
#   The grip bias itself -- without it, action 0 means "target := measurement", i.e. zero PD error
#   and zero grip force, and every *EEGrasped reset discarded its grasp one control step in.
#
#   The Carry reward variant -- adds the mid-range (std 0.08 m) goal-distance term, without which
#   the 200 mm -> 20 mm transport is steered on ~0.29 of episode reward against per-step velocity
#   and action-rate penalties, making "hold still" near-optimal over exactly the carry interval.
set -uo pipefail
WS=${WS:-$HOME/rubitek}
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
RUNS=logs/rsl_rl/ur10e_delto_omnireset_agent
TASK=${TASK:-OmniReset-UR10eDelto-CubeStackTwoFingerGripCarry-RelCartesianOSC-State-v0}
N=${N:-4096}; SEED=${SEED:-42}; NAME=${NAME:-h100_pinch}; ENTROPY=${ENTROPY:-0.006}
IT_A=${IT_A:-250}; IT_B=${IT_B:-900}; IT_C=${IT_C:-400}; IT_D=${IT_D:-0}
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
# $WS/compat FIRST and non-negotiable: it carries the pytorch3d shim that uwlab_tasks imports at
# module scope. Dropping it when this script was adapted from the vast.ai version cost a full
# curriculum -- both stages died in 30 seconds with ModuleNotFoundError and the box sat idle.
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-0.15
export UWLAB_LATCHED_HAND=1
export UWLAB_HAND_EFFORT_SCALE=${UWLAB_HAND_EFFORT_SCALE:-1}
export UWLAB_TUCK_OTHERS=${UWLAB_TUCK_OTHERS:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
# This box is SHARED. mdp/utils.get_temp_dir builds /tmp/uwlab/<uid>/..., and /tmp/uwlab here is
# already owned by another user's uid, so creating the per-uid subdirectory dies with
# PermissionError and takes the whole curriculum with it. UWLAB_TMP_ROOT exists for exactly this.
export UWLAB_TMP_ROOT=${UWLAB_TMP_ROOT:-$WS/tmp}
mkdir -p "$UWLAB_TMP_ROOT"
C1=ObjectAnywhereEEAnywhere; C2=ObjectRestingEEGrasped; C3=ObjectAnywhereEEGrasped; C4=ObjectPartiallyAssembledEEGrasped
say(){ echo "TRAIN $* $(date -Is)"; }
newest_ck(){ find "$RUNS" -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-; }

say "TASK=$TASK envs=$N effort=x$UWLAB_HAND_EFFORT_SCALE tuck=$UWLAB_TUCK_OTHERS"
# Every checkpoint on disk was trained while the hand ramped itself shut and the arm fell, against
# dynamics that no longer exist. Resuming would carry that damage forward.
FRESH=${FRESH:-1}
PUSHED=$(ls -t "$WS"/ck/model_*.pt 2>/dev/null | head -1)
if [ "$FRESH" = "1" ]; then
  CK=""; say "FRESH_START (ignoring ${PUSHED:-no checkpoint}) -- action semantics changed"
elif [ -n "$PUSHED" ]; then CK="$PUSHED"; say "RESUMING_FROM $PUSHED"; else CK=""; say "NO_CHECKPOINT starting from scratch"; fi

stage(){  # stage <name> <iters> <types> <probs> <resume-ck>
  local s=$1 iters=$2 types=$3 probs=$4 ck=$5
  [ "$iters" -le 0 ] && { say "STAGE_${s}_SKIPPED"; return 0; }
  local res=(); [ -n "$ck" ] && [ -f "$ck" ] && res=(--resume_path "$ck")
  say "STAGE_${s}_START iters=$iters families=$types envs=$N resume=${ck:-none}"
  "$PY" scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$TASK" \
    --num_envs "$N" --max_iterations "$iters" --seed "$SEED" --headless \
    --run_name "${NAME}_$s" "${res[@]}" \
    "env.events.reset_from_reset_states.params.reset_types=$types" \
    "env.events.reset_from_reset_states.params.probs=$probs" \
    agent.algorithm.entropy_coef="$ENTROPY" agent.save_interval=25 2>&1 | tr '\r' '\n' | grep -vE '^\[[0-9]'
  say "STAGE_${s}_END newest=$(newest_ck)"
}

# Only advance the checkpoint after a stage that actually trained: newest_ck on a skipped stage can
# hand the next stage the model_0.pt an aborted run leaves behind, which is random init.
advance_ck(){ local n; n=$(newest_ck); [ -n "$n" ] && CK="$n"; return 0; }

stage A "$IT_A" "[$C4]"             "[1.0]"                 "$CK"; [ "$IT_A" -gt 0 ] && advance_ck
stage B "$IT_B" "[$C4,$C3]"         "[0.5,0.5]"             "$CK"; [ "$IT_B" -gt 0 ] && advance_ck
stage C "$IT_C" "[$C4,$C3,$C2]"     "[0.34,0.33,0.33]"      "$CK"; [ "$IT_C" -gt 0 ] && advance_ck
stage D "$IT_D" "[$C4,$C3,$C2,$C1]" "[0.25,0.25,0.25,0.25]" "$CK"
say "CURRICULUM_DONE newest=$(newest_ck)"
