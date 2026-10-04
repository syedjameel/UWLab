#!/usr/bin/env bash
# DELTO two-finger pinch cube-stack training ON THE FIXED TASK ID.
#
# The previous run (job_train_curriculum.sh) trained
#   OmniReset-UR10eDelto-CubeStackTwoFinger-RelCartesianOSC-State-v0
# whose hand action is RelativeJointPositionAction with use_zero_offset=True, so
#   joint_position_target = measured_joint_pos + 0.1 * action
# and a zero action means "target := measurement": zero PD error, zero grip force. Every EEGrasped
# reset therefore threw the recorded grasp away one control step after the reset. Measured on the
# box: replaying the C3 bank with zero actions held the cube in 4.7% of envs (7.0% with friction
# pinned at 1.0 -- friction was never the cause), and the cube was gone by step 5 and never moved
# again. That is exactly why Metrics/task_1_success_rate sat at 0.0000 for 212 iterations while the
# near-goal family reached 0.38-0.48: near-goal never needed a grasp, its cube rests on the base cube.
#
# The repo already ships the fix as a sibling task id: _apply_grip_bias sets offset=0.05 rad on every
# driven hand joint with use_zero_offset=False, so raw action 0 means "hold a 0.05 rad squeeze" while
# the policy keeps the full [-0.1, +0.1] rad/step authority around it (opening stays available).
# Checkpoints transfer between a task and its Grip twin -- observation and action dims are unchanged.
#
# The Carry variant additionally adds the mid-range (std 0.08 m) goal-distance term, without which
# the transport between 200 mm and 20 mm is steered on ~0.29 of episode reward against per-step
# velocity and action-rate penalties -- i.e. holding still is near-optimal over exactly the interval
# where C3/C2 have to carry the cube.
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
[ -f "$WS/ISAACSIM_OK" ] || { echo "SKIP: /workspace/ISAACSIM_OK missing -- Isaac Sim provisioning not verified"; exit 1; }
RUNS=logs/rsl_rl/ur10e_delto_omnireset_agent
TASK=${TASK:-OmniReset-UR10eDelto-CubeStackTwoFingerGripCarry-RelCartesianOSC-State-v0}
N=${N:-2048}; SEED=${SEED:-42}; NAME=${NAME:-pinch_grip}; ENTROPY=${ENTROPY:-0.006}
# Stage A is now a real stage, not a re-adaptation: FRESH=1 starts from random init.
IT_A=${IT_A:-250}; IT_B=${IT_B:-600}; IT_C=${IT_C:-400}; IT_D=${IT_D:-1200}
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
# THE TWO FIXES THIS RUN DEPENDS ON, both measured on the C3 bank with a null action (128 envs):
#
#   UWLAB_OSC_GRAVITY_COMP / _SIGN -- the OSC computes tau = J^T f and nothing else, while the arm
#   joints carry stiffness 0.0 / damping 0.0, so a zero action commanded ZERO torque and the arm
#   fell 114 mm in 12 s, riding any grasp into the table. Adding PhysX's gravity term at scale
#   -0.15 holds the hand to within 20 mm. (Sign is measured: +1 made the fall 8x worse. The raw
#   accessor returns the gravity LOAD, not the correction, despite its name.)
#
#   UWLAB_LATCHED_HAND -- RelativeJointPositionAction re-adds its offset to the LIVE MEASUREMENT on
#   every one of the 12 physics substeps, so a "0.08 rad bias" was a ramp: fingers travelled
#   +0.453 rad past the reset posture, crushed through the 34 mm cube and ejected it (in_hand 0.461
#   at step 5 -> 0.000 by step 120, with film-level 0.9 N of force the whole time). Latching the
#   target instead: 0.867 at step 5, fingers settling at +0.044 rad.
export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-0.15
export UWLAB_LATCHED_HAND=1
C1=ObjectAnywhereEEAnywhere; C2=ObjectRestingEEGrasped; C3=ObjectAnywhereEEGrasped; C4=ObjectPartiallyAssembledEEGrasped
say(){ echo "TRAIN $* $(date -Is)"; }
newest_ck(){ find "$RUNS" -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-; }

say "TASK=$TASK"
# FRESH=1 by default: every existing checkpoint was trained while the hand ramped itself shut and
# the arm fell, so its hand policy was optimised against dynamics that no longer exist. Resuming
# would carry that damage forward. Set FRESH=0 to resume anyway.
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
    agent.algorithm.entropy_coef="$ENTROPY" agent.save_interval=50 2>&1 | tr '\r' '\n' | grep -vE '^\[[0-9]'
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
