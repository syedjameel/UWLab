#!/usr/bin/env bash
# DELTO two-finger pinch cube-stack training on the rented box: STAGED BACKWARDS CURRICULUM.
#
# The flat 25%-per-family mixture plateaued locally (1224 iters: success ~0.13, per-family C4 ~0.48
# and C3/C2/C1 exactly 0.000) because three quarters of every batch went to families the policy
# cannot reach yet. Each stage here adds the next family backwards along the task and resumes the
# previous stage's weights.
#
# RESUME: if /workspace/ck/model_*.pt exists (pushed from the local run, which already learned
# near-goal), stage A is skipped and stage B resumes from it.
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
# Refuse to burn rented time on a half-provisioned box: job_fix_isaacsim.sh leaves this marker.
[ -f "$WS/ISAACSIM_OK" ] || { echo "SKIP: /workspace/ISAACSIM_OK missing -- Isaac Sim provisioning not verified"; exit 1; }
RUNS=logs/rsl_rl/ur10e_delto_omnireset_agent
N=${N:-2048}; SEED=${SEED:-42}; NAME=${NAME:-pinch_curr_remote}; ENTROPY=${ENTROPY:-0.006}
IT_A=${IT_A:-140}; IT_B=${IT_B:-400}; IT_C=${IT_C:-500}; IT_D=${IT_D:-2000}
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
# $WS/compat first: uwlab_is6_bridge + the pytorch3d shim uwlab_tasks imports.
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
C1=ObjectAnywhereEEAnywhere; C2=ObjectRestingEEGrasped; C3=ObjectAnywhereEEGrasped; C4=ObjectPartiallyAssembledEEGrasped
say(){ echo "TRAIN $* $(date -Is)"; }
newest_ck(){ find "$RUNS" -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-; }

PUSHED=$(ls -t "$WS"/ck/model_*.pt 2>/dev/null | head -1)
if [ -n "$PUSHED" ]; then IT_A=0; CK="$PUSHED"; say "RESUMING_FROM $PUSHED (stage A skipped)"; else CK=""; fi

stage(){  # stage <name> <iters> <types> <probs> <resume-ck>
  local s=$1 iters=$2 types=$3 probs=$4 ck=$5
  [ "$iters" -le 0 ] && { say "STAGE_${s}_SKIPPED"; return 0; }
  local res=(); [ -n "$ck" ] && [ -f "$ck" ] && res=(--resume_path "$ck")
  say "STAGE_${s}_START iters=$iters families=$types envs=$N resume=${ck:-none}"
  "$PY" scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-UR10eDelto-CubeStackTwoFinger-RelCartesianOSC-State-v0 \
    --num_envs "$N" --max_iterations "$iters" --seed "$SEED" --headless \
    --run_name "${NAME}_$s" "${res[@]}" \
    "env.events.reset_from_reset_states.params.reset_types=$types" \
    "env.events.reset_from_reset_states.params.probs=$probs" \
    agent.algorithm.entropy_coef="$ENTROPY" agent.save_interval=50 2>&1 | tr '\r' '\n' | grep -vE '^\[[0-9]'
  say "STAGE_${s}_END newest=$(newest_ck)"
}

# Keep the previous checkpoint when newest_ck finds nothing: the code rsync excludes logs/, so on a
# fresh box $RUNS is empty and an unguarded `CK=$(newest_ck)` wiped the PUSHED checkpoint path --
# stage B then trained from scratch while the log still said "RESUMING_FROM ...".
advance_ck(){ local n; n=$(newest_ck); [ -n "$n" ] && CK="$n"; return 0; }

# Advance ONLY after a stage that actually trained. A skipped stage (iters=0) must not advance:
# newest_ck would then hand the next stage whatever model_*.pt happens to be newest -- including the
# model_0.pt an aborted run leaves behind, which is random init. That silently discarded the pushed
# checkpoint twice.
stage A "$IT_A" "[$C4]"             "[1.0]"                 "$CK"; [ "$IT_A" -gt 0 ] && advance_ck
stage B "$IT_B" "[$C4,$C3]"         "[0.5,0.5]"             "$CK"; [ "$IT_B" -gt 0 ] && advance_ck
stage C "$IT_C" "[$C4,$C3,$C2]"     "[0.34,0.33,0.33]"      "$CK"; [ "$IT_C" -gt 0 ] && advance_ck
stage D "$IT_D" "[$C4,$C3,$C2,$C1]" "[0.25,0.25,0.25,0.25]" "$CK"
say "CURRICULUM_DONE newest=$(newest_ck)"
