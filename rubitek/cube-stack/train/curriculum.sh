#!/usr/bin/env bash
# DELTO two-finger pinch cube-stack, STAGED BACKWARDS CURRICULUM.
#
# WHY NOT THE FLAT MIXTURE. Run pinch_local_a trained 1224 iterations with all four reset families at
# 25% each: end-of-episode success plateaued at ~0.13 (mean of the last 400 iterations 0.132 vs 0.130
# in the 400 before), and the per-family split never moved off near-goal -- C4 ~0.48, C3/C2/C1 exactly
# 0.000. Three quarters of every batch was spent on families the policy cannot reach yet.
#
# Each stage adds the next family backwards along the task and resumes the previous stage's weights.
# Stage lengths follow cube-stack/vast/job_twofinger.sh, which set them from measured curves: near-goal
# saturates fast, so the budget belongs at the far end where the task is actually unsolved.
#
#   N=1024 IT_A=0 RESUME=<ck> NAME=pinch_curr ./curriculum.sh
# IT_A=0 skips stage A (use when resuming a run that already learned near-goal).
set -uo pipefail
REPO=${REPO:-/home/mattephi/Work/Rubitek/UWLab-cubestack}
RUNNER=${RUNNER:-/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh}
OUT=${OUT:-/home/mattephi/Work/Rubitek/cube-stack/train/out}
RUNS=${RUNS:-$REPO/logs/rsl_rl/ur10e_delto_omnireset_agent}
N=${N:-1024}; SEED=${SEED:-42}; NAME=${NAME:-pinch_curr}
ENTROPY=${ENTROPY:-0.006}
IT_A=${IT_A:-140}; IT_B=${IT_B:-400}; IT_C=${IT_C:-500}; IT_D=${IT_D:-1500}
MIN_GPU_MIB=${MIN_GPU_MIB:-6000}; MIN_RAM_MIB=${MIN_RAM_MIB:-7000}
cd "$REPO" || exit 1
mkdir -p "$OUT"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=${UWLAB_GPU_COLLISION_STACK_SIZE:-2147483648}
export UWLAB_CUBE_DATASET_DIR=${UWLAB_CUBE_DATASET_DIR:-./Datasets_ur10e_delto/OmniReset_pinch2f}
export PYTHONPATH="$REPO/source/uwlab_tasks:$REPO/source/uwlab_assets:$REPO/source/uwlab:$REPO/source/uwlab_rl${PYTHONPATH:+:$PYTHONPATH}"
LOG="$OUT/curriculum_$NAME.log"
C1=ObjectAnywhereEEAnywhere; C2=ObjectRestingEEGrasped; C3=ObjectAnywhereEEGrasped; C4=ObjectPartiallyAssembledEEGrasped
say(){ echo "CURR $* $(date -Is)" | tee -a "$LOG"; }
newest_ck(){ find "$RUNS" -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-; }
wait_mem(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt "$MIN_GPU_MIB" ] || \
                  [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -lt "$MIN_RAM_MIB" ]; do sleep 30; done; }

stage(){  # stage <name> <iters> <types> <probs> <resume-ck>
  local s=$1 iters=$2 types=$3 probs=$4 ck=$5
  [ "$iters" -le 0 ] && { say "STAGE_${s}_SKIPPED"; return 0; }
  local res=(); [ -n "$ck" ] && [ -f "$ck" ] && res=(--resume_path "$ck")
  wait_mem
  say "STAGE_${s}_START iters=$iters families=$types resume=${ck:-none}"
  "$RUNNER" scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-UR10eDelto-CubeStackTwoFinger-RelCartesianOSC-State-v0 \
    --num_envs "$N" --max_iterations "$iters" --seed "$SEED" --headless \
    --run_name "${NAME}_$s" "${res[@]}" \
    "env.events.reset_from_reset_states.params.reset_types=$types" \
    "env.events.reset_from_reset_states.params.probs=$probs" \
    agent.algorithm.entropy_coef="$ENTROPY" agent.save_interval=50 >> "$LOG" 2>&1
  say "STAGE_${s}_END rc=$? newest=$(newest_ck)"
}

CK=${RESUME:-}
stage A "$IT_A" "[$C4]"             "[1.0]"                 "$CK"; CK=$(newest_ck)
stage B "$IT_B" "[$C4,$C3]"         "[0.5,0.5]"             "$CK"; CK=$(newest_ck)
stage C "$IT_C" "[$C4,$C3,$C2]"     "[0.34,0.33,0.33]"      "$CK"; CK=$(newest_ck)
stage D "$IT_D" "[$C4,$C3,$C2,$C1]" "[0.25,0.25,0.25,0.25]" "$CK"
say "CURRICULUM_DONE newest=$(newest_ck)"
