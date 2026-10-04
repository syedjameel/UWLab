#!/usr/bin/env bash
# LOCAL fallback for the pinch curriculum, on the 4070 Ti.
#
# Exists because the whole path to the H100 runs through a ~10 KB/s tailnet link: if the transfer or
# the remote provisioning fails, the night should not be lost waiting for it. The local box already
# has the full Isaac Sim 6 stack (.venv-is6), the banks, and a GPU that renders -- it is slower, not
# less capable.
#
# Sized for 16 GB: 1024 envs against the H100 job's 4096, and a 1 GB collision stack rather than 2 GB.
# Expect roughly a quarter of the H100's iteration rate, so stage budgets are cut to match and the
# curriculum stops after stage B -- getting C3 off zero is the result that matters tonight.
#
# The env vars below are the four measured fixes plus the effort scale the endurance probe selected.
# See train_pinch_local.sh's H100 twin (cube-stack/vast/job_train_h100.sh) for what each one repairs.
set -uo pipefail
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv-is6/bin/python
cd "$REPO" || exit 1
RUNS=logs/rsl_rl/ur10e_delto_omnireset_agent
TASK=${TASK:-OmniReset-UR10eDelto-CubeStackTwoFingerGripCarry-RelCartesianOSC-State-v0}
N=${N:-1024}; SEED=${SEED:-42}; NAME=${NAME:-local_pinch}; ENTROPY=${ENTROPY:-0.006}
IT_A=${IT_A:-200}; IT_B=${IT_B:-700}

export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="/home/mattephi/Work/Rubitek/uwlab-env/compat:$REPO/source/uwlab_assets:$REPO/source/uwlab_tasks:$REPO/source/uwlab:$REPO/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-0.15
export UWLAB_LATCHED_HAND=1
export UWLAB_HAND_EFFORT_SCALE=${UWLAB_HAND_EFFORT_SCALE:-3}
export UWLAB_TUCK_OTHERS=${UWLAB_TUCK_OTHERS:-0}
export UWLAB_GRIP_BIAS_RAD=${UWLAB_GRIP_BIAS_RAD:-0.05}

C3=ObjectAnywhereEEGrasped; C4=ObjectPartiallyAssembledEEGrasped
say(){ echo "TRAIN $* $(date -Is)"; }
newest_ck(){ find "$RUNS" -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-; }

say "LOCAL TASK=$TASK envs=$N effort=x$UWLAB_HAND_EFFORT_SCALE tuck=$UWLAB_TUCK_OTHERS"

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

CK=""
stage A "$IT_A" "[$C4]"     "[1.0]"     "$CK"
n=$(newest_ck); [ -n "$n" ] && CK="$n"
stage B "$IT_B" "[$C4,$C3]" "[0.5,0.5]" "$CK"
say "LOCAL_CURRICULUM_DONE newest=$(newest_ck)"
