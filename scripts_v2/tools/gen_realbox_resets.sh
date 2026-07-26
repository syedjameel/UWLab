#!/usr/bin/env bash
# Generate the box-assembly reset-state datasets for the RealBox models (all 3 stage pairs).
#
# Inputs (arm-independent, live in ./Datasets/OmniReset):
#   Grasps/<Obj>/grasps.pt            - from record_grasps.py (bottom/mid/caprim)
#   Resets/<pair>/partial_assemblies.pt - recorded here first
# Outputs (UR10e-specific, consumed by training via LOCAL_UR10E_DATASET):
#   source/uwlab_assets/data/Datasets_ur10e/OmniReset/Resets/<pair>/resets_<type>.pt
#
# Order: partials first (OPAEG input), then AAEA for all pairs (fast, OREG input),
# then the grasped types (slow, yield-limited).
#
#   bash scripts_v2/tools/gen_realbox_resets.sh [num_envs]
set -u
cd "$(dirname "$0")/../.."
source env_uwlab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

NE=${1:-512}
GEN_DS=./Datasets/OmniReset
UR10E_DS=source/uwlab_assets/data/Datasets_ur10e/OmniReset
LOG_DIR=${LOG_DIR:-./logs/realbox_resets}
mkdir -p "$LOG_DIR"

# NOTE: no GPU cleanup between runs on purpose — other processes may own the GPU.
RUN() { # RUN <timeout_s> <log_name> <args...>
  local to=$1 log=$2; shift 2
  echo ">>> $(date '+%H:%M:%S') $log"
  timeout -s KILL "$to" ./uwlab.sh -p "$@" < /dev/null > "$LOG_DIR/$log.log" 2>&1
  echo "<<< exit $? $(date '+%H:%M:%S')"
  grep -E "Success rate|Successful reset|Successful partial|Total" "$LOG_DIR/$log.log" | tail -2
}

declare -A INS=( [A]=bottom [B]=mid [C]=caprim )
declare -A REC=( [A]=target [B]=bottom [C]=bottom )

for S in A B C; do
  OBJ="env.scene.insertive_object=${INS[$S]} env.scene.receptive_object=${REC[$S]}"
  RUN 1800 "partial_$S" scripts_v2/tools/record_partial_assemblies.py \
    --task OmniReset-PartialAssemblies-v0 --num_envs 32 --num_trajectories 32 --headless \
    --dataset_dir "$GEN_DS" $OBJ
done

for S in A B C; do
  OBJ="env.scene.insertive_object=${INS[$S]} env.scene.receptive_object=${REC[$S]}"
  RUN 3600 "aaea_$S" scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR10eLinearGripper-ObjectAnywhereEEAnywhere-v0 \
    --num_envs "$NE" --num_reset_states 3000 --headless --dataset_dir "$UR10E_DS" $OBJ
done

for S in A B C; do
  OBJ="env.scene.insertive_object=${INS[$S]} env.scene.receptive_object=${REC[$S]}"
  RUN 9000 "oaeg_$S" scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR10eLinearGripper-ObjectAnywhereEEGrasped-v0 \
    --num_envs "$NE" --num_reset_states 1000 --headless --dataset_dir "$UR10E_DS" $OBJ \
    env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir="$GEN_DS"
done

for S in A B C; do
  OBJ="env.scene.insertive_object=${INS[$S]} env.scene.receptive_object=${REC[$S]}"
  RUN 9000 "opaeg_$S" scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR10eLinearGripper-ObjectPartiallyAssembledEEGrasped-v0 \
    --num_envs "$NE" --num_reset_states 1000 --headless --dataset_dir "$UR10E_DS" $OBJ \
    env.events.reset_insertive_object_pose_from_partial_assembly_dataset.params.dataset_dir="$GEN_DS" \
    env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir="$GEN_DS"
done

for S in A B C; do
  OBJ="env.scene.insertive_object=${INS[$S]} env.scene.receptive_object=${REC[$S]}"
  RUN 9000 "oreg_$S" scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR10eLinearGripper-ObjectRestingEEGrasped-v0 \
    --num_envs "$NE" --num_reset_states 500 --headless --dataset_dir "$UR10E_DS" $OBJ \
    env.events.reset_insertive_object_pose_from_reset_states.params.dataset_dir="$UR10E_DS" \
    env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir="$GEN_DS"
done

echo "REALBOX_RESETS_DONE"
find "$UR10E_DS/Resets" -name "resets_*.pt" -exec ls -la {} \;
