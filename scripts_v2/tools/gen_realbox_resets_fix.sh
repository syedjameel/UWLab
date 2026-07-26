#!/usr/bin/env bash
# Regenerate the *EEGrasped* reset datasets after three fixes. AAEA involves no grasp and is left
# on disk.
#   1. record_reset_states.py commands the gripper closed for every "EEGrasped" type.
#      ObjectPartiallyAssembledEEGrasped matched neither of the two hard-coded names and took the
#      random open/close branch, so 65-86% of its states had open jaws.
#   2. The UR10e-linear EEGrasped reset cfgs now set gripper_close_joint_max, so a state whose jaws
#      shut past 0.050 m (nothing between the pads) is rejected instead of banked.
#   3. Datasets/OmniReset/Grasps/CapRim/grasps.pt trimmed to the outer-wall mode.
#
# Stage-A OPAEG is time-capped rather than count-capped. Its partial-assembly poses hover a median
# 32 mm above the target, so the state is only valid while the gripper genuinely carries the 39 g
# tray in mid-air; that succeeds on ~8% of envs, against ~40% for Stage B whose object sits inside
# the tray cavity. The recorder flushes incrementally, so a killed run still leaves a usable
# dataset, and training auto-trims the reset-type mix to whatever is on disk.
#
#   bash scripts_v2/tools/gen_realbox_resets_fix.sh [num_envs]
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
  grep -aE "Success rate|Successful reset conditions|Total reset" "$LOG_DIR/$log.log" | tail -3
}

OBJ_C="env.scene.insertive_object=caprim env.scene.receptive_object=bottom"
OBJ_A="env.scene.insertive_object=bottom env.scene.receptive_object=target"

RUN 7200 "oaeg_C" scripts_v2/tools/record_reset_states.py \
  --task OmniReset-UR10eLinearGripper-ObjectAnywhereEEGrasped-v0 \
  --num_envs "$NE" --num_reset_states 1000 --headless --dataset_dir "$UR10E_DS" $OBJ_C \
  env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir="$GEN_DS"

RUN 5400 "opaeg_C" scripts_v2/tools/record_reset_states.py \
  --task OmniReset-UR10eLinearGripper-ObjectPartiallyAssembledEEGrasped-v0 \
  --num_envs "$NE" --num_reset_states 1000 --headless --dataset_dir "$UR10E_DS" $OBJ_C \
  env.events.reset_insertive_object_pose_from_partial_assembly_dataset.params.dataset_dir="$GEN_DS" \
  env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir="$GEN_DS"

RUN 3600 "opaeg_A" scripts_v2/tools/record_reset_states.py \
  --task OmniReset-UR10eLinearGripper-ObjectPartiallyAssembledEEGrasped-v0 \
  --num_envs "$NE" --num_reset_states 1000 --headless --dataset_dir "$UR10E_DS" $OBJ_A \
  env.events.reset_insertive_object_pose_from_partial_assembly_dataset.params.dataset_dir="$GEN_DS" \
  env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir="$GEN_DS"

echo "REALBOX_RESETS_FIX_DONE"
find "$UR10E_DS/Resets" -name "resets_*.pt" -exec ls -la {} \;
