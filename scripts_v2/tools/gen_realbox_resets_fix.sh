#!/usr/bin/env bash
# Regenerate the six *EEGrasped* reset datasets after three fixes. AAEA (all pairs) involves no
# grasp and is left on disk.
#   1. record_reset_states.py commands the gripper closed for every "EEGrasped" type.
#      ObjectPartiallyAssembledEEGrasped matched neither of the two hard-coded names and took the
#      random open/close branch, so 65-86% of its states had open jaws.       -> opaeg A/B/C
#   2. The UR10e-linear EEGrasped reset cfgs now set gripper_close_joint_max, so a state whose jaws
#      shut past 0.050 m (nothing between the pads) is rejected instead of banked. Without it 68%
#      of Bottom__CapRim grasped states held nothing.                         -> oaeg A/B/C
#   3. Datasets/OmniReset/Grasps/CapRim/grasps.pt trimmed to the outer-wall mode (the louver-band
#      grasps clamp two 0.54 mm shells and eject the cap).                    -> oaeg/opaeg C
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

declare -A INS=( [A]=bottom [B]=mid [C]=caprim )
declare -A REC=( [A]=target [B]=bottom [C]=bottom )

# Stage C is the slow one (the cap is light and ejects easily, so most candidates are now rejected)
# and B is the fastest, so walk the pairs cheapest-first and do both types per pair before moving on.
# That way a bad reset type shows up in minutes instead of behind the hour-long Stage-C run.
for S in B A C; do
  OBJ="env.scene.insertive_object=${INS[$S]} env.scene.receptive_object=${REC[$S]}"
  RUN 14400 "oaeg_$S" scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR10eLinearGripper-ObjectAnywhereEEGrasped-v0 \
    --num_envs "$NE" --num_reset_states 1000 --headless --dataset_dir "$UR10E_DS" $OBJ \
    env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir="$GEN_DS"

  RUN 14400 "opaeg_$S" scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR10eLinearGripper-ObjectPartiallyAssembledEEGrasped-v0 \
    --num_envs "$NE" --num_reset_states 1000 --headless --dataset_dir "$UR10E_DS" $OBJ \
    env.events.reset_insertive_object_pose_from_partial_assembly_dataset.params.dataset_dir="$GEN_DS" \
    env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir="$GEN_DS"
done

echo "REALBOX_RESETS_FIX_DONE"
find "$UR10E_DS/Resets" -name "resets_*.pt" -exec ls -la {} \;
