#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"; P=$PWD
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
echo "=== $(date -Is) grasp sampler, pinch ON" | tee -a out/stage10.log
UWLAB_DELTO_PINCH=1 ./run6.sh "$REPO/scripts_v2/tools/record_grasps.py" \
    --task OmniReset-Delto-GraspSampling-v0 --num_envs 64 --num_grasps 40 --headless \
    --dataset_dir "$REPO/Datasets_ur10e_delto/OmniReset_pinchon" \
    env.scene.object=cube34 > out/grasps_pinchon.log 2>&1
echo "    rc=$? | $(grep -E 'placement collision gate rejected' out/grasps_pinchon.log | tail -1)" | tee -a out/stage10.log
echo "STAGE10_DONE $(date -Is)" | tee -a out/stage10.log
