#!/usr/bin/env bash
# Stage 9: run the grasp sampler with the pinch posture enabled, into a SEPARATE dataset dir so
# nothing existing is overwritten. Baseline first (pinch off) so the yields are comparable.
set -uo pipefail
cd "$(dirname "$0")"; P=$PWD
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
log(){ echo "=== $(date -Is) $*" | tee -a out/stage9.log; }

run_sampler () {  # run_sampler <tag> <pinch 0|1>
  local tag="$1" pinch="$2"
  log "grasp sampler tag=$tag pinch=$pinch"
  UWLAB_DELTO_PINCH="$pinch" ./run6.sh "$REPO/scripts_v2/tools/record_grasps.py" \
      --task OmniReset-Delto-GraspSampling-v0 --num_envs 64 --num_grasps 40 --headless \
      --dataset_dir "$REPO/Datasets_ur10e_delto/OmniReset_$tag" \
      env.scene.object=cube34 > "out/grasps_$tag.log" 2>&1
  echo "    rc=$? $(grep -iE 'grasps|saved|recorded|WROTE' "out/grasps_$tag.log" | tail -3 | tr '\n' ' | ')" | tee -a out/stage9.log
}

run_sampler pinchon 1
run_sampler pinchoff 0
echo "STAGE9_DONE $(date -Is)" | tee -a out/stage9.log
