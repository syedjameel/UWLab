#!/usr/bin/env bash
# Full grasp bank for the hold-optimised posture, then the three grasped reset families.
#
# The reset families run at the SHIPPED max_object_pos_deviation = 0.01. That threshold was the
# blocker for the previous posture, and the question of whether to relax it was going to be a
# judgement call. It no longer is: the pinch now holds 83.6% of the time under gravity and a
# 3 Hz shake, so whether ten millimetres is achievable is something to measure, not to decide.
set -uo pipefail
cd /home/mattephi/Work/Rubitek/cube-stack/pinch
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
DS=Datasets_ur10e_delto/OmniReset_hold
PAIR=InsertiveCube34__ReceptiveCube34
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
export UWLAB_DELTO_PINCH=1
export UWLAB_CUBE_DATASET_DIR="./$DS"
log(){ echo "=== $(date -Is) $*" | tee -a out/stage17.log; }

log "grasp bank, 300 grasps"
./run6.sh "$REPO/scripts_v2/tools/record_grasps.py" \
    --task OmniReset-Delto-GraspSampling-v0 --num_envs 64 --num_grasps 300 --headless \
    --dataset_dir "$REPO/$DS" env.scene.object=cube34 > out/grasps_hold.log 2>&1
log "grasp bank rc=$? | $(grep -E 'BATCH_END' out/grasps_hold.log | tail -1)"

mkdir -p "$REPO/$DS/Resets/$PAIR"
# C1 and the partial assemblies do not depend on the grasp bank, so they are reused rather than
# re-recorded. Everything else here does depend on it.
cp -n "$REPO/Datasets_ur10e_delto/OmniReset/Resets/$PAIR/resets_ObjectAnywhereEEAnywhere.pt" "$REPO/$DS/Resets/$PAIR/" 2>/dev/null
cp -n "$REPO/Datasets_ur10e_delto/OmniReset/Resets/$PAIR/partial_assemblies.pt" "$REPO/$DS/Resets/$PAIR/" 2>/dev/null
log "inputs: $(ls "$REPO/$DS/Resets/$PAIR" | tr '\n' ' ')"

# C3 first, then C4, then C2 -- C2 is by far the most expensive family (the repo's own figure is
# 0.83% acceptance), so it goes last where a slow run cannot starve the other two.
for T in ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped ObjectRestingEEGrasped; do
  rm -f "$REPO/$DS/Resets/$PAIR/resets_$T.pt"   # a killed run leaves a short, valid-looking file
  log "reset states $T"
  timeout 3600 ./run6.sh "$REPO/scripts_v2/tools/record_reset_states.py" \
      --task "OmniReset-UR10eDelto-CubeStack-$T-v0" --reset_type "$T" \
      --num_envs 128 --num_reset_states 40 --headless --dataset_dir "./$DS" \
      env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 \
      > "out/resets_hold_$T.log" 2>&1
  log "$T rc=$? | $(ls -la "$REPO/$DS/Resets/$PAIR/resets_$T.pt" 2>/dev/null | awk '{print $5}') bytes"
done
echo "STAGE17_DONE $(date -Is)" | tee -a out/stage17.log
