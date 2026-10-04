#!/usr/bin/env bash
# Stage 11: a real pinch grasp bank, then the three grasped reset families built from it.
# C1 does not involve the grasp, so the existing bank is reused rather than re-recorded.
set -uo pipefail
cd "$(dirname "$0")"; P=$PWD
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
DS="$REPO/Datasets_ur10e_delto/OmniReset_pinchon"
SRC="$REPO/Datasets_ur10e_delto/OmniReset"
PAIR=InsertiveCube34__ReceptiveCube34
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
export UWLAB_DELTO_PINCH=1
log(){ echo "=== $(date -Is) $*" | tee -a out/stage11.log; }

log "grasp bank, 300 grasps"
./run6.sh "$REPO/scripts_v2/tools/record_grasps.py" \
    --task OmniReset-Delto-GraspSampling-v0 --num_envs 64 --num_grasps 300 --headless \
    --dataset_dir "$DS" env.scene.object=cube34 > out/grasps_big.log 2>&1
echo "    rc=$? | $(grep -E 'BATCH_END' out/grasps_big.log | tail -1)" | tee -a out/stage11.log

# C1 and the partial assemblies are grasp-independent; reuse them so C2/C4 have their inputs.
mkdir -p "$DS/Resets/$PAIR"
cp -n "$SRC/Resets/$PAIR/resets_ObjectAnywhereEEAnywhere.pt" "$DS/Resets/$PAIR/" 2>/dev/null
cp -n "$SRC/Resets/$PAIR/partial_assemblies.pt" "$DS/Resets/$PAIR/" 2>/dev/null
log "reused C1 + partial_assemblies: $(ls "$DS/Resets/$PAIR" | tr '\n' ' ')"

for T in ObjectRestingEEGrasped ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped; do
  log "reset states $T"
  ./run6.sh "$REPO/scripts_v2/tools/record_reset_states.py" \
      --task "OmniReset-UR10eDelto-CubeStack-$T-v0" --reset_type "$T" \
      --num_envs 256 --num_reset_states 120 --headless --dataset_dir "$DS" \
      env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 \
      > "out/resets_$T.log" 2>&1
  echo "    rc=$? | $(grep -iE 'recorded|saved|states' "out/resets_$T.log" | tail -1)" | tee -a out/stage11.log
done
echo "STAGE11_DONE $(date -Is)" | tee -a out/stage11.log
