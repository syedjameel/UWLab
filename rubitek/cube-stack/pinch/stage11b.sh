#!/usr/bin/env bash
# Stage 11b: the three grasped reset families, built from the PINCH grasp bank.
#
# 40 states, not 120. C2 is the expensive family -- this repo's own measurement is 0.83%
# acceptance (40 of 4799) -- and at 256 envs it was taking 5-10 minutes per ACCEPTED state,
# i.e. 11+ hours for 120. 128 envs also keeps warp off the OOM ceiling this box hit when two
# recorders overlapped. 40 per family is enough to film and to prove the pipeline; the shipped
# C2 bank has 103.
# Split from stage 11 because CUBE_DATASET_DIR had to become overridable first -- without
# UWLAB_CUBE_DATASET_DIR the reset recorders read the OLD five-finger grasp bank no matter what
# --dataset_dir says, and would have produced non-pinch states with nothing in the log saying so.
set -uo pipefail
cd "$(dirname "$0")"; P=$PWD
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
DS=./Datasets_ur10e_delto/OmniReset_pinchon
PAIR=InsertiveCube34__ReceptiveCube34
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
export UWLAB_DELTO_PINCH=1
export UWLAB_CUBE_DATASET_DIR="$DS"
# Per-condition acceptance counters. check_reset_state_success ANDs six conditions and reports
# only the conjunction, so a family that accepts nothing looks identical whether the grasp is
# slipping, the object drifted out of bounds, or the collision analyzer is vetoing everything.
# C3 sat at 0/40 for an hour without this.
export UWLAB_GRASP_DEBUG=1
log(){ echo "=== $(date -Is) $*" | tee -a out/stage11b.log; }

mkdir -p "$REPO/$DS/Resets/$PAIR"
cp -n "$REPO/Datasets_ur10e_delto/OmniReset/Resets/$PAIR/resets_ObjectAnywhereEEAnywhere.pt" "$REPO/$DS/Resets/$PAIR/" 2>/dev/null
cp -n "$REPO/Datasets_ur10e_delto/OmniReset/Resets/$PAIR/partial_assemblies.pt" "$REPO/$DS/Resets/$PAIR/" 2>/dev/null
log "inputs: $(ls "$REPO/$DS/Resets/$PAIR" | tr '\n' ' ') | grasps: $(ls "$REPO/$DS/Grasps/InsertiveCube34" 2>/dev/null | tr '\n' ' ')"

# C3 first, then C4, then C2. C2 is by far the slowest -- this repo measured 0.83% acceptance
# for it, and two runs here sat at 0/40 for six minutes -- so recording it first means nothing
# is verifiable for hours. C3 needs no other bank as input and C4 only the partial assemblies,
# so both can produce a checkable pinch bank long before C2 finishes.
for T in ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped ObjectRestingEEGrasped; do
  # Delete any stale output first. A killed run leaves a short, VALID-LOOKING .pt behind, and a
  # 3-state file recorded against the wrong grasp bank is indistinguishable from a good one
  # downstream -- it just quietly makes the bank wrong.
  rm -f "$REPO/$DS/Resets/$PAIR/resets_$T.pt"
  log "$T"
  ./run6.sh "$REPO/scripts_v2/tools/record_reset_states.py" \
      --task "OmniReset-UR10eDelto-CubeStack-$T-v0" --reset_type "$T" \
      --num_envs 128 --num_reset_states 40 --headless --dataset_dir "$DS" \
      env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 \
      > "out/resets_$T.log" 2>&1
  echo "    rc=$? | $(grep -ivE 'warning|^\[INFO\]' "out/resets_$T.log" | tail -2 | tr '\n' ' | ')" | tee -a out/stage11b.log
  # Verify it is actually made of grasps before anything downstream trusts it.
  if [ -f "$REPO/$DS/Resets/$PAIR/resets_$T.pt" ]; then
    ./run6.sh "$P/banklab.py" --bank "$REPO/$DS/Resets/$PAIR/resets_$T.pt" \
        --states 40 --settle-s 0.02 --verify-grasped > "out/verify_$T.log" 2>&1
    echo "    VERIFY rc=$? | $(grep -E '^VERIFY' "out/verify_$T.log" | tr '\n' ' ')" | tee -a out/stage11b.log
  fi
done
echo "STAGE11B_DONE $(date -Is)" | tee -a out/stage11b.log
