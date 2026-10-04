#!/usr/bin/env bash
# Rebuild everything on the UNFLIPPED rig, in dependency order.
#
# Every artefact produced before the (x,y,z,w) quaternion fix encodes a rig whose table stood
# upside down and whose arm was bolted on inverted, so nothing is reusable: not the pinch posture
# (its hold was measured with gravity pulling the wrong way through the hand), not the grasp bank,
# not C1, not the partial assemblies, not any reset family. This runs the lot, serially, because
# one Isaac Sim instance already fills the 16 GB card.
set -uo pipefail
cd /home/mattephi/Work/Rubitek/cube-stack/pinch
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
DS=Datasets_ur10e_delto/OmniReset_fixed
PAIR=InsertiveCube34__ReceptiveCube34
P=/home/mattephi/Work/Rubitek/cube-stack/pinch
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
export UWLAB_CUBE_DATASET_DIR="./$DS"
mkdir -p "$REPO/$DS/Resets/$PAIR" out/films
LOG=out/pipeline.log
POSE="${POSE:-$P/out/pinchhold_v2.json}"     # posture to build everything else from
SKIP_S1="${SKIP_S1:-}"                        # set to reuse an already-derived POSE
say(){ echo "PIPE $* $(date -Is)" | tee -a "$LOG"; }

# S1 -- the posture, re-derived against a hand that is the right way up.
if [ -z "$SKIP_S1" ]; then
say S1_START posture
timeout 5400 ./run6.sh "$P/pinchhold.py" --out "$POSE" > out/pipe_s1.log 2>&1
rc=$?; say S1_END rc=$rc "$(grep -a 'WINNER re-evaluated' out/pipe_s1.log | tail -1)"
else say S1_SKIP "reusing $POSE"; fi

# S2 -- the three posture-derived axes, then write posture + axes into the repo together.
say S2_START axes
./run6.sh "$P/pinchaxes.py" --pose "$POSE" --out "$P/out/pinchaxes_v2.json" > out/pipe_s2.log 2>&1
rc=$?; say S2_AXES rc=$rc
python3 "$P/applypose.py" --pose "$POSE" --axes "$P/out/pinchaxes_v2.json" >> out/pipe_s2.log 2>&1
rc=$?; say S2_END rc=$rc

# S3 -- grasp bank. UWLAB_DELTO_PINCH=1 makes the sampler close with the pinch, not the fist.
export UWLAB_DELTO_PINCH=1
say S3_START grasps
timeout 5400 ./run6.sh "$REPO/scripts_v2/tools/record_grasps.py" \
    --task OmniReset-Delto-GraspSampling-v0 --num_envs 64 --num_grasps 300 --headless \
    --dataset_dir "$REPO/$DS" env.scene.object=cube34 > out/pipe_s3.log 2>&1
rc=$?; say S3_END rc=$rc "$(grep -aE 'BATCH_END' out/pipe_s3.log | tail -1)"
[ "$rc" -eq 0 ] || { say PIPELINE_ABORT "grasp bank failed -- every downstream family consumes it"; exit 1; }

# S4 -- C1 and the partial assemblies. Neither depends on the grasp bank, both depend on the rig,
# so both are re-recorded rather than copied forward from the flipped tree.
say S4_START c1
timeout 3600 ./run6.sh "$REPO/scripts_v2/tools/record_reset_states.py" \
    --task OmniReset-UR10eDelto-CubeStack-ObjectAnywhereEEAnywhere-v0 \
    --reset_type ObjectAnywhereEEAnywhere --num_envs 128 --num_reset_states 200 --headless \
    --dataset_dir "./$DS" env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 \
    > out/pipe_s4.log 2>&1
rc=$?; say S4_END rc=$rc "$(stat -c%s "$REPO/$DS/Resets/$PAIR/resets_ObjectAnywhereEEAnywhere.pt" 2>/dev/null) bytes"

# S5 -- the grasped families. C3 first (cheapest), C2 last (the repo's own acceptance figure for
# it is 0.83%), so a slow C2 cannot starve the others.
export UWLAB_GRAVITY_COMP=1
for T in ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped ObjectRestingEEGrasped; do
  rm -f "$REPO/$DS/Resets/$PAIR/resets_$T.pt"
  say S5_START "$T"
  timeout 5400 ./run6.sh "$REPO/scripts_v2/tools/record_reset_states.py" \
      --task "OmniReset-UR10eDelto-CubeStack-$T-v0" --reset_type "$T" \
      --num_envs 128 --num_reset_states 40 --headless --dataset_dir "./$DS" \
      env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 \
      > "out/pipe_s5_$T.log" 2>&1
  rc=$?; say S5_END "$T" rc=$rc "$(stat -c%s "$REPO/$DS/Resets/$PAIR/resets_$T.pt" 2>/dev/null) bytes"
done

# S6 -- film all four, from a camera that is above the work surface rather than level with it.
say S6_START films
UWLAB_VIEW_EYE="1.55,-1.35,0.95" UWLAB_VIEW_AT="0.45,0.00,0.10" \
  ./stage18.sh "./$DS" fixed >> out/pipe_s6.log 2>&1
rc=$?; say S6_END rc=$rc
echo "PIPELINE_DONE $(date -Is)" | tee -a "$LOG"
