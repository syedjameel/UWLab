#!/usr/bin/env bash
# Record the four OmniReset families against the FIST grasp bank, then film them.
# The fist is what grasps on the corrected rig; the two-finger pinch banks nothing (see the report).
set -uo pipefail
cd /home/mattephi/Work/Rubitek/cube-stack/pinch
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
DS=Datasets_ur10e_delto/OmniReset_fist
PAIR=InsertiveCube34__ReceptiveCube34
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
export UWLAB_CUBE_DATASET_DIR="./$DS"
export UWLAB_DELTO_PINCH=0
export UWLAB_GRAVITY_COMP=1
mkdir -p "$REPO/$DS/Resets/$PAIR"
LOG=out/fistpipe.log
say(){ echo "FIST $* $(date -Is)" | tee -a "$LOG"; }

say C1_START
timeout 3600 ./run6.sh "$REPO/scripts_v2/tools/record_reset_states.py" \
    --task OmniReset-UR10eDelto-CubeStack-ObjectAnywhereEEAnywhere-v0 \
    --reset_type ObjectAnywhereEEAnywhere --num_envs 128 --num_reset_states 200 --headless \
    --dataset_dir "./$DS" env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 \
    > out/fist_c1.log 2>&1
rc=$?; say C1_END rc=$rc "$(stat -c%s "$REPO/$DS/Resets/$PAIR/resets_ObjectAnywhereEEAnywhere.pt" 2>/dev/null) bytes"

for T in ObjectAnywhereEEGrasped ObjectRestingEEGrasped ObjectPartiallyAssembledEEGrasped; do
  rm -f "$REPO/$DS/Resets/$PAIR/resets_$T.pt"
  say "${T}_START"
  timeout 5400 ./run6.sh "$REPO/scripts_v2/tools/record_reset_states.py" \
      --task "OmniReset-UR10eDelto-CubeStack-$T-v0" --reset_type "$T" \
      --num_envs 128 --num_reset_states 40 --headless --dataset_dir "./$DS" \
      env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 \
      > "out/fist_$T.log" 2>&1
  rc=$?; say "${T}_END" rc=$rc "$(stat -c%s "$REPO/$DS/Resets/$PAIR/resets_$T.pt" 2>/dev/null) bytes"
done

say FILMS_START
UWLAB_VIEW_EYE="1.55,-1.35,0.95" UWLAB_VIEW_AT="0.45,0.00,0.10" ./stage18.sh "./$DS" fist >> out/fistpipe_films.log 2>&1
rc=$?; say FILMS_END rc=$rc
echo "FISTPIPE_DONE $(date -Is)" | tee -a "$LOG"
