#!/usr/bin/env bash
# Small filmed check of linear-gripper C2 under the jaw-closure in-hand check, before the full run.
set -uo pipefail
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
O=/home/mattephi/Work/Rubitek/cube-stack/linear/out/c2_check
SRC=Datasets_ur10e_linear/OmniReset
DS=Datasets_ur10e_linear/OmniReset_c2check
PAIR=InsertiveCube34__ReceptiveCube34
say(){ echo "C2CHECK $* $(date -Is)" | tee -a $O/check.log; }
rm -rf $DS; mkdir -p $DS/Grasps/InsertiveCube34 $DS/Resets/$PAIR
cp $SRC/Grasps/InsertiveCube34/grasps.pt $DS/Grasps/InsertiveCube34/
cp $SRC/Resets/$PAIR/resets_ObjectAnywhereEEAnywhere.pt $DS/Resets/$PAIR/
unset UWLAB_DELTO_PINCH
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 UWLAB_LINEAR_CUBE_DATASET_DIR="./$DS"
say RECORD_START
UWLAB_GRASP_DEBUG=1 timeout 2400 $R scripts_v2/tools/record_reset_states.py --task OmniReset-UR10eLinearGripper-CubeStack-ObjectRestingEEGrasped-v0 \
    --reset_type ObjectRestingEEGrasped --num_envs 128 --num_reset_states 6 --headless --dataset_dir "./$DS" > $O/record.log 2>&1
say RECORD_END rc=$? "$(stat -c%s $DS/Resets/$PAIR/resets_ObjectRestingEEGrasped.pt 2>/dev/null || echo MISSING)"
[ -f $DS/Resets/$PAIR/resets_ObjectRestingEEGrasped.pt ] || exit 1
say FILM_START
UWLAB_VIEW_SCRIPTED_HOLD=1 UWLAB_VIEW_EYE="1.55,-1.35,0.95" UWLAB_VIEW_AT="0.45,0.00,0.10" timeout 1500 $R scripts_v2/tools/render_reset_states_viewport.py \
    --reset-states $DS/Resets/$PAIR/resets_ObjectRestingEEGrasped.pt --task OmniReset-UR10eLinearGripper-CubeStack-ObjectRestingEEGrasped-v0 \
    --dataset-dir "./$DS" --insertive-variant cube34 --receptive-variant cube34 --out $O/film --count 6 --settle-steps 20 --repeats 1 \
    --video $O/c2_check.mp4 --video-freeze 14 --video-settle-steps 60 --video-every 3 > $O/film.log 2>&1
say FILM_END rc=$?
grep -aE "^  state |OPPOSED [0-9]" $O/film.log | tee -a $O/check.log
echo "C2CHECK_DONE $(date -Is)" | tee -a $O/check.log
