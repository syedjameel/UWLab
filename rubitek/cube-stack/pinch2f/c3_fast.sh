#!/usr/bin/env bash
# Small filmed check of pinch C3 with the robust acceptance (grasp-centre box, arm pinned), before scaling.
set -uo pipefail
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
O=/home/mattephi/Work/Rubitek/cube-stack/pinch2f/out/c3_fast
SRC=Datasets_ur10e_delto/OmniReset_pinch2f
DS=Datasets_ur10e_delto/OmniReset_pinch2f_c3fast
PAIR=InsertiveCube34__ReceptiveCube34
T=OmniReset-UR10eDelto-CubeStack-ObjectAnywhereEEGrasped-v0
mkdir -p $O; say(){ echo "P3FAST $* $(date -Is)" | tee -a $O/check.log; }
until ! ps -eo args | grep -q "[c]3_refilm.sh"; do sleep 15; done
rm -rf $DS; mkdir -p $DS/Grasps/InsertiveCube34 $DS/Resets/$PAIR
cp $SRC/Grasps/InsertiveCube34/grasps.pt $DS/Grasps/InsertiveCube34/
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_PIN_ARM=1 UWLAB_HOLD_OBJECT_STEPS=8
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 UWLAB_CUBE_DATASET_DIR="./$DS"
say RECORD_START
UWLAB_GRASP_DEBUG=1 timeout 900 $R scripts_v2/tools/record_reset_states.py --task $T \
    --reset_type ObjectAnywhereEEGrasped --num_envs 128 --num_reset_states 6 --headless --dataset_dir "./$DS" > $O/record.log 2>&1
say RECORD_END rc=$? "$(grep -ao '[0-9]*/6 \[[0-9:]*' $O/record.log | tail -1)"
[ -f $DS/Resets/$PAIR/resets_ObjectAnywhereEEGrasped.pt ] || exit 1
say FILM_START
UWLAB_VIEW_SCRIPTED_HOLD=1 UWLAB_VIEW_EYE="1.55,-1.35,0.95" UWLAB_VIEW_AT="0.45,0.00,0.10" timeout 900 $R scripts_v2/tools/render_reset_states_viewport.py \
    --reset-states $DS/Resets/$PAIR/resets_ObjectAnywhereEEGrasped.pt --task $T \
    --dataset-dir "./$DS" --insertive-variant cube34 --receptive-variant cube34 --out $O/film --count 6 --settle-steps 60 --repeats 1 \
    --video $O/c3_fast.mp4 --video-freeze 14 --video-settle-steps 60 --video-every 3 > $O/film.log 2>&1
say FILM_END rc=$?
grep -aE "^  state |OPPOSED [0-9]" $O/film.log | tee -a $O/check.log
echo "P3FAST_DONE $(date -Is)" | tee -a $O/check.log
