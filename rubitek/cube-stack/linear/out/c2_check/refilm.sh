#!/usr/bin/env bash
set -uo pipefail
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
O=/home/mattephi/Work/Rubitek/cube-stack/linear/out/c2_check
DS=Datasets_ur10e_linear/OmniReset_c2check
PAIR=InsertiveCube34__ReceptiveCube34
unset UWLAB_DELTO_PINCH
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 UWLAB_LINEAR_CUBE_DATASET_DIR="./$DS"
rm -rf $O/film2 $O/c2_check2.mp4
UWLAB_VIEW_SCRIPTED_HOLD=1 UWLAB_VIEW_EYE="1.55,-1.35,0.95" UWLAB_VIEW_AT="0.45,0.00,0.10" timeout 1500 $R scripts_v2/tools/render_reset_states_viewport.py \
    --reset-states $DS/Resets/$PAIR/resets_ObjectRestingEEGrasped.pt --task OmniReset-UR10eLinearGripper-CubeStack-ObjectRestingEEGrasped-v0 \
    --dataset-dir "./$DS" --insertive-variant cube34 --receptive-variant cube34 --out $O/film2 --count 6 --settle-steps 60 --repeats 1 \
    --video $O/c2_check2.mp4 --video-freeze 14 --video-settle-steps 60 --video-every 3 > $O/film2.log 2>&1
echo "REFILM_DONE rc=$? $(date -Is)" >> $O/film2.log
