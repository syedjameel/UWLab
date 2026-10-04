#!/usr/bin/env bash
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
O=/home/mattephi/Work/Rubitek/cube-stack/pinch2f/out/pad_filter
UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f_padfilter timeout 900 /home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh scripts_v2/tools/film_grasp_sampling.py --task OmniReset-UR10eDelto-CubeStack-ObjectAnywhereEEGrasped-v0 --num_envs 1 --episodes 3 --max-steps 40 --hold-object-steps 8 --world-camera --eye 1.05 -0.60 0.70 --lookat 0.45 0.0 0.20 --out $O/hold8.mp4 > $O/hold8.log 2>&1
echo "HOLD8_RC=$?" >> $O/hold8.log
