#!/usr/bin/env bash
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
O=/home/mattephi/Work/Rubitek/cube-stack/pinch2f/out/c3_check
UWLAB_GRASP_DEBUG=1 UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f_c3check timeout 900 /home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh scripts_v2/tools/film_grasp_sampling.py --task OmniReset-UR10eDelto-CubeStack-ObjectAnywhereEEGrasped-v0 --num_envs 4 --episodes 3 --max-steps 120 --eye 0.95 -0.55 0.60 --lookat 0.45 0.0 0.15 --out $O/c3_record_episode2.mp4 > $O/c3_record_episode2.log 2>&1
echo "REFILM_RC=$?" >> $O/c3_record_episode2.log
