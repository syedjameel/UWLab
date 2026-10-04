#!/usr/bin/env bash
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
O=/home/mattephi/Work/Rubitek/cube-stack/pinch2f/out/pad_measure
UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 timeout 1500 /home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh scripts_v2/tools/record_grasps.py --task OmniReset-Delto-GraspSampling-v0 --num_envs 64 --num_grasps 64 --headless --dataset_dir $O/ds env.scene.object=cube34 > $O/grasps.log 2>&1
echo "MEASURE_RC=$?" >> $O/grasps.log
