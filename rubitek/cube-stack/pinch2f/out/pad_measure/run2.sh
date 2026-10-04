#!/usr/bin/env bash
O=/home/mattephi/Work/Rubitek/cube-stack/pinch2f/out/pad_measure
until grep -qa "MEASURE_RC=" $O/grasps.log; do sleep 10; done
sleep 5
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
rm -rf $O/ds2
UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 timeout 1500 /home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh scripts_v2/tools/record_grasps.py --task OmniReset-Delto-GraspSampling-v0 --num_envs 64 --num_grasps 64 --headless --dataset_dir $O/ds2 env.scene.object=cube34 > $O/grasps2.log 2>&1
echo "MEASURE2_RC=$?" >> $O/grasps2.log
