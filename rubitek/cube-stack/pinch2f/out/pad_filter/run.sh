#!/usr/bin/env bash
O=/home/mattephi/Work/Rubitek/cube-stack/pinch2f/out/pad_filter
M=/home/mattephi/Work/Rubitek/cube-stack/pinch2f/out/pad_measure
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
until grep -qa "MEASURE2_RC=" $M/grasps2.log; do sleep 10; done
sleep 5
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
DS=Datasets_ur10e_delto/OmniReset_pinch2f_padfilter
rm -rf $DS; mkdir -p $DS
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
echo "STEP grasps $(date -Is)" >> $O/run.log
UWLAB_PINCH_MAX_PAD_OFFSET=0.007 timeout 1800 $R scripts_v2/tools/record_grasps.py --task OmniReset-Delto-GraspSampling-v0 --num_envs 64 --num_grasps 16 --headless --dataset_dir /home/mattephi/Work/Rubitek/UWLab-cubestack/$DS env.scene.object=cube34 > $O/grasps.log 2>&1
echo "GRASPS_RC=$? $(stat -c%s $DS/Grasps/InsertiveCube34/grasps.pt 2>/dev/null || echo MISSING)" >> $O/run.log
[ -f $DS/Grasps/InsertiveCube34/grasps.pt ] || exit 1
echo "STEP replay_film $(date -Is)" >> $O/run.log
UWLAB_GRASP_DEBUG=1 UWLAB_CUBE_DATASET_DIR=./$DS timeout 900 $R scripts_v2/tools/film_grasp_sampling.py --task OmniReset-UR10eDelto-CubeStack-ObjectAnywhereEEGrasped-v0 --num_envs 1 --episodes 4 --max-steps 80 --world-camera --eye 1.05 -0.60 0.70 --lookat 0.45 0.0 0.20 --out $O/replay.mp4 > $O/replay.log 2>&1
echo "REPLAY_RC=$?" >> $O/run.log
echo "PADFILTER_DONE $(date -Is)" >> $O/run.log
