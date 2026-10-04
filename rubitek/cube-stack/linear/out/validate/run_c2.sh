#!/usr/bin/env bash
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
unset UWLAB_DELTO_PINCH
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 UWLAB_LINEAR_CUBE_DATASET_DIR=./Datasets_ur10e_linear/OmniReset
DS=Datasets_ur10e_linear/OmniReset; PAIR=InsertiveCube34__ReceptiveCube34
timeout 900 /home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh scripts_v2/tools/validate_grasped_bank.py \
  --reset-states $DS/Resets/$PAIR/resets_ObjectRestingEEGrasped.pt --task OmniReset-UR10eLinearGripper-CubeStack-ObjectRestingEEGrasped-v0 \
  --dataset-dir ./$DS --out /home/mattephi/Work/Rubitek/cube-stack/linear/out/validate/resets_ObjectRestingEEGrasped.validated.pt \
  > /home/mattephi/Work/Rubitek/cube-stack/linear/out/validate/c2.log 2>&1
echo "VALIDATE_DONE rc=$?" >> /home/mattephi/Work/Rubitek/cube-stack/linear/out/validate/c2.log
