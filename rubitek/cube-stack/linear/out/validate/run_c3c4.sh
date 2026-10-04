#!/usr/bin/env bash
# Replay-validate the linear C3 and C4 banks (all states), after the C2 run agreed with the films.
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
unset UWLAB_DELTO_PINCH
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 UWLAB_LINEAR_CUBE_DATASET_DIR=./Datasets_ur10e_linear/OmniReset
DS=Datasets_ur10e_linear/OmniReset; PAIR=InsertiveCube34__ReceptiveCube34
O=/home/mattephi/Work/Rubitek/cube-stack/linear/out/validate
for T in ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped; do
  timeout 900 /home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh scripts_v2/tools/validate_grasped_bank.py \
    --reset-states $DS/Resets/$PAIR/resets_$T.pt --task OmniReset-UR10eLinearGripper-CubeStack-$T-v0 \
    --dataset-dir ./$DS --out $O/resets_$T.validated.pt > $O/$T.log 2>&1
  echo "VALIDATE_DONE rc=$?" >> $O/$T.log
done
echo ALL_DONE >> $O/c3c4.done
