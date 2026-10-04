#!/usr/bin/env bash
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
DS=Datasets_ur10e_delto/OmniReset_pinch2f; PAIR=InsertiveCube34__ReceptiveCube34
export UWLAB_CUBE_DATASET_DIR=./$DS
O=/home/mattephi/Work/Rubitek/cube-stack/pinch2f/out/validate
while [ "$(awk "/MemAvailable/ {print int(\$2/1024)}" /proc/meminfo)" -lt 6000 ]; do sleep 20; done
timeout 900 /home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh scripts_v2/tools/validate_grasped_bank.py \
  --reset-states $DS/Resets/$PAIR/resets_ObjectAnywhereEEGrasped.unfiltered.pt --task OmniReset-UR10eDelto-CubeStack-ObjectAnywhereEEGrasped-v0 \
  --dataset-dir ./$DS --margin 0.012 --out $O/c3_contact.pt > $O/c3_contact.log 2>&1
echo "VALIDATE_DONE rc=$?" >> $O/c3_contact.log
