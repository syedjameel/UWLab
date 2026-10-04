#!/usr/bin/env bash
# Sweep arm PD gains and read the acceptance gate's two VELOCITY conditions.
# Baseline (shipped, stiffness=damping=0): not_abnormal 32/128, stable 29/128, success 0.
# Gains 2000/200 made both WORSE (16/14), which is the signature of a stiff PD oscillating at
# dt=1/120 rather than of a missing command -- so walk the gains down and measure.
set -uo pipefail
cd /home/mattephi/Work/Rubitek/cube-stack/pinch
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
export UWLAB_DELTO_PINCH=1
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_hold
export UWLAB_GRASP_DEBUG=1
for KD in "60 12" "200 40" "600 100"; do
  set -- $KD
  export UWLAB_ARM_STIFFNESS=$1 UWLAB_ARM_DAMPING=$2
  echo "=== stiffness=$1 damping=$2" | tee -a out/sweepgain.log
  timeout 240 ./run6.sh "$REPO/scripts_v2/tools/record_reset_states.py" \
      --task "OmniReset-UR10eDelto-CubeStack-ObjectAnywhereEEGrasped-v0" \
      --reset_type ObjectAnywhereEEGrasped --num_envs 128 --num_reset_states 40 --headless \
      --dataset_dir ./Datasets_ur10e_delto/OmniReset_sweep \
      env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 \
      > "out/sweep_$1.log" 2>&1
  grep -ao "reset dbg\][^\\n]\{0,120\}" "out/sweep_$1.log" | head -3 | tee -a out/sweepgain.log
done
echo "SWEEP_DONE" | tee -a out/sweepgain.log
