#!/usr/bin/env bash
# Smoke test of the DELTO two-finger pinch cube-stack training task on the QA'd pinch banks:
# does the env build, do all four reset banks load, are rewards/losses finite, what is the FPS.
set -uo pipefail
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
N=${N:-512}; IT=${IT:-30}
O=/home/mattephi/Work/Rubitek/cube-stack/train/out; mkdir -p $O
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 7000 ] || \
      [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -lt 7000 ]; do sleep 20; done
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
# train.py's proof-of-tree guard: this worktree's packages must be first on sys.path.
W=/home/mattephi/Work/Rubitek/UWLab-cubestack
export PYTHONPATH="$W/source/uwlab_tasks:$W/source/uwlab_assets:$W/source/uwlab:$W/source/uwlab_rl${PYTHONPATH:+:$PYTHONPATH}"
echo "SMOKE_START N=$N IT=$IT $(date -Is)" | tee $O/smoke_$N.log
timeout 2400 /home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-UR10eDelto-CubeStackTwoFinger-RelCartesianOSC-State-v0 --num_envs $N --headless \
  --max_iterations $IT --run_name pinch_smoke_$N >> $O/smoke_$N.log 2>&1
echo "SMOKE_DONE rc=$? $(date -Is)" | tee -a $O/smoke_$N.log
