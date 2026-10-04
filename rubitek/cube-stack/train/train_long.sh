#!/usr/bin/env bash
# DELTO two-finger pinch cube-stack PPO run, same script locally and on a rented GPU.
#   N=2048 IT=3000 NAME=pinch_a ./train_long.sh            fresh run
#   N=2048 IT=3000 NAME=pinch_b RESUME=<model.pt> ./train_long.sh
# REPO and PY can be overridden for a remote box.
set -uo pipefail
REPO=${REPO:-/home/mattephi/Work/Rubitek/UWLab-cubestack}
RUNNER=${RUNNER:-/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh}
OUT=${OUT:-/home/mattephi/Work/Rubitek/cube-stack/train/out}
N=${N:-2048}; IT=${IT:-3000}; NAME=${NAME:-pinch_long}; SEED=${SEED:-42}
MIN_GPU_MIB=${MIN_GPU_MIB:-9000}; MIN_RAM_MIB=${MIN_RAM_MIB:-9000}
cd "$REPO" || exit 1
mkdir -p "$OUT"
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt "$MIN_GPU_MIB" ] || \
      [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -lt "$MIN_RAM_MIB" ]; do sleep 30; done
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=${UWLAB_GPU_COLLISION_STACK_SIZE:-2147483648}
export UWLAB_CUBE_DATASET_DIR=${UWLAB_CUBE_DATASET_DIR:-./Datasets_ur10e_delto/OmniReset_pinch2f}
export PYTHONPATH="$REPO/source/uwlab_tasks:$REPO/source/uwlab_assets:$REPO/source/uwlab:$REPO/source/uwlab_rl${PYTHONPATH:+:$PYTHONPATH}"
RES=(); [ -n "${RESUME:-}" ] && RES=(--resume_path "$RESUME")
LOG="$OUT/train_$NAME.log"
echo "TRAIN_START name=$NAME N=$N IT=$IT seed=$SEED resume=${RESUME:-none} $(date -Is)" | tee "$LOG"
"$RUNNER" scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-UR10eDelto-CubeStackTwoFinger-RelCartesianOSC-State-v0 --num_envs "$N" --headless \
  --max_iterations "$IT" --seed "$SEED" --run_name "$NAME" "${RES[@]}" >> "$LOG" 2>&1
echo "TRAIN_DONE rc=$? $(date -Is)" | tee -a "$LOG"
