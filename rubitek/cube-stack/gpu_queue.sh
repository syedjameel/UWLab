#!/usr/bin/env bash
# Remaining GPU work, run when the card has room (another project's job may hold most of it).
#  1. re-record the linear grasped families under the new in-hand acceptance check
#  2. film the linear banks
#  3. the DELTO two-finger pinch pipeline (grasps, PA, C2-C4 under the same check, films)
set -uo pipefail
cd /home/mattephi/Work/Rubitek/cube-stack
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
DS=Datasets_ur10e_linear/OmniReset
PAIR=InsertiveCube34__ReceptiveCube34
LOG=/home/mattephi/Work/Rubitek/cube-stack/gpu_queue.log
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 UWLAB_LINEAR_CUBE_DATASET_DIR="./$DS"
say(){ echo "Q $* $(date -Is)" | tee -a "$LOG"; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
ram_avail_mib(){ awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
# An Isaac instance needs ~12 GB of GPU memory AND several GB of host RAM; starting into a host
# with no RAM left gets the process (or its parent) killed by the OOM guard.
wait_gpu(){ while [ "$(free_mib)" -lt 12000 ] || [ "$(ram_avail_mib)" -lt 12000 ]; do sleep 30; done; }

say WAIT_FOR_GPU free=$(free_mib)
for T in ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped ObjectRestingEEGrasped; do
  wait_gpu
  B="$REPO/$DS/Resets/$PAIR/resets_$T.pt"
  [ -f "$B" ] && [ ! -f "${B%.pt}.pre_inhand.pt" ] && mv "$B" "${B%.pt}.pre_inhand.pt"
  rm -f "$B"
  say "LINEAR_${T}_START" free=$(free_mib)
  (cd "$REPO" && timeout 5400 $R scripts_v2/tools/record_reset_states.py --task "OmniReset-UR10eLinearGripper-CubeStack-$T-v0" \
      --reset_type "$T" --num_envs 128 --num_reset_states 40 --headless --dataset_dir "./$DS" \
      > "/home/mattephi/Work/Rubitek/cube-stack/linear/out/$T.inhand.log" 2>&1)
  say "LINEAR_${T}_END" rc=$? "$(stat -c%s "$B" 2>/dev/null || echo MISSING)"
done

wait_gpu; say LINEAR_FILMS_START
(cd linear && ./film.sh "./$DS" linear OmniReset-UR10eLinearGripper-CubeStack >> out/films.log 2>&1)
say LINEAR_FILMS_END rc=$?
echo "LINEAR_PIPELINE_DONE $(date -Is)" >> linear/out/pipeline.log

wait_gpu; say PINCH2F_START free=$(free_mib)
(cd pinch2f && : > out/pipeline.log && ./pipeline.sh > out/pipeline.stdout 2>&1)
say PINCH2F_END rc=$?
echo "QUEUE_DONE $(date -Is)" | tee -a "$LOG"
