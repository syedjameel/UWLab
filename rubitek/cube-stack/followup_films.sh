#!/usr/bin/env bash
# After the C2 follow-up: re-film linear C1/C3/C4, whose first-state freeze frames were rendered
# before the viewport-tick fix (they show the pre-write scene).
set -uo pipefail
cd /home/mattephi/Work/Rubitek/cube-stack
LOG=/home/mattephi/Work/Rubitek/cube-stack/gpu_queue.log
say(){ echo "Q $* $(date -Is)" | tee -a "$LOG"; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
ram_avail_mib(){ awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
wait_gpu(){ while [ "$(free_mib)" -lt 12000 ] || [ "$(ram_avail_mib)" -lt 12000 ]; do sleep 30; done; }
until grep -q FOLLOWUP_DONE "$LOG"; do sleep 60; done
for T in ObjectAnywhereEEAnywhere ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped; do
  wait_gpu; say "REFILM_$T"
  (cd linear && UWLAB_FILM_ONLY=$T ./film.sh ./Datasets_ur10e_linear/OmniReset linear OmniReset-UR10eLinearGripper-CubeStack >> out/films.log 2>&1)
done
echo "REFILM_DONE $(date -Is)" | tee -a "$LOG"
