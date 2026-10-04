#!/usr/bin/env bash
# After the main queue: re-record linear ObjectRestingEEGrasped under the joint-closure in-hand check,
# then re-film it. The bank recorded at 15:08 predates that check (72% of its states have closed jaws).
set -uo pipefail
cd /home/mattephi/Work/Rubitek/cube-stack
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
DS=Datasets_ur10e_linear/OmniReset
B="$REPO/$DS/Resets/InsertiveCube34__ReceptiveCube34/resets_ObjectRestingEEGrasped.pt"
LOG=/home/mattephi/Work/Rubitek/cube-stack/gpu_queue.log
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 UWLAB_LINEAR_CUBE_DATASET_DIR="./$DS"
say(){ echo "Q $* $(date -Is)" | tee -a "$LOG"; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
ram_avail_mib(){ awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
wait_gpu(){ while [ "$(free_mib)" -lt 12000 ] || [ "$(ram_avail_mib)" -lt 12000 ]; do sleep 30; done; }
until grep -q QUEUE_DONE "$LOG"; do sleep 60; done
wait_gpu
mv -f "$B" "${B%.pt}.pre_jointcheck.pt"
say FOLLOWUP_LINEAR_C2_START
(cd "$REPO" && timeout 7200 $R scripts_v2/tools/record_reset_states.py --task OmniReset-UR10eLinearGripper-CubeStack-ObjectRestingEEGrasped-v0 \
    --reset_type ObjectRestingEEGrasped --num_envs 128 --num_reset_states 40 --headless --dataset_dir "./$DS" \
    > linear_c2_jointcheck.log 2>&1)
say FOLLOWUP_LINEAR_C2_END rc=$? "$(stat -c%s "$B" 2>/dev/null || echo MISSING)"
mv linear_c2_jointcheck.log /home/mattephi/Work/Rubitek/cube-stack/linear/out/ 2>/dev/null
wait_gpu
say FOLLOWUP_FILM_C2_START
(cd linear && UWLAB_FILM_ONLY=ObjectRestingEEGrasped ./film.sh "./$DS" linear OmniReset-UR10eLinearGripper-CubeStack >> out/films.log 2>&1)
say FOLLOWUP_FILM_C2_END rc=$?
echo "FOLLOWUP_DONE $(date -Is)" | tee -a "$LOG"
