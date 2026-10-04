#!/usr/bin/env bash
# Finish the UR10e linear-gripper cube banks, AFTER the small filmed C2 check passed:
#   1. re-record C2 (ObjectRestingEEGrasped) to 40 states under the jaw-closure in-hand check
#   2. render films for C1-C4 with the fixed renderer; grasped families hold the jaws with the
#      scripted close (a zero action opens the binary jaws), C1 replays its own mixed jaw states
set -uo pipefail
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
L=/home/mattephi/Work/Rubitek/cube-stack/linear
DS=Datasets_ur10e_linear/OmniReset
PAIR=InsertiveCube34__ReceptiveCube34
PFX=OmniReset-UR10eLinearGripper-CubeStack
LOG=$L/out/finish.log
say(){ echo "FINISH $* $(date -Is)" | tee -a $LOG; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
ram_avail_mib(){ awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
wait_mem(){ while [ "$(free_mib)" -lt 12000 ] || [ "$(ram_avail_mib)" -lt 12000 ]; do sleep 30; done; }
unset UWLAB_DELTO_PINCH
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 UWLAB_LINEAR_CUBE_DATASET_DIR="./$DS"

B=$DS/Resets/$PAIR/resets_ObjectRestingEEGrasped.pt
wait_mem
# Partial bank from the stopped strict-check run; keep it aside, never over an earlier backup.
[ -f "$B" ] && mv -f "$B" "${B%.pt}.partial_strict.pt"
# Robust acceptance (jaw-pad in-hand, in the task cfg); arm pinned at the IK pose. Validated on c2_fast.
export UWLAB_PIN_ARM=1
say C2_START
timeout 7200 $R scripts_v2/tools/record_reset_states.py --task $PFX-ObjectRestingEEGrasped-v0 \
    --reset_type ObjectRestingEEGrasped --num_envs 128 --num_reset_states 40 --headless --dataset_dir "./$DS" \
    > $L/out/c2_final.log 2>&1
say C2_END rc=$? "$(stat -c%s $B 2>/dev/null || echo MISSING)"

mkdir -p $L/out/films
for T in ObjectAnywhereEEAnywhere ObjectRestingEEGrasped ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped; do
  wait_mem
  HOLD=1; [ "$T" = ObjectAnywhereEEAnywhere ] && HOLD=
  EXTRA=; [ "$T" = ObjectAnywhereEEAnywhere ] && EXTRA=--allow-no-dataset-dir
  rm -rf "$L/out/films/linear_$T" "$L/out/films/linear_$T.mp4"
  say "FILM_${T}_START"
  UWLAB_VIEW_SCRIPTED_HOLD=$HOLD UWLAB_VIEW_EYE="1.55,-1.35,0.95" UWLAB_VIEW_AT="0.45,0.00,0.10" timeout 1500 \
    $R scripts_v2/tools/render_reset_states_viewport.py --reset-states "$DS/Resets/$PAIR/resets_$T.pt" \
    --task "$PFX-$T-v0" --dataset-dir "./$DS" --insertive-variant cube34 --receptive-variant cube34 \
    --out "$L/out/films/linear_$T" --count 6 --settle-steps 20 --repeats 1 \
    --video "$L/out/films/linear_$T.mp4" --video-freeze 14 --video-settle-steps 60 --video-every 3 $EXTRA \
    > "$L/out/film_final_$T.log" 2>&1
  say "FILM_${T}_END" rc=$? "$(grep -a 'OPPOSED [0-9]' $L/out/film_final_$T.log | tail -1 | cut -c1-80)"
done
echo "FINISH_DONE $(date -Is)" | tee -a $LOG
