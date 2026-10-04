#!/usr/bin/env bash
# Re-film the existing 6-state DELTO pinch C3 check bank with the fixed renderer (no time_out
# auto-reset mid-settle). Waits for the linear finish.sh run so the two sim jobs don't share the GPU.
set -uo pipefail
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
O=/home/mattephi/Work/Rubitek/cube-stack/pinch2f/out/c3_check
DS=Datasets_ur10e_delto/OmniReset_pinch2f_c3check
PAIR=InsertiveCube34__ReceptiveCube34
LIN=/home/mattephi/Work/Rubitek/cube-stack/linear/out/finish.log
say(){ echo "C3REFILM $* $(date -Is)" | tee -a "$O/refilm.log"; }
until grep -q FINISH_DONE "$LIN" 2>/dev/null; do sleep 30; done
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 12000 ] || \
      [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -lt 12000 ]; do sleep 30; done
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 UWLAB_CUBE_DATASET_DIR="./$DS"
rm -rf "$O/film2" "$O/c3_check2.mp4"
say FILM_START
UWLAB_VIEW_SCRIPTED_HOLD=1 UWLAB_VIEW_EYE="1.55,-1.35,0.95" UWLAB_VIEW_AT="0.45,0.00,0.10" timeout 1500 $R scripts_v2/tools/render_reset_states_viewport.py \
    --reset-states "$DS/Resets/$PAIR/resets_ObjectAnywhereEEGrasped.pt" --task OmniReset-UR10eDelto-CubeStack-ObjectAnywhereEEGrasped-v0 \
    --dataset-dir "./$DS" --insertive-variant cube34 --receptive-variant cube34 --out "$O/film2" --count 6 --settle-steps 60 --repeats 1 \
    --video "$O/c3_check2.mp4" --video-freeze 14 --video-settle-steps 60 --video-every 3 > "$O/film2.log" 2>&1
say FILM_END rc=$?
grep -aE "^  state |OPPOSED [0-9]" "$O/film2.log" | tee -a "$O/refilm.log"
echo "C3REFILM_DONE $(date -Is)" | tee -a "$O/refilm.log"
