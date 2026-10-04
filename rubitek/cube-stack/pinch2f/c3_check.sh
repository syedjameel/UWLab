#!/usr/bin/env bash
# Small, filmed check of the DELTO pinch C3 bank BEFORE any full C2-C4 recording.
# Waits for the full grasp bank to finish, records 6 C3 states, then films them holding with the
# recorder's own scripted close (absolute hand action).
set -uo pipefail
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
O=/home/mattephi/Work/Rubitek/cube-stack/pinch2f/out/c3_check
DS=Datasets_ur10e_delto/OmniReset_pinch2f_c3check
PAIR=InsertiveCube34__ReceptiveCube34
mkdir -p "$O" "$DS/Grasps/InsertiveCube34" "$DS/Resets/$PAIR"
say(){ echo "C3CHECK $* $(date -Is)" | tee -a "$O/check.log"; }
until grep -qa "^rc=" /tmp/claude-1000/-home-mattephi-Work-Rubitek/d817c68b-2665-4f13-965b-515b6cadcd04/tasks/bxnvo38mv.output 2>/dev/null; do sleep 20; done
cp Datasets_ur10e_delto/OmniReset_pinch2f/Grasps/InsertiveCube34/grasps.pt "$DS/Grasps/InsertiveCube34/" || { say NO_GRASP_BANK; exit 1; }
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 UWLAB_CUBE_DATASET_DIR="./$DS"
say RECORD_START
timeout 1800 $R scripts_v2/tools/record_reset_states.py --task OmniReset-UR10eDelto-CubeStack-ObjectAnywhereEEGrasped-v0 \
    --reset_type ObjectAnywhereEEGrasped --num_envs 32 --num_reset_states 6 --headless --dataset_dir "./$DS" > "$O/record.log" 2>&1
say RECORD_END rc=$? "$(stat -c%s "$DS/Resets/$PAIR/resets_ObjectAnywhereEEGrasped.pt" 2>/dev/null || echo MISSING)"
[ -f "$DS/Resets/$PAIR/resets_ObjectAnywhereEEGrasped.pt" ] || exit 1
say FILM_START
UWLAB_VIEW_SCRIPTED_HOLD=1 UWLAB_VIEW_EYE="1.55,-1.35,0.95" UWLAB_VIEW_AT="0.45,0.00,0.10" timeout 1500 $R scripts_v2/tools/render_reset_states_viewport.py \
    --reset-states "$DS/Resets/$PAIR/resets_ObjectAnywhereEEGrasped.pt" --task OmniReset-UR10eDelto-CubeStack-ObjectAnywhereEEGrasped-v0 \
    --dataset-dir "./$DS" --insertive-variant cube34 --receptive-variant cube34 --out "$O/film" --count 6 --settle-steps 20 --repeats 1 \
    --video "$O/c3_check.mp4" --video-freeze 14 --video-settle-steps 60 --video-every 3 > "$O/film.log" 2>&1
say FILM_END rc=$?
grep -aE "^  state |OPPOSED [0-9]" "$O/film.log" | tee -a "$O/check.log"
echo "C3CHECK_DONE $(date -Is)" | tee -a "$O/check.log"
