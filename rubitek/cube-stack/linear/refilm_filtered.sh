#!/usr/bin/env bash
# After qa_all.sh: re-film every grasped family whose bank the QA filter changed, from the FILTERED
# bank, so the report's films show exactly what ships. Unchanged banks keep their existing film.
set -uo pipefail
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
L=/home/mattephi/Work/Rubitek/cube-stack/linear/out
DS=Datasets_ur10e_linear/OmniReset; PAIR=InsertiveCube34__ReceptiveCube34
PFX=OmniReset-UR10eLinearGripper-CubeStack
until grep -q QA_DONE $L/qa/qa.log 2>/dev/null; do sleep 30; done
unset UWLAB_DELTO_PINCH
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 UWLAB_LINEAR_CUBE_DATASET_DIR="./$DS"
for T in ObjectRestingEEGrasped ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped; do
  grep -a "resets_$T.unfiltered.pt: kept" $L/qa/qa.log | grep -q "dropped {}" && { echo "REFILM $T unchanged, skip" >> $L/qa/qa.log; continue; }
  rm -rf "$L/films/linear_$T" "$L/films/linear_$T.mp4"
  UWLAB_VIEW_SCRIPTED_HOLD=1 UWLAB_VIEW_EYE="1.55,-1.35,0.95" UWLAB_VIEW_AT="0.45,0.00,0.10" timeout 1500 \
    $R scripts_v2/tools/render_reset_states_viewport.py --reset-states "$DS/Resets/$PAIR/resets_$T.pt" \
    --task "$PFX-$T-v0" --dataset-dir "./$DS" --insertive-variant cube34 --receptive-variant cube34 \
    --out "$L/films/linear_$T" --count 6 --settle-steps 60 --repeats 1 \
    --video "$L/films/linear_$T.mp4" --video-freeze 14 --video-settle-steps 60 --video-every 3 \
    > "$L/film_filtered_$T.log" 2>&1
  echo "REFILM $T rc=$? $(grep -a 'OPPOSED [0-9]' $L/film_filtered_$T.log | tail -1 | cut -c1-60) $(date -Is)" >> $L/qa/qa.log
done
echo "REFILM_DONE $(date -Is)" >> $L/qa/qa.log
