#!/usr/bin/env bash
# DELTO two-finger pinch C1-C4, robust pipeline: record C2-C4 (arm pinned, grasp-centre in-hand box),
# replay EVERY grasped state with the film renderer and keep only OPPOSED, then film C1-C4 where the
# report reads them (linear/out/films/pinch2f_<family>.mp4; QA verdicts in pinch2f/out/qa).
# Launch only after c3_fast.sh's film shows the pinch holding.
set -uo pipefail
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv-is6/bin/python
O=/home/mattephi/Work/Rubitek/cube-stack/pinch2f/out
FILMS=/home/mattephi/Work/Rubitek/cube-stack/linear/out/films
DS=Datasets_ur10e_delto/OmniReset_pinch2f; PAIR=InsertiveCube34__ReceptiveCube34
PFX=OmniReset-UR10eDelto-CubeStack
mkdir -p $O/qa $FILMS; say(){ echo "P2FULL $* $(date -Is)" | tee -a $O/full.log; }
# Arm held by a stiff PD, never by per-step state writes (those teleport the articulation and drop the
# pinch's friction anchors: 0 held vs 40/128 accepted in the first batch). Same hold in the film replay.
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_DROP_ARM_ACTION=1 UWLAB_ARM_PD_HOLD=1 UWLAB_VIEW_ARM_PD_ONLY=1
export UWLAB_HOLD_OBJECT_STEPS=${UWLAB_HOLD_OBJECT_STEPS:-8}
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 UWLAB_CUBE_DATASET_DIR="./$DS"
VIEW="UWLAB_VIEW_EYE=1.55,-1.35,0.95 UWLAB_VIEW_AT=0.45,0.00,0.10"

for T in ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped ObjectRestingEEGrasped; do
  B=$DS/Resets/$PAIR/resets_$T.pt
  say "REC_${T}_START"
  timeout 3600 $R scripts_v2/tools/record_reset_states.py --task "$PFX-$T-v0" --reset_type "$T" \
      --num_envs 128 --num_reset_states 40 --headless --dataset_dir "./$DS" > "$O/rec_$T.log" 2>&1
  say "REC_${T}_END rc=$? $(grep -ao '[0-9]*/40 \[[0-9:]*' $O/rec_$T.log | tail -1)"
  [ -f "$B" ] || { say "NO_BANK $T"; continue; }
  cp -f "$B" "${B%.pt}.unfiltered.pt"
  rm -rf "$O/qa/render_$T"
  env $VIEW UWLAB_VIEW_SCRIPTED_HOLD=1 timeout 2400 $R scripts_v2/tools/render_reset_states_viewport.py \
      --reset-states "${B%.pt}.unfiltered.pt" --task "$PFX-$T-v0" --dataset-dir "./$DS" \
      --insertive-variant cube34 --receptive-variant cube34 --out "$O/qa/render_$T" --count 1000 \
      --settle-steps 60 --repeats 1 > "$O/qa/render_$T.log" 2>&1
  say "QA_${T} $(grep -a 'OPPOSED [0-9]' $O/qa/render_$T.log | tail -1 | cut -c1-70)"
  $PY scripts_v2/tools/filter_bank_by_render.py --bank "${B%.pt}.unfiltered.pt" --render-dir "$O/qa/render_$T" \
      --out "$B" 2>&1 | tee -a $O/full.log
done

for T in ObjectAnywhereEEAnywhere ObjectRestingEEGrasped ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped; do
  HOLD=1; EXTRA=; [ "$T" = ObjectAnywhereEEAnywhere ] && { HOLD=; EXTRA=--allow-no-dataset-dir; }
  [ -f "$DS/Resets/$PAIR/resets_$T.pt" ] || { say "FILM_SKIP $T no bank"; continue; }
  rm -rf "$FILMS/pinch2f_$T" "$FILMS/pinch2f_$T.mp4"
  env $VIEW UWLAB_VIEW_SCRIPTED_HOLD=$HOLD timeout 1500 $R scripts_v2/tools/render_reset_states_viewport.py \
      --reset-states "$DS/Resets/$PAIR/resets_$T.pt" --task "$PFX-$T-v0" --dataset-dir "./$DS" \
      --insertive-variant cube34 --receptive-variant cube34 --out "$FILMS/pinch2f_$T" --count 6 \
      --settle-steps 60 --repeats 1 --video "$FILMS/pinch2f_$T.mp4" --video-freeze 14 \
      --video-settle-steps 60 --video-every 3 $EXTRA > "$O/film_$T.log" 2>&1
  say "FILM_${T} rc=$? $(grep -a 'OPPOSED [0-9]' $O/film_$T.log | tail -1 | cut -c1-70)"
done
echo "P2FULL_DONE $(date -Is)" | tee -a $O/full.log
