#!/usr/bin/env bash
# Full-bank replay QA for the linear grasped banks: render EVERY state (stills only, 60-step settle,
# arm held, scripted close), then keep only OPPOSED states. Originals are kept as *.unfiltered.pt.
set -uo pipefail
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv-is6/bin/python
L=/home/mattephi/Work/Rubitek/cube-stack/linear/out/qa
DS=Datasets_ur10e_linear/OmniReset; PAIR=InsertiveCube34__ReceptiveCube34
PFX=OmniReset-UR10eLinearGripper-CubeStack
mkdir -p $L; say(){ echo "QA $* $(date -Is)" | tee -a $L/qa.log; }
unset UWLAB_DELTO_PINCH
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 UWLAB_LINEAR_CUBE_DATASET_DIR="./$DS"
for T in ObjectRestingEEGrasped ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped; do
  B=$DS/Resets/$PAIR/resets_$T.pt
  [ -f "${B%.pt}.unfiltered.pt" ] || cp "$B" "${B%.pt}.unfiltered.pt"
  rm -rf $L/render_$T
  say "RENDER_${T}_START"
  UWLAB_VIEW_SCRIPTED_HOLD=1 UWLAB_VIEW_EYE="1.55,-1.35,0.95" UWLAB_VIEW_AT="0.45,0.00,0.10" timeout 2400 \
    $R scripts_v2/tools/render_reset_states_viewport.py --reset-states "${B%.pt}.unfiltered.pt" \
    --task "$PFX-$T-v0" --dataset-dir "./$DS" --insertive-variant cube34 --receptive-variant cube34 \
    --out $L/render_$T --count 1000 --settle-steps 60 --repeats 1 > $L/render_$T.log 2>&1
  say "RENDER_${T}_END rc=$? $(grep -a 'OPPOSED [0-9]' $L/render_$T.log | tail -1 | cut -c1-70)"
  $PY scripts_v2/tools/filter_bank_by_render.py --bank "${B%.pt}.unfiltered.pt" --render-dir $L/render_$T \
    --out "$B" 2>&1 | tee -a $L/qa.log
done
echo "QA_DONE $(date -Is)" | tee -a $L/qa.log
