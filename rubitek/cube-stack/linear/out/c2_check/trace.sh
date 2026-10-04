#!/usr/bin/env bash
# Traced 2-state replay of the small C2 check bank: per-step jaw q/target/torque and cube z/v.
set -uo pipefail
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
O=/home/mattephi/Work/Rubitek/cube-stack/linear/out/c2_check
DS=Datasets_ur10e_linear/OmniReset_c2check
PAIR=InsertiveCube34__ReceptiveCube34
unset UWLAB_DELTO_PINCH
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 UWLAB_LINEAR_CUBE_DATASET_DIR="./$DS"
UWLAB_VIEW_TRACE=1 UWLAB_VIEW_SCRIPTED_HOLD=1 UWLAB_VIEW_EYE="1.55,-1.35,0.95" UWLAB_VIEW_AT="0.45,0.00,0.10" timeout 1200 $R scripts_v2/tools/render_reset_states_viewport.py \
    --reset-states $DS/Resets/$PAIR/resets_ObjectRestingEEGrasped.pt --task OmniReset-UR10eLinearGripper-CubeStack-ObjectRestingEEGrasped-v0 \
    --dataset-dir "./$DS" --insertive-variant cube34 --receptive-variant cube34 --out $O/trace_film --count 2 --settle-steps 20 --repeats 1 \
    > $O/trace.log 2>&1
echo "TRACE_DONE rc=$? $(date -Is)" >> $O/trace.log
