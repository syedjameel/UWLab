#!/usr/bin/env bash
# UR10e + linear gripper on cube34: the parallel-jaw baseline, C1-C4 + films, on the Isaac Lab 3.0 port.
set -uo pipefail
cd /home/mattephi/Work/Rubitek/cube-stack/linear
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
DS=Datasets_ur10e_linear/OmniReset
PAIR=InsertiveCube34__ReceptiveCube34
PFX=OmniReset-UR10eLinearGripper-CubeStack
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 UWLAB_LINEAR_CUBE_DATASET_DIR="./$DS"
mkdir -p "$REPO/$DS/Resets/$PAIR"
LOG=out/pipeline.log
say(){ echo "LIN $* $(date -Is)" | tee -a "$LOG"; }
sz(){ stat -c%s "$1" 2>/dev/null || echo MISSING; }

if [ -f "$REPO/$DS/Grasps/InsertiveCube34/grasps.pt" ]; then say GRASPS_SKIP "bank exists"; else
say GRASPS_START
timeout 3600 $R "$REPO/scripts_v2/tools/record_grasps.py" --task OmniReset-LinearGripper-GraspSampling-v0 \
    --num_envs 64 --num_grasps 300 --headless --dataset_dir "$REPO/$DS" env.scene.object=cube34 > out/grasps.log 2>&1
rc=$?; say GRASPS_END rc=$rc "$(sz $REPO/$DS/Grasps/InsertiveCube34/grasps.pt)"; fi

say C1_START
timeout 3600 $R "$REPO/scripts_v2/tools/record_reset_states.py" --task $PFX-ObjectAnywhereEEAnywhere-v0 \
    --reset_type ObjectAnywhereEEAnywhere --num_envs 128 --num_reset_states 200 --headless --dataset_dir "./$DS" > out/c1.log 2>&1
rc=$?; say C1_END rc=$rc "$(sz $REPO/$DS/Resets/$PAIR/resets_ObjectAnywhereEEAnywhere.pt)"

if [ -f "$REPO/$DS/Resets/$PAIR/partial_assemblies.pt" ]; then say PA_SKIP "bank exists"; else
say PA_START
timeout 3600 $R "$REPO/scripts_v2/tools/record_partial_assemblies.py" --task OmniReset-CubeStack-PartialAssemblies-v0 \
    --num_envs 64 --headless --dataset_dir "./$DS" --num_trajectories 64 \
    env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 > out/pa.log 2>&1
rc=$?; say PA_END rc=$rc "$(sz $REPO/$DS/Resets/$PAIR/partial_assemblies.pt)"; fi

for T in ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped ObjectRestingEEGrasped; do
  say "${T}_START"
  timeout 5400 $R "$REPO/scripts_v2/tools/record_reset_states.py" --task "$PFX-$T-v0" --reset_type "$T" \
      --num_envs 128 --num_reset_states 40 --headless --dataset_dir "./$DS" > "out/$T.log" 2>&1
  rc=$?; say "${T}_END" rc=$rc "$(sz $REPO/$DS/Resets/$PAIR/resets_$T.pt)"
done

say FILMS_START; ./film.sh "./$DS" linear "$PFX" >> out/films.log 2>&1; say FILMS_END rc=$?
echo "LINEAR_PIPELINE_DONE $(date -Is)" | tee -a "$LOG"
