#!/usr/bin/env bash
# DELTO two-finger pinch (fingers 1+2 close, 3/4/5 held open) on cube34: C1-C4 + films.
# Waits for the linear-gripper baseline to release the GPU (one Isaac instance fills the card).
set -uo pipefail
cd /home/mattephi/Work/Rubitek/cube-stack/pinch2f
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
DS=Datasets_ur10e_delto/OmniReset_pinch2f
PAIR=InsertiveCube34__ReceptiveCube34
PFX=OmniReset-UR10eDelto-CubeStack
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824 UWLAB_CUBE_DATASET_DIR="./$DS" UWLAB_DELTO_PINCH=1 UWLAB_GRAVITY_COMP=1
LOG=out/pipeline.log
say(){ echo "P2F $* $(date -Is)" | tee -a "$LOG"; }
sz(){ stat -c%s "$1" 2>/dev/null || echo MISSING; }

until grep -q LINEAR_PIPELINE_DONE /home/mattephi/Work/Rubitek/cube-stack/linear/out/pipeline.log 2>/dev/null; do sleep 20; done
sleep 10
mkdir -p "$REPO/$DS/Resets/$PAIR"

say GRASPS_START
timeout 5400 $R "$REPO/scripts_v2/tools/record_grasps.py" --task OmniReset-Delto-GraspSampling-v0 \
    --num_envs 64 --num_grasps 300 --headless --dataset_dir "$REPO/$DS" env.scene.object=cube34 > out/grasps.log 2>&1
rc=$?; say GRASPS_END rc=$rc "$(sz $REPO/$DS/Grasps/InsertiveCube34/grasps.pt)" "$(grep -a BATCH_END out/grasps.log | tail -1 | grep -o 'attempts_consumed_so_far=[0-9/]*\|cumulative_successes=[0-9]*' | tr '\n' ' ')"

# C1 does not read the grasp bank; the corrected-rig bank is reused rather than re-recorded.
cp "$REPO/Datasets_ur10e_delto/OmniReset_fist/Resets/$PAIR/resets_ObjectAnywhereEEAnywhere.pt" "$REPO/$DS/Resets/$PAIR/" && say C1_REUSED

say PA_START
timeout 3600 $R "$REPO/scripts_v2/tools/record_partial_assemblies.py" --task OmniReset-CubeStack-PartialAssemblies-v0 \
    --num_envs 64 --headless --dataset_dir "./$DS" --num_trajectories 64 \
    env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 > out/pa.log 2>&1
rc=$?; say PA_END rc=$rc "$(sz $REPO/$DS/Resets/$PAIR/partial_assemblies.pt)"

if [ -f "$REPO/$DS/Grasps/InsertiveCube34/grasps.pt" ]; then
  for T in ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped ObjectRestingEEGrasped; do
    say "${T}_START"
    timeout 5400 $R "$REPO/scripts_v2/tools/record_reset_states.py" --task "$PFX-$T-v0" --reset_type "$T" \
        --num_envs 128 --num_reset_states 40 --headless --dataset_dir "./$DS" > "out/$T.log" 2>&1
    rc=$?; say "${T}_END" rc=$rc "$(sz $REPO/$DS/Resets/$PAIR/resets_$T.pt)"
  done
else
  say GRASPED_SKIPPED "no grasp bank"
fi

say FILMS_START
UWLAB_DELTO_PINCH=0 /home/mattephi/Work/Rubitek/cube-stack/linear/film.sh "./$DS" pinch2f "$PFX" >> out/films.log 2>&1
say FILMS_END rc=$?
echo "PINCH2F_PIPELINE_DONE $(date -Is)" | tee -a "$LOG"
