#!/usr/bin/env bash
# C2 only. C4 (ObjectPartiallyAssembledEEGrasped) fails structurally -- above_ground = 0 of 128 in
# every episode and orient_down = 4, with no environment passing even five of the six conditions.
# It consumes partial_assemblies.pt, which was copied from the OLD dataset tree and encodes seated
# poses from the pre-fix pipeline; that is a separate job from this one, not a tuning matter.
set -uo pipefail
cd /home/mattephi/Work/Rubitek/cube-stack/pinch
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
DS=./Datasets_ur10e_delto/OmniReset_hold
PAIR=InsertiveCube34__ReceptiveCube34
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
export UWLAB_DELTO_PINCH=1
export UWLAB_CUBE_DATASET_DIR="$DS"
export UWLAB_GRASP_DEBUG=1
export UWLAB_GRAVITY_COMP=1
T=ObjectRestingEEGrasped
rm -f "$REPO/${DS#./}/Resets/$PAIR/resets_$T.pt"
echo "=== $(date -Is) $T" | tee -a out/stage20.log
timeout 5400 ./run6.sh "$REPO/scripts_v2/tools/record_reset_states.py" \
    --task "OmniReset-UR10eDelto-CubeStack-$T-v0" --reset_type "$T" \
    --num_envs 128 --num_reset_states 40 --headless --dataset_dir "$DS" \
    env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 \
    > "out/resets20_$T.log" 2>&1
echo "=== $(date -Is)   rc=$? bytes=$(ls -la "$REPO/${DS#./}/Resets/$PAIR/resets_$T.pt" 2>/dev/null | awk '{print $5}')" | tee -a out/stage20.log
echo "STAGE20_DONE $(date -Is)" | tee -a out/stage20.log
