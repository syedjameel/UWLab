#!/usr/bin/env bash
# The before/after pair: the SAME C3 reset, filmed under the shipped effort cap and under the raised
# one, both with a NULL ACTION.
#
# Null action matters. For this action space zero means "hold the current pose" for the arm and "hold
# the latched grasp" for the hand, so whatever the cube does in the clip is the RESET's doing and no
# policy's. That is the only way to show the grasp itself failing without the objection that a bad
# policy dropped it.
#
# It also needs no checkpoint, which makes this the cheapest possible end-to-end test of the render
# path -- camera, RTX, video writer -- hours before a trained policy needs filming. Rendering is
# local-only by necessity: the H100 doing the training has no graphics engine and segfaults on any
# camera-enabled run.
#
# Expected, from the endurance probe (128 envs, in_hand at step 150): x1 holds 0.047 of cubes and x20
# holds 0.586. So the x1 clip should show the cube falling out of the hand, and the x20 clip should
# show it held.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv-is6/bin/python

RT=${RT:-ObjectAnywhereEEGrasped}
EPS=${EPS:-3}
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="/home/mattephi/Work/Rubitek/uwlab-env/compat:$REPO/source/uwlab_assets:$REPO/source/uwlab_tasks:$REPO/source/uwlab:$REPO/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=268435456
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-0.15
export UWLAB_LATCHED_HAND=1 UWLAB_TUCK_OTHERS=0 UWLAB_GRIP_BIAS_RAD=0.05

cd "$REPO" || exit 1
mkdir -p "$CS/logs"
for EFF in 1 20; do
  OUT=$CS/results/reset_eff${EFF}
  rm -rf "$OUT"
  echo "=== rendering reset at effort x$EFF $(date -Is)"
  # Small footprint on purpose: the local GPU is also carrying the hedge training run, and an OOM
  # there would cost more than this clip is worth.
  UWLAB_HAND_EFFORT_SCALE=$EFF timeout 2400 "$PY" "$CS/render_rollout.py" \
    --task OmniReset-UR10eDelto-CubeStackTwoFingerGripCarry-RelCartesianOSC-State-v0 \
    --null_action --num_envs 1 --episodes "$EPS" --reset_type "$RT" \
    --dataset_dir ./Datasets_ur10e_delto/OmniReset_pinch2f \
    --view closeup --track_asset insertive_object \
    --width 960 --height 540 --collision_stack_mb 256 \
    --out "$OUT" > "$CS/logs/render_reset_eff${EFF}.log" 2>&1
  echo "=== effort x$EFF exit $? ; artefacts:"
  find "$OUT" -name '*.mp4' -printf '%s %p\n' 2>/dev/null | awk '{printf "%8.2f MB %s\n", $1/1048576, $2}'
done
echo "=== RESET_PAIR_DONE $(date -Is)"
