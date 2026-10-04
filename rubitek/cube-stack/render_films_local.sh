#!/usr/bin/env bash
# Render the trained pinch policy to video, LOCALLY.
#
# WHY LOCALLY. Any camera-enabled Isaac run needs an RTX-class GPU: the H100 that does the training
# has no graphics engine and segfaults about a second into startup. The 4070 Ti SUPER here is the
# only machine in the loop that can render, so films are always a pull-then-render step.
#
# THE ENV VARS BELOW ARE NOT DECORATION. The policy was trained with the latched hand, the OSC
# gravity term, and whichever effort/tuck setting the endurance probe selected. Rendering without
# them films a DIFFERENT ROBOT from the one that learned -- the hand would ramp itself shut and the
# arm would sag -- and the clip would show a failure the policy is not responsible for. Pass the
# same values the training job used.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv-is6/bin/python

CK=${CK:?set CK to a checkpoint path}
TASK=${TASK:-OmniReset-UR10eDelto-CubeStackTwoFingerGripCarry-RelCartesianOSC-State-v0}
RT=${RT:-ObjectAnywhereEEGrasped}
N=${N:-40}
PREFIX=${PREFIX:-pinch}
VIEW=${VIEW:-closeup}
# The grasped families start with the cube ~180 mm from the receptive cube, in the hand. Tracking the
# RECEPTIVE cube there leaves the hand out of frame for most of the episode and the clip shows only a
# cube dropping into view; track the one being carried instead.
TRACK=${TRACK:-insertive_object}

export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="/home/mattephi/Work/Rubitek/uwlab-env/compat:$REPO/source/uwlab_assets:$REPO/source/uwlab_tasks:$REPO/source/uwlab:$REPO/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-0.15
export UWLAB_LATCHED_HAND=1
export UWLAB_HAND_EFFORT_SCALE=${UWLAB_HAND_EFFORT_SCALE:-1}
export UWLAB_TUCK_OTHERS=${UWLAB_TUCK_OTHERS:-0}
export UWLAB_GRIP_BIAS_RAD=${UWLAB_GRIP_BIAS_RAD:-0.05}

OUT=$CS/results/video_${PREFIX}_${RT}
rm -rf "$OUT"; mkdir -p "$CS/logs"
echo "=== rendering $N episodes / $RT / $(basename "$CK") effort=x$UWLAB_HAND_EFFORT_SCALE tuck=$UWLAB_TUCK_OTHERS"
cd "$REPO" || exit 1
timeout 5400 "$PY" "$CS/render_rollout.py" \
  --task "$TASK" --checkpoint "$CK" --num_envs 1 --episodes "$N" --reset_type "$RT" \
  --dataset_dir ./Datasets_ur10e_delto/OmniReset_pinch2f \
  --view "$VIEW" --track_asset "$TRACK" --collision_stack_mb 1024 \
  --out "$OUT" > "$CS/logs/render_${PREFIX}_${RT}.log" 2>&1
echo "=== exit $? ; artefacts:"
ls -la "$OUT" 2>/dev/null | head
