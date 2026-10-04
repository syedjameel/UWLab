#!/usr/bin/env bash
# Render the report's clips through the CameraCfg SENSOR path (film_camera.py).
#
# This replaces render_films_local.sh / render_reset_pair.sh, which both go through env.render() and
# therefore through the Kit viewport -- which under Isaac Sim 6 produces one good frame per run and
# blank sky for every frame after it, no matter how the camera is handled. Isaac 5.1 is not an escape
# either: it crashes in libomni.hydra.rtx on this box's 610 driver.
#
# Two kinds of clip:
#   PAIR   -- the same C3 reset under the shipped effort cap and the raised one, null action. Null
#             action means the arm holds its pose and the hand holds its latched grasp, so the cube's
#             behaviour is the reset's doing and no policy's. This is the before/after.
#   POLICY -- a trained checkpoint acting. Needs CK set.
#
# Rendering is local-only: the H100 doing the training has no graphics engine.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv-is6/bin/python

export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="/home/mattephi/Work/Rubitek/uwlab-env/compat:$REPO/source/uwlab_assets:$REPO/source/uwlab_tasks:$REPO/source/uwlab:$REPO/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=268435456
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-0.15
export UWLAB_LATCHED_HAND=1 UWLAB_TUCK_OTHERS=0 UWLAB_GRIP_BIAS_RAD=0.05

MODE=${MODE:-pair}
EPS=${EPS:-2}
RT=${RT:-ObjectAnywhereEEGrasped}
cd "$REPO" || exit 1
mkdir -p "$CS/logs"

run() {  # run <name> <effort> <extra args...>
  local name=$1 eff=$2; shift 2
  local out=$CS/results/film_$name
  echo "=== film_$name (effort x$eff) $(date -Is)"
  UWLAB_HAND_EFFORT_SCALE="$eff" timeout 3600 "$PY" "$CS/film_camera.py" \
    --episodes "$EPS" --reset_type "$RT" --out "$out" "$@" \
    > "$CS/logs/film_$name.log" 2>&1
  # Grade on frames that CONTAIN something, not on exit status: every broken render tonight exited 0.
  grep -aE "frames have real content|WARNING|Traceback" "$CS/logs/film_$name.log" | tail -2
}

case "$MODE" in
  pair)
    run eff1  1  --null_action
    run eff20 20 --null_action
    ;;
  policy)
    CK=${CK:?set CK to a checkpoint path}
    run "policy_${RT}" 20 --checkpoint "$CK"
    ;;
  *) echo "MODE must be pair or policy"; exit 2;;
esac
echo "=== FILMS_DONE $(date -Is)"
