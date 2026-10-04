#!/usr/bin/env bash
# Collect the before/after state traces and animate them.
#
# Substitutes for the photoreal clips, which are unavailable on this box: Isaac Sim 6 aims the camera
# correctly (verified cam_pos and look-at quaternion) and still draws nothing but sky, Isaac Sim 5.1
# crashes in libomni.hydra.rtx, and the renderer reports "IOMMU is enabled" -- a documented cause of
# broken Omniverse rendering and a BIOS setting, not something to change on someone's workstation
# unattended. Physics is unaffected, so the measured trajectories are still fully available.
#
# One process per effort scale: UWLAB_HAND_EFFORT_SCALE is consumed when the actuator config is built
# at import time, so it cannot be varied inside a single run.
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
export TRACE_ENVS=${TRACE_ENVS:-64} TRACE_STEPS=${TRACE_STEPS:-160}

cd "$REPO" || exit 1
mkdir -p "$CS/results" "$CS/logs"
for EFF in 1 20; do
  echo "=== trace effort x$EFF $(date -Is)"
  UWLAB_HAND_EFFORT_SCALE=$EFF TRACE_NAME="eff$EFF" \
    timeout 1800 "$PY" "$CS/trace_collect.py" > "$CS/logs/trace_eff$EFF.log" 2>&1
  grep -aE "TRACE wrote|Traceback|Error" "$CS/logs/trace_eff$EFF.log" | tail -3
done

# Animation needs matplotlib + ffmpeg, not Isaac. Use whichever interpreter actually has matplotlib.
APY=""
for cand in "$PY" python3; do
  "$cand" -c "import matplotlib" 2>/dev/null && { APY=$cand; break; }
done
if [ -z "$APY" ]; then echo "NO MATPLOTLIB -- traces collected, animation skipped"; exit 1; fi
echo "=== animating with $APY $(date -Is)"
"$APY" "$CS/trace_animate.py" eff1 eff20 "$CS/results/film_trace/rollout.mp4" 2>&1 | tail -3
ls -la "$CS/results/film_trace/rollout.mp4" 2>/dev/null | awk '{printf "%.2f MB %s\n", $5/1048576, $9}'
echo "=== TRACE_PAIR_DONE $(date -Is)"
