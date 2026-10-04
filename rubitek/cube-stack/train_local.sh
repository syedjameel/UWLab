#!/usr/bin/env bash
# Train a cube-stacking policy, with the expensive run gated behind two cheap checks.
#
#   1. preflight  -- all four sampled reset banks present AND above a state-count floor
#   2. smoke      -- a short run that proves the env constructs and measures seconds/iteration,
#                    so the full run's duration is a MEASUREMENT rather than a guess
#   3. full run   -- only if both pass
#
# Usage:  ./train_local.sh <variant> [iters] [num_envs]
#   variant: noorient | oriented
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack

VARIANT=${1:-noorient}
ITERS=${2:-2000}
ENVS=${3:-2048}
SEED=${SEED:-42}
SMOKE_ITERS=${SMOKE_ITERS:-15}

case "$VARIANT" in
  noorient) TASK=OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-v0 ;;
  oriented) TASK=OmniReset-UR10eDelto-CubeStack-RelCartesianOSC-State-v0 ;;
  *) echo "unknown variant '$VARIANT' (want noorient|oriented)"; exit 2 ;;
esac

cd "$REPO" || exit 1

echo "############ preflight ############"
MIN_STATES=${MIN_STATES:-100} "$CS/preflight_train.sh" || {
  echo "ABORT: banks not ready. Not spending a training run on stub coverage."; exit 1; }

run_train () {  # run_train <iters> <logfile>
  "$CS/run.sh" scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$TASK" --num_envs "$ENVS" --max_iterations "$1" --seed "$SEED" --headless \
    env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 \
    env.sim.physx.gpu_collision_stack_size=1073741824 \
    > "$2" 2>&1
}

echo
echo "############ smoke: $SMOKE_ITERS iterations ############"
SMOKE_LOG="$CS/logs/train_${VARIANT}_smoke.log"
t0=$(date +%s)
run_train "$SMOKE_ITERS" "$SMOKE_LOG"
t1=$(date +%s)

# Grade on positive evidence, never on the wrapper's exit code.
if ! grep -qE 'Learning iteration|Mean reward' "$SMOKE_LOG"; then
  echo "ABORT: smoke produced no learning output. Last lines:"
  tail -25 "$SMOKE_LOG" | tr '\r' '\n' | tail -12
  exit 1
fi
elapsed=$((t1 - t0))
per_iter=$(python3 -c "print(f'{$elapsed/$SMOKE_ITERS:.2f}')")
proj_h=$(python3 -c "print(f'{$elapsed/$SMOKE_ITERS*$ITERS/3600:.2f}')")
echo "[SMOKE] $SMOKE_ITERS iters in ${elapsed}s -> ${per_iter}s/iter (includes ~60-90s startup)"
echo "[SMOKE] projected $ITERS iters ~= ${proj_h} h"
grep -E 'Mean reward|Mean episode length' "$SMOKE_LOG" | tail -4

echo
echo "############ full run: $ITERS iterations, $ENVS envs, seed $SEED ############"
FULL_LOG="$CS/logs/train_${VARIANT}.log"
run_train "$ITERS" "$FULL_LOG"
echo "[train] wrapper exit $? (not trusted)"
echo "--- newest checkpoints ---"
find logs/rsl_rl -name 'model_*.pt' -newermt '-2 days' -printf '%T@ %s %p\n' 2>/dev/null | sort -rn | head -5
echo "=== train_local done $(date +%H:%M:%S) ==="
