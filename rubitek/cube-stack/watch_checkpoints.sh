#!/usr/bin/env bash
# Evaluate every new checkpoint as it lands, instead of waiting for the run to finish.
#
# WHY. Measured throughput is 47.4 s/iteration at 2048 envs -- collection 45.9 s against 0.109 s of
# learning, i.e. 99.8% physics. 2000 iterations would be 26 h, which is not reachable here. But
# save_interval is 100, so a checkpoint lands roughly every 78 minutes, and each one is a usable
# artefact. Evaluating as they appear means there is always a current best result with video,
# rather than nothing until a run that will not finish does.
#
# Eval runs at a small env count so it can share the GPU with training. If it OOMs, that is
# reported and training is left alone -- the training run is the thing worth protecting.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
RUNS=$REPO/logs/rsl_rl/ur10e_delto_omnireset_agent
TASK=${TASK:-OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-Play-v0}
EVAL_ENVS=${EVAL_ENVS:-64}
EVAL_EPISODES=${EVAL_EPISODES:-128}
seen=""

mkdir -p "$CS/results/evals"
echo "=== checkpoint watcher armed $(date +%H:%M:%S) ==="
for i in $(seq 1 500); do
  ckpt=$(find "$RUNS" -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  if [ -n "${ckpt:-}" ] && ! echo "$seen" | grep -qF "$ckpt"; then
    seen="$seen
$ckpt"
    tag=$(basename "$ckpt" .pt)
    echo "=== [$(date +%H:%M:%S)] evaluating $tag ==="
    "$CS/run.sh" "$CS/eval_policy.py" \
      --task "$TASK" --checkpoint "$ckpt" \
      --num_envs "$EVAL_ENVS" --episodes "$EVAL_EPISODES" \
      --collision_stack_mb 512 --headless \
      --out "$CS/results/evals/${tag}.json" \
      > "$CS/logs/eval_${tag}.log" 2>&1
    if grep -q 'EVAL_RESULT' "$CS/logs/eval_${tag}.log"; then
      grep 'EVAL_RESULT' "$CS/logs/eval_${tag}.log"
    else
      echo "  eval failed for $tag:"; tail -6 "$CS/logs/eval_${tag}.log" | tr '\r' '\n' | tail -4
    fi
  fi
  sleep 120
done
