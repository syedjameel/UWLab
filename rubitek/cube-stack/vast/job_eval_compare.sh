#!/usr/bin/env bash
# Controlled comparison: untrained model_0 vs model_200, identical deterministic evaluation.
#
# WHY THIS RUNS HERE. The dev card is saturated by its own training (2048 envs leaves ~500 MiB), so
# an eval alongside it does not fit. This box has 24 GB free beside its training run, so the
# comparison can happen NOW rather than after the local run stops -- which is the difference between
# knowing at 07:00 and knowing at 09:30.
#
# WHY THE COMPARISON AT ALL. The training reward falls over the run, but that fall is the
# action_rate penalty tracking gSDE's exploration noise (std 0.75 -> 2.65) while the task terms stay
# flat. Evaluation never sees that noise: rsl_rl's get_inference_policy returns act_inference, the
# actor's MEAN action. So the training curve cannot answer "did it learn"; only this can.
set -uo pipefail
WS=/workspace
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
cd "$WS/UWLab" || exit 1
PY=$WS/venv/bin/python
mkdir -p "$WS/eval_out"

# Near-goal first: it is where a compute-limited policy is most likely to show a difference at all.
for RT in ObjectPartiallyAssembledEEGrasped ObjectAnywhereEEAnywhere; do
  for CK in model_0 model_200; do
    echo "=== $CK on $RT ==="
    "$PY" "$WS/cube-stack/eval_policy.py" \
      --task OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-Play-v0 \
      --checkpoint "$WS/ckpt_from_local/${CK}.pt" \
      --num_envs 96 --episodes 192 --reset_type "$RT" --headless \
      --out "$WS/eval_out/${CK}_${RT}.json" 2>&1 | grep -E 'EVAL_RESULT|thresholds|Error|Traceback' | tail -4
  done
done
echo "=== eval compare done ==="
