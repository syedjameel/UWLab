#!/usr/bin/env bash
# Second seed of the near-goal-only diagnostic. See train_diagnostic.sh for the full rationale.
#
# Why a second seed rather than a second hyperparameter: the paper (§A.3.10) records that seeds
# produce BEHAVIOURALLY different policies, not merely noisier ones, and selects among them with an
# offline proxy. If one seed learns near-goal and the other does not, that is itself the answer to
# "can this pipeline learn" -- and one arm learning is enough to rule out a structural defect.
set -uo pipefail
WS=/workspace
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
cd "$WS/UWLab" || exit 1
PY=$WS/venv/bin/python

ITERS=${ITERS:-2000}
ENVS=${ENVS:-2048}
SEED=${SEED:-7}
ENTROPY=${ENTROPY:-0.0005}

echo "=== diagnostic seed $SEED: near-goal only, entropy_coef=$ENTROPY, $ENVS envs ==="
"$PY" scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-UR10eDelto-CubeStackNearGoal-RelCartesianOSC-State-v0 \
  --num_envs "$ENVS" --max_iterations "$ITERS" --seed "$SEED" --headless \
  env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 \
  env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e_delto/OmniReset \
  agent.algorithm.entropy_coef="$ENTROPY" 2>&1 \
  | tr '\r' '\n' | grep -vE '^\[[0-9]' | tail -800
echo "=== diagnostic seed $SEED done ==="
