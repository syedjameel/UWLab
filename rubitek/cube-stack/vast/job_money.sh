#!/usr/bin/env bash
# THE MAIN RUN: paper-faithful four-family mixture, position-only success criterion.
#
# WHY THIS DIFFERS FROM THE FIRST ATTEMPT. The first four-family run used the config default
# entropy_coef=0.006 and learned nothing measurable in 402 iterations. The near-goal-only
# diagnostic isolated the cause: at 0.006 the gSDE noise std climbed 0.75 -> 2.65 while every task
# reward term stayed flat, i.e. the entropy bonus dominated the advantage signal and the sampled
# actions saturated against the +-1 clip. At 0.0005 the same pipeline moved near-goal success from
# 1.9% to ~20% within 60 iterations and held std at ~0.73. That is a fix to the optimiser
# configuration, not to the method, so the paper's flat Uniform(D) mixture is restored here.
set -uo pipefail
WS=/workspace
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
cd "$WS/UWLab" || exit 1

ITERS=${ITERS:-1500}
ENVS=${ENVS:-2048}
SEED=${SEED:-42}
ENTROPY=${ENTROPY:-0.0005}

echo "=== MONEY RUN: 4-family mixture, entropy=$ENTROPY, envs=$ENVS, seed=$SEED ==="
"$WS/venv/bin/python" scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-v0 \
  --num_envs "$ENVS" --max_iterations "$ITERS" --seed "$SEED" --headless \
  env.scene.insertive_object=cube34 \
  env.scene.receptive_object=cube34 \
  env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e_delto/OmniReset \
  agent.algorithm.entropy_coef="$ENTROPY" \
  agent.save_interval=50 2>&1 | tr '\r' '\n' | grep -vE '^\[[0-9]'
echo "=== money run exited ==="
find logs/rsl_rl -name 'model_*.pt' -newermt '-1 day' -printf '%s %p\n' | sort -rn | head -5
