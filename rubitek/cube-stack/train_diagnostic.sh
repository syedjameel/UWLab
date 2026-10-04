#!/usr/bin/env bash
# DIAGNOSTIC RUN: can this pipeline learn anything at all?
#
# After 402 iterations on the paper's four-family mixture the policy scored 9.05% near-goal against
# an untrained 10.13% (p=0.454) and 0.0% everywhere else, while hand action magnitude rose 11.5x.
# It moved a long way and learned nothing measurable. Two explanations fit equally well from
# outside: the compute budget is ~2 orders below the paper's, or something in this port prevents
# learning at all. This run distinguishes them.
#
# Two changes from the main run, each with its reason:
#
#   1. NEAR-GOAL RESETS ONLY. The easiest family -- cube starts ~13 mm from a 5 mm tolerance and
#      needs a small local correction -- and 4x the near-goal experience per iteration by dropping
#      the other three from the mixture. THIS IS NOT THE PAPER'S METHOD: OmniReset's claim rests on
#      the flat four-family mixture producing an emergent backwards curriculum, and removing three
#      families removes the mechanism under test. Diagnostic only, reported as such.
#
#   2. entropy_coef 0.006 -> 0.0005. Evidence, not taste: gSDE noise std climbed monotonically
#      0.75 -> 2.65 across the main run while every task reward term stayed flat -- the entropy
#      bonus dominating a weak advantage signal. At std 2.65 against an action clip of +-1 the
#      sampled actions are largely saturated, so the policy cannot exploit what it finds.
#
# Everything else is held fixed so the comparison stays interpretable: same banks, same reward,
# same 2048 envs, same seed, same deterministic evaluation afterwards.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
cd "$REPO" || exit 1

ITERS=${ITERS:-2000}
ENVS=${ENVS:-2048}
SEED=${SEED:-42}
ENTROPY=${ENTROPY:-0.0005}
TASK=OmniReset-UR10eDelto-CubeStackNearGoal-RelCartesianOSC-State-v0

echo "=== diagnostic: near-goal only, entropy_coef=$ENTROPY, $ENVS envs, seed $SEED ==="
"$CS/run.sh" scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$TASK" --num_envs "$ENVS" --max_iterations "$ITERS" --seed "$SEED" --headless \
  env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 \
  env.sim.physx.gpu_collision_stack_size=1073741824 \
  agent.algorithm.entropy_coef="$ENTROPY" \
  > "$CS/logs/train_diagnostic.log" 2>&1 &
TRAIN_PID=$!
echo "training pid $TRAIN_PID"

# Confirm the override actually took. A silently ignored hydra override would make this whole run
# uninterpretable, and it would look completely normal in the log.
sleep 180
if grep -qE 'entropy_coef' "$CS/logs/train_diagnostic.log"; then
  grep -oE 'entropy_coef[^,}]*' "$CS/logs/train_diagnostic.log" | head -2
fi
grep -oE 'Learning iteration [0-9]+' "$CS/logs/train_diagnostic.log" | tail -1 || echo "no iterations yet"
echo "=== launched; monitor logs/train_diagnostic.log ==="
