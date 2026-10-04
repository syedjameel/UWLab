#!/usr/bin/env bash
# Does the near-goal policy actually STACK the cube, or does it inherit resets that were already
# stacked?
#
# The near-goal-only diagnostic moved end-of-episode success from 1.9% to ~20% on two independent
# seeds (42 local, 7 remote), which settles that the pipeline can learn. It does NOT settle that the
# successes are real: the termination rule
# consecutive_success_state_with_min_length(num_consecutive_successes=5, min_episode_length=10)
# lets an episode end successfully at step 10, and it does so exactly when the predicate already
# held at reset. An earlier run's three "successes" were all exactly 10 steps long.
#
# So the number that matters is the success rate restricted to episodes that ran LONGER than 10
# steps. eval_policy.py now reports it directly. Untrained model_0 and trained model_300 are graded
# identically, from the same run directory, so the initialisation is shared and the comparison is
# about training alone.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
RUN=/home/mattephi/Work/Rubitek/UWLab-cubestack/logs/rsl_rl/ur10e_delto_omnireset_agent/2026-08-31_11-16-08
TASK=OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-Play-v0
N=${N:-800}

for CK in model_0 model_300; do
  echo "=== $CK / ObjectPartiallyAssembledEEGrasped / n=$N ==="
  # Delete first. Otherwise a crashed eval leaves the previous JSON untouched, the console prints
  # FAILED, and the report keeps serving the old numbers as if they were current.
  rm -f "$CS/results/verify_neargoal_$CK.json"
  "$CS/run.sh" "$CS/eval_policy.py" \
    --task "$TASK" --checkpoint "$RUN/$CK.pt" \
    --num_envs 256 --episodes "$N" \
    --reset_type ObjectPartiallyAssembledEEGrasped \
    --collision_stack_mb 1536 --headless \
    --out "$CS/results/verify_neargoal_$CK.json" \
    > "$CS/logs/verify_$CK.log" 2>&1
  grep -E 'GENUINE|EVAL_RESULT' "$CS/logs/verify_$CK.log" \
    || { echo "  FAILED:"; tail -8 "$CS/logs/verify_$CK.log" | tr '\r' '\n' | tail -5; }
done
echo "=== verify done ==="
