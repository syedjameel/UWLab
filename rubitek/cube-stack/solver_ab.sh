#!/usr/bin/env bash
# Is the physics cost the SOLVE or the NARROWPHASE? A short A/B that decides where to spend effort.
#
# WHY IT MATTERS. Collection is 98% of wall-clock (56 s of a 56.5 s iteration at 2048 envs), and the
# UR10e DELTO asset is the reason: local/Robots/Ur10eDelto/ur10e_delto.usd authors 28 SDF mesh
# colliders and 12 convex hulls, with enabled_self_collisions=True and
# solver_position_iteration_count=36 over decimation 12 -- 432 position-solve iterations per control
# step across 40 colliders on a 26-DOF articulation. The UR5e branch of this same family already
# migrated off SDF (ur5e_delto_hullfix3.usd: 25 convexHull, 3 convexDecomposition, zero SDF) and
# ur5e_delto_cfg.py refuses to start on the legacy plant. The UR10e path was never migrated.
#
# The two fixes cost very differently:
#   solver iterations   one config field, reversible, no asset work
#   collider migration  re-author the USD, re-validate stability, and RE-RECORD every reset bank,
#                       because the banks were recorded on the SDF plant
# So measure which one the time is actually in before paying for the second.
#
# WHAT IS MEASURED. Throughput at four solver-iteration counts, everything else identical. If
# throughput scales with the count, the solve dominates and the cheap knob is the answer. If it
# barely moves, narrowphase dominates and only the collider swap will help.
#
# THIS IS A THROUGHPUT PROBE, NOT A DECISION. Lowering solver iterations changes the plant the reset
# banks were recorded on, so a win here still has to pass a fidelity check (see solver_fidelity.sh)
# before it is adopted: a grasp that was stable at 36 iterations may slip at 8, which would show up
# as a drop in evaluated success rather than as an error.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
cd "$REPO" || exit 1

ENVS=${ENVS:-2048}
ITERS=${ITERS:-4}
OUT=$CS/results/solver_ab.md

echo "# Solver-iteration throughput A/B" > "$OUT"
echo >> "$OUT"
echo "$ENVS envs, $ITERS iterations each, everything else identical." >> "$OUT"
echo >> "$OUT"
echo "| solver_position_iteration_count | steps/s | s/iteration |" >> "$OUT"
echo "|---|---|---|" >> "$OUT"

for IT in 36 16 8 4; do
  echo "=== solver_position_iteration_count=$IT ==="
  LOG=$CS/logs/solver_ab_$IT.log
  "$CS/run.sh" scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-UR10eDelto-CubeStackNearGoal-RelCartesianOSC-State-v0 \
    --num_envs "$ENVS" --max_iterations "$ITERS" --seed 42 --headless \
    --run_name "solver_ab_$IT" \
    env.sim.physx.gpu_collision_stack_size=1610612736 \
    env.scene.robot.spawn.articulation_props.solver_position_iteration_count="$IT" \
    agent.algorithm.entropy_coef=0.0005 \
    > "$LOG" 2>&1
  SPS=$(tr '\r' '\n' < "$LOG" | grep -oE 'Computation: [0-9]+ steps/s' | tail -1 | grep -oE '[0-9]+')
  SEC=$(tr '\r' '\n' < "$LOG" | grep -oE 'Iteration time: [0-9.]+s' | tail -1 | grep -oE '[0-9.]+')
  if [ -z "${SPS:-}" ]; then
    echo "| $IT | FAILED | see logs/solver_ab_$IT.log |" >> "$OUT"
    echo "  FAILED:"; tail -6 "$LOG" | tr '\r' '\n' | tail -4
  else
    echo "| $IT | $SPS | $SEC |" >> "$OUT"
    echo "  $IT -> $SPS steps/s, ${SEC}s/iter"
  fi
done

echo
cat "$OUT"
