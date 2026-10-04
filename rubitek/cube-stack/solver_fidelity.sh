#!/usr/bin/env bash
# Does lowering solver_position_iteration_count change the TASK, or only its cost?
#
# solver_ab.sh measures throughput. It cannot be acted on alone, because every reset-state bank was
# recorded on the 36-iteration plant: a grasp that PhysX held stable at 36 may slip at 8, and that
# would not raise an error -- it would quietly lower the success rate and invalidate every number
# measured since. The banks store joint positions and object poses, not contact state, so they are
# replayed under whatever contact physics is configured at load.
#
# This grades the SAME checkpoint under each setting, on the same reset family, with the same episode
# count. model_300 is the right probe because its behaviour is already pinned down precisely:
# 9.26% genuine success (episodes running past the 10-step pre-solved boundary) over n=821, against
# an untrained 1.43% over n=840. A setting that leaves genuine success inside sampling error of
# 9.26% is replaying the banks faithfully; one that drops it is changing the task.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
RUN=/home/mattephi/Work/Rubitek/UWLab-cubestack/logs/rsl_rl/ur10e_delto_omnireset_agent/2026-08-31_11-16-08
TASK=OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-Play-v0
N=${N:-600}
OUT=$CS/results/solver_fidelity.md

echo "# Solver-iteration fidelity check" > "$OUT"
echo >> "$OUT"
echo "Same checkpoint (model_300), same reset family, n=$N each. Reference at 36 iterations:" >> "$OUT"
echo "9.26% genuine over n=821." >> "$OUT"
echo >> "$OUT"
echo "| solver iterations | held at end | genuine | median pos err |" >> "$OUT"
echo "|---|---|---|---|" >> "$OUT"

for IT in 36 16 8; do
  echo "=== fidelity at solver_position_iteration_count=$IT ==="
  LOG=$CS/logs/solver_fid_$IT.log
  # Delete first: a crashed eval used to leave the PREVIOUS run's JSON in place, and "the file
  # exists" was the only success test, so stale numbers were reported as a fresh measurement.
  rm -f "$CS/results/solver_fid_$IT.json"
  "$CS/run.sh" "$CS/eval_policy.py" \
    --task "$TASK" --checkpoint "$RUN/model_300.pt" \
    --num_envs 256 --episodes "$N" \
    --reset_type ObjectPartiallyAssembledEEGrasped \
    --collision_stack_mb 1536 --headless \
    --out "$CS/results/solver_fid_$IT.json" \
    --solver_iters "$IT" \
    > "$LOG" 2>&1
  if [ -f "$CS/results/solver_fid_$IT.json" ]; then
    /home/mattephi/Work/Rubitek/uwlab-env/.venv/bin/python - "$IT" "$CS/results/solver_fid_$IT.json" "$OUT" <<'PY'
import json, sys
it, path, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path))
with open(out, "a") as f:
    f.write(f"| {it} | {d['success_held_at_end_pct']:.1f}% | {d['genuine_success_pct']:.2f}% | "
            f"{d['position_error_m']['median']*1000:.1f} mm |\n")
print(f"  {it}: held {d['success_held_at_end_pct']:.1f}%  genuine {d['genuine_success_pct']:.2f}%")
PY
  else
    echo "| $IT | FAILED | | |" >> "$OUT"
    echo "  FAILED:"; tail -6 "$LOG" | tr '\r' '\n' | tail -4
  fi
done

echo
cat "$OUT"
