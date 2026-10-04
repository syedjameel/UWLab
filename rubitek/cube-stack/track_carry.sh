#!/usr/bin/env bash
# Track whether the CARRY is being learned, on a family whose success rate may stay at zero for
# hundreds of iterations.
#
# Stable-Grasp success is a step function: nothing until the policy can both hold and place, then
# something. Waiting on it gives no gradient of information, and it is the number that has already
# read 0.0000 for a thousand iterations across two configurations. Three others move earlier and
# say what is actually happening:
#
#   final cube height   -- 21 mm is the table. Above it means the policy still has the cube at the
#                          end of the episode, which before tonight's grip fix was never true.
#   frac_on_table       -- the same thing as a fraction, so a trend is visible.
#   |a| arm             -- whether the policy has started MOVING. A carry cannot begin until it does,
#                          and a rising |a| with a collapsing cube height is the failure mode that
#                          matters: the policy shaking the cube loose rather than carrying it.
#
# Together they distinguish "learning the carry" from "learning to stand still", long before either
# shows up in a success rate.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
UWLAB_PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv/bin/python
BOX=${BOX:-b5}
TASK=${TASK:-OmniReset-UR10eDelto-CubeStackGrip-RelCartesianOSC-State-Play-v0}
GAINS=${GAINS:-"--motion_stiffness 1000 --motion_damping 1.0"}
EVERY=${EVERY:-2700}
S=/home/mattephi/.cache/tmp/claude-1000/-home-mattephi-Work-Rubitek-UWLab/1df539be-ae09-4745-af83-4e088ecd3da9/scratchpad
EV=$S/eval_policy_${BOX}.py

# A REAL LOCK, NOT A POLL. `while pgrep ...; do sleep; done` is check-then-act: three trackers and
# the harvest can all observe an idle GPU in the same instant and launch together. That is exactly
# what happened -- an eval died with "CUDA out of memory ... Process 623387 has 8.80 GiB in use"
# while another job held the card. flock serialises them properly.
LOCK=/tmp/cubestack_gpu.lock
touch "$LOCK"

while true; do
  # RE-COPY THE EVALUATOR EVERY CYCLE. Copying once at startup means every fix to eval_policy.py
  # needs a manual restart of every tracker to take effect -- which is how a zero-initialised
  # progress metric kept reporting "-186 mm, 0% closer" from an evaluator that had already been
  # fixed on disk. Frozen copies protect a RUN in flight; they must not outlive it.
  cp "$CS/eval_policy.py" "$EV"
  "$CS/pull_checkpoint.sh" "$BOX" >/dev/null 2>&1
  CK=$(ls -t "$CS/results/ck_$BOX"/model_*.pt 2>/dev/null | head -1)
  if [ -n "${CK:-}" ]; then
    OUT="$CS/results/track_${BOX}_$(basename "$CK" .pt).json"
    # STAMP BEFORE THE RUN. If the eval dies, the JSON at this path is a PREVIOUS run's and printing
    # it reports a stale measurement as a fresh one -- which is how an OOM crash came back looking
    # like a repeat reading identical to three decimals.
    STAMP=$(mktemp)
    rc=0
    flock "$LOCK" "$CS/run.sh" "$EV" --task "$TASK" --checkpoint "$CK" --num_envs 256 --episodes 256 \
      --reset_type ObjectAnywhereEEGrasped --collision_stack_mb 1536 --headless $GAINS \
      --out "$OUT" > "$CS/logs/track_$BOX.log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ] || [ ! "$OUT" -nt "$STAMP" ]; then
      rm -f "$STAMP"
      echo "$BOX: eval did not produce a fresh result (exit $rc) -- $(tr '\r' '\n' < "$CS/logs/track_$BOX.log" | grep -aoE 'OutOfMemoryError|CUDA error[^,]*|Traceback' | tail -1)"
      sleep "$EVERY"
      continue
    fi
    rm -f "$STAMP"
    "$UWLAB_PY" - "$CS/results/track_${BOX}_$(basename "$CK" .pt).json" "$BOX" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"{sys.argv[2]}: eval produced no JSON ({e})"); raise SystemExit
z = d["final_object_z_m"]
pr = d.get("goal_progress_m") or {}
if pr:
    print("%s %s: success %.2f%% | goal progress %+.1f mm (p90 %+.1f, closer %.0f%%) "
          "| height %+.1f mm | on table %.0f%% | |a| arm %.3f"
          % (sys.argv[2], d["checkpoint"].split("/")[-1], d["success_held_at_end_pct"],
             pr["median"] * 1000, pr["p90"] * 1000, (pr.get("frac_closer") or 0) * 100,
             (d.get("object_dz_m") or {}).get("median", 0) * 1000,
             z["frac_on_table"] * 100, d["mean_abs_action_arm"]))
else:
    # JSON written before the progress fields existed
    print("%s %s: success %.2f%% | cube height %.1f mm | on table %.1f%% | |a| arm %.3f hand %.3f"
          % (sys.argv[2], d["checkpoint"].split("/")[-1],
             d["success_held_at_end_pct"], z["median"] * 1000, z["frac_on_table"] * 100,
             d["mean_abs_action_arm"], d["mean_abs_action_hand"]))
PY
  else
    echo "$BOX: no checkpoint pulled yet"
  fi
  sleep "$EVERY"
done
