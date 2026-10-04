#!/usr/bin/env bash
# Wait for the 900-iteration carry run to finish, then do the whole downstream pipeline unattended:
# acceptance rollout -> A/B diff -> video render -> report regeneration -> queue the reproduction arm.
#
# WHY THIS EXISTS. Tonight the GPU sat idle for 2h11m because the acceptance rollout finished and
# nothing was watching for it: the monitor was watching a wrapper process, the wrapper exited, and the
# watch died with it. A finished checkpoint that nobody acts on is worth exactly as much as no
# checkpoint. So the actions that follow the run are scripted rather than left to whoever happens to
# be awake.
#
# It waits on the REAL python process, matched on comm=python plus the train.py argv. It must never
# use `pkill -f` or `pgrep -f` on a pattern that appears in its own command line -- that has killed
# the wrong process here before.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
RUNS=/home/mattephi/Work/Rubitek/UWLab-cubestack/logs/rsl_rl/ur10e_delto_omnireset_agent
RVENV=/tmp/claude-1000/-home-mattephi-Work-Rubitek/d817c68b-2665-4f13-965b-515b6cadcd04/scratchpad/rvenv/bin/python
cd "$CS" || exit 1
say(){ echo "AFTER $* $(date -Is)"; }

train_pid(){ ps -eo pid,comm,args | awk '$2 ~ /^python/ && /train\.py/ {print $1; exit}'; }

say "WAITING for the training process to exit"
while :; do
  p=$(train_pid)
  [ -z "$p" ] && break
  sleep 60
done
say "TRAINING GONE"

# The run is only usable if it actually reached the end. A crash at iteration 300 leaves a checkpoint
# too, and silently evaluating it as though it were the finished article would be a lie by omission.
LAST_IT=$(grep -aoE "Learning iteration [0-9]+/[0-9]+" "$CS/logs/long_carry.log" 2>/dev/null | tail -1)
DONE_LINE=$(grep -a "AB_DONE long_carry" "$CS/logs/long_carry.out" 2>/dev/null | tail -1)
say "LAST=$LAST_IT  DONE=${DONE_LINE:-<none: the run did NOT finish cleanly>}"

D=$(ls -dt "$RUNS"/*_long_carry 2>/dev/null | head -1)
[ -n "$D" ] || { say "NO RUN DIR -- stopping"; exit 1; }
CK=$(ls -t "$D"/model_*.pt 2>/dev/null | head -1)
[ -n "$CK" ] || { say "NO CHECKPOINT in $D -- stopping"; exit 1; }
say "CHECKPOINT $CK"

say "STEP 1/4 acceptance rollout on C3"
"$CS/accept_policy.sh" "$CK" ObjectAnywhereEEGrasped longcarry 2>&1 | tail -40

say "STEP 2/4 A/B diff against the weight-0 baseline"
python3 "$CS/ab_diff.py" "baseline_w0=$CS/results/poses_ablc3.npz" \
                         "long_carry=$CS/results/poses_longcarry.npz" 2>&1 | tail -30

say "STEP 3/4 render the median environment"
if [ -x "$RVENV" ] && [ -f "$CS/results/poses_longcarry.npz" ]; then
  mkdir -p "$CS/results/film_longcarry_c3"
  RENDER_W=960 RENDER_H=720 RENDER_FPS=15 timeout 900 "$RVENV" "$CS/render_meshes.py" \
    longcarry "$CS/results/film_longcarry_c3/rollout.mp4" median 2>&1 | grep -aE "RENDER|Error" | tail -6
else
  say "RENDER SKIPPED (no pyrender venv at $RVENV, or no poses file)"
fi

say "STEP 4/4 regenerate the report"
python3 "$CS/make_report.py" 2>&1 | tail -2

# The reproduction arm has been queued behind the GPU all night. Run it last, so a finished result is
# never waiting on a confirmation run.
say "QUEUING the weight-0 reproduction arm (200 iters)"
"$CS/ab_hold_bonus.sh" 0 ab_nobonus2 200 512 > "$CS/logs/ab_nobonus2.out" 2>&1
say "ALL DONE"
