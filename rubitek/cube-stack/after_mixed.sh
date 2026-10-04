#!/usr/bin/env bash
# Wait for the mixed-curriculum resume run to finish, then harvest it unattended:
# acceptance rollout on BOTH families -> A/B diff -> video render -> report regeneration.
#
# WHY. Twice tonight a finished job sat idle because nothing was watching: once for 2h11m after an
# acceptance rollout completed, and again when a monitor died with the wrapper it was watching. A
# checkpoint nobody evaluates is worth the same as no checkpoint. So everything downstream of the run
# is scripted rather than left to whoever is awake.
#
# It waits on the REAL python process (comm=python plus the train.py argv). It must never use
# `pkill -f`/`pgrep -f` on a pattern that appears in its own command line -- that has killed the wrong
# process in this project before.
#
# C4 IS EVALUATED TOO, AND THAT NEEDS A WARNING ATTACHED. The C4 reset (ObjectPartiallyAssembled)
# starts the cube ALREADY SEATED within tolerance, so its success rate is mostly the reset
# distribution, not an achievement -- median cube displacement over 150 steps has been measured at
# 0.00000 m with success going 39/64 at step 0 to 38/64 at step 149. Any C4 number below must be read
# as "did it avoid destroying a finished stack", never as "did it do the task". C3 is the real task.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
RUNS=/home/mattephi/Work/Rubitek/UWLab-cubestack/logs/rsl_rl/ur10e_delto_omnireset_agent
RVENV=/tmp/claude-1000/-home-mattephi-Work-Rubitek/d817c68b-2665-4f13-965b-515b6cadcd04/scratchpad/rvenv/bin/python
cd "$CS" || exit 1
say(){ echo "AFTER_MIXED $* $(date -Is)"; }
train_pid(){ ps -eo pid,comm,args | awk '$2 ~ /^python/ && /train\.py/ {print $1; exit}'; }

say "WAITING for training to exit"
while :; do p=$(train_pid); [ -z "$p" ] && break; sleep 60; done
say "TRAINING GONE"

LAST_IT=$(grep -aoE "Learning iteration [0-9]+/[0-9]+" "$CS/logs/mixed_resume.log" 2>/dev/null | tail -1)
DONE=$(grep -a "RESUME_DONE mixed_resume" "$CS/logs/mixed_resume.out" 2>/dev/null | tail -1)
say "LAST=$LAST_IT DONE=${DONE:-<none: run did NOT finish cleanly>}"

D=$(ls -dt "$RUNS"/*_mixed_resume 2>/dev/null | head -1)
[ -n "$D" ] || { say "NO RUN DIR -- stopping"; exit 1; }
CK=$(ls -t "$D"/model_*.pt 2>/dev/null | head -1)
[ -n "$CK" ] || { say "NO CHECKPOINT -- stopping"; exit 1; }
say "CHECKPOINT $CK"

say "STEP 1/4 acceptance on C3 (the real carry task)"
"$CS/accept_policy.sh" "$CK" ObjectAnywhereEEGrasped mixedc3 2>&1 | tail -40

say "STEP 2/4 acceptance on C4 (starts already seated -- read as 'did not wreck the stack')"
"$CS/accept_policy.sh" "$CK" ObjectPartiallyAssembledEEGrasped mixedc4 2>&1 | tail -30

say "STEP 3/4 diff C3 against the weight-0 C3-only baseline"
python3 "$CS/ab_diff.py" "baseline_w0=$CS/results/poses_ablc3.npz" \
                         "mixed_c3=$CS/results/poses_mixedc3.npz" 2>&1 | tail -30

say "STEP 4/4 render C3 (median environment, not a flattering one) and rebuild the report"
if [ -x "$RVENV" ] && [ -f "$CS/results/poses_mixedc3.npz" ]; then
  mkdir -p "$CS/results/film_mixed_c3"
  RENDER_W=960 RENDER_H=720 RENDER_FPS=15 timeout 900 "$RVENV" "$CS/render_meshes.py" \
    mixedc3 "$CS/results/film_mixed_c3/rollout.mp4" median 2>&1 | grep -aE "RENDER|Error" | tail -6
else
  say "RENDER SKIPPED (no pyrender venv, or no poses file)"
fi
python3 "$CS/make_report.py" 2>&1 | tail -2
say "ALL DONE"
