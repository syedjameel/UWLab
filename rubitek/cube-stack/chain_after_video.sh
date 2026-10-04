#!/usr/bin/env bash
# GPU queue: video, then the headline re-measurement, then the throughput A/B.
#
# ORDER IS DELIBERATE. The re-measurement has to come before anything else that wants the card,
# because ProgressContext.reset was just scoped to env_ids: it used to zero EVERY environment's
# consecutive-success counter whenever ANY environment reset, so
# `consecutive_success_state_with_min_length` fired only in the synchronised wave after a global
# reset. Every episode LENGTH measured before that fix is therefore a function of --num_envs, and
# the genuine/pre-solved split is conditioned on episode length. The headline numbers are not
# trustworthy until they are taken again.
#
# LAUNCH THIS FROM A FROZEN COPY. It sleeps for tens of minutes, and bash reads a script
# incrementally by byte offset -- rewriting the file underneath a running instance resumes execution
# mid-statement and kills it with a syntax error on a line that is perfectly valid. That is exactly
# how the b3 autostart died. Use:
#     cp chain_after_video.sh "$SCRATCH/run.sh" && nohup bash "$SCRATCH/run.sh" &
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack

echo "=== waiting for the rollout to finish ==="
while ! grep -qE 'CUT\]|nothing to cut|RENDER FAILED' "$CS/logs/make_video_neargoal.log" 2>/dev/null; do
  sleep 30
done
echo "=== rollout done $(date +%H:%M) ==="
tail -20 "$CS/logs/make_video_neargoal.log"

echo
echo "=== re-measuring the headline with the success-counter fix ==="
N=800 "$CS/verify_neargoal.sh" 2>&1 | tail -20

echo
# The solver throughput A/B was the previous occupant of this slot. It is deferred: a 2-4x speedup
# would compound, but acting on it also means re-recording every reset bank (they were recorded at
# 36 solver iterations), and the question below is worth more right now.
echo "=== do the fingers close? visual check on the newest money-run checkpoint ==="
BOX=b1 N=14 "$CS/grasp_check.sh" 2>&1 | tail -25
echo "=== chain done $(date +%H:%M) ==="
