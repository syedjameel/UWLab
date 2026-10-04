#!/usr/bin/env bash
# Wait for mixed_resume2 to finish, then harvest its final checkpoint via harvest_ck.sh.
# Watches the REAL python process (comm=python + train.py argv) -- never pkill -f, which would match
# this script's own command line.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
RUNS=/home/mattephi/Work/Rubitek/UWLab-cubestack/logs/rsl_rl/ur10e_delto_omnireset_agent
cd "$CS" || exit 1
say(){ echo "AFTER2 $* $(date -Is)"; }
say "WAITING for mixed_resume2 to exit"
while :; do
  p=$(ps -eo pid,comm,args | awk '$2 ~ /^python/ && /train\.py/ {print $1; exit}')
  [ -z "$p" ] && break
  sleep 60
done
say "TRAINING GONE last=$(grep -aoE 'Learning iteration [0-9]+/[0-9]+' "$CS/logs/mixed_resume2.log" 2>/dev/null | tail -1)"
D=$(ls -dt "$RUNS"/*_mixed_resume2 2>/dev/null | head -1)
[ -n "$D" ] || { say "NO RUN DIR"; exit 1; }
CK=$(ls -t "$D"/model_*.pt 2>/dev/null | head -1)
[ -n "$CK" ] || { say "NO CHECKPOINT"; exit 1; }
say "HARVESTING $CK"
"$CS/harvest_ck.sh" "$CK" mixed2 ObjectAnywhereEEGrasped
say "ALL DONE"
