#!/usr/bin/env bash
# Pull the H100's newest curriculum log to where make_report.py expects it.
#
# The report reads cube-stack/logs/train_h100.log, but the training log lives on the remote box and
# its name moves (train_curriculum.log, then _2, then _3, as failed runs were re-queued). Resolving
# "newest by mtime" here rather than hardcoding a name is the same mistake that left the H100 monitor
# watching a file nothing was writing to any more.
#
# Also pulls any checkpoint newer than what has already been fetched, so films can be rendered
# locally -- the H100 has no graphics engine and segfaults on any camera-enabled run.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
DEST_LOG=$CS/logs/train_h100.log
CK_DIR=$CS/train/ck_pull
mkdir -p "$CS/logs" "$CK_DIR"

F=$(timeout 40 ssh -o BatchMode=yes h100-server 'ls -t ~/rubitek/logs/train_curriculum*.log 2>/dev/null | head -1' | tr -d '\r')
if [ -z "$F" ]; then echo "no remote curriculum log yet"; exit 0; fi
echo "newest remote log: $F"
timeout 300 ssh -o BatchMode=yes h100-server "cat '$F'" > "$DEST_LOG" 2>/dev/null \
  && echo "pulled $(wc -l < "$DEST_LOG") lines -> $DEST_LOG"

# SKIP_CK=1 fetches the log ONLY. The checkpoint pull below is ~8 MB over a ~12 KB/s link -- about
# eleven minutes of blocking -- which is fine at 02:00 and unacceptable at 06:00 when the report is
# being assembled. Use SKIP_CK=1 for a quick log refresh and run the pull separately in background.
if [ "${SKIP_CK:-0}" = "1" ]; then
  echo "SKIP_CK=1: log only, checkpoint not pulled"
  exit 0
fi

# Checkpoints: newest first, and only if it is not already here. These are ~8 MB each over a
# ~12 KB/s link (about 11 minutes), so pull one, not the set.
NEWEST=$(timeout 40 ssh -o BatchMode=yes h100-server \
  'find ~/rubitek/UWLab/logs/rsl_rl -name "model_*.pt" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d" " -f2-' | tr -d '\r')
if [ -n "$NEWEST" ]; then
  B=$(basename "$NEWEST")
  if [ -f "$CK_DIR/h100_$B" ]; then
    echo "checkpoint h100_$B already local"
  else
    echo "pulling $NEWEST ($(date -Is))"
    timeout 3600 ssh -o BatchMode=yes h100-server "cat '$NEWEST'" > "$CK_DIR/h100_$B.part" 2>/dev/null \
      && mv "$CK_DIR/h100_$B.part" "$CK_DIR/h100_$B" \
      && echo "pulled $(stat -c%s "$CK_DIR/h100_$B") bytes -> $CK_DIR/h100_$B"
  fi
else
  echo "no checkpoint on the box yet"
fi
