#!/usr/bin/env bash
# Wait for the endurance probe, extend it upward, pick the winner FROM THE LOG, then train locally.
#
# Round 1 showed endurance is monotonic in the effort cap -- in_hand at step 150 went 0.047 (shipped)
# -> 0.336 (x3) -> 0.523 (x10) -- which is the signature of an actuator that was saturated, not weak.
# Round 2 pushes further to find where it stops helping. It must stop somewhere: a hand that closes
# hard enough will EJECT a 34 mm cube rather than hold it, which is the original failure mode from
# the substep ramp. The probe measures ejection directly (in_hand collapses while tip force stays
# high), so the data decides rather than an argument about what "reasonable" force is.
#
# For scale: x10 is ~22 N at the fingertip, against a DG-5F rated at ~7 kg (~68 N) of grip and a USD
# that authors 30 N*m per joint. Raising this cap is not exceeding the hardware -- the shipped
# 0.06-0.17 N*m is a hand-picked backstop, ~100x below what the real hand does.
#
# Why pick from the log rather than hard-coding: writing the answer in advance is how a "fix" gets
# applied that the measurement never supported.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
LOG=$CS/logs/probe_local.log

echo "=== CHAIN waiting for probe round 1 $(date -Is)"
while pgrep -f '[p]robe_grip_local.sh' >/dev/null 2>&1; do sleep 20; done
echo "=== CHAIN round 1 finished $(date -Is)"

echo "=== CHAIN round 2 (upper end of the effort sweep) $(date -Is)"
CONDS="eff10:10:0:0.05 eff20:20:0:0.05 eff40:40:0:0.05 eff10bias:10:0:0.15" \
  "$CS/probe_grip_local.sh" >> "$LOG" 2>&1
echo "=== CHAIN round 2 finished $(date -Is)"

# Best condition = highest in_hand at step 150. That is the endurance number the whole night turns
# on: the grasp has to survive to step 160, where success is scored.
BEST=$(grep -aE "step= 150" "$LOG" | sed -E 's/PROBE ([^ ]+) .*in_hand=([0-9.]+).*/\2 \1/' | sort -rn | head -1)
SCORE=${BEST%% *}; NAME=${BEST##* }
echo "=== CHAIN best condition: $NAME (in_hand@150=$SCORE)"

COND=$(grep -aE "^=== COND $NAME " "$LOG" | head -1)
EFF=$(sed -E 's/.*effort=x([0-9.]+).*/\1/' <<<"$COND")
TUCK=$(sed -E 's/.*tuck=([0-9.]+).*/\1/' <<<"$COND")
BIAS=$(sed -E 's/.*bias=([0-9.]+).*/\1/' <<<"$COND")
echo "=== CHAIN config: effort=x$EFF tuck=$TUCK bias=$BIAS"

# A condition that never cleared the shipped baseline is not worth training on; say so loudly rather
# than burning the night proving it.
awk -v s="$SCORE" 'BEGIN{ if (s+0 < 0.15) print "=== CHAIN WARNING: best endurance " s " is near the 0.047 baseline; C3 may still fail" }'

export UWLAB_HAND_EFFORT_SCALE="$EFF" UWLAB_TUCK_OTHERS="$TUCK" UWLAB_GRIP_BIAS_RAD="$BIAS"
export NAME=local_${NAME}
echo "=== CHAIN starting local training $(date -Is)"
exec "$CS/train_pinch_local.sh"
