#!/usr/bin/env bash
# Run both A/B arms' final checkpoints through the IDENTICAL acceptance harness and print them
# side by side.
#
# WHY A SCRIPT AND NOT AN AD-HOC COMMAND. The two things that have gone wrong repeatedly on this
# task are (a) comparing runs that silently differed in configuration, and (b) losing the exact
# invocation afterwards. Both arms here come from ab_hold_bonus.sh, which varies only
# UWLAB_HOLD_BONUS_WEIGHT; this script then evaluates them through one code path
# (accept_policy.sh -> trace_full_poses.py -> the scene's own object_hand_s contact sensor) so the
# evaluation cannot differ either. It also re-prints each run's stored config so the claim
# "these are matched" is checkable from the output rather than asserted.
#
# Runs are SERIALISED: one 4070 Ti, and two concurrent Isaac processes contend for it and corrupt
# both measurements.
#
# Usage:  ./compare_ab.sh [family]
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
RUNS=/home/mattephi/Work/Rubitek/UWLab-cubestack/logs/rsl_rl/ur10e_delto_omnireset_agent
FAMILY=${1:-ObjectAnywhereEEGrasped}

newest_for(){ ls -dt "$RUNS"/*_"$1" 2>/dev/null | head -1; }

echo "############ A/B ACCEPTANCE  family=$FAMILY  $(date -Is)"
for arm in ab_bonus05 ab_nobonus2; do
  d=$(newest_for "$arm")
  if [ -z "$d" ] || [ ! -d "$d" ]; then echo "!! no run dir for $arm -- skipping"; continue; fi
  ck="$d/model_199.pt"
  [ -f "$ck" ] || ck=$(ls -t "$d"/model_*.pt 2>/dev/null | head -1)
  [ -f "$ck" ] || { echo "!! no checkpoint in $d"; continue; }
  echo
  echo "======== ARM $arm"
  echo "  run dir   : $d"
  echo "  checkpoint: $(basename "$ck")"
  # Re-print the stored config so "matched" is verifiable from this output, not asserted.
  echo "  -- stored config --"
  grep -aA3 "reset_types:" "$d/params/env.yaml" 2>/dev/null | grep -aE "^      -" | sed 's/^/     family:/'
  grep -aoE "num_envs: [0-9]+" "$d/params/env.yaml" 2>/dev/null | sed 's/^/     /'
  grep -aoE "(seed|max_iterations|entropy_coef): [0-9.]+" "$d/params/agent.yaml" 2>/dev/null | sed 's/^/     /'
  grep -aA4 "rj_dg_\[1-5\]_1:" "$d/params/env.yaml" 2>/dev/null | head -1 | sed 's/^/     effort_limit_sim /'
  echo "  -- hold_bonus present in training log? --"
  echo "     Episode_Reward/hold_bonus lines: $(grep -acE 'Episode_Reward/hold_bonus' "$CS/logs/$arm.log" 2>/dev/null)"
  echo "     'bonus DISABLED' lines        : $(grep -acE 'hold.carry bonus DISABLED' "$CS/logs/$arm.log" 2>/dev/null)"
  echo
  "$CS/accept_policy.sh" "$ck" "$FAMILY" "$arm"
done
echo
echo "############ A/B ACCEPTANCE DONE $(date -Is)"
