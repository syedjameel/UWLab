#!/usr/bin/env bash
# Roll out ONE checkpoint through the contact harness, render it, and rebuild the report.
#
# WHY A SEPARATE SCRIPT. after_mixed.sh is already running (waiting on the training PID) and harvests
# the MIXED run's final checkpoint. But if the mixed curriculum fails its pre-committed bar, the best
# policy of the night is long_carry/model_400 -- the C3-only policy that grips at 11.6 N and moves the
# cube 236 mm -- and that checkpoint has never been rolled out or rendered. This harvests any
# checkpoint on demand.
#
# It is deliberately NOT an edit to after_mixed.sh: bash reads a script incrementally as it executes,
# so editing a running script can corrupt its execution mid-flight.
#
# Usage:  ./harvest_ck.sh <checkpoint.pt> <name> [family]
#   ./harvest_ck.sh .../long_carry/model_400.pt long400
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
RVENV=/tmp/claude-1000/-home-mattephi-Work-Rubitek/d817c68b-2665-4f13-965b-515b6cadcd04/scratchpad/rvenv/bin/python
cd "$CS" || exit 1

CK=${1:?usage: harvest_ck.sh <checkpoint.pt> <name> [family]}
NAME=${2:?usage: harvest_ck.sh <checkpoint.pt> <name> [family]}
FAMILY=${3:-ObjectAnywhereEEGrasped}
[ -f "$CK" ] || { echo "HARVEST no such checkpoint: $CK"; exit 1; }
say(){ echo "HARVEST $* $(date -Is)"; }

# Refuse to run while an Isaac job holds the GPU: two concurrent Isaac processes on one 4070 Ti
# contend and corrupt BOTH measurements, which is not a failure that announces itself.
busy=$(ps -eo pid,comm,args | awk '$2 ~ /^python/ && (/train\.py/ || /trace_full_poses/) {print $1; exit}')
[ -n "$busy" ] && { say "REFUSING: python job $busy still holds the GPU"; exit 1; }

say "START $NAME from $(basename "$CK") family=$FAMILY"

say "STEP 1/3 acceptance rollout"
"$CS/accept_policy.sh" "$CK" "$FAMILY" "$NAME" 2>&1 | tail -40

say "STEP 2/3 diff against the weight-0 C3-only baseline"
if [ -f "$CS/results/poses_$NAME.npz" ]; then
  python3 "$CS/ab_diff.py" "baseline_w0=$CS/results/poses_ablc3.npz" \
                           "$NAME=$CS/results/poses_$NAME.npz" 2>&1 | tail -28
else
  say "NO POSES FILE -- the rollout did not produce results/poses_$NAME.npz"
fi

say "STEP 3/3 render the MEDIAN environment and rebuild the report"
# median, never the best environment: picking a flattering rollout is how an earlier set of clips
# came to misrepresent what the policy actually does.
if [ -x "$RVENV" ] && [ -f "$CS/results/poses_$NAME.npz" ]; then
  mkdir -p "$CS/results/film_$NAME"
  RENDER_W=960 RENDER_H=720 RENDER_FPS=15 timeout 900 "$RVENV" "$CS/render_meshes.py" \
    "$NAME" "$CS/results/film_$NAME/rollout.mp4" median 2>&1 | grep -aE "RENDER|Error" | tail -6
else
  say "RENDER SKIPPED (no pyrender venv, or no poses file)"
fi
python3 "$CS/make_report.py" 2>&1 | tail -3
say "DONE $NAME"
