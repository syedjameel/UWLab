#!/usr/bin/env bash
# Assemble the morning deliverable in one command.
#
# Written at 01:45 precisely so it is not improvised at 05:30. Every component here has already been
# run and verified tonight; this only sequences them against whatever the newest checkpoint is.
#
# It does NOT publish blindly: it prints the numbers and the artefact paths, and the contact sheets
# still have to be looked at before the result is trusted. Three separate times tonight a step
# "succeeded" while producing something empty or misleading -- blank video that passed a content
# check, a caption contradicting its own curve, a success metric measured 34 mm from the real target.
#
# Usage:  ./finalize_report.sh            (traces C3 and C4 from the newest local checkpoint)
#         FAMILIES="ObjectAnywhereEEGrasped" ./finalize_report.sh      (just C3)
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv-is6/bin/python
S=${SHEETS:-$CS/results/sheets}
mkdir -p "$S" "$CS/logs"

export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="/home/mattephi/Work/Rubitek/uwlab-env/compat:$REPO/source/uwlab_assets:$REPO/source/uwlab_tasks:$REPO/source/uwlab:$REPO/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=268435456
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-0.15
export UWLAB_LATCHED_HAND=1 UWLAB_TUCK_OTHERS=0 UWLAB_GRIP_BIAS_RAD=0.05 UWLAB_HAND_EFFORT_SCALE=20
export TRACE_ENVS=${TRACE_ENVS:-64} TRACE_STEPS=160

CK=${CK:-$(ls -t "$REPO"/logs/rsl_rl/ur10e_delto_omnireset_agent/*local_eff20*/model_*.pt 2>/dev/null | head -1)}
[ -n "$CK" ] || { echo "no local checkpoint found"; exit 1; }
echo "=== checkpoint: $CK"
echo "=== iteration reached: $(grep -ac 'Learning iteration' "$CS/logs/train_local.log")"

FAMILIES=${FAMILIES:-"ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped"}
for FAM in $FAMILIES; do
  case "$FAM" in
    ObjectAnywhereEEGrasped) TAG=c3 ;;
    ObjectPartiallyAssembledEEGrasped) TAG=c4 ;;
    *) TAG=$(echo "$FAM" | tr '[:upper:]' '[:lower:]') ;;
  esac
  echo "=== tracing $FAM ($TAG) $(date -Is)"
  ( cd "$REPO" && TRACE_NAME="policy_$TAG" TRACE_CK="$CK" TRACE_FAMILY="$FAM" \
      timeout 1800 "$PY" "$CS/trace_collect_policy.py" > "$CS/logs/final_trace_$TAG.log" 2>&1 )
  grep -aE "TRACE (thresholds|wrote)|Traceback" "$CS/logs/final_trace_$TAG.log" | tail -2

  echo "=== animating $TAG"
  "$PY" "$CS/trace_animate_policy.py" "policy_$TAG" "$CS/results/film_policy_$TAG/rollout.mp4" 2>&1 | tail -1
  # Contact sheet for the mandatory eye check -- numbers alone have been wrong three times tonight.
  timeout 150 ffmpeg -y -loglevel error -i "$CS/results/film_policy_$TAG/rollout.mp4" \
    -vf "select='not(mod(n\,49))',scale=680:-1,tile=1x3" -frames:v 1 "$S/policy_${TAG}_sheet.png" 2>&1 | tail -1
  echo "    sheet: $S/policy_${TAG}_sheet.png"
done

echo "=== refreshing the H100 log (log only; the checkpoint pull blocks ~11 min on this link)"
SKIP_CK=1 "$CS/fetch_h100_log.sh" 2>&1 | tail -2

echo "=== regenerating the report"
python3 "$CS/make_report.py" 2>&1 | tail -1
timeout 10 curl -s -o /dev/null -w "report HTTP %{http_code}\n" http://127.0.0.1:8797/delto_pinch/

echo
echo "=== NOW LOOK AT THE SHEETS BEFORE TRUSTING ANY OF IT:"
ls -la "$S"/*.png 2>/dev/null | awk '{printf "  %6.0f KB %s\n", $5/1024, $9}'
echo "=== report: https://mattecache.tailcc6bd2.ts.net/reports/delto_pinch/"
