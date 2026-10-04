#!/usr/bin/env bash
# Watch the curriculum boxes and, the moment one reaches the FULL-TASK stage, measure it.
#
# WHY AUTOMATE THIS. The thing worth knowing is whether the policy can stack a cube it has to pick
# up first, and that only becomes answerable when a curriculum run reaches stage D -- the flat
# four-family mixture that includes Reaching. Stage D is hours away and arrives at an unpredictable
# time; without this, the answer waits for whenever someone next looks.
#
# When it fires it does the full sequence: pull the checkpoint, evaluate all four reset families
# (the paper's Fig. 8 decomposition), record a rollout from Reaching, cut whatever genuine successes
# are in it, and regenerate the report. If there are none, that is recorded too -- the clip cutter
# writes nothing rather than labelling failures as results.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv/bin/python
KEY=/home/mattephi/Work/IRiS/ip/meaningful-latents/.secrets/id_ed25519_ml_vast
POLL=${POLL:-300}

declare -A PORT=( [b5]=45350 [b4]=40523 )
declare -A HOST=( [b5]=root@115.73.217.180 [b4]=root@38.117.87.51 )
declare -A TASK=(
  [b5]=OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-Play-v0
  [b4]=OmniReset-UR10eDelto-CubeStackTwoFinger-RelCartesianOSC-State-Play-v0
)
declare -A DONE=( [b5]=0 [b4]=0 )

reached_stage_d () {  # reached_stage_d <box>
  local b=$1
  timeout 40 ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes \
    -p "${PORT[$b]}" "${HOST[$b]}" \
    "grep -acE 'STAGE D' /workspace/logs/job_*.log 2>/dev/null | paste -sd+ | bc" 2>/dev/null
}

harvest () {  # harvest <box>
  local b=$1
  echo "=== $b reached the full-task stage $(date +%H:%M) -- harvesting ==="
  "$CS/pull_checkpoint.sh" "$b" 2>&1 | tail -2
  local ck
  ck=$(ls -t "$CS/results/ck_$b"/model_*.pt 2>/dev/null | head -1)
  [ -z "${ck:-}" ] && { echo "  no checkpoint"; return 1; }
  echo "  checkpoint $ck"

  for rt in ObjectAnywhereEEAnywhere ObjectRestingEEGrasped ObjectAnywhereEEGrasped \
            ObjectPartiallyAssembledEEGrasped; do
    echo "--- $b / $rt ---"
    "$CS/run.sh" "$CS/eval_policy.py" --task "${TASK[$b]}" --checkpoint "$ck" \
      --num_envs 256 --episodes 400 --reset_type "$rt" \
      --collision_stack_mb 1536 --headless \
      --out "$CS/results/final_${b}_${rt}.json" > "$CS/logs/final_${b}_${rt}.log" 2>&1
    grep -aE 'GENUINE|EVAL_RESULT' "$CS/logs/final_${b}_${rt}.log" \
      || { echo "  FAILED:"; tail -6 "$CS/logs/final_${b}_${rt}.log" | tr '\r' '\n' | tail -4; }
  done

  # The full task is the one worth filming. If nothing genuine is in it the cutter writes nothing.
  local out=$CS/results/video_fulltask_$b
  rm -rf "$out"
  "$CS/run.sh" "$CS/render_rollout.py" --task "${TASK[$b]}" --checkpoint "$ck" \
    --num_envs 1 --episodes 30 --reset_type ObjectAnywhereEEAnywhere \
    --collision_stack_mb 1024 --out "$out" > "$CS/logs/render_fulltask_$b.log" 2>&1
  grep -aE 'RENDER_RESULT' "$CS/logs/render_fulltask_$b.log" \
    || { echo "  RENDER FAILED"; tail -5 "$CS/logs/render_fulltask_$b.log" | tr '\r' '\n' | tail -3; }
  "$PY" "$CS/make_montage.py" --dir "$out" \
    --out "$CS/report/media/fulltask_${b}_stacks.mp4" 2>&1 | tail -4 || echo "  no genuine success to montage"
  "$PY" "$CS/collect_status.py"
  echo "=== $b harvest done $(date +%H:%M) ==="
}

echo "=== armed $(date +%H:%M): watching b5 and b4 for the full-task stage ==="
while true; do
  for b in b5 b4; do
    [ "${DONE[$b]}" = "1" ] && continue
    n=$(reached_stage_d "$b")
    if [ "${n:-0}" -gt 0 ] 2>/dev/null; then
      DONE[$b]=1
      harvest "$b"
    fi
  done
  [ "${DONE[b5]}" = "1" ] && [ "${DONE[b4]}" = "1" ] && { echo "=== both harvested ==="; break; }
  sleep "$POLL"
done
