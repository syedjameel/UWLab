#!/usr/bin/env bash
# Watch every box on the grip-fixed curriculum and measure the first one that reaches the full task.
#
# STAGE D is the only stage containing ObjectAnywhereEEAnywhere -- the cube on the table, nothing in
# the hand -- and that is the whole question. It arrives at an unpredictable hour, so this fires on
# it rather than on a clock.
#
# It also fires on STAGE B, which is new: stage B is the first stage that contains
# ObjectAnywhereEEGrasped, the family the grip fix unblocks. Under the old action term that family
# scored exactly 0.0000 for 200 iterations on two boxes. Whether it moves off zero now is knowable
# hours before stage D, and it is the fix's falsifiable prediction -- so it gets measured the moment
# the stage starts producing checkpoints, not at the end.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv/bin/python
KEY=/home/mattephi/Work/IRiS/ip/meaningful-latents/.secrets/id_ed25519_ml_vast
SSHO="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes"
POLL=${POLL:-420}

declare -A PORT=( [b1]=40624 [b3]=17626 [b4]=40523 [b5]=45350 )
declare -A HOST=( [b1]=root@38.29.145.10 [b3]=root@89.221.67.158 [b4]=root@38.117.87.51 [b5]=root@115.73.217.180 )
# Five-finger boxes and two-finger boxes evaluate under different Play ids; a 14-dim checkpoint does
# not load a 26-dim actor and the failure is a shape error, not a bad number.
# The 2x2: hand size x transport shaping, every cell with the grip fix.
#            grip only                    grip + carry
#   5-finger b5  CubeStackGrip            b1  CubeStackGripCarry
#   2-finger b3  CubeStackTwoFingerGrip   b4  CubeStackTwoFingerGripCarry
declare -A TASK=(
  [b5]=OmniReset-UR10eDelto-CubeStackGrip-RelCartesianOSC-State-Play-v0
  [b1]=OmniReset-UR10eDelto-CubeStackGripCarry-RelCartesianOSC-State-Play-v0
  [b3]=OmniReset-UR10eDelto-CubeStackTwoFingerGrip-RelCartesianOSC-State-Play-v0
  [b4]=OmniReset-UR10eDelto-CubeStackTwoFingerGripCarry-RelCartesianOSC-State-Play-v0
)
declare -A SEEN_B=( [b1]=0 [b3]=0 [b4]=0 [b5]=0 )
declare -A SEEN_D=( [b1]=0 [b3]=0 [b4]=0 [b5]=0 )

# NO `bc`. The vast.ai images do not ship it, so the previous form -- `grep -ahc ... | paste -sd+ |
# bc` -- returned an empty string on every poll, `${n:-0}` made that a 0, and this watcher could
# never fire. It sat armed all night while a box passed the stage it was watching for. Concatenating
# the logs first and counting once removes the arithmetic entirely.
stage_count () {  # stage_count <box> <stage-letter>
  timeout 40 ssh -n -i "$KEY" $SSHO -p "${PORT[$1]}" "${HOST[$1]}" \
    "cat /workspace/logs/job_*.log 2>/dev/null | grep -ac 'STAGE $2 '" 2>/dev/null | tr -dc '0-9'
}

# ONE ISAAC PROCESS AT A TIME ON THE LOCAL GPU. An eval, a render and a probe each want several GB
# of VRAM, and a stale chain firing on top of a running render has already cost a crashed run once
# ("getRigidDynamicData: CUDA error, code 2", then an OOM). Everything here goes through run.sh,
# which execs the uwlab venv's python, so waiting on that interpreter covers every GPU job this
# project starts -- including ones launched by hand from another shell.
UWLAB_PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv/bin/python

# GRADE ON THE PLANT THE POLICY WAS TRAINED AGAINST. job_stiff.sh passes these same values to
# train.py as hydra overrides; the Play configs ship the softer defaults, so evaluating without them
# would measure a stiff-arm policy on a soft arm -- a different robot from the one it learned.
EVAL_GAINS=${EVAL_GAINS:-"--motion_stiffness 1000 --motion_damping 1.0"}

# The trackers hold /tmp/cubestack_gpu.lock around every eval. Polling pgrep instead is
# check-then-act and loses the race: an eval died with "CUDA out of memory ... Process 623387 has
# 8.80 GiB in use" because two jobs saw an idle card in the same instant. Everything that touches
# the GPU takes the same lock; `run_locked` wraps a command in it.
LOCK=/tmp/cubestack_gpu.lock
touch "$LOCK"
run_locked () { flock "$LOCK" "$@"; }

wait_for_gpu () { :; }  # superseded by the lock; kept so existing call sites stay valid

measure () {  # measure <box> <label> <families...>
  local b=$1 label=$2; shift 2
  wait_for_gpu
  echo "=== $b $label $(date +%H:%M) ==="
  "$CS/pull_checkpoint.sh" "$b" 2>&1 | tail -1
  local ck
  ck=$(ls -t "$CS/results/ck_$b"/model_*.pt 2>/dev/null | head -1)
  [ -z "${ck:-}" ] && { echo "  no checkpoint yet"; return 1; }
  echo "  checkpoint $ck"
  for rt in "$@"; do
    run_locked "$CS/run.sh" "$CS/eval_policy.py" --task "${TASK[$b]}" --checkpoint "$ck" \
      --num_envs 256 --episodes 400 --reset_type "$rt" \
      --collision_stack_mb 1536 --headless $EVAL_GAINS \
      --out "$CS/results/grip_${b}_${label}_${rt}.json" > "$CS/logs/grip_${b}_${label}_${rt}.log" 2>&1
    grep -aE 'GENUINE|EVAL_RESULT' "$CS/logs/grip_${b}_${label}_${rt}.log" \
      || { echo "  $rt FAILED:"; tr '\r' '\n' < "$CS/logs/grip_${b}_${label}_${rt}.log" | tail -4; }
  done
  "$PY" "$CS/collect_status.py" 2>&1 | tail -2
}

echo "=== grip harvest armed $(date +%H:%M): b1 b3 b4 b5, on stage B and stage D ==="
while true; do
  for b in b1 b3 b4 b5; do
    if [ "${SEEN_D[$b]}" = "0" ]; then
      n=$(stage_count "$b" D)
      if [ "${n:-0}" -gt 0 ] 2>/dev/null; then
        SEEN_D[$b]=1
        # The full task first, then the rest of the decomposition.
        measure "$b" stageD ObjectAnywhereEEAnywhere ObjectAnywhereEEGrasped \
                ObjectRestingEEGrasped ObjectPartiallyAssembledEEGrasped
        out=$CS/results/video_grip_fulltask_$b
        rm -rf "$out"
        wait_for_gpu
        ck=$(ls -t "$CS/results/ck_$b"/model_*.pt 2>/dev/null | head -1)
        [ -n "${ck:-}" ] && run_locked "$CS/run.sh" "$CS/render_rollout.py" --task "${TASK[$b]}" \
          --checkpoint "$ck" --num_envs 1 --episodes 30 --reset_type ObjectAnywhereEEAnywhere \
          --collision_stack_mb 1024 $EVAL_GAINS --out "$out" > "$CS/logs/render_grip_$b.log" 2>&1
        grep -aE 'RENDER_RESULT' "$CS/logs/render_grip_$b.log" 2>/dev/null || echo "  render did not report"
        "$PY" "$CS/make_montage.py" --dir "$out" \
          --out "$CS/report/media/grip_fulltask_${b}_stacks.mp4" 2>&1 | tail -2
        "$PY" "$CS/collect_status.py" 2>&1 | tail -1
        continue
      fi
    fi
    if [ "${SEEN_B[$b]}" = "0" ]; then
      n=$(stage_count "$b" B)
      if [ "${n:-0}" -gt 0 ] 2>/dev/null; then
        SEEN_B[$b]=1
        # Only the family the fix is about. One number, as early as it can be had.
        measure "$b" stageB ObjectAnywhereEEGrasped
      fi
    fi
  done
  sleep "$POLL"
done
