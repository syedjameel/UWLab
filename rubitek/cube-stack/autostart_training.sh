#!/usr/bin/env bash
# Wait until BOTH boxes have finished recording, then hand off to start_training.sh.
#
# THE BUG THIS VERSION FIXES. The first version ran preflight itself and only then called
# start_training.sh. Preflight checks LOCAL files, but C2 and C3 are recorded on the REMOTE box and
# the local copies are stale stubs from killed runs. Local C2 would have sat at 17 states forever,
# preflight would never have passed, and training would never have started -- a silent stall lasting
# however long nobody looked. The pull has to happen BEFORE the check, and start_training.sh is what
# does the pull, so this script must not gate on a check that precedes it.
#
# What it gates on instead: both recorders being gone. start_training.sh then pulls, runs preflight
# (which is where a genuine "banks too small" failure surfaces), and launches.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
SSHKEY=/home/mattephi/Work/IRiS/ip/meaningful-latents/.secrets/id_ed25519_ml_vast
HOSTARGS="-i $SSHKEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=12 -p 40624"
HOST=root@38.29.145.10
REMOTE_C2=/workspace/UWLab/Datasets_ur10e_delto/OmniReset/Resets/InsertiveCube34__ReceptiveCube34/resets_ObjectRestingEEGrasped.pt
MARKER=$CS/.training_started
DEADLINE=${DEADLINE:-9000}
C2_MIN_BYTES=$(( ${C2_MIN_STATES:-100} * 4200 ))

[ -f "$MARKER" ] && { echo "training already started (marker present)"; exit 0; }

t0=$(date +%s)
echo "=== autostart armed $(date +%H:%M:%S) -- waiting for both recorders to finish ==="
while true; do
  el=$(( $(date +%s) - t0 ))
  local_busy=$(pgrep -cf '[r]ecord_reset_states' || true)
  rc2=$(timeout 20 ssh $HOSTARGS $HOST "stat -c %s $REMOTE_C2 2>/dev/null || echo 0" 2>/dev/null | tr -dc '0-9')
  rc2=${rc2:-0}
  remote_busy=$(timeout 20 ssh $HOSTARGS $HOST "pgrep -cf '[r]ecord_reset_states' || true" 2>/dev/null | tr -dc '0-9')
  remote_busy=${remote_busy:-0}

  echo "[$(date +%H:%M:%S)] local_recorders=${local_busy:-0} remote_recorders=${remote_busy} remote_C2=${rc2}B (~$(( rc2 / 4200 )) states) el=${el}s"

  # Remote C2 has enough: stop it so the GPU is free and start_training can proceed.
  if [ "$rc2" -ge "$C2_MIN_BYTES" ] && [ "$remote_busy" -gt 0 ]; then
    echo "=== remote C2 reached the floor -- capping it (SIGKILL; Isaac ignores SIGTERM) ==="
    timeout 25 ssh $HOSTARGS $HOST "pkill -9 -f 'record_reset_states.*ObjectRestingEEGrasped'" 2>/dev/null || true
    sleep 10
    continue
  fi

  if [ "${local_busy:-0}" -eq 0 ] && [ "$remote_busy" -eq 0 ]; then
    echo "=== [$(date +%H:%M:%S)] both boxes idle after ${el}s -- handing off to start_training.sh ==="
    touch "$MARKER"
    exec "$CS/start_training.sh"
  fi

  if [ "$el" -ge "$DEADLINE" ]; then
    echo "=== [$(date +%H:%M:%S)] deadline ${DEADLINE}s reached; forcing both recorders down and proceeding ==="
    pkill -9 -f 'record_reset_states' 2>/dev/null || true
    timeout 25 ssh $HOSTARGS $HOST "pkill -9 -f record_reset_states" 2>/dev/null || true
    sleep 10
    touch "$MARKER"
    exec "$CS/start_training.sh"
  fi
  sleep 45
done
