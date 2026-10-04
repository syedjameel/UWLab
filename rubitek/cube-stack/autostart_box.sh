#!/usr/bin/env bash
# Wait for a freshly rented box to finish provisioning, then push the repo and queue a job.
#
# Provisioning is ~15 minutes of apt, uv, a 10 GB Isaac download and an IsaacLab clone, and it
# HALTS halfway waiting for the working tree to be rsynced (onstart blocks on .repo_ready, because
# the cube-stacking branch is deliberately not published to any remote). So the sequence is: wait
# for ssh, push code, wait for READY, queue the job. Doing it by hand means either watching a
# terminal for a quarter of an hour or coming back to an idle box that has been billing the whole
# time.
#
#   ./autostart_box.sh b3 vast/job_oriented.sh
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
KEY=/home/mattephi/Work/IRiS/ip/meaningful-latents/.secrets/id_ed25519_ml_vast
VAST=/home/mattephi/Work/IRiS/ip/meaningful-latents/scripts/vast
BOX=${1:?usage: autostart_box.sh <box-tag> <job.sh>}
JOB=${2:?usage: autostart_box.sh <box-tag> <job.sh>}
IDFILE=$CS/vast/instance_id_$BOX
[ -f "$IDFILE" ] || { echo "no $IDFILE"; exit 1; }
ID=$(cat "$IDFILE")

endpoint () {
  "$VAST" show instance "$ID" --raw 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
p = d.get('direct_port_start')
print(p if p and int(p) > 0 else '', d.get('public_ipaddr') or '')
"
}

echo "=== waiting for $BOX ($ID) to expose an ssh endpoint ==="
for _ in $(seq 1 120); do
  read -r PORT IP <<< "$(endpoint)"
  # Both, and PORT must be numeric. When direct_port_start is absent the helper printed a leading
  # blank, word-splitting dropped it, and the IP landed in PORT -- which passed a bare -n test and
  # then spent an hour running `ssh -p <ip>` before failing.
  case "${PORT:-}" in (*[!0-9]*|"") PORT="" ;; esac
  [ -n "${PORT:-}" ] && [ -n "${IP:-}" ] && break
  sleep 20
done
[ -z "${PORT:-}" ] && { echo "no endpoint after 40 min"; exit 1; }
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes -p $PORT"
echo "endpoint root@$IP:$PORT"

echo "=== waiting for sshd ==="
SSH_UP=0
for _ in $(seq 1 90); do
  if $SSH "root@$IP" true 2>/dev/null; then SSH_UP=1; break; fi
  sleep 20
done
[ "$SSH_UP" = "1" ] || { echo "sshd never answered on $IP:$PORT"; exit 1; }

# The driver guard in onstart writes this and exits rather than provisioning a box that cannot run
# CUDA at all. Surface it here instead of waiting out a READY that will never come.
if $SSH "root@$IP" "test -f /workspace/PROVISION_FAILED" 2>/dev/null; then
  echo "=== $BOX FAILED provisioning: $($SSH "root@$IP" 'cat /workspace/PROVISION_FAILED') ==="
  echo "=== destroy it and rent another ==="
  exit 1
fi

echo "=== pushing the working tree (onstart blocks until this lands) ==="
# Not piped into tail: that would make the exit status tail's, and this script would report a
# started box that received no code.
BOX=$BOX "$CS/vast/launch.sh" push || { echo "push FAILED"; exit 1; }

echo "=== waiting for READY ==="
for _ in $(seq 1 180); do
  $SSH "root@$IP" "test -f /workspace/READY" 2>/dev/null && { echo "READY"; break; }
  sleep 20
done
$SSH "root@$IP" "test -f /workspace/READY" 2>/dev/null || { echo "never became READY"; exit 1; }

echo "=== queueing $(basename "$JOB") ==="
BOX=$BOX "$CS/vast/launch.sh" job "$JOB" || { echo "queueing FAILED"; exit 1; }
echo "=== $BOX started; add it to probe_runs.py as ('$PORT', 'root@$IP') ==="
