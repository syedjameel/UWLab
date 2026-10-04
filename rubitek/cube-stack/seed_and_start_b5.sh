#!/usr/bin/env bash
# Seed b5 with the trained near-goal checkpoint, then retire the A6000 it replaces.
#
# WHY MIGRATE. b2 is an RTX A6000 at 73 s/iteration; b4 is an RTX 4090 at 33 s and costs $0.335/h
# against $0.4556. The A6000 is 2.2x slower AND more expensive for this workload, which is physics
# bound rather than memory bound. b2's curriculum had 409 iterations left in stage B alone -- 8.3
# hours there, and it would not reach the full-task stage for about twenty. The same work on a 4090
# is under four.
#
# The seed checkpoint has to be uploaded BEFORE the job runs, because job_curriculum.sh skips its
# first stage only when /workspace/seed_neargoal/model_300.pt is present -- and defaulting that path
# to "newest checkpoint on the box" is exactly the bug that had it resuming from an untrained
# model_0 earlier today.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
KEY=/home/mattephi/Work/IRiS/ip/meaningful-latents/.secrets/id_ed25519_ml_vast
VAST=/home/mattephi/Work/IRiS/ip/meaningful-latents/scripts/vast
SEED=/home/mattephi/Work/Rubitek/UWLab-cubestack/logs/rsl_rl/ur10e_delto_omnireset_agent/2026-08-31_11-16-08/model_300.pt

echo "=== waiting for b5 to be READY ==="
for _ in $(seq 1 200); do
  read -r PORT IP <<< "$("$VAST" show instance "$(cat "$CS/vast/instance_id_b5")" --raw 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
p = d.get('direct_port_start')
print(p if p and int(p) > 0 else '', d.get('public_ipaddr') or '')
")"
  case "${PORT:-}" in (*[!0-9]*|"") PORT="" ;; esac
  if [ -n "${PORT:-}" ]; then
    SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes -p $PORT"
    if $SSH "root@$IP" "test -f /workspace/READY" 2>/dev/null; then
      echo "b5 READY at root@$IP:$PORT"
      break
    fi
  fi
  sleep 20
done
[ -z "${PORT:-}" ] && { echo "b5 never came up"; exit 1; }

echo "=== uploading the near-goal seed ==="
$SSH "root@$IP" "mkdir -p /workspace/seed_neargoal"
scp -q -i "$KEY" -o StrictHostKeyChecking=accept-new -P "$PORT" "$SEED" "root@$IP:/workspace/seed_neargoal/model_300.pt" \
  || { echo "seed upload FAILED"; exit 1; }
$SSH "root@$IP" "ls -la /workspace/seed_neargoal/"

echo "=== queueing the curriculum ==="
BOX=b5 "$CS/vast/launch.sh" job "$CS/vast/job_curriculum.sh" || exit 1

echo "=== waiting for b5 to reach an iteration before retiring b2 ==="
for _ in $(seq 1 60); do
  n=$($SSH "root@$IP" "grep -acE 'Learning iteration' /workspace/logs/job_curriculum.log 2>/dev/null" 2>/dev/null)
  [ "${n:-0}" -gt 0 ] 2>/dev/null && { echo "b5 is training"; break; }
  sleep 30
done

# Only now retire the box it replaces. Destroying first would have left a gap with no curriculum
# running at all if b5 had failed to provision.
echo "=== retiring b2 (the A6000) ==="
BOX=b2 "$CS/vast/launch.sh" stop 2>&1 | tail -1
echo y | "$VAST" destroy instance "$(cat "$CS/vast/instance_id_b2")" 2>&1 | tail -1
rm -f "$CS/vast/instance_id_b2"
"$VAST" show instances
echo "=== migration done $(date +%H:%M) ==="
