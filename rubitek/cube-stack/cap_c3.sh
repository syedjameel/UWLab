#!/usr/bin/env bash
# Hand the remote GPU from C3 to C2 as soon as C3 has ENOUGH states -- not on a fixed timer.
#
# A fixed deadline wastes whatever gap sits between "enough" and the deadline. C3 accepts about
# 2 states/min, so a 45-minute misjudgement is ~90 states of C2 never recorded. Poll the bank size
# and cut over the moment the floor is crossed.
#
# The bank file is written incrementally, so killing the recorder keeps every state accepted so far
# and the job script advances to C2 on its own.
set -uo pipefail
SSHKEY=/home/mattephi/Work/IRiS/ip/meaningful-latents/.secrets/id_ed25519_ml_vast
HOSTARGS="-i $SSHKEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=12 -p 40624 root@38.29.145.10"
F=/workspace/UWLab/Datasets_ur10e_delto/OmniReset/Resets/InsertiveCube34__ReceptiveCube34/resets_ObjectAnywhereEEGrasped.pt

TARGET=${TARGET:-120}                 # states
BYTES_PER_STATE=4200                  # measured: 16.72 MB / 4008 states on C1
MIN_BYTES=$(( TARGET * BYTES_PER_STATE ))
HARD_DEADLINE=${HARD_DEADLINE:-5400}  # cut over regardless, so C2 always gets a slot

t0=$(date +%s)
echo "=== C3 cap armed: cut over at >= ${TARGET} states (~${MIN_BYTES} bytes) or ${HARD_DEADLINE}s ==="
while true; do
  sz=$(timeout 20 ssh $HOSTARGS "stat -c %s $F 2>/dev/null || echo 0" 2>/dev/null | tr -dc '0-9')
  sz=${sz:-0}
  el=$(( $(date +%s) - t0 ))
  echo "[$(date +%H:%M:%S)] C3 ${sz} bytes (~$(( sz / BYTES_PER_STATE )) states), ${el}s elapsed"
  if [ "$sz" -ge "$MIN_BYTES" ]; then
    echo "=== C3 reached the floor after ${el}s -- capping now ==="
    break
  fi
  if [ "$el" -ge "$HARD_DEADLINE" ]; then
    echo "=== hard deadline hit at ${sz} bytes -- capping anyway so C2 gets a slot ==="
    break
  fi
  sleep 45
done

timeout 25 ssh $HOSTARGS "stat -c 'C3 final: %s bytes' $F; pkill -f 'record_reset_states.*ObjectAnywhereEEGrasped'" 2>&1 | tail -2
echo "=== C3 capped $(date +%H:%M:%S); job script advances to C2 ==="
