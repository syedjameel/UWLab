#!/usr/bin/env bash
# Stop the local C4 recording once it has enough states, so the dev GPU is free for training.
#
# Same reasoning as cap_c3.sh: --num_reset_states 300 is a target, not a commitment, because the
# bank is written incrementally. C4 accepts ~1.3 states/min, so running to 300 would occupy the
# only render-capable GPU until roughly 06:00 for states the policy will each see hundreds of times
# anyway. 150 is the floor worth waiting for.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv/bin/python
F=/home/mattephi/Work/Rubitek/UWLab-cubestack/Datasets_ur10e_delto/OmniReset/Resets/InsertiveCube34__ReceptiveCube34/resets_ObjectPartiallyAssembledEEGrasped.pt
TARGET=${TARGET:-150}
HARD_DEADLINE=${HARD_DEADLINE:-5400}

t0=$(date +%s)
echo "=== C4 cap armed: stop at >= ${TARGET} states or ${HARD_DEADLINE}s ==="
while true; do
  n=$($PY "$CS/count_bank.py" "$F" 2>/dev/null | awk '{print $(NF-1)}')
  n=${n:-0}
  el=$(( $(date +%s) - t0 ))
  echo "[$(date +%H:%M:%S)] C4 ${n} states, ${el}s elapsed"
  if [ "$n" -ge "$TARGET" ] 2>/dev/null; then echo "=== C4 reached ${n} states -- capping ==="; break; fi
  if [ "$el" -ge "$HARD_DEADLINE" ]; then echo "=== hard deadline at ${n} states -- capping ==="; break; fi
  # If the recorder already exited on its own there is nothing left to cap.
  pgrep -f 'record_reset_states.*ObjectPartiallyAssembledEEGrasped' >/dev/null || {
    echo "=== C4 recorder already gone (${n} states) ==="; exit 0; }
  sleep 45
done
# SIGKILL, not SIGTERM. Observed on the C3 cap: plain `pkill` sent SIGTERM, Isaac ignored it, the
# recorder kept running past its cap and the next family never got the GPU. Isaac installs its own
# signal handling and a graceful stop is not reliable here.
pkill -9 -f 'record_reset_states.*ObjectPartiallyAssembledEEGrasped'
sleep 8
$PY "$CS/count_bank.py" "$F"
echo "=== C4 capped $(date +%H:%M:%S); dev GPU free for training ==="
