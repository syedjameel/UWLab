#!/usr/bin/env bash
# Start the local diagnostic arm as soon as the harvest's rendering releases the GPU.
# Rendering and training cannot share a 16 GB card, so this waits rather than competing.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
while pgrep -f '[r]ender_rollout.py' >/dev/null 2>&1 || pgrep -f '[e]val_policy.py' >/dev/null 2>&1; do sleep 60; done
grep -q 'harvest done' "$CS/logs/harvest.log" 2>/dev/null || sleep 60
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "=== [$(date +%H:%M:%S)] GPU free ${free_mb} MiB -- starting local diagnostic (seed 42) ==="
exec "$CS/train_diagnostic.sh"
