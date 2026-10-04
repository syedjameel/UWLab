#!/usr/bin/env bash
# Wait for the grasp bank, QA it, then build every remaining bank. Unattended.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
GRASPS=/home/mattephi/Work/Rubitek/UWLab-cubestack/Datasets_ur10e_delto/OmniReset/Grasps/InsertiveCube34/grasps.pt

while pgrep -f 'tools/rec''ord_grasps.py' >/dev/null 2>&1; do sleep 30; done
echo "=== grasp bank stage ended $(date +%H:%M:%S) ==="

if [ ! -f "$GRASPS" ]; then
  echo "[FAIL] no grasps.pt at $GRASPS -- cannot continue to reset banks"
  ls -laR /home/mattephi/Work/Rubitek/UWLab-cubestack/Datasets_ur10e_delto 2>/dev/null | head -30
  exit 1
fi
echo "[PASS] grasps.pt present: $(stat -c%s "$GRASPS") bytes"

echo; echo "=== grasp bank QA ==="
/home/mattephi/Work/Rubitek/uwlab-env/.venv/bin/python "$CS/qa_grasp_bank.py" --grasps "$GRASPS" 2>&1 | tee "$CS/results/grasp_bank_qa.txt"

echo; echo "=== building reset banks ==="
NUM_ENVS=1024 NUM_STATES=4000 "$CS/run_banks.sh" 2>&1 | tee "$CS/logs/run_banks.log"
echo "=== chain finished $(date +%H:%M:%S) ==="
