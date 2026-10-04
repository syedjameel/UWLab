#!/usr/bin/env bash
# Wait for the running posture optimiser to finish, then rebuild everything from its winner.
cd /home/mattephi/Work/Rubitek/cube-stack/pinch
while ps -eo comm=,args= | awk '$1=="python"' | grep -q pinchhold.py; do sleep 20; done
sleep 5
export SKIP_S1=1 POSE=/home/mattephi/Work/Rubitek/cube-stack/pinch/out/pinchhold_v3.json
exec ./pipeline.sh
