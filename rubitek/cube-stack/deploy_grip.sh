#!/usr/bin/env bash
# Move every rented box onto the grip-fixed task, in one pass.
#
# WHY ALL OF THEM. The three "EEGrasped" reset families were hollow: the bank stores the validated
# closed hand posture, but its recorded PD set point is only 3.6 mrad from the recorded joint
# position, so there is no squeeze in it -- and the relative action term then holds target =
# measured, which is zero grip force forever. Measured with the null action, the C3 cube falls from
# 102 mm to 34 mm. Every run currently training spends most of its samples on families where the
# cube is not actually in the hand, which is exactly why task_1_success_rate has sat at 0.0000 on
# two boxes for hundreds of iterations while near-goal -- the one family that never needed a grasp
# -- trains to 0.20.
#
# ORDER MATTERS. Stop the job wrapper BEFORE train.py, or the wrapper marches on to its next
# curriculum stage with no resume while the replacement job sits queued. `launch.sh stop` does this.
# Push code before queueing, or the box runs the new job script against the old task registry.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
BOXES=${BOXES:-"b1 b3 b4 b5"}
JOB=${JOB:-$CS/vast/job_grip.sh}

for b in $BOXES; do
  [ -f "$CS/vast/instance_id_$b" ] || { echo "$b: no instance file, skipping"; continue; }
  echo "=== $b: stopping current job ==="
  BOX=$b "$CS/vast/launch.sh" stop 2>&1 | tail -2
  echo "=== $b: pushing code ==="
  BOX=$b "$CS/vast/launch.sh" push 2>&1 | tail -2
  echo "=== $b: queueing $(basename "$JOB") ==="
  BOX=$b FORCE=1 "$CS/vast/launch.sh" job "$JOB" 2>&1 | tail -2
done
echo "=== deploy done $(date +%H:%M) ==="
