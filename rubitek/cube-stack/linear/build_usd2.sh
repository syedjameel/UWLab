#!/usr/bin/env bash
set -uo pipefail
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
L=source/uwlab_assets/uwlab_assets/local/Robots
say(){ echo "BUILD $* $(date -Is)"; }
say flatten; $R /home/mattephi/Work/Rubitek/cube-stack/linear/flatten.py $L/UR10e/ur10e/ur10e.usda $L/UR10e/ur10e.usd $L/LinearGripper/gripper/gripper.usda $L/LinearGripper/linear_gripper.usd; say flatten rc=$?
say mimic; $R scripts_v2/tools/conversions/add_gripper_mimic.py --headless --usd $L/LinearGripper/linear_gripper.usd --dual-drive; say mimic rc=$?
say 2c; $R scripts_v2/tools/conversions/graft_gripper_on_ur10e.py; say 2c rc=$?
ls -la $L/UR10e $L/LinearGripper $L/Ur10eLinearGripper
say DONE
