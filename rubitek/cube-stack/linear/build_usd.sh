#!/usr/bin/env bash
# Build the UR10e + linear-gripper USDs (UR10E_OMNIRESET_GUIDE.md section 2) under Isaac Sim 6.
set -uo pipefail
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
L=source/uwlab_assets/uwlab_assets/local/Robots
say(){ echo "BUILD $* $(date -Is)"; }
say 2a; $R scripts_v2/tools/conversions/convert_gripper_urdf.py --headless --fix-base --input $L/UR10e/ur10e.urdf --output $L/UR10e/ur10e.usd; say 2a rc=$?
say 2b; $R scripts_v2/tools/conversions/convert_gripper_urdf.py --headless --input $L/LinearGripper/gripper.urdf --output $L/LinearGripper/linear_gripper.usd; say 2b rc=$?
say mimic; $R scripts_v2/tools/conversions/add_gripper_mimic.py --headless --usd $L/LinearGripper/linear_gripper.usd --dual-drive; say mimic rc=$?
say 2c; $R scripts_v2/tools/conversions/graft_gripper_on_ur10e.py; say 2c rc=$?
ls -la $L/UR10e $L/LinearGripper $L/Ur10eLinearGripper
say DONE
