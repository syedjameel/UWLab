#!/usr/bin/env bash
# Rebuild the linear-gripper USDs with the jaw-box colliders actually applied, then verify physically.
set -uo pipefail
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
S=/tmp/claude-1000/-home-mattephi-Work-Rubitek/d817c68b-2665-4f13-965b-515b6cadcd04/scratchpad
O=/home/mattephi/Work/Rubitek/cube-stack/linear/out
L=source/uwlab_assets/uwlab_assets/local/Robots
say(){ echo "BUILD3 $* $(date -Is)"; }
say convert; $R scripts_v2/tools/conversions/convert_gripper_urdf.py --headless --input $L/LinearGripper/gripper.urdf --output $L/LinearGripper/linear_gripper.usd > $O/b3_convert.log 2>&1; say convert rc=$?
say flatten; $R /home/mattephi/Work/Rubitek/cube-stack/linear/flatten.py $L/LinearGripper/gripper/gripper.usda $L/LinearGripper/linear_gripper.usd 2>&1 | grep -a "FLATTEN\|WROTE\|Error"; say flatten
rm -rf $L/LinearGripper/gripper $L/LinearGripper/config.yaml $L/LinearGripper/.asset_hash
say unnest; $R scripts_v2/tools/conversions/unnest_rigid_bodies.py --usd $L/LinearGripper/linear_gripper.usd 2>&1 | grep -av "WARNING: All"; rc=${PIPESTATUS[0]}; say unnest rc=$rc; [ $rc -eq 0 ] || { echo BUILD3_ABORT unnest; exit 1; }
say prep; $R scripts_v2/tools/conversions/add_gripper_mimic.py --headless --usd $L/LinearGripper/linear_gripper.usd --dual-drive > $O/b3_prep.log 2>&1; rc=$?; tail -12 $O/b3_prep.log | grep -av "absl\|WARNING: All"; say prep rc=$rc; [ $rc -eq 0 ] || { echo BUILD3_ABORT prep; exit 1; }
say graft; $R scripts_v2/tools/conversions/graft_gripper_on_ur10e.py 2>&1 | grep -v "^WARNING: All"; say graft
say verify_prims; $R $S/boxes.py 2>&1 | grep -v "WARNING: All"
say grip40; timeout 300 $R $S/grip_is6.py --headless --usd $L/LinearGripper/linear_gripper.usd --finger-offset 0.10 --slab-x 0.040 --slab-y 0.040 --slab-z 0.040 --slab-mass 0.001 > $O/grip40.log 2>&1
grep -aE "^\s+0\.[01][0-9]{2}\s+" $O/grip40.log; say grip40 done
say pcb_dbg; UWLAB_GRASP_DEBUG=1 timeout 240 $R scripts_v2/tools/record_grasps.py --task OmniReset-LinearGripper-GraspSampling-v0 --num_envs 32 --num_grasps 16 --headless --dataset_dir $O/dbg_pcb3 env.scene.object=pcb > $O/grasp_pcb3.log 2>&1; echo "rc=$?"
grep -a "grasp dbg0" $O/grasp_pcb3.log | head -14; grep -a "BATCH_END" $O/grasp_pcb3.log | tail -2; grep -a "Traceback" -A3 $O/grasp_pcb3.log | head
echo BUILD3_DONE
