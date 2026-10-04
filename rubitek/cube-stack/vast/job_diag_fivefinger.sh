#!/usr/bin/env bash
# CAN THE FIVE-FINGER GRIP HOLD WHAT THE TWO-FINGER PINCH CANNOT?
#
# Two-finger, every knob tested, in_hand at step 120: 0.047-0.180, i.e. at best one cube in five.
# Cube mass is directional but not a rescue -- light (0.02 kg) 0.125, heavy (0.06 kg) 0.047, against
# a control spread of 0.062-0.086. Even the LIGHTEST cube in the shipped range fails.
#
# The repo's own grip_probe drives all twenty hand joints closed and holds a mid-air C3 cube for a
# whole episode at 0.48-2.56 N fingertip force. The CubeStackGripCarry id biases all twenty joints
# (the TwoFinger variant narrows the action term to the 8 joints of fingers 1-2, so its bias reaches
# only those). The reset banks store FULL joint state, so fingers 3-5 start in their open pose and
# the bias closes them onto the cube -- no re-recording should be required.
#
# Same banks, same arm fix, same null action. Only the hand configuration differs.
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=40 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-1.0

echo "=== DIAG_5F_START $(date -Is)"
for TASKID in CubeStackTwoFingerGripCarry CubeStackGripCarry; do
  echo "=== TASK=$TASKID $(date -Is)"
  UWLAB_DIAG_TASK="$TASKID" timeout 1800 "$PY" - <<'PYEOF' 2>&1 | grep -aE "DIAG|Error|Traceback" | tail -16
import os

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
import uwlab_is6_bridge  # noqa: E402

uwlab_is6_bridge.apply()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

NAME = os.environ["UWLAB_DIAG_TASK"]
TASK = f"OmniReset-UR10eDelto-{NAME}-RelCartesianOSC-State-v0"
N = 128
TIPS = ("rl_dg_1_tip", "rl_dg_2_tip")
_t = lambda x: getattr(x, "torch", x)  # noqa: E731

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
ev = cfg.events.reset_from_reset_states
ev.params["reset_types"] = ["ObjectAnywhereEEGrasped"]
ev.params["probs"] = [1.0]

env = gym.make(TASK, cfg=cfg).unwrapped
robot = env.scene["robot"]
obj = env.scene["insertive_object"]
names = list(robot.body_names)
tip_ids = [names.index(t) for t in TIPS if t in names]
gterm = env.action_manager.get_term("gripper")
print(f"DIAG task={NAME} gripper_action_dim={gterm.action_dim} total={env.action_manager.total_action_dim}")

env.reset()
pos0 = _t(obj.data.root_pos_w).clone()
tips0 = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1).clone()
d0 = (pos0 - tips0).norm(dim=-1)
zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)

for step in range(120):
    env.step(zero)
    if step in (4, 19, 79, 119):
        pos = _t(obj.data.root_pos_w)
        tips = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1)
        d = (pos - tips).norm(dim=-1)
        in_hand = float(((d - d0).abs() < 0.02).float().mean())
        print(f"DIAG task={NAME} step={step + 1:4d} in_hand={in_hand:.3f} cube_to_tip={float(d.mean()):.4f}")
env.close()
app.close()
PYEOF
done
echo "=== DIAG_5F_DONE $(date -Is)"
