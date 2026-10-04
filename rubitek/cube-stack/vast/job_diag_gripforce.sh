#!/usr/bin/env bash
# CAN THE TWO-FINGER PINCH ACTUALLY HOLD THE CUBE FOR A FULL EPISODE?
#
# The arm defect is fixed: gravity comp at any negative scale removes the 114 mm fall
# (off -0.114 m -> -0.25 +0.037, -0.50 +0.053, -0.75 +0.058, -1.00 +0.062 m at step 120).
# But holding barely moved: in_hand@120 went 0.008 -> 0.031 / 0.070 / 0.109 / 0.117. It is also NOT
# monotone with drift -- -1.00 has the LARGEST residual drift and the BEST holding -- so drift is no
# longer what loses the cube. The hand is.
#
# Why the hand is weak here: _apply_two_finger_hand narrows the action term to the 8 joints of
# fingers 1-2, so _apply_grip_bias biases only those. The repo's own probe found the `_3` joints --
# the stiffest in the hand at 4.0 N*m/rad -- supply most of the grip, and that driving all twenty at
# +0.1 rad/step settles fingertip forces at 0.48-2.56 N and holds the C3 cube for a whole episode.
# A 0.05 rad bias on 8 joints is a far weaker squeeze than that against a 0.29 N cube.
#
# Sweep the two knobs that set pinch force and report in_hand at step 120 -- the quantity that
# decides whether C3 is trainable at all. Arm held at the best-holding scale (-1.00) throughout so
# only the grip varies.
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-1.0

echo "=== DIAG_GRIPFORCE_START $(date -Is)"
# BIAS: the per-joint offset _apply_grip_bias installs (shipped 0.05; must stay < 0.1 rad so the
# policy's +-0.1 rad/step authority can still open the hand).
# STIFF: UWLAB_PINCH_SEG1_STIFFNESS, the proximal-segment stiffness of the pinching fingers (10 today).
for COMBO in "0.05 10" "0.08 10" "0.05 40" "0.08 40" "0.08 80"; do
  set -- $COMBO; BIAS=$1; STIFF=$2
  echo "=== GRIPFORCE bias=$BIAS seg1_stiffness=$STIFF $(date -Is)"
  export UWLAB_PINCH_SEG1_STIFFNESS="$STIFF"
  UWLAB_DIAG_BIAS="$BIAS" UWLAB_DIAG_STIFF="$STIFF" timeout 1800 "$PY" - <<'PYEOF' 2>&1 | grep -aE "DIAG|Error|Traceback" | tail -20
import os

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
import uwlab_is6_bridge  # noqa: E402

uwlab_is6_bridge.apply()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "OmniReset-UR10eDelto-CubeStackTwoFingerGripCarry-RelCartesianOSC-State-v0"
N = 128
TIPS = ("rl_dg_1_tip", "rl_dg_2_tip")
BIAS = float(os.environ["UWLAB_DIAG_BIAS"])
STIFF = os.environ["UWLAB_DIAG_STIFF"]
_t = lambda x: getattr(x, "torch", x)  # noqa: E731

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
ev = cfg.events.reset_from_reset_states
ev.params["reset_types"] = ["ObjectAnywhereEEGrasped"]
ev.params["probs"] = [1.0]
g = cfg.actions.gripper
cfg.actions.gripper = g.replace(use_zero_offset=False, offset={n: BIAS for n in g.scale})

env = gym.make(TASK, cfg=cfg).unwrapped
robot = env.scene["robot"]
obj = env.scene["insertive_object"]
names = list(robot.body_names)
tip_ids = [names.index(t) for t in TIPS if t in names]

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
        sag = float((tips0[:, 2] - tips[:, 2]).mean())
        print(
            f"DIAG bias={BIAS} stiff={STIFF} step={step + 1:4d} in_hand={in_hand:.3f} "
            f"hand_sag_m={sag:+.4f} cube_to_tip={float(d.mean()):.4f}"
        )
env.close()
app.close()
PYEOF
done
echo "=== DIAG_GRIPFORCE_DONE $(date -Is)"
