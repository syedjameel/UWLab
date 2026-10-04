#!/usr/bin/env bash
# VERIFY THE LATCHED HAND TERM THROUGH THE REAL TASK CFG.
#
# Everything measured so far swapped the action term inside the probe. This runs the SHIPPED cfg
# chain with UWLAB_LATCHED_HAND=1, so it also proves the term constructs correctly inside
# _apply_grip_bias -- import path, per-joint scale dict, joint resolution, reset latching.
#
# Expected from the probe-local version (C3 bank, null action, 128 envs):
#     relative bias 0.08 : in_hand 0.461 -> 0.000 by step 120, fingers travel +0.453 rad (ramp)
#     latched  bias 0.05 : in_hand 0.867 -> 0.117 by step 120, fingers settle +0.044 rad
# `off` is the control; `on` must reproduce the latched curve or the wiring is wrong.
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-0.15

echo "=== DIAG_LATCHED_START $(date -Is)"
for MODE in off on; do
  echo "=== LATCHED=$MODE $(date -Is)"
  export UWLAB_LATCHED_HAND=$([ "$MODE" = "on" ] && echo 1 || echo 0)
  UWLAB_DIAG_MODE="$MODE" timeout 1800 "$PY" - <<'PYEOF' 2>&1 | grep -aE "DIAG|hand action ->|Error|Traceback" | tail -18
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
MODE = os.environ["UWLAB_DIAG_MODE"]
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
print(f"DIAG mode={MODE} term={type(gterm).__name__} dim={gterm.action_dim}")
fidx = list(gterm._joint_ids) if not isinstance(gterm._joint_ids, slice) else list(range(robot.num_joints))


def tip_force():
    try:
        f = _t(env.scene["object_hand_s"].data.force_matrix_w)
        return float(f.norm(dim=-1).sum(dim=-1).mean())
    except Exception:
        return float("nan")


env.reset()
pos0 = _t(obj.data.root_pos_w).clone()
tips0 = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1).clone()
d0 = (pos0 - tips0).norm(dim=-1)
q0 = _t(robot.data.joint_pos)[:, fidx].clone()
zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)

for step in range(120):
    env.step(zero)
    if step in (4, 19, 79, 119):
        pos = _t(obj.data.root_pos_w)
        tips = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1)
        d = (pos - tips).norm(dim=-1)
        in_hand = float(((d - d0).abs() < 0.02).float().mean())
        dq = float((_t(robot.data.joint_pos)[:, fidx] - q0).mean())
        print(
            f"DIAG mode={MODE} step={step + 1:4d} in_hand={in_hand:.3f} "
            f"finger_dq_rad={dq:+.4f} tip_force_N={tip_force():.3f} cube_to_tip={float(d.mean()):.4f}"
        )
env.close()
app.close()
PYEOF
done
echo "=== DIAG_LATCHED_DONE $(date -Is)"
