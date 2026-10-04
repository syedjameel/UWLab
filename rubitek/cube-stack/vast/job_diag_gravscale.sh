#!/usr/bin/env bash
# WHAT SCALE ON THE GRAVITY TERM ACTUALLY NULLS THE ARM DRIFT?
#
# Measured, C3 bank, null action, fingertip sag at step 120 (negative = falling):
#     gravity comp off   -0.114 m     the shipped defect
#     sign = +1          -0.903 m     adding it DOUBLES the fall
#     sign = -1          +0.059 m     right direction, but the arm now RISES
#
# -1 over-compensates, so the correct term is a FRACTION of it. This sweeps the scale and reports
# which value holds the hand nearest zero drift across a full episode. The target is not "less sag"
# but |sag| ~ 0 at step 120 AND in_hand staying high: any residual drift is a bias the policy has to
# cancel on every step, which is the defect being removed.
#
# 0.0 is the control (identical to off) and validates that the sweep harness itself is sane.
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f

echo "=== DIAG_GRAVSCALE_START $(date -Is)"
for S in -0.25 -0.50 -0.75 -1.00; do
  echo "=== GRAVSCALE=$S $(date -Is)"
  export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN="$S"
  UWLAB_DIAG_S="$S" timeout 1800 "$PY" - <<'PYEOF' 2>&1 | grep -aE "DIAG|gravity compensation|Error|Traceback" | tail -20
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
S = os.environ.get("UWLAB_DIAG_S", "?")
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

env.reset()
pos0 = _t(obj.data.root_pos_w).clone()
tips0 = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1).clone()
d0 = (pos0 - tips0).norm(dim=-1)
zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)

# No step-159 sample: the 160-step episode boundary resets the envs before it is read.
for step in range(120):
    env.step(zero)
    if step in (0, 4, 19, 39, 79, 119):
        pos = _t(obj.data.root_pos_w)
        tips = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1)
        d = (pos - tips).norm(dim=-1)
        in_hand = float(((d - d0).abs() < 0.02).float().mean())
        sag = float((tips0[:, 2] - tips[:, 2]).mean())
        print(
            f"DIAG scale={S} step={step + 1:4d} in_hand={in_hand:.3f} "
            f"hand_sag_m={sag:+.4f} cube_to_tip={float(d.mean()):.4f}"
        )
env.close()
app.close()
PYEOF
done
echo "=== DIAG_GRAVSCALE_DONE $(date -Is)"
