#!/usr/bin/env bash
# POSITIVE CONTROL FIRST, THEN THE FINE NULL SEARCH.
#
# THE BIAS SWEEP WAS FLAT, AND THAT IS THE CLUE. Raising the bias 5x (0.08 -> 0.40 rad) changed
# fingertip force not at all: 0.745 / 0.519 / 0.737 / 0.548 N at step 5, in_hand@120 0.070 / 0.078 /
# 0.078 / 0.148. A commanded offset that produces no extra force means the torque is SATURATED --
# the DELTO finger actuators cap out -- so bias was never the knob, and neither was seg1 stiffness.
#
# TWO THINGS THIS RUN FIXES, BOTH METHODOLOGICAL:
#
# 1. NO POSITIVE CONTROL. Every number I have came from a harness that has never once reproduced a
#    successful hold, so I cannot distinguish "the pinch fails" from "the harness cannot measure a
#    success". The films hold the cube at ~1.2 N with the ARM PINNED. Cell `stiffarm` gives the arm
#    real PD gains (the actuator reads UWLAB_ARM_STIFFNESS/DAMPING, default 0.0/0.0) to recreate
#    that condition. If this cell does NOT hold, the harness is wrong and every earlier conclusion
#    built on it is void.
#
# 2. THE GRAVITY NULL WAS NEVER BRACKETED. off = -114 mm, and the FIRST point I tested, -0.25,
#    was already +37 mm -- an overshoot. The null sits between 0 and -0.25, near -0.15..-0.20, and
#    that interval was never sampled. Residual drift of 40-60 mm across an episode is a plausible
#    cause of the cube working loose, and "less bad than -114 mm" is not the same as "held still".
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f

echo "=== DIAG_CONTROL_START $(date -Is)"
for CELL in stiffarm g010 g015 g020; do
  echo "=== CELL=$CELL $(date -Is)"
  unset UWLAB_ARM_STIFFNESS UWLAB_ARM_DAMPING
  case "$CELL" in
    # Positive control: hold the arm with joint PD instead of the OSC, gravity term OFF.
    # 600/60 rather than the 2000/200 the repo called unstable at dt = 1/120.
    stiffarm) export UWLAB_ARM_STIFFNESS=600 UWLAB_ARM_DAMPING=60 UWLAB_OSC_GRAVITY_COMP=0 UWLAB_OSC_GRAVITY_SIGN=1 ;;
    g010)     export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-0.10 ;;
    g015)     export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-0.15 ;;
    g020)     export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-0.20 ;;
  esac
  UWLAB_DIAG_CELL="$CELL" timeout 1800 "$PY" - <<'PYEOF' 2>&1 | grep -aE "DIAG|Error|Traceback" | tail -16
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
BIAS = 0.08
TIPS = ("rl_dg_1_tip", "rl_dg_2_tip")
CELL = os.environ["UWLAB_DIAG_CELL"]
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

try:
    arm_ids, _ = robot.find_joints(["shoulder.*", "elbow.*", "wrist.*"])
    k = _t(robot.data.joint_stiffness)[:, arm_ids].mean()
    print(f"DIAG cell={CELL} arm_joint_stiffness_mean={float(k):.1f}")
except Exception as exc:
    print(f"DIAG cell={CELL} stiffness_readback_skipped {type(exc).__name__}")


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
            f"DIAG cell={CELL} step={step + 1:4d} in_hand={in_hand:.3f} "
            f"hand_sag_m={sag:+.4f} tip_force_N={tip_force():.3f} cube_to_tip={float(d.mean()):.4f}"
        )
env.close()
app.close()
PYEOF
done
echo "=== DIAG_CONTROL_DONE $(date -Is)"
