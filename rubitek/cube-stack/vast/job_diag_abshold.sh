#!/usr/bin/env bash
# THE CUBE IS BEING EJECTED, NOT DROPPED -- AND THE CAUSE IS A RELATIVE ACTION ON A SUBSTEP LOOP.
#
# The positive control settled it. With the arm held rigid (joint stiffness 600) and again with the
# gravity null properly bracketed (-0.15/-0.20 hold the hand to within 20 mm over 12 s), the cube
# STILL leaves: in_hand 0.59 -> 0.07 -> 0.016. So it is not arm drift, not friction, not cube mass,
# not finger count, not bias magnitude -- every one of those is now excluded by measurement.
#
# What the force trace shows instead: tip force at step 5 is 0.89-1.15 N, matching the films' ~1.2 N,
# then collapses to ~0.05 N by step 20 with the cube 0.20 m away. Film-level grip force IS present,
# and the cube leaves anyway. That is ejection, not slipping.
#
# MECHANISM: RelativeJointPositionAction.apply_actions() runs EVERY PHYSICS SUBSTEP and recomputes
#     target = measured_joint_pos + offset
# so a constant 0.08 rad "bias" is not a set point at all -- it is a RAMP of 0.08 rad per substep,
# 12 substeps per control step. The fingers drive continuously further closed until a 34 mm cube is
# squeezed out from between them, after which the hand is a closed fist (force ~0) and the cube is
# long gone. This predicts every flat sweep: more bias just ejects it sooner, which is why 0.40 rad
# was worse at step 5 than 0.08.
#
# THE TEST: give the hand an ABSOLUTE position target -- the posture the bank recorded plus a fixed
# squeeze -- instead of a per-substep relative offset. Absolute means the target stops advancing once
# reached, so the fingers hold a set point rather than crushing through it. Finger joint angles are
# printed at every sample: under the relative bias they should climb monotonically past closure;
# under the absolute hold they should settle.
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-0.15

echo "=== DIAG_ABSHOLD_START $(date -Is)"
for MODE in rel_bias abs_hold0 abs_squeeze; do
  echo "=== MODE=$MODE $(date -Is)"
  UWLAB_DIAG_MODE="$MODE" timeout 1800 "$PY" - <<'PYEOF' 2>&1 | grep -aE "DIAG|Error|Traceback" | tail -20
import os

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
import uwlab_is6_bridge  # noqa: E402

uwlab_is6_bridge.apply()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "OmniReset-UR10eDelto-CubeStackTwoFingerGripCarry-RelCartesianOSC-State-v0"
N = 128
MODE = os.environ["UWLAB_DIAG_MODE"]
SQUEEZE = {"abs_hold0": 0.0, "abs_squeeze": 0.05}.get(MODE, 0.0)
TIPS = ("rl_dg_1_tip", "rl_dg_2_tip")
_t = lambda x: getattr(x, "torch", x)  # noqa: E731

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
ev = cfg.events.reset_from_reset_states
ev.params["reset_types"] = ["ObjectAnywhereEEGrasped"]
ev.params["probs"] = [1.0]

g = cfg.actions.gripper
if MODE == "rel_bias":
    cfg.actions.gripper = g.replace(use_zero_offset=False, offset={n: 0.08 for n in g.scale})
else:
    # ABSOLUTE joint position: target = action * 1.0 + 0.0, so the action IS the commanded angle and
    # it stops advancing once reached -- unlike the relative term, which re-adds its offset to the
    # live measurement on every one of the 12 substeps.
    cfg.actions.gripper = JointPositionActionCfg(
        asset_name="robot", joint_names=list(g.scale), scale=1.0, offset=0.0, use_default_offset=False
    )

env = gym.make(TASK, cfg=cfg).unwrapped
robot = env.scene["robot"]
obj = env.scene["insertive_object"]
names = list(robot.body_names)
tip_ids = [names.index(t) for t in TIPS if t in names]
gterm = env.action_manager.get_term("gripper")
gdim = gterm.action_dim
fidx = list(gterm._joint_ids) if not isinstance(gterm._joint_ids, slice) else list(range(robot.num_joints))
print(f"DIAG mode={MODE} gripper_dim={gdim} squeeze={SQUEEZE}")


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
# Latch the posture the reset produced; the absolute modes hold exactly this (plus a squeeze).
q_latched = _t(robot.data.joint_pos)[:, fidx].clone()

act = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
if MODE != "rel_bias":
    act[:, -gdim:] = q_latched + SQUEEZE

for step in range(120):
    env.step(act)
    if step in (4, 19, 79, 119):
        pos = _t(obj.data.root_pos_w)
        tips = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1)
        d = (pos - tips).norm(dim=-1)
        in_hand = float(((d - d0).abs() < 0.02).float().mean())
        q = _t(robot.data.joint_pos)[:, fidx]
        dq = float((q - q_latched).mean())
        print(
            f"DIAG mode={MODE} step={step + 1:4d} in_hand={in_hand:.3f} "
            f"finger_dq_rad={dq:+.4f} tip_force_N={tip_force():.3f} cube_to_tip={float(d.mean()):.4f}"
        )
env.close()
app.close()
PYEOF
done
echo "=== DIAG_ABSHOLD_DONE $(date -Is)"
