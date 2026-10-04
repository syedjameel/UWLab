#!/usr/bin/env bash
# THE BIAS WAS NEVER STRONG ENOUGH -- TEST THE RANGE I WRONGLY RULED OUT.
#
# The reset films show the two-finger pinch holding the cube steadily at ~1.2 N per fingertip. They
# hold because the replay tool commands an ABSOLUTE CLOSED POSTURE, driving the fingers well past
# contact: a large PD error, hence real force. Every probe I ran instead used a 0.05-0.08 rad
# relative bias -- on 0.30 N*m/rad distal joints that is ~0.015 N*m, a fraction of a newton. So the
# pinch was never given the force the films demonstrate it can apply. "Two fingers cannot hold this
# cube" was my error, not a measurement.
#
# WHY I CAPPED IT AT 0.1: the hand action documents +-0.1 rad/step of policy authority, so a bias at
# or above that looked like it would prevent the policy from ever opening the hand. But the clip is
# applied to the PROCESSED value (raw * scale = raw * 0.1) and clipped to (-1, 1) rad -- so the real
# per-step authority is +-1.0 rad, ten times the documented figure. A 0.25 rad bias still leaves the
# policy [-0.75, +1.25] rad/step: strong grip AND fully overridable. That range was never tested.
#
# Reports fingertip CONTACT FORCE next to in_hand, so the result is directly comparable to the
# films' ~1.2 N rather than to a proxy.
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-1.0

echo "=== DIAG_HIGHBIAS_START $(date -Is)"
for BIAS in 0.08 0.15 0.25 0.40; do
  echo "=== HIGHBIAS=$BIAS $(date -Is)"
  UWLAB_DIAG_BIAS="$BIAS" timeout 1800 "$PY" - <<'PYEOF' 2>&1 | grep -aE "DIAG|Error|Traceback" | tail -16
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
BIAS = float(os.environ["UWLAB_DIAG_BIAS"])
TIPS = ("rl_dg_1_tip", "rl_dg_2_tip")
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


def tip_force():
    """Mean fingertip contact force, guarded: a readback must never kill the measurement."""
    try:
        sensor = env.scene["object_hand_s"]
        f = _t(sensor.data.force_matrix_w)
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
        print(
            f"DIAG bias={BIAS} step={step + 1:4d} in_hand={in_hand:.3f} "
            f"tip_force_N={tip_force():.3f} cube_to_tip={float(d.mean()):.4f}"
        )
env.close()
app.close()
PYEOF
done
echo "=== DIAG_HIGHBIAS_DONE $(date -Is)"
