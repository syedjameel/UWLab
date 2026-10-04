#!/usr/bin/env bash
# GRIP ENDURANCE SWEEP, run on the local 4070 Ti instead of the rented box.
#
# The whole Isaac Sim 6 stack is present locally in .venv-is6, the banks are local, and the probe is
# 128 envs for 150 steps -- so the measurement that decides the night does not have to wait behind a
# 64 KB/s link. The H100 is for the long training run, not for this.
#
# WHAT IS BEING DECIDED. Four environment defects are fixed; what remains is ENDURANCE. With the
# latched hand and a null action the C3 bank holds the cube in ~90% of envs at step 5 and ~5% by step
# 120, and success is scored ONLY at step 160. A policy cannot learn a carry whose grasp is gone
# five-sixths of the way through the episode.
#
#   UWLAB_HAND_EFFORT_SCALE -- fingertip force measured 2.07 N against caps permitting ~2.2 N, so the
#   hand is torque-SATURATED. That is why raising the grip bias 5x moved force not at all. Anything
#   above 1.0 makes the sim hand stronger than the real DG-5F: a fidelity trade, to be reported.
#
#   UWLAB_TUCK_OTHERS -- fingers 3/4/5 are driven by nothing here and splay open beside the cube.
#   Curling them is Matte's suggestion and had never been tested.
#
# READ AS: which condition keeps in_hand at step 150 above ~0.5. Run-to-run variance on identical
# configs has measured 0.062-0.180, so ignore differences below ~0.15.
set -uo pipefail
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv-is6/bin/python
cd "$REPO" || exit 1
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="/home/mattephi/Work/Rubitek/uwlab-env/compat:$REPO/source/uwlab_assets:$REPO/source/uwlab_tasks:$REPO/source/uwlab:$REPO/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-0.15
export UWLAB_LATCHED_HAND=1

N=${N:-128}
# name:HAND_EFFORT_SCALE:TUCK_OTHERS:GRIP_BIAS
CONDS=${CONDS:-"base:1:0:0.05 tuck:1:0.6:0.05 eff3:3:0:0.05 eff3tuck:3:0.6:0.05 eff10tuck:10:0.6:0.05 eff3tuckbias:3:0.6:0.15"}

echo "=== PROBE_START $(date -Is) envs=$N"
for c in $CONDS; do
  IFS=: read -r NAME EFF TUCK BIAS <<<"$c"
  echo "=== COND $NAME effort=x$EFF tuck=$TUCK bias=$BIAS $(date -Is)"
  UWLAB_HAND_EFFORT_SCALE="$EFF" UWLAB_TUCK_OTHERS="$TUCK" UWLAB_GRIP_BIAS_RAD="$BIAS" \
  UWLAB_PROBE_NAME="$NAME" UWLAB_PROBE_ENVS="$N" \
  timeout 1800 "$PY" - <<'PYEOF' 2>&1 | grep -aE "PROBE|hand action ->|Error|Traceback|Fatal" | tail -20
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
N = int(os.environ.get("UWLAB_PROBE_ENVS", "128"))
NAME = os.environ["UWLAB_PROBE_NAME"]
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
held = len(getattr(gterm, "_hold_ids", []) or [])
print(f"PROBE {NAME} term={type(gterm).__name__} dim={gterm.action_dim} held_joints={held}")
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
z0 = pos0[:, 2].clone()
zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)

# Stop at 150: the episode is 160 control steps, and sampling at or past the boundary reads envs the
# time-out has already reset -- which fakes a recovery (in_hand rebounding) that never happened.
SAMPLES = (4, 19, 59, 119, 149)
for step in range(150):
    env.step(zero)
    if step in SAMPLES:
        pos = _t(obj.data.root_pos_w)
        tips = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1)
        d = (pos - tips).norm(dim=-1)
        in_hand = float(((d - d0).abs() < 0.02).float().mean())
        dq = float((_t(robot.data.joint_pos)[:, fidx] - q0).mean())
        drop = float((pos[:, 2] - z0).mean())
        print(
            f"PROBE {NAME} step={step + 1:4d} in_hand={in_hand:.3f} finger_dq_rad={dq:+.4f} "
            f"tip_force_N={tip_force():.3f} cube_to_tip={float(d.mean()):.4f} cube_dz={drop:+.4f}"
        )
env.close()
app.close()
PYEOF
done
echo "=== PROBE_DONE $(date -Is)"
