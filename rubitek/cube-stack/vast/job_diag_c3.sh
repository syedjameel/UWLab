#!/usr/bin/env bash
# WHY C3 IS STUCK AT ZERO. Stage B trained 212 iterations with C4 and C3 at 50/50 and C3
# (ObjectAnywhereEEGrasped) never once succeeded, while C4 (ObjectPartiallyAssembledEEGrasped) sat at
# 0.38-0.48. The asymmetry points at the grip: a C3 state holds the cube in MID-AIR, so a grip that
# slips ends the episode unrecoverably, while a C4 state has the cube already resting partly on the
# receptive cube and survives a weak grip.
#
# The suspect is friction. The pinch banks were RECORDED with robot friction pinned at 1.0 (the value
# the grasps were sampled under), but rl_state_cfg randomizes robot material friction over
# 0.3-1.2 at startup. At the low end a 0.06 N*m two-finger pinch cannot hold a 30 g cube.
#
# This measures it directly in the TRAINING env: reset from the C3 bank, step with zero action, and
# report the fraction of envs whose cube is still in the hand after N steps -- once with the shipped
# friction DR, once with friction pinned at 1.0.
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f

for MODE in dr pinned; do
  echo "=== MODE=$MODE $(date -Is)"
  UWLAB_DIAG_FRICTION_MODE=$MODE timeout 900 "$PY" - <<'PYEOF' 2>&1 | grep -aE "DIAG|Error|Traceback" | tail -12
import os
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
import uwlab_is6_bridge  # noqa: E402

uwlab_is6_bridge.apply()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "OmniReset-UR10eDelto-CubeStackTwoFinger-RelCartesianOSC-State-v0"
N = 128
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)

mode = os.environ.get("UWLAB_DIAG_FRICTION_MODE", "dr")
mat = getattr(cfg.events, "robot_material", None)
if mat is not None:
    lo, hi = mat.params["static_friction_range"]
    if mode == "pinned":
        mat.params["static_friction_range"] = (1.0, 1.0)
        mat.params["dynamic_friction_range"] = (0.9, 0.9)
    print(f"DIAG friction mode={mode} shipped_static_range=({lo}, {hi}) -> {mat.params['static_friction_range']}")

# Only C3 starts, so every env exercises the held-in-air family.
ev = cfg.events.reset_from_reset_states
ev.params["reset_types"] = ["ObjectAnywhereEEGrasped"]
ev.params["probs"] = [1.0]

env = gym.make(TASK, cfg=cfg).unwrapped
obj = env.scene["insertive_object"]
_t = lambda x: getattr(x, "torch", x)  # noqa: E731

env.reset()
z0 = _t(obj.data.root_pos_w)[:, 2].clone()
zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
held_hist = []
for step in range(40):
    env.step(zero)
    z = _t(obj.data.root_pos_w)[:, 2]
    # "still up" = cube has not fallen more than 3 cm from where the reset placed it
    held = (z > z0 - 0.03)
    if step in (4, 9, 19, 39):
        held_hist.append((step + 1, float(held.float().mean())))
for s, f in held_hist:
    print(f"DIAG mode={mode} step={s:3d} still_held_fraction={f:.3f}")
print(f"DIAG mode={mode} start_heights_mean={float(z0.mean()):.3f} n_envs={N}")
env.close()
app.close()
PYEOF
done
echo "=== DIAG_DONE $(date -Is)"
