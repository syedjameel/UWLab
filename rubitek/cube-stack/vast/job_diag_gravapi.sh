#!/usr/bin/env bash
# WHAT DOES PHYSX ACTUALLY RETURN FOR GENERALIZED GRAVITY FORCES?
#
# The gravity-comp patch enabled cleanly ("gravity compensation ENABLED") and then died with
#   TypeError: '<' not supported between instances of 'list' and 'int'
# The job log grep kept only the last traceback line, so the frame is unknown. The prime suspect is
# the return type: `getattr(g, "torch", g)` handles an IsaacLab ProxyArray but NOT a raw warp array,
# and `g[:, self._joint_ids]` with a LIST of joint ids is exactly the kind of index a warp array
# rejects with a list-vs-int comparison.
#
# This prints the accessor names that exist, the concrete return type, and which conversion yields a
# torch tensor of the right shape -- with the FULL traceback, unfiltered.
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
export UWLAB_OSC_GRAVITY_COMP=0   # inspect the API directly; do not route through the patch

echo "=== DIAG_GRAVAPI_START $(date -Is)"
timeout 900 "$PY" - <<'PYEOF' 2>&1 | tail -60
import traceback

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
import uwlab_is6_bridge  # noqa: E402

uwlab_is6_bridge.apply()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "OmniReset-UR10eDelto-CubeStackTwoFingerGripCarry-RelCartesianOSC-State-v0"
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=8)
env = gym.make(TASK, cfg=cfg).unwrapped
env.reset()
robot = env.scene["robot"]
view = robot.root_physx_view

print("GRAVAPI view_type:", type(view))
cands = [n for n in dir(view) if "grav" in n.lower() or "coriolis" in n.lower() or "mass_matrix" in n.lower()]
print("GRAVAPI candidates:", cands)

term = env.action_manager.get_term("arm")
jids = term._joint_ids
print("GRAVAPI joint_ids:", jids, type(jids))

for name in cands:
    try:
        fn = getattr(view, name)
        g = fn()
        print(f"GRAVAPI {name} -> type={type(g)} shape={getattr(g, 'shape', None)}")
        for conv in ("torch_attr", "wp.to_torch", "as_tensor"):
            try:
                if conv == "torch_attr":
                    t = getattr(g, "torch", None)
                    if t is None:
                        print(f"GRAVAPI   {conv}: absent")
                        continue
                elif conv == "wp.to_torch":
                    import warp as wp

                    t = wp.to_torch(g)
                else:
                    t = torch.as_tensor(g)
                print(f"GRAVAPI   {conv}: OK type={type(t)} shape={tuple(t.shape)}")
                sub = t[:, jids] if not isinstance(jids, slice) else t
                print(f"GRAVAPI   {conv}: indexed shape={tuple(sub.shape)} mean={float(sub.float().mean()):+.4f}")
            except Exception as exc:
                print(f"GRAVAPI   {conv}: FAILED {type(exc).__name__}: {exc}")
    except Exception:
        print(f"GRAVAPI {name} call FAILED:")
        traceback.print_exc()

env.close()
app.close()
PYEOF
echo "=== DIAG_GRAVAPI_DONE $(date -Is)"
