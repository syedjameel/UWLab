#!/usr/bin/env bash
# DOES THE GRIP-BIAS TASK VARIANT FIX THE C3 COLLAPSE?
#
# job_diag_grip.sh established that on the plain TwoFinger id a zero action means "target := measured
# joint position", i.e. zero PD error and zero grip force, so a mid-air C3 reset loses the cube on the
# first control step (held fraction 0.047 with shipped DR, 0.070 with friction pinned -- friction was
# never the cause). The repo already ships the fix as a sibling task id whose hand action carries
# offset=GRIP_BIAS_RAD with use_zero_offset=False, so raw 0 means "keep squeezing".
#
# Same C3-only zero-action replay, shipped friction DR, run on BOTH ids back to back.
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f

echo "=== DIAG_GRIPBIAS_START $(date -Is)"
timeout 1800 "$PY" - <<'PYEOF' 2>&1 | grep -aE "DIAG|Error|Traceback" | tail -60
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
import uwlab_is6_bridge  # noqa: E402

uwlab_is6_bridge.apply()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

BASE = "OmniReset-UR10eDelto-CubeStackTwoFinger-RelCartesianOSC-State-v0"
GRIP = "OmniReset-UR10eDelto-CubeStackTwoFingerGrip-RelCartesianOSC-State-v0"
N = 128
_t = lambda x: getattr(x, "torch", x)  # noqa: E731


def run(task, family):
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=N)
    ev = cfg.events.reset_from_reset_states
    ev.params["reset_types"] = [family]
    ev.params["probs"] = [1.0]
    env = gym.make(task, cfg=cfg).unwrapped
    obj = env.scene["insertive_object"]
    tag = "GRIP" if "TwoFingerGrip" in task else "BASE"

    gterm = env.action_manager.get_term("gripper")
    off = getattr(gterm, "_offset", None)
    off_s = float(_t(off).abs().max()) if off is not None and hasattr(off, "shape") else off
    print(f"DIAG {tag} family={family} gripper_offset_max={off_s}")

    env.reset()
    z0 = _t(obj.data.root_pos_w)[:, 2].clone()
    act = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    for step in range(40):
        env.step(act)
        if step in (0, 4, 19, 39):
            z = _t(obj.data.root_pos_w)[:, 2]
            held = float((z > z0 - 0.03).float().mean())
            print(f"DIAG {tag} family={family} step={step + 1:3d} still_held_fraction={held:.3f}")
    print(f"DIAG {tag} family={family} start_heights_mean={float(z0.mean()):.3f} n_envs={N}")
    env.close()


for task in (BASE, GRIP):
    for family in ("ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped"):
        try:
            run(task, family)
        except Exception as exc:
            print(f"DIAG FAILED task={task} family={family} {type(exc).__name__}: {exc}")

app.close()
PYEOF
echo "=== DIAG_GRIPBIAS_DONE $(date -Is)"
