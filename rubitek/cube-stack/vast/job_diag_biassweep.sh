#!/usr/bin/env bash
# HOW LONG DOES THE GRIP BIAS ACTUALLY HOLD, AND AT WHAT VALUE?
#
# job_diag_gripbias.sh proved the 0.05 rad bias fixes the first-step release on C3:
#   held@1 0.078 -> 1.000, held@5 0.078 -> 0.711.
# But it decays: held@20 0.141, held@40 0.102. Two things that measurement could not separate:
#
#   1. ARM DROOP vs REAL SLIP. "held" was z > z0 - 0.03, so the OSC arm sagging past 3 cm scores as a
#      drop even with the cube still pinched. The C4 row is the tell -- a cube RESTING on the base
#      cube cannot fall 3 cm, yet it scored 0.266 at step 40. So this run tracks the cube RELATIVE TO
#      THE FINGERTIPS (that is the grasp), and reports hand height separately as its own number.
#   2. HOW MUCH BIAS. 0.05 is the shipped GRIP_BIAS_RAD. The bias stays a bias rather than a
#      constraint for any value below the 0.1 rad/step action authority, so 0.08 and 0.095 are still
#      policy-overridable. If holding time scales with bias, the carry gets the horizon it needs.
#
# C3 only, zero action, 60 steps (longer than the diagnostics above, to cover a full carry).
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f

echo "=== DIAG_BIASSWEEP_START $(date -Is)"
timeout 2400 "$PY" - <<'PYEOF' 2>&1 | grep -aE "DIAG|Error|Traceback" | tail -80
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
STEPS = 60
TIPS = ("rl_dg_1_tip", "rl_dg_2_tip")
_t = lambda x: getattr(x, "torch", x)  # noqa: E731


def run(bias):
    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
    ev = cfg.events.reset_from_reset_states
    ev.params["reset_types"] = ["ObjectAnywhereEEGrasped"]
    ev.params["probs"] = [1.0]
    g = cfg.actions.gripper
    cfg.actions.gripper = g.replace(use_zero_offset=False, offset={n: bias for n in g.scale})

    env = gym.make(TASK, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    obj = env.scene["insertive_object"]

    names = list(robot.body_names)
    tip_ids = [names.index(t) for t in TIPS if t in names]
    print(f"DIAG bias={bias} tips_found={[names[i] for i in tip_ids]}")

    env.reset()
    pos0 = _t(obj.data.root_pos_w).clone()
    tips0 = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1).clone()
    # Cube position in the hand's own frame at reset -- the grasp we must not lose.
    d0 = (pos0 - tips0).norm(dim=-1)
    print(f"DIAG bias={bias} cube_to_tip_mean_at_reset={float(d0.mean()):.4f} m")

    act = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    for step in range(STEPS):
        env.step(act)
        if step in (0, 4, 19, 39, 59):
            pos = _t(obj.data.root_pos_w)
            tips = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1)
            d = (pos - tips).norm(dim=-1)
            # IN HAND: the cube has not moved away from the fingertips by more than 2 cm.
            in_hand = float(((d - d0).abs() < 0.02).float().mean())
            # The old z-only metric, for comparison with the earlier runs.
            z_held = float((pos[:, 2] > pos0[:, 2] - 0.03).float().mean())
            hand_drop = float((tips0[:, 2] - tips[:, 2]).mean())
            print(
                f"DIAG bias={bias} step={step + 1:3d} in_hand={in_hand:.3f} "
                f"z_held={z_held:.3f} hand_sag_m={hand_drop:+.4f} cube_to_tip={float(d.mean()):.4f}"
            )
    env.close()


for bias in (0.05, 0.08, 0.095):
    try:
        run(bias)
    except Exception as exc:
        print(f"DIAG bias={bias} FAILED {type(exc).__name__}: {exc}")

app.close()
PYEOF
echo "=== DIAG_BIASSWEEP_DONE $(date -Is)"
