#!/usr/bin/env bash
# WHY C3 RESETS LOSE THE CUBE IN THE TRAINING ENV.
#
# job_diag_c3.sh measured the C3 bank replayed in the TRAINING env with zero actions: only 4.7% of
# envs still held the cube, and pinning the ROBOT material friction to 1.0 moved that to just 7.0%.
# Two things that first pass did not establish, both tested here:
#
#   1. FRICTION, PROPERLY. It pinned only `events.robot_material`. If the OBJECT's material (or any
#      other friction-bearing event term) is randomized too, the grasp can still slip. This run
#      enumerates EVERY event param whose name contains "friction" and pins them all.
#   2. THE GRIPPER ACTION. The held fraction was identical at steps 5/10/20/40 -- the cube is gone
#      within the first half-second, which is a release, not a slip. A zero action vector may command
#      the hand OPEN on step 1, overriding the finger joint state the reset wrote. So: zero action vs
#      gripper dims driven to +1 and to -1.
#
# Prints finger joint positions and targets at reset and after one step, so a release is visible
# directly rather than inferred from the cube falling.
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f

echo "=== DIAG_GRIP_START $(date -Is)"
timeout 1800 "$PY" - <<'PYEOF' 2>&1 | grep -aE "DIAG|Error|Traceback|error" | tail -80
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
_t = lambda x: getattr(x, "torch", x)  # noqa: E731


def friction_terms(cfg):
    """Every event param whose name mentions friction, as (term, param, value)."""
    out = []
    for term_name in dir(cfg.events):
        if term_name.startswith("_"):
            continue
        term = getattr(cfg.events, term_name, None)
        params = getattr(term, "params", None)
        if not isinstance(params, dict):
            continue
        for p, v in params.items():
            if "friction" in p.lower():
                out.append((term_name, p, v))
    return out


def build_cfg(pin_friction):
    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
    # Only C3 starts, so every env exercises the held-in-air family.
    ev = cfg.events.reset_from_reset_states
    ev.params["reset_types"] = ["ObjectAnywhereEEGrasped"]
    ev.params["probs"] = [1.0]
    found = friction_terms(cfg)
    for term_name, p, v in found:
        print(f"DIAG friction_param {term_name}.{p} = {v}")
    if pin_friction:
        for term_name, p, v in found:
            if not (isinstance(v, (tuple, list)) and len(v) == 2):
                continue
            hi = 0.9 if "dynamic" in p.lower() else 1.0
            getattr(cfg.events, term_name).params[p] = (hi, hi)
            print(f"DIAG friction_pinned {term_name}.{p} -> ({hi}, {hi})")
    return cfg


def run(mode, pin_friction, grip_value):
    cfg = build_cfg(pin_friction)
    env = gym.make(TASK, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    obj = env.scene["insertive_object"]

    am = env.action_manager
    names = list(am.active_terms)
    dims = []
    for n in names:
        try:
            dims.append(int(am.get_term(n).action_dim))
        except Exception:
            dims.append(-1)
    print(f"DIAG action_terms mode={mode} {list(zip(names, dims))} total={am.total_action_dim}")

    # Finger dims = every term that is not the cartesian/arm term.
    slices, off = {}, 0
    for n, d in zip(names, dims):
        if d <= 0:
            break
        slices[n] = (off, off + d)
        off += d
    hand_terms = [n for n in slices if not any(k in n.lower() for k in ("arm", "cartesian", "osc", "ee_", "pose"))]
    print(f"DIAG hand_terms mode={mode} {hand_terms}")

    fidx = [i for i, jn in enumerate(robot.joint_names) if "dg" in jn.lower()]
    fnames = [robot.joint_names[i] for i in fidx]

    env.reset()
    z0 = _t(obj.data.root_pos_w)[:, 2].clone()
    q_reset = _t(robot.data.joint_pos)[:, fidx].mean(dim=0).clone()
    print(f"DIAG mode={mode} finger_joints={fnames}")
    print(f"DIAG mode={mode} q_after_reset={[round(float(v), 4) for v in q_reset]}")

    act = torch.zeros((env.num_envs, am.total_action_dim), device=env.device)
    for n in hand_terms:
        a, b = slices[n]
        act[:, a:b] = grip_value

    for step in range(40):
        env.step(act)
        if step == 0:
            q1 = _t(robot.data.joint_pos)[:, fidx].mean(dim=0)
            tgt = _t(robot.data.joint_pos_target)[:, fidx].mean(dim=0)
            print(f"DIAG mode={mode} q_after_step1={[round(float(v), 4) for v in q1]}")
            print(f"DIAG mode={mode} target_after_step1={[round(float(v), 4) for v in tgt]}")
        if step in (0, 4, 19, 39):
            z = _t(obj.data.root_pos_w)[:, 2]
            held = float((z > z0 - 0.03).float().mean())
            print(f"DIAG mode={mode} step={step + 1:3d} still_held_fraction={held:.3f}")
    print(f"DIAG mode={mode} start_heights_mean={float(z0.mean()):.3f} n_envs={N}")
    env.close()


for mode, pin, grip in [
    ("zero_dr", False, 0.0),
    ("zero_pinnedall", True, 0.0),
    ("close_pos_dr", False, 1.0),
    ("close_neg_dr", False, -1.0),
]:
    try:
        run(mode, pin, grip)
    except Exception as exc:  # keep the remaining modes running
        print(f"DIAG mode={mode} FAILED {type(exc).__name__}: {exc}")

app.close()
PYEOF
echo "=== DIAG_GRIP_DONE $(date -Is)"
