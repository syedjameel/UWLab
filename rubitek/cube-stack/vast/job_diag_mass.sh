#!/usr/bin/env bash
# CUBE MASS, DONE CORRECTLY -- AND HOW NOISY ARE THESE NUMBERS ANYWAY?
#
# TWO CORRECTIONS TO job_diag_dr.sh, WHICH PRODUCED INVALID CELLS:
#
# 1. Its mass cell pinned EVERY event param whose name contained "mass", which caught
#    randomize_robot_mass and randomize_table_mass. Those carry SCALE FACTORS (U(0.7,1.3)), not
#    kilograms, so setting them to 0.03 made the robot and the table 3% of their mass. The cube then
#    flew to 0.59 m from the tips. The `mass` and `both` cells measured a broken scene, not cube
#    mass. Only `randomize_insertive_object_mass` -- the cube being carried -- is pinned here.
#
# 2. Identical configs disagreed across runs: bias 0.08 / seg1 40 / gravity -1.0 gave in_hand@120
#    = 0.180 in the grip sweep and 0.094 as job_diag_dr's control. A 2x spread means the grip
#    sweep's "best setting" may have been noise and nothing measured so far can be ranked. So the
#    control is run TWICE here, first and last, to bound the run-to-run spread directly.
#
# Cube mass ships as U(0.02, 0.06) kg = 0.20-0.59 N. If the light end holds and the heavy end does
# not, the pinch works and the mass range is what makes C3 unlearnable.
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=40 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-1.0

echo "=== DIAG_MASS_START $(date -Is)"
for CELL in control_a light heavy control_b; do
  echo "=== MASSCELL=$CELL $(date -Is)"
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
MASS = {"light": 0.02, "heavy": 0.06}.get(CELL)
_t = lambda x: getattr(x, "torch", x)  # noqa: E731

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
ev = cfg.events.reset_from_reset_states
ev.params["reset_types"] = ["ObjectAnywhereEEGrasped"]
ev.params["probs"] = [1.0]
g = cfg.actions.gripper
cfg.actions.gripper = g.replace(use_zero_offset=False, offset={n: BIAS for n in g.scale})

# ONLY the carried cube. Not the robot, not the table -- those params are scale factors.
if MASS is not None:
    term = getattr(cfg.events, "randomize_insertive_object_mass", None)
    if term is None:
        print(f"DIAG cell={CELL} ERROR randomize_insertive_object_mass missing")
    else:
        term.params["mass_distribution_params"] = (MASS, MASS)
        print(f"DIAG cell={CELL} cube_mass_pinned_kg={MASS} weight_N={MASS * 9.81:.3f}")

env = gym.make(TASK, cfg=cfg).unwrapped
robot = env.scene["robot"]
obj = env.scene["insertive_object"]
names = list(robot.body_names)
tip_ids = [names.index(t) for t in TIPS if t in names]

env.reset()
# Readback of the ACTUAL sampled mass, so the pin is verified rather than trusted. Guarded:
# get_masses() returns a raw warp array with no .torch and no .float, and an unguarded
# float(m.float().mean()) threw AttributeError AFTER reset -- killing all four cells before a
# single measurement was taken. A verification line must never be able to kill the measurement.
try:
    import warp as wp

    _m = obj.root_physx_view.get_masses()
    _mt = getattr(_m, "torch", None)
    if _mt is None:
        _mt = wp.to_torch(_m)
    print(f"DIAG cell={CELL} actual_cube_mass_kg_mean={float(_mt.float().mean()):.4f}")
except Exception as _exc:
    print(f"DIAG cell={CELL} mass_readback_skipped {type(_exc).__name__}: {_exc}")
pos0 = _t(obj.data.root_pos_w).clone()
tips0 = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1).clone()
d0 = (pos0 - tips0).norm(dim=-1)
zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)

for step in range(120):
    env.step(zero)
    if step in (4, 19, 119):
        pos = _t(obj.data.root_pos_w)
        tips = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1)
        d = (pos - tips).norm(dim=-1)
        in_hand = float(((d - d0).abs() < 0.02).float().mean())
        print(f"DIAG cell={CELL} step={step + 1:4d} in_hand={in_hand:.3f} cube_to_tip={float(d.mean()):.4f}")
env.close()
app.close()
PYEOF
done
echo "=== DIAG_MASS_DONE $(date -Is)"
