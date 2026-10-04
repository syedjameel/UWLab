#!/usr/bin/env bash
# IS DOMAIN RANDOMIZATION WHAT BREAKS THE PINCH, NOW THAT THE PINCH HAS FORCE?
#
# The grip-force sweep found NO setting that holds: in_hand@120 ran 0.086-0.180 across
# bias 0.05/0.08 x seg1 stiffness 10/40/80, and MORE squeeze was worse early (bias 0.08 stiff 80
# gave the worst step-5 hold, 0.297) -- consistent with over-squeezing ejecting the cube rather than
# gripping it. So raw force is not the knob.
#
# The one interaction never validly tested: friction. This morning's friction probe ran with a LIMP
# hand (action 0 meant zero grip force), so friction could not possibly have mattered -- that test
# was uninformative by construction, not evidence. Now there IS force, and:
#   robot friction  U(0.3, 1.2)   -- at the low end a two-finger pinch on a smooth cube just slides
#   cube mass       U(0.02, 0.06) kg = 0.20-0.59 N  -- the heavy tail is 3x the light one
# Either could be what decides whether the pinch holds, and both are shipped training DR.
#
# Four cells: control, friction pinned, mass pinned, both pinned. Grip at the sweep's best
# (bias 0.08, seg1 40), arm at the best-holding gravity scale. If "both pinned" holds and the
# control does not, the pinch works and the DR ranges are what make C3 unlearnable.
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=40 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-1.0

echo "=== DIAG_DR_START $(date -Is)"
for CELL in control fric mass both; do
  echo "=== DRCELL=$CELL $(date -Is)"
  UWLAB_DIAG_CELL="$CELL" timeout 1800 "$PY" - <<'PYEOF' 2>&1 | grep -aE "DIAG|Error|Traceback" | tail -24
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


def pin(kind):
    """Collapse every event param whose name mentions friction / mass to a single value."""
    for term_name in dir(cfg.events):
        if term_name.startswith("_"):
            continue
        term = getattr(cfg.events, term_name, None)
        params = getattr(term, "params", None)
        if not isinstance(params, dict):
            continue
        for p, v in list(params.items()):
            low = p.lower()
            hit = ("friction" in low) if kind == "fric" else ("mass" in low and "range" in low or "mass_distribution" in low)
            if not hit or not isinstance(v, (tuple, list)) or len(v) != 2:
                continue
            if kind == "fric":
                val = 0.9 if "dynamic" in low else 1.0
            else:
                val = 0.03  # mid of the shipped U(0.02, 0.06) kg cube-mass range
            params[p] = (val, val)
            print(f"DIAG cell={CELL} pinned {term_name}.{p} -> ({val}, {val})")


if CELL in ("fric", "both"):
    pin("fric")
if CELL in ("mass", "both"):
    pin("mass")

env = gym.make(TASK, cfg=cfg).unwrapped
robot = env.scene["robot"]
obj = env.scene["insertive_object"]
names = list(robot.body_names)
tip_ids = [names.index(t) for t in TIPS if t in names]

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
            f"DIAG cell={CELL} step={step + 1:4d} in_hand={in_hand:.3f} "
            f"cube_to_tip={float(d.mean()):.4f}"
        )
env.close()
app.close()
PYEOF
done
echo "=== DIAG_DR_DONE $(date -Is)"
