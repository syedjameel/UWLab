#!/usr/bin/env bash
# DOES GRAVITY COMPENSATION STOP THE ARM SAGGING?
#
# The OSC action term computes tau = J^T @ (Kp*pose_err + Kd*vel_err) and NOTHING else, while the
# arm joints carry stiffness 0.0 / damping 0.0. So a zero action commands exactly zero torque to a
# 6-DOF arm holding a 1.7735 kg hand, and it falls: measured 91 mm of fingertip sag over 60 control
# steps, with the grasped cube left 0.26 m behind. 17.4 N against Kp_z = 200 N/m predicts 87 mm.
# `process_actions` also re-latches the desired pose to the MEASURED pose each control step, so the
# accumulated sag never produces a restoring error -- no constant command can mean "stay here".
#
# ur10e_delto.py:214-222 concluded "the fix belongs in the action term, not here" after stiff joint
# gains proved unstable at dt = 1/120. The action term never got it. UWLAB_OSC_GRAVITY_COMP=1 adds
# tau += g(q) from PhysX's generalized gravity forces. The deployed robot compensates gravity via
# setPayload, so OFF is the sim2real mismatch, not ON.
#
# Success here = sag collapses toward 0 and the cube stays in hand across a FULL 160-step episode.
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f

echo "=== DIAG_GRAVCOMP_START $(date -Is)"
# Three modes in one job. PhysX's accessor is `get_gravity_compensation_forces` (measured: raw warp
# array, shape (envs, 26), mean -12.96 N*m over the six arm joints). The NAME implies it is already
# the cancelling torque, so sign=+1 should null the droop -- but if the convention is the gravity
# LOAD instead, +1 doubles the sag and -1 is correct. Testing both costs one extra 40 s run and
# saves a whole push/queue/poll cycle on a 50/50 guess.
for MODE in off on_plus on_minus; do
  echo "=== GRAVCOMP=$MODE $(date -Is)"
  case "$MODE" in
    off)      export UWLAB_OSC_GRAVITY_COMP=0 UWLAB_OSC_GRAVITY_SIGN=1 ;;
    on_plus)  export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=1 ;;
    on_minus) export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-1 ;;
  esac
  UWLAB_DIAG_GC="$MODE" timeout 1800 "$PY" - <<'PYEOF' 2>&1 | grep -aE "DIAG|gravity compensation|Error|Traceback" | tail -30
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
STEPS = 160
TIPS = ("rl_dg_1_tip", "rl_dg_2_tip")
MODE = os.environ.get("UWLAB_DIAG_GC", "?")
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

env.reset()
pos0 = _t(obj.data.root_pos_w).clone()
tips0 = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1).clone()
d0 = (pos0 - tips0).norm(dim=-1)
zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)

for step in range(STEPS):
    env.step(zero)
    # NOTE: no step-159 sample. episode_length_s = 16.0 at 0.1 s/step means the episode boundary
    # falls exactly at step 160 and auto-resets the envs BEFORE that read, which is what produced the
    # bogus "recovery" rows in the earlier probes (in_hand rebounding to 0.820, sag to -0.022 m).
    # 120 is the last sample that still describes the episode it belongs to.
    if step in (0, 4, 19, 39, 79, 119):
        pos = _t(obj.data.root_pos_w)
        tips = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1)
        d = (pos - tips).norm(dim=-1)
        in_hand = float(((d - d0).abs() < 0.02).float().mean())
        sag = float((tips0[:, 2] - tips[:, 2]).mean())
        print(
            f"DIAG gc={MODE} step={step + 1:4d} in_hand={in_hand:.3f} "
            f"hand_sag_m={sag:+.4f} cube_to_tip={float(d.mean()):.4f}"
        )
env.close()
app.close()
PYEOF
done
echo "=== DIAG_GRAVCOMP_DONE $(date -Is)"
