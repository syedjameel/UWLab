#!/usr/bin/env bash
# DOES THE TRAINED POLICY HOLD THE CUBE, OR DOES ONLY THE ZERO ACTION FAIL?
#
# Every diagnostic so far drove a ZERO action, and the bias sweep showed the dominant failure is not
# the hand: the fingertips SAG ~91 mm over 60 steps at every bias (0.05 / 0.08 / 0.095 gave
# -0.091 / -0.092 / -0.097 m), with the cube ending ~0.26 m from the tips -- the hand descends and
# leaves the cube behind. But a zero action is a regime TRAINING NEVER ENTERS: the policy commands
# arm deltas every step. Stage A reached C4 0.476, its best ever, which a 91 mm sag would not permit.
#
# So: replay the ACTUAL POLICY over a FULL 160-step episode (the horizon at which success is
# sampled), with the zero action as the control, and report in_hand / hand sag / distance to goal.
#
# TWO FAILURES FROM THE FIRST ATTEMPT, BOTH FIXED HERE:
#   1. OnPolicyRunner(agent_cfg) died with "PPO.__init__() got an unexpected keyword argument
#      'optimizer'" -- the registered cfg is written for a different PPO signature than the installed
#      rsl_rl fork. Fixed by not constructing a runner at all: the actor MLP is rebuilt directly from
#      the checkpoint's model_state_dict, which is immune to cfg-signature drift.
#   2. The first failure cascaded into "Simulation context already exists" for every later case,
#      because a failed gym.make leaves the sim context alive. Fixed by creating ONE env and
#      reusing it -- the reset family is swapped on the live event term between conditions.
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
CK=${CK:-$(ls -t "$WS"/UWLab/logs/rsl_rl/ur10e_delto_omnireset_agent/*_pinch_grip_B/model_*.pt 2>/dev/null | head -1)}
echo "=== DIAG_POLICY_START $(date -Is) ck=$CK"
[ -n "$CK" ] || { echo "DIAG no checkpoint found"; exit 1; }

CK="$CK" timeout 2400 "$PY" - <<'PYEOF' 2>&1 | grep -aE "DIAG|Error|Traceback" | tail -80
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
CK = os.environ["CK"]
_t = lambda x: getattr(x, "torch", x)  # noqa: E731


def build_actor(device):
    """Rebuild the actor MLP straight from the checkpoint, bypassing the runner/cfg entirely.

    rsl_rl stores the ActorCritic weights under 'model_state_dict' as actor.0/2/4..., i.e. a plain
    nn.Sequential of Linear+activation. Reading the shapes off the state dict avoids having to agree
    with the registered agent cfg about hidden sizes, activation or PPO kwargs.
    """
    import torch.nn as nn

    sd = torch.load(CK, map_location=device, weights_only=False)
    model = sd.get("model_state_dict", sd)
    keys = sorted(
        (k for k in model if k.startswith("actor.") and k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1]),
    )
    if not keys:
        raise RuntimeError(f"no actor.* weights in checkpoint; keys={list(model)[:8]}")
    layers, prev_idx = [], None
    for k in keys:
        w = model[k]
        if prev_idx is not None:
            layers.append(nn.ELU())
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(model[k.replace(".weight", ".bias")])
        layers.append(lin)
        prev_idx = k
    actor = nn.Sequential(*layers).to(device).eval()
    print(f"DIAG actor_rebuilt layers={[tuple(model[k].shape) for k in keys]}")
    return actor


def obs_vector(env, obs_dict):
    """The policy group as a flat tensor, whatever the manager hands back."""
    if isinstance(obs_dict, dict):
        o = obs_dict.get("policy", next(iter(obs_dict.values())))
    else:
        o = obs_dict
    return _t(o)


cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
ev_params = None
env = gym.make(TASK, cfg=cfg).unwrapped
robot = env.scene["robot"]
obj = env.scene["insertive_object"]
names = list(robot.body_names)
tip_ids = [names.index(t) for t in TIPS if t in names]
ev_params = env.event_manager.get_term_cfg("reset_from_reset_states").params

actor = None
try:
    actor = build_actor(env.device)
except Exception as exc:
    print(f"DIAG actor_build_FAILED {type(exc).__name__}: {exc}")


def run(family, use_policy):
    tag = "POLICY" if use_policy else "ZERO"
    if use_policy and actor is None:
        print(f"DIAG {tag} family={family} SKIPPED no_actor")
        return
    ev_params["reset_types"] = [family]
    ev_params["probs"] = [1.0]
    obs_dict, _ = env.reset()
    pos0 = _t(obj.data.root_pos_w).clone()
    tips0 = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1).clone()
    d0 = (pos0 - tips0).norm(dim=-1)
    zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)

    for step in range(STEPS):
        if use_policy:
            with torch.inference_mode():
                o = obs_vector(env, obs_dict)
                act = actor(o)
            obs_dict = env.step(act)[0]
        else:
            obs_dict = env.step(zero)[0]
        if step in (0, 4, 19, 39, 79, 119, 159):
            pos = _t(obj.data.root_pos_w)
            tips = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1)
            d = (pos - tips).norm(dim=-1)
            in_hand = float(((d - d0).abs() < 0.02).float().mean())
            sag = float((tips0[:, 2] - tips[:, 2]).mean())
            print(
                f"DIAG {tag} family={family} step={step + 1:4d} in_hand={in_hand:.3f} "
                f"hand_sag_m={sag:+.4f} cube_to_tip={float(d.mean()):.4f}"
            )


for family in ("ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped"):
    for use_policy in (True, False):
        try:
            run(family, use_policy)
        except Exception as exc:
            print(f"DIAG FAILED family={family} policy={use_policy} {type(exc).__name__}: {exc}")

env.close()
app.close()
PYEOF
echo "=== DIAG_POLICY_DONE $(date -Is)"
