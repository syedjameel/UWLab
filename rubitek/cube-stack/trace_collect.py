"""Record what the hand and cube actually DO, as numbers, so the result can be shown without RTX.

WHY THIS EXISTS. Photoreal rendering is unavailable on this machine tonight: Isaac Sim 6 places and
aims the camera correctly (verified: cam_pos [0.780,-0.280,0.160], look-at quaternion exact) and
still draws nothing but sky, while Isaac Sim 5.1 crashes outright in libomni.hydra.rtx. The common
factor is the 610 driver -- this stack wants the 580 branch -- and swapping a driver on someone's
workstation unattended is not a call to make at 01:00.

Physics is completely unaffected. So record the trajectories and animate them instead: a side view
of the two fingertips and the cube through a full 160-step episode is a faithful picture of the
result, provided it is labelled as a state trace and not passed off as a render.

One process per effort scale, because UWLAB_HAND_EFFORT_SCALE is read when the actuator config is
built at import time and cannot be changed within a run.
"""

from __future__ import annotations

import json
import os

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

try:
    import uwlab_is6_bridge

    uwlab_is6_bridge.apply()
except ImportError:
    pass

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "OmniReset-UR10eDelto-CubeStackTwoFingerGripCarry-RelCartesianOSC-State-v0"
N = int(os.environ.get("TRACE_ENVS", "64"))
STEPS = int(os.environ.get("TRACE_STEPS", "160"))
NAME = os.environ["TRACE_NAME"]
OUT = os.environ.get("TRACE_OUT", f"/home/mattephi/Work/Rubitek/cube-stack/results/trace_{NAME}.json")
TIPS = ("rl_dg_1_tip", "rl_dg_2_tip")
_t = lambda x: getattr(x, "torch", x)  # noqa: E731

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
# Configurable, because a policy must be rolled out on the family it was TRAINED on. Stage A trains
# C4 (ObjectPartiallyAssembledEEGrasped); replaying such a checkpoint on C3 would show it failing at
# a task it has never seen and would misrepresent the policy.
FAMILY = os.environ.get("TRACE_FAMILY", "ObjectAnywhereEEGrasped")
ev = cfg.events.reset_from_reset_states
ev.params["reset_types"] = [FAMILY]
ev.params["probs"] = [1.0]

env = gym.make(TASK, cfg=cfg).unwrapped
robot = env.scene["robot"]
obj = env.scene["insertive_object"]
names = list(robot.body_names)
tip_ids = [names.index(t) for t in TIPS if t in names]

CK = os.environ.get("TRACE_CK", "")
actor = None
obs_norm = None
if CK:
    # Rebuild the actor MLP straight from the state dict rather than going through OnPolicyRunner:
    # the UW-Lab rsl_rl fork's PPO rejects keys the registered config passes it ("PPO.__init__() got
    # an unexpected keyword argument 'optimizer'"), and acting needs only the actor.
    sd = torch.load(CK, map_location="cuda:0", weights_only=False)
    msd = sd.get("model_state_dict", sd)
    ws = [(k, v) for k, v in msd.items() if k.startswith("actor.") and k.endswith(".weight")]
    bs = {k.rsplit(".", 1)[0]: v for k, v in msd.items() if k.startswith("actor.") and k.endswith(".bias")}
    layers = []
    for i, (k, w) in enumerate(ws):
        lin = torch.nn.Linear(w.shape[1], w.shape[0]).to("cuda:0")
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(bs[k.rsplit(".", 1)[0]])
        if i:
            layers.append(torch.nn.ELU())
        layers.append(lin)
    actor = torch.nn.Sequential(*layers).eval()
    # Observation normalisation, if the run used it. Feeding raw observations to an actor trained on
    # normalised ones produces confident nonsense, which would look like a bad policy rather than a
    # loading bug.
    # The normaliser lives INSIDE model_state_dict as actor_obs_normalizer._mean/._std, not under a
    # top-level key. Probing for the wrong names returned "no normaliser" and fed raw observations to
    # an actor trained on normalised ones: the rollout destroyed the grasp in 5 steps and left the
    # cube frozen 37 mm from the goal, i.e. it looked like a broken policy when the policy was fine
    # (training reports ~0.27 success on this same family).
    if "actor_obs_normalizer._mean" in msd:
        mean = msd["actor_obs_normalizer._mean"].to("cuda:0").float()
        if "actor_obs_normalizer._std" in msd:
            std = msd["actor_obs_normalizer._std"].to("cuda:0").float()
        else:
            std = torch.sqrt(msd["actor_obs_normalizer._var"].to("cuda:0").float() + 1e-8)
        obs_norm = (mean, std.clamp_min(1e-6))
    in_dim = ws[0][1].shape[1]
    print(f"TRACE policy from {CK}: {len(ws)} layers, in_dim={in_dim}, "
          f"obs_norm={'yes' if obs_norm else 'NO'}")
    if obs_norm is None:
        # Loud, because silently skipping this is exactly what produced a misleading result once.
        print("TRACE WARNING: checkpoint has no actor_obs_normalizer -- if training normalised "
              "observations, this rollout is NOT trustworthy", flush=True)


def act_from(obs):
    """Policy action, or zeros when rolling out a reset."""
    if actor is None:
        return zero
    x = obs["policy"] if isinstance(obs, dict) else obs
    x = _t(x)
    if obs_norm is not None:
        mean, std = obs_norm  # _std is stored directly; no sqrt
        x = (x - mean) / std
    a = actor(x)
    return a[:, : zero.shape[1]]


obs, _ = env.reset()
pos0 = _t(obj.data.root_pos_w).clone()
tips0 = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1).clone()
d0 = (pos0 - tips0).norm(dim=-1)
zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)

# Record EVERY environment, not just env 0. Logging only env 0 produced an animation whose "after"
# panel showed the cube being dropped -- because env 0 happened to be one of the ~25% that fail at
# effort x20, while the aggregate says 80% hold. A single arbitrary environment is not evidence; the
# animator picks a representative one (median final cube-to-tip) from the full set instead.
#
# Positions are stored RELATIVE to each environment's origin. The raw world x sits near 5.8 m because
# of the environment grid offset, which made the plot axes meaningless and stopped the two conditions
# sharing a frame.
origins = _t(env.scene.env_origins)[:, :3]
rec: dict = {"step": [], "in_hand": [], "cube_z": [], "cube_to_tip": [],
             "cube_xyz": [], "tip1_xyz": [], "tip2_xyz": []}
with torch.inference_mode():
    for step in range(STEPS):
        obs, _, _, _, _ = env.step(act_from(obs))
        pos = _t(obj.data.root_pos_w)
        # Where the cube has to end up. Success is the insertive cube's centre within 5 mm of the
        # seated-stack pose, so distance-to-goal is the only number that says whether the POLICY is
        # doing the task, as opposed to merely not dropping the cube.
        goal = _t(env.scene["receptive_object"].data.root_pos_w)
        rec.setdefault("cube_to_goal", []).append(float((pos[:, :3] - goal[:, :3]).norm(dim=-1).mean()))
        rec.setdefault("goal_xyz", []).append(
            [[round(v, 4) for v in p] for p in (goal[:, :3] - origins).cpu().tolist()]
        )
        tips = _t(robot.data.body_pos_w)[:, tip_ids]
        mid = tips.mean(dim=1)
        d = (pos - mid).norm(dim=-1)
        rec["step"].append(step + 1)
        # Fraction of environments whose cube is still the same distance from the fingertips as at
        # reset, i.e. still held. This is the number the whole night turns on.
        rec["in_hand"].append(float(((d - d0).abs() < 0.02).float().mean()))
        rec["cube_z"].append(float((pos[:, 2] - pos0[:, 2]).mean()))
        rec["cube_to_tip"].append(float(d.mean()))
        rel_c = (pos[:, :3] - origins).cpu().tolist()
        rel_1 = (tips[:, 0, :3] - origins).cpu().tolist()
        rel_2 = (tips[:, 1, :3] - origins).cpu().tolist()
        rec["cube_xyz"].append([[round(v, 4) for v in p] for p in rel_c])
        rec["tip1_xyz"].append([[round(v, 4) for v in p] for p in rel_1])
        rec["tip2_xyz"].append([[round(v, 4) for v in p] for p in rel_2])

# Per-environment outcome at step 150 (NOT 160: the time-out resets every env on the final step and
# the cube reappears in the hand). This is what lets the animator choose a representative env.
_pos = _t(obj.data.root_pos_w)
_tips = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1)
_d = (_pos - _tips).norm(dim=-1)
rec["final_cube_to_tip"] = [float(v) for v in _d]
rec["held_at_end"] = [bool(v) for v in ((_d - d0).abs() < 0.02).tolist()]

rec["meta"] = {
    "name": NAME,
    "effort_scale": os.environ.get("UWLAB_HAND_EFFORT_SCALE", "1"),
    "tuck": os.environ.get("UWLAB_TUCK_OTHERS", "0"),
    "grip_bias": os.environ.get("UWLAB_GRIP_BIAS_RAD", "0.05"),
    "envs": N,
    "steps": STEPS,
    "reset_family": FAMILY,
    "checkpoint": CK or None,
}
with open(OUT, "w") as f:
    json.dump(rec, f)
# Report step 150, NOT the last step. The episode is 160 control steps: the time-out fires on the
# final one and resets every env before the sample, so in_hand rebounds to ~0.9 and the summary reads
# as a success. That artifact already made this line claim "1.000 -> 0.906" for a run whose grasp was
# measured gone (0.047) ten steps earlier.
_i150 = min(149, len(rec["in_hand"]) - 1)
print(f"TRACE wrote {OUT}: in_hand {rec['in_hand'][0]:.3f} -> {rec['in_hand'][_i150]:.3f} at step "
      f"{rec['step'][_i150]} (last step {rec['step'][-1]} reads {rec['in_hand'][-1]:.3f}, "
      "which is the post-time-out reset, not a recovery)")
env.close()
app.close()
