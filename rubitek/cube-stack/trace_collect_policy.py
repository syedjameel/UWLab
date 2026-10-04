"""Record a POLICY rollout through the repo's own inference path, not a hand-rebuilt actor.

WHY A SECOND COLLECTOR. `trace_collect.py` rebuilds the actor from `model_state_dict` and steps the
raw env. That path produced a rollout in which the policy destroyed its own grasp by step 5 and left
the cube frozen 36 mm from the goal, 0/64 within tolerance -- while training reports ~0.27 success on
the very same family. Adding the observation normaliser (`actor_obs_normalizer._mean/._std`, which
lives inside model_state_dict) was a genuine fix but changed nothing material, so something else the
training wrapper does is still missing: `RslRlVecEnvWrapper` applies `clip_actions`, and the UW-Lab
fork may squash actions. Guessing at each of those in turn is slower and less reliable than using the
path the repo itself uses to play checkpoints.

So this mirrors `render_rollout.py`: sanitize the agent cfg (without it the fork's PPO dies on
`unexpected keyword argument 'optimizer'`), wrap the env, let OnPolicyRunner load the checkpoint, and
take `get_inference_policy()`. Normalisation, clipping and squashing are then whatever training used,
by construction.

Writes the same JSON schema as `trace_collect.py` so `trace_animate_policy.py` consumes it unchanged.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, "/home/mattephi/Work/Rubitek/UWLab-cubestack/scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa: E402

TASK = "OmniReset-UR10eDelto-CubeStackTwoFingerGripCarry-RelCartesianOSC-State-v0"
N = int(os.environ.get("TRACE_ENVS", "64"))
STEPS = int(os.environ.get("TRACE_STEPS", "160"))
NAME = os.environ["TRACE_NAME"]
CK = os.environ["TRACE_CK"]
FAMILY = os.environ.get("TRACE_FAMILY", "ObjectPartiallyAssembledEEGrasped")
OUT = os.environ.get("TRACE_OUT", f"/home/mattephi/Work/Rubitek/cube-stack/results/trace_{NAME}.json")
TIPS = ("rl_dg_1_tip", "rl_dg_2_tip")
_t = lambda x: getattr(x, "torch", x)  # noqa: E731

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
ev = cfg.events.reset_from_reset_states
ev.params["reset_types"] = [FAMILY]
ev.params["probs"] = [1.0]

agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)

env_raw = gym.make(TASK, cfg=cfg)
u = env_raw.unwrapped
env = RslRlVecEnvWrapper(env_raw, clip_actions=getattr(agent_cfg, "clip_actions", None))

# TRACE_CK=none rolls out the NULL ACTION through this identical wrapper and reward path, so a
# null-action episode and a policy episode can be compared on the one number that drives learning:
# accumulated reward. The policy drops the cube within ~5 steps on C3 while a null action holds it in
# ~72% of environments at step 150 -- if the dropping behaviour nonetheless earns MORE reward, the
# objective is provably misaligned and "add more iterations" cannot fix it.
if CK.lower() in ("none", "null", ""):
    _z = torch.zeros((N, u.action_manager.total_action_dim), device=u.device)

    def policy(_obs):  # noqa: D401 -- same call signature as the inference policy
        return _z

    print("TRACE NULL ACTION (no checkpoint); same wrapper and reward path as the policy rollout")
else:
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(CK)
    policy = runner.get_inference_policy(device=u.device)
    print(f"TRACE runner policy loaded from {CK}; clip_actions={getattr(agent_cfg, 'clip_actions', None)}")

# THE AUTHORITATIVE SUCCESS SIGNAL -- the same one training reports as task_N_success_rate.
#
# Hand-rolled proxies scored this policy at 0/64 and looked like total failure. Both were wrong:
#   * "cube to receptive-object centre" is ~34 mm from the real target, because commands.py:244
#     measures the insertive cube IN THE RECEPTIVE ASSET'S FRAME, and a correctly seated cube sits
#     one cube-height above that centre. A perfect stack scores ~34 mm on the naive measure.
#   * "in_hand" counts a release as a dropped grasp -- but releasing the cube onto the stack is the
#     whole task on C4. Measured: the cube's median displacement from its own start is 0.0 mm.
ctx = u.reward_manager.get_term_cfg("progress_context").func
cmd = u.command_manager.get_term("task_command")
print(f"TRACE thresholds pos={cmd.success_position_threshold} rot={cmd.success_orientation_threshold}")

robot = u.scene["robot"]
obj = u.scene["insertive_object"]
goal_asset = u.scene["receptive_object"]
names = list(robot.body_names)
tip_ids = [names.index(t) for t in TIPS if t in names]
origins = _t(u.scene.env_origins)[:, :3]

# RESET FIRST. `get_observations()` alone does NOT start a fresh episode: the environments keep
# whatever episode phase they were left in, so they hit their individual time-outs partway through the
# recording window. Measured consequence, 64 envs over 150 steps: 35 of them showed a single-step cube
# jump above 50 mm (up to 326 mm -- impossible under gravity in one 0.1 s step), i.e. a mid-window
# reset. That silently corrupts everything downstream -- a reset puts the cube back in the hand, so an
# environment that dropped it reads as one that held it, and per-step reward keeps accumulating across
# the reset while reward_manager._episode_sums is zeroed at it. It produced a confident, published,
# WRONG result ("dropping pays 2x holding"). trace_collect.py always reset and was never affected.
_r = env.reset()
obs = _r[0] if isinstance(_r, tuple) else _r
if isinstance(obs, tuple):
    obs = obs[0]
pos0 = _t(obj.data.root_pos_w).clone()
tips0 = _t(robot.data.body_pos_w)[:, tip_ids].mean(dim=1).clone()
d0 = (pos0 - tips0).norm(dim=-1)

rec: dict = {"step": [], "in_hand": [], "cube_z": [], "cube_to_tip": [], "cube_to_goal": [],
             "cube_xyz": [], "tip1_xyz": [], "tip2_xyz": [], "goal_xyz": []}
with torch.inference_mode():
    for step in range(STEPS):
        actions = policy(obs)
        out = env.step(actions)
        obs = out[0]
        if isinstance(obs, tuple):
            obs = obs[0]
        # Reward is the quantity the policy actually optimises; everything else on this page is a
        # diagnostic. Recorded per step so a holding rollout and a dropping rollout can be compared
        # on it directly.
        try:
            r = _t(out[1]).float().flatten()
            rec.setdefault("reward", []).append(float(r.mean()))
            # PER-ENV reward too. The null-vs-policy comparison is confounded -- the null arm holds
            # but transports nothing, so it forgoes the transport shaping the policy collects. Per-env
            # reward inside a SINGLE policy rollout removes that: it asks whether, among environments
            # running the same behaviour, the ones that kept the grasp were paid more than the ones
            # that dropped it. That is the question the recommendation actually rests on.
            rec.setdefault("reward_env", []).append([round(float(v), 5) for v in r])
        except Exception:  # noqa: BLE001 -- a diagnostic must never kill the rollout
            pass
        # PER-TERM episode sums at step 150, per environment. Knowing that the objective pays more
        # for dropping than holding is actionable only once the offending TERM is named; this turns
        # "fix the objective" into "this term pays for dropping". Captured at 149 because the
        # time-out at step 160 resets the accumulators.
        if step == 149:
            try:
                rec["term_sums"] = {
                    k: [round(float(x), 5) for x in _t(v)]
                    for k, v in u.reward_manager._episode_sums.items()
                }
            except Exception as exc:  # noqa: BLE001
                print(f"TRACE term sums unavailable ({exc.__class__.__name__}: {exc})", flush=True)
        pos = _t(obj.data.root_pos_w)
        tips = _t(robot.data.body_pos_w)[:, tip_ids]
        mid = tips.mean(dim=1)
        d = (pos - mid).norm(dim=-1)
        goal = _t(goal_asset.data.root_pos_w)
        rec["step"].append(step + 1)
        rec["in_hand"].append(float(((d - d0).abs() < 0.02).float().mean()))
        rec["cube_z"].append(float((pos[:, 2] - pos0[:, 2]).mean()))
        rec["cube_to_tip"].append(float(d.mean()))
        rec["cube_to_goal"].append(float((pos[:, :3] - goal[:, :3]).norm(dim=-1).mean()))
        rec["cube_xyz"].append([[round(v, 4) for v in p] for p in (pos[:, :3] - origins).cpu().tolist()])
        rec["tip1_xyz"].append([[round(v, 4) for v in p] for p in (tips[:, 0, :3] - origins).cpu().tolist()])
        rec["tip2_xyz"].append([[round(v, 4) for v in p] for p in (tips[:, 1, :3] - origins).cpu().tolist()])
        rec["goal_xyz"].append([[round(v, 4) for v in p] for p in (goal[:, :3] - origins).cpu().tolist()])
        # The env's own numbers, not a proxy.
        xyz = _t(cmd.xyz_distance)
        rec.setdefault("xyz_distance", []).append(float(xyz.mean()))
        rec.setdefault("xyz_dist_env", []).append([round(float(v), 5) for v in xyz])
        rec.setdefault("success", []).append(float(_t(ctx.success).float().mean()))
        rec.setdefault("success_env", []).append([bool(v) for v in _t(ctx.success).tolist()])

rec["meta"] = {"name": NAME, "effort_scale": os.environ.get("UWLAB_HAND_EFFORT_SCALE", "1"),
               "tuck": os.environ.get("UWLAB_TUCK_OTHERS", "0"),
               "grip_bias": os.environ.get("UWLAB_GRIP_BIAS_RAD", "0.05"),
               "envs": N, "steps": STEPS, "reset_family": FAMILY, "checkpoint": CK,
               "loader": "OnPolicyRunner.get_inference_policy",
               "success_position_threshold": float(cmd.success_position_threshold),
               "success_orientation_threshold": float(cmd.success_orientation_threshold)}
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(rec, f)
# Report at 150, not 160: the time-out resets every env on the final step and the cube reappears in
# the hand, which reads as a recovery that never happened.
i150 = min(149, len(rec["in_hand"]) - 1)
print(f"TRACE wrote {OUT}: in_hand@{rec['step'][i150]}={rec['in_hand'][i150]:.3f} "
      f"cube_to_goal {rec['cube_to_goal'][0]:.4f} -> {rec['cube_to_goal'][i150]:.4f} m")
env.close()
app.close()
