# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""STEP 1 diagnostic for bead UWLab-dwx.6/dwx.7: is the 0/32 generator result a genuine policy
failure, or a sampling-timing mismatch against the episode-end moment?

The certified Stage-2 checkpoint (0.922 pass@30mm) is graded by a STICKY-OR minimum over the whole
episode -- it need only be correct at SOME point, not at termination. The generator's held-check
(dexlift/mdp/held_check.py) is wired as a terminating DoneTerm, which, given how RecorderManager's
"success" predicate has to be delivered (a term literally named "success", and ANY registered
termination term unconditionally ORs into the episode-ending signal -- termination_manager.py:182),
ALREADY ends the episode the instant all 4 gates align. That is "record when held becomes true"
(option a from the bead) -- it did not need to be newly built, it falls out of the plumbing.

WHAT COULD STILL BE WRONG, independent of that: the probe window (SETTLE_STEPS..SETTLE_STEPS+
PROBE_STEPS, i.e. steps 60-70 only) fires EXACTLY ONCE per episode, at a FIXED early point. If the
policy's grasp only stabilises LATER (e.g. after actively lifting, which plausibly takes longer than
70 steps), the probe measures the wrong moment and NEVER RE-FIRES -- held can then never latch that
episode even though the policy is doing fine by the time it matters, and the sticky-OR certified
number never sees that failure mode because it isn't gated on any single probe window at all.

This script does NOT register held_with_probe as a termination (so episodes run their natural
course to time_out/abnormal/object_out_of_bound, unmodified) and instead evaluates, every step,
the two gates that do NOT depend on the probe -- settled & opposed_contact & co_move, i.e.
"is this a moment the policy is plausibly holding it, ignoring the probe entirely" -- logging per
episode:
  * whether that ever became true, and at which step (relative to episode start)
  * whether it was true specifically at the step the episode naturally ended

If "ever" is materially above "at natural end", that confirms the sampling-timing diagnosis, and
separately, if the step distribution of first-true events clusters near or past step 70, that
confirms the probe-window-too-early sub-hypothesis specifically.

Run (one Isaac process; never via uwlab.sh):
    PYTHONPATH=... timeout -s KILL 400 <python> -u scripts_v2/tools/diagnose_held_timing.py \\
        --checkpoint <path>.pth --num_envs 64 --num_steps 800
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="DexLift-UR5eDelto-RelJointPos-TableLeg-Lift-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--agent_yaml", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=800)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import yaml  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from uwlab_tasks.manager_based.manipulation.dexlift.mdp.held_check_core import SETTLE_STEPS  # noqa: E402
from uwlab_tasks.manager_based.manipulation.dexlift.mdp.rewards import _sensor_force_magnitudes  # noqa: E402

THUMB_NAMES = ("rl_dg_1_tip", "rl_dg_5_tip")
TIP_NAMES = ("rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip")
FORCE_THRESH = 0.2
COMOVE_SPEED_THRESH = 0.05
COMOVE_VZ_THRESH = 0.1


def unwrap(o):
    if isinstance(o, tuple):
        o = o[0]
    if isinstance(o, dict):
        o = o["obs"]
    return o


def main() -> None:
    agent_yaml = args_cli.agent_yaml or os.path.join(
        os.path.dirname(os.path.dirname(args_cli.checkpoint)), "params", "agent.yaml"
    )
    with open(agent_yaml) as f:
        agent_cfg = yaml.safe_load(f)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.sim.physx.gpu_collision_stack_size = 2**24
    # Deliberately NOT touching env_cfg.terminations: episodes run their natural course
    # (time_out / abnormal_robot / object_out_of_bound only), so we can observe the FULL
    # trajectory instead of latching out early.
    env_cfg.seed = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    palm_id = env.scene["robot"].body_names.index("rl_dg_mount")

    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_act = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    wrapped_env = RlGamesVecEnvWrapper(env, args_cli.device, clip_obs, clip_act, obs_groups, concate_obs_groups)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: wrapped_env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    agent_cfg["params"]["config"]["num_actors"] = env.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(args_cli.checkpoint)
    player.reset()
    player.has_batch_dimension = True
    print("[diag] POLICY_LOADED", flush=True)

    obs = unwrap(wrapped_env.reset())
    n = env.num_envs
    device = env.device

    ever_true = torch.zeros(n, dtype=torch.bool, device=device)
    first_true_step = torch.full((n,), -1, dtype=torch.long, device=device)

    n_episodes = 0
    n_episodes_ever_true = 0
    n_episodes_true_at_end = 0
    first_true_steps_log: list[int] = []

    for _ in range(args_cli.num_steps):
        with torch.inference_mode():
            act = player.get_action(obs, is_deterministic=True)
            ret = wrapped_env.step(act)
            obs = unwrap(ret[0])
            dones = ret[2].flatten().to(torch.bool)

            robot = env.scene["robot"]
            obj = env.scene["object"]
            thumb_loaded = _sensor_force_magnitudes(env, THUMB_NAMES).gt(FORCE_THRESH).any(dim=-1)
            tip_loaded = _sensor_force_magnitudes(env, TIP_NAMES).gt(FORCE_THRESH).any(dim=-1)
            relative_speed = torch.linalg.vector_norm(
                obj.data.root_lin_vel_w - robot.data.body_lin_vel_w[:, palm_id, :], dim=-1
            )
            obj_vz = obj.data.root_lin_vel_w[:, 2]
            settled = env.episode_length_buf > SETTLE_STEPS
            grasp_moment = (
                settled
                & thumb_loaded
                & tip_loaded
                & (relative_speed < COMOVE_SPEED_THRESH)
                & (obj_vz.abs() < COMOVE_VZ_THRESH)
            )

            newly_true = grasp_moment & ~ever_true
            if newly_true.any():
                first_true_step[newly_true] = env.episode_length_buf[newly_true].long()
            ever_true |= grasp_moment

            if dones.any():
                done_idx = torch.nonzero(dones).flatten()
                n_episodes += done_idx.numel()
                for idx in done_idx.tolist():
                    if bool(ever_true[idx]):
                        n_episodes_ever_true += 1
                        first_true_steps_log.append(int(first_true_step[idx].item()))
                    if bool(grasp_moment[idx]):
                        n_episodes_true_at_end += 1
                # reset per-episode trackers for envs that just finished
                ever_true[done_idx] = False
                first_true_step[done_idx] = -1

    print("\n=== DIAGNOSTIC RESULT (probe-independent: settled & opposed_contact & co_move) ===", flush=True)
    print(f"episodes observed: {n_episodes}", flush=True)
    if n_episodes > 0:
        print(f"ever-true rate:     {n_episodes_ever_true}/{n_episodes} = {n_episodes_ever_true/n_episodes:.2%}", flush=True)
        print(f"at-natural-end rate:{n_episodes_true_at_end}/{n_episodes} = {n_episodes_true_at_end/n_episodes:.2%}", flush=True)
    if first_true_steps_log:
        steps_t = torch.tensor(first_true_steps_log, dtype=torch.float32)
        print(f"first-true step distribution over {len(first_true_steps_log)} episodes that were ever true:", flush=True)
        print(f"  min={steps_t.min():.0f} p25={steps_t.quantile(0.25):.0f} median={steps_t.median():.0f} "
              f"p75={steps_t.quantile(0.75):.0f} max={steps_t.max():.0f}", flush=True)
        frac_after_probe_window = float((steps_t > 70).float().mean())
        print(f"fraction of first-true events AFTER step 70 (the probe window's own end): {frac_after_probe_window:.2%}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
