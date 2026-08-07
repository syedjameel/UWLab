# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roll out a checkpoint on ONE reset type and bin success by the episode's STARTING conditions.

A per-type success number says a reset type is lagging; it cannot say WHICH states fail. This
records, per episode, the board's initial height, tilt and distance to the goal, then reports the
success rate within each bin -- turning "task_2 plateaus at 0.89" into "the states that fail are
the ones with <property>", or showing the failures are spread evenly and the type is simply hard.

    ./uwlab.sh -p scripts_v2/tools/conversions/eval_success_by_start.py \
      --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-Play-v0 \
      --checkpoint logs/.../model_3300.pt --num_envs 64 --episodes 8 --headless \
      --reset_type ObjectAnywhereEEGrasped \
      env.scene.insertive_object=realpcb env.scene.receptive_object=jigenclosure \
      env.events.reset_from_reset_states.params.dataset_dir=./Datasets_realpcb_jig/OmniReset
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--reset_type", type=str, default="ObjectAnywhereEEGrasped")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from typing import cast

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose

sys.path.insert(0, os.path.join(os.getcwd(), "scripts/reinforcement_learning/rsl_rl"))
import cli_args  # noqa: E402


def _report(name, vals, ok, edges):
    print(f"\n  success by {name}:", flush=True)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (vals >= lo) & (vals < hi)
        if m.sum() == 0:
            continue
        print(f"    [{lo:7.1f}, {hi:7.1f}) : {ok[m].mean():5.3f}   n={int(m.sum()):5d}", flush=True)


@hydra_task_compose(args_cli.task, "rsl_rl_cfg_entry_point", hydra_args=hydra_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    env_cfg.events.reset_from_reset_states.params["reset_types"] = [args_cli.reset_type]
    env_cfg.events.reset_from_reset_states.params["probs"] = [1.0]

    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    wrapped = RslRlVecEnvWrapper(env)

    ckpt = retrieve_file_path(args_cli.checkpoint)
    sanitized = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    runner = OnPolicyRunner(wrapped, sanitized.to_dict(), log_dir=None, device=env.device)
    sd = torch.load(ckpt, map_location=env.device)["model_state_dict"]
    actor_sd = {k: v for k, v in sd.items() if k.startswith(("actor", "std", "actor_obs_normalizer"))}
    runner.alg.policy.load_state_dict(actor_sd, strict=False)
    live = dict(runner.alg.policy.named_parameters()); live.update(dict(runner.alg.policy.named_buffers()))
    checked = sum(1 for k, v in actor_sd.items()
                  if k in live and live[k].shape == v.shape
                  and torch.allclose(live[k].float(), v.float().to(live[k].device)))
    assert checked >= 4, f"only {checked} actor tensors verified"
    print(f"[eval] actor VERIFIED ({checked}/{len(actor_sd)}); reset_type={args_cli.reset_type}", flush=True)
    runner.alg.policy.eval()
    policy = runner.get_inference_policy(device=env.device)

    success_fn = env.reward_manager.get_term_cfg("progress_context").func
    obj = env.scene["insertive_object"]
    rec = env.scene["receptive_object"]
    ep_len = int(env.max_episode_length)

    z0 = torch.zeros(env.num_envs, device=env.device)
    tilt0 = torch.zeros(env.num_envs, device=env.device)
    dist0 = torch.zeros(env.num_envs, device=env.device)
    fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    Z, T, D, OK = [], [], [], []

    obs = wrapped.get_observations()
    for _ in range(args_cli.episodes * ep_len):
        if fresh.any():   # latch the starting condition of any env that just reset
            q = obj.data.root_quat_w
            tl = torch.rad2deg(torch.arccos(torch.clamp(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)))
            dd = torch.linalg.norm(obj.data.root_pos_w[:, :2] - rec.data.root_pos_w[:, :2], dim=1)
            z0 = torch.where(fresh, obj.data.root_pos_w[:, 2], z0)
            tilt0 = torch.where(fresh, tl, tilt0)
            dist0 = torch.where(fresh, dd, dist0)
            fresh[:] = False
        with torch.inference_mode():
            obs, _, dones, _ = wrapped.step(policy(obs))
        d = dones.nonzero(as_tuple=False).flatten()
        if d.numel():
            Z.append(z0[d].cpu().numpy() * 1000)
            T.append(tilt0[d].cpu().numpy())
            D.append(dist0[d].cpu().numpy() * 1000)
            OK.append(success_fn.success[d].float().cpu().numpy())
            fresh[d] = True

    Z, T, D, OK = map(np.concatenate, (Z, T, D, OK))
    print(f"\n=== {args_cli.reset_type}: {len(OK)} episodes, overall success {OK.mean():.3f}", flush=True)
    _report("board start HEIGHT (mm)", Z, OK, [0, 30, 50, 80, 120, 180, 250, 1e9])
    _report("board start TILT (deg)", T, OK, [0, 2, 5, 10, 20, 45, 1e9])
    _report("start DISTANCE to goal (mm)", D, OK, [0, 100, 200, 300, 400, 1e9])
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    import gymnasium as gym
    main()
