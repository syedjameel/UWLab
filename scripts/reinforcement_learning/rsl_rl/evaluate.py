# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministically evaluate an RSL-RL checkpoint against a named success termination."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--episodes", type=int, default=1000)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--success_term", default="success")
parser.add_argument("--output", type=Path)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.checkpoint is None:
    parser.error("--checkpoint is required")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config


def _wilson(successes: int, episodes: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (p + z * z / (2.0 * episodes)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / episodes + z * z / (4.0 * episodes**2)) / denominator
    return center - radius, center + radius


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    if args_cli.episodes <= 0 or args_cli.num_envs <= 0:
        raise ValueError("--episodes and --num_envs must be positive")

    agent_cfg = cli_args.sanitize_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli))
    env_cfg.scene.num_envs = min(args_cli.num_envs, args_cli.episodes)
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    checkpoint = Path(retrieve_file_path(args_cli.checkpoint)).resolve()
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy

    completed = 0
    successes = 0
    obs, _ = env.reset()
    while completed < args_cli.episodes:
        batch_size = min(env.num_envs, args_cli.episodes - completed)
        active = torch.arange(env.num_envs, device=env.unwrapped.device) < batch_size
        finished = torch.zeros(env.num_envs, device=env.unwrapped.device, dtype=torch.bool)
        while not bool(finished[active].all()):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                done = dones.bool()
                newly_finished = done & active & ~finished
                success = env.unwrapped.termination_manager.get_term(args_cli.success_term)
                successes += int(success[newly_finished].sum().item())
                finished |= newly_finished
                policy_nn.reset(dones)
        completed += batch_size
        if completed < args_cli.episodes:
            obs, _ = env.reset()
            policy_nn.reset(torch.ones(env.num_envs, device=env.unwrapped.device, dtype=torch.bool))

    low, high = _wilson(successes, completed)
    result = {
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "seed": args_cli.seed,
        "episodes": completed,
        "parallel_envs": env.num_envs,
        "protocol": "one_terminal_outcome_per_environment_slot",
        "successes": successes,
        "success_rate": successes / completed,
        "wilson_95": [low, high],
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args_cli.output is not None:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(encoded + "\n", encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
