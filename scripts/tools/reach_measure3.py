"""Third pass: the commanded goal, measured after the command manager has populated it."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="DexLift-UR5eDelto-RelJointPos-Lift-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--out", type=str, default="/root/reach_measure3.json")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import json  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def stats(t):
    t = t.float().flatten()
    return {"min": float(t.min()), "median": float(t.median()), "max": float(t.max()), "mean": float(t.mean())}


def box(p):
    return {
        "x": [float(p[:, 0].min()), float(p[:, 0].max())],
        "y": [float(p[:, 1].min()), float(p[:, 1].max())],
        "z": [float(p[:, 2].min()), float(p[:, 2].max())],
    }


def main():
    OUT = {}
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    uenv.reset()
    act = torch.zeros((uenv.num_envs, uenv.action_manager.total_action_dim), device=uenv.device)
    uenv.step(act)

    origins = uenv.scene.env_origins
    robot = uenv.scene["robot"]
    obj = uenv.scene["object"]
    palm_id = robot.find_bodies("rl_dg_mount")[0][0]
    tip_ids, _ = robot.find_bodies(".*tip.*")

    term = uenv.command_manager.get_term("object_pose")
    goal_e = term.pose_command_w[:, :3] - origins
    palm_e = robot.data.body_pos_w[:, palm_id, :] - origins
    tips_e = robot.data.body_pos_w[:, tip_ids, :] - origins[:, None, :]
    obj_e = obj.data.root_pos_w - origins

    OUT["goal_box_env_frame"] = box(goal_e)
    OUT["goal_mean_env_frame"] = [float(v) for v in goal_e.mean(0)]
    OUT["dist_goal_to_palm"] = stats(torch.linalg.norm(goal_e - palm_e, dim=-1))
    OUT["dist_goal_to_nearest_fingertip"] = stats(
        torch.linalg.norm(tips_e - goal_e[:, None, :], dim=-1).min(dim=1).values
    )
    OUT["dist_goal_to_object"] = stats(torch.linalg.norm(goal_e - obj_e, dim=-1))
    OUT["dist_goal_to_base"] = stats(torch.linalg.norm(goal_e, dim=-1))
    OUT["palm_box_after_1_step"] = box(palm_e)
    OUT["object_box_after_1_step"] = box(obj_e)
    OUT["dist_object_to_nearest_fingertip_after_1_step"] = stats(
        torch.linalg.norm(tips_e - obj_e[:, None, :], dim=-1).min(dim=1).values
    )

    with open(args_cli.out, "w") as f:
        json.dump(OUT, f, indent=2)
    print("WROTE", args_cli.out)
    print(json.dumps(OUT, indent=2))


main()
print("SCRIPT_COMPLETE")
