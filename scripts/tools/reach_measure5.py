"""Fifth pass: test the two candidate repairs of the yaw-pi root, and hold them for 60 steps.

reach_measure4 localised the defect: ``/World/envs/env_0/Robot/base_link`` carries an AUTHORED
``xformOp:orient`` of yaw-pi, the articulation root body IS ``base_link``, and nothing writes the
root pose because this env nulls dexsuite's ``reset_root``. ``default_root_state`` therefore says
identity while the root really sits at yaw-pi.

  --mode none  as shipped
  --mode pin   restore dexsuite's zero-range ``reset_root``, which writes the root pose to
               ``default_root_state`` every reset. Question: does the arm's fixed ``root_joint``
               fight that write?
  --mode flip  spawn the /Robot prim at yaw-pi so the composed base_link lands at identity.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="DexLift-UR5eDelto-RelJointPos-Lift-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--mode", type=str, default="none", choices=["none", "pin", "flip", "unpin"])
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--out", type=str, default="/root/reach_measure5.json")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import json  # noqa: E402
import torch  # noqa: E402

import isaaclab.envs.mdp as base_mdp  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab.managers import EventTermCfg, SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import uwlab_tasks  # noqa: F401,E402


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
    OUT = {"mode": args_cli.mode}
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    if args_cli.mode == "flip":
        env_cfg.scene.robot.init_state = env_cfg.scene.robot.init_state.replace(rot=(0.0, 0.0, 0.0, 1.0))
    elif args_cli.mode == "pin":
        env_cfg.events.reset_root = EventTermCfg(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot")},
        )

    elif args_cli.mode == "unpin":
        # NEGATIVE TEST for the two guards added with the fix. Reproduce the defect by nulling the
        # root pin, and require BOTH guards to notice.
        from uwlab_tasks.manager_based.manipulation.dexlift.dexlift_ur5e_delto_env_cfg import (
            _assert_root_pin_is_a_pin,
        )

        env_cfg.events.reset_root = None
        try:
            _assert_root_pin_is_a_pin(env_cfg)
            OUT["construction_guard"] = "DID NOT FIRE"
        except ValueError as exc:
            OUT["construction_guard"] = f"fired: {exc}"
        try:
            env = gym.make(args_cli.task, cfg=env_cfg)
            env.unwrapped.reset()
            OUT["runtime_guard"] = "DID NOT FIRE"
        except RuntimeError as exc:
            OUT["runtime_guard"] = f"fired: {exc}"
        with open(args_cli.out, "w") as f:
            json.dump(OUT, f, indent=2)
        print(json.dumps(OUT, indent=2))
        return

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

    def snapshot(tag):
        r, p, y = euler_xyz_from_quat(robot.data.root_quat_w)
        term = uenv.command_manager.get_term("object_pose")
        goal_e = term.pose_command_w[:, :3] - origins
        palm_e = robot.data.body_pos_w[:, palm_id, :] - origins
        tips_e = robot.data.body_pos_w[:, tip_ids, :] - origins[:, None, :]
        obj_e = obj.data.root_pos_w - origins
        base_e = robot.data.root_pos_w - origins

        # table footprint, from the world bbox measured in reach_measure4: x [-0.35, 1.05], |y| < 0.35
        def over_table(p):
            hit = (p[..., 0] > -0.35) & (p[..., 0] < 1.05) & (p[..., 1].abs() < 0.35)
            return float(hit.float().mean())

        OUT[tag] = {
            "root_yaw_rad": stats(torch.atan2(torch.sin(y), torch.cos(y))),
            "root_pos_env": box(base_e),
            "palm_box_env": box(palm_e),
            "tips_box_env": box(tips_e.reshape(-1, 3)),
            "object_box_env": box(obj_e),
            "goal_box_env": box(goal_e),
            "dist_object_to_nearest_fingertip": stats(
                torch.linalg.norm(tips_e - obj_e[:, None, :], dim=-1).min(dim=1).values
            ),
            "dist_object_to_palm": stats(torch.linalg.norm(obj_e - palm_e, dim=-1)),
            "dist_goal_to_palm": stats(torch.linalg.norm(goal_e - palm_e, dim=-1)),
            "dist_object_to_base": stats(torch.linalg.norm(obj_e - base_e, dim=-1)),
            "dist_goal_to_base": stats(torch.linalg.norm(goal_e - base_e, dim=-1)),
            "max_abs_joint_vel": float(robot.data.joint_vel.abs().max()),
            "frac_object_over_table": over_table(obj_e),
            "frac_goal_over_table": over_table(goal_e),
            "frac_tips_over_table": over_table(tips_e),
        }

    snapshot("after_1_step")
    for _ in range(args_cli.steps):
        uenv.step(act)
    snapshot(f"after_{args_cli.steps + 1}_steps")

    with open(args_cli.out, "w") as f:
        json.dump(OUT, f, indent=2)
    print("WROTE", args_cli.out)
    print(json.dumps(OUT, indent=2))


main()
print("SCRIPT_COMPLETE")
