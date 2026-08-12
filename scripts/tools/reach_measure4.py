"""Fourth pass: WHERE IS THE TABLE, and where does base_link's yaw-pi come from.

Answers the two open questions left by reach_measure3:
  1. is the object spawn / the commanded goal over the work surface at all?
  2. why is ``root_quat_w`` yaw-pi when ``init_state.rot`` is identity?

``--flip`` spawns the robot prim at yaw-pi so the composed base_link lands at identity, and
re-measures every distance, which is the verification of the proposed fix rather than an argument
for it.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="DexLift-UR5eDelto-RelJointPos-Lift-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--flip", action="store_true")
parser.add_argument("--out", type=str, default="/root/reach_measure4.json")
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
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402


def stats(t):
    t = t.float().flatten()
    return {"min": float(t.min()), "median": float(t.median()), "max": float(t.max()), "mean": float(t.mean())}


def box(p):
    return {
        "x": [float(p[:, 0].min()), float(p[:, 0].max())],
        "y": [float(p[:, 1].min()), float(p[:, 1].max())],
        "z": [float(p[:, 2].min()), float(p[:, 2].max())],
    }


def world_bbox(stage, path, purpose):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [purpose], useExtentsHint=False)
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    lo, hi = rng.GetMin(), rng.GetMax()
    return {"min": [float(v) for v in lo], "max": [float(v) for v in hi]}


def xform_of(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat()
    im = q.GetImaginary()
    return {
        "pos": [float(v) for v in t],
        "quat_wxyz": [float(q.GetReal()), float(im[0]), float(im[1]), float(im[2])],
        "local_ops": {op.GetOpName(): str(op.Get()) for op in UsdGeom.Xformable(prim).GetOrderedXformOps()},
    }


def main():
    OUT = {"flip": args_cli.flip}
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    OUT["cfg_robot_init_rot_before"] = list(env_cfg.scene.robot.init_state.rot)
    if args_cli.flip:
        env_cfg.scene.robot.init_state = env_cfg.scene.robot.init_state.replace(rot=(0.0, 0.0, 0.0, 1.0))
    OUT["cfg_robot_init_rot_used"] = list(env_cfg.scene.robot.init_state.rot)
    OUT["cfg_object_init_pos"] = list(env_cfg.scene.object.init_state.pos)
    OUT["cfg_table_init"] = {
        "pos": list(env_cfg.scene.table.init_state.pos),
        "rot": list(env_cfg.scene.table.init_state.rot),
    }
    OUT["cfg_cmd_ranges"] = {
        "pos_x": list(env_cfg.commands.object_pose.ranges.pos_x),
        "pos_y": list(env_cfg.commands.object_pose.ranges.pos_y),
        "pos_z": list(env_cfg.commands.object_pose.ranges.pos_z),
    }
    OUT["cfg_oob"] = {k: list(v) for k, v in env_cfg.terminations.object_out_of_bound.params["in_bound_range"].items()}

    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    uenv.reset()
    act = torch.zeros((uenv.num_envs, uenv.action_manager.total_action_dim), device=uenv.device)
    uenv.step(act)

    stage = uenv.sim.stage
    # -- USD side: where the prims actually are, env_0 only (env origin is 0,0,0 for env_0)
    OUT["usd"] = {}
    for path in (
        "/World/envs/env_0",
        "/World/envs/env_0/Robot",
        "/World/envs/env_0/Table",
        "/World/envs/env_0/Object",
    ):
        OUT["usd"][path] = xform_of(stage, path)
    robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    # find every prim named base_link / base anywhere under the robot
    for p in Usd.PrimRange(robot_prim):
        if p.GetName() in ("base_link", "base", "base_link_inertia", "shoulder_link"):
            OUT["usd"][str(p.GetPath())] = xform_of(stage, str(p.GetPath()))
    OUT["table_world_bbox_default"] = world_bbox(stage, "/World/envs/env_0/Table", "default")
    OUT["table_frame_bbox_default"] = world_bbox(
        stage, "/World/envs/env_0/Table/custom_lab_table/visuals/table_frame", "default"
    )

    # -- physics side
    origins = uenv.scene.env_origins
    robot = uenv.scene["robot"]
    obj = uenv.scene["object"]
    table = uenv.scene["table"]
    palm_id = robot.find_bodies("rl_dg_mount")[0][0]
    tip_ids, tip_names = robot.find_bodies(".*tip.*")

    r, p, y = euler_xyz_from_quat(robot.data.root_quat_w[:1])
    OUT["robot_root_body_name"] = robot.body_names[0]
    OUT["robot_root_quat_w"] = [float(v) for v in robot.data.root_quat_w[0]]
    OUT["robot_root_euler_xyz"] = [float(r[0]), float(p[0]), float(y[0])]
    OUT["robot_root_pos_env"] = [float(v) for v in (robot.data.root_pos_w[0] - origins[0])]
    OUT["robot_default_root_state_rot"] = [float(v) for v in robot.data.default_root_state[0, 3:7]]
    OUT["table_root_pos_env"] = [float(v) for v in (table.data.root_pos_w[0] - origins[0])]
    OUT["table_root_quat_w"] = [float(v) for v in table.data.root_quat_w[0]]

    term = uenv.command_manager.get_term("object_pose")
    goal_e = term.pose_command_w[:, :3] - origins
    palm_e = robot.data.body_pos_w[:, palm_id, :] - origins
    tips_e = robot.data.body_pos_w[:, tip_ids, :] - origins[:, None, :]
    obj_e = obj.data.root_pos_w - origins

    OUT["tip_names"] = tip_names
    OUT["palm_box_env"] = box(palm_e)
    OUT["tips_box_env"] = box(tips_e.reshape(-1, 3))
    OUT["object_box_env"] = box(obj_e)
    OUT["goal_box_env"] = box(goal_e)
    OUT["dist_object_to_nearest_fingertip"] = stats(
        torch.linalg.norm(tips_e - obj_e[:, None, :], dim=-1).min(dim=1).values
    )
    OUT["dist_object_to_palm"] = stats(torch.linalg.norm(obj_e - palm_e, dim=-1))
    OUT["dist_goal_to_palm"] = stats(torch.linalg.norm(goal_e - palm_e, dim=-1))
    OUT["dist_goal_to_object"] = stats(torch.linalg.norm(goal_e - obj_e, dim=-1))
    OUT["dist_object_to_base"] = stats(torch.linalg.norm(obj_e - (robot.data.root_pos_w - origins), dim=-1))
    OUT["dist_goal_to_base"] = stats(torch.linalg.norm(goal_e - (robot.data.root_pos_w - origins), dim=-1))

    # is the object over the table footprint?
    bb = OUT["table_frame_bbox_default"]
    if bb is not None:
        inside = (
            (obj_e[:, 0] >= bb["min"][0])
            & (obj_e[:, 0] <= bb["max"][0])
            & (obj_e[:, 1] >= bb["min"][1])
            & (obj_e[:, 1] <= bb["max"][1])
        )
        OUT["frac_object_over_table_footprint"] = float(inside.float().mean())
        gin = (
            (goal_e[:, 0] >= bb["min"][0])
            & (goal_e[:, 0] <= bb["max"][0])
            & (goal_e[:, 1] >= bb["min"][1])
            & (goal_e[:, 1] <= bb["max"][1])
        )
        OUT["frac_goal_over_table_footprint"] = float(gin.float().mean())
        tin = (
            (tips_e[..., 0] >= bb["min"][0])
            & (tips_e[..., 0] <= bb["max"][0])
            & (tips_e[..., 1] >= bb["min"][1])
            & (tips_e[..., 1] <= bb["max"][1])
        )
        OUT["frac_fingertips_over_table_footprint"] = float(tin.float().mean())

    with open(args_cli.out, "w") as f:
        json.dump(OUT, f, indent=2)
    print("WROTE", args_cli.out)
    print(json.dumps(OUT, indent=2))


main()
print("SCRIPT_COMPLETE")
