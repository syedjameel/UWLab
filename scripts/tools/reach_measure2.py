"""Follow-up: gravity actually in force at reset, whether the object falls, and the palm
position at the EXACT default (unrandomized) arm posture."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="DexLift-UR5eDelto-RelJointPos-Lift-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--out", type=str, default="/root/reach_measure2.json")
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

OUT: dict = {}


def stats(t):
    t = t.float().flatten()
    return {
        "min": float(t.min()),
        "median": float(t.median()),
        "max": float(t.max()),
        "mean": float(t.mean()),
    }


def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    OUT["cfg_object_init_pos"] = list(env_cfg.scene.object.init_state.pos)
    OUT["cfg_reset_object_pose_range"] = {
        k: list(v) for k, v in env_cfg.events.reset_object.params["pose_range"].items()
    }
    OUT["cfg_reset_robot_joints_range"] = list(env_cfg.events.reset_robot_joints.params["position_range"])
    OUT["cfg_reset_robot_wrist_range"] = list(env_cfg.events.reset_robot_wrist_joint.params["position_range"])
    OUT["cfg_variable_gravity_params"] = str(env_cfg.events.variable_gravity.params["gravity_distribution_params"])
    OUT["cfg_gravity_adr_present"] = env_cfg.curriculum.gravity_adr is not None
    OUT["cfg_command_ranges"] = {
        "pos_x": list(env_cfg.commands.object_pose.ranges.pos_x),
        "pos_y": list(env_cfg.commands.object_pose.ranges.pos_y),
        "pos_z": list(env_cfg.commands.object_pose.ranges.pos_z),
    }
    OUT["cfg_sim_gravity"] = list(env_cfg.sim.gravity)

    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    uenv.reset()

    scene = uenv.scene
    robot = scene["robot"]
    obj = scene["object"]
    origins = scene.env_origins

    pc = uenv.sim.get_physics_context()
    try:
        OUT["gravity_at_reset"] = str(pc.get_gravity())
    except Exception as e:  # noqa: BLE001
        OUT["gravity_at_reset"] = f"UNREADABLE: {e}"

    palm_id = robot.find_bodies("rl_dg_mount")[0][0]
    tip_ids, tip_names = robot.find_bodies(".*tip.*")

    z0 = (obj.data.root_pos_w - origins)[:, 2].clone()
    OUT["object_z_at_reset"] = stats(z0)

    # -- let physics run 2 s of sim with a zero action; does the object fall?
    act = torch.zeros((uenv.num_envs, uenv.action_manager.total_action_dim), device=uenv.device)
    for _ in range(120):  # 120 policy steps @60 Hz = 2.0 s
        uenv.step(act)
    z1 = (obj.data.root_pos_w - origins)[:, 2]
    OUT["object_z_after_2s_zero_action"] = stats(z1)
    OUT["object_z_drop_2s"] = stats(z0 - z1)

    # -- palm at the EXACT default posture, no reset randomization
    q = robot.data.default_joint_pos.clone()
    v = torch.zeros_like(q)
    robot.write_joint_state_to_sim(q, v)
    robot.set_joint_position_target(q)
    robot.write_data_to_sim()
    uenv.sim.step(render=False)
    robot.update(uenv.sim.get_physics_dt())
    bp = robot.data.body_pos_w - origins[:, None, :]
    OUT["default_posture"] = {
        "arm_joint_pos_rad": {n: float(q[0, i]) for i, n in enumerate(robot.joint_names[:6])},
        "palm_env_frame": [float(x) for x in bp[0, palm_id]],
        "fingertips_env_frame": {n: [float(x) for x in bp[0, i]] for i, n in zip(tip_ids, tip_names)},
        "palm_radius_from_base": float(torch.linalg.norm(bp[0, palm_id])),
    }
    # mirrored: shoulder_pan + pi
    q2 = q.clone()
    q2[:, 0] += 3.14159265
    robot.write_joint_state_to_sim(q2, v)
    robot.set_joint_position_target(q2)
    robot.write_data_to_sim()
    uenv.sim.step(render=False)
    robot.update(uenv.sim.get_physics_dt())
    bp2 = robot.data.body_pos_w - origins[:, None, :]
    OUT["default_posture_pan_plus_pi"] = {
        "palm_env_frame": [float(x) for x in bp2[0, palm_id]],
        "fingertips_env_frame": {n: [float(x) for x in bp2[0, i]] for i, n in zip(tip_ids, tip_names)},
    }

    with open(args_cli.out, "w") as f:
        json.dump(OUT, f, indent=2)
    print("WROTE", args_cli.out)
    print(json.dumps(OUT, indent=2))


main()
print("SCRIPT_COMPLETE")
