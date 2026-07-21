# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnostic: measure which world direction the WRIST CAMERA faces at settle-like (near-goal)
poses, to check/fix the "camera settles toward old +X instead of +X" bug.

The wrist camera is rigidly mounted on robotiq_base_link with the calibrated link-relative
rotation `CAM_ROT` (from ur10e_linear_gripper_rgb_cfg `_UR10E_CAMERA_POSES['wrist_camera']`), and
uses the OpenGL convention (optical axis = camera-local -Z). So:

    cam_axis_world = R(base_link_world_quat) @ R(CAM_ROT) @ (0,0,-1)

We drive the ObjectPartiallyAssembledEEGrasped reset (object near the goal, gripper grasping) and
report the horizontal heading of cam_axis_world: atan2(y, x) in degrees. 0deg = +X (toward the
operator, desired); +-90 / 180 = pointing sideways / away.

    ./uwlab.sh -p scripts_v2/tools/conversions/measure_camera_facing.py \
      --task OmniReset-UR10eLinearGripper-ObjectPartiallyAssembledEEGrasped-v0 --num_envs 32 --headless \
      env.scene.insertive_object=realpcb env.scene.receptive_object=realopenbox \
      env.events.reset_insertive_object_pose_from_partial_assembly_dataset.params.dataset_dir=./Datasets_realpcb/OmniReset \
      env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets_realpcb/OmniReset <TRIMS>
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--settle_steps", type=int, default=30)
parser.add_argument("--yaw_shift_deg", type=float, default=0.0,
                    help="Hypothetical extra yaw (deg) applied about world +Z to the base quat, to "
                         "preview how a reset-yaw re-center would move the camera heading.")
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from typing import cast

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_mul, quat_from_angle_axis

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose

# robotiq_base_link -> wrist camera, calibrated (opengl). wxyz.
CAM_ROT = (0.0018127, -0.0175937, 0.91723, -0.3979652)


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = None
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    robot = env.scene["robot"]
    base_idx = robot.data.body_names.index("robotiq_base_link")

    cam_rot = torch.tensor(CAM_ROT, device=env.device).expand(env.num_envs, 4)
    optical_local = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    yaw_shift = quat_from_angle_axis(
        torch.full((env.num_envs,), math.radians(args_cli.yaw_shift_deg), device=env.device),
        torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3),
    )

    actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
    actions[:, -1] = -1.0
    env.reset()
    for _ in range(args_cli.settle_steps):
        env.step(actions)

    base_quat = robot.data.body_link_quat_w[:, base_idx, :]                 # world<-base_link
    base_quat = quat_mul(yaw_shift, base_quat)                              # optional preview yaw
    cam_quat = quat_mul(base_quat, cam_rot)                                 # world<-camera
    axis_w = quat_apply(cam_quat, optical_local)                           # camera optical axis (world)

    heading = torch.atan2(axis_w[:, 1], axis_w[:, 0]) * 180.0 / math.pi     # deg from +X in XY plane
    pitch = torch.asin(axis_w[:, 2].clamp(-1, 1)) * 180.0 / math.pi         # +up / -down

    # Circular stats for heading.
    c, s = torch.cos(heading * math.pi / 180), torch.sin(heading * math.pi / 180)
    mean_heading = math.degrees(math.atan2(float(s.mean()), float(c.mean())))
    R = float(torch.sqrt(c.mean() ** 2 + s.mean() ** 2))  # concentration (1 = tight)

    print(f"\n[camface] yaw_shift={args_cli.yaw_shift_deg:.0f}deg  n={env.num_envs}")
    print(f"[camface] camera heading from +X (0=+X/toward operator, 180=away, +-90=side):")
    print(f"[camface]   mean={mean_heading:+.1f}deg  concentration R={R:.2f}  "
          f"(pitch mean={float(pitch.mean()):+.0f}deg, -=looking down)")
    # Histogram over 8 sectors.
    import collections
    sect = collections.Counter(((torch.round(heading / 45.0).long() * 45) % 360).tolist())
    print("[camface]   heading histogram (deg:count): "
          + "  ".join(f"{int(k):>4}:{v}" for k, v in sorted(sect.items())))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
