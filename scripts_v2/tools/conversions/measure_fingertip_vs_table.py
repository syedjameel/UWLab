# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnostic: measure the gripper jaw/fingertip world-Z vs the table top after an
ObjectRestingEEGrasped reset, to decide whether recorded thin-object grasps bury the
fingertips in the table (unrealizable on the real robot) or rest them at the surface.

Not a recorder -- reuses the reset env, applies the recorded grasp+resting states, lets
physics settle, then prints per-env: min finger-body Z, table-top Z, and the gap.

    ./uwlab.sh -p scripts_v2/tools/conversions/measure_fingertip_vs_table.py \
      --task OmniReset-UR10eLinearGripper-ObjectRestingEEGrasped-v0 --num_envs 16 --headless \
      env.scene.insertive_object=realpcb env.scene.receptive_object=openbox \
      env.events.reset_insertive_object_pose_from_reset_states.params.dataset_dir=./Datasets_realpcb_smoke \
      env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets_realpcb_smoke <TRIMS>
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--settle_steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from typing import cast

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = None
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    robot = env.scene["robot"]

    # Identify finger/jaw bodies by name.
    body_names = robot.data.body_names
    finger_ids = [i for i, n in enumerate(body_names) if "finger" in n.lower()]
    print(f"[measure] all bodies: {body_names}")
    print(f"[measure] finger bodies: {[body_names[i] for i in finger_ids]}")

    env.reset()
    actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
    actions[:, -1] = -1.0  # keep gripper closed (grasped)
    for _ in range(args_cli.settle_steps):
        env.step(actions)

    # True jaw TIP: robotiq_base_link pose + metadata tip offset (0.144 m) along the gripper's
    # local +Z approach axis (fingers reach +Z). This is the actual lowest gripping point, unlike
    # the inner_finger link origins (which sit well above the tip).
    from isaaclab.utils.math import quat_apply

    base_idx = body_names.index("robotiq_base_link")
    body_pos_w = robot.data.body_link_pos_w  # (num_envs, num_bodies, 3)
    base_pos = body_pos_w[:, base_idx, :]  # (num_envs, 3)
    base_quat = robot.data.body_link_quat_w[:, base_idx, :]  # (num_envs, 4) wxyz
    TIP_LOCAL = torch.tensor([0.0, 0.0, 0.144], device=env.device).expand(env.num_envs, 3)
    tip_world = base_pos + quat_apply(base_quat, TIP_LOCAL)  # (num_envs, 3)
    min_finger_z = tip_world[:, 2]  # jaw tip world Z
    print(f"[measure] robotiq_base_link mean Z: {base_pos[:,2].mean().item():.5f}")

    # Table top: table root Z + its half-height is unknown here; use the insertive object's
    # bottom as the table-surface proxy (the PCB rests ON the table, so PCB_bottom == table top).
    pcb = env.scene["insertive_object"]
    pcb_z = pcb.data.root_pos_w[:, 2]
    pcb_bottom = pcb_z - 0.0015  # realpcb half-thickness = 1.5 mm
    table = env.scene["table"]
    table_z = table.data.root_pos_w[:, 2]

    print("\n[measure] per-env (meters):")
    print(f"{'env':>3} {'min_finger_z':>13} {'pcb_z':>9} {'pcb_bottom(=table top)':>22} {'finger-below-surface':>21}")
    for e in range(env.num_envs):
        below = pcb_bottom[e].item() - min_finger_z[e].item()  # >0 means finger is BELOW the surface
        print(f"{e:>3} {min_finger_z[e].item():>13.5f} {pcb_z[e].item():>9.5f} "
              f"{pcb_bottom[e].item():>22.5f} {below*1000:>18.2f} mm")

    below_all = (pcb_bottom - min_finger_z)  # meters, >0 = buried
    print(f"\n[measure] table root Z (all ~equal): {table_z.mean().item():.5f}")
    print(f"[measure] fingers BELOW surface: {(below_all > 0).sum().item()}/{env.num_envs} envs")
    print(f"[measure] burial depth mm -> median={below_all.median().item()*1000:.2f} "
          f"max={below_all.max().item()*1000:.2f} min={below_all.min().item()*1000:.2f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
