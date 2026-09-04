# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Render the jig-removal-v2 table layout from the three real camera poses (RTX GPU required).

Answers the only question that matters for the plate's position: can the cameras SEE it? The
FOV can be computed, but the computation is easy to get wrong (a first pass here truncated the
front camera's footprint by 180 mm because rays missing the mat plane were dropped), so this
puts the actual rendered frames on disk instead.

Uses the CameraAlign env: it inherits RlStateSceneCfg -- so it carries the pedestal variant and
the enclosure_pcb prop -- and has NO randomisation, so the layout is exactly what is asked for.
Poses are written straight to the sim after reset; no reset datasets are needed.

    ./uwlab.sh -p scripts_v2/tools/conversions/snapshot_layout.py --enable_cameras --headless \
      --pedestal-x 0.80 --enclosure-x 0.50 --out /tmp/layout \
      env.scene.insertive_object=jigv2c env.scene.receptive_object=pedestal <TRIMS>
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="OmniReset-UR10eLinearGripper-CameraAlign-v0")
parser.add_argument("--pedestal-x", type=float, default=0.80)
parser.add_argument("--pedestal-y", type=float, default=0.0)
parser.add_argument("--enclosure-x", type=float, default=0.50)
parser.add_argument("--enclosure-y", type=float, default=0.0)
parser.add_argument("--enclosure-yaw", type=float, default=0.0, help="degrees")
parser.add_argument("--settle", type=int, default=12)
parser.add_argument("--jig-on-plate", action="store_true")
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose

MAT_Z = 0.004


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=hydra_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = 1
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    dev = env.device

    def place(name, x, y, z, yaw_deg=0.0):
        asset = env.scene[name]
        pose = asset.data.default_root_state.clone()[:, :7]
        pose[:, 0] = x + env.scene.env_origins[:, 0]
        pose[:, 1] = y + env.scene.env_origins[:, 1]
        pose[:, 2] = z + env.scene.env_origins[:, 2]
        h = math.radians(yaw_deg) / 2.0
        pose[:, 3] = math.cos(h)
        pose[:, 4] = 0.0
        pose[:, 5] = 0.0
        pose[:, 6] = math.sin(h)
        asset.write_root_pose_to_sim(pose)
        asset.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=dev))

    # Pedestal: 10 mm plate resting on the mat -> root at mat + half-thickness.
    place("receptive_object", args_cli.pedestal_x, args_cli.pedestal_y, MAT_Z + 0.005)
    # Enclosure+PCB stack: 36.07 mm tall, bottom at -18.037 from its root.
    place("enclosure_pcb", args_cli.enclosure_x, args_cli.enclosure_y,
          MAT_Z + 0.018037, args_cli.enclosure_yaw)
    # Jig seated on it (rel z 11.563, derived in make_seated_partial_assembly.py).
    place("insertive_object", args_cli.enclosure_x, args_cli.enclosure_y,
          MAT_Z + 0.018037 + 0.011563, args_cli.enclosure_yaw)

    if args_cli.jig_on_plate:
        # jig bottom exactly on the plate top: plate root + 5 mm + jig half-height 12 mm
        place("insertive_object", args_cli.pedestal_x, args_cli.pedestal_y, MAT_Z + 0.005 + 0.005 + 0.012)
    jig = env.scene["insertive_object"]; ped = env.scene["receptive_object"]
    for i in range(args_cli.settle):
        env.sim.step(render=True)
        env.scene.update(env.physics_dt)
        if args_cli.jig_on_plate:
            gap = ((jig.data.root_pos_w[0, 2] - 0.012) - (ped.data.root_pos_w[0, 2] + 0.005)) * 1000
            print(f"[settle {i:2d}] jig root z {jig.data.root_pos_w[0,2]:.5f}  "
                  f"bottom vs plate top {gap:+7.2f} mm", flush=True)

    os.makedirs(args_cli.out, exist_ok=True)
    tag = f"ped{args_cli.pedestal_x:.2f}_enc{args_cli.enclosure_x:.2f}"
    frames = {}
    for cam in ("front_camera", "side_camera", "wrist_camera"):
        img = env.scene[cam].data.output["rgb"][0, ..., :3].cpu().numpy()
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        frames[cam] = img
        plt.imsave(os.path.join(args_cli.out, f"{tag}_{cam}.png"), img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (cam, img) in zip(axes, frames.items()):
        ax.imshow(img)
        ax.set_title(cam)
        ax.axis("off")
    fig.suptitle(f"pedestal x={args_cli.pedestal_x:.2f}  enclosure x={args_cli.enclosure_x:.2f} "
                 f"yaw={args_cli.enclosure_yaw:.0f} deg")
    fig.tight_layout()
    fig.savefig(os.path.join(args_cli.out, f"{tag}_ALL.png"), dpi=100)
    print(f"[snapshot] wrote {args_cli.out}/{tag}_*.png")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
