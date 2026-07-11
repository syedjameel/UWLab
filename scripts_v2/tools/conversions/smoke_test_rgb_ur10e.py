# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Smoke-test the UR10e RGB / camera-align envs: build one, step it, confirm the 3 cameras
render, and save a frame per camera. Catches config-wiring errors; the render itself needs a
GPU that fits 3 TiledCameras (A100/4090 -- the 6 GB laptop may OOM on the camera buffers,
which is a capacity note, not a config error).

    ./uwlab.sh -p scripts_v2/tools/conversions/smoke_test_rgb_ur10e.py \
        --task OmniReset-UR10eLinearGripper-CameraAlign-v0 --num_envs 1
    ./uwlab.sh -p scripts_v2/tools/conversions/smoke_test_rgb_ur10e.py \
        --task OmniReset-UR10eLinearGripper-RelCartesianOSC-RGB-DataCollection-v0 --num_envs 2
"""
from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--out", type=str, default="/home/syed/.claude/jobs/68e1e7df/tmp")
parser.add_argument("--no_trims", action="store_true", help="Skip the laptop PhysX buffer trims.")
parser.add_argument(
    "--no_dr",
    action="store_true",
    help="Null the appearance/HDRI randomization terms (they cloud-download ~957 texture "
    "assets on first run). Use to validate reset+camera wiring fast without the download.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True  # required for TiledCamera rendering
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import sys
import traceback

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import uwlab_tasks  # noqa: F401

os.makedirs(args.out, exist_ok=True)
IS_RGB = "RGB" in args.task


def save_frame(name, arr):
    """Save an HxWxC uint8/float image to PNG via matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = arr.detach().cpu().numpy() if torch.is_tensor(arr) else np.asarray(arr)
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8) if a.max() <= 1.0 else a.clip(0, 255).astype(np.uint8)
    fig = plt.figure(figsize=(4, 3))
    plt.imshow(a)
    plt.axis("off")
    fig.savefig(os.path.join(args.out, name), bbox_inches="tight", dpi=80)
    plt.close(fig)


fails = []
try:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)

    # Laptop PhysX buffer trims (6 GB GPU) -- the RGB cfgs request 2 GB collision stacks.
    if not args.no_trims:
        cfg.sim.physx.gpu_collision_stack_size = 67108864
        cfg.sim.physx.gpu_max_rigid_contact_count = 2097152
        cfg.sim.physx.gpu_max_rigid_patch_count = 2097152
        cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2097152
        cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2097152

    # RGB collection: pcb/openbox + UR10e reset datasets (camera-align has no such fields).
    if IS_RGB:
        cfg.scene.insertive_object = cfg.variants["scene.insertive_object"]["pcb"]
        cfg.scene.receptive_object = cfg.variants["scene.receptive_object"]["openbox"]

    # Optionally null the appearance/HDRI DR (cloud-downloads ~957 textures on first run).
    if args.no_dr and getattr(cfg, "events", None) is not None:
        for name in list(vars(cfg.events).keys()):
            if any(t in name for t in ("appearance", "sky_light", "hdri")):
                setattr(cfg.events, name, None)

    env = gym.make(args.task, cfg=cfg).unwrapped
    print(f"[SMOKE] built {args.task}: {env.num_envs} env(s)")
    obs, _ = env.reset()
    print("[SMOKE] reset OK")

    # Find camera obs across obs groups.
    cam_keys = {}
    for group, gobs in obs.items():
        if isinstance(gobs, dict):
            for k, v in gobs.items():
                if "rgb" in k.lower():
                    cam_keys[f"{group}/{k}"] = v
    if not cam_keys:
        fails.append("no *rgb* obs terms found after reset")
    else:
        for k, v in cam_keys.items():
            print(f"[SMOKE] camera obs {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    a = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
    for _ in range(args.steps):
        obs = env.step(a)[0]
    print(f"[SMOKE] stepped {args.steps}x OK")

    # Save one frame per camera (from env 0) to eyeball the render.
    saved = 0
    for group, gobs in obs.items():
        if isinstance(gobs, dict):
            for k, v in gobs.items():
                if "rgb" in k.lower() and torch.is_tensor(v) and v.ndim >= 3:
                    img = v[0]
                    if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW -> HWC
                        img = img.permute(1, 2, 0)
                    tag = args.task.split("-")[1]
                    save_frame(f"rgb_{tag}_{group}_{k}.png".replace("/", "_"), img)
                    saved += 1
    print(f"[SMOKE] saved {saved} camera frame(s) to {args.out}")

    if fails:
        print("[SMOKE_RESULT] [FAIL] " + " | ".join(fails))
    else:
        print(f"[SMOKE_RESULT] [PASS] {args.task}: built + stepped + {len(cam_keys)} cameras render")
except Exception as e:  # noqa: BLE001
    print(f"[SMOKE_RESULT] [FAIL] {type(e).__name__}: {e}")
    traceback.print_exc()

sys.stdout.flush()
os._exit(0)
