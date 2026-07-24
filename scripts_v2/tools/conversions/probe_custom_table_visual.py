# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Visual smoke of the CUSTOM LAB TABLE: spawn snapshots + object-drop datum probe.

Builds the UR10e CameraAlign env (3 TiledCameras, custom table, robot, objects; no DR, no
dataset resets), repoints the cameras at three wide vantages of the table, and saves
user-reviewable PNGs to ./table_swap_snaps/:

  01_spawn_<view>.png   -- robot on the new table (overview / front / top view)
  02_objectdrop_<view>.png -- pcb + openbox resting on the mats after a drop
plus prints each object's settled root z (expected: bottom on the +0.004 work surface).

    ./uwlab.sh -p scripts_v2/tools/conversions/probe_custom_table_visual.py
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default="./table_swap_snaps")
parser.add_argument("--no_trims", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math
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
TASK = "OmniReset-UR10eLinearGripper-CameraAlign-v0"


def look_at_quat_wxyz(eye, target, up=(0.0, 0.0, 1.0)):
    """Quaternion (wxyz) for an OpenGL-convention camera at `eye` looking at `target`."""
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    # camera axes: x=right, y=up, z=-forward (OpenGL)
    R = np.stack([r, u, -f], axis=1)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
        w, x, y, z = q
    return (w, x, y, z)


def save_frame(path, arr):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = arr.detach().cpu().numpy() if torch.is_tensor(arr) else np.asarray(arr)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8) if a.max() <= 1.0 else a.clip(0, 255).astype(np.uint8)
    fig = plt.figure(figsize=(6, 4.5))
    plt.imshow(a)
    plt.axis("off")
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def grab_cams(obs, prefix):
    n = 0
    for group, gobs in obs.items():
        if isinstance(gobs, dict):
            for k, v in gobs.items():
                if "rgb" in k.lower() and torch.is_tensor(v) and v.ndim >= 3:
                    name = k.replace("_rgb", "").replace("rgb_", "")
                    save_frame(os.path.join(args.out, f"{prefix}_{name}.png"), v[0])
                    n += 1
    return n


try:
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    if not args.no_trims:
        cfg.sim.physx.gpu_collision_stack_size = 67108864
        cfg.sim.physx.gpu_max_rigid_contact_count = 2097152
        cfg.sim.physx.gpu_max_rigid_patch_count = 2097152
        cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2097152
        cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2097152

    # pcb/openbox if the cfg exposes variants (falls back to the defaults otherwise)
    variants = getattr(cfg, "variants", None)
    if variants and "scene.insertive_object" in variants:
        cfg.scene.insertive_object = variants["scene.insertive_object"]["pcb"]
        cfg.scene.receptive_object = variants["scene.receptive_object"]["openbox"]

    # Null the curtains for clean snapshots (they sit at the AUTHORS' positions -- to be
    # moved to the real backdrop placement at collection time; they block the wide views).
    for cname in ("curtain_left", "curtain_back", "curtain_right"):
        if getattr(cfg.scene, cname, None) is not None:
            setattr(cfg.scene, cname, None)

    # Repoint the three cameras at wide vantages of the table (snapshot-only override;
    # table spans x -0.35..1.05, y +-0.35 around the robot at (0, -0.039, 0)).
    center = (0.45, -0.039, 0.05)
    views = {
        "front_camera": ((1.85, -0.9, 0.85), center),   # 3/4 view from the front-left
        "side_camera": ((0.45, 1.35, 0.75), center),    # side-on across the table
        "wrist_camera": None,                            # leave (on-gripper)
    }
    for cam_name, v in views.items():
        cam = getattr(cfg.scene, cam_name, None)
        if cam is None or v is None:
            continue
        eye, tgt = v
        cam.offset.pos = eye
        cam.offset.rot = look_at_quat_wxyz(eye, tgt)
    # the wrist camera sees its own gripper; the two wide views carry the snapshot value

    env = gym.make(TASK, cfg=cfg).unwrapped
    print(f"[PROBE] built {TASK}")
    obs, _ = env.reset()

    ins = env.scene["insertive_object"]
    rec = env.scene["receptive_object"]

    def place(asset, xy, z):
        pose = asset.data.default_root_state[:, :7].clone()
        pose[:, 0] = xy[0]
        pose[:, 1] = xy[1]
        pose[:, 2] = z
        pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
        asset.write_root_pose_to_sim(pose)
        asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))

    a = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)

    # --- 01: spawn views (objects parked on the green mat) ---
    place(rec, (0.55, 0.05), 0.06)
    place(ins, (0.42, -0.15), 0.05)
    for _ in range(90):
        env.step(a)
    obs = env.step(a)[0]
    n = grab_cams(obs, "01_spawn")
    print(f"[PROBE] saved {n} spawn view(s)")

    # --- 02: drop probe across the mats (incl. the black/green boundary + y-trim edge).
    # pcb bottom_offset z = -0.01 -> expected rest root z = 0.004 + 0.010 = 0.014.
    # Low drop (2 cm) so the thin board can't bounce onto an edge. The openbox is
    # KINEMATIC (a fixture, like the real command-stripped box) -> placed, must NOT move.
    drops = [
        ("pcb", ins, (0.32, -0.08), 0.03),   # black mat, near x-min of the object range
        ("pcb", ins, (0.45, 0.28), 0.03),    # green mat, near the TRIMMED y edge (0.30)
        ("pcb", ins, (0.55, 0.0), 0.10),     # higher drop for comparison
    ]
    results = []
    for name, asset, xy, z0 in drops:
        place(asset, xy, z0)
        for _ in range(150):
            env.step(a)
        z = float(asset.data.root_pos_w[0, 2] - env.scene.env_origins[0, 2])
        v = float(asset.data.root_vel_w[0, :3].norm())
        q = asset.data.root_quat_w[0].tolist()
        results.append((name, xy, z, v))
        print(f"[PROBE] drop {name} @ {xy} from z={z0}: rest root z={z:+.4f} |v|={v:.4f} "
              f"quat=({q[0]:+.2f},{q[1]:+.2f},{q[2]:+.2f},{q[3]:+.2f})")
    # kinematic openbox: place ON the surface, verify it stays put
    place(rec, (0.50, 0.10), 0.011)  # openbox bottom_offset -0.007 -> root at 0.004+0.007
    for _ in range(60):
        env.step(a)
    zb = float(rec.data.root_pos_w[0, 2] - env.scene.env_origins[0, 2])
    print(f"[PROBE] kinematic openbox placed at 0.011: z={zb:+.4f} (must stay ~0.011)")
    obs = env.step(a)[0]
    n = grab_cams(obs, "02_objectdrop")
    print(f"[PROBE] saved {n} drop view(s)")

    # verdict: objects SETTLE on the mats (not floating high / not sunk through) and the
    # kinematic box stays put. Exact datum arithmetic is human-reviewed from the printed
    # numbers (default peg: standing rest root 0.034 = bottom EXACTLY at the +0.004
    # surface [half-length 0.030]; tipped 0.019 = +0.004 + radius 0.015) and re-verified
    # statistically by the reset-state QC after recording.
    ok = all(0.004 < z <= 0.06 and v < 0.05 for _, _, z, v in results) and abs(zb - 0.011) < 0.003
    print(f"[PROBE_RESULT] [{'PASS' if ok else 'FAIL'}] snapshots in {os.path.abspath(args.out)}")
except Exception as e:  # noqa: BLE001
    print(f"[PROBE_RESULT] [FAIL] {type(e).__name__}: {e}")
    traceback.print_exc()

sys.stdout.flush()
os._exit(0)
