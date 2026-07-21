# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experiment: for a thin insertive object (realpcb) grasped with the D1 object-aware standoff,
sweep the ObjectRestingEEGrasped reset augmentation (`pose_range_b` on the grasp-EE event) and
measure how often the jaw tip is driven below the table.

This quantifies (a) Option A's reject rate under each augmentation (the tip-below-table fraction)
and (b) which reduced augmentation keeps burial ~0 while retaining diversity -> sets Option B.

`pose_range_b` is cached as `self.ranges` in the event's __init__, so we mutate that tensor
between resets and sweep every config in ONE sim session.

    ./uwlab.sh -p scripts_v2/tools/conversions/sweep_thin_reset_augmentation.py \
      --task OmniReset-UR10eLinearGripper-ObjectRestingEEGrasped-v0 --num_envs 32 --headless \
      env.scene.insertive_object=realpcb env.scene.receptive_object=openbox \
      env.events.reset_insertive_object_pose_from_reset_states.params.dataset_dir=./Datasets_realpcb_smoke \
      env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets_realpcb_smoke <TRIMS>
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--settle_steps", type=int, default=40)
parser.add_argument("--resets_per_cfg", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from typing import cast

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose

P16 = math.pi / 16.0   # 11.25 deg (baseline)
P60 = math.pi / 60.0   # 3.0 deg
P120 = math.pi / 120.0  # 1.5 deg
# NOTE: pose_range_b is in the gripper BODY frame; local +Z is the (downward) approach axis, so
# a POSITIVE z draw drives the tip DEEPER (toward/into the table). To keep the tip clear we bias z
# NEGATIVE (shallower / away from table). yaw is about the vertical approach -> does not change tip Z,
# so we keep full yaw diversity. x/y are lateral along the face -> harmless, kept for diversity.

# name -> {x,y,z,roll,pitch,yaw : (lo,hi)}  (missing key -> (0,0))
CONFIGS = [
    ("baseline ±2cm/±11.25°",   dict(x=(-.02, .02), y=(-.02, .02), z=(-.02, .02), roll=(-P16, P16), pitch=(-P16, P16), yaw=(-P16, P16))),
    ("none (0/0)",              dict()),
    ("shallow z[-1cm,0]/rp±3",  dict(x=(-.005, .005), y=(-.005, .005), z=(-.01, 0.0), roll=(-P60, P60), pitch=(-P60, P60), yaw=(-P16, P16))),
    ("shallow z[-2cm,0]/rp±3",  dict(x=(-.005, .005), y=(-.005, .005), z=(-.02, 0.0), roll=(-P60, P60), pitch=(-P60, P60), yaw=(-P16, P16))),
    ("shallow z[-1cm,0]/rp±1.5", dict(x=(-.005, .005), y=(-.005, .005), z=(-.01, 0.0), roll=(-P120, P120), pitch=(-P120, P120), yaw=(-P16, P16))),
    ("minimal z[-5mm,0]/rp±1.5", dict(x=(-.003, .003), y=(-.003, .003), z=(-.005, 0.0), roll=(-P120, P120), pitch=(-P120, P120), yaw=(-P16, P16))),
]
KEYS = ["x", "y", "z", "roll", "pitch", "yaw"]


def find_grasp_ee_term(env):
    em = env.event_manager
    cands = []
    for attr in ["_mode_class_term_cfgs", "_mode_term_cfgs"]:
        d = getattr(em, attr, None)
        if isinstance(d, dict):
            for terms in d.values():
                for t in terms:
                    cands.append(getattr(t, "func", t))
    try:
        cfg = em.get_term_cfg("reset_end_effector_pose_from_grasp_dataset")
        cands.append(getattr(cfg, "func", None))
    except Exception:
        pass
    for fn in cands:
        r = getattr(fn, "ranges", None)
        if r is not None and hasattr(r, "shape") and tuple(r.shape) == (6, 2):
            return fn
    return None


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = None
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    robot = env.scene["robot"]
    pcb = env.scene["insertive_object"]
    body_names = robot.data.body_names
    base_idx = body_names.index("robotiq_base_link")
    TIP_LOCAL = torch.tensor([0.0, 0.0, 0.144], device=env.device).expand(env.num_envs, 3)

    term = find_grasp_ee_term(env)
    if term is None:
        print("[sweep] ERROR: could not locate grasp-EE event term (.ranges).")
        env.close()
        return
    lf_idx = body_names.index("left_inner_finger")
    rf_idx = body_names.index("right_inner_finger")

    actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
    actions[:, -1] = -1.0  # keep gripper closed

    def measure():
        base_pos = robot.data.body_link_pos_w[:, base_idx, :]
        base_quat = robot.data.body_link_quat_w[:, base_idx, :]
        tip_z = (base_pos + quat_apply(base_quat, TIP_LOCAL))[:, 2]
        jaw_gap = (robot.data.body_link_pos_w[:, lf_idx, :] - robot.data.body_link_pos_w[:, rf_idx, :]).norm(dim=1)
        return tip_z, jaw_gap, pcb.data.root_pos_w[:, 2]

    # Establish a FIXED table-top height from a zero-jitter warmup (object rests on the table).
    term.ranges = torch.zeros((6, 2), device=env.device, dtype=term.ranges.dtype)
    env.reset()
    for _ in range(args_cli.settle_steps):
        env.step(actions)
    _, gap0, pcb_z0 = measure()
    TABLE_TOP = (pcb_z0 - 0.0015).median().item()  # PCB bottom when resting == table surface
    GRIP_GAP = 0.04  # jaws holding the 100 mm object keep the fingers far apart; empty-closed ~ few mm
    print(f"[sweep] TABLE_TOP={TABLE_TOP:.5f}  warmup median jaw_gap={gap0.median().item():.4f} "
          f"(gripped if gap>{GRIP_GAP})")
    print(f"\n[sweep] num_envs={env.num_envs} resets_per_cfg={args_cli.resets_per_cfg} settle={args_cli.settle_steps}")
    print("  Valid resting edge-grasp = gripped AND tip within [-tol, object_top]. tip clr vs FIXED table top.")
    header = (f"{'config':<26} {'N':>4} {'grip%':>6} {'good%':>6} "
              f"{'bur>1mm%':>8} {'bur>2mm%':>8} {'medClr_mm':>10} {'medGap':>7}")
    print(header)
    print("-" * len(header))

    for name, cfg in CONFIGS:
        range_list = [cfg.get(k, (0.0, 0.0)) for k in KEYS]
        term.ranges = torch.tensor(range_list, device=env.device, dtype=term.ranges.dtype)
        tips, gaps = [], []
        for _ in range(args_cli.resets_per_cfg):
            env.reset()
            for _ in range(args_cli.settle_steps):
                env.step(actions)
            tz, gp, _ = measure()
            tips.append(tz)
            gaps.append(gp)
        tip = torch.cat(tips)
        gap = torch.cat(gaps)
        clr = (tip - TABLE_TOP) * 1000.0  # mm, >0 above table
        gripped = gap > GRIP_GAP
        # good = gripped AND tip in [-1mm, object_top(3mm)+2mm tol] -> pinching the edge near the table
        good = gripped & (clr > -1.0) & (clr < 5.0)
        n = clr.numel()
        pg = gripped.float().mean().item() * 100
        pgood = good.float().mean().item() * 100
        b1 = (gripped & (clr < -1.0)).float().mean().item() * 100
        b2 = (gripped & (clr < -2.0)).float().mean().item() * 100
        med_clr = clr[gripped].median().item() if gripped.any() else float("nan")
        med_gap = gap.median().item()
        print(f"{name:<26} {n:>4} {pg:>5.1f}% {pgood:>5.1f}% {b1:>7.1f}% {b2:>7.1f}% {med_clr:>10.2f} {med_gap:>7.3f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
