# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the SEATING CAPTURE BASIN: from how far off-centre does the insertive object still
settle into a successful assembly?

Built for the jig-v2 investigation. ``visualize_perfect_mate`` only probes 7 hand-picked
configurations, which is enough to catch a broken seat but NOT enough to catch a *shrunken*
capture basin -- and a shrunken basin puts a hard CEILING on every task that has to seat
(offsets outside it can never succeed however good the policy is).

Each env is placed at a different (dx, dy) offset from the perfect mating point, dropped from a
small height, settled, and scored with the task's own success term. Prints a 2-D map plus the
per-axis capture radius, so two assets can be compared directly:

    ./uwlab.sh -p scripts_v2/tools/sweep_capture_basin.py \
      --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-v0 \
      --headless --span 10 --step 2 --drop 0.004 \
      env.scene.insertive_object=jigv2 env.scene.receptive_object=bottomenclosure <TRIMS>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--span", type=float, default=10.0, help="max |offset| swept, mm")
parser.add_argument("--step", type=float, default=2.0, help="grid step, mm")
parser.add_argument("--drop", type=float, default=0.004, help="height above the seat to drop from, m")
parser.add_argument("--settle_steps", type=int, default=25)
parser.add_argument("--out", type=str, default="/tmp/capture_basin.json")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from typing import cast

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_conjugate

import uwlab_tasks  # noqa: F401
import uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils as task_utils
from uwlab_tasks.utils.hydra import hydra_task_compose


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=hydra_args)
def main(env_cfg, agent_cfg) -> None:
    offs = np.arange(-args_cli.span, args_cli.span + 1e-6, args_cli.step)
    grid = [(float(dx), float(dy)) for dx in offs for dy in offs]
    env_cfg.scene.num_envs = len(grid)
    env_cfg.seed = 0
    env_cfg.events.reset_from_reset_states = None  # this script poses the objects itself

    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    ins, rec = env.scene["insertive_object"], env.scene["receptive_object"]
    dev = env.device

    ins_meta = task_utils.read_metadata_from_usd_directory(ins.cfg.spawn.usd_path)
    rec_meta = task_utils.read_metadata_from_usd_directory(rec.cfg.spawn.usd_path)
    ins_off = torch.tensor(ins_meta["assembled_offset"]["pos"], device=dev, dtype=torch.float32)
    rec_off = torch.tensor(rec_meta["assembled_offset"]["pos"], device=dev, dtype=torch.float32)
    print(f"[sweep] asset={os.path.basename(os.path.dirname(ins.cfg.spawn.usd_path))} "
          f"grid={len(offs)}x{len(offs)}={len(grid)} envs  drop={args_cli.drop*1000:.1f}mm", flush=True)

    env.reset()
    # Enclosure: identical pose in every env (yaw 0) so dx/dy mean the same thing everywhere.
    n = len(grid)
    rq = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).repeat(n, 1)
    rp = torch.tensor([0.55, 0.0, 0.0113], device=dev).repeat(n, 1)
    rec.write_root_pose_to_sim(torch.cat([rp, rq], dim=1))
    rec.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))

    d = torch.tensor([[dx / 1000.0, dy / 1000.0, args_cli.drop] for dx, dy in grid], device=dev)
    mate_w = rp + quat_apply(rq, rec_off.repeat(n, 1))
    jig_root = mate_w + d - quat_apply(rq, ins_off.repeat(n, 1))
    ins.write_root_pose_to_sim(torch.cat([jig_root, rq], dim=1))
    ins.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))

    actions = torch.zeros(env.action_space.shape, device=dev, dtype=torch.float32)
    success_fn = env.reward_manager.get_term_cfg("progress_context").func
    # Envs that terminate mid-settle get re-randomised by the reset machinery, which silently
    # corrupts their offset (seen as rel z ~ -675 mm, i.e. dropped out of the world). Track them
    # and exclude, rather than reporting garbage as a failed seat.
    dirty = torch.zeros(n, dtype=torch.bool, device=dev)
    for _ in range(args_cli.settle_steps):
        _, _, term, trunc, _ = env.step(actions)
        dirty |= (term | trunc)

    ok = success_fn.success.detach().cpu().numpy().astype(bool)
    bad = dirty.detach().cpu().numpy().astype(bool)
    if bad.any():
        print(f"  [warn] {bad.sum()}/{n} envs terminated mid-settle -- excluded from the map", flush=True)
    dp = ins.data.root_pos_w - rec.data.root_pos_w
    rel = quat_apply(quat_conjugate(rec.data.root_quat_w), dp)
    relz = (rel[:, 2] * 1000.0).detach().cpu().numpy()

    k = len(offs)
    ok = ok & ~bad
    S = ok.reshape(k, k)          # [ix, iy]
    Z = relz.reshape(k, k)
    B = bad.reshape(k, k)
    print("\n  seat success map   rows = dx (mm), cols = dy (mm)   '#'=seated  '.'=failed")
    print("        " + "".join(f"{o:+5.0f}" for o in offs))
    for i, dx in enumerate(offs):
        print(f"  {dx:+5.0f} " + "".join(f"{'    #' if S[i,j] else ('    ?' if B[i,j] else '    .')}" for j in range(k)))
    print("\n  settled rel z (mm)  rows = dx, cols = dy")
    print("        " + "".join(f"{o:+6.0f}" for o in offs))
    for i, dx in enumerate(offs):
        print(f"  {dx:+5.0f} " + "".join(f"{Z[i, j]:6.1f}" for j in range(k)))

    ax0 = np.argmin(np.abs(offs))
    def radius(mask):
        r = 0.0
        for i, o in enumerate(offs):
            if mask[i] and abs(o) > r:
                # only count if every smaller offset also seats (contiguous basin)
                lo, hi = sorted((ax0, i))
                if mask[lo:hi + 1].all():
                    r = abs(o)
        return r
    rx, ry = radius(S[:, ax0]), radius(S[ax0, :])
    print(f"\n  CAPTURE RADIUS along x: {rx:.0f} mm   along y: {ry:.0f} mm   "
          f"overall seated {100*S.mean():.1f}% of {len(grid)} offsets")
    json.dump({"asset": ins.cfg.spawn.usd_path, "offsets_mm": offs.tolist(),
               "success": S.tolist(), "rel_z_mm": Z.tolist(),
               "capture_radius_x_mm": rx, "capture_radius_y_mm": ry,
               "seated_fraction": float(S.mean())}, open(args_cli.out, "w"), indent=2)
    print(f"  wrote {args_cli.out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
