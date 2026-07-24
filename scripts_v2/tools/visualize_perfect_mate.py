# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize the PERFECT MATE of the insertive object on the receptive object (GUI).

Cycles through labeled configurations -- the metadata-defined assembled pose (yaw 0 and 180),
small in-tolerance offsets, and deliberate out-of-tolerance ones -- settles physics, and prints
the task's success flag next to each, so the success thresholds can be judged visually:

    ./uwlab.sh -p scripts_v2/tools/visualize_perfect_mate.py \
      --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-v0 \
      env.scene.insertive_object=jig env.scene.receptive_object=bottomenclosure <TRIMS>
"""

from __future__ import annotations

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--settle_steps", type=int, default=25)
parser.add_argument("--hold_seconds", type=float, default=4.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import contextlib
import math

import gymnasium as gym
import torch
from typing import cast

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

import uwlab_tasks  # noqa: F401
import uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils as task_utils
from uwlab_tasks.utils.hydra import hydra_task_compose


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=hydra_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    # No dataset-based resets: this viewer poses the objects itself.
    env_cfg.events.reset_from_reset_states = None

    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    ins = env.scene["insertive_object"]
    rec = env.scene["receptive_object"]
    dev = env.device

    ins_meta = task_utils.read_metadata_from_usd_directory(ins.cfg.spawn.usd_path)
    rec_meta = task_utils.read_metadata_from_usd_directory(rec.cfg.spawn.usd_path)
    ins_off = torch.tensor(ins_meta["assembled_offset"]["pos"], device=dev, dtype=torch.float32)
    rec_off = torch.tensor(rec_meta["assembled_offset"]["pos"], device=dev, dtype=torch.float32)
    thr = rec_meta.get("success_thresholds", {})
    print(f"[mate] insertive assembled_offset={ins_off.tolist()}  receptive={rec_off.tolist()}  "
          f"thresholds={thr}")

    # (label, expected, dx, dy, dz, droll, dpitch, dyaw) -- offsets applied to the perfect mate
    CONFIGS = [
        ("PERFECT MATE (yaw 0)",          True,  0.0,   0.0,  0.0,   0.0,  0.0, 0.0),
        ("PERFECT MATE (yaw 180)",        True,  0.0,   0.0,  0.0,   0.0,  0.0, math.pi),
        ("+3 mm x (in tolerance)",        True,  0.003, 0.0,  0.0,   0.0,  0.0, 0.0),
        ("+8 mm x (stays offset: FAIL)",  False, 0.008, 0.0,  0.0,   0.0,  0.0, 0.0),
        ("dropped from 10 mm: SELF-SEATS", True, 0.0,   0.0,  0.010, 0.0,  0.0, 0.0),
        ("tilted 3 deg: settles seated",  True,  0.0,   0.0,  0.002, 0.052, 0.0, 0.0),
        ("yaw 90 (rests ON pillars: FAIL)", False, 0.0, 0.0,  0.004, 0.0,  0.0, math.pi / 2),
    ]

    def set_pose(dx, dy, dz, dr, dp, dy_):
        rp = rec.data.root_pos_w[0]
        rq = rec.data.root_quat_w[0]
        dq = quat_from_euler_xyz(torch.tensor(dr, device=dev), torch.tensor(dp, device=dev),
                                 torch.tensor(dy_, device=dev))
        jq = quat_mul(rq.unsqueeze(0), dq.unsqueeze(0))[0]
        # jig root so that its assembled point lands on the receptive mating point (+ offset)
        mate_w = rp + quat_apply(rq.unsqueeze(0), rec_off.unsqueeze(0))[0]
        jig_root = mate_w + torch.tensor([dx, dy, dz], device=dev) - quat_apply(jq.unsqueeze(0), ins_off.unsqueeze(0))[0]
        pose = torch.cat([jig_root, jq]).unsqueeze(0)
        ins.write_root_pose_to_sim(pose)
        ins.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))

    actions = torch.zeros(env.action_space.shape, device=dev, dtype=torch.float32)
    success_fn = env.reward_manager.get_term_cfg("progress_context").func

    with contextlib.suppress(KeyboardInterrupt):
        while simulation_app.is_running():
            for label, expected, *ofs in CONFIGS:
                env.reset()
                # place the enclosure somewhere sensible on the green mat
                rq = quat_from_euler_xyz(torch.zeros(1, device=dev), torch.zeros(1, device=dev),
                                         torch.rand(1, device=dev) * 2 * math.pi - math.pi)[0]
                rec.write_root_pose_to_sim(torch.cat([torch.tensor([0.55, 0.0, 0.0113], device=dev), rq]).unsqueeze(0))
                rec.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
                set_pose(*ofs)
                for _ in range(args_cli.settle_steps):
                    env.step(actions)
                ok = bool(success_fn.success[0])
                # settled jig root in the enclosure frame (z tells the achieved seat depth)
                from isaaclab.utils.math import quat_conjugate
                dp = ins.data.root_pos_w[0] - rec.data.root_pos_w[0]
                rel = quat_apply(quat_conjugate(rec.data.root_quat_w[0]).unsqueeze(0), dp.unsqueeze(0))[0]
                verdict = "OK" if ok == expected else "  <-- MISMATCH vs expectation!"
                print(f"[mate] {label:<34s} expected={expected!s:<5s} success={ok!s:<5s} "
                      f"settled rel z={rel[2].item()*1000:6.1f} mm {verdict}", flush=True)
                time.sleep(args_cli.hold_seconds)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
