# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose WHY ObjectRestingEEGrasped (C2) recording produces zero held grips on the thin PCB.

Replays the C2 reset (object resting + EE placed at a recorded grasp) and steps with the same
close command the recorder uses, logging per episode:

  * fj0        finger_joint right after the reset event writes the grasp joints
  * tip0       jaw-tip height above the table at reset (must be within [0, 3mm] to catch the board)
  * obj_disp   object XY displacement from reset to episode end
  * fj_end     finger_joint at episode end

and classifies each ended episode:
  HELD          fj_end in [0.012, 0.045]     -> a true width grip survived settle
  ABOVE (H1)    closed past + tip0 > 4 mm    -> jaws closed above the 3 mm board (never engaged)
  SQUIRT (H2)   closed past + obj moved >5mm -> board squeezed out sideways along the table
  PASSED (H3)   closed past + tip low + obj still -> jaws slid past the edges without catching
  OPEN          fj_end < 0.012

    ./uwlab.sh -p scripts_v2/tools/conversions/diagnose_c2_grip_slip.py \
      --task OmniReset-UR10eLinearGripper-ObjectRestingEEGrasped-v0 --num_envs 32 --headless \
      env.scene.insertive_object=realpcb env.scene.receptive_object=realopenbox \
      env.events.reset_insertive_object_pose_from_reset_states.params.dataset_dir=./Datasets_realpcb/OmniReset \
      env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets_realpcb/OmniReset <TRIMS>
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--rounds", type=int, default=3, help="Reset rounds (each = num_envs episodes).")
parser.add_argument("--max_steps", type=int, default=40)
parser.add_argument("--table_top", type=float, default=0.004)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from typing import cast

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = None
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    robot = env.scene["robot"]
    obj = env.scene["insertive_object"]
    base_idx = robot.data.body_names.index("robotiq_base_link")
    fj_id = robot.find_joints(["finger_joint"])[0][0]
    rfj_id = robot.find_joints(["right_finger_joint"])[0][0]
    TIP = torch.tensor([0.0, 0.0, 0.144], device=env.device).expand(env.num_envs, 3)

    actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
    actions[:, -1] = -1.0  # the recorder's close command for EEGrasped types

    def snap():
        tip_z = (robot.data.body_link_pos_w[:, base_idx, :]
                 + quat_apply(robot.data.body_link_quat_w[:, base_idx, :], TIP))[:, 2]
        fj = robot.data.joint_pos[:, fj_id]
        rfj = robot.data.joint_pos[:, rfj_id]
        oxy = obj.data.root_pos_w[:, :2].clone()
        return tip_z, fj, rfj, oxy

    cls = {"HELD": 0, "ABOVE(H1)": 0, "SQUIRT(H2)": 0, "PASSED(H3)": 0, "OPEN": 0}
    held_inband = held_total = inband_total = 0
    fj0_all, tip0_all, engaged_band = [], [], []
    fj_traj_sum = np.zeros(args_cli.max_steps)
    fj_traj_cnt = np.zeros(args_cli.max_steps)

    for _ in range(args_cli.rounds):
        env.reset()
        tip0, fj0, _, oxy0 = snap()
        tip0_rel = tip0 - args_cli.table_top
        fj0_all.extend(fj0.tolist())
        tip0_all.extend(tip0_rel.tolist())
        engaged_band.extend(((tip0_rel >= -0.001) & (tip0_rel <= 0.003)).tolist())
        done_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        max_disp = torch.zeros(env.num_envs, device=env.device)
        # NOTE: env.step auto-resets terminated envs BEFORE returning, so terminal-state values
        # must come from the PRE-step snapshot of that env, not the post-step read.
        prev_fj = fj0.clone()
        for s in range(args_cli.max_steps):
            _, _, terminated, truncated, _ = env.step(actions)
            tip_z, fj, rfj, oxy = snap()
            live = ~done_mask
            newly = (terminated | truncated).flatten() & live
            if newly.any():
                for i in torch.where(newly)[0].tolist():
                    f = float(prev_fj[i])  # last pre-termination value
                    inband = -0.001 <= float(tip0_rel[i]) <= 0.003
                    inband_total += int(inband)
                    if 0.012 <= f <= 0.045:
                        cls["HELD"] += 1
                        held_total += 1
                        held_inband += int(inband)
                    elif f < 0.012:
                        cls["OPEN"] += 1
                    elif float(tip0_rel[i]) > 0.004:
                        cls["ABOVE(H1)"] += 1
                    elif float(max_disp[i]) > 0.005:
                        cls["SQUIRT(H2)"] += 1
                    else:
                        cls["PASSED(H3)"] += 1
                done_mask |= newly
            live = ~done_mask
            fj_traj_sum[s] += float(fj[live].sum())
            fj_traj_cnt[s] += int(live.sum())
            disp = (oxy - oxy0).norm(dim=1)
            max_disp = torch.where(live, torch.maximum(max_disp, disp), max_disp)
            prev_fj = fj.clone()
            if bool(done_mask.all()):
                break

    n = sum(cls.values())
    fj0 = np.array(fj0_all)
    tip0 = np.array(tip0_all) * 1000
    print(f"\n[c2diag] episodes classified: {n}")
    print(f"[c2diag] fj AFTER RESET: med={np.median(fj0):.4f}  frac at grasp-width(<0.03)={np.mean(fj0 < 0.03) * 100:.0f}%  "
          f"frac already-closed(>0.045)={np.mean(fj0 > 0.045) * 100:.0f}%")
    print(f"[c2diag] tip height above table at reset (mm): med={np.median(tip0):+.1f}  "
          f"p10={np.percentile(tip0, 10):+.1f}  p90={np.percentile(tip0, 90):+.1f}  "
          f"in catch band [-1,+3]mm: {np.mean(engaged_band) * 100:.0f}%")
    med_traj = np.where(fj_traj_cnt > 0, fj_traj_sum / np.maximum(fj_traj_cnt, 1), np.nan)
    print("[c2diag] mean fj by step (live envs): "
          + "  ".join(f"s{s}:{med_traj[s]:.3f}" for s in range(0, min(args_cli.max_steps, 16), 2)))
    print("[c2diag] classification: " + "  ".join(f"{k}={v} ({v / max(n,1) * 100:.0f}%)" for k, v in cls.items()))
    print(f"[c2diag] catch-band episodes: {inband_total}  of which HELD: {held_inband} "
          f"({held_inband / max(inband_total, 1) * 100:.0f}%)  [held total: {held_total}]")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
