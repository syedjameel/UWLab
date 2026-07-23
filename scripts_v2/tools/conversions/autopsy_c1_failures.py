# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Failure autopsy for C1 (from-scratch) episodes of a trained state expert.

Plays a checkpoint on ObjectAnywhereEEAnywhere-only resets and classifies every episode end:

  SUCCESS           aligned (position & orientation) at episode end
  EARLY_KILL        terminated before timeout (wrist_camera_window / abnormal)
  NEVER_LIFTED      object max z-rise < 15 mm -> the pick never happened
  LIFTED_LOST       object was lifted but ends neither gripped nor aligned -> pick broke mid-way
  HELD_NOT_ALIGNED  ends gripped but not aligned -> pick fine, insertion/alignment failed

    ./uwlab.sh -p scripts_v2/tools/conversions/autopsy_c1_failures.py \
      --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-v0 \
      --checkpoint logs/rsl_rl/.../model_3600.pt --num_envs 32 --rounds 2 --headless \
      env.scene.insertive_object=realpcb env.scene.receptive_object=realopenbox \
      env.events.reset_from_reset_states.params.dataset_dir=./Datasets_realpcb/OmniReset <TRIMS>
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--rounds", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config

GRIP_LO, GRIP_HI = 0.012, 0.046


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    # C1-only resets from the local dataset (dataset_dir supplied via hydra override).
    env_cfg.events.reset_from_reset_states.params["reset_types"] = ["ObjectAnywhereEEAnywhere"]
    env_cfg.events.reset_from_reset_states.params["probs"] = [1.0]

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    dev = env.unwrapped.device

    # Make the (possibly newer) training cfg compatible with the installed rsl-rl, like play.py does.
    import os as _os
    sys.path.append(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  "..", "..", "..", "scripts", "reinforcement_learning", "rsl_rl"))
    import cli_args  # noqa: E402

    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(dev))
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=dev)

    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["insertive_object"]
    fj_id = robot.find_joints(["finger_joint"])[0][0]
    cmd = env.unwrapped.command_manager.get_term("task_command")
    max_len = int(env.unwrapped.max_episode_length)

    from isaaclab.utils.math import euler_xyz_from_quat

    def obj_yaw():
        _, _, yz = euler_xyz_from_quat(obj.data.root_quat_w)
        return yz

    box = env.unwrapped.scene["receptive_object"]
    ee_idx = robot.data.body_names.index("robotiq_base_link")

    cls = {"SUCCESS": 0, "EARLY_KILL": 0, "NEVER_LIFTED": 0, "LIFTED_LOST": 0, "HELD_NOT_ALIGNED": 0}
    zrise_all = []
    yaw_outcomes = []  # (board yaw at reset [rad], outcome str)
    geo_outcomes = []  # (outcome, board_box_dist0, ee_board_dist_end, board_x0, board_y0)

    obs = env.get_observations()
    n = env.unwrapped.num_envs
    z0 = obj.data.root_pos_w[:, 2].clone()
    yaw0 = obj_yaw().clone()
    p0 = obj.data.root_pos_w[:, :2].clone()
    boxp0 = box.data.root_pos_w[:, :2].clone()
    zmax = z0.clone()
    cnt = torch.zeros(n, dtype=torch.long, device=dev)
    episodes_done = 0
    target = args_cli.rounds * n

    prev_fj = robot.data.joint_pos[:, fj_id].clone()
    prev_aligned = (cmd.position_aligned & cmd.orientation_aligned).clone()
    prev_ee_obj = (robot.data.body_link_pos_w[:, ee_idx] - obj.data.root_pos_w).norm(dim=1).clone()

    for _ in range(args_cli.rounds * (max_len + 10)):
        with torch.no_grad():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        d = dones.bool().flatten()
        # post-step (possibly post-reset for done envs) readings:
        z = obj.data.root_pos_w[:, 2]
        fj = robot.data.joint_pos[:, fj_id]
        aligned = cmd.position_aligned & cmd.orientation_aligned
        cnt += 1
        if d.any():
            yaw_now = obj_yaw()
            for i in torch.where(d)[0].tolist():
                zr = float(zmax[i] - z0[i])
                zrise_all.append(zr)
                if bool(prev_aligned[i]):
                    out = "SUCCESS"
                elif int(cnt[i]) < max_len - 1:
                    out = "EARLY_KILL"
                elif zr < 0.015:
                    out = "NEVER_LIFTED"
                elif GRIP_LO <= float(prev_fj[i]) <= GRIP_HI:
                    out = "HELD_NOT_ALIGNED"
                else:
                    out = "LIFTED_LOST"
                cls[out] += 1
                yaw_outcomes.append((float(yaw0[i]), out))
                geo_outcomes.append((out,
                                     float((p0[i] - boxp0[i]).norm()),
                                     float(prev_ee_obj[i]),
                                     float(p0[i][0]), float(p0[i][1])))
                episodes_done += 1
                # re-baseline this env for its next episode (post-reset state)
                z0[i] = z[i]
                zmax[i] = z[i]
                yaw0[i] = yaw_now[i]
                p0[i] = obj.data.root_pos_w[i, :2]
                boxp0[i] = box.data.root_pos_w[i, :2]
                cnt[i] = 0
        live = ~d
        zmax = torch.where(live, torch.maximum(zmax, z), zmax)
        prev_fj = fj.clone()
        prev_aligned = aligned.clone()
        prev_ee_obj = (robot.data.body_link_pos_w[:, ee_idx] - obj.data.root_pos_w).norm(dim=1).clone()
        if episodes_done >= target:
            break

    import numpy as np

    zr = np.array(zrise_all)
    print(f"\n[autopsy] C1 episodes: {episodes_done}  (checkpoint: {args_cli.checkpoint.split('/')[-1]})")
    for k, v in cls.items():
        print(f"[autopsy]   {k:<18s} {v:>4d}  ({v / max(episodes_done, 1) * 100:5.1f}%)")
    print(f"[autopsy] object z-rise (cm): med={np.median(zr) * 100:.1f}  p25={np.percentile(zr, 25) * 100:.1f}  "
          f"p75={np.percentile(zr, 75) * 100:.1f}  frac lifted>1.5cm: {np.mean(zr > 0.015) * 100:.0f}%")
    # Board-yaw vs outcome: the width-grip closing axis rotates 1:1 with wrist_3, which the D3
    # window confines to a 120 deg span -- if failures cluster in a ~60 deg (mod 180) yaw band,
    # those picks are geometrically impossible without the camera leaving the window.
    import math
    bins = 12  # 15 deg bins over yaw mod 180
    succ = np.zeros(bins); fail = np.zeros(bins)
    for y, out in yaw_outcomes:
        b = int(((math.degrees(y) % 180.0) // 15.0)) % bins
        if out == "SUCCESS":
            succ[b] += 1
        elif out == "NEVER_LIFTED":
            fail[b] += 1
    print("[autopsy] board yaw (mod 180, 15deg bins) success/never-lifted:")
    print("[autopsy]   bin_start: " + " ".join(f"{i*15:>5d}" for i in range(bins)))
    print("[autopsy]   success  : " + " ".join(f"{int(v):>5d}" for v in succ))
    print("[autopsy]   nolift   : " + " ".join(f"{int(v):>5d}" for v in fail))
    tot = succ + fail
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(tot > 0, succ / np.maximum(tot, 1), np.nan)
    print("[autopsy]   pick rate: " + " ".join(("  nan" if np.isnan(r) else f"{r:5.2f}") for r in rate))
    # Spatial signature: per outcome, board->box distance at reset, EE->board distance at episode
    # end (did the policy even go to the board?), and board reset position.
    print("[autopsy] spatial per outcome (median [p10,p90]):")
    for want in ("SUCCESS", "NEVER_LIFTED"):
        g = [t for t in geo_outcomes if t[0] == want]
        if not g:
            continue
        bb = np.array([t[1] for t in g]); eo = np.array([t[2] for t in g])
        bx = np.array([t[3] for t in g]); by = np.array([t[4] for t in g])
        print(f"[autopsy]   {want:<14s} n={len(g):<4d} board-box dist: {np.median(bb):.3f} "
              f"[{np.percentile(bb,10):.3f},{np.percentile(bb,90):.3f}]  "
              f"EE-board@end: {np.median(eo):.3f} [{np.percentile(eo,10):.3f},{np.percentile(eo,90):.3f}]  "
              f"board x: {np.median(bx):.3f} [{np.percentile(bx,10):.3f},{np.percentile(bx,90):.3f}]  "
              f"y: {np.median(by):+.3f} [{np.percentile(by,10):+.3f},{np.percentile(by,90):+.3f}]")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
