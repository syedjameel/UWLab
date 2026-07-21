# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained state-expert (exported jit actor) and measure where the WRIST CAMERA settles.

Uses the exported actor (`exported/policy.pt`, obs 195) so the critic-obs mismatch from disabling
the corner pillars (172 vs 160) is irrelevant -- inference only needs the actor. At each episode
end we record the wrist-camera optical-axis heading (world XY, 0deg=+X toward the operator).

    ./uwlab.sh -p scripts_v2/tools/conversions/measure_expert_camera_settle.py \
      --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-Finetune-Play-v0 --num_envs 32 --headless \
      --jit logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/2026-07-13_14-24-16/exported/policy.pt \
      env.scene.insertive_object=pcb env.scene.receptive_object=openbox \
      env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e/OmniReset <TRIMS>
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--jit", type=str, required=True, help="exported policy.pt (jit actor)")
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--cam_rot", type=float, nargs=4, default=None,
                    help="wrist-camera rot quat wxyz (robotiq_base_link->cam, opengl). "
                         "Default = UR10e linear-gripper D405; pass the 2F-85 D415 rot for the authors' model.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import collections
import math

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply, quat_mul

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config

CAM_ROT = (0.0018127, -0.0175937, 0.91723, -0.3979652)  # robotiq_base_link -> wrist cam (opengl)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = 0

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    dev = env.unwrapped.device

    policy = torch.jit.load(args_cli.jit, map_location=dev)
    policy.eval()

    robot = env.unwrapped.scene["robot"]
    base_idx = robot.data.body_names.index("robotiq_base_link")
    w3_jid = robot.find_joints(["wrist_3_joint"])[0][0]
    cam_rot = torch.tensor(args_cli.cam_rot if args_cli.cam_rot is not None else CAM_ROT, device=dev)
    optical = torch.tensor([0.0, 0.0, -1.0], device=dev)

    def _heading_and_w3():
        n = env.unwrapped.num_envs
        bq = robot.data.body_link_quat_w[:, base_idx, :]
        cq = quat_mul(bq, cam_rot.expand(n, 4))
        ax = quat_apply(cq, optical.expand(n, 3))
        hd = torch.atan2(ax[:, 1], ax[:, 0]) * 180.0 / math.pi
        w3 = robot.data.joint_pos[:, w3_jid]
        w3 = (torch.remainder(w3 + math.pi, 2 * math.pi) - math.pi) * 180.0 / math.pi
        return hd, w3

    def policy_obs(o):
        """Extract the flat actor obs (195) from the wrapper's return (TensorDict / tuple / tensor)."""
        if isinstance(o, tuple):
            o = o[0]
        if hasattr(o, "keys") and "policy" in o.keys():
            o = o["policy"]
        return o

    headings, w3s = [], []
    obs = env.get_observations()

    # Reset-state reference (camera should be +X here): measure BEFORE the policy drifts it.
    _h0, _w30 = _heading_and_w3()
    print(f"[camsettle] RESET reference: camera heading mean={float(_h0.mean()):+.1f}deg  "
          f"wrist_3 mean={float(_w30.mean()):+.1f}deg (median={float(_w30.median()):+.1f})", flush=True)

    for _ in range(args_cli.max_steps):
        h, w3 = _heading_and_w3()
        with torch.no_grad():
            actions = policy(policy_obs(obs))
            obs, _, dones, _ = env.step(actions)
        d = dones.bool().flatten()
        if d.any():
            headings.extend(h[d].tolist())
            w3s.extend(w3[d].tolist())
        if len(headings) >= 200:
            break

    def _circ_mean(xs):
        t = torch.tensor(xs) * math.pi / 180
        return math.degrees(math.atan2(float(torch.sin(t).mean()), float(torch.cos(t).mean())))

    t = torch.tensor(headings)
    w = torch.tensor(w3s)
    mean_h = _circ_mean(headings)
    mean_w = _circ_mean(w3s)
    c, s = torch.cos(t * math.pi / 180), torch.sin(t * math.pi / 180)
    R = float(torch.sqrt(c.mean() ** 2 + s.mean() ** 2))
    # per-episode camera-vs-wrist offset: heading = wrist_3 + offset (mod 360). Stable at the
    # top-down settle pose -> gives the wrist_3 that yields camera=+X, and the +-60deg window.
    off = ((t - w + 180.0) % 360.0) - 180.0
    mean_off = _circ_mean(off.tolist())
    target_w3 = (((0.0 - mean_off) + 180.0) % 360.0) - 180.0  # wrist_3 giving camera heading 0 (+X)
    sect = collections.Counter(((torch.round(t / 45.0).long() * 45) % 360).tolist())
    print(f"\n[camsettle] TRAINED EXPERT (model_5300) settle over {len(headings)} episodes")
    print(f"[camsettle]   camera heading mean = {mean_h:+.1f}deg (0=+X)   concentration R = {R:.2f}")
    print(f"[camsettle]   wrist_3 mean = {mean_w:+.1f}deg   (heading - wrist_3) offset mean = {mean_off:+.1f}deg")
    print(f"[camsettle]   => wrist_3 for camera=+X is ~{target_w3:+.1f}deg; window +-60 = "
          f"[{target_w3-60:+.1f}, {target_w3+60:+.1f}] deg", flush=True)
    print("[camsettle]   heading histogram (deg:count): "
          + "  ".join(f"{int(k):>4}:{v}" for k, v in sorted(sect.items())))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
