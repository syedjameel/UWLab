# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hold-test recorded EEGrasped reset states: reload them and step with the CLOSE command
(the same command the recorder used), measuring whether the object stays in the jaws.

Built after the jig 'always falls in the visualizer' investigation: visualize_reset_states
decides open-vs-close from the jaw CLOSED FRACTION (>10%), which misclassifies wide objects
(the 129 mm jig closes the jaws only ~5%) and actively opens the gripper. This script applies
the correct command and reports the physical truth.

    ./uwlab.sh -p scripts_v2/tools/conversions/holdtest_grasped_resets.py \
      --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-v0 --num_envs 32 --headless \
      --reset_type ObjectAnywhereEEGrasped \
      env.scene.insertive_object=jig env.scene.receptive_object=bottomenclosure \
      env.events.reset_from_reset_states.params.dataset_dir=./Datasets_jig_smoke2/OmniReset <TRIMS>
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--reset_type", type=str, default="ObjectAnywhereEEGrasped")
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--hold_steps", type=int, default=50)
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

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=hydra_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = None
    env_cfg.events.reset_from_reset_states.params["reset_types"] = [args_cli.reset_type]
    env_cfg.events.reset_from_reset_states.params["probs"] = [1.0]

    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    robot = env.scene["robot"]
    obj = env.scene["insertive_object"]
    base_idx = robot.data.body_names.index("robotiq_base_link")
    fj_id = robot.find_joints(["finger_joint"])[0][0]

    actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
    actions[:, -1] = -1.0  # CLOSE -- what the recorder held during these states

    drops, rels, fj0s = [], [], []
    for _ in range(args_cli.rounds):
        env.reset()
        rel0 = obj.data.root_pos_w[:, 2] - robot.data.body_link_pos_w[:, base_idx, 2]
        fj0s.extend(robot.data.joint_pos[:, fj_id].tolist())
        for _ in range(args_cli.hold_steps):
            env.step(actions)
        rel = obj.data.root_pos_w[:, 2] - robot.data.body_link_pos_w[:, base_idx, 2]
        d = (rel0 - rel)  # positive = object moved DOWN relative to the gripper
        drops.extend(d.tolist())
        rels.extend(rel.tolist())

    d = np.array(drops) * 1000
    fj0 = np.array(fj0s)
    held = (d < 20).mean() * 100  # fell = slipped >2 cm relative to the hand over 5 s
    print(f"\n[holdtest] {args_cli.reset_type}  episodes={len(d)}  hold_steps={args_cli.hold_steps} (5 s, close cmd)")
    print(f"[holdtest]   fj at reset: med={np.median(fj0):.4f}")
    print(f"[holdtest]   relative drop (mm): med={np.median(d):.1f}  p90={np.percentile(d, 90):.1f}  max={d.max():.1f}")
    print(f"[holdtest]   HELD (slip < 20 mm): {held:.0f}%   FELL: {(d >= 20).mean() * 100:.0f}%")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
