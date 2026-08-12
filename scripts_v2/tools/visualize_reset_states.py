# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to visualize saved reset states from a dataset directory."""

from __future__ import annotations

import argparse
import time
import torch
from typing import cast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Visualize saved reset states from a dataset directory.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="./Datasets/OmniReset",
    help="Base dataset directory (contains Resets/<Pair>/ subdirectories).",
)
parser.add_argument(
    "--reset_type",
    type=str,
    default=None,
    help="Single reset type to visualize (e.g. ObjectAnywhereEEAnywhere). If omitted, all four types are loaded.",
)
parser.add_argument("--reset_interval", type=float, default=0.1, help="Time interval between resets in seconds.")

AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import contextlib
import gymnasium as gym
import inspect

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase

import uwlab_tasks.manager_based.manipulation.omnireset.mdp as task_mdp
from uwlab_tasks.utils.hydra import hydra_task_compose

# A gripper joint this far from its OPEN posture counts as engaged. Replaces a 2F-85-specific
# "fraction of the close command" test, which cannot be written for a gripper that has no close
# command. 0.1 rad is the same 10% threshold on the 2F-85's ~0.8 rad driver stroke.
CLOSED_DISPLACEMENT_RAD = 0.1

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # make sure environment is non-deterministic for diverse pose discovery
    env_cfg.seed = None

    # Override existing MultiResetManager params to use the CLI-specified dataset/types
    ALL_RESET_TYPES = [
        "ObjectAnywhereEEAnywhere",
        "ObjectRestingEEGrasped",
        "ObjectAnywhereEEGrasped",
        "ObjectPartiallyAssembledEEGrasped",
    ]
    reset_types = [args_cli.reset_type] if args_cli.reset_type else ALL_RESET_TYPES
    env_cfg.events.reset_from_reset_states.params["dataset_dir"] = args_cli.dataset_dir
    env_cfg.events.reset_from_reset_states.params["reset_types"] = reset_types
    env_cfg.events.reset_from_reset_states.params["probs"] = [1.0] * len(reset_types)

    # create environment
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped

    # The EventManager is created before sim.play(), so ManagerTermBase classes
    # are deferred to a timeline callback that can silently fail.  Force-init any
    # class-based event terms that the callback missed.
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)

    env.reset()

    # Initialize variables
    print(f"Starting visualization of saved states from {args_cli.dataset_dir}")
    print("Press Ctrl+C to stop")

    with contextlib.suppress(KeyboardInterrupt):
        while True:
            asset = env.unwrapped.scene["robot"]
            # Whether each replayed state has the gripper engaged, so the viewer keeps holding
            # what the recorded state was holding instead of dropping it on the first step.
            #
            # This used to read the 2F-85's ``finger_joint`` and
            # ``env_cfg.actions.gripper.close_command_expr``, then write the command into
            # ``action[:, -1]``. All three are parallel-jaw-specific: on the DELTO the joint does
            # not exist, the action term has no close command at all (it is twenty independent
            # joints), and the last dimension is ``rj_dg_5_4`` alone. Ask the gripper's joints how
            # far they are from their OPEN posture instead, and let the scripted-gripper helper
            # write whichever dimensions this gripper actually owns.
            # The task's own per-gripper declaration -- the same list the gripper seam writes and
            # the startup check validates -- so this is right for every gripper by construction.
            gripper_joint_ids, _ = asset.find_joints(list(env_cfg.events.check_gripper_joints.params["joint_names"]))
            open_posture = asset.data.default_joint_pos[:, gripper_joint_ids]
            displacement = (asset.data.joint_pos[:, gripper_joint_ids] - open_posture).abs().max(dim=1).values
            gripper_mask = displacement > CLOSED_DISPLACEMENT_RAD
            # Step the simulation
            for _ in range(5):
                env.step(task_mdp.scripted_gripper_actions_for(env.unwrapped, gripper_mask))
            for _ in range(5):
                env.unwrapped.sim.step()
            success = env.unwrapped.reward_manager.get_term_cfg("progress_context").func.success
            print("Success: ", success)

            # Wait for the specified interval
            time.sleep(args_cli.reset_interval)

            # Reset the environment to load a new state
            env.reset()

    env.close()


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
