# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run grasp sampling using IsaacLab framework."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import time
from tqdm import tqdm
from typing import cast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Grasp sampling for end effector on objects.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="OmniReset-Robotiq2f85-GraspSampling-v0", help="Name of the task.")
parser.add_argument(
    "--dataset_dir", type=str, default="./Datasets/OmniReset/", help="Root Datasets/OmniReset/ directory."
)
parser.add_argument("--num_grasps", type=int, default=500, help="Number of grasp candidates to evaluate.")

AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.recorder_manager import DatasetExportMode

from uwlab.utils.datasets.torch_dataset_file_handler import TorchDatasetFileHandler

import uwlab_tasks  # noqa: F401
import uwlab_tasks.manager_based.manipulation.omnireset.mdp as task_mdp
from uwlab_tasks.utils.hydra import hydra_task_compose

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    """Main function to run grasp sampling."""
    # create directory if it does not exist
    if not os.path.exists(args_cli.dataset_dir):
        os.makedirs(args_cli.dataset_dir, exist_ok=True)

    # Derive object name for output path
    object_usd_path = env_cfg.scene.object.spawn.usd_path
    obj_name = task_mdp.utils.object_name_from_usd(object_usd_path)
    output_dir = os.path.join(args_cli.dataset_dir, "Grasps", obj_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Recording grasps for: {obj_name}")
    print(f"Object: {object_usd_path}")
    print(f"Output: {output_dir}/grasps.pt")

    # The body the recorded grasp pose is expressed in. Taken from the task's own grasp-sampling
    # term rather than hard-coded, so it is right for every gripper by construction: that term is
    # what POSES the gripper, so recording against any other body would record a pose the sampler
    # never produced.
    #
    # This is a no-op for the shipped grippers -- ``grasp_sampling_cfg`` binds
    # ``robotiq_base_link``, and the linear gripper renames its links to that same contract, so
    # both resolve to exactly the value that used to be written here. The DELTO does not: its palm
    # is ``rl_dg_mount``, and ``DeltoGraspSamplingCfg`` repoints the term accordingly.
    #
    # Getting this wrong is not a clean failure. ``GraspRelativePoseRecorder`` matches the body by
    # substring and leaves the index as ``None`` when nothing matches; ``body_state_w[ids, None, :3]``
    # then INSERTS an axis instead of raising, and the run dies later inside
    # ``subtract_frame_transforms`` with a tensor-size error that names no body at all.
    gripper_body_name = env_cfg.events.grasp_sampling.params["gripper_cfg"].body_names
    if isinstance(gripper_body_name, (list, tuple)):
        gripper_body_name = gripper_body_name[0]
    print(f"Gripper body: {gripper_body_name}")

    # Configure recorder
    env_cfg.recorders = task_mdp.GraspRelativePoseRecorderManagerCfg(
        robot_name="robot",
        object_name="object",
        gripper_body_name=gripper_body_name,
    )
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = "grasps.pt"
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = TorchDatasetFileHandler

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # make sure environment is non-deterministic, so we don't get redundant trajectories between datasets!
    env_cfg.seed = None

    # Create environment
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped

    # Reset environment (this will trigger grasp sampling event)
    env.reset()

    # Run grasp sampling
    num_grasps_evaluated = 0
    current_successful_grasps = 0

    # Create progress bar for successful grasps
    pbar = tqdm(total=args_cli.num_grasps, desc="Successful grasps", unit="grasps")

    start_time = time.time()

    # --- stall-vs-slow instrumentation (bead UWLab-z9j.11 follow-on) -----------------------------
    # The pbar above counts SUCCESSES (exported_successful_episode_count), so it stays at zero for
    # arbitrarily long stretches whenever every attempt in that stretch fails -- indistinguishable
    # from outside between "the rollout is hung" and "it is running fine and just has not produced
    # a success yet". These prints exist to tell those two apart without guessing.
    #
    # NOTE ON "BATCH": the loop below is a flat while/env.step() -- there is no batch object
    # anywhere in this file or in grasp_sampling_event. Each reset draws its candidate via
    # torch.randint (WITH replacement) from the precomputed pool (events.py, _generate_grasp_
    # candidates / _filter_placement_collisions run once, lazily, inside the FIRST call), not a
    # sweep through it in order, and per-env episodes desync over the run as some finish early and
    # restart before others -- only the very first reset (line ~115, above this loop) has every env
    # synchronized. "Batch" here is therefore DEFINED, not observed: a fixed-size accounting window
    # of ``env.max_episode_length`` env-steps (the nominal per-episode step count). Treat the
    # batch_end/batch_start pair below as a coarse round-number marker, not a real synchronization
    # boundary -- by the last few batches of a run, individual envs can be anywhere in their own
    # episode. Candidate counts reported as "consumed" are cumulative ATTEMPTS (dones), not pool
    # depletion -- the same candidate can be (and typically will be) redrawn.
    max_episode_length = env.max_episode_length
    total_candidate_pool = len(env.grasp_candidates) if hasattr(env, "grasp_candidates") else None
    coarse_interval = 30  # env-steps; within the 30-60 range requested

    env_step_index = 0
    batch_index = 0
    batch_start_time = start_time
    batch_start_attempts = 0
    batch_start_successes = 0

    print(
        f"[grasp record] {time.strftime('%H:%M:%S')} START env-step=0 batch=0 "
        f"candidate_pool={total_candidate_pool} num_envs={env.num_envs} "
        f"max_episode_length={max_episode_length} target_successes={args_cli.num_grasps}",
        flush=True,
    )

    while current_successful_grasps < args_cli.num_grasps:
        # The scripted CLOSE, asked of the env's own action terms rather than assumed.
        #
        # This used to be ``-torch.ones(env.action_space.shape)`` hoisted out of the loop: correct
        # for a BinaryJointAction, where the single dimension follows Isaac Lab's negative=close
        # convention, and exactly wrong for the fully actuated DELTO hand, where -1 on all twenty
        # dimensions commands 0.1 rad of EXTENSION per joint per step -- it would have opened the
        # hand while the sampler waited for a grasp. ``scripted_gripper_actions`` servos each joint
        # toward a measured closed posture by NAME, so it must be recomputed every step.
        actions = task_mdp.scripted_gripper_actions(env, close=True)
        _, _, terminated, truncated, _ = env.step(actions)
        dones = terminated | truncated
        env_step_index += 1

        # Update progress based on successful grasps
        new_successful_count = env.recorder_manager.exported_successful_episode_count
        if new_successful_count > current_successful_grasps:
            increment = new_successful_count - current_successful_grasps
            current_successful_grasps = new_successful_count
            pbar.update(increment)

        # Count total grasps evaluated (sum across all environments)
        num_grasps_evaluated += dones.sum().item()

        # Coarse per-batch physics-step progress: proves env.step() is still advancing (env-step
        # index and elapsed time both increasing) even while successes sit at zero, and shows where
        # individual envs currently sit within their own episode (episode_length_buf), which is
        # where a genuinely stuck batch would show up -- a min/mean/max that stops changing.
        if env_step_index % coarse_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"[grasp record] {time.strftime('%H:%M:%S')} PROGRESS env-step={env_step_index} "
                f"batch={batch_index} step_in_batch={env_step_index - batch_index * max_episode_length} "
                f"elapsed={elapsed:.1f}s attempts={num_grasps_evaluated} successes={current_successful_grasps} "
                f"episode_len_buf(min/mean/max)="
                f"{env.episode_length_buf.min().item()}/{env.episode_length_buf.float().mean().item():.1f}/"
                f"{env.episode_length_buf.max().item()}",
                flush=True,
            )

        # Batch end/start markers, on the fixed-size accounting window defined above.
        if env_step_index % max_episode_length == 0:
            batch_wall_clock = time.time() - batch_start_time
            batch_attempts = num_grasps_evaluated - batch_start_attempts
            batch_successes = current_successful_grasps - batch_start_successes
            pool_str = "?" if total_candidate_pool is None else str(total_candidate_pool)
            print(
                f"[grasp record] {time.strftime('%H:%M:%S')} BATCH_END batch={batch_index} "
                f"batch_size={env.num_envs} attempts_in_batch={batch_attempts} "
                f"attempts_consumed_so_far={num_grasps_evaluated}/{pool_str} "
                f"wall_clock_for_batch={batch_wall_clock:.1f}s successes_in_batch={batch_successes} "
                f"cumulative_successes={current_successful_grasps}",
                flush=True,
            )
            batch_index += 1
            batch_start_time = time.time()
            batch_start_attempts = num_grasps_evaluated
            batch_start_successes = current_successful_grasps
            print(
                f"[grasp record] {time.strftime('%H:%M:%S')} BATCH_START batch={batch_index} "
                f"batch_size={env.num_envs}",
                flush=True,
            )

        # Check if simulation should stop
        if env.sim.is_stopped():
            break

    pbar.close()

    # Get final statistics
    final_successful_grasps = env.recorder_manager.exported_successful_episode_count

    print("Grasp sampling complete!")
    print(f"Total grasps evaluated: {num_grasps_evaluated}")
    print(f"Successful grasps: {final_successful_grasps}")
    if num_grasps_evaluated > 0:
        print(f"Success rate: {final_successful_grasps / num_grasps_evaluated:.2%}")
        print(f"Time taken: {(time.time() - start_time) / 60:.2f} minutes")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
