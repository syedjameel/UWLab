# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to record reset states using IsaacLab framework."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import re
import torch
from tqdm import tqdm
from typing import cast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record reset states for object pairs.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task", type=str, default="OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0", help="Name of the task."
)
parser.add_argument(
    "--dataset_dir", type=str, default="./Datasets/OmniReset/", help="Root Datasets/OmniReset/ directory."
)
parser.add_argument(
    "--reset_type",
    type=str,
    default=None,
    help="Reset type name (e.g. ObjectAnywhereEEAnywhere). Auto-inferred from --task if omitted.",
)
parser.add_argument(
    "--num_reset_states", type=int, default=100, help="Number of reset states to record. Set to 0 for infinite."
)

AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ISAAC SIM 6 PORT: register the deprecated isaacsim.core.utils ext path and wrap
# Isaac Lab's jit-scripted math helpers so they accept warp ProxyArray asset data.
# Must run after the app starts and before any uwlab import. Same two lines
# record_grasps.py already carries -- this script simply had not been run under Isaac Sim 6 yet.
import uwlab_is6_bridge  # noqa: E402
uwlab_is6_bridge.apply()

"""Rest everything else."""

import gymnasium as gym
import time

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


RESET_TYPE_PATTERN = re.compile(r"-(Object[A-Za-z0-9]+)-v\d+$")


def registered_reset_types() -> dict[str, str]:
    """Return reset-state task ids and their output type, derived from Gym's live registry."""
    reset_types: dict[str, str] = {}
    for spec in gym.registry.values():
        env_entry_point = (spec.kwargs or {}).get("env_cfg_entry_point")
        match = RESET_TYPE_PATTERN.search(spec.id)
        if isinstance(env_entry_point, str) and env_entry_point.endswith("ResetStatesCfg") and match:
            reset_types[spec.id] = match.group(1)
    return reset_types


def infer_reset_type(task_name: str) -> str:
    """Infer a reset type only when ``task_name`` is a registered reset-state task."""
    try:
        task_id = gym.spec(task_name).id
    except gym.error.Error as exc:
        raise ValueError(f"Cannot infer reset_type: task {task_name!r} is not registered.") from exc

    reset_types = registered_reset_types()
    if task_id not in reset_types:
        registered = ", ".join(sorted(reset_types))
        raise ValueError(
            f"Cannot infer reset_type: task {task_name!r} is not a registered reset-state task. "
            f"Registered reset-state tasks: {registered}"
        )
    return reset_types[task_id]


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    """Main function to record reset states."""
    # create directory if it does not exist
    if not os.path.exists(args_cli.dataset_dir):
        os.makedirs(args_cli.dataset_dir, exist_ok=True)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # make sure environment is non-deterministic for diverse pose discovery
    env_cfg.seed = None

    # OPTIONAL: drop the arm's task-space action term for recording.
    #
    # No policy acts during reset-state recording -- the reset event solves IK to place the end
    # effector at a sampled grasp, writes the joint state, and reconciles the PD targets to it
    # (events.py, "RECONCILE THE PD TARGETS"). The arm should then simply hold that pose. It does
    # not: the action term runs every control step and overwrites those reconciled targets from a
    # frame that is stale immediately after the teleport, which measured as a 2.8 rad swing in a
    # standalone replay of the same states.
    #
    # This was invisible while the arm actuator carried stiffness = damping = 0, because then the
    # overwritten targets drove nothing at all and the arm merely sagged. Give the actuator real
    # gains and the stale command becomes a real motion, which is why not_abnormal and stable got
    # WORSE with gains alone (32 -> 16 and 29 -> 14 of 128).
    if os.environ.get("UWLAB_DROP_ARM_ACTION"):
        for _name, _term in list(vars(env_cfg.actions).items()):
            if _term is None:
                continue
            _cls = type(_term).__name__
            if "Cartesian" in _cls or "InverseKinematics" in _cls or "OperationalSpace" in _cls:
                setattr(env_cfg.actions, _name, None)
                print(f"[record] dropped arm action term '{_name}' ({_cls})", flush=True)

    # Derive pair directory and reset type for output path
    insertive_usd_path = env_cfg.scene.insertive_object.spawn.usd_path
    receptive_usd_path = env_cfg.scene.receptive_object.spawn.usd_path
    pair = task_mdp.utils.compute_pair_dir(insertive_usd_path, receptive_usd_path)

    # Auto-infer reset_type from task name if not provided
    reset_type = args_cli.reset_type
    if reset_type is None:
        reset_type = infer_reset_type(args_cli.task)

    print(f"Recording reset states for: {pair} / {reset_type}")
    print(f"Insertive: {insertive_usd_path}")
    print(f"Receptive: {receptive_usd_path}")

    # Setup recording configuration
    output_dir = os.path.join(args_cli.dataset_dir, "Resets", pair)
    os.makedirs(output_dir, exist_ok=True)
    output_file_name = f"resets_{reset_type}.pt"

    env_cfg.recorders = task_mdp.StableStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = TorchDatasetFileHandler

    # create environment
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    env.reset()

    # Run reset state sampling
    num_reset_conditions_evaluated = 0
    current_successful_reset_conditions = 0
    # Every "*EEGrasped" reset type promises the end-effector is holding the object, so the gripper
    # must be commanded shut for all of them. This used to test the two literal names
    # "ObjectAnywhereEEGrasped"/"ObjectRestingEEGrasped", which silently excluded any other grasped
    # type -- notably "ObjectPartiallyAssembledEEGrasped", which then fell through to the random
    # open/close branch below. The collision-free success gate favours the open-jaw samples, so ~65%
    # of that dataset was recorded with fully-open jaws and was not grasping anything. Matching on
    # the "EEGrasped" substring covers present and future grasped variants.
    #
    # THE COMMAND IS NO LONGER WRITTEN INTO ``actions[:, -1]``. That index is the whole gripper for
    # a BinaryJointAction and nothing but ``rj_dg_5_4`` for the fully actuated DELTO hand, where -1
    # additionally means EXTENSION rather than closure -- so the same line that closed a parallel
    # jaw quietly recorded an open, non-grasping hand, with no shape error to catch it (the buffer
    # is sized from ``env.action_space.shape``, which stays right while the meaning goes wrong).
    # ``scripted_gripper_actions`` asks the env's own action terms what "close" means for them and
    # writes every dimension they own -- all twenty, by joint name, for the DELTO. It is recomputed
    # each step because a per-joint servo has to read the measured joint positions.
    grasped = "EEGrasped" in args_cli.task
    close_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    if not grasped:
        close_mask = torch.randint(0, 2, (env.num_envs,), device=env.device) == 1

    # Create progress bar
    pbar = tqdm(total=args_cli.num_reset_states, desc="Successful reset states", unit="reset states")

    # PIN THE ARM AT THE POSE THE RESET EVENT PLACED IT IN.
    #
    # A reset state is a STATIC initial condition: the reset event solves IK to put the end effector
    # on a sampled grasp and reconciles the PD targets to it, and the arm is then supposed to stay
    # there while the check decides whether the HAND holds the object. It does not stay: the arm
    # actuator carries stiffness = damping = 0 by design (it is meant to be torque-controlled), no
    # policy is running to supply those torques, and so the arm sags -- dragging the grasped object
    # with it. That is what the shipped banks recorded, and it is why arms sit under the table in
    # them.
    #
    # Gains are not the fix; that was measured. Swept against this same gate: 60/12 reproduces the
    # baseline but is far too soft to hold a UR10e, 2000/200 is stiff enough but unstable at
    # dt = 1/120 (not_abnormal 32 -> 16). Stiff enough to hold is unstable, soft enough to be stable
    # sags -- a PD standing in for a controller that should be supplying torques.
    #
    # Writing the arm's joint state back each control step sidesteps that entirely and matches what
    # the data is supposed to mean. The hand and the object stay fully dynamic, so the acceptance
    # check still measures exactly what it should.
    # GRAVITY COMPENSATION for the arm during recording.
    #
    # This is the missing piece the other three attempts were each working around. The arm actuator
    # is zero-gain BY DESIGN because it is meant to be torque-controlled, and during reset-state
    # recording no policy is running to supply those torques -- so the arm sags out of the pose the
    # IK solver placed it in, dragging the grasped object with it. Measured alternatives, all
    # rejected against this same gate (of 128 envs):
    #
    #     PD gains 2000/200   not_abnormal 32 -> 16   stiff enough to hold, unstable at dt = 1/120
    #     PD gains 60/12      not_abnormal 32 -> 31   stable, but far too soft to hold a UR10e
    #     kinematic pin       not_abnormal 47 -> 11   writing joint state mid-step is non-physical,
    #                                                 and the velocity conditions measure exactly
    #                                                 that discontinuity
    #
    # The torque needed is not a tuning parameter: PhysX already computes it. Feeding the
    # generalized gravity forces back as the arm's effort target holds the arm where it was placed
    # without any stiffness to oscillate and without violating the dynamics the checks read.
    grav_comp = bool(os.environ.get("UWLAB_GRAVITY_COMP"))
    if grav_comp:
        _robot = env.scene["robot"]
        _arm_ids, _arm_names = _robot.find_joints(["shoulder.*", "elbow.*", "wrist.*"])
        _arm_ids_t = torch.tensor(_arm_ids, device=env.device, dtype=torch.long)
        print(f"[record] gravity-compensating {len(_arm_ids)} arm joints: {_arm_names}", flush=True)

    pin_arm = bool(os.environ.get("UWLAB_PIN_ARM"))
    if pin_arm:
        _robot = env.scene["robot"]
        _arm_ids, _arm_names = _robot.find_joints(["shoulder.*", "elbow.*", "wrist.*"])
        _pinned = _robot.data.joint_pos[:, _arm_ids].clone()
        _zero_v = torch.zeros_like(_pinned)
        print(f"[record] pinning {len(_arm_ids)} arm joints at the reset pose: {_arm_names}",
              flush=True)

    start_time = time.time()

    while current_successful_reset_conditions < args_cli.num_reset_states:
        # Step environment (this will evaluate grasps in parallel across environments)
        actions = task_mdp.scripted_gripper_actions_for(env, close_mask)
        _, _, terminated, truncated, _ = env.step(actions)
        dones = terminated | truncated
        done_idx = torch.where(dones)[0]

        if grav_comp:
            # data.gravity_compensation_forces is g(q) in joint space: applying tau = g(q) at
            # q_dot = 0 with no external load gives static equilibrium, which is precisely the
            # "arm stays where the IK solver put it" this recording needs. Its DoF axis may carry
            # leading base DoFs, so index from the END, where the joint DoFs live.
            _g = _robot.data.gravity_compensation_forces
            _g = getattr(_g, "torch", _g)
            _g = _g[:, -_robot.num_joints:]
            _robot.set_joint_effort_target(_g[:, _arm_ids_t], joint_ids=_arm_ids)

        if pin_arm:
            # Environments that just reset carry a NEW arm pose from the reset event -- adopt it as
            # their pin. Everything else is held at the pose it was given.
            if done_idx.numel() > 0:
                _pinned[done_idx] = _robot.data.joint_pos[done_idx][:, _arm_ids]
            _robot.write_joint_state_to_sim(_pinned, _zero_v, joint_ids=_arm_ids)
            _robot.set_joint_position_target(_pinned, joint_ids=_arm_ids)

        # Re-randomise the open/closed choice for environments that are done. Same substring test
        # as above: a grasped reset type must keep the gripper closed on re-randomisation, never
        # flip back to open.
        if done_idx.numel() > 0 and not grasped:
            close_mask[done_idx] = torch.randint(0, 2, (done_idx.numel(),), device=env.device) == 1

        # Update progress based on successful reset conditions
        new_successful_count = env.recorder_manager.exported_successful_episode_count
        if new_successful_count > current_successful_reset_conditions:
            increment = new_successful_count - current_successful_reset_conditions
            current_successful_reset_conditions = new_successful_count
            pbar.update(increment)

        # Count total reset conditions evaluated (sum across all environments)
        num_reset_conditions_evaluated += dones.sum().item()

        if env.sim.is_stopped():
            break

    pbar.close()

    # Get final statistics
    final_successful_reset_conditions = env.recorder_manager.exported_successful_episode_count
    print("Reset state recording complete!")
    print(f"Total reset conditions evaluated: {num_reset_conditions_evaluated}")
    print(f"Successful reset conditions: {final_successful_reset_conditions}")
    if num_reset_conditions_evaluated > 0:
        print(f"Success rate: {final_successful_reset_conditions / num_reset_conditions_evaluated:.2%}")
        print(f"Time taken: {(time.time() - start_time) / 60:.2f} minutes")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
