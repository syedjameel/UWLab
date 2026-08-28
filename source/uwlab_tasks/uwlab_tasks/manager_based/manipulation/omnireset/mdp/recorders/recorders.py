# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers.recorder_manager import RecorderTerm

from ..reset_state_schema import add_joint_targets


class StableStateRecorder(RecorderTerm):
    def record_pre_reset(self, env_ids):
        def extract_env_ids_values(value):
            nonlocal env_ids
            if isinstance(value, dict):
                return {k: extract_env_ids_values(v) for k, v in value.items()}
            return value[env_ids]

        state = self._env.scene.get_state(is_relative=True)
        # -- PD SET POINT (bead UWLab-algw.7). ``scene.get_state()`` above carries only the
        # CONFIGURATION (root_pose/root_velocity/joint_position/joint_velocity) -- not what the PD
        # actuator was actually commanded to hold, which is what determines applied grip force on
        # replay. Read directly off each live Articulation's OWN data buffer -- the un-indexed,
        # full-batch ``joint_pos_target``/``joint_vel_target`` are exactly what the last
        # set_joint_position_target/set_joint_velocity_target call (or the actuator's own init
        # value, if neither was ever called) wrote, not a re-derivation of it. Added BEFORE
        # extract_env_ids_values runs, so the same recursion indexes these two new leaves by
        # env_ids identically to every other field -- including the multi-env-per-call case (this
        # recorder's own record_pre_reset can bundle more than one accepted state under one call).
        for name, articulation in self._env.scene._articulations.items():
            add_joint_targets(
                state["articulation"][name],
                articulation.data.joint_pos_target,
                articulation.data.joint_vel_target,
            )

        return "initial_state", extract_env_ids_values(state)


class GraspRelativePoseRecorder(RecorderTerm):
    """Recorder term that records relative position, orientation, and gripper joint states for grasp evaluation."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Configuration for which robot and object to track
        self.robot_name = cfg.robot_name
        self.object_name = cfg.object_name
        self.gripper_body_name = cfg.gripper_body_name

    def record_pre_reset(self, env_ids):
        """Record relative pose between object and gripper, plus gripper joint states before reset."""
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self._env.device)

        # Get robot articulation and object rigid body
        robot = self._env.scene[self.robot_name]
        obj = self._env.scene[self.object_name]

        # Get object pose (root pose contains position and orientation)
        obj_root_state = obj.data.root_state_w[env_ids]  # Shape: (num_envs, 13) - pos(3) + quat(4) + vel(6)
        obj_pos = obj_root_state[:, :3]  # Position
        obj_quat = obj_root_state[:, 3:7]  # Quaternion (w, x, y, z)

        # Get gripper body pose from the robot articulation
        # Find the gripper body index
        gripper_body_idx = None
        for idx, body_name in enumerate(robot.body_names):
            if self.gripper_body_name in body_name:
                gripper_body_idx = idx
                break
        if gripper_body_idx is None:
            # Without this the ``None`` index does not raise -- ``body_state_w[ids, None, :3]``
            # inserts an axis and slices the BODY dimension, so the run dies much later inside
            # subtract_frame_transforms with a tensor-size error that never names the body. This
            # can only fire where the configuration was already wrong (e.g. the 2F-85's
            # ``robotiq_base_link`` against a gripper whose palm is named something else).
            raise ValueError(
                f"gripper_body_name {self.gripper_body_name!r} matches no body on articulation "
                f"{self.robot_name!r}; available bodies: {list(robot.body_names)}"
            )

        # Get specific body pose
        gripper_pos = robot.data.body_state_w[env_ids, gripper_body_idx, :3]
        gripper_quat = robot.data.body_state_w[env_ids, gripper_body_idx, 3:7]

        # Calculate relative transform: T_gripper_in_object = T_object^{-1} * T_gripper
        relative_pos, relative_quat = math_utils.subtract_frame_transforms(obj_pos, obj_quat, gripper_pos, gripper_quat)

        # Get gripper joint states as dict mapping joint names to positions
        gripper_joint_pos = robot.data.joint_pos[env_ids].clone()
        gripper_joint_dict = {joint_name: gripper_joint_pos[:, i] for i, joint_name in enumerate(robot.joint_names)}

        # Prepare data to record
        grasp_data = {
            "relative_position": relative_pos,
            "relative_orientation": relative_quat,
            "gripper_joint_positions": gripper_joint_dict,
        }

        return "grasp_relative_pose", grasp_data


class PreStepDataCollectionObservationsRecorder(RecorderTerm):
    """Recorder term that records data collection observations from the data_collection observation group."""

    def record_pre_step(self):
        """Record data collection observations from the data_collection observation group."""
        return "obs", self._env.obs_buf["data_collection"]
