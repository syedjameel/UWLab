# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Environment wrapper that publishes table-leg task diagnostics to RL loggers."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from . import mdp
from .table_leg_env_cfg import (
    CONTACT_THRESHOLD,
    FINGER_CONTACT_GROUPS,
    FINGER_CONTACT_NAMES,
    FINGER_JOINT_GROUPS,
    MAX_SUCCESS_OBJECT_SPEED,
    MINIMUM_ARTICULATED_FINGERS,
    MINIMUM_JOINT_DISPLACEMENT,
    NON_FINGER_HAND_CONTACT_NAMES,
    TARGET_POSITION_TOLERANCE,
    THUMB_CONTACT_NAMES,
    TIP_CONTACT_NAMES,
)


class TableLegGraspLiftEnv(ManagerBasedRLEnv):
    """Manager-based environment with inexpensive per-step grasp/lift telemetry."""

    def step(self, action: torch.Tensor):
        result = super().step(action)
        height = mdp.root_height_above_table(self)
        contacts = mdp.logical_finger_contacts(self, FINGER_CONTACT_GROUPS, CONTACT_THRESHOLD).sum(dim=-1)
        articulated = mdp.articulated_finger_count(
            self, FINGER_JOINT_GROUPS, MINIMUM_JOINT_DISPLACEMENT
        )
        target_error = mdp.target_position_error(self, "object_pose")
        speed = torch.linalg.vector_norm(self.scene["object"].data.root_lin_vel_w, dim=-1)
        max_force = mdp.max_finger_contact_force(self, FINGER_CONTACT_NAMES)
        unwanted_force = mdp.max_finger_contact_force(self, NON_FINGER_HAND_CONTACT_NAMES)
        opposed_contact = mdp.geometric_opposed_contacts(
            self,
            CONTACT_THRESHOLD,
            THUMB_CONTACT_NAMES,
            TIP_CONTACT_NAMES,
            -0.1,
            NON_FINGER_HAND_CONTACT_NAMES,
            CONTACT_THRESHOLD,
        )
        if not hasattr(self, "_table_leg_first_contact_step"):
            self._table_leg_first_contact_step = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            )
        reset = self.episode_length_buf == 0
        self._table_leg_first_contact_step[reset] = -1
        any_hand_contact = (contacts > 0) | (unwanted_force > CONTACT_THRESHOLD)
        new_contact = any_hand_contact & (self._table_leg_first_contact_step < 0)
        self._table_leg_first_contact_step[new_contact] = self.episode_length_buf[new_contact]
        observed_contact = self._table_leg_first_contact_step >= 0
        ordered_contact = self._table_leg_first_contact_step >= 30
        valid = (
            (target_error <= TARGET_POSITION_TOLERANCE)
            & (contacts >= 2)
            & opposed_contact
            & (articulated >= MINIMUM_ARTICULATED_FINGERS)
            & (speed <= MAX_SUCCESS_OBJECT_SPEED)
            & (unwanted_force <= CONTACT_THRESHOLD)
        )
        log = self.extras.setdefault("log", {})
        log.update(
            {
                "Metrics/table_leg/mean_clearance_m": height.mean(),
                "Metrics/table_leg/max_clearance_m": height.max(),
                "Metrics/table_leg/mean_contact_fingers": contacts.float().mean(),
                "Metrics/table_leg/two_finger_contact_fraction": (contacts >= 2).float().mean(),
                "Metrics/table_leg/mean_articulated_fingers": articulated.float().mean(),
                "Metrics/table_leg/mean_target_position_error_m": target_error.mean(),
                "Metrics/table_leg/target_position_fraction": (
                    target_error <= TARGET_POSITION_TOLERANCE
                ).float().mean(),
                "Metrics/table_leg/strict_hold_candidate_fraction": valid.float().mean(),
                "Metrics/table_leg/mean_object_speed_mps": speed.mean(),
                "Metrics/table_leg/mean_max_contact_force_n": max_force.clamp(max=100.0).mean(),
                "Metrics/table_leg/mean_nonfinger_force_n": unwanted_force.clamp(max=100.0).mean(),
                "Metrics/table_leg/reset_overlap_fraction": (
                    any_hand_contact & (self.episode_length_buf <= 1)
                ).float().mean(),
                "Metrics/table_leg/ordered_first_contact_fraction": (
                    ordered_contact & observed_contact
                ).float().sum() / observed_contact.float().sum().clamp(min=1.0),
            }
        )
        return result
