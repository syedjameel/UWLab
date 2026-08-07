# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Environment wrapper that publishes table-leg task diagnostics to RL loggers."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from . import mdp
from .table_leg_env_cfg import CONTACT_THRESHOLD, FINGER_CONTACT_GROUPS, FINGER_CONTACT_NAMES, SUCCESS_HEIGHT


class TableLegGraspLiftEnv(ManagerBasedRLEnv):
    """Manager-based environment with inexpensive per-step grasp/lift telemetry."""

    def step(self, action: torch.Tensor):
        result = super().step(action)
        height = mdp.root_height_above_table(self)
        contacts = mdp.logical_finger_contacts(self, FINGER_CONTACT_GROUPS, CONTACT_THRESHOLD).sum(dim=-1)
        speed = torch.linalg.vector_norm(self.scene["object"].data.root_lin_vel_w, dim=-1)
        max_force = mdp.max_finger_contact_force(self, FINGER_CONTACT_NAMES)
        valid = (height >= SUCCESS_HEIGHT) & (contacts >= 2) & (speed <= 0.5)
        log = self.extras.setdefault("log", {})
        log.update(
            {
                "Metrics/table_leg/mean_clearance_m": height.mean(),
                "Metrics/table_leg/max_clearance_m": height.max(),
                "Metrics/table_leg/mean_contact_fingers": contacts.float().mean(),
                "Metrics/table_leg/two_finger_contact_fraction": (contacts >= 2).float().mean(),
                "Metrics/table_leg/clearance_fraction": (height >= SUCCESS_HEIGHT).float().mean(),
                "Metrics/table_leg/held_lift_candidate_fraction": valid.float().mean(),
                "Metrics/table_leg/mean_object_speed_mps": speed.mean(),
                "Metrics/table_leg/mean_max_contact_force_n": max_force.clamp(max=100.0).mean(),
            }
        )
        return result
