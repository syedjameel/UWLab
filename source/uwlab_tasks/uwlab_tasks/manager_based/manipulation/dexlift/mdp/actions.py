# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific action terms for dexterous lifting."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import BinaryJointPositionAction, BinaryJointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class ContinuousSynergyJointPositionAction(BinaryJointPositionAction):
    """Interpolate one scalar continuously between calibrated open and close postures.

    The sign convention matches Isaac Lab's binary gripper action: zero and
    positive values are open, while ``-1`` is fully closed. Intermediate
    negative values provide the gentle closure required by the long table leg.
    """

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        close_fraction = (-actions).clamp(0.0, 1.0)
        self._processed_actions = self._open_command + close_fraction * (
            self._close_command - self._open_command
        )


@configclass
class ContinuousSynergyJointPositionActionCfg(BinaryJointPositionActionCfg):
    """Configuration for a one-dimensional continuous joint-position synergy."""

    class_type: type[ActionTerm] = ContinuousSynergyJointPositionAction
