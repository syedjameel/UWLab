# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
MDP terminations.
"""


def invalid_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return true if the RigidBody position reads nan"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.isnan(asset.data.body_pos_w).any(dim=-1).any(dim=-1)


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # Restrict the |vel| > 2x limit check to ``asset_cfg.joint_ids`` (default slice(None) = all
    # joints, so existing callers are unchanged). This lets a caller exclude joints whose velocity
    # limit is deliberately clamped low (e.g. a real-jaw speed cap), where 2x the clamp is grazed
    # by benign drive/noise overshoot and would fire a spurious termination -- the check is meant
    # to catch runaway ARM dynamics, not a force-/speed-limited gripper.
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    return (
        robot.data.joint_vel[:, joint_ids].abs() > (robot.data.joint_vel_limits[:, joint_ids] * 2)
    ).any(dim=1)
