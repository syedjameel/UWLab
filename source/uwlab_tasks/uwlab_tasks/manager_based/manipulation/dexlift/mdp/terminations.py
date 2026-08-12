# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the dexterous lifting tasks.

Only one term lives here, and it exists because of a specific asymmetry introduced by grafting a
twenty-joint hand onto an identified arm. See :func:`abnormal_robot_state_scoped`.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def abnormal_robot_state_scoped(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Velocity-limit violation, restricted to the joints ``asset_cfg`` selects.

    This is the upstream ``abnormal_robot_state`` -- "any joint moving at more than twice its
    velocity limit means the physics has gone unstable, end the episode" -- with one difference:
    it honours ``asset_cfg.joint_ids``. Both the dexsuite and the uwlab versions take an
    ``asset_cfg`` and then ignore its joint selection, reducing over EVERY joint of the
    articulation, so there is no way to express "watch the arm" with either of them.

    Why that matters on this robot. The instability this term is meant to catch is an arm one: a
    six-joint arm whose solver has diverged. The twenty DELTO joints are a different regime
    entirely -- they are soft, force-limited, and they snap when a fingertip breaks contact. Their
    ``velocity_limit_sim`` is 3.0 rad/s while the USD independently authors
    ``physxJoint:maxJointVelocity`` at 7.31 rad/s, so PhysX itself permits a finger to pass the
    2x threshold during perfectly healthy contact transients. Unscoped, a normal release therefore
    ends the episode AND pays the early-termination penalty, and the policy learns not to let go.

    Args:
        env: The environment.
        asset_cfg: The articulation and the joints to watch. Pass ``joint_names=`` to scope it;
            the default watches every joint, exactly like the upstream term.

    Returns:
        Boolean tensor, one entry per environment.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    joint_vel_limits = robot.data.joint_vel_limits[:, asset_cfg.joint_ids]
    return (joint_vel.abs() > (joint_vel_limits * 2)).any(dim=1)
