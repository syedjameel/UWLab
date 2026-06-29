# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actions for the UR5e + linear gripper robot.

The arm actions are identical to the 2F-85 robot's (same arm, same IK body
``robotiq_base_link``), so they are reused. Only the gripper action differs: a binary
open/close on the single driver joint ``finger_joint`` -- but in METERS (prismatic),
0.0 = OPEN, 0.068 = CLOSED. The PhysX mimic makes ``right_finger_joint`` follow, so only
the driver joint is commanded (exactly like the 2F-85 commands only ``finger_joint``).
"""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.utils import configclass

# Reuse the arm action cfgs (identical arm + IK body) from the 2F-85 robot.
from uwlab_assets.robots.ur5e_robotiq_gripper.actions import (
    UR5E_JOINT_POSITION,
    UR5E_MC_IKABSOLUTE_ARM,
    UR5E_MC_IKDELTA_ARM,
    UR5E_RELATIVE_JOINT_POSITION,
)

LINEAR_GRIPPER_BINARY_ACTIONS = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["finger_joint"],
    open_command_expr={"finger_joint": 0.0},      # jaws fully open
    close_command_expr={"finger_joint": 0.068},   # jaws fully closed (meters)
)


@configclass
class Ur5eLinearGripperIkAbsoluteAction:
    arm = UR5E_MC_IKABSOLUTE_ARM
    gripper = LINEAR_GRIPPER_BINARY_ACTIONS


@configclass
class Ur5eLinearGripperMcIkDeltaAction:
    arm = UR5E_MC_IKDELTA_ARM
    gripper = LINEAR_GRIPPER_BINARY_ACTIONS


@configclass
class Ur5eLinearGripperJointPositionAction:
    arm = UR5E_JOINT_POSITION
    gripper = LINEAR_GRIPPER_BINARY_ACTIONS


@configclass
class Ur5eLinearGripperRelativeJointPositionAction:
    arm = UR5E_RELATIVE_JOINT_POSITION
    gripper = LINEAR_GRIPPER_BINARY_ACTIONS


@configclass
class LinearGripperBinaryGripperAction:
    gripper = LINEAR_GRIPPER_BINARY_ACTIONS
