# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actions for the UR5e + linear gripper robot.

The arm actions are identical to the 2F-85 robot's (same arm, same IK body
``robotiq_base_link``), so they are reused. Only the gripper action differs: a binary
open/close in METERS (prismatic), 0.0 = OPEN, 0.068 = CLOSED.

Two gripper-action variants, because the jaw coupling differs by context (see
``ur5e_linear_gripper.py``):

* :obj:`LINEAR_GRIPPER_BINARY_ACTIONS` -- STANDALONE gripper (grasp sampling). Commands ONLY
  ``finger_joint``; the PhysX prismatic mimic makes ``right_finger_joint`` follow (it works
  there because the finger joints are the articulation root DOFs).
* :obj:`LINEAR_GRIPPER_DUAL_BINARY_ACTIONS` -- FULL ROBOT (reset/RL). Commands BOTH jaws to the
  same target, because the prismatic mimic is inert once the gripper is embedded in the full
  arm articulation (the graft strips it and drives both jaws instead). Still ONE binary command,
  so the follower is slaved, not an independent policy DOF.
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

# STANDALONE (grasp sampling): drive the driver only; the mimic follows.
LINEAR_GRIPPER_BINARY_ACTIONS = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["finger_joint"],
    open_command_expr={"finger_joint": 0.0},      # jaws fully open
    close_command_expr={"finger_joint": 0.068},   # jaws fully closed (meters)
)

# FULL ROBOT (reset/RL): drive BOTH jaws to the same target (mimic is inert in the full
# articulation). gearing was -1 with right_finger_joint framed so q_right = q_finger gives
# symmetric closure -> both jaws go to +0.068 closed (verified |follower-driver|=0.0000).
LINEAR_GRIPPER_DUAL_BINARY_ACTIONS = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["finger_joint", "right_finger_joint"],
    open_command_expr={"finger_joint": 0.0, "right_finger_joint": 0.0},
    close_command_expr={"finger_joint": 0.068, "right_finger_joint": 0.068},
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


@configclass
class LinearGripperDualBinaryGripperAction:
    # Grasp sampling: dual-drive both jaws (the prismatic mimic pins the gripper at 0 otherwise).
    gripper = LINEAR_GRIPPER_DUAL_BINARY_ACTIONS
