# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from uwlab_assets.robots.ur5e_linear_gripper.actions import LINEAR_GRIPPER_DUAL_BINARY_ACTIONS
from uwlab_assets.robots.ur5e_robotiq_gripper.actions import ROBOTIQ_GRIPPER_BINARY_ACTIONS

from ...mdp.actions.actions_cfg import RelCartesianOSCActionCfg

# Pre-train gains (soft initial Kp; curriculum ramps to stiff terminal)
UR5E_ROBOTIQ_2F85_RELATIVE_OSC = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="wrist_3_link",
    scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    motion_stiffness=(200.0, 200.0, 200.0, 3.0, 3.0, 3.0),
    motion_damping_ratio=(3.0, 3.0, 3.0, 1.0, 1.0, 1.0),
    torque_limit=(150.0, 150.0, 150.0, 28.0, 28.0, 28.0),
)

# Eval / sim2real gains (high Kp matched to sysid friction, end-of-curriculum values)
UR5E_ROBOTIQ_2F85_RELATIVE_OSC_EVAL = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="wrist_3_link",
    scale_xyz_axisangle=(0.01, 0.01, 0.002, 0.02, 0.02, 0.2),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    torque_limit=(150.0, 150.0, 150.0, 28.0, 28.0, 28.0),
)

# Unscaled (for sysid scripts)
UR5E_ROBOTIQ_2F85_RELATIVE_OSC_UNSCALED = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="wrist_3_link",
    scale_xyz_axisangle=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    torque_limit=(150.0, 150.0, 150.0, 28.0, 28.0, 28.0),
)


@configclass
class Ur5eRobotiq2f85RelativeOSCAction:
    """Action config using the analytical OSC + binary gripper."""

    arm = UR5E_ROBOTIQ_2F85_RELATIVE_OSC
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS


@configclass
class Ur5eRobotiq2f85RelativeOSCEvalAction:
    """Action config with high Kp gains (end-of-curriculum values) for eval / data-collection."""

    arm = UR5E_ROBOTIQ_2F85_RELATIVE_OSC_EVAL
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS


@configclass
class Ur5eRobotiq2f85SysidOSCAction:
    """Unscaled arm action (Cartesian delta) + binary gripper. For Sysid env / scripts."""

    arm = UR5E_ROBOTIQ_2F85_RELATIVE_OSC_UNSCALED
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS


# Linear-gripper action variants: IDENTICAL arm OSC (same UR5e arm + wrist_3_link IK body),
# only the binary gripper sub-action differs. These run on the FULL ROBOT (reset/RL), where the
# prismatic mimic is inert, so they use LINEAR_GRIPPER_DUAL_BINARY_ACTIONS (commands BOTH jaws to
# the same target in meters). Grasp sampling (standalone) is ALSO dual-driven now (the prismatic
# mimic pins the gripper at 0), so it uses the dual action too -- see linear_gripper_cfg.py.
@configclass
class Ur5eLinearGripperRelativeOSCAction:
    """Pre-train / train gains: analytical OSC + linear-gripper dual-jaw binary action."""

    arm = UR5E_ROBOTIQ_2F85_RELATIVE_OSC
    gripper = LINEAR_GRIPPER_DUAL_BINARY_ACTIONS


@configclass
class Ur5eLinearGripperRelativeOSCEvalAction:
    """Eval / sim2real gains: high-Kp OSC + linear-gripper dual-jaw binary action."""

    arm = UR5E_ROBOTIQ_2F85_RELATIVE_OSC_EVAL
    gripper = LINEAR_GRIPPER_DUAL_BINARY_ACTIONS
