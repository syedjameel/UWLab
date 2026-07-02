# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR
from uwlab_assets.robots.ur5e_linear_gripper.actions import LINEAR_GRIPPER_BINARY_ACTIONS
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


# Linear-gripper train OSC: same as the 2F-85 train OSC BUT rotational damping_ratio 1.0 -> 0.2
# (rotational stiffness Kp stays 3.0, like the 2F-85).
#
# ROOT CAUSE of the "vibration": the OSC is a mass-less Cartesian PD (task_space_actions: "No mass
# matrix") with kd = 2*sqrt(kp)*damping_ratio. Our gripper (stiff dual-drive jaws + large inertia)
# produces NOISY wrist velocities; the derivative (kd) term AMPLIFIES that velocity noise into arm
# jitter -> visible vibration. It scales with kd, NOT with under-damping: a sweep showed jitter
# 0.02 -> 59 -> 487 mrad/s as rot damping went 1 -> ... -> 8 at Kp 3 (more damping = WORSE). The
# compact/soft 2F-85 gripper has clean velocities, so it is fine at the stock gains.
#
# The earlier "fix" (rot Kp -> 0.1) killed the jitter only by killing kd, but it ALSO killed the
# rotational STIFFNESS, so the OSC lost orientation authority and the policy could only grasp at
# whatever angle it drifted into (weird, non-top-down grasps). The correct fix keeps firm rot Kp=3
# (control) but LOW damping_ratio 0.2 -> kd~0.69 (same low kd as the steady Kp=0.1 case) -> ZERO
# vibration AND more orientation authority than the 2F-85's stock gains (validated: jitter 0.00
# mrad/s, +33% control vs Kp3/dr1). Use rot damping ~0.15 with a higher Kp (e.g. 10) if even more
# top-down authority is wanted. Requires RE-TRAINING (the old policy learned the weird angles).
UR5E_LINEAR_GRIPPER_RELATIVE_OSC = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="wrist_3_link",
    scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    motion_stiffness=(200.0, 200.0, 200.0, 3.0, 3.0, 3.0),
    motion_damping_ratio=(3.0, 3.0, 3.0, 0.2, 0.2, 0.2),
    torque_limit=(150.0, 150.0, 150.0, 28.0, 28.0, 28.0),
)


@configclass
class Ur5eLinearGripperRelativeOSCAction:
    """Pre-train / train gains: analytical OSC (soft rotation, no vibration) + linear-gripper binary."""

    arm = UR5E_LINEAR_GRIPPER_RELATIVE_OSC
    gripper = LINEAR_GRIPPER_BINARY_ACTIONS


@configclass
class Ur5eLinearGripperRelativeOSCEvalAction:
    """Eval / sim2real gains: high-Kp OSC + linear-gripper binary action.

    NOTE: this still uses the 2F-85 EVAL gains (rotational Kp=50), which will vibrate on our large
    gripper the same way the train Kp=3 did (only the train task has been validated so far). When
    you get to eval/finetune, either drop its rotational Kp the same way, or better, address the
    root cause by shrinking the gripper's rotational inertia in the full-robot USD so the stiff
    eval gains stay precise AND stable.
    """

    arm = UR5E_ROBOTIQ_2F85_RELATIVE_OSC_EVAL
    gripper = LINEAR_GRIPPER_BINARY_ACTIONS


# ---------------------------------------------------------------------------------------
# UR10e + linear gripper
# ---------------------------------------------------------------------------------------
# The OSC's analytical kinematics/mass-matrix are data-driven: calibration_dir points them at
# the UR10e's calibrated_joints/link_inertials (nominal from the URDF, FK cross-checked to
# <0.5 mm against the Isaac articulation). Left as None they would silently use the UR5e's.
_UR10E_CALIBRATION_DIR = f"{UWLAB_LOCAL_ASSETS_DIR}/Robots/UR10e"

# UR10e torque limits (URDF): the UR10e is a much stronger arm than the UR5e (150/28).
_UR10E_TORQUE_LIMIT = (330.0, 330.0, 150.0, 56.0, 56.0, 56.0)

# Train OSC: same gains as the UR5e linear-gripper train OSC (rot Kp 3, damping_ratio 0.2).
#
# Rot Kp MUST stay at 3: raising it to 6 (to close the UR10e's free-space rotational
# authority gap, ~2/3 of the UR5e's at Kp 3) made the wrist UNSTABLE IN CONTACT -- the
# moment the stiff dual-drive jaws (1500 N/m) close on an object, the mass-less rotational
# PD enters a high-frequency limit cycle: wrist_3 velocity swings at its +/-pi limit while
# the held object chatters at ~3.5 rad/s, so EVERY EEGrasped reset fails the stability
# check (~250x slower recording). At Kp 3 the same close is dead quiet (dq ~0.003). The
# instability is purely the gain: verified by A/B with only Kp changed; the UR10e analytical
# Jacobian matches a finite-difference of the sim to 3 decimals on every column. Lesson: any
# OSC gain change must be validated IN CONTACT (jaws closed on an object), not just with
# free-space hold-jitter/authority probes.
UR10E_LINEAR_GRIPPER_RELATIVE_OSC = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="wrist_3_link",
    scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    motion_stiffness=(200.0, 200.0, 200.0, 3.0, 3.0, 3.0),
    motion_damping_ratio=(3.0, 3.0, 3.0, 0.2, 0.2, 0.2),
    torque_limit=_UR10E_TORQUE_LIMIT,
    calibration_dir=_UR10E_CALIBRATION_DIR,
)

# Eval / sim2real gains (mirrors the 2F-85 eval OSC; same rotational-vibration caveat as the
# UR5e linear-gripper eval action -- revisit the rot gains when eval/finetune is reached).
UR10E_LINEAR_GRIPPER_RELATIVE_OSC_EVAL = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="wrist_3_link",
    scale_xyz_axisangle=(0.01, 0.01, 0.002, 0.02, 0.02, 0.2),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    torque_limit=_UR10E_TORQUE_LIMIT,
    calibration_dir=_UR10E_CALIBRATION_DIR,
)


@configclass
class Ur10eLinearGripperRelativeOSCAction:
    """Pre-train / train gains: UR10e analytical OSC + linear-gripper binary action."""

    arm = UR10E_LINEAR_GRIPPER_RELATIVE_OSC
    gripper = LINEAR_GRIPPER_BINARY_ACTIONS


@configclass
class Ur10eLinearGripperRelativeOSCEvalAction:
    """Eval / sim2real gains: high-Kp UR10e OSC + linear-gripper binary action."""

    arm = UR10E_LINEAR_GRIPPER_RELATIVE_OSC_EVAL
    gripper = LINEAR_GRIPPER_BINARY_ACTIONS
