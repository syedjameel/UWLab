# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR
from uwlab_assets.robots.ur5e_linear_gripper.actions import LINEAR_GRIPPER_BINARY_ACTIONS
from uwlab_assets.robots.ur5e_robotiq_gripper.actions import ROBOTIQ_GRIPPER_BINARY_ACTIONS
from uwlab_assets.robots.ur5e_robotiq_gripper.kinematics import ARM_JOINT_NAMES
from uwlab_assets.robots.ur5e_robotiq_gripper.ur5e_robotiq_2f85_gripper import UR5E_EFFORT_LIMITS
from uwlab_assets.robots.ur10e_delto.actions import DELTO_FULL_HAND_ACTIONS

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

# Train OSC: UR5e linear-gripper train gains (rot Kp 3) with rot damping_ratio 0.1
# (the UR5e ran 0.2; the 2026-07-10 note below explains why the lighter gripper halves it).
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
#
# 2026-07-10: the wrist_3 velocity-limit "runaway" during reset recording was TWO stacked
# defects, both diagnosed by per-step torque/velocity traces:
#   (1) the reset events' differential IK is joint-limit-unaware and wrote wrist solutions
#       BEYOND the graft's +-180 deg limits (ee915f8 hardening; the pre-hardening +-360
#       graft made the same solutions legal, which is why the 07-03 datasets were clean).
#       PhysX then resolves the violated limit DURING the episode -- dragging the joint at
#       the velocity cap with ZERO applied torque. Fixed at the source: the events wrap
#       the final IK joints into the limits (events._wrap_joints_into_limits; a full-turn
#       wrap is physically exact).
#   (2) the rotational kd term is EXPLICIT velocity feedback at 120 Hz: stability needs
#       kd*dt/I < 2. The graft's gripper-mass fix (d1cc46b, 1.1 -> real 0.575 kg) halved
#       the wrist_3 reflected inertia (I ~ 0.002), putting the UR5e-lineage dr 0.2
#       (kd 0.69) at ~2.7 -> alternating-sign divergence growing x1.7/step from noise.
#       dr 0.1 (kd 0.35) restores kd*dt/I ~ 1.4 -- the same margin the UR5e had at dr 0.2
#       with its 1.1 kg gripper (0.69/(0.004*120) ~ 1.4). kd scales with inertia; the mass
#       halving halves dr.
UR10E_LINEAR_GRIPPER_RELATIVE_OSC = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="wrist_3_link",
    scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    motion_stiffness=(200.0, 200.0, 200.0, 3.0, 3.0, 3.0),
    motion_damping_ratio=(3.0, 3.0, 3.0, 0.1, 0.1, 0.1),
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


# Unscaled (for sysid scripts) -- mirrors UR5E_ROBOTIQ_2F85_RELATIVE_OSC_UNSCALED with the
# UR10e torque limits + calibration.
UR10E_LINEAR_GRIPPER_RELATIVE_OSC_UNSCALED = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="wrist_3_link",
    scale_xyz_axisangle=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    torque_limit=_UR10E_TORQUE_LIMIT,
    calibration_dir=_UR10E_CALIBRATION_DIR,
)


@configclass
class Ur10eLinearGripperSysidOSCAction:
    """Unscaled arm action (Cartesian delta) + binary gripper. For the UR10e Sysid env."""

    arm = UR10E_LINEAR_GRIPPER_RELATIVE_OSC_UNSCALED
    gripper = LINEAR_GRIPPER_BINARY_ACTIONS


# ---------------------------------------------------------------------------------------
# UR10e + Tesollo DELTO DG-5F hand
# ---------------------------------------------------------------------------------------
# SAME ARM, SAME EVERYTHING, except the rotational damping ratio. The arm is the identical
# calibrated UR10e, the IK body is still ``wrist_3_link``, and ``calibration_dir`` still points at
# ``Robots/UR10e`` -- the analytical kinematics/mass matrix are an ARM property and the end
# effector does not enter them. Translational gains, scales and torque limits are the linear
# gripper's verbatim.
#
# WHAT THE HAND COULD HAVE CHANGED: the rotational damping ratio. The OSC is a mass-less Cartesian
# PD (``task_space_actions``: "No mass matrix") with ``kd = 2*sqrt(kp)*damping_ratio``, so kd is
# EXPLICIT velocity feedback at the env rate and stability needs ``kd*dt/I < 2``, where I is the
# wrist's reflected inertia -- an END-EFFECTOR property. That is why the linear gripper's own
# number moved (UR5e, 1.1 kg -> 0.2; UR10e, 0.575 kg -> 0.1, halved when the gripper mass halved).
#
# IT DOES NOT CHANGE HERE. Named rather than inlined so that stays a deliberate statement rather
# than an oversight. The DELTO's measured wrist_3 reflected inertia is
# directly instead of scaling by mass: 2.66e-3 to 4.67e-3 kg*m^2 depending on posture, against the
# 1.1 kg linear gripper's 3.97e-3. So kd*dt/I lands at 0.62-1.09 against the bound of 2 -- inside
# the margin the shipped grippers run at, at the linear gripper's 0.1.
#
# The mass-scaled guess this line first carried (0.1 * 1.685/0.575 = 0.29) was wrong, and wrong in
# the unsafe direction. Mass is a bad proxy for reflected inertia: the DELTO is 2.9x heavier than
# the UR10e's jaw but its mass is spread over five fingers close to the flange rather than
# concentrated in a jaw out at the end, so the inertia barely moves.
#
# NON-NEGOTIABLE if this is ever revised: validate IN CONTACT, with the hand closed on an object.
# The UR10e linear gripper's rot Kp 3 -> 6 attempt passed every free-space jitter and authority
# probe and then entered a high-frequency limit cycle the moment the jaws touched an object,
# failing every EEGrasped reset. Free-space probes do not detect this class of instability.
_UR10E_DELTO_ROT_DAMPING_RATIO = 0.1

UR10E_DELTO_RELATIVE_OSC = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="wrist_3_link",
    scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    motion_stiffness=(200.0, 200.0, 200.0, 3.0, 3.0, 3.0),
    motion_damping_ratio=(3.0, 3.0, 3.0) + (_UR10E_DELTO_ROT_DAMPING_RATIO,) * 3,
    torque_limit=_UR10E_TORQUE_LIMIT,
    calibration_dir=_UR10E_CALIBRATION_DIR,
)

# Eval / sim2real gains. Mirrors UR10E_LINEAR_GRIPPER_RELATIVE_OSC_EVAL, INCLUDING its rotational
# damping ratio of 1.0: the eval OSC's rotational Kp is 50, not 3, and the two gains were fitted as
# a pair -- rescaling only the damping of a stiffer controller is not the same correction. The
# linear gripper's own caveat therefore applies here too and is, if anything, larger on a heavier
# hand: revisit the eval rotational gains before trusting eval/finetune numbers.
UR10E_DELTO_RELATIVE_OSC_EVAL = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="wrist_3_link",
    scale_xyz_axisangle=(0.01, 0.01, 0.002, 0.02, 0.02, 0.2),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    torque_limit=_UR10E_TORQUE_LIMIT,
    calibration_dir=_UR10E_CALIBRATION_DIR,
)

# Unscaled (for sysid scripts / camera alignment) -- mirrors the linear gripper's.
UR10E_DELTO_RELATIVE_OSC_UNSCALED = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="wrist_3_link",
    scale_xyz_axisangle=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    torque_limit=_UR10E_TORQUE_LIMIT,
    calibration_dir=_UR10E_CALIBRATION_DIR,
)


# The three groups below pair an UNCHANGED 6-D Cartesian OSC arm half with the fully actuated
# hand. Every hand joint is an independent policy action, so the group's action dimension is
# 6 + 20 = 26, not the 6 + 1 = 7 of the deleted one-scalar closure. Any checkpoint, ONNX export or
# recorded dataset produced against the 7-D groups is incompatible and must be regenerated.


@configclass
class Ur10eDeltoRelativeOSCAction:
    """Pre-train / train gains: UR10e analytical OSC + fully actuated DELTO hand. Dim 6 + 20 = 26."""

    arm = UR10E_DELTO_RELATIVE_OSC
    gripper = DELTO_FULL_HAND_ACTIONS


@configclass
class Ur10eDeltoRelativeOSCEvalAction:
    """Eval / sim2real gains: high-Kp UR10e OSC + fully actuated DELTO hand. Dim 6 + 20 = 26."""

    arm = UR10E_DELTO_RELATIVE_OSC_EVAL
    gripper = DELTO_FULL_HAND_ACTIONS


@configclass
class Ur10eDeltoSysidOSCAction:
    """Unscaled arm action (Cartesian delta) + fully actuated DELTO hand. Dim 6 + 20 = 26."""

    arm = UR10E_DELTO_RELATIVE_OSC_UNSCALED
    gripper = DELTO_FULL_HAND_ACTIONS


# ---------------------------------------------------------------------------------------
# UR5e + Tesollo DELTO DG-5F
# ---------------------------------------------------------------------------------------
# The UR5e half is the calibrated UR5e's, verbatim: scales, translational gains and torque limits
# are the 2F-85 pre-train action's. Only the ROTATIONAL DAMPING RATIO is re-derived, because it is
# the one gain in this controller that is a property of the END EFFECTOR. See below.

# The analytical kinematics read ``metadata.yaml`` from this directory. Pointing them at the
# GRAFT's own asset dir rather than leaving ``calibration_dir=None`` (which falls back to the
# cloud 2F-85 UR5e asset) is deliberate and is NOT a change of numbers: the ``calibrated_joints``
# and ``link_inertials`` blocks in ``Robots/Ur5eDelto/metadata.yaml`` were compared against the
# cloud UR5e's and are identical, key for key. Two reasons to name it anyway. (1) The fallback
# resolves over the network on every process start; this file is committed, so the env spawns on a
# machine with no route to HuggingFace -- the failure mode that already cost this project a
# training box. (2) It makes the arm identification a stated property of THIS robot: if the graft's
# arm is ever re-identified, one file changes and this action follows it, instead of silently
# continuing to describe a different robot's arm.
_UR5E_DELTO_CALIBRATION_DIR = f"{UWLAB_LOCAL_ASSETS_DIR}/Robots/Ur5eDelto"

# (150, 150, 150, 28, 28, 28) N*m -- the UR5e's own effort limits, read out of the articulation
# config IN THE ORDER THE ANALYTICAL JACOBIAN USES. Spelled as a lookup rather than as a literal so
# that a renamed arm joint raises a KeyError at import instead of silently shifting a limit onto
# the wrong axis. The graft did not touch the arm, so these are the same limits the 2F-85 UR5e
# actions carry as literals a few hundred lines above.
_UR5E_TORQUE_LIMIT = tuple(UR5E_EFFORT_LIMITS[name] for name in ARM_JOINT_NAMES)

# THE ONE NUMBER THAT IS NOT INHERITED.
#
# This controller is a MASS-LESS Cartesian PD (``task_space_actions``: "No inertial dynamics
# decoupling, no mass matrix"). Its damping term ``kd = 2*sqrt(kp)*damping_ratio`` is therefore
# EXPLICIT velocity feedback evaluated once per PHYSICS step -- ``apply_actions`` runs at
# ``sim.dt``, not at the policy rate. Linearising the rotational channel about the wrist axis gives
# ``I * dw/dt = -kd * w``, whose explicit-Euler update is ``w <- w * (1 - kd*dt/I)``: the loop
# diverges, with alternating sign, as soon as ``kd*dt/I > 2``. ``I`` there is the rotational
# inertia of everything DISTAL to the wrist about the wrist axis -- an end-effector property. That
# is the entire reason this ratio has been re-derived for every gripper on this arm family
# (2F-85 1.0, linear/UR5e 0.2, linear/UR10e 0.1) and why none of those numbers may be copied here.
#
# I FOR THIS ROBOT, computed from ``Robots/Ur5eDelto/ur5e_delto.usd`` itself (pxr only, no
# simulator): the composite rigid-body inertia of ``wrist_3_link`` plus all 28 hand bodies, about
# the ``wrist_3_joint`` axis. That joint authors ``axis = Z``, ``localPos1 = (0,0,0)`` and
# ``localRot1 = identity``, so the axis IS wrist_3_link's own +Z through its origin, and the hand's
# MountJoint sits at that same origin with identity rotation. Per body: rotate the authored
# ``diagonalInertia`` by its ``principalAxes`` quaternion into the wrist frame, then parallel-axis
# it out to the wrist origin. Distal mass 1.9614 kg (1.7735 kg hand + 0.1879 kg wrist_3_link).
#
#   posture                                     I about the wrist axis
#   open (the articulation's reset posture)     2.757e-3 kg*m^2
#   the validated scripted close                4.376e-3 kg*m^2
#   4000 postures drawn uniformly in the USD
#     joint limits                              min 2.841e-3, median 4.000e-3, max 5.322e-3
#   per-joint greedy minimum over those limits  1.931e-3 kg*m^2   <-- worst case used below
#
# THE ARITHMETIC, at the rotational stiffness this action uses (kp = 3.0) and dt = 1/120 s:
#
#   damping_ratio 0.1  ->  kd = 2*sqrt(3)*0.1 = 0.3464 N*m*s/rad
#     kd*dt/I  =  0.3464 / (120 * I)  =  1.05 (open)   0.66 (closed)   1.50 (greedy worst case)
#   damping_ratio 0.2  ->  kd = 0.6928
#     kd*dt/I  =  2.09 (open)  --  ALREADY DIVERGENT AT THE RESET POSTURE.
#
# So 0.1, and 0.2 -- the value the UR5e's own linear gripper runs -- is not merely aggressive here,
# it is over the bound before the policy takes its first action. The margin 0.1 buys (1.05 at the
# reset posture, 1.50 at a posture the hand probably cannot even reach through its own
# self-collisions) is the same margin every shipped configuration on this arm family runs at: the
# UR5e's 1.1 kg linear gripper sits at 1.45, the UR10e's 0.575 kg one at 1.44, and the UR10e+DELTO
# at 0.62-1.09.
#
# That last line is the cross-check worth reading twice. The UR10e+DELTO's ratio was fixed at 0.1
# from a reflected inertia MEASURED IN SIMULATION (2.66e-3 to 4.67e-3 kg*m^2); this file's number
# comes from arithmetic on a different robot's USD with no simulator involved, and lands on
# 2.757e-3 to 4.376e-3. Two independent methods, two different arms, the same band -- which is
# exactly what should happen, because the quantity is dominated by the hand and the hand is the
# same graft bolted to the same flange. The agreement is evidence for the number; it is NOT a
# licence to inherit it, and the arithmetic above is what this constant actually rests on.
#
# WHAT ``I`` IS AND IS NOT, because one omission is large enough to change the conclusion if it is
# ever quietly folded in. ``I`` above is the RIGID-BODY inertia of the distal chain and excludes
# JOINT ARMATURE. The UR5e's identified wrist_3 armature is 0.3825 kg*m^2 -- 139 times the whole
# hand's rotational inertia about that axis -- and PhysX adds armature straight onto the joint-space
# mass-matrix diagonal, so with it applied the same expression reads 0.0075 instead of 1.05 and no
# rotational damping ratio would ever look unsafe. That is not a reason to raise this number:
#
#   * the USD authors no armature on ``wrist_3_joint`` (drive attributes only). Armature exists
#     solely because ``randomize_arm_from_sysid`` writes it, ON RESET, and the ADR variant SCALES
#     IT BY CURRICULUM PROGRESS -- so it is zero at the start of training and zero between spawn
#     and the first reset even in the ``_fixed`` variant this env uses;
#   * zero armature is therefore a regime this controller genuinely runs in, and it is the regime
#     the vibration and runaway on the two linear grippers were actually observed in;
#   * and the whole point of holding the criterion fixed is that its four calibration points --
#     2F-85 at 1.0, linear/UR5e at 0.2, linear/UR10e at 0.1, DELTO/UR10e at 0.1 -- were all read
#     with I computed this same way. A criterion is only worth the consistency of its inputs. Read
#     "1.05" as "the same margin the shipped grippers run at", not as a proof of stability.
#
# Two things the bound above deliberately does not include, both checked and both slack:
#   * the STIFFNESS half of the same discretisation, kp*dt^2/I = 3*(1/120)^2/2.757e-3 = 0.076,
#     three orders under the kd term -- which is why a single-parameter criterion is honest here;
#   * the TRANSLATIONAL channel, kd_xyz*dt/m = 2*sqrt(200)*3/(120*1.9614) = 0.36 against the same
#     bound of 2. Mass, unlike rotational inertia, barely moved. That is why only the rotational
#     ratio gets re-derived per end effector.
#
# NON-NEGOTIABLE if this is ever revised: validate IN CONTACT, with the hand closed on an object.
# The UR10e linear gripper's rot Kp 3 -> 6 attempt passed every free-space jitter and authority
# probe and then entered a high-frequency limit cycle the moment the jaws touched an object.
# Free-space probes do not detect that class of instability, and neither does the algebra above.
_UR5E_DELTO_ROT_DAMPING_RATIO = 0.1

UR5E_DELTO_RELATIVE_OSC = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="wrist_3_link",
    scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    motion_stiffness=(200.0, 200.0, 200.0, 3.0, 3.0, 3.0),
    motion_damping_ratio=(3.0, 3.0, 3.0) + (_UR5E_DELTO_ROT_DAMPING_RATIO,) * 3,
    torque_limit=_UR5E_TORQUE_LIMIT,
    calibration_dir=_UR5E_DELTO_CALIBRATION_DIR,
)

# Eval / sim2real gains. RECORDED AS SPECIFIED AND KNOWN TO FAIL THE BOUND ABOVE: at rotational
# kp = 50 and damping_ratio 1.0, kd = 2*sqrt(50) = 14.14, so kd*dt/I = 14.14/(120*2.757e-3) = 42.7
# against a bound of 2 -- twenty-one times over, at the reset posture. That is not a typo
# introduced here; it is the state every eval OSC group on this arm family is in (the UR10e+DELTO
# and both linear-gripper eval groups carry the same caveat in prose, and the same arithmetic gives
# 58.9 for the UR10e linear gripper). The eval gains were fitted as a stiff PAIR against the 2F-85
# and have never been re-derived for a heavy multi-finger hand; rescaling only the damping of a
# stiffer controller is a different correction, not this one.
#
# It is defined so the spec is written down in one place with its number attached, and it is
# deliberately NOT reachable from any registered environment: nothing imports the group below.
# Before an eval or fine-tune run on this robot, re-derive the pair -- do not simply run this.
UR5E_DELTO_RELATIVE_OSC_EVAL = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="wrist_3_link",
    scale_xyz_axisangle=(0.01, 0.01, 0.002, 0.02, 0.02, 0.2),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    torque_limit=_UR5E_TORQUE_LIMIT,
    calibration_dir=_UR5E_DELTO_CALIBRATION_DIR,
)

# NOTE there is deliberately no ``Ur5eDeltoRelativeOSC*Action`` GROUP here, unlike every other
# robot in this module. The groups above exist because OmniReset environments mount them. The only
# environment that mounts the two terms above is the dexlift task, so its group -- which is what
# pairs this arm half with the hand half and is where the resulting action dimension is asserted --
# lives with that task, in ``dexlift/dexlift_ur5e_delto_osc_actions.py``, exactly as the other
# variant's group does. When an OmniReset UR5e+DELTO env is written, define its group here and
# import these same two terms; do not restate the gains.
