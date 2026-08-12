# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Articulation configurations for the UR10e with a Tesollo DELTO DG-5F hand.

The committed assets retain all 28 hand rigid bodies and align each fingertip collider with its
visible mesh. ``rl_dg_mount`` is attached directly to the UR flange with zero standoff. Arm-side
posture and actuator constants are shared with the calibrated UR10e linear-gripper configuration.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR
from uwlab_assets.robots.ur5e_linear_gripper.ur5e_linear_gripper import LINEAR_GRIPPER_DEFAULT_JOINT_POS

# Same arm as the linear-gripper robot: reuse its posture and limits verbatim.
from uwlab_assets.robots.ur10e_linear_gripper.ur10e_linear_gripper import (
    UR10E_DEFAULT_JOINT_POS,
    UR10E_EFFORT_LIMITS,
    UR10E_VELOCITY_LIMITS,
)

# UR10E_DEFAULT_JOINT_POS bundles the arm posture WITH the parallel jaw's two joints, which do not
# exist on this robot -- an init_state key that matches no joint is an error at spawn. Drop exactly
# those two rather than restating the six arm values.
_UR10E_ARM_DEFAULT_JOINT_POS = {
    name: value for name, value in UR10E_DEFAULT_JOINT_POS.items() if name not in LINEAR_GRIPPER_DEFAULT_JOINT_POS
}

# Open posture in radians. Keep it identical to ``finger_open_joint_angles`` in the hand metadata.
DELTO_HAND_DEFAULT_JOINT_POS = {
    "rj_dg_1_1": 0.181879,
    "rj_dg_1_2": -0.373635,
    "rj_dg_1_3": 0.303144,
    "rj_dg_1_4": 0.159307,
    "rj_dg_2_1": -0.058794,
    "rj_dg_2_2": 0.9,
    "rj_dg_2_3": 0.695646,
    "rj_dg_2_4": 0.573853,
    "rj_dg_3_1": -0.032855,
    "rj_dg_3_2": 0.9,
    "rj_dg_3_3": 0.695646,
    "rj_dg_3_4": 0.573853,
    "rj_dg_4_1": 0.174352,
    "rj_dg_4_2": 0.9,
    "rj_dg_4_3": 0.695646,
    "rj_dg_4_4": 0.573853,
    "rj_dg_5_1": 0.193785,
    "rj_dg_5_2": 0.9,
    "rj_dg_5_3": 0.695646,
    "rj_dg_5_4": 0.573853,
}

UR10E_DELTO_DEFAULT_JOINT_POS = {
    **_UR10E_ARM_DEFAULT_JOINT_POS,
    **DELTO_HAND_DEFAULT_JOINT_POS,
}

# Hand actuator. Per-joint gains preserve the serial-chain stiffness profile; effort and speed
# limits prevent the fingers from ejecting lightweight parts on first contact.
#
# What the USD authors, for reference: stiffness 0.096-2.45 N*m/rad, damping pinned at exactly
# stiffness/2500 (3.8e-5 - 9.8e-4), maxForce 30 N*m, type "force". The stiffness column is good and
# is kept; the damping column is auto-generated and is not. The USD also authors
# ``physxJoint:maxJointVelocity`` 419 deg/s, which PhysX enforces independently of the actuator's
# velocity_limit_sim below.
_DELTO_HAND_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=[r"rj_dg_[1-5]_[1-4]"],
    # The USD's OWN per-joint stiffness, except for the three distal _4 joints whose authored
    # 0.0957 N*m/rad cannot overcome the reference actuator friction during this small commanded
    # stroke. They use 0.30: still softer than the _3 joints (0.4112), with force independently
    # capped by the 0.06 N*m effort_limit_sim below. The 0.01 N*m friction is the reference
    # implementation's value, not a hardware measurement.
    #
    # The two regimes are not the same problem. The reference commands small relative deltas at
    # 60 Hz and lets the policy regulate force; we command a fixed posture and HOLD it against
    # contact, so stiffness maps straight onto grip force. A uniform gain is also the wrong SHAPE
    # on a serial chain: the distal joint works a 25.5 mm lever against the proximal joint's
    # ~93 mm, so identical joint stiffness is 3.6x harsher in fingertip force. These authored
    # values fall off distally in near-proportion to lever arm, which is the correct shaping.
    # Measured on one posture pair: 40 N total with these, 288 N with a flat 3.0 -- on a 30-52 g
    # part. The flat gain's worst offender is rj_dg_2_4 at 60 N, from a joint whose authored
    # stiffness gives 1.9 N.
    stiffness={
        "rj_dg_1_1": 0.8294,
        "rj_dg_1_2": 0.6286,
        "rj_dg_1_3": 4.0,
        "rj_dg_1_4": 0.1698,
        "rj_dg_2_1": 2.4381,
        "rj_dg_2_2": 0.8276,
        "rj_dg_2_3": 4.0,
        "rj_dg_2_4": 0.30,
        "rj_dg_3_1": 2.4536,
        "rj_dg_3_2": 0.8264,
        "rj_dg_3_3": 4.0,
        "rj_dg_3_4": 0.30,
        "rj_dg_4_1": 2.3093,
        "rj_dg_4_2": 0.8273,
        "rj_dg_4_3": 4.0,
        "rj_dg_4_4": 0.30,
        "rj_dg_5_1": 1.7342,
        "rj_dg_5_2": 1.1075,
        "rj_dg_5_3": 4.0,
        "rj_dg_5_4": 0.1694,
    },
    # zeta ~ 0.7 against the composite subtree inertias computed from the USD mass properties. The
    # USD's authored damping is zeta 0.009-0.016 -- essentially undamped, which is tolerable for
    # 60 Hz corrections but bad for holding a static grip, where the fingers ring against the part
    # and modulate contact force.
    damping={
        r"rj_dg_[1-5]_1": 0.040,
        r"rj_dg_[1-5]_2": 0.025,
        r"rj_dg_[1-5]_3": 0.0374,
        r"rj_dg_[1-5]_4": 0.0025,
    },
    # ~2.2 N fingertip cap, scaled per joint class by that class's smallest lever arm. The USD
    # authors maxForce 30 N*m on all 20 joints (the DG-5F's ~7 kg rated grip) and the reference
    # copies it; at these levers that permits 323-1176 N at the fingertip. This is a BACKSTOP
    # against a runaway target, not a substitute for commanding a sane stroke.
    effort_limit_sim={
        r"rj_dg_[1-5]_1": 0.13,
        r"rj_dg_[1-5]_2": 0.17,
        r"rj_dg_[1-5]_3": 0.14,
        r"rj_dg_[1-5]_4": 0.06,
    },
    # Closing speed sets the kinetic energy delivered to a ~40 g part on first touch, which is the
    # ejection mode directly; the reference's 10000 rad/s is effectively unbounded. 3.0 rad/s is
    # still generous against real DG-5F hardware (~1.5 rad in ~1 s).
    velocity_limit_sim={r"rj_dg_[1-5]_[1-4]": 3.0},
    friction={r"rj_dg_[1-5]_[1-4]": 0.01},
)

UR10E_DELTO_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_LOCAL_ASSETS_DIR}/Robots/Ur10eDelto/ur10e_delto.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            # The soft 20-joint hand otherwise injects persistent rigid-link ringing into the
            # wrist while the fingers close. This is the package value used by the D2/D3 datasets
            # and the D4 training run; changing it invalidates those artifacts. (The ~10 s figure
            # this once quoted was the duration of the deleted scripted binary close; a policy
            # driving twenty joints has no such fixed timescale.)
            linear_damping=500.0,
            angular_damping=500.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Self-collisions ON, as for the linear gripper, and as the hand asset itself asks for:
            # it comes from the reference's ``..._self_collision.usd`` variant, which authors
            # ``physxArticulation:enabledSelfCollisions = True``. Preparation filters only the five
            # palm <-> first-phalanx structural pairs whose original source joints were collapsed;
            # every other non-adjacent hand pair remains collision-enabled.
            enabled_self_collisions=True,
            solver_position_iteration_count=36,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0), rot=(1, 0, 0, 0), joint_pos=UR10E_DELTO_DEFAULT_JOINT_POS
    ),
    soft_joint_pos_limit_factor=1,
)

EXPLICIT_UR10E_DELTO = UR10E_DELTO_ARTICULATION.copy()  # type: ignore
EXPLICIT_UR10E_DELTO.actuators = {
    "arm": DelayedPDActuatorCfg(
        joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
        stiffness=0.0,
        damping=0.0,
        effort_limit=UR10E_EFFORT_LIMITS,
        effort_limit_sim=UR10E_EFFORT_LIMITS,
        velocity_limit=UR10E_VELOCITY_LIMITS,
        velocity_limit_sim=UR10E_VELOCITY_LIMITS,
        min_delay=0,
        max_delay=1,
    ),
    "hand": _DELTO_HAND_ACTUATOR,
}

IMPLICIT_UR10E_DELTO = UR10E_DELTO_ARTICULATION.copy()  # type: ignore
IMPLICIT_UR10E_DELTO.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
        stiffness=0.0,
        damping=0.0,
        effort_limit_sim=UR10E_EFFORT_LIMITS,
        velocity_limit_sim=UR10E_VELOCITY_LIMITS,
    ),
    "hand": _DELTO_HAND_ACTUATOR,
}

# Hand-only articulation for grasp sampling (mirrors LINEAR_GRIPPER): the sampler teleports the
# hand alone, with no arm, and closes it on the object. Same USD the full robot is grafted from, so
# the sampled grasps and the deployed hand are the same geometry and the same 1.7735 kg.
#
# Two deliberate departures from LINEAR_GRIPPER:
#
# * NO ``mass_props``. LINEAR_GRIPPER passes ``MassPropertiesCfg(mass=0.5)`` under a comment saying
#   it normalises the TOTAL gripper mass, but ``modify_mass_properties`` is decorated
#   ``@apply_nested``: it sets that mass on EVERY nested rigid body, so the parallel jaw's three
#   links end up at 0.5 kg each (1.5 kg total, not 0.5). Applied to this hand's 28 bodies it would
#   produce a 14 kg hand. The authored per-link masses are correct and are left alone.
# * No extra root damping. LINEAR_GRIPPER needs ``linear_damping``/``angular_damping`` 50 because
#   its firm prismatic dual-drive close recoils hard enough to fling the unanchored gripper across
#   the scene. The DELTO's 20 soft revolute drives close far more gently. Add damping only if
#   grasp export shows the hand drifting (the symptom was ``grip_disp`` around 0.5 m).
#
# ``enabled_self_collisions`` is True to match the source asset, which authors
# ``physxArticulation:enabledSelfCollisions = True`` on its own root -- this hand comes from the
# reference's ``..._self_collision.usd`` variant specifically.
DELTO_HAND = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/DeltoHand",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_LOCAL_ASSETS_DIR}/Robots/DeltoHand/delto_hand.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=36,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0.1), rot=(1, 0, 0, 0), joint_pos=DELTO_HAND_DEFAULT_JOINT_POS
    ),
    actuators={"hand": _DELTO_HAND_ACTUATOR},
    soft_joint_pos_limit_factor=1,
)
