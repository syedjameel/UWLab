# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the UR10e + custom linear parallel-jaw gripper robot.

This mirrors ``ur5e_linear_gripper.ur5e_linear_gripper`` with the UR10e arm in place of the
UR5e (see ``scripts_v2/tools/conversions/graft_gripper_on_ur10e.py`` for the graft). The two
arms share the exact joint/link naming contract (``shoulder_pan/lift_joint``, ``elbow_joint``,
``wrist_1/2/3_joint``), so only the numbers differ:

* the spawn USD is the local UR10e + linear gripper graft,
* effort/velocity limits are the UR10e's (from the URDF): 330/330/150/56/56/56 N*m,
  120/120/180/180/180/180 deg/s.

The gripper (joints, dual-drive actuator, binary action) is IDENTICAL to the UR5e variant and
is reused from ``ur5e_linear_gripper`` -- there is deliberately ONE definition of the tuned
gripper actuator so a retune propagates to both robots. There is no gripper-only articulation
here either: grasp sampling is arm-independent and uses ``ur5e_linear_gripper.LINEAR_GRIPPER``.

Configurations:
* :obj:`UR10E_LINEAR_ARTICULATION`     - base articulation (USD, init state).
* :obj:`EXPLICIT_UR10E_LINEAR_GRIPPER` - DelayedPD arm (sim2real finetuning).
* :obj:`IMPLICIT_UR10E_LINEAR_GRIPPER` - Implicit arm (RL training).
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR

# Same gripper as the UR5e variant: reuse its default pose and the tuned dual-drive actuator
# (stiffness 1500 / damping 80 / effort 60 -- see ur5e_linear_gripper.py for the derivation).
from uwlab_assets.robots.ur5e_linear_gripper.ur5e_linear_gripper import (
    _LINEAR_GRIPPER_DUAL_ACTUATOR,
    LINEAR_GRIPPER_DEFAULT_JOINT_POS,
)

UR10E_DEFAULT_JOINT_POS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": -1.5708,
    **LINEAR_GRIPPER_DEFAULT_JOINT_POS,
}

# From the UR10e URDF: pan/lift 120 deg/s, elbow + wrists 180 deg/s.
UR10E_VELOCITY_LIMITS = {
    "shoulder_pan_joint": 2.0944,
    "shoulder_lift_joint": 2.0944,
    "elbow_joint": 3.1415,
    "wrist_1_joint": 3.1415,
    "wrist_2_joint": 3.1415,
    "wrist_3_joint": 3.1415,
}

# From the UR10e URDF (the UR10e is a much stronger arm than the UR5e's 150/28).
UR10E_EFFORT_LIMITS = {
    "shoulder_pan_joint": 330.0,
    "shoulder_lift_joint": 330.0,
    "elbow_joint": 150.0,
    "wrist_1_joint": 56.0,
    "wrist_2_joint": 56.0,
    "wrist_3_joint": 56.0,
}

UR10E_LINEAR_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_LOCAL_ASSETS_DIR}/Robots/Ur10eLinearGripper/ur10e_linear_gripper.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=36, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), joint_pos=UR10E_DEFAULT_JOINT_POS),
    soft_joint_pos_limit_factor=1,
)

EXPLICIT_UR10E_LINEAR_GRIPPER = UR10E_LINEAR_ARTICULATION.copy()  # type: ignore
EXPLICIT_UR10E_LINEAR_GRIPPER.actuators = {
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
    "gripper": _LINEAR_GRIPPER_DUAL_ACTUATOR,
}

IMPLICIT_UR10E_LINEAR_GRIPPER = UR10E_LINEAR_ARTICULATION.copy()  # type: ignore
IMPLICIT_UR10E_LINEAR_GRIPPER.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
        stiffness=0.0,
        damping=0.0,
        effort_limit_sim=UR10E_EFFORT_LIMITS,
        velocity_limit_sim=UR10E_VELOCITY_LIMITS,
    ),
    "gripper": _LINEAR_GRIPPER_DUAL_ACTUATOR,
}
