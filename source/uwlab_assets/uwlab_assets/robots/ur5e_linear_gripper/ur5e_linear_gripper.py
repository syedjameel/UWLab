# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the UR5e + custom linear parallel-jaw gripper robot.

This mirrors ``ur5e_robotiq_gripper.ur5e_robotiq_2f85_gripper`` exactly, except the gripper
is the custom linear gripper grafted onto the same calibrated UR5e arm (see
``scripts_v2/tools/conversions/graft_gripper_on_ur5e.py``). The arm joints, sysid and the
DelayedPD/Implicit actuator setup are identical -- only the gripper differs:

* one actuated driver joint ``finger_joint`` (prismatic, meters; 0 = OPEN, 0.068 = CLOSED),
* ``right_finger_joint`` is a PhysX *mimic* of ``finger_joint`` (rigid coupling, per the
  OmniReset paper A.3.3) -- it is NOT actuated, exactly like the 2F-85's passive joints.

Configurations:
* :obj:`UR5E_LINEAR_ARTICULATION`     - base articulation (USD, init state).
* :obj:`EXPLICIT_UR5E_LINEAR_GRIPPER` - DelayedPD arm (sim2real finetuning).
* :obj:`IMPLICIT_UR5E_LINEAR_GRIPPER` - Implicit arm (RL training).
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR

# The linear gripper exposes one driver joint + one mimic; both start at 0 (= OPEN).
LINEAR_GRIPPER_DEFAULT_JOINT_POS = {
    "finger_joint": 0.0,
    "right_finger_joint": 0.0,
}

UR5E_DEFAULT_JOINT_POS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": -1.5708,
    **LINEAR_GRIPPER_DEFAULT_JOINT_POS,
}

UR5E_VELOCITY_LIMITS = {
    "shoulder_pan_joint": 1.5708,
    "shoulder_lift_joint": 1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": 3.1415,
    "wrist_2_joint": 3.1415,
    "wrist_3_joint": 3.1415,
}

UR5E_EFFORT_LIMITS = {
    "shoulder_pan_joint": 150.0,
    "shoulder_lift_joint": 150.0,
    "elbow_joint": 150.0,
    "wrist_1_joint": 28.0,
    "wrist_2_joint": 28.0,
    "wrist_3_joint": 28.0,
}

UR5E_LINEAR_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_LOCAL_ASSETS_DIR}/Robots/Ur5eLinearGripper/ur5e_linear_gripper.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=36, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), joint_pos=UR5E_DEFAULT_JOINT_POS),
    soft_joint_pos_limit_factor=1,
)

# Gripper actuator: drive ONLY the finger_joint (the mimic makes right_finger_joint follow).
# Prismatic position drive (N/m, N). Stiffness is deliberately LOW: a stiff/fast close slams
# the jaws shut and FLINGS a light object out (verified -- stiffness>=200 ejects the slab,
# stiffness 50 grips it: finger_joint stops on the object instead of closing fully).
_LINEAR_GRIPPER_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=["finger_joint"],
    stiffness=50.0,
    damping=5.0,
    effort_limit_sim=120.0,
)

# Gripper-only articulation for grasp sampling (mirrors ROBOTIQ_2F85): the sampler teleports
# the gripper alone (no arm). Uses the standalone gripper USD (with the PhysX mimic baked in).
LINEAR_GRIPPER = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/RobotiqGripper",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_LOCAL_ASSETS_DIR}/Robots/LinearGripper/linear_gripper.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=36, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0.1), rot=(1, 0, 0, 0), joint_pos=LINEAR_GRIPPER_DEFAULT_JOINT_POS
    ),
    actuators={"gripper": _LINEAR_GRIPPER_ACTUATOR},
    soft_joint_pos_limit_factor=1,
)

EXPLICIT_UR5E_LINEAR_GRIPPER = UR5E_LINEAR_ARTICULATION.copy()  # type: ignore
EXPLICIT_UR5E_LINEAR_GRIPPER.actuators = {
    "arm": DelayedPDActuatorCfg(
        joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
        stiffness=0.0,
        damping=0.0,
        effort_limit=UR5E_EFFORT_LIMITS,
        effort_limit_sim=UR5E_EFFORT_LIMITS,
        velocity_limit=UR5E_VELOCITY_LIMITS,
        velocity_limit_sim=UR5E_VELOCITY_LIMITS,
        min_delay=0,
        max_delay=1,
    ),
    "gripper": _LINEAR_GRIPPER_ACTUATOR,
}

IMPLICIT_UR5E_LINEAR_GRIPPER = UR5E_LINEAR_ARTICULATION.copy()  # type: ignore
IMPLICIT_UR5E_LINEAR_GRIPPER.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
        stiffness=0.0,
        damping=0.0,
        effort_limit_sim=UR5E_EFFORT_LIMITS,
        velocity_limit_sim=UR5E_VELOCITY_LIMITS,
    ),
    "gripper": _LINEAR_GRIPPER_ACTUATOR,
}
