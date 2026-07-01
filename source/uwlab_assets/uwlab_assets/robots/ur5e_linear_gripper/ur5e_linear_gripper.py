# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the UR5e + custom linear parallel-jaw gripper robot.

This mirrors ``ur5e_robotiq_gripper.ur5e_robotiq_2f85_gripper`` exactly, except the gripper
is the custom linear gripper grafted onto the same calibrated UR5e arm (see
``scripts_v2/tools/conversions/graft_gripper_on_ur5e.py``). The arm joints, sysid and the
DelayedPD/Implicit actuator setup are identical -- only the gripper differs:

* driver joint ``finger_joint`` (prismatic, meters; 0 = OPEN, 0.068 = CLOSED),
* ``right_finger_joint`` -- the opposite jaw. Both jaws are DUAL-DRIVEN to the same binary
  target (the PhysX prismatic mimic is unreliable: it lets the driver outrun the free
  follower and the solver pins both jaws at 0, so the gripper never grips). One binary
  command slaves both jaws, so the follower is not an independent policy DOF (paper A.3.3).

Configurations:
* :obj:`UR5E_LINEAR_ARTICULATION`     - base articulation (USD, init state).
* :obj:`EXPLICIT_UR5E_LINEAR_GRIPPER` - DelayedPD arm (sim2real finetuning).
* :obj:`IMPLICIT_UR5E_LINEAR_GRIPPER` - Implicit arm (RL training).
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR

# The linear gripper has two jaw joints (driver + follower); both start at 0 (= OPEN).
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

# Gripper actuator (STANDALONE gripper, grasp sampling): drive BOTH jaws. The PhysX prismatic
# mimic is NOT reliable even in the standalone (where the finger joints are the root DOFs): with
# the soft drive the driver outruns the free follower, the mimic equality constraint accumulates
# error, and at ~step 5 the solver slams both jaws to the lower limit (0) and PINS them -- the
# gripper never grips, so the recorder snapshots finger_joint = 0 for every grasp. So the
# standalone now DUAL-DRIVES both jaws to the same binary target (mimic stripped from the USD,
# exactly like the full robot), giving a symmetric, stable close.
#
# STIFFNESS 200 (not the reference 2F-85's 17): 17 (N/m on our prismatic jaws) gives only a ~0.4 N
# squeeze at the grip equilibrium -- a marginal hold. 200/damping 20 gives a firm ~6 N clamp. (The
# main reason grasps failed to export before was the gripper DRIFT, fixed by the root damping on the
# LINEAR_GRIPPER spawn below; the firmer clamp is complementary.) The old ">=200 ejects a light
# object" finding was for SINGLE-jaw drive (one jaw races in and punts the object); dual-drive closes
# both jaws symmetrically so the forces cancel and it does not fling. effort_limit_sim 60 N caps it.
_LINEAR_GRIPPER_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=["finger_joint", "right_finger_joint"],
    stiffness=200.0,
    damping=20.0,
    effort_limit_sim=60.0,
)

# Gripper actuator (FULL ROBOT, reset/RL): drive BOTH jaws. The PhysX prismatic mimic is INERT
# once the gripper is embedded in the full arm articulation (verified exhaustively: follower gets
# ~zero coupling force; revolute mimic like the 2F-85 would couple, prismatic does not, and there
# is no compliance knob), AND the inert mimic blocks any actuator drive -- so the graft strips it
# and re-activates the follower's DriveAPI. We then command BOTH jaws to the same target here; the
# binary gripper action slaves both jaws to ONE command (not an independent policy DOF, A.3.3 holds).
#
# STIFFNESS 1500 (NOT 17): 17 is the reference 2F-85 gain, but that gripper is REVOLUTE (N*m/rad);
# applied to our PRISMATIC jaws (N/m) it is far too soft -- the jaws then SLOSH under the arm's
# motion (inertial load on the held jaws) and decouple by up to ~0.05 m, which looked like
# "vibration + one finger open/other closed" in the reset GUI. Gravity is disabled on the robot, so
# the disturbance is arm acceleration, not weight. Sweep under aggressive wrist oscillation
# (test_fullrobot_mimic.py --dual-drive --arm-wiggle): 17 -> |diff| 0.05 (FAILS); 1500 -> 0.0035
# (coupled). effort_limit_sim 60 N caps the squeeze force on a grasped object.
_LINEAR_GRIPPER_DUAL_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=["finger_joint", "right_finger_joint"],
    stiffness=1500.0,
    damping=80.0,
    effort_limit_sim=60.0,
)

# Gripper-only articulation for grasp sampling (mirrors ROBOTIQ_2F85): the sampler teleports
# the gripper alone (no arm). Uses the standalone gripper USD (dual-drive: both jaws driven,
# no mimic) plus root damping (below) to stop the free gripper drifting during the grip.
LINEAR_GRIPPER = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/RobotiqGripper",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_LOCAL_ASSETS_DIR}/Robots/LinearGripper/linear_gripper.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            # The grasp sampler teleports this gripper to a candidate and leaves it FREE (no arm,
            # no fixed base) while it closes on the object and gravity/perturbation are applied.
            # Our firm prismatic dual-drive close produces an asymmetric contact reaction that, on
            # an unanchored gravity-disabled body, makes the whole gripper DRIFT ~0.5-0.8 m over the
            # episode -- carrying/dropping the object so EVERY grasp fails (grip_disp ~0.5 measured).
            # High linear/angular damping arrests that drift (the 2F-85's gentler revolute close
            # recoils less, so it does not need this). The recorded relative grasp pose is unaffected.
            linear_damping=50.0,
            angular_damping=50.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=36, solver_velocity_iteration_count=0
        ),
        # Normalize total gripper mass to 0.5 kg, exactly like the reference ROBOTIQ_2F85 spawn
        # (our baked link masses sum to ~1.1 kg; a lighter gripper matches the 2F-85 dynamics).
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
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
    "gripper": _LINEAR_GRIPPER_DUAL_ACTUATOR,
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
    "gripper": _LINEAR_GRIPPER_DUAL_ACTUATOR,
}
