# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FurnitureBench table-leg grasp and lift with UR10e + Tesollo DELTO."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.dexsuite import dexsuite_env_cfg as dexsuite
from isaaclab_tasks.manager_based.manipulation.dexsuite.adr_curriculum import CurriculumCfg as DexsuiteCurriculumCfg

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR
from uwlab_assets.robots.ur10e_delto.ur10e_delto import DELTO_HAND_DEFAULT_JOINT_POS

from . import mdp
from .dexlift_ur10e_delto_env_cfg import (
    ALL_TIP_NAMES,
    ARM_JOINT_NAMES,
    HAND_JOINT_REGEX,
    PALM_BODY,
    TIP_BODY_REGEX,
    UR10eDeltoEventCfg,
    UR10eDeltoMixinCfg,
)

ASSET_ROOT = f"{UWLAB_LOCAL_ASSETS_DIR}/Props/FurnitureBench/SquareTableOneLeg"
# Put the near table edge at x=0.45 m, leaving the UR a little more operating
# room than the measured OmniReset work surface's x=0.35 m near edge.
TABLE_CENTER_X = 0.85
WORKSPACE_X = 0.98
WORKSPACE_Y = 0.12
TABLE_CENTER_Z = 0.235
TABLE_THICKNESS = 0.04
TABLE_TOP_Z = TABLE_CENTER_Z + 0.5 * TABLE_THICKNESS
# The leg starts airborne and falls freely onto the table before acquisition.
LEG_SPAWN_POS = (1.00035097, 0.12072423, 0.40887862)
# The horizontal 30 mm leg rests with its root 15 mm above the tabletop. Success
# is an 80 mm physical lift from that support pose; the termination independently
# measures the same minimum displacement from each episode's observed low point.
MINIMUM_LIFT = 0.08
TARGET_OBJECT_POS_B = (WORKSPACE_X, WORKSPACE_Y, 0.65)
TARGET_CLEARANCE = TARGET_OBJECT_POS_B[2] - TABLE_TOP_Z
TARGET_POSITION_TOLERANCE = 0.05
CONTACT_THRESHOLD = 0.05
MINIMUM_ARTICULATED_FINGERS = 2
MINIMUM_JOINT_DISPLACEMENT = 0.10
MAX_SUCCESS_OBJECT_SPEED = 0.15
MAX_SUCCESS_RELATIVE_SPEED = 0.15
# Palm-relative pose reached only after the leg has fallen to the table and the
# open hand has approached it. Smooth replay establishes opposed phalange
# contact from this pose without palm, base, or mount contact.
DESIRED_OBJECT_POS_P = (0.06681, 0.06487, 0.16058)
# Post-reset object orientation in the nominal palm frame (w, x, y, z).
DESIRED_OBJECT_QUAT_P = (-0.05087, -0.71751, -0.00219, 0.69469)
# Collision-checked reset pose. The open palm is 407 mm from the airborne leg
# root and remains clear while the leg falls; the policy must lower the palm
# about 456 mm before first finger contact is possible.
RESET_ARM_JOINT_POS = {
    "shoulder_pan_joint": -0.15457708,
    "shoulder_lift_joint": -1.32246149,
    "elbow_joint": 1.09441864,
    "wrist_1_joint": 0.30196786,
    "wrist_2_joint": 1.40933454,
    "wrist_3_joint": -1.66120136,
}
# Collision-checked table-supported approach endpoint. The open hand reaches
# this only after the airborne leg has settled; closure then causes first
# contact, rather than inheriting contact from reset.
APPROACH_ARM_JOINT_POS = {
    "shoulder_pan_joint": -0.15458396,
    "shoulder_lift_joint": -1.19841504,
    "elbow_joint": 1.75789106,
    "wrist_1_joint": -0.48564836,
    "wrist_2_joint": 1.40925562,
    "wrist_3_joint": -1.66108799,
}
# Raw hand command found by smooth, full-gravity search and verified to sustain
# geometrically opposed contact while the arm lifts. Converting through the
# task's default-centered action scales gives the target used by dense shaping.
GRASP_HAND_ACTION = {
    name: action
    for name, action in zip(
        (f"rj_dg_{finger}_{joint}" for finger in range(1, 6) for joint in range(1, 5)),
        (
            0.43454090,
            -0.14072552,
            0.55655491,
            0.33865815,
            -0.83503151,
            -0.43959865,
            1.0,
            0.69505721,
            -0.42886350,
            -0.65505075,
            0.45063210,
            0.61510897,
            0.23848540,
            0.93663901,
            -0.06144656,
            0.08257288,
            0.56994998,
            0.23781885,
            -0.63208628,
            -0.08513815,
        ),
        strict=True,
    )
}
GRASP_HAND_JOINT_POS = {
    name: DELTO_HAND_DEFAULT_JOINT_POS[name] + (1.50 if name.endswith("_4") else 0.30) * action
    for name, action in GRASP_HAND_ACTION.items()
}
# Smooth replay carries the root 372 mm from the table while preserving the
# palm-down orientation and triggers the strict 30-step success hold.
LIFT_ARM_JOINT_POS = {
    "shoulder_pan_joint": -0.15457757,
    "shoulder_lift_joint": -1.31123459,
    "elbow_joint": 1.03634858,
    "wrist_1_joint": 0.34880942,
    "wrist_2_joint": 1.40933466,
    "wrist_3_joint": -1.66120088,
}
FULL_OBJECT_POSE_RANGE = {
    # Local robustness around the collision-validated airborne acquisition.
    # Larger offsets are a separate search-and-approach task: they are outside
    # this fixed pre-grasp environment's reachable opposed-contact basin.
    "x": (-0.002, 0.002),
    "y": (-0.003, 0.003),
    "z": (0.0, 0.001),
    "roll": (-0.02, 0.02),
    "pitch": (-0.02, 0.02),
    "yaw": (-0.02, 0.02),
}
FINGER_CONTACT_NAMES = tuple(f"rl_dg_{finger}_{link}" for finger in range(1, 6) for link in ("1", "2", "3", "4", "tip"))
FINGER_CONTACT_GROUPS = tuple(
    tuple(name for name in FINGER_CONTACT_NAMES if name.startswith(f"rl_dg_{finger}_")) for finger in range(1, 6)
)
FINGER_JOINT_GROUPS = tuple(tuple(f"rj_dg_{finger}_{joint}" for joint in range(1, 5)) for finger in range(1, 6))
THUMB_CONTACT_NAMES = tuple(name for name in FINGER_CONTACT_NAMES if name.startswith(("rl_dg_1_", "rl_dg_5_")))
TIP_CONTACT_NAMES = tuple(
    name for name in FINGER_CONTACT_NAMES if name.startswith(("rl_dg_2_", "rl_dg_3_", "rl_dg_4_"))
)
NON_FINGER_HAND_CONTACT_NAMES = (PALM_BODY, "rl_dg_base", "rl_dg_palm")


@configclass
class TableLegJointPositionActionCfg:
    """Default-centered arm control plus a validated whole-hand position synergy."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale={
            # Absolute zero still holds the validated reset pose. Keep arm
            # exploration conservative enough to refine contact instead of
            # sweeping the hand past the leg in a single action update.
            "shoulder_pan_joint": 0.50,
            "shoulder_lift_joint": 0.375,
            "elbow_joint": 0.375,
            "wrist_1_joint": 0.75,
            "wrist_2_joint": 0.75,
            "wrist_3_joint": 1.60,
        },
        use_default_offset=True,
    )
    # One scalar selects the open or collision-validated close posture. All 20
    # joints remain actuated and move to their individually calibrated targets;
    # the reduced policy action avoids rediscovering a fragile 20-D synergy.
    hand_action = mdp.ContinuousSynergyJointPositionActionCfg(
        asset_name="robot",
        joint_names=[HAND_JOINT_REGEX],
        open_command_expr=DELTO_HAND_DEFAULT_JOINT_POS,
        close_command_expr=GRASP_HAND_JOINT_POS,
    )


def _leg_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=f"{ASSET_ROOT}/leg_200mm/square_table_leg4_200mm_matchedmass_sdf_hybrid.urdf",
            fix_base=False,
            joint_drive=None,
            merge_fixed_joints=True,
            # The URDF already supplies twelve CoACD convex body pieces.  Asking
            # the importer to decompose each piece again makes every launch run
            # VHACD for minutes and multiplies contact patches at 4096 envs.
            # A convex hull preserves each authored convex piece; only the small
            # threaded cap is conservatively hulled, away from the grasp zone.
            collider_type="convex_hull",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=LEG_SPAWN_POS,
            # Rotate the nominal leg with the requested +90-degree top-view
            # hand yaw so the collision-validated grasp relationship is kept.
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )


def _table_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.0, TABLE_THICKNESS),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.34, 0.24)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(TABLE_CENTER_X, 0.0, TABLE_CENTER_Z)),
    )


@configclass
class TableLegRewardsCfg(dexsuite.RewardsCfg):
    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.001)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.003)
    # The former term penalized every arm action before contact and therefore
    # explicitly trained the invalid reset-in-hand closure.  The pose terms
    # below now supply the approach gradient.
    pregrasp_arm_motion = None
    approach_arm_action = RewTerm(
        func=mdp.PrecontactArmActionReward,
        weight=20000.0,
        params={
            "target_joint_pos": APPROACH_ARM_JOINT_POS,
            "action_term_name": "arm_action",
            "std": 0.20,
            "minimum_episode_steps": 45,
            "contact_groups": FINGER_CONTACT_GROUPS,
            "threshold": CONTACT_THRESHOLD,
            "unwanted_contact_names": NON_FINGER_HAND_CONTACT_NAMES,
        },
    )
    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        weight=8.0,
        params={"std": 0.20, "asset_cfg": SceneEntityCfg("robot", body_names=[TIP_BODY_REGEX])},
    )
    grasp_position = RewTerm(
        func=mdp.GraspPoseReward,
        # Keep translation learnable even while a random wrist action degrades
        # orientation; the full-pose term then refines the approach.
        weight=1500.0,
        params={
            "desired_object_pos_p": DESIRED_OBJECT_POS_P,
            "position_std": 0.20,
            "orientation_std": 1.5,
            "position_only": True,
            "unwanted_contact_names": NON_FINGER_HAND_CONTACT_NAMES,
            "palm_body": PALM_BODY,
        },
    )
    grasp_pose = RewTerm(
        func=mdp.GraspPoseReward,
        weight=1500.0,
        params={
            "desired_object_pos_p": DESIRED_OBJECT_POS_P,
            "desired_object_quat_p": DESIRED_OBJECT_QUAT_P,
            "position_std": 0.20,
            "orientation_std": 1.0,
            "unwanted_contact_names": NON_FINGER_HAND_CONTACT_NAMES,
            "palm_body": PALM_BODY,
        },
    )
    precise_grasp_pose = RewTerm(
        func=mdp.GraspPoseReward,
        weight=300.0,
        params={
            "desired_object_pos_p": DESIRED_OBJECT_POS_P,
            "desired_object_quat_p": DESIRED_OBJECT_QUAT_P,
            "position_std": 0.05,
            "orientation_std": 0.5,
            "unwanted_contact_names": NON_FINGER_HAND_CONTACT_NAMES,
            "palm_body": PALM_BODY,
        },
    )
    grasp_posture_progress = RewTerm(
        func=mdp.synergy_grasp_action,
        weight=20000.0,
        params={
            "desired_object_pos_p": DESIRED_OBJECT_POS_P,
            "desired_object_quat_p": DESIRED_OBJECT_QUAT_P,
            "action_term_name": "hand_action",
            "position_std": 0.08,
            "orientation_std": 0.75,
            "unwanted_contact_names": NON_FINGER_HAND_CONTACT_NAMES,
            "palm_body": PALM_BODY,
        },
    )
    position_tracking = RewTerm(
        func=mdp.target_position_tracking,
        weight=1000.0,
        params={
            "command_name": "object_pose",
            "std": 0.10,
            "thumb_contact_names": THUMB_CONTACT_NAMES,
            "tip_contact_names": TIP_CONTACT_NAMES,
            "contact_threshold": CONTACT_THRESHOLD,
            "unwanted_contact_names": NON_FINGER_HAND_CONTACT_NAMES,
        },
    )
    postgrasp_lift_action = RewTerm(
        func=mdp.PostgraspArmActionReward,
        weight=20000.0,
        params={
            "target_joint_pos": LIFT_ARM_JOINT_POS,
            "action_term_name": "arm_action",
            "std": 0.5,
            "memory_steps": 20,
            "threshold": CONTACT_THRESHOLD,
            "thumb_contact_name": THUMB_CONTACT_NAMES,
            "tip_contact_names": TIP_CONTACT_NAMES,
            "max_pair_cosine": -0.1,
            "unwanted_contact_names": NON_FINGER_HAND_CONTACT_NAMES,
        },
    )
    palm_contact_penalty = RewTerm(
        func=mdp.unwanted_contact_force,
        weight=-2000.0,
        params={"contact_names": NON_FINGER_HAND_CONTACT_NAMES, "clip": 5.0},
    )
    orientation_tracking = None
    success = RewTerm(func=mdp.is_terminated_term, weight=10000.0, params={"term_keys": "success"})
    early_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-10000.0,
        params={"term_keys": ["object_out_of_bound", "dropped"]},
    )
    finger_contacts = RewTerm(
        func=mdp.finger_contact_count,
        weight=15.0,
        params={
            "threshold": CONTACT_THRESHOLD,
            "contact_groups": FINGER_CONTACT_GROUPS,
            "unwanted_contact_names": NON_FINGER_HAND_CONTACT_NAMES,
        },
    )
    opposition_contact = RewTerm(
        func=mdp.geometric_opposed_contacts,
        weight=150.0,
        params={
            "threshold": CONTACT_THRESHOLD,
            "thumb_contact_names": THUMB_CONTACT_NAMES,
            "tip_contact_names": TIP_CONTACT_NAMES,
            "max_pair_cosine": -0.1,
            "unwanted_contact_names": NON_FINGER_HAND_CONTACT_NAMES,
        },
    )
    stable_grasp = RewTerm(
        func=mdp.stable_grasp,
        weight=60.0,
        params={
            "threshold": CONTACT_THRESHOLD,
            "thumb_contact_name": THUMB_CONTACT_NAMES,
            "tip_contact_names": TIP_CONTACT_NAMES,
            "max_pair_cosine": -0.1,
            "max_object_speed": 0.5,
            "unwanted_contact_names": NON_FINGER_HAND_CONTACT_NAMES,
        },
    )
    object_upward_motion = RewTerm(
        func=mdp.upward_velocity_until_height,
        weight=300.0,
        params={
            "target_height": TARGET_CLEARANCE,
            "std": 0.15,
            "threshold": CONTACT_THRESHOLD,
            "thumb_contact_name": THUMB_CONTACT_NAMES,
            "tip_contact_names": TIP_CONTACT_NAMES,
            "max_pair_cosine": -0.1,
            "unwanted_contact_names": NON_FINGER_HAND_CONTACT_NAMES,
        },
    )
    held_lift_stability = RewTerm(
        func=mdp.held_lift_stability,
        weight=1000.0,
        params={
            "minimum_height": TARGET_CLEARANCE - TARGET_POSITION_TOLERANCE,
            "linear_speed_std": 0.15,
            "relative_speed_std": 0.10,
            "threshold": CONTACT_THRESHOLD,
            "thumb_contact_name": THUMB_CONTACT_NAMES,
            "tip_contact_names": TIP_CONTACT_NAMES,
            "max_pair_cosine": -0.1,
            "robot_cfg": SceneEntityCfg("robot", body_names=[PALM_BODY]),
            "unwanted_contact_names": NON_FINGER_HAND_CONTACT_NAMES,
        },
    )
    lift_progress = RewTerm(
        func=mdp.ActualLiftProgress,
        weight=1000.0,
        params={
            "target_height": TARGET_CLEARANCE,
            "threshold": CONTACT_THRESHOLD,
            "thumb_contact_name": THUMB_CONTACT_NAMES,
            "tip_contact_names": TIP_CONTACT_NAMES,
            "max_pair_cosine": -0.1,
            "unwanted_contact_names": NON_FINGER_HAND_CONTACT_NAMES,
        },
    )
    table_contact_penalty = RewTerm(
        func=mdp.table_contact_penalty,
        weight=-0.5,
        params={
            "table_contact_name": "table_s",
            "threshold": 0.05,
            "thumb_contact_name": THUMB_CONTACT_NAMES,
            "tip_contact_names": TIP_CONTACT_NAMES,
        },
    )
    horizontal_displacement_penalty = RewTerm(
        func=mdp.excessive_horizontal_displacement,
        weight=-200.0,
        # Do not penalize valid samples from the 8 x 12 cm airborne reset range.
        # The term should only discourage launches beyond that workspace.
        params={"center_x": WORKSPACE_X, "center_y": WORKSPACE_Y, "free_radius": 0.15},
    )
    object_speed_penalty = RewTerm(
        func=mdp.excessive_object_speed,
        weight=-20.0,
        params={"max_speed": 0.3},
    )
    contact_force_penalty = RewTerm(
        func=mdp.excessive_contact_force,
        weight=-5.0,
        params={"contact_names": FINGER_CONTACT_NAMES, "max_force": 10.0},
    )


@configclass
class TableLegTerminationsCfg(dexsuite.TerminationsCfg):
    success = DoneTerm(
        func=mdp.SustainedLiftSuccess,
        params={
            "minimum_height": TARGET_CLEARANCE - TARGET_POSITION_TOLERANCE,
            "hold_steps": 30,
            "command_name": "object_pose",
            "position_tolerance": TARGET_POSITION_TOLERANCE,
            "contact_groups": FINGER_CONTACT_GROUPS,
            "finger_joint_groups": FINGER_JOINT_GROUPS,
            "minimum_contact_groups": 2,
            "thumb_contact_names": THUMB_CONTACT_NAMES,
            "tip_contact_names": TIP_CONTACT_NAMES,
            "minimum_articulated_fingers": MINIMUM_ARTICULATED_FINGERS,
            "minimum_joint_displacement": MINIMUM_JOINT_DISPLACEMENT,
            "contact_threshold": CONTACT_THRESHOLD,
            "unwanted_contact_names": NON_FINGER_HAND_CONTACT_NAMES,
            "max_unwanted_contact_force": CONTACT_THRESHOLD,
            "max_object_speed": MAX_SUCCESS_OBJECT_SPEED,
            "minimum_lift": MINIMUM_LIFT,
            "max_palm_distance": 0.30,
            "max_relative_speed": MAX_SUCCESS_RELATIVE_SPEED,
            "minimum_episode_steps": 180,
            "minimum_first_contact_step": 30,
            "max_opposition_pair_cosine": -0.1,
            "palm_body": PALM_BODY,
        },
    )
    dropped = DoneTerm(func=mdp.object_dropped, params={"minimum_height": -0.04})
    abnormal_robot = None


@configclass
class TableLegEventCfg(UR10eDeltoEventCfg):
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            # Keep the standard DexSuite approach pose deterministic; robustness
            # comes from the airborne object's pose variation.
            "position_range": [0.0, 0.0],
            "velocity_range": [0.0, 0.0],
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_robot_wrist_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="wrist_3_joint"),
            "position_range": [0.0, 0.0],
            "velocity_range": [0.0, 0.0],
        },
    )
    reset_robot_elbow_joint = None
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": FULL_OBJECT_POSE_RANGE,
            "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class TableLegCurriculumCfg(DexsuiteCurriculumCfg):
    """Success-adaptive airborne pose randomization at full gravity."""

    adr = CurrTerm(
        func=mdp.SuccessDifficultyScheduler,
        params={
            "success_term": "success",
            "initial_level": 0.0,
            "min_level": 0.0,
            "max_level": 20.0,
            # Preserve the 80% fixed point while traversing the staged task
            # four times faster than the legacy 0.25/-1.0 increments.
            "promotion_step": 1.0,
            "demotion_step": 4.0,
        },
    )

    # Point-cloud perception is intentionally absent from this state-only task.
    object_obs_unoise_min_adr = None
    object_obs_unoise_max_adr = None

    # Zero-gravity resets leave the leg hovering in the closing envelope and
    # recreate the invalid spawn-in-hand behavior.  Keep physical gravity fixed.
    gravity_adr = None

    object_pose_range = CurrTerm(
        func=mdp.adaptive_object_pose_curriculum,
        params={
            "event_term_name": "reset_object",
            "difficulty_term": "adr",
            "stage_start": 10.0,
            "stage_mid": 15.0,
            "stage_end": 20.0,
            "full_pose_range": FULL_OBJECT_POSE_RANGE,
        },
    )


@configclass
class TableLegGraspLiftEnvCfg(UR10eDeltoMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    rewards: TableLegRewardsCfg = TableLegRewardsCfg()
    actions: TableLegJointPositionActionCfg = TableLegJointPositionActionCfg()
    terminations: TableLegTerminationsCfg = TableLegTerminationsCfg()
    events: TableLegEventCfg = TableLegEventCfg()
    curriculum = None

    def __post_init__(self):
        # Dexsuite treats every non-null curriculum as its own ADR schema.
        # Hide task-specific curricula until its initializer has completed.
        task_curriculum = self.curriculum
        self.curriculum = None
        super().__post_init__()
        self.curriculum = task_curriculum

        self.scene.object = _leg_cfg()
        self.scene.table = _table_cfg()
        self.scene.robot.init_state.joint_pos.update(RESET_ARM_JOINT_POS)
        self.scene.robot.actuators["arm"] = self.scene.robot.actuators["arm"].replace(stiffness=1600.0, damping=80.0)
        hand_actuator = self.scene.robot.actuators["hand"]
        # Preserve the identified per-joint gain and effort-limit shapes.  The
        # lightweight 57 g leg needs enough authority to maintain opposed
        # contact throughout the long UR lift; measured forces settle near 3 N
        # per contacted phalanx and peak around 18 N during closure.
        self.scene.robot.actuators["hand"] = hand_actuator.replace(
            stiffness={name: 32.0 * value for name, value in hand_actuator.stiffness.items()},
            damping={name: (32.0**0.5) * value for name, value in hand_actuator.damping.items()},
            effort_limit_sim={name: 24.0 * value for name, value in hand_actuator.effort_limit_sim.items()},
            velocity_limit_sim={name: min(value, 3.0) for name, value in hand_actuator.velocity_limit_sim.items()},
        )
        self.scene.robot.spawn.articulation_props = self.scene.robot.spawn.articulation_props.replace(
            solver_velocity_iteration_count=2
        )
        # Production launches use two ranks, so this is 4096 environments
        # globally while keeping each 4090's contact-rich scene at 2048.
        self.scene.num_envs = 2048
        self.scene.env_spacing = 2.0
        self.scene.replicate_physics = True
        # The exact concave-looking leg is represented by convex decomposition and
        # creates substantially more narrow-phase work than the primitive Dexsuite
        # objects.  The 64 MiB PhysX default overflows at only 256 environments and
        # drops contacts, invalidating both learning and evaluation.
        self.sim.physx.gpu_collision_stack_size = 2**31

        # Disable generic object scaling/mass ADR: this task uses one exact measured asset.
        self.events.randomize_object_scale = None
        self.events.object_scale_mass = None
        # Establish the nominal FurnitureBench policy at the deterministic center
        # of the calibrated dynamics ranges.  Keep reset-pose variation; a later
        # sim-to-real finetune can widen these ranges again.
        # The identified torque-friction model creates a 9-12 mrad deadband in
        # this position-controlled task and can pin the arm at reset.  Retain
        # deterministic position servos here; torque-control tasks still use
        # the sysid model through their own configurations.
        self.events.randomize_arm_sysid.params["scale_range"] = (0.0, 0.0)
        self.events.randomize_arm_sysid.params["delay_range"] = (0, 0)
        for material_event in (self.events.robot_physics_material, self.events.object_physics_material):
            material_event.params["static_friction_range"] = [2.0, 2.0]
            material_event.params["dynamic_friction_range"] = [2.0, 2.0]
        self.events.joint_stiffness_and_damping.params["stiffness_distribution_params"] = [1.0, 1.0]
        self.events.joint_stiffness_and_damping.params["damping_distribution_params"] = [1.0, 1.0]
        self.events.joint_friction.params["friction_distribution_params"] = [1.0, 1.0]
        # UR10eDeltoMixinCfg disables the generic early-termination term. Restore
        # this task's explicit launch/drop penalty after the mixin has run.
        self.rewards.early_termination = RewTerm(
            func=mdp.is_terminated_term,
            weight=-10000.0,
            params={"term_keys": ["object_out_of_bound", "dropped"]},
        )
        self.events.reset_table.params["pose_range"] = {
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "yaw": [0.0, 0.0],
        }

        # Gravity is never disabled: the leg visibly falls to the support table
        # before the hand approaches in training, evaluation, and playback.
        self.events.variable_gravity.params["gravity_distribution_params"] = (
            (0.0, 0.0, -9.81),
            (0.0, 0.0, -9.81),
        )

        # Generic fingertip sensors target the old primitive root. Replace them
        # with one object-centric one-to-many view over every hand body. PhysX
        # supports one source body against many filters; this reports the same
        # equal-and-opposite per-phalanx forces while avoiding 28 replicated
        # filtered-contact views per environment.
        for link_name in ALL_TIP_NAMES:
            setattr(self.scene, f"{link_name}_object_s", None)
        self.scene.object_hand_s = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Object/base_link",
            filter_prim_paths_expr=[
                f"{{ENV_REGEX_NS}}/Robot/gripper/{link_name}"
                for link_name in (*FINGER_CONTACT_NAMES, *NON_FINGER_HAND_CONTACT_NAMES)
            ],
        )
        self.scene.table_s = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/table",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object/base_link"],
        )

        self.observations.policy.object_pos_b = ObsTerm(func=mdp.object_pos_b)
        self.observations.policy.object_velocity_b = ObsTerm(func=mdp.object_velocity_b, clip=(-3.0, 3.0))
        # The airborne settle, approach, close, and lift targets are deliberately
        # phase-gated.  Exposing finite-horizon progress makes that schedule
        # Markov instead of forcing the actor to infer it from action history.
        self.observations.policy.episode_phase = ObsTerm(func=mdp.episode_phase)
        self.observations.proprio.contact = ObsTerm(
            func=mdp.finger_contact_strengths,
            params={"contact_names": FINGER_CONTACT_NAMES, "clip": 5.0},
        )
        self.observations.perception = None

        self.commands.object_pose.ranges.pos_x = (TARGET_OBJECT_POS_B[0], TARGET_OBJECT_POS_B[0])
        self.commands.object_pose.ranges.pos_y = (TARGET_OBJECT_POS_B[1], TARGET_OBJECT_POS_B[1])
        self.commands.object_pose.ranges.pos_z = (TARGET_OBJECT_POS_B[2], TARGET_OBJECT_POS_B[2])
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)
        self.commands.object_pose.debug_vis = False

        self.terminations.object_out_of_bound.params["in_bound_range"] = {
            "x": (0.2, 1.15),
            "y": (-0.45, 0.45),
            "z": (0.0, 1.0),
        }
        # The airborne leg needs time to fall, be acquired, lift 80 mm, and remain
        # stable for the held-success window.
        self.episode_length_s = 24.0
        self.viewer.eye = (1.55, -0.75, 1.05)
        self.viewer.lookat = (WORKSPACE_X, WORKSPACE_Y, 0.50)


@configclass
class TableLegGraspLiftCurriculumEnvCfg(TableLegGraspLiftEnvCfg):
    """Training variant; evaluation always uses the full-range base environment."""

    curriculum: TableLegCurriculumCfg = TableLegCurriculumCfg()


@configclass
class TableLegGraspLiftEnvCfg_PLAY(TableLegGraspLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.events.reset_robot_joints.params["position_range"] = [0.0, 0.0]
        self.commands.object_pose.debug_vis = True
