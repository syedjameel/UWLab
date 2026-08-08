# Copyright (c) 2024-2026, The UW Lab Project Developers.
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
from isaaclab_tasks.manager_based.manipulation.dexsuite.adr_curriculum import CurriculumCfg as DexsuiteCurriculumCfg

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR
from uwlab_assets.robots.ur10e_delto.actions import DELTO_BINARY_ACTIONS

from . import mdp
from .dexlift_ur10e_delto_env_cfg import (
    ALL_TIP_NAMES,
    ARM_JOINT_NAMES,
    PALM_BODY,
    TIP_BODY_REGEX,
    UR10eDeltoEventCfg,
    UR10eDeltoMixinCfg,
)
from isaaclab_tasks.manager_based.manipulation.dexsuite import dexsuite_env_cfg as dexsuite


ASSET_ROOT = f"{UWLAB_LOCAL_ASSETS_DIR}/Props/FurnitureBench/SquareTableOneLeg"
TABLE_CENTER_X = 0.55
WORKSPACE_X = 0.75
WORKSPACE_Y = 0.10
TABLE_CENTER_Z = 0.235
TABLE_THICKNESS = 0.04
TABLE_TOP_Z = TABLE_CENTER_Z + 0.5 * TABLE_THICKNESS
# The palm resets around z=0.68 m.  Spawn the leg below every finger collider
# but still airborne above the table, so the hand must descend before grasping.
LEG_SPAWN_ROOT_Z = 0.42
# Lift shaping measures progress from the episode's observed minimum height.
SUCCESS_HEIGHT = 0.31
CONTACT_THRESHOLD = 0.05
FULL_OBJECT_POSE_RANGE = {
    "x": (-0.08, 0.08),
    "y": (-0.12, 0.12),
    "z": (0.0, 0.03),
    "roll": (-0.35, 0.35),
    "pitch": (-0.35, 0.35),
    "yaw": (-3.14159, 3.14159),
}
FINGER_CONTACT_NAMES = tuple(
    f"rl_dg_{finger}_{link}" for finger in range(1, 6) for link in ("1", "2", "3", "4", "tip")
)
FINGER_CONTACT_GROUPS = tuple(
    tuple(name for name in FINGER_CONTACT_NAMES if name.startswith(f"rl_dg_{finger}_"))
    for finger in range(1, 6)
)
THUMB_CONTACT_NAMES = tuple(name for name in FINGER_CONTACT_NAMES if name.startswith(("rl_dg_1_", "rl_dg_5_")))
TIP_CONTACT_NAMES = tuple(name for name in FINGER_CONTACT_NAMES if name.startswith(("rl_dg_2_", "rl_dg_3_", "rl_dg_4_")))


@configclass
class TableLegBinaryGraspActionCfg:
    """Six relative arm joints plus the calibrated DELTO open/close synergy."""

    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale={
            "shoulder_pan_joint": 0.10,
            "shoulder_lift_joint": 0.10,
            "elbow_joint": 0.10,
            "wrist_1_joint": 0.05,
            "wrist_2_joint": 0.05,
            "wrist_3_joint": 0.05,
        },
    )
    gripper_action = DELTO_BINARY_ACTIONS


def _leg_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=f"{ASSET_ROOT}/leg_200mm/square_table_leg4_200mm_matchedmass_sdf_hybrid.urdf",
            fix_base=False,
            joint_drive=None,
            merge_fixed_joints=True,
            collider_type="convex_decomposition",
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
            pos=(WORKSPACE_X, WORKSPACE_Y, LEG_SPAWN_ROOT_Z),
            rot=(0.707107, 0.0, 0.0, 0.707107),
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
    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        weight=8.0,
        params={"std": 0.20, "asset_cfg": SceneEntityCfg("robot", body_names=[TIP_BODY_REGEX])},
    )
    grasp_position = RewTerm(
        func=mdp.GraspPoseReward,
        weight=50.0,
        params={
            # The URDF root is at the thread/body junction.  A collider sweep
            # places the cage 30 mm into the square body, where all five digits
            # maintain contact without requiring an interpenetrating reset.
            "desired_object_pos_p": (0.0586, 0.0900, 0.1700),
            "position_std": 0.15,
            "orientation_std": 1.5,
            "position_only": True,
            "palm_body": PALM_BODY,
        },
    )
    grasp_pose = RewTerm(
        func=mdp.GraspPoseReward,
        weight=1500.0,
        params={
            "desired_object_pos_p": (0.0586, 0.0900, 0.1700),
            "position_std": 0.20,
            "orientation_std": 1.0,
            "palm_body": PALM_BODY,
        },
    )
    precise_grasp_pose = RewTerm(
        func=mdp.GraspPoseReward,
        weight=300.0,
        params={
            "desired_object_pos_p": (0.0586, 0.0900, 0.1700),
            "position_std": 0.05,
            "orientation_std": 0.5,
            "palm_body": PALM_BODY,
        },
    )
    close_near_grasp_position = RewTerm(
        func=mdp.GraspPoseReward,
        weight=200.0,
        params={
            "desired_object_pos_p": (0.0586, 0.0900, 0.1700),
            "position_std": 0.05,
            "orientation_std": 1.0,
            "position_only": True,
            "close_only": True,
            "palm_body": PALM_BODY,
        },
    )
    close_at_grasp_pose = RewTerm(
        func=mdp.GraspPoseReward,
        weight=500.0,
        params={
            "desired_object_pos_p": (0.0586, 0.0900, 0.1700),
            "position_std": 0.05,
            "orientation_std": 0.5,
            "close_only": True,
            "palm_body": PALM_BODY,
        },
    )
    position_tracking = None
    orientation_tracking = None
    success = RewTerm(func=mdp.is_terminated_term, weight=3000.0, params={"term_keys": "success"})
    early_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-3000.0,
        params={"term_keys": ["object_out_of_bound", "dropped"]},
    )
    finger_contacts = RewTerm(
        func=mdp.finger_contact_count,
        weight=15.0,
        params={"threshold": CONTACT_THRESHOLD, "contact_groups": FINGER_CONTACT_GROUPS},
    )
    opposition_contact = RewTerm(
        func=mdp.contacts,
        weight=150.0,
        params={
            "threshold": CONTACT_THRESHOLD,
            "thumb_contact_name": THUMB_CONTACT_NAMES,
            "tip_contact_names": TIP_CONTACT_NAMES,
        },
    )
    stable_grasp = RewTerm(
        func=mdp.stable_grasp,
        weight=60.0,
        params={
            "threshold": CONTACT_THRESHOLD,
            "thumb_contact_name": THUMB_CONTACT_NAMES,
            "tip_contact_names": TIP_CONTACT_NAMES,
            "max_object_speed": 0.5,
        },
    )
    object_upward_motion = RewTerm(
        func=mdp.object_upward_velocity_bonus,
        weight=150.0,
        params={
            "std": 0.15,
            "threshold": CONTACT_THRESHOLD,
            "thumb_contact_name": THUMB_CONTACT_NAMES,
            "tip_contact_names": TIP_CONTACT_NAMES,
        },
    )
    lift_progress = RewTerm(
        func=mdp.ActualLiftProgress,
        weight=300.0,
        params={
            "target_height": SUCCESS_HEIGHT,
            "threshold": CONTACT_THRESHOLD,
            "thumb_contact_name": THUMB_CONTACT_NAMES,
            "tip_contact_names": TIP_CONTACT_NAMES,
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
        weight=-0.5,
        # Do not penalize valid samples from the 8 x 12 cm airborne reset range.
        # The term should only discourage launches beyond that workspace.
        params={"center_x": WORKSPACE_X, "center_y": WORKSPACE_Y, "free_radius": 0.15},
    )
    object_speed_penalty = RewTerm(
        func=mdp.excessive_object_speed,
        weight=-0.5,
        params={"max_speed": 0.5},
    )
    contact_force_penalty = RewTerm(
        func=mdp.excessive_contact_force,
        weight=-0.05,
        params={"contact_names": FINGER_CONTACT_NAMES, "max_force": 10.0},
    )


@configclass
class TableLegTerminationsCfg(dexsuite.TerminationsCfg):
    success = DoneTerm(
        func=mdp.SustainedLiftSuccess,
        params={
            "minimum_height": SUCCESS_HEIGHT,
            "hold_steps": 12,
            "contact_groups": FINGER_CONTACT_GROUPS,
            "minimum_contact_groups": 2,
            "contact_threshold": CONTACT_THRESHOLD,
            "max_object_speed": 0.5,
            "minimum_lift": 0.08,
            "max_palm_distance": 0.30,
            "max_relative_speed": 0.20,
            "minimum_episode_steps": 60,
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
    """Success-adaptive gravity followed by airborne pose randomization."""

    adr = CurrTerm(
        func=mdp.SuccessDifficultyScheduler,
        params={
            "success_term": "success",
            "initial_level": 0.0,
            "min_level": 0.0,
            "max_level": 20.0,
            "promotion_step": 0.25,
            "demotion_step": 1.0,
        },
    )

    # Point-cloud perception is intentionally absent from this state-only task.
    object_obs_unoise_min_adr = None
    object_obs_unoise_max_adr = None

    gravity_adr = CurrTerm(
        func=mdp.adaptive_gravity_curriculum,
        params={
            "event_term_name": "variable_gravity",
            "difficulty_term": "adr",
            "stage_start": 0.0,
            "stage_end": 10.0,
            "final_gravity": -9.81,
        },
    )

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
    actions: TableLegBinaryGraspActionCfg = TableLegBinaryGraspActionCfg()
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
        self.scene.robot.actuators["arm"] = self.scene.robot.actuators["arm"].replace(stiffness=1600.0, damping=80.0)
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
        self.events.randomize_arm_sysid.params["scale_range"] = (1.0, 1.0)
        self.events.randomize_arm_sysid.params["delay_range"] = (0, 0)
        for material_event in (self.events.robot_physics_material, self.events.object_physics_material):
            material_event.params["static_friction_range"] = [0.75, 0.75]
            material_event.params["dynamic_friction_range"] = [0.75, 0.75]
        self.events.joint_stiffness_and_damping.params["stiffness_distribution_params"] = [1.0, 1.0]
        self.events.joint_stiffness_and_damping.params["damping_distribution_params"] = [1.0, 1.0]
        self.events.joint_friction.params["friction_distribution_params"] = [1.0, 1.0]
        # UR10eDeltoMixinCfg disables the generic early-termination term. Restore
        # this task's explicit launch/drop penalty after the mixin has run.
        self.rewards.early_termination = RewTerm(
            func=mdp.is_terminated_term,
            weight=-3000.0,
            params={"term_keys": ["object_out_of_bound", "dropped"]},
        )
        self.events.reset_table.params["pose_range"] = {
            "x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0], "yaw": [0.0, 0.0]
        }

        if task_curriculum is None:
            # Evaluation/play always runs at full gravity. Training starts at zero
            # and lets the inherited DexSuite gravity ADR promote it toward this value.
            self.events.variable_gravity.params["gravity_distribution_params"] = (
                (0.0, 0.0, -9.81),
                (0.0, 0.0, -9.81),
            )

        # Generic fingertip sensors target the old primitive root. Replace them with one
        # sensor per phalange: PhysX filtered-contact views require one source body per
        # sensor, and this also makes missing intermediate/distal collision reporting visible.
        for link_name in ALL_TIP_NAMES:
            setattr(self.scene, f"{link_name}_object_s", None)
        for link_name in FINGER_CONTACT_NAMES:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/gripper/{link_name}",
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object/base_link"],
                ),
            )
        self.scene.table_s = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/table",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object/base_link"],
        )

        self.observations.policy.object_pos_b = ObsTerm(func=mdp.object_pos_b)
        self.observations.policy.object_velocity_b = ObsTerm(func=mdp.object_velocity_b, clip=(-3.0, 3.0))
        self.observations.proprio.contact = ObsTerm(
            func=mdp.finger_contact_strengths,
            params={"contact_names": FINGER_CONTACT_NAMES, "clip": 5.0},
        )
        self.observations.perception = None

        self.commands.object_pose.ranges.pos_x = (WORKSPACE_X, WORKSPACE_X)
        self.commands.object_pose.ranges.pos_y = (WORKSPACE_Y, WORKSPACE_Y)
        self.commands.object_pose.ranges.pos_z = (0.65, 0.65)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)
        self.commands.object_pose.debug_vis = False

        self.terminations.object_out_of_bound.params["in_bound_range"] = {
            "x": (0.2, 0.9), "y": (-0.45, 0.45), "z": (0.0, 1.0)
        }
        self.episode_length_s = 8.0
        self.viewer.eye = (1.35, -0.8, 0.75)
        self.viewer.lookat = (WORKSPACE_X, WORKSPACE_Y, 0.36)


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
