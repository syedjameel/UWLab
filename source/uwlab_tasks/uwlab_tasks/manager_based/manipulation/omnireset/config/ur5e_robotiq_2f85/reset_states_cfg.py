# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets import UWLAB_ASSETS_DATA_DIR, UWLAB_CLOUD_ASSETS_DIR, UWLAB_LOCAL_ASSETS_DIR
from uwlab_assets.robots.ur5e_robotiq_gripper import IMPLICIT_UR5E_ROBOTIQ_2F85

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.actions import (
    Ur5eRobotiq2f85RelativeOSCAction,
)

from ... import mdp as task_mdp
from .gripper_seam import ROBOTIQ_2F85_GRIPPER_JOINTS, robotiq_2f85_gripper_joints


@configclass
class ResetStatesSceneCfg(InteractiveSceneCfg):
    """Scene configuration for reset states environment."""

    robot = IMPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

    insertive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            # assume very light
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    receptive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                # receptive object does not move
                kinematic_enabled=True,
            ),
            # since kinematic_enabled=True, mass does not matter
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Environment -- the REAL lab table (see rl_state_cfg.py for the full rationale;
    # this block is duplicated there -- keep both in sync). Asset frame == robot base
    # frame; work surface at +0.004; reset_robot_pose jitters robot+support+table
    # TOGETHER (rigid assembly -- the base is bolted to this table).
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Mounts/CustomLabTable/custom_lab_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # Flush proxy plate (placement datum): ROOT z = work surface (+0.004).
    ur5_metal_support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/UR5MetalSupport",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0, 0.004), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Mounts/CustomLabTable/custom_mount_plate.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.676)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=10000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class ResetStatesBaseEventCfg:
    """Configuration for randomization."""

    # startup: fail immediately if this task's gripper joint selection does not fit the robot it
    # spawns (a variant that forgot its gripper seam). Draws no randomness; see
    # gripper_seam.override_gripper_joints, which keeps this in step with the events below.
    check_gripper_joints = EventTerm(
        func=task_mdp.check_gripper_joint_selection,
        mode="startup",
        params={"joint_names": ROBOTIQ_2F85_GRIPPER_JOINTS, "asset_name": "robot"},
    )

    # startup: low friction to avoid slip
    reset_robot_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.3),
            "dynamic_friction_range": (0.2, 0.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("robot"),
            "make_consistent": True,
        },
    )

    insertive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.3),
            "dynamic_friction_range": (0.2, 0.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "make_consistent": True,
        },
    )

    receptive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.3),
            "dynamic_friction_range": (0.2, 0.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "make_consistent": True,
        },
    )

    # reset

    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})

    reset_robot_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.01, 0.01),
                # AUTHORS' design verbatim: robot + support jitter against a STATIC
                # table (their base-placement DR, z included). Only the y CENTER differs:
                # theirs was -0.039 (their rig's measured base placement); ours is 0 by
                # construction (table asset frame == robot base frame). Width identical.
                # Authors-authentic artifact: the jittered base can visually clip the
                # static mat's cutout edge -- their kinematic plate clipped their table
                # the same way.
                "y": (-0.02, 0.02),
                "z": (-0.01, 0.01),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfgs": {
                "robot": SceneEntityCfg("robot"),
                "ur5_metal_support": SceneEntityCfg("ur5_metal_support"),
            },
        },
    )

    reset_receptive_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                # x (authors: 0.3, 0.55) shifted +5 cm, band size kept: on our rig the
                # black mat (x < 0.35, with the base cutout) is the base's zone and the
                # green mat starting at x = 0.35 is the physical workspace -- objects
                # spawn fully on the green mat, clear of the base. Same shift in the two
                # insertive events below; C3/C4 inherit (C1 restores / receptive-relative).
                "x": (0.35, 0.60),
                # y centered (authors: -0.1, 0.5 -- their rig's workspace was off to one
                # side; ours is symmetric): same 0.40 m width, centered on the base/mat
                # centerline, 12 cm margin to both mat edges (y +-0.35).
                "y": (-0.2, 0.2),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-np.pi / 12, np.pi / 12),
            },
            "velocity_range": {},
            "asset_cfgs": {"receptive_object": SceneEntityCfg("receptive_object")},
            "offset_asset_cfg": SceneEntityCfg("ur5_metal_support"),
            "use_bottom_offset": True,
        },
    )


@configclass
class ObjectAnywhereEEAnywhereEventCfg(ResetStatesBaseEventCfg):
    reset_insertive_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                # x shifted +5 cm onto the green workspace mat (see reset_receptive_object_pose).
                "x": (0.35, 0.60),
                # y centered, same 0.40 m width (see reset_receptive_object_pose).
                "y": (-0.2, 0.2),
                "z": (0.0, 0.3),
                # PCB stays essentially top-up: only a small +/-0.1 rad (~6 deg) roll/pitch
                # jitter for robustness to placement error; yaw stays free (in-plane spin
                # is realistic and does not affect success, which checks roll+pitch only).
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-np.pi, np.pi),
            },
            "velocity_range": {},
            "asset_cfgs": {"insertive_object": SceneEntityCfg("insertive_object")},
            "offset_asset_cfg": SceneEntityCfg("ur5_metal_support"),
            "use_bottom_offset": True,
        },
    )

    reset_end_effector_pose = EventTerm(
        func=task_mdp.reset_end_effector_round_fixed_asset,
        mode="reset",
        params={
            "fixed_asset_cfg": SceneEntityCfg("robot"),
            "fixed_asset_offset": None,
            "pose_range_b": {
                "x": (0.3, 0.7),
                "y": (-0.4, 0.4),
                # floor 0.017 (authors' literal: 0.0) = the authors' EFFECT exactly: their
                # surface was at -0.013, so their floor gave 13 mm minimum clearance; our
                # surface is at +0.004 -> 0.017 gives the identical 13 mm. Their literal 0.0
                # wedges fingers into our mat (H100 C1: 1.9% fingertips below the top ->
                # wrist chatter at eval gains).
                "z": (0.017, 0.5),
                "roll": (0.0, 0.0),
                "pitch": (np.pi / 4, 3 * np.pi / 4),
                "yaw": (np.pi / 2, 3 * np.pi / 2),
            },
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
        },
    )


@configclass
class ObjectRestingEEGraspedEventCfg(ResetStatesBaseEventCfg):
    reset_insertive_object_pose_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": ["ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
            # "table"/"ur5_metal_support" are kinematic AND their own init_state IS their real,
            # permanent pose here too -- ResetStatesSceneCfg.table (:77-84) and .ur5_metal_support
            # (:87-94) above, explicitly "duplicated [from rl_state_cfg.py] -- keep both in sync"
            # (:73-76). See MultiResetManager's coverage guard (omnireset/mdp/events.py,
            # _assert_reset_file_covers_scene) for why this must be an explicit claim, not inferred
            # from kinematic_enabled. In practice the "ObjectAnywhereEEAnywhere" files loaded here
            # are recorded by StableStateRecorder off THIS SAME ResetStatesSceneCfg (scene.get_state,
            # recorders.py:23), which dumps every rigid_object including untouched ones -- so both
            # names are already present in every such file today; this declaration is defense in
            # depth for a future dataset that drops one, not a fix for a currently-missing key.
            "assumed_static_assets": ["table", "ur5_metal_support"],
        },
    )

    reset_end_effector_pose_from_grasp_dataset = EventTerm(
        func=task_mdp.reset_end_effector_from_grasp_dataset,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "fixed_asset_cfg": SceneEntityCfg("insertive_object"),
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
            # 2F-85 joints (driver + passive linkage). Another gripper MUST replace this via
            # gripper_seam.override_gripper_joints -- see linear_gripper_cfg._apply_linear_gripper.
            "gripper_cfg": robotiq_2f85_gripper_joints(),
            "pose_range_b": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (-0.02, 0.02),
                "roll": (-np.pi / 16, np.pi / 16),
                "pitch": (-np.pi / 16, np.pi / 16),
                "yaw": (-np.pi / 16, np.pi / 16),
            },
        },
    )


@configclass
class ObjectAnywhereEEGraspedEventCfg(ResetStatesBaseEventCfg):
    reset_insertive_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                # x shifted +5 cm onto the green workspace mat (see reset_receptive_object_pose).
                "x": (0.35, 0.60),
                # y centered, same 0.40 m width (see reset_receptive_object_pose).
                "y": (-0.2, 0.2),
                "z": (0.0, 0.3),
                # PCB stays essentially top-up: only a small +/-0.1 rad (~6 deg) roll/pitch
                # jitter for robustness to placement error; yaw stays free (in-plane spin
                # is realistic and does not affect success, which checks roll+pitch only).
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-np.pi, np.pi),
            },
            "velocity_range": {},
            "asset_cfgs": {"insertive_object": SceneEntityCfg("insertive_object")},
            "offset_asset_cfg": SceneEntityCfg("ur5_metal_support"),
            "use_bottom_offset": True,
        },
    )

    reset_end_effector_pose_from_grasp_dataset = EventTerm(
        func=task_mdp.reset_end_effector_from_grasp_dataset,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "fixed_asset_cfg": SceneEntityCfg("insertive_object"),
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
            # 2F-85 joints (driver + passive linkage). Another gripper MUST replace this via
            # gripper_seam.override_gripper_joints -- see linear_gripper_cfg._apply_linear_gripper.
            "gripper_cfg": robotiq_2f85_gripper_joints(),
            "pose_range_b": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class ObjectPartiallyAssembledEEAnywhereEventCfg(ResetStatesBaseEventCfg):
    reset_insertive_object_pose_from_partial_assembly_dataset = EventTerm(
        func=task_mdp.reset_insertive_object_from_partial_assembly_dataset,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "pose_range_b": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_end_effector_pose = EventTerm(
        func=task_mdp.reset_end_effector_round_fixed_asset,
        mode="reset",
        params={
            "fixed_asset_cfg": SceneEntityCfg("robot"),
            "fixed_asset_offset": None,
            "pose_range_b": {
                "x": (0.3, 0.7),
                "y": (-0.4, 0.4),
                "z": (0.5, 0.5),
                "roll": (0.0, 0.0),
                "pitch": (np.pi / 4, 3 * np.pi / 4),
                "yaw": (np.pi / 2, 3 * np.pi / 2),
            },
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
        },
    )


@configclass
class ObjectPartiallyAssembledEEGraspedEventCfg(ResetStatesBaseEventCfg):
    reset_insertive_object_pose_from_partial_assembly_dataset = EventTerm(
        func=task_mdp.reset_insertive_object_from_partial_assembly_dataset,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "pose_range_b": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_end_effector_pose_from_grasp_dataset = EventTerm(
        func=task_mdp.reset_end_effector_from_grasp_dataset,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "fixed_asset_cfg": SceneEntityCfg("insertive_object"),
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
            # 2F-85 joints (driver + passive linkage). Another gripper MUST replace this via
            # gripper_seam.override_gripper_joints -- see linear_gripper_cfg._apply_linear_gripper.
            "gripper_cfg": robotiq_2f85_gripper_joints(),
            "pose_range_b": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
                "roll": (-np.pi / 32, np.pi / 32),
                "pitch": (-np.pi / 32, np.pi / 32),
                "yaw": (-np.pi / 32, np.pi / 32),
            },
        },
    )


@configclass
class ResetStatesTerminationCfg:
    """Configuration for reset states termination conditions."""

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)

    success = DoneTerm(
        func=task_mdp.check_reset_state_success,
        params={
            "object_cfgs": [SceneEntityCfg("insertive_object"), SceneEntityCfg("receptive_object")],
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_body_name": "robotiq_base_link",
            "collision_analyzer_cfgs": [
                task_mdp.CollisionAnalyzerCfg(
                    num_points=1024,
                    max_dist=0.5,
                    min_dist=-0.0005,
                    asset_cfg=SceneEntityCfg("robot"),
                    obstacle_cfgs=[SceneEntityCfg("insertive_object")],
                ),
                task_mdp.CollisionAnalyzerCfg(
                    num_points=1024,
                    max_dist=0.5,
                    min_dist=0.0,
                    asset_cfg=SceneEntityCfg("robot"),
                    obstacle_cfgs=[SceneEntityCfg("receptive_object")],
                ),
                task_mdp.CollisionAnalyzerCfg(
                    num_points=1024,
                    max_dist=0.5,
                    min_dist=-0.001,
                    asset_cfg=SceneEntityCfg("insertive_object"),
                    obstacle_cfgs=[SceneEntityCfg("receptive_object")],
                ),
            ],
            "max_robot_pos_deviation": 0.1,
            "max_object_pos_deviation": MISSING,
            "pos_z_threshold": -0.02,
            "consecutive_stability_steps": 5,
        },
        time_out=True,
    )


@configclass
class ResetStatesObservationsCfg:
    """Configuration for reset states observations."""

    pass


@configclass
class ResetStatesRewardsCfg:
    """Configuration for reset states rewards."""

    pass


def make_insertive_object(usd_path: str, override_mass: bool = True):
    """Build an insertive-object config, optionally preserving its authored mass."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001) if override_mass else None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_receptive_object(usd_path: str, disable_articulation_root: bool = False):
    """Build a receptive-object config, optionally disabling a baked-in articulation root.

    ``disable_articulation_root``: every receptive asset up to this one was authored directly as a
    plain rigid-body USD. An asset run through ``isaaclab.sim.converters.UrdfConverter`` with
    ``fix_base=True`` (as ``OneLegInsertionFixture`` was, see UWLab-3o5.3) gets an
    ArticulationRootAPI + a fixed ``root_joint`` to the world baked into the USD BY THE CONVERTER,
    even for a single-link fixture with no moving joints. ``RigidObjectCfg`` construction hard-fails
    against that ("Found an articulation root when resolving ... for rigid objects") -- IsaacLab's
    own error message names the fix: ``ArticulationRootPropertiesCfg.articulation_enabled = False``
    in the SPAWN CONFIG, not a USD edit. This parameter is that opt-in, off by default so every
    existing variant is byte-for-byte unaffected.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            articulation_props=(
                sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False)
                if disable_articulation_root
                else None
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


variants = {
    "scene.insertive_object": {
        "fbleg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareLeg/square_leg.usd"),
        "fbdrawerbottom": make_insertive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBottom/drawer_bottom.usd"
        ),
        "peg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd"),
        "cupcake": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/CupCake/cupcake.usd"),
        "cube": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/InsertiveCube/insertive_cube.usd"),
        "rectangle": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Rectangle/rectangle.usd"),
        # Local dev asset (PCB slab). Switch to UWLAB_CLOUD_ASSETS_DIR when sharing.
        "pcb": make_insertive_object(f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/Pcb/pcb.usd"),
        # Local dev asset (telescoping cover/lid). Switch to UWLAB_CLOUD_ASSETS_DIR when sharing.
        "cover": make_insertive_object(f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/Cover/cover.usd"),
        # DELTO-sized part: a 34 mm cube, 0.03 kg, inside the hand's validated grasp window.
        # Pairs with the "deltoslot" fixture.
        # override_mass=False, and here it MATTERS: unlike the reference props this asset carries
        # a MassAPI on its ROOT prim, so the default override really does apply and would replace
        # the deliberate 0.03 kg with 1 g -- the mass the hand's grip-force budget was sized
        # against. See make_insertive_object.
        "deltoblock": make_insertive_object(
            f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/DeltoBlock/delto_block.usd", override_mass=False
        ),
        # Our table-leg pair (bead UWLab-zvd.8), for the DELTO thread-insertion task. Pairs with the
        # "onelegfixture" receptive entry below. SAME shipped USD dexlift spawns
        # (dexlift_ur5e_delto_tableleg_env_cfg.py:TABLE_LEG_USD_PATH) -- SquareTableLeg200mmDecomp,
        # not the plain SquareTableLeg200mm sibling directory (that one is unused; do not swap it in
        # by mistake, the "Decomp" one is the certified-checkpoint one).
        #
        # override_mass=False IS THE TRAP THIS BEAD WAS FILED TO CATCH. The default
        # (override_mass=True, matching upstream's un-parameterized make_insertive_object) forces
        # mass_props to 0.001 kg regardless of what the USD authors -- a 1 g leg is flung by
        # ordinary contact, and a policy "grasping" it can look successful while holding nothing.
        # This leg's root prim (/square_table_leg4_200mm_merged) carries its own MassAPI at
        # 0.02275 kg (measured directly off the USD with pxr, not inferred from a script or a
        # config file); override_mass=False is what keeps that number instead of overwriting it.
        # disable_gravity is NOT set here or anywhere in make_insertive_object (it is hard-False,
        # same as every other variant) -- no gravity override applies to this object.
        "leg200mm": make_insertive_object(
            f"{UWLAB_LOCAL_ASSETS_DIR}/Props/FurnitureBench/SquareTableLeg200mmDecomp/square_table_leg4_200mm.usd",
            override_mass=False,
        ),
    },
    "scene.receptive_object": {
        "fbtabletop": make_receptive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareTableTop/square_table_top.usd"
        ),
        "fbdrawerbox": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBox/drawer_box.usd"),
        "peghole": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd"),
        "plate": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Plate/plate.usd"),
        "cube": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/ReceptiveCube/receptive_cube.usd"),
        "wall": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Wall/wall.usd"),
        # Local dev asset (open-top box). Switch to UWLAB_CLOUD_ASSETS_DIR when sharing.
        "openbox": make_receptive_object(f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/OpenBox/open_box.usd"),
        # Local dev asset (box with seated PCB; lid task receptive, mating point at the top rim).
        "boxwithpcb": make_receptive_object(f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/BoxWithPcb/box_with_pcb.usd"),
        # Receptive fixture for "deltoblock": 56 x 56 x 18 mm outer with a 36 x 36 mm pocket
        # 12 mm deep (1 mm clearance per side), pocket authored as five convex pieces. Kinematic,
        # so the mass override above is the usual no-op. Its metadata.yaml carries
        # success_thresholds, which a receptive object REQUIRES -- commands.TaskCommand reads
        # position/orientation from it with no default.
        "deltoslot": make_receptive_object(f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/DeltoSlot/delto_slot.usd"),
        # Receptive fixture for "leg200mm" (bead UWLab-zvd.8): the one-leg insertion fixture built
        # in UWLab-3o5.3 (source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/),
        # hence UWLAB_ASSETS_DATA_DIR rather than UWLAB_LOCAL_ASSETS_DIR -- a different asset root
        # than every other entry in this dict. Kinematic by construction (make_receptive_object
        # always sets kinematic_enabled=True), matching how upstream spawns its own tabletop.
        # metadata.yaml (authored separately, not by this bead) carries success_thresholds, which a
        # receptive object requires the same way "deltoslot" does.
        #
        # disable_articulation_root=True IS REQUIRED, NOT COSMETIC: the fixture was produced by
        # UrdfConverter with fix_base=True (UWLab-3o5.3), which bakes an ArticulationRootAPI +
        # fixed root_joint into the USD even for a single-link fixture. Without this flag,
        # RigidObjectCfg construction hard-fails ("Found an articulation root when resolving ...
        # for rigid objects") -- reproduced and confirmed while wiring this variant. See
        # make_receptive_object's docstring; the fix is a spawn-config parameter, not a USD edit.
        "onelegfixture": make_receptive_object(
            f"{UWLAB_ASSETS_DATA_DIR}/Props/FurnitureBench/OneLegInsertionFixture/one_leg_insertion_fixture.usd",
            disable_articulation_root=True,
        ),
    },
}


@configclass
class UR5eRobotiq2f85ResetStatesCfg(ManagerBasedRLEnvCfg):
    """Configuration for reset states environment with UR5e Robotiq 2F85 gripper."""

    scene: ResetStatesSceneCfg = ResetStatesSceneCfg(num_envs=1, env_spacing=1.5)
    events: ResetStatesBaseEventCfg = MISSING
    terminations: ResetStatesTerminationCfg = ResetStatesTerminationCfg()
    observations: ResetStatesObservationsCfg = ResetStatesObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: ResetStatesRewardsCfg = ResetStatesRewardsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

    def __post_init__(self):
        self.decimation = 12
        self.episode_length_s = 2.0
        # simulation settings
        self.sim.dt = 1 / 120.0

        # Contact and solver settings
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**31

        # Render settings
        self.sim.render.enable_dlssg = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True


@configclass
class ObjectAnywhereEEAnywhereResetStatesCfg(UR5eRobotiq2f85ResetStatesCfg):
    events: ObjectAnywhereEEAnywhereEventCfg = ObjectAnywhereEEAnywhereEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = np.inf


@configclass
class ObjectRestingEEGraspedResetStatesCfg(UR5eRobotiq2f85ResetStatesCfg):
    events: ObjectRestingEEGraspedEventCfg = ObjectRestingEEGraspedEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = 0.01


@configclass
class ObjectAnywhereEEGraspedResetStatesCfg(UR5eRobotiq2f85ResetStatesCfg):
    events: ObjectAnywhereEEGraspedEventCfg = ObjectAnywhereEEGraspedEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = 0.05


@configclass
class ObjectPartiallyAssembledEEAnywhereResetStatesCfg(UR5eRobotiq2f85ResetStatesCfg):
    events: ObjectPartiallyAssembledEEAnywhereEventCfg = ObjectPartiallyAssembledEEAnywhereEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = 0.025
        self.terminations.success.params["insertive_asset_cfg"] = SceneEntityCfg("insertive_object")
        self.terminations.success.params["receptive_asset_cfg"] = SceneEntityCfg("receptive_object")
        self.terminations.success.params["assembly_success_prob"] = 0.5
        self.terminations.success.params["assembly_threshold_scale"] = 1.5


@configclass
class ObjectPartiallyAssembledEEGraspedResetStatesCfg(UR5eRobotiq2f85ResetStatesCfg):
    events: ObjectPartiallyAssembledEEGraspedEventCfg = ObjectPartiallyAssembledEEGraspedEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = 0.025
        self.terminations.success.params["insertive_asset_cfg"] = SceneEntityCfg("insertive_object")
        self.terminations.success.params["receptive_asset_cfg"] = SceneEntityCfg("receptive_object")
        self.terminations.success.params["assembly_success_prob"] = 0.5
        self.terminations.success.params["assembly_threshold_scale"] = 1.5
