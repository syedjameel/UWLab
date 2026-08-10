"""Configuration for the Delto direct RL environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR
from uwlab_assets.robots.ur10e_delto import IMPLICIT_UR10E_DELTO


_OBJECT_PRIM_PATH_EXPR = "/World/envs/env_.*/Object"
_HAND_JOINT_EXPR = r"rj_dg_[1-5]_[1-4]"


def _robot_asset_cfg() -> ArticulationCfg:
    """Use the OmniReset articulation with the reference task's position controller."""
    asset = IMPLICIT_UR10E_DELTO.replace(prim_path="/World/envs/env_.*/Robot")
    # The shared asset's 500 s^-1 link damping is deliberate for OmniReset's
    # quasi-static grasp hold, but it effectively freezes a policy-controlled
    # 20-DoF hand.  This direct task keeps the same robot geometry and inertias
    # while using light damping: 5 s^-1 keeps randomized self-colliding starts
    # stable but still lets every finger track a 0.2 rad command.
    asset.spawn = asset.spawn.replace(
        activate_contact_sensors=True,
        rigid_props=asset.spawn.rigid_props.replace(
            # The committed OmniReset USD does not use the reference asset's
            # fixed-root convention, so its robot bodies must remain
            # gravity-disabled for the arm to hold the commanded posture.
            disable_gravity=True,
            linear_damping=5.0,
            angular_damping=5.0,
        ),
        articulation_props=asset.spawn.articulation_props.replace(
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
    )
    asset.actuators = {
        "arm": asset.actuators["arm"].replace(stiffness=800.0, damping=40.0),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=[_HAND_JOINT_EXPR],
            effort_limit_sim=30.0,
            velocity_limit_sim=10000.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    }
    return asset


@configclass
class RobotCfg:
    """Robot asset, joint layout, initial pose, and joint limits."""

    asset: ArticulationCfg = _robot_asset_cfg()
    arm_joint_names: tuple[str, ...] = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    # Collision-checked palm-down reset shared with the table-leg task.
    arm_start_pos: tuple[float, ...] = (
        -0.15457708,
        -1.32246149,
        1.09441864,
        0.30196786,
        1.40933454,
        -1.66120136,
    )
    hand_joint_names: tuple[str, ...] = (
        "rj_dg_1_1", "rj_dg_1_2", "rj_dg_1_3", "rj_dg_1_4",
        "rj_dg_2_1", "rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4",
        "rj_dg_3_1", "rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4",
        "rj_dg_4_1", "rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4",
        "rj_dg_5_1", "rj_dg_5_2", "rj_dg_5_3", "rj_dg_5_4",
    )


@configclass
class ObjectCfg:
    """Manipulated object assets and domain-randomization parameters."""

    asset: RigidObjectCfg = RigidObjectCfg(
        prim_path=_OBJECT_PRIM_PATH_EXPR,
        # Objects are spawned individually in setup_scene() so every environment
        # can use a different primitive and color.
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            # Match the reference task's 25--30 cm lateral approach rather
            # than dropping an object through the initially open fingers.
            pos=(1.10, -0.12, 0.35), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    randomize_shape: bool = True
    shape_names: tuple[str, ...] = ("cuboid", "sphere", "cylinder", "capsule")
    randomize_size: bool = True
    cuboid_size: tuple[float, float, float] = (0.07, 0.07, 0.1)
    cuboid_size_min: tuple[float, float, float] = (0.03, 0.03, 0.04)
    cuboid_size_max: tuple[float, float, float] = (0.1, 0.1, 0.12)
    sphere_radius: float = 0.05
    sphere_radius_range: tuple[float, float] = (0.025, 0.06)
    cylinder_radius: float = 0.04
    cylinder_height: float = 0.1
    cylinder_radius_range: tuple[float, float] = (0.02, 0.055)
    cylinder_height_range: tuple[float, float] = (0.03, 0.12)
    capsule_radius: float = 0.035
    capsule_height: float = 0.04
    capsule_radius_range: tuple[float, float] = (0.02, 0.045)
    capsule_height_range: tuple[float, float] = (0.01, 0.06)

    randomize_mass: bool = True
    mass_range: tuple[float, float] = (0.1, 0.4)
    default_mass: float = 0.2

    randomize_color: bool = True
    default_color: tuple[float, float, float] = (0.0, 1.0, 0.0)
    color_min: tuple[float, float, float] = (0.1, 0.1, 0.1)
    color_max: tuple[float, float, float] = (1.0, 1.0, 1.0)
    static_friction: float = 2.0
    dynamic_friction: float = 2.0
    restitution: float = 0.0

    randomize_friction: bool = True
    static_friction_range: tuple[float, float] = (0.5, 2.0)
    dynamic_friction_range: tuple[float, float] = (0.4, 1.5)
    restitution_range: tuple[float, float] = (0.0, 0.1)
    friction_num_buckets: int = 32
    fixed_half_height: float = 0.0
    fixed_dimensions: tuple[float, float, float] = (0.0, 0.0, 0.0)

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, kinematic_enabled=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=15.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.1, 0.1), metallic=0.1
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.85, 0.0, 0.235)),
    )


@configclass
class SensorCfg:
    """Fingertip contact sensors and policy point-cloud size."""

    fingertip_names: tuple[str, ...] = (
        "rl_dg_1_tip",
        "rl_dg_2_tip",
        "rl_dg_3_tip",
        "rl_dg_4_tip",
        "rl_dg_5_tip",
    )
    contact_threshold: float = 1.5
    point_cloud_size: int = 25
    contacts: dict[str, ContactSensorCfg] = {
        name: ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/gripper/{name}",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            # ``force_matrix_w`` will contain only fingertip-object contacts.
            filter_prim_paths_expr=[_OBJECT_PRIM_PATH_EXPR],
            track_air_time=False,
        )
        for name in fingertip_names
    }
@configclass
class ActionCfg:
    """Incremental joint-position action parameters."""

    scale: tuple[float, ...] = (0.05,) * 6 + (0.01,) * 20


@configclass
class ObservationCfg:
    """Policy observation history parameters."""

    history_length: int = 5
    flatten_history: bool = True
    asymmetric_critic: bool = True


@configclass
class RewardCfg:
    """Reward parameters."""

    approach_scale: float = 1.0
    approach_distance_scale: float = 0.4
    contact_scale: float = 0.3
    lift_scale: float = 3.0
    lift_distance_scale: float = 0.15
    lift_contact_power: float = 2.0
    # A discontinuous signed force reward made avoiding contact optimal for the
    # OmniReset Tesollo colliders.  Keep force telemetry, but do not reward or
    # penalize force magnitude directly.
    force_scale: float = 0.0
    force_threshold: float = 10.0
    action_rate_scale: float = 0.001
    action_scale: float = 0.001
    success_bonus: float = 30.0
    success_height: float = 0.4
    grasp_contact_min: int = 3
    hold_steps: int = 30


@configclass
class ResetCfg:
    """Episode reset randomization parameters."""

    joint_noise_deg: int = 10
    timeout_fraction_range: tuple[float, float] = (0.7, 1.0)
    object_xy_range: float = 0.15
    object_z_min: float = 0.0
    object_z_range: float = 0.2


@configclass
class TerminationCfg:
    """Object workspace limits [m]."""

    object_position_min: tuple[float, float, float] = (0.2, -0.8, 0.0)
    object_position_max: tuple[float, float, float] = (1.6, 0.8, 2.0)


@configclass
class CurriculumCfg:
    """Success-driven observation-noise and disturbance curriculum."""

    enabled: bool = True
    success_rate_threshold: float = 0.25
    success_consecutive_steps: int = 10
    level_cooldown_steps: int = 600
    max_level: int = 10

    joint_pos_noise_std_max: float = 0.02
    contact_force_noise_std_max: float = 1.0
    spatial_noise_std_max: float = 0.005

    external_force_max: float = 5.0
    external_force_probability_max: float = 0.25
    external_force_lift_height: float = 0.03


# =======================================================================


@configclass
class DeltoGraspEnvCfg(DirectRLEnvCfg):
    """Reference MDP using the shared OmniReset UR10e + Tesollo articulation."""

    decimation: int = 2
    episode_length_s: float = 5.0
    action_space: int = 26
    observation_space: int = 121 * 5
    # Derived from the policy observation and privileged-state layout at init.
    state_space: int = 121 * 5 + 30

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0
        ),
        # The 39-body OmniReset articulation retains hand self-collisions and
        # needs the full collision stack even at the production 512 envs/GPU.
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_patch_count=2**21,
            gpu_collision_stack_size=2**31,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32, env_spacing=3.0, replicate_physics=True
    )

    robot: RobotCfg = RobotCfg()
    object: ObjectCfg = ObjectCfg()
    sensors: SensorCfg = SensorCfg()
    action: ActionCfg = ActionCfg()
    observation: ObservationCfg = ObservationCfg()
    reward: RewardCfg = RewardCfg()
    reset: ResetCfg = ResetCfg()
    termination: TerminationCfg = TerminationCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class TableLegObjectCfg(ObjectCfg):
    """The 200 mm FurnitureBench leg used by the table-leg task."""

    asset: RigidObjectCfg = RigidObjectCfg(
        prim_path=_OBJECT_PRIM_PATH_EXPR,
        spawn=sim_utils.UrdfFileCfg(
            asset_path=(
                f"{UWLAB_LOCAL_ASSETS_DIR}/Props/FurnitureBench/SquareTableOneLeg/"
                "leg_200mm/square_table_leg4_200mm_matchedmass_sdf_hybrid.urdf"
            ),
            fix_base=False,
            joint_drive=None,
            merge_fixed_joints=True,
            collider_type="convex_decomposition",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.10, -0.12, 0.35), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    randomize_shape: bool = False
    shape_names: tuple[str, ...] = ("table_leg_200mm",)
    randomize_size: bool = False
    randomize_color: bool = False
    mass_range: tuple[float, float] = (0.04, 0.08)
    default_mass: float = 0.0573543915
    fixed_half_height: float = 0.015
    fixed_dimensions: tuple[float, float, float] = (0.2, 0.03, 0.03)


@configclass
class TableLegGraspEnvCfg(DeltoGraspEnvCfg):
    """Reference MDP with the randomized primitives replaced by the target leg."""

    object: TableLegObjectCfg = TableLegObjectCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep the calibrated palm-down reset exact.  Independent 10-degree
        # noise on all 26 joints can start the self-colliding hand in internal
        # contact and rotates the palm away from the airborne leg.
        self.reset.joint_noise_deg = 0
        # The falling-object task needs the complete five-second horizon for
        # approach, grasp, lift, and the required 30-step hold.  Random early
        # truncation can end otherwise valid acquisition attempts at 3.5 s.
        self.reset.timeout_fraction_range = (1.0, 1.0)
        # Every episode begins with a real free-fall phase above the table.
        self.reset.object_xy_range = 0.12
        self.reset.object_z_min = 0.05
        self.reset.object_z_range = 0.05
        # The URDF importer places the sole rigid body below the Object Xform.
        # Contact filtering must address that rigid body, not its parent.
        for sensor in self.sensors.contacts.values():
            sensor.filter_prim_paths_expr = [
                "/World/envs/env_.*/Object/base_link"
            ]


__all__ = [
    "ActionCfg",
    "CurriculumCfg",
    "DeltoGraspEnvCfg",
    "ObjectCfg",
    "ObservationCfg",
    "ResetCfg",
    "RewardCfg",
    "RobotCfg",
    "SensorCfg",
    "TerminationCfg",
    "TableLegGraspEnvCfg",
]
