# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task functions for the reference-style UR10e + Tesollo grasp task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import (
    sample_object_point_cloud,
)

if TYPE_CHECKING:
    from .delto_grasp_env import DeltoGraspEnv
    from .delto_grasp_env_cfg import DeltoGraspEnvCfg

    DeltoEnv = DeltoGraspEnv
    DeltoEnvCfg = DeltoGraspEnvCfg


def compute_observation_dim(cfg: DeltoEnvCfg) -> int | tuple[int, int]:
    """Return the flattened policy-observation dimension."""
    joint_dim = cfg.action_space
    force_dim = len(cfg.sensors.fingertip_names)
    fingertip_position_dim = 3 * force_dim
    point_cloud_dim = 3 * cfg.sensors.point_cloud_size
    single_observation_dim = (
        joint_dim + force_dim + fingertip_position_dim + point_cloud_dim
    )
    if cfg.observation.flatten_history:
        return single_observation_dim * cfg.observation.history_length
    return (cfg.observation.history_length, single_observation_dim)


def compute_state_dim(cfg: DeltoEnvCfg) -> int:
    """Return the asymmetric critic-state dimension."""
    if not cfg.observation.asymmetric_critic:
        return 0
    policy_dim = compute_observation_dim(cfg)
    if not isinstance(policy_dim, int):
        raise ValueError("The asymmetric critic requires flattened observation history.")
    privileged_dim = 26 + len(cfg.object.shape_names)
    return policy_dim + privileged_dim


def setup_scene(env: DeltoEnv) -> None:
    """Create and register the assets and sensors used by the task."""
    spawn_ground_plane("/World/ground", GroundPlaneCfg())

    env.hand = Articulation(env.cfg.robot.asset)
    env.scene.articulations["hand"] = env.hand
    env.table = RigidObject(env.cfg.object.table)
    env.scene.rigid_objects["table"] = env.table

    env.scene.clone_environments(copy_from_source=False)
    if env.device == "cpu":
        env.scene.filter_collisions(global_prim_paths=[])

    _spawn_objects(env)
    env.object = RigidObject(env.cfg.object.asset)
    env.scene.rigid_objects["object"] = env.object

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    for name, sensor_cfg in env.cfg.sensors.contacts.items():
        env.scene.sensors[name] = ContactSensor(sensor_cfg)

    if env.cfg.object.asset.spawn is None:
        env.point_cloud = sample_object_point_cloud(
            num_envs=env.scene.num_envs,
            num_points=env.cfg.sensors.point_cloud_size,
            prim_path_pattern="/World/envs/env_.*/Object",
            device=env.device,
        )
        if env.point_cloud is None:
            raise RuntimeError("The manipulated object has no collider point cloud.")
    else:
        env.point_cloud = fixed_cuboid_point_cloud(
            env.cfg.object.fixed_dimensions,
            env.cfg.sensors.point_cloud_size,
            env.scene.num_envs,
            env.device,
        )


def fixed_cuboid_point_cloud(
    dimensions: tuple[float, float, float],
    num_points: int,
    num_envs: int,
    device: str,
) -> torch.Tensor:
    """Return deterministic surface points for the fixed 200 mm leg body."""
    axis = torch.linspace(-0.5, 0.5, 3, device=device)
    points = torch.cartesian_prod(axis, axis, axis)
    points = points[(points.abs() == 0.5).any(dim=1)]
    if num_points > points.shape[0]:
        repeats = (num_points + points.shape[0] - 1) // points.shape[0]
        points = points.repeat(repeats, 1)
    scale = torch.tensor(dimensions, device=device)
    points = points[:num_points] * scale
    return points.unsqueeze(0).expand(num_envs, -1, -1).clone()


def object_point_cloud_w(env: DeltoEnv) -> torch.Tensor:
    """Transform the cached object-local point cloud into world coordinates."""
    count = env.point_cloud.shape[1]
    object_pos = env.object.data.body_link_pose_w[:, 0, :3].unsqueeze(1)
    object_quat = env.object.data.body_link_pose_w[:, 0, 3:].unsqueeze(1)
    return quat_apply(object_quat.expand(-1, count, -1), env.point_cloud) + object_pos


def _spawn_objects(env: DeltoEnv) -> None:
    """Spawn one randomized primitive in every environment."""
    cfg = env.cfg.object
    num_shapes = len(cfg.shape_names)
    if num_shapes == 0:
        raise ValueError("ObjectCfg.shape_names must contain at least one primitive.")

    # A configured file spawner is instantiated by ``RigidObject`` below.  We
    # only need to provide the shape metadata otherwise produced by the
    # heterogeneous primitive spawner.
    if cfg.asset.spawn is not None:
        env.object_shape_ids = torch.zeros(env.scene.num_envs, device=env.device)
        env.object_half_heights = torch.full(
            (env.scene.num_envs,), cfg.fixed_half_height, device=env.device
        )
        env.object_dimensions = torch.tensor(
            cfg.fixed_dimensions, device=env.device
        ).repeat(env.scene.num_envs, 1)
        return

    if cfg.randomize_shape:
        shape_ids = torch.randint(num_shapes, (env.scene.num_envs,)).tolist()
    else:
        shape_ids = [0] * env.scene.num_envs

    if cfg.randomize_color:
        color_min = torch.tensor(cfg.color_min)
        color_max = torch.tensor(cfg.color_max)
        colors = color_min + (color_max - color_min) * torch.rand(
            (env.scene.num_envs, 3)
        )
    else:
        colors = torch.tensor(cfg.default_color).repeat(env.scene.num_envs, 1)

    spawn_cfgs = []
    half_heights = []
    dimensions = []
    for env_id, shape_id in enumerate(shape_ids):
        shape_name = cfg.shape_names[shape_id]
        spawn_cfg, half_height, shape_dimensions = _make_object_spawn_cfg(
            cfg, shape_name, tuple(colors[env_id].tolist())
        )
        spawn_cfgs.append(spawn_cfg)
        half_heights.append(half_height)
        dimensions.append(shape_dimensions)

    # Spawn all heterogeneous objects through one regex operation.  This is
    # compatible with replicated physics in current Isaac Lab; editing env_0
    # and then manually creating every object duplicates inherited prims.
    multi_spawn_cfg = sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=spawn_cfgs, random_choice=False
    )
    multi_spawn_cfg.func("/World/envs/env_.*/Object", multi_spawn_cfg)

    env.object_shape_ids = torch.tensor(shape_ids, device=env.device)
    env.object_half_heights = torch.tensor(half_heights, device=env.device)
    env.object_dimensions = torch.tensor(dimensions, device=env.device)


def _make_object_spawn_cfg(cfg, shape_name: str, color: tuple[float, float, float]):
    """Create a randomized primitive spawner and return its half-height [m]."""
    common = {
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        "physics_material": sim_utils.RigidBodyMaterialCfg(
            static_friction=cfg.static_friction,
            dynamic_friction=cfg.dynamic_friction,
            restitution=cfg.restitution,
        ),
        "mass_props": sim_utils.MassPropertiesCfg(mass=cfg.default_mass),
        "collision_props": sim_utils.CollisionPropertiesCfg(),
        "visual_material": sim_utils.PreviewSurfaceCfg(
            diffuse_color=color, metallic=0.1
        ),
    }
    if shape_name == "cuboid":
        size = (
            _sample_vector(cfg.cuboid_size_min, cfg.cuboid_size_max)
            if cfg.randomize_size
            else cfg.cuboid_size
        )
        return sim_utils.CuboidCfg(size=size, **common), 0.5 * size[2], size
    if shape_name == "sphere":
        radius = (
            _sample_scalar(cfg.sphere_radius_range)
            if cfg.randomize_size
            else cfg.sphere_radius
        )
        dimensions = (2.0 * radius,) * 3
        return sim_utils.SphereCfg(radius=radius, **common), radius, dimensions
    if shape_name == "cylinder":
        radius = (
            _sample_scalar(cfg.cylinder_radius_range)
            if cfg.randomize_size
            else cfg.cylinder_radius
        )
        height = (
            _sample_scalar(cfg.cylinder_height_range)
            if cfg.randomize_size
            else cfg.cylinder_height
        )
        return (
            sim_utils.CylinderCfg(radius=radius, height=height, axis="Z", **common),
            0.5 * height,
            (2.0 * radius, 2.0 * radius, height),
        )
    if shape_name == "capsule":
        radius = (
            _sample_scalar(cfg.capsule_radius_range)
            if cfg.randomize_size
            else cfg.capsule_radius
        )
        height = (
            _sample_scalar(cfg.capsule_height_range)
            if cfg.randomize_size
            else cfg.capsule_height
        )
        return (
            sim_utils.CapsuleCfg(radius=radius, height=height, axis="Z", **common),
            radius + 0.5 * height,
            (2.0 * radius, 2.0 * radius, height + 2.0 * radius),
        )
    raise ValueError(f"Unsupported primitive in ObjectCfg.shape_names: {shape_name!r}")


def _sample_scalar(value_range: tuple[float, float]) -> float:
    """Sample one scalar uniformly from a configured range."""
    lower, upper = value_range
    if lower <= 0.0 or upper < lower:
        raise ValueError(f"Invalid positive range: {value_range}")
    return lower + (upper - lower) * torch.rand(()).item()


def _sample_vector(
    lower: tuple[float, float, float], upper: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Sample a three-dimensional vector uniformly component-wise."""
    lower_tensor = torch.tensor(lower)
    upper_tensor = torch.tensor(upper)
    if (lower_tensor <= 0.0).any() or (upper_tensor < lower_tensor).any():
        raise ValueError(f"Invalid positive vector range: lower={lower}, upper={upper}")
    sample = lower_tensor + (upper_tensor - lower_tensor) * torch.rand(3)
    return tuple(sample.tolist())


def allocate_buffers(env: DeltoEnv) -> None:
    """Resolve scene handles and allocate task tensors."""
    env.env_origins = env.scene.env_origins
    env.hand_joint_ids, _ = env.hand.find_joints(env.cfg.robot.hand_joint_names)
    env.arm_joint_ids, _ = env.hand.find_joints(env.cfg.robot.arm_joint_names)
    env.fingertip_body_ids = [
        env.hand.body_names.index(name) for name in env.cfg.sensors.fingertip_names
    ]

    env.joint_pos = env.hand.data.joint_pos
    env.joint_vel = env.hand.data.joint_vel
    # Isaac Lab 2.3 exposes parsed rigid-body defaults directly on the data
    # object rather than through the removed ``body_mass/body_inertia`` views.
    env.object_default_mass = env.object.data.default_mass.to(env.device).clone()
    env.object_default_inertia = env.object.data.default_inertia.to(env.device).clone()
    env.object_mass = env.object_default_mass.clone()
    env.object_material_properties = torch.tensor(
        (
            env.cfg.object.static_friction,
            env.cfg.object.dynamic_friction,
            env.cfg.object.restitution,
        ),
        device=env.device,
    ).repeat(env.num_envs, 1)
    env.object_material_buckets = _create_material_buckets(env)
    # PhysX rigid-body/material setters perform a full CPU tensor round-trip.
    # Give every environment one independently sampled parameter set, but do
    # not repeat that expensive operation on every episodic reset.
    env.object_mass_initialized = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.bool
    )
    env.object_friction_initialized = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.bool
    )
    env.curriculum_level = 0
    env.curriculum_success_streak = 0
    env.curriculum_last_update_step = -env.cfg.curriculum.level_cooldown_steps
    env.object_external_force = torch.zeros((env.num_envs, 3), device=env.device)

    # Joint coordinate conventions and ordering come from the committed
    # OmniReset articulation, not from the reference repository's older USD.
    env.robot_start_position = env.hand.data.default_joint_pos[0].clone()
    env.robot_start_position[env.arm_joint_ids] = torch.tensor(
        env.cfg.robot.arm_start_pos, device=env.device
    )
    env.robot_lower_limits = env.hand.data.soft_joint_pos_limits[0, :, 0].clone()
    env.robot_upper_limits = env.hand.data.soft_joint_pos_limits[0, :, 1].clone()
    env.action_scale = torch.tensor(env.cfg.action.scale, device=env.device).unsqueeze(0)

    shape = (env.num_envs, env.cfg.action_space)
    env.raw_actions = torch.zeros(shape, device=env.device)
    env.raw_prev_actions = torch.zeros_like(env.raw_actions)
    env.target_pos = env.robot_start_position.repeat(env.num_envs, 1)
    env.per_env_timeout = torch.full(
        (env.num_envs,), env.max_episode_length, device=env.device, dtype=torch.long
    )
    env.success_hold_counter = torch.zeros(
        (env.num_envs,), device=env.device, dtype=torch.long
    )

    single_obs_dim = (
        env.cfg.action_space
        + 4 * len(env.cfg.sensors.fingertip_names)
        + 3 * env.cfg.sensors.point_cloud_size
    )
    env.observation_history = torch.zeros(
        (env.num_envs, env.cfg.observation.history_length, single_obs_dim),
        device=env.device,
    )
    env.critic_observation_history = torch.zeros_like(env.observation_history)


def _create_material_buckets(env: DeltoEnv) -> torch.Tensor:
    """Pre-sample a finite set of PhysX material parameters on the CPU."""
    cfg = env.cfg.object
    if cfg.friction_num_buckets < 1:
        raise ValueError("ObjectCfg.friction_num_buckets must be positive.")

    ranges = torch.tensor(
        (
            cfg.static_friction_range,
            cfg.dynamic_friction_range,
            cfg.restitution_range,
        ),
        dtype=torch.float32,
    )
    if (ranges[:, 0] < 0.0).any() or (ranges[:, 1] < ranges[:, 0]).any():
        raise ValueError(
            "Object material randomization ranges must be non-negative and ordered."
        )

    buckets = ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * torch.rand(
        (cfg.friction_num_buckets, 3)
    )
    buckets[:, 1] = torch.minimum(buckets[:, 0], buckets[:, 1])
    return buckets


def process_actions(env: DeltoEnv, actions: torch.Tensor) -> None:
    """Convert normalized actions to bounded joint-position targets."""
    env.raw_prev_actions.copy_(env.raw_actions)
    env.raw_actions.copy_(actions)
    env.target_pos.add_(env.action_scale * actions)
    env.target_pos.clamp_(env.robot_lower_limits, env.robot_upper_limits)
    _apply_curriculum_force(env)


def _apply_curriculum_force(env: DeltoEnv) -> None:
    """Apply a level-scaled random force to lifted objects."""
    cfg = env.cfg.curriculum
    level_scale = env.curriculum_level / max(cfg.max_level, 1)
    forces = torch.zeros((env.num_envs, 1, 3), device=env.device)

    if cfg.enabled and level_scale > 0.0:
        table_top_height = (
            env.cfg.object.table.init_state.pos[2]
            + 0.5 * env.cfg.object.table.spawn.size[2]
        )
        object_height = (
            env.object.data.body_com_pos_w[:, 0, 2]
            - env.env_origins[:, 2]
            - table_top_height
            - env.object_half_heights
        )
        lifted = object_height > cfg.external_force_lift_height
        selected = lifted & (
            torch.rand(env.num_envs, device=env.device)
            < cfg.external_force_probability_max * level_scale
        )

        directions = torch.randn((env.num_envs, 3), device=env.device)
        directions /= directions.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
        magnitudes = (
            cfg.external_force_max
            * level_scale
            * torch.rand((env.num_envs, 1), device=env.device)
        )
        forces[:, 0] = directions * magnitudes * selected.unsqueeze(1)

    body_ids = torch.tensor([0], device=env.device, dtype=torch.int32)
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
    env.object.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        body_ids=body_ids,
        env_ids=env_ids,
        is_global=True,
    )
    env.object_external_force.copy_(forces[:, 0])


def apply_actions(env: DeltoEnv) -> None:
    """Write the current joint-position targets to the articulation."""
    env.hand.set_joint_position_target(env.target_pos)


def read_contacts(env: DeltoEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return object-only fingertip forces, contact flags, and positions [m]."""
    forces = []
    for name in env.cfg.sensors.fingertip_names:
        # Shape: (num_envs, sensor_bodies, filtered_bodies, 3). The sensor
        # configuration contains only the manipulated object in its filter.
        force_matrix = env.scene.sensors[name].data.force_matrix_w
        if force_matrix is None:
            raise RuntimeError(
                f"Contact sensor '{name}' has no filtered force matrix. "
                "Set filter_prim_paths_expr to the manipulated object path."
            )
        object_force = force_matrix.sum(dim=(1, 2))
        forces.append(torch.linalg.vector_norm(object_force, dim=-1))

    force_magnitudes = torch.stack(forces, dim=1)
    contact_flags = force_magnitudes > env.cfg.sensors.contact_threshold
    fingertip_pos = (
        env.hand.data.body_pos_w[:, env.fingertip_body_ids] - env.env_origins.unsqueeze(1)
    )
    return force_magnitudes, contact_flags, fingertip_pos


def build_observations(env: DeltoEnv) -> dict[str, torch.Tensor]:
    """Build the policy observation and append it to the history window."""
    forces, _, fingertip_pos = read_contacts(env)
    points = object_point_cloud_w(env) - env.env_origins.unsqueeze(1)
    clean_observation = torch.cat(
        (
            env.joint_pos.float(),
            forces.float(),
            fingertip_pos.float().flatten(start_dim=1),
            points.float().flatten(start_dim=1),
        ),
        dim=1,
    )
    noisy_observation = _add_curriculum_observation_noise(
        env, env.joint_pos.float(), forces.float(), fingertip_pos.float(), points.float()
    )

    env.observation_history = torch.roll(env.observation_history, shifts=-1, dims=1)
    env.critic_observation_history = torch.roll(
        env.critic_observation_history, shifts=-1, dims=1
    )
    env.observation_history[:, -1] = noisy_observation
    env.critic_observation_history[:, -1] = clean_observation
    if env.cfg.observation.flatten_history:
        policy_observation = env.observation_history.flatten(start_dim=1)
        critic_observation = env.critic_observation_history.flatten(start_dim=1)
    else:
        policy_observation = env.observation_history
        critic_observation = env.critic_observation_history
    observations = {"policy": policy_observation}
    if env.cfg.observation.asymmetric_critic:
        observations["critic"] = torch.cat(
            (critic_observation, _build_privileged_state(env)), dim=1
        )
    return observations


def _add_curriculum_observation_noise(
    env: DeltoEnv,
    joint_pos: torch.Tensor,
    forces: torch.Tensor,
    fingertip_pos: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """Add level-scaled Gaussian noise to actor observations."""
    cfg = env.cfg.curriculum
    level_scale = env.curriculum_level / max(cfg.max_level, 1)
    if not cfg.enabled or level_scale == 0.0:
        return torch.cat(
            (joint_pos, forces, fingertip_pos.flatten(1), points.flatten(1)), dim=1
        )

    joint_pos = joint_pos + torch.randn_like(joint_pos) * (
        cfg.joint_pos_noise_std_max * level_scale
    )
    forces = forces + torch.randn_like(forces) * (
        cfg.contact_force_noise_std_max * level_scale
    )
    fingertip_pos = fingertip_pos + torch.randn_like(fingertip_pos) * (
        cfg.spatial_noise_std_max * level_scale
    )
    points = points + torch.randn_like(points) * (
        cfg.spatial_noise_std_max * level_scale
    )
    return torch.cat(
        (joint_pos, forces, fingertip_pos.flatten(1), points.flatten(1)), dim=1
    )


def _build_privileged_state(env: DeltoEnv) -> torch.Tensor:
    """Build simulator-only object state and domain metadata for the critic."""
    object_pos = env.object.data.body_com_pos_w[:, 0, :3] - env.env_origins
    object_quat = env.object.data.body_com_quat_w[:, 0]
    object_lin_vel = env.object.data.body_com_lin_vel_w[:, 0]
    object_ang_vel = env.object.data.body_com_ang_vel_w[:, 0]
    shape = torch.nn.functional.one_hot(
        env.object_shape_ids.long(), num_classes=len(env.cfg.object.shape_names)
    ).float()
    episode_progress = (
        env.episode_length_buf.float() / env.max_episode_length
    ).unsqueeze(1)
    timeout_fraction = (
        env.per_env_timeout.float() / env.max_episode_length
    ).unsqueeze(1)
    curriculum_level = torch.full(
        (env.num_envs, 1),
        env.curriculum_level / max(env.cfg.curriculum.max_level, 1),
        device=env.device,
    )
    return torch.cat(
        (
            object_pos,
            object_quat,
            object_lin_vel,
            object_ang_vel,
            env.object_mass,
            shape,
            env.object_dimensions,
            env.object_material_properties,
            episode_progress,
            timeout_fraction,
            curriculum_level,
            env.object_external_force,
        ),
        dim=1,
    )


def compute_rewards(env: DeltoEnv) -> torch.Tensor:
    """Compute the grasping and lifting reward for every environment."""
    cfg = env.cfg.reward
    forces, contact_flags, fingertip_pos = read_contacts(env)
    object_pos = env.object.data.body_com_pos_w[:, 0, :3] - env.env_origins
    object_distance = (fingertip_pos - object_pos.unsqueeze(1)).norm(dim=2).amax(dim=1)
    approach = cfg.approach_scale * (
        1.0 - torch.tanh(object_distance / cfg.approach_distance_scale)
    )

    contact_count = contact_flags.sum(dim=1)
    contact_reward = cfg.contact_scale * contact_count
    table_top_height = (
        env.cfg.object.table.init_state.pos[2]
        + 0.5 * env.cfg.object.table.spawn.size[2]
    )
    initial_height = table_top_height + env.object_half_heights
    lifted_height = (object_pos[:, 2] - initial_height).clamp_min(0.0)
    # A hard three-contact gate left PPO with no gradient between reaching and
    # a complete grasp.  Smoothly increase lift credit as more fingertips make
    # object-only contact; the success predicate below remains strictly gated.
    contact_fraction = (
        contact_count.float() / cfg.grasp_contact_min
    ).clamp(max=1.0)
    lift_reward = (
        cfg.lift_scale
        * torch.tanh(lifted_height / cfg.lift_distance_scale)
        * contact_fraction.pow(cfg.lift_contact_power)
    )

    max_force = forces.amax(dim=1)
    force_term = cfg.force_scale * torch.where(
        max_force > cfg.force_threshold, -max_force, max_force
    )
    action_rate_penalty = -cfg.action_rate_scale * (
        env.raw_prev_actions - env.raw_actions
    ).square().sum(dim=1).clamp_max(1000.0)
    action_penalty = (
        -cfg.action_scale * env.raw_actions.square().sum(dim=1).clamp_max(1000.0)
    )

    success = (object_pos[:, 2] >= cfg.success_height) & (
        contact_count >= cfg.grasp_contact_min
    )
    env.success_hold_counter = torch.where(
        success, env.success_hold_counter + 1, torch.zeros_like(env.success_hold_counter)
    )
    held_success = env.success_hold_counter >= cfg.hold_steps
    _update_curriculum(env, success)
    success_bonus = cfg.success_bonus * success.float()
    reward = (
        approach
        + contact_reward
        + lift_reward
        + force_term
        + action_rate_penalty
        + action_penalty
        + success_bonus
    )

    log = env.extras.setdefault("log", {})
    log["Reward/approach"] = approach.mean().item()
    log["Reward/contact"] = contact_reward.mean().item()
    log["Reward/lift"] = lift_reward.mean().item()
    log["Reward/force"] = force_term.mean().item()
    log["Reward/action_rate_penalty"] = action_rate_penalty.mean().item()
    log["Reward/action_penalty"] = action_penalty.mean().item()
    log["Reward/success_bonus"] = success_bonus.mean().item()
    log["Reward/total"] = reward.mean().item()
    log["Metrics/contact_count_mean"] = contact_count.float().mean().item()
    log["Metrics/contact_force_mean"] = max_force.mean().item()
    log["Metrics/contact_force_p95"] = torch.quantile(max_force, 0.95).item()
    log["Metrics/contact_force_over_threshold"] = (
        max_force > cfg.force_threshold
    ).float().mean().item()
    log["Metrics/lifted_height_mean"] = lifted_height.mean().item()
    log["Metrics/held_success_rate"] = held_success.float().mean().item()
    return reward


def _update_curriculum(env: DeltoEnv, success: torch.Tensor) -> None:
    """Advance curriculum after a sustained population-level success rate."""
    cfg = env.cfg.curriculum
    success_rate = success.float().mean().item()
    if cfg.enabled and env.curriculum_level < cfg.max_level:
        if success_rate >= cfg.success_rate_threshold:
            env.curriculum_success_streak += 1
        else:
            env.curriculum_success_streak = 0

        step = env._sim_step_counter // env.cfg.decimation
        cooldown_finished = (
            step - env.curriculum_last_update_step >= cfg.level_cooldown_steps
        )
        if (
            env.curriculum_success_streak >= cfg.success_consecutive_steps
            and cooldown_finished
        ):
            env.curriculum_level += 1
            env.curriculum_success_streak = 0
            env.curriculum_last_update_step = step

    log = env.extras.setdefault("log", {})
    log["Metrics/success_rate"] = success_rate
    log["Curriculum/level"] = env.curriculum_level
    log["Curriculum/noise_scale"] = env.curriculum_level / max(cfg.max_level, 1)
    log["Curriculum/external_force_mean"] = (
        env.object_external_force.norm(dim=1).mean().item()
    )


def compute_terminations(env: DeltoEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute out-of-bounds termination and per-environment timeout."""
    object_pos = env.object.data.body_com_pos_w[:, 0, :3] - env.env_origins
    lower = torch.tensor(env.cfg.termination.object_position_min, device=env.device)
    upper = torch.tensor(env.cfg.termination.object_position_max, device=env.device)
    terminated = ((object_pos < lower) | (object_pos > upper)).any(dim=1)
    truncated = env.episode_length_buf >= env.per_env_timeout
    return terminated, truncated


def reset_envs(env: DeltoEnv, env_ids: torch.Tensor) -> None:
    """Reset robot, task buffers, and object poses for selected environments."""
    count = env_ids.numel()
    noise_deg = env.cfg.reset.joint_noise_deg
    joint_noise = torch.randint(
        -noise_deg, noise_deg + 1, (count, env.cfg.action_space), device=env.device
    )
    joint_pos = env.robot_start_position + torch.deg2rad(joint_noise.float())
    joint_pos.clamp_(env.robot_lower_limits, env.robot_upper_limits)
    joint_vel = torch.zeros_like(joint_pos)

    env.target_pos[env_ids] = joint_pos
    env.raw_actions[env_ids] = 0.0
    env.raw_prev_actions[env_ids] = 0.0
    env.observation_history[env_ids] = 0.0
    env.critic_observation_history[env_ids] = 0.0
    env.object_external_force[env_ids] = 0.0
    env.success_hold_counter[env_ids] = 0

    low_fraction, high_fraction = env.cfg.reset.timeout_fraction_range
    low = int(low_fraction * env.max_episode_length)
    high = int(high_fraction * env.max_episode_length)
    env.per_env_timeout[env_ids] = torch.randint(
        low, high + 1, (count,), device=env.device
    )

    env.hand.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    env.hand.set_joint_position_target(joint_pos, env_ids=env_ids)
    _randomize_object_mass(env, env_ids)
    _randomize_object_friction(env, env_ids)
    _reset_object(env, env_ids)


def _randomize_object_mass(env: DeltoEnv, env_ids: torch.Tensor) -> None:
    """Randomize each object's mass [kg] and inertia once at initialization."""
    cfg = env.cfg.object
    if not cfg.randomize_mass:
        return

    env_ids = env_ids[~env.object_mass_initialized[env_ids]]
    count = env_ids.numel()
    if count == 0:
        return
    mass_min, mass_max = cfg.mass_range
    if mass_min <= 0.0 or mass_max < mass_min:
        raise ValueError(f"Invalid ObjectCfg.mass_range: {cfg.mass_range}")

    masses = mass_min + (mass_max - mass_min) * torch.rand(
        (count, 1), device=env.device
    )
    physics_env_ids = env_ids.to(device="cpu", dtype=torch.int32)
    all_masses = env.object.root_physx_view.get_masses()
    all_masses[physics_env_ids] = masses.to(device="cpu")
    env.object.root_physx_view.set_masses(all_masses, physics_env_ids)
    env.object_mass[env_ids] = masses

    default_mass = env.object_default_mass[env_ids]
    inertia = env.object_default_inertia[env_ids] * (masses / default_mass)
    all_inertias = env.object.root_physx_view.get_inertias()
    all_inertias[physics_env_ids] = inertia.to(device="cpu")
    env.object.root_physx_view.set_inertias(all_inertias, physics_env_ids)
    env.object_mass_initialized[env_ids] = True


def _randomize_object_friction(env: DeltoEnv, env_ids: torch.Tensor) -> None:
    """Assign each object bucketed material properties once at initialization."""
    cfg = env.cfg.object
    if not cfg.randomize_friction:
        return

    env_ids = env_ids[~env.object_friction_initialized[env_ids]]
    if env_ids.numel() == 0:
        return
    env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
    num_shapes = env.object.root_physx_view.max_shapes
    bucket_ids = torch.randint(
        0,
        cfg.friction_num_buckets,
        (env_ids.numel(), num_shapes),
        device="cpu",
    )
    samples = env.object_material_buckets[bucket_ids]

    materials = env.object.root_physx_view.get_material_properties()
    materials[env_ids_cpu] = samples
    env.object.root_physx_view.set_material_properties(
        materials, env_ids_cpu.to(torch.int32)
    )
    env.object_material_properties[env_ids] = samples[:, 0].to(env.device)
    env.object_friction_initialized[env_ids] = True


def _reset_object(env: DeltoEnv, env_ids: torch.Tensor) -> None:
    """Randomize object position and yaw for selected environments."""
    count = env_ids.numel()
    root_state = env.object.data.default_root_state[env_ids].clone()
    xy_range = env.cfg.reset.object_xy_range
    root_state[:, :2] += 2.0 * xy_range * (torch.rand((count, 2), device=env.device) - 0.5)
    table_top_height = (
        env.cfg.object.table.init_state.pos[2]
        + 0.5 * env.cfg.object.table.spawn.size[2]
    )
    root_state[:, 2] = (
        table_top_height
        + env.object_half_heights[env_ids]
        + env.cfg.reset.object_z_min
        + env.cfg.reset.object_z_range * torch.rand((count,), device=env.device)
    )

    half_yaw = torch.pi * torch.rand((count,), device=env.device)
    root_state[:, 3:7] = 0.0
    root_state[:, 3] = torch.cos(half_yaw)
    root_state[:, 6] = torch.sin(half_yaw)
    root_state[:, 7:13] = 0.0
    root_state[:, :3] += env.env_origins[env_ids]
    env.object.write_root_state_to_sim(root_state, env_ids)
