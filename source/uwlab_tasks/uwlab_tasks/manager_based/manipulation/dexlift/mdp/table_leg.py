# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Task terms for the FurnitureBench table-leg grasp and lift environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply_inverse,
    quat_conjugate,
    quat_error_magnitude,
    quat_mul,
)

from .rewards import _sensor_force_magnitudes, contacts

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def finger_contact_strengths(
    env: ManagerBasedRLEnv,
    contact_names: tuple[str, ...],
    clip: float = 5.0,
) -> torch.Tensor:
    """Object-contact force per named phalange, clipped in newtons."""
    return _sensor_force_magnitudes(env, contact_names).clamp(max=clip)


def logical_finger_contacts(
    env: ManagerBasedRLEnv,
    contact_groups: tuple[tuple[str, ...], ...],
    threshold: float,
) -> torch.Tensor:
    """Whether each logical finger has object contact on any of its phalanges."""
    return torch.stack(
        [_sensor_force_magnitudes(env, group).gt(threshold).any(dim=-1) for group in contact_groups],
        dim=-1,
    )


def finger_contact_count(
    env: ManagerBasedRLEnv,
    contact_groups: tuple[tuple[str, ...], ...],
    threshold: float,
    unwanted_contact_names: tuple[str, ...] | None = None,
    max_unwanted_contact_force: float = 0.05,
) -> torch.Tensor:
    """Number of distinct fingers touching the leg, independent of the contacted phalanx."""
    count = logical_finger_contacts(env, contact_groups, threshold).float().sum(dim=-1)
    if unwanted_contact_names:
        count = count * (max_finger_contact_force(env, unwanted_contact_names) <= max_unwanted_contact_force).float()
    return count


def pregrasp_arm_action_l2(
    env: ManagerBasedRLEnv,
    arm_action_dim: int,
    thumb_contact_name: str | tuple[str, ...],
    tip_contact_names: tuple[str, ...],
    threshold: float,
    unwanted_contact_names: tuple[str, ...] | None = None,
    max_unwanted_contact_force: float = 0.05,
) -> torch.Tensor:
    """Penalize arm motion until an opposed grasp frees the policy to lift."""
    opposed = contacts(
        env,
        threshold,
        thumb_contact_name,
        tip_contact_names,
        unwanted_contact_names,
        max_unwanted_contact_force,
    )
    arm_action = env.action_manager.action[:, :arm_action_dim]
    return arm_action.square().sum(dim=-1) * (~opposed).float()


def max_finger_contact_force(
    env: ManagerBasedRLEnv,
    contact_names: tuple[str, ...],
) -> torch.Tensor:
    """Maximum object-contact force across all observed phalanges."""
    return _sensor_force_magnitudes(env, contact_names).amax(dim=-1)


def unwanted_contact_force(
    env: ManagerBasedRLEnv,
    contact_names: tuple[str, ...],
    clip: float,
) -> torch.Tensor:
    """Clipped force from robot bodies that must not carry the object."""
    return max_finger_contact_force(env, contact_names).clamp(max=clip)


def root_height_above_table(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    table_top_offset: float = 0.02,
) -> torch.Tensor:
    """Object-root clearance above the static support table's top surface."""
    obj: RigidObject = env.scene[object_cfg.name]
    table: RigidObject = env.scene[table_cfg.name]
    return obj.data.root_pos_w[:, 2] - (table.data.root_pos_w[:, 2] + table_top_offset)


def object_velocity_b(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Object linear and angular velocity in the robot root frame."""
    obj: RigidObject = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    quat_w = robot.data.root_quat_w
    return torch.cat(
        (
            quat_apply_inverse(quat_w, obj.data.root_lin_vel_w),
            quat_apply_inverse(quat_w, obj.data.root_ang_vel_w),
        ),
        dim=-1,
    )


def target_position_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_name: str = "robot",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Distance from the object root to the commanded position in world coordinates."""
    robot = env.scene[robot_name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3]
    )
    return torch.linalg.vector_norm(obj.data.root_pos_w - target_pos_w, dim=-1)


def articulated_finger_count(
    env: ManagerBasedRLEnv,
    finger_joint_groups: tuple[tuple[str, ...], ...],
    minimum_joint_displacement: float,
    robot_name: str = "robot",
) -> torch.Tensor:
    """Count fingers with a positive flexion displacement from their reset posture."""
    robot = env.scene[robot_name]
    articulated = []
    for group in finger_joint_groups:
        joint_ids, _ = robot.find_joints(list(group), preserve_order=True)
        displacement = robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]
        articulated.append(displacement.clamp(min=0.0).amax(dim=-1) >= minimum_joint_displacement)
    return torch.stack(articulated, dim=-1).sum(dim=-1)


def target_position_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    contact_groups: tuple[tuple[str, ...], ...],
    minimum_contact_groups: int,
    contact_threshold: float,
    unwanted_contact_names: tuple[str, ...],
    max_unwanted_contact_force: float = 0.05,
) -> torch.Tensor:
    """Reward commanded-position tracking only for a finger-supported object."""
    contact_count = logical_finger_contacts(env, contact_groups, contact_threshold).sum(dim=-1)
    unwanted_force = max_finger_contact_force(env, unwanted_contact_names)
    valid_grasp = (contact_count >= minimum_contact_groups) & (
        unwanted_force <= max_unwanted_contact_force
    )
    error = target_position_error(env, command_name)
    return (1.0 - torch.tanh(error / std)) * valid_grasp.float()


class ActualLiftProgress(ManagerTermBase):
    """Contact-gated lift progress measured from each episode's lowest height."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._lowest_height = torch.full((env.num_envs,), torch.inf, device=env.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._lowest_height.fill_(torch.inf)
        else:
            self._lowest_height[env_ids] = torch.inf

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_height: float,
        thumb_contact_name: str | tuple[str, ...],
        tip_contact_names: tuple[str, ...],
        threshold: float,
        unwanted_contact_names: tuple[str, ...] | None = None,
        max_unwanted_contact_force: float = 0.05,
    ) -> torch.Tensor:
        height = root_height_above_table(env)
        self._lowest_height = torch.minimum(self._lowest_height, height)
        distance_to_target = (target_height - self._lowest_height).clamp(min=1.0e-6)
        progress = ((height - self._lowest_height) / distance_to_target).clamp(0.0, 1.0)
        return progress * contacts(
            env,
            threshold,
            thumb_contact_name,
            tip_contact_names,
            unwanted_contact_names,
            max_unwanted_contact_force,
        ).float()


def stable_grasp(
    env: ManagerBasedRLEnv,
    thumb_contact_name: str | tuple[str, ...],
    tip_contact_names: tuple[str, ...],
    threshold: float,
    max_object_speed: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    unwanted_contact_names: tuple[str, ...] | None = None,
    max_unwanted_contact_force: float = 0.05,
) -> torch.Tensor:
    """Opposition contact discounted when the leg is moving violently."""
    obj: RigidObject = env.scene[object_cfg.name]
    speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=-1)
    calm = (1.0 - (speed / max_object_speed).clamp(0.0, 1.0))
    return calm * contacts(
        env,
        threshold,
        thumb_contact_name,
        tip_contact_names,
        unwanted_contact_names,
        max_unwanted_contact_force,
    ).float()


def upward_velocity_until_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    thumb_contact_name: str | tuple[str, ...],
    tip_contact_names: tuple[str, ...],
    threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    unwanted_contact_names: tuple[str, ...] | None = None,
    max_unwanted_contact_force: float = 0.05,
) -> torch.Tensor:
    """Reward upward motion only until the held leg reaches the lift target."""
    obj: RigidObject = env.scene[object_cfg.name]
    below_target = root_height_above_table(env, object_cfg=object_cfg) < target_height
    reward = torch.tanh(obj.data.root_lin_vel_w[:, 2] / max(std, 1.0e-6))
    return reward * below_target.float() * contacts(
        env,
        threshold,
        thumb_contact_name,
        tip_contact_names,
        unwanted_contact_names,
        max_unwanted_contact_force,
    ).float()


def held_lift_stability(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    linear_speed_std: float,
    relative_speed_std: float,
    thumb_contact_name: str | tuple[str, ...],
    tip_contact_names: tuple[str, ...],
    threshold: float,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    unwanted_contact_names: tuple[str, ...] | None = None,
    max_unwanted_contact_force: float = 0.05,
) -> torch.Tensor:
    """Reward a calm finger-only hold after reaching the lift target."""
    obj: RigidObject = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    palm_id = robot_cfg.body_ids[0]
    object_speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=-1)
    relative_speed = torch.linalg.vector_norm(
        obj.data.root_lin_vel_w - robot.data.body_lin_vel_w[:, palm_id], dim=-1
    )
    calm = (1.0 - torch.tanh(object_speed / max(linear_speed_std, 1.0e-6))) * (
        1.0 - torch.tanh(relative_speed / max(relative_speed_std, 1.0e-6))
    )
    above_target = root_height_above_table(env, object_cfg=object_cfg) >= minimum_height
    return calm * above_target.float() * contacts(
        env,
        threshold,
        thumb_contact_name,
        tip_contact_names,
        unwanted_contact_names,
        max_unwanted_contact_force,
    ).float()


def excessive_object_speed(
    env: ManagerBasedRLEnv,
    max_speed: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Squared, bounded penalty above a safe object linear speed."""
    obj: RigidObject = env.scene[object_cfg.name]
    speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=-1)
    return ((speed - max_speed).clamp(min=0.0) / max_speed).square().clamp(max=25.0)


def excessive_horizontal_displacement(
    env: ManagerBasedRLEnv,
    center_x: float,
    center_y: float,
    free_radius: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Squared penalty once the leg moves materially away from its receiver."""
    obj: RigidObject = env.scene[object_cfg.name]
    # Asset root poses are expressed in world coordinates, while task workspace
    # constants are local to each replicated environment.
    object_pos_e = obj.data.root_pos_w - env.scene.env_origins
    dx = object_pos_e[:, 0] - center_x
    dy = object_pos_e[:, 1] - center_y
    distance = torch.sqrt(dx.square() + dy.square())
    return ((distance - free_radius).clamp(min=0.0) / free_radius).square().clamp(max=25.0)


def excessive_contact_force(
    env: ManagerBasedRLEnv,
    contact_names: tuple[str, ...],
    max_force: float,
) -> torch.Tensor:
    """Squared, bounded penalty for impact-like phalange/object contact."""
    force = max_finger_contact_force(env, contact_names)
    return ((force - max_force).clamp(min=0.0) / max_force).square().clamp(max=25.0)


def object_dropped(
    env: ManagerBasedRLEnv,
    minimum_height: float,
) -> torch.Tensor:
    """Terminate once the leg falls materially below the receptive table root."""
    return root_height_above_table(env) < minimum_height


class GraspPoseReward(ManagerTermBase):
    """Reward the complete palm-relative grasp pose, optionally gated by closing.

    Multiplying the position and orientation scores prevents the policy from
    collecting either reward while approaching the leg with an unusable wrist
    orientation.  The desired relative orientation is the collision-validated
    nominal spawn orientation.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        robot = env.scene[self.cfg.params.get("robot_name", "robot")]
        palm_body = self.cfg.params.get("palm_body", "rl_dg_mount")
        palm_ids, _ = robot.find_bodies(palm_body)
        if len(palm_ids) != 1:
            raise ValueError(f"Expected one palm body matching {palm_body!r}, found {palm_ids}")
        self._palm_body_id = palm_ids[0]
        desired_quat = self.cfg.params.get("desired_object_quat_p")
        position_only = self.cfg.params.get("position_only", False)
        if desired_quat is None and not position_only:
            raise ValueError("desired_object_quat_p is required for a full grasp-pose reward")
        self._desired_object_quat_p = torch.tensor(
            desired_quat or (1.0, 0.0, 0.0, 0.0), device=env.device
        ).repeat(env.num_envs, 1)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        desired_object_pos_p: tuple[float, float, float],
        position_std: float,
        orientation_std: float,
        desired_object_quat_p: tuple[float, float, float, float] | None = None,
        position_only: bool = False,
        unwanted_contact_names: tuple[str, ...] | None = None,
        max_unwanted_contact_force: float = 0.05,
        robot_name: str = "robot",
        object_name: str = "object",
        palm_body: str = "rl_dg_mount",
    ) -> torch.Tensor:
        del desired_object_quat_p, palm_body
        robot = env.scene[robot_name]
        object_asset = env.scene[object_name]
        palm_pos_w = robot.data.body_pos_w[:, self._palm_body_id]
        palm_quat_w = robot.data.body_quat_w[:, self._palm_body_id]

        object_pos_p = quat_apply_inverse(palm_quat_w, object_asset.data.root_pos_w - palm_pos_w)
        desired_pos = object_pos_p.new_tensor(desired_object_pos_p)
        position_error = torch.linalg.vector_norm(object_pos_p - desired_pos, dim=-1)
        position_score = 1.0 - torch.tanh(position_error / position_std)

        object_quat_p = quat_mul(quat_conjugate(palm_quat_w), object_asset.data.root_quat_w)
        orientation_error = quat_error_magnitude(object_quat_p, self._desired_object_quat_p)
        orientation_score = 1.0 - torch.tanh(orientation_error / orientation_std)
        score = position_score if position_only else position_score * orientation_score
        if unwanted_contact_names:
            valid_contact = max_finger_contact_force(env, unwanted_contact_names) <= max_unwanted_contact_force
            score = score * valid_contact.float()
        return score


class GraspPostureProgressReward(ManagerTermBase):
    """Reward closing toward a collision-validated hand posture at the grasp pose.

    Independent hand control removes the old binary ``close`` action, so contact
    rewards alone are too sparse to teach which of the 20 joints should flex.
    This term supplies that missing dense bridge while leaving every finger
    independently actuated.  Progress is measured only along joints that differ
    from the default posture and is gated by the palm-relative object pose.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        robot = env.scene[self.cfg.params.get("robot_name", "robot")]
        palm_body = self.cfg.params.get("palm_body", "rl_dg_mount")
        palm_ids, _ = robot.find_bodies(palm_body)
        if len(palm_ids) != 1:
            raise ValueError(f"Expected one palm body matching {palm_body!r}, found {palm_ids}")
        self._palm_body_id = palm_ids[0]

        target_joint_pos = self.cfg.params["target_joint_pos"]
        joint_ids, joint_names = robot.find_joints(list(target_joint_pos), preserve_order=True)
        if len(joint_ids) != len(target_joint_pos):
            raise ValueError(
                f"Expected {len(target_joint_pos)} grasp-posture joints, found {len(joint_ids)}: {joint_names}"
            )
        self._joint_ids = joint_ids
        self._target_joint_pos = torch.tensor(
            [target_joint_pos[name] for name in joint_names], device=env.device
        ).unsqueeze(0)
        default_joint_pos = robot.data.default_joint_pos[:, joint_ids]
        travel = self._target_joint_pos - default_joint_pos
        self._travel_direction = travel.sign()
        self._travel_magnitude = travel.abs().clamp(min=1.0e-6)
        self._moving_joints = travel.abs() > 1.0e-4

        desired_quat = self.cfg.params.get("desired_object_quat_p")
        self._desired_object_quat_p = torch.tensor(
            desired_quat or (1.0, 0.0, 0.0, 0.0), device=env.device
        ).repeat(env.num_envs, 1)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        desired_object_pos_p: tuple[float, float, float],
        target_joint_pos: dict[str, float],
        position_std: float,
        orientation_std: float,
        desired_object_quat_p: tuple[float, float, float, float] | None = None,
        unwanted_contact_names: tuple[str, ...] | None = None,
        max_unwanted_contact_force: float = 0.05,
        robot_name: str = "robot",
        object_name: str = "object",
        palm_body: str = "rl_dg_mount",
    ) -> torch.Tensor:
        del target_joint_pos, desired_object_quat_p, palm_body
        robot = env.scene[robot_name]
        object_asset = env.scene[object_name]

        joint_displacement = (
            robot.data.joint_pos[:, self._joint_ids]
            - robot.data.default_joint_pos[:, self._joint_ids]
        )
        # Retain a gradient for motion away from the target.  Clamping at zero
        # makes every wrong-direction command equally unrewarded and lets half
        # the independent fingers drift open without a corrective signal.
        joint_progress = (joint_displacement * self._travel_direction / self._travel_magnitude).clamp(-1.0, 1.0)
        posture_progress = (
            joint_progress.masked_select(self._moving_joints).reshape(env.num_envs, -1).mean(dim=-1)
        )

        palm_pos_w = robot.data.body_pos_w[:, self._palm_body_id]
        palm_quat_w = robot.data.body_quat_w[:, self._palm_body_id]
        object_pos_p = quat_apply_inverse(palm_quat_w, object_asset.data.root_pos_w - palm_pos_w)
        desired_pos = object_pos_p.new_tensor(desired_object_pos_p)
        position_error = torch.linalg.vector_norm(object_pos_p - desired_pos, dim=-1)
        position_score = 1.0 - torch.tanh(position_error / position_std)

        object_quat_p = quat_mul(quat_conjugate(palm_quat_w), object_asset.data.root_quat_w)
        orientation_error = quat_error_magnitude(object_quat_p, self._desired_object_quat_p)
        orientation_score = 1.0 - torch.tanh(orientation_error / orientation_std)
        score = posture_progress * position_score * orientation_score
        if unwanted_contact_names:
            valid_contact = max_finger_contact_force(env, unwanted_contact_names) <= max_unwanted_contact_force
            score = score * valid_contact.float()
        return score


class SuccessDifficultyScheduler(ManagerTermBase):
    """Adaptive two-stage difficulty driven by the real held-lift termination.

    A failure demotes four times as much as a success promotes.  Away from the
    bounds, an 80% success rate is therefore the fixed point.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        initial = float(self.cfg.params.get("initial_level", 0.0))
        self.levels = torch.full((env.num_envs,), initial, device=env.device)
        self.mean_level = initial
        self.difficulty_frac = torch.tensor(
            initial / max(float(self.cfg.params.get("max_level", 20.0)), 1.0), device=env.device
        )

    def get_state(self) -> torch.Tensor:
        return self.levels

    def set_state(self, state: torch.Tensor) -> None:
        self.levels = state.clone().to(self._env.device)
        self.mean_level = float(self.levels.mean().item())
        max_level = max(float(self.cfg.params.get("max_level", 20.0)), 1.0)
        self.difficulty_frac = self.levels.mean() / max_level

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | torch.Tensor,
        success_term: str = "success",
        initial_level: float = 0.0,
        min_level: float = 0.0,
        max_level: float = 20.0,
        promotion_step: float = 0.25,
        demotion_step: float = 1.0,
    ) -> float:
        del initial_level
        if int(env.common_step_counter) > 0 and len(env_ids) > 0:
            success = env.termination_manager.get_term(success_term)[env_ids]
            delta = torch.where(
                success,
                torch.full_like(self.levels[env_ids], promotion_step),
                torch.full_like(self.levels[env_ids], -demotion_step),
            )
            self.levels[env_ids] = (self.levels[env_ids] + delta).clamp(min=min_level, max=max_level)
        mean_level = self.levels.mean()
        self.mean_level = float(mean_level.item())
        # DexSuite's interpolation helper calls ``.item()`` on expressions
        # involving this value, so preserve its scalar-tensor contract.
        self.difficulty_frac = mean_level / max(max_level, 1.0)
        return self.mean_level


def _adaptive_stage_progress(env: ManagerBasedRLEnv, difficulty_term: str, start: float, end: float) -> float:
    scheduler: SuccessDifficultyScheduler = getattr(env.curriculum_manager.cfg, difficulty_term).func
    return min(max((scheduler.mean_level - start) / (end - start), 0.0), 1.0)


def adaptive_gravity_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor,
    event_term_name: str,
    difficulty_term: str,
    stage_start: float,
    stage_end: float,
    final_gravity: float = -9.81,
) -> float:
    """Map adaptive difficulty levels onto zero-to-Earth gravity."""
    del env_ids
    progress = _adaptive_stage_progress(env, difficulty_term, stage_start, stage_end)
    gravity = final_gravity * progress
    term_cfg = env.event_manager.get_term_cfg(event_term_name)
    term_cfg.params["gravity_distribution_params"] = (
        (0.0, 0.0, gravity),
        (0.0, 0.0, gravity),
    )
    env.event_manager.set_term_cfg(event_term_name, term_cfg)
    return progress


def adaptive_object_pose_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor,
    event_term_name: str,
    difficulty_term: str,
    stage_start: float,
    stage_mid: float,
    stage_end: float,
    full_pose_range: dict[str, tuple[float, float]],
) -> float:
    """Widen translation first, then orientation, after gravity is mastered."""
    del env_ids
    translation_progress = _adaptive_stage_progress(env, difficulty_term, stage_start, stage_mid)
    orientation_progress = _adaptive_stage_progress(env, difficulty_term, stage_mid, stage_end)
    term_cfg = env.event_manager.get_term_cfg(event_term_name)
    term_cfg.params["pose_range"] = {
        axis: [
            float(bounds[0]) * (translation_progress if axis in ("x", "y", "z") else orientation_progress),
            float(bounds[1]) * (translation_progress if axis in ("x", "y", "z") else orientation_progress),
        ]
        for axis, bounds in full_pose_range.items()
    }
    env.event_manager.set_term_cfg(event_term_name, term_cfg)
    return _adaptive_stage_progress(env, difficulty_term, stage_start, stage_end)


class SustainedLiftSuccess(ManagerTermBase):
    """True only after the hand carries a multi-finger grasp through a real lift."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self._lowest_height = torch.full((env.num_envs,), torch.inf, device=env.device)
        robot = env.scene[self.cfg.params.get("robot_name", "robot")]
        palm_body = self.cfg.params.get("palm_body", "rl_dg_mount")
        palm_ids, _ = robot.find_bodies(palm_body)
        if len(palm_ids) != 1:
            raise ValueError(f"Expected one palm body matching {palm_body!r}, found {palm_ids}")
        self._palm_body_id = palm_ids[0]
        self._finger_joint_ids = [
            robot.find_joints(list(group), preserve_order=True)[0]
            for group in self.cfg.params["finger_joint_groups"]
        ]

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._counter.zero_()
            self._lowest_height.fill_(torch.inf)
        else:
            self._counter[env_ids] = 0
            self._lowest_height[env_ids] = torch.inf

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        minimum_height: float,
        hold_steps: int,
        command_name: str,
        position_tolerance: float,
        contact_groups: tuple[tuple[str, ...], ...],
        finger_joint_groups: tuple[tuple[str, ...], ...],
        minimum_contact_groups: int,
        thumb_contact_names: tuple[str, ...],
        tip_contact_names: tuple[str, ...],
        minimum_articulated_fingers: int,
        minimum_joint_displacement: float,
        contact_threshold: float,
        unwanted_contact_names: tuple[str, ...],
        max_unwanted_contact_force: float,
        max_object_speed: float,
        minimum_lift: float,
        max_palm_distance: float,
        max_relative_speed: float,
        minimum_episode_steps: int,
        palm_body: str = "rl_dg_mount",
        robot_name: str = "robot",
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        del palm_body
        obj: RigidObject = env.scene[object_cfg.name]
        robot = env.scene[robot_name]
        height = root_height_above_table(env)
        self._lowest_height = torch.minimum(self._lowest_height, height)
        speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=-1)
        palm_pos = robot.data.body_pos_w[:, self._palm_body_id]
        palm_velocity = robot.data.body_lin_vel_w[:, self._palm_body_id]
        palm_distance = torch.linalg.vector_norm(obj.data.root_pos_w - palm_pos, dim=-1)
        relative_speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w - palm_velocity, dim=-1)
        position_error = target_position_error(env, command_name, robot_name, object_cfg)
        contact_count = logical_finger_contacts(env, contact_groups, contact_threshold).sum(dim=-1)
        opposed_contact = contacts(
            env,
            contact_threshold,
            thumb_contact_names,
            tip_contact_names,
            unwanted_contact_names,
            max_unwanted_contact_force,
        )
        del finger_joint_groups
        articulated_count = torch.stack(
            [
                (
                    robot.data.joint_pos[:, joint_ids]
                    - robot.data.default_joint_pos[:, joint_ids]
                )
                .clamp(min=0.0)
                .amax(dim=-1)
                >= minimum_joint_displacement
                for joint_ids in self._finger_joint_ids
            ],
            dim=-1,
        ).sum(dim=-1)
        unwanted_force = max_finger_contact_force(env, unwanted_contact_names)
        valid = (
            (height >= minimum_height)
            & ((height - self._lowest_height) >= minimum_lift)
            & (position_error <= position_tolerance)
            & (speed <= max_object_speed)
            & (palm_distance <= max_palm_distance)
            & (relative_speed <= max_relative_speed)
            & (contact_count >= minimum_contact_groups)
            & opposed_contact
            & (articulated_count >= minimum_articulated_fingers)
            & (unwanted_force <= max_unwanted_contact_force)
            & (env.episode_length_buf >= minimum_episode_steps)
        )
        self._counter = torch.where(valid, self._counter + 1, torch.zeros_like(self._counter))
        return self._counter >= hold_steps
