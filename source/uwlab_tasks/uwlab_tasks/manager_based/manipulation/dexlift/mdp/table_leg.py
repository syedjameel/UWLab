# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Task terms for the FurnitureBench table-leg grasp and lift environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

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
) -> torch.Tensor:
    """Number of distinct fingers touching the leg, independent of the contacted phalanx."""
    return logical_finger_contacts(env, contact_groups, threshold).float().sum(dim=-1)


def max_finger_contact_force(
    env: ManagerBasedRLEnv,
    contact_names: tuple[str, ...],
) -> torch.Tensor:
    """Maximum object-contact force across all observed phalanges."""
    return _sensor_force_magnitudes(env, contact_names).amax(dim=-1)


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


def lift_progress(
    env: ManagerBasedRLEnv,
    start_height: float,
    target_height: float,
    thumb_contact_name: str | tuple[str, ...],
    tip_contact_names: tuple[str, ...],
    threshold: float,
) -> torch.Tensor:
    """Normalized lift progress, paid only while an opposing grasp is present."""
    height = root_height_above_table(env)
    progress = ((height - start_height) / (target_height - start_height)).clamp(0.0, 1.0)
    return progress * contacts(env, threshold, thumb_contact_name, tip_contact_names).float()


def stable_grasp(
    env: ManagerBasedRLEnv,
    thumb_contact_name: str | tuple[str, ...],
    tip_contact_names: tuple[str, ...],
    threshold: float,
    max_object_speed: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Opposition contact discounted when the leg is moving violently."""
    obj: RigidObject = env.scene[object_cfg.name]
    speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=-1)
    calm = (1.0 - (speed / max_object_speed).clamp(0.0, 1.0))
    return calm * contacts(env, threshold, thumb_contact_name, tip_contact_names).float()


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
    """True only after a multi-finger, contact-held lift persists for a fixed window."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._counter.zero_()
        else:
            self._counter[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        minimum_height: float,
        hold_steps: int,
        contact_groups: tuple[tuple[str, ...], ...],
        minimum_contact_groups: int,
        contact_threshold: float,
        max_object_speed: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        obj: RigidObject = env.scene[object_cfg.name]
        speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=-1)
        contact_count = logical_finger_contacts(env, contact_groups, contact_threshold).sum(dim=-1)
        valid = (
            (root_height_above_table(env) >= minimum_height)
            & (speed <= max_object_speed)
            & (contact_count >= minimum_contact_groups)
        )
        self._counter = torch.where(valid, self._counter + 1, torch.zeros_like(self._counter))
        return self._counter >= hold_steps
