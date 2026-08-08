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


def object_pose_reset_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor,
    event_term_name: str,
    full_pose_range: dict[str, tuple[float, float]],
    warmup_steps: int,
    ramp_steps: int,
) -> float:
    """Smoothly widen object reset offsets from centered to the evaluation range."""
    del env_ids
    if ramp_steps <= 0:
        raise ValueError("ramp_steps must be positive")
    progress = min(max((int(env.common_step_counter) - warmup_steps) / ramp_steps, 0.0), 1.0)
    term_cfg = env.event_manager.get_term_cfg(event_term_name)
    term_cfg.params["pose_range"] = {
        axis: [float(bounds[0]) * progress, float(bounds[1]) * progress]
        for axis, bounds in full_pose_range.items()
    }
    env.event_manager.set_term_cfg(event_term_name, term_cfg)
    return progress


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
