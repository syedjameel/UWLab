# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the dexterous lifting task.

The dexsuite base package vendored in IsaacLab already provides ``action_l2_clamped``,
``action_rate_l2_clamped`` and ``object_ee_distance``; those are re-exported unchanged by
:mod:`.mdp`. Everything defined here is either absent from the vendored package or present
in a form that cannot be reused:

* ``contacts`` exists upstream but hardcodes the four Kuka-Allegro sensor names and exposes no
  parameters, so every contact-gated reward has to be reimplemented alongside it.
* ``success_reward`` exists upstream but is **not** contact gated at all. The gate matters: it is
  what stops the policy from collecting tracking reward while merely batting the object around.
* ``position_command_error_tanh`` and ``orientation_command_error_tanh`` exist upstream but call
  the hardcoded ``contacts``.

The contact-name arguments are deliberately required rather than defaulted. A hand that silently
falls back to another robot's sensor names would read zero force forever and quietly disable every
gate; a missing argument instead raises at manager construction time.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _sensor_force_magnitudes(env: ManagerBasedRLEnv, contact_names: list[str] | tuple[str, ...]) -> torch.Tensor:
    """Magnitude of the filtered contact force reported by each named fingertip sensor.

    Each name refers to a :class:`~isaaclab.sensors.ContactSensor` registered in the scene as
    ``f"{name}_object_s"`` and filtered to the object, so ``force_matrix_w`` holds a single
    contact pair per environment.

    Returns:
        Tensor of shape ``(num_envs, len(contact_names))``.
    """
    magnitudes = []
    for name in contact_names:
        sensor: ContactSensor = env.scene.sensors[f"{name}_object_s"]
        # A sensor may cover one rigid body (the original fingertip setup) or every
        # phalange of a finger (the FurnitureBench leg task).  Reduce all matched
        # body/filter pairs to the strongest force for that logical finger.
        force_w = sensor.data.force_matrix_w.reshape(env.num_envs, -1, 3)
        magnitudes.append(torch.norm(force_w, dim=-1).amax(dim=-1))
    return torch.stack(magnitudes, dim=-1)


def any_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    contact_names: tuple[str, ...],
) -> torch.Tensor:
    """Whether any of the listed fingertips presses on the object harder than ``threshold`` newtons."""
    return _sensor_force_magnitudes(env, contact_names).gt(threshold).any(dim=-1)


def contacts(
    env: ManagerBasedRLEnv,
    threshold: float,
    thumb_contact_name: str | list[str] | tuple[str, ...],
    tip_contact_names: tuple[str, ...],
    unwanted_contact_names: tuple[str, ...] | None = None,
    max_unwanted_contact_force: float = 0.05,
) -> torch.Tensor:
    """The opposition gate shared by every task reward below.

    True where at least one thumb **and** at least one non-thumb fingertip press on the object
    harder than ``threshold`` newtons, i.e. where the object is pinched rather than merely touched.
    The DELTO hand declares two opposable digits, so ``thumb_contact_name`` accepts a sequence.
    """
    if isinstance(thumb_contact_name, str):
        thumb_contact_name = [thumb_contact_name]
    thumb_contact = _sensor_force_magnitudes(env, thumb_contact_name).gt(threshold).any(dim=-1)
    tip_contact = _sensor_force_magnitudes(env, tip_contact_names).gt(threshold).any(dim=-1)
    valid = thumb_contact & tip_contact
    if unwanted_contact_names:
        unwanted_force = _sensor_force_magnitudes(env, unwanted_contact_names).amax(dim=-1)
        valid = valid & (unwanted_force <= max_unwanted_contact_force)
    return valid


def success_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    pos_std: float,
    thumb_contact_name: str | list[str] | tuple[str, ...],
    tip_contact_names: tuple[str, ...],
    rot_std: float | None = None,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward matching the commanded object pose, gated by contact once orientation counts.

    With ``rot_std`` unset (the Lift task) the position term is squared instead of being multiplied
    by an orientation term, which keeps the magnitude comparable to the Reorient task, and no
    contact gate applies.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, des_quat_w = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3], command[:, 3:7]
    )
    pos_err, rot_err = compute_pose_error(des_pos_w, des_quat_w, object.data.root_pos_w, object.data.root_quat_w)
    pos_dist = torch.norm(pos_err, dim=1)
    if not rot_std:
        return (1 - torch.tanh(pos_dist / pos_std)) ** 2
    rot_dist = torch.norm(rot_err, dim=1)
    return (
        (1 - torch.tanh(pos_dist / pos_std))
        * (1 - torch.tanh(rot_dist / rot_std))
        * contacts(env, threshold, thumb_contact_name, tip_contact_names).float()
    )


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    thumb_contact_name: str | list[str] | tuple[str, ...],
    tip_contact_names: tuple[str, ...],
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward tracking of the commanded object position with a tanh kernel, gated by contact."""
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * contacts(env, threshold, thumb_contact_name, tip_contact_names).float()


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    thumb_contact_name: str | list[str] | tuple[str, ...],
    tip_contact_names: tuple[str, ...],
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward tracking of the commanded object orientation with a tanh kernel, gated by contact.

    Note that the reference implementation hardcodes the gate at 1.0 N here while accepting a
    configurable threshold on the sibling position term. ``threshold`` is exposed so the two terms
    can be kept consistent; leaving it at the default reproduces the reference exactly.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = math_utils.quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    quat_distance = math_utils.quat_error_magnitude(object.data.root_quat_w, des_quat_w)
    return (1 - torch.tanh(quat_distance / std)) * contacts(
        env, threshold, thumb_contact_name, tip_contact_names
    ).float()


def finger_distance_tanh(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
    thumb_contact_name: str | list[str] | tuple[str, ...],
    tip_contact_names: tuple[str, ...],
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward keeping the fingertips spread apart while gripping, via the smallest pairwise distance."""
    asset: RigidObject = env.scene[asset_cfg.name]
    finger_tips_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    num_fingers = finger_tips_pos.shape[1]
    if num_fingers < 2:
        return torch.zeros(env.num_envs, device=env.device)

    dists = torch.cdist(finger_tips_pos, finger_tips_pos, p=2)
    # self-distances sit on the diagonal and would always win the minimum
    mask = torch.eye(num_fingers, device=env.device).bool().unsqueeze(0)
    dists = dists.masked_fill(mask, float("inf")).reshape(dists.shape[0], -1)
    min_dists = dists.min(dim=1).values

    return torch.tanh(min_dists / std) * contacts(env, threshold, thumb_contact_name, tip_contact_names).float()


def thumb2finger_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    thumb_asset_cfg: SceneEntityCfg,
    tip_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward keeping the thumbs away from the nearest other fingertip. Not contact gated.

    The reference implementation squeezes the thumb axis away before taking the minimum, which
    assumes a single thumb and returns a per-fingertip tensor instead of a per-environment one for
    a hand like the DELTO that declares two. Reducing over both axes keeps the single-thumb result
    identical while giving the two-thumb case the shape the reward manager expects.
    """
    thumb_asset: RigidObject = env.scene[thumb_asset_cfg.name]
    thumb_pos = thumb_asset.data.body_pos_w[:, thumb_asset_cfg.body_ids]

    tip_asset: RigidObject = env.scene[tip_asset_cfg.name]
    tip_pos = tip_asset.data.body_pos_w[:, tip_asset_cfg.body_ids]

    dists = torch.cdist(thumb_pos, tip_pos, p=2)
    min_dists = dists.flatten(start_dim=1).min(dim=1).values
    return torch.tanh(min_dists / std)


def table_contact_penalty(
    env: ManagerBasedRLEnv,
    thumb_contact_name: str | list[str] | tuple[str, ...],
    tip_contact_names: tuple[str, ...],
    table_contact_name: str = "table_s",
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize gripping the object while it is still resting on the table.

    Fires only when both conditions hold, so it discourages dragging rather than approaching.
    The gate here always runs at 1.0 N; ``threshold`` applies to the table-object force.
    """
    table_contact: ContactSensor = env.scene.sensors[table_contact_name]
    contact_force = table_contact.data.force_matrix_w.view(env.num_envs, 3)
    contact_mag = torch.norm(contact_force, dim=-1)
    contact_penalty = (contact_mag > threshold).float()
    return contact_penalty * contacts(env, 1.0, thumb_contact_name, tip_contact_names).float()


def object_upward_velocity_bonus(
    env: ManagerBasedRLEnv,
    std: float,
    thumb_contact_name: str | list[str] | tuple[str, ...],
    tip_contact_names: tuple[str, ...],
    threshold: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    unwanted_contact_names: tuple[str, ...] | None = None,
    max_unwanted_contact_force: float = 0.05,
) -> torch.Tensor:
    """Reward lifting the object while gripping it.

    Signed, so lowering a gripped object is penalized. This is the shaping that actually produces a
    lift; the base task has no explicit lift reward and relies on the gravity curriculum instead.
    """
    object: RigidObject = env.scene[object_cfg.name]
    vel_z = object.data.root_lin_vel_w[:, 2]
    reward = torch.tanh(vel_z / max(std, 1.0e-6))
    return reward * contacts(
        env,
        threshold,
        thumb_contact_name,
        tip_contact_names,
        unwanted_contact_names,
        max_unwanted_contact_force,
    ).float()
