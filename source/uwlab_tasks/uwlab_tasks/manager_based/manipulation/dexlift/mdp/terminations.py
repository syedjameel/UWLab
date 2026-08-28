# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the dexterous lifting task.

RECREATES a module this package used to have and deleted (see ``mdp/__init__.py``'s own comment on
that history) -- for an unrelated, new reason: the C4 seating-aware training variant (DELIVERABLE 2,
gated behind ``DEXLIFT_C4_GROSS_UNSEATING_TERM``, wired in
``dexlift_ur5e_delto_tableleg_env_cfg.py``'s ``_apply_c4_seating_training``) needs a termination
predicate the base dexsuite package has no equivalent of: "the object has drifted GROSSLY from the
goal command", which under ``DEXLIFT_GOAL_AT_SPAWN``/``DEXLIFT_PARTIAL_ASSEMBLY`` means "the leg has
been pulled far enough out of the fixture bore (or tilted far enough) that this episode cannot
recover a valid C4 seated state."
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gross_seating_loss(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    pos_threshold: float,
    rot_threshold: float,
) -> torch.Tensor:
    """True once the object's pose has drifted GROSSLY from the goal command.

    Reuses the exact same pose-error composition ``mdp.success_reward`` already does (``asset_cfg``
    is the frame the command is expressed in -- ``robot``; ``align_asset_cfg`` is the object being
    checked) -- not reimplemented, just evaluated against a much LOOSER pair of thresholds meant to
    catch "clearly gone", not "not perfectly still".

    Under ``DEXLIFT_GOAL_AT_SPAWN=1`` (implied by ``DEXLIFT_PARTIAL_ASSEMBLY=1``), the command is
    pinned to the object's own spawn -- i.e. seated -- pose for the whole episode (see
    ``partial_assembly.GoalAtSpawnPoseCommand`` / ``episode_mixture.MixtureGoalPoseCommand``), so
    "drifted from the goal" and "drifted from where it was inserted" are the same question. Calling
    this with any OTHER goal source (a uniformly resampled ``object_pose``) would terminate on
    proximity to an arbitrary point instead -- ``_apply_c4_seating_training`` asserts the
    precondition before ever wiring this in; this function itself does not check it (it has no way
    to know which command source is live, only what the command's current value is).

    NOT CONTACT-GATED, deliberately, unlike every reward term in this module. A termination that
    only fired while gripped would miss the episode where the policy grasped, pulled the leg mostly
    out, and then LOST the grip -- that episode is exactly as unrecoverable as one where the grip
    never breaks, and both should end the same way: early, not run out the clock.

    THIS IS A TRAINING-TIME SHAPING SIGNAL, NOT AN EVALUATION PREDICATE. See
    ``_apply_c4_seating_training``'s own docstring (`dexlift_ur5e_delto_tableleg_env_cfg.py`) for
    the full argument: whether a checkpoint actually holds seating is measured by
    ``generate_reset_states_policy.py --c4_seating_gate`` over many full generation episodes, never
    by how often (or how late) this termination fires during training.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, des_quat_w = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3], command[:, 3:7]
    )
    pos_err, rot_err = compute_pose_error(des_pos_w, des_quat_w, obj.data.root_pos_w, obj.data.root_quat_w)
    pos_dist = torch.norm(pos_err, dim=1)
    rot_dist = torch.norm(rot_err, dim=1)
    return (pos_dist > pos_threshold) | (rot_dist > rot_threshold)
