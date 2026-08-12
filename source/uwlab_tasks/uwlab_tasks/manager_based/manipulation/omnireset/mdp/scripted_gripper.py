# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""How an OFFLINE RECORDER tells an environment's gripper to open or close.

The two dataset recorders -- ``scripts_v2/tools/record_grasps.py`` and
``scripts_v2/tools/record_reset_states.py`` -- are scripted rollouts, not policies. Both need to
hold the gripper shut while the environment shakes the object and decides whether the grasp
survived. Neither trains anything, so neither is a "training path"; what they are is the PRODUCER
for the datasets a trainable environment resets from, which is why getting this wrong is silent
and expensive rather than loud.

WHAT WAS WRONG. Both scripts wrote the close command into the LAST action dimension
(``actions[:, -1] = -1.0``, and ``-torch.ones(...)`` respectively). That is correct for exactly one
kind of gripper: a ``BinaryJointAction``, which occupies one dimension and follows Isaac Lab's
convention of negative = close. On the fully actuated DELTO hand the last dimension is
``rj_dg_5_4`` alone, and -1 commands 0.1 rad of EXTENSION on it per step -- the opposite of
closing, on one joint of twenty, with the other nineteen left at zero. Nothing would have raised:
the buffer is sized from ``env.action_space.shape``, so the shape is right and the meaning is
wrong. Every regenerated DELTO ``*EEGrasped`` dataset would have been recorded with an open hand.

WHAT THIS MODULE DOES INSTEAD. It asks the environment's own action terms what "close" means for
them, and returns a FULL-WIDTH action vector -- zeros for the arm, the gripper's command in the
gripper's own slice, at the gripper's own indices:

* a ``BinaryJointAction`` term takes -1 to close and +1 to open (Isaac Lab's convention, read off
  the term's class, not assumed from the action width);
* a per-joint term (one action dimension per joint, e.g. the DELTO's twenty) is SERVOED, joint by
  joint and BY NAME, toward a target posture: ``clamp((target - measured) / scale, lo, hi)``. There
  is no scalar and no shared command anywhere in that -- all twenty dimensions are written
  independently, every step, which is also why it must be recomputed each step rather than filled
  into a buffer once.

The postures come from the gripper asset (``uwlab_assets.robots.ur10e_delto.actions``), keyed by
joint NAME. Position-keyed lookup is impossible here on purpose: this YAML-and-metadata family is
finger-major while the articulation reports joints level-major, and the two orders differ.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointAction

from uwlab_assets.robots.ur10e_delto.actions import (
    DELTO_HAND_OPEN_JOINT_POS,
    DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS,
)

# Joint name -> (open angle, closed angle), for every gripper whose action term is per-joint and
# therefore has no open/close command of its own. Keyed by JOINT NAME, so a gripper is recognised
# by the joints its action term resolved to; ``rj_dg_*`` exists on no other robot in this tree.
#
# A gripper missing from here fails LOUDLY in :func:`scripted_gripper_actions` rather than being
# driven with zeros, which for a relative action term means "hold wherever you are" -- i.e. an
# open, non-grasping hand recorded as a grasp.
_PER_JOINT_POSTURES: dict[str, tuple[float, float]] = {
    name: (float(DELTO_HAND_OPEN_JOINT_POS[name]), float(DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS[name]))
    for name in DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS
}


def _term_scale(term, num_joints: int, device) -> torch.Tensor:
    """Per-joint action scale of a joint action term, as a (num_envs, num_joints) tensor."""
    scale = getattr(term, "_scale", 1.0)
    if isinstance(scale, torch.Tensor):
        return scale
    return torch.full((term.num_envs, num_joints), float(scale), device=device)


def _clip_bounds(term, num_joints: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The term's own action clip, or +-inf when it declares none."""
    clip = getattr(term, "_clip", None)
    if isinstance(clip, torch.Tensor):
        return clip[..., 0], clip[..., 1]
    inf = torch.full((term.num_envs, num_joints), float("inf"), device=device)
    return -inf, inf


def _per_joint_command(term, close: bool, device) -> torch.Tensor:
    """Servo command driving every joint of ``term`` toward its open or closed posture."""
    joint_names: list[str] = list(getattr(term, "_joint_names", []))
    unknown = [name for name in joint_names if name not in _PER_JOINT_POSTURES]
    if unknown:
        raise ValueError(
            f"No scripted open/closed posture is declared for gripper joint(s) {unknown}, which the"
            f" per-joint action term '{type(term).__name__}' drives. The offline recorders cannot"
            " command this gripper shut. Declare the postures in"
            " omnireset/mdp/scripted_gripper.py::_PER_JOINT_POSTURES, keyed by joint name."
            "\nDriving the term with zeros instead is NOT a fallback: a relative joint action reads"
            " zero as 'hold the measured position', so every recorded grasp would be an open hand."
        )
    index = 1 if close else 0
    target = torch.tensor([_PER_JOINT_POSTURES[name][index] for name in joint_names], device=device)
    measured = term._asset.data.joint_pos[:, term._joint_ids]
    scale = _term_scale(term, len(joint_names), device)
    lo, hi = _clip_bounds(term, len(joint_names), device)
    # Recomputed from the MEASURED position every step, so the command saturates against contact
    # and keeps a standing position error -- a sustained squeeze rather than a one-shot target.
    return torch.clamp((target.unsqueeze(0) - measured) / scale, lo, hi)


def scripted_gripper_actions(env, close: bool) -> torch.Tensor:
    """Full-width action vector that opens or closes this environment's gripper.

    Args:
        env: The (unwrapped) manager-based env. Its action manager is read, not its config.
        close: ``True`` for the close command, ``False`` for open.

    Returns:
        ``(num_envs, total_action_dim)``. Every non-gripper term (an OSC arm, for instance) is left
        at zero.

    Raises:
        ValueError: if no gripper term is found, or a per-joint gripper has no declared posture.
    """
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    offset = 0
    found = False
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        width = term.action_dim
        joint_names = list(getattr(term, "_joint_names", []))
        if isinstance(term, BinaryJointAction):
            # Isaac Lab's convention, taken from the class rather than from the action width.
            actions[:, offset : offset + width] = -1.0 if close else 1.0
            found = True
        elif joint_names and width == len(joint_names) and joint_names[0] in _PER_JOINT_POSTURES:
            actions[:, offset : offset + width] = _per_joint_command(term, close, env.device)
            found = True
        offset += width
    if not found:
        raise ValueError(
            "No gripper action term was recognised on this environment, so there is no way to"
            f" command it {'shut' if close else 'open'}. Action terms:"
            f" {list(env.action_manager.active_terms)}. A binary gripper is recognised by its"
            " BinaryJointAction class; a fully actuated hand by its joint names appearing in"
            " omnireset/mdp/scripted_gripper.py::_PER_JOINT_POSTURES."
        )
    return actions


def scripted_gripper_actions_for(env, close_mask: torch.Tensor) -> torch.Tensor:
    """Per-environment mix of the open and close commands, selected by ``close_mask``.

    Used by the reset-state recorder's non-grasped types, which deliberately record both open and
    closed end-effector states.
    """
    close = scripted_gripper_actions(env, close=True)
    open_ = scripted_gripper_actions(env, close=False)
    return torch.where(close_mask.view(-1, 1).to(torch.bool), close, open_)


def is_gripper_posture_declared(joint_names: Mapping[str, object] | list[str]) -> bool:
    """Whether every named joint has a declared scripted posture. For tooling diagnostics."""
    return all(name in _PER_JOINT_POSTURES for name in joint_names)
