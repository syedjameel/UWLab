# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaaclab-free helpers for the reset-state PD-target schema (bead UWLab-algw.7).

THE DEFECT THIS SCHEMA CLOSES. A saved reset state (``StableStateRecorder``, ``recorders.py``)
used to carry only ``root_pose``/``root_velocity``/``joint_position``/``joint_velocity`` per
articulation -- the CONFIGURATION, not what the PD actuator was actually commanded to hold it at.
Applied effort on a PD joint is proportional to ``(joint_position_target - joint_position)``, and
replay (``MultiResetManager._reset_to``, ``events.py``) used to set ``target := q`` -- exactly
zero commanded squeeze, regardless of what was true at capture. Measured gap at the accepted step
(two independent runs): 20 DELTO hand joints median 0.155-0.161 rad, p90 0.33, max 0.82; 6 arm
joints median 0.016 rad. A four-arm sweep over the SAME 34 states then isolated target as the
fix: restoring it took 26/34 held (today's target := q convention) to 34/34 -- restoring/zeroing
velocity made no difference on its own.

WHY THIS LIVES IN ITS OWN MODULE, NOT ``recorders.py``/``events.py`` DIRECTLY. Both of those files
import ``carb``/``isaaclab``/``omni``/``warp``/``pxr`` at module scope and cannot be imported
outside a running Isaac process. The three functions below touch nothing but plain dicts and
``torch`` tensors -- pulling them out here is what makes bead UWLab-algw.7's schema logic (as
opposed to the physics that reads/writes it) unit-testable on CPU, with no Isaac import, via
``test_reset_state_schema.py``.
"""

from __future__ import annotations

import torch

JOINT_POSITION_TARGET_KEY = "joint_position_target"
JOINT_VELOCITY_TARGET_KEY = "joint_velocity_target"


def add_joint_targets(
    articulation_state: dict[str, torch.Tensor],
    joint_position_target: torch.Tensor,
    joint_velocity_target: torch.Tensor,
) -> None:
    """Mutate ONE articulation's state dict in place, adding the PD set point alongside the
    configuration fields ``scene.get_state()`` already writes (``root_pose``/``root_velocity``/
    ``joint_position``/``joint_velocity``).

    Callers (``StableStateRecorder.record_pre_reset``) must pass the UN-INDEXED, full-batch
    target tensors -- i.e. read directly off ``Articulation.data.joint_pos_target`` /
    ``joint_vel_target`` BEFORE any per-``env_ids`` slicing, exactly as every other field in this
    dict is un-indexed at this point. The caller's own ``extract_env_ids_values`` recursion then
    indexes this field identically to every other leaf, including the multi-env-per-call case
    (a single ``record_pre_reset`` call can carry more than one accepted state under a leading
    batch dim -- e.g. 32 capture events yielding 34 states in one measured run -- and this
    function does not assume otherwise: it never reads ``articulation_state``'s existing batch
    size, it only writes two more same-shaped tensors into it).
    """
    articulation_state[JOINT_POSITION_TARGET_KEY] = joint_position_target
    articulation_state[JOINT_VELOCITY_TARGET_KEY] = joint_velocity_target


def resolve_joint_targets(
    articulation_state: dict[str, torch.Tensor],
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
) -> tuple[torch.Tensor, bool, torch.Tensor, bool]:
    """Given ONE already-sampled (env_ids-indexed) articulation state, return the
    ``(joint_position_target, had_position_target, joint_velocity_target, had_velocity_target)``
    to restore: the STORED target when the file provides it, else today's ``target := q``/``qdot``
    fallback (bead UWLab-algw.7 backward-compat requirement -- banks recorded before this schema
    existed have no target field and must keep loading, not raise).

    ``joint_position``/``joint_velocity`` are the just-restored CONFIGURATION for this same
    articulation (``asset_state["joint_position"]``/``["joint_velocity"]``, already cloned by the
    caller) -- passed in rather than re-read from ``articulation_state`` so the fallback is
    unambiguous even if a caller has already popped those keys for its own use.
    """
    has_position_target = JOINT_POSITION_TARGET_KEY in articulation_state
    joint_position_target = articulation_state[JOINT_POSITION_TARGET_KEY] if has_position_target else joint_position
    has_velocity_target = JOINT_VELOCITY_TARGET_KEY in articulation_state
    joint_velocity_target = articulation_state[JOINT_VELOCITY_TARGET_KEY] if has_velocity_target else joint_velocity
    return joint_position_target, has_position_target, joint_velocity_target, has_velocity_target


def missing_target_asset_names(articulation_states: dict[str, dict]) -> list[str]:
    """Names of articulation entries in a LOADED (whole-file, not yet ``env_ids``-sampled)
    reset-state dict that have no ``joint_position_target``.

    Meant to be called ONCE, at ``MultiResetManager.__init__`` load time, to log a single
    "this bank predates the target field" warning per file (bead UWLab-algw.7: a silent fallback
    is how the zeroed-squeeze defect survived undetected in the first place) -- rather than
    re-checking, and potentially re-logging, on every reset drawn from that file.
    """
    return [name for name, asset_state in articulation_states.items() if JOINT_POSITION_TARGET_KEY not in asset_state]
