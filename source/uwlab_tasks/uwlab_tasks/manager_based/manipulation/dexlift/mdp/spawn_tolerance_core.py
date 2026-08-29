# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python/torch core for the C3(S_t) spawn-pose-tolerance addon (bead dr-sj6.22).

``V2_C3_DESIGN.md`` sec 5, ``V2_ACCEPTANCE_CRITERIA.md`` sec 4: S_t's acceptance criterion is
"held, plus the leg within a tolerance of its own spawn pose" -- deliberately NOT the bore-mating
``_SeatingGateAddon`` (``scripts_v2/tools/generate_reset_states_policy.py:1272-1338``), which has no
mating frame to project S_t into and would reject ~100% of valid S_t states (same design doc,
same section -- the exact trap the retracted v1 ``stays_seated`` proposal fell into for S2').

Needs only ``math``/``dataclasses`` and plain ``torch`` (no Isaac Sim, no GPU, no env construction)
-- same split, and same reason, as ``c1_hand_pose_core.py``/``held_check_core.py``/
``c3_transport_core.py`` next to this file: the ISAAC-TOUCHING half (the actual termination-term
addon reading ``env.scene``) lives in ``spawn_tolerance.py``, which imports ``isaaclab`` at module
scope and therefore needs a running Isaac Sim process just to import. This module has none of that
dependency, so ``source/uwlab_tasks/test/test_spawn_tolerance_stage.py`` can load it with plain
``python3`` -- see that test's own docstring for how (loaded by file path, not via
``import uwlab_tasks...``, same technique ``test_c1_hand_pose_stage.py`` uses and for the same
reason).

TOLERANCES ARE OPEN, WITH NO DEFAULT -- READ THIS BEFORE CHANGING :class:`SpawnToleranceConfig`.
No source document states a numeric position or rotation tolerance for "within tolerance of own
spawn pose"; ``V2_ACCEPTANCE_CRITERIA.md`` sec 4 marks both OPEN explicitly, and bead dr-sj6.24
says they are to be DERIVED from the R4 validation run's own measured grasp-induced displacement
distribution -- which is exactly what :func:`pose_distance` below exists to produce, not a value
this module invents in the meantime. Passing a plausible-looking number here would be exactly the
failure ``RESET_SPEC_V2.md`` R7 exists to prevent, and this campaign has already shipped one
invented constant (``RESET_SPEC_V2.md`` sec 6 item 0: the withdrawn ``stays_seated``
6.02%->43.19% pair). :class:`SpawnToleranceConfig` therefore has NO field defaults and raises in
``__post_init__`` if either is missing or non-positive -- see that class's own docstring.

QUATERNION HELPERS ARE REPRODUCED HERE, not imported from ``isaaclab.utils.math``, for the same
reason ``c1_hand_pose_core.py``'s own ``quat_apply``/``quat_from_two_vectors`` are: importing
``isaaclab`` needs a running Isaac Sim process. :func:`pose_distance`'s rotation-distance formula
(relative quaternion, ``2*acos(|w|)``) is the SAME formula ``c1_hand_pose.py``'s own IK-residual
measurement uses (``c1_hand_pose.py:266-268``: ``quat_err = quat_mul(quat_inv(cmd_quat_w),
achieved_quat_w); ori_err = 2*acos(quat_err[:,0].abs().clamp(max=1.0))``) -- reproduced here with a
local ``(w,x,y,z)`` Hamilton product/conjugate rather than restated with different arithmetic, so a
reader who already trusts that formula there can verify this one by comparison rather than by
re-deriving it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "SpawnPoseDisplacement",
    "SpawnToleranceConfig",
    "pose_distance",
    "within_spawn_tolerance",
]


@dataclass(frozen=True)
class SpawnToleranceConfig:
    """Validated pos/rot tolerance for "within tolerance of own spawn pose" (S_t's acceptance
    criterion, ``V2_C3_DESIGN.md`` sec 5). See this module's own docstring, "TOLERANCES ARE OPEN,
    WITH NO DEFAULT", for why neither field has one.

    Args:
        pos_tol_m: REQUIRED. Max position drift from spawn, metres. Raises if ``None`` or not
            strictly positive.
        rot_tol_rad: Max rotation drift from spawn, radians, or ``None`` to disable the rotation
            gate entirely (tested for truthiness, same convention ``success.py``'s own
            ``within_success_tolerance`` uses -- ``None`` and ``0.0`` both drop the gate, which is
            worth stating because "tolerance zero" reads like the opposite). Optional because a
            caller deriving only a position tolerance from R4 first (bead dr-sj6.24) should not be
            forced to invent a rotation number to unblock it -- but if given, it must be positive:
            an explicit ``0.0`` would silently mean "disabled", not "no rotation allowed", the same
            trap ``within_success_tolerance``'s own docstring calls out.
    """

    pos_tol_m: float
    rot_tol_rad: float | None = None

    def __post_init__(self) -> None:
        if self.pos_tol_m is None:
            raise ValueError(
                "SpawnToleranceConfig.pos_tol_m is REQUIRED -- there is no sourced default. "
                "V2_ACCEPTANCE_CRITERIA.md sec 4 marks the S_t position tolerance OPEN; bead "
                "dr-sj6.24 derives it from the R4 validation run's own measured grasp-induced "
                "displacement distribution. Do not pass a guessed value -- collect the "
                "distribution (see pose_distance/SpawnPoseDisplacement in this module) first."
            )
        if not self.pos_tol_m > 0.0:
            raise ValueError(f"SpawnToleranceConfig.pos_tol_m must be > 0, got {self.pos_tol_m!r}")
        if self.rot_tol_rad is not None and not self.rot_tol_rad > 0.0:
            raise ValueError(
                f"SpawnToleranceConfig.rot_tol_rad must be > 0 or None (None/0.0 both disable the "
                f"rotation gate -- see this class's own docstring), got {self.rot_tol_rad!r}"
            )


@dataclass(frozen=True)
class SpawnPoseDisplacement:
    """One measured (position, rotation) displacement from spawn -- what R4 needs to collect to
    derive :class:`SpawnToleranceConfig`'s numbers (bead dr-sj6.24). Plain floats, not tensors, so
    a caller can accumulate these into a plain list/csv/npz without any torch dependency surviving
    past the measurement itself."""

    pos_dist_m: float
    rot_dist_rad: float


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two ``(..., 4)`` quaternions, ``(w, x, y, z)`` convention -- same
    convention and composition order as ``isaaclab.utils.math.quat_mul`` (``q1`` applied after
    ``q2``), reproduced here so this module needs no ``isaaclab`` import (see this module's own
    docstring, "QUATERNION HELPERS ARE REPRODUCED HERE")."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a UNIT quaternion ``(..., 4)``, ``(w, x, y, z)`` convention -- its inverse.
    Every pose this module handles comes straight off the physics sim (``RigidObject.data`` is
    always unit-normalised), so conjugate suffices; no norm division is performed."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def pose_distance(
    spawn_pos_w: torch.Tensor,
    spawn_quat_w: torch.Tensor,
    live_pos_w: torch.Tensor,
    live_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(pos_dist_m, rot_dist_rad)``, each ``(...,)``: how far ``live_pos_w``/``live_quat_w`` has
    drifted from a recorded ``spawn_pos_w``/``spawn_quat_w``, in the leg's own world frame.

    Position distance is a plain L2 norm. Rotation distance is the angle of the relative
    quaternion ``spawn^-1 * live``, via ``2*acos(|w|)`` -- the SAME formula
    ``c1_hand_pose.py:266-268`` already uses for its own achieved-vs-commanded orientation
    residual (see this module's own docstring for the direct line comparison); reproduced with
    this module's local :func:`_quat_mul`/:func:`_quat_inv` rather than ``isaaclab.utils.math``'s,
    for the Isaac-import reason stated there.
    """
    pos_dist = torch.linalg.norm(live_pos_w - spawn_pos_w, dim=-1)
    rel_quat = _quat_mul(_quat_inv(spawn_quat_w), live_quat_w)
    rot_dist = 2.0 * torch.acos(rel_quat[..., 0].abs().clamp(max=1.0))
    return pos_dist, rot_dist


def within_spawn_tolerance(
    pos_dist_m: torch.Tensor, rot_dist_rad: torch.Tensor, cfg: SpawnToleranceConfig
) -> torch.Tensor:
    """Per-env boolean: is this state within tolerance of its own recorded spawn pose?

    Deliberately the SAME shape as ``success.py``'s ``within_success_tolerance`` -- strict
    less-than, ``rot_tol`` tested for truthiness so ``None`` drops the orientation gate -- reused
    conceptually rather than imported, because that function is coupled to the training/ADR
    curriculum's live goal-command plumbing (``goal_pose_error`` reads
    ``env.command_manager.get_command(...)``), which this generator-time acceptance check
    deliberately does not depend on: S_t's spawn pose is captured directly off the object at reset,
    not read back through whatever goal-command wiring a given run happens to have active. See
    ``spawn_tolerance.py``'s own module docstring, "WHY NOT env.command_manager", for the full
    reasoning (this project's own F27 discipline: a value correct under one config's wiring must
    not be silently consumed under a different one).
    """
    if cfg.rot_tol_rad:
        return (pos_dist_m < cfg.pos_tol_m) & (rot_dist_rad < cfg.rot_tol_rad)
    return pos_dist_m < cfg.pos_tol_m
