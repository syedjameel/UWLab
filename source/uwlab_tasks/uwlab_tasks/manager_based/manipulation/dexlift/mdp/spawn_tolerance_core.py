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
addon reading ``env.scene``) is ``_SpawnPoseToleranceAddon``/``SpawnToleranceHeldWithProbe`` in
``scripts_v2/tools/generate_reset_states_policy.py``, alongside that script's own
``_SeatingGateAddon`` -- GENERATION-side, per the team-lead layer split with ``dexlift/mdp/c3_rung.py``
(bead dr-ai1.4, ENV-side: draws which half of C3 an episode is, sets the spawn/goal, defines no
acceptance predicate of its own). That script imports ``isaaclab`` at module scope and therefore
needs a running Isaac Sim process just to import. This module has none of that dependency, so
``source/uwlab_tasks/test/test_spawn_tolerance_stage.py`` can load it with plain ``python3`` -- see
that test's own docstring for how (loaded by file path, not via ``import uwlab_tasks...``, same
technique ``test_c1_hand_pose_stage.py`` uses and for the same reason).

CHECKED AGAINST ``c3_rung_core.py`` BEFORE WRITING THIS MODULE'S CONVENTIONS: that module defines
no "distance from spawn pose" or rotation-metric convention of its own to reuse or conflict with --
its only frame arithmetic is the tip/root ``cos(tilt)`` Z-conversion for banner/logging
(``goal_tip_z_from_root_z``), a different quantity from the direct 3D pose delta
:func:`pose_distance` computes here.

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

TWO ROTATION METRICS, DELIBERATELY BOTH KEPT, NEITHER CHOSEN (team-lead ask, 2026-08-29). For a
horizontal S_t peg, spin about the leg's own long axis is physically unconstrained (F51) -- a peg
rotated 90deg about its own axis is the same state for every purpose this rung cares about, so
:func:`pose_distance`'s full-quaternion ``rot_dist_rad`` (position AND orientation together,
including axial spin) MAY be the wrong metric to gate acceptance on; an axis-only tilt,
:func:`axis_tilt_rad`, ignores that spin and may be the more defensible one (``V2_C3_DESIGN.md``
sec 7). But "may be" is not a source, and picking one now would be exactly the invented-constant
failure ``RESET_SPEC_V2.md`` R7 exists to prevent, the same discipline that keeps
:class:`SpawnToleranceConfig` free of a default. So R4 records BOTH, per accepted and rejected
state, and bead dr-sj6.24 chooses the metric AND the tolerance together from that one distribution,
rather than the metric being fixed by whichever was easiest to compute first. Neither is used to
gate acceptance today -- :func:`within_spawn_tolerance` still gates on ``pose_distance``'s full
angle alone, unchanged, because switching gates would itself be the premature decision this
paragraph says not to make.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "LEG_TIP_LOCAL_AXIS",
    "SpawnPoseDisplacement",
    "SpawnToleranceConfig",
    "axis_tilt_rad",
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
    """One measured displacement from the commanded goal -- what R4 needs to collect to derive
    :class:`SpawnToleranceConfig`'s numbers AND choose between the two rotation metrics (bead
    dr-sj6.24; see this module's own docstring, "TWO ROTATION METRICS"). Plain floats, not tensors,
    so a caller can accumulate these into a plain list/csv/npz without any torch dependency
    surviving past the measurement itself.

    ``rot_dist_rad`` and ``axis_tilt_rad`` are BOTH recorded, deliberately -- neither is dropped in
    favour of the other. Which one ends up gating acceptance is bead dr-sj6.24's decision, made from
    the distribution these fields exist to build, not from this dataclass's shape.
    """

    pos_dist_m: float
    rot_dist_rad: float
    axis_tilt_rad: float


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


LEG_TIP_LOCAL_AXIS: tuple[float, float, float] = (-1.0, 0.0, 0.0)
"""The leg's insertion-tip direction in its OWN local/body frame -- the SAME fixed convention
``_MatingFrameGeometry._tip_local_axis`` uses
(``scripts_v2/tools/generate_reset_states_policy.py``, ``_MatingFrameGeometry.__init__``), reused
here rather than re-derived: this project has been burned before by a geometry constant quoted from
memory instead of imported. NOT read from ``metadata.yaml`` -- ``assembled_offset`` there gives the
mating feature's position/orientation, not this axis; this is a fixed modeling convention for this
leg asset family, validated in that class's own docstring by reproducing a known spawn distribution
before being trusted. Rotation ABOUT this axis is exactly the "leg's own long axis" spin
:func:`axis_tilt_rad` is deliberately blind to."""


def _quat_rotate(quat_wxyz: torch.Tensor, v: tuple[float, float, float]) -> torch.Tensor:
    """Rotate a fixed LOCAL unit vector ``v`` into world frame by ``quat_wxyz`` (``(..., 4) ->
    (..., 3)``), via the quaternion sandwich ``q * (0, v) * q^-1``. Mathematically identical to
    ``_MatingFrameGeometry``'s rotation-matrix formulation (``_quat_wxyz_to_rotmat``/``_rotate`` in
    ``generate_reset_states_policy.py``) -- expressed with THIS module's own quaternion helpers
    (:func:`_quat_mul`/:func:`_quat_inv`) instead of adding a second rotation representation here."""
    v_t = torch.as_tensor(v, dtype=quat_wxyz.dtype, device=quat_wxyz.device).expand(*quat_wxyz.shape[:-1], 3)
    v_quat = torch.cat([torch.zeros_like(v_t[..., :1]), v_t], dim=-1)
    rotated = _quat_mul(_quat_mul(quat_wxyz, v_quat), _quat_inv(quat_wxyz))
    return rotated[..., 1:]


def axis_tilt_rad(
    goal_quat_w: torch.Tensor,
    live_quat_w: torch.Tensor,
    local_axis: tuple[float, float, float] = LEG_TIP_LOCAL_AXIS,
) -> torch.Tensor:
    """``(...,)``: angle (radians) between ``local_axis`` rotated by ``goal_quat_w`` and the SAME
    local axis rotated by ``live_quat_w`` -- spin-INVARIANT about that axis, unlike
    :func:`pose_distance`'s ``rot_dist_rad`` (the full quaternion angle, which includes spin about
    every axis). See this module's own docstring, "TWO ROTATION METRICS, DELIBERATELY BOTH KEPT,
    NEITHER CHOSEN" -- this is the second of the two, recorded for R4 alongside the first, not a
    replacement for it.
    """
    goal_axis_w = _quat_rotate(goal_quat_w, local_axis)
    live_axis_w = _quat_rotate(live_quat_w, local_axis)
    goal_axis_w = goal_axis_w / goal_axis_w.norm(dim=-1, keepdim=True)
    live_axis_w = live_axis_w / live_axis_w.norm(dim=-1, keepdim=True)
    cosang = (goal_axis_w * live_axis_w).sum(-1).clamp(-1.0, 1.0)
    return torch.acos(cosang)


def within_spawn_tolerance(
    pos_dist_m: torch.Tensor, rot_dist_rad: torch.Tensor, cfg: SpawnToleranceConfig
) -> torch.Tensor:
    """Per-env boolean: is this state within tolerance of its own recorded spawn pose?

    Deliberately the SAME shape as ``success.py``'s ``within_success_tolerance`` -- strict
    less-than, ``rot_tol`` tested for truthiness so ``None`` drops the orientation gate -- reused
    conceptually rather than imported, because that function is coupled to the training/ADR
    curriculum's own goal-command plumbing in a way this generator-time acceptance check need not
    be: this module's caller (``_SpawnPoseToleranceAddon``,
    ``scripts_v2/tools/generate_reset_states_policy.py``) passes in whatever
    ``spawn_pos_w``/``spawn_quat_w`` it resolved (as of the second correction, the COMMANDED GOAL
    read via ``CommandManager.get_term``, gated on ``goal_is_final`` -- see that class's own
    docstring); this function itself is agnostic to where that pose came from, by design (this
    project's own F27 discipline: a value correct under one config's wiring must not be silently
    consumed under a different one, so this pure-math half makes no assumption about the source at
    all).
    """
    if cfg.rot_tol_rad:
        return (pos_dist_m < cfg.pos_tol_m) & (rot_dist_rad < cfg.rot_tol_rad)
    return pos_dist_m < cfg.pos_tol_m
