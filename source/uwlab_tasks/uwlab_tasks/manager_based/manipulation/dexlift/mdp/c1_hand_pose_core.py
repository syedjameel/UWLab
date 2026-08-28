# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python/torch core for the C1 (free, arbitrary) hand-pose reset stage.

RESET_SPEC_V2.md sec 1 C1, sec 1a (frames), sec 2 (R1). V2_POSE_FINDINGS.md F10.

Needs only ``os``/``math`` for the env-var parsing half and plain ``torch`` (no Isaac Sim, no GPU,
no env construction) for the quaternion half -- same split as ``held_check_core.py`` next to this
file, and for the same reason: the ISAAC-TOUCHING half (the actual IK-solving event term) lives in
``c1_hand_pose.py`` and in ``dexlift_ur5e_delto_env_cfg.py``'s ``_apply_c1_hand_pose_stage``, both
of which import ``isaaclab`` at module scope and therefore need a running Isaac Sim process just to
import. This module has none of that, so ``source/uwlab_tasks/test/test_c1_hand_pose_stage.py`` can
load it with plain ``python3`` -- see that test's own docstring for how (loaded by file path, not
via ``import uwlab_tasks...``, for the same reason ``test_held_check_core.py`` does).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

DEXRESET_C1_HAND_ENV = "DEXRESET_C1_HAND"
DEXRESET_C1_HAND_Z_ENV = "DEXRESET_C1_HAND_Z"
DEXRESET_C1_HAND_XY_ENV = "DEXRESET_C1_HAND_XY"
DEXRESET_C1_HAND_TILT_ENV = "DEXRESET_C1_HAND_TILT"

DEFAULT_Z_RAW = "0.10,0.20"
DEFAULT_XY_RAW = "0.15"
DEFAULT_TILT_RAW = "0.7854"  # ~pi/4, 45 deg


@dataclass(frozen=True)
class C1HandPoseStage:
    """One fully-validated set of staged C1 hand-pose ranges, in the units RESET_SPEC_V2.md states
    them: metres above the work surface (z), metres half-width (xy), radians half-angle (tilt).
    """

    z_lo: float
    z_hi: float
    xy_half_width: float
    tilt: float


def parse_c1_hand_pose_env(env: dict) -> C1HandPoseStage | None:
    """Parse and validate the ``DEXRESET_C1_HAND*`` env vars.

    Returns ``None`` (a complete no-op) unless ``DEXRESET_C1_HAND`` is exactly ``"1"`` -- an unset
    or any-other-value master switch means CURRENT BEHAVIOUR, BYTE FOR BYTE. Raises ``ValueError``,
    quoting the offending raw string, on any malformed value -- same idiom as
    ``_apply_pose_tilt_stage`` / the ``DEXLIFT_GOAL_VERTICAL_Z`` parser in
    ``dexlift_ur5e_delto_env_cfg.py``, deliberately matched so a reader who already knows that
    idiom recognises this one.

    Args:
        env: a mapping to read the four env vars from (pass ``os.environ`` in production; tests
            pass a plain ``dict`` so no process env var pollutes or is polluted by a test run).
    """
    if env.get(DEXRESET_C1_HAND_ENV) != "1":
        return None

    z_raw = env.get(DEXRESET_C1_HAND_Z_ENV, DEFAULT_Z_RAW)
    try:
        z_lo, z_hi = (float(part) for part in z_raw.split(","))
    except ValueError as exc:
        raise ValueError(
            f"{DEXRESET_C1_HAND_Z_ENV} must be two comma-separated metres, e.g. '{DEFAULT_Z_RAW}';"
            f" got {z_raw!r}"
        ) from exc
    if not z_lo < z_hi:
        raise ValueError(f"{DEXRESET_C1_HAND_Z_ENV} must satisfy lo < hi; got {z_lo} >= {z_hi}")

    xy_raw = env.get(DEXRESET_C1_HAND_XY_ENV, DEFAULT_XY_RAW)
    try:
        xy_half_width = float(xy_raw)
    except ValueError as exc:
        raise ValueError(
            f"{DEXRESET_C1_HAND_XY_ENV} must be a single metres value, e.g. '{DEFAULT_XY_RAW}';"
            f" got {xy_raw!r}"
        ) from exc
    if not xy_half_width > 0.0:
        raise ValueError(f"{DEXRESET_C1_HAND_XY_ENV} must be > 0; got {xy_half_width}")

    tilt_raw = env.get(DEXRESET_C1_HAND_TILT_ENV, DEFAULT_TILT_RAW)
    try:
        tilt = float(tilt_raw)
    except ValueError as exc:
        raise ValueError(
            f"{DEXRESET_C1_HAND_TILT_ENV} must be a single radians value, e.g."
            f" '{DEFAULT_TILT_RAW}'; got {tilt_raw!r}"
        ) from exc
    if not 0.0 <= tilt <= math.pi:
        raise ValueError(f"{DEXRESET_C1_HAND_TILT_ENV} must be in [0, pi] radians; got {tilt}")

    return C1HandPoseStage(z_lo=z_lo, z_hi=z_hi, xy_half_width=xy_half_width, tilt=tilt)


def anchor_xy_from_ranges(pos_x: tuple[float, float], pos_y: tuple[float, float]) -> tuple[float, float]:
    """Midpoint of the ``commands.object_pose.ranges`` XY box -- the C1 hand's XY anchor.

    RESET_SPEC_V2.md sec 1 wants the hand centred "above the insertion hole (goal/bore XY), not
    above the leg". The dexlift scene has no physical bore (V2_POSE_FINDINGS.md F2); this range's
    midpoint is the closest static stand-in for one available in this scene, and it is STATIC
    (the config's range, not a live per-episode command draw) on purpose -- see
    ``_apply_c1_hand_pose_stage``'s own docstring for why reading the live command here would be
    stale by one episode.
    """
    return 0.5 * (pos_x[0] + pos_x[1]), 0.5 * (pos_y[0] + pos_y[1])


def quat_from_two_vectors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Shortest-arc unit quaternion (w, x, y, z) rotating unit vector ``a`` onto unit vector ``b``.

    Copied verbatim (formula and convention) from
    ``scripts_v2/tools/analyze_grasp_orientation_distribution.py``'s own ``quat_from_two_vectors``,
    which this module's ``c1_hand_pose.py`` consumer relies on producing the SAME "palm pointing
    down" nominal that ``omnireset/mdp/terminations.py::check_reset_state_success``'s
    ``orient_down`` gate scores against -- both apply this quaternion to
    ``gripper_approach_direction`` and check the result lands at (or near) world -Z.
    """
    dot = (a * b).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    if dot.min() <= -0.999:
        raise ValueError("quat_from_two_vectors: a and b are (near-)antiparallel -- degenerate case not handled")
    axis = torch.cross(a.expand_as(b), b, dim=-1)
    m = torch.sqrt(2.0 + 2.0 * dot)
    w = m / 2.0
    xyz = axis / m
    return torch.cat([w, xyz], dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector ``v`` (..., 3) by unit quaternion ``q`` (..., 4), (w, x, y, z) convention.

    Same formula ``analyze_grasp_orientation_distribution.py`` uses, kept local so this module's
    self-check (:func:`palm_down_self_check`) needs no ``isaaclab`` import either.
    """
    qw = q[..., 0:1]
    qxyz = q[..., 1:4]
    t = 2.0 * torch.cross(qxyz, v, dim=-1)
    return v + qw * t + torch.cross(qxyz, t, dim=-1)


def palm_down_self_check(approach_local: torch.Tensor, atol: float = 1e-4) -> torch.Tensor:
    """Return the quaternion that rotates ``approach_local`` onto world -Z, asserting it actually
    does. Mirrors the self-check ``analyze_grasp_orientation_distribution.py`` runs before trusting
    its own ``quat_from_two_vectors`` call.
    """
    approach_local = approach_local / torch.linalg.vector_norm(approach_local)
    world_down = torch.tensor([0.0, 0.0, -1.0], dtype=approach_local.dtype)
    quat = quat_from_two_vectors(approach_local, world_down)
    check = quat_apply(quat, approach_local)
    if not torch.allclose(check, world_down, atol=atol):
        raise AssertionError(f"palm_down_self_check failed: approach_local -> {check}, expected {world_down}")
    return quat
