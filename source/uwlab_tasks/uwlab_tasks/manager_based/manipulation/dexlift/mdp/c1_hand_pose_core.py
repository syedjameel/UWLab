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
DEXRESET_C1_HAND_MAX_POS_ERR_MM_ENV = "DEXRESET_C1_HAND_MAX_POS_ERR_MM"
DEXRESET_C1_HAND_MAX_ORI_ERR_DEG_ENV = "DEXRESET_C1_HAND_MAX_ORI_ERR_DEG"
DEXRESET_C1_HAND_MIN_JOINT_MARGIN_DEG_ENV = "DEXRESET_C1_HAND_MIN_JOINT_MARGIN_DEG"
DEXRESET_C1_HAND_MAX_RETRIES_ENV = "DEXRESET_C1_HAND_MAX_RETRIES"

DEFAULT_Z_RAW = "0.10,0.20"
DEFAULT_XY_RAW = "0.15"
DEFAULT_TILT_RAW = "0.7854"  # ~pi/4, 45 deg

# POST-SOLVE GATE DEFAULTS -- critic review of commit 1654e2c (RESET_SPEC_V2.md sec 1 C1) found the
# original event wrote whatever the 10-iteration damped IK converged to, unchecked. Measured on the
# H100 (DL_H100, 2x H100 PCIe, 256 envs, 2 forced resets each, n=512): 95/512 (18.6%, run B,
# xy_half_width=0.15) and 89/512 (17.4%, run C, xy_half_width=0.10) resets landed with the achieved
# palm HEIGHT outside the commanded [z_lo, z_hi] band -- including a minimum achieved height of
# -0.317 m, i.e. BELOW the tabletop. |dx|/|dy| exceeded xy_half_width on 21/512 (4.1%, run B) and
# 34/512 (6.6%, run C) -- a HIGHER rate at the TIGHTER band, ruling out a proportional-to-box-size
# error and pointing at absolute-scale IK divergence instead.
#
# DEXRESET_C1_HAND_MAX_POS_ERR_MM=100: the commanded-vs-achieved position residual's sorted-value
# gaps start widening around the 95th-97th percentile in both runs (run B: p95=96.6mm,
# p97=152.6mm; run C: p95=72.9mm, p97=117.6mm), well past the median the IK actually converges
# well for (~24-26mm). 100mm sits just past that elbow in both runs -- loose enough not to reject
# an ordinary DLS residual on a healthy solve, tight enough to catch the population that plainly
# diverged (residuals up to 678mm/693mm were observed).
#
# DEXRESET_C1_HAND_MAX_ORI_ERR_DEG=20: sits at run B's own measured p95 orientation residual
# (20.2 deg) and just above run C's (15.7 deg) -- both runs' orientation residual "elbows" land in
# the same place their position residual does.
#
# NOTE ON WHAT THIS GATE DOES AND DOES NOT GUARANTEE: measured against the SAME data, the
# commanded-vs-achieved RESIDUAL only partially predicts whether the ACHIEVED pose lands inside the
# height/XY band -- some in-band samples had a large residual (a lucky solve into a different but
# still in-band configuration) and some out-of-band samples had a small one (a well-converged solve
# whose COMMANDED target itself sat right at the band edge). The residual+joint-margin gate is a
# genuine IK-QUALITY check (did the solver actually reach what was asked, independent of any
# particular band) and is what was asked for; it is deliberately NOT the only thing this stage now
# gates on -- see :func:`~.c1_hand_pose.reset_end_effector_c1_hand_pose`'s docstring for the
# achieved-height/XY band check added alongside it, which is what actually drives violations to
# zero.
#
# DEXRESET_C1_HAND_MIN_JOINT_MARGIN_DEG=1.0: matches scripts_v2/tools/gen_ik_c4_reset_bank.py's own
# --joint-limit-margin-deg default verbatim, on instruction ("in the style of
# gen_ik_c4_reset_bank.py's --joint-limit-margin-deg").
#
# DEXRESET_C1_HAND_MAX_RETRIES=5: a bounded resample-and-resolve budget (6 total attempts including
# the first). Retries must be bounded and the exhaustion count must be visible, not silently
# absorbed -- see the event class's own docstring for how an exhausted env is handled (best-of-
# attempts kept, count printed every reset).
DEFAULT_MAX_POS_ERR_MM_RAW = "100.0"
DEFAULT_MAX_ORI_ERR_DEG_RAW = "20.0"
DEFAULT_MIN_JOINT_MARGIN_DEG_RAW = "1.0"
DEFAULT_MAX_RETRIES_RAW = "5"


@dataclass(frozen=True)
class C1HandPoseStage:
    """One fully-validated set of staged C1 hand-pose ranges, in the units RESET_SPEC_V2.md states
    them: metres above the work surface (z), metres half-width (xy), radians half-angle (tilt).
    Also carries the post-solve gate's thresholds (radians/metres internally, parsed from mm/deg
    env vars -- see :func:`parse_c1_hand_pose_env`) and the bounded retry budget.
    """

    z_lo: float
    z_hi: float
    xy_half_width: float
    tilt: float
    max_pos_err_m: float
    max_ori_err_rad: float
    min_joint_margin_rad: float
    max_retries: int


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

    max_pos_err_mm_raw = env.get(DEXRESET_C1_HAND_MAX_POS_ERR_MM_ENV, DEFAULT_MAX_POS_ERR_MM_RAW)
    try:
        max_pos_err_mm = float(max_pos_err_mm_raw)
    except ValueError as exc:
        raise ValueError(
            f"{DEXRESET_C1_HAND_MAX_POS_ERR_MM_ENV} must be a single millimetres value, e.g."
            f" '{DEFAULT_MAX_POS_ERR_MM_RAW}'; got {max_pos_err_mm_raw!r}"
        ) from exc
    if not max_pos_err_mm > 0.0:
        raise ValueError(f"{DEXRESET_C1_HAND_MAX_POS_ERR_MM_ENV} must be > 0; got {max_pos_err_mm}")

    max_ori_err_deg_raw = env.get(DEXRESET_C1_HAND_MAX_ORI_ERR_DEG_ENV, DEFAULT_MAX_ORI_ERR_DEG_RAW)
    try:
        max_ori_err_deg = float(max_ori_err_deg_raw)
    except ValueError as exc:
        raise ValueError(
            f"{DEXRESET_C1_HAND_MAX_ORI_ERR_DEG_ENV} must be a single degrees value, e.g."
            f" '{DEFAULT_MAX_ORI_ERR_DEG_RAW}'; got {max_ori_err_deg_raw!r}"
        ) from exc
    if not 0.0 < max_ori_err_deg <= 180.0:
        raise ValueError(
            f"{DEXRESET_C1_HAND_MAX_ORI_ERR_DEG_ENV} must be in (0, 180] degrees; got {max_ori_err_deg}"
        )

    min_joint_margin_deg_raw = env.get(
        DEXRESET_C1_HAND_MIN_JOINT_MARGIN_DEG_ENV, DEFAULT_MIN_JOINT_MARGIN_DEG_RAW
    )
    try:
        min_joint_margin_deg = float(min_joint_margin_deg_raw)
    except ValueError as exc:
        raise ValueError(
            f"{DEXRESET_C1_HAND_MIN_JOINT_MARGIN_DEG_ENV} must be a single degrees value, e.g."
            f" '{DEFAULT_MIN_JOINT_MARGIN_DEG_RAW}'; got {min_joint_margin_deg_raw!r}"
        ) from exc
    if not min_joint_margin_deg >= 0.0:
        raise ValueError(
            f"{DEXRESET_C1_HAND_MIN_JOINT_MARGIN_DEG_ENV} must be >= 0; got {min_joint_margin_deg}"
        )

    max_retries_raw = env.get(DEXRESET_C1_HAND_MAX_RETRIES_ENV, DEFAULT_MAX_RETRIES_RAW)
    try:
        # int(str) rejects "1.5" (ValueError) same as it rejects "nope" -- exactly what a retry
        # BUDGET (a count, not a measurement) should do; a fractional retry count is as malformed
        # as a non-numeric one.
        max_retries = int(max_retries_raw)
    except ValueError as exc:
        raise ValueError(
            f"{DEXRESET_C1_HAND_MAX_RETRIES_ENV} must be a non-negative integer, e.g."
            f" '{DEFAULT_MAX_RETRIES_RAW}'; got {max_retries_raw!r}"
        ) from exc
    if max_retries < 0:
        raise ValueError(f"{DEXRESET_C1_HAND_MAX_RETRIES_ENV} must be >= 0; got {max_retries_raw!r}")

    return C1HandPoseStage(
        z_lo=z_lo,
        z_hi=z_hi,
        xy_half_width=xy_half_width,
        tilt=tilt,
        max_pos_err_m=max_pos_err_mm / 1000.0,
        max_ori_err_rad=math.radians(max_ori_err_deg),
        min_joint_margin_rad=math.radians(min_joint_margin_deg),
        max_retries=max_retries,
    )


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


def worst_case_composed_angle_rad(tilt: float) -> float:
    """The total rotation angle (from the palm-down nominal) at the WORST corner of the per-axis
    tilt box (roll = pitch = yaw = +tilt), under the SAME Tait-Bryan composition
    ``isaaclab.utils.math.quat_from_euler_xyz`` and the event's own ``quat_mul(perturb, nominal)``
    use.

    Critic review of commit 1654e2c: RESET_SPEC_V2.md sec 1 C1 asks for "+-45 deg variation,
    applied per-axis about the palm-down nominal", which the event reads as three INDEPENDENTLY
    sampled per-axis angles composed into ONE rotation. Euler-angle composition is not additive,
    so nothing bounds the RESULTING single-rotation angle to the per-axis tilt -- at the shipped
    45 deg default the worst corner composes to ~64.74 deg, not 45. This is NOT a cone bound (the
    achieved-angle distribution is not spherically symmetric about the nominal -- see
    ``test_per_axis_tilt_extremes_do_not_bound_the_composed_rotation_to_tilt``), so it must not be
    used as a rejection threshold (that would silently substitute a different definition than the
    one RESET_SPEC_V2.md's user chose -- per-axis, not cone). Its only sanctioned use is
    INFORMATIONAL: printed in the staging banner alongside "+-tilt deg per axis" so nobody later
    reads that number as a bound on the achieved cone.
    """
    half = tilt * 0.5
    c, s = math.cos(half), math.sin(half)
    # Tait-Bryan roll-pitch-yaw -> quaternion, roll = pitch = yaw = tilt (the same formula
    # isaaclab.utils.math.quat_from_euler_xyz documents; reproduced here because that module needs
    # a running Isaac Sim process just to import -- see this module's own docstring).
    qw = c * c * c + s * s * s
    # |axis| is not needed: the rotation angle from identity is recovered from qw alone via
    # angle = 2*acos(|qw|), the same relation :func:`quat_from_two_vectors`'s callers already use.
    return 2.0 * math.acos(min(1.0, abs(qw)))


def ik_gate_pass(
    pos_err_m: torch.Tensor,
    ori_err_rad: torch.Tensor,
    joint_margin_rad: torch.Tensor,
    height_m: torch.Tensor,
    dx_m: torch.Tensor,
    dy_m: torch.Tensor,
    stage: C1HandPoseStage,
) -> torch.Tensor:
    """Per-env boolean: does this attempt satisfy every post-solve acceptance criterion?

    Critic review of commit 1654e2c's headline finding: the original event wrote whatever IK
    converged to, unchecked. Measured on the H100 (n=512 each), 17-19% of resets landed with the
    ACHIEVED palm height outside the commanded band (min -0.317 m -- below the tabletop) and
    4-7% outside the XY band, despite the SAMPLED target always being drawn inside both. Two
    independent checks are combined here, both needed (see :data:`DEFAULT_MAX_POS_ERR_MM_RAW`'s
    own comment for why residual alone under-catches band violations and vice versa):

    1. IK QUALITY -- did the solver actually reach what was asked: ``pos_err_m`` /
       ``ori_err_rad`` (achieved vs. the COMMANDED target, not vs. the anchor) within
       ``stage.max_pos_err_m`` / ``stage.max_ori_err_rad``, and ``joint_margin_rad`` (min over the
       controlled arm joints) at least ``stage.min_joint_margin_rad`` away from either limit --
       same style as ``gen_ik_c4_reset_bank.py``'s ``--joint-limit-margin-deg``.
    2. SPEC COMPLIANCE -- does the ACHIEVED pose actually land in RESET_SPEC_V2.md sec 1's own
       height/XY band: this is the literal numeric criterion, not a substitution (contrast the
       palm-angle cone, which sec 1 states per-axis and which this function deliberately does NOT
       gate on -- see :func:`worst_case_composed_angle_rad`'s own docstring).

    All bounds are INCLUSIVE (``<=``/``>=``): a threshold copied verbatim from a measured
    percentile must not reject the very samples that defined it.
    """
    return (
        (pos_err_m <= stage.max_pos_err_m)
        & (ori_err_rad <= stage.max_ori_err_rad)
        & (joint_margin_rad >= stage.min_joint_margin_rad)
        & (height_m >= stage.z_lo)
        & (height_m <= stage.z_hi)
        & (dx_m.abs() <= stage.xy_half_width)
        & (dy_m.abs() <= stage.xy_half_width)
    )


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
