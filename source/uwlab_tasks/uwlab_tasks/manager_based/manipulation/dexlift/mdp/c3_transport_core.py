# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python core for the episode mixture's TRANSPORT branch (RESET_SPEC_V2.md sec 1 C3,
V2_POSE_FINDINGS.md F43 / bead dr-ai1.13: "no mixture branch asks the policy to TRANSPORT the leg
to tip-down").

Needs only ``math`` (no torch, no Isaac Sim, no GPU, no env construction) -- same split, and same
reason, as ``c1_hand_pose_core.py`` next to this file: the ISAAC-TOUCHING half (the actual mixture
event/command terms) lives in ``episode_mixture.py`` and in
``dexlift_ur5e_delto_tableleg_env_cfg.py``, both of which import ``isaaclab`` at module scope and
therefore need a running Isaac Sim process just to import. This module has none of that dependency,
so ``source/uwlab_tasks/test/test_c3_transport_stage.py`` can load it with plain ``python3``.

WHAT THIS BRANCH IS, AND WHY IT IS NOT S1 OR S_t (bead dr-ai1.4 is the separate, harder bead for
those). F43 measured that classic (48%) and low-goal (25%) branches never command a goal within
72.8 deg of tip-down, and the one branch that IS tip-down -- partial-assembly (26%) -- has the
object already AT the goal (spawned pre-inserted), so there is nothing to transport. "The policy is
asked to hold the leg near-horizontal in 73 percent of episodes, and tip-down only in the 26 percent
where it is already there." This module supplies the missing case: a goal that is tip-down AND a
spawn that is NOT already there, so a real transport task exists. It reuses the classic/low-goal
branch's ORDINARY spawn (arbitrary orientation, no fixture) unchanged -- only the GOAL differs.

FRAME, verified against measurement, not assumed (V2_C3_DESIGN.md sec 3's own standing warning).
``ROOT_ABOVE_TIP_M`` is the leg's ``assembled_offset`` position x-component
(``omnireset/mdp/commands.py``'s ``_PINNED_OFFSET_LITERALS["SquareTableLeg200mmDecomp"]["pos"]`` ==
``(-0.106203, 0.0, 0.0)``), the SAME constant ``_apply_goal_vertical_mixture``'s docstring cites
("the leg's root sits 106.2 mm ABOVE its tip when tip-down") and the one F43's own measurement
confirmed live: "on branch 2, goal_root_z - goal_tip_z = 0.1062 m exactly." This module treats that
relation as exact ONLY at (or near) the tip-down orientation this branch's goal is centred on -- the
same approximation ``_apply_goal_vertical_mixture``'s own banner already makes for its identical
tip-down-plus-tilt band, not a new one invented here.

ORIENTATION CONVENTION, copied from ``goal_mixture.MixedGoalPoseCommand`` /
``_apply_goal_vertical_mixture`` (epic UWLab-nnlv), not re-derived: roll centred on 0, pitch centred
on ``-pi/2`` (Ry(-90) is the leg's ASSEMBLED/tip-down root orientation), yaw pinned to 0, all
sampled as ABSOLUTE Euler angles (not composed as a perturbation onto a base quaternion) -- exactly
``MixedGoalPoseCommand._resample_command``'s own ``euler[:, 1].uniform_(*rng.pitch)`` shape. THE SAME
GIMBAL WARNING APPLIES: at ``pitch == -pi/2`` the XYZ-Euler parameterisation is degenerate (roll and
yaw drive the same degree of freedom, rotation about the now-vertical leg axis), so the ``tilt`` band
is a FAN in the pitch plane crossed with an axial spin, not a cone about vertical. That is adequate
here for the same reason it was adequate there: the band's centre is exactly tip-down and the spread
is padding against a razor-thin target, not a coverage requirement.

X/Y ANCHOR, load-bearing, NOT a free design choice (epic UWLab-nnlv, bead UWLab-nnlv.4, measured).
``MixedGoalPoseCommandCfg.vertical_anchor_xy``'s docstring records that inheriting an INDEPENDENT
x/y for a tip-down goal taught the policy to "hold the leg vertical SOMEWHERE" rather than at a
commanded place, and reset-state generation against that policy produced ZERO states from 148
attempts (banked poses decomposed to lateral median 141 mm, tilt median 89.9 deg -- "the policy
withdraws by flipping the leg horizontal and carrying it away, because nothing in its objective ever
rewarded staying over the spot it started from"). This branch therefore ALWAYS anchors the goal's
x/y to the object's own just-spawned position -- "hold it vertical straight up from where it lies" --
which is a commanded relation (it moves with the object every episode) rather than a fixed point, and
it is the setting the prior mixture had to add BACK IN after measuring the alternative fail. There is
no toggle here to turn that back off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# The leg's root sits this far ABOVE its tip when the leg is tip-down (Ry(-90) root orientation).
# See the module docstring's "FRAME" section for the citation chain; do not restate this number
# anywhere else -- import it from here.
ROOT_ABOVE_TIP_M = 0.106203

DEFAULT_TRANSPORT_GOAL_PROB = 0.0
DEFAULT_TRANSPORT_GOAL_TILT_RAD = 0.35  # ~20 deg, same default _apply_goal_vertical_mixture shipped
DEFAULT_TRANSPORT_GOAL_Z_RANGE_M = (0.13, 0.27)  # root frame; tip 24-164 mm, same band as GOAL_VERTICAL


def tip_z_from_root_z(root_z_m: float, *, tilt_rad: float) -> float:
    """Convert a ROOT-frame height to the TIP-frame height it implies at axis-only tilt
    ``tilt_rad`` from tip-down (the SAME "angle between the rotated local tip axis and world -Z"
    metric ``scripts_v2/tools/measure_v2_pose_distribution.py``'s ``axis_tilt_from_tipdown_deg``
    already uses -- not roll/spin about the tip axis).

    EXACT for any tilt: ``root_z - tip_z = ROOT_ABOVE_TIP_M * cos(tilt_rad)`` (F49,
    V2_POSE_FINDINGS.md, team-lead review 2026-08-29). ``tilt_rad`` is REQUIRED, not defaulted --
    the bug this fixes was exactly a silent ``cos(tilt) == 1`` assumption (equivalently, a bare
    ``root_z - ROOT_ABOVE_TIP_M``) reused outside the tip-down pose it is only exact at: measured
    against F43's own data, that assumption was off by up to 20 mm even on THIS module's own
    +-20 deg default tilt band, and by 87-125 mm on the mixture's near-horizontal (classic/
    low-goal) branches, where it should never have been applied at all. Requiring the caller to
    name ``tilt_rad`` here is the guard against a future reader lifting this function into a
    context this module's own tip-down-by-construction goals do not share.
    """
    if not 0.0 <= tilt_rad <= math.pi:
        raise ValueError(f"tilt_rad must be in [0, pi] radians; got {tilt_rad}")
    return root_z_m - ROOT_ABOVE_TIP_M * math.cos(tilt_rad)


def root_z_from_tip_z(tip_z_m: float, *, tilt_rad: float) -> float:
    """Convert a TIP-frame height to the ROOT-frame height that commands it at axis-only tilt
    ``tilt_rad`` from tip-down. Exact inverse of :func:`tip_z_from_root_z` -- see its docstring
    for the F49 citation and why ``tilt_rad`` has no default."""
    if not 0.0 <= tilt_rad <= math.pi:
        raise ValueError(f"tilt_rad must be in [0, pi] radians; got {tilt_rad}")
    return tip_z_m + ROOT_ABOVE_TIP_M * math.cos(tilt_rad)


@dataclass(frozen=True)
class TransportGoalRanges:
    """Euler-angle bounds (radians) for the transport branch's goal orientation, ROBOT ROOT frame.

    Same shape as ``MixedGoalPoseCommandCfg.VerticalRanges`` -- see the module docstring's
    "ORIENTATION CONVENTION" section for why these three ranges, centred where they are, are correct
    rather than arbitrary.
    """

    roll: tuple[float, float]
    pitch: tuple[float, float]
    yaw: tuple[float, float]


def transport_goal_ranges(tilt: float) -> TransportGoalRanges:
    """Build the roll/pitch/yaw ranges for a given tilt half-width, validated.

    Raises ``ValueError`` for a tilt outside ``[0, pi/2]`` -- the same bound
    ``_apply_goal_vertical_mixture`` enforces on ``DEXLIFT_GOAL_VERTICAL_TILT``, for the same
    reason: past ``pi/2`` the "band centred on tip-down" framing stops meaning what it says.
    """
    if not 0.0 <= tilt <= math.pi / 2:
        raise ValueError(f"transport goal tilt must be in [0, pi/2] radians; got {tilt}")
    return TransportGoalRanges(
        roll=(-tilt, tilt),
        pitch=(-math.pi / 2 - tilt, -math.pi / 2 + tilt),
        yaw=(0.0, 0.0),
    )


def validate_transport_goal_z(z_lo: float, z_hi: float) -> None:
    """Fail loudly on a malformed root-frame z band, same check ``_apply_goal_vertical_mixture``
    applies to ``DEXLIFT_GOAL_VERTICAL_Z``."""
    if not z_lo < z_hi:
        raise ValueError(f"transport goal z range must satisfy lo < hi; got ({z_lo}, {z_hi})")


def validate_episode_mixture_fractions(
    classic_goal_prob: float,
    low_goal_prob: float,
    partial_assembly_prob: float,
    transport_goal_prob: float,
) -> None:
    """Fail loudly rather than train silently on a broken mixture -- the 4-branch extension of
    ``episode_mixture.assert_episode_mixture_is_sane``'s two checks (see that function's docstring
    for the measured 55%/89%/pass@30mm-0.0000 collapse the ``classic_goal_prob > 0`` check guards
    against; this module adds the fourth term to the sum without changing either check's intent).

    Factored out here, rather than left inline in ``episode_mixture.py``, so it is unit-testable
    without an Isaac Sim process -- see the module docstring.
    """
    total = classic_goal_prob + low_goal_prob + partial_assembly_prob + transport_goal_prob
    assert abs(total - 1.0) < 1e-6, (
        "episode-mixture fractions must sum to 1.0 (a CLI override of one field without the others"
        f" desyncs them silently otherwise); got classic_goal_prob={classic_goal_prob}"
        f" low_goal_prob={low_goal_prob} partial_assembly_prob={partial_assembly_prob}"
        f" transport_goal_prob={transport_goal_prob}, sum={total}"
    )
    assert classic_goal_prob > 0.0, (
        "classic_goal_prob must be > 0. Driving it to 0 (100% low-goal/goal-at-spawn) has been"
        " measured to destroy this policy: 55% of the skill gone in 50 epochs, 89% by 300, reaching"
        " pass@30mm 0.0000, because the objective no longer contains the transport task. Keep a"
        f" majority (or at least a real fraction) of episodes on the classic goal; got {classic_goal_prob}."
    )


def transport_goal_banner(
    transport_goal_prob: float, tilt: float, z_lo: float, z_hi: float
) -> str:
    """The exact banner text printed when the transport branch is wired in (R5).

    Returned as a string (not printed here) so a test can assert on it byte-for-byte, same
    technique as the C1 gate's corrected banner (V2_POSE_FINDINGS.md F46b).

    THE TIP-Z NUMBERS ARE A FLOOR, NOT THE BAND (F49). ``tip_z_from_root_z`` is called here at
    ``tilt_rad=0.0`` -- exact tip-down, the deepest the tip can be for a given root z, since
    ``root_z - tip_z = ROOT_ABOVE_TIP_M * cos(tilt)`` is maximal at tilt 0 and shrinks toward 0 as
    tilt grows. An env drawn anywhere in this branch's own +-tilt band sits at a tip z somewhat
    ABOVE this number, never below it. Said explicitly in the banner text so a reader cannot mistake
    "leg TIP ... above the work surface" for the achieved band the way an earlier version of this
    same conversion was mistaken for one on the classic/low-goal branches (F49).
    """
    ranges = transport_goal_ranges(tilt)
    return (
        f"[dexlift] TRANSPORT GOAL branch staged: {transport_goal_prob:.3f} of episodes draw a"
        f" tip-down goal (pitch {ranges.pitch[0]:.4f} to {ranges.pitch[1]:.4f} rad = -90 +-"
        f" {math.degrees(tilt):.1f} deg, roll +-{math.degrees(tilt):.1f} deg, yaw 0) at root height"
        f" [{z_lo:.3f}, {z_hi:.3f}] m, i.e. leg TIP (at tilt=0, the floor of this branch's own tilt"
        f" band -- F49) {tip_z_from_root_z(z_lo, tilt_rad=0.0):.3f} to"
        f" {tip_z_from_root_z(z_hi, tilt_rad=0.0):.3f} m above the work surface, ANCHORED to the"
        " object's own spawn x/y (never independent -- see this module's core docstring for the"
        " 148-attempts/0-states measurement that anchoring fixes). The object spawns via the"
        " ORDINARY (classic/low-goal) draw, arbitrary orientation, so this is a genuine transport"
        " task: the goal is tip-down and the leg does not start there."
    )
