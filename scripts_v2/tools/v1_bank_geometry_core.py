# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Pure-numpy, Isaac-free geometry for filtering v1 reset banks against the v2 acceptance
criteria that are readable from STORED STATE ALONE (bead R3, ``RESET_SPEC_V2.md`` sec 6: "Reuse
old-bank states only where they pass the v2 filter; regenerate the rest.", C1-C4).

THE CRITICAL DISTINCTION THIS WHOLE MODULE IS BUILT AROUND (team-lead instruction, 2026-08-29).
Some v2 criteria are GEOMETRIC and readable off a stored state: tip-frame position relative to the
mating frame (depth/lateral/tilt), and root LINEAR VELOCITY (a snapshot field the recorder already
stores, not something requiring a running sim). Others are DYNAMIC and CANNOT be evaluated offline
at all: the held-state gate chain's ``settled``/``opposed_contact``/``co_move`` gates need live
contact-sensor forces and step-count history a stored root pose does not carry. **This module
therefore can only ever help produce an UPPER BOUND on reusable states -- "geometrically
compatible", never "reusable".** A state that passes every check here may still fail the held-state
chain when replayed, and that chain is where the large majority of v1 attempts died (per team-lead:
at least 90.8%). Every caller of this module must report results in exactly those words.

SCOPE OF THIS PASS -- STATED, NOT HIDDEN.
  - **C4**: mating-frame depth/lateral/tilt band, code defaults (``--c4_depth_min_mm``=5,
    ``--c4_depth_max_mm``=20, ``--c4_lateral_max_mm``=8, ``--c4_tilt_max_deg``=20,
    ``--c4_engaged_span_mm``=25) -- NOT open, one count.
  - **C3(S1)**: the SAME mating-frame decomposition, but the v2 band itself is OPEN (bead
    dr-sj6.23) -- reported as a SURVIVAL CURVE (:func:`band_survival_curve`) over a family of
    candidate bands built by scaling the v1 precedent shape (depth [0,10]mm, lateral<=5mm,
    tilt<=15deg -- ``launch_dexreset_s1_s2_bank_gen.sh``'s own S1 definition), never a single
    picked count.
  - **C2**: resting speed, code default ``--c2_max_resting_speed``=0.05 m/s, absolute LINEAR
    speed (not the held-chain's relative co_move speed) -- one count.
  - **C3(S_t)**: deliberately NOT reported as a survival number. S_t's own goal is defined as "the
    leg's own pose, zero delta" (``V2_C3_DESIGN.md`` sec 5/7) -- for an EXISTING stored state,
    that goal IS the stored pose itself, so "distance from the goal" is trivially zero for every
    state by construction. The entire discriminating power of S_t's acceptance criterion lives in
    the held-state chain (was the leg actually held stably at that pose), which this module cannot
    evaluate. Reporting a flat 100%-survival number here would be exactly the "geometrically
    compatible" read-as-"reusable" defect this module exists to prevent -- so callers should not
    compute or print one.
  - **C1**: OUT OF SCOPE for this pass, stated rather than silently skipped. C1's own acceptance
    criterion (``c1_hand_pose_core.py``) is about the ACHIEVED HAND pose relative to a target,
    height/XY banded plus an IK residual -- not a property of the stored OBJECT/receptive_object
    root state this module reads. Recovering it would need forward kinematics from the stored
    robot joint state, a separate undertaking not attempted here.

NO cos(tilt)/TIP-Z-FROM-TABLETOP CONVERSION ANYWHERE IN THIS MODULE (team-lead's FRAME RULES,
2026-08-29, "already cost this campaign a defect" x3: F49's bare Z-subtraction bug is the same
shape). Depth is computed via the FULL rigid-transform projection into the fixture's own target
frame (``leg_tip_in_target = R_fix^T @ (leg_tip_world - fix_pos) - fixture_offset_pos``), exactly
as ``generate_reset_states_policy.py``'s own F49b-audited ``_MatingFrameGeometry.decompose`` does
-- never a scalar Z subtraction or an approximate cos(tilt) formula, so F49's trap does not apply
to depth here at all. Tilt is the angle between two WORLD-frame axis vectors, both derived from
each state's OWN live measured quaternion -- never a nominal/nameplate tilt.

GEOMETRY CONSTANTS -- SOURCE, NOT GUESSED. ``LEG_OFFSET_POS_M``/``LEG_TIP_LOCAL_AXIS`` and
``FIXTURE_OFFSET_POS_M``/``FIXTURE_OFFSET_QUAT_WXYZ``/``BORE_DEEP_LOCAL_AXIS`` are read verbatim
from ``source/uwlab_assets/data/Props/FurnitureBench/{leg}/metadata.yaml`` and
``.../OneLegInsertionFixture/metadata.yaml`` on DL_H100 (confirmed 2026-08-29), and reproduced --
not re-derived -- the same discipline ``spawn_tolerance_core.py``'s own quaternion helpers already
use for the identical Isaac-import reason. **Every leg variant's metadata.yaml states "MUST stay
byte-identical to the Decomp/Sdf sibling copy" (checked directly for
SquareTableLeg200mmThreadSdfHybrid and OneLegInsertionFixture)** -- so ONE constant set is valid
for every ``SquareTableLeg200mm*`` bank regardless of which leg-asset variant generated it; this
module does not need per-bank metadata lookups. ``LEG_TIP_LOCAL_AXIS = [-1, 0, 0]`` and
``BORE_DEEP_LOCAL_AXIS = [0, 0, -1]`` are the SAME hardcoded convention
``_MatingFrameGeometry.__init__`` uses, independently corroborated by the leg metadata's own
comment: "Leg bbox x spans -106.203..+93.797 mm, so the tip is at local -X."

Isaac-free: only ``numpy`` + stdlib, no ``torch`` import at module scope -- runs on the plain
system ``python3`` (no torch there) as well as any venv that has torch, so the pure-math half of
this tool's own tests can execute on BOTH.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = [
    "BORE_DEEP_LOCAL_AXIS",
    "DEFAULT_C2_MAX_RESTING_SPEED_MPS",
    "DEFAULT_C4_DEPTH_MAX_M",
    "DEFAULT_C4_DEPTH_MIN_M",
    "DEFAULT_C4_LATERAL_MAX_M",
    "DEFAULT_C4_TILT_MAX_DEG",
    "DEFAULT_ENGAGED_SPAN_M",
    "FIXTURE_OFFSET_POS_M",
    "FIXTURE_OFFSET_QUAT_WXYZ",
    "LEG_OFFSET_POS_M",
    "LEG_TIP_LOCAL_AXIS",
    "V1_S1_PRECEDENT_DEPTH_MIN_M",
    "V1_S1_PRECEDENT_DEPTH_MAX_M",
    "V1_S1_PRECEDENT_LATERAL_MAX_M",
    "V1_S1_PRECEDENT_TILT_MAX_DEG",
    "BandCriteria",
    "band_survival_curve",
    "decompose_mating_frame",
    "leg_asset_from_path",
    "quat_wxyz_to_rotmat",
    "resting_speed_survival",
    "rotate",
    "within_band",
]

# -- geometry constants, see module docstring "GEOMETRY CONSTANTS -- SOURCE, NOT GUESSED" --------
LEG_OFFSET_POS_M = np.array([-0.106203, 0.0, 0.0])
LEG_TIP_LOCAL_AXIS = np.array([-1.0, 0.0, 0.0])
FIXTURE_OFFSET_POS_M = np.array([-0.056250, 0.056250, -0.009374])
FIXTURE_OFFSET_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
BORE_DEEP_LOCAL_AXIS = np.array([0.0, 0.0, -1.0])

# -- v2 code-default bands (generate_reset_states_policy.py argparse defaults) --------------------
DEFAULT_ENGAGED_SPAN_M = 0.025  # --c4_engaged_span_mm
DEFAULT_C4_DEPTH_MIN_M = 0.005  # --c4_depth_min_mm
DEFAULT_C4_DEPTH_MAX_M = 0.020  # --c4_depth_max_mm
DEFAULT_C4_LATERAL_MAX_M = 0.008  # --c4_lateral_max_mm
DEFAULT_C4_TILT_MAX_DEG = 20.0  # --c4_tilt_max_deg
DEFAULT_C2_MAX_RESTING_SPEED_MPS = 0.05  # --c2_max_resting_speed, ABSOLUTE linear speed

# -- v1 precedent S1 band shape (launch_dexreset_s1_s2_bank_gen.sh's own S1 definition) -- a
# REFERENCE SHAPE for the survival-CURVE sweep below, not itself the v2 answer (dr-sj6.23 is OPEN).
V1_S1_PRECEDENT_DEPTH_MIN_M = 0.000
V1_S1_PRECEDENT_DEPTH_MAX_M = 0.010
V1_S1_PRECEDENT_LATERAL_MAX_M = 0.005
V1_S1_PRECEDENT_TILT_MAX_DEG = 15.0


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """``(..., 4)`` WXYZ quaternions -> ``(..., 3, 3)`` rotation matrices. Verbatim port (formula,
    not re-derived) of ``generate_reset_states_policy.py``'s own ``_quat_wxyz_to_rotmat`` -- that
    function's own docstring: validated by reproducing the KNOWN spawn distribution before being
    trusted. Reproduced in numpy here (Isaac-free constraint) rather than imported."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotate(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``R: (..., 3, 3), v: (3,) or (..., 3) -> (..., 3)``. Same helper as
    ``generate_reset_states_policy.py``'s own ``_rotate``, ported to numpy."""
    R = np.asarray(R, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        v = np.broadcast_to(v, R.shape[:-2] + (3,))
    return np.einsum("...ij,...j->...i", R, v)


def decompose_mating_frame(
    leg_pos: np.ndarray,
    leg_quat: np.ndarray,
    fix_pos: np.ndarray,
    fix_quat: np.ndarray,
    *,
    engaged_span_m: float = DEFAULT_ENGAGED_SPAN_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(depth_m, lateral_m, tilt_deg)``, each ``(...,)`` -- SAME formula, same constants'
    provenance, as ``generate_reset_states_policy.py``'s own ``_MatingFrameGeometry.decompose``.
    See this module's own docstring for why no cos(tilt)/tip-Z-from-tabletop conversion is used
    anywhere here.
    """
    leg_pos = np.asarray(leg_pos, dtype=np.float64)
    fix_pos = np.asarray(fix_pos, dtype=np.float64)
    R_leg = quat_wxyz_to_rotmat(leg_quat)
    R_fix = quat_wxyz_to_rotmat(fix_quat)

    leg_tip_world = leg_pos + rotate(R_leg, LEG_OFFSET_POS_M)
    leg_tip_axis_world = rotate(R_leg, LEG_TIP_LOCAL_AXIS)
    leg_tip_axis_world = leg_tip_axis_world / np.linalg.norm(leg_tip_axis_world, axis=-1, keepdims=True)
    bore_deep_axis_world = rotate(R_fix, BORE_DEEP_LOCAL_AXIS)
    bore_deep_axis_world = bore_deep_axis_world / np.linalg.norm(bore_deep_axis_world, axis=-1, keepdims=True)

    R_fix_T = np.swapaxes(R_fix, -1, -2)
    leg_tip_in_fix_root = rotate(R_fix_T, leg_tip_world - fix_pos)
    leg_tip_in_target = leg_tip_in_fix_root - FIXTURE_OFFSET_POS_M
    x_t, y_t, z_t = np.moveaxis(leg_tip_in_target, -1, 0)

    depth_m = engaged_span_m - z_t
    lateral_m = np.sqrt(x_t**2 + y_t**2)
    cosang = np.clip((leg_tip_axis_world * bore_deep_axis_world).sum(-1), -1.0, 1.0)
    tilt_deg = np.degrees(np.arccos(cosang))
    return depth_m, lateral_m, tilt_deg


def within_band(
    depth_m: np.ndarray,
    lateral_m: np.ndarray,
    tilt_deg: np.ndarray,
    *,
    depth_min_m: float,
    depth_max_m: float,
    lateral_max_m: float,
    tilt_max_deg: float,
) -> np.ndarray:
    """Per-state boolean: inside ``[depth_min_m, depth_max_m]``, ``lateral_m <= lateral_max_m``,
    ``tilt_deg <= tilt_max_deg`` -- the SAME inclusive convention ``_SeatingGateAddon`` uses."""
    depth_m = np.asarray(depth_m)
    lateral_m = np.asarray(lateral_m)
    tilt_deg = np.asarray(tilt_deg)
    return (depth_m >= depth_min_m) & (depth_m <= depth_max_m) & (lateral_m <= lateral_max_m) & (tilt_deg <= tilt_max_deg)


def resting_speed_survival(lin_speed_mps: np.ndarray, *, max_speed_mps: float = DEFAULT_C2_MAX_RESTING_SPEED_MPS) -> np.ndarray:
    """Per-state boolean: absolute linear speed (root velocity magnitude, a value the recorder
    already stores -- no simulation needed to read it) is at or under ``max_speed_mps``. This is
    C2's OWN hard filter (``--c2_max_resting_speed``), NOT the held-chain's relative co_move speed
    -- different quantity, different threshold's own provenance."""
    return np.asarray(lin_speed_mps) <= max_speed_mps


class BandCriteria:
    """One (depth_min, depth_max, lateral_max, tilt_max) band, plus a human label. Plain data, no
    behaviour -- used by :func:`band_survival_curve` to sweep a family of candidate bands."""

    __slots__ = ("label", "depth_min_m", "depth_max_m", "lateral_max_m", "tilt_max_deg")

    def __init__(self, label: str, depth_min_m: float, depth_max_m: float, lateral_max_m: float, tilt_max_deg: float):
        self.label = label
        self.depth_min_m = depth_min_m
        self.depth_max_m = depth_max_m
        self.lateral_max_m = lateral_max_m
        self.tilt_max_deg = tilt_max_deg


def band_survival_curve(
    depth_m: np.ndarray,
    lateral_m: np.ndarray,
    tilt_deg: np.ndarray,
    *,
    scale_factors: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0),
    reference_depth_min_m: float = V1_S1_PRECEDENT_DEPTH_MIN_M,
    reference_depth_max_m: float = V1_S1_PRECEDENT_DEPTH_MAX_M,
    reference_lateral_max_m: float = V1_S1_PRECEDENT_LATERAL_MAX_M,
    reference_tilt_max_deg: float = V1_S1_PRECEDENT_TILT_MAX_DEG,
) -> list[dict]:
    """Survival count AS A FUNCTION OF CANDIDATE BAND WIDTH (team-lead requirement, for any v2
    criterion that is still OPEN -- C3(S1)'s band, bead dr-sj6.23, is the one this is built for).
    Rather than pick one guessed band, sweep a family built by scaling a REFERENCE shape's three
    widths by each of ``scale_factors`` -- the reference defaults to the v1 precedent S1 band
    (depth [0,10]mm, lateral<=5mm, tilt<=15deg), one defensible parameterisation among several, not
    a claim that scaling THIS shape is the correct answer. Depth is scaled around its own midpoint
    (widening both ends), lateral/tilt are scaled from zero (both are already `<= max` bands).
    Returns a list of ``{scale_factor, depth_min_m, depth_max_m, lateral_max_m, tilt_max_deg, n,
    n_survive, survive_fraction}`` dicts, one per scale factor, in the order given.
    """
    depth_m = np.asarray(depth_m)
    lateral_m = np.asarray(lateral_m)
    tilt_deg = np.asarray(tilt_deg)
    n = int(depth_m.size)
    depth_mid = (reference_depth_min_m + reference_depth_max_m) / 2.0
    depth_half_width = (reference_depth_max_m - reference_depth_min_m) / 2.0
    rows = []
    for scale in scale_factors:
        depth_min_m = depth_mid - depth_half_width * scale
        depth_max_m = depth_mid + depth_half_width * scale
        lateral_max_m = reference_lateral_max_m * scale
        tilt_max_deg = reference_tilt_max_deg * scale
        survive = within_band(
            depth_m, lateral_m, tilt_deg,
            depth_min_m=depth_min_m, depth_max_m=depth_max_m,
            lateral_max_m=lateral_max_m, tilt_max_deg=tilt_max_deg,
        )
        n_survive = int(survive.sum())
        rows.append({
            "scale_factor": scale,
            "depth_min_m": depth_min_m,
            "depth_max_m": depth_max_m,
            "lateral_max_m": lateral_max_m,
            "tilt_max_deg": tilt_max_deg,
            "n": n,
            "n_survive": n_survive,
            "survive_fraction": (n_survive / n) if n > 0 else None,
        })
    return rows


_LEG_ASSET_RE = re.compile(r"OneLegInsertionFixture__([A-Za-z0-9]+)")


def leg_asset_from_path(bank_path: str) -> str:
    """Recover the leg asset name from a bank's OWN directory path (``compute_pair_dir``'s own
    naming: ``.../Resets/<receptive>__<leg>/resets_....pt``). Returns the matched name, or the
    literal string ``"UNKNOWN"`` if the path does not contain a recognisable
    ``OneLegInsertionFixture__<asset>`` component -- NEVER a guess (team-lead: "a bank whose asset
    nobody can name may not be reusable regardless of geometry")."""
    m = _LEG_ASSET_RE.search(bank_path)
    return m.group(1) if m else "UNKNOWN"
