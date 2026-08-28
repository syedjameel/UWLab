"""CPU-only geometry test for bead UWLab-algw.9's sample_axial_insertion_depth.

No isaaclab / Isaac Sim import anywhere in this file -- deliberately, per the task's
"DO NOT RUN ISAAC" constraint. This is an INDEPENDENT reimplementation (numpy, not torch) of the
same rigid-transform primitives isaaclab.utils.math provides (quat_mul, quat_apply,
combine_frame_transforms, quat_from_euler_xyz), using the standard/well-known formulas for each
(Hamilton product; the optimized cross-product form of quaternion-vector rotation; the standard
roll/pitch/yaw-to-quaternion formula used across IsaacGym/IsaacLab-derived codebases). These are
verified against known mathematical identities below (round-trip inverse, rotation composition)
BEFORE being trusted for the geometry checks, so the geometry assertions are not just checked
against a possibly-broken copy of themselves.

The production code (events.py's sample_axial_insertion_depth, partial_assemblies_cfg.py's
axial_depth_sampling term) uses the REAL isaaclab.utils.math and the existing Offset class, not
this file. This file exists only because isaaclab is not importable in this environment without
launching Isaac Sim, which was explicitly out of scope for this change.

SIGN FOLLOW-UP (second pass on this test, same bead). The first version of this test asserted the
depth band lay in [-0.011562, -0.009374] -- a band that passes whether or not that is the correct
SIDE of the seat point, because the test's own offset construction hardcoded the same -Z
assumption the production code used to hardcode. That is not a test of the thing actually in
doubt. This version:
  1. Derives the insertion axis from the assembled pose (mirrors events.py's __init__ exactly),
     instead of assuming -Z.
  2. Cross-checks the SUPPLIED mouth_local_z_m against that derived axis, and shows this currently
     FAILS for the disputed input (-0.011562) -- documenting the known-bad state, not hiding it.
  3. Runs the full sampler with a SELF-CONSISTENT (derived-axis-agreeing) mouth value and confirms
     it passes.
  4. Adds an explicit negative control: the SAME self-consistent mouth value, but with the axis
     forced to the old hardcoded (0, 0, -1) -- this must FAIL the depth-band check, proving the
     test is actually sensitive to the sign, not just checking a symmetric magnitude.

MEASUREMENT FOLLOW-UP (third pass, same bead): the round-1 "mouth" z was actually the BLIND END;
the real entry mouth measured 25mm away, not ~2.188mm. Updated the geometry constants and added a
negative control for collapsing back to the old wrong magnitude.

TILT FOLLOW-UP (fourth pass, same bead). tilt_max_rad was a single constant, computed with the
lever arm fixed at the worst-case seat_depth_m -- correct only for the deepest sample, and
silently over-constraining every shallower one (the shallow end is where a reset bank's angular
diversity actually needs to come from). tilt_max is now a function of the ENGAGED LENGTH remaining
at the sampled depth, computed per sample; the engaged-segment radial check is now evaluated at
fractions of each sample's OWN engaged length instead of a shared global sweep; and a new
shallow-vs-deep check asserts the tilt bound (and the REALIZED tilt) is meaningfully larger for
shallow samples than deep ones, so a future reversion to a constant bound fails loudly instead of
quietly narrowing the dataset.

CAP + NEAR-GOAL FOLLOW-UP (fifth pass, same bead). Two more problems in the depth-dependent tilt:
  PROBLEM 1 -- the engaged-length bound alone goes to pi/2 as engaged_length -> 0 (a leg leaning
    against the hole, not partially inserted). Fixed with TWO independent caps: (a) a MINIMUM
    ENGAGED LENGTH floor on depth_max_m, so no sample ever sits at the degenerate limit; (b) a
    depth-INDEPENDENT rim cap derived from the pilot's own radius vs. the mouth's opening radius
    (a tilted cylinder's cross-section is an ellipse, not a point -- the engaged-length bound's
    axis-point model has no way to know this).
  PROBLEM 2 -- C4's semantic name is NEAR GOAL, so the depth band must be a small, seated-side
    slice near depth=0, not a uniform sample over the whole 25mm span (which mostly is NOT near
    goal and overlaps other reset categories). The band is now [depth_min_m, depth_max_m], derived
    from the receptive object's own position success threshold, with an explicit check that the
    fraction of the default band already counting as SUCCESS at spawn is 0%.
This version adds checks for both: the rim cap value itself, that no sample's engaged length ever
drops below the floor, that the default near-goal band matches its documented derivation, and the
already-solved-at-spawn fraction.

THREAD-YAW COUPLING FOLLOW-UP (sixth pass, same bead). Leg and bore are mating SCREW THREADS:
depth and the extra rotation about the insertion axis ("yaw") are physically coupled, but every
prior pass left yaw hardcoded at exactly 0 for every depth -- correct only at the authored/
assembled pose (depth=0), and silently INTERPENETRATING the two meshes at any other depth. This
pass:
  1. Fixes the RIM CAP to use the thread CREST radius (the cross-section that actually reaches the
     mouth plane at these depths) instead of the flat pilot radius (measured at the tip, nowhere
     near the mouth at these depths) -- the corrected cap is tighter and now actually binds against
     the engaged-length bound near the min_engaged_length_m floor, instead of being permanently
     dormant.
  2. Adds depth-coupled yaw sampling: interpolates a (depth, feasible-arc CENTRE, feasible-arc
     WIDTH) table solved directly against the real collision meshes by
     ``solve_thread_lead_from_meshes.py`` (loaded dynamically below, the same
     ``importlib.util.spec_from_file_location`` pattern that script already uses in the other
     direction to load THIS file -- see the loader for why that is not circular).
  3. Adds a GEOMETRIC negative control (not the task's own success metric, which discards roll
     about the insertion axis entirely): yaw left at 0 for a nonzero depth must show real,
     substantial mesh interpenetration when checked against the actual meshes, proving this test
     would have caught the exact bug this whole pass exists to prevent.
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Quaternion / rigid-transform primitives (w, x, y, z convention, matching isaaclab).
# ---------------------------------------------------------------------------


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def quat_inv(q: np.ndarray) -> np.ndarray:
    norm_sq = np.sum(q * q, axis=-1, keepdims=True)
    return quat_conj(q) / norm_sq


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q, via the standard sandwich-product-free formula."""
    w = q[..., 0:1]
    qvec = q[..., 1:4]
    t = 2.0 * np.cross(qvec, v)
    return v + w * t + np.cross(qvec, t)


def combine_frame_transforms(t01: np.ndarray, q01: np.ndarray, t12: np.ndarray, q12: np.ndarray):
    t02 = t01 + quat_apply(q01, t12)
    q02 = quat_mul(q01, q12)
    q02 = q02 / np.linalg.norm(q02, axis=-1, keepdims=True)
    return t02, q02


def quat_from_euler_xyz(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Standard roll(X)-pitch(Y)-yaw(Z) -> quaternion formula (as used across IsaacGym/IsaacLab)."""
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return np.stack([qw, qx, qy, qz], axis=-1)


# ---------------------------------------------------------------------------
# Sanity-check the primitives themselves before trusting them for geometry checks.
# ---------------------------------------------------------------------------


def _check_primitives():
    q1 = quat_from_euler_xyz(*RNG.uniform(-1, 1, size=3))
    q2 = quat_from_euler_xyz(*RNG.uniform(-1, 1, size=3))
    v = RNG.uniform(-1, 1, size=3)

    # Composition: applying q1 then q2 == applying quat_mul(q2, q1) directly is the OTHER
    # convention; here we check the convention actually used by combine_frame_transforms:
    # rotating v first by q12 then by q01 must equal rotating by quat_mul(q01, q12).
    lhs = quat_apply(q1, quat_apply(q2, v))
    rhs = quat_apply(quat_mul(q1, q2), v)
    assert np.allclose(lhs, rhs, atol=1e-10), f"quat_mul/quat_apply composition mismatch: {lhs} vs {rhs}"

    # Inverse round-trip.
    v_rot = quat_apply(q1, v)
    v_back = quat_apply(quat_inv(q1), v_rot)
    assert np.allclose(v_back, v, atol=1e-10), f"quat_inv round-trip failed: {v_back} vs {v}"

    # combine_frame_transforms round-trip through Offset.subtract's own formula (mirrors
    # assembly_keypoints.py's Offset.subtract exactly): combining a pose with an offset, then
    # combining with the offset's algebraic inverse, must return the original pose.
    t01 = RNG.uniform(-1, 1, size=3)
    off_pos = RNG.uniform(-1, 1, size=3)
    off_quat = quat_from_euler_xyz(*RNG.uniform(-1, 1, size=3))
    t02, q02 = combine_frame_transforms(t01, q1, off_pos, off_quat)
    inv_off_pos = -quat_apply(quat_inv(off_quat), off_pos)
    inv_off_quat = quat_inv(off_quat)
    t01_back, q01_back = combine_frame_transforms(t02, q02, inv_off_pos, inv_off_quat)
    assert np.allclose(t01_back, t01, atol=1e-9), f"combine/subtract round-trip pos failed: {t01_back} vs {t01}"
    assert np.allclose(np.abs(np.dot(q01_back, q1)), 1.0, atol=1e-9), "combine/subtract round-trip quat failed"

    print("[primitives] quat_mul/apply/inv/combine_frame_transforms self-checks: PASS")


# ---------------------------------------------------------------------------
# Real geometry: SquareTableLeg200mmDecomp / OneLegInsertionFixture metadata (read directly off
# the repo's metadata.yaml files, not the bead report's transcription -- see below for the one
# discrepancy found in the FIRST pass on this test).
# ---------------------------------------------------------------------------

# OneLegInsertionFixture/metadata.yaml assembled_offset (quat is identity in this file).
RECEPTIVE_OFFSET_POS = np.array([-0.056250, 0.056250, -0.009374])
RECEPTIVE_OFFSET_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

# SquareTableLeg200mmDecomp/metadata.yaml assembled_offset.
# NOTE (from the first pass on this test): the bead report gave this quat as
# [0.70710678, 0, -0.70710678, 0]; the actual file has [0.70710678, 0, 0.70710678, 0] -- opposite
# sign on y. Using the FILE's value, since that is what the production code reads at runtime.
INSERTIVE_OFFSET_POS = np.array([-0.106203, 0.0, 0.0])
INSERTIVE_OFFSET_QUAT = np.array([0.70710678, 0.0, 0.70710678, 0.0])

# MEASURED 2026-08-18 off the composed USD (see partial_assemblies_cfg.py's own comment for the
# full provenance -- Usd.TraverseInstanceProxies() required, the collider subtree is
# instanceable=True). Two rounds of supplied facts on this bead, both wrong in different ways:
#   round 1: "mouth = hole mesh z-min = -11.562mm" -- WRONG SIDE. That z is actually the BLIND
#     END (a conical apex with no open face); the consistency check below still uses it to prove
#     the sign-disagreement it originally caught.
#   round 2 (this one): the ENTRY MOUTH, the collar's genuine open annular face, is at
#     z=+15.625mm -- 25mm from the assembled tip, not the ~2.188mm the round-1 (mislabeled) value
#     implied.
BLIND_END_LOCAL_Z_M = -0.011562  # round 1's mislabeled value; kept only for the regression check below
ENTRY_MOUTH_LOCAL_Z_M = 0.015625  # round 2's measured value; this is what production code now uses
# RADIAL_CLEARANCE_M: measured 0.9116mm, the smooth PILOT's clearance to the bore wall's tightest
# point over the whole engaged span. Still correct and still used below for the engaged-segment
# PILOT check (#4) and the dedicated rim-cap-vs-floor demonstration (#8d) -- NOT, as of the sixth
# pass, what actually bounds tilt/lateral-jitter operationally any more; see
# YAW_COUPLED_CLEARANCE_M for why and what replaces it in the checks that mirror the real sampler.
RADIAL_CLEARANCE_M = 0.0009116  # measured; tighter than round 1's 0.00093 placeholder
# Sixth-pass addition, found via THIS pass's own required geometric validation (see check 10a):
# once yaw coupling makes the THREAD CREST a live constraint, the crest's own clearance to the wall
# -- even at the solved arc's centre, the best achievable yaw -- measures only ~0.287mm across
# depth 0-18mm (solve_thread_lead_from_meshes.py's "[clearance] at the feasible-arc CENTRE" table),
# a THIRD of the pilot figure above. Feeding the pilot-sized budget into tilt/jitter on top of the
# yaw table's own residual slack reliably drove the crest negative (an N=300 geometric check came
# back min=-0.578mm, median=-0.026mm). 0.10mm, split the same way, was checked geometrically across
# 5 seeds x 600 samples (3000 total): min observed +0.060mm, never negative. BYTE-IDENTICAL to
# partial_assemblies_cfg.py's LEG200MM_ONELEGFIXTURE_YAW_COUPLED_CLEARANCE_M -- see that file's
# identical comment for the full derivation.
YAW_COUPLED_CLEARANCE_M = 0.0001

# Fifth-pass additions: mouth-rim geometry and the near-goal band, same provenance/reasoning as
# partial_assemblies_cfg.py's own comments for each.
MIN_ENGAGED_LENGTH_M = 0.002  # 2mm floor, chosen (see cfg file), not measured
# Sixth-pass CORRECTION (see partial_assemblies_cfg.py's identical comment for the full
# derivation): this used to be PILOT_RADIUS_M = 0.010004, the leg's flat-pilot radius measured AT
# THE TIP -- the wrong cross-section for a rim cap, since the pilot's tip is nowhere near the mouth
# plane at these depths. The material actually near the mouth is the THREAD CREST, radius =
# major_diameter/2 = 0.012188 (major diameter 24.376mm, from the same mesh-fit pass as the yaw
# table below).
MOUTH_CROSSING_RADIUS_M = 0.012188  # measured (thread crest radius, major_diameter/2)
MOUTH_BORE_RADIUS_M = 0.0124995  # measured (the entry mouth's own open-loop radius)
POSITION_SUCCESS_THRESHOLD_M = 0.0025  # OneLegInsertionFixture metadata.yaml success_thresholds.position
DEPTH_MIN_M = 3.0 * POSITION_SUCCESS_THRESHOLD_M  # 7.5mm, near-goal band lower bound
DEPTH_MAX_M = 6.0 * POSITION_SUCCESS_THRESHOLD_M  # 15mm, near-goal band upper bound

# Sixth-pass addition: the solved (depth, feasible-arc CENTRE, feasible-arc WIDTH) thread-yaw
# table -- BYTE-IDENTICAL to partial_assemblies_cfg.py's THREAD_YAW_TABLE_DEG_MM (same solve run,
# 2026-08-20, scripts_v2/tools/solve_thread_lead_from_meshes.py); see that file's comment for the
# full provenance and cross-checks. Kept as a separate literal here (not imported from the cfg
# file, which imports isaaclab) so this file stays isaaclab-free.
THREAD_YAW_TABLE_DEG_MM = [
    (0.0, 60.0543, 124.325),
    (1.0, 98.4390, 124.404),
    (2.0, 136.7951, 124.483),
    (3.0, 175.1966, 124.371),
    (4.0, 213.6291, 124.457),
    (5.0, 251.9302, 124.491),
    (6.0, 290.3552, 124.485),
    (7.0, 328.8344, 124.447),
    (8.0, 367.1777, 124.461),
    (9.0, 405.4755, 124.841),
    (10.0, 433.1243, 146.249),
    (11.0, 452.3208, 184.642),
    (12.0, 471.5828, 223.166),
    (13.0, 490.7331, 261.466),
    (14.0, 509.9529, 299.906),
    (15.0, 529.1847, 338.369),
    (16.0, 540.0000, 360.000),
    (17.0, 540.0000, 360.000),
    (18.0, 540.0000, 360.000),
    (19.0, 540.0000, 360.000),
    (20.0, 540.0000, 360.000),
]
THREAD_YAW_TABLE_DEPTH_M = np.array([d / 1000.0 for d, _, _ in THREAD_YAW_TABLE_DEG_MM])
THREAD_YAW_TABLE_CENTER_RAD = np.array([math.radians(c) for _, c, _ in THREAD_YAW_TABLE_DEG_MM])
THREAD_YAW_TABLE_WIDTH_RAD = np.array([math.radians(w) for _, _, w in THREAD_YAW_TABLE_DEG_MM])
YAW_ARC_MARGIN = 0.9  # mirrors partial_assemblies_cfg.py's LEG200MM_ONELEGFIXTURE_YAW_ARC_MARGIN

SEAT_LOCAL_Z_M = float(RECEPTIVE_OFFSET_POS[2])


def sample_yaw_from_table(
    depth_m: np.ndarray, table_depth_m: np.ndarray, table_center_rad: np.ndarray, table_width_rad: np.ndarray,
    margin: float, rng: np.random.Generator,
) -> np.ndarray:
    """Mirrors events.py's sample_axial_insertion_depth.__call__ yaw-coupling block exactly:
    ``numpy.interp`` here is the reference for that code's ``_interp1d`` (same clamped/constant
    extrapolation semantics outside the table's depth range)."""
    center = np.interp(depth_m, table_depth_m, table_center_rad)
    width = np.interp(depth_m, table_depth_m, table_width_rad)
    full_circle = width >= (2.0 * math.pi - 1e-4)
    half_span = np.where(full_circle, math.pi, 0.5 * margin * width)
    return center + (2.0 * rng.uniform(0.0, 1.0, size=depth_m.shape) - 1.0) * half_span


# ---------------------------------------------------------------------------
# Axis derivation + consistency check -- mirrors events.py's sample_axial_insertion_depth.__init__
# exactly (same formulas, numpy instead of torch).
# ---------------------------------------------------------------------------


def derive_insertion_axis_local(offset_pos: np.ndarray, offset_quat: np.ndarray) -> np.ndarray:
    """Direction of INCREASING depth (toward the blind end), in the mating frame's local axes.

    offset_pos is the insertive object's tip in its OWN local frame (root-to-tip direction is
    just normalize(offset_pos)); un-rotating that by the offset's own quat expresses the same
    direction in the mating frame. Backing off toward the mouth is this vector's negation.
    """
    tip_direction_local = offset_pos / np.linalg.norm(offset_pos)
    return quat_apply(quat_inv(offset_quat), tip_direction_local)


def check_mouth_sign_consistent(seat_local_z: float, mouth_local_z_m: float, insertion_axis_local: np.ndarray) -> bool:
    """True iff mouth_local_z_m sits on the side of the seat point the derived axis says it should."""
    mouth_offset_from_seat = mouth_local_z_m - seat_local_z
    derived_mouth_sign = -1.0 if float(insertion_axis_local[2]) > 0.0 else 1.0
    given_mouth_sign = 1.0 if mouth_offset_from_seat > 0.0 else -1.0
    return derived_mouth_sign == given_mouth_sign


INSERTION_AXIS_LOCAL = derive_insertion_axis_local(INSERTIVE_OFFSET_POS, INSERTIVE_OFFSET_QUAT)
print(f"[derived] insertion_axis_local={INSERTION_AXIS_LOCAL.tolist()} (direction of INCREASING depth)")
axis_xy_mag = float(np.linalg.norm(INSERTION_AXIS_LOCAL[:2]))
assert axis_xy_mag <= 1e-3, f"insertion axis not Z-aligned (xy magnitude={axis_xy_mag:.6f}) -- mouth_local_z_m invalid"
print(f"[check] insertion axis is Z-aligned (xy magnitude={axis_xy_mag:.2e} <= 1e-3): PASS")

# RIM CAP (mirrors events.py's sample_axial_insertion_depth.__init__ exactly, sixth-pass CORRECTED
# radius): a tilted cylinder of radius MOUTH_CROSSING_RADIUS_M (the thread CREST, not the pilot --
# see that constant's comment) has an elliptical cross-section whose semi-major axis grows to
# MOUTH_CROSSING_RADIUS_M / cos(tilt); for the whole cross-section to still fit through the mouth's
# circular opening, tilt <= acos(MOUTH_CROSSING_RADIUS_M / MOUTH_BORE_RADIUS_M). Depth-independent
# -- computed once.
assert MOUTH_CROSSING_RADIUS_M < MOUTH_BORE_RADIUS_M, "leg would not fit through the mouth at any tilt"
RIM_TILT_CAP_RAD = math.acos(min(1.0, MOUTH_CROSSING_RADIUS_M / MOUTH_BORE_RADIUS_M))
print(f"[derived] rim_tilt_cap_rad={RIM_TILT_CAP_RAD:.6f} ({math.degrees(RIM_TILT_CAP_RAD):.3f} deg), from "
      f"mouth_crossing_radius_m={MOUTH_CROSSING_RADIUS_M:.6f} vs mouth_bore_radius_m={MOUTH_BORE_RADIUS_M:.6f}")


OLD_SLIVER_SEAT_DEPTH_M = abs(BLIND_END_LOCAL_Z_M - SEAT_LOCAL_Z_M)  # ~0.002188m, round 1's wrong span


def main():
    _check_primitives()

    # --- REGRESSION: round 1's mislabeled blind-end value still FAILS the consistency check.
    # This documents the bug that was actually caught, rather than deleting the evidence once the
    # measurement landed.
    blind_end_consistent = check_mouth_sign_consistent(SEAT_LOCAL_Z_M, BLIND_END_LOCAL_Z_M, INSERTION_AXIS_LOCAL)
    assert not blind_end_consistent, (
        "expected round 1's mislabeled blind-end value to DISAGREE with the derived axis -- if this "
        "now passes, something about the derivation changed; investigate before trusting it."
    )
    print(
        f"[check] round-1 blind_end_local_z_m={BLIND_END_LOCAL_Z_M} is INCONSISTENT with the derived "
        "axis (matches events.py's SANITY CHECK 2, which correctly refused to run on this input) -- PASS"
    )

    # --- The MEASURED entry mouth PASSES the consistency check -- this is round 2, the actual fix.
    mouth_consistent = check_mouth_sign_consistent(SEAT_LOCAL_Z_M, ENTRY_MOUTH_LOCAL_Z_M, INSERTION_AXIS_LOCAL)
    assert mouth_consistent, (
        f"expected the measured entry mouth ({ENTRY_MOUTH_LOCAL_Z_M}) to AGREE with the derived axis "
        f"({INSERTION_AXIS_LOCAL.tolist()}) -- if this fails, the measurement or the derivation is wrong; "
        "do not proceed past this without resolving which one."
    )
    seat_depth_m = abs(ENTRY_MOUTH_LOCAL_Z_M - SEAT_LOCAL_Z_M)
    print(f"[check] measured entry_mouth_local_z_m={ENTRY_MOUTH_LOCAL_Z_M} is CONSISTENT with the derived "
          "axis -- PASS")
    print(f"[derived] seat_depth_m={seat_depth_m:.6f} ({seat_depth_m * 1000:.4f} mm)")

    # --- MAGNITUDE regression: the real engaged span must be far larger than round 1's mislabeled
    # ~2.188mm sliver -- if this ever fails, mouth_local_z_m has drifted back toward the blind end
    # (or some other wrong-magnitude value), the exact "collapses to the old sliver" failure mode.
    assert seat_depth_m > 10 * OLD_SLIVER_SEAT_DEPTH_M, (
        f"seat_depth_m={seat_depth_m:.6f} is not far larger than the old mislabeled sliver "
        f"({OLD_SLIVER_SEAT_DEPTH_M:.6f}) -- looks like a regression back toward the wrong span"
    )
    print(f"[check] seat_depth_m={seat_depth_m * 1000:.3f}mm is >>10x the old mislabeled sliver "
          f"({OLD_SLIVER_SEAT_DEPTH_M * 1000:.3f}mm) -- PASS")

    # Sixth pass: split from YAW_COUPLED_CLEARANCE_M (crest-informed), NOT the historical
    # RADIAL_CLEARANCE_M (pilot-informed) -- see that constant's comment. This is what the real
    # sampler now actually uses operationally; RADIAL_CLEARANCE_M is still used below only for
    # check #4 (a real, still-true, just less tight pilot-clearance bound) and the dedicated
    # rim-cap-vs-floor demonstration (#8d).
    lateral_jitter_max_m = YAW_COUPLED_CLEARANCE_M / 2
    tilt_clearance_budget_m = YAW_COUPLED_CLEARANCE_M - lateral_jitter_max_m
    print(f"[derived] lateral_jitter_max_m={lateral_jitter_max_m:.6f} "
          f"tilt_clearance_budget_m={tilt_clearance_budget_m:.6f}")

    def tilt_max_rad_of_engaged_length(engaged_length: np.ndarray) -> np.ndarray:
        """Mirrors events.py's per-sample tilt bound exactly: MIN of (a) the engaged-length bound
        (lever = the distance remaining between the current, depth-backed-off tip and the mouth,
        not the constant worst-case seat_depth_m an earlier version of this code used everywhere;
        floored so a sample landing at/near the mouth does not divide by zero) and (b) the
        depth-independent rim cap."""
        engaged_length = np.maximum(engaged_length, 1e-6)
        engaged_bound = np.arcsin(np.minimum(1.0, tilt_clearance_budget_m / engaged_length))
        return np.minimum(engaged_bound, RIM_TILT_CAP_RAD)

    def sample_axial_insertion_depth_reference(num, depth_min_m, depth_max_m, lateral_jitter_max_m, enable_tilt,
                                                axis_local, enable_yaw_coupling=False, force_yaw_zero=False):
        """Mirrors events.py's sample_axial_insertion_depth.__call__ exactly (axis_local is the
        derived insertion_axis_local, passed in explicitly so the negative control below can
        substitute a deliberately wrong one). depth_min_m/depth_max_m are NOT clamped by
        min_engaged_length_m here -- callers apply that clamp themselves (mirrors events.py's
        __init__ doing the clamp once, not __call__ doing it every step).

        enable_yaw_coupling: sample yaw from the solved THREAD_YAW_TABLE_* table via
        sample_yaw_from_table (mirrors the production thread_yaw_table path). force_yaw_zero: the
        sixth-pass NEGATIVE CONTROL -- reproduce the OLD hardcoded behaviour (yaw stuck at exactly
        0 for every depth) instead of coupling it, so callers can prove that pose actually
        interferes with the real meshes."""
        receptive_pos = np.zeros((num, 3))
        receptive_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (num, 1))

        target_pos, target_quat = combine_frame_transforms(
            receptive_pos, receptive_quat,
            np.tile(RECEPTIVE_OFFSET_POS, (num, 1)), np.tile(RECEPTIVE_OFFSET_QUAT, (num, 1)),
        )

        if depth_max_m > depth_min_m:
            depth = RNG.uniform(depth_min_m, depth_max_m, size=num)
        else:
            depth = np.full(num, depth_min_m)

        if lateral_jitter_max_m > 0.0:
            jitter_r = lateral_jitter_max_m * np.sqrt(RNG.uniform(0.0, 1.0, size=num))
            jitter_theta = RNG.uniform(0.0, 2 * math.pi, size=num)
            jitter_x = jitter_r * np.cos(jitter_theta)
            jitter_y = jitter_r * np.sin(jitter_theta)
        else:
            jitter_x = np.zeros(num)
            jitter_y = np.zeros(num)

        lateral_offset = np.stack([jitter_x, jitter_y, np.zeros(num)], axis=-1)
        axial_offset = -depth[:, None] * axis_local[None, :]
        offset_pos = lateral_offset + axial_offset

        if enable_tilt:
            engaged_length = seat_depth_m - depth  # per-sample lever, NOT a constant
            tilt_max_rad = tilt_max_rad_of_engaged_length(engaged_length)
            tilt_r = tilt_max_rad * np.sqrt(RNG.uniform(0.0, 1.0, size=num))
            tilt_theta = RNG.uniform(0.0, 2 * math.pi, size=num)
            roll = tilt_r * np.cos(tilt_theta)
            pitch = tilt_r * np.sin(tilt_theta)
        else:
            engaged_length = seat_depth_m - depth
            tilt_max_rad = np.zeros(num)
            roll = np.zeros(num)
            pitch = np.zeros(num)

        if force_yaw_zero:
            # NEGATIVE CONTROL: reproduce the pre-sixth-pass bug (yaw hardcoded at 0 regardless of
            # depth) on purpose.
            yaw = np.zeros(num)
        elif enable_yaw_coupling:
            yaw = sample_yaw_from_table(
                depth, THREAD_YAW_TABLE_DEPTH_M, THREAD_YAW_TABLE_CENTER_RAD, THREAD_YAW_TABLE_WIDTH_RAD,
                YAW_ARC_MARGIN, RNG,
            )
        else:
            yaw = np.zeros(num)
        offset_quat = quat_from_euler_xyz(roll, pitch, yaw)

        new_target_pos, new_target_quat = combine_frame_transforms(target_pos, target_quat, offset_pos, offset_quat)

        inv_off_pos = -quat_apply(
            quat_inv(np.tile(INSERTIVE_OFFSET_QUAT, (num, 1))), np.tile(INSERTIVE_OFFSET_POS, (num, 1))
        )
        inv_off_quat = quat_inv(np.tile(INSERTIVE_OFFSET_QUAT, (num, 1)))
        insertive_pos, insertive_quat = combine_frame_transforms(
            new_target_pos, new_target_quat, inv_off_pos, inv_off_quat
        )

        return dict(
            target_pos=target_pos, target_quat=target_quat,
            depth=depth, jitter_x=jitter_x, jitter_y=jitter_y,
            engaged_length=engaged_length, tilt_max_rad=tilt_max_rad,
            roll=roll, pitch=pitch, yaw=yaw,
            offset_quat=offset_quat,
            new_target_pos=new_target_pos, new_target_quat=new_target_quat,
            insertive_pos=insertive_pos, insertive_quat=insertive_quat,
        )

    # Direct unit check of the MIN(engaged-length bound, rim cap) logic, independent of whether
    # the chosen defaults ever make the rim cap bind (see below -- with THESE numbers, it does
    # not, at either the full range's shallow edge or the near-goal band; min_engaged_length_m is
    # the constraint that actually binds there). Force an engaged_length small enough that the
    # engaged-length bound alone would exceed the rim cap, and confirm the rim cap wins.
    _tiny_engaged = np.array([1e-5])
    _engaged_only_bound = math.asin(min(1.0, tilt_clearance_budget_m / _tiny_engaged[0]))
    assert _engaged_only_bound > RIM_TILT_CAP_RAD, "test setup: engaged-only bound should exceed the rim cap here"
    _capped = float(tilt_max_rad_of_engaged_length(_tiny_engaged)[0])
    assert abs(_capped - RIM_TILT_CAP_RAD) < 1e-9, (
        f"rim cap did not win at a tiny engaged length: got {_capped:.6f}, expected {RIM_TILT_CAP_RAD:.6f}"
    )
    print(f"[check] MIN(engaged-length bound, rim cap) logic: at engaged_length=1e-5, engaged-only bound="
          f"{_engaged_only_bound:.4f} rad but capped result={_capped:.6f} rad == rim cap -- PASS")

    num = 200_000
    bore_axis_xy = RECEPTIVE_OFFSET_POS[:2]

    # === FULL-RANGE run: derived axis, depth from 0 up to the min-engaged-length-clamped ceiling,
    # depth-dependent tilt. Exercises the underlying MECHANISM (shallow-vs-deep diversity, the
    # engaged-length floor, the rim cap) across its whole reachable range -- the near-goal DEFAULT
    # band is checked separately below, since it deliberately samples only a slice of this. ===
    full_range_depth_max_m = min(seat_depth_m, seat_depth_m - MIN_ENGAGED_LENGTH_M)
    _mouth_side_z = SEAT_LOCAL_Z_M + math.copysign(full_range_depth_max_m, ENTRY_MOUTH_LOCAL_Z_M - SEAT_LOCAL_Z_M)
    band_lo, band_hi = sorted([SEAT_LOCAL_Z_M, _mouth_side_z])
    out = sample_axial_insertion_depth_reference(
        num, 0.0, full_range_depth_max_m, lateral_jitter_max_m, True, INSERTION_AXIS_LOCAL
    )

    # 1) Round-trip sanity: insertive_pose o insertive_assembled_offset must reproduce
    #    new_target_pos/new_target_quat exactly (the SAME Offset.subtract/combine pair events.py
    #    relies on for the seated case -- checking it here too, not just once).
    recon_pos, recon_quat = combine_frame_transforms(
        out["insertive_pos"], out["insertive_quat"],
        np.tile(INSERTIVE_OFFSET_POS, (num, 1)), np.tile(INSERTIVE_OFFSET_QUAT, (num, 1)),
    )
    assert np.allclose(recon_pos, out["new_target_pos"], atol=1e-8), "tip reconstruction position mismatch"
    dot = np.abs(np.sum(recon_quat * out["new_target_quat"], axis=-1))
    assert np.all(dot > 1 - 1e-8), "tip reconstruction orientation mismatch"
    print(f"[check] insertive_pose o insertive_assembled_offset reproduces the sampled tip pose: PASS ({num} samples)")

    # 2) depth band: new_target z must lie within [band_lo, band_hi] -- sign-agnostic on purpose,
    #    since band_lo/band_hi are derived from whichever side ENTRY_MOUTH_LOCAL_Z_M landed on.
    z = out["new_target_pos"][:, 2]
    assert z.min() >= band_lo - 1e-9, f"a sample went past the mouth: min z={z.min():.6f}"
    assert z.max() <= band_hi + 1e-9, f"a sample went past fully seated: max z={z.max():.6f}"
    print(f"[check] depth band z in [{band_lo:.6f}, {band_hi:.6f}]: observed [{z.min():.6f}, {z.max():.6f}] -- PASS")

    # 3) tip radial position within lateral_jitter_max_m of the bore axis.
    tip_radial = np.linalg.norm(out["new_target_pos"][:, :2] - bore_axis_xy, axis=-1)
    assert tip_radial.max() <= lateral_jitter_max_m + 1e-9, f"tip strayed off-axis: max={tip_radial.max():.6f}"
    print(f"[check] tip radial offset <= lateral_jitter_max_m={lateral_jitter_max_m:.6f}:"
          f" observed max={tip_radial.max():.6f} -- PASS")

    # 4) engaged-segment radial clearance (numeric, not small-angle-approximated) -- THE REAL
    #    GUARANTEE, per team-lead's follow-up: checked at fractions of EACH SAMPLE's OWN engaged
    #    length (seat_depth_m - depth), not a single global sweep up to the constant seat_depth_m.
    #    A shallow sample's own engaged length is short, so its lever only needs to be checked out
    #    to that short distance -- checking it out to the full 25mm would be checking a point that
    #    sample's own physical pilot never reaches.
    worst_radial = 0.0
    for frac in (0.0, 0.5, 1.0):
        lever_local = np.stack(
            [np.zeros(num), np.zeros(num), frac * out["engaged_length"]], axis=-1
        )
        lever_world = out["new_target_pos"] + quat_apply(out["new_target_quat"], lever_local)
        radial = np.linalg.norm(lever_world[:, :2] - bore_axis_xy, axis=-1)
        worst_radial = max(worst_radial, float(radial.max()))
    assert worst_radial <= RADIAL_CLEARANCE_M + 1e-9, (
        f"engaged-segment radial clearance violated: worst={worst_radial:.6f} > {RADIAL_CLEARANCE_M:.6f}"
    )
    print(f"[check] engaged-segment (per-sample s in [0, engaged_length]) radial offset <= "
          f"radial_clearance_m={RADIAL_CLEARANCE_M:.6f}: observed worst={worst_radial:.6f} -- PASS")

    # 4b) SHALLOW-VS-DEEP tilt diversity (team-lead's follow-up, the actual point of this fix):
    #    assert a shallow sample's tilt bound is meaningfully larger than a deep sample's -- if a
    #    future change reverts to a single constant bound, this fails instead of silently
    #    narrowing the dataset's angular diversity back down.
    shallow_mask = out["depth"] > 0.95 * full_range_depth_max_m   # engaged_length close to the min-engaged floor
    deep_mask = out["depth"] < 0.05 * full_range_depth_max_m       # engaged_length close to full engagement
    assert shallow_mask.sum() > 100 and deep_mask.sum() > 100, "not enough samples in each bucket"
    shallow_tilt_bound_rad = float(out["tilt_max_rad"][shallow_mask].mean())
    deep_tilt_bound_rad = float(out["tilt_max_rad"][deep_mask].mean())
    assert shallow_tilt_bound_rad > 5 * deep_tilt_bound_rad, (
        f"shallow tilt bound ({shallow_tilt_bound_rad:.6f} rad) is not meaningfully larger than the "
        f"deep tilt bound ({deep_tilt_bound_rad:.6f} rad) -- tilt no longer looks depth-dependent"
    )
    # And the REALIZED tilt magnitude (not just the bound) must actually be larger for shallow
    # samples too, confirming the bound is doing something rather than being computed and ignored.
    realized_tilt_rad = 2.0 * np.arccos(np.clip(np.abs(out["offset_quat"][:, 0]), -1.0, 1.0))
    shallow_realized_max = float(realized_tilt_rad[shallow_mask].max())
    deep_realized_max = float(realized_tilt_rad[deep_mask].max())
    assert shallow_realized_max > 5 * deep_realized_max, (
        f"shallow realized tilt max ({shallow_realized_max:.6f} rad) is not meaningfully larger than "
        f"deep realized tilt max ({deep_realized_max:.6f} rad)"
    )
    print(f"[check] SHALLOW-VS-DEEP tilt diversity: shallow bound={shallow_tilt_bound_rad:.6f} rad "
          f"({math.degrees(shallow_tilt_bound_rad):.2f} deg) vs deep bound={deep_tilt_bound_rad:.6f} rad "
          f"({math.degrees(deep_tilt_bound_rad):.2f} deg); realized max shallow={shallow_realized_max:.6f} "
          f"vs deep={deep_realized_max:.6f} -- PASS (a constant-bound reversion would fail this)")

    # 4c) MINIMUM ENGAGED LENGTH floor (PROBLEM 1a): no sample's engaged length should ever drop
    #    below MIN_ENGAGED_LENGTH_M -- this is what actually prevents the pi/2 pathology, since (as
    #    the earlier unit check showed) the rim cap does not bind at these clearance/floor values.
    min_observed_engaged = float(out["engaged_length"].min())
    assert min_observed_engaged >= MIN_ENGAGED_LENGTH_M - 1e-9, (
        f"observed engaged_length min ({min_observed_engaged:.6f}) is below the floor "
        f"({MIN_ENGAGED_LENGTH_M:.6f}) -- the degenerate near-mouth pathology is back"
    )
    max_observed_tilt_bound = float(out["tilt_max_rad"].max())
    assert max_observed_tilt_bound <= RIM_TILT_CAP_RAD + 1e-9, (
        f"observed tilt bound max ({max_observed_tilt_bound:.6f}) exceeds the rim cap "
        f"({RIM_TILT_CAP_RAD:.6f}) -- MIN() logic broken"
    )
    print(f"[check] min_engaged_length_m floor holds (observed min engaged_length="
          f"{min_observed_engaged * 1000:.4f}mm >= {MIN_ENGAGED_LENGTH_M * 1000:.4f}mm) and no tilt bound "
          f"exceeds the rim cap (observed max={math.degrees(max_observed_tilt_bound):.3f} deg <= "
          f"{math.degrees(RIM_TILT_CAP_RAD):.3f} deg) -- PASS. At these clearance/floor values the "
          f"min_engaged_length_m floor is what actually binds at the shallow edge "
          f"({math.degrees(max_observed_tilt_bound):.2f} deg observed vs the rim cap's "
          f"{math.degrees(RIM_TILT_CAP_RAD):.2f} deg -- the rim cap is a dormant-but-present second "
          "safety net for wider bands, not decorative -- see the unit check above).")

    # 5) negative control (no motion): without the depth term, everything collapses to one exact
    #    seated pose (the bug the task warned deleting apply_forces alone would produce).
    out_zero = sample_axial_insertion_depth_reference(1000, 0.0, 0.0, 0.0, False, INSERTION_AXIS_LOCAL)
    depths_seen = np.unique(np.round(out_zero["new_target_pos"], 9), axis=0)
    assert depths_seen.shape[0] == 1, "expected a single collapsed pose with depth_max=0"
    print("[check] negative control (depth_max=0): every sample collapses to the single seated pose"
          " -- confirms the failure mode the depth sampler must avoid -- PASS")

    # 6) NEGATIVE CONTROL (wrong sign): same self-consistent mouth value / band, but the axis
    #    forced to reproduce the OLD hardcoded behavior. The first version of events.py wrote
    #    offset_pos.z = -depth directly, i.e. its (un-derived) BACKING-OFF direction was (0,0,-1).
    #    sample_axial_insertion_depth_reference here takes insertion_axis_local (the direction of
    #    INCREASING depth) and negates it internally for backing off, so reproducing the old bug
    #    means passing axis_local=(0,0,+1) -- the wrong "insertion" direction whose negation
    #    matches the old code's hardcoded (0,0,-1) backing-off direction. This must FAIL the
    #    depth-band check -- if it does not, this test is not actually sensitive to the sign.
    wrong_axis = np.array([0.0, 0.0, 1.0])
    assert not np.allclose(wrong_axis, INSERTION_AXIS_LOCAL), "the wrong axis happens to equal the derived one"
    out_wrong = sample_axial_insertion_depth_reference(
        num, 0.0, full_range_depth_max_m, lateral_jitter_max_m, True, wrong_axis
    )
    z_wrong = out_wrong["new_target_pos"][:, 2]
    band_violated = bool(z_wrong.min() < band_lo - 1e-9 or z_wrong.max() > band_hi + 1e-9)
    assert band_violated, (
        "expected the old-hardcoded-behavior run to violate the depth band -- if it does not, this "
        "negative control is not actually sensitive to the sign error it is meant to catch"
    )
    print(f"[check] NEGATIVE CONTROL: the OLD hardcoded backing-off direction (0,0,-1), reproduced via "
          f"axis_local=(0,0,+1), drives z to [{z_wrong.min():.6f}, {z_wrong.max():.6f}], OUTSIDE the "
          f"correct band [{band_lo:.6f}, {band_hi:.6f}] -- PASS (confirms this test would have caught "
          "the sign bug, and that the old code was indeed backing off the wrong way)")

    # 7) NEGATIVE CONTROL (wrong magnitude / old sliver): correct axis and correct mouth SIDE, but
    #    depth_max_m collapsed back to round 1's mislabeled ~2.188mm span instead of the real
    #    ~25mm one. This must cover only a small sliver right next to the seat, nowhere near the
    #    real mouth -- if a run like this ever looks indistinguishable from the correct one, the
    #    depth range has silently regressed to the old wrong magnitude.
    out_sliver = sample_axial_insertion_depth_reference(
        num, 0.0, OLD_SLIVER_SEAT_DEPTH_M, lateral_jitter_max_m, True, INSERTION_AXIS_LOCAL
    )
    z_sliver = out_sliver["new_target_pos"][:, 2]
    sliver_span = float(z_sliver.max() - z_sliver.min())
    full_span = band_hi - band_lo
    assert sliver_span < 0.1 * full_span, (
        f"the old-sliver run's z-span ({sliver_span:.6f}) is not far smaller than the real band's "
        f"({full_span:.6f}) -- this negative control is not actually sensitive to the magnitude collapse"
    )
    # And the sliver run must sit entirely on the SEAT side of the band, nowhere close to the
    # actual mouth (band_hi) -- distance from the mouth end should be close to the FULL span,
    # not close to zero, i.e. it must not accidentally still cover the mouth.
    assert abs(z_sliver.max() - band_hi) > 0.5 * full_span, (
        "the old-sliver run reaches too close to the real mouth -- not a meaningful negative control"
    )
    print(f"[check] NEGATIVE CONTROL: depth_max_m collapsed to the old mislabeled sliver "
          f"({OLD_SLIVER_SEAT_DEPTH_M * 1000:.3f}mm) covers only [{z_sliver.min():.6f}, {z_sliver.max():.6f}] "
          f"({sliver_span * 1000:.3f}mm span), far short of the real mouth at {band_hi:.6f} "
          f"({full_span * 1000:.3f}mm span) -- PASS (confirms this test would catch a regression back to "
          "the old wrong magnitude)")

    # === 8) NEAR-GOAL DEFAULT BAND (PROBLEM 2) -- the actual product-level fix, not just the
    # underlying mechanism verified above. Runs the DEFAULT [DEPTH_MIN_M, DEPTH_MAX_M] band, not
    # the full range. ===
    assert DEPTH_MAX_M <= full_range_depth_max_m, "near-goal band exceeds the min-engaged-length-clamped ceiling"
    out_near_goal = sample_axial_insertion_depth_reference(
        num, DEPTH_MIN_M, DEPTH_MAX_M, lateral_jitter_max_m, True, INSERTION_AXIS_LOCAL
    )

    # 8a) the band is what it claims to be.
    assert out_near_goal["depth"].min() >= DEPTH_MIN_M - 1e-9
    assert out_near_goal["depth"].max() <= DEPTH_MAX_M + 1e-9
    print(f"[check] near-goal DEFAULT band depth in [{DEPTH_MIN_M:.6f}, {DEPTH_MAX_M:.6f}] "
          f"({DEPTH_MIN_M * 1000:.2f}mm-{DEPTH_MAX_M * 1000:.2f}mm, {DEPTH_MIN_M * 1000 / (seat_depth_m * 1000):.0%}"
          f"-{DEPTH_MAX_M * 1000 / (seat_depth_m * 1000):.0%} of the full {seat_depth_m * 1000:.1f}mm span): "
          f"observed [{out_near_goal['depth'].min():.6f}, {out_near_goal['depth'].max():.6f}] -- PASS")

    # 8b) ALREADY-SOLVED-AT-SPAWN FRACTION must be 0 -- the exact failure mode that has already
    #    inflated a success rate and reversed a conclusion on this project once (per the bead
    #    report). position_error is the distance from the fully-seated GOAL pose: depth (axial) and
    #    lateral jitter (radial) combine as a Euclidean norm, so position_error >= depth always --
    #    checked with the ACTUAL combined value, not just depth alone, so jitter cannot quietly
    #    close the gap.
    position_error_m = np.linalg.norm(out_near_goal["new_target_pos"] - out_near_goal["target_pos"], axis=-1)
    already_solved = position_error_m <= POSITION_SUCCESS_THRESHOLD_M
    already_solved_fraction = float(already_solved.mean())
    assert already_solved_fraction == 0.0, (
        f"already-solved-at-spawn fraction is {already_solved_fraction:.4%}, not 0 -- the near-goal band "
        f"is partly self-solving (min position_error={position_error_m.min():.6f} vs threshold="
        f"{POSITION_SUCCESS_THRESHOLD_M:.6f})"
    )
    print(f"[check] already-solved-at-spawn fraction = {already_solved_fraction:.4%} "
          f"(min position_error={position_error_m.min() * 1000:.3f}mm vs success threshold="
          f"{POSITION_SUCCESS_THRESHOLD_M * 1000:.3f}mm) -- PASS, by construction (depth_min_m > threshold)")

    # 8c) the band's own tilt range is modest and matches the hand-derived endpoints (~1.49-2.61deg
    #    for this pair) -- neither degenerate (near 90deg, PROBLEM 1) nor collapsed to a single
    #    constant (PROBLEM the earlier fourth-pass fix addressed).
    band_tilt_min = float(out_near_goal["tilt_max_rad"].min())
    band_tilt_max = float(out_near_goal["tilt_max_rad"].max())
    assert band_tilt_max < math.radians(10.0), (
        f"near-goal band tilt bound max ({math.degrees(band_tilt_max):.2f} deg) is surprisingly large for "
        "a band this close to fully seated -- re-check the band/clearance numbers"
    )
    assert band_tilt_max > band_tilt_min, "near-goal band shows no tilt variation at all"
    print(f"[check] near-goal band tilt bound range: [{math.degrees(band_tilt_min):.3f}, "
          f"{math.degrees(band_tilt_max):.3f}] deg -- PASS")

    # 8d) DEDICATED rim-cap-vs-floor demonstration (bead UWLab-algw.9 sixth pass, PROBLEM 1b
    # CORRECTED). Reproduces the exact comparison this fix was specified against: at
    # engaged_length=min_engaged_length_m, under the HISTORICAL pilot-based clearance budget
    # (RADIAL_CLEARANCE_M -- independent of whatever smaller operational budget the shipped
    # sampler now ALSO uses, for the separate crest-interference reason in check 10a below), the
    # OLD (pilot-radius) rim cap never bound; the CORRECTED (crest-radius) rim cap does.
    historical_tilt_budget_m = RADIAL_CLEARANCE_M / 2  # lateral jitter gets the other half
    engaged_bound_at_floor_deg = math.degrees(
        math.asin(min(1.0, historical_tilt_budget_m / MIN_ENGAGED_LENGTH_M))
    )
    old_pilot_rim_cap_rad = math.acos(min(1.0, 0.010004 / MOUTH_BORE_RADIUS_M))  # the WRONG, pre-fix cap
    assert math.degrees(RIM_TILT_CAP_RAD) < engaged_bound_at_floor_deg, (
        f"expected the CORRECTED rim cap ({math.degrees(RIM_TILT_CAP_RAD):.3f}deg) to be tighter than "
        f"the floor-side engaged-length bound ({engaged_bound_at_floor_deg:.3f}deg) under the historical "
        "clearance budget -- if this fails, fix (b) no longer changes anything at the floor"
    )
    assert math.degrees(old_pilot_rim_cap_rad) > engaged_bound_at_floor_deg, (
        "expected the OLD (pre-fix, pilot-radius) rim cap to be LOOSER than the floor-side "
        "engaged-length bound -- i.e. permanently dormant, which is the bug this fix corrects"
    )
    print(
        f"[check] rim-cap-vs-floor (HISTORICAL pilot-based clearance budget, formula-correctness "
        f"demonstration): at engaged_length=min_engaged_length_m={MIN_ENGAGED_LENGTH_M * 1000:.1f}mm, "
        f"engaged-length-only bound={engaged_bound_at_floor_deg:.3f}deg. OLD (pilot-radius) rim cap="
        f"{math.degrees(old_pilot_rim_cap_rad):.3f}deg (LOOSER -- never bound). CORRECTED (crest-radius) "
        f"rim cap={math.degrees(RIM_TILT_CAP_RAD):.3f}deg (TIGHTER -- now binds) -- PASS. (The sampler's "
        "actual OPERATING clearance budget is smaller still, for the separate crest-interference reason "
        "in check 10a below, under which the rim cap stays dormant again -- see check 4c's own numbers.)"
    )

    # === 9) THREAD-YAW COUPLING mechanism (sixth pass) -- pure-numpy sanity of the
    # interpolation/sampling logic itself, no mesh access (that is check 10 below). ===
    num_yaw = 200_000
    depth_yaw_check = RNG.uniform(0.0, THREAD_YAW_TABLE_DEPTH_M[-1], size=num_yaw)
    yaw_samples = sample_yaw_from_table(
        depth_yaw_check, THREAD_YAW_TABLE_DEPTH_M, THREAD_YAW_TABLE_CENTER_RAD, THREAD_YAW_TABLE_WIDTH_RAD,
        YAW_ARC_MARGIN, RNG,
    )
    center_at = np.interp(depth_yaw_check, THREAD_YAW_TABLE_DEPTH_M, THREAD_YAW_TABLE_CENTER_RAD)
    width_at = np.interp(depth_yaw_check, THREAD_YAW_TABLE_DEPTH_M, THREAD_YAW_TABLE_WIDTH_RAD)
    full_circle_at = width_at >= (2.0 * math.pi - 1e-4)

    # 9a) narrow-arc depths: every sampled yaw lies within yaw_arc_margin of the interpolated
    #     centre -- i.e. this test's own sampler cannot silently drift outside the solved table's
    #     feasible arc.
    narrow = ~full_circle_at
    dev = np.abs(yaw_samples[narrow] - center_at[narrow])
    bound = 0.5 * YAW_ARC_MARGIN * width_at[narrow] + 1e-9
    assert np.all(dev <= bound), (
        f"a table-coupled yaw sample fell outside its margin-shrunk feasible arc: max deviation "
        f"{math.degrees(dev.max()):.4f}deg vs bound {math.degrees(bound[dev.argmax()]):.4f}deg"
    )
    print(f"[check] thread-yaw coupling: {narrow.sum()} narrow-arc samples all land within "
          f"yaw_arc_margin={YAW_ARC_MARGIN} of the interpolated centre -- PASS")

    # 9b) wide/unconstrained depths (>=~16mm): sampling must cover close to the full circle, not
    #     collapse toward a meaningless centre.
    wide = full_circle_at
    assert wide.sum() > 1000, "not enough full-circle samples to check circle coverage"
    wide_span_deg = math.degrees(float(np.ptp(np.mod(yaw_samples[wide], 2 * math.pi))))
    assert wide_span_deg > 300.0, (
        f"full-circle-eligible depths only spanned {wide_span_deg:.1f}deg of sampled yaw -- expected "
        "near-360deg coverage"
    )
    print(f"[check] thread-yaw coupling: {wide.sum()} unconstrained-depth samples span "
          f"{wide_span_deg:.1f}deg of yaw (near the full circle) -- PASS")

    # 9c) margin actually shrinks the sampled range: a margin=1.0 run's max |deviation from centre|
    #     (at the SAME depths, narrow-arc only) must exceed the margin=0.9 run's -- otherwise
    #     yaw_arc_margin is being computed but silently ignored.
    yaw_full_margin = sample_yaw_from_table(
        depth_yaw_check, THREAD_YAW_TABLE_DEPTH_M, THREAD_YAW_TABLE_CENTER_RAD, THREAD_YAW_TABLE_WIDTH_RAD,
        1.0, np.random.default_rng(0),
    )
    dev_full_margin = np.abs(yaw_full_margin[narrow] - center_at[narrow])
    assert dev_full_margin.max() > dev.max(), (
        "yaw_arc_margin=1.0 run did not sample a wider deviation from centre than "
        f"yaw_arc_margin={YAW_ARC_MARGIN} -- the margin looks inert"
    )
    print(f"[check] yaw_arc_margin={YAW_ARC_MARGIN} measurably shrinks the sampled sub-arc: max "
          f"deviation {math.degrees(dev.max()):.3f}deg vs {math.degrees(dev_full_margin.max()):.3f}deg "
          "at margin=1.0 -- PASS (margin is not decorative)")

    # === 10) GEOMETRIC checks against the REAL collision meshes (sixth pass) -- the part of this
    # bead's own instructions that a success-metric or an assertion-only check cannot stand in for.
    # Loads scripts_v2/tools/solve_thread_lead_from_meshes.py the SAME way that script loads THIS
    # file (importlib.util.spec_from_file_location by path, under a distinct sys.modules name) --
    # not circular: that script's own module-level code does not call main() or otherwise recurse,
    # it only defines the loader functions and constants reused below. ===
    print("\n[mesh] loading solve_thread_lead_from_meshes.py for geometric validation ...")
    _solver_spec = importlib.util.spec_from_file_location(
        "_solve_thread_lead_from_meshes", Path(__file__).with_name("solve_thread_lead_from_meshes.py")
    )
    _solver = importlib.util.module_from_spec(_solver_spec)
    sys.modules[_solver_spec.name] = _solver
    _solver_spec.loader.exec_module(_solver)

    bore_pts, bore_tris = _solver.load_bore_mesh()
    leg_thread_root = _solver.load_leg_thread_obj_in_root_frame()
    r_local = np.sqrt(leg_thread_root[:, 1] ** 2 + leg_thread_root[:, 2] ** 2)
    crest_local_pts_full = leg_thread_root[r_local > _solver.CREST_LOCAL_RADIUS_MIN_M]
    bore_z = bore_pts[:, 2]
    BORE_AXIS_XY = RECEPTIVE_OFFSET_POS[:2].copy()
    wall_map = _solver.BoreWallRadiusMap(bore_pts, bore_tris, BORE_AXIS_XY, float(bore_z.min()), float(bore_z.max()))
    # Stride-subsampled crest for the many-sample band report below (speed); the negative control
    # further down re-checks with the FULL crest for maximum fidelity on the few poses it evaluates.
    stride = max(1, crest_local_pts_full.shape[0] // 700)
    crest_local_pts_fast = crest_local_pts_full[::stride]
    print(f"[mesh] bore: {len(bore_pts)}pts/{len(bore_tris)}tris; leg crest (local radius > "
          f"{_solver.CREST_LOCAL_RADIUS_MIN_M * 1000:.1f}mm): {crest_local_pts_full.shape[0]} full / "
          f"{crest_local_pts_fast.shape[0]} stride-subsampled points")

    def geometric_clearance(crest_pts: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> tuple[float, int]:
        """Radial clearance of ONE fully-composed leg pose (position + orientation, INCLUDING
        lateral jitter and tilt, not just depth/yaw) against the real bore wall -- generalises
        solve_thread_lead_from_meshes.py's ``clearance_at`` (which only ever varies depth/yaw) to
        the full 6-DoF sampled pose this term actually writes."""
        world_pts = pos[None, :] + quat_apply(quat[None, :], crest_pts)
        z = world_pts[:, 2]
        mask = (z >= wall_map.z_min) & (z <= wall_map.z_max)
        n_engaged = int(mask.sum())
        if n_engaged == 0:
            return float("inf"), 0
        pts = world_pts[mask]
        dx, dy = pts[:, 0] - BORE_AXIS_XY[0], pts[:, 1] - BORE_AXIS_XY[1]
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)
        r_wall = wall_map.query(theta, pts[:, 2])
        return float((r_wall - r).min()), n_engaged

    # --- 10a) BAND REPORT (this bead's requested deliverable): for depth sampled over [1, 20]mm of
    # backoff with the actual fixed sampler (tilt + thread-coupled yaw + lateral jitter all
    # enabled), report the resulting distribution of lateral miss, tilt, and MINIMUM CLEARANCE --
    # the last one computed geometrically against the real meshes, not asserted. ---
    NUM_REPORT = 300
    out_band = sample_axial_insertion_depth_reference(
        NUM_REPORT, 0.001, 0.020, lateral_jitter_max_m, True, INSERTION_AXIS_LOCAL, enable_yaw_coupling=True,
    )
    clearances_mm, engaged_counts = [], []
    for i in range(NUM_REPORT):
        c, n = geometric_clearance(crest_local_pts_fast, out_band["insertive_pos"][i], out_band["insertive_quat"][i])
        clearances_mm.append(c * 1000.0)
        engaged_counts.append(n)
    clearances_mm = np.array(clearances_mm)
    lateral_miss_mm = np.sqrt(out_band["jitter_x"] ** 2 + out_band["jitter_y"] ** 2) * 1000.0
    tilt_deg = np.degrees(np.sqrt(out_band["roll"] ** 2 + out_band["pitch"] ** 2))
    finite_clearances_mm = clearances_mm[np.isfinite(clearances_mm)]
    print(
        f"\n[REPORT] band = depth in [1, 20]mm of backoff, N={NUM_REPORT} samples, tilt+thread-coupled-yaw+"
        f"lateral-jitter all enabled:\n"
        f"  lateral miss (mm):  median={np.median(lateral_miss_mm):.4f}  max={lateral_miss_mm.max():.4f}\n"
        f"  tilt (deg):         median={np.median(tilt_deg):.4f}  max={tilt_deg.max():.4f}\n"
        f"  min clearance (mm): median={np.median(finite_clearances_mm):.4f}  "
        f"min={finite_clearances_mm.min():.4f}  max={finite_clearances_mm.max():.4f}  "
        f"({(~np.isfinite(clearances_mm)).sum()}/{NUM_REPORT} samples had no crest material in the "
        "bore's z-span at all -- disengaged, not interfering by construction)"
    )
    assert finite_clearances_mm.min() > 0.0, (
        f"GEOMETRIC clearance went negative for at least one sampled pose in the band report "
        f"(min={finite_clearances_mm.min():.4f}mm) -- the thread-yaw coupling is not actually "
        "keeping the meshes clear; re-check yaw_arc_margin / the table itself"
    )
    print(f"[check] every one of the {len(finite_clearances_mm)} engaged band-report samples clears the "
          "real bore mesh (min clearance > 0mm), checked geometrically -- PASS")

    # --- 10b) NEGATIVE CONTROL (the actual point of this pass): yaw left at 0 for a nonzero depth
    # -- reproducing the bug this whole pass exists to prevent -- must show REAL, substantial mesh
    # interpenetration, AND the task's own success predicate
    # (omnireset/mdp/terminations.py:483-490, ``e_x, e_y, _ = euler_xyz_from_quat(rel_quat)``,
    # ``euler_xy_dist = |e_x| + |e_y|`` -- e_z, the roll/yaw error about the insertion axis, is
    # never even read) must be UNABLE to tell the two poses apart. No lateral jitter or tilt here,
    # to isolate the yaw effect cleanly.
    #
    # e_x/e_y/e_z below are read directly off out["roll"]/out["pitch"]/out["yaw"] rather than
    # re-extracted from new_target_quat via a fresh euler-from-quat implementation: in THIS
    # synthetic setup receptive_quat and RECEPTIVE_OFFSET_QUAT are both identity, so target_quat is
    # identity and new_target_quat = quat_mul(target_quat, offset_quat) = offset_quat EXACTLY --
    # i.e. new_target_quat is quat_from_euler_xyz(roll, pitch, yaw) by construction, so
    # (roll, pitch, yaw) already ARE (e_x, e_y, e_z) here, with no inversion needed or extra
    # unverified machinery introduced just for this check. ---
    # Probe depths are chosen PROGRAMMATICALLY from the table, not hand-picked: yaw=0 is not
    # uniformly unsafe at every depth -- because the arc CENTRE advances by ~38.4deg per mm, it
    # periodically wraps back near 0 (e.g. depth~=8-9mm, ~=17-18mm, ...), where yaw=0 happens to
    # already sit inside the feasible arc by coincidence of phase, not because it is safe in
    # general. Pick depths where 0 sits comfortably OUTSIDE the feasible arc (>=20deg of margin),
    # so the control is testing the real bug, not an accidental phase alignment.
    def _out_of_arc_margin_rad(depth_m: float) -> float:
        c = float(np.interp(depth_m, THREAD_YAW_TABLE_DEPTH_M, THREAD_YAW_TABLE_CENTER_RAD))
        w = float(np.interp(depth_m, THREAD_YAW_TABLE_DEPTH_M, THREAD_YAW_TABLE_WIDTH_RAD))
        circ_dist_to_zero = abs((0.0 - c + math.pi) % (2.0 * math.pi) - math.pi)
        return circ_dist_to_zero - 0.5 * w  # positive => 0 lies outside the feasible arc by this much

    _candidate_depths = THREAD_YAW_TABLE_DEPTH_M[THREAD_YAW_TABLE_DEPTH_M > 0.0]
    _margins = [(float(d), _out_of_arc_margin_rad(float(d))) for d in _candidate_depths]
    CONTROL_DEPTHS_M = sorted(d for d, m in _margins if m > math.radians(20.0))[:4]
    assert len(CONTROL_DEPTHS_M) >= 3, (
        f"expected at least 3 table depths with yaw=0 comfortably outside the feasible arc; got "
        f"{len(CONTROL_DEPTHS_M)} -- margins were {[(d*1000, math.degrees(m)) for d, m in _margins]}"
    )
    print(f"\n[NEGATIVE CONTROL] yaw forced to 0 vs thread-coupled, at depths {[d*1000 for d in CONTROL_DEPTHS_M]}mm "
          "(auto-selected: yaw=0 lies >=20deg outside the feasible arc at each):")
    worst_zero_yaw_clearance_mm = float("inf")
    for depth_probe in CONTROL_DEPTHS_M:
        out_zero_yaw = sample_axial_insertion_depth_reference(
            1, depth_probe, depth_probe, 0.0, False, INSERTION_AXIS_LOCAL, force_yaw_zero=True,
        )
        out_coupled = sample_axial_insertion_depth_reference(
            1, depth_probe, depth_probe, 0.0, False, INSERTION_AXIS_LOCAL, enable_yaw_coupling=True,
        )
        c_zero, n_zero = geometric_clearance(crest_local_pts_full, out_zero_yaw["insertive_pos"][0], out_zero_yaw["insertive_quat"][0])
        c_coup, n_coup = geometric_clearance(crest_local_pts_full, out_coupled["insertive_pos"][0], out_coupled["insertive_quat"][0])
        worst_zero_yaw_clearance_mm = min(worst_zero_yaw_clearance_mm, c_zero * 1000.0)

        # The task's own position/orientation success signals, computed identically for both poses.
        for label, out, c, n in (
            ("yaw=0 (BUG)", out_zero_yaw, c_zero, n_zero),
            ("thread-coupled (FIXED)", out_coupled, c_coup, n_coup),
        ):
            rel_pos = out["new_target_pos"][0] - out["target_pos"][0]
            xyz_dist_mm = float(np.linalg.norm(rel_pos)) * 1000.0
            e_x, e_y, e_z = float(out["roll"][0]), float(out["pitch"][0]), float(out["yaw"][0])
            euler_xy_dist_deg = math.degrees(abs(e_x) + abs(e_y))
            print(
                f"  depth={depth_probe * 1000:5.1f}mm  {label:23s}  geometric_clearance={c * 1000:+9.4f}mm  "
                f"n_engaged={n:5d}  xyz_dist={xyz_dist_mm:7.4f}mm  euler_xy_dist={euler_xy_dist_deg:7.4f}deg  "
                f"(e_z/yaw={math.degrees(e_z):+8.3f}deg, NEVER READ by the success predicate)"
            )
        assert c_zero < -0.3e-3, (
            f"expected yaw=0 at depth={depth_probe*1000:.1f}mm to interfere by at least 0.3mm -- got "
            f"{c_zero * 1000:.4f}mm; the negative control did not fire"
        )
        assert c_coup > 0.0, (
            f"expected the thread-coupled yaw at depth={depth_probe*1000:.1f}mm to clear the mesh -- got "
            f"{c_coup * 1000:.4f}mm"
        )
        # The two poses differ ONLY in yaw (same depth, zero jitter/tilt in both) -- confirm the
        # task's own metrics are numerically blind to that difference, i.e. genuinely could not
        # have caught this bug.
        rel_pos_zero = out_zero_yaw["new_target_pos"][0] - out_zero_yaw["target_pos"][0]
        rel_pos_coup = out_coupled["new_target_pos"][0] - out_coupled["target_pos"][0]
        assert np.allclose(rel_pos_zero, rel_pos_coup, atol=1e-9), (
            "expected identical position error between the yaw=0 and thread-coupled poses (yaw does "
            "not move the tip) -- if this fails, the two are not actually isolating yaw alone"
        )
        assert float(out_zero_yaw["yaw"][0]) == 0.0, "force_yaw_zero did not actually zero yaw"
        assert float(out_coupled["yaw"][0]) != 0.0, (
            "thread-coupled yaw sampled exactly 0 -- vanishingly unlikely; re-check the table/sampler"
        )
        euler_xy_dist_zero = abs(float(out_zero_yaw["roll"][0])) + abs(float(out_zero_yaw["pitch"][0]))
        euler_xy_dist_coup = abs(float(out_coupled["roll"][0])) + abs(float(out_coupled["pitch"][0]))
        assert euler_xy_dist_zero == 0.0 and euler_xy_dist_coup == 0.0, (
            "expected euler_xy_dist == 0 for both poses (tilt disabled here) -- if nonzero, this "
            "probe is not actually isolating yaw"
        )

    assert worst_zero_yaw_clearance_mm < -0.3, (
        f"expected at least one yaw=0 depth to interfere by > 0.3mm; worst observed "
        f"{worst_zero_yaw_clearance_mm:.4f}mm"
    )
    print(
        f"\n[check] NEGATIVE CONTROL FIRED: yaw=0 at nonzero depth interferes with the real bore mesh by "
        f"up to {abs(worst_zero_yaw_clearance_mm):.4f}mm of radial solid-body overlap at every probed depth, "
        "while the thread-coupled yaw clears it at the same depths -- and xyz_dist / euler_xy_dist "
        "(the task's own position + orientation success signals) are IDENTICAL between the two poses "
        "at each depth, because both are computed with the same combine_frame_transforms chain that "
        "makes new_target_pos independent of yaw and euler_xy_dist never reads e_z. PASS: this test "
        "would have caught the exact bug this pass exists to prevent, and the task's own success "
        "metric provably would not have."
    )

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
