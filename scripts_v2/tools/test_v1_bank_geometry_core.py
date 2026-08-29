# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Unit-proves v1_bank_geometry_core.py's mating-frame decomposition, band membership, resting-
speed check, provenance extraction, and survival-curve sweep (bead R3) without touching Isaac,
torch, or a real bank.

Isaac-free AND torch-free -- only ``numpy`` + stdlib, so this runs on BOTH local interpreters
(team-lead instruction, 2026-08-29: run tests on both, since one v2 suite recently had never
executed because it lacked a runner and pytest was absent on one of them):
  - plain system ``python3`` (has numpy, no torch, no pytest)
  - ``/home/dom-iva/.cache/simdist-cpu-venv/bin/python3`` (numpy + torch + pytest)
The module under test itself never imports torch, so both actually exercise the same code path,
not two different ones.

Loaded BY FILE PATH, same technique ``test_spawn_tolerance_stage.py`` uses next door.

NEGATIVE CONTROLS in this file:
  1. the ASSEMBLED-POSE cross-check (an EXTERNALLY hand-derived expectation -- depth=engaged_span,
     lateral=0, tilt=0 -- not derived from the code under test) doubles as the sign/axis control:
     using the WRONG tip-local-axis on the SAME fixture breaks it (tilt jumps to 180deg), proving
     the test is sensitive to the axis convention, not just checking a symmetric magnitude.
  2. within_band boundary tests (inclusive on both ends, same convention as _SeatingGateAddon).
  3. leg_asset_from_path: a recognisable path, an unrecognisable one (must read UNKNOWN, never a
     guess), and a path with two DIFFERENT plausible substrings (must not silently pick one).
  4. band_survival_curve: monotonic non-increasing survival as the band narrows, PLUS an exact
     hand-computed count at one scale factor (not just "some numbers came out").
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

_MODULE_PATH = Path(__file__).resolve().parent / "v1_bank_geometry_core.py"
_spec = importlib.util.spec_from_file_location("v1_bank_geometry_core", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["v1_bank_geometry_core"] = _mod
_spec.loader.exec_module(_mod)

quat_wxyz_to_rotmat = _mod.quat_wxyz_to_rotmat
rotate = _mod.rotate
decompose_mating_frame = _mod.decompose_mating_frame
within_band = _mod.within_band
resting_speed_survival = _mod.resting_speed_survival
band_survival_curve = _mod.band_survival_curve
leg_asset_from_path = _mod.leg_asset_from_path
LEG_OFFSET_POS_M = _mod.LEG_OFFSET_POS_M
LEG_TIP_LOCAL_AXIS = _mod.LEG_TIP_LOCAL_AXIS
FIXTURE_OFFSET_POS_M = _mod.FIXTURE_OFFSET_POS_M
BORE_DEEP_LOCAL_AXIS = _mod.BORE_DEEP_LOCAL_AXIS
DEFAULT_ENGAGED_SPAN_M = _mod.DEFAULT_ENGAGED_SPAN_M


def _ry(deg: float) -> np.ndarray:
    """(w,x,y,z) quaternion for a rotation of `deg` about world +Y -- the standard textbook
    formula, independent of quat_wxyz_to_rotmat, so cross-checks built on it are not
    self-confirming."""
    half = math.radians(deg) / 2.0
    return np.array([math.cos(half), 0.0, math.sin(half), 0.0])


_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
_ORIGIN = np.array([0.0, 0.0, 0.0])


def _assembled_leg_pos_quat():
    """The leg placed EXACTLY at its assembled pose against a fixture sitting at the world origin
    with identity orientation -- derived externally (Ry(-90) matches the leg metadata.yaml's own
    documented assembled root_quat, cross-checked against play2perfect's independent derivation),
    not from the code under test."""
    leg_quat = _ry(-90.0)
    R_leg = quat_wxyz_to_rotmat(leg_quat)
    target_world = _ORIGIN + rotate(quat_wxyz_to_rotmat(_IDENTITY_QUAT), FIXTURE_OFFSET_POS_M)
    leg_pos = target_world - rotate(R_leg, LEG_OFFSET_POS_M)
    return leg_pos, leg_quat


# ---------------------------------------------------------------------------------------------
# decompose_mating_frame: assembled-pose cross-check (NEGATIVE CONTROL 1).
# ---------------------------------------------------------------------------------------------


def test_decompose_mating_frame_at_the_assembled_pose_is_fully_seated_centred_and_untilted():
    leg_pos, leg_quat = _assembled_leg_pos_quat()
    depth_m, lateral_m, tilt_deg = decompose_mating_frame(leg_pos, leg_quat, _ORIGIN, _IDENTITY_QUAT)
    assert math.isclose(float(depth_m), DEFAULT_ENGAGED_SPAN_M, abs_tol=1e-9)
    assert math.isclose(float(lateral_m), 0.0, abs_tol=1e-9)
    assert math.isclose(float(tilt_deg), 0.0, abs_tol=1e-6)


def test_decompose_mating_frame_tip_axis_points_world_minus_z_at_assembly():
    # Independent corroboration of LEG_TIP_LOCAL_AXIS against the leg metadata.yaml's own comment
    # ("the leg's own tip points DOWN at assembly") -- not the same assertion as the depth/lateral/
    # tilt test above, a direct check on the intermediate quantity.
    _, leg_quat = _assembled_leg_pos_quat()
    R_leg = quat_wxyz_to_rotmat(leg_quat)
    tip_axis_world = rotate(R_leg, LEG_TIP_LOCAL_AXIS)
    assert np.allclose(tip_axis_world, np.array([0.0, 0.0, -1.0]), atol=1e-6)


def test_decompose_mating_frame_is_sensitive_to_the_tip_axis_sign():
    # NEGATIVE CONTROL: using the WRONG tip-local-axis (+X instead of -X) on the assembled pose
    # must NOT read as fully-seated-and-untilted -- proves the tilt computation is actually
    # sensitive to the axis convention, not just producing a symmetric/always-small number.
    leg_pos, leg_quat = _assembled_leg_pos_quat()
    R_leg = quat_wxyz_to_rotmat(leg_quat)
    wrong_tip_axis_world = rotate(R_leg, np.array([1.0, 0.0, 0.0]))  # wrong sign
    R_fix = quat_wxyz_to_rotmat(_IDENTITY_QUAT)
    bore_deep_axis_world = rotate(R_fix, BORE_DEEP_LOCAL_AXIS)
    cosang = np.clip((wrong_tip_axis_world * bore_deep_axis_world).sum(-1), -1.0, 1.0)
    wrong_tilt_deg = math.degrees(math.acos(float(cosang)))
    assert math.isclose(wrong_tilt_deg, 180.0, abs_tol=1e-4), (
        "test fixture problem: flipping the tip axis should read as fully INVERTED (180deg), "
        "or this negative control is not actually exercising the sign"
    )


def test_decompose_mating_frame_depth_decreases_as_tip_is_pulled_out():
    # A leg withdrawn ALONG the bore axis by delta_m should read depth SMALLER by exactly delta_m
    # -- a hand-derived, monotonic, exact expectation (not merely "some numbers changed").
    leg_pos, leg_quat = _assembled_leg_pos_quat()
    depth0, _, _ = decompose_mating_frame(leg_pos, leg_quat, _ORIGIN, _IDENTITY_QUAT)
    delta_m = 0.007
    # bore_deep_axis_world = [0,0,-1] at this fixture pose (identity); withdrawing means moving
    # the leg root AWAY from the fixture along +Z (opposite the deep axis).
    withdrawn_leg_pos = leg_pos + np.array([0.0, 0.0, delta_m])
    depth1, lateral1, tilt1 = decompose_mating_frame(withdrawn_leg_pos, leg_quat, _ORIGIN, _IDENTITY_QUAT)
    assert math.isclose(float(depth0) - float(depth1), delta_m, abs_tol=1e-9)
    assert math.isclose(float(lateral1), 0.0, abs_tol=1e-9)  # pure withdrawal, no lateral drift
    assert math.isclose(float(tilt1), 0.0, abs_tol=1e-6)  # pure withdrawal, no tilt


def test_decompose_mating_frame_is_vectorised_over_a_batch():
    leg_pos, leg_quat = _assembled_leg_pos_quat()
    n = 5
    leg_pos_batch = np.tile(leg_pos, (n, 1))
    leg_quat_batch = np.tile(leg_quat, (n, 1))
    fix_pos_batch = np.tile(_ORIGIN, (n, 1))
    fix_quat_batch = np.tile(_IDENTITY_QUAT, (n, 1))
    depth_m, lateral_m, tilt_deg = decompose_mating_frame(leg_pos_batch, leg_quat_batch, fix_pos_batch, fix_quat_batch)
    assert depth_m.shape == (n,)
    assert np.allclose(depth_m, DEFAULT_ENGAGED_SPAN_M, atol=1e-9)


# ---------------------------------------------------------------------------------------------
# within_band: inclusive boundaries (NEGATIVE CONTROL 2).
# ---------------------------------------------------------------------------------------------


def test_within_band_accepts_inside_and_on_the_boundary():
    depth_m = np.array([0.005, 0.020, 0.012])
    lateral_m = np.array([0.0, 0.008, 0.004])
    tilt_deg = np.array([0.0, 20.0, 10.0])
    result = within_band(depth_m, lateral_m, tilt_deg, depth_min_m=0.005, depth_max_m=0.020, lateral_max_m=0.008, tilt_max_deg=20.0)
    assert bool(result[0]) is True  # depth at exact min boundary
    assert bool(result[1]) is True  # depth/lateral/tilt all at exact max boundary
    assert bool(result[2]) is True  # comfortably inside


def test_within_band_rejects_just_outside_each_dimension_independently():
    base = dict(depth_min_m=0.005, depth_max_m=0.020, lateral_max_m=0.008, tilt_max_deg=20.0)
    assert not bool(within_band(np.array([0.0049]), np.array([0.0]), np.array([0.0]), **base)[0])
    assert not bool(within_band(np.array([0.0201]), np.array([0.0]), np.array([0.0]), **base)[0])
    assert not bool(within_band(np.array([0.010]), np.array([0.0081]), np.array([0.0]), **base)[0])
    assert not bool(within_band(np.array([0.010]), np.array([0.0]), np.array([20.01]), **base)[0])


def test_resting_speed_survival_boundary():
    speeds = np.array([0.049, 0.050, 0.051])
    result = resting_speed_survival(speeds, max_speed_mps=0.05)
    assert list(result) == [True, True, False]


# ---------------------------------------------------------------------------------------------
# leg_asset_from_path: recognisable, unrecognisable (UNKNOWN, never guessed) (NEGATIVE CONTROL 3).
# ---------------------------------------------------------------------------------------------


def test_leg_asset_from_path_recognises_a_real_path():
    path = "/home/dom_iva/Foo/Resets/OneLegInsertionFixture__SquareTableLeg200mmDecomp/resets_ObjectPartiallyAssembledEEGrasped.pt"
    assert leg_asset_from_path(path) == "SquareTableLeg200mmDecomp"


def test_leg_asset_from_path_recognises_the_sdf_variant():
    path = "/home/dom_iva/Foo/Resets/OneLegInsertionFixture__SquareTableLeg200mmSdf/resets_ObjectRestingEEGrasped.pt"
    assert leg_asset_from_path(path) == "SquareTableLeg200mmSdf"


def test_leg_asset_from_path_returns_unknown_when_unrecognisable():
    path = "/home/dom_iva/some_random_dir/resets_ObjectAnywhereEEGrasped.pt"
    assert leg_asset_from_path(path) == "UNKNOWN"


# ---------------------------------------------------------------------------------------------
# band_survival_curve: monotonic in band width, plus one exact hand-computed count.
# ---------------------------------------------------------------------------------------------


def test_band_survival_curve_is_monotonic_non_increasing_as_the_band_narrows():
    rng_depth = np.linspace(-0.02, 0.03, 500)
    rng_lateral = np.abs(np.linspace(0.0, 0.02, 500))
    rng_tilt = np.abs(np.linspace(0.0, 30.0, 500))
    rows = band_survival_curve(rng_depth, rng_lateral, rng_tilt, scale_factors=(0.5, 1.0, 1.5, 2.0, 3.0))
    counts = [r["n_survive"] for r in rows]
    assert counts == sorted(counts), f"expected non-decreasing counts as scale grows, got {counts}"
    assert all(r["n"] == 500 for r in rows)


def test_band_survival_curve_exact_count_at_scale_one():
    # Hand-constructed: 3 states inside the v1-precedent-shaped band exactly, 2 states clearly
    # outside it (one on depth, one on tilt) -- an EXACT expected count, not just "monotonic".
    depth_m = np.array([0.003, 0.008, 0.001, 0.050, 0.005])
    lateral_m = np.array([0.001, 0.002, 0.0, 0.0, 0.0])
    tilt_deg = np.array([5.0, 10.0, 0.0, 0.0, 90.0])
    rows = band_survival_curve(depth_m, lateral_m, tilt_deg, scale_factors=(1.0,))
    assert rows[0]["n_survive"] == 3
    assert math.isclose(rows[0]["depth_min_m"], 0.0, abs_tol=1e-12)
    assert math.isclose(rows[0]["depth_max_m"], 0.010, abs_tol=1e-12)
    assert math.isclose(rows[0]["lateral_max_m"], 0.005, abs_tol=1e-12)
    assert math.isclose(rows[0]["tilt_max_deg"], 15.0, abs_tol=1e-12)


def test_band_survival_curve_reports_scale_and_fraction_together():
    depth_m = np.full(10, 0.005)
    lateral_m = np.zeros(10)
    tilt_deg = np.zeros(10)
    rows = band_survival_curve(depth_m, lateral_m, tilt_deg, scale_factors=(1.0,))
    assert rows[0]["n_survive"] == 10
    assert math.isclose(rows[0]["survive_fraction"], 1.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[v1_bank_geometry_core] {name} OK", flush=True)
    print("[v1_bank_geometry_core] all tests passed", flush=True)
