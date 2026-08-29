# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit-proves the episode mixture's TRANSPORT branch core math (RESET_SPEC_V2.md sec 1 C3,
V2_POSE_FINDINGS.md F43, bead dr-ai1.13) without touching Isaac at all.

Needs only plain ``python3`` -- no Isaac Sim, no GPU, no env construction. Same reason and same
technique as ``test_c1_hand_pose_stage.py`` next to this file: ``c3_transport_core.py`` has no
``isaaclab`` import by design, so it is loaded here BY FILE PATH.

This test covers ONLY the frame conversion, the orientation-range construction, the 4-way mixture
validation, and the banner text. It cannot prove the Isaac-touching half (``episode_mixture.py``'s
new TRANSPORT kind, and the config wiring in ``dexlift_ur5e_delto_tableleg_env_cfg.py``) actually
draws from these ranges at runtime -- that needs Isaac Sim and is out of scope here.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

_CORE_PATH = (
    Path(__file__).resolve().parents[1]
    / "uwlab_tasks/manager_based/manipulation/dexlift/mdp/c3_transport_core.py"
)


def _load_by_path(name: str, path: Path):
    """Load an Isaac-free module by FILE PATH, COMPILING THE SOURCE TEXT (bead dr-76w.22).

    NOT ``spec_from_file_location(...).loader.exec_module(...)``, which this file used to use:
    that consults and writes ``__pycache__``, and CPython's staleness check compares the source
    mtime at ONE-SECOND granularity against the ``.pyc`` header. An edit / run / restore cycle
    completed inside one second therefore leaves a ``.pyc`` that looks valid for the restored
    source, and the next run silently executes the MUTATED bytecode -- reporting failures for code
    that is correct on disk, or worse, passes for code that is not.

    That is not hypothetical: it happened while writing ``test_c3_rung_stage.py``'s negative
    controls, where it produced four phantom failures against correct restored source. Reproduced
    deterministically side by side (same mtime, same size): the old loader returned the mutated
    value while the file on disk held the original; this one returned the original.

    Compiling the text every time costs microseconds, never reads a ``.pyc`` and never writes one.
    """
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)  # noqa: S102
    return module


_c3_transport_core = _load_by_path("c3_transport_core", _CORE_PATH)

ROOT_ABOVE_TIP_M = _c3_transport_core.ROOT_ABOVE_TIP_M
tip_z_from_root_z = _c3_transport_core.tip_z_from_root_z
root_z_from_tip_z = _c3_transport_core.root_z_from_tip_z
transport_goal_ranges = _c3_transport_core.transport_goal_ranges
validate_transport_goal_z = _c3_transport_core.validate_transport_goal_z
validate_episode_mixture_fractions = _c3_transport_core.validate_episode_mixture_fractions
transport_goal_banner = _c3_transport_core.transport_goal_banner


def _raises(exc_type, fn, *args, **kwargs):
    """``pytest.raises`` without pytest (bead dr-76w.22).

    This file used to ``import pytest`` inside five cases. The local plain ``python3`` has no
    pytest, so the ``__main__`` runner below ABORTED at the first such case -- executing 2 of 20
    tests and exiting non-zero on a ModuleNotFoundError, which reads as an environment problem
    rather than as 18 unrun tests. With this helper the suite runs end-to-end under bare
    ``python3`` AND still collects normally under pytest, so both interpreters execute all 20.

    Returns the exception so a caller can assert on its message.
    """
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"expected {exc_type.__name__} from {getattr(fn, '__name__', fn)}(...)")


def test_root_above_tip_offset_matches_f43_measurement():
    # F43: "on branch 2, goal_root_z - goal_tip_z = 0.1062 m exactly."
    assert math.isclose(ROOT_ABOVE_TIP_M, 0.106203, abs_tol=1e-6)


def test_tip_z_from_root_z_at_exact_tip_down_subtracts_the_full_offset():
    # F49: root_z - tip_z = ROOT_ABOVE_TIP_M * cos(tilt); at tilt=0 (exact tip-down) cos(tilt)=1,
    # so the full offset applies -- this is the ONLY tilt at which the old scalar-only form was
    # exact (V2_POSE_FINDINGS.md F49, team-lead review).
    assert math.isclose(tip_z_from_root_z(0.13, tilt_rad=0.0), 0.13 - 0.106203, abs_tol=1e-9)
    assert math.isclose(tip_z_from_root_z(0.27, tilt_rad=0.0), 0.27 - 0.106203, abs_tol=1e-9)


def test_tip_z_from_root_z_requires_an_explicit_tilt():
    # No default: a caller must state the pose it is converting for, so this cannot be silently
    # reused in a near-horizontal context the way the pre-F49 scalar-only version was (team-lead
    # review: "the next reader cannot lift it into a near-horizontal context").
    _raises(TypeError, tip_z_from_root_z, 0.13)


def test_tip_z_from_root_z_scales_the_offset_by_cos_tilt():
    # F49's sharpened rule, reproduced from the team-lead's own measured table: root_z - tip_z =
    # 0.106203 * cos(tilt). At tilt=pi/2 (fully horizontal) the offset vanishes to 0.
    for tilt in (0.35, math.pi / 4, math.pi / 2):
        root_z = 0.20
        expected_tip_z = root_z - 0.106203 * math.cos(tilt)
        assert math.isclose(tip_z_from_root_z(root_z, tilt_rad=tilt), expected_tip_z, abs_tol=1e-9)


def test_tip_z_from_root_z_rejects_tilt_outside_zero_to_pi():
    for bad in (-0.01, math.pi + 0.01):
        _raises(ValueError, tip_z_from_root_z, 0.13, tilt_rad=bad)


def test_root_z_from_tip_z_is_the_inverse_at_a_range_of_tilts():
    for tip_z in (0.0, 0.024, 0.164):
        for tilt in (0.0, 0.35, math.pi / 2):
            root_z = root_z_from_tip_z(tip_z, tilt_rad=tilt)
            assert math.isclose(tip_z_from_root_z(root_z, tilt_rad=tilt), tip_z, abs_tol=1e-9)


def test_root_z_from_tip_z_requires_an_explicit_tilt():
    _raises(TypeError, root_z_from_tip_z, 0.024)


def test_transport_goal_ranges_centres_pitch_on_tip_down():
    ranges = transport_goal_ranges(0.35)
    assert math.isclose(ranges.pitch[0], -math.pi / 2 - 0.35, abs_tol=1e-9)
    assert math.isclose(ranges.pitch[1], -math.pi / 2 + 0.35, abs_tol=1e-9)


def test_transport_goal_ranges_roll_is_symmetric_about_zero():
    ranges = transport_goal_ranges(0.35)
    assert ranges.roll == (-0.35, 0.35)


def test_transport_goal_ranges_yaw_is_pinned_to_zero():
    ranges = transport_goal_ranges(0.35)
    assert ranges.yaw == (0.0, 0.0)


def test_transport_goal_ranges_rejects_tilt_outside_zero_to_half_pi():
    for bad in (-0.01, math.pi / 2 + 0.01, math.pi):
        _raises(ValueError, transport_goal_ranges, bad)


def test_transport_goal_ranges_accepts_tilt_at_the_boundaries():
    transport_goal_ranges(0.0)
    transport_goal_ranges(math.pi / 2)


def test_validate_transport_goal_z_accepts_lo_less_than_hi():
    validate_transport_goal_z(0.13, 0.27)


def test_validate_transport_goal_z_rejects_lo_ge_hi():
    _raises(ValueError, validate_transport_goal_z, 0.27, 0.13)
    _raises(ValueError, validate_transport_goal_z, 0.20, 0.20)


def test_validate_episode_mixture_fractions_accepts_the_shipped_default():
    # Default transport_goal_prob is 0.0, so the existing 0.50/0.25/0.25 default must still pass.
    validate_episode_mixture_fractions(0.50, 0.25, 0.25, 0.0)


def test_validate_episode_mixture_fractions_accepts_a_real_transport_share():
    validate_episode_mixture_fractions(0.40, 0.20, 0.20, 0.20)


def test_validate_episode_mixture_fractions_rejects_a_sum_off_by_the_transport_share():
    # A CLI override that adds a transport share without shrinking the other three.
    try:
        validate_episode_mixture_fractions(0.50, 0.25, 0.25, 0.20)
        raise AssertionError("expected AssertionError for a sum of 1.20")
    except AssertionError as exc:
        assert "sum=1.2" in str(exc)


def test_validate_episode_mixture_fractions_still_rejects_classic_prob_zero():
    try:
        validate_episode_mixture_fractions(0.0, 0.4, 0.3, 0.3)
        raise AssertionError("expected AssertionError for classic_goal_prob=0")
    except AssertionError as exc:
        assert "classic_goal_prob must be > 0" in str(exc)


def test_transport_goal_banner_names_the_probability_tilt_and_tip_band():
    text = transport_goal_banner(0.20, 0.35, 0.13, 0.27)
    assert "0.200" in text
    assert f"{math.degrees(0.35):.1f} deg" in text
    assert "0.130" in text and "0.270" in text
    # tip band reported AT tilt=0 (nominal tip-down), the only tilt where the pre-F49 scalar-only
    # form is exact: 0.13 - 0.106203 = 0.023797, 0.27 - 0.106203 = 0.163797. Away from tilt=0 the
    # true tip z is HIGHER (root_z - 0.106203*cos(tilt)), never lower, so this is a floor, not the
    # whole band -- the banner must say so (F49, team-lead review).
    assert "0.024" in text
    assert "0.164" in text
    assert "at tilt=0, the floor" in text
    assert "ANCHORED to the object's own" in text


def test_transport_goal_banner_names_pitch_range_in_radians():
    text = transport_goal_banner(0.20, 0.35, 0.13, 0.27)
    pitch_lo = -math.pi / 2 - 0.35
    pitch_hi = -math.pi / 2 + 0.35
    assert f"{pitch_lo:.4f}" in text
    assert f"{pitch_hi:.4f}" in text


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[c3_transport] {name} OK", flush=True)
    print("[c3_transport] all tests passed", flush=True)
