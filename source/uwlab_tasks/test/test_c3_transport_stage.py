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
_spec = importlib.util.spec_from_file_location("c3_transport_core", _CORE_PATH)
_c3_transport_core = importlib.util.module_from_spec(_spec)
sys.modules["c3_transport_core"] = _c3_transport_core
_spec.loader.exec_module(_c3_transport_core)

ROOT_ABOVE_TIP_M = _c3_transport_core.ROOT_ABOVE_TIP_M
tip_z_from_root_z = _c3_transport_core.tip_z_from_root_z
root_z_from_tip_z = _c3_transport_core.root_z_from_tip_z
transport_goal_ranges = _c3_transport_core.transport_goal_ranges
validate_transport_goal_z = _c3_transport_core.validate_transport_goal_z
validate_episode_mixture_fractions = _c3_transport_core.validate_episode_mixture_fractions
transport_goal_banner = _c3_transport_core.transport_goal_banner


def test_root_above_tip_offset_matches_f43_measurement():
    # F43: "on branch 2, goal_root_z - goal_tip_z = 0.1062 m exactly."
    assert math.isclose(ROOT_ABOVE_TIP_M, 0.106203, abs_tol=1e-6)


def test_tip_z_from_root_z_subtracts_the_offset():
    assert math.isclose(tip_z_from_root_z(0.13), 0.13 - 0.106203, abs_tol=1e-9)
    assert math.isclose(tip_z_from_root_z(0.27), 0.27 - 0.106203, abs_tol=1e-9)


def test_root_z_from_tip_z_is_the_inverse():
    for tip_z in (0.0, 0.024, 0.164):
        root_z = root_z_from_tip_z(tip_z)
        assert math.isclose(tip_z_from_root_z(root_z), tip_z, abs_tol=1e-9)


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
    import pytest

    for bad in (-0.01, math.pi / 2 + 0.01, math.pi):
        with pytest.raises(ValueError):
            transport_goal_ranges(bad)


def test_transport_goal_ranges_accepts_tilt_at_the_boundaries():
    transport_goal_ranges(0.0)
    transport_goal_ranges(math.pi / 2)


def test_validate_transport_goal_z_accepts_lo_less_than_hi():
    validate_transport_goal_z(0.13, 0.27)


def test_validate_transport_goal_z_rejects_lo_ge_hi():
    import pytest

    with pytest.raises(ValueError):
        validate_transport_goal_z(0.27, 0.13)
    with pytest.raises(ValueError):
        validate_transport_goal_z(0.20, 0.20)


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
    # tip band: 0.13 - 0.106203 = 0.023797, 0.27 - 0.106203 = 0.163797
    assert "0.024" in text
    assert "0.164" in text
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
