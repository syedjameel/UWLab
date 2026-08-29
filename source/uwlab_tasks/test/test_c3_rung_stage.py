# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit-proves the C3 RUNG stage's core -- ``RESET_SPEC_V2.md`` sec 1 C3, bead ``dr-ai1.4``,
**C3 = 50% S1 + 50% S_t** -- without touching Isaac at all.

Needs only plain ``python3`` -- no Isaac Sim, no GPU, no env construction. Same reason and same
technique as ``test_c3_transport_stage.py`` next to this file: ``c3_rung_core.py`` has no
``isaaclab`` import by design, so it is loaded here BY FILE PATH. ``c3_transport_core`` is loaded
first and registered in ``sys.modules`` because ``c3_rung_core`` delegates its root/tip conversion
to it (and must -- see below).

RUNNABLE WITH BARE ``python3``, DELIBERATELY. ``test_c3_transport_stage.py``'s ``__main__`` runner
aborts partway on a host without ``pytest`` installed, because several of its cases ``import
pytest`` for ``pytest.raises``. This file uses :func:`_raises` instead, so the same file runs
end-to-end under bare ``python3`` AND collects normally under ``pytest``.

WHAT THIS COVERS -- the three things bead dr-ai1.4 names, plus the frame rule they all sit on:

1. **The 50/50 split ratio** -- :func:`test_the_split_is_fifty_fifty_at_the_spec_default` and the
   band-edge cases around it.
2. **S_t's goal equals the object's OWN pose** -- position AND orientation, zero delta, no
   reorientation: :func:`test_st_goal_is_the_objects_own_pose_exactly`.
3. **S1's goal is tip-down and offset along the bore axis** --
   :func:`test_s1_goal_is_offset_along_the_bore_axis` and
   :func:`test_s1_goal_orientation_is_the_spawn_orientation_tip_down`.
4. **S_t is HORIZONTAL, so no 106.203 mm subtraction applies to it** (F51 + F49) --
   :func:`test_st_tip_z_equals_root_z_because_the_peg_is_horizontal`. This is the case that would
   fail if a future reader re-inferred a tip-down S_t, which is exactly what happened once already.

WHAT IT CANNOT COVER. The Isaac-touching half (``mdp/c3_rung.py``'s two terms and the wiring in
``dexlift_ur5e_delto_tableleg_env_cfg._apply_c3_rung_stage``) needs Isaac Sim and is out of scope
here. In particular this file does NOT prove that the runtime tensor draw agrees with
:func:`c3_kind_for_draw`, only that the scalar rule those two share is right.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

_MDP_DIR = Path(__file__).resolve().parents[1] / "uwlab_tasks/manager_based/manipulation/dexlift/mdp"


def _load(name: str):
    """Load an Isaac-free ``mdp`` module by FILE PATH, compiling the source text directly.

    NOT ``spec_from_file_location(...).loader.exec_module(...)`` (what ``test_c3_transport_stage.py``
    uses), because that consults and writes ``__pycache__``, and CPython's staleness check compares
    the source mtime at ONE-SECOND granularity. Observed while writing this file: a negative-control
    edit, a run, and the restore all landed inside the same second, so the next run silently
    executed the MUTATED bytecode against the restored source and reported 4 failures for code that
    was correct on disk. A test that can report yesterday's bytecode is worse than no test --
    compiling the text every time costs microseconds and removes the failure mode.
    """
    path = _MDP_DIR / f"{name}.py"
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)  # noqa: S102
    return module


# Order matters: c3_rung_core falls back to a plain ``import c3_transport_core`` when loaded outside
# a package, so the sibling must already be in sys.modules.
_c3_transport_core = _load("c3_transport_core")
_c3_rung_core = _load("c3_rung_core")

C3_KIND_S1 = _c3_rung_core.C3_KIND_S1
C3_KIND_ST = _c3_rung_core.C3_KIND_ST
DEFAULT_S1_FRACTION = _c3_rung_core.DEFAULT_S1_FRACTION
DEFAULT_S1_GOAL_DELTA_MM = _c3_rung_core.DEFAULT_S1_GOAL_DELTA_MM
S1_NOMINAL_TILT_RAD = _c3_rung_core.S1_NOMINAL_TILT_RAD
ST_NOMINAL_TILT_RAD = _c3_rung_core.ST_NOMINAL_TILT_RAD
c3_kind_counts = _c3_rung_core.c3_kind_counts
c3_kind_for_draw = _c3_rung_core.c3_kind_for_draw
c3_rung_banner = _c3_rung_core.c3_rung_banner
goal_tip_z_from_root_z = _c3_rung_core.goal_tip_z_from_root_z
nominal_tilt_rad = _c3_rung_core.nominal_tilt_rad
parse_c3_rung_env = _c3_rung_core.parse_c3_rung_env
s1_goal_orientation = _c3_rung_core.s1_goal_orientation
s1_goal_position = _c3_rung_core.s1_goal_position
st_goal_pose = _c3_rung_core.st_goal_pose
validate_s1_fraction = _c3_rung_core.validate_s1_fraction
validate_s1_goal_delta_mm = _c3_rung_core.validate_s1_goal_delta_mm

ROOT_ABOVE_TIP_M = _c3_transport_core.ROOT_ABOVE_TIP_M


def _raises(exc_type, fn, *args, **kwargs):
    """``pytest.raises`` without pytest -- see the module docstring. Returns the exception so a
    caller can assert on its message."""
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"expected {exc_type.__name__} from {getattr(fn, '__name__', fn)}(...)")


# ---------------------------------------------------------------------------------------------
# 1. THE 50/50 SPLIT
# ---------------------------------------------------------------------------------------------


def test_the_spec_default_is_fifty_fifty():
    # RESET_SPEC_V2.md sec 1: "Composition: C3 = 50% S1 + 50% S_t."
    assert DEFAULT_S1_FRACTION == 0.5


def test_the_split_is_fifty_fifty_at_the_spec_default():
    # A deterministic sweep of the unit interval, not a random sample: with 1000 evenly spaced
    # draws, exactly half must land in each half. Deterministic on purpose -- a random draw would
    # make this test's pass/fail depend on a seed, and the property under test (the band boundary)
    # is exact, not statistical.
    draws = [i / 1000.0 for i in range(1000)]
    counts = c3_kind_counts(draws, DEFAULT_S1_FRACTION)
    assert counts[C3_KIND_S1] == 500, counts
    assert counts[C3_KIND_ST] == 500, counts


def test_the_split_reports_both_halves_even_when_one_is_empty():
    # Both keys always present, zeros included: "no S_t envs this reset" must not be readable as
    # "S_t is not configured".
    counts = c3_kind_counts([0.1, 0.2], 1.0)
    assert counts == {C3_KIND_S1: 2, C3_KIND_ST: 0}


def test_the_band_is_half_open_so_a_zero_fraction_yields_no_s1():
    # ``draw < s1_fraction`` with draws in [0, 1): nothing is < 0.0, so a zero fraction really is
    # zero S1 envs. Same convention MixtureResetObject.__call__'s bands already use.
    assert c3_kind_for_draw(0.0, 0.0) == C3_KIND_ST
    counts = c3_kind_counts([i / 100.0 for i in range(100)], 0.0)
    assert counts[C3_KIND_S1] == 0


def test_the_band_is_half_open_so_a_unit_fraction_yields_no_st():
    counts = c3_kind_counts([i / 100.0 for i in range(100)], 1.0)
    assert counts[C3_KIND_ST] == 0


def test_a_draw_exactly_on_the_boundary_goes_to_st():
    # The boundary belongs to S_t (strict ``<`` for S1). Pinned so a future refactor to ``<=``
    # shows up as a failure rather than as a half-percent drift nobody measures.
    assert c3_kind_for_draw(0.5, 0.5) == C3_KIND_ST
    assert c3_kind_for_draw(0.4999999, 0.5) == C3_KIND_S1


def test_an_uneven_split_is_allowed_for_characterising_one_half_alone():
    counts = c3_kind_counts([i / 1000.0 for i in range(1000)], 0.25)
    assert counts[C3_KIND_S1] == 250
    assert counts[C3_KIND_ST] == 750


def test_a_fraction_outside_zero_to_one_is_rejected():
    for bad in (-0.01, 1.01, 2.0):
        _raises(ValueError, validate_s1_fraction, bad)


def test_the_boundary_fractions_are_accepted():
    validate_s1_fraction(0.0)
    validate_s1_fraction(1.0)


# ---------------------------------------------------------------------------------------------
# 2. S_t -- THE GOAL IS THE OBJECT'S OWN POSE, ZERO DELTA
# ---------------------------------------------------------------------------------------------


def test_st_goal_is_the_objects_own_pose_exactly():
    # "The target is placed exactly where the leg is" -- RESET_SPEC_V2.md sec 1 C3. Position AND
    # orientation, zero delta: the policy acquires and holds, it does not transport and it does not
    # reorient.
    spawn_pos = (0.31, -0.07, 0.0153)
    spawn_quat = (0.7071067811865476, 0.0, 0.7071067811865476, 0.0)  # some horizontal-lying pose
    goal_pos, goal_quat = st_goal_pose(spawn_pos, spawn_quat)
    assert goal_pos == tuple(spawn_pos)
    assert goal_quat == tuple(spawn_quat)


def test_st_goal_moves_nothing_whatever_the_spawn_is():
    # Exercised across poses so this cannot pass by coincidence on one hand-picked pose.
    for spawn_pos in ((0.0, 0.0, 0.0), (0.4, 0.2, 0.015), (-0.1, 0.9, 0.3)):
        for spawn_quat in ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.5, 0.5, 0.5, 0.5)):
            goal_pos, goal_quat = st_goal_pose(spawn_pos, spawn_quat)
            assert goal_pos == spawn_pos
            assert goal_quat == spawn_quat


def test_st_tip_z_equals_root_z_because_the_peg_is_horizontal():
    # THE CASE THAT GUARDS THE F51 CORRECTION. S_t's peg is HORIZONTAL (user clarification
    # 2026-08-29; measured baseline n=2048 settled: 99.02% lie flat with the tip within 20 mm of the
    # table). Horizontal means the 106.203 mm root-above-tip offset lies in the horizontal plane and
    # projects to ZERO on world z (F49: root_z - tip_z = ROOT_ABOVE_TIP_M * cos(tilt), cos(pi/2)=0).
    # A bare ``root_z - 0.106203`` here would be wrong by the full 106 mm -- and re-inferring a
    # tip-down S_t would break this assertion, which is the point of writing it.
    for root_z in (0.0153, 0.05, 0.20):
        assert math.isclose(goal_tip_z_from_root_z(root_z, C3_KIND_ST), root_z, abs_tol=1e-9)


def test_st_nominal_tilt_is_horizontal():
    assert math.isclose(ST_NOMINAL_TILT_RAD, math.pi / 2, abs_tol=1e-12)
    assert math.isclose(nominal_tilt_rad(C3_KIND_ST), math.pi / 2, abs_tol=1e-12)


# ---------------------------------------------------------------------------------------------
# 3. S1 -- TIP-DOWN, OFFSET ALONG THE BORE AXIS
# ---------------------------------------------------------------------------------------------


def test_s1_goal_is_offset_along_the_bore_axis():
    # The bore's deep axis points INTO the bore -- world (0, 0, -1) at the fixture's nominal,
    # yaw-only orientation (live_bore_deep_axis asserts its world z is < -0.9). A POSITIVE delta
    # therefore moves the goal DOWN, deeper into the bore, and only along that axis.
    spawn_pos = (0.30, 0.05, 0.1362)
    axis = (0.0, 0.0, -1.0)
    goal = s1_goal_position(spawn_pos, axis, 0.005)
    assert math.isclose(goal[0], 0.30, abs_tol=1e-12)
    assert math.isclose(goal[1], 0.05, abs_tol=1e-12)
    assert math.isclose(goal[2], 0.1362 - 0.005, abs_tol=1e-12)


def test_s1_goal_offset_follows_a_tilted_bore_axis_rather_than_world_z():
    # The axis is read off the FIXTURE's live orientation, so the displacement must follow whatever
    # that axis is -- not world -Z. A version that hardcoded "subtract delta from z" would pass the
    # test above and fail this one.
    axis = (0.6, 0.0, -0.8)  # unit, tilted
    goal = s1_goal_position((0.0, 0.0, 0.0), axis, 0.010)
    assert math.isclose(goal[0], 0.006, abs_tol=1e-12)
    assert math.isclose(goal[2], -0.008, abs_tol=1e-12)


def test_s1_goal_offset_sign_reverses_along_the_same_axis():
    # One expression for both signs, as GoalBelowSpawnPoseCommand does it: a negative delta goes
    # back OUT of the mouth along the same axis. Nothing branches on the sign.
    axis = (0.0, 0.0, -1.0)
    assert math.isclose(s1_goal_position((0.0, 0.0, 0.1), axis, -0.005)[2], 0.105, abs_tol=1e-12)


def test_s1_goal_orientation_is_the_spawn_orientation_tip_down():
    # S1's spawn is the partial-assembly pose -- leg pre-inserted in the bore, hence TIP-DOWN by
    # construction (F43 measured that branch's goal tilt at 0.00-0.28 deg from tip-down). The goal
    # returns that orientation UNCHANGED, which is what makes S1 a DEPTH task about the mating frame
    # and never a reorientation task.
    tip_down_quat = (0.7071067811865476, 0.0, -0.7071067811865476, 0.0)  # Ry(-90)
    assert s1_goal_orientation(tip_down_quat) == tip_down_quat


def test_s1_tip_z_subtracts_the_full_offset_because_the_peg_is_tip_down():
    # The other extreme of the same cosine: tip-down, cos(0) = 1, the full 106.203 mm applies. Read
    # against the S_t case above, these two are the whole of F49's rule.
    for root_z in (0.1325, 0.1362, 0.1404):  # F43's measured kind-2 (partial-assembly) root band
        assert math.isclose(goal_tip_z_from_root_z(root_z, C3_KIND_S1), root_z - ROOT_ABOVE_TIP_M, abs_tol=1e-9)


def test_s1_nominal_tilt_is_tip_down():
    assert S1_NOMINAL_TILT_RAD == 0.0
    assert nominal_tilt_rad(C3_KIND_S1) == 0.0


def test_the_two_halves_convert_root_to_tip_differently():
    # The single fact that makes a shared bare subtraction impossible: at the same root height the
    # two halves imply tip heights 106.203 mm apart.
    root_z = 0.20
    assert math.isclose(
        goal_tip_z_from_root_z(root_z, C3_KIND_ST) - goal_tip_z_from_root_z(root_z, C3_KIND_S1),
        ROOT_ABOVE_TIP_M,
        abs_tol=1e-9,
    )


def test_the_conversion_delegates_to_the_shared_f49_helper():
    # Not a private copy of the arithmetic: c3_rung_core must produce exactly what
    # c3_transport_core.tip_z_from_root_z produces at the matching tilt. If someone reimplements the
    # conversion locally, this is what catches it.
    for kind in (C3_KIND_S1, C3_KIND_ST):
        for root_z in (0.02, 0.15, 0.30):
            assert goal_tip_z_from_root_z(root_z, kind) == _c3_transport_core.tip_z_from_root_z(
                root_z, tilt_rad=nominal_tilt_rad(kind)
            )


def test_an_unknown_kind_is_rejected_rather_than_silently_converted():
    _raises(ValueError, nominal_tilt_rad, 7)
    _raises(ValueError, goal_tip_z_from_root_z, 0.2, 7)


# ---------------------------------------------------------------------------------------------
# 4. THE ENV-VAR STAGE: OFF BY DEFAULT, VALIDATED WHEN ON
# ---------------------------------------------------------------------------------------------


def test_the_stage_is_off_when_the_variable_is_unset():
    assert parse_c3_rung_env({}) is None


def test_the_stage_is_off_for_anything_that_is_not_exactly_one():
    for raw in ("0", "", "true", "True", "yes", "1 ", "01"):
        assert parse_c3_rung_env({"DEXRESET_C3_RUNG": raw}) is None, raw


def test_the_stage_defaults_to_the_spec_fifty_fifty_and_the_shipped_s1_delta():
    staging = parse_c3_rung_env({"DEXRESET_C3_RUNG": "1"})
    assert staging is not None
    assert staging.s1_fraction == 0.5
    assert staging.st_fraction == 0.5
    # +5.0 mm is what launch_dexreset_s1_s2_bank_gen.sh already runs S1 at (S1_GOAL_BELOW_SPAWN_MM
    # default 5), carried over rather than re-picked -- expressed in METRES on the staging object.
    assert math.isclose(staging.s1_goal_delta_m, 0.005, abs_tol=1e-12)
    assert DEFAULT_S1_GOAL_DELTA_MM == 5.0


def test_the_st_fraction_is_always_the_complement():
    for frac in ("0.0", "0.25", "0.5", "1.0"):
        staging = parse_c3_rung_env({"DEXRESET_C3_RUNG": "1", "DEXRESET_C3_S1_FRACTION": frac})
        assert math.isclose(staging.s1_fraction + staging.st_fraction, 1.0, abs_tol=1e-12)


def test_the_fraction_and_delta_can_be_overridden():
    staging = parse_c3_rung_env(
        {"DEXRESET_C3_RUNG": "1", "DEXRESET_C3_S1_FRACTION": "0.25", "DEXRESET_C3_S1_GOAL_DELTA_MM": "-60"}
    )
    assert staging.s1_fraction == 0.25
    assert math.isclose(staging.s1_goal_delta_m, -0.060, abs_tol=1e-12)


def test_a_non_numeric_fraction_is_rejected_with_the_variable_named():
    exc = _raises(ValueError, parse_c3_rung_env, {"DEXRESET_C3_RUNG": "1", "DEXRESET_C3_S1_FRACTION": "half"})
    assert "DEXRESET_C3_S1_FRACTION" in str(exc)


def test_an_out_of_range_fraction_is_rejected():
    _raises(ValueError, parse_c3_rung_env, {"DEXRESET_C3_RUNG": "1", "DEXRESET_C3_S1_FRACTION": "1.5"})


def test_a_non_numeric_delta_is_rejected_with_the_variable_named():
    exc = _raises(
        ValueError, parse_c3_rung_env, {"DEXRESET_C3_RUNG": "1", "DEXRESET_C3_S1_GOAL_DELTA_MM": "5mm"}
    )
    assert "DEXRESET_C3_S1_GOAL_DELTA_MM" in str(exc)


def test_the_s1_delta_bounds_match_the_legacy_shaping_command():
    # Same signed bounds GoalBelowSpawnPoseCommand / _apply_partial_assembly_and_goal_toggles
    # enforce: +25 mm is the bore's engaged span, -200 mm is headroom around the S2' band.
    validate_s1_goal_delta_mm(25.0)
    validate_s1_goal_delta_mm(-200.0)
    _raises(ValueError, validate_s1_goal_delta_mm, 25.01)
    _raises(ValueError, validate_s1_goal_delta_mm, -200.01)


def test_an_out_of_bounds_delta_is_rejected_at_parse_time_before_isaac_starts():
    _raises(ValueError, parse_c3_rung_env, {"DEXRESET_C3_RUNG": "1", "DEXRESET_C3_S1_GOAL_DELTA_MM": "40"})


def test_the_stage_does_not_read_the_surplus_st_tipdown_toggle():
    # F51: DEXRESET_ST_SPAWN_TIPDOWN (commit dffe5de) is surplus to S_t and stays OFF. Setting it
    # must change nothing about this stage -- S_t needs NO spawn change.
    on = parse_c3_rung_env({"DEXRESET_C3_RUNG": "1"})
    also_on = parse_c3_rung_env({"DEXRESET_C3_RUNG": "1", "DEXRESET_ST_SPAWN_TIPDOWN": "1"})
    assert on == also_on


# ---------------------------------------------------------------------------------------------
# 5. THE BANNER (R5: a run must state its staging)
# ---------------------------------------------------------------------------------------------


def test_the_banner_names_both_fractions_and_the_s1_delta():
    text = c3_rung_banner(parse_c3_rung_env({"DEXRESET_C3_RUNG": "1"}))
    assert "0.500 of envs draw S1" in text
    assert "0.500 draw S_t" in text
    assert "+5.00 mm" in text
    assert "bead dr-ai1.4" in text


def test_the_banner_states_that_st_is_horizontal_and_zero_delta():
    text = c3_rung_banner(parse_c3_rung_env({"DEXRESET_C3_RUNG": "1"}))
    assert "HORIZONTAL" in text
    assert "ZERO delta" in text
    assert "99.02%" in text
    assert "DEXRESET_ST_SPAWN_TIPDOWN is not read here" in text


def test_the_banner_prints_both_root_to_tip_conversions_so_they_cannot_be_confused():
    text = c3_rung_banner(parse_c3_rung_env({"DEXRESET_C3_RUNG": "1"}))
    # root 0.200 -> tip 0.0938 for S1 (tip-down, full offset) and 0.2000 for S_t (horizontal, none).
    assert "0.0938" in text
    assert "0.2000" in text
    assert "F49" in text


if __name__ == "__main__":
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
            except Exception as _exc:  # noqa: BLE001 -- a runner, it must report every failure
                _failures += 1
                print(f"[c3_rung] {_name} FAILED: {type(_exc).__name__}: {_exc}", flush=True)
            else:
                print(f"[c3_rung] {_name} OK", flush=True)
    if _failures:
        print(f"[c3_rung] {_failures} test(s) FAILED", flush=True)
        raise SystemExit(1)
    print("[c3_rung] all tests passed", flush=True)
