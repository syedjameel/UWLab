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

import ast
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


# ---------------------------------------------------------------------------------------------
# 6. S_t's DEFERRED RE-PIN AT THE SETTLED POSE (bead dr-ai1.18, V2_C3_DESIGN.md sec 7)
# ---------------------------------------------------------------------------------------------

# A scripted drop: the leg spawns airborne in a RANDOMIZED orientation, bounces once (its linear
# speed passes through zero at the apex -- the trap a speed-only predicate falls into), and finally
# comes to rest lying flat. Steps are (step, speed_mps, pos, quat). Orientations are stand-ins; only
# "spawn orientation differs from resting orientation" matters, which is the rung-inverting case
# V2_C3_DESIGN.md sec 7 describes.
_SPAWN_QUAT = (0.5, 0.5, 0.5, 0.5)  # randomized, ~90 deg from flat
_RESTING_QUAT = (0.7071067811865476, 0.0, 0.7071067811865476, 0.0)  # lying flat
_SPAWN_POS = (0.31, -0.07, 0.0480)  # mid-air, inside the [0, 0.05] m drop clamp
_APEX_POS = (0.32, -0.07, 0.0300)  # bounce apex: speed 0.0 but NOT settled
_RESTING_POS = (0.33, -0.06, 0.0153)  # at rest, tip on the table

_PIVOT_POS = (0.33, -0.06, 0.0155)  # linearly still, but pivoting on a corner
_PIVOT_QUAT = (0.66, 0.25, 0.66, 0.24)  # mid-pivot: NOT the orientation it will come to rest at

# (step, linear speed m/s, ANGULAR speed rad/s, pos, quat)
_DROP_TRAJECTORY = [
    (0, 0.00, 0.00, _SPAWN_POS, _SPAWN_QUAT),  # the instant CommandManager.reset() reads
    (10, 1.20, 6.00, (0.31, -0.07, 0.030), (0.5, 0.5, 0.5, 0.5)),
    (30, 0.00, 5.00, _APEX_POS, (0.6, 0.4, 0.5, 0.48)),  # apex: zero LINEAR speed, still tumbling
    (55, 0.90, 3.00, (0.33, -0.06, 0.020), (0.68, 0.1, 0.70, 0.05)),
    (61, 0.30, 1.50, (0.33, -0.06, 0.016), (0.70, 0.02, 0.71, 0.01)),  # past the floor, still moving
    (65, 0.01, 0.80, _PIVOT_POS, _PIVOT_QUAT),  # LINEARLY STILL but ROTATING -- must not fire
    (70, 0.02, 0.01, _RESTING_POS, _RESTING_QUAT),  # FIRST genuinely settled step
    (90, 0.00, 0.00, (0.34, -0.06, 0.0153), (0.70, 0.0, 0.71, 0.0)),  # later; must NOT re-pin again
]

_MIN_STEPS = 60  # held_check_core.SETTLE_STEPS; passed in, never imported here (torch dependency)


def _play_drop(trajectory, *, settle_speed_mps=None, settle_ang_speed_rad_s=None, min_steps=_MIN_STEPS):
    """Run a scripted drop through st_should_repin and return (fire_count, pinned_pose_or_None).

    The pinned pose is what st_goal_pose returns at the step the re-pin fires -- i.e. exactly what
    the runtime writes into pose_command_b.
    """
    if settle_speed_mps is None:
        settle_speed_mps = _c3_rung_core.DEFAULT_ST_SETTLE_SPEED_MPS
    if settle_ang_speed_rad_s is None:
        settle_ang_speed_rad_s = _c3_rung_core.DEFAULT_ST_SETTLE_ANG_SPEED_RAD_S
    already = False
    fires = 0
    pinned = None
    for step, speed, ang_speed, pos, quat in trajectory:
        if _c3_rung_core.st_should_repin(
            already_repinned=already,
            steps_since_reset=step,
            object_lin_speed_mps=speed,
            object_ang_speed_rad_s=ang_speed,
            settle_speed_mps=settle_speed_mps,
            settle_ang_speed_rad_s=settle_ang_speed_rad_s,
            min_steps=min_steps,
        ):
            fires += 1
            pinned = st_goal_pose(pos, quat)
            already = True
    return fires, pinned


def test_the_repin_lands_on_the_settled_pose_not_the_spawn_pose():
    # THE TEST THE WHOLE RE-PIN EXISTS FOR. Under a randomized spawn orientation the pre-settle pin
    # and the post-settle pin MUST differ -- in orientation above all, since a spawn-pinned goal
    # commands a reorientation, which is the one thing S_t must never ask for (V2_C3_DESIGN.md
    # sec 7). This test FAILS against the old spawn-pinned behaviour, which is the point of it.
    fires, pinned = _play_drop(_DROP_TRAJECTORY)
    assert fires == 1, fires
    pinned_pos, pinned_quat = pinned
    assert pinned_pos == _RESTING_POS
    assert pinned_quat == _RESTING_QUAT
    # And explicitly NOT the spawn pose -- both components.
    assert pinned_pos != _SPAWN_POS
    assert pinned_quat != _SPAWN_QUAT


def test_the_repin_fires_exactly_once_per_episode():
    # The latch. Later settled steps must not move the goal again -- after the re-pin the target is
    # constant, so nothing downstream ever sees it move.
    fires, _ = _play_drop(_DROP_TRAJECTORY)
    assert fires == 1


def test_the_repin_does_not_fire_at_the_bounce_apex():
    # A dropped leg's linear speed passes through ZERO at every apex. A speed-only predicate would
    # re-pin there -- mid-air, mid-tumble -- reintroducing the exact defect this mechanism removes.
    # The step floor is what prevents it; this proves the floor is load-bearing.
    apex_only = [t for t in _DROP_TRAJECTORY if t[0] <= 30]
    fires, pinned = _play_drop(apex_only)
    assert fires == 0, f"re-pinned at the apex: {pinned}"


def test_the_step_floor_alone_is_not_enough_either():
    # Symmetric check: past the step floor but still moving (step 61, 0.30 m/s) must not fire.
    assert not _c3_rung_core.st_should_repin(
        already_repinned=False,
        steps_since_reset=61,
        object_lin_speed_mps=0.30,
        object_ang_speed_rad_s=0.0,
        settle_speed_mps=0.05,
        settle_ang_speed_rad_s=0.05,
        min_steps=_MIN_STEPS,
    )


def test_the_repin_fires_on_the_first_step_where_both_conditions_hold():
    assert _c3_rung_core.st_should_repin(
        already_repinned=False,
        steps_since_reset=61,
        object_lin_speed_mps=0.05,  # exactly at the ceiling -- inclusive
        object_ang_speed_rad_s=0.05,  # likewise
        settle_speed_mps=0.05,
        settle_ang_speed_rad_s=0.05,
        min_steps=_MIN_STEPS,
    )
    # ... and not one step earlier: the floor is strict (steps > min_steps), matching held_check's
    # own `settled = steps > self.settle_steps`.
    assert not _c3_rung_core.st_should_repin(
        already_repinned=False,
        steps_since_reset=60,
        object_lin_speed_mps=0.0,
        object_ang_speed_rad_s=0.0,
        settle_speed_mps=0.05,
        settle_ang_speed_rad_s=0.05,
        min_steps=_MIN_STEPS,
    )


def test_the_latch_blocks_every_later_step():
    assert not _c3_rung_core.st_should_repin(
        already_repinned=True,
        steps_since_reset=999,
        object_lin_speed_mps=0.0,
        object_ang_speed_rad_s=0.0,
        settle_speed_mps=0.05,
        settle_ang_speed_rad_s=0.05,
        min_steps=_MIN_STEPS,
    )


def test_a_never_settling_leg_simply_keeps_the_provisional_goal():
    # No fire, no crash, no exception -- the provisional reset-time pin stands. Stated as a test so
    # the degenerate case is a known, benign outcome rather than an unexamined one.
    never = [(s, 0.90, 2.0, _SPAWN_POS, _SPAWN_QUAT) for s in (10, 60, 120, 300)]
    fires, pinned = _play_drop(never)
    assert fires == 0
    assert pinned is None


def test_the_repin_does_not_fire_while_the_leg_is_still_rotating():
    # THE ANGULAR TERM'S OWN CASE (team-lead decision 2026-08-29). Step 65 of the trajectory is
    # linearly still (0.01 m/s, under the 0.05 ceiling) and past the step floor, but still ROTATING
    # at 0.80 rad/s in an orientation it will not come to rest at. A linear-only predicate pins
    # _PIVOT_QUAT there -- a WRONG orientation captured at a moment of zero linear speed, which is
    # the exact failure this bead removes, reached by a narrower path.
    up_to_pivot = [t for t in _DROP_TRAJECTORY if t[0] <= 65]
    fires, pinned = _play_drop(up_to_pivot)
    assert fires == 0, f"re-pinned mid-pivot: {pinned}"
    # and the full run still lands on the resting pose, never the pivot pose
    _, pinned_full = _play_drop(_DROP_TRAJECTORY)
    assert pinned_full[1] != _PIVOT_QUAT


def test_the_angular_ceiling_is_inclusive_and_rejects_just_above():
    common = dict(
        already_repinned=False,
        steps_since_reset=61,
        object_lin_speed_mps=0.0,
        settle_speed_mps=0.05,
        settle_ang_speed_rad_s=0.05,
        min_steps=_MIN_STEPS,
    )
    assert _c3_rung_core.st_should_repin(object_ang_speed_rad_s=0.05, **common)
    assert not _c3_rung_core.st_should_repin(object_ang_speed_rad_s=0.0501, **common)


def test_the_angular_default_comes_from_the_f50_f51_settled_pair():
    # F50/F51: "settled (lin < 0.01 m/s, ang < 0.05 rad/s)". The ANGULAR half is taken verbatim.
    assert _c3_rung_core.DEFAULT_ST_SETTLE_ANG_SPEED_RAD_S == 0.05


def test_the_two_settle_thresholds_are_independent_not_a_shared_constant():
    # Both read 0.05 but in DIFFERENT UNITS from DIFFERENT sources -- linear from
    # --c2_max_resting_speed (cross-layer agreement with the generation side), angular from
    # F50/F51. They are deliberately not linked; the docstring says so, and this pins that
    # overriding one leaves the other alone, so nobody "unifies" them later.
    staging = parse_c3_rung_env({"DEXRESET_C3_RUNG": "1", "DEXRESET_C3_ST_SETTLE_SPEED": "0.01"})
    assert staging.st_settle_speed_mps == 0.01
    assert staging.st_settle_ang_speed_rad_s == 0.05
    staging = parse_c3_rung_env({"DEXRESET_C3_RUNG": "1", "DEXRESET_C3_ST_SETTLE_ANG_SPEED": "0.2"})
    assert staging.st_settle_ang_speed_rad_s == 0.2
    assert staging.st_settle_speed_mps == 0.05
    # And the linear bound must NOT have been "fixed" to F50/F51's tighter 0.01 -- cross-layer
    # agreement with --c2_max_resting_speed is the deliberate choice.
    assert _c3_rung_core.DEFAULT_ST_SETTLE_SPEED_MPS == 0.05


def test_a_zero_angular_settle_ceiling_is_rejected():
    _raises(ValueError, parse_c3_rung_env, {"DEXRESET_C3_RUNG": "1", "DEXRESET_C3_ST_SETTLE_ANG_SPEED": "0"})
    _raises(ValueError, parse_c3_rung_env, {"DEXRESET_C3_RUNG": "1", "DEXRESET_C3_ST_SETTLE_ANG_SPEED": "-1"})


def test_a_non_numeric_angular_ceiling_is_rejected_with_the_variable_named():
    exc = _raises(
        ValueError, parse_c3_rung_env, {"DEXRESET_C3_RUNG": "1", "DEXRESET_C3_ST_SETTLE_ANG_SPEED": "slow"}
    )
    assert "DEXRESET_C3_ST_SETTLE_ANG_SPEED" in str(exc)


def test_the_settle_speed_default_matches_the_generation_side_resting_convention():
    # 0.05 m/s is --c2_max_resting_speed / --c4_rewind_max_speed in generate_reset_states_policy.py:
    # "object linear velocity magnitude ... at the REWOUND step", measured medians 0.000-0.049 m/s.
    # Matching it means the env side and the generation side call the same states resting.
    assert _c3_rung_core.DEFAULT_ST_SETTLE_SPEED_MPS == 0.05


def test_the_min_steps_argument_is_required_so_the_number_lives_in_one_place():
    # No default: c3_rung.py passes held_check_core.SETTLE_STEPS. Same API shape as
    # c3_transport_core.tip_z_from_root_z's required tilt_rad (F49b: "the fix worth copying is the
    # API shape, not the arithmetic"). This module must not carry a second copy of 60.
    _raises(
        TypeError,
        _c3_rung_core.st_should_repin,
        already_repinned=False,
        steps_since_reset=61,
        object_lin_speed_mps=0.0,
        object_ang_speed_rad_s=0.0,
        settle_speed_mps=0.05,
        settle_ang_speed_rad_s=0.05,
    )
    src = (_MDP_DIR / "c3_rung_core.py").read_text()
    assert "SETTLE_STEPS" in src, "the pointer to the source of the number must be named"
    # No EXECUTABLE copy of 60 -- checked on the AST, not by grepping text. A docstring may cite
    # held_check_core's value (and does, as provenance); what must not exist is an assignment or a
    # parameter default carrying it, because only those can drift into use.
    import ast

    # BOTH modules of the stage. c3_rung.py is where SETTLE_STEPS is actually imported and used,
    # so it is at least as likely a place for the number to be restated, and the earlier version of
    # this test did not look at it at all -- a SCOPE gap rather than a node-type one.
    for module_name in ("c3_rung_core.py", "c3_rung.py"):
        tree = ast.parse((_MDP_DIR / module_name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == 60 and not isinstance(node.value, bool):
                raise AssertionError(
                    f"{module_name} restates held_check's 60 as an executable literal at line"
                    f" {node.lineno}; it must be passed in from held_check_core.SETTLE_STEPS instead"
                )


def test_the_settle_knobs_are_parameterised_not_hardcoded():
    staging = parse_c3_rung_env(
        {
            "DEXRESET_C3_RUNG": "1",
            "DEXRESET_C3_ST_SETTLE_SPEED": "0.01",
            "DEXRESET_C3_ST_SETTLE_STEPS": "120",
        }
    )
    assert staging.st_settle_speed_mps == 0.01
    assert staging.st_settle_min_steps == 120


def test_the_settle_steps_default_to_none_meaning_the_shared_constant():
    staging = parse_c3_rung_env({"DEXRESET_C3_RUNG": "1"})
    assert staging.st_settle_min_steps is None
    assert staging.st_settle_speed_mps == 0.05


def test_a_zero_settle_speed_is_rejected_because_it_would_silently_never_fire():
    _raises(ValueError, parse_c3_rung_env, {"DEXRESET_C3_RUNG": "1", "DEXRESET_C3_ST_SETTLE_SPEED": "0"})
    _raises(ValueError, parse_c3_rung_env, {"DEXRESET_C3_RUNG": "1", "DEXRESET_C3_ST_SETTLE_SPEED": "-0.1"})


def test_a_negative_settle_step_floor_is_rejected():
    _raises(ValueError, parse_c3_rung_env, {"DEXRESET_C3_RUNG": "1", "DEXRESET_C3_ST_SETTLE_STEPS": "-1"})


def test_the_banner_states_the_repin_and_both_of_its_conditions():
    text = c3_rung_banner(parse_c3_rung_env({"DEXRESET_C3_RUNG": "1"}))
    assert "RE-PINNED ONCE at the SETTLED pose" in text
    assert "held_check_core.SETTLE_STEPS" in text
    assert "0.050 m/s" in text
    assert "0.050 rad/s" in text
    assert "dr-ai1.18" in text


# ---------------------------------------------------------------------------------------------
# 7. DRIFT GUARD: our goal expression vs the episode mixture's (team-lead decision, task 2)
# ---------------------------------------------------------------------------------------------
#
# The team lead's call was: KEEP the duplicated goal expression rather than subclass
# MixtureGoalPoseCommand (subclassing would drag in that class's gating and banners for the sake of
# ~10 lines), but PIN the two copies against each other, because two statements of one convention is
# the F27 defect class this campaign keeps rediscovering. When one drifts, this fails -- instead of a
# bank silently filling with wrong states.
#
# The comparison is STRUCTURAL, on the real source of both methods, via `ast`. It cannot be numeric:
# both methods are torch code operating on Isaac scene handles, and neither torch nor isaaclab is
# importable in this test's environment. Normalising the delta operand is the only licensed
# difference -- ours is a parameter (`delta_m`), the mixture's is an attribute
# (`self._partial_delta_m`) -- and everything else must match token for token.

_MIXTURE_PATH = _MDP_DIR / "episode_mixture.py"
_C3_RUNG_PATH = _MDP_DIR / "c3_rung.py"
_DELTA_TOKENS = ("self._partial_delta_m", "delta_m")


# Node types the walker below must see. The original pair -- (Assign, If) -- had the AnnAssign
# blind spot twice over, and worse: a statement ADDED to either method in a type the walker did not
# collect was invisible to every assertion in this section, because they all test MEMBERSHIP. The
# routes that actually matter in torch code are AugAssign (`goal_pos_w += ...`, the natural way to
# bolt on an extra offset), AnnAssign (the annotated style this file already uses 18 times), and
# bare Expr -- an in-place tensor op such as `goal_pos_w.clamp_(...)` is an expression statement,
# not an assignment, and it would silently change the goal. Return/For/While/With are collected for
# the same reason: cheap, and each one is a way to change what runs.
_COMPARED_STATEMENTS = (
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.Expr,
    ast.Return,
    ast.If,
    ast.For,
    ast.While,
    ast.With,
)


def _normalised_statements(path, class_name: str, method_name: str) -> list[tuple[str, str]]:
    """Unparse one method's body to (node-kind, normalised source), delta operand canonicalised.

    The node KIND travels with the text so that rewriting a statement into a different form -- the
    same arithmetic as an AugAssign, say -- is a difference this guard reports rather than one it
    launders into an identical string.
    """
    import ast

    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    out = []
                    for stmt in ast.walk(item):
                        # Docstrings are Expr-of-Constant; they are prose, not behaviour.
                        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                            continue
                        if isinstance(stmt, _COMPARED_STATEMENTS):
                            text = ast.unparse(stmt.test if isinstance(stmt, ast.If) else stmt)
                            for token in _DELTA_TOKENS:
                                text = text.replace(token, "DELTA")
                            out.append((type(stmt).__name__, text))
                    return out
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


_OURS_TAGGED = _normalised_statements(_C3_RUNG_PATH, "C3RungGoalPoseCommand", "_pin_goal_at_object_pose")
_THEIRS_TAGGED = _normalised_statements(_MIXTURE_PATH, "MixtureGoalPoseCommand", "_resample_goal_at_spawn")
# Text-only views, for the per-statement membership tests below.
_OURS = [text for _kind, text in _OURS_TAGGED]
_THEIRS = [text for _kind, text in _THEIRS_TAGGED]


def _canonical(stmt_src: str) -> str:
    """Round-trip an EXPECTED statement through THIS interpreter's own ``ast.unparse``.

    ``ast.unparse`` is not stable across CPython versions. On 3.10 a tuple target unparses as
    ``(pos_b, quat_b) = ...``; on 3.12 it is ``pos_b, quat_b = ...``. Comparing a hand-written
    string against unparsed source therefore encodes the version of whoever wrote the string --
    and it did: this suite passed on 3.12 and failed on 3.10 on exactly that statement, for a
    difference in PARENTHESES with no bearing on the goal expression it exists to guard.

    Sending both sides through the same unparser removes the interpreter from the comparison
    without weakening it: the expected source still has to parse to the same tree.
    """
    return ast.unparse(ast.parse(stmt_src).body[0])


def _assert_in_both(stmt_src: str) -> None:
    """Assert one statement is present in BOTH normalised bodies, interpreter-independently."""
    want = _canonical(stmt_src)
    assert want in _OURS, (want, _OURS)
    assert want in _THEIRS, (want, _THEIRS)


def test_neither_method_contains_a_statement_the_other_does_not():
    """THE ONE THE MEMBERSHIP TESTS CANNOT DO. Every other test in this section asks "is this
    statement present in both". None of them can see a statement ADDED to one side -- an extra
    clamp, an extra offset, a second displacement -- which is precisely the drift this guard exists
    to catch, and which a negative control confirmed passed all 73 tests silently. Compare the two
    normalised bodies as ORDERED SEQUENCES: same statements, same kinds, same traversal order.
    """
    assert _OURS_TAGGED == _THEIRS_TAGGED, (
        "the two goal expressions have diverged:\n"
        f"  only in c3_rung:        {[s for s in _OURS_TAGGED if s not in _THEIRS_TAGGED]}\n"
        f"  only in episode_mixture: {[s for s in _THEIRS_TAGGED if s not in _OURS_TAGGED]}\n"
        f"  ours={_OURS_TAGGED}\n  theirs={_THEIRS_TAGGED}"
    )


def test_both_read_the_objects_pose_the_same_way():
    for stmt in (
        "object_pos_w = self.object.data.root_pos_w[env_ids]",
        "object_quat_w = self.object.data.root_quat_w[env_ids]",
    ):
        _assert_in_both(stmt)


def test_both_guard_the_displacement_on_a_nonzero_delta():
    _assert_in_both("DELTA != 0.0")


def test_both_take_the_bore_axis_from_the_same_shared_helper():
    # Same function, same arguments -- so the two cannot disagree about which way "deeper" points,
    # and both inherit live_bore_deep_axis's runtime axis guard.
    _assert_in_both("axis_world = live_bore_deep_axis(self._fixture, self._fixture_local_deep_axis, env_ids)")


def test_both_displace_the_goal_with_the_identical_expression():
    # THE ONE THAT MATTERS. One expression, both signs, no branch on the sign.
    _assert_in_both("goal_pos_w = object_pos_w + DELTA * axis_world")


def test_both_start_from_the_undisplaced_pose():
    _assert_in_both("goal_pos_w = object_pos_w")


def test_both_convert_to_the_robot_root_frame_identically():
    # Including that the goal QUATERNION is the object's own in both -- neither half of either
    # mechanism commands a reorientation.
    _assert_in_both(
        "pos_b, quat_b = subtract_frame_transforms(self.robot.data.root_pos_w[env_ids],"
        " self.robot.data.root_quat_w[env_ids], goal_pos_w, object_quat_w)"
    )


def test_both_write_the_same_command_slots():
    for stmt in ("self.pose_command_b[env_ids, 0:3] = pos_b", "self.pose_command_b[env_ids, 3:7] = quat_b"):
        _assert_in_both(stmt)


def test_the_numeric_model_matches_the_mixture_on_a_tilted_bore_axis():
    # The lead asked for a tilted axis explicitly. c3_rung_core.s1_goal_position is the pure model
    # of the expression both methods share; check it against an independently written reference on
    # a tilted, non-axis-aligned unit vector and on BOTH signs, so an implementation that only works
    # for world -Z (or only for a positive delta) fails here.
    axis = (0.4, -0.48, -0.78)  # roughly unit, deliberately not axis-aligned
    spawn = (0.30, 0.05, 0.1362)
    for delta in (0.005, -0.060, 0.0):
        expected = tuple(s + delta * a for s, a in zip(spawn, axis))
        assert s1_goal_position(spawn, axis, delta) == expected


# ---------------------------------------------------------------------------------------------
# 8. THE PUBLIC goal_is_final ACCESSOR (team-lead decision, bead dr-ai1.18 follow-on)
# ---------------------------------------------------------------------------------------------
#
# C3RungGoalPoseCommand.goal_is_final is the ONE public read of "has this env's S_t goal been
# re-pinned yet / is its commanded goal trustworthy". It exists because the private latch grew three
# consumers in a day and two of them were reaching into it or recomputing st_should_repin's
# conditions -- two layers computing one condition, the shape that caused the pre-settle capture bug
# twice. These tests bind the accessor to the single latch so the two can never disagree.
#
# STRUCTURAL, via `ast`, for the same reason section 7 is: c3_rung.py imports isaaclab and torch,
# neither of which is importable in this environment, so the property cannot be executed here. What
# CAN be proved without a GPU is that it is derived from the one buffer rather than duplicating it.

_C3_RUNG_SRC = (_MDP_DIR / "c3_rung.py").read_text()
_LATCH = "_st_awaiting_repin"


def _assignment_targets(node) -> list:
    """Targets of ANY assignment form, or [] for a non-assignment node.

    One place, because getting this wrong is how the original writer-set tests missed
    ``ast.AnnAssign`` entirely: an annotated write has ``.target`` (singular) rather than
    ``.targets``, so a walker filtering on ``ast.Assign`` alone never sees it -- and this file
    uses the annotated style 18 times, so it is the LIKELY spelling, not an exotic one.
    """
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        return [node.target]
    return []


def _command_class():
    import ast

    for node in ast.walk(ast.parse(_C3_RUNG_SRC)):
        if isinstance(node, ast.ClassDef) and node.name == "C3RungGoalPoseCommand":
            return node
    raise AssertionError("C3RungGoalPoseCommand not found")


def _method(name):
    import ast

    for item in _command_class().body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    raise AssertionError(f"{name} not found on C3RungGoalPoseCommand")


def test_goal_is_final_is_exactly_the_negation_of_the_single_latch():
    # THE ONE THAT MATTERS: the accessor is a derived view, not a second buffer, so it cannot drift
    # from the latch. If someone reimplements it -- caches it, recomputes the settle conditions,
    # or adds a parallel bool -- the unparsed body stops being this exact expression and this fails.
    import ast

    fn = _method("goal_is_final")
    body = [st for st in fn.body if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant))]
    assert len(body) == 1, f"goal_is_final should be a one-liner over the latch; got {len(body)} statements"
    # Through _canonical for the same reason section 7 is: an exact-unparse comparison against a
    # hand-written string is only as portable as the interpreter that wrote the string.
    assert ast.unparse(body[0]) == _canonical(f"return ~self.{_LATCH}"), ast.unparse(body[0])


def test_goal_is_final_is_a_read_only_property():
    import ast

    fn = _method("goal_is_final")
    decorators = [ast.unparse(d) for d in fn.decorator_list]
    assert decorators == ["property"], decorators
    # No setter: nothing anywhere may define goal_is_final.setter, and nothing may assign to it.
    assert "goal_is_final.setter" not in _C3_RUNG_SRC
    # Every assignment form, because the route that matters here is the annotated one. A CLASS-LEVEL
    # `goal_is_final: bool = False` silently SHADOWS the property -- unlike `self.goal_is_final = x`,
    # which raises AttributeError at runtime and so cannot survive to production. A negative control
    # confirmed the ast.Assign-only version did not see it.
    for node in ast.walk(ast.parse(_C3_RUNG_SRC)):
        for tgt in _assignment_targets(node):
            assert "goal_is_final" not in ast.unparse(tgt), f"goal_is_final assigned: {ast.unparse(node)}"


def test_there_is_exactly_one_latch_buffer():
    # A second stored bool tensor tracking the same thing is the failure mode this whole change
    # removes. The latch must be allocated exactly once, in __init__.
    #
    # Scanned CLASS-WIDE, not just __init__, and over every assignment form. The earlier version
    # walked ast.Assign inside __init__ only, and a negative control confirmed BOTH holes: a second
    # allocation written as `self._st_awaiting_repin: torch.Tensor = torch.zeros(...)` passed all 73
    # tests, and a reallocation in another method was outside the search entirely.
    #
    # A BARE attribute target allocates the buffer. A Subscript target -- the three legitimate
    # element writes, `self._st_awaiting_repin[ids] = ...` -- mutates the existing one and is not an
    # allocation; that distinction is what lets this assert a hard count of 1.
    allocations = []
    for item in _command_class().body:
        if not isinstance(item, ast.FunctionDef):
            continue
        for n in ast.walk(item):
            for tgt in _assignment_targets(n):
                if isinstance(tgt, ast.Attribute) and tgt.attr == _LATCH:
                    allocations.append((item.name, ast.unparse(n)))
    assert len(allocations) == 1, allocations
    assert allocations[0][0] == "__init__", f"the latch is allocated outside __init__: {allocations}"
    assert "torch.zeros" in allocations[0][1], allocations


def test_the_latch_is_written_only_in_the_three_documented_methods():
    # Arming/disarming lives in _resample_command, clearing in _update_command, allocation in
    # __init__. A write appearing anywhere else can desync the accessor from reality without any
    # of the above tests noticing, so pin the set of writers rather than a bare count.
    import ast

    writers = set()
    for item in _command_class().body:
        if not isinstance(item, ast.FunctionDef):
            continue
        for n in ast.walk(item):
            # _assignment_targets covers Assign, AugAssign AND AnnAssign in one place -- see its
            # docstring for why the annotated form is the one that got missed.
            if any(_LATCH in ast.unparse(t) for t in _assignment_targets(n)):
                writers.add(item.name)
    assert writers == {"__init__", "_resample_command", "_update_command"}, writers


def test_the_accessor_docstring_pins_the_s1_semantics():
    # The lead's explicit requirement: a caller who gets S1 backwards silently rejects or accepts an
    # entire rung, so the S1 rule must be stated where the caller reads it, not only in a message.
    import ast

    # Whitespace-normalised: the docstring is wrapped, so "never re-pinned" spans a line break in
    # the source. Checking the raw text would fail on reflow rather than on meaning.
    doc = " ".join((ast.get_docstring(_method("goal_is_final")) or "").split())
    assert "S1" in doc and "S_t" in doc
    assert "never re-pinned" in doc, "the docstring must say S1 is never re-pinned"
    assert "trustworthy" in doc, "the docstring must state what the flag actually means"
    # Both directions of the misreading the lead warned about are named explicitly.
    assert 'Do not read S1\'s ``True`` as "the re-pin has happened"' in doc
    assert 'never read ``False`` as "this env is S1"' in doc


def _reader_methods(attr_name: str) -> set[str]:
    """Methods of the command class that READ ``attr_name``.

    ctx=Load only -- the __init__ assignment TARGET is an Attribute in Store context and is not a
    read of the value.
    """
    readers = set()
    for item in _command_class().body:
        if isinstance(item, ast.FunctionDef):
            for n in ast.walk(item):
                if isinstance(n, ast.Attribute) and n.attr == attr_name and isinstance(n.ctx, ast.Load):
                    readers.add(item.name)
    return readers


def test_the_command_does_not_recompute_the_settle_conditions_anywhere_else():
    # st_should_repin's three conditions are expressed ONCE in the tensor mask inside
    # _update_command. If a second place in this class starts comparing against the settle
    # thresholds, that is the duplication this change exists to prevent.
    #
    # ALL THREE thresholds, not just the linear one. The earlier version tracked
    # _st_settle_speed_mps alone, so a second settle test written against the ANGULAR threshold or
    # the step floor was invisible -- and the angular term is the one added last, so it is the
    # likeliest to sprout a second consumer. A negative control confirmed a helper reading only
    # _st_settle_ang_speed_rad_s passed all 73 tests.
    #
    # __init__ passes each to its validator; _update_command builds the mask; st_settle_thresholds
    # reports them. A bare COUNT was the wrong instrument here -- it cannot distinguish a new
    # accessor from a new recomputation. Pinning the method set can: a read appearing in any OTHER
    # method is a second place deciding "is it settled yet" and fails.
    expected = {"__init__", "_update_command", "st_settle_thresholds"}
    for attr in ("_st_settle_speed_mps", "_st_settle_ang_speed_rad_s", "_st_settle_min_steps"):
        assert _reader_methods(attr) == expected, (attr, _reader_methods(attr))


def test_the_unresolved_cfg_thresholds_are_read_only_in_init():
    # The OTHER way to reach the same numbers, which the private-field scan above cannot see:
    # cfg.st_settle_min_steps may be None, meaning "use held_check_core.SETTLE_STEPS". Reading the
    # cfg outside __init__ is how a second place ends up deciding the step floor -- the exact defect
    # st_settle_thresholds was introduced to remove.
    for attr in ("st_settle_speed_mps", "st_settle_ang_speed_rad_s", "st_settle_min_steps"):
        readers = _reader_methods(attr)
        assert readers <= {"__init__"}, f"cfg.{attr} read outside __init__: {sorted(readers)}"


def test_st_settle_thresholds_returns_the_resolved_values_not_the_cfg():
    # The point of the accessor: min_steps is resolved in __init__ (cfg's value may be None meaning
    # "use held_check_core.SETTLE_STEPS"), so it must come from the private resolved field, not from
    # cfg. If someone "simplifies" this to read cfg.st_settle_min_steps, a caller gets None and
    # re-implements the fallback -- a second place deciding the step floor.
    import ast

    fn = _method("st_settle_thresholds")
    body = [st for st in fn.body if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant))]
    assert len(body) == 1, ast.unparse(fn)
    got = ast.unparse(body[0])
    assert got == _canonical(
        "return (self._st_settle_speed_mps, self._st_settle_ang_speed_rad_s, self._st_settle_min_steps)"
    ), got
    assert "self.cfg" not in got, "must return the RESOLVED fields, never the cfg's unresolved ones"


def test_st_settle_thresholds_is_read_only():
    import ast

    assert [ast.unparse(d) for d in _method("st_settle_thresholds").decorator_list] == ["property"]
    assert "st_settle_thresholds.setter" not in _C3_RUNG_SRC


def test_the_settle_thresholds_are_written_only_in_init():
    # Config, not state: fixed at __init__ and never mutated. A write anywhere else would make the
    # accessor a moving target and would mean the mask and the reported values could disagree.
    import ast

    writers = set()
    for item in _command_class().body:
        if not isinstance(item, ast.FunctionDef):
            continue
        for n in ast.walk(item):
            if any("_st_settle_" in ast.unparse(t) for t in _assignment_targets(n)):
                writers.add(item.name)
    assert writers == {"__init__"}, writers


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
