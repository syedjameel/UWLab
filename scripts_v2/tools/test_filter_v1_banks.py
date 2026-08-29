"""Tests for filter_v1_banks.py. Builds tiny fake .pt banks with torch.save (never touches a real
bank -- this suite is offline/fixture-only) and checks: reset_type parsing from filename, the
rigid_object key-naming fallback (insertive_object vs. raw object), the per-criterion counts for
both bank shapes, the "no criterion in scope" path, and negative controls for each failure mode
this script is supposed to catch (missing key, unreadable bank, unrecognised filename). Also
checks the upper-bound banner discipline directly, since that wording is a project requirement,
not an implementation detail.

Run with: python3 test_filter_v1_banks.py   (bare __main__ runner, no pytest needed)
      or:  <a venv with torch> -m pytest test_filter_v1_banks.py -q
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("filter_v1_banks", os.path.join(_HERE, "filter_v1_banks.py"))
fvb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fvb)


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------

def _pose_list(n, pos, quat_wxyz):
    """n identical (7,) root_pose tensors -- enough for these tests, which check counts/branching,
    not per-state geometric correctness (that's v1_bank_geometry_core's own test suite's job)."""
    row = torch.tensor(list(pos) + list(quat_wxyz), dtype=torch.float32)
    return [row.clone() for _ in range(n)]


def _vel_list(n, lin_speed):
    row = torch.tensor([lin_speed, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    return [row.clone() for _ in range(n)]


def _write_bank(path, *, leg_key="insertive_object", include_fixture=True, n=4, leg_speed=0.0):
    rigid = {
        leg_key: {
            "root_pose": _pose_list(n, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            "root_velocity": _vel_list(n, leg_speed),
        }
    }
    if include_fixture:
        rigid["receptive_object"] = {
            "root_pose": _pose_list(n, (0.5, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            "root_velocity": _vel_list(n, 0.0),
        }
    torch.save({"initial_state": {"articulation": {"robot": {}}, "rigid_object": rigid}}, path)


# ---------------------------------------------------------------------------
# reset_type_from_filename
# ---------------------------------------------------------------------------

def test_reset_type_from_filename_recognises_partially_assembled():
    got = fvb.reset_type_from_filename("/x/y/resets_ObjectPartiallyAssembledEEGrasped.pt")
    assert got == "ObjectPartiallyAssembledEEGrasped", got


def test_reset_type_from_filename_strips_rewind_suffix():
    got = fvb.reset_type_from_filename("/x/y/resets_ObjectRestingEEGrasped_off0.10s.pt")
    assert got == "ObjectRestingEEGrasped", got


def test_reset_type_from_filename_strips_dotted_suffix():
    got = fvb.reset_type_from_filename("/x/y/resets_ObjectRestingEEGrasped.clean949.pt")
    assert got == "ObjectRestingEEGrasped", got


def test_reset_type_from_filename_returns_unknown_for_non_matching_name():
    # negative control: not a resets_*.pt filename at all
    got = fvb.reset_type_from_filename("/x/y/backup_dump.pt")
    assert got == "UNKNOWN", got


# ---------------------------------------------------------------------------
# load_bank_states
# ---------------------------------------------------------------------------

def test_load_bank_states_reads_insertive_object_naming():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "resets_ObjectPartiallyAssembledEEGrasped.pt")
        _write_bank(p, leg_key="insertive_object", n=5)
        states = fvb.load_bank_states(p)
        assert states["n"] == 5
        assert states["leg_pos"].shape == (5, 3)
        assert "fix_pos" in states


def test_load_bank_states_falls_back_to_bare_object_naming():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "resets_ObjectRestingEEGrasped.pt")
        _write_bank(p, leg_key="object", include_fixture=False, n=3)
        states = fvb.load_bank_states(p)
        assert states["n"] == 3
        assert "fix_pos" not in states


def test_load_bank_states_computes_linear_speed_from_root_velocity():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "resets_ObjectRestingEEGrasped.pt")
        _write_bank(p, leg_key="insertive_object", include_fixture=False, n=2, leg_speed=0.3)
        states = fvb.load_bank_states(p)
        assert np.allclose(states["leg_lin_speed_mps"], 0.3), states["leg_lin_speed_mps"]


def test_load_bank_states_raises_on_missing_leg_key():
    # negative control: rigid_object present but neither insertive_object nor object exists
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "resets_ObjectRestingEEGrasped.pt")
        torch.save({"initial_state": {"rigid_object": {"receptive_object": {"root_pose": [], "root_velocity": []}}}}, p)
        try:
            fvb.load_bank_states(p)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "insertive_object" in str(e)


def test_load_bank_states_raises_on_missing_initial_state_key():
    # negative control: not shaped like a reset bank at all
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "resets_ObjectRestingEEGrasped.pt")
        torch.save({"something_else": {}}, p)
        try:
            fvb.load_bank_states(p)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "initial_state" in str(e)


# ---------------------------------------------------------------------------
# report_bank
# ---------------------------------------------------------------------------

def test_report_bank_applies_c4_and_c3s1_curve_for_partially_assembled():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "OneLegInsertionFixture__SquareTableLeg200mmDecomp", "resets_ObjectPartiallyAssembledEEGrasped.pt")
        os.makedirs(os.path.dirname(p))
        _write_bank(p, n=6)
        text = fvb.report_bank(p)
        assert "C4 band" in text
        assert "ALL THREE (joint)" in text
        assert "C3(S1) band, OPEN" in text
        assert "scale=1.00" in text
        assert "leg asset (provenance): SquareTableLeg200mmDecomp" in text


def test_report_bank_applies_c2_speed_for_resting():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "resets_ObjectRestingEEGrasped.pt")
        _write_bank(p, include_fixture=False, n=4, leg_speed=0.01)
        text = fvb.report_bank(p)
        assert "C2 resting speed" in text
        assert "pass: 4/4" in text  # 0.01 m/s is well under the 0.05 default


def test_report_bank_notes_no_criterion_for_unrecognised_reset_type():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "resets_ObjectAnywhereEEAnywhere.pt")
        _write_bank(p, include_fixture=False, n=2)
        text = fvb.report_bank(p)
        assert "no offline v2 criterion in this pass's scope" in text
        assert "ObjectAnywhereEEAnywhere" in text


def test_report_bank_reports_read_error_without_raising():
    # negative control: report_bank must not raise on an unreadable/malformed bank -- it must
    # report the failure inline so one bad bank does not kill a whole batch run.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "resets_ObjectRestingEEGrasped.pt")
        with open(p, "wb") as f:
            f.write(b"not a torch file")
        text = fvb.report_bank(p)  # must not raise
        assert "COULD NOT READ" in text


def test_report_bank_provenance_is_unknown_when_path_has_no_leg_asset_component():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "some_flat_dir", "resets_ObjectRestingEEGrasped.pt")
        os.makedirs(os.path.dirname(p))
        _write_bank(p, include_fixture=False, n=1)
        text = fvb.report_bank(p)
        assert "leg asset (provenance): UNKNOWN" in text


def test_report_bank_always_includes_upper_bound_banner():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "resets_ObjectRestingEEGrasped.pt")
        _write_bank(p, include_fixture=False, n=1)
        text = fvb.report_bank(p)
        assert fvb.UPPER_BOUND_BANNER in text


def test_report_bank_never_labels_a_count_reusable():
    # negative control on the wording discipline itself: "reusable" must never appear unqualified
    # as a label for a count -- only inside the banner's own "NOT 'reusable'" phrasing, and inside
    # the explicit "<- upper bound for X reuse" annotations, both of which are checked here rather
    # than allowing a bare "reusable" to slip in unqualified.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "resets_ObjectRestingEEGrasped.pt")
        _write_bank(p, include_fixture=False, n=1)
        text = fvb.report_bank(p)
        for line in text.splitlines():
            if "reusable" in line:
                assert "NOT 'reusable'" in line, f"unqualified 'reusable' in report line: {line!r}"


def test_report_bank_survival_curve_fraction_decreases_with_smaller_scale():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "resets_ObjectPartiallyAssembledEEGrasped.pt")
        _write_bank(p, n=8)
        text = fvb.report_bank(p, scale_factors=(0.1, 5.0))
        lines = [l for l in text.splitlines() if l.strip().startswith("scale=")]
        assert len(lines) == 2, lines
        # sanity: both rows must surface (monotonicity itself is v1_bank_geometry_core's own
        # test suite's job) -- this just checks the wiring passes scale_factors through.
        assert "scale=0.10" in lines[0] and "scale=5.00" in lines[1]


if __name__ == "__main__":
    failures = []
    tests = [(name, obj) for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    for name, fn in tests:
        try:
            fn()
            print(f"OK   {name}")
        except Exception as e:  # noqa: BLE001
            failures.append((name, e))
            print(f"FAIL {name}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        sys.exit(1)
