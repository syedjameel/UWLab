# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Unit-proves analyze_c3_st_measure_only.py's pure math/validation (bead dr-sj6.24) without
touching Isaac, a real generator run, or a real .pt bank produced by one.

Needs only ``torch``/``numpy`` -- no Isaac Sim, no GPU. Loaded BY FILE PATH, same technique
``test_spawn_tolerance_stage.py`` uses next door, so this can run standalone even though the
module under test lives in ``scripts_v2/tools/`` rather than an ``mdp`` package.

Every fixture bank below is a MINIMAL synthetic dict shaped like what ``torch.load`` would return
for a real ``.pt`` (``{"initial_state": {"measure_only": [<tensor>, ...]}}``) -- not a real bank,
since no GPU is available here; the module's own docstring is explicit about this being the
untested seam (its "INPUT FORMAT, PART 2" note).

REQUIRED NEGATIVE CONTROLS (team-lead's own list, verbatim):
  1. a bimodal fixture where the mean falls in the empty gap between the modes
  2. a fixture where the two rotation metrics diverge sharply
  3. one that is not marked measure-only
  4. one below the minimum n
Each must trip EXACTLY its own check -- not a nearby one, not all of them at once.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

_MODULE_PATH = Path(__file__).resolve().parent / "analyze_c3_st_measure_only.py"
_spec = importlib.util.spec_from_file_location("analyze_c3_st_measure_only", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["analyze_c3_st_measure_only"] = _mod
_spec.loader.exec_module(_mod)

RefuseToAnalyze = _mod.RefuseToAnalyze
read_jsonl_log = _mod.read_jsonl_log
attempted_vs_recorded = _mod.attempted_vs_recorded
validate_measure_only_bank = _mod.validate_measure_only_bank
distribution_report = _mod.distribution_report
rotation_disagreement_report = _mod.rotation_disagreement_report
propose_tolerances = _mod.propose_tolerances
format_text_report = _mod.format_text_report


def _measure_only_bank(n: int = 5, all_true: bool = True) -> dict:
    marker = torch.ones(n, dtype=torch.bool) if all_true else torch.zeros(n, dtype=torch.bool)
    return {"initial_state": {"measure_only": [marker]}}


# ---------------------------------------------------------------------------------------------
# NEGATIVE CONTROL 1: bimodal fixture -- mean falls in the empty gap between the modes.
# ---------------------------------------------------------------------------------------------


def test_distribution_report_flags_bimodal_mean_as_uninformative():
    # 90 states tightly at 0.010, 10 states tightly at 0.090 -- deterministic, no randomness.
    values = [0.010] * 90 + [0.090] * 10
    report = distribution_report(values, name="pos_dist_m", min_n=50, percentiles=(50.0, 99.0))

    mean = report["mean"]
    assert math.isclose(mean, 0.018, abs_tol=1e-9), mean
    # The mean sits STRICTLY BETWEEN the two modes, in the empty gap (0.011, 0.089) -- describing
    # neither mode, exactly the "outcome that occurred for no leg" shape team-lead named.
    assert 0.011 < mean < 0.089, "test fixture problem: mean is not actually in the empty gap"
    assert not any(math.isclose(v, mean, abs_tol=1e-6) for v in values), (
        "test fixture problem: some observed value equals the mean -- the gap is not actually empty"
    )
    # The percentiles land ON a real mode, unlike the mean -- p50 squarely in the 90%-weight low
    # mode, p99 squarely in the 10%-weight high mode (both chosen clear of numpy's linear
    # interpolation crossing the mode boundary, which happens near p90 for this exact 90/10 split).
    assert math.isclose(report["percentiles"][50.0], 0.010, abs_tol=1e-9)
    assert math.isclose(report["percentiles"][99.0], 0.090, abs_tol=1e-9)
    # The histogram has an EMPTY bin between the two modes -- the shape a bare mean would hide.
    counts = report["histogram"]["counts"]
    assert any(c == 0 for c in counts), "expected at least one empty histogram bin between the two modes"
    assert sum(counts) == 100


def test_distribution_report_histogram_is_never_collapsed_to_a_single_number():
    report = distribution_report(list(range(1, 301)), name="x", min_n=100, bins=10)
    assert len(report["histogram"]["counts"]) == 10
    assert len(report["histogram"]["edges"]) == 11
    assert "mean" in report and "percentiles" in report and "max" in report


# ---------------------------------------------------------------------------------------------
# NEGATIVE CONTROL 2: two rotation metrics diverge sharply vs. agree closely.
# ---------------------------------------------------------------------------------------------


def test_rotation_disagreement_flags_significant_divergence():
    n = 200
    axis_tilt = [0.02] * n  # small, real tilt error
    # rot_dist_rad = axis_tilt + a LARGE constant spin about the leg's own axis (1.2 rad ~ 69deg)
    # -- axis_tilt_rad is blind to it by construction (spawn_tolerance_core.py's own contract).
    rot_dist = [0.02 + 1.2] * n
    report = rotation_disagreement_report(rot_dist, axis_tilt, min_n=100)
    assert report["n"] == n
    assert math.isclose(report["median_diff_rad"], 1.2, abs_tol=1e-9)
    assert report["significant_divergence"] is True


def test_rotation_disagreement_does_not_flag_close_agreement():
    n = 200
    axis_tilt = [0.02] * n
    rot_dist = [0.021] * n  # 1 millirad difference -- far under any reasonable threshold
    report = rotation_disagreement_report(rot_dist, axis_tilt, min_n=100)
    assert report["significant_divergence"] is False


def test_rotation_disagreement_threshold_is_overridable_and_honoured():
    n = 200
    axis_tilt = [0.0] * n
    rot_dist = [0.05] * n  # ~2.86deg
    loose = rotation_disagreement_report(rot_dist, axis_tilt, min_n=100, significant_threshold_rad=math.radians(10.0))
    tight = rotation_disagreement_report(rot_dist, axis_tilt, min_n=100, significant_threshold_rad=math.radians(1.0))
    assert loose["significant_divergence"] is False
    assert tight["significant_divergence"] is True


def test_rotation_disagreement_rejects_mismatched_lengths():
    import pytest

    with pytest.raises(RefuseToAnalyze):
        rotation_disagreement_report([0.1, 0.2, 0.3], [0.1, 0.2], min_n=1)


# ---------------------------------------------------------------------------------------------
# NEGATIVE CONTROL 3: bank not marked measure-only -- filename, in-file marker, or both.
# ---------------------------------------------------------------------------------------------


def test_validate_measure_only_bank_accepts_a_properly_marked_bank():
    bank = _measure_only_bank(n=5, all_true=True)
    validate_measure_only_bank("/x/resets_ObjectRestingEEGrasped_MEASUREONLY.pt", bank)  # must not raise


def test_validate_measure_only_bank_rejects_missing_filename_marker():
    import pytest

    bank = _measure_only_bank(n=5, all_true=True)
    with pytest.raises(RefuseToAnalyze) as exc_info:
        validate_measure_only_bank("/x/resets_ObjectRestingEEGrasped.pt", bank)
    assert "FILENAME marker missing" in str(exc_info.value)
    assert "IN-FILE marker" not in str(exc_info.value)  # names ONLY the marker that is missing


def test_validate_measure_only_bank_rejects_missing_in_file_marker():
    import pytest

    bank = {"initial_state": {}}  # no 'measure_only' key at all -- a gated/production bank's shape
    with pytest.raises(RefuseToAnalyze) as exc_info:
        validate_measure_only_bank("/x/resets_ObjectRestingEEGrasped_MEASUREONLY.pt", bank)
    assert "IN-FILE marker missing" in str(exc_info.value)
    assert "FILENAME marker" not in str(exc_info.value)  # names ONLY the marker that is missing


def test_validate_measure_only_bank_rejects_both_markers_missing():
    import pytest

    bank = {"initial_state": {}}
    with pytest.raises(RefuseToAnalyze) as exc_info:
        validate_measure_only_bank("/x/resets_ObjectRestingEEGrasped.pt", bank)
    msg = str(exc_info.value)
    assert "FILENAME marker missing" in msg
    assert "IN-FILE marker missing" in msg


def test_validate_measure_only_bank_rejects_mixed_true_false_marker():
    import pytest

    bank = _measure_only_bank(n=5, all_true=False)
    with pytest.raises(RefuseToAnalyze) as exc_info:
        validate_measure_only_bank("/x/resets_ObjectRestingEEGrasped_MEASUREONLY.pt", bank)
    assert "NOT all True" in str(exc_info.value)


# ---------------------------------------------------------------------------------------------
# NEGATIVE CONTROL 4: below the minimum n.
# ---------------------------------------------------------------------------------------------


def test_distribution_report_refuses_below_min_n():
    import pytest

    with pytest.raises(RefuseToAnalyze) as exc_info:
        distribution_report([0.01] * 40, name="pos_dist_m", min_n=200)
    msg = str(exc_info.value)
    assert "n=40" in msg
    assert "min_n=200" in msg


def test_distribution_report_accepts_exactly_at_min_n():
    report = distribution_report([0.01] * 200, name="pos_dist_m", min_n=200)
    assert report["n"] == 200  # boundary: exactly min_n must NOT raise


def test_rotation_disagreement_refuses_below_min_n():
    import pytest

    with pytest.raises(RefuseToAnalyze):
        rotation_disagreement_report([0.1] * 40, [0.05] * 40, min_n=200)


def test_propose_tolerances_refuses_below_min_n():
    import pytest

    with pytest.raises(RefuseToAnalyze):
        propose_tolerances([0.01] * 40, [0.05] * 40, [0.03] * 40, min_n=200)


def test_propose_tolerances_rejects_mismatched_lengths():
    import pytest

    with pytest.raises(RefuseToAnalyze):
        propose_tolerances([0.01, 0.02], [0.05], [0.03, 0.04], min_n=1)


# ---------------------------------------------------------------------------------------------
# attempted_vs_recorded: recorded/attempted counted separately, never folded into a rate alone.
# ---------------------------------------------------------------------------------------------


def test_attempted_vs_recorded_counts_both_explicitly():
    records = [{"success": True}] * 30 + [{"success": False}] * 70
    counts = attempted_vs_recorded(records)
    assert counts["attempted"] == 100
    assert counts["recorded"] == 30
    assert math.isclose(counts["recorded_fraction"], 0.30)


def test_attempted_vs_recorded_zero_states_does_not_divide_by_zero():
    counts = attempted_vs_recorded([])
    assert counts["attempted"] == 0
    assert counts["recorded"] == 0
    assert counts["recorded_fraction"] is None


# ---------------------------------------------------------------------------------------------
# propose_tolerances: marginal fraction matches the percentile; joint fraction is reported too.
# ---------------------------------------------------------------------------------------------


def test_propose_tolerances_marginal_fraction_matches_the_percentile():
    n = 1000
    pos = [float(i) for i in range(n)]  # 0..999, uniform -- p90 tolerance should accept ~90%
    rot = [float(i) for i in range(n)]
    axis = [float(i) for i in range(n)]
    result = propose_tolerances(pos, rot, axis, min_n=100, percentiles=(90.0,))
    frac = result["proposals"]["pos_dist_m"][90.0]["marginal_accept_fraction"]
    assert math.isclose(frac, 0.901, abs_tol=0.01)  # np.percentile interpolation, <=  -> ~90.1%


def test_propose_tolerances_joint_fraction_present_for_both_rotation_metrics():
    n = 300
    pos = [0.01] * n
    rot = [0.05] * n
    axis = [0.03] * n
    result = propose_tolerances(pos, rot, axis, min_n=100, percentiles=(90.0, 99.0))
    for p in (90.0, 99.0):
        j = result["joint"][p]
        assert "pos_and_full_quat_accept_fraction" in j
        assert "pos_and_axis_tilt_accept_fraction" in j
        # constant arrays -> every candidate tolerance is >= every value -> 100% joint acceptance
        assert math.isclose(j["pos_and_full_quat_accept_fraction"], 1.0)
        assert math.isclose(j["pos_and_axis_tilt_accept_fraction"], 1.0)


# ---------------------------------------------------------------------------------------------
# read_jsonl_log: parses valid lines, refuses (not silently skips) a malformed one.
# ---------------------------------------------------------------------------------------------


def test_read_jsonl_log_parses_valid_lines(tmp_path):
    log_path = tmp_path / "log.jsonl"
    log_path.write_text(
        '{"success": true, "pos_dist_m": 0.01, "rot_dist_rad": 0.02, "axis_tilt_rad": 0.015}\n'
        "\n"  # blank line, must be skipped
        '{"success": false, "pos_dist_m": 0.5, "rot_dist_rad": 0.9, "axis_tilt_rad": 0.8}\n'
    )
    records = read_jsonl_log(str(log_path))
    assert len(records) == 2
    assert records[0]["success"] is True
    assert records[1]["success"] is False


def test_read_jsonl_log_refuses_malformed_line_with_its_line_number(tmp_path):
    import pytest

    log_path = tmp_path / "log.jsonl"
    log_path.write_text('{"success": true}\n' "not json at all\n")
    with pytest.raises(RefuseToAnalyze) as exc_info:
        read_jsonl_log(str(log_path))
    assert ":2:" in str(exc_info.value)  # names the OFFENDING line, not just "malformed somewhere"


# ---------------------------------------------------------------------------------------------
# format_text_report: assembles without raising, and surfaces the divergence banner when present.
# ---------------------------------------------------------------------------------------------


def test_format_text_report_includes_significant_divergence_banner():
    n = 200
    pos = [0.01] * n
    rot = [1.0] * n
    axis = [0.02] * n
    counts = {"attempted": 250, "recorded": n, "recorded_fraction": n / 250}
    pos_r = distribution_report(pos, name="pos_dist_m", min_n=100)
    rot_r = distribution_report(rot, name="rot_dist_rad", min_n=100)
    axis_r = distribution_report(axis, name="axis_tilt_rad", min_n=100)
    disagreement = rotation_disagreement_report(rot, axis, min_n=100)
    proposal = propose_tolerances(pos, rot, axis, min_n=100)
    text = format_text_report(counts, pos_r, rot_r, axis_r, disagreement, proposal)
    assert "SIGNIFICANT DIVERGENCE" in text
    assert "PROPOSAL ONLY" in text
    assert "attempted=250" in text and "recorded=200" in text


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            import inspect

            sig = inspect.signature(fn)
            if "tmp_path" in sig.parameters:
                import shutil
                import tempfile

                tmp_dir = tempfile.mkdtemp(prefix="analyze_c3_st_measure_only_test_")
                try:
                    fn(Path(tmp_dir))
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                fn()
            print(f"[analyze_c3_st_measure_only] {name} OK", flush=True)
    print("[analyze_c3_st_measure_only] all tests passed", flush=True)
