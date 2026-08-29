#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Analyser for a C3(S_t) measure-only run (bead dr-sj6.24, team-lead instruction 2026-08-29) --
turns the grasp-induced displacement a ``--c3_st_tolerance_measure_only`` run produces into a
PROPOSED tolerance, at a stated percentile, with its own ``n``. This script never DECIDES: the
proposal is printed, never written back into ``V2_ACCEPTANCE_CRITERIA.md`` -- a person adopts a
number, this script only computes what there is to adopt.

WHY A DISTRIBUTION, NEVER A BARE MEAN. This campaign has already shipped one invented constant
derived from a mean over a BIMODAL distribution that described an outcome that occurred for no leg
(``RESET_SPEC_V2.md`` sec 6 item 0, the withdrawn ``stays_seated`` 6.02%->43.19% pair). Every report
this module produces is a full histogram plus p50/p90/p95/p99/max, never a headline mean --
``test_distribution_report_flags_bimodal_mean_as_uninformative`` below is the negative control that
would have caught that exact class of number.

INPUT FORMAT, PART 1 -- THE BANK (``.pt``), already produced by
``generate_reset_states_policy.py``'s ``--c3_st_tolerance_measure_only`` +
``_MeasureOnlyBankRecorder`` (commit ``92ea54b``). Used HERE only to CONFIRM the run was actually
measure-only, via the two markers that class writes:

  1. FILENAME: ``resets_<reset_type>_MEASUREONLY.pt``.
  2. IN-FILE: ``bank["initial_state"]["measure_only"] == True`` on every recorded state.

:func:`validate_measure_only_bank` REFUSES (:class:`RefuseToAnalyze`) unless BOTH markers are
present, and says explicitly which is missing if either is (team-lead requirement 4). Deriving a
tolerance from a GATED bank would be circular: the gate would have already removed the tail this
analysis exists to measure.

INPUT FORMAT, PART 2 -- THE LOG. *** THIS IS A KNOWN GAP, FLAGGED HERE RATHER THAN HIDDEN. ***
``generate_reset_states_policy.py``'s own ``gate_breakdown()`` ALREADY computes
``spawn_pos_dist_m``/``spawn_rot_dist_rad``/``spawn_axis_tilt_rad`` every step
(``_SpawnPoseToleranceAddon.check()``) -- but nothing in that script currently PERSISTS them
anywhere: the main loop reads ``gate_breakdown()`` only for the BOOLEAN gates in ``gate_names``
(the first-failing-gate/reach-count reduction), never for these three FLOAT diagnostic fields,
which are overwritten every step and never reach disk. Neither the ``.pt`` bank
(``record_pre_reset_states`` only captures scene state, not termination-manager diagnostics) nor
any existing print line carries per-episode displacement. **This script therefore consumes a NEW
log format that has no producer yet** -- a small, scoped, ``--c3_st_tolerance_measure_only``-only
patch to ``generate_reset_states_policy.py`` (append one JSON line per ``dones.any()`` episode) is
the minimal fix, proposed but NOT applied here, because that file was explicitly declared
off-limits for this task ("clean and committed... I would rather it stay that way until it has
actually run"). Every function below is fully testable TODAY against synthetic log data (see
``test_analyze_c3_st_measure_only.py``); only the end-to-end CLI needs a real log to exist.

Expected log: JSON LINES, one object per DONE episode, with (at least) these fields:

  ``success``        bool  -- ``held_with_probe``'s own decision. In measure-only mode this IS the
                              recorder-export decision (the spawn-tolerance addon gates nothing).
  ``pos_dist_m``      float -- ``_SpawnPoseToleranceAddon.last_pos_dist_m`` at the step this
                              episode ended.
  ``rot_dist_rad``    float -- ...\\ ``last_rot_dist_rad`` (full quaternion angle).
  ``axis_tilt_rad``   float -- ...\\ ``last_axis_tilt_rad`` (spin-invariant about the leg's own
                              long axis).

Extra keys are ignored, not rejected -- this format is meant to be a strict subset of whatever a
producer's own per-episode diagnostic line already carries.

Isaac-free: stdlib (``json``/``argparse``/``math``/``os``) + ``numpy`` for every statistic, and
``torch`` ONLY to ``torch.load`` the ``.pt`` bank -- the same and only reason every
``validate_c4_bank*.py`` script in this directory also needs ``torch``: there is no way to read a
torch-serialized file without it. Runnable with
``/home/dom-iva/.cache/simdist-cpu-venv/bin/python3``, no GPU, no Isaac Sim.
"""

from __future__ import annotations

import argparse
import importlib.util as _importlib_util
import json
import math
import os

import numpy as np
import torch

# -- c3_st_measure_only_log_schema.py: loaded BY FILE PATH (same technique this project's test
# suites use for a same-directory sibling module) rather than a bare `import
# c3_st_measure_only_log_schema`, so this still works regardless of whether this file is run
# directly, imported as a module, or loaded by file path itself (as the test suite does). Field-name
# constants ONLY -- this is the ONE place both this consumer and generate_reset_states_policy.py's
# producer get them from (bead dr-sj6.24, team-lead instruction 2026-08-29: "a second literal list
# is not" a fix).
_schema_spec = _importlib_util.spec_from_file_location(
    "c3_st_measure_only_log_schema", os.path.join(os.path.dirname(os.path.abspath(__file__)), "c3_st_measure_only_log_schema.py")
)
c3_st_measure_only_log_schema = _importlib_util.module_from_spec(_schema_spec)
_schema_spec.loader.exec_module(c3_st_measure_only_log_schema)

DEFAULT_MIN_N = 200
"""NOT derived from a formal power calculation -- a round number chosen so a p99 estimate has a
few points at/above the tail with some margin (n=200 -> ~2 points at/above p99) while a p50
estimate is comfortably oversampled at the same n. Override via --min_n if a different bound is
wanted; this default is a stated rule of thumb, not a citation, and is printed on every refusal so
it is never mistaken for one."""

DEFAULT_CANDIDATE_PERCENTILES: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0)
DEFAULT_HISTOGRAM_BINS = 20
DEFAULT_SIGNIFICANT_DIVERGENCE_THRESHOLD_RAD = math.radians(5.0)
"""Default threshold for "the two rotation metrics disagree enough that the choice matters"
(team-lead requirement 2). A round, defensible default in the same single-digit-degree range this
campaign already uses for angular tolerances (v1's C4 tilt bands: 15/25deg; F50/F51's own settled
angular-speed ceiling, 0.05 rad/s ~= 2.9deg/s) -- NOT derived from this metric itself, and
overridable via --significant_threshold_deg."""

__all__ = [
    "DEFAULT_CANDIDATE_PERCENTILES",
    "DEFAULT_HISTOGRAM_BINS",
    "DEFAULT_MIN_N",
    "DEFAULT_SIGNIFICANT_DIVERGENCE_THRESHOLD_RAD",
    "RefuseToAnalyze",
    "attempted_vs_recorded",
    "distribution_report",
    "format_text_report",
    "propose_tolerances",
    "read_jsonl_log",
    "rotation_disagreement_report",
    "validate_measure_only_bank",
]


class RefuseToAnalyze(Exception):
    """Raised whenever this script declines to produce a distribution or a proposal. A refusal is
    a FEATURE, not a bug to work around (bead dr-sj6.24: a percentile from too few states, or one
    derived from a gated bank, is worse than no number at all) -- never caught and silently
    downgraded to a smaller/looser check by any code in this module."""


def read_jsonl_log(path: str) -> list[dict]:
    """Parse a JSON-lines log (see this module's own docstring, "INPUT FORMAT, PART 2") into a
    list of dicts, one per line. Blank lines are skipped; a malformed JSON line, or a well-formed
    one that fails :func:`c3_st_measure_only_log_schema.validate_log_record` (missing/wrong-typed
    field), raises :class:`RefuseToAnalyze` naming its own line number rather than silently
    dropping it -- a silently-shorter distribution is exactly the "looks like a tight distribution,
    is actually a low count" failure this whole script exists to prevent (team-lead's own framing,
    requirement 6: "an addon that records nothing and an addon that records zeros look identical in
    a summary line"). SCHEMA VALIDATION (team-lead requirement, 2026-08-29): a producer that
    silently emits four fields where five are expected must not read as a distribution with one
    metric quietly missing -- it must refuse, here, at parse time, not fail confusingly three
    functions later when a KeyError names a symptom instead of the cause. Extra/unrecognised keys
    are NOT an error (see validate_log_record's own docstring) -- only MISSING or wrong-typed
    required fields are.
    """
    records: list[dict] = []
    with open(path) as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise RefuseToAnalyze(f"{path}:{lineno}: malformed JSON line -- {e}") from e
            try:
                c3_st_measure_only_log_schema.validate_log_record(record, lineno=lineno)
            except c3_st_measure_only_log_schema.LogSchemaError as e:
                raise RefuseToAnalyze(f"{path}:{lineno}: {e}") from e
            records.append(record)
    return records


def attempted_vs_recorded(records: list[dict]) -> dict:
    """Team-lead requirement 6: how many states were RECORDED (``success`` True -- these are the
    ones ``generate_reset_states_policy.py``'s own recorder actually exports) versus how many the
    run ATTEMPTED (every logged episode, accepted or not). Reported as an explicit COUNT pair, not
    folded into a rate, so a low count is visibly a low count rather than a plausible-looking
    fraction."""
    field = c3_st_measure_only_log_schema.FIELD_SUCCESS
    n_attempted = len(records)
    n_recorded = sum(1 for r in records if r.get(field))
    return {
        "attempted": n_attempted,
        "recorded": n_recorded,
        "recorded_fraction": (n_recorded / n_attempted) if n_attempted > 0 else None,
    }


def validate_measure_only_bank(bank_path: str, bank: dict) -> None:
    """Refuse (:class:`RefuseToAnalyze`) unless BOTH markers ``_MeasureOnlyBankRecorder`` writes
    are present (team-lead requirement 4): the ``_MEASUREONLY`` filename suffix, AND
    ``bank["initial_state"]["measure_only"] == True`` on every recorded state. Names explicitly
    which marker is missing if either is -- deriving a tolerance from a GATED bank would be
    circular (the gate would already have removed the tail this analysis exists to measure).
    """
    problems: list[str] = []
    basename = os.path.basename(bank_path)
    if "_MEASUREONLY" not in basename:
        problems.append(
            f"FILENAME marker missing: {basename!r} has no '_MEASUREONLY' substring. "
            "generate_reset_states_policy.py's --c3_st_tolerance_measure_only names its output "
            "resets_<reset_type>_MEASUREONLY.pt -- this does not look like that output."
        )
    initial_state = bank.get("initial_state", {})
    marker = initial_state.get("measure_only")
    if marker is None:
        problems.append(
            "IN-FILE marker missing: bank['initial_state'] has no 'measure_only' key. "
            "_MeasureOnlyBankRecorder stamps initial_state.measure_only=True on every recorded "
            "state -- this bank was not recorded with --c3_st_tolerance_measure_only (or predates "
            "that recorder)."
        )
    elif len(marker) == 0:
        problems.append(
            "IN-FILE marker present but EMPTY: bank['initial_state']['measure_only'] has zero "
            "entries -- an empty bank proves nothing about how it was recorded."
        )
    else:
        all_true = all(bool(torch.as_tensor(chunk).all()) for chunk in marker)
        if not all_true:
            problems.append(
                "IN-FILE marker present but NOT all True: some recorded states have "
                "initial_state.measure_only=False. A single run should never mix gated and "
                "measure-only states -- check this bank's provenance before trusting it."
            )
    if problems:
        raise RefuseToAnalyze(
            "REFUSING to analyze -- this bank is not confirmed measure-only:\n  - " + "\n  - ".join(problems)
        )


def distribution_report(
    values,
    *,
    name: str,
    min_n: int = DEFAULT_MIN_N,
    percentiles: tuple[float, ...] = DEFAULT_CANDIDATE_PERCENTILES,
    bins: int = DEFAULT_HISTOGRAM_BINS,
) -> dict:
    """Full distribution for ONE metric -- histogram + percentiles + max, NEVER a bare mean as the
    thing to trust (see this module's own docstring, "WHY A DISTRIBUTION, NEVER A BARE MEAN").
    Refuses (:class:`RefuseToAnalyze`) if ``n < min_n`` -- a percentile from too few states is not
    a percentile, and ``min_n`` is printed in the refusal message so the threshold is never a
    mystery.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    n = int(arr.size)
    if n < min_n:
        raise RefuseToAnalyze(
            f"{name}: REFUSING to report a distribution from n={n} states (< --min_n={min_n}). "
            "A percentile from too few states is not a percentile -- collect more measure-only "
            "states before deriving anything from this metric."
        )
    counts, edges = np.histogram(arr, bins=bins)
    pct = {p: float(np.percentile(arr, p)) for p in percentiles}
    return {
        "name": name,
        "n": n,
        "mean": float(arr.mean()),  # reported for completeness -- NEVER the headline number
        "percentiles": pct,
        "max": float(arr.max()),
        "histogram": {"edges": edges.tolist(), "counts": counts.tolist()},
    }


def rotation_disagreement_report(
    rot_dist_rad,
    axis_tilt_rad,
    *,
    min_n: int = DEFAULT_MIN_N,
    significant_threshold_rad: float = DEFAULT_SIGNIFICANT_DIVERGENCE_THRESHOLD_RAD,
) -> dict:
    """Per-state ``rot_dist_rad - axis_tilt_rad``: how much the two recorded rotation metrics
    disagree (team-lead requirement 2). Since ``axis_tilt_rad`` is spin-invariant about the leg's
    own long axis, this difference IS (to first order) that axial spin's contribution to the
    full-quaternion angle -- the exact evidence bead dr-sj6.24 is meant to choose a metric from. If
    ``median_diff_rad`` stays under ``significant_threshold_rad``, the two metrics agree closely
    and the choice does not matter; if it exceeds it, the divergence IS the finding, and its size
    is how many good S_t states a full-quaternion gate would reject for axial spin that physically
    does not matter (V2_C3_DESIGN.md sec 7).
    """
    rot = np.asarray(list(rot_dist_rad), dtype=np.float64)
    axis = np.asarray(list(axis_tilt_rad), dtype=np.float64)
    if rot.shape != axis.shape:
        raise RefuseToAnalyze(
            f"rotation_disagreement_report: rot_dist_rad ({rot.shape}) and axis_tilt_rad "
            f"({axis.shape}) must be the SAME per-state arrays -- one value of each per episode."
        )
    n = int(rot.size)
    if n < min_n:
        raise RefuseToAnalyze(f"rotation disagreement: REFUSING to report from n={n} states (< --min_n={min_n}).")
    diff = rot - axis
    median_diff = float(np.median(diff))
    return {
        "n": n,
        "mean_diff_rad": float(diff.mean()),
        "median_diff_rad": median_diff,
        "p90_diff_rad": float(np.percentile(diff, 90)),
        "max_diff_rad": float(diff.max()),
        # min_diff_rad is reported, NOT enforced >= 0 -- axis_tilt_rad is a lower bound on
        # rot_dist_rad as a matter of rotation geometry (the minimal single-axis rotation mapping
        # one vector to another has the smallest angle of any rotation achieving that mapping), but
        # this function reports empirical data, it does not assert physics on it.
        "min_diff_rad": float(diff.min()),
        "significant_threshold_rad": significant_threshold_rad,
        "significant_divergence": median_diff > significant_threshold_rad,
    }


def propose_tolerances(
    pos_dist_m,
    rot_dist_rad,
    axis_tilt_rad,
    *,
    min_n: int = DEFAULT_MIN_N,
    percentiles: tuple[float, ...] = DEFAULT_CANDIDATE_PERCENTILES,
) -> dict:
    """PROPOSE, never decide (team-lead requirement 3). For each of the three metrics and each
    candidate percentile: the tolerance value AT that percentile, plus the MARGINAL accept
    fraction it implies (equal to the percentile itself by construction -- reported anyway so a
    reader sees the number, not just the label). Also reports the JOINT accept fraction of gating
    on position AND each rotation metric together at MATCHING percentiles -- the number that
    actually matters for a production run's yield, since ``P(pos<=p AND rot<=p) != p`` in general;
    it depends on how correlated the two displacements are.
    """
    pos = np.asarray(list(pos_dist_m), dtype=np.float64)
    rot = np.asarray(list(rot_dist_rad), dtype=np.float64)
    axis = np.asarray(list(axis_tilt_rad), dtype=np.float64)
    n = int(pos.size)
    if not (rot.size == n and axis.size == n):
        raise RefuseToAnalyze(
            "propose_tolerances: pos_dist_m/rot_dist_rad/axis_tilt_rad must be the SAME length "
            f"(one triple per episode); got {pos.size}, {rot.size}, {axis.size}."
        )
    if n < min_n:
        raise RefuseToAnalyze(f"propose_tolerances: REFUSING to propose from n={n} states (< --min_n={min_n}).")

    def _per_metric(arr: np.ndarray) -> dict:
        out = {}
        for p in percentiles:
            tol = float(np.percentile(arr, p))
            out[p] = {"tolerance": tol, "marginal_accept_fraction": float((arr <= tol).mean())}
        return out

    proposals = {
        "pos_dist_m": _per_metric(pos),
        "rot_dist_rad": _per_metric(rot),
        "axis_tilt_rad": _per_metric(axis),
    }
    joint = {}
    for p in percentiles:
        pos_tol = proposals["pos_dist_m"][p]["tolerance"]
        rot_tol = proposals["rot_dist_rad"][p]["tolerance"]
        axis_tol = proposals["axis_tilt_rad"][p]["tolerance"]
        joint[p] = {
            "pos_and_full_quat_accept_fraction": float(((pos <= pos_tol) & (rot <= rot_tol)).mean()),
            "pos_and_axis_tilt_accept_fraction": float(((pos <= pos_tol) & (axis <= axis_tol)).mean()),
        }
    return {"n": n, "candidate_percentiles": list(percentiles), "proposals": proposals, "joint": joint}


def _format_distribution(report: dict) -> str:
    lines = [
        f"  {report['name']}  n={report['n']}  mean={report['mean']:.6f} (context only, NOT the number to use)"
        f"  max={report['max']:.6f}",
        "  percentiles: " + "  ".join(f"p{p:g}={v:.6f}" for p, v in sorted(report["percentiles"].items())),
    ]
    edges = report["histogram"]["edges"]
    counts = report["histogram"]["counts"]
    lines.append("  histogram:")
    total = sum(counts) or 1
    for i, c in enumerate(counts):
        bar = "#" * max(1, round(40 * c / max(counts))) if c > 0 else ""
        lines.append(f"    [{edges[i]:.5f}, {edges[i + 1]:.5f}) {c:6d} ({c / total:5.1%})  {bar}")
    return "\n".join(lines)


def format_text_report(
    counts: dict,
    pos_report: dict,
    rot_report: dict,
    axis_report: dict,
    disagreement: dict,
    proposal: dict,
) -> str:
    """Assemble the full human-readable report from the pure-function outputs above. A separate
    function from ``main()`` so tests can assert on its structure without capturing stdout."""
    out: list[str] = []
    out.append("=== C3(S_t) MEASURE-ONLY ANALYSIS (bead dr-sj6.24) ===")
    out.append("PROPOSAL ONLY -- no number here is adopted until a person writes it into V2_ACCEPTANCE_CRITERIA.md.")
    out.append("")
    out.append(
        f"attempted={counts['attempted']}  recorded={counts['recorded']}  "
        f"recorded_fraction={counts['recorded_fraction']:.2%}"
        if counts["recorded_fraction"] is not None
        else f"attempted={counts['attempted']}  recorded={counts['recorded']}"
    )
    out.append("")
    out.append("--- position displacement (pos_dist_m) ---")
    out.append(_format_distribution(pos_report))
    out.append("")
    out.append("--- rotation, full quaternion angle (rot_dist_rad) ---")
    out.append(_format_distribution(rot_report))
    out.append("")
    out.append("--- rotation, axis-only tilt (axis_tilt_rad) ---")
    out.append(_format_distribution(axis_report))
    out.append("")
    out.append("--- rotation metric disagreement (rot_dist_rad - axis_tilt_rad) ---")
    out.append(
        f"  n={disagreement['n']}  mean={disagreement['mean_diff_rad']:.6f}rad "
        f"({math.degrees(disagreement['mean_diff_rad']):.2f}deg)  "
        f"median={disagreement['median_diff_rad']:.6f}rad "
        f"({math.degrees(disagreement['median_diff_rad']):.2f}deg)  "
        f"p90={disagreement['p90_diff_rad']:.6f}rad  max={disagreement['max_diff_rad']:.6f}rad  "
        f"min={disagreement['min_diff_rad']:.6f}rad"
    )
    if disagreement["significant_divergence"]:
        out.append(
            f"  *** SIGNIFICANT DIVERGENCE *** median disagreement "
            f"({math.degrees(disagreement['median_diff_rad']):.2f}deg) exceeds the "
            f"{math.degrees(disagreement['significant_threshold_rad']):.2f}deg threshold -- the "
            "choice of rotation metric MATTERS. A full-quaternion gate would reject states an "
            "axis-only gate would accept, purely for axial spin about the leg's own long axis."
        )
    else:
        out.append(
            f"  Median disagreement stays under the "
            f"{math.degrees(disagreement['significant_threshold_rad']):.2f}deg threshold -- the two "
            "metrics agree closely here; the choice of rotation metric likely does not matter much."
        )
    out.append("")
    out.append(f"--- proposed tolerances (n={proposal['n']}) -- PROPOSALS, not decisions ---")
    for metric in ("pos_dist_m", "rot_dist_rad", "axis_tilt_rad"):
        out.append(f"  {metric}:")
        for p in proposal["candidate_percentiles"]:
            entry = proposal["proposals"][metric][p]
            out.append(
                f"    p{p:g}: tolerance={entry['tolerance']:.6f}  "
                f"marginal_accept_fraction={entry['marginal_accept_fraction']:.2%}"
            )
    out.append("  joint accept fraction (pos AND rotation together, matching percentiles):")
    for p in proposal["candidate_percentiles"]:
        j = proposal["joint"][p]
        out.append(
            f"    p{p:g}: pos+full_quat={j['pos_and_full_quat_accept_fraction']:.2%}  "
            f"pos+axis_tilt={j['pos_and_axis_tilt_accept_fraction']:.2%}"
        )
    return "\n".join(out)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank_path", type=str, required=True, help="Path to the measure-only .pt bank.")
    parser.add_argument(
        "--log_path", type=str, required=True,
        help="Path to the JSON-lines per-episode displacement log (see this module's own "
        "docstring, 'INPUT FORMAT, PART 2' -- no producer for this log exists yet).",
    )
    parser.add_argument("--min_n", type=int, default=DEFAULT_MIN_N, help=f"Minimum n to report anything (default {DEFAULT_MIN_N}).")
    parser.add_argument(
        "--percentiles", type=str, default=",".join(str(p) for p in DEFAULT_CANDIDATE_PERCENTILES),
        help="Comma-separated candidate percentiles.",
    )
    parser.add_argument("--bins", type=int, default=DEFAULT_HISTOGRAM_BINS, help="Histogram bin count.")
    parser.add_argument(
        "--significant_threshold_deg", type=float,
        default=math.degrees(DEFAULT_SIGNIFICANT_DIVERGENCE_THRESHOLD_RAD),
        help="Median rotation-metric disagreement above this (degrees) is flagged significant.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    percentiles = tuple(float(p) for p in args.percentiles.split(","))

    bank = torch.load(args.bank_path, map_location="cpu", weights_only=False)
    validate_measure_only_bank(args.bank_path, bank)
    print(f"[analyzer] bank confirmed measure-only: {args.bank_path}")

    schema = c3_st_measure_only_log_schema
    records = read_jsonl_log(args.log_path)
    counts = attempted_vs_recorded(records)
    accepted = [r for r in records if r.get(schema.FIELD_SUCCESS)]

    pos_report = distribution_report(
        [r[schema.FIELD_POS_DIST_M] for r in accepted], name="pos_dist_m", min_n=args.min_n, percentiles=percentiles, bins=args.bins
    )
    rot_report = distribution_report(
        [r[schema.FIELD_ROT_DIST_RAD] for r in accepted], name="rot_dist_rad", min_n=args.min_n, percentiles=percentiles, bins=args.bins
    )
    axis_report = distribution_report(
        [r[schema.FIELD_AXIS_TILT_RAD] for r in accepted], name="axis_tilt_rad", min_n=args.min_n, percentiles=percentiles, bins=args.bins
    )
    disagreement = rotation_disagreement_report(
        [r[schema.FIELD_ROT_DIST_RAD] for r in accepted],
        [r[schema.FIELD_AXIS_TILT_RAD] for r in accepted],
        min_n=args.min_n,
        significant_threshold_rad=math.radians(args.significant_threshold_deg),
    )
    proposal = propose_tolerances(
        [r[schema.FIELD_POS_DIST_M] for r in accepted],
        [r[schema.FIELD_ROT_DIST_RAD] for r in accepted],
        [r[schema.FIELD_AXIS_TILT_RAD] for r in accepted],
        min_n=args.min_n,
        percentiles=percentiles,
    )
    print(format_text_report(counts, pos_report, rot_report, axis_report, disagreement, proposal))


if __name__ == "__main__":
    main()
