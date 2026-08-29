# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Analysis half of the C3 RUNG GPU smoke (bead ``dr-ai1.4``, commit ``922c3d3``) -- reads the npz
``smoke_c3_rung_isaac.py`` wrote and the run log it was launched under, and answers exactly one
question: **did the per-env S1/S_t draw and dispatch (``c3_rung.py:195-231``) actually route each
env's spawn and goal to the half its own recorded ``kind`` says it drew, or is something silently
swapped.**

Isaac-FREE. Needs only ``numpy`` and the standard library -- no torch, no isaaclab, no GPU -- so it
runs under plain ``python3`` (or the CPU venv used for local syntax checks) once an npz + log exist,
exactly like ``validate_c4_bank.py`` / ``analyze_grasp_orientation_distribution.py`` next to it in
this directory are Isaac-free post-hoc analysis scripts.

``c3_rung_core.py`` (``goal_tip_z_from_root_z``, ``S1_NOMINAL_TILT_RAD``, ``ST_NOMINAL_TILT_RAD``,
``C3_KIND_S1``, ``C3_KIND_ST``) is IMPORTED here, by file path, with the exact loader idiom
``source/uwlab_tasks/test/test_c3_rung_stage.py`` already uses (compile-and-exec the source text
directly, not ``spec_from_file_location(...).loader.exec_module``, to sidestep that loader's
one-second ``__pycache__`` staleness granularity -- see the test's own docstring for the false-pass
this caused once already). This module is reused, not restated, for exactly the reason its own
docstring gives: "the fix worth copying is the API shape, not the arithmetic."

FOUR GATES, each named so a failure says which one and why, not just "assert failed":

  1. **SPLIT RATIO** -- the observed S1 fraction over all ``n`` draws is within a binomial
     tolerance of ``--expected_s1_fraction`` (default 0.5). Tolerance = 3 standard errors of a
     Binomial(n, p) proportion (``sqrt(p*(1-p)/n)``), i.e. a ~99.7% band under the normal
     approximation -- derived from ``n``, never eyeballed. See :func:`binomial_3sigma_tolerance`.

  2. **NOT SWAPPED** -- the thing this whole smoke exists to catch. For each kind, independently
     from the OTHER kind's numbers:
       * tilt-from-tip-down falls on the correct side of the SWAP THRESHOLD, the midpoint between
         :data:`S1_NOMINAL_TILT_RAD` (0 deg) and :data:`ST_NOMINAL_TILT_RAD` (90 deg) = 45 deg --
         S1's median must be BELOW it, S_t's ABOVE it. 45 deg is not an eyeballed number: it is the
         one point equidistant from both nominal values, so "median lands on the correct side" is
         exactly "the routing did not invert," independent of how tightly either half's spawn
         happens to cluster around its own nominal value.
       * the fixture is NOT parked for every S1 sample and IS parked for every S_t sample -- exact,
         not statistical, since fixture placement is deterministic per branch
         (``C3RungResetObject.__call__``), so ANY disagreement here is a routing bug, not noise.
       * the commanded goal matches what each half's own arithmetic promises:
         S_t's goal equals the leg's own pose (position AND orientation, zero delta --
         ``c3_rung_core.st_goal_pose``); S1's goal is offset from the leg's own position by
         ``|s1_goal_delta_m|`` (from the npz's recorded ``meta.s1_goal_delta_m``, never a restated
         literal) with orientation UNCHANGED (``c3_rung_core.s1_goal_orientation``).

  3. **TIP/ROOT CONVERSION SANITY** -- ``c3_rung_core.goal_tip_z_from_root_z``'s NOMINAL-tilt tip z
     (imported, not reimplemented) is compared against the GEOMETRIC tip z the smoke script already
     computed by rotating ``assembled_offset`` through the leg's ACTUAL measured quaternion. A gross
     disagreement (order of ROOT_ABOVE_TIP_M, i.e. ~106 mm) is exactly the F49 bare-subtraction
     failure mode and fails loudly by name.

  4. **STAGING TOOK EFFECT, PER THE RUN LOG, NOT THE COMMAND LINE** -- ``RESET_SPEC_V2.md`` R5 /
     Trap 3: "read the staged value back out of the run log; never assume the command-line value
     took effect." Greps the log for the two banners this stage's own code prints
     (``c3_rung_core.c3_rung_banner`` and ``dexlift_ur5e_delto_env_cfg._apply_pose_tilt_stage``'s
     ``"[dexlift] POSE_TILT staged"`` line) and checks the STAGED numbers therein against
     ``--expected_s1_fraction`` / ``--expected_pose_tilt``, not against what was merely passed on
     the command line.

TILT IS REPORTED AS A HISTOGRAM, NOT A MEAN -- deliberately, because a mean over the bimodal
S1-vs-S_t tilt distribution describes an outcome that occurs for no single leg, and this project has
already paid for that exact mistake once (an experiment cost on a mean read over a bimodal
distribution). :func:`histogram_report` is used for both kinds; the median (not the mean) is what
gate 2 above actually gates on.

Run (no GPU, no Isaac):
    python3 scripts_v2/tools/analyze_c3_rung_smoke.py \\
        --npz /path/to/out.npz --log /path/to/run.log \\
        --expected_s1_fraction 0.5 --expected_pose_tilt 0.3
Exit code is nonzero iff any gate failed; every gate's outcome is printed either way.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

_MDP_DIR = (
    Path(__file__).resolve().parents[2]
    / "source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/dexlift/mdp"
)


def _load(name: str):
    """Load an Isaac-free ``mdp`` module by FILE PATH, compiling the source text directly.

    Identical idiom, and identical reason, as ``test_c3_rung_stage.py``'s own ``_load``: NOT
    ``spec_from_file_location(...).loader.exec_module(...)``, because that loader's staleness check
    against ``__pycache__`` is only one-second granular and has already produced one false pass in
    this campaign (a mutated-bytecode run reported as a passing restored-source run). Compiling the
    text fresh every call costs microseconds and removes that failure mode.
    """
    path = _MDP_DIR / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"expected an Isaac-free mdp module at {path} -- is this script still next to the dexlift"
            " mdp package (scripts_v2/tools/../../source/uwlab_tasks/.../dexlift/mdp)?"
        )
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)  # noqa: S102
    return module


# Order matters: c3_rung_core falls back to a plain ``import c3_transport_core`` when loaded outside
# a package, so the sibling must already be in sys.modules -- same requirement the test states.
_c3_transport_core = _load("c3_transport_core")
_c3_rung_core = _load("c3_rung_core")

C3_KIND_S1 = _c3_rung_core.C3_KIND_S1
C3_KIND_ST = _c3_rung_core.C3_KIND_ST
S1_NOMINAL_TILT_DEG = math.degrees(_c3_rung_core.S1_NOMINAL_TILT_RAD)
ST_NOMINAL_TILT_DEG = math.degrees(_c3_rung_core.ST_NOMINAL_TILT_RAD)
# The one point equidistant from both nominal tilts -- see the module docstring's gate 2. Derived,
# not restated: changing either nominal constant upstream moves this threshold automatically.
SWAP_THRESHOLD_DEG = (S1_NOMINAL_TILT_DEG + ST_NOMINAL_TILT_DEG) / 2.0
goal_tip_z_from_root_z = _c3_rung_core.goal_tip_z_from_root_z


def binomial_3sigma_tolerance(n: int, p: float) -> float:
    """3 standard errors of a Binomial(n, p) proportion under the normal approximation
    (``sqrt(p*(1-p)/n)``) -- a ~99.7% band. 3-sigma, not 1 or 2, because this gate runs once per
    smoke invocation (not repeatedly), so the operating point that matters is "does a genuine routing
    bug get through," not "what is the tightest band that still mostly holds" -- a false ratio-gate
    failure on an otherwise-correct run would send someone chasing a phantom for the wrong reason.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"binomial_3sigma_tolerance needs p in (0, 1); got {p}")
    if n <= 0:
        raise ValueError(f"binomial_3sigma_tolerance needs n > 0; got {n}")
    return 3.0 * math.sqrt(p * (1.0 - p) / n)


def histogram_report(x: np.ndarray, bins: int = 18) -> dict:
    if x.size == 0:
        return {"n": 0}
    counts, edges = np.histogram(x, bins=bins)
    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "counts": counts.tolist(),
        "edges": edges.tolist(),
    }


def parse_run_log(log_text: str) -> dict:
    """Read the STAGED values back out of the run log -- RESET_SPEC_V2.md R5 / Trap 3: "never assume
    the command-line value took effect." Regexes match the banners ``c3_rung_core.c3_rung_banner``
    and ``_apply_pose_tilt_stage`` print byte-for-byte today; if either format ever changes, the
    corresponding key below comes back ``None`` and gate 4 fails loudly rather than silently passing
    on a banner it no longer understands.
    """
    result = {"c3_s1_fraction_logged": None, "c3_st_fraction_logged": None, "pose_tilt_logged": None}

    m = re.search(
        r"\[dexreset\] C3 RUNG staged.*?:\s*([0-9.]+)\s*of envs draw S1 and\s*([0-9.]+)\s*draw S_t",
        log_text,
    )
    if m:
        result["c3_s1_fraction_logged"] = float(m.group(1))
        result["c3_st_fraction_logged"] = float(m.group(2))

    m = re.search(
        r"\[dexlift\] POSE_TILT staged: .*?\+-([0-9.]+) rad",
        log_text,
    )
    if m:
        result["pose_tilt_logged"] = float(m.group(1))

    result["c3_wiring_line_present"] = "[dexreset] C3 RUNG wiring: reset_object -> C3RungResetObject" in log_text
    result["staging_verified_line_present"] = "[smoke_c3_rung] staging verified:" in log_text
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--log", type=str, required=True, help="The run log smoke_c3_rung_isaac.py's stdout was redirected to.")
    parser.add_argument("--expected_s1_fraction", type=float, default=0.5)
    parser.add_argument("--expected_pose_tilt", type=float, default=0.3)
    parser.add_argument(
        "--log_number_atol",
        type=float,
        default=0.0015,
        help="Absolute tolerance for comparing a --expected_* value against the run-log banner's"
        " OWN printed precision (the banners print s1_fraction to 3 decimals and tilt to 4), not a"
        " physical tolerance.",
    )
    args = parser.parse_args()

    failures: list[str] = []

    data = np.load(args.npz, allow_pickle=False)
    meta = json.loads(str(data["meta_json"]))
    n = int(meta["n_samples"])
    print(f"[analyze_c3_rung] loaded {args.npz}: n={n} task={meta['task']}", flush=True)
    print(f"[analyze_c3_rung] meta = {json.dumps(meta, indent=2)}", flush=True)

    kind = data["kind"]
    leg_pos = data["leg_root_pos_w"]
    goal_pos = data["goal_pos_w"]
    leg_quat = data["leg_root_quat_w_wxyz"]
    goal_quat = data["goal_quat_w_wxyz"]
    leg_root_z = leg_pos[:, 2]
    leg_tip_z = data["leg_tip_pos_w"][:, 2]
    leg_tilt = data["leg_tilt_from_tipdown_deg"]
    fixture_parked = data["fixture_parked"].astype(bool)

    s1_mask = kind == C3_KIND_S1
    st_mask = kind == C3_KIND_ST
    n_s1, n_st = int(s1_mask.sum()), int(st_mask.sum())
    if n_s1 + n_st != n:
        failures.append(f"GATE 0 (SANITY): n_s1({n_s1}) + n_st({n_st}) != n({n}) -- unexpected kind value present.")

    # ==================================== GATE 1: SPLIT RATIO ====================================
    observed_s1_fraction = n_s1 / n
    tol = binomial_3sigma_tolerance(n, args.expected_s1_fraction)
    lo, hi = args.expected_s1_fraction - tol, args.expected_s1_fraction + tol
    print(
        f"[analyze_c3_rung] GATE 1 (SPLIT RATIO): observed S1 fraction = {observed_s1_fraction:.4f}"
        f" ({n_s1}/{n}), expected {args.expected_s1_fraction:.4f} +- {tol:.4f} (3-sigma band"
        f" [{lo:.4f}, {hi:.4f}], derived from n={n}: 3*sqrt(p*(1-p)/n))",
        flush=True,
    )
    if not (lo <= observed_s1_fraction <= hi):
        failures.append(
            f"GATE 1 (SPLIT RATIO): observed S1 fraction {observed_s1_fraction:.4f} is outside the"
            f" 3-sigma band [{lo:.4f}, {hi:.4f}] around expected {args.expected_s1_fraction:.4f}."
        )

    # ==================================== GATE 2: NOT SWAPPED ====================================
    print(
        f"[analyze_c3_rung] GATE 2 (NOT SWAPPED): swap threshold = {SWAP_THRESHOLD_DEG:.2f} deg"
        f" (midpoint of S1 nominal {S1_NOMINAL_TILT_DEG:.1f} deg and S_t nominal"
        f" {ST_NOMINAL_TILT_DEG:.1f} deg). S1 median must be BELOW it, S_t ABOVE it.",
        flush=True,
    )
    s1_hist = histogram_report(leg_tilt[s1_mask])
    st_hist = histogram_report(leg_tilt[st_mask])
    print(f"[analyze_c3_rung] S1 leg tilt-from-tipdown histogram (deg): {json.dumps(s1_hist)}", flush=True)
    print(f"[analyze_c3_rung] S_t leg tilt-from-tipdown histogram (deg): {json.dumps(st_hist)}", flush=True)

    if n_s1 > 0 and s1_hist["median"] >= SWAP_THRESHOLD_DEG:
        failures.append(
            f"GATE 2 (NOT SWAPPED, TILT): S1's median tilt-from-tipdown is {s1_hist['median']:.2f} deg,"
            f" >= the {SWAP_THRESHOLD_DEG:.2f} deg swap threshold (S1 should be near"
            f" {S1_NOMINAL_TILT_DEG:.0f} deg, TIP-DOWN). S1 legs look HORIZONTAL -- this is the S1/S_t"
            " tilt SWAP this smoke exists to catch."
        )
    if n_st > 0 and st_hist["median"] <= SWAP_THRESHOLD_DEG:
        failures.append(
            f"GATE 2 (NOT SWAPPED, TILT): S_t's median tilt-from-tipdown is {st_hist['median']:.2f} deg,"
            f" <= the {SWAP_THRESHOLD_DEG:.2f} deg swap threshold (S_t should be near"
            f" {ST_NOMINAL_TILT_DEG:.0f} deg, HORIZONTAL). S_t legs look TIP-DOWN -- this is the"
            " S1/S_t tilt SWAP this smoke exists to catch."
        )

    if n_s1 > 0 and bool(fixture_parked[s1_mask].any()):
        n_bad = int(fixture_parked[s1_mask].sum())
        failures.append(
            f"GATE 2 (NOT SWAPPED, FIXTURE): {n_bad}/{n_s1} S1 envs have the fixture PARKED; S1 must"
            " place the fixture at its normal RECEPTIVE_POSE_RANGE pose (the leg composes against it)."
            " Fixture placement is deterministic per branch, so this is a routing bug, not noise."
        )
    if n_st > 0 and not bool(fixture_parked[st_mask].all()):
        n_bad = int((~fixture_parked[st_mask]).sum())
        failures.append(
            f"GATE 2 (NOT SWAPPED, FIXTURE): {n_bad}/{n_st} S_t envs do NOT have the fixture parked;"
            " S_t must park it on every reset (episode_mixture.py's 'THE FIXTURE IS WRITTEN EVERY"
            " RESET' section) or a stale S1 fixture pose sits in the leg's own workspace."
        )

    goal_pos_err = np.linalg.norm(goal_pos - leg_pos, axis=-1)
    goal_quat_dot = np.abs(np.sum(goal_quat * leg_quat, axis=-1))  # abs: q and -q are the same rotation

    ST_GOAL_POS_TOL_M = 0.001  # provisional pin is delta_m=0.0 read at the same instant as leg_pos
    ST_GOAL_QUAT_DOT_TOL = 0.999
    if n_st > 0:
        bad_pos = goal_pos_err[st_mask] > ST_GOAL_POS_TOL_M
        bad_quat = goal_quat_dot[st_mask] < ST_GOAL_QUAT_DOT_TOL
        if bool(bad_pos.any()):
            failures.append(
                f"GATE 2 (NOT SWAPPED, S_t GOAL POSITION): {int(bad_pos.sum())}/{n_st} S_t envs have"
                f" |goal_pos - leg_pos| > {ST_GOAL_POS_TOL_M} m (max"
                f" {goal_pos_err[st_mask].max():.5f} m). S_t's goal must equal the leg's OWN pose,"
                " zero delta (c3_rung_core.st_goal_pose) -- looks like the S1 delta-offset branch fired"
                " instead."
            )
        if bool(bad_quat.any()):
            failures.append(
                f"GATE 2 (NOT SWAPPED, S_t GOAL ORIENTATION): {int(bad_quat.sum())}/{n_st} S_t envs have"
                f" goal/leg quat dot < {ST_GOAL_QUAT_DOT_TOL} (min {goal_quat_dot[st_mask].min():.5f})."
                " S_t's goal orientation must equal the leg's own, unchanged."
            )

    s1_goal_delta_m = float(meta["s1_goal_delta_m"])
    S1_GOAL_POS_TOL_M = 0.001
    S1_GOAL_QUAT_DOT_TOL = 0.999
    if n_s1 > 0:
        s1_pos_err_from_expected = np.abs(goal_pos_err[s1_mask] - abs(s1_goal_delta_m))
        bad_pos = s1_pos_err_from_expected > S1_GOAL_POS_TOL_M
        bad_quat = goal_quat_dot[s1_mask] < S1_GOAL_QUAT_DOT_TOL
        if bool(bad_pos.any()):
            failures.append(
                f"GATE 2 (NOT SWAPPED, S1 GOAL POSITION): {int(bad_pos.sum())}/{n_s1} S1 envs have"
                f" |goal_pos - leg_pos| differing from the expected |s1_goal_delta_m|="
                f"{abs(s1_goal_delta_m):.5f} m by more than {S1_GOAL_POS_TOL_M} m (max deviation"
                f" {s1_pos_err_from_expected.max():.5f} m). S1's goal must be the spawn pose displaced"
                " exactly s1_goal_delta_m along the bore axis -- looks like the S_t zero-delta branch"
                " fired instead, or the axis/magnitude is wrong."
            )
        if bool(bad_quat.any()):
            failures.append(
                f"GATE 2 (NOT SWAPPED, S1 GOAL ORIENTATION): {int(bad_quat.sum())}/{n_s1} S1 envs have"
                f" goal/leg quat dot < {S1_GOAL_QUAT_DOT_TOL} (min {goal_quat_dot[s1_mask].min():.5f})."
                " S1's goal orientation must equal the spawn orientation, unchanged"
                " (c3_rung_core.s1_goal_orientation)."
            )

    # ============================ GATE 3: TIP/ROOT CONVERSION SANITY ============================
    # goal_tip_z_from_root_z is the NOMINAL-tilt conversion (imported, not reimplemented); leg_tip_z
    # is the GEOMETRIC (exact, actual-quaternion) tip z the smoke script already computed. A gross
    # (~ROOT_ABOVE_TIP_M, ~106 mm) disagreement is the F49 bare-subtraction failure mode.
    F49_GROSS_ERROR_M = 0.05  # well under the 106.203 mm a bare subtraction would be wrong by
    for label, mask, kind_const in (("S1", s1_mask, C3_KIND_S1), ("S_t", st_mask, C3_KIND_ST)):
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        nominal_tip_z = np.array([goal_tip_z_from_root_z(float(z), kind_const) for z in leg_root_z[mask]])
        diff = np.abs(nominal_tip_z - leg_tip_z[mask])
        print(
            f"[analyze_c3_rung] GATE 3 ({label} tip/root conversion): nominal-vs-geometric tip z"
            f" |diff| median={float(np.median(diff)):.5f} m, max={float(diff.max()):.5f} m"
            f" (n={cnt}, F49 gross-error threshold={F49_GROSS_ERROR_M} m)",
            flush=True,
        )
        if bool((diff > F49_GROSS_ERROR_M).any()):
            n_bad = int((diff > F49_GROSS_ERROR_M).sum())
            failures.append(
                f"GATE 3 (TIP/ROOT CONVERSION, {label}): {n_bad}/{cnt} envs have"
                f" |goal_tip_z_from_root_z(root_z, {label}) - geometric_tip_z| > {F49_GROSS_ERROR_M} m"
                f" (max {diff.max():.4f} m) -- of the same order as ROOT_ABOVE_TIP_M (0.106203 m)."
                " This is the F49 bare-subtraction failure mode: the wrong nominal tilt is being"
                " applied for this kind, or the conversion has regressed to a bare subtraction."
            )

    # ==================== GATE 4: STAGING TOOK EFFECT, PER THE RUN LOG ====================
    log_text = Path(args.log).read_text()
    logged = parse_run_log(log_text)
    print(f"[analyze_c3_rung] GATE 4 (RUN LOG): {json.dumps(logged)}", flush=True)

    if logged["c3_s1_fraction_logged"] is None:
        failures.append(
            "GATE 4 (RUN LOG): the '[dexreset] C3 RUNG staged' banner was not found in the run log --"
            " DEXRESET_C3_RUNG=1 may not have taken effect, or the banner text changed and this"
            " script's regex is stale."
        )
    elif abs(logged["c3_s1_fraction_logged"] - args.expected_s1_fraction) > args.log_number_atol:
        failures.append(
            f"GATE 4 (RUN LOG): banner logged s1_fraction={logged['c3_s1_fraction_logged']}, expected"
            f" {args.expected_s1_fraction} (+-{args.log_number_atol}) -- the command-line"
            " --s1_fraction did not take effect as staged; do not trust the split-ratio result above."
        )

    if logged["pose_tilt_logged"] is None:
        failures.append(
            "GATE 4 (RUN LOG): the '[dexlift] POSE_TILT staged' banner was not found in the run log --"
            " DEXLIFT_POSE_TILT may not have taken effect, or the banner text changed and this"
            " script's regex is stale."
        )
    elif abs(logged["pose_tilt_logged"] - args.expected_pose_tilt) > args.log_number_atol:
        failures.append(
            f"GATE 4 (RUN LOG): banner logged POSE_TILT={logged['pose_tilt_logged']}, expected"
            f" {args.expected_pose_tilt} (+-{args.log_number_atol}) -- the command-line --pose_tilt"
            " did not take effect as staged; the S_t at-reset tilt distribution above describes a"
            " different (un-staged, or differently-staged) run than intended."
        )

    if not logged["c3_wiring_line_present"]:
        failures.append(
            "GATE 4 (RUN LOG): the '[dexreset] C3 RUNG wiring: reset_object -> C3RungResetObject'"
            " line is missing from the run log -- _apply_c3_rung_stage may not have returned True."
        )
    if not logged["staging_verified_line_present"]:
        failures.append(
            "GATE 4 (RUN LOG): smoke_c3_rung_isaac.py's own 'staging verified' line is missing from"
            " the run log -- either the script exited before that check, or it never ran."
        )

    print("", flush=True)
    if failures:
        print(f"[analyze_c3_rung] FAIL -- {len(failures)} gate(s) failed:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        return 1

    print(
        f"[analyze_c3_rung] PASS -- all gates OK. n={n} (S1={n_s1}, S_t={n_st}),"
        f" observed S1 fraction={observed_s1_fraction:.4f}, S1 median tilt={s1_hist['median']:.2f} deg,"
        f" S_t median tilt={st_hist['median']:.2f} deg.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
