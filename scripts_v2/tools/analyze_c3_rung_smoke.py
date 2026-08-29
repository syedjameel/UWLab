# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Analysis half of the C3 RUNG GPU smoke (bead ``dr-ai1.4`` / ``dr-ai1.20``, commits ``922c3d3`` /
``9b51f56`` / ``4217ed8``) -- reads the npz ``smoke_c3_rung_isaac.py`` wrote and the run log it was
launched under, and answers TWO questions depending on which mode produced the npz (read from
``meta.mode``, never assumed from the filename):

* ``--mode reset`` (bead ``dr-ai1.4``): did the per-env S1/S_t draw and dispatch
  (``c3_rung.py:195-231``) actually route each env's spawn and goal to the half its own recorded
  ``kind`` says it drew, or is something silently swapped.
* ``--mode settle`` (bead ``dr-ai1.20``): does the DEFERRED S_t goal re-pin (``_st_awaiting_repin``,
  ``C3RungGoalPoseCommand._update_command``, commits ``9b51f56``/``4217ed8``) actually fire, exactly
  once, no earlier than the step floor, landing on the leg's own settled pose rather than leaving
  the goal at its mid-air spawn pin -- and never fires for an S1 env, which is never armed.

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

SHARED GATES (both modes), each named so a failure says which one and why, not just "assert failed":

  1. **SPLIT RATIO** -- the observed S1 fraction over all ``n`` draws is within a binomial
     tolerance of ``--expected_s1_fraction`` (default 0.5). Tolerance = 3 standard errors of a
     Binomial(n, p) proportion (``sqrt(p*(1-p)/n)``), i.e. a ~99.7% band under the normal
     approximation -- derived from ``n``, never eyeballed. See :func:`binomial_3sigma_tolerance`.

  4. **STAGING TOOK EFFECT, PER THE RUN LOG, NOT THE COMMAND LINE** -- ``RESET_SPEC_V2.md`` R5 /
     Trap 3: "read the staged value back out of the run log; never assume the command-line value
     took effect." Greps the log for the two banners this stage's own code prints
     (``c3_rung_core.c3_rung_banner`` and ``dexlift_ur5e_delto_env_cfg._apply_pose_tilt_stage``'s
     ``"[dexlift] POSE_TILT staged"`` line) and checks the STAGED numbers therein against
     ``--expected_s1_fraction`` / ``--expected_pose_tilt``, not against what was merely passed on
     the command line.

``--mode reset`` GATES 2-3:

  2. **NOT SWAPPED** -- the thing the phase-1 smoke exists to catch. For each kind, independently
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
       * the commanded goal matches what each half's own arithmetic promises: S_t's goal equals the
         leg's own pose (``c3_rung_core.st_goal_pose``); S1's goal is offset from the leg's own
         position by ``|s1_goal_delta_m|`` (from the npz's recorded ``meta.s1_goal_delta_m``, never
         a restated literal) with orientation UNCHANGED (``c3_rung_core.s1_goal_orientation``).

  3. **TIP/ROOT CONVERSION SANITY** -- ``c3_rung_core.goal_tip_z_from_root_z``'s NOMINAL-tilt tip z
     (imported, not reimplemented) is compared against the GEOMETRIC tip z the smoke script already
     computed by rotating ``assembled_offset`` through the leg's ACTUAL measured quaternion. A gross
     disagreement (order of ROOT_ABOVE_TIP_M, i.e. ~106 mm) is exactly the F49 bare-subtraction
     failure mode and fails loudly by name.

``--mode settle`` GATES S1-S4 (bead ``dr-ai1.20``), CONTAMINATED envs (``data['contaminated']`` --
a mid-window auto-reset spliced a second episode's draw into that row, per
``smoke_c3_rung_isaac.py``'s own module docstring) are EXCLUDED from every one of these, count
reported separately:

  S1. **RE-PIN LANDS ON THE SETTLED POSE, AND MOVED TO GET THERE** -- for every non-contaminated
      S_t env that ever re-pinned, the FINAL commanded goal equals the leg's own FINAL (settled)
      pose (position + orientation) -- not swapped for "equals the provisional pin", which would
      pass vacuously if the re-pin hook silently never fired anything. To make sure the test is not
      vacuous the OTHER way, this gate also requires the goal to have actually MOVED from its
      provisional (t=0) pin by a stated minimum -- team-lead's own framing: "these must be DIFFERENT
      from each other, otherwise the test proves nothing."
  S2. **AT MOST ONE RE-PIN, NEVER FOR S1** -- structurally guaranteed by the smoke script's own
      edge-detection loop (a `repin_step < 0` guard records only the first True->False transition)
      and by ``_st_awaiting_repin`` only ever being ARMED in ``_resample_command``, i.e. at reset,
      which this phase does not repeat mid-window -- so this gate checks the OBSERVABLE consequence
      instead: no S1 env's ``ever_repinned`` is ever True, and no S1 env's final goal differs from
      its t=0 goal (S1 is never armed, so it must never move).
  S3. **NO RE-PIN BEFORE THE STEP FLOOR** -- ``repin_step`` is the EXACT internal
      ``episode_length_buf`` value the predicate itself used at the instant it fired (recorded by
      the smoke script, not re-derived here), so this is an EXACT check, not a tolerance: every
      recorded ``repin_step`` must be strictly greater than ``meta.settle_steps``
      (``held_check_core.SETTLE_STEPS``, imported by the smoke script, never restated).
  S4. **RE-PIN TIMING ADVISORY (not a hard gate)** -- team-lead's own triage order: re-pins
      clustering suspiciously early (immediately after the step floor opens) point at the LATCH or
      the HOOK misbehaving, not the predicate (which carries 64 tests and seven negative controls of
      its own) -- but distinguishing "the physics genuinely settled fast" from "the hook fired
      without checking speed" needs a human looking at the histogram, not a bright-line assert. This
      script reports the full ``repin_step`` histogram and prints an ADVISORY (never a FAIL) if more
      than half of the observed re-pins land within the first 5 steps after the floor opens.

TILT/REPIN-STEP ARE REPORTED AS HISTOGRAMS, NOT MEANS -- deliberately, because a mean over a bimodal
distribution (S1-vs-S_t tilt; settled-vs-still-bouncing repin timing) describes an outcome that
occurs for no single leg, and this project has already paid for that exact mistake once (an
experiment cost on a mean read over a bimodal distribution). :func:`histogram_report` is used
throughout; medians, not means, are what any gate above actually gates on.

Run (no GPU, no Isaac):
    python3 scripts_v2/tools/analyze_c3_rung_smoke.py \\
        --npz /path/to/out.npz --log /path/to/run.log \\
        --expected_s1_fraction 0.5 --expected_pose_tilt 0.3
The same invocation works for a ``--mode settle`` npz+log -- the mode is read from the npz's own
``meta.mode``, not passed on this script's command line. Exit code is nonzero iff any gate failed;
every gate's outcome is printed either way.
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
    corresponding key below comes back ``None`` and the log gate fails loudly rather than silently
    passing on a banner it no longer understands.
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


def run_split_ratio_gate(kind: np.ndarray, n: int, expected_s1_fraction: float) -> tuple[list[str], dict]:
    """GATE 1 (both modes): observed S1 fraction within a 3-sigma binomial band. Returns
    ``(failures, {n_s1, n_st, observed_s1_fraction})``."""
    failures: list[str] = []
    n_s1 = int((kind == C3_KIND_S1).sum())
    n_st = int((kind == C3_KIND_ST).sum())
    if n_s1 + n_st != n:
        failures.append(f"GATE 0 (SANITY): n_s1({n_s1}) + n_st({n_st}) != n({n}) -- unexpected kind value present.")

    observed_s1_fraction = n_s1 / n if n > 0 else float("nan")
    tol = binomial_3sigma_tolerance(n, expected_s1_fraction)
    lo, hi = expected_s1_fraction - tol, expected_s1_fraction + tol
    print(
        f"[analyze_c3_rung] GATE 1 (SPLIT RATIO): observed S1 fraction = {observed_s1_fraction:.4f}"
        f" ({n_s1}/{n}), expected {expected_s1_fraction:.4f} +- {tol:.4f} (3-sigma band"
        f" [{lo:.4f}, {hi:.4f}], derived from n={n}: 3*sqrt(p*(1-p)/n))",
        flush=True,
    )
    if not (lo <= observed_s1_fraction <= hi):
        failures.append(
            f"GATE 1 (SPLIT RATIO): observed S1 fraction {observed_s1_fraction:.4f} is outside the"
            f" 3-sigma band [{lo:.4f}, {hi:.4f}] around expected {expected_s1_fraction:.4f}."
        )
    return failures, {"n_s1": n_s1, "n_st": n_st, "observed_s1_fraction": observed_s1_fraction}


def run_log_gate(log_text: str, expected_s1_fraction: float, expected_pose_tilt: float, atol: float) -> list[str]:
    """GATE 4 (both modes): staging read back from the run log, not the command line."""
    failures: list[str] = []
    logged = parse_run_log(log_text)
    print(f"[analyze_c3_rung] GATE 4 (RUN LOG): {json.dumps(logged)}", flush=True)

    if logged["c3_s1_fraction_logged"] is None:
        failures.append(
            "GATE 4 (RUN LOG): the '[dexreset] C3 RUNG staged' banner was not found in the run log --"
            " DEXRESET_C3_RUNG=1 may not have taken effect, or the banner text changed and this"
            " script's regex is stale."
        )
    elif abs(logged["c3_s1_fraction_logged"] - expected_s1_fraction) > atol:
        failures.append(
            f"GATE 4 (RUN LOG): banner logged s1_fraction={logged['c3_s1_fraction_logged']}, expected"
            f" {expected_s1_fraction} (+-{atol}) -- the command-line --s1_fraction did not take"
            " effect as staged; do not trust the split-ratio result above."
        )

    if logged["pose_tilt_logged"] is None:
        failures.append(
            "GATE 4 (RUN LOG): the '[dexlift] POSE_TILT staged' banner was not found in the run log --"
            " DEXLIFT_POSE_TILT may not have taken effect, or the banner text changed and this"
            " script's regex is stale."
        )
    elif abs(logged["pose_tilt_logged"] - expected_pose_tilt) > atol:
        failures.append(
            f"GATE 4 (RUN LOG): banner logged POSE_TILT={logged['pose_tilt_logged']}, expected"
            f" {expected_pose_tilt} (+-{atol}) -- the command-line --pose_tilt did not take effect"
            " as staged; the tilt/repin distributions above describe a different (un-staged, or"
            " differently-staged) run than intended."
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
    return failures


def run_reset_gates(data: np.lib.npyio.NpzFile, meta: dict, n: int, args: argparse.Namespace) -> list[str]:
    """``--mode reset`` (bead dr-ai1.4): GATES 1-4 -- see module docstring."""
    failures: list[str] = []
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

    ratio_failures, _ = run_split_ratio_gate(kind, n, args.expected_s1_fraction)
    failures += ratio_failures

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

    log_text = Path(args.log).read_text()
    failures += run_log_gate(log_text, args.expected_s1_fraction, args.expected_pose_tilt, args.log_number_atol)

    if not failures:
        print(
            f"[analyze_c3_rung] (reset mode) n={n} (S1={n_s1}, S_t={n_st}), S1 median tilt="
            f"{s1_hist['median']:.2f} deg, S_t median tilt={st_hist['median']:.2f} deg.",
            flush=True,
        )
    return failures


def run_settle_gates(data: np.lib.npyio.NpzFile, meta: dict, n: int, args: argparse.Namespace) -> list[str]:
    """``--mode settle`` (bead dr-ai1.20): GATES S1-S4 -- see module docstring."""
    failures: list[str] = []
    kind = data["kind"]
    contaminated = data["contaminated"].astype(bool)
    ever_repinned = data["ever_repinned"].astype(bool)
    repin_step = data["repin_step"]
    goal_pos_t0 = data["goal_pos_t0_w"]
    goal_quat_t0 = data["goal_quat_t0_w_wxyz"]
    goal_pos_final = data["goal_pos_final_w"]
    goal_quat_final = data["goal_quat_final_w_wxyz"]
    leg_pos_final = data["leg_pos_final_w"]
    leg_quat_final = data["leg_quat_final_w_wxyz"]

    n_contaminated = int(contaminated.sum())
    clean = ~contaminated
    print(
        f"[analyze_c3_rung] {n_contaminated}/{n} envs CONTAMINATED (mid-window auto-reset) -- excluded"
        " from every gate below.",
        flush=True,
    )

    ratio_failures, _ = run_split_ratio_gate(kind[clean], int(clean.sum()), args.expected_s1_fraction)
    failures += [f.replace("GATE 1", "GATE 1 (settle, clean envs only)") for f in ratio_failures]

    s1_mask = clean & (kind == C3_KIND_S1)
    st_mask = clean & (kind == C3_KIND_ST)
    n_s1, n_st = int(s1_mask.sum()), int(st_mask.sum())

    st_repinned = st_mask & ever_repinned
    st_never = st_mask & ~ever_repinned
    n_st_repinned, n_st_never = int(st_repinned.sum()), int(st_never.sum())
    print(
        f"[analyze_c3_rung] S_t envs (clean): {n_st}, ever repinned: {n_st_repinned}, never settled in"
        f" the window (goal correctly still at the provisional pin, by design -- not a failure):"
        f" {n_st_never}",
        flush=True,
    )

    # ================ GATE S1: RE-PIN LANDS ON THE SETTLED POSE, AND MOVED TO GET THERE ================
    REPIN_POS_TOL_M = 0.002
    REPIN_QUAT_DOT_TOL = 0.999
    MOVED_MIN_POS_M = 0.003  # "these must be DIFFERENT from each other, otherwise the test proves nothing"
    if n_st_repinned > 0:
        final_pos_err = np.linalg.norm(goal_pos_final[st_repinned] - leg_pos_final[st_repinned], axis=-1)
        final_quat_dot = np.abs(np.sum(goal_quat_final[st_repinned] * leg_quat_final[st_repinned], axis=-1))
        bad_pos = final_pos_err > REPIN_POS_TOL_M
        bad_quat = final_quat_dot < REPIN_QUAT_DOT_TOL
        if bool(bad_pos.any()):
            failures.append(
                f"GATE S1 (RE-PIN TARGET): {int(bad_pos.sum())}/{n_st_repinned} repinned S_t envs have"
                f" |final_goal_pos - final_leg_pos| > {REPIN_POS_TOL_M} m (max {final_pos_err.max():.5f}"
                " m). The re-pin fired but did not land on the leg's own settled pose."
            )
        if bool(bad_quat.any()):
            failures.append(
                f"GATE S1 (RE-PIN TARGET): {int(bad_quat.sum())}/{n_st_repinned} repinned S_t envs have"
                f" final goal/leg quat dot < {REPIN_QUAT_DOT_TOL} (min {final_quat_dot.min():.5f})."
                " The re-pin fired but did not land on the leg's own settled orientation."
            )

        moved = np.linalg.norm(goal_pos_final[st_repinned] - goal_pos_t0[st_repinned], axis=-1)
        moved_hist = histogram_report(moved)
        print(f"[analyze_c3_rung] GATE S1 provisional-to-final goal movement (m): {json.dumps(moved_hist)}", flush=True)
        if moved_hist["median"] < MOVED_MIN_POS_M:
            failures.append(
                f"GATE S1 (RE-PIN IS NOT VACUOUS): median provisional-to-final goal movement is"
                f" {moved_hist['median']:.5f} m, below the {MOVED_MIN_POS_M} m floor this gate requires"
                " to consider the re-pin OBSERVED rather than coincidental (the goal barely moved, so"
                " a re-pin that silently never fired would look the same as one that did)."
            )
    elif n_st > 0:
        print("[analyze_c3_rung] GATE S1: no S_t env ever repinned in this window -- see GATE S3/window sizing.", flush=True)

    # ======================== GATE S2: AT MOST ONE RE-PIN, NEVER FOR S1 ========================
    # "At most once" is structurally guaranteed by the smoke script's own edge-detection loop and by
    # _st_awaiting_repin only ever being armed at reset (not repeated mid-window) -- not re-tested
    # here. What IS checked: the observable consequence for S1, which is never armed.
    if n_s1 > 0:
        bad_s1_repinned = ever_repinned[s1_mask]
        if bool(bad_s1_repinned.any()):
            failures.append(
                f"GATE S2 (S1 NEVER RE-PINNED): {int(bad_s1_repinned.sum())}/{n_s1} S1 envs show"
                " ever_repinned=True. S1 is never armed (C3RungGoalPoseCommand.__init__'s latch"
                " comment) -- this means the awaiting-repin latch or the predicate is firing for the"
                " wrong kind."
            )
        s1_pos_drift = np.linalg.norm(goal_pos_final[s1_mask] - goal_pos_t0[s1_mask], axis=-1)
        s1_quat_dot = np.abs(np.sum(goal_quat_final[s1_mask] * goal_quat_t0[s1_mask], axis=-1))
        S1_DRIFT_TOL_M = 0.0005
        bad_drift = s1_pos_drift > S1_DRIFT_TOL_M
        if bool(bad_drift.any()):
            failures.append(
                f"GATE S2 (S1 GOAL NEVER MOVES): {int(bad_drift.sum())}/{n_s1} S1 envs have"
                f" |final_goal - t0_goal| > {S1_DRIFT_TOL_M} m (max {s1_pos_drift.max():.5f} m)."
                " S1's goal is pinned once at reset and must not move."
            )
        bad_quat_drift = s1_quat_dot < 0.9999
        if bool(bad_quat_drift.any()):
            failures.append(
                f"GATE S2 (S1 GOAL NEVER MOVES): {int(bad_quat_drift.sum())}/{n_s1} S1 envs have"
                f" final/t0 goal quat dot < 0.9999 (min {s1_quat_dot.min():.6f}) -- S1's goal"
                " orientation moved when it must not."
            )

    # ============================ GATE S3: NO RE-PIN BEFORE THE STEP FLOOR ============================
    settle_steps = int(meta["settle_steps"])
    if n_st_repinned > 0:
        fired_steps = repin_step[st_repinned]
        repin_hist = histogram_report(fired_steps.astype(float))
        print(
            f"[analyze_c3_rung] GATE S3/S4 repin_step histogram (env-steps, floor={settle_steps}):"
            f" {json.dumps(repin_hist)}",
            flush=True,
        )
        too_early = fired_steps <= settle_steps
        if bool(too_early.any()):
            failures.append(
                f"GATE S3 (STEP FLOOR): {int(too_early.sum())}/{n_st_repinned} repinned S_t envs have"
                f" repin_step <= settle_steps={settle_steps} (min observed {int(fired_steps.min())})."
                " The predicate must require steps_since_reset > settle_steps, strictly -- this is an"
                " EXACT check (repin_step is the internal counter the predicate itself read), not a"
                " tolerance."
            )

        # ==================== GATE S4: RE-PIN TIMING ADVISORY (not a hard gate) ====================
        early_window_hi = settle_steps + 5
        n_early = int(((fired_steps > settle_steps) & (fired_steps <= early_window_hi)).sum())
        frac_early = n_early / n_st_repinned
        if frac_early > 0.5:
            print(
                f"[analyze_c3_rung] ADVISORY (GATE S4, not a failure): {n_early}/{n_st_repinned}"
                f" ({frac_early:.0%}) of re-pins fired within 5 steps of the floor opening"
                f" ({settle_steps}, {early_window_hi}]. Per team-lead's triage order: clustering this"
                " early points at the LATCH or the HOOK misbehaving, not the predicate (64 tests + 7"
                " negative controls of its own) -- worth a human look at the histogram above before"
                " trusting these re-pins, but this script does not have enough information (no"
                " per-step speed trace) to fail this on its own.",
                flush=True,
            )

    log_text = Path(args.log).read_text()
    failures += run_log_gate(log_text, args.expected_s1_fraction, args.expected_pose_tilt, args.log_number_atol)

    if not failures:
        print(
            f"[analyze_c3_rung] (settle mode) n={n} clean={int(clean.sum())} contaminated={n_contaminated},"
            f" S_t repinned={n_st_repinned}/{n_st}, S_t never settled={n_st_never}.",
            flush=True,
        )
    return failures


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

    data = np.load(args.npz, allow_pickle=False)
    meta = json.loads(str(data["meta_json"]))
    n = int(meta["n_samples"])
    mode = meta.get("mode", "reset")
    print(f"[analyze_c3_rung] loaded {args.npz}: mode={mode} n={n} task={meta['task']}", flush=True)
    print(f"[analyze_c3_rung] meta = {json.dumps(meta, indent=2)}", flush=True)

    if mode == "reset":
        failures = run_reset_gates(data, meta, n, args)
    elif mode == "settle":
        failures = run_settle_gates(data, meta, n, args)
    else:
        failures = [f"unknown meta['mode']={mode!r} -- expected 'reset' or 'settle'."]

    print("", flush=True)
    if failures:
        print(f"[analyze_c3_rung] FAIL -- {len(failures)} gate(s) failed:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        return 1

    print(f"[analyze_c3_rung] PASS -- all gates OK (mode={mode}).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
