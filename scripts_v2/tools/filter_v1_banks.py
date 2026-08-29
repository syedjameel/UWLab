#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Reads existing (v1-era) reset banks and reports, per bank, how many stored states survive
every v2 acceptance criterion that is evaluable from STORED STATE ALONE (bead R3,
``RESET_SPEC_V2.md`` sec 6: "Reuse old-bank states only where they pass the v2 filter; regenerate
the rest.").

*** UPPER BOUND ONLY -- READ THIS BEFORE READING ANY NUMBER THIS SCRIPT PRINTS. ***
Every count here is "geometrically compatible", NEVER "reusable". The held-state gate chain
(settled/opposed_contact/co_move) needs a running sim -- live contact-sensor forces and step-count
history a stored root pose does not carry -- and cannot be evaluated by this script at all. A
state that passes every check here may still fail that chain when replayed, and that chain is
where the large majority of v1 attempts died. Every print statement below repeats this because a
number silently re-labelled "reusable" downstream is exactly this campaign's own defect class,
repeated (and would inflate the reuse estimate by roughly an order of magnitude if read as final).

WHAT THIS SCRIPT ACTUALLY CHECKS, per bank, by its reset_type (see
v1_bank_geometry_core.py's own docstring, "SCOPE OF THIS PASS", for the full reasoning):
  - ObjectPartiallyAssembledEEGrasped banks (have receptive_object): C4's mating-frame band
    (code defaults, ONE count) AND C3(S1)'s band as a SURVIVAL CURVE (OPEN, bead dr-sj6.23).
  - ObjectRestingEEGrasped* banks: C2's resting-speed filter (code default, ONE count).
  - Anything else: reported (path/n/provenance) with an explicit "no offline v2 criterion in this
    pass's scope" note -- not silently skipped.
  - C1: out of scope for this whole pass (see v1_bank_geometry_core.py's docstring for why).
  - C3(S_t): never computed here -- its geometric criterion is VACUOUS for an existing stored
    state (the goal IS the stored pose by definition), so any number would carry no information.

PROVENANCE: the leg asset name is read off the bank's OWN directory path
(``v1_bank_geometry_core.leg_asset_from_path``) -- ``UNKNOWN`` if the path does not contain a
recognisable ``OneLegInsertionFixture__<asset>`` component, never guessed.

READ-ONLY. This script only ever calls ``torch.load`` -- it never writes, moves, renames, or
deletes anything at ``--bank_path``.

Needs ``torch`` to read the ``.pt`` bank (same reason every ``validate_c4_bank*.py`` script in
this directory needs it) and ``numpy`` for the actual geometry/statistics, via
``v1_bank_geometry_core.py`` (Isaac-free, numpy-only, loaded by file path). Meant to run on
DL_H100 with ``/home/dom_iva/venv_uwlab/bin/python3`` (confirmed: torch.load works there without a
GPU) via ``tools/h100.sh``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re

import numpy as np
import torch

_CORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v1_bank_geometry_core.py")
_spec = importlib.util.spec_from_file_location("v1_bank_geometry_core", _CORE_PATH)
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)

UPPER_BOUND_BANNER = (
    "*** UPPER BOUND ONLY: 'geometrically compatible', NOT 'reusable' -- the held-state chain "
    "(settled/opposed_contact/co_move) needs a running sim and is not evaluated here; it is where "
    "the large majority of v1 attempts died. ***"
)

_RESET_TYPE_RE = re.compile(r"resets_([A-Za-z0-9]+(?:EEGrasped|EEAnywhere)?)")


def reset_type_from_filename(bank_path: str) -> str:
    """Best-effort reset_type stem from the filename (``resets_<type>...pt``); returns the raw
    stem between ``resets_`` and the first ``.``/``_off``/``_fwd`` suffix marker, or the literal
    string ``"UNKNOWN"`` if the filename does not match the expected ``resets_*.pt`` shape."""
    base = os.path.basename(bank_path)
    if not base.startswith("resets_") or not base.endswith(".pt"):
        return "UNKNOWN"
    stem = base[len("resets_"):-len(".pt")]
    # strip known C2-rewind suffixes (_off0.10s, _fwd0.05s, etc.) and ad hoc bank-specific suffixes
    # (.rekeyed, .clean949, .slabclear2mm) so e.g. "ObjectRestingEEGrasped_off0.10s" and
    # "ObjectRestingEEGrasped.clean949" both still match the base reset_type.
    stem = stem.split(".")[0]
    m = re.match(r"^(Object[A-Za-z]+)", stem)
    return m.group(1) if m else stem


def load_bank_states(bank_path: str) -> dict:
    """Read one ``.pt`` bank and return the plain-numpy arrays this script needs. Read-only --
    calls ``torch.load`` and nothing else. Raises ``KeyError``/``ValueError`` with a clear message
    (not a bare stack trace) if the bank does not have the expected ``initial_state.rigid_object``
    shape.
    """
    raw = torch.load(bank_path, map_location="cpu", weights_only=False)
    if "initial_state" not in raw:
        raise ValueError(f"{bank_path}: no 'initial_state' key -- not a reset bank this script recognises.")
    rigid = raw["initial_state"].get("rigid_object", {})
    leg_key = "insertive_object" if "insertive_object" in rigid else ("object" if "object" in rigid else None)
    fixture_key = "receptive_object" if "receptive_object" in rigid else None
    if leg_key is None:
        raise ValueError(
            f"{bank_path}: initial_state.rigid_object has no 'insertive_object' or 'object' key "
            f"(got {sorted(rigid.keys())}) -- cannot locate the leg's own stored state."
        )
    out: dict = {}
    leg_pose = torch.stack(rigid[leg_key]["root_pose"]).numpy()
    leg_vel = torch.stack(rigid[leg_key]["root_velocity"]).numpy()
    out["n"] = leg_pose.shape[0]
    out["leg_pos"] = leg_pose[:, :3]
    out["leg_quat"] = leg_pose[:, 3:]
    out["leg_lin_speed_mps"] = np.linalg.norm(leg_vel[:, :3], axis=-1)
    if fixture_key is not None:
        fix_pose = torch.stack(rigid[fixture_key]["root_pose"]).numpy()
        out["fix_pos"] = fix_pose[:, :3]
        out["fix_quat"] = fix_pose[:, 3:]
    return out


def report_bank(bank_path: str, *, scale_factors: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)) -> str:
    """Build the full per-bank report text. A separate function from ``main()`` so a caller (or a
    test with a monkeypatched loader) can get the report without going through the CLI."""
    lines: list[str] = []
    leg_asset = core.leg_asset_from_path(bank_path)
    reset_type = reset_type_from_filename(bank_path)
    lines.append(f"=== BANK: {bank_path}")
    lines.append(f"    leg asset (provenance): {leg_asset}")
    lines.append(f"    reset_type (from filename): {reset_type}")

    try:
        states = load_bank_states(bank_path)
    except Exception as e:  # noqa: BLE001 -- reported per-bank, not fatal to the whole run
        lines.append(f"    COULD NOT READ: {e}")
        return "\n".join(lines)

    n = states["n"]
    lines.append(f"    total states: {n}")
    lines.append(f"    {UPPER_BOUND_BANNER}")

    if reset_type == "ObjectPartiallyAssembledEEGrasped" and "fix_pos" in states:
        depth_m, lateral_m, tilt_deg = core.decompose_mating_frame(
            states["leg_pos"], states["leg_quat"], states["fix_pos"], states["fix_quat"]
        )
        # -- C4, code-default band, ONE count -- per-criterion counts reported SEPARATELY first,
        # so a single dominant rejector is visible rather than hidden in a three-way conjunction.
        depth_ok = (depth_m >= core.DEFAULT_C4_DEPTH_MIN_M) & (depth_m <= core.DEFAULT_C4_DEPTH_MAX_M)
        lateral_ok = lateral_m <= core.DEFAULT_C4_LATERAL_MAX_M
        tilt_ok = tilt_deg <= core.DEFAULT_C4_TILT_MAX_DEG
        joint_ok = depth_ok & lateral_ok & tilt_ok
        lines.append(
            f"    [C4 band, code defaults: depth[{core.DEFAULT_C4_DEPTH_MIN_M * 1000:.1f},"
            f"{core.DEFAULT_C4_DEPTH_MAX_M * 1000:.1f}]mm lateral<={core.DEFAULT_C4_LATERAL_MAX_M * 1000:.1f}mm "
            f"tilt<={core.DEFAULT_C4_TILT_MAX_DEG:.1f}deg]"
        )
        lines.append(f"      depth-only pass:   {int(depth_ok.sum())}/{n}")
        lines.append(f"      lateral-only pass: {int(lateral_ok.sum())}/{n}")
        lines.append(f"      tilt-only pass:    {int(tilt_ok.sum())}/{n}")
        lines.append(f"      ALL THREE (joint): {int(joint_ok.sum())}/{n}  <- upper bound for C4 reuse")

        # -- C3(S1), band is OPEN (dr-sj6.23): survival CURVE, never a single picked count.
        curve = core.band_survival_curve(depth_m, lateral_m, tilt_deg, scale_factors=scale_factors)
        lines.append(
            "    [C3(S1) band, OPEN (bead dr-sj6.23) -- survival vs. scale of the v1 precedent "
            f"shape (depth[{core.V1_S1_PRECEDENT_DEPTH_MIN_M * 1000:.0f},"
            f"{core.V1_S1_PRECEDENT_DEPTH_MAX_M * 1000:.0f}]mm "
            f"lateral<={core.V1_S1_PRECEDENT_LATERAL_MAX_M * 1000:.0f}mm "
            f"tilt<={core.V1_S1_PRECEDENT_TILT_MAX_DEG:.0f}deg @ scale=1.0)]"
        )
        for row in curve:
            lines.append(
                f"      scale={row['scale_factor']:.2f}  depth[{row['depth_min_m'] * 1000:.2f},"
                f"{row['depth_max_m'] * 1000:.2f}]mm lateral<={row['lateral_max_m'] * 1000:.2f}mm "
                f"tilt<={row['tilt_max_deg']:.2f}deg  ->  {row['n_survive']}/{row['n']} "
                f"({row['survive_fraction']:.2%})"
            )
        lines.append(
            "    [C3(S_t): not computed -- its own goal is the stored pose itself, so a "
            "geometric distance-from-goal number here is vacuous by construction, not a finding.]"
        )
    elif reset_type.startswith("ObjectRestingEEGrasped"):
        speed_ok = core.resting_speed_survival(states["leg_lin_speed_mps"])
        lines.append(
            f"    [C2 resting speed, code default <= {core.DEFAULT_C2_MAX_RESTING_SPEED_MPS:.2f} m/s "
            f"(absolute linear, from stored root_velocity)]"
        )
        lines.append(f"      pass: {int(speed_ok.sum())}/{n}  <- upper bound for C2 reuse")
    else:
        lines.append(
            f"    [no offline v2 criterion in this pass's scope for reset_type={reset_type!r} -- "
            "reported for its state count and provenance only]"
        )

    lines.append(
        "    [C1: out of scope for this pass -- needs forward kinematics from stored joint state "
        "to recover an achieved hand pose, not attempted here.]"
    )
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank_path", type=str, action="append", required=True, help="Path to one .pt bank. Repeat for multiple banks.")
    parser.add_argument(
        "--s1_scale_factors", type=str, default="0.5,0.75,1.0,1.25,1.5,2.0,3.0",
        help="Comma-separated scale factors swept for C3(S1)'s survival curve.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    scale_factors = tuple(float(s) for s in args.s1_scale_factors.split(","))
    print(UPPER_BOUND_BANNER)
    print(f"[filter_v1_banks] {len(args.bank_path)} bank(s) requested")
    for bank_path in args.bank_path:
        print()
        print(report_bank(bank_path, scale_factors=scale_factors))
    print()
    print(UPPER_BOUND_BANNER)


if __name__ == "__main__":
    main()
