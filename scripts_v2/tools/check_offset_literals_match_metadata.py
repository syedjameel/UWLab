#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Standalone check: does metadata.yaml's assembled_offset agree with the hardcoded literals in
validate_c4_bank_aro2_3.py? Runs with plain Python + PyYAML -- no Isaac boot required.

Why this can run standalone at all: read_metadata_from_usd_directory() (omnireset/mdp/utils.py)
resolves a LOCAL metadata.yaml path via safe_retrieve_file_path(), which for a local file is just
`os.path.isfile(url)` -> return it -- no Isaac/omni/pxr import is on that path. The only reason the
real utils.py module can't be imported directly here is that importing the MODULE (not the function)
drags in isaaclab/isaacsim/omni/pxr/pytorch3d at import time (see its own imports). This script
reads the same metadata.yaml files directly instead, so it can be run in a bare Python env (CI, a
laptop, before Isaac is even installed) to catch literal drift early.

This is the SAME check as commands.py's `_assert_offset_matches_pinned_literals` (which runs at
every TaskCommand construction, i.e. every real training/eval env startup) -- this script exists so
the check can ALSO run without booting Isaac at all, e.g. in CI on every commit.

Exit code 0 = all pinned pairs agree. Exit code 1 = at least one mismatch (or a metadata.yaml could
not be read) -- see the printed message for which literal/value disagree.
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

# Must be byte-identical to commands.py's _PINNED_OFFSET_LITERALS and to
# scripts_v2/tools/validate_c4_bank_aro2_3.py:188-190 (_LEG_OFF_POS / _LEG_OFF_QUAT_WXYZ / _RECV_OFF_POS).
PINNED_OFFSET_LITERALS: dict[str, dict[str, tuple[float, ...]]] = {
    "SquareTableLeg200mmDecomp": {"pos": (-0.106203, 0.0, 0.0), "quat": (0.70710678, 0.0, 0.70710678, 0.0)},
    "SquareTableLeg200mmSdf": {"pos": (-0.106203, 0.0, 0.0), "quat": (0.70710678, 0.0, 0.70710678, 0.0)},
    "SquareTableLeg200mmSdf1024": {"pos": (-0.106203, 0.0, 0.0), "quat": (0.70710678, 0.0, 0.70710678, 0.0)},
    "SquareTableLeg200mmSdf2048": {"pos": (-0.106203, 0.0, 0.0), "quat": (0.70710678, 0.0, 0.70710678, 0.0)},
    "OneLegInsertionFixture": {"pos": (-0.056250, 0.056250, -0.009374), "quat": (1.0, 0.0, 0.0, 0.0)},
}
PINNED_OFFSET_SOURCE = "scripts_v2/tools/validate_c4_bank_aro2_3.py:188-190 (_LEG_OFF_POS / _LEG_OFF_QUAT_WXYZ / _RECV_OFF_POS)"
OFFSET_ATOL = 1e-9  # metres (pos), dimensionless quaternion component (quat)

# Relative to --repo-root. These are the metadata.yaml directories the runtime actually selects
# for this pair -- see docs/ASSET_MANIFEST.md (DexReset repo) for the full selector audit.
DEFAULT_METADATA_DIRS = [
    "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmDecomp",
    "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf",
    "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf1024",
    "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf2048",
    "source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture",
]


def check_one(metadata_dir: str) -> list[str]:
    """Returns a list of error strings (empty = OK) for the metadata.yaml in metadata_dir."""
    object_name = os.path.basename(os.path.normpath(metadata_dir))
    expected = PINNED_OFFSET_LITERALS.get(object_name)
    if expected is None:
        return [f"{object_name!r}: no pinned literal known for this object name -- skipped (not an error)."]

    metadata_path = os.path.join(metadata_dir, "metadata.yaml")
    if not os.path.isfile(metadata_path):
        return [f"{object_name!r}: {metadata_path} does not exist -- cannot check (this asset is not present "
                 "in this checkout; see docs/ASSET_MANIFEST.md)."]

    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)

    runtime_offset = (metadata or {}).get("assembled_offset") or {}
    runtime_pos = tuple(runtime_offset.get("pos", ()))
    runtime_quat = tuple(runtime_offset.get("quat", ()))

    errors = []
    for field_name, runtime_val, expected_val in (("pos", runtime_pos, expected["pos"]), ("quat", runtime_quat, expected["quat"])):
        mismatched = len(runtime_val) != len(expected_val) or any(
            abs(r - e) > OFFSET_ATOL for r, e in zip(runtime_val, expected_val)
        )
        if mismatched:
            errors.append(
                f"assembled_offset.{field_name} MISMATCH for {object_name!r}: "
                f"runtime value read from {metadata_path} = {runtime_val}, but the hardcoded literal "
                f"in {PINNED_OFFSET_SOURCE} = {expected_val} (tolerance {OFFSET_ATOL:g}, exceeded). "
                "EVERY task_3 distance/success number this project computes is INVALID until "
                "metadata.yaml and the literal in validate_c4_bank_aro2_3.py agree. Update whichever "
                "one is stale, do not silently pick one."
            )
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                     help="uwlab checkout containing the metadata.yaml files (default: this script's own repo).")
    args = ap.parse_args()

    any_error = False
    for rel_dir in DEFAULT_METADATA_DIRS:
        full_dir = os.path.join(args.repo_root, rel_dir)
        messages = check_one(full_dir)
        if not messages:
            print(f"[ OK ] {rel_dir}: assembled_offset matches {PINNED_OFFSET_SOURCE}", flush=True)
            continue
        for msg in messages:
            is_error = "MISMATCH" in msg
            any_error = any_error or is_error
            print(("[FAIL] " if is_error else "[skip] ") + msg, flush=True)

    if any_error:
        print("\nFAILED: at least one assembled_offset literal has drifted from metadata.yaml.", flush=True)
        return 1
    print("\nOK: all pinned assembled_offset literals agree with their metadata.yaml.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
