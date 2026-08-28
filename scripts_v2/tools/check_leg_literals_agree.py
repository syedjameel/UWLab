#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Standalone check: do the five hardcoded SquareTableLeg200mm* literals (rl_state_cfg.py,
reset_states_cfg.py, grasp_sampling_cfg.py, partial_assemblies_cfg.py,
dexlift_ur5e_delto_tableleg_env_cfg.py's TABLE_LEG_USD_PATH) all name the same leg variant?

Unlike the assembled_offset check, this one needs no separate reimplementation: the real check
(``uwlab_assets.assert_omnireset_leg_literals_agree``) already lives in a lightweight module with no
Isaac/omni/pxr imports, so this script just imports and calls it -- there is exactly one copy of the
check's logic, which is the point (duplicating it here would recreate the same "two copies that can
silently drift" defect class this whole audit exists to close).

Exit code 0 = all five agree. Exit code 1 = mismatch (or a file could not be read) -- see the printed
RuntimeError for path:line -> variant per file.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                     help="uwlab checkout containing the five literal-bearing files (default: this script's own repo).")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo_root, "source", "uwlab_assets"))
    import uwlab_assets  # noqa: E402  (path must be set up first)

    try:
        uwlab_assets.assert_omnireset_leg_literals_agree(repo_root=args.repo_root)
    except RuntimeError as exc:
        print(f"[FAIL] {exc}", flush=True)
        return 1

    print("[ OK ] all five SquareTableLeg200mm* literals name the same leg variant.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
