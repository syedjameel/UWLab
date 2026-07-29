# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify an RGB demo .zarr written by ``collect_demos.py`` -- specifically, whether a dataset
left behind by a CRASHED collection run is intact and safe to keep.

Plain python (zarr + numpy). No Isaac, no GPU. Safe to run while another collection is going.

What can actually go wrong, from ``ZarrDatasetFileHandler._save_episode_to_zarr``: each episode
first EXTENDS every data array, and only then appends to ``meta/episode_ends``. So a process
killed mid-episode leaves one of:
  * trailing rows in ``data/*`` that no episode indexes  -> harmless, consumers slice by
    episode_ends and never read them;
  * data arrays of UNEQUAL length (some extended, some not) -> the real corruption case;
  * an unwritten trailing chunk -> reads raise, which is why this script actually READS the
    last episodes rather than trusting the metadata.

    python scripts_v2/tools/conversions/verify_zarr_demos.py --dataset datasets/x/rgb0.zarr
    python scripts_v2/tools/conversions/verify_zarr_demos.py --dataset ... --read-all   # slow, thorough
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import zarr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="path to the .zarr")
    ap.add_argument("--probe-episodes", type=int, default=25,
                    help="how many episodes to fully read at each end of the file")
    ap.add_argument("--read-all", action="store_true",
                    help="read EVERY episode (forces every chunk off disk; slow but exhaustive)")
    args = ap.parse_args()

    z = zarr.open(args.dataset, mode="r")
    data, meta = z["data"], z["meta"]
    ends = np.asarray(meta["episode_ends"][:])
    n_ep = len(ends)
    total = int(ends[-1]) if n_ep else 0

    print(f"=== {args.dataset}")
    print(f"  env_name          : {z.attrs.get('env_name')}")
    print(f"  EPISODES          : {n_ep}")
    print(f"  frames (indexed)  : {total}")
    if n_ep:
        lens = np.diff(np.concatenate([[0], ends]))
        print(f"  episode length    : min {lens.min()}  median {int(np.median(lens))}  max {lens.max()}")

    problems: list[str] = []

    # 1. episode_ends must be strictly increasing (no zero-length / rewound episodes).
    if n_ep:
        bad = np.where(np.diff(np.concatenate([[0], ends])) <= 0)[0]
        if len(bad):
            problems.append(f"episode_ends not strictly increasing at episode index/indices {bad[:10].tolist()}")

    # 2. Every data array must cover the indexed frames, and they should all agree.
    print("\n  data arrays:")
    keys: list[str] = []

    def walk(group, prefix=""):
        for k in group.array_keys():
            keys.append(prefix + k)
        for k in group.group_keys():
            walk(group[k], prefix + k + "/")

    walk(data)
    lengths = {}
    for k in sorted(keys):
        arr = data[k]
        lengths[k] = arr.shape[0]
        flag = ""
        if arr.shape[0] < total:
            flag = "  <-- SHORTER than episode_ends: TRUNCATED"
            problems.append(f"array '{k}' has {arr.shape[0]} rows but episode_ends claims {total}")
        elif arr.shape[0] > total:
            flag = f"  <-- {arr.shape[0]-total} un-indexed trailing rows (harmless partial episode)"
        print(f"    {k:38s} shape={str(arr.shape):26s} dtype={arr.dtype}{flag}")
    if len(set(lengths.values())) > 1:
        problems.append(f"data arrays disagree on length: {lengths}")

    # 3. Actually READ episodes -- metadata can look fine while a chunk is missing on disk.
    def read_episode(i: int):
        lo = int(ends[i - 1]) if i > 0 else 0
        hi = int(ends[i])
        for k in keys:
            a = data[k][lo:hi]
            if a.shape[0] != hi - lo:
                raise ValueError(f"episode {i}: '{k}' returned {a.shape[0]} rows, expected {hi-lo}")
            if np.issubdtype(a.dtype, np.floating) and not np.isfinite(np.asarray(a)).all():
                raise ValueError(f"episode {i}: '{k}' contains NaN/Inf")

    if n_ep:
        if args.read_all:
            probe = list(range(n_ep))
        else:
            p = min(args.probe_episodes, n_ep)
            probe = sorted(set(list(range(p)) + list(range(max(0, n_ep - p), n_ep))))
        print(f"\n  reading {len(probe)} episodes off disk (the LAST ones are what a crash endangers)...")
        n_bad = 0
        for i in probe:
            try:
                read_episode(i)
            except Exception as e:  # noqa: BLE001 - report and continue, we want the full tally
                n_bad += 1
                problems.append(f"episode {i} unreadable: {e}")
                if n_bad <= 5:
                    print(f"    episode {i}: FAILED -- {e}")
        print(f"    {len(probe)-n_bad}/{len(probe)} episodes read cleanly")

    print()
    if problems:
        print(f"  RESULT: {len(problems)} PROBLEM(S)")
        for p in problems[:20]:
            print(f"    - {p}")
        print("\n  Un-indexed trailing rows alone are FINE (consumers slice by episode_ends).")
        print("  Truncated/disagreeing array lengths or unreadable episodes are NOT -- in that")
        print("  case treat the last episode as suspect; there is no in-place repair tool, so")
        print("  keep this file for the earlier episodes and collect the remainder into rgb1.zarr.")
        return 1
    print(f"  RESULT: OK -- {n_ep} episodes intact and readable. Safe to keep and merge.")
    print("  Top up the remainder into rgb1.zarr in the SAME directory; training merges the dir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
