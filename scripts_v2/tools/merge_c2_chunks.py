#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Merge C2-via-rewind per-offset .pt files from several CHUNKED generate_reset_states_policy.py
runs into one bank per offset (bead UWLab-weyl, chunking follow-up).

WHY CHUNKED IN THE FIRST PLACE. The generator rewrites its ENTIRE accept-time bank on every
accepted state (measured on the C3 paper-scale run: windowed rate fell 0.97 -> 0.645 -> 0.395 ->
0.327 accepted/s as n grew, fitting rate ~ 1034/(466+n) -- per-state cost is linear in n, so total
cost is quadratic). C2-via-rewind pays that SAME cost on its accept-time bank even though that bank
is a throwaway for this purpose. Splitting a large --num_reset_states target into several smaller
chunk runs, each writing to its OWN isolated scratch directory, avoids paying the quadratic tail
and bounds a crash's blast radius to one chunk instead of the whole run (this tool's C2 files are
written ONCE, by write(), at the very end -- a crash loses everything accumulated in that process).
This script is the other half of that plan: it recombines the chunks' independent per-offset
outputs back into one bank per offset, since nothing downstream should have to know the paper-scale
run was chunked.

THE SCHEMA THIS RELIES ON, get this wrong and it silently produces garbage or loudly crashes with
"unexpected type list". Each resets_<reset_type>_off<X>s.pt file is:

    {"initial_state": {
        "articulation": {name: {field: [tensor, tensor, ...]}},
        "rigid_object":  {name: {field: [tensor, tensor, ...]}},
    }}

-- LISTS of per-state tensors (no leading batch dim on each tensor; confirmed against an existing
bank via this bead's own torch.load probe and TorchDatasetFileHandler._extend_dicts_last_entry,
which is what produced this shape in the first place). Merging two chunks' files for the SAME
offset means CONCATENATING those lists end to end, never stacking or torch.cat-ing the tensors
inside them -- there is no batch dimension to concatenate along, only a python list to extend.

NO ISAAC IMPORT. Plain torch, runs with any interpreter that has torch installed -- does not need
the Isaac-enabled python this project's other generator/rekey tools require.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch


def merge_leaf_lists(dest: dict, src: dict, path: str = "") -> None:
    """Mutate `dest` in place, extending every leaf LIST with the matching leaf list from `src`.
    Recurses through the nested articulation/rigid_object/name/field structure; raises loudly
    (rather than silently misbehaving) the moment a leaf is not a list, since that means the
    upstream schema changed underneath this script."""
    for key, value in src.items():
        sub_path = f"{path}/{key}"
        if isinstance(value, dict):
            merge_leaf_lists(dest.setdefault(key, {}), value, sub_path)
        else:
            if not isinstance(value, list):
                raise TypeError(
                    f"expected a LIST of per-state tensors at {sub_path!r} (this on-disk schema "
                    "stores un-stacked lists -- see this script's own module docstring), got "
                    f"{type(value).__name__} instead -- did the upstream schema change?"
                )
            dest.setdefault(key, []).extend(value)


def leaf_lengths(d: dict, path: str = "") -> dict[str, int]:
    """Every leaf list's length, keyed by its full path -- used to assert every field of one
    file/merge carries the SAME state count (a torn write would desync them silently otherwise)."""
    out: dict[str, int] = {}
    for key, value in d.items():
        sub_path = f"{path}/{key}"
        if isinstance(value, dict):
            out.update(leaf_lengths(value, sub_path))
        else:
            out[sub_path] = len(value)
    return out


def _single_length(lengths: dict[str, int], context: str) -> int:
    distinct = set(lengths.values())
    if len(distinct) > 1:
        raise ValueError(f"{context}: leaf lists disagree on length: {lengths}")
    return next(iter(distinct)) if distinct else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--chunk_dirs", nargs="+", required=True,
        help="Each chunk run's --dataset_dir root (NOT the Resets/ subdir -- same value passed to generate_reset_states_policy.py for that chunk).",
    )
    parser.add_argument("--output_dir", required=True, help="Merged --dataset_dir root; created if absent.")
    parser.add_argument(
        "--c2_reset_type", default="ObjectAnywhereEENear",
        help="Must match the --c2_reset_type every chunk was generated with.",
    )
    args = parser.parse_args()

    # Discover each chunk's per-offset files: <chunk_dir>/Resets/<pair>/resets_<c2_reset_type>_off*.pt
    per_offset_paths: dict[str, list[str]] = {}
    pair_dirs_seen: set[str] = set()
    for chunk_dir in args.chunk_dirs:
        pattern = os.path.join(chunk_dir, "Resets", "*", f"resets_{args.c2_reset_type}_off*.pt")
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"[merge] WARNING: no C2 files found under {chunk_dir!r} (pattern {pattern!r})", file=sys.stderr)
            continue
        for path in matches:
            pair_dir = os.path.basename(os.path.dirname(path))
            pair_dirs_seen.add(pair_dir)
            fname = os.path.basename(path)
            per_offset_paths.setdefault(fname, []).append(path)

    if not per_offset_paths:
        print("[merge] FATAL: no C2 files found in ANY chunk dir -- nothing to merge.", file=sys.stderr)
        sys.exit(1)
    if len(pair_dirs_seen) != 1:
        print(
            f"[merge] FATAL: chunks disagree on the pair directory name: {sorted(pair_dirs_seen)} -- "
            "refusing to guess which is correct (a mismatched insertive/receptive USD path between "
            "chunk invocations would produce this).",
            file=sys.stderr,
        )
        sys.exit(1)
    pair_dir = next(iter(pair_dirs_seen))

    out_pair_dir = os.path.join(args.output_dir, "Resets", pair_dir)
    os.makedirs(out_pair_dir, exist_ok=True)
    print(f"[merge] pair dir: {pair_dir}")

    for fname, paths in sorted(per_offset_paths.items()):
        merged: dict = {}
        chunk_counts: list[int] = []
        expected_total = 0
        for path in paths:
            data = torch.load(path, map_location="cpu", weights_only=False)
            n = _single_length(leaf_lengths(data["initial_state"]), context=path)
            chunk_counts.append(n)
            expected_total += n
            merge_leaf_lists(merged.setdefault("initial_state", {}), data["initial_state"])

        merged_total = _single_length(leaf_lengths(merged["initial_state"]), context=f"{fname} (merged)")
        # HARD count check (team-lead requirement): verify the merged count equals the sum of the
        # parts EXACTLY -- a silent partial merge (e.g. from the list-vs-stack schema mistake this
        # script's docstring warns about) would otherwise produce a plausible-looking but wrong bank.
        assert merged_total == expected_total, (
            f"{fname}: merged count {merged_total} != sum of chunk counts {expected_total} "
            f"(per-chunk: {chunk_counts}) -- refusing to write a bank whose count does not add up."
        )

        out_path = os.path.join(out_pair_dir, fname)
        torch.save(merged, out_path)
        print(f"[merge] {fname}: {' + '.join(str(c) for c in chunk_counts)} = {merged_total} -> {out_path}")

    print(f"[merge] DONE. Merged output at: {out_pair_dir}")


if __name__ == "__main__":
    main()
