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

CROSS-CHUNK DUPLICATE CHECK (team-lead requirement, chunking follow-up #2). The merge above proves
nothing about DIVERSITY -- if every chunk is a fresh process launched with an identical command
line (same checkpoint, same env count, same task, same offsets), a per-process-seeded RNG could
make two or three chunks explore the same trajectory distribution from the same initial conditions
and emit largely or entirely IDENTICAL states. The count check would still pass (count is count),
the schema would still validate, and the bank would silently carry N copies of one chunk's
diversity. This script therefore ALWAYS compares the manipulated object's root_pose across chunks
(position + orientation) and reports how many states are exact or near-exact duplicates of a state
in ANOTHER chunk, as a headline number -- reporting, not failing, since interpreting "how much
overlap is too much" is a judgment call for whoever is watching the run, not a hardcoded gate here.
Also runnable standalone with --report_only (no --output_dir needed, nothing written) for an
early check -- e.g. chunk 1 vs chunk 2 as soon as both exist, before a third chunk's GPU time is
spent on what might be a clone.

EXTENSION (bead UWLab-hls2, ported from GenRunner's independent 3d7168f on feat/realbox-assembly,
reconciled here rather than left as two diverging copies): this tool's own C2-via-rewind splits
its target into several OFFSETS per run, hence resets_<type>_off<N>s.pt and the glob below. C3
chunks come from plain generate_reset_states_policy.py, which writes exactly one
resets_<reset_type>.pt per --dataset_dir with NO offset suffix. Rather than fork a second merge
script for that shape, --filename accepts an exact basename and bypasses the c2_reset_type/_off*
glob entirely -- mutually exclusive with --c2_reset_type (an ERROR to pass both, not a silent
precedence rule -- the earlier GenRunner-only version let --c2_reset_type sit unused whenever
--filename was given, which is exactly the kind of silently-ignored flag this project has been
bitten by all day). Every other code path -- the pair-dir consistency guard, the per-file
list-concat merge, the hard count assertion, and the cross-chunk duplicate check above -- is
identical and untouched for both entry points; C3's chunks need the duplicate check at least as
much as C2's, since the seed is confirmed globally pinned across every process too.
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


def extract_object_root_poses(initial_state: dict) -> list[torch.Tensor]:
    """Every rigid_object's root_pose list, concatenated (in this schema there is exactly one --
    ``insertive_object`` -- but this does not assume that, in case a future bank ever carries
    more than one manipulated body)."""
    poses: list[torch.Tensor] = []
    for asset in initial_state.get("rigid_object", {}).values():
        poses.extend(asset["root_pose"])
    return poses


def report_cross_chunk_duplicates(
    chunk_poses: dict[str, list[torch.Tensor]], pos_tol_m: float, ang_tol_deg: float
) -> dict:
    """Compare every state's object root_pose (position + quaternion, IsaacLab's own
    xyz + wxyz layout -- see root_pose_w) against every OTHER chunk's states for the SAME offset,
    and report how many have at least one near/exact match in a DIFFERENT chunk.

    NOT a coincidence detector at these tolerances: the reset distribution this bead measured
    spans tens to hundreds of millimeters and full-range orientation (or +-0.3 rad under
    DEXLIFT_POSE_TILT) -- two INDEPENDENT random draws landing within 1 mm and 0.5 deg of each
    other by chance, in that continuous a space, is not a plausible coincidence. A large count
    here means shared RNG state across chunk processes, not luck.

    EXACT means bit-identical to float32 precision (pos_tol effectively 0); NEAR uses the caller's
    tolerances and INCLUDES exact matches (a looser superset, not a disjoint bucket) -- report both
    so "some near-misses" (expected, e.g. a common table-rest pose) reads differently from
    "everything is exact" (a cloned RNG stream).
    """
    chunk_names = list(chunk_poses.keys())
    flat_poses: list[torch.Tensor] = []
    chunk_idx: list[int] = []
    for ci, name in enumerate(chunk_names):
        for pose in chunk_poses[name]:
            flat_poses.append(pose)
            chunk_idx.append(ci)

    n_total = len(flat_poses)
    result = {
        "chunk_names": chunk_names,
        "chunk_sizes": [len(chunk_poses[c]) for c in chunk_names],
        "n_total": n_total,
        "n_near_dup": 0,
        "n_exact_dup": 0,
    }
    if n_total == 0 or len(chunk_names) < 2:
        return result

    stacked = torch.stack(flat_poses).float()  # (n_total, 7): xyz + wxyz
    pos = stacked[:, :3]
    quat = torch.nn.functional.normalize(stacked[:, 3:7], dim=-1)
    chunk_idx_t = torch.tensor(chunk_idx)

    pos_dist = torch.cdist(pos, pos)
    dot = (quat @ quat.T).clamp(-1.0, 1.0).abs()
    ang_dist_deg = torch.rad2deg(2.0 * torch.acos(dot))

    different_chunk = chunk_idx_t.unsqueeze(0) != chunk_idx_t.unsqueeze(1)
    not_self = ~torch.eye(n_total, dtype=torch.bool)
    cross_chunk = different_chunk & not_self

    near = (pos_dist < pos_tol_m) & (ang_dist_deg < ang_tol_deg) & cross_chunk
    exact = (pos_dist < 1e-6) & (ang_dist_deg < 1e-4) & cross_chunk

    result["n_near_dup"] = int(near.any(dim=1).sum().item())
    result["n_exact_dup"] = int(exact.any(dim=1).sum().item())
    return result


def print_duplicate_report(fname: str, report: dict, pos_tol_m: float, ang_tol_deg: float) -> None:
    n_total = report["n_total"]
    n_near = report["n_near_dup"]
    n_exact = report["n_exact_dup"]
    sizes = " + ".join(f"{name}={size}" for name, size in zip(report["chunk_names"], report["chunk_sizes"]))
    near_pct = (100.0 * n_near / n_total) if n_total else 0.0
    exact_pct = (100.0 * n_exact / n_total) if n_total else 0.0
    print(
        f"[dup-check] {fname}: {sizes}  ->  near-dup (<{pos_tol_m * 1000:.1f}mm, <{ang_tol_deg:.1f}deg, "
        f"cross-chunk)={n_near}/{n_total} ({near_pct:.1f}%)  exact-dup={n_exact}/{n_total} ({exact_pct:.1f}%)"
    )
    if n_total > 0 and near_pct > 20.0:
        print(
            f"[dup-check] WARNING: {near_pct:.1f}% of states in {fname} have a near/exact match in "
            "a DIFFERENT chunk -- this many coincidental matches is not plausible in this reset "
            "distribution's continuous state space. The chunks are likely sharing RNG state "
            "(per-process rather than per-run seeding) and largely duplicating each other's "
            "diversity, not adding to it. Investigate before trusting this bank at its face count."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--chunk_dirs", nargs="+", required=True,
        help="Each chunk run's --dataset_dir root (NOT the Resets/ subdir -- same value passed to generate_reset_states_policy.py for that chunk).",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Merged --dataset_dir root; created if absent. Required unless --report_only.",
    )
    parser.add_argument(
        "--c2_reset_type", default=None,
        help=(
            "Must match the --c2_reset_type every chunk was generated with. Mutually exclusive "
            "with --filename (an error to pass both). Defaults to 'ObjectAnywhereEENear' when "
            "neither this nor --filename is given."
        ),
    )
    parser.add_argument(
        "--filename", default=None,
        help=(
            "Exact resets_<Type>.pt basename to merge from each chunk dir's Resets/<pair>/ -- "
            "bypasses the c2_reset_type/_off* glob entirely. Use for non-offset-split generators "
            "(e.g. C3's plain generate_reset_states_policy.py output, resets_<ResetType>.pt with "
            "no _off<N>s suffix). Mutually exclusive with --c2_reset_type (an error to pass both)."
        ),
    )
    parser.add_argument(
        "--report_only", action="store_true",
        help=(
            "Run the load + duplicate check but do NOT merge/write anything -- for an early, "
            "no-side-effects diversity check (e.g. chunk 1 vs chunk 2 as soon as both exist, "
            "before spending GPU time on a third chunk). --output_dir not required with this."
        ),
    )
    parser.add_argument("--dup_pos_tol_m", type=float, default=1e-3, help="NEAR-duplicate position tolerance, meters.")
    parser.add_argument("--dup_ang_tol_deg", type=float, default=0.5, help="NEAR-duplicate orientation tolerance, degrees.")
    args = parser.parse_args()

    if not args.report_only and args.output_dir is None:
        print("[merge] FATAL: --output_dir is required unless --report_only is set.", file=sys.stderr)
        sys.exit(1)

    # --filename and --c2_reset_type select DIFFERENT discovery patterns (exact basename vs.
    # c2_reset_type/_off* glob) -- an ERROR to pass both, not a silent precedence rule (the
    # earlier GenRunner-only version let --c2_reset_type sit unused whenever --filename was given;
    # see this script's own module docstring for why that shape of bug is not acceptable here).
    if args.filename is not None and args.c2_reset_type is not None:
        print(
            "[merge] FATAL: --filename and --c2_reset_type are mutually exclusive -- pass exactly "
            "one (or neither, for the 'ObjectAnywhereEENear' offset-glob default).",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.filename is None and args.c2_reset_type is None:
        args.c2_reset_type = "ObjectAnywhereEENear"

    # Discover each chunk's files: <chunk_dir>/Resets/<pair>/<--filename OR resets_<c2_reset_type>_off*.pt>
    # Chunk label = "<1-based index in --chunk_dirs>:<basename>" -- used ONLY to attribute states
    # to a chunk for the duplicate check below, never written into the merged output itself.
    #
    # BUG FIXED HERE (team-lead, caught on the real C3 comparison): this used to be JUST the
    # basename (e.g. "OmniReset"), which silently COLLIDES whenever two chunk dirs differ higher up
    # the path than their last component -- exactly C3's own layout, <root>_c3chunkN/OmniReset for
    # every N. chunk_poses is a dict keyed by this label; a collision means the second chunk's poses
    # OVERWRITE the first's, so that chunk vanishes from the duplicate comparison while the merge
    # (which works off the path list, not this label) still proceeds correctly and reports the
    # right count -- a check that silently measures less than it claims. The 1-based index prefix
    # makes every label unique BY CONSTRUCTION regardless of what the chunk dirs' basenames are,
    # since args.chunk_dirs is a fixed-order list and this loop visits each position exactly once.
    per_offset_chunk_paths: dict[str, list[tuple[str, str]]] = {}  # fname -> [(chunk_label, path), ...]
    pair_dirs_seen: set[str] = set()
    for i, chunk_dir in enumerate(args.chunk_dirs, start=1):
        chunk_label = f"{i}:{os.path.basename(os.path.normpath(chunk_dir))}"
        if args.filename is not None:
            pattern = os.path.join(chunk_dir, "Resets", "*", args.filename)
        else:
            pattern = os.path.join(chunk_dir, "Resets", "*", f"resets_{args.c2_reset_type}_off*.pt")
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"[merge] WARNING: no C2 files found under {chunk_dir!r} (pattern {pattern!r})", file=sys.stderr)
            continue
        for path in matches:
            pair_dir = os.path.basename(os.path.dirname(path))
            pair_dirs_seen.add(pair_dir)
            fname = os.path.basename(path)
            per_offset_chunk_paths.setdefault(fname, []).append((chunk_label, path))

    if not per_offset_chunk_paths:
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

    if not args.report_only:
        out_pair_dir = os.path.join(args.output_dir, "Resets", pair_dir)
        os.makedirs(out_pair_dir, exist_ok=True)
        print(f"[merge] pair dir: {pair_dir}")
    else:
        print(f"[merge] pair dir: {pair_dir}  (--report_only: nothing will be written)")

    any_dup_warning = False
    for fname, chunk_paths in sorted(per_offset_chunk_paths.items()):
        merged: dict = {}
        chunk_counts: list[int] = []
        expected_total = 0
        chunk_poses: dict[str, list[torch.Tensor]] = {}
        for chunk_label, path in chunk_paths:
            data = torch.load(path, map_location="cpu", weights_only=False)
            n = _single_length(leaf_lengths(data["initial_state"]), context=path)
            chunk_counts.append(n)
            expected_total += n
            chunk_poses[chunk_label] = extract_object_root_poses(data["initial_state"])
            merge_leaf_lists(merged.setdefault("initial_state", {}), data["initial_state"])

        # LOUD, NOT SILENT, if this ever recurs under some layout neither of us has thought of
        # (team-lead requirement): the index-prefixed label above makes a collision impossible by
        # construction, so this should never fire -- but "should never" is exactly the class of
        # claim this whole campaign has spent today disproving. Checking the INVARIANT (one pose
        # list per chunk file, no fewer) rather than trusting the labeling scheme is what catches a
        # future regression here instead of silently under-counting again.
        assert len(chunk_poses) == len(chunk_paths), (
            f"{fname}: {len(chunk_paths)} chunk files but only {len(chunk_poses)} distinct labels in "
            f"chunk_poses ({sorted(chunk_poses.keys())}) -- a chunk label collided and its states "
            "were silently dropped from the duplicate check. This should be structurally impossible "
            "with index-prefixed labels; if it fires, the labeling logic itself has a new bug."
        )

        merged_total = _single_length(leaf_lengths(merged["initial_state"]), context=f"{fname} (merged)")
        # HARD count check (team-lead requirement): verify the merged count equals the sum of the
        # parts EXACTLY -- a silent partial merge (e.g. from the list-vs-stack schema mistake this
        # script's docstring warns about) would otherwise produce a plausible-looking but wrong bank.
        assert merged_total == expected_total, (
            f"{fname}: merged count {merged_total} != sum of chunk counts {expected_total} "
            f"(per-chunk: {chunk_counts}) -- refusing to write a bank whose count does not add up."
        )

        # CROSS-CHUNK DUPLICATE CHECK (team-lead requirement) -- reports, does not fail. See
        # report_cross_chunk_duplicates's own docstring for why a large overlap here is not a
        # plausible coincidence.
        dup_report = report_cross_chunk_duplicates(chunk_poses, args.dup_pos_tol_m, args.dup_ang_tol_deg)
        print_duplicate_report(fname, dup_report, args.dup_pos_tol_m, args.dup_ang_tol_deg)
        if dup_report["n_total"] and dup_report["n_near_dup"] / dup_report["n_total"] > 0.20:
            any_dup_warning = True

        if args.report_only:
            print(f"[merge] {fname}: {' + '.join(str(c) for c in chunk_counts)} = {merged_total}  (--report_only: not written)")
            continue

        out_path = os.path.join(out_pair_dir, fname)
        torch.save(merged, out_path)
        print(f"[merge] {fname}: {' + '.join(str(c) for c in chunk_counts)} = {merged_total} -> {out_path}")

    if args.report_only:
        print("[merge] REPORT-ONLY DONE -- nothing written.")
    else:
        print(f"[merge] DONE. Merged output at: {out_pair_dir}")
    if any_dup_warning:
        print(
            "[merge] SUMMARY: at least one offset crossed the 20% near-duplicate threshold -- see "
            "the [dup-check] WARNING lines above before trusting this bank's state count as real "
            "diversity.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
