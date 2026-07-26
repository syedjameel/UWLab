#!/usr/bin/env python3
"""Keep only the grasps whose commanded jaw aperture falls inside a window.

The linear-gripper grasp sampler proposes every antipodal pair it finds on the visual mesh,
including pairs that clamp thin hollow shells. On the RealBox cap those are the louver-band
grasps (aperture ~70 mm, two 0.54 mm walls with air between): the jaws squeeze straight through
and the 30 g cap is ejected, so ~100% of them record as empty-jaw reset states. The outer-wall
grasps (aperture ~127 mm, solid 1.5 mm side walls) hold. This trims the dataset to the modes
that survive contact.

    python scripts_v2/tools/filter_grasps_by_aperture.py \
        --input Datasets/OmniReset/Grasps/CapRim/grasps.pt --min-aperture 0.100 --in-place
"""

from __future__ import annotations

import argparse
import shutil

import numpy as np
import torch

STROKE = 0.137  # fully-open jaw aperture of the linear gripper (m)


def apply_mask(node, keep: np.ndarray, n: int):
    """Recursively filter every list of length n by the keep mask."""
    if isinstance(node, dict):
        return {k: apply_mask(v, keep, n) for k, v in node.items()}
    if isinstance(node, list) and len(node) == n:
        return [item for item, k in zip(node, keep) if k]
    return node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None, help="Default: <input>.filtered.pt")
    ap.add_argument("--min-aperture", type=float, default=None, help="Drop grasps narrower than this (m).")
    ap.add_argument("--max-aperture", type=float, default=None, help="Drop grasps wider than this (m).")
    ap.add_argument("--in-place", action="store_true", help="Overwrite the input file (keeps a .bak copy).")
    args = ap.parse_args()

    data = torch.load(args.input, map_location="cpu", weights_only=False)
    joints = data["grasp_relative_pose"]["gripper_joint_positions"]
    fj = np.array([float(x) for x in joints["finger_joint"]])
    rf = np.array([float(x) for x in joints["right_finger_joint"]])
    aperture = STROKE - fj - rf
    n = len(aperture)
    keep = np.ones(n, dtype=bool)

    if args.min_aperture is not None:
        too_narrow = aperture < args.min_aperture
        keep &= ~too_narrow
        print(f"[FILTER] aperture < {1000 * args.min_aperture:.0f} mm: dropping {int(too_narrow.sum())}/{n}")
    if args.max_aperture is not None:
        too_wide = aperture > args.max_aperture
        keep &= ~too_wide
        print(f"[FILTER] aperture > {1000 * args.max_aperture:.0f} mm: dropping {int(too_wide.sum())}/{n}")

    kept = int(keep.sum())
    print(
        f"[FILTER] keeping {kept}/{n} grasps ({100 * kept / n:.1f}%) | "
        f"aperture {1000 * aperture[keep].min():.1f}..{1000 * aperture[keep].max():.1f} mm"
        if kept
        else f"[FILTER] keeping 0/{n}"
    )
    if kept == n:
        print("[FILTER] nothing to drop -- no output written")
        return
    if kept == 0:
        print("[FILTER] would drop EVERYTHING -- aborting, no output written")
        return

    filtered = apply_mask(data, keep, n)
    out = args.input if args.in_place else (args.output or args.input.replace(".pt", ".filtered.pt"))
    if args.in_place:
        shutil.copy2(args.input, args.input + ".bak")
        print(f"[FILTER] backup written to {args.input}.bak")
    torch.save(filtered, out)
    print(f"[FILTER] wrote {out}")


if __name__ == "__main__":
    main()
