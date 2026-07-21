# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Normalize recorded reset states into the wrist-camera window by the gripper's 2-fold symmetry.

The D3 wrist-camera anchor (``wrist_camera_window`` termination, window [-84,+36] deg) instantly
terminates training episodes whose reset state STARTS outside the window -- and the reset datasets
were recorded without it (measured: only 0-44% of states start inside; C4 near-goal is 0% inside,
one +151 deg cluster). Because the parallel-jaw gripper is 2-fold symmetric about its approach
axis, rotating wrist_3 by 180 deg and SWAPPING the two jaw joint values yields the physically
identical grasp with the camera mount flipped to the other side. This tool applies that flip to
every out-of-window state whose image lands inside the window, and drops the rest (angles in the
uncoverable 120 deg band). Run it on all four resets_*.pt before Stage-1 training with D3.

    python scripts_v2/tools/conversions/flip_wrist_into_window.py --in-place \
      --input Datasets_realpcb/OmniReset/Resets/RealOpenBox__RealPcb/resets_ObjectAnywhereEEAnywhere.pt
"""

from __future__ import annotations

import argparse
import math
import os
import shutil

import numpy as np
import torch

WRIST3_COL = 5          # UR joint order: pan, lift, elbow, w1, w2, w3, finger, right_finger
FINGER_COLS = (6, 7)    # swapped for flipped states (2-fold jaw symmetry)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def apply_mask(node, keep: np.ndarray, n: int):
    if isinstance(node, dict):
        return {k: apply_mask(v, keep, n) for k, v in node.items()}
    if isinstance(node, list) and len(node) == n:
        return [item for item, k in zip(node, keep) if k]
    return node


def main():
    ap = argparse.ArgumentParser(description="Flip wrist_3 by 180 deg (+ jaw swap) into the camera window.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None, help="Default: <input>.flipped.pt")
    ap.add_argument("--window", type=float, nargs=2, default=(-84.0, 36.0), metavar=("LO", "HI"),
                    help="Wrist_3 window in DEG (default: the D3 camera window -84 36).")
    ap.add_argument("--in-place", action="store_true", help="Overwrite the input (keeps a .bak copy).")
    args = ap.parse_args()

    lo, hi = math.radians(args.window[0]), math.radians(args.window[1])
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    robot = data["initial_state"]["articulation"]["robot"]
    jp = robot["joint_position"]
    n = len(jp)

    flipped = kept_asis = 0
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        q = jp[i]
        w3 = wrap(float(q[WRIST3_COL]))
        if lo <= w3 <= hi:
            q[WRIST3_COL] = w3  # store wrapped (equivalent angle, inside +-180 limits)
            kept_asis += 1
            continue
        w3f = wrap(w3 + math.pi)
        if lo <= w3f <= hi:
            q[WRIST3_COL] = w3f
            a, b = FINGER_COLS
            q[a], q[b] = q[b].clone(), q[a].clone()
            jv = robot["joint_velocity"][i]
            jv[a], jv[b] = jv[b].clone(), jv[a].clone()
            flipped += 1
        else:
            keep[i] = False

    dropped = n - kept_asis - flipped
    data["initial_state"] = apply_mask(data["initial_state"], keep, n)

    out = args.input if args.in_place else (args.output or args.input.replace(".pt", ".flipped.pt"))
    if args.in_place:
        shutil.copy2(args.input, args.input + ".bak")
    torch.save(data, out)
    print(f"[FLIP] {os.path.basename(args.input)}: N={n}  in-window as-is={kept_asis}  "
          f"flipped-in={flipped}  dropped={dropped}  -> kept {kept_asis + flipped} ({(kept_asis + flipped) / n * 100:.1f}%)")
    print(f"[FLIP] wrote {out}" + (f" (backup: {args.input}.bak)" if args.in_place else ""))


if __name__ == "__main__":
    main()
