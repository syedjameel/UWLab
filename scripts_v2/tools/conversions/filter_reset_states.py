# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Post-hoc filter for recorded reset-state files (CPU-only, no Isaac).

Salvages datasets without re-recording:
  * ``--drop-wrist-beyond``: drop states with any |wrist_1/2/3| > 180 deg + 0.1 deg.
    States recorded on the old +-360 USD but WITHIN +-180 load identically on the new
    USD (limits do not change dynamics away from limits); only the beyond-limit states
    clamp mid-teleport and must go.
  * ``--wrist3-window LO HI`` (deg): keep only states with wrapped wrist_3 inside the
    window -- real-rig cable constraint: the wrist-camera mount must face the viewer
    side (wrist_3 within +-60 deg of the -90 home -> ``--wrist3-window -150 -30``).
  * ``--min-grip Q``: drop states whose finger_joint < Q (e.g. 0.03) -- removes the
    open-jaw "Near Goal" hovers that check_reset_state_success accepts because it has
    no jaws-on-object condition.
  * ``--min-object-z Z`` (m): for the *EEGrasped types, drop states whose insertive
    object is lying on the table instead of being HELD. Same root cause as --min-grip:
    check_reset_state_success validates STABILITY, and a dropped object lying still is
    maximally stable, so a failed grasp is accepted. --min-grip cannot catch it on a thin
    board (jaws closed on a 3 mm board vs closed on air differ by 3 mm of finger travel),
    but height separates the cases with huge margin -- measured on realpcb C3: dropped
    boards at z = 4.96 mm, held ones spanning 17-300 mm. 0.012 is a safe cut (a flat board
    sits at ~5.5 mm; C2 boards held on the 40 mm pedestal sit at 45.5 mm and are kept).

Writes ``<input>.filtered.pt`` next to the input (or ``--output``); prints the kept
fraction. Run the QC afterwards to confirm.

    python scripts_v2/tools/conversions/filter_reset_states.py \
        --input Datasets_ur10e/OmniReset/Resets/OpenBox__Pcb/resets_ObjectAnywhereEEAnywhere.pt \
        --drop-wrist-beyond
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

# Isaac articulation joint order for the graft: pan, lift, elbow, w1, w2, w3, finger, right_finger
WRIST_COLS = [3, 4, 5]
FINGER_COL = 6


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
    ap.add_argument("--drop-wrist-beyond", action="store_true")
    ap.add_argument(
        "--wrist3-window", type=float, nargs=2, default=None, metavar=("LO_DEG", "HI_DEG"),
        help="Keep only states whose wrist_3 (wrapped to (-180,180]) lies inside [LO,HI] deg. "
        "Real-rig cable constraint (2026-07-17): the wrist-camera mount must face the viewer/"
        "front-camera side, i.e. wrist_3 within +-60 deg of the -90 deg home -> '-150 -30'. "
        "The recorded grasped-EE yaw spans 180 deg, so about half the states point the mount "
        "away; the student would imitate that.",
    )
    ap.add_argument("--min-grip", type=float, default=None, help="Drop states with finger_joint below this (m).")
    ap.add_argument("--max-pedestal-offset", type=float, default=None, metavar="D",
                    help="Drop states where the insertive object is further than D metres (in xy) "
                         "from the PEDESTAL, i.e. no longer sitting on it. This is the invariant "
                         "the C1/C2 states are supposed to satisfy, and it catches what height "
                         "and tilt cannot: a board that slid onto the jig's top face rests at "
                         "4 + 41.6 + 1.5 = 47.1 mm, inside any sane z-band around the 45.5 mm "
                         "pedestal height, yet its footprint intersects the goal fixture. "
                         "0.02 = the 40 mm cube's half-width.")
    ap.add_argument("--max-object-z", type=float, default=None, metavar="Z",
                    help="Drop states whose object is ABOVE this height (m). With --min-object-z "
                         "this forms a band: for realpcb C1/C2 the board rests on the 40 mm cube "
                         "at 0.0455, so 0.0425..0.0485 keeps only boards still ON the pedestal. "
                         "Boards that tip off during settle land anywhere -- measured at 10k "
                         "scale: z -1.19..186.23 mm, tilt up to 180 deg, pedestal separating by "
                         "up to 650 mm, and 22 landing on the goal fixture.")
    ap.add_argument("--min-object-z", type=float, default=None, metavar="Z",
                    help="Drop *EEGrasped states whose object is below this height (m) -- i.e. "
                         "lying on the table rather than held. 0.012 for realpcb.")
    ap.add_argument("--require-upright", type=float, default=None, metavar="MAXDEG",
                    help="Drop states whose insertive object is tilted more than this (deg roll/pitch). "
                         "E.g. 20 removes flipped/tipped parts (spawn-interpenetration launches).")
    ap.add_argument("--obj-x-min", type=float, default=None,
                    help="Drop states with the insertive object's x below this (m) -- e.g. 0.34 "
                         "removes parts launched off the workspace toward the robot base.")
    ap.add_argument("--in-place", action="store_true", help="Overwrite the input file (keeps a .bak copy).")
    args = ap.parse_args()

    data = torch.load(args.input, map_location="cpu", weights_only=False)
    robot = data["initial_state"]["articulation"]["robot"]
    jp = torch.stack([t.cpu() for t in robot["joint_position"]]).numpy()
    n = jp.shape[0]
    keep = np.ones(n, dtype=bool)

    if args.drop_wrist_beyond:
        tol = np.radians(0.1)
        beyond = (np.abs(jp[:, WRIST_COLS]) > np.pi + tol).any(axis=1)
        keep &= ~beyond
        print(f"[FILTER] wrist beyond +-180: dropping {int(beyond.sum())}/{n}")
    if args.wrist3_window is not None:
        lo, hi = np.radians(args.wrist3_window)
        w3 = jp[:, WRIST_COLS[2]]
        w3_wrapped = np.mod(w3 + np.pi, 2 * np.pi) - np.pi  # wrap to (-pi, pi]
        outside = (w3_wrapped < lo) | (w3_wrapped > hi)
        keep &= ~outside
        print(
            f"[FILTER] wrist_3 outside [{args.wrist3_window[0]:.0f}, {args.wrist3_window[1]:.0f}] deg: "
            f"dropping {int(outside.sum())}/{n}"
        )
    if args.max_pedestal_offset is not None:
        ro = data["initial_state"]["rigid_object"]
        if "pedestal" not in ro:
            print("[FILTER] --max-pedestal-offset: no pedestal in this dataset, skipping")
        else:
            O = torch.stack([t.cpu() for t in ro["insertive_object"]["root_pose"]]).numpy()
            Pd = torch.stack([t.cpu() for t in ro["pedestal"]["root_pose"]]).numpy()
            off = np.linalg.norm(O[:, :2] - Pd[:, :2], axis=1)
            bad = off > args.max_pedestal_offset
            keep &= ~bad
            print(f"[FILTER] object further than {args.max_pedestal_offset} m from the pedestal "
                  f"(slid off it): dropping {int(bad.sum())}/{n}")

    if args.min_object_z is not None or args.max_object_z is not None:
        rp_list = data["initial_state"]["rigid_object"]["insertive_object"]["root_pose"]
        Pz = torch.stack([t.cpu() for t in rp_list]).numpy()[:, 2]
        if args.min_object_z is not None:
            bad = Pz < args.min_object_z
            keep &= ~bad
            print(f"[FILTER] insertive object z < {args.min_object_z} (on the table, not held): "
                  f"dropping {int(bad.sum())}/{n}")
        if args.max_object_z is not None:
            bad = Pz > args.max_object_z
            keep &= ~bad
            print(f"[FILTER] insertive object z > {args.max_object_z} (off the pedestal): "
                  f"dropping {int(bad.sum())}/{n}")

    if args.require_upright is not None or args.obj_x_min is not None:
        import math
        rp_list = data["initial_state"]["rigid_object"]["insertive_object"]["root_pose"]
        P = torch.stack([t.cpu() for t in rp_list]).numpy()
        if args.require_upright is not None:
            qw, qx, qy, qz = P[:, 3], P[:, 4], P[:, 5], P[:, 6]
            # roll/pitch magnitude via the world-z of the object's z-axis: cos(tilt)
            cz = 1.0 - 2.0 * (qx * qx + qy * qy)
            tilt = np.degrees(np.arccos(np.clip(cz, -1.0, 1.0)))
            bad = tilt > args.require_upright
            keep &= ~bad
            print(f"[FILTER] insertive object tilted > {args.require_upright} deg: dropping {int(bad.sum())}/{n}")
        if args.obj_x_min is not None:
            bad = P[:, 0] < args.obj_x_min
            keep &= ~bad
            print(f"[FILTER] insertive object x < {args.obj_x_min}: dropping {int(bad.sum())}/{n}")

    if args.min_grip is not None:
        open_jaw = jp[:, FINGER_COL] < args.min_grip
        keep &= ~open_jaw
        print(f"[FILTER] finger_joint < {args.min_grip}: dropping {int(open_jaw.sum())}/{n}")

    kept = int(keep.sum())
    print(f"[FILTER] keeping {kept}/{n} states ({100 * kept / n:.1f}%)")
    if kept == n:
        print("[FILTER] nothing to drop -- no output written")
        return
    if kept == 0:
        print("[FILTER] would drop EVERYTHING -- aborting, no output written")
        return

    filtered = apply_mask(data, keep, n)
    if args.in_place:
        import shutil

        shutil.copyfile(args.input, args.input + ".bak")
        out = args.input
        print(f"[FILTER] backup: {args.input}.bak")
    else:
        out = args.output or args.input.replace(".pt", ".filtered.pt")
    torch.save(filtered, out)
    print(f"[FILTER] wrote {out}")


if __name__ == "__main__":
    main()
