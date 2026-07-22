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
    ap.add_argument("--max-grip", type=float, default=None,
                    help="Drop states with finger_joint above this (m) -- e.g. 0.045 removes jaws "
                         "closed PAST a thin object (empty grip) while keeping true width grips.")
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
    if args.min_grip is not None:
        open_jaw = jp[:, FINGER_COL] < args.min_grip
        keep &= ~open_jaw
        print(f"[FILTER] finger_joint < {args.min_grip}: dropping {int(open_jaw.sum())}/{n}")

    if args.max_grip is not None:
        closed_past = jp[:, FINGER_COL] > args.max_grip
        keep &= ~closed_past
        print(f"[FILTER] finger_joint > {args.max_grip}: dropping {int(closed_past.sum())}/{n}")

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
