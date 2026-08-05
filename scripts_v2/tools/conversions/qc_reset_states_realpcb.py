# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""QC gate for the realpcb-onto-jigenclosure reset datasets (CPU only, no Isaac).

``qc_reset_states_ur10e.py`` covers the robot side (wrist limits, FK top-down, jaw symmetry) but
its grip thresholds are calibrated for the 40 mm cube / 2 mm pcb, and it has no concept of a
pedestal or of a seat depth. This adds the GEOMETRY checks that the failures we actually hit
would have caught -- each threshold below is a measured number, not a guess:

  P  pedestal placement    C1/C2 the cube must be UNDER the board; C3/C4 it must be PARKED.
                           A kinematic cube inside the fixture cannot be pushed out by physics.
  H  board actually HELD   *EEGrasped types. check_reset_state_success validates STABILITY, not
                           attachment, and a dropped board lying still is maximally stable
                           (documented for the jig: "Near Goal 66% open-jaw ... NO jaws-on-object
                           condition"). Measured here: 1/69 C3 boards at z=4.96 mm, flat on the
                           table, while held ones span 17-300 mm.
  S  board ON the pedestal C1/C2. The +-0.1 rad roll/pitch jitter tips a 3 mm board off a 40 mm
                           cube; measured 8/57 in an earlier batch.
  A  board not pre-seated  C4. If a "partial" assembly already satisfies the success threshold
                           the episode starts solved and teaches nothing.
  U  board upright         C1/C2 -- mirrors the jig's --require-upright filter.
  F  clear of the fixture  C1/C2 board and pedestal vs the goal fixture footprint.

Exit code 1 on any FAIL, so it can gate a recording run.

    python scripts_v2/tools/conversions/qc_reset_states_realpcb.py \
      --dataset_dir ./Datasets_realpcb_jig/OmniReset
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

GRASPED = {"ObjectRestingEEGrasped", "ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped"}
RESTING_ON_PEDESTAL = {"ObjectAnywhereEEAnywhere", "ObjectRestingEEGrasped"}
PARKED = {"ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped"}

# --- measured geometry (mm) -------------------------------------------------------------
TABLE_TOP = 4.0          # work surface in the robot base frame
CUBE_H = 40.0            # pedestal height (Props/Custom/Pcb = 40 mm cube)
BOARD_HALF_T = 1.5       # slab collider half-thickness
BOARD_ON_CUBE_Z = TABLE_TOP + CUBE_H + BOARD_HALF_T          # 45.5, verified across 65/65 C2
CUBE_HALF = 20.0
BOARD_HALF = (70.0, 50.0)
FIXTURE_HALF_DIAG = 104.0
BOARD_HALF_DIAG = 86.0
MIN_SEP = FIXTURE_HALF_DIAG + BOARD_HALF_DIAG                # 190
SEAT_Z = 13.6            # assembled seat above the enclosure bottom
SUCCESS_POS = 5.0        # position success threshold (metadata)


def load(path):
    st = torch.load(path, map_location="cpu", weights_only=False)["initial_state"]
    ro = st["rigid_object"]
    out = {}
    for name in ("insertive_object", "receptive_object", "pedestal"):
        if name in ro:
            out[name] = torch.stack([p.detach().cpu().float() for p in ro[name]["root_pose"]]).numpy()
    q = st["articulation"]["robot"]["joint_position"]
    out["q"] = torch.stack([t.detach().cpu().float() for t in q]).numpy()
    return out


def _corners(pos, quat, hx, hy):
    """Footprint corners (N,4,2) of a box, honouring each state's yaw."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    a = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s_ = np.cos(a), np.sin(a)
    R = np.stack([np.stack([c, -s_], -1), np.stack([s_, c], -1)], -2)
    loc = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]]) / 1000.0
    return pos[:, None, :2] + np.einsum("nij,kj->nki", R, loc)


def overlaps(A, B):
    """Separating-axis test per state. Centre distance is NOT a valid proxy: gating on the sum
    of half-DIAGONALS (190 mm here) rejects poses whose real footprints are 8 mm apart, which
    produced 9/65 phantom C2 failures until this was measured properly."""
    out = np.ones(len(A), bool)
    for P, Q_ in ((A, B), (B, A)):
        for i in range(4):
            e = P[:, (i + 1) % 4] - P[:, i]
            ax = np.stack([-e[:, 1], e[:, 0]], -1)
            ax = ax / np.linalg.norm(ax, axis=1)[:, None]
            pa = np.einsum("nki,ni->nk", P, ax)
            pb = np.einsum("nki,ni->nk", Q_, ax)
            out &= ~((pa.max(1) < pb.min(1)) | (pb.max(1) < pa.min(1)))
    return out


def rp_deg(quat):
    """roll/pitch magnitude in degrees from wxyz quats (N,4): angle of body +Z off world +Z."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    zz = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(zz, -1.0, 1.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="./Datasets_realpcb_jig/OmniReset")
    ap.add_argument("--pair", default="JigEnclosure__RealPcb")
    ap.add_argument("--held-max-mm", type=float, default=120.0,
                    help="max board-to-fingertip distance for a *EEGrasped state")
    ap.add_argument("--upright-max-deg", type=float, default=20.0)
    ap.add_argument("--seat-margin-mm", type=float, default=5.0,
                    help="C4 partial assemblies must be at least this far from the seated pose")
    args = ap.parse_args()

    base = os.path.join(args.dataset_dir, "Resets", args.pair)
    files = sorted(glob.glob(os.path.join(base, "resets_*.pt")))
    if not files:
        print(f"[QC] no reset files under {base}")
        sys.exit(1)

    fail = False
    for f in files:
        t = os.path.basename(f)[len("resets_"):-len(".pt")]
        d = load(f)
        obj, rec, ped = d["insertive_object"], d["receptive_object"], d.get("pedestal")
        n = len(obj)
        print(f"\n=== {t}  (n={n})")
        msgs = []

        # ---- P: pedestal placement -------------------------------------------------
        if ped is None:
            msgs.append("FAIL P: no pedestal recorded in this dataset")
        else:
            d_op = np.linalg.norm(obj[:, :2] - ped[:, :2], axis=1) * 1000
            d_pr = np.linalg.norm(ped[:, :2] - rec[:, :2], axis=1) * 1000
            print(f"  P pedestal: under-board {d_op.min():6.1f}..{d_op.max():7.1f} mm | "
                  f"to-fixture min {d_pr.min():6.1f} mm")
            if t in RESTING_ON_PEDESTAL:
                bad = d_op > 10.0
                if bad.any():
                    msgs.append(f"FAIL P: {bad.sum()}/{n} pedestals not under the board (>10 mm)")
            if t in PARKED:
                bad = overlaps(_corners(ped, ped[:, 3:7], CUBE_HALF, CUBE_HALF),
                               _corners(rec, rec[:, 3:7], 82.0, 64.5))
                if bad.any():
                    msgs.append(f"FAIL P: {bad.sum()}/{n} parked pedestals intersect the fixture")

        # ---- S: board actually ON the pedestal -------------------------------------
        if t in RESTING_ON_PEDESTAL:
            z = obj[:, 2] * 1000
            off = np.abs(z - BOARD_ON_CUBE_Z) > 3.0
            inxy = (np.abs(obj[:, 0] - ped[:, 0]) * 1000 <= CUBE_HALF + 2) & \
                   (np.abs(obj[:, 1] - ped[:, 1]) * 1000 <= CUBE_HALF + 2) if ped is not None else np.ones(n, bool)
            print(f"  S on-pedestal: z {z.min():6.2f}..{z.max():6.2f} mm (want {BOARD_ON_CUBE_Z:.1f}) | "
                  f"off {int(off.sum())}/{n}")
            if off.any():
                msgs.append(f"FAIL S: {off.sum()}/{n} boards not resting on the cube (z off by >3 mm)")
            if (~inxy).any():
                msgs.append(f"WARN S: {int((~inxy).sum())}/{n} boards overhang the cube centre by >20 mm")

        # ---- U: upright ------------------------------------------------------------
        if t in RESTING_ON_PEDESTAL:
            tilt = rp_deg(obj[:, 3:7])
            print(f"  U upright: tilt median {np.median(tilt):5.2f} deg, max {tilt.max():5.2f}")
            bad = tilt > args.upright_max_deg
            if bad.any():
                msgs.append(f"FAIL U: {bad.sum()}/{n} boards tilted beyond {args.upright_max_deg} deg")

        # ---- F: clear of the fixture ------------------------------------------------
        if t in RESTING_ON_PEDESTAL:
            fx = _corners(rec, rec[:, 3:7], 82.0, 64.5)
            ob = overlaps(_corners(obj, obj[:, 3:7], *BOARD_HALF), fx)
            pb = overlaps(_corners(ped, ped[:, 3:7], CUBE_HALF, CUBE_HALF), fx) if ped is not None \
                else np.zeros(n, bool)
            d_or = np.linalg.norm(obj[:, :2] - rec[:, :2], axis=1) * 1000
            print(f"  F fixture overlap: board {int(ob.sum())}/{n}, pedestal {int(pb.sum())}/{n} "
                  f"(centre-dist min {d_or.min():.1f} mm, informational)")
            if ob.any():
                msgs.append(f"FAIL F: {ob.sum()}/{n} board footprints intersect the fixture")
            if pb.any():
                msgs.append(f"FAIL F: {pb.sum()}/{n} pedestal footprints intersect the fixture")

        # ---- H: actually held -------------------------------------------------------
        if t in GRASPED:
            # cheap, robot-free proxy: a held board cannot be sitting on the table surface
            z = obj[:, 2] * 1000
            dropped = z < (TABLE_TOP + 2 * BOARD_HALF_T + 5.0)
            print(f"  H held: board z {z.min():7.2f}..{z.max():7.2f} mm | "
                  f"on-table {int(dropped.sum())}/{n}")
            if dropped.any():
                msgs.append(f"FAIL H: {dropped.sum()}/{n} boards lying on the table, not held")

        # ---- A: C4 must not already be seated ---------------------------------------
        if t == "ObjectPartiallyAssembledEEGrasped":
            rel = (obj[:, :3] - rec[:, :3]) * 1000
            # distance from the seated pose: seat is directly above the fixture centre
            dz = np.abs(rel[:, 2] - (SEAT_Z - 20.8))
            dxy = np.linalg.norm(rel[:, :2], axis=1)
            seated = (dxy < SUCCESS_POS) & (dz < SUCCESS_POS)
            print(f"  A pre-seated: {int(seated.sum())}/{n} already within the {SUCCESS_POS} mm "
                  f"success threshold (xy min {dxy.min():.1f}, dz min {dz.min():.1f})")
            fj = d["q"][:, 6] * 1000
            openj = int((fj < 1.0).sum())
            print(f"  A jaws OPEN (grasp never engaged): {openj}/{n} -- a board inside the "
                  f"102 mm window cannot be gripped (jaws need +-50 mm plus pads)")
            if openj > 0.1 * n:
                msgs.append(f"FAIL A: {openj}/{n} states have jaws OPEN -- failed grasps")
            if seated.sum() > 0.25 * n:
                msgs.append(f"FAIL A: {seated.sum()}/{n} partial assemblies are ALREADY SEATED "
                            f"-- those episodes start solved")

        for m in msgs:
            print(f"      {m}")
            if m.startswith("FAIL"):
                fail = True

    print(f"\n[QC_RESULT] {'[FAIL] see FAIL lines above' if fail else '[PASS] all checks passed'}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
