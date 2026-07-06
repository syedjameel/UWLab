# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Quality-check UR10e + linear-gripper reset-state datasets. CPU-only (torch + numpy +
yaml, NO Isaac) -- runs anywhere, including the A100 while other jobs hold the GPUs.

Checks per reset file (thresholds from the P7 QC + the wrist-limit hardening):
  * joint columns sane (2 finger columns in [0, 0.068], 6 arm columns = URDF order)
  * WRIST LIMITS: 0 states with |wrist_1/2/3| > 180 deg (post ee915f8 USDs enforce +-180;
    a violating state clamps mid-teleport when loaded)
  * top-down fraction (gripper +Z tilt from straight-down; EEAnywhere ~64% @45deg expected,
    grasped types ~96-100%)
  * grasped types: finger_joint ~0.0487 (a real grip ON the pcb, not closed past it),
    jaw asymmetry |finger - right_finger| < 1 mm
  * fingertip-below-support fraction (inherited EEAnywhere artifact, expect ~1-2% / worst
    few cm; NOT a blocker, just tracked)

Usage (from the repo root; conda env only needs torch/numpy/yaml, e.g. env_uwlab):
    python scripts_v2/tools/conversions/qc_reset_states_ur10e.py \
        --dataset_dir ./Datasets_ur10e/OmniReset [--pair OpenBox__Pcb]
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
META = os.path.join(REPO, "source/uwlab_assets/uwlab_assets/local/Robots/UR10e/metadata.yaml")

# Isaac articulation joint order for the graft (serial chain, then the two jaw leaves).
ARM_COLS = list(range(6))  # pan, lift, elbow, w1, w2, w3
FINGER_COLS = [6, 7]  # finger_joint, right_finger_joint
# fingertip = wrist_3 origin + 0.193 m along wrist_3 +Z (0.049 standoff + 0.144 tip reach)
FINGERTIP_OFFSET = 0.193
GRASPED_TYPES = {"ObjectRestingEEGrasped", "ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped"}


def rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def load_fk_chain():
    meta = yaml.safe_load(open(META))
    xyz = np.array(meta["calibrated_joints"]["xyz"])
    rpy = np.array(meta["calibrated_joints"]["rpy"])
    fixed = []
    for i in range(6):
        Tf = np.eye(4)
        Tf[:3, :3] = rpy_to_R(rpy[i])
        Tf[:3, 3] = xyz[i]
        fixed.append(Tf)
    return fixed


def fk_wrist3(q, fixed):
    """FK base_link_inertia -> wrist_3 (q: (N,6)) -> positions (N,3), +Z axes (N,3)."""
    N = q.shape[0]
    pos = np.zeros((N, 3))
    zax = np.zeros((N, 3))
    for n in range(N):
        T = np.eye(4)
        for i in range(6):
            c, s = np.cos(q[n, i]), np.sin(q[n, i])
            Tj = np.eye(4)
            Tj[0, 0] = c
            Tj[0, 1] = -s
            Tj[1, 0] = s
            Tj[1, 1] = c
            T = T @ fixed[i] @ Tj
        pos[n] = T[:3, 3]
        zax[n] = T[:3, 2]
    return pos, zax


def quat_to_R(quat):
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="./Datasets_ur10e/OmniReset")
    ap.add_argument("--pair", default=None, help="Object pair dir (default: every pair found)")
    args = ap.parse_args()

    fixed = load_fk_chain()
    pat = os.path.join(args.dataset_dir, "Resets", args.pair or "*", "resets_*.pt")
    files = sorted(glob.glob(pat))
    if not files:
        print(f"[QC] no reset files match {pat}")
        return
    overall_fail = False

    for f in files:
        rtype = os.path.basename(f)[len("resets_") : -len(".pt")]
        data = torch.load(f, map_location="cpu", weights_only=False)
        robot = data["initial_state"]["articulation"]["robot"]
        jp = torch.stack([t.cpu() for t in robot["joint_position"]]).numpy()
        root = torch.stack([t.cpu() for t in robot["root_pose"]]).numpy()  # (N,7) pos+quat wxyz
        N = jp.shape[0]
        fails = []

        # column sanity
        fingers = jp[:, FINGER_COLS]
        if not ((fingers > -1e-4).all() and (fingers < 0.0685).all()):
            fails.append(
                f"finger columns {FINGER_COLS} out of [0,0.068] "
                f"(min {fingers.min():.4f} max {fingers.max():.4f}) -- joint order assumption broken?"
            )

        # wrist limits
        wrists = jp[:, 3:6]
        n_viol = int((np.abs(wrists) > np.pi).any(axis=1).sum())
        worst = np.degrees(np.abs(wrists).max())
        if n_viol > 0:
            fails.append(f"{n_viol}/{N} states have |wrist| > 180 deg (worst {worst:.1f}) -- STALE recording")

        # FK-based checks (robot root ~ identity in these scenes, but apply it anyway)
        w3_pos_b, w3_z_b = fk_wrist3(jp[:, ARM_COLS], fixed)
        Rr = quat_to_R(root[0, 3:7])
        w3_z_w = w3_z_b @ Rr.T
        w3_pos_w = w3_pos_b @ Rr.T + root[:, :3]
        cos_down = -w3_z_w[:, 2].clip(-1, 1)  # 1 = approach straight down
        tilt = np.degrees(np.arccos(cos_down))
        topdown45 = float((tilt <= 45).mean()) * 100
        topdown30 = float((tilt <= 30).mean()) * 100

        fingertip_z = w3_pos_w[:, 2] + FINGERTIP_OFFSET * w3_z_w[:, 2]
        below = fingertip_z < 0.0
        below_pct = float(below.mean()) * 100
        below_worst = float(fingertip_z.min())

        line = (
            f"[QC] {rtype:<38s} N={N:<6d} wrist>180: {n_viol} (worst {worst:6.1f} deg)  "
            f"topdown<=45/30 deg: {topdown45:5.1f}%/{topdown30:5.1f}%  "
            f"fingertip<0: {below_pct:4.1f}% (worst {below_worst:+.3f} m)"
        )
        if rtype in GRASPED_TYPES:
            asym_mm = np.abs(fingers[:, 0] - fingers[:, 1]).max() * 1000
            fj = fingers[:, 0]
            line += f"  grip q: {fj.min():.4f}/{np.median(fj):.4f}/{fj.max():.4f}  jaw asym max {asym_mm:.2f} mm"
            # Grip-width semantics differ per type (measured on the old full datasets):
            #  * AnywhereEEGrasped holds the pcb mid-air at the canonical grasp width
            #    (q ~ 0.0487, symmetric jaws)
            #  * RestingEEGrasped grips the pcb lying on the table, mostly across its
            #    ~2 mm THICKNESS (q -> ~0.067-0.068) and often off-center (large asym)
            #  * PartiallyAssembledEEGrasped grips at the assembly pose (mixed widths,
            #    tilted approaches) -- stats reported, not gated
            if rtype == "ObjectAnywhereEEGrasped":
                asym_p99 = float(np.percentile(np.abs(fingers[:, 0] - fingers[:, 1]), 99)) * 1000
                if asym_p99 > 1.0:
                    fails.append(f"jaw asymmetry p99 {asym_p99:.2f} mm > 1 mm")
                if np.median(fj) < 0.045 or np.median(fj) > 0.055:
                    fails.append(f"median finger_joint {np.median(fj):.4f} not a pcb-width grip (~0.0487)")
            if rtype == "ObjectRestingEEGrasped" and np.median(fj) < 0.045:
                fails.append(f"median finger_joint {np.median(fj):.4f} -- jaws not closed on anything?")
            if rtype in ("ObjectAnywhereEEGrasped", "ObjectRestingEEGrasped") and topdown45 < 85:
                fails.append(f"top-down fraction {topdown45:.0f}% < 85% for a grasped type")
        print(line)
        if N < 9000:
            print(f"      NOTE: N={N} < ~10000 -- partial copy or short recording?")
        for msg in fails:
            print(f"      FAIL: {msg}")
            overall_fail = True

    print(f"\n[QC_RESULT] {'[FAIL] see FAIL lines above' if overall_fail else '[PASS] all checks passed'}")


if __name__ == "__main__":
    main()
