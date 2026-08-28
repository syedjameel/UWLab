#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""INDEPENDENT validator for a generated deep-C4 partial_assemblies.pt (gen_deep_c4_partial_assemblies_v1.py).

"The generator agreeing with itself is not evidence." This script re-derives depth / lateral / tilt
/ roll from the SAVED FILE ONLY (never imports or calls the generator), using a DIFFERENT
composition path -- scipy.spatial.transform.Rotation (rotation MATRICES) instead of
isaaclab.utils.math's quaternion combine_frame_transforms/subtract_frame_transforms -- so a
frame-order bug shared between "generate" and "check the same way" cannot hide.

It then runs the SAME success-predicate scorer training actually uses (ProgressContext, rewards.py:
pos_err = ||relative_pos||; rot_err = |wrap_to_pi(e_x)| + |wrap_to_pi(e_y)| from
euler_xyz_from_quat(relative_quat), e_z discarded) via isaaclab.utils.math -- deliberately the real
implementation here, not a reimplementation, because the number this produces (the fraction already
below 0.0025m / 0.025rad) IS the predicted task_3 baseline and must match what training will
actually compute. A SEPARATE, independent scipy Euler extraction of the same rotation is also
computed and compared against it, as a cross-check that the two libraries agree on what "orientation
error" means for these poses (not a substitute for using the real scorer for the reported number).

CPU-only. No Isaac app, no scene, no GPU.
"""

from __future__ import annotations

import argparse
import hashlib

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import isaaclab.utils.math as math_utils

LEG_OFF_POS = np.array([-0.106203, 0.0, 0.0])
LEG_OFF_QUAT_WXYZ = np.array([0.70710678, 0.0, 0.70710678, 0.0])
RECV_OFF_POS = np.array([-0.056250, 0.056250, -0.009374])
RECV_OFF_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])

SEAT_DEPTH_MM = 25.0
POS_THRESHOLD_M = 0.0025
ROT_THRESHOLD_RAD = 0.025


def wxyz_to_xyzw(q):
    return q[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(q):
    return q[..., [3, 0, 1, 2]]


def pct(a, ps=(0, 5, 10, 25, 50, 75, 90, 95, 100)):
    return {p: float(np.percentile(a, p)) for p in ps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True)
    args = ap.parse_args()

    sha = hashlib.sha256()
    with open(args.path, "rb") as f:
        raw = f.read()
        sha.update(raw)
    print(f"[validate] file={args.path}")
    print(f"[validate] sha256={sha.hexdigest()}")
    print(f"[validate] size_bytes={len(raw)}")

    data = torch.load(args.path, map_location="cpu", weights_only=False)
    for k in ("relative_position", "relative_orientation", "relative_pose", "receptive_object_pose", "insertive_object_pose"):
        assert k in data, f"missing expected key {k!r} -- not the schema fk_match_composed_c4.py / reset_insertive_object_from_partial_assembly_dataset expect"

    ins_root_pos = data["relative_position"].numpy()  # insertive ROOT in receptive ROOT frame (world identity here)
    ins_root_quat_wxyz = data["relative_orientation"].numpy()
    n = ins_root_pos.shape[0]
    print(f"[validate] n_poses={n}")

    # ================= INDEPENDENT REPROJECTION (scipy Rotation matrices, NOT isaaclab quat math) =================
    R_ins_root = Rotation.from_quat(wxyz_to_xyzw(ins_root_quat_wxyz))
    R_leg_off = Rotation.from_quat(wxyz_to_xyzw(np.tile(LEG_OFF_QUAT_WXYZ, (n, 1))))

    # alignment pose = insertive ROOT combined (forward) with the leg's OWN assembled_offset --
    # the exact inverse operation of what the generator's Offset.subtract did, computed here via
    # matrix multiplication + matrix-vector application, never combine_frame_transforms.
    align_pos = ins_root_pos + R_ins_root.apply(np.tile(LEG_OFF_POS, (n, 1)))
    R_align = R_ins_root * R_leg_off

    # target (seat) pose = receptive ROOT (world identity here) combined with the fixture's own
    # assembled_offset.
    R_recv_off = Rotation.from_quat(wxyz_to_xyzw(RECV_OFF_QUAT_WXYZ))
    target_pos = RECV_OFF_POS.copy()  # receptive root is identity, so target = offset directly
    R_target = R_recv_off

    # relative = target^{-1} o align, expressed in the target/mating frame -- done here with
    # R_target.inv() and manual vector rotation, the matrix-algebra equivalent of
    # subtract_frame_transforms, not a call to it.
    R_target_inv = R_target.inv()
    rel_pos = R_target_inv.apply(align_pos - target_pos)
    R_rel = R_target_inv * R_align

    # ---- geometric quantities, all from the independent reprojection above ----
    depth_into_bore_mm = SEAT_DEPTH_MM - rel_pos[:, 2] * 1000.0
    lateral_mm = np.hypot(rel_pos[:, 0], rel_pos[:, 1]) * 1000.0
    pos_err_mm_indep = np.linalg.norm(rel_pos, axis=1) * 1000.0

    # tilt magnitude: rotation-invariant angle between the rotated local-Z axis and world Z --
    # does not depend on any Euler decomposition order, so it cannot inherit an Euler-convention bug.
    R_rel_mat = R_rel.as_matrix()
    local_z_in_target = R_rel_mat[:, :, 2]  # column 3 = where local Z lands
    tilt_deg_indep = np.degrees(np.arccos(np.clip(local_z_in_target[:, 2], -1.0, 1.0)))

    # roll about the insertion axis (thread yaw): scipy intrinsic-xyz Euler extraction, z component
    # -- an approximate/diagnostic measure only (roll does not gate success), computed independently
    # of the generator's own recorded roll_axis_deg / center/width table lookups.
    euler_xyz_scipy = R_rel.as_euler("xyz", degrees=True)
    roll_axis_deg_indep = euler_xyz_scipy[:, 2]

    print("\n=== INDEPENDENT REPROJECTION (scipy Rotation, NOT the generator's own math) ===")
    print("depth_into_bore_mm:", pct(depth_into_bore_mm))
    print("lateral_mm:", pct(lateral_mm))
    print("tilt_deg (rotation-invariant, local-Z vs world-Z):", pct(tilt_deg_indep))
    print("roll_axis_deg (scipy xyz-euler z-component, diagnostic only):", pct(roll_axis_deg_indep))

    # ================= PRODUCTION SUCCESS-PREDICATE SCORER (isaaclab.utils.math, the REAL formula) =================
    rel_pos_t = torch.as_tensor(rel_pos, dtype=torch.float32)
    rel_quat_t = xyzw_to_wxyz(R_rel.as_quat())
    rel_quat_t = torch.as_tensor(rel_quat_t, dtype=torch.float32)

    pos_err_m = torch.norm(rel_pos_t, dim=1)
    e_x, e_y, e_z = math_utils.euler_xyz_from_quat(rel_quat_t)
    rot_err_rad = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()

    pos_err_mm = (pos_err_m * 1000.0).numpy()
    rot_err_deg = np.degrees(rot_err_rad.numpy())

    print("\n=== CROSS-CHECK: production scorer's pos_err vs independent reprojection's pos_err ===")
    max_abs_diff_mm = float(np.max(np.abs(pos_err_mm - pos_err_mm_indep)))
    print(f"max |pos_err_mm (production) - pos_err_mm (independent)| = {max_abs_diff_mm:.6f} mm")
    assert max_abs_diff_mm < 1e-2, "position reprojection disagrees with production scorer by >0.01mm -- STOP, do not trust the baseline below"

    # independent orientation-error cross-check: scipy's own e_x,e_y (intrinsic xyz), wrapped the
    # same way, summed the same way -- NOT the number reported as the baseline, only a sanity check
    # that isaaclab's euler_xyz_from_quat and scipy's 'xyz' extraction agree on this task's poses.
    ex_s = np.deg2rad(euler_xyz_scipy[:, 0])
    ey_s = np.deg2rad(euler_xyz_scipy[:, 1])
    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi
    rot_err_rad_scipy = np.abs(wrap(ex_s)) + np.abs(wrap(ey_s))
    rot_err_deg_scipy = np.degrees(rot_err_rad_scipy)
    max_abs_diff_deg = float(np.max(np.abs(rot_err_deg - rot_err_deg_scipy)))
    median_abs_diff_deg = float(np.median(np.abs(rot_err_deg - rot_err_deg_scipy)))
    print(f"orientation error, production (isaaclab euler_xyz_from_quat) vs independent (scipy 'xyz'):")
    print(f"  max |diff| = {max_abs_diff_deg:.6f} deg, median |diff| = {median_abs_diff_deg:.6f} deg")
    if max_abs_diff_deg > 1.0:
        print("  WARNING: >1deg disagreement on at least one sample -- Euler-convention mismatch, investigate before trusting rot_err")
    else:
        print("  agreement within 1deg on every sample -- both methods describe the same orientation error here")

    print("\n=== DEPTH-VS-SUCCESS GRADIENT (production scorer, independent reprojection's depth) ===")
    bin_lo = np.floor(depth_into_bore_mm.min())
    bins = np.arange(bin_lo, 26, 1.0)
    idx = np.digitize(depth_into_bore_mm, bins)
    for b in range(1, len(bins)):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            continue
        frac_pos = float((pos_err_mm[m] < POS_THRESHOLD_M * 1000.0).mean())
        frac_rot = float((rot_err_deg[m] < np.degrees(ROT_THRESHOLD_RAD)).mean())
        frac_both = float(((pos_err_mm[m] < POS_THRESHOLD_M * 1000.0) & (rot_err_deg[m] < np.degrees(ROT_THRESHOLD_RAD))).mean())
        print(f"  depth_into_bore in [{bins[b-1]:.0f},{bins[b]:.0f})mm: n={cnt}, pos_ok={frac_pos:.3f}, rot_ok={frac_rot:.3f}, BOTH={frac_both:.3f}")

    pos_ok = pos_err_mm < (POS_THRESHOLD_M * 1000.0)
    rot_ok = rot_err_deg < np.degrees(ROT_THRESHOLD_RAD)
    both_ok = pos_ok & rot_ok

    print("\n=== PREDICTED task_3 BASELINE (production 0.0025m / 0.025rad gate, over the whole n={} bank) ===".format(n))
    print(f"pos_err_mm distribution: {pct(pos_err_mm)}")
    print(f"rot_err_deg distribution: {pct(rot_err_deg)}")
    print(f"fraction pos_ok (<2.5mm):        {float(pos_ok.mean()):.4f}")
    print(f"fraction rot_ok (<1.4324deg):    {float(rot_ok.mean()):.4f}")
    print(f"fraction BOTH (predicted baseline): {float(both_ok.mean()):.4f}")


if __name__ == "__main__":
    main()
