# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Add the 180-degree-flipped twin of every recorded grasp (CPU, no Isaac).

A top-down straddle on a rectangular part is 2-FOLD SYMMETRIC about the approach axis: rotating
the gripper 180 deg about its own +Z gives an identical grasp -- same jaws, same two walls, same
contact patches -- differing only in which way round the wrist sits.

The sampler emits only ONE of the two. Measured on the jig-removal v3 set: all 8191 grasps had the
gripper's +Y axis pointing the same way (100.0% toward the jig's +x, 0.0% toward -x). Its
uniqueness filter compares poses, and the flipped twin is a distinct pose that simply never gets
generated, so the whole dataset is one-sided.

WHY THAT MATTERS. The jig spawns at free yaw, so the wrist must rotate to match it; with only one
branch available that rotation runs the full circle and hits the +-180 deg wrist limit. Measured
across the four v3 reset sets (60953 states):

    49.3% needed the wrist MORE than 90 deg from the home pose
    C4 (the placement phase) was worst: 56.7% beyond 90 deg, 41.0% beyond 150 deg,
    and 2.7% sitting AT the +-180 deg limit, where the wrist cannot rotate any further

With both branches present the policy can always take the one within 90 deg of home, halving the
required wrist travel. Nothing rewards wrist economy directly, but ``action_magnitude`` and
``action_rate`` are penalised, so the cheaper branch is the one PPO should prefer.

The flip is exact and analytic -- no simulation, no re-sampling:
  * ``relative_position``    UNCHANGED. Rotating the gripper about its own approach axis does not
    move the EE frame origin, so the grasp point is identical.
  * ``relative_orientation`` q -> q (x) q_z180, where q_z180 = (0, 0, 0, 1) in wxyz.
  * ``gripper_joint_positions`` UNCHANGED -- the jaws close by the same amount on the same walls.

Every downstream set that bakes EE poses in (C1 pre-grasp seeds, C2, C3, C4) must be RE-RECORDED
after running this, and Stage-1 retrained: the reset states carry absolute joint angles, so they
cannot be flipped after the fact.
"""

from __future__ import annotations

import argparse
import os
import shutil

import torch


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product, WXYZ (IsaacLab convention)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _as_tensor(seq) -> torch.Tensor:
    if torch.is_tensor(seq):
        return seq.float()
    return torch.stack([t if torch.is_tensor(t) else torch.tensor(t) for t in seq]).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps", required=True, help="path to grasps.pt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = torch.load(args.grasps, map_location="cpu", weights_only=False)
    g = data["grasp_relative_pose"]
    pos = _as_tensor(g["relative_position"])
    quat = _as_tensor(g["relative_orientation"])
    n = len(pos)

    # 180 deg about the gripper's own +Z (the approach axis), applied on the RIGHT so it is a
    # rotation in the gripper's LOCAL frame.
    z180 = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(n, 4)
    quat_f = _quat_mul(quat, z180)

    def _axis(q, v):
        w, x, y, z = q.unbind(-1)
        vx, vy, vz = v
        return torch.stack([
            (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - w * z) * vy + 2 * (x * z + w * y) * vz,
            2 * (x * y + w * z) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z - w * x) * vz,
            2 * (x * z - w * y) * vx + 2 * (y * z + w * x) * vy + (1 - 2 * (x * x + y * y)) * vz,
        ], dim=-1)

    # verify: approach (+Z) preserved, +Y reversed -- that IS the flip
    a0, a1 = _axis(quat, (0, 0, 1.0)), _axis(quat_f, (0, 0, 1.0))
    y0, y1 = _axis(quat, (0, 1.0, 0)), _axis(quat_f, (0, 1.0, 0))
    approach_err = (a0 - a1).norm(dim=1).max().item()
    y_flip_err = (y0 + y1).norm(dim=1).max().item()
    print(f"grasps in: {n}")
    print(f"  approach axis preserved   max err {approach_err:.2e}  (want ~0)")
    print(f"  +Y axis exactly reversed  max err {y_flip_err:.2e}  (want ~0)")
    if approach_err > 1e-5 or y_flip_err > 1e-5:
        raise SystemExit("flip verification FAILED -- not writing")
    print(f"  before: +Y toward +x {(y0[:, 0] > 0).float().mean() * 100:.1f}%")
    print(f"  after : +Y toward +x {(y1[:, 0] > 0).float().mean() * 100:.1f}%")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    out = dict(data)
    ng = dict(g)
    ng["relative_position"] = torch.cat([pos, pos], dim=0)          # unchanged by the flip
    ng["relative_orientation"] = torch.cat([quat, quat_f], dim=0)
    gj = g.get("gripper_joint_positions")
    if isinstance(gj, dict):
        ng["gripper_joint_positions"] = {k: torch.cat([_as_tensor(v), _as_tensor(v)], dim=0)
                                         for k, v in gj.items()}
    out["grasp_relative_pose"] = ng

    backup = args.grasps + ".oneSided"
    if not os.path.exists(backup):
        shutil.copy2(args.grasps, backup)
    torch.save(out, args.grasps)
    print(f"\nwrote {args.grasps}: {n} -> {2 * n} grasps (both wrist branches)")
    print(f"  backup of the one-sided set: {backup}")
    print("\nRE-RECORD C1 (seeds), C2, C3, C4 and retrain Stage-1 -- reset states store absolute")
    print("joint angles and cannot be flipped retrospectively.")


if __name__ == "__main__":
    main()
