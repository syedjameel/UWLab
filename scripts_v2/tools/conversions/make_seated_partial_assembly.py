# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Author the SEATED-start partial_assemblies.pt for jig-removal v2 (CPU, no Isaac).

The removal task's deployment start is "jig seated on the enclosure+PCB stack". The stock C1
recorder (``ObjectAnywhereEEAnywhere``) samples the jig INDEPENDENTLY of anything else and so
can never produce that state. The stock mechanism for a DEPENDENT placement is the
partial-assembly event, which composes a stored RELATIVE pose with the target asset's CURRENT
randomised pose, quaternion included (see reset_insertive_object_from_partial_assembly_dataset).

So the seated start is recorded with:

    --task OmniReset-UR10eLinearGripper-ObjectPartiallyAssembledEEAnywhere-v0
    --reset_type ObjectPartiallyAssembledEEAnywhere
    env.events.reset_insertive_object_pose_from_partial_assembly_dataset.params\\
        .receptive_object_cfg.name=enclosure_pcb
    env.events.reset_insertive_object_pose_from_partial_assembly_dataset.params.dataset_dir=<dir>

Retargeting ``receptive_object_cfg`` is what makes the loader resolve the pair against the
enclosure instead of the Pedestal, so the file belongs under ``EnclosurePcb__JigV2c``
(compute_pair_dir sorts the two names ALPHABETICALLY -- it is not insertive-then-receptive).

The relative pose is DERIVED from the two metadata files rather than hardcoded: at a seat the
insertive's assembled point must coincide with the receptive's, so

    rel_pos.z = receptive.assembled_offset.z - insertive.assembled_offset.z

Data preparation in the authors' file format -- no code or config changes.
"""

from __future__ import annotations

import argparse
import os

import torch
import yaml

_LOCAL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..",
    "source/uwlab_assets/uwlab_assets/local/Props/Custom",
)


def _assembled_z(asset_dir: str) -> float:
    meta = yaml.safe_load(open(os.path.join(_LOCAL, asset_dir, "metadata.yaml")))
    return float(meta["assembled_offset"]["pos"][2])


def _save(out, pair, rel_pos, rel_quat, label):
    d = os.path.join(out, "Resets", pair)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "partial_assemblies.pt")
    torch.save({
        "relative_position": rel_pos,
        "relative_orientation": rel_quat,
        "relative_pose": torch.cat([rel_pos, rel_quat], dim=1),
    }, path)
    print(f"wrote {path}\n  {len(rel_pos)} entries -- {label}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help=".../OmniReset (the Resets/<Pair>/partial_assemblies.pt path is derived)")
    ap.add_argument("--mode", choices=("seated", "neargoal"), default="seated")
    ap.add_argument("--insertive", default="JigV2c")
    ap.add_argument("--receptive", default="EnclosurePcb")
    ap.add_argument("--pair", default=None, help="default: the two names sorted alphabetically")
    ap.add_argument("--n", type=int, default=256, help="neargoal only: how many poses")
    ap.add_argument("--lateral-mm", type=float, default=35.0)
    ap.add_argument("--height-mm", type=float, default=70.0)
    ap.add_argument("--yaw-rad", type=float, default=0.6,
                    help="neargoal only: yaw spread about each twin. The Pedestal's success gate "
                         "is 0.35 rad with yaw_symmetry 2, so 0.6 straddles it -- some states "
                         "start inside the gate, some need correcting.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rel_z = _assembled_z(args.receptive) - _assembled_z(args.insertive)
    pair = args.pair or "__".join(sorted([args.insertive, args.receptive]))

    if args.mode == "neargoal":
        # C4 (near the goal, EE grasped) must be AUTHORED for a flat plate. The stock
        # record_partial_assemblies.py walks the insertive out of a SOCKET, recording the
        # continuum of partial insertions. A plate has no such continuum -- the jig is either
        # on it or off it -- so the recorder returns the assembled pose repeated plus the
        # occasional slid-off outlier (measured: 11 of 12 identical, 1 off the plate).
        # What C4 actually needs is the arm's approach: held above the plate, roughly aligned.
        g = torch.Generator().manual_seed(args.seed)
        n = args.n
        xy = (torch.rand((n, 2), generator=g) * 2 - 1) * (args.lateral_mm / 1000.0)
        z = rel_z + torch.rand((n, 1), generator=g) * (args.height_mm / 1000.0)
        rel_pos = torch.cat([xy, z], dim=1).float()
        twin = torch.randint(0, 2, (n,), generator=g) * torch.pi          # 0 or pi
        yaw = twin + (torch.rand(n, generator=g) * 2 - 1) * args.yaw_rad
        rel_quat = torch.stack([torch.cos(yaw / 2), torch.zeros(n), torch.zeros(n),
                                torch.sin(yaw / 2)], dim=1).float()
        _save(args.out, pair, rel_pos, rel_quat,
              f"near-goal: +-{args.lateral_mm:.0f} mm lateral, {rel_z*1000:.1f}"
              f"-{rel_z*1000+args.height_mm:.0f} mm high, yaw within +-{args.yaw_rad} rad of "
              f"each twin, in the {args.receptive} frame")
        return

    # Both yaw twins: the pillar pattern is 2-fold symmetric, and success_thresholds carry
    # yaw_symmetry 2, so 0 and pi are the same seat.
    rel_pos = torch.tensor([[0.0, 0.0, rel_z]] * 2, dtype=torch.float32)
    rel_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0],       # yaw 0
                             [0.0, 0.0, 0.0, 1.0]],      # yaw pi
                            dtype=torch.float32)

    d = os.path.join(args.out, "Resets", pair)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "partial_assemblies.pt")
    torch.save({
        "relative_position": rel_pos,
        "relative_orientation": rel_quat,
        "relative_pose": torch.cat([rel_pos, rel_quat], dim=1),
    }, path)
    print(f"wrote {path}")
    print(f"  {len(rel_pos)} entries -- seated: rel pos (0, 0, {rel_z:.6f}) in the "
          f"{args.receptive} frame, yaw 0 and pi")
    print(f"  derived from metadata: {args.receptive}.assembled_z={_assembled_z(args.receptive):+.6f} "
          f"- {args.insertive}.assembled_z={_assembled_z(args.insertive):+.6f}")


if __name__ == "__main__":
    main()
