# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Author the SEATED-pose partial_assemblies.pt for the jig-removal task (CPU, no Isaac).

The removal task's deployment start is "jig seated on the fixture". The stock C1 recorder
samples the jig INDEPENDENTLY of the fixture and so never produces that state; the stock
dependent-placement mechanism is the partial-assembly event, which composes a RELATIVE pose
with the fixture's CURRENT randomised pose, yaw included (verified in
reset_insertive_object_from_partial_assembly_dataset).

This writes a partial_assemblies.pt in the authors' format containing exactly the seated
relative pose (both yaw twins -- the pillar pattern is 2-fold symmetric):

    rel z = enc_bottom(-0.018037) + jig_seat(0.0176) + jig_half_height(0.012) = 0.011563

Feed it to the stock recorder via
    --task OmniReset-UR10eLinearGripper-ObjectPartiallyAssembledEEAnywhere-v0
    env.events.reset_insertive_object_pose_from_partial_assembly_dataset.params.dataset_dir=<dir>

Data preparation in the authors' file format -- no code or config changes. Ledger: R6.
"""

from __future__ import annotations

import argparse
import os

import torch

# EnclosurePcb frame: bottom face at -0.018037 (its metadata bottom_offset). Jig seats with its
# bottom 17.6 mm above that face; jig root = bottom + half-height (12 mm).
REL_Z = -0.018037 + 0.0176 + 0.012


JIG_HALF_H = 0.012      # jig root above its bottom face


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
    ap.add_argument("--pair", default=None,
                    help="default: EnclosurePcb__JigBlocked (seated) / JigBlocked__ParkingSpot "
                         "(neargoal -- the event is retargeted at parking_marker; compute_pair_dir "
                         "names the pair INSERTIVE__RECEPTIVE, verified from the loader error)")
    ap.add_argument("--n", type=int, default=256, help="neargoal only: how many poses")
    ap.add_argument("--lateral-mm", type=float, default=35.0)
    ap.add_argument("--height-mm", type=float, default=70.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "seated":
        pair = args.pair or "EnclosurePcb__JigBlocked"
        rel_pos = torch.tensor([[0.0, 0.0, REL_Z]] * 2, dtype=torch.float32)
        rel_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0],      # yaw 0
                                 [0.0, 0.0, 0.0, 1.0]],     # yaw pi (2-fold symmetric part)
                                dtype=torch.float32)
        _save(args.out, pair, rel_pos, rel_quat,
              f"seated: rel pos (0, 0, {REL_Z:.6f}) in the FIXTURE frame, yaw 0 and pi")
    else:
        # NEAR-GOAL for C4, in the PARKING MARKER's frame. The marker is world-fixed at the
        # parking spot with its origin ON the work surface, so "relative to the marker" is a
        # fixed region around the spot. Authored rather than sampled: assembly_sampling_event
        # slides the part under zero friction, which on an open mat just scatters it -- the
        # useful near-goal distribution here is "held at/above the spot, roughly aligned".
        g = torch.Generator().manual_seed(args.seed)
        n = args.n
        xy = (torch.rand((n, 2), generator=g) * 2 - 1) * (args.lateral_mm / 1000.0)
        z = JIG_HALF_H + torch.rand((n, 1), generator=g) * (args.height_mm / 1000.0)
        rel_pos = torch.cat([xy, z], dim=1).float()
        yaw = (torch.rand(n, generator=g) * 2 - 1) * torch.pi   # free: no yaw gate when parking
        rel_quat = torch.stack([torch.cos(yaw / 2), torch.zeros(n), torch.zeros(n),
                                torch.sin(yaw / 2)], dim=1).float()
        _save(args.out, args.pair or "JigBlocked__ParkingSpot", rel_pos, rel_quat,
              f"near-goal: +-{args.lateral_mm:.0f} mm lateral, {JIG_HALF_H*1000:.0f}"
              f"-{JIG_HALF_H*1000+args.height_mm:.0f} mm high, free yaw, in the MARKER frame")


if __name__ == "__main__":
    main()
