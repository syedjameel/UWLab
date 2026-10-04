"""Write a distance-filtered copy of a reset bank, for a curriculum that has no gap in it.

THE GAP
-------
Measured off the banks themselves, initial cube-to-goal distance:

    Near-Goal      n=125   min 0 mm   median 12 mm   max 25 mm
    Stable-Grasp   n=122   min 15 mm  median 215 mm  max 392 mm   (13 states under 100 mm)

The curriculum asks the policy to go from a 12 mm nudge straight to a 215 mm carry. There is nothing
in between, and the far tail is beyond what the arm can traverse inside the step budget at all --
those episodes cannot terminate successfully however good the policy is, so they contribute gradient
that says "this family is impossible" rather than "try harder".

That is an exploration failure, not an incentive failure. The transport reward is already worth
about 15x the cost of moving in accumulated return; the policy simply never reaches the goal from
209 mm, so it never observes that value, and holding still is locally optimal. Two independent
boxes -- one with transport shaping, one without -- both show arm action magnitude DECAYING over
training, which is what settling into stillness looks like.

WHAT THIS DOES
--------------
Writes a filtered copy of a reset bank keeping only the states whose insertive object starts within
`--max_dist` of its seated pose, under a separate dataset root with the SAME filename, so a training
stage can select it with nothing but `dataset_dir=`. No code change, no new task id, no edit to the
original bank -- which stays exactly as recorded.

The seated pose is computed the way the reward computes it: bottom face to top face, both assembled
offsets rotated by their own body's orientation.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

INSERTIVE_OFFSET = (0.0, 0.0, -0.017)
RECEPTIVE_OFFSET = (0.0, 0.0, 0.017)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by quaternion q, both (N, ...). Matches isaaclab's (w, x, y, z) convention."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset root holding Resets/<pair>/")
    ap.add_argument("--dst", required=True, help="dataset root to write the filtered copy into")
    ap.add_argument("--pair", default="InsertiveCube34__ReceptiveCube34")
    ap.add_argument("--reset_type", default="ObjectAnywhereEEGrasped")
    ap.add_argument("--max_dist", type=float, required=True, help="metres, initial cube-to-goal")
    ap.add_argument("--min_clearance", type=float, default=None,
                    help="metres. Drop states whose success predicate would require the carried "
                         "cube's CENTRE to sit lower than this above the base cube's top face.\n\n"
                         "WHY THIS MATTERS MORE THAN DISTANCE. Success compares the insertive "
                         "bottom face to the receptive top face, both offsets rotated by their own "
                         "body. A cube held inverted has its bottom face pointing UP, so zeroing "
                         "the predicate would put its centre inside the base cube -- impossible at "
                         "that attitude, and the NoOrient task pays nothing for rotating out of it. "
                         "Measured on this bank: 42 of 122 Stable-Grasp states (34%) are in that "
                         "condition, against 13 of 125 for Near-Goal. A third of every batch was "
                         "teaching the policy that the family cannot be solved.\n\n"
                         "A normally seated stack clears by ~17 mm, so 0.012 keeps the seatable "
                         "states and drops the rest.")
    ap.add_argument("--min_states", type=int, default=8,
                    help="Refuse to write a bank thinner than this. A stage that samples from a "
                         "handful of states memorises them instead of learning the family, and the "
                         "success rate it reports is not about the task.")
    args = ap.parse_args()

    src = Path(args.src) / "Resets" / args.pair / f"resets_{args.reset_type}.pt"
    blob = torch.load(src, map_location="cpu", weights_only=False)
    state = blob["initial_state"]

    ins = torch.stack([t.reshape(-1)[:7] for t in state["rigid_object"]["insertive_object"]["root_pose"]])
    rec = torch.stack([t.reshape(-1)[:7] for t in state["rigid_object"]["receptive_object"]["root_pose"]])
    n = ins.shape[0]

    ins_off = torch.tensor(INSERTIVE_OFFSET).expand(n, 3)
    rec_off = torch.tensor(RECEPTIVE_OFFSET).expand(n, 3)
    seated = rec[:, :3] + quat_apply(rec[:, 3:7], rec_off) - quat_apply(ins[:, 3:7], ins_off)
    dist = torch.norm(ins[:, :3] - seated, dim=1)

    ok = dist <= args.max_dist
    print(f"{args.reset_type}: {int(ok.sum())}/{n} states within {args.max_dist * 1000:.0f} mm "
          f"(bank median {float(dist.median()) * 1000:.0f} mm)")
    if args.min_clearance is not None:
        top = rec[:, 2] + 0.017
        clearance = seated[:, 2] - top
        seatable = clearance >= args.min_clearance
        print(f"  seatable at their own attitude (clearance >= {args.min_clearance * 1000:.0f} mm): "
              f"{int(seatable.sum())}/{n}"
              f"   -- {int((~seatable).sum())} would need the cube inside the base")
        ok = ok & seatable
    keep = ok.nonzero(as_tuple=True)[0].tolist()
    print(f"  keeping {len(keep)}/{n}")
    if len(keep) < args.min_states:
        raise SystemExit(
            f"only {len(keep)} states survive the filter, under the {args.min_states} floor -- "
            "widen --max_dist rather than training a stage on a handful of memorisable resets")

    def subset(node):
        if isinstance(node, dict):
            return {k: subset(v) for k, v in node.items()}
        if isinstance(node, list):
            return [node[i] for i in keep]
        return node

    out = dict(blob)
    out["initial_state"] = subset(state)

    dst = Path(args.dst) / "Resets" / args.pair / f"resets_{args.reset_type}.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    kept = dist[torch.tensor(keep)]
    print(f"wrote {dst}  ({len(keep)} states, "
          f"distance {float(kept.min())*1000:.0f}-{float(kept.max())*1000:.0f} mm)")

    # Every other file the reset manager may open for this pair has to exist alongside it, or a
    # stage pointed at this root fails at construction on a missing file rather than a bad number.
    for extra in (Path(args.src) / "Resets" / args.pair).glob("*.pt"):
        target = dst.parent / extra.name
        if not target.exists():
            shutil.copy2(extra, target)
            print(f"  copied through {extra.name}")
    grasps = Path(args.src) / "Grasps"
    if grasps.is_dir() and not (Path(args.dst) / "Grasps").exists():
        shutil.copytree(grasps, Path(args.dst) / "Grasps")
        print("  copied through Grasps/")


if __name__ == "__main__":
    main()
