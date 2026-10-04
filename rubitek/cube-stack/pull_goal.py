"""Build the missing curriculum rung by moving the GOAL to the cube, not the cube to the goal.

THE GAP THIS FILLS
------------------
Measured on one checkpoint, same policy, two families:

    Near-Goal      places but does not carry -- 50.8% of episodes end within 20 mm, best 0.5 mm,
                   yet median progress is -2.3 mm and only 6.4% close any distance. It ends near the
                   goal because it STARTS there (median 12 mm).
    Stable-Grasp   carries but does not place -- 51.2% of episodes make real progress, median
                   +5.9 mm, and NOT ONE of 256 finishes within 20 mm. Nearest state starts 30 mm out.

Two halves of one task, each practised on a family that does not require the other. Nothing in the
data sits between 12 mm and 30 mm, so no curriculum built from these banks can teach both together
-- which is why a distance ladder over Stable-Grasp states failed: its lowest rung was still
entirely on the transport side of the gap.

WHY MOVE THE GOAL
-----------------
Moving the cube toward the goal would break the grasp: cube and hand would have to travel together,
which needs inverse kinematics on a stored joint configuration. The RECEPTIVE cube is kinematic and
its pose is simply stored in the reset state, so relocating it changes the target while leaving the
robot, the grasp and every other asset exactly as recorded.

THE GEOMETRY, AND ITS LIMIT
---------------------------
Success compares the insertive bottom face to the receptive top face, both offsets rotated by their
own body:

    seated_centre = rec_pos + R_rec @ [0,0,+0.017] - R_ins @ [0,0,-0.017]

Solving for a receptive position that puts `seated_centre` a chosen distance from where the cube
already is, subject to the base cube resting on the table (z = 0.021), gives a circle -- and it only
exists when the requested distance is at least |p_z - 0.021|, where p is the cube position less the
two offsets. A cube held 100 mm up therefore cannot be closer than ~45-79 mm to any seated pose on
the table. States holding the cube near 55 mm can reach a few millimetres. That floor is physical,
not a modelling choice, so this script reports it per state rather than silently producing something
unreachable.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import torch

INS_OFFSET = (0.0, 0.0, -0.017)
REC_OFFSET = (0.0, 0.0, 0.017)
TABLE_Z = 0.021
CUBE_M = 0.034


def qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--pair", default="InsertiveCube34__ReceptiveCube34")
    ap.add_argument("--reset_type", default="ObjectAnywhereEEGrasped")
    ap.add_argument("--target_dist", type=float, default=0.025,
                    help="Desired initial cube-to-goal distance, metres. Where geometry forbids it "
                         "the state gets its own achievable minimum instead, and the spread is "
                         "reported.")
    ap.add_argument("--clearance", type=float, default=0.004,
                    help="Vertical gap required between the base cube's top face and the held "
                         "cube's bottom face, metres.")
    ap.add_argument("--max_dist", type=float, default=0.060,
                    help="Drop states whose achievable minimum exceeds this -- they are still "
                         "transport problems and belong in the existing banks, not in this rung.")
    args = ap.parse_args()

    src = Path(args.src) / "Resets" / args.pair / f"resets_{args.reset_type}.pt"
    blob = torch.load(src, map_location="cpu", weights_only=False)
    state = blob["initial_state"]

    ins = torch.stack([t.reshape(-1)[:7] for t in state["rigid_object"]["insertive_object"]["root_pose"]])
    rec = torch.stack([t.reshape(-1)[:7] for t in state["rigid_object"]["receptive_object"]["root_pose"]])
    n = ins.shape[0]

    ins_off = torch.tensor(INS_OFFSET).expand(n, 3)
    rec_off = torch.tensor(REC_OFFSET).expand(n, 3)

    # p is where the receptive centre would have to be for a ZERO-distance seat.
    p = ins[:, :3] - qapply(rec[:, 3:7], rec_off) + qapply(ins[:, 3:7], ins_off)
    dz = (p[:, 2] - TABLE_Z).abs()                 # the unavoidable floor for each state
    want = torch.clamp(torch.full_like(dz, args.target_dist), min=dz)
    radius = torch.sqrt(torch.clamp(want ** 2 - dz ** 2, min=0.0))

    # Place the base cube on the circle, in the direction that keeps it clear of the arm: away from
    # the robot base at the origin.
    away = p[:, :2].clone()
    norm = away.norm(dim=1, keepdim=True).clamp(min=1e-6)
    away = away / norm

    new_rec = rec.clone()
    new_rec[:, 0] = p[:, 0] + away[:, 0] * radius
    new_rec[:, 1] = p[:, 1] + away[:, 1] * radius
    new_rec[:, 2] = TABLE_Z

    seated = new_rec[:, :3] + qapply(new_rec[:, 3:7], rec_off) - qapply(ins[:, 3:7], ins_off)
    achieved = (ins[:, :3] - seated).norm(dim=1)

    # NON-OVERLAP, TESTED DIRECTLY. Distance to the seated pose says nothing about whether the two
    # bodies currently occupy the same space: the first version of this script produced states whose
    # held cube sat 17.5 mm off the floor while the relocated base's top face reached 38 mm. PhysX
    # resolved the interpenetration explosively -- 29 N at a fingertip on the first step, cubes
    # ejected to -659 mm -- and every number measured on it was meaningless.
    #
    # Two 34 mm axis-aligned cubes overlap when they are within 34 mm on ALL THREE axes. Testing the
    # separation itself is safer than deriving a height inequality and trusting the derivation.
    sep = (ins[:, :3] - new_rec[:, :3]).abs()
    overlapping = (sep < CUBE_M).all(dim=1)
    # And the held cube must be ABOVE the base to be seatable onto it at all.
    above = ins[:, 2] > (new_rec[:, 2] + CUBE_M + args.clearance)
    ok = (achieved <= args.max_dist) & (~overlapping) & above
    print(f"  interpenetrating placements rejected: {int(overlapping.sum())}/{n}")
    print(f"  cube not above the base, rejected:     {int((~above).sum())}/{n}")
    keep = ok.nonzero(as_tuple=True)[0].tolist()
    print(f"{args.reset_type}: {n} states")
    print(f"  geometric floor |p_z - table|: median {float(dz.median())*1000:.1f} mm, "
          f"min {float(dz.min())*1000:.1f} mm, max {float(dz.max())*1000:.1f} mm")
    print(f"  achieved goal distance:        median {float(achieved.median())*1000:.1f} mm")
    print(f"  states within {args.max_dist*1000:.0f} mm: {len(keep)}/{n}")
    if len(keep) < 8:
        raise SystemExit("too few states survive; raise --max_dist rather than train on a handful")

    def subset(node, idx):
        if isinstance(node, dict):
            return {k: subset(v, idx) for k, v in node.items()}
        if isinstance(node, list):
            return [node[i] for i in idx]
        return node

    out_state = subset(state, keep)
    # Write the relocated receptive poses back, preserving each entry's original tensor shape.
    tmpl = state["rigid_object"]["receptive_object"]["root_pose"][0]
    out_state["rigid_object"]["receptive_object"]["root_pose"] = [
        new_rec[i].reshape(tmpl.shape).clone() for i in keep
    ]
    out = dict(blob)
    out["initial_state"] = out_state

    dst = Path(args.dst) / "Resets" / args.pair / f"resets_{args.reset_type}.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    kept = achieved[torch.tensor(keep)]
    print(f"wrote {dst}  ({len(keep)} states, goal distance "
          f"{float(kept.min())*1000:.1f}-{float(kept.max())*1000:.1f} mm)")

    for extra in (Path(args.src) / "Resets" / args.pair).glob("*.pt"):
        t = dst.parent / extra.name
        if not t.exists():
            shutil.copy2(extra, t)
    g = Path(args.src) / "Grasps"
    if g.is_dir() and not (Path(args.dst) / "Grasps").exists():
        shutil.copytree(g, Path(args.dst) / "Grasps")


if __name__ == "__main__":
    main()
