"""Numeric QA of a recorded grasp bank. No Isaac, no GPU -- pure torch on the saved tensors.

WHY THIS EXISTS. The cube34 bank is built at a ~5% acceptance rate. Low yield is fine for
wall-clock (more envs fixes it) but it is NOT automatically fine for coverage: if the only
candidates that survive the shake test are one face at one approach angle, the bank is narrow, and
the paper's own ablation (§4.3, Fig. 9) shows narrowing the grasp distribution costs both sample
efficiency and final success rate -- "Broad" beats "Moderate" beats "Narrow", and the method uses
Broad. So the bank has to be checked for mode collapse, not just for size.

Reports, for the recorded relative grasp poses:
  * count
  * per-axis position spread (the object is a 34 mm cube, so ~+-17 mm is the physical bound)
  * how the approach directions distribute over the cube's six faces
  * pairwise-distance summary, and the largest empty gap, as a collapse detector
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument("--grasps", type=Path, required=True)
p.add_argument("--edge-mm", type=float, default=34.0, help="cube edge, for context in the printout")
args = p.parse_args()

d = torch.load(args.grasps, map_location="cpu", weights_only=False)

# Schema per render_delto_grasps_raw.py:9-16
node = d["grasp_relative_pose"] if "grasp_relative_pose" in d else d
pos = node["relative_position"]
quat = node["relative_orientation"]
if isinstance(pos, list):
    pos = torch.stack(pos)
if isinstance(quat, list):
    quat = torch.stack(quat)
pos = pos.float().reshape(-1, 3)
quat = quat.float().reshape(-1, 4)

n = pos.shape[0]
print(f"grasps            : {n}")
print(f"file              : {args.grasps}")
print(f"object            : {args.edge_mm:.1f} mm cube (half-extent {args.edge_mm/2:.1f} mm)")

print("\nrelative position, mm")
mm = pos * 1000.0
for i, ax in enumerate("xyz"):
    c = mm[:, i]
    print(f"  {ax}: min {c.min():8.2f}  max {c.max():8.2f}  mean {c.mean():8.2f}  std {c.std():7.2f}  range {c.max()-c.min():7.2f}")
print(f"  |p|: min {mm.norm(dim=1).min():8.2f}  max {mm.norm(dim=1).max():8.2f}  mean {mm.norm(dim=1).mean():8.2f}")


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


R = quat_to_mat(quat)
# The gripper's local +Z is the approach direction (metadata: "quat is (w,x,y,z), with +Z along
# approach"). Expressed in the OBJECT frame, this says which face the hand comes at.
approach = R[:, :, 2]
faces = {"+x": 0, "-x": 0, "+y": 0, "-y": 0, "+z": 0, "-z": 0}
names = ["+x", "-x", "+y", "-y", "+z", "-z"]
axes = torch.tensor(
    [[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]]
)
best = (approach @ axes.T).argmax(dim=1)
for b in best.tolist():
    faces[names[b]] += 1
print("\napproach direction, binned to the nearest cube face (object frame)")
for k in names:
    v = faces[k]
    bar = "#" * int(40 * v / max(1, n))
    print(f"  {k}: {v:5d}  {100.0*v/max(1,n):5.1f}%  {bar}")
occupied = sum(1 for v in faces.values() if v > 0)
print(f"  faces represented: {occupied}/6")

# Orientation spread: pairwise geodesic angle on a subsample.
m = min(n, 400)
idx = torch.randperm(n)[:m]
Rs = R[idx]
rel = Rs.transpose(1, 2) @ Rs.unsqueeze(1)  # (m, m, 3, 3)
tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
ang = torch.arccos(torch.clamp((tr - 1) / 2, -1.0, 1.0)) * 180.0 / math.pi
iu = torch.triu_indices(m, m, offset=1)
pw = ang[iu[0], iu[1]]
print(f"\npairwise orientation angle, deg (subsample n={m})")
print(f"  min {pw.min():6.2f}  median {pw.median():6.2f}  mean {pw.mean():6.2f}  max {pw.max():6.2f}")

# Collapse detector: nearest-neighbour orientation distance. If most grasps have a near-zero
# nearest neighbour, the bank is a handful of modes duplicated, not a distribution.
big = ang + torch.eye(m) * 1e6
nn = big.min(dim=1).values
print(f"  nearest-neighbour angle: median {nn.median():6.2f}  p90 {nn.quantile(0.9):6.2f}  max {nn.max():6.2f}")

verdict = []
if occupied >= 4:
    verdict.append(f"PASS faces ({occupied}/6 represented)")
else:
    verdict.append(f"WARN faces (only {occupied}/6 represented -- narrow)")
if pw.median() > 30.0:
    verdict.append(f"PASS orientation spread (median {pw.median():.1f} deg)")
else:
    verdict.append(f"WARN orientation spread (median {pw.median():.1f} deg -- clustered)")
print("\n[BANK_QA] " + " | ".join(verdict))
