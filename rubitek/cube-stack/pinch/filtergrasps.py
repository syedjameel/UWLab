"""Keep only the grasps a table-mounted arm can actually reach.

The sampler draws grasp ORIENTATIONS uniformly and gates candidates on collision with the OBJECT.
Nothing in that loop knows the object sits on a table, so slightly over half the bank approaches
the part from underneath -- measured on the 321-grasp pinch bank, 53.6% of approach axes point up,
and the hand is placed as much as 161 mm BELOW the object. For a 34 mm cube resting 21 mm above the
work surface, those grasps are reachable only by putting the wrist through the tabletop.

That is what fills the reset banks with arms under the table, and it is why check_reset_state_success
rejects on orient_down (27% pass): the gate is catching unreachable samples one episode at a time,
after the expensive part has already run. Filtering the bank up front is the same decision made
earlier and far more cheaply.

The threshold is on the approach axis rotated into the object frame: keep grasps whose approach has
a downward component of at least --min-down. 0.0 would keep every horizontal approach, which is
still reachable; the default is deliberately mild.
"""

from __future__ import annotations

import argparse
import shutil

import torch

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--approach", type=float, nargs=3, default=[0.67381, 0.268005, 0.688589],
                help="the gripper's local approach axis (pinchaxes_hold.json)")
ap.add_argument("--min-down", type=float, default=0.10,
                help="require approach_world_z <= -min_down, i.e. actually pointing downward")
args = ap.parse_args()

bank = torch.load(args.inp, map_location="cpu")
d = bank["grasp_relative_pose"]
f = lambda x: torch.stack(x) if isinstance(x, list) else x  # noqa: E731
pos, quat = f(d["relative_position"]), f(d["relative_orientation"])

w, x, y, z = quat.unbind(-1)
R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                -1).reshape(-1, 3, 3)
a = torch.tensor(args.approach, dtype=pos.dtype)
approach_z = (R @ a.view(1, 3, 1)).squeeze(-1)[:, 2]
keep = approach_z <= -args.min_down

print(f"in  : {pos.shape[0]} grasps")
print(f"keep: {int(keep.sum())} ({keep.float().mean() * 100:.1f}%) with approach_z <= {-args.min_down:.2f}")
print(f"drop: {int((~keep).sum())} that approach the part from below")
if int(keep.sum()) == 0:
    raise SystemExit("REFUSING: filter would empty the bank")

d["relative_position"] = pos[keep]
d["relative_orientation"] = quat[keep]
gj = d.get("gripper_joint_positions")
if isinstance(gj, dict):
    for k in gj:
        v = f(gj[k])
        if torch.is_tensor(v) and v.shape[0] == pos.shape[0]:
            gj[k] = v[keep]
elif gj is not None:
    gj = f(gj)
    d["gripper_joint_positions"] = gj[keep]

shutil.copy(args.inp, args.inp + ".prefilter")
torch.save(bank, args.out)
print(f"WROTE {args.out}  (original kept at {args.inp}.prefilter)")
