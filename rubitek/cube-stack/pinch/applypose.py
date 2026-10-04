"""Write a derived pinch posture and its axes into the repo constants.

Doing this by hand is how a posture and the vectors derived FROM that posture drift apart: the
grasp centre, the closing axis and the approach direction are all functions of the closed joint
angles, and the moment one is edited without the others the sampler places the hand using a
geometry the hand no longer has. That failure is silent -- the numbers stay plausible -- and it
already cost a full sampler run once.

So this takes pinchhold.py's posture and pinchaxes.py's vectors, checks they were derived from
the same posture, and rewrites all four constants together or not at all.
"""

from __future__ import annotations

import argparse
import numpy as np
import json
import os
import re
from pathlib import Path

REPO = Path("/home/mattephi/Work/Rubitek/UWLab-cubestack")
ACTIONS = REPO / "source/uwlab_assets/uwlab_assets/robots/ur10e_delto/actions.py"
DELTO_CFG = (REPO / "source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset"
             / "config/ur5e_robotiq_2f85/delto_cfg.py")
JOINTS = [f"rj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--pose", default="out/pinchhold.json")
ap.add_argument("--axes", default="out/pinchaxes_hold.json")
ap.add_argument("--clearance-radius-mm", type=float, default=0.0,
                help="object half-width + fingertip collider reach; the standoff the grasp "
                     "sampler's placement gate demands of each pad body origin (34 mm cube: "
                     "17.0 + 16.5, both measured rather than assumed)")
ap.add_argument("--centre", choices=("closed", "open"), default="closed",
                help="which pad midpoint the placement datum is measured at (see above)")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

pose = json.load(open(args.pose))
axes_all = json.load(open(args.axes))
axes = axes_all["closed"]
# grasp_center_offset is a PLACEMENT datum, not a closing one. The sampler drops the object at it
# with the hand held OPEN and then rejects any candidate whose collider points intersect the part,
# so the value has to be the midpoint of the OPEN pads. Writing the closed midpoint puts the object
# 5.5 mm off the jaw axis, which shortens one tip's standoff from 25.0 mm to 19.6 mm -- less than
# the fingertip collider's own reach past a 34 mm cube's face -- and the gate then rejects all
# 61440 candidates with rl_dg_1_tip 14.21 mm inside the object. Measured live in the sampler, the
# open pad midpoint is [0.0860, 0.0336, 0.1694]; axes["open"] gives it to 0.46 mm, axes["closed"]
# misses by 5.48 mm. The closing axis and the approach direction stay closed-posture quantities.
#
# It is not the open pad midpoint either. The gate requires ZERO intersection between any gripper
# collider point and the object, fingertips included, and a DELTO fingertip's collider reaches
# 16.5 mm past its body origin (measured in the sampler). Against a 34 mm cube (half-width 17 mm)
# each pad origin therefore has to stand at least 16.5 + 17 = 33.5 mm off the object centre, while
# the open jaw is only 50 mm wide -- 25 mm per side. The missing distance comes from backing the
# centre off along the approach axis, and with the closing axis perpendicular to the approach (which
# the optimiser now enforces) the geometry is a right triangle:
#
#     k = sqrt(clearance_radius^2 - (aperture/2)^2) = sqrt(33.5^2 - 25.0^2) = 22.3 mm
#
# The value that worked before this rebuild sat 22.6 mm back along the approach from the open pad
# midpoint, which is this same number arrived at by hand. Derived rather than inherited, it also
# tracks a change of object size or posture instead of silently going stale.
# Which of the two midpoints, though, is a real tension and not a preference. The closed midpoint
# is where the pads actually meet, so it is the only centre at which closing the hand grips the
# part -- delto_cfg.py:668 records an earlier iteration that used the fist metadata's centre
# instead and got 0 successes in 6144 attempts with all 25 hand bodies reading 0.000 N. The open
# midpoint (and any back-off from it) is what the placement gate wants. Both are satisfiable at
# once only by opening the PRE-GRASP wide enough that the gate clears while the object still sits
# at the closed midpoint, which is what --open-scale on the pose file is for.
_open = axes_all["closed"] if args.centre == "closed" else axes_all.get("open", axes)
_c = np.asarray(_open["grasp_center_offset"], dtype=float)
_a = np.asarray(axes["gripper_approach_direction"], dtype=float)
_a = _a / (np.linalg.norm(_a) + 1e-12)
_half_aperture = float(_open["pad_separation_mm"]) / 2000.0
_k = float(np.sqrt(max((args.clearance_radius_mm / 1000.0) ** 2 - _half_aperture**2, 0.0)))
placement_centre = [round(float(v), 6) for v in (_c - _k * _a)]

# The axes must come from THIS posture, not a previous one. Identity is the source file, not the
# measured pad separation: pinchaxes runs a bare hand with no ground plane and no cube, and the
# fingers settle about a millimetre differently there than on the bench, so apertures that
# legitimately match still differ by ~5%.
src = axes_all.get("source_pose")
if src is None:
    raise SystemExit(f"REFUSING: {args.axes} carries no source_pose stamp; re-run pinchaxes.py")
if os.path.abspath(src) != os.path.abspath(args.pose):
    raise SystemExit(f"REFUSING: axes derived from {src}, but writing posture {args.pose}")
ap_pose = float(pose["close"]["aperture_mm"])
ap_axes = float(axes["pad_separation_mm"])
if abs(ap_pose - ap_axes) > 3.0:
    raise SystemExit(
        f"REFUSING: axes measured a {ap_axes:.2f} mm jaw against the posture's {ap_pose:.2f} mm. "
        "Same source file, so this is not a mix-up -- something about the posture is unstable.")


def block(name: str, joints: dict) -> str:
    body = "".join(f'    "{n}": {float(joints[n]):.6f},\n' for n in JOINTS)
    return f"{name} = {{\n{body}}}"


def sub_block(text: str, name: str, joints: dict) -> str:
    pat = re.compile(rf"^{re.escape(name)} = \{{.*?^\}}", re.S | re.M)
    out, n = pat.subn(lambda _: block(name, joints), text, count=1)
    if n != 1:
        raise SystemExit(f"REFUSING: found {n} definitions of {name}, expected exactly 1")
    return out


src = ACTIONS.read_text()
src = sub_block(src, "DELTO_HAND_PINCH_CLOSE_JOINT_POS", pose["close"]["joints"])
src = sub_block(src, "DELTO_HAND_PINCH_OPEN_JOINT_POS", pose["open"]["joints"])
c = placement_centre
src, n = re.subn(r"^DELTO_PINCH_GRASP_CENTER_OFFSET = \([^)]*\)",
                 f"DELTO_PINCH_GRASP_CENTER_OFFSET = ({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f})",
                 src, count=1, flags=re.M)
if n != 1:
    raise SystemExit("REFUSING: DELTO_PINCH_GRASP_CENTER_OFFSET not found")

cfg = DELTO_CFG.read_text()
for key, vec in (("grasp_align_axis", axes["grasp_align_axis"]),
                 ("orientation_sample_axis", axes["grasp_align_axis"]),
                 ("gripper_approach_direction", axes["gripper_approach_direction"])):
    lit = f"({vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f})"
    pat = rf'^(\s*self\.events\.grasp_sampling\.params\["{key}"\] = )\([^)]*\)'
    cfg, n = re.subn(pat, rf"\g<1>{lit}", cfg, count=1, flags=re.M | re.S)
    if n != 1:
        raise SystemExit(
            f'REFUSING: self.events.grasp_sampling.params["{key}"] = (...) not found '
            f"in delto_cfg.py (matched {n})")

# The SUCCESS CHECK's copy of the approach axis lives in a third file and was being maintained by
# hand. It drifted: the committed value was derived from a posture two revisions old. Same class of
# bug as the sampler axes -- a posture-derived constant kept somewhere the posture update does not
# reach -- so it is written from the same source here.
CUBESTACK = (REPO / "source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset"
             / "config/ur5e_robotiq_2f85/ur10e_delto_cubestack_cfg.py")
cs = CUBESTACK.read_text()
a = axes["gripper_approach_direction"]
lit = f"(\n                {a[0]:.6f}, {a[1]:.6f}, {a[2]:.6f})"
cs, n_cs = re.subn(
    r'(self\.terminations\.success\.params\["gripper_approach_direction"\] = )\([^)]*\)',
    lambda m: m.group(1) + lit, cs, flags=re.S)
if n_cs == 0:
    raise SystemExit("REFUSING: no terminations.success gripper_approach_direction found")

print(f"posture   closed jaw {ap_pose:.2f} mm, held {pose['held_rate'] * 100:.1f}%, "
      f"true pinch {pose['pinch_rate'] * 100:.1f}%")
print(f"centre    {c}")
print(f"align     {axes['grasp_align_axis']}")
print(f"centre    {placement_centre}   (open pad midpoint backed off {_k * 1000:.1f} mm "
      f"along the approach -- the placement datum)")
print(f"approach  {axes['gripper_approach_direction']}")
if args.dry_run:
    print("DRY RUN, nothing written")
else:
    ACTIONS.write_text(src)
    DELTO_CFG.write_text(cfg)
    CUBESTACK.write_text(cs)
    print(f"WROTE {ACTIONS.name}, {DELTO_CFG.name}, {CUBESTACK.name} ({n_cs} success axes)")
