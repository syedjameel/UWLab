"""Re-author our DELTO hand colliders from SDF to convexHull, matching the certified reference.

MEASURED, not assumed (scratchpad/probe_colliders.py):
    ours ur5e_delto.usd   23 hand bodies 'sdf',  5 fingertips 'convexHull'
    ref  ur10e_delto_...  25 hand bodies 'convexHull', 1 'convexDecomposition', ZERO sdf

WHY THIS IS THE SUSPECT. At matched epoch our good_finger_contact is at reference parity (0.77-0.80
vs 0.845) while object_upward_motion is 11-15x behind (0.0021 vs 0.0321 at ep 866). The contact
reward reads ONLY the five fingertip sensors -- convexHull on both sides -- so it is blind to the
phalanges and palm, which are the bodies that cage an object. Those are the 23 that differ. The
signature is therefore "touches like the reference, grips unlike it", which is what a different
collision representation on the non-tip bodies predicts and what an actuator or reward difference
does not.

WRITES A NEW FILE. The original is left untouched so the A6000 control runs keep the asset they were
launched against; a run that silently changes its own robot mid-flight is not a control.
"""

from __future__ import annotations

import shutil
import sys

from pxr import Usd, UsdPhysics

SRC = "/home/dom-iva/github.com/orel/lerobot/UWLab_ur5edelto/source/uwlab_assets/uwlab_assets/local/Robots/Ur5eDelto/ur5e_delto.usd"
DST = "/home/dom-iva/github.com/orel/lerobot/UWLab_ur5edelto/source/uwlab_assets/uwlab_assets/local/Robots/Ur5eDelto/ur5e_delto_convexhull.usd"

shutil.copyfile(SRC, DST)
stage = Usd.Stage.Open(DST)
if stage is None:
    raise SystemExit(f"cannot open {DST}")

changed = []
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    mesh = UsdPhysics.MeshCollisionAPI(prim)
    attr = mesh.GetApproximationAttr()
    if attr is None or not attr.HasAuthoredValue():
        continue
    if attr.Get() != "sdf":
        continue
    attr.Set("convexHull")
    changed.append(str(prim.GetPath()))

stage.GetRootLayer().Save()

# VERIFY BY RE-READING FROM DISK, not by trusting the in-memory stage. A save that silently did not
# land would otherwise pass -- the defect class where the check is the bug.
verify = Usd.Stage.Open(DST)
remaining = [
    str(p.GetPath())
    for p in verify.Traverse()
    if p.HasAPI(UsdPhysics.CollisionAPI)
    and UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr()
    and UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == "sdf"
]

print(f"rewrote {len(changed)} sdf -> convexHull")
for p in changed[:6]:
    print(f"  {p}")
print(f"  ... ({len(changed)} total)" if len(changed) > 6 else "")
print(f"sdf prims remaining after save: {len(remaining)}")
print(f"wrote {DST}")
if remaining or len(changed) != 23:
    print(f"FAIL: expected 23 rewritten and 0 remaining, got {len(changed)} and {len(remaining)}")
    sys.exit(1)
print("REAUTHOR_OK")
