"""The reference-faithful shipping asset: filter ONLY the two bodies the reference does not have.

hullfix  filters {base,palm} x {N_1}            -- minimal, measured to cap closure 25 percent
hullfix2 filters {mount,base,palm} x {all 25}   -- diagnostic, finger-vs-finger only
hullfix3 filters {base,palm}  x {all 25}        -- THIS ONE

The argument for hullfix3 is topological, not empirical. The reference hand has exactly one shell
body, rl_dg_mount, and PhysX checks it against every finger body except its own jointed children
(the N_1 roots). Our graft invented rl_dg_base and rl_dg_palm -- extra rigid bodies, welded to the
mount by fixed joints, that no reference robot and no VALIK robot has. Filtering those two against
every finger therefore REMOVES geometry the reference never had, while leaving mount-vs-finger live
exactly as the reference leaves it. Finger-vs-finger stays fully live, which is the constraint the
whole exercise is about: the 94.9 percent policy fused finger 3 through finger 4 on 92.5 percent of
steps.

Honest limitation, to state in any writeup: our rl_dg_mount is a SMALLER volume than the reference's,
because the reference's mount also contains the geometry our base and palm carry. So a finger can
curl slightly deeper here than it could there. That is over-permissive relative to the reference and
strictly better than the alternative, which is fingers passing through each other.
"""

from __future__ import annotations

import shutil
import sys

from pxr import Usd, UsdPhysics

ROOT = "/home/dom-iva/github.com/orel/lerobot/UWLab_ur5edelto/source/uwlab_assets/uwlab_assets/local/Robots/Ur5eDelto"
# Optional CLI overrides, added 2026-08-17 to allow a rebuild-and-compare run without ever writing
# to the live asset paths (see the "Verified reproducibility" section of README.md in this
# directory). No args -> SRC/DST are byte-identical to what this script has always defaulted to;
# DST is independent of SRC on purpose, so a caller can redirect the output without touching the
# input, unlike reauthor_leg_decomp.py's DST-derived-from-SRC pattern.
SRC = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/ur5e_delto.usd"
DST = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/ur5e_delto_hullfix3.usd"

DECOMP = {"rl_dg_mount", "rl_dg_base", "rl_dg_palm"}
SHELLS = ("rl_dg_base", "rl_dg_palm")   # the two bodies the reference does not have
FINGERS = tuple(f"rl_dg_{i}_{j}" for i in range(1, 6) for j in range(1, 5)) + \
          tuple(f"rl_dg_{i}_tip" for i in range(1, 6))

shutil.copyfile(SRC, DST)
stage = Usd.Stage.Open(DST)


def owning_body(prim):
    p = prim
    while p and p.IsValid():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p
        p = p.GetParent()
    return None


hull = decomp = 0
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    attr = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
    if attr is None or not attr.HasAuthoredValue():
        continue
    b = owning_body(prim)
    if b is None:
        continue
    if b.GetName() in DECOMP:
        attr.Set("convexDecomposition"); decomp += 1
    elif attr.Get() == "sdf":
        attr.Set("convexHull"); hull += 1

bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
missing = [n for n in SHELLS + FINGERS if n not in bodies]
if missing:
    raise SystemExit(f"REFUSING: bodies not found: {missing}")
for s in SHELLS:
    rel = UsdPhysics.FilteredPairsAPI.Apply(bodies[s]).CreateFilteredPairsRel()
    for f in FINGERS:
        rel.AddTarget(bodies[f].GetPath())
stage.GetRootLayer().Save()

v = Usd.Stage.Open(DST)
ap, filt = {}, {}
for prim in v.Traverse():
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        a = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
        if a and a.HasAuthoredValue():
            ap[a.Get()] = ap.get(a.Get(), 0) + 1
    if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
        filt[prim.GetName()] = len(UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets())
print(f"rewrote {hull} -> convexHull, {decomp} -> convexDecomposition")
print(f"on disk approximations: {ap}")
print(f"on disk filtered pairs: {filt}")
errs = []
if ap.get("sdf"):
    errs.append(f"{ap['sdf']} sdf remain")
if ap.get("convexHull") != 25 or ap.get("convexDecomposition") != 3:
    errs.append(f"expected 25 hull + 3 decomp, got {ap}")
if sorted(filt) != sorted(SHELLS) or any(filt[s] != 25 for s in filt):
    errs.append(f"filtered pairs wrong: {filt}")
if errs:
    for e in errs:
        print(f"FAIL: {e}")
    sys.exit(1)
print(f"wrote {DST}")
print("HULLFIX3_OK")
