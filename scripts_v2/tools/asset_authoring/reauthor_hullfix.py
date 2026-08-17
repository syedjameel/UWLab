"""Author the collider set that lets self-collisions be turned ON.

WHAT THE MEASUREMENT SAID, and it is a topology difference, not a tuning one.

Our graft splits the DELTO palm assembly into THREE rigid bodies welded by fixed joints:

    wrist_3_link --fixed--> rl_dg_mount --fixed--> rl_dg_base --fixed--> rl_dg_palm
    rl_dg_mount  --revolute--> rl_dg_{1..5}_1

The certified reference (ur10e_delto_optimized_separate_tips_limited_jnts_self_collision.usd)
has ONE body there -- rl_dg_mount, approximation convexDecomposition -- and 25 convexHull
everywhere else, zero sdf.

That difference is the whole bug. Each proximal phalanx joints to rl_dg_mount, so PhysX
auto-filters phalanx-vs-mount. It does NOT filter phalanx-vs-palm or phalanx-vs-base, because
those are separate bodies two joints away. In the reference that same geometry IS the mount, so
the pair does not exist. probe_pairs.py measured the consequence at the authored spawn posture,
with convexHull on everything:

    38.5%  rl_dg_1_1 <-> rl_dg_palm        20.9%  rl_dg_2_1 <-> rl_dg_palm
    28.4%  rl_dg_1_1 <-> rl_dg_base        13.0%  rl_dg_1_2 <-> rl_dg_palm
    23.6%  rl_dg_4_1 <-> rl_dg_palm        12.3%  rl_dg_5_1 <-> rl_dg_palm
    22.3%  rl_dg_3_1 <-> rl_dg_palm

(22 ADJACENT pairs also overlap and are harmless -- PhysX always filters jointed pairs. That is
the control which proves overlap alone is not the problem; being non-adjacent is.)

TWO FIXES, both needed, for two different reasons:

  1. convexDecomposition on rl_dg_mount + rl_dg_base + rl_dg_palm. The palm shell is CONCAVE --
     the finger roots sit in recesses. A convex hull fills those recesses and swallows the roots.
     Decomposition follows the concavity, which is why the reference can afford self-collisions
     and is also why our sdf build never diverged. This is what kills rl_dg_1_2 <-> palm, a pair
     the reference genuinely does check (mount-vs-_2 is non-adjacent there too).

  2. UsdPhysics.FilteredPairsAPI on {rl_dg_base, rl_dg_palm} x {rl_dg_N_1}. Decomposition alone
     may not fully separate a phalanx root that is seated INSIDE the palm collar. Filtering these
     is reference-faithful rather than a cheat: base and palm are rigidly welded to rl_dg_mount,
     which is the phalanx's own joint parent, so the pair is kinematically identical to
     mount-vs-N_1 -- a pair PhysX already filters as adjacent, and which the reference cannot
     even express. Every OTHER pair stays live, including palm vs the distal phalanges and tips,
     which is a real constraint (a finger curling into the palm) and stays enforced.

WRITES A NEW FILE. ur5e_delto.usd and ur5e_delto_convexhull.usd are left untouched so the running
control runs keep the exact asset they were launched against.
"""

from __future__ import annotations

import shutil
import sys

from pxr import Usd, UsdPhysics

ROOT = "/home/dom-iva/github.com/orel/lerobot/UWLab_ur5edelto/source/uwlab_assets/uwlab_assets/local/Robots/Ur5eDelto"
SRC = f"{ROOT}/ur5e_delto.usd"
DST = f"{ROOT}/ur5e_delto_hullfix.usd"

# The three welded bodies that the reference expresses as one.
DECOMP_BODIES = {"rl_dg_mount", "rl_dg_base", "rl_dg_palm"}
# Filter only the welded-shell-vs-phalanx-root pairs, i.e. what the reference's topology filters.
FILTER_A = ("rl_dg_base", "rl_dg_palm")
FILTER_B = tuple(f"rl_dg_{i}_1" for i in range(1, 6))

shutil.copyfile(SRC, DST)
stage = Usd.Stage.Open(DST)
if stage is None:
    raise SystemExit(f"cannot open {DST}")


def owning_body(prim):
    """Nearest ancestor carrying RigidBodyAPI -- the body PhysX attributes this collider to."""
    p = prim
    while p and p.IsValid():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p
        p = p.GetParent()
    return None


# ---- 1. approximations -------------------------------------------------------------------
to_hull, to_decomp, unattributed = [], [], []
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    attr = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
    if attr is None or not attr.HasAuthoredValue():
        continue
    body = owning_body(prim)
    if body is None:
        unattributed.append(str(prim.GetPath()))
        continue
    if body.GetName() in DECOMP_BODIES:
        attr.Set("convexDecomposition")
        to_decomp.append(body.GetName())
    elif attr.Get() == "sdf":
        attr.Set("convexHull")
        to_hull.append(body.GetName())

# ---- 2. filtered pairs -------------------------------------------------------------------
bodies = {}
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        bodies[prim.GetName()] = prim

missing = [n for n in FILTER_A + FILTER_B if n not in bodies]
if missing:
    raise SystemExit(f"REFUSING: bodies not found in {DST}: {missing}")

pairs = []
for a in FILTER_A:
    api = UsdPhysics.FilteredPairsAPI.Apply(bodies[a])
    rel = api.CreateFilteredPairsRel()
    for b in FILTER_B:
        rel.AddTarget(bodies[b].GetPath())
        pairs.append((a, b))

stage.GetRootLayer().Save()

# ---- 3. verify by RE-READING FROM DISK ---------------------------------------------------
# Never trust the in-memory stage: a save that silently did not land would otherwise pass. This
# is the defect class where the check is the bug.
v = Usd.Stage.Open(DST)
approx, filt = {}, {}
for prim in v.Traverse():
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        attr = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
        if attr and attr.HasAuthoredValue():
            b = owning_body(prim)
            if b is not None:
                approx.setdefault(attr.Get(), []).append(b.GetName())
    if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
        tg = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
        filt[prim.GetName()] = [t.name for t in tg]

print(f"rewrote {len(to_hull)} sdf -> convexHull")
print(f"set {len(to_decomp)} -> convexDecomposition: {sorted(set(to_decomp))}")
print(f"authored {len(pairs)} filtered pairs")
for k, vv in sorted(approx.items()):
    print(f"  on disk: {k:22s} {len(vv)} colliders")
for k, vv in sorted(filt.items()):
    print(f"  on disk: FilteredPairs {k} -> {vv}")

errs = []
if approx.get("sdf"):
    errs.append(f"{len(approx['sdf'])} sdf colliders remain")
if sorted(set(approx.get("convexDecomposition", []))) != sorted(DECOMP_BODIES):
    errs.append(f"convexDecomposition bodies wrong: {sorted(set(approx.get('convexDecomposition', [])))}")
if len(approx.get("convexHull", [])) != 25:
    errs.append(f"expected 25 convexHull colliders (20 phalanges + 5 tips), got {len(approx.get('convexHull', []))}")
if sorted(filt) != sorted(FILTER_A) or any(sorted(filt[a]) != sorted(FILTER_B) for a in filt):
    errs.append(f"filtered pairs wrong: {filt}")
if unattributed:
    errs.append(f"unattributed colliders: {unattributed}")

if errs:
    for e in errs:
        print(f"FAIL: {e}")
    sys.exit(1)
print(f"wrote {DST}")
print("HULLFIX_OK")
