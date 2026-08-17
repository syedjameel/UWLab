import sys, collections
from pxr import Usd, UsdGeom, UsdPhysics

path = sys.argv[1]
stage = Usd.Stage.Open(path)
print("=" * 90)
print("STAGE:", path)
print("defaultPrim:", stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None)

# ---- articulation root / self collisions
print("\n--- ARTICULATION ROOT / SELF COLLISIONS ---")
for prim in stage.Traverse():
    for attr in prim.GetAttributes():
        n = attr.GetName()
        if "enabledSelfCollisions" in n or "ArticulationRootAPI" in n:
            print(f"  {prim.GetPath()}  {n} = {attr.Get()}  (authored={attr.HasAuthoredValue()})")
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        a = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
        print(f"  [ArticulationRootAPI] {prim.GetPath()}  enabledSelfCollisions="
              f"{a.Get() if a else 'ATTR-ABSENT'} authored={a.HasAuthoredValue() if a else False}")
        for nm in ("physxArticulation:solverPositionIterationCount",
                   "physxArticulation:solverVelocityIterationCount"):
            b = prim.GetAttribute(nm)
            if b:
                print(f"      {nm} = {b.Get()}")

# ---- collider approximations
print("\n--- COLLIDER APPROXIMATIONS ---")
approx = collections.Counter()
by_approx = collections.defaultdict(list)
hulls = collections.Counter()
ncoll = 0
offsets = collections.Counter()
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        ncoll += 1
        p = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            a = prim.GetAttribute("physics:approximation")
            v = a.Get() if a and a.HasAuthoredValue() else "<UNAUTHORED>"
        else:
            v = "<no MeshCollisionAPI: " + prim.GetTypeName() + ">"
        approx[str(v)] += 1
        if len(by_approx[str(v)]) < 6:
            by_approx[str(v)].append(p)
        h = prim.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls")
        if h and h.HasAuthoredValue():
            hulls[h.Get()] += 1
        for nm in ("physxCollision:contactOffset", "physxCollision:restOffset"):
            c = prim.GetAttribute(nm)
            if c and c.HasAuthoredValue():
                offsets[(nm, c.Get())] += 1
print(f"  total prims with CollisionAPI: {ncoll}")
for k, v in approx.most_common():
    print(f"    approximation={k:35s} count={v}")
    for s in by_approx[k]:
        print(f"        e.g. {s}")
print("  maxConvexHulls authored:", dict(hulls) or "NONE AUTHORED")
print("  contact/rest offsets authored:", dict(offsets) or "NONE AUTHORED")

# ---- prim hierarchy: top-level children of default prim (for contact sensor paths)
print("\n--- TOP-LEVEL CHILDREN OF DEFAULT PRIM ---")
dp = stage.GetDefaultPrim()
if dp:
    for c in dp.GetChildren():
        print(f"    {c.GetName():30s} type={c.GetTypeName()}")

# ---- fingertip prims
print("\n--- PRIMS MATCHING *_tip / *tip* ---")
tips = [str(p.GetPath()) for p in stage.Traverse() if "tip" in p.GetName().lower()]
for t in tips[:40]:
    print("   ", t)
print(f"   ({len(tips)} total)")

# ---- physics materials
print("\n--- PHYSICS MATERIALS ---")
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.MaterialAPI) or prim.HasAPI(UsdPhysics.MaterialAPI):
        sf = prim.GetAttribute("physics:staticFriction")
        df = prim.GetAttribute("physics:dynamicFriction")
        r = prim.GetAttribute("physics:restitution")
        print(f"    {prim.GetPath()}  static={sf.Get() if sf else None} dyn={df.Get() if df else None} rest={r.Get() if r else None}")
