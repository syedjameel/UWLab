"""Author the convex decomposition the table leg's own converter asked for and never got.

THE DEFECT, measured rather than suspected. Every table-leg run in this campaign logs::

    PhysicsUSD: Parse collision - triangle mesh collision (approximation None/MeshSimplification)
    cannot be a part of a dynamic body, falling back to convexHull approximation:
    /World/envs/env_0/Object/geometry/mesh

and the asset explains why: the mesh prim inside ``Props/instanceable_meshes.usd`` carries ONLY
``PhysicsCollisionAPI``. ``UsdPhysics.MeshCollisionAPI`` is absent, so the ``physics:approximation``
attribute does not exist, and ``PhysxConvexDecompositionCollisionAPI`` is absent too. PhysX therefore
sees an unapproximated triangle mesh on a dynamic body and substitutes a convex hull.

Meanwhile the shipped ``config.yaml`` records what was REQUESTED::

    mesh_collision_props:
      usd_func: pxr.UsdPhysics:MeshCollisionAPI
      physx_func: pxr.PhysxSchema:PhysxConvexDecompositionCollisionAPI
      hull_vertex_limit: 64
      max_convex_hulls: 32
      voxel_resolution: 500000

So the request was recorded and the result was not applied. This script applies it.

WHY IT PLAUSIBLY MATTERS FOR THIS TASK, and why it was not worth fixing for Lift. A convex hull of a
threaded rod FILLS THE THREAD RELIEF -- the converter's own docstring rejects convexHull for exactly
that reason, calling the relief "the one grippable feature". Lifting does not need it: the leg Lift
policy reached 0.929 with the hull. But the thing the Reorient policy cannot do is HOLD an
orientation, and holding a rod against gravity-torque is precisely where a finger seated in a relief
beats a finger on a smooth convex surface.

PROVENANCE NOTE THAT MUST TRAVEL WITH ANY RESULT FROM THIS ASSET: the shipped USD is md5-identical to
the reference's (4880401d3bee54867dbe0780143d8645), so the certified 92.87 percent reference run ALSO
ran on the hull. Fixing this makes our asset BETTER than the reference's, not equal to it -- any
number measured on it is no longer strictly comparable to that 92.87 percent.

Writes a SEPARATE asset directory and leaves the original untouched, so the two can be A/B'd.
"""

import shutil
import sys
from pathlib import Path

from pxr import Sdf, Usd, UsdPhysics

try:  # ships with Isaac Sim, not with a bare pxr install
    from pxr import PhysxSchema
except ImportError:  # pragma: no cover - environment dependent
    PhysxSchema = None

SRC = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/dom-iva/github.com/orel/lerobot/UWLab_ur5edelto/source/uwlab_assets/uwlab_assets"
    "/local/Props/FurnitureBench/SquareTableLeg200mm"
)
DST = SRC.with_name("SquareTableLeg200mmDecomp")
MESH_PATH = "/square_table_leg4_200mm_merged/geometry/mesh"
# Straight from the converter's own config.yaml, so this reproduces the intent rather than inventing
# a new one.
HULL_VERTEX_LIMIT = 64
MAX_CONVEX_HULLS = 32
VOXEL_RESOLUTION = 500000


def main() -> int:
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    # The top-level USD references ./Props/instanceable_meshes.usd by RELATIVE path, so copying the
    # whole directory keeps that reference resolving inside the copy and nothing needs re-pathing.
    payload = DST / "Props" / "instanceable_meshes.usd"

    stage = Usd.Stage.Open(str(payload))
    prim = stage.GetPrimAtPath(MESH_PATH)
    if not prim or not prim.IsValid():
        print("REFUSING: %s not found in %s" % (MESH_PATH, payload))
        return 1

    before = list(prim.GetAppliedSchemas())
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)
    # AUTHOR THE PhysX API BY NAME. PhysxSchema is an Isaac EXTENSION module, not part of a plain
    # pxr install, and it is not importable from either venv here without booting the app. An applied
    # API schema is just its name in the prim's apiSchemas list plus its namespaced attributes, which
    # is precisely what the generated class writes -- so authoring it directly produces a
    # byte-equivalent result and drops the dependency. The check that matters is behavioural anyway:
    # PhysX must stop logging the convexHull fallback.
    if PhysxSchema is not None:
        decomp = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
        decomp.CreateHullVertexLimitAttr().Set(HULL_VERTEX_LIMIT)
        decomp.CreateMaxConvexHullsAttr().Set(MAX_CONVEX_HULLS)
        decomp.CreateVoxelResolutionAttr().Set(VOXEL_RESOLUTION)
    else:
        prim.AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
        for attr_name, value in (
            ("physxConvexDecompositionCollision:hullVertexLimit", HULL_VERTEX_LIMIT),
            ("physxConvexDecompositionCollision:maxConvexHulls", MAX_CONVEX_HULLS),
            ("physxConvexDecompositionCollision:voxelResolution", VOXEL_RESOLUTION),
        ):
            prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.UInt).Set(value)
    stage.GetRootLayer().Save()

    # RE-READ FROM DISK. An in-memory stage reports what was requested; only a fresh open reports
    # what was written, and this whole script exists because a converter's request and its output
    # disagreed.
    check = Usd.Stage.Open(str(payload))
    got = check.GetPrimAtPath(MESH_PATH)
    schemas = list(got.GetAppliedSchemas())
    approx = got.GetAttribute("physics:approximation").Get()
    print("mesh prim   :", MESH_PATH)
    print("schemas before:", before)
    print("schemas after :", schemas)
    print("approximation :", approx)
    for name, attr in (
        ("hullVertexLimit", "physxConvexDecompositionCollision:hullVertexLimit"),
        ("maxConvexHulls", "physxConvexDecompositionCollision:maxConvexHulls"),
        ("voxelResolution", "physxConvexDecompositionCollision:voxelResolution"),
    ):
        print("  %-16s %s" % (name, got.GetAttribute(attr).Get()))

    # WHAT IS LOAD-BEARING vs WHAT IS TUNING. The convexHull fallback is triggered purely by the
    # ABSENCE of physics:approximation, so MeshCollisionAPI + the convexDecomposition token is the
    # whole fix. PhysxConvexDecompositionCollisionAPI only TUNES the decomposition (hull count,
    # vertex limit, voxel resolution).
    #
    # AddAppliedSchema silently drops a schema type this USD build does not know, and that is what
    # happens here -- the three physxConvexDecompositionCollision:* attributes are written but the
    # API name does not survive a re-read, so PhysX will decompose at ITS DEFAULTS rather than at the
    # converter's 64/32/500000. Say so instead of pretending otherwise: a run on this asset is a run
    # on a default-tuned decomposition.
    tuned = "PhysxConvexDecompositionCollisionAPI" in schemas
    ok = approx == UsdPhysics.Tokens.convexDecomposition and "PhysicsMeshCollisionAPI" in schemas
    print("fallback fixed :", ok, "(physics:approximation is what PhysX reads)")
    print("physx tuning   :", tuned, "" if tuned else "-> decomposition runs at PhysX DEFAULT tuning")
    print("RESULT:", "OK" if ok else "FAILED", "->", DST)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
