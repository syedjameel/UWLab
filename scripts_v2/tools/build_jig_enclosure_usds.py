# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the JIG (insertive) and BOTTOM ENCLOSURE (receptive) task assets from their STLs.

New task: the robot picks the alignment JIG and seats it onto the (kinematic, open-side-up)
bottom enclosure. Assets follow the standard OmniReset structure (see ``omnireset_asset_utils``
and ``build_pcb_usd.py`` / ``build_openbox_usd.py``): a single Xform root with RigidBodyAPI
(default prim), ``visuals`` mesh + invisible ``collisions`` mesh, a baked PhysicsMaterial
(friction 0.5; grasp sampling relies on it), NO MassAPI (PhysX auto-computes mass, matching
the reference assets), and a ``metadata.yaml`` beside the USD.

STL handling (source meshes committed next to the outputs):
* units mm -> m (x0.001);
* enclosure STL is exported Y-up -> rotated to Z-up;
* origin moved to the bounding-box center (both objects, like the pcb/openbox assets);
* collision uses PhysX convexDecomposition (both parts are concave: window frame / open shell).

ASSEMBLED OFFSET: the enclosure's mating point is set to its TOP-RIM plane center and the
jig's to its bottom-center -- a PROVISIONAL convention pending the CAD assembly-position
export (the jig's registration lips may seat it a few mm lower). Refine with
``--seat-drop`` once the real seating depth is known.

    ./uwlab.sh -p scripts_v2/tools/build_jig_enclosure_usds.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import trimesh

from omnireset_asset_utils import add_box, create_stage, write_metadata
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

_LOCAL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "source/uwlab_assets/uwlab_assets/local/Props/Custom",
)


def add_trimesh(stage, prim_path, mesh: trimesh.Trimesh, *, collision: bool,
                color=None, material_path=None, approximation="convexDecomposition"):
    """Author a trimesh as a USD Mesh (visual, or a collider with the given approximation).

    Approximations used here (mate-fidelity matters -- convex decomposition floats the jig
    ~3.5+ mm proud of the true seat by filling the corner cone holes / wrapping the pillars):
    * "none" -- exact triangle mesh. Valid for KINEMATIC bodies only (the enclosure is
      kinematic in every task) -> real pillars, real recesses.
    * "sdf"  -- PhysX SDF collider, exact concave collision for DYNAMIC bodies (the jig's
      cone holes), with a PhysxSDFMeshCollisionAPI resolution of 256.
    """
    m = UsdGeom.Mesh.Define(stage, prim_path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in mesh.vertices.astype(float)]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mesh.faces)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(mesh.faces.flatten().tolist()))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = mesh.bounds
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))
    m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in np.repeat(mesh.face_normals, 3, axis=0)]))
    m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    if collision:
        UsdGeom.Imageable(m).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        mc = UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim())
        mc.CreateApproximationAttr(approximation)
        if approximation == "convexDecompositionHQ":
            # High-quality decomposition: SDF jaw contacts fling the torque-controlled arm
            # (all knobs tried); default decomposition seats the skirt LOPSIDED (yaw0 22.6 vs
            # yaw180 14.6 mm). Many small hulls + shrink-wrap keep convex-contact stability
            # while capturing the skirt/lip geometry symmetrically.
            mc.CreateApproximationAttr("convexDecomposition")
            m.GetPrim().AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
            for name, vt, val in [("maxConvexHulls", Sdf.ValueTypeNames.Int, 128),
                                  ("hullVertexLimit", Sdf.ValueTypeNames.Int, 64),
                                  ("errorPercentage", Sdf.ValueTypeNames.Float, 0.5),
                                  ("voxelResolution", Sdf.ValueTypeNames.Int, 1000000),
                                  ("shrinkWrap", Sdf.ValueTypeNames.Bool, True)]:
                m.GetPrim().CreateAttribute(f"physxConvexDecompositionCollision:{name}", vt).Set(val)
        if approximation == "sdf":
            # Raw authoring (PhysxSchema isn't in the bare pxr install): apply the API by
            # name and write its resolution attribute; Isaac's PhysX parser reads both.
            m.GetPrim().AddAppliedSchema("PhysxSDFMeshCollisionAPI")
            # Resolution 64 (not 256): high-res SDFs produce spiky contact normals under the
            # stiff jaw drive + torque-controlled arm (measured: arm flung, |qd| to 576 rad/s).
            # 64 -> ~2.6 mm cells, still resolves the skirt/pillar seating. Contact offset
            # raised for gentler engagement on reload.
            m.GetPrim().CreateAttribute("physxSDFMeshCollision:sdfResolution",
                                        Sdf.ValueTypeNames.Int).Set(64)
            m.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
            m.GetPrim().CreateAttribute("physxCollision:contactOffset",
                                        Sdf.ValueTypeNames.Float).Set(0.003)
            m.GetPrim().CreateAttribute("physxCollision:restOffset",
                                        Sdf.ValueTypeNames.Float).Set(0.0)
        if material_path is not None:
            binding = UsdShade.MaterialBindingAPI.Apply(m.GetPrim())
            binding.Bind(UsdShade.Material(stage.GetPrimAtPath(material_path)),
                         bindingStrength=UsdShade.Tokens.weakerThanDescendants)
    elif color is not None:
        m.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    return m


# ---------------------------------------------------------------------------------------
# HAND-BUILT jig collider (exact axis-aligned boxes, convex-hull each -- the openbox
# approach). Why: SDF jaw contacts fling the torque-controlled arm; convex decomposition
# seats the skirt lopsided. Boxes give stable convex contacts AND an exact, symmetric seat.
#
# Structure measured from jig.stl (object frame, origin at bbox center; heights below are
# mm above the jig BOTTOM plane, half-extents in meters):
# * long walls y in +-[54.5, 64.5], x in [-72, 72]: solid 0-24 at the sides, 0-9 in the
#   middle |x|<25 (the wall cutout -> matches the graspable-region map used by the sampler);
# * end walls x in +-[72, 82], full y: solid 0-24 with two vertical cone through-holes at
#   y = +-32 (pillar sockets). Each cone is emulated as a TWO-STAGE rectangular hole:
#   an 11 mm mouth for z 0-5 (funnel capture) narrowing to a 5x5 mm pocket for z 5-24 --
#   the r=3.6 mm pillar (top at enclosure z 22.6) enters the mouth and JAMS at the step,
#   seating the jig bottom at 22.6 - 5.0 = 17.6 mm above the enclosure bottom, which is the
#   SDF-measured truth (jig root rel z 18.3 mm).
# Format: (cx, cy, cz, hx, hy, hz) in mm, z measured from the jig bottom.
_JIG_BOXES_MM = []
_STEP = 5.0     # mouth height = pillar engagement depth (sets the seat)
_MOUTH = 5.5    # half-width of the lower mouth (11 mm)
_POCKET = 2.5   # half-width of the upper pocket (5 mm)
# The enclosure has 4 corner POSTS at (+-71, +-54), r~3.4, tops ~1 mm below the pillar
# tops -- the real jig clears them via hollow window corners. The box ring must carve the
# same clearances or the ring lands on the posts 4.3 mm proud of the true seat (measured).
# Wall thicknesses MEASURED (mesh probes, confirmed by caliper on the real jig): both walls
# are TIERED with a step at z ~10.5 mm:
#   long walls: 14 mm thick below the step (inner edge y 50.75), 12 mm above (inner 52.5)
#   end walls: 11.35 mm below (inner edge x 70.65), 9.5 mm above (inner 72.5)
for sy in (+1, -1):  # long walls
    # lower tier z 0-9 (14 mm thick): outer sub-band full length; inner sub-band shortened
    # to x +-66 so the (+-71, +-54) posts pass under at the corners
    _JIG_BOXES_MM.append((0.0, sy * 61.75, 4.5, 72.0, 2.75, 4.5))       # outer y 59..64.5
    _JIG_BOXES_MM.append((0.0, sy * 54.875, 4.5, 66.0, 4.125, 4.5))     # inner y 50.75..59
    for sxa, sxb in ((-72.0, -25.0), (25.0, 72.0)):                     # upper tier z 9-24 (12 mm)
        _JIG_BOXES_MM.append(((sxa + sxb) / 2, sy * 58.5, 16.5, (sxb - sxa) / 2, 6.0, 7.5))
for sx in (+1, -1):  # end walls with two-stage pillar sockets at +-32 and post-cleared corners
    xlo = sx * 76.325   # lower/mid tier center (x 70.65..82, 11.35 mm thick)
    xup = sx * 77.25    # upper tier center (x 72.5..82, 9.5 mm thick)
    # mouth tier z 0-STEP: mouths 11 mm at +-32; corner segments y +-[50, 64.5] narrowed to
    # x [75, 82] so the (+-71, +-54) posts pass under
    for ya, yb in ((-50.0, -32 - _MOUTH), (-32 + _MOUTH, 32 - _MOUTH), (32 + _MOUTH, 50.0)):
        _JIG_BOXES_MM.append((xlo, (ya + yb) / 2, _STEP / 2, 5.675, (yb - ya) / 2, _STEP / 2))
    for sy in (+1, -1):
        _JIG_BOXES_MM.append((sx * 78.5, sy * 57.25, _STEP / 2, 3.5, 7.25, _STEP / 2))
    # mid tier z STEP-10.5 (still 11.35 mm) with 5 mm pockets at +-32
    for ya, yb in ((-64.5, -32 - _POCKET), (-32 + _POCKET, 32 - _POCKET), (32 + _POCKET, 64.5)):
        _JIG_BOXES_MM.append((xlo, (ya + yb) / 2, (10.5 + _STEP) / 2, 5.675, (yb - ya) / 2, (10.5 - _STEP) / 2))
    # upper tier z 10.5-24 (9.5 mm) with 5 mm pockets at +-32
    for ya, yb in ((-64.5, -32 - _POCKET), (-32 + _POCKET, 32 - _POCKET), (32 + _POCKET, 64.5)):
        _JIG_BOXES_MM.append((xup, (ya + yb) / 2, (24 + 10.5) / 2, 4.75, (yb - ya) / 2, (24 - 10.5) / 2))
    # x-cheeks: close the pockets in x (5x5 mm, centered on the pillar at x=+-76 like the
    # real cone) -> 2-axis registration. Mid tier reaches in to 70.65, upper to 72.5.
    for sy in (+1, -1):
        _JIG_BOXES_MM.append((sx * 72.075, sy * 32.0, (10.5 + _STEP) / 2, 1.425, _POCKET, (10.5 - _STEP) / 2))
        _JIG_BOXES_MM.append((sx * 80.25, sy * 32.0, (10.5 + _STEP) / 2, 1.75, _POCKET, (10.5 - _STEP) / 2))
        _JIG_BOXES_MM.append((sx * 73.0, sy * 32.0, (24 + 10.5) / 2, 0.5, _POCKET, (24 - 10.5) / 2))
        _JIG_BOXES_MM.append((sx * 80.25, sy * 32.0, (24 + 10.5) / 2, 1.75, _POCKET, (24 - 10.5) / 2))


def _add_jig_box_collider(stage, root_name, material_path):
    for i, (cx, cy, cz, hx, hy, hz) in enumerate(_JIG_BOXES_MM):
        add_box(stage, f"/{root_name}/collisions/box_{i:02d}",
                center=(cx / 1000.0, cy / 1000.0, (cz - 12.0) / 1000.0),
                half_extents=(hx / 1000.0, hy / 1000.0, hz / 1000.0),
                collision=True, material_path=material_path)


def build(stl_path, usd_path, root_name, *, y_up=False, color, metadata_extra=None,
          mate="bottom", approximation="convexDecomposition"):
    mesh = trimesh.load(stl_path, force="mesh")
    mesh.apply_scale(0.001)  # mm -> m
    if y_up:  # rotate STL Y-up -> Z-up (+90 deg about X: y->z)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0]))
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-center)  # origin at bbox center
    hz = mesh.extents[2] / 2.0

    if os.path.exists(usd_path):
        os.remove(usd_path)
    stage, _, mat = create_stage(usd_path, root_name=root_name)
    add_trimesh(stage, f"/{root_name}/visuals/mesh", mesh, collision=False, color=color)
    if approximation == "handBoxes":
        _add_jig_box_collider(stage, root_name, mat)
    else:
        add_trimesh(stage, f"/{root_name}/collisions/mesh", mesh, collision=True, material_path=mat,
                    approximation=approximation)
    stage.GetRootLayer().Save()

    mate_z = -hz if mate == "bottom" else hz
    metadata = {
        "assembled_offset": {"pos": [0.0, 0.0, round(float(mate_z), 6)], "quat": [1.0, 0.0, 0.0, 0.0]},
        "bottom_offset": {"pos": [0.0, 0.0, round(float(-hz), 6)], "quat": [1.0, 0.0, 0.0, 0.0]},
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    meta_path = write_metadata(usd_path, metadata)
    print(f"Wrote {usd_path}")
    print(f"Wrote {meta_path}")
    print(f"  extents (m): {np.round(mesh.extents, 4)}   bottom_offset.z={-hz:.6f}  assembled_offset.z={mate_z:.6f}")
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Build jig + bottom-enclosure USDs from STLs.")
    parser.add_argument("--enclosure-seat-z", type=float, default=0.0063,
                        help="Enclosure-frame z of the seated jig's BOTTOM-CENTER (the mating "
                             "point). Default 0.0063 is SIM-MEASURED with the final collision "
                             "model (jig HQ convex decomposition + enclosure exact trimesh): the "
                             "perfect mate settles at jig-root rel z 19.6 mm (yaw 0) / 22.6 mm "
                             "(yaw 180; 3 mm decomposition asymmetry -- the part itself is "
                             "180-symmetric, SDF seats both at 18.3) -> mating point centered "
                             "between them, both orientations within 1.5 mm. SDF was rejected: "
                             "its jaw contacts fling the torque-controlled arm. Re-measure with "
                             "visualize_perfect_mate after any collision-model change.")
    parser.add_argument("--show-colliders", action="store_true",
                        help="Debug build: make the collision prims VISIBLE (tinted red) so the "
                             "collider can be inspected in the GUI. Re-run WITHOUT this flag for "
                             "the final assets.")
    args = parser.parse_args()

    build(
        f"{_LOCAL}/Jig/jig.stl", f"{_LOCAL}/Jig/jig.usd", "Jig",
        y_up=False, color=(0.10, 0.35, 0.13), mate="bottom",  # real jig: dark (goblin) green  # insertive: mating point = bottom-center
        approximation="handBoxes",  # hand-built box collider: stable convex contacts + exact symmetric pillar-socket seat
    )
    build(
        f"{_LOCAL}/BottomEnclosure/bottom_enclosure.stl", f"{_LOCAL}/BottomEnclosure/bottom_enclosure.usd",
        "BottomEnclosure", y_up=True, color=(0.02, 0.02, 0.022), mate="top",  # real enclosure: black
        metadata_extra={"success_thresholds": {
            # authors' position/orientation values; yaw gate is jig-specific (the pillar
            # pattern is 2-fold symmetric: yaw 0/180 valid, 90 wedges -> false success without it)
            "position": 0.005, "orientation": 0.025, "yaw": 0.35, "yaw_symmetry": 2}},
        approximation="none",  # kinematic in every task -> exact triangle mesh (real pillars)
    )
    # Set the enclosure's mating point to the (sim-measured) seated height of the jig's
    # bottom-center -- success then means "jig seated on the pillars" within the thresholds.
    import yaml
    p = f"{_LOCAL}/BottomEnclosure/metadata.yaml"
    with open(p) as f:
        meta = yaml.safe_load(f)
    meta["assembled_offset"]["pos"][2] = round(args.enclosure_seat_z, 6)
    with open(p, "w") as f:
        yaml.safe_dump(meta, f, default_flow_style=None, sort_keys=False)
    print(f"  enclosure assembled_offset.z (seated mating point) = {meta['assembled_offset']['pos'][2]}")

    if args.show_colliders:
        from pxr import UsdGeom as _UsdGeom
        for usd in (f"{_LOCAL}/Jig/jig.usd", f"{_LOCAL}/BottomEnclosure/bottom_enclosure.usd"):
            stage = Usd.Stage.Open(usd)
            for prim in stage.Traverse():
                if "/collisions/" in str(prim.GetPath()) and prim.IsA(_UsdGeom.Mesh):
                    g = _UsdGeom.Mesh(prim)
                    g.CreateVisibilityAttr("inherited")
                    g.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.9, 0.1, 0.1)]))
            stage.GetRootLayer().Save()
        print("  [show-colliders] collision prims made VISIBLE (red) -- debug build, do not commit")


if __name__ == "__main__":
    main()
