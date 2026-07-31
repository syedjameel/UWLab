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


# ---------------------------------------------------------------------------------------
# V2 ONLY -- INTERIOR BLOCKER (training scaffold, not part of the real jig).
#
# Why: v1's task_0 (reach + grasp from scratch) learned a ONE-SIDED RIM PINCH -- one jaw
# descends INTO the open middle of the frame and squeezes a single ~13 mm long wall against
# the outer jaw. It succeeds (~0.91) but is less stable than the two-sided STRADDLE the grasp
# sampler produces (~0.98), and because deployment always starts jig-on-mat/gripper-open (the
# C1 scenario), task_0's grasp is the one that ships AND the one the RGB student imitates.
# Filling the interior makes the one-sided pinch PHYSICALLY IMPOSSIBLE from iteration 1,
# which is more reliable than reward-shaping (RL is already in the one-sided basin).
#
# Sim2real: the straddle grips the OUTER walls only, so the same motion works on the real
# (open-middle) jig. This collider is a SIM TRAINING SCAFFOLD; its absence at deploy is fine.
#
# Geometry (measured from jig.stl by containment cross-sections, jig frame, origin at bbox
# center, z -12..+12): the interior void is TIERED, matching the tiered walls --
#   z -12..-2.5 (lower, 9.5 mm): x +-70.5, y +-50.5
#   z -2.5..+12 (upper, 14.5 mm): x +-72.5, y +-52.5
# The UPPER tier mirrors the void exactly: it is flush with the upper inner wall faces and
# therefore caps the ENTIRE interior mouth -- anything descending from above is stopped by
# it regardless of what sits below. The LOWER tier is deliberately INSET to x +-68 / y +-48
# (it only has to plug the remaining volume, not block an approach) which buys seat
# clearance cheaply; the 2.65/2.75 mm slot this opens against the lower inner walls is
# unusable because the jaw pad is 28 mm thick along the closing axis (measured from
# tip.stl + the URDF joint frames). Neither tier protrudes past the outer walls.
#
# Seat clearance (jig bottom sits 17.6 mm above the enclosure bottom when seated; measured
# box-vs-box, 0 hard overlaps): the corner PILLARS (x +-76, top 22.55) clear the lower tier
# by 4.4 mm in x; the POST TOWERS (x 69..73.5, y 51.5..56, top 21.6) clear it by 3.5 mm in
# y; the long-wall RIDGES (y 48.5..50.5, top 13.6) fall outside the lower-tier footprint
# entirely and 4.0 mm below its underside. The upper tier starts 5.5 mm above the tallest
# enclosure feature. Verified with visualize_perfect_mate.py -- re-verify after any change.
#
# MASS: the blocker is 283% of the jig's existing collider volume, and the jig authors NO
# MassAPI on its root (PhysX auto-computes mass from collider volume -- see
# omnireset_asset_utils.create_stage). Left as-is it would ~4x the jig mass and silently
# change every contact/dynamics result. Each blocker box therefore carries its OWN MassAPI
# with a near-zero density so it contributes no mass or inertia. Density is 1e-9 and NOT 0
# because UsdPhysics treats a density of 0 as "unauthored" (ignored -> falls back to the
# default), which would reintroduce the full mass. MassAPI is applied to the CHILD collider
# prims only, never the root, so the root stays MassAPI-free and the spawn config's
# mass=0.001 keeps harmlessly failing to apply exactly as it does for v1.
# Format matches _JIG_BOXES_MM: (cx, cy, cz, hx, hy, hz) in mm, z from the jig bottom.
#
# v2b (--v2b-jig) DROPS THE LOWER TIER. Measured cost of keeping it, v2 vs v1 Stage-1:
#   * total_fps 6951 vs 10807 (1.56x slower) -- its bottom face is COPLANAR with the jig's
#     underside, so a resting jig presents a solid ~136x96 mm contact patch to the mat where
#     v1 had only a thin wall rim;
#   * Episode_Reward/abnormal_robot -0.0013 vs -0.0004 (3.2x more penalty);
#   * it is the only part of the blocker that can foul the enclosure: at x offsets >=5 mm it
#     overlaps the corner pillars, so the jig cannot descend (upper tier alone: 0 overlaps at
#     every offset tested, both axes).
# It contributes NOTHING to blocking -- the upper tier alone caps the whole interior mouth,
# and below z 9.5 the jig's own walls are solid all round so no jaw can enter from the side.
_JIG_INTERIOR_BOXES_MM = [
    (0.0, 0.0, 9.5 / 2, 68.0, 48.0, 9.5 / 2),                    # lower tier z 0..9.5 (inset)
    (0.0, 0.0, (24 + 9.5) / 2, 72.5, 52.5, (24 - 9.5) / 2),      # upper tier z 9.5..24 (flush)
]
_JIG_INTERIOR_BOXES_V2B_MM = _JIG_INTERIOR_BOXES_MM[1:]          # upper tier only

# ---------------------------------------------------------------------------------------
# v2c -- SHAPED blocker. v2/v2b were WRONG and this is why.
#
# The long walls are NOT uniformly 24 mm tall. Measured from jig.stl (wall-top vs x, probed
# across the wall band):   |x| <= 28  ->  8.5 mm (the notch, bottom strip only)
#                          |x| >= 32  -> 23.5 mm (full height)
# A blocker authored as one full-height box therefore stands in OPEN AIR right across the
# middle band, presenting two clean vertical faces 105 mm apart -- inside the 136.8 mm jaw
# opening. Measured with test_blocker_exploit.py: the v2 jig is HELD at a jaw gap of 107.7 mm
# at x=0 (that is the blocker's width, not the jig's 129 mm), while the v1 jig correctly falls
# 1.5 m. That fake grasp is also EASIER than the true straddle (~15 mm clearance per side vs
# 3.9 mm), so RL prefers it -- the v2 Stage-1 run learned it instead of the straddle.
#
# So the blocker must never exceed the LOCAL wall height:
#   low  slab z 0..8.5 across the whole interior  (walls exist all round down here, so a jaw
#        reaching in to pinch the bottom strip one-sided is blocked);
#   high slabs z 8.5..23.5 ONLY for |x| >= 30      (where the long wall is full height);
#   nothing above 8.5 in the notch band            (the real jig is empty there).
# y/x extents are flush with the hand-built WALL boxes (inner faces 50.75 low / 52.5 high,
# 70.65 / 72.5), so there is no slot to enter and no protrusion past the collider's material.
#
# The high slabs still expose two step faces at x = +-30. Those are made UNGRASPABLE by the
# near-zero-friction material (see _add_interior_blocker): the jaws can still be BLOCKED by
# the blocker, but cannot HOLD it, so any residual exposed face affords no grasp. Legitimate
# straddles grip the outer walls at +-64.5 and never touch the blocker, so they are unaffected.
_NOTCH_X = 30.0          # blocker stays below the notch wall-top inboard of this
_WALL_TOP = 23.5         # measured full-wall height
_NOTCH_TOP = 8.5         # measured notch (bottom-strip) height
#
# The LOW slab is INSET to x +-68 / y +-48 rather than run flush to the wall faces. Flush
# (+-70.5 / +-50.75) leaves only 0.75 mm to the enclosure's post towers and starts FOULING them
# at just 2 mm of lateral offset -- inside the ~8 mm capture basin, which would wreck seating
# (measured box-vs-box; the inset restores ~3.5 mm and pushes first fouling past 6 mm). The
# inset is free: below 8.5 mm the jig's own walls enclose the slab on every side, so insetting
# opens a 2.7 mm SLOT rather than exposing material, and the jaw pad is 28 mm thick along the
# closing axis so it cannot enter. (This is a different argument from the one that failed for
# v2: there the problem was material standing in OPEN AIR, not a slot.)
_JIG_INTERIOR_BOXES_V2C_MM = [
    (0.0, 0.0, _NOTCH_TOP / 2, 68.0, 48.0, _NOTCH_TOP / 2),      # low slab, whole interior, inset
]
for _sx in (+1, -1):     # high slabs, only where the long wall is full height
    _x0, _x1 = _sx * _NOTCH_X, _sx * 72.5
    _JIG_INTERIOR_BOXES_V2C_MM.append(
        ((_x0 + _x1) / 2, 0.0, (_WALL_TOP + _NOTCH_TOP) / 2,
         abs(_x1 - _x0) / 2, 52.5, (_WALL_TOP - _NOTCH_TOP) / 2)
    )
_INTERIOR_DENSITY = 1e-9  # kg/m^3; non-zero so UsdPhysics honours it, small enough to vanish


def _add_slippery_material(stage, root_name):
    """A near-frictionless PhysicsMaterial for the interior blocker.

    The blocker exists to BLOCK a jaw, never to be held by one. Any face of it that a jaw can
    reach is a grasp the real jig does not afford (v2 was picked up by its blocker at a jaw gap
    of 107.7 mm -- see _JIG_INTERIOR_BOXES_V2C_MM). Shaping removes most such faces; this
    removes the rest by making them impossible to grip: with no friction the jig simply squirts
    out from between flat parallel jaws, while normal forces still stop a jaw entering.

    ``frictionCombineMode = min`` is essential. PhysX's default is AVERAGE, which would combine
    friction 0 with the jaw's ~1.0 into 0.5 -- still perfectly grippable. It is a PhysX
    extension (UsdPhysics.MaterialAPI exposes only static/dynamic friction, restitution and
    density), so it is authored raw, the same way the SDF attributes are above.
    """
    path = f"/{root_name}/BlockerMaterial"
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(0.0)
    api.CreateDynamicFrictionAttr(0.0)
    mat.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    mat.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode",
                                  Sdf.ValueTypeNames.Token).Set("min")
    return path


def _add_interior_blocker(stage, root_name, material_path, bottom_mm, boxes=None, slippery=False):
    """Author the interior blocker as massless colliders under ``collisions``."""
    if slippery:
        material_path = _add_slippery_material(stage, root_name)
    for i, (cx, cy, cz, hx, hy, hz) in enumerate(_JIG_INTERIOR_BOXES_MM if boxes is None else boxes):
        mesh = add_box(stage, f"/{root_name}/collisions/interior_{i:02d}",
                       center=(cx / 1000.0, cy / 1000.0, (cz - bottom_mm) / 1000.0),
                       half_extents=(hx / 1000.0, hy / 1000.0, hz / 1000.0),
                       collision=True, material_path=material_path)
        mass_api = UsdPhysics.MassAPI.Apply(mesh.GetPrim())
        mass_api.CreateDensityAttr(_INTERIOR_DENSITY)


# HAND-BUILT enclosure collider. The exact-trimesh enclosure explodes the PhysX GPU
# collision stack (35 jig boxes x 8878 triangles x thousands of in-contact envs: >2.3 GB
# demand at 4096-env C4 resets -- above the int32 setting ceiling; training would drop
# contacts silently). Structures measured from the mesh (heights above the enclosure
# BOTTOM; only the pillars and post towers rise above the 17.6 mm seat plane, both of
# which the jig's carved clearances/sockets accommodate -- verified vs the SDF seat):
#   interior plateau top 9.4 | end shelves (|x| 69-78.3) top 9.4 | long wall ridges
#   (y ~+-49.5, 13.6 tall, x +-66) | 4 pillars (+-76,+-32) r3.6 top 22.55 | 4 post
#   towers (x 69-73.5, y 51.5-56) top 21.6. Small interior cutouts/fins (<15.2 mm,
#   below the seat plane) are approximated away.
_ENC_BOXES_MM = [
    (0.0, 0.0, 4.7, 70.0, 48.0, 4.7),          # interior plateau slab z 0-9.4
]
for sx in (+1, -1):
    _ENC_BOXES_MM.append((sx * 73.65, 0.0, 4.7, 4.65, 60.8, 4.7))   # end shelf x 69..78.3, z 0-9.4
    for sy in (+1, -1):
        _ENC_BOXES_MM.append((sx * 76.0, sy * 32.0, 15.975, 3.6, 3.6, 6.575))    # pillar z 9.4-22.55
        _ENC_BOXES_MM.append((sx * 71.25, sy * 53.75, 15.5, 2.25, 2.25, 6.1))    # post tower z 9.4-21.6
for sy in (+1, -1):
    _ENC_BOXES_MM.append((0.0, sy * 49.5, 6.8, 66.0, 1.0, 6.8))     # long wall ridge z 0-13.6

_BOX_TABLES = {"Jig": (_JIG_BOXES_MM, 12.0), "BottomEnclosure": (_ENC_BOXES_MM, 11.3)}


def _add_hand_box_collider(stage, root_name, material_path):
    boxes, bottom_mm = _BOX_TABLES[root_name]
    for i, (cx, cy, cz, hx, hy, hz) in enumerate(boxes):
        add_box(stage, f"/{root_name}/collisions/box_{i:02d}",
                center=(cx / 1000.0, cy / 1000.0, (cz - bottom_mm) / 1000.0),
                half_extents=(hx / 1000.0, hy / 1000.0, hz / 1000.0),
                collision=True, material_path=material_path)


def build(stl_path, usd_path, root_name, *, y_up=False, color, metadata_extra=None,
          mate="bottom", approximation="convexDecomposition", interior_blocker=False,
          slippery_blocker=False):
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
        _add_hand_box_collider(stage, root_name, mat)
        if interior_blocker:
            _add_interior_blocker(stage, root_name, mat, _BOX_TABLES[root_name][1],
                                  boxes=interior_blocker if isinstance(interior_blocker, list) else None,
                                  slippery=slippery_blocker)
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


def _reveal_colliders(usd_paths):
    """Debug: make every collision prim visible (red) so colliders can be eyeballed in the GUI."""
    from pxr import UsdGeom as _UsdGeom
    for usd in usd_paths:
        stage = Usd.Stage.Open(usd)
        for prim in stage.Traverse():
            if "/collisions/" in str(prim.GetPath()) and prim.IsA(_UsdGeom.Mesh):
                g = _UsdGeom.Mesh(prim)
                g.CreateVisibilityAttr("inherited")
                g.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.9, 0.1, 0.1)]))
        stage.GetRootLayer().Save()
    print("  [show-colliders] collision prims made VISIBLE (red) -- debug build, do not commit")


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
    parser.add_argument("--v2c-jig", action="store_true",
                        help="Build ONLY the v2c jig (JigV2c/jig_v2c.usd): the SHAPED, "
                             "near-frictionless blocker. Follows the measured local wall height "
                             "(8.5 mm in the notch, 23.5 mm elsewhere) so it never stands in open "
                             "air, and cannot be gripped. This is the one to use -- v2/v2b are "
                             "both pickable BY THE BLOCKER (test_blocker_exploit.py).")
    parser.add_argument("--v2b-jig", action="store_true",
                        help="Build ONLY the v2b jig (JigV2b/jig_v2b.usd): the interior blocker "
                             "with the LOWER TIER DROPPED. Same jaw blocking (the upper tier caps "
                             "the whole mouth) but no mat contact patch, no pillar fouling and "
                             "~1.5x the throughput. See _JIG_INTERIOR_BOXES_V2B_MM.")
    parser.add_argument("--v2-jig", action="store_true",
                        help="Build ONLY the v2 jig (JigV2/jig_v2.usd) -- same geometry as v1 plus "
                             "the massless INTERIOR BLOCKER that forbids the one-sided rim pinch. "
                             "Leaves Jig/jig.usd and the enclosure untouched (v1 keeps training / "
                             "collecting against them); the v2 task pairs jigv2 with the SAME "
                             "bottomenclosure. See _JIG_INTERIOR_BOXES_MM.")
    args = parser.parse_args()

    if args.v2_jig or args.v2b_jig or args.v2c_jig:
        out = (f"{_LOCAL}/JigV2c/jig_v2c.usd" if args.v2c_jig else
               f"{_LOCAL}/JigV2b/jig_v2b.usd" if args.v2b_jig else
               f"{_LOCAL}/JigV2/jig_v2.usd")
        build(
            f"{_LOCAL}/Jig/jig.stl", out, "Jig",
            y_up=False, color=(0.10, 0.35, 0.13), mate="bottom",
            approximation="handBoxes",
            interior_blocker=(_JIG_INTERIOR_BOXES_V2C_MM if args.v2c_jig else
                              _JIG_INTERIOR_BOXES_V2B_MM if args.v2b_jig else True),
            slippery_blocker=args.v2c_jig,
        )
        if args.show_colliders:
            _reveal_colliders([out])
        return

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
        approximation="handBoxes",  # measured boxes: trimesh blew the GPU contact stack (see _ENC_BOXES_MM)
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
        _reveal_colliders([f"{_LOCAL}/Jig/jig.usd", f"{_LOCAL}/BottomEnclosure/bottom_enclosure.usd"])


if __name__ == "__main__":
    main()
