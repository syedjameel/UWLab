# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedurally generate the custom lab-table USDs from measured dimensions.

Replaces the authors' cloud ``pat_vention.usd`` (table) + ``ur5plate.usd`` (support) with
assets built from ``table_dims.yaml`` -- structurally IDENTICAL to the decoded author
assets (same prim anatomy, physics APIs, MDL OmniPBR constant-color materials, and
hand-authored invisible Cube colliders; no textures, no UVs), only the geometry differs:

* ``custom_lab_table.usd`` -- root Xform ``/custom_lab_table`` [RigidBody+Mass APIs,
  kinematic, mass 100 -- as the authors']:
    - ``visuals/mat_black``   700x700x4 mm mat over the robot half, with the circular
                              base cutout (ring-triangulated; the RGB DR retextures it)
    - ``visuals/mat_green``   700x700x4 mm workspace mat (DR-retextured too)
    - ``visuals/table_frame`` structural top + body + 4 corner pillars in ONE mesh
                              (authors' single ``vention_metal`` pattern; not randomized)
    - ``visuals/Looks/*``     constant-color OmniPBR (authors' exact constants)
    - ``collisions/*``        invisible unit Cubes scaled to slabs/body/pillars
* ``custom_mount_plate.usd`` -- the ``ur5_metal_support`` replacement: a flush proxy disk
  filling the mat cutout (no physical plate exists on the real rig). Its ROOT is authored
  at the WORK-SURFACE level (authors' convention: plate root z == mat-top height), so the
  scene places the root at the placement datum used by the object-reset events.

Asset frame == ROBOT BASE frame (origin at the base flange, z=0 at the flange plane,
+x toward the workspace). Pure pxr + numpy, NO Isaac app required:

    conda activate leisaac && python scripts_v2/tools/conversions/make_custom_table_usd.py

Verify with the pxr probe printed at the end (or scripts_v2/tools/conversions/view_usd.py).
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import yaml
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ASSET_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..",
    "source", "uwlab_assets", "uwlab_assets", "local", "Props", "Mounts", "CustomLabTable",
)


# ---------------------------------------------------------------------------------------
# mesh builders (numpy -> (points, faceVertexCounts, faceVertexIndices))
# ---------------------------------------------------------------------------------------

def box_mesh(x0, x1, y0, y1, z0, z1):
    """Axis-aligned box as 8 points / 12 triangles."""
    pts = np.array([
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ])
    quads = [
        (3, 2, 1, 0),  # bottom (-z)
        (4, 5, 6, 7),  # top (+z)
        (0, 1, 5, 4),  # -y
        (2, 3, 7, 6),  # +y
        (1, 2, 6, 5),  # +x
        (3, 0, 4, 7),  # -x
    ]
    idx = []
    for a, b, c, d in quads:
        idx += [a, b, c, a, c, d]
    return pts, [3] * (len(idx) // 3), idx


def plate_with_hole_mesh(x0, x1, y0, y1, z0, z1, hole_cx, hole_cy, hole_r, segments=64):
    """Rectangular plate with a circular through-hole (ring triangulation).

    For each circle angle the matching outer point is the radial projection onto the
    rectangle boundary -> a clean quad strip between the circle ring and the boundary
    ring, top + bottom faces, outer wall, and the hole wall.
    """
    th = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    # circle ring (hole edge)
    circ = np.stack([hole_cx + hole_r * c, hole_cy + hole_r * s], axis=1)
    # radial projection of each angle onto the rectangle boundary (from the hole center)
    tx = np.where(c > 0, (x1 - hole_cx) / np.where(c == 0, np.inf, c),
                  np.where(c < 0, (x0 - hole_cx) / np.where(c == 0, np.inf, c), np.inf))
    ty = np.where(s > 0, (y1 - hole_cy) / np.where(s == 0, np.inf, s),
                  np.where(s < 0, (y0 - hole_cy) / np.where(s == 0, np.inf, s), np.inf))
    t = np.minimum(tx, ty)
    outer = np.stack([hole_cx + t * c, hole_cy + t * s], axis=1)

    n = segments
    pts = []
    pts += [(p[0], p[1], z1) for p in circ]    # 0..n-1    circle top
    pts += [(p[0], p[1], z1) for p in outer]   # n..2n-1   outer top
    pts += [(p[0], p[1], z0) for p in circ]    # 2n..3n-1  circle bottom
    pts += [(p[0], p[1], z0) for p in outer]   # 3n..4n-1  outer bottom

    idx = []

    def quad(a, b, cc, d):  # two CCW triangles
        idx.extend([a, b, cc, a, cc, d])

    for i in range(n):
        j = (i + 1) % n
        quad(i, j, n + j, n + i)                            # top ring (normal +z)
        quad(2 * n + j, 2 * n + i, 3 * n + i, 3 * n + j)    # bottom ring (normal -z)
        quad(n + i, n + j, 3 * n + j, 3 * n + i)            # outer wall
        quad(j, i, 2 * n + i, 2 * n + j)                    # hole wall
    return np.array(pts), [3] * (len(idx) // 3), idx


def disk_mesh(cx, cy, r, z0, z1, segments=64):
    """Closed cylinder (disk with thickness)."""
    th = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    ring = np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], axis=1)
    n = segments
    pts = [(cx, cy, z1)] + [(p[0], p[1], z1) for p in ring]          # 0 center-top, 1..n top ring
    pts += [(cx, cy, z0)] + [(p[0], p[1], z0) for p in ring]         # n+1 center-bot, n+2..2n+1 bottom
    idx = []
    for i in range(n):
        j = (i + 1) % n
        idx += [0, 1 + i, 1 + j]                                     # top fan (+z)
        idx += [n + 1, n + 2 + j, n + 2 + i]                         # bottom fan (-z)
        a, b = 1 + i, 1 + j
        c2, d2 = n + 2 + i, n + 2 + j
        idx += [a, c2, d2, a, d2, b]                                 # side wall
    return np.array(pts), [3] * (len(idx) // 3), idx


def merge_meshes(meshes):
    """Merge (points, counts, indices) tuples into one mesh."""
    pts_all, counts_all, idx_all = [], [], []
    offset = 0
    for pts, counts, idx in meshes:
        pts_all.append(pts)
        counts_all += counts
        idx_all += [i + offset for i in idx]
        offset += len(pts)
    return np.concatenate(pts_all), counts_all, idx_all


# ---------------------------------------------------------------------------------------
# USD authoring (mirrors the decoded author patterns)
# ---------------------------------------------------------------------------------------

def author_mesh(stage, path, mesh, material_path=None):
    pts, counts, idx = mesh
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateDoubleSidedAttr(True)
    m.CreateSubdivisionSchemeAttr("none")
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    m.CreateExtentAttr([Gf.Vec3f(*map(float, mn)), Gf.Vec3f(*map(float, mx))])
    if material_path is not None:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(
            UsdShade.Material(stage.GetPrimAtPath(material_path))
        )
    return m


def author_omnipbr(stage, path, diffuse, metallic, roughness):
    """Constant-color MDL OmniPBR material -- the authors' exact shader layout."""
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(metallic))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(roughness))
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
    mat.CreateDisplacementOutput("mdl").ConnectToSource(out)
    mat.CreateVolumeOutput("mdl").ConnectToSource(out)
    return mat


def author_collision_cube(stage, path, center, size):
    """Invisible unit Cube scaled to a box collider (authors' exact pattern)."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    xf.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    xf.AddScaleOp().Set(Gf.Vec3d(*map(float, size)))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    cube.GetPrim().GetAttribute("physics:collisionEnabled").Set(True)
    UsdGeom.Imageable(cube).MakeInvisible()
    return cube


def new_stage(path, default_prim_name, rigid_body=True, mass=None, kinematic=True):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetMetadata("kilogramsPerUnit", 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{default_prim_name}")
    prim = root.GetPrim()
    stage.SetDefaultPrim(prim)
    if rigid_body:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        prim.GetAttribute("physics:kinematicEnabled").Set(bool(kinematic))
        if mass is not None:
            UsdPhysics.MassAPI.Apply(prim)
            prim.GetAttribute("physics:mass").Set(float(mass))
    return stage, root


# ---------------------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------------------

def make_table(dims, out_path):
    t = dims["table"]
    top, body, mats, pil = t["top"], t["body"], t["mats"], t["pillars"]
    x0, x1, yh, thick = top["x_min"], top["x_max"], top["y_half"], top["thickness"]
    mat_t, split = mats["thickness"], mats["split_x"]
    (hcx, hcy), hr = mats["hole"]["center"], mats["hole"]["diameter"] / 2.0
    pc, ph, pin = pil["cross_section"], pil["height"], pil["inset"]
    m = dims["materials"]

    stage, root = new_stage(out_path, "custom_lab_table", mass=100.0, kinematic=True)
    vis = UsdGeom.Xform.Define(stage, "/custom_lab_table/visuals")
    UsdGeom.Scope.Define(stage, "/custom_lab_table/visuals/Looks")
    col = UsdGeom.Xform.Define(stage, "/custom_lab_table/collisions")

    for name in ("mat_black", "mat_green", "table_frame"):
        author_omnipbr(stage, f"/custom_lab_table/visuals/Looks/{name}",
                       m[name]["diffuse"], m[name]["metallic"], m[name]["roughness"])

    # --- visuals ---
    # black mat (rear half) with the circular base cutout; top = work surface (+mat_t)
    author_mesh(stage, "/custom_lab_table/visuals/mat_black",
                plate_with_hole_mesh(x0, split, -yh, yh, 0.0, mat_t, hcx, hcy, hr),
                "/custom_lab_table/visuals/Looks/mat_black")
    # green mat (front/workspace half)
    author_mesh(stage, "/custom_lab_table/visuals/mat_green",
                box_mesh(split, x1, -yh, yh, 0.0, mat_t),
                "/custom_lab_table/visuals/Looks/mat_green")
    # frame: structural top slab + body + 4 corner pillars, merged into ONE mesh
    inset = body["inset"]
    pxs = (x0 + pin, x1 - pin)
    pys = (-yh + pin, yh - pin)
    frame_parts = [
        box_mesh(x0, x1, -yh, yh, -thick, 0.0),                                   # top slab
        box_mesh(x0 + inset, x1 - inset, -yh + inset, yh - inset,
                 body["z_bottom"], -thick),                                        # body
    ]
    for px in pxs:
        for py in pys:
            frame_parts.append(box_mesh(px - pc / 2, px + pc / 2, py - pc / 2, py + pc / 2,
                                        mat_t, mat_t + ph))                        # pillars
    author_mesh(stage, "/custom_lab_table/visuals/table_frame", merge_meshes(frame_parts),
                "/custom_lab_table/visuals/Looks/table_frame")

    # --- collisions (invisible Cubes; authors' pattern) ---
    cx, cy = (x0 + x1) / 2.0, 0.0
    author_collision_cube(stage, "/custom_lab_table/collisions/mats",
                          (cx, cy, mat_t / 2.0), (x1 - x0, 2 * yh, mat_t))
    author_collision_cube(stage, "/custom_lab_table/collisions/top_slab",
                          (cx, cy, -thick / 2.0), (x1 - x0, 2 * yh, thick))
    author_collision_cube(stage, "/custom_lab_table/collisions/body",
                          (cx, cy, (body["z_bottom"] - thick) / 2.0),
                          (x1 - x0 - 2 * inset, 2 * (yh - inset), abs(body["z_bottom"]) - thick))
    for k, px in enumerate(pxs):
        for j, py in enumerate(pys):
            author_collision_cube(stage, f"/custom_lab_table/collisions/pillar_{k}{j}",
                                  (px, py, mat_t + ph / 2.0), (pc, pc, ph))
    stage.Save()
    return out_path


def make_mount_plate(dims, out_path):
    p = dims["mount_plate"]
    m = dims["materials"]["mount_plate"]
    r, th = p["diameter"] / 2.0, p["thickness"]
    # Root at the WORK-SURFACE level: the plate's top face (= structural tabletop, where
    # the base bolts) sits mat_thickness BELOW the root -> local z of the top face:
    mat_t = dims["table"]["mats"]["thickness"]
    z_top = -mat_t          # plate top face (base flange level), root at work surface
    z_bot = z_top - th

    stage, root = new_stage(out_path, "custom_mount_plate", mass=None, kinematic=True)
    UsdGeom.Xform.Define(stage, "/custom_mount_plate/visuals")
    UsdGeom.Scope.Define(stage, "/custom_mount_plate/visuals/Looks")
    author_omnipbr(stage, "/custom_mount_plate/visuals/Looks/mount_plate",
                   m["diffuse"], m["metallic"], m["roughness"])
    author_mesh(stage, "/custom_mount_plate/visuals/mount_plate",
                disk_mesh(0.0, 0.0, r, z_bot, z_top),
                "/custom_mount_plate/visuals/Looks/mount_plate")
    col = UsdGeom.Xform.Define(stage, "/custom_mount_plate/collisions")
    author_collision_cube(stage, "/custom_mount_plate/collisions/plate",
                          (0.0, 0.0, (z_top + z_bot) / 2.0), (2 * r * 0.7, 2 * r * 0.7, th))
    stage.Save()
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", default=os.path.normpath(ASSET_DIR_DEFAULT),
                        help="Directory holding table_dims.yaml; USDs are written here.")
    args = parser.parse_args()

    dims_path = os.path.join(args.asset_dir, "table_dims.yaml")
    with open(dims_path) as f:
        dims = yaml.safe_load(f)

    table_usd = make_table(dims, os.path.join(args.asset_dir, "custom_lab_table.usd"))
    plate_usd = make_mount_plate(dims, os.path.join(args.asset_dir, "custom_mount_plate.usd"))
    print(f"[make_custom_table_usd] wrote {table_usd}")
    print(f"[make_custom_table_usd] wrote {plate_usd}")
    print("[make_custom_table_usd] verify: python -c \"from pxr import Usd,...\" bbox probe, "
          "or scripts_v2/tools/conversions/view_usd.py")


if __name__ == "__main__":
    main()
