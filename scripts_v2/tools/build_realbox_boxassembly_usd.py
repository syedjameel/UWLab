# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the RealBox (customer enclosure) box-assembly prop USDs.

Replaces the synthetic Bottom / Mid / CapRim props with the real enclosure parts:

* ``Bottom``  — the tray (Низ, 145 x 93.5 x 12.7 mm), visual mesh + 16 hand-built box colliders.
* ``Mid``     — the circuit board (Плата, 123 x 89.5 x 25.5 mm), 37 box colliders that double
  as the visuals (the full 317k-triangle CAD mesh is deliberately not shipped).
* ``CapRim``  — the cap rev-2 (Крышка, 125 x 90 x 28.5 mm, 4 corner tabs), visual mesh +
  17 hand-built box colliders.
* ``TableCenterTarget`` — the flat goal marker, resized for the bigger tray footprint.

Hand-built boxes (``realbox_boxassembly_boxes.json``, measured from the customer STEP meshes)
instead of trimesh/SDF colliders: an exact-trimesh collider explodes the PhysX GPU collision
stack when thousands of envs reset in contact (see the jig-enclosure branch, commit 0bcd80b),
and convex decomposition inflates the 0.2-0.3 mm cap-tab/cavity fits.

Frames: every part's origin sits at its lowest point (tray: base centre; circuit: lowest
point; cap: tab bottoms), xy at the part bbox centre, so ``bottom_offset`` is zero for all
three. Assembled offsets (metadata.yaml) place the parts in the measured closed pose:
circuit origin at (0, +2.5, +5.0) mm and cap origin at (0, +1.75, +11.3) mm in the tray frame
(the tray cavity is 1.75 mm off-centre in +y; the extra 0.75 mm on the circuit centres the
plate rather than the bbox).

Run with any python that has ``pxr`` + ``trimesh`` + ``yaml``::

    ./uwlab.sh -p scripts_v2/tools/build_realbox_boxassembly_usd.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import trimesh
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from omnireset_asset_utils import add_box, create_stage, write_metadata


def add_cube_collider(stage, prim_path: str, center, half_extents, material_path: str) -> None:
    """Author an analytic box collider as a UsdGeom.Cube prim (native PhysX box shape).

    add_box authors colliders as meshes with a convexHull approximation; PhysX GPU convex
    cooking FAILS on extreme-aspect boxes (e.g. the 122x88x1 mm tray floor) and silently
    falls back to CPU-only collision, which never collides with GPU-pipeline dynamics —
    the circuit dropped straight through the tray floor. A Cube prim is an exact box shape
    with no cooking at all (this is also how the original BoxAssembly props were authored).
    """
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.GetSizeAttr().Set(2.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in half_extents]))
    UsdGeom.Imageable(cube).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    # contact/rest offsets like the original RealBox circuit cubes (PhysxSchema is not
    # importable outside kit; apply the API schema by name)
    cube.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
    cube.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.001)
    cube.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    binding.Bind(material, bindingStrength=UsdShade.Tokens.weakerThanDescendants)

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TOOLS_DIR))
_DEFAULT_OUT = os.path.join(_REPO_ROOT, "source/uwlab_assets/data/Props/BoxAssembly")
_BOXES_JSON = os.path.join(_TOOLS_DIR, "realbox_boxassembly_boxes.json")

# Assembled offsets, object frame (see mdp/events.py assembly_sampling_event: the insertive
# origin is placed at mating_frame - R * assembled_offset.pos, mating frame = Bottom origin).
_ASSEMBLED = {
    "Mid": [0.0, -0.0025, -0.005],
    "CapRim": [0.0, -0.00175, -0.0113],
}
_COLORS = {
    "Bottom": (0.72, 0.52, 0.30),
    "Mid": (0.20, 0.62, 0.30),
    "CapRim": (0.30, 0.52, 0.78),
}
# Target marker: tray footprint (145 x 93.5) + ~5 mm margin per side.
_TARGET_HALF = (0.0775, 0.0525, 0.001)


def add_visual_mesh(stage, prim_path: str, mesh: trimesh.Trimesh, color) -> None:
    """Author a triangle mesh as a rendered visual (no physics)."""
    prim = UsdGeom.Mesh.Define(stage, prim_path)
    prim.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in mesh.vertices]))
    prim.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mesh.faces)))
    prim.CreateFaceVertexIndicesAttr(Vt.IntArray(mesh.faces.flatten().tolist()))
    prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    prim.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    lo, hi = mesh.bounds
    prim.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))


def build_part(name: str, usd_name: str, boxes: list[dict], out_dir: str, mesh_stl: str | None) -> None:
    part_dir = os.path.join(out_dir, name)
    usd_path = os.path.join(part_dir, usd_name)
    if os.path.exists(usd_path):
        os.remove(usd_path)
    stage, _, mat = create_stage(usd_path, root_name=name)

    color = _COLORS[name]
    if mesh_stl is not None and os.path.exists(mesh_stl):
        mesh = trimesh.load(mesh_stl, force="mesh")
        add_visual_mesh(stage, f"/{name}/visuals/mesh", mesh, color)
    else:
        if mesh_stl is not None:
            print(f"[{name}] WARNING: {mesh_stl} missing — falling back to box visuals")
        for b in boxes:
            mn, mx = np.array(b["min"]), np.array(b["max"])
            add_box(stage, f"/{name}/visuals/{b['name']}", center=((mn + mx) / 2).tolist(),
                    half_extents=((mx - mn) / 2).tolist(), collision=False, color=color)

    for b in boxes:
        mn, mx = np.array(b["min"]), np.array(b["max"])
        add_cube_collider(stage, f"/{name}/collisions/{b['name']}", center=((mn + mx) / 2).tolist(),
                          half_extents=((mx - mn) / 2).tolist(), material_path=mat)
    stage.GetRootLayer().Save()

    metadata = {
        "assembled_offset": {"pos": _ASSEMBLED.get(name, [0.0, 0.0, 0.0]), "quat": [1.0, 0.0, 0.0, 0.0]},
        "bottom_offset": {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
    }
    if name == "Bottom":
        # receptive asset: consumed by TaskCommand / ProgressContext
        metadata["success_thresholds"] = {"position": 0.005, "orientation": 0.05}
    meta_path = write_metadata(usd_path, metadata)
    vol = sum(float(np.prod(np.array(b["max"]) - np.array(b["min"]))) for b in boxes)
    print(f"[{name}] wrote {usd_path} ({len(boxes)} colliders, box volume {vol*1e6:.1f} cm^3 "
          f"-> auto mass ~{vol*1000*1000:.0f} g at default density) + {meta_path}")


def build_target(out_dir: str) -> None:
    part_dir = os.path.join(out_dir, "TableCenterTarget")
    usd_path = os.path.join(part_dir, "target.usd")
    if os.path.exists(usd_path):
        os.remove(usd_path)
    stage, _, mat = create_stage(usd_path, root_name="TableCenterTarget")
    hx, hy, hz = _TARGET_HALF
    # thin plate, origin at its top surface so the tray base sits flush on the marker
    add_box(stage, "/TableCenterTarget/visuals/plate", center=(0, 0, -hz), half_extents=(hx, hy, hz),
            collision=False, color=(0.45, 0.45, 0.48))
    add_cube_collider(stage, "/TableCenterTarget/collisions/plate", center=(0, 0, -hz),
                      half_extents=(hx, hy, hz), material_path=mat)
    stage.GetRootLayer().Save()
    metadata = {
        "assembled_offset": {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
        "bottom_offset": {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
        "success_thresholds": {"position": 0.005, "orientation": 0.05},
    }
    meta_path = write_metadata(usd_path, metadata)
    print(f"[TableCenterTarget] wrote {usd_path} ({2*hx*1000:.0f} x {2*hy*1000:.0f} mm marker) + {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RealBox box-assembly prop USDs.")
    parser.add_argument("--out-dir", type=str, default=_DEFAULT_OUT, help="Props/BoxAssembly directory.")
    parser.add_argument("--boxes-json", type=str, default=_BOXES_JSON, help="Collision boxes JSON.")
    args = parser.parse_args()

    spec = json.load(open(args.boxes_json))
    build_part("Bottom", "bottom.usd", spec["tray"], args.out_dir,
               os.path.join(args.out_dir, "Bottom/source_mesh.stl"))
    build_part("Mid", "mid.usd", spec["circuit"], args.out_dir, None)
    build_part("CapRim", "caprim.usd", spec["cap"], args.out_dir,
               os.path.join(args.out_dir, "CapRim/source_mesh.stl"))
    build_target(args.out_dir)


if __name__ == "__main__":
    main()
