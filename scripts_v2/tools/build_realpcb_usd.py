# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the real thin-PCB insertive slab asset (``realpcb.usd`` + ``metadata.yaml``).

This mirrors ``build_pcb_usd.py`` but defaults to the user's *real* PCB footprint:
140 mm x 100 mm x 3 mm. Unlike the deployed ``pcb`` proxy (a 40 mm cube), this is a
genuinely thin slab lying flat, so its vertical side faces are only 3 mm tall -- the
worst case for a top-down parallel-jaw grasp. All dimensions are in meters.

Run with any python that has ``pxr`` (e.g. ``./uwlab.sh -p``)::

    ./uwlab.sh -p scripts_v2/tools/build_realpcb_usd.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import trimesh

from omnireset_asset_utils import add_box, create_stage, write_metadata
from pxr import Gf, UsdGeom, UsdPhysics, Vt

_DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "source/uwlab_assets/uwlab_assets/local/Props/Custom/RealPcb/realpcb.usd",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the real thin-PCB insertive slab USD asset.")
    parser.add_argument("--length", type=float, default=0.140, help="PCB length along X (m).")
    parser.add_argument("--width", type=float, default=0.100, help="PCB width along Y (m).")
    parser.add_argument("--thickness", type=float, default=0.003, help="PCB thickness along Z (m).")
    parser.add_argument("--output", type=str, default=_DEFAULT_OUT, help="Output .usd path.")
    parser.add_argument("--visual-stl", type=str, default=None,
                        help="Use this decimated STL as the VISUAL mesh instead of a plain slab. "
                             "The COLLIDER stays the box slab: the real board carries 20 mm "
                             "connectors, but a parallel jaw closes on the board's edge PLANE "
                             "and never reaches them, so no extra collision geometry is needed "
                             "-- and adding any would be a deviation from the real part.")
    args = parser.parse_args()

    hx, hy, hz = args.length / 2.0, args.width / 2.0, args.thickness / 2.0

    if os.path.exists(args.output):
        os.remove(args.output)
    stage, _, mat = create_stage(args.output, root_name="RealPcb")

    # Origin at the slab center -> visual and collision boxes are both centered at z=0.
    # The +Z (top) face is dark green so the top side is visually identifiable in sim/playback.
    if args.visual_stl:
        # Real board geometry for RENDERING only. Costs nothing in training: Stage-1 and the
        # finetune run headless (no cameras -> never rasterised) and PhysX only ever integrates
        # the collider below. Aligned so the mesh's UNDERSIDE coincides with the slab collider's
        # underside, keeping bottom_offset (and therefore the measured seat) exact.
        vm = trimesh.load(args.visual_stl, force="mesh")
        vm.apply_scale(0.001)
        vm.apply_translation(-vm.bounds.mean(axis=0))
        vm.apply_translation([0.0, 0.0, -hz - vm.bounds[0][2]])
        m = UsdGeom.Mesh.Define(stage, "/RealPcb/visuals/board")
        m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in vm.vertices.astype(float)]))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(vm.faces)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray(vm.faces.flatten().tolist()))
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        lo, hi = vm.bounds
        m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))
        m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in np.repeat(vm.face_normals, 3, axis=0)]))
        m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        m.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.20, 0.55, 0.30)]))
        print(f"  visual: {args.visual_stl} ({len(vm.faces):,} faces), underside aligned to the slab")
    else:
        add_box(stage, "/RealPcb/visuals/slab", center=(0, 0, 0), half_extents=(hx, hy, hz),
                collision=False, color=(0.20, 0.55, 0.30), top_color=(0.0, 0.20, 0.05))
    add_box(stage, "/RealPcb/collisions/slab", center=(0, 0, 0), half_extents=(hx, hy, hz),
            collision=True, material_path=mat)
    stage.GetRootLayer().Save()

    # bottom_offset.z = -(origin -> lowest point) = -thickness/2.
    # assembled_offset point = slab bottom-center (rests on the cavity floor when assembled).
    bottom_z = -hz
    metadata = {
        "assembled_offset": {"pos": [0.0, 0.0, round(bottom_z, 6)], "quat": [1.0, 0.0, 0.0, 0.0]},
        "bottom_offset": {"pos": [0.0, 0.0, round(bottom_z, 6)], "quat": [1.0, 0.0, 0.0, 0.0]},
    }
    meta_path = write_metadata(args.output, metadata)

    print(f"Wrote {args.output}")
    print(f"Wrote {meta_path}")
    print(f"  PCB size (LxWxT): {args.length} x {args.width} x {args.thickness} m")
    print(f"  bottom_offset.z = assembled_offset.z = {bottom_z:.6f}")


if __name__ == "__main__":
    main()
