# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the insertive telescoping "cover" (lid) asset (``cover.usd`` + ``metadata.yaml``).

The cover is a shoebox-style cap: a flat top plate plus four downward skirt walls (open
bottom), whose inner cavity is slightly larger than the box's outer footprint so it slides
down *over the outside* of the box walls. It is the *insertive* object for the box-closing
task (receptive = ``box_with_pcb``). Geometry is built from exact axis-aligned boxes; each
box gets its own convex-hull collider (so the cavity stays open and the box can telescope
in). The origin sits at the outer bounding-box center. All dimensions are in meters.

Derived footprint (mirrors ``build_openbox_usd.py`` conventions)::

    lid_inner = box_outer + 2 * clearance     # cavity that slips over the box
    lid_outer = lid_inner + 2 * wall           # skirt outer footprint
    lid_height = top + skirt                    # plate + telescoping skirt

When seated the lid's top-plate underside rests on the box top rim and the skirt overlaps
the top ``skirt`` mm of the box walls.

Run with any python that has ``pxr`` (e.g. ``./uwlab.sh -p``)::

    ./uwlab.sh -p scripts_v2/tools/build_lid_usd.py
"""

from __future__ import annotations

import argparse
import os

from omnireset_asset_utils import add_box, create_stage, write_metadata

_DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "source/uwlab_assets/uwlab_assets/local/Props/Custom/Cover/cover.usd",
)

_COLOR = (0.30, 0.30, 0.35)
_TOP_COLOR = (0.15, 0.15, 0.18)  # darker top so the lid's up-face is identifiable


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the telescoping cover (lid) insertive USD asset.")
    # Box outer footprint the lid must fit over (must match build_openbox_usd output: 58x58x14 mm).
    parser.add_argument("--box-outer-x", type=float, default=0.058, help="Box outer size along X (m).")
    parser.add_argument("--box-outer-y", type=float, default=0.058, help="Box outer size along Y (m).")
    parser.add_argument("--clearance", type=float, default=0.002, help="Gap per side between skirt and box wall (m).")
    parser.add_argument("--wall", type=float, default=0.003, help="Skirt wall thickness (m).")
    parser.add_argument("--top", type=float, default=0.003, help="Top plate thickness (m).")
    parser.add_argument("--skirt", type=float, default=0.006, help="Skirt depth = overlap over box walls (m).")
    parser.add_argument("--output", type=str, default=_DEFAULT_OUT, help="Output .usd path.")
    args = parser.parse_args()

    # Derived dimensions.
    inner_x = args.box_outer_x + 2 * args.clearance
    inner_y = args.box_outer_y + 2 * args.clearance
    outer_x = inner_x + 2 * args.wall
    outer_y = inner_y + 2 * args.wall
    height = args.top + args.skirt

    hox, hoy = outer_x / 2.0, outer_y / 2.0  # outer half-footprint
    hix, hiy = inner_x / 2.0, inner_y / 2.0  # cavity half-footprint
    z_top = height / 2.0  # outer bbox centered on the origin
    z_bottom = -height / 2.0
    plate_bottom = z_top - args.top  # underside of the top plate = where the box rim seats

    # Box definitions: (name, center, half_extents).
    plate_cz = (plate_bottom + z_top) / 2.0
    skirt_cz = (z_bottom + plate_bottom) / 2.0
    skirt_hz = (plate_bottom - z_bottom) / 2.0
    wall_cx = (hix + hox) / 2.0  # mid-thickness of the +/-X skirt walls
    wall_cy = (hiy + hoy) / 2.0
    wall_hwx = (hox - hix) / 2.0  # half thickness of a skirt wall
    wall_hwy = (hoy - hiy) / 2.0

    boxes = [
        ("plate", (0.0, 0.0, plate_cz), (hox, hoy, args.top / 2.0)),
        ("skirt_px", (wall_cx, 0.0, skirt_cz), (wall_hwx, hoy, skirt_hz)),
        ("skirt_nx", (-wall_cx, 0.0, skirt_cz), (wall_hwx, hoy, skirt_hz)),
        ("skirt_py", (0.0, wall_cy, skirt_cz), (hix, wall_hwy, skirt_hz)),
        ("skirt_ny", (0.0, -wall_cy, skirt_cz), (hix, wall_hwy, skirt_hz)),
    ]

    if os.path.exists(args.output):
        os.remove(args.output)
    stage, _, mat = create_stage(args.output, root_name="Cover")
    for name, center, half in boxes:
        # Only the top plate carries the darker top color so the lid's up-face is visible.
        top_c = _TOP_COLOR if name == "plate" else None
        add_box(stage, f"/Cover/visuals/{name}", center=center, half_extents=half,
                collision=False, color=_COLOR, top_color=top_c)
        add_box(stage, f"/Cover/collisions/{name}", center=center, half_extents=half,
                collision=True, material_path=mat)
    stage.GetRootLayer().Save()

    # assembled_offset point = top-plate underside center (rests on the box top rim when seated).
    # bottom_offset.z = -(origin -> lowest point) = z_bottom (skirt bottom).
    metadata = {
        "assembled_offset": {"pos": [0.0, 0.0, round(plate_bottom, 6)], "quat": [1.0, 0.0, 0.0, 0.0]},
        "bottom_offset": {"pos": [0.0, 0.0, round(z_bottom, 6)], "quat": [1.0, 0.0, 0.0, 0.0]},
    }
    meta_path = write_metadata(args.output, metadata)

    print(f"Wrote {args.output}")
    print(f"Wrote {meta_path}")
    print(f"  lid inner footprint   : {inner_x:.3f} x {inner_y:.3f} m (fits over {args.box_outer_x:.3f} x {args.box_outer_y:.3f})")
    print(f"  lid outer size (LxWxH): {outer_x:.3f} x {outer_y:.3f} x {height:.3f} m")
    print(f"  skirt overlap         : {args.skirt:.3f} m")
    print(f"  assembled_offset.z    = {plate_bottom:.6f} (top-plate underside / seats on box rim)")
    print(f"  bottom_offset.z       = {z_bottom:.6f}")


if __name__ == "__main__":
    main()
