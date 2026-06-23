# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the receptive open-top box asset (``open_box.usd`` + ``metadata.yaml``).

The box is an open-top rectangular tray (floor + 4 walls, no lid) with a cavity slightly
larger than the PCB. It is the *receptive* object for the insertion task. Geometry is built
from exact axis-aligned boxes; collision uses per-box convex hulls (no PhysX SDF needed).
The origin sits at the outer bounding-box center. All dimensions are in meters.

Derived footprint::

    cavity_inner = pcb + 2 * clearance
    box_outer    = cavity_inner + 2 * wall
    box_height   = floor + depth

Run with any python that has ``pxr`` (e.g. ``./uwlab.sh -p``)::

    ./uwlab.sh -p scripts_v2/tools/build_openbox_usd.py
"""

from __future__ import annotations

import argparse
import os

from omnireset_asset_utils import add_box, create_stage, write_metadata

_DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "source/uwlab_assets/uwlab_assets/local/Props/Custom/OpenBox/open_box.usd",
)

_COLOR = (0.55, 0.55, 0.60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the open-top box receptive USD asset.")
    parser.add_argument("--pcb-length", type=float, default=0.050, help="Mating PCB length along X (m).")
    parser.add_argument("--pcb-width", type=float, default=0.040, help="Mating PCB width along Y (m).")
    parser.add_argument("--clearance", type=float, default=0.005, help="Gap per side between PCB and wall (m).")
    parser.add_argument("--wall", type=float, default=0.004, help="Side wall thickness (m).")
    parser.add_argument("--floor", type=float, default=0.004, help="Floor thickness (m).")
    parser.add_argument("--depth", type=float, default=0.050, help="Cavity depth above the floor (m).")
    parser.add_argument("--output", type=str, default=_DEFAULT_OUT, help="Output .usd path.")
    args = parser.parse_args()

    # Derived dimensions.
    inner_x = args.pcb_length + 2 * args.clearance
    inner_y = args.pcb_width + 2 * args.clearance
    outer_x = inner_x + 2 * args.wall
    outer_y = inner_y + 2 * args.wall
    height = args.floor + args.depth

    hox, hoy = outer_x / 2.0, outer_y / 2.0  # outer half-footprint
    hix, hiy = inner_x / 2.0, inner_y / 2.0  # cavity half-footprint
    z_bottom = -height / 2.0  # outer bbox is centered on the origin
    floor_top = z_bottom + args.floor  # top surface of the floor = where the PCB rests
    z_top = height / 2.0

    # Box definitions: (name, center, half_extents).
    floor_cz = (z_bottom + floor_top) / 2.0
    wall_cz = (floor_top + z_top) / 2.0
    wall_hz = (z_top - floor_top) / 2.0
    wall_cx = (hix + hox) / 2.0  # mid-thickness of the +/-X walls
    wall_cy = (hiy + hoy) / 2.0
    wall_hwx = (hox - hix) / 2.0  # half thickness of a wall
    wall_hwy = (hoy - hiy) / 2.0

    boxes = [
        ("floor", (0.0, 0.0, floor_cz), (hox, hoy, args.floor / 2.0)),
        ("wall_px", (wall_cx, 0.0, wall_cz), (wall_hwx, hoy, wall_hz)),
        ("wall_nx", (-wall_cx, 0.0, wall_cz), (wall_hwx, hoy, wall_hz)),
        ("wall_py", (0.0, wall_cy, wall_cz), (hix, wall_hwy, wall_hz)),
        ("wall_ny", (0.0, -wall_cy, wall_cz), (hix, wall_hwy, wall_hz)),
    ]

    if os.path.exists(args.output):
        os.remove(args.output)
    stage, _, mat = create_stage(args.output, root_name="OpenBox")
    for name, center, half in boxes:
        add_box(stage, f"/OpenBox/visuals/{name}", center=center, half_extents=half,
                collision=False, color=_COLOR)
        add_box(stage, f"/OpenBox/collisions/{name}", center=center, half_extents=half,
                collision=True, material_path=mat)
    stage.GetRootLayer().Save()

    # bottom_offset.z = -height/2 (origin -> lowest point).
    # assembled_offset point = cavity-floor center (PCB bottom-center lands here when assembled).
    metadata = {
        "assembled_offset": {"pos": [0.0, 0.0, round(floor_top, 6)], "quat": [1.0, 0.0, 0.0, 0.0]},
        "bottom_offset": {"pos": [0.0, 0.0, round(z_bottom, 6)], "quat": [1.0, 0.0, 0.0, 0.0]},
        # Loose fit (PCB + 5mm/side clearance); tighten for harder tasks.
        "success_thresholds": {"position": 0.005, "orientation": 0.05},
    }
    meta_path = write_metadata(args.output, metadata)

    print(f"Wrote {args.output}")
    print(f"Wrote {meta_path}")
    print(f"  cavity inner footprint : {inner_x:.3f} x {inner_y:.3f} m")
    print(f"  box outer size (LxWxH) : {outer_x:.3f} x {outer_y:.3f} x {height:.3f} m")
    print(f"  bottom_offset.z   = {z_bottom:.6f}")
    print(f"  assembled_offset.z= {floor_top:.6f} (cavity floor top)")


if __name__ == "__main__":
    main()
