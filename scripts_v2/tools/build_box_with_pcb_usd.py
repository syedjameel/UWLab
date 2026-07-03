# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the receptive "box with PCB" asset (``box_with_pcb.usd`` + ``metadata.yaml``).

This is the receptive object for the box-closing task (insertive = ``cover``). It is the
same open-top box as ``build_openbox_usd.py`` (floor + 4 walls) but with a PCB slab modeled
seated on the cavity floor, recessed below the rim, and the mating point moved to the box
**top rim** (where the telescoping lid seats) instead of the cavity floor.

The modeled PCB is part of the (kinematic) receptive geometry and is purely representative:
the lid telescopes over the *outside* of the box and never contacts the PCB. Keep the
modeled PCB thickness <= cavity depth so it stays below the rim and the lid can close.

Derived footprint matches ``build_openbox_usd.py``::

    cavity_inner = pcb + 2 * clearance
    box_outer    = cavity_inner + 2 * wall
    box_height   = floor + depth

Run with any python that has ``pxr`` (e.g. ``./uwlab.sh -p``)::

    ./uwlab.sh -p scripts_v2/tools/build_box_with_pcb_usd.py
"""

from __future__ import annotations

import argparse
import os

from omnireset_asset_utils import add_box, create_stage, write_metadata

_DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "source/uwlab_assets/uwlab_assets/local/Props/Custom/BoxWithPcb/box_with_pcb.usd",
)

_BOX_COLOR = (0.55, 0.55, 0.60)
_PCB_COLOR = (0.20, 0.55, 0.30)
_PCB_TOP_COLOR = (0.0, 0.20, 0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the box-with-PCB receptive USD asset (for the lid task).")
    parser.add_argument("--pcb-length", type=float, default=0.040, help="PCB length along X (m).")
    parser.add_argument("--pcb-width", type=float, default=0.040, help="PCB width along Y (m).")
    parser.add_argument("--clearance", type=float, default=0.005, help="Gap per side between PCB and wall (m).")
    parser.add_argument("--wall", type=float, default=0.004, help="Side wall thickness (m).")
    parser.add_argument("--floor", type=float, default=0.004, help="Floor thickness (m).")
    parser.add_argument("--depth", type=float, default=0.010, help="Cavity depth above the floor (m).")
    parser.add_argument("--pcb-thickness", type=float, default=0.005,
                        help="Modeled seated PCB thickness (m); must be <= depth so it stays below the rim.")
    parser.add_argument("--output", type=str, default=_DEFAULT_OUT, help="Output .usd path.")
    args = parser.parse_args()

    if args.pcb_thickness > args.depth:
        raise SystemExit(
            f"--pcb-thickness ({args.pcb_thickness}) exceeds cavity --depth ({args.depth}); "
            "the PCB would protrude above the rim and the lid could not close."
        )

    # Derived box dimensions (identical to build_openbox_usd.py).
    inner_x = args.pcb_length + 2 * args.clearance
    inner_y = args.pcb_width + 2 * args.clearance
    outer_x = inner_x + 2 * args.wall
    outer_y = inner_y + 2 * args.wall
    height = args.floor + args.depth

    hox, hoy = outer_x / 2.0, outer_y / 2.0
    hix, hiy = inner_x / 2.0, inner_y / 2.0
    z_bottom = -height / 2.0
    floor_top = z_bottom + args.floor
    z_top = height / 2.0

    floor_cz = (z_bottom + floor_top) / 2.0
    wall_cz = (floor_top + z_top) / 2.0
    wall_hz = (z_top - floor_top) / 2.0
    wall_cx = (hix + hox) / 2.0
    wall_cy = (hiy + hoy) / 2.0
    wall_hwx = (hox - hix) / 2.0
    wall_hwy = (hoy - hiy) / 2.0

    boxes = [
        ("floor", (0.0, 0.0, floor_cz), (hox, hoy, args.floor / 2.0)),
        ("wall_px", (wall_cx, 0.0, wall_cz), (wall_hwx, hoy, wall_hz)),
        ("wall_nx", (-wall_cx, 0.0, wall_cz), (wall_hwx, hoy, wall_hz)),
        ("wall_py", (0.0, wall_cy, wall_cz), (hix, wall_hwy, wall_hz)),
        ("wall_ny", (0.0, -wall_cy, wall_cz), (hix, wall_hwy, wall_hz)),
    ]

    # Modeled PCB: rests on the cavity floor, footprint pcb_length x pcb_width, recessed below rim.
    pcb_hz = args.pcb_thickness / 2.0
    pcb_cz = floor_top + pcb_hz
    pcb_half = (args.pcb_length / 2.0, args.pcb_width / 2.0, pcb_hz)

    if os.path.exists(args.output):
        os.remove(args.output)
    stage, _, mat = create_stage(args.output, root_name="BoxWithPcb")
    for name, center, half in boxes:
        add_box(stage, f"/BoxWithPcb/visuals/{name}", center=center, half_extents=half,
                collision=False, color=_BOX_COLOR)
        add_box(stage, f"/BoxWithPcb/collisions/{name}", center=center, half_extents=half,
                collision=True, material_path=mat)
    # Seated PCB (dark-green top), modeled as part of the receptive geometry.
    add_box(stage, "/BoxWithPcb/visuals/pcb", center=(0.0, 0.0, pcb_cz), half_extents=pcb_half,
            collision=False, color=_PCB_COLOR, top_color=_PCB_TOP_COLOR)
    add_box(stage, "/BoxWithPcb/collisions/pcb", center=(0.0, 0.0, pcb_cz), half_extents=pcb_half,
            collision=True, material_path=mat)
    stage.GetRootLayer().Save()

    # assembled_offset point = box top rim center (the telescoping lid's plate underside seats here).
    # bottom_offset.z = -height/2 (origin -> lowest point).
    metadata = {
        "assembled_offset": {"pos": [0.0, 0.0, round(z_top, 6)], "quat": [1.0, 0.0, 0.0, 0.0]},
        "bottom_offset": {"pos": [0.0, 0.0, round(z_bottom, 6)], "quat": [1.0, 0.0, 0.0, 0.0]},
        "success_thresholds": {"position": 0.005, "orientation": 0.025},
    }
    meta_path = write_metadata(args.output, metadata)

    print(f"Wrote {args.output}")
    print(f"Wrote {meta_path}")
    print(f"  box outer size (LxWxH): {outer_x:.3f} x {outer_y:.3f} x {height:.3f} m")
    print(f"  modeled PCB           : {args.pcb_length:.3f} x {args.pcb_width:.3f} x {args.pcb_thickness:.3f} m"
          f" (top {args.depth - args.pcb_thickness:.3f} m below rim)")
    print(f"  assembled_offset.z    = {z_top:.6f} (box top rim / lid seats here)")
    print(f"  bottom_offset.z       = {z_bottom:.6f}")


if __name__ == "__main__":
    main()
