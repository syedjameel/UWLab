# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the insertive "PCB" slab asset (``pcb.usd`` + ``metadata.yaml``).

The PCB is a flat rectangular slab whose origin sits at its bounding-box center. It is the
*insertive* object for the open-box insertion task. All dimensions are in meters.

Run with any python that has ``pxr`` (e.g. ``./uwlab.sh -p``)::

    ./uwlab.sh -p scripts_v2/tools/build_pcb_usd.py
"""

from __future__ import annotations

import argparse
import os

from omnireset_asset_utils import add_box, create_stage, write_metadata

_DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "source/uwlab_assets/uwlab_assets/local/Props/Custom/Pcb/pcb.usd",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the PCB insertive slab USD asset.")
    parser.add_argument("--length", type=float, default=0.040, help="PCB length along X (m).")
    parser.add_argument("--width", type=float, default=0.040, help="PCB width along Y (m).")
    parser.add_argument("--thickness", type=float, default=0.020, help="PCB thickness along Z (m).")
    parser.add_argument("--output", type=str, default=_DEFAULT_OUT, help="Output .usd path.")
    args = parser.parse_args()

    hx, hy, hz = args.length / 2.0, args.width / 2.0, args.thickness / 2.0

    if os.path.exists(args.output):
        os.remove(args.output)
    stage, _, mat = create_stage(args.output, root_name="Pcb")

    # Origin at the slab center -> visual and collision boxes are both centered at z=0.
    # The +Z (top) face is dark green so the top side is visually identifiable in sim/playback.
    add_box(stage, "/Pcb/visuals/slab", center=(0, 0, 0), half_extents=(hx, hy, hz),
            collision=False, color=(0.20, 0.55, 0.30), top_color=(0.0, 0.20, 0.05))
    add_box(stage, "/Pcb/collisions/slab", center=(0, 0, 0), half_extents=(hx, hy, hz),
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
