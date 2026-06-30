# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematically animate the gripper opening/closing in the GUI, with a slab drawn between the
jaws -- so you can SEE whether the closing jaws go around the slab or miss it.

This is physics-FREE: it just slides the finger prims and renders, so it will NOT trigger the
physics-stepping hang on the laptop. The red slab is fixed at ``--finger-offset`` on the
gripper +Z axis (where grasp sampling would place the object). Watch whether the jaw faces
close ONTO the slab or slide past it. Close the window (or Ctrl+C) to exit.

Run on the laptop (GUI)::

    ./uwlab.sh -p scripts_v2/tools/conversions/view_gripper_close.py \
        --usd source/uwlab_assets/uwlab_assets/local/Robots/LinearGripper/linear_gripper.usd \
        --finger-offset 0.13
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Kinematically view the gripper open/close in the GUI.")
parser.add_argument("--usd", type=str, required=True)
parser.add_argument("--finger-offset", type=float, default=0.13, help="Slab height on the gripper +Z axis (m).")
parser.add_argument("--slab-x", type=float, default=0.040)
parser.add_argument("--slab-y", type=float, default=0.040)
parser.add_argument("--slab-z", type=float, default=0.020)
parser.add_argument("--close-value", type=float, default=0.068)
parser.add_argument("--speed", type=float, default=0.04, help="Animation speed (rad/frame of the cycle).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False  # GUI

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math
import os

import omni.usd
from pxr import Gf, UsdGeom

omni.usd.get_context().open_stage(os.path.abspath(args.usd))
stage = omni.usd.get_context().get_stage()

# Draw the slab (red) at the grasp height on the gripper +Z axis (base is at the origin).
slab_xf = UsdGeom.Xform.Define(stage, "/Slab")
sx = slab_xf.AddTranslateOp()
sx.Set(Gf.Vec3d(0.0, 0.0, args.finger_offset))
cube = UsdGeom.Cube.Define(stage, "/Slab/box")
cube.GetSizeAttr().Set(1.0)
UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(Gf.Vec3f(args.slab_x, args.slab_y, args.slab_z))
cube.GetDisplayColorAttr().Set([(0.9, 0.1, 0.1)])

left = UsdGeom.Xformable(stage.GetPrimAtPath("/linear_gripper/left_inner_finger"))
right = UsdGeom.Xformable(stage.GetPrimAtPath("/linear_gripper/right_inner_finger"))
left_t = left.GetOrderedXformOps()[0]
right_t = right.GetOrderedXformOps()[0]
L0 = left_t.Get()
R0 = right_t.Get()

print(f"Animating jaws 0..{args.close_value} (red slab at z={args.finger_offset}). Close the window to exit.")
t = 0.0
while simulation_app.is_running():
    v = 0.5 * args.close_value * (1.0 - math.cos(t))  # smooth oscillation 0..close..0
    left_t.Set(Gf.Vec3d(L0[0] + v, L0[1], L0[2]))     # left jaw slides +X toward center
    right_t.Set(Gf.Vec3d(R0[0] - v, R0[1], R0[2]))    # right jaw slides -X toward center
    simulation_app.update()
    t += args.speed

simulation_app.close()
