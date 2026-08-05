# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Can a CLOSED gripper reach into the jig window and press the PCB down to its seat?

The board is gripped across its 100 mm axis, so the jaws sit at y = +-50 while the window is only
101.5 mm wide -- an OPEN gripper cannot follow the board in. Closed, the pads span ~40 mm and
should fit with room to spare, but the finger carriages above them are far wider (~149 mm when
closed, from USD bounds), so whether the gripper actually reaches the seat depends on how far the
narrow pad section protrudes past the carriage. Bounding boxes cannot answer that; this does.

Method mirrors test_pcb_seat.py: drop the standalone gripper, jaws CLOSED, straight down over the
window centre and record the lowest point it reaches. Free-fall onto a kinematic fixture, on a raw
SimulationContext -- no arm, no IK, no controller to confound the result.

    ./uwlab.sh -p scripts_v2/tools/conversions/test_gripper_reach.py --headless
    ./uwlab.sh -p scripts_v2/tools/conversions/test_gripper_reach.py           # watch it
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--settle_steps", type=int, default=500)
parser.add_argument("--start_mm", type=float, default=60.0, help="start height above the jig top")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR

_ASM = f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/JigEnclosure/jig_enclosure.usd"
_GRIP = f"{UWLAB_LOCAL_ASSETS_DIR}/Robots/LinearGripper/linear_gripper.usd"

ENC_HALF_H = 0.0113
ASM_HALF_H = 0.0208          # assembly half-height (41.6 mm tall)
SEAT_ABOVE_ENC_BOTTOM = 0.0136
JAWS = ["finger_joint", "right_finger_joint"]


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args_cli.device))
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))

    # assembly, kinematic, enclosure bottom at -ENC_HALF_H (same convention as test_pcb_seat)
    asm_z = ASM_HALF_H - ENC_HALF_H
    cfg = sim_utils.UsdFileCfg(
        usd_path=_ASM, rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True))
    cfg.func("/World/Assembly", cfg, translation=(0.0, 0.0, asm_z))

    jig_top = asm_z + ASM_HALF_H
    seat = -ENC_HALF_H + SEAT_ABOVE_ENC_BOTTOM
    start_z = jig_top + args_cli.start_mm / 1000.0

    grip = Articulation(ArticulationCfg(
        prim_path="/World/Gripper",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_GRIP,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, start_z)),
        actuators={},
    ))

    sim.set_camera_view(eye=(0.28, -0.28, 0.16), target=(0.0, 0.0, 0.02))
    sim.reset()

    # jaws CLOSED (joint pos 0 = shut for this dual-drive gripper; open is positive)
    ids, names = grip.find_joints(JAWS)
    q = grip.data.default_joint_pos.clone()
    q[:, ids] = 0.0
    grip.write_joint_state_to_sim(q, torch.zeros_like(q))
    print(f"[reach] closed jaws: {names}", flush=True)

    lowest = 1e9
    for _ in range(args_cli.settle_steps):
        grip.write_data_to_sim()
        sim.step()
        grip.update(sim.get_physics_dt())
        lowest = min(lowest, float(grip.data.body_pos_w[0, :, 2].min()))

    final = float(grip.data.body_pos_w[0, :, 2].min())
    print("\n=== CLOSED-GRIPPER REACH (mm, world) ===", flush=True)
    print(f"  jig top face   : {jig_top*1000:8.2f}", flush=True)
    print(f"  PCB seat       : {seat*1000:8.2f}", flush=True)
    print(f"  gripper lowest : {final*1000:8.2f}   (min over the drop: {lowest*1000:.2f})", flush=True)
    depth = (jig_top - final) * 1000.0
    need = (jig_top - seat) * 1000.0
    print(f"  entered the window by {depth:.2f} mm of the {need:.2f} mm needed to touch the seat",
          flush=True)
    if final <= seat + 0.002:
        print("  -> REACHES the seat: a closed gripper CAN press the board home", flush=True)
    elif depth > 2.0:
        print(f"  -> PARTIAL: enters {depth:.1f} mm but stops {(final-seat)*1000:.1f} mm short", flush=True)
    else:
        print("  -> BLOCKED at the rim: the insert is gravity-only", flush=True)

    sys.stdout.flush()
    if not args_cli.headless:
        print("\n  GUI open -- close the window to exit.", flush=True)
        while simulation_app.is_running():
            sim.step()
    os._exit(0)


if __name__ == "__main__":
    main()
