# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drop the real PCB into the assembled jig+enclosure and record where it actually settles.

Defines the assembly seat EMPIRICALLY rather than from CAD intent. Mesh probing of the STL
says the PCB's 140 mm ends graze corner bosses at 20.05 mm with ~0 mm overlap, while the
hand-built BOX COLLIDERS -- which are what PhysX actually uses -- let it fall to the long-wall
ridges at 13.60 mm. Only a physics drop settles which one the sim will really produce, and
whether the resulting pose is stable and repeatable enough to be a success criterion.

Runs on a raw SimulationContext, NOT inside the task env: the task's action manager and reset
events would re-place the object every reset and mask the result (this exact mistake voided two
earlier blocker experiments).

    ./uwlab.sh -p scripts_v2/tools/conversions/test_pcb_seat.py --trials 16 --headless
    ./uwlab.sh -p scripts_v2/tools/conversions/test_pcb_seat.py --trials 4      # watch it
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=16)
parser.add_argument("--jitter_mm", type=float, default=2.0, help="+/- x,y spawn jitter")
parser.add_argument("--jitter_yaw_deg", type=float, default=2.0)
parser.add_argument("--drop_mm", type=float, default=45.0, help="spawn height above the jig top")
parser.add_argument("--settle_steps", type=int, default=600)
parser.add_argument("--jig_seat_mm", type=float, default=18.3, help="jig centre above enclosure centre")
parser.add_argument("--repeats", type=int, default=1, help="re-drop this many times (watch it in the GUI)")
parser.add_argument("--assembly", action="store_true",
                    help="drop onto the COMBINED JigEnclosure asset instead of enclosure+jig "
                         "spawned separately -- must reproduce the same seat")
parser.add_argument("--hold_steps", type=int, default=120, help="pause on the settled pose between drops")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext, SimulationCfg

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR

_ENC = f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/BottomEnclosure/bottom_enclosure.usd"
_JIG = f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/Jig/jig.usd"
_PCB = f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/RealPcb/realpcb.usd"
_ASM = f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/JigEnclosure/jig_enclosure.usd"

ENC_HALF_H = 0.0113   # enclosure half-height (m); its bottom face is -ENC_HALF_H from centre
PCB_HALF_T = 0.0015   # PCB half-thickness (m)


def _static(prim_path: str, usd: str, z: float) -> None:
    """Spawn an immovable copy (kinematic rigid body -- collides, never moves)."""
    sim_utils.UsdFileCfg(
        usd_path=usd,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    ).func(prim_path, sim_utils.UsdFileCfg(
        usd_path=usd,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    ), translation=(0.0, 0.0, z))


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args_cli.device))
    spacing = 0.6
    rng = np.random.default_rng(0)

    # GUI needs a light, and a camera pointed at the first trial -- otherwise it opens black.
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.95)).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.95)))
    sim.set_camera_view(eye=(0.32, -0.34, 0.22), target=(0.0, 0.0, 0.02))

    jig_z = args_cli.jig_seat_mm / 1000.0
    jig_top = jig_z + 0.012                      # jig half-height 12 mm
    spawn_z = jig_top + args_cli.drop_mm / 1000.0

    jit = args_cli.jitter_mm / 1000.0
    dx = rng.uniform(-jit, jit, args_cli.trials)
    dy = rng.uniform(-jit, jit, args_cli.trials)
    dyaw = np.deg2rad(rng.uniform(-args_cli.jitter_yaw_deg, args_cli.jitter_yaw_deg, args_cli.trials))

    for i in range(args_cli.trials):
        ox = (i % 8) * spacing
        oy = (i // 8) * spacing
        root = f"/World/trial_{i:02d}"
        if args_cli.assembly:
            # the assembly's origin is the PAIR's bbox centre, so its enclosure bottom sits
            # 20.8 mm below it; lift it so that bottom lands at -ENC_HALF_H like the 2-part case
            _static(f"{root}/Assembly", _ASM, 0.0208 - ENC_HALF_H)
            subs = ("Assembly",)
        else:
            _static(f"{root}/Enclosure", _ENC, 0.0)
            _static(f"{root}/Jig", _JIG, jig_z)
            subs = ("Enclosure", "Jig")
        # shift the whole trial sideways by translating each prim
        from pxr import UsdGeom, Gf
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        for sub in subs:
            p = stage.GetPrimAtPath(f"{root}/{sub}")
            x = UsdGeom.Xformable(p)
            tr = [op for op in x.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
            cur = tr[0].Get() if tr else Gf.Vec3d(0, 0, 0)
            (tr[0] if tr else x.AddTranslateOp()).Set(Gf.Vec3d(cur[0] + ox, cur[1] + oy, cur[2]))

    pcb_cfg = RigidObjectCfg(
        prim_path="/World/trial_.*/Pcb",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_PCB,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                disable_gravity=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, spawn_z)),
    )
    pcb = RigidObject(pcb_cfg)

    sim.reset()

    def drop(seed: int):
        """Re-spawn every PCB high above its trial with fresh jitter, then let it settle."""
        r = np.random.default_rng(seed)
        jx = r.uniform(-jit, jit, args_cli.trials)
        jy = r.uniform(-jit, jit, args_cli.trials)
        jw = np.deg2rad(r.uniform(-args_cli.jitter_yaw_deg, args_cli.jitter_yaw_deg, args_cli.trials))
        st = pcb.data.default_root_state.clone()
        for i in range(args_cli.trials):
            st[i, 0] = (i % 8) * spacing + jx[i]
            st[i, 1] = (i // 8) * spacing + jy[i]
            st[i, 2] = spawn_z
            half = jw[i] / 2.0
            st[i, 3] = float(np.cos(half)); st[i, 4] = 0.0; st[i, 5] = 0.0
            st[i, 6] = float(np.sin(half))
        st[:, 7:] = 0.0                      # zero linear+angular velocity
        pcb.write_root_state_to_sim(st)
        pcb.reset()
        for _ in range(args_cli.settle_steps):
            pcb.write_data_to_sim()
            sim.step()
            pcb.update(sim.get_physics_dt())
        for _ in range(args_cli.hold_steps):  # dwell so the settled pose is visible
            sim.step()
        return jx, jy

    for rep in range(args_cli.repeats):
        dx, dy = drop(rep)
        z = pcb.data.root_pos_w[:, 2].cpu().numpy()
        s = (z - PCB_HALF_T + ENC_HALF_H) * 1000.0
        print(f"  drop {rep + 1}/{args_cli.repeats}: seat median {np.median(s):6.2f} mm  "
              f"[{s.min():6.2f}, {s.max():6.2f}]", flush=True)

    pos = pcb.data.root_pos_w.clone().cpu().numpy()
    quat = pcb.data.root_quat_w.clone().cpu().numpy()
    # height of the PCB's BOTTOM face above the enclosure's bottom face, in mm
    seat = (pos[:, 2] - PCB_HALF_T + ENC_HALF_H) * 1000.0
    # tilt: angle of the body +Z from world +Z
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    zz = 1.0 - 2.0 * (x * x + y * y)
    tilt = np.rad2deg(np.arccos(np.clip(zz, -1.0, 1.0)))
    slide = np.hypot(pos[:, 0] - ((np.arange(args_cli.trials) % 8) * spacing + dx),
                     pos[:, 1] - ((np.arange(args_cli.trials) // 8) * spacing + dy)) * 1000.0

    print("\n=== PCB SEAT (mm above the enclosure's bottom face) ===", flush=True)
    print(f"  trials {args_cli.trials}   jitter +/-{args_cli.jitter_mm} mm, "
          f"+/-{args_cli.jitter_yaw_deg} deg", flush=True)
    print(f"  seat  : median {np.median(seat):7.2f}   min {seat.min():7.2f}   max {seat.max():7.2f}"
          f"   spread {seat.max()-seat.min():.2f}", flush=True)
    print(f"  tilt  : median {np.median(tilt):7.2f} deg   max {tilt.max():7.2f} deg", flush=True)
    print(f"  slide : median {np.median(slide):7.2f} mm    max {slide.max():7.2f} mm", flush=True)
    print("\n  reference: STL corner bosses = 20.05 mm, box-collider ridges = 13.60 mm", flush=True)
    settled = np.abs(seat - np.median(seat)) < 1.0
    print(f"  {int(settled.sum())}/{args_cli.trials} trials landed within 1 mm of the median "
          f"-> {'REPEATABLE' if settled.sum() >= 0.9 * args_cli.trials else 'NOT repeatable'}",
          flush=True)
    sys.stdout.flush()

    if not args_cli.headless:
        print("\n  GUI open -- close the window to exit.", flush=True)
        while simulation_app.is_running():
            sim.step()
    os._exit(0)


if __name__ == "__main__":
    main()
