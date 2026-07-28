# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the v2 jig's INTERIOR BLOCKER collider is truly massless.

The jig authors no MassAPI on its root, so PhysX auto-computes the body mass from the
collider volume (see omnireset_asset_utils.create_stage). The v2 interior blocker adds 283%
more collider volume, which -- if its per-shape density were not honoured -- would silently
~4x the jig mass and invalidate every dynamics/contact result against v1.

This spawns BOTH jigs through the REAL ``make_insertive_object`` spawn config (same
rigid_props / mass_props=0.001 path the task uses) and compares the runtime PhysX mass,
inertia and centre of mass. v1 and v2 must match to within float noise.

    ./uwlab.sh -p scripts_v2/tools/conversions/verify_jig_v2_mass.py --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--tol", type=float, default=1e-4, help="relative tolerance for the match")
parser.add_argument("--out", type=str, default="/tmp/jig_v2_mass_report.json",
                    help="where to write the JSON report (written BEFORE teardown)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.reset_states_cfg import (  # noqa: E501
    variants,
)

# Pull the REAL variant configs, so this checks exactly what training spawns (including the
# jigv2 override_mass=False that keeps the nested spawn override off the blocker prims).
_CFGS = {tag: variants["scene.insertive_object"][name] for tag, name in (("v1", "jig"), ("v2", "jigv2"))}
_JIGS = {tag: cfg.spawn.usd_path for tag, cfg in _CFGS.items()}


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg())

    objs = {}
    for i, (tag, src) in enumerate(_CFGS.items()):
        cfg = copy.deepcopy(src)
        cfg.prim_path = f"/World/Jig_{tag}"
        cfg.init_state.pos = (0.5 * i, 0.0, 0.5)
        print(f"[spawn] {tag}: mass_props={cfg.spawn.mass_props}  usd={cfg.spawn.usd_path}", flush=True)
        objs[tag] = RigidObject(cfg)

    sim.reset()

    stats = {}
    report = {}
    for tag, obj in objs.items():
        view = obj.root_physx_view
        mass = view.get_masses().clone()[0]
        inertia = view.get_inertias().clone()[0]
        com = view.get_coms().clone()[0]
        stats[tag] = (mass, inertia, com)
        report[tag] = {
            "usd": _JIGS[tag],
            "mass_kg": float(mass),
            "inertia": [float(x) for x in inertia.flatten()[:9]],
            "com": [float(x) for x in com[:3]],
        }
        print(f"\n=== jig {tag}  ({_JIGS[tag].split('/')[-1]})", flush=True)
        print(f"  mass    = {float(mass):.9f} kg", flush=True)
        print(f"  inertia = {[round(float(x), 12) for x in inertia.flatten()[:9]]}", flush=True)
        print(f"  com     = {[round(float(x), 9) for x in com[:3]]}", flush=True)

    m1, i1, c1 = stats["v1"]
    m2, i2, c2 = stats["v2"]
    dm = abs(float(m2 - m1)) / max(float(m1), 1e-12)
    di = float((i2 - i1).abs().max()) / max(float(i1.abs().max()), 1e-12)
    dc = float((c2[:3] - c1[:3]).abs().max())
    ok = dm <= args_cli.tol and di <= args_cli.tol and dc <= 1e-6
    report["diff"] = {"rel_mass": dm, "rel_inertia": di, "abs_com_m": dc, "pass": ok}

    print("\n=== v2 vs v1", flush=True)
    print(f"  relative mass    diff = {dm:.3e}", flush=True)
    print(f"  relative inertia diff = {di:.3e}", flush=True)
    print(f"  absolute com     diff = {dc:.3e} m", flush=True)
    print(f"\n  {'PASS -- interior blocker is massless' if ok else 'FAIL -- interior blocker CHANGES the jig dynamics'}", flush=True)
    if not ok:
        print("  -> the per-shape MassAPI density is NOT being honoured; the blocker must be", flush=True)
        print("     neutralised another way before ANY v2 training run.", flush=True)

    # Persist BEFORE teardown: Isaac's simulation_app.close() can hang indefinitely on this
    # box, and a hang would strand the numbers in the block-buffered stdout (measured).
    with open(args_cli.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  wrote {args_cli.out}", flush=True)
    sys.stdout.flush()

    # os._exit skips the (hanging) Kit teardown -- this is a read-only diagnostic, there is
    # nothing to flush but the report we just wrote.
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
