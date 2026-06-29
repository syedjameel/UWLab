# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the linear gripper's grasp geometry for its metadata.yaml and flange mount.

The contact pad of the L-shaped finger cannot be read from mesh bounding boxes, so we
measure from the *articulated* gripper: set the driver joint to OPEN and CLOSED via
forward kinematics (no long physics loop), read the finger contact-body world poses and
their collision-mesh extents, and report:

* maximum_aperture        - jaw gap at OPEN (max grip width)
* finger_offset           - base-origin -> contact-pad center along the approach axis
* gripper_approach_direction (sanity: which base axis the fingers extend along)
* the base->contact distance the flange mount uses to match the 2F-85 grasp-point location

Run on a machine with stable Isaac physics (the A100 server; physics stepping hangs on the
laptop 3060)::

    ./uwlab.sh -p scripts_v2/tools/conversions/measure_gripper.py \
        --usd source/uwlab_assets/uwlab_assets/local/Robots/LinearGripper/linear_gripper.usd
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure linear gripper grasp geometry.")
parser.add_argument("--usd", type=str, required=True, help="Gripper-only USD (with mimic).")
parser.add_argument("--driver-joint", type=str, default="finger_joint")
parser.add_argument("--open-value", type=float, default=0.0, help="Driver joint value at OPEN.")
parser.add_argument("--close-value", type=float, default=0.068, help="Driver joint value at CLOSED.")
parser.add_argument("--left-finger", type=str, default="left_inner_finger")
parser.add_argument("--right-finger", type=str, default="right_inner_finger")
parser.add_argument("--base", type=str, default="robotiq_base_link")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext


def body_world_pos(robot: Articulation, name: str) -> np.ndarray:
    idx = robot.body_names.index(name)
    return robot.data.body_link_pos_w[0, idx].cpu().numpy()


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device="cpu"))
    cfg = ArticulationCfg(
        prim_path="/World/Gripper",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.abspath(args.usd)),
        actuators={"g": ImplicitActuatorCfg(joint_names_expr=[args.driver_joint], stiffness=200.0, damping=20.0)},
    )
    robot = Articulation(cfg)
    sim.reset()

    didx = robot.joint_names.index(args.driver_joint)
    print("joint_names:", robot.joint_names)
    print("body_names :", robot.body_names)

    results = {}
    for label, val in [("OPEN", args.open_value), ("CLOSED", args.close_value)]:
        q = robot.data.default_joint_pos.clone()
        q[0, didx] = val
        # forward-kinematics only: write joint state and let the kinematics update (a couple of steps,
        # NOT a long control loop -- this avoids the hang seen on weak GPUs).
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        for _ in range(3):
            sim.step()
            robot.update(1 / 120.0)
        base = body_world_pos(robot, args.base)
        lf = body_world_pos(robot, args.left_finger)
        rf = body_world_pos(robot, args.right_finger)
        gap = float(np.linalg.norm(lf - rf))
        results[label] = dict(base=base, lf=lf, rf=rf, gap=gap)
        print(f"\n[{label}] driver={val:.4f}")
        print(f"  base   world = {base.round(4)}")
        print(f"  L finger     = {lf.round(4)}   R finger = {rf.round(4)}")
        print(f"  finger-center gap = {gap*1000:.1f} mm   along axis = {(rf-lf).round(4)}")

    # Approach axis = base -> midpoint of fingers (the direction the gripper reaches).
    mid = 0.5 * (results["OPEN"]["lf"] + results["OPEN"]["rf"])
    approach = mid - results["OPEN"]["base"]
    finger_offset = float(np.linalg.norm(approach))
    print("\n================= METADATA =================")
    print(f"maximum_aperture (OPEN finger-center gap) = {results['OPEN']['gap']:.4f} m")
    print(f"finger_offset (base->finger-center)       = {finger_offset:.4f} m")
    print(f"approach (base->finger-center, normalized)= {(approach/finger_offset).round(3).tolist()}")
    print("NOTE: aperture above is finger-CENTER gap; subtract finger pad thickness for usable width.")
    print("Use these to fill metadata.yaml and the flange mount (place base so finger-center")
    print("matches the 2F-85 grasp point ~0.146 m along wrist_3 +Z).")


if __name__ == "__main__":
    main()
    simulation_app.close()
