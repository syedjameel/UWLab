# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime test: drive finger_joint CLOSED via the actuator; does the mimic pull the other jaw?

Grasp sampling drives ONLY finger_joint (binary action); the PhysX mimic is supposed to make
right_finger_joint follow. If the mimic isn't coupling at runtime, only one jaw closes and no
object is ever gripped -> 0% grasp success (the sampler then loops forever). This spawns the
gripper, commands the actuator to close, steps physics, and prints both jaw positions.

Run on a machine with stable Isaac physics (the A100 server)::

    ./uwlab.sh -p scripts_v2/tools/conversions/test_gripper_mimic.py \
        --usd source/uwlab_assets/uwlab_assets/local/Robots/LinearGripper/linear_gripper.usd
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Runtime mimic-coupling test for the linear gripper.")
parser.add_argument("--usd", type=str, required=True)
parser.add_argument("--driver-joint", type=str, default="finger_joint")
parser.add_argument("--mimic-joint", type=str, default="right_finger_joint")
parser.add_argument("--close-value", type=float, default=0.068)
parser.add_argument("--stiffness", type=float, default=500.0)
parser.add_argument("--damping", type=float, default=50.0)
parser.add_argument("--effort", type=float, default=120.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device="cpu"))
    cfg = ArticulationCfg(
        prim_path="/World/Gripper",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.abspath(args.usd)),
        actuators={
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[args.driver_joint],
                stiffness=args.stiffness,
                damping=args.damping,
                effort_limit_sim=args.effort,
            )
        },
    )
    robot = Articulation(cfg)
    sim.reset()

    d = robot.joint_names.index(args.driver_joint)
    m = robot.joint_names.index(args.mimic_joint)
    print("joint_names:", robot.joint_names)

    # Command the actuator to CLOSE only the driver joint (this is what the gripper action does).
    target = robot.data.default_joint_pos.clone()
    target[0, d] = args.close_value
    print(f"\nCommanding {args.driver_joint} -> {args.close_value} via actuator (mimic should pull {args.mimic_joint}):")
    for step in range(300):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(1 / 120.0)
        if step % 30 == 0 or step == 299:
            q = robot.data.joint_pos[0]
            print(f"  step {step:3d}:  {args.driver_joint}={q[d]:+.4f}   {args.mimic_joint}={q[m]:+.4f}")

    q = robot.data.joint_pos[0]
    drv, mim = float(q[d]), float(q[m])
    print("\n==================== VERDICT ====================")
    if abs(mim - drv) < 0.01:
        print(f"MIMIC OK: both jaws closed together ({args.mimic_joint} followed {args.driver_joint}).")
    elif abs(mim) < 0.01:
        print(f"MIMIC NOT WORKING: {args.mimic_joint} stayed at 0 while driver moved to {drv:.4f}.")
        print("  -> only one jaw closes; nothing gets gripped -> 0% grasp success.")
    elif (mim < 0) != (drv < 0) or abs(mim + drv) < 0.01:
        print(f"MIMIC INVERTED: {args.mimic_joint}={mim:.4f} moved opposite the driver ({drv:.4f}).")
        print("  -> jaws splay apart; flip --gearing in add_gripper_mimic (-1.0 <-> +1.0).")
    else:
        print(f"PARTIAL: driver={drv:.4f} mimic={mim:.4f} -- coupling weak/soft.")


if __name__ == "__main__":
    main()
    simulation_app.close()
