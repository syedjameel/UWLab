# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Controlled grasp test: put a slab between the jaws, close, and see if it's held.

Grasp sampling shows the object ejected/dropped before gravity. This isolates the grip:
spawn the gripper + a box at finger_offset on the gripper's +Z axis (between the jaws),
close the gripper, and report:
  * where finger_joint stops (0.068 = jaws closed all the way -> they MISSED the slab;
    < 0.068 = they're pinching it),
  * whether the slab stays put (held) or moves/falls (ejected/dropped),
  * the min gripper->slab gap.

Run on the A100 (stable physics)::

    ./uwlab.sh -p scripts_v2/tools/conversions/test_gripper_grasp.py \
        --usd source/uwlab_assets/uwlab_assets/local/Robots/LinearGripper/linear_gripper.usd \
        --finger-offset 0.13 --slab-x 0.040 --slab-z 0.020
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Controlled grasp test for the linear gripper.")
parser.add_argument("--usd", type=str, required=True)
parser.add_argument("--finger-offset", type=float, default=0.13, help="Slab center along gripper +Z from base.")
parser.add_argument("--slab-x", type=float, default=0.040, help="Slab size along X (closing axis).")
parser.add_argument("--slab-y", type=float, default=0.040)
parser.add_argument("--slab-z", type=float, default=0.020)
parser.add_argument("--close-value", type=float, default=0.068)
parser.add_argument("--stiffness", type=float, default=500.0)
parser.add_argument("--gravity-step", type=int, default=150, help="Step at which to enable gravity on the slab.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device="cpu", gravity=(0.0, 0.0, -9.81)))

    gripper = Articulation(
        ArticulationCfg(
            prim_path="/World/Gripper",
            spawn=sim_utils.UsdFileCfg(usd_path=os.path.abspath(args.usd)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.5)),  # lift base so jaws are in the air
            actuators={"g": ImplicitActuatorCfg(joint_names_expr=["finger_joint"], stiffness=args.stiffness, damping=50.0, effort_limit_sim=120.0)},
        )
    )
    # A box slab as the object, placed on the gripper's +Z axis, between the jaws.
    slab = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Slab",
            spawn=sim_utils.CuboidCfg(
                size=(args.slab_x, args.slab_y, args.slab_z),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5 + args.finger_offset)),
        )
    )
    sim.reset()

    d = gripper.joint_names.index("finger_joint")
    rfj = gripper.joint_names.index("right_finger_joint") if "right_finger_joint" in gripper.joint_names else None
    lf = gripper.body_names.index("left_inner_finger")
    rf = gripper.body_names.index("right_inner_finger")
    base = gripper.body_names.index("robotiq_base_link")

    # Report where the finger BODIES sit relative to the base when fully OPEN (joint=0), so we
    # see the real jaw geometry (closing axis + height range).
    sim.step(); gripper.update(1 / 120.0)
    bz = float(gripper.data.body_link_pos_w[0, base, 2])
    print("\n=== gripper geometry (OPEN, base-relative, m) ===")
    for name, idx in [("left_inner_finger", lf), ("right_inner_finger", rf)]:
        p = gripper.data.body_link_pos_w[0, idx] - gripper.data.body_link_pos_w[0, base]
        print(f"  {name:18s} body origin (base frame) = ({float(p[0]):+.4f}, {float(p[1]):+.4f}, {float(p[2]):+.4f})")

    # Sweep the slab height (finger_offset) and, at each, close the jaws and see where they stop.
    offsets = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14]
    print("\n=== sweep: slab at z=base+offset, close jaws, where does finger_joint stop? ===")
    print("  offset   finger_joint_stop   slab_disp   verdict")
    for off in offsets:
        # reset gripper OPEN and slab at this height
        qopen = torch.zeros_like(gripper.data.joint_pos)
        gripper.write_joint_state_to_sim(qopen, torch.zeros_like(qopen))
        sp = torch.tensor([[0.0, 0.0, bz + off, 1.0, 0.0, 0.0, 0.0]], device=slab.device)
        slab.write_root_pose_to_sim(sp)
        slab.write_root_velocity_to_sim(torch.zeros((1, 6), device=slab.device))
        for _ in range(5):
            sim.step(); gripper.update(1 / 120.0); slab.update(1 / 120.0)
        slab_start = slab.data.root_pos_w[0].clone()
        target = torch.zeros_like(gripper.data.joint_pos)
        target[0, d] = args.close_value
        for _ in range(150):
            gripper.set_joint_position_target(target); gripper.write_data_to_sim()
            sim.step(); gripper.update(1 / 120.0); slab.update(1 / 120.0)
        fj = float(gripper.data.joint_pos[0, d])
        disp = float((slab.data.root_pos_w[0] - slab_start).norm())
        if fj > args.close_value - 0.004:
            v = "MISSED (closed fully)"
        elif disp < 0.02:
            v = "GRIPPED (held)"
        else:
            v = f"contacted but slab moved {disp*1000:.0f}mm"
        print(f"  {off:.3f}     {fj:.4f}             {disp*1000:5.1f}mm     {v}")
    print("\nWhere finger_joint stops SHORT and slab is held = the real grip height -> set finger_offset there.")


if __name__ == "__main__":
    main()
    simulation_app.close()
