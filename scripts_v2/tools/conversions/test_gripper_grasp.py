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
    slab0 = slab.data.root_pos_w[0].clone()

    target = gripper.data.default_joint_pos.clone()  # start OPEN (0)
    gravity_on = False
    for step in range(360):
        # close after a short settle
        if step == 20:
            target[0, d] = args.close_value
        gripper.set_joint_position_target(target)
        gripper.write_data_to_sim()
        if step == args.gravity_step and not gravity_on:
            slab.root_physx_view.set_disable_gravities(torch.zeros(1, dtype=torch.bool), torch.arange(1))
            gravity_on = True
        sim.step()
        gripper.update(1 / 120.0)
        slab.update(1 / 120.0)
        if step % 30 == 0 or step in (19, 21, args.gravity_step):
            fj = float(gripper.data.joint_pos[0, d])
            rj = float(gripper.data.joint_pos[0, rfj]) if rfj is not None else float("nan")
            sp = slab.data.root_pos_w[0]
            disp = float((sp - slab0).norm())
            tag = "(closing)" if step >= 20 else "(open)"
            print(f"  step {step:3d} {tag:9s} finger_joint={fj:+.4f} right={rj:+.4f}  slab_disp={disp*1000:5.1f}mm  slab_z={float(sp[2]):.3f}")

    fj = float(gripper.data.joint_pos[0, d])
    disp = float((slab.data.root_pos_w[0] - slab0).norm())
    print("\n==================== VERDICT ====================")
    if fj > args.close_value - 0.003:
        print(f"JAWS MISSED: finger_joint closed fully to {fj:.4f} with no resistance -> slab not between the jaws.")
        print("  -> fix gripper positioning (finger_offset / approach / which axis the slab spans).")
    elif disp < 0.02:
        print(f"GRIPPED: finger_joint stopped at {fj:.4f} (pinching) and slab stayed put (disp {disp*1000:.1f}mm).")
    else:
        print(f"CONTACTED BUT EJECTED: finger_joint stopped at {fj:.4f} but slab moved {disp*1000:.1f}mm.")
        print("  -> jaws hit the slab off-center / too hard, or grip is unstable.")


if __name__ == "__main__":
    main()
    simulation_app.close()
