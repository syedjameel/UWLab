# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apply a PhysX mimic joint to the linear gripper so the two jaws are RIGIDLY coupled.

The OmniReset paper (A.3.3) requires grippers be modeled with mimic joints, not
independent joints: independent joints behave as a compliant spring-damper the policy
exploits unphysically (no sim2real transfer). The URDF importer drops the URDF <mimic>,
so we author a PhysX mimic joint matching the reference 2F-85 USD exactly:
on the passive joint -> physxMimicJoint:<axis>:{gearing, offset} + referenceJoint rel.

Run in-app (server or local leisaac)::

    ./uwlab.sh -p scripts_v2/tools/conversions/add_gripper_mimic.py \
        --usd source/uwlab_assets/uwlab_assets/local/Robots/LinearGripper/linear_gripper.usd \
        --mimic-joint right_finger_joint --driver-joint finger_joint \
        --axis transX --gearing -1.0 --test
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Apply a PhysX mimic joint to a gripper USD.")
parser.add_argument("--usd", type=str, required=True, help="Gripper USD to edit in place.")
parser.add_argument("--mimic-joint", type=str, default="right_finger_joint", help="Passive joint name.")
parser.add_argument("--driver-joint", type=str, default="finger_joint", help="Reference (driver) joint name.")
parser.add_argument("--axis", type=str, default="transX", help="Mimic axis token (e.g. transX, rotZ).")
parser.add_argument("--gearing", type=float, default=-1.0, help="Mimic gearing (sign verified by --test).")
parser.add_argument("--offset", type=float, default=0.0, help="Mimic offset.")
parser.add_argument("--test", action="store_true", help="After authoring, drive the joint in sim to verify coupling.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os

from pxr import PhysxSchema, Usd, UsdPhysics


def _find_joint(stage, name):
    for p in stage.Traverse():
        if p.GetName() == name and "Joint" in str(p.GetTypeName()):
            return p
    raise RuntimeError(f"joint '{name}' not found")


def author_mimic() -> None:
    print("PhysxMimicJointAPI methods:", [m for m in dir(PhysxSchema.PhysxMimicJointAPI) if not m.startswith("_")])
    stage = Usd.Stage.Open(os.path.abspath(args.usd))
    mimic_prim = _find_joint(stage, args.mimic_joint)
    driver_prim = _find_joint(stage, args.driver_joint)

    api = PhysxSchema.PhysxMimicJointAPI.Apply(mimic_prim, args.axis)
    api.CreateGearingAttr(args.gearing)
    api.CreateOffsetAttr(args.offset)
    ref_rel = api.CreateReferenceJointRel()
    ref_rel.SetTargets([driver_prim.GetPath()])
    # referenceJointAxis defaults to the same axis; set explicitly to match the reference USD.
    if hasattr(api, "CreateReferenceJointAxisAttr"):
        api.CreateReferenceJointAxisAttr(args.axis)

    stage.GetRootLayer().Save()
    print(f"Applied mimic on '{args.mimic_joint}' (axis {args.axis}) -> reference '{args.driver_joint}', "
          f"gearing={args.gearing}, offset={args.offset}")
    print("applied schemas now:", list(mimic_prim.GetAppliedSchemas()))
    for a in mimic_prim.GetAttributes():
        if "imic" in a.GetName().lower():
            print("   ", a.GetName(), "=", a.Get())
    for r in mimic_prim.GetRelationships():
        if "imic" in r.GetName().lower():
            print("   ", r.GetName(), "->", [t.pathString for t in r.GetTargets()])


def main() -> None:
    author_mimic()
    if args.test:
        _test_in_sim()


def _test_in_sim() -> None:
    """Spawn the gripper, drive the driver joint, and check both jaws follow rigidly."""
    import torch

    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    import isaaclab.sim as sim_utils

    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device="cpu"))
    # ground + light not needed; gripper has gravity disabled is fine for a coupling check
    cfg = ArticulationCfg(
        prim_path="/World/Gripper",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.abspath(args.usd)),
        actuators={
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[args.driver_joint], stiffness=200.0, damping=20.0
            ),
        },
    )
    robot = Articulation(cfg)
    sim.reset()
    driver_idx = robot.joint_names.index(args.driver_joint)
    mimic_idx = robot.joint_names.index(args.mimic_joint)
    print("joint_names:", robot.joint_names)

    target = torch.zeros((1, robot.num_joints))
    target[0, driver_idx] = 0.05  # close
    for _ in range(240):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(1 / 120.0)
    q = robot.data.joint_pos[0]
    print(f"\nDriven {args.driver_joint}->0.05:  driver={q[driver_idx]:.4f}  mimic={q[mimic_idx]:.4f}")
    print("COUPLING:", "RIGID/symmetric OK" if abs(q[mimic_idx] - 0.05) < 0.01 else
          ("OPPOSITE SIGN (flip --gearing)" if abs(q[mimic_idx] + 0.05) < 0.02 else "NOT COUPLED"))


if __name__ == "__main__":
    main()
    simulation_app.close()
