# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Graft the linear gripper onto the UR10e arm (sibling of graft_gripper_on_ur5e.py).

Same pure-USD graft (pxr only -- no Isaac app), but onto our LOCAL UR10e arm USD instead of
the cloud UR5e+2F-85 asset. Differences vs the UR5e graft:
  * arm USD is the local ``UR10e/ur10e.usd`` (converted from the URDF), root prim ``/ur10e``;
  * the UR10e arm is BARE (no gripper), so there is nothing to strip -- we only add the gripper;
  * the 6 arm links/joints + the articulation root (``root_joint``) are left untouched, so the
    UR10e kinematics/inertials (which the analytical OSC reads from UR10e/metadata.yaml) are
    preserved.

Steps:
  1. Open UR10e/ur10e.usd.
  2. Reference the linear gripper under ``/ur10e/gripper`` and place it at the wrist_3 flange
     (identity rotation: gripper approach +Z aligned to wrist_3 +Z; +standoff along wrist_3 +Z).
  3. Remove the gripper's nested ArticulationRootAPI so it joins the arm's single articulation.
  4. Dual-drive the follower jaw (re-activate right_finger_joint's linear DriveAPI; strip the inert
     prismatic mimic post-flatten) -- identical rationale to the UR5e full-robot graft.
  5. Add a FixedJoint wrist_3_link -> robotiq_base_link.
  6. Flatten + export a self-contained combined USD.

The mount --standoff places the gripper base along wrist_3 +Z; eyeball it in the GUI and re-run
with a different value if the jaws sit too close/far. Run::

    ./uwlab.sh -p scripts_v2/tools/conversions/graft_gripper_on_ur10e.py   # (plain python works too)
"""

from __future__ import annotations

import argparse
import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ROOT = "/ur10e"


def main() -> None:
    ap = argparse.ArgumentParser(description="Graft the linear gripper onto the UR10e arm.")
    ap.add_argument("--arm-usd",
                    default=os.path.join(_REPO, "source/uwlab_assets/uwlab_assets/local/Robots/UR10e/ur10e.usd"),
                    help="UR10e arm USD (input), converted from the URDF via convert_gripper_urdf.py --fix-base.")
    ap.add_argument("--gripper-usd",
                    default=os.path.join(_REPO, "source/uwlab_assets/uwlab_assets/local/Robots/LinearGripper/linear_gripper.usd"))
    ap.add_argument("--output",
                    default=os.path.join(_REPO, "source/uwlab_assets/uwlab_assets/local/Robots/Ur10eLinearGripper/ur10e_linear_gripper.usd"))
    ap.add_argument("--standoff", type=float, default=0.049, help="Mount offset along wrist_3 +Z (m).")
    args = ap.parse_args()

    if not os.path.exists(args.arm_usd):
        raise SystemExit(f"arm USD not found: {args.arm_usd}\nRun convert_gripper_urdf.py --fix-base on UR10e/ur10e.urdf first.")
    if not os.path.exists(args.gripper_usd):
        raise SystemExit(f"gripper USD not found: {args.gripper_usd}\nRun convert_gripper_urdf.py + add_gripper_mimic.py first.")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    stage = Usd.Stage.Open(args.arm_usd)

    # 1) reference the gripper under a new prim and place it at the flange.
    gpath = f"{ROOT}/gripper"
    gprim = stage.DefinePrim(gpath, "Xform")
    gprim.GetReferences().AddReference(os.path.abspath(args.gripper_usd))  # references its defaultPrim

    # world pose of the gripper = wrist_3 world * mount(translate +standoff along wrist_3 +Z).
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    wrist = stage.GetPrimAtPath(f"{ROOT}/wrist_3_link")
    if not wrist or not wrist.IsValid():
        raise SystemExit(f"wrist_3_link not found under {ROOT}; is this a UR10e arm USD?")
    T_wrist_w = xc.GetLocalToWorldTransform(wrist)
    T_mount = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, 0.0, args.standoff))
    T_grip_w = T_mount * T_wrist_w  # pre-multiply: mount is in wrist frame
    T_root_w = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(ROOT))
    T_grip_local = T_grip_w * T_root_w.GetInverse()
    UsdGeom.Xformable(gprim).MakeMatrixXform().Set(T_grip_local)

    # 2) the referenced gripper's robotiq_base_link carries ArticulationRootAPI; remove it so the
    #    gripper joins the arm's single articulation (root stays /ur10e/root_joint).
    rb = stage.GetPrimAtPath(f"{gpath}/robotiq_base_link")
    if rb.HasAPI(UsdPhysics.ArticulationRootAPI):
        rb.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    # 3) dual-drive the follower (the prismatic mimic is inert once embedded in the arm articulation
    #    and blocks the actuator; drive both jaws instead -- see graft_gripper_on_ur5e.py step 3b).
    rfj = stage.GetPrimAtPath(f"{gpath}/joints/right_finger_joint")
    UsdPhysics.DriveAPI.Apply(rfj, "linear")  # re-activate the orphaned linear drive
    print("  full-robot follower: re-activated linear DriveAPI (dual-drive)")

    # 4) mount FixedJoint wrist_3 -> robotiq_base_link.
    fj = UsdPhysics.FixedJoint.Define(stage, f"{gpath}/robotiq_base_link/MountJoint")
    fp = fj.GetPrim()
    fj.CreateBody0Rel().SetTargets([Sdf.Path(f"{ROOT}/wrist_3_link")])
    fj.CreateBody1Rel().SetTargets([Sdf.Path(f"{gpath}/robotiq_base_link")])
    fp.CreateAttribute("physics:localPos0", Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(0.0, 0.0, args.standoff))
    fp.CreateAttribute("physics:localRot0", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(1, 0, 0, 0))
    fp.CreateAttribute("physics:localPos1", Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fp.CreateAttribute("physics:localRot1", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(1, 0, 0, 0))
    fp.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    fp.CreateAttribute("physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool).Set(False)
    fp.CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    fp.CreateAttribute("physics:breakForce", Sdf.ValueTypeNames.Float).Set(float("inf"))
    fp.CreateAttribute("physics:breakTorque", Sdf.ValueTypeNames.Float).Set(float("inf"))

    # 4b) give the URDF importer's massless frame links a small mass. The importer leaves the
    #     URDF's pure-frame links (base, base_link, flange, tool0 -- no inertial) at mass 0.0;
    #     zero-mass links in a PhysX articulation are ill-conditioned (the calibrated UR5e cloud
    #     asset has no such links -- they were merged away). Asset hygiene: 0.01 kg is negligible
    #     against the 28.7 kg arm and keeps the mass matrix well-posed.
    for name in ("base", "base_link", "flange", "tool0"):
        prim = stage.GetPrimAtPath(f"{ROOT}/{name}")
        if prim and prim.IsValid():
            api = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) else UsdPhysics.MassAPI.Apply(prim)
            if not api.GetMassAttr().Get():
                api.GetMassAttr().Set(0.01)
                print(f"  frame link {name}: mass 0.0 -> 0.01 kg")

    # 5) flatten (inlines the gripper + its meshes) and export a self-contained USD.
    flat = stage.Flatten()
    frfj = flat.GetPrimAtPath(f"{gpath}/joints/right_finger_joint")  # Sdf.PrimSpec (Flatten -> Sdf.Layer)
    stripped = [n for n in list(frfj.properties.keys()) if "physxMimicJoint" in n]
    for name in stripped:
        del frfj.properties[name]
    print(f"  full-robot follower: stripped {len(stripped)} inert mimic prop(s) post-flatten")
    flat.Export(args.output)
    print(f"Wrote {args.output}")
    print(f"  standoff along wrist_3 +Z = {args.standoff} m (identity rotation; approach +Z = wrist_3 +Z)")


if __name__ == "__main__":
    main()
