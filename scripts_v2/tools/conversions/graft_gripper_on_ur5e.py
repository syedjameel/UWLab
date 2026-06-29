# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Graft the linear gripper onto the calibrated UR5e arm (replace the Robotiq 2F-85).

Pure-USD edit (pxr only -- no Isaac app, fast, safe locally):
  1. Open the calibrated UR5e+2F-85 USD.
  2. Strip the 9 2F-85 bodies + 10 2F-85 joints (the mount FixedJoint is nested under
     robotiq_base_link, so it goes too). The 6 arm links/joints + the articulation root
     (root_joint) are kept untouched -> the UR5e sysid/FK calibration is preserved.
  3. Reference the linear gripper, remove its nested ArticulationRootAPI (so it joins the
     arm's single articulation), and place it at the wrist_3 flange.
  4. Add a FixedJoint wrist_3_link -> robotiq_base_link, authored exactly like the 2F-85's
     mount joint (same attribute set) but with our mount transform: identity rotation
     (gripper approach +Z aligned to wrist_3 +Z) and a +standoff offset along wrist_3 +Z.
  5. Flatten + export a self-contained combined USD.

The mount --standoff places the gripper base along wrist_3 +Z; eyeball it in the GUI and
re-run with a different value if the jaws sit too close/far. Run::

    ./uwlab.sh -p scripts_v2/tools/conversions/graft_gripper_on_ur5e.py   # (plain python works too)
"""

from __future__ import annotations

import argparse
import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

_CACHE = os.path.expanduser(
    "~/.cache/uwlab/assets/Robots/UniversalRobots/Ur5e2f85RobotiqGripperCalibrated/"
    "ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd"
)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ROOT = "/ur5e_robotiq_gripper_d415_mount"
F85_BODIES = [
    "robotiq_base_link", "left_outer_knuckle", "left_outer_finger", "left_inner_finger",
    "left_inner_knuckle", "right_outer_knuckle", "right_outer_finger", "right_inner_finger",
    "right_inner_knuckle",
]
F85_JOINTS = [
    "finger_joint", "right_outer_knuckle_joint", "right_inner_finger_joint",
    "right_inner_knuckle_joint", "right_inner_finger_knuckle_joint",
    "left_inner_finger_knuckle_joint", "left_inner_finger_joint", "left_inner_knuckle_joint",
    "left_outer_finger_joint", "right_outer_finger_joint",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Graft the linear gripper onto the UR5e arm.")
    ap.add_argument("--arm-usd", default=_CACHE, help="Calibrated UR5e+2F-85 USD (input).")
    ap.add_argument("--gripper-usd",
                    default=os.path.join(_REPO, "source/uwlab_assets/uwlab_assets/local/Robots/LinearGripper/linear_gripper.usd"))
    ap.add_argument("--output",
                    default=os.path.join(_REPO, "source/uwlab_assets/uwlab_assets/local/Robots/Ur5eLinearGripper/ur5e_linear_gripper.usd"))
    ap.add_argument("--standoff", type=float, default=0.049, help="Mount offset along wrist_3 +Z (m).")
    args = ap.parse_args()

    if not os.path.exists(args.arm_usd):
        raise SystemExit(f"arm USD not found: {args.arm_usd}\nDownload the calibrated USD from the cloud first.")
    if not os.path.exists(args.gripper_usd):
        raise SystemExit(f"gripper USD not found: {args.gripper_usd}\nRun convert_gripper_urdf.py + add_gripper_mimic.py first.")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    stage = Usd.Stage.Open(args.arm_usd)

    # 1) strip the 2F-85
    for name in F85_BODIES + F85_JOINTS:
        p = f"{ROOT}/{name}"
        if stage.GetPrimAtPath(p):
            stage.RemovePrim(p)

    # 2) reference the gripper under a new prim and place it at the flange.
    gpath = f"{ROOT}/gripper"
    gprim = stage.DefinePrim(gpath, "Xform")
    gprim.GetReferences().AddReference(os.path.abspath(args.gripper_usd))  # references its defaultPrim

    # world pose of the gripper = wrist_3 world * mount(translate +standoff along wrist_3 +Z).
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    wrist = stage.GetPrimAtPath(f"{ROOT}/wrist_3_link")
    T_wrist_w = xc.GetLocalToWorldTransform(wrist)
    T_mount = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, 0.0, args.standoff))
    T_grip_w = T_mount * T_wrist_w  # pre-multiply: mount is in wrist frame
    # gripper prim local = (root world)^-1 * gripper world
    T_root_w = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(ROOT))
    T_grip_local = T_grip_w * T_root_w.GetInverse()
    UsdGeom.Xformable(gprim).MakeMatrixXform().Set(T_grip_local)

    # 3) the referenced gripper's robotiq_base_link carries ArticulationRootAPI; remove it so
    #    the gripper joins the arm's single articulation (root stays root_joint).
    rb = stage.GetPrimAtPath(f"{gpath}/robotiq_base_link")
    if rb.HasAPI(UsdPhysics.ArticulationRootAPI):
        rb.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    # 4) mount FixedJoint wrist_3 -> robotiq_base_link, authored like the 2F-85's.
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

    # 5) flatten (inlines the gripper + its meshes) and export a self-contained USD.
    flat = stage.Flatten()
    flat.Export(args.output)
    print(f"Wrote {args.output}")
    print(f"  standoff along wrist_3 +Z = {args.standoff} m (identity rotation; approach +Z = wrist_3 +Z)")


if __name__ == "__main__":
    main()
