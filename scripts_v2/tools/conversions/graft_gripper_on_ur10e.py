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
    ap.add_argument("--gripper-mass", type=float, default=0.575,
                    help="Total gripper mass (kg) to scale the grafted gripper links to. The URDF "
                         "masses total ~1.1 kg but the REAL assembly weighs 0.575 kg (measured "
                         "2026-07-03, matches the UR pendant payload) -- the phantom extra ~0.5 kg "
                         "at the wrist skews the sysid fit (wrist_1 armature pinned at 0) and every "
                         "dynamic the policy feels. Pass 0 to keep the URDF masses.")
    ap.add_argument("--wrist-limit", type=float, default=180.0,
                    help="Wrist joint limit (deg, symmetric). Default 180 (paper A.3.1 sim2real "
                         "hardening vs the URDF's 360). Pass 0 to keep the URDF limits.")
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

    # 4a) scale the gripper link masses to the real measured total (see --gripper-mass help).
    #     Inertia tensors scale with mass for a rigid body of fixed geometry, so scale them too.
    if args.gripper_mass > 0:
        glinks = ["robotiq_base_link", "left_inner_finger", "right_inner_finger"]
        masses = {}
        for name in glinks:
            prim = stage.GetPrimAtPath(f"{gpath}/{name}")
            api = UsdPhysics.MassAPI(prim)
            masses[name] = api.GetMassAttr().Get() or 0.0
        total = sum(masses.values())
        if total > 0:
            scale = args.gripper_mass / total
            for name in glinks:
                prim = stage.GetPrimAtPath(f"{gpath}/{name}")
                api = UsdPhysics.MassAPI(prim)
                api.GetMassAttr().Set(masses[name] * scale)
                inertia = api.GetDiagonalInertiaAttr().Get()
                if inertia:
                    api.GetDiagonalInertiaAttr().Set(inertia * scale)
            print(f"  gripper mass: {total:.3f} kg (URDF) -> {args.gripper_mass:.3f} kg (real), scale {scale:.4f}")

    # 4a2) reduce the wrist joint limits +/-360 -> +/-180 deg (paper A.3.1 sim2real hardening,
    #      NOT present in the released assets): prevents the policy exploiting extreme wrist
    #      rotations that would hit real joint limits / trigger safety stops on hardware.
    #      Reset states recorded with |wrist q| > 180 deg become invalid -> re-record after
    #      changing this. Pass --wrist-limit 0 to keep the URDF's +/-360.
    if args.wrist_limit > 0:
        for name in ("wrist_1_joint", "wrist_2_joint", "wrist_3_joint"):
            joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"{ROOT}/joints/{name}"))
            joint.GetLowerLimitAttr().Set(-args.wrist_limit)
            joint.GetUpperLimitAttr().Set(args.wrist_limit)
        print(f"  wrist joint limits -> +/-{args.wrist_limit:.0f} deg (was +/-360)")

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

    # 5b) de-instance the gripper VISUAL prims: the URDF importer marks them instanceable,
    #     but instance proxies are read-only -- the RGB collection's per-episode gripper
    #     appearance DR (the authors' randomize_wrist_mount/inner_finger_appearance
    #     equivalents) cannot bind per-env materials to them. Visual-only change: rendering
    #     memory grows by 3 small meshes per env; physics/collisions untouched.
    deinstanced = 0
    for link in ("robotiq_base_link", "left_inner_finger", "right_inner_finger"):
        spec = flat.GetPrimAtPath(f"{gpath}/{link}/visuals")
        if spec is not None and spec.instanceable:
            spec.instanceable = False
            deinstanced += 1
    print(f"  gripper visuals: de-instanced {deinstanced} prim(s) (per-env appearance DR)")
    if deinstanced == 0:
        print("  WARNING: no gripper visuals were de-instanced -- if they are also not plain "
              "prims, the RGB collection's gripper-appearance DR will fail with 'No prims "
              "found matching ... visuals/.*'.")
    flat.Export(args.output)
    print(f"Wrote {args.output}")
    print(f"  standoff along wrist_3 +Z = {args.standoff} m (identity rotation; approach +Z = wrist_3 +Z)")


if __name__ == "__main__":
    main()
