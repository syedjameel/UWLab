# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Graft the Tesollo DELTO DG-5F hand onto the calibrated UR5e arm.

This is the UR5e sibling of ``graft_gripper_on_ur10e.py`` and the DELTO sibling of
``graft_gripper_on_ur5e.py``. It takes the arm half from the latter and the hand half from the
former:

  * ARM half (from ``graft_gripper_on_ur5e.py``, unchanged): the input is the CLOUD
    ``ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd``, resolved/downloaded through
    ``resolve_cloud_path``. That asset -- not a bare UR5e URDF conversion -- is what carries the
    UR5e sysid and FK calibration, so it is stripped of its 2F-85 rather than replaced. The 9
    2F-85 bodies + 10 2F-85 joints go (the 2F-85's own mount FixedJoint is nested under
    ``robotiq_base_link``, so it goes with the body); the 6 arm links/joints and the articulation
    root ``root_joint`` are left untouched.
  * HAND half (from ``graft_gripper_on_ur10e.py``): the hand's base link is ``rl_dg_mount``, which
    bolts directly to the UR flange, so the mount standoff resolves to 0 rather than the linear
    gripper's 49 mm adapter plate. The hand's own free root joint is stripped, its nested
    ``ArticulationRootAPI`` is removed so it joins the arm's single articulation, its massless
    ``rl_dg_*_tip`` frame bodies are seeded with a small mass, and the subtree is re-placed so that
    ``rl_dg_mount`` -- not the referenced subtree's root prim -- lands exactly on the flange.

The referenced hand is nested at ``{ROOT}/gripper``. That name is NOT cosmetic: dexlift's
``HAND_PRIM`` constant is literally ``"gripper"`` and the fingertip ``ContactSensorCfg`` prim paths
address it. Renaming it breaks those sensors SILENTLY -- they resolve to nothing and simply never
fire -- so the prim name is fixed here rather than exposed as a flag.

WHAT IS DELIBERATELY *NOT* CARRIED OVER FROM THE TWO SOURCE SCRIPTS

  * The dual-drive / mimic-strip block (``graft_gripper_on_ur5e.py`` step 3b and its UR10e
    ``--dual-drive-joint`` / ``--strip-mimic-joint`` flags). That block exists because the linear
    gripper's PRISMATIC PhysX mimic goes inert once embedded in the arm articulation while still
    capturing the follower joint's control. The DELTO has 20 independently actuated revolute joints
    and no mimic at all, so re-activating a "follower" drive and deleting "inert" mimic properties
    would be operating on machinery that does not exist here.
  * The ``--gripper-mass`` rescale. The linear gripper is rescaled from its ~1.1 kg URDF total to
    the 0.575 kg measured assembly. The DELTO's masses are AUTHORED by ``prepare_delto_hand.py``
    from the vendor model and are verified below against that authored total; there is no
    independent measurement here to rescale toward.
  * The UR10e graft's wrist-limit clamp and its ``base``/``base_link``/``flange``/``tool0``
    zero-mass hygiene fixes. Both are properties of the locally URDF-converted UR10e. The
    calibrated UR5e cloud asset has no such massless frame links (they were merged away), and its
    joint limits are part of the calibration this script exists to preserve.

Pure-USD edit (``pxr`` only -- no Isaac app, no GPU). It self-verifies the written asset and exits
non-zero on any mismatch. Run::

    python scripts_v2/tools/conversions/graft_delto_on_ur5e.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR, resolve_cloud_path

# The calibrated UR5e+2F-85 arm USD is a CLOUD asset. resolve_cloud_path downloads it once to
# ~/.cache/uwlab/assets/... and returns the local path (so this works on a fresh cache, not only
# where the 2F-85 tasks were already run). Same URL the 2F-85 robot cfg spawns from.
_ARM_USD_URL = (
    f"{UWLAB_CLOUD_ASSETS_DIR}/Robots/UniversalRobots/Ur5e2f85RobotiqGripperCalibrated/"
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

# The prim the hand is referenced under. Fixed, not a flag -- see the module docstring.
HAND_PRIM = "gripper"

# The 6 UR5e revolute DOFs that must survive the strip untouched.
_ARM_JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
_FINGERS = (1, 2, 3, 4, 5)
# The 20 actuated DELTO revolutes: four phalanx joints on each of five fingers.
_DELTO_JOINT_RE = re.compile(r"^rj_dg_[1-5]_[1-4]$")
_DELTO_JOINTS = tuple(f"rj_dg_{finger}_{phalanx}" for finger in _FINGERS for phalanx in (1, 2, 3, 4))
_DELTO_TIP_LINKS = tuple(f"rl_dg_{finger}_tip" for finger in _FINGERS)
# Every rigid body of the prepared hand, in the order prepare_delto_hand.py authors them.
_DELTO_LINKS = ("rl_dg_mount", "rl_dg_base", "rl_dg_palm") + tuple(
    f"rl_dg_{finger}_{part}" for finger in _FINGERS for part in ("1", "2", "3", "4", "tip")
)

# The AUTHORED total explicit mass of the prepared DELTO hand (prepare_delto_hand.py's
# _EXPECTED_TOTAL_MASS). The DG-5F spec sheet says 1.4 kg; the vendor USD's per-link inertials sum
# to 1.7735 kg. That discrepancy is KNOWN and RECORDED -- it is not silently rescaled away here,
# because the number the dynamics actually see must stay traceable to the asset it came from.
_EXPECTED_HAND_MASS = 1.7735
_MASS_TOL = 1e-4


def _csv(value: str) -> list[str]:
    """Parse a comma-separated flag value into a list, treating "" / "none" as empty."""
    if not value or value.strip().lower() == "none":
        return []
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def verify(output: str, base_link: str) -> list[str]:
    """Re-open the written asset and check every property the consumers depend on."""
    failures: list[str] = []
    stage = Usd.Stage.Open(output)
    if not stage:
        return [f"cannot re-open the written USD: {output}"]
    # Instance proxies must be expanded or instanced subtrees read as empty -- the exact blind spot
    # that hides missing tip geometry (see prepare_delto_hand.py).
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    prims = list(Usd.PrimRange.Stage(stage, predicate))
    gpath = f"{ROOT}/{HAND_PRIM}"

    # 1) the hand must be nested under the exact prim name dexlift's contact sensors address.
    if not stage.GetPrimAtPath(gpath):
        failures.append(f"hand prim {gpath} missing -- dexlift's HAND_PRIM/ContactSensorCfg paths "
                        f"would resolve to nothing and the fingertip sensors would never fire")

    # 2) 26 actuated joints: the 6 UR5e revolutes + the DELTO's 20 rj_dg_[1-5]_[1-4].
    revolutes = sorted(p.GetName() for p in prims if p.GetTypeName() == "PhysicsRevoluteJoint")
    expected_revolutes = sorted(_ARM_JOINTS + _DELTO_JOINTS)
    if revolutes != expected_revolutes:
        missing = sorted(set(expected_revolutes) - set(revolutes))
        extra = sorted(set(revolutes) - set(expected_revolutes))
        failures.append(f"expected {len(expected_revolutes)} revolute joints (6 arm + 20 hand), "
                        f"found {len(revolutes)}; missing={missing} unexpected={extra}")
    hand_revolutes = [n for n in revolutes if _DELTO_JOINT_RE.match(n)]
    if len(hand_revolutes) != len(_DELTO_JOINTS):
        failures.append(f"{len(hand_revolutes)} joints match rj_dg_[1-5]_[1-4], expected {len(_DELTO_JOINTS)}")

    # 3) exactly ONE articulation root, still the arm's own root_joint. A surviving nested root on
    #    the hand would split the robot into two articulations that the solver couples only through
    #    the mount joint -- the arm would appear to work while the hand went limp.
    roots = sorted(p.GetPath().pathString for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI))
    expected_root = f"{ROOT}/root_joint"
    if roots != [expected_root]:
        failures.append(f"expected exactly one ArticulationRootAPI on {expected_root}, found {roots}")
    nested = [r for r in roots if r.startswith(gpath + "/") or r == gpath]
    if nested:
        failures.append(f"nested ArticulationRootAPI survived inside the hand subtree: {nested}")

    # 4) the hand's AUTHORED explicit mass, unrescaled. See _EXPECTED_HAND_MASS.
    hand_bodies = {
        p.GetPath().pathString: p
        for p in prims
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetPath().pathString.startswith(gpath + "/")
    }
    hand_mass = sum(UsdPhysics.MassAPI(p).GetMassAttr().Get() or 0.0 for p in hand_bodies.values())
    if abs(hand_mass - _EXPECTED_HAND_MASS) > _MASS_TOL:
        failures.append(f"hand explicit mass {hand_mass:.6f} kg, expected the AUTHORED "
                        f"{_EXPECTED_HAND_MASS} kg (+/-{_MASS_TOL}) -- do NOT 'fix' this by "
                        f"rescaling to the 1.4 kg spec-sheet figure; the discrepancy is recorded")

    # 5) all five fingertip frame bodies present under the hand prim (the contact-sensor targets).
    for name in _DELTO_TIP_LINKS:
        tip = stage.GetPrimAtPath(f"{gpath}/{name}")
        if not tip or not tip.IsValid():
            failures.append(f"missing fingertip body {gpath}/{name}")

    # 6) the mount FixedJoint really connects the arm wrist to the hand base.
    mount_joint = stage.GetPrimAtPath(f"{gpath}/{base_link}/MountJoint")
    if not mount_joint or not mount_joint.IsValid():
        failures.append(f"mount FixedJoint not found at {gpath}/{base_link}/MountJoint")
    else:
        body0 = [t.pathString for t in mount_joint.GetRelationship("physics:body0").GetTargets()]
        body1 = [t.pathString for t in mount_joint.GetRelationship("physics:body1").GetTargets()]
        if body0 != [f"{ROOT}/wrist_3_link"]:
            failures.append(f"MountJoint body0 {body0}, expected [{ROOT}/wrist_3_link]")
        if body1 != [f"{gpath}/{base_link}"]:
            failures.append(f"MountJoint body1 {body1}, expected [{gpath}/{base_link}]")

    # 7) no 2F-85 residue. A leftover jaw body is an invisible extra collider bolted to the wrist.
    residue = []
    for prim in prims:
        name = prim.GetName()
        if name in F85_BODIES and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            residue.append(prim.GetPath().pathString)
        elif name in F85_JOINTS and "Joint" in str(prim.GetTypeName()):
            residue.append(prim.GetPath().pathString)
    if residue:
        failures.append(f"2F-85 residue survived the strip: {sorted(residue)}")

    print(f"  verify: {len(hand_bodies)} hand bodies, {len(revolutes)} revolute joints "
          f"({len(hand_revolutes)} hand), hand mass {hand_mass:.6f} kg, articulation root {roots}")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description="Graft the Tesollo DELTO DG-5F hand onto the UR5e arm.")
    ap.add_argument("--arm-usd", default=None,
                    help="Calibrated UR5e+2F-85 USD (input). Default: resolve/download from the cloud.")
    ap.add_argument("--gripper-usd",
                    default=os.path.join(_REPO, "source/uwlab_assets/uwlab_assets/local/Robots/DeltoHand/delto_hand.usd"),
                    help="Prepared DELTO hand USD, the output of prepare_delto_hand.py.")
    ap.add_argument("--output",
                    default=os.path.join(_REPO, "source/uwlab_assets/uwlab_assets/local/Robots/Ur5eDelto/ur5e_delto.usd"))
    ap.add_argument(
        "--standoff",
        type=float,
        default=None,
        help="Mount offset along wrist_3 +Z (m). Default: 0 for rl_dg_mount (the DELTO bolts "
        "directly to the flange, whose frame is coincident with wrist_3_link), 0.049 for other "
        "grippers using the adapter plate. An explicit value always overrides the inference.",
    )
    ap.add_argument("--gripper-root-prim", default="",
                    help="Prim path to reference out of --gripper-usd (e.g. /DeltoHand). Empty (the "
                         "default) references the file's defaultPrim, which prepare_delto_hand.py "
                         "sets; a hand authored without a defaultPrim needs this.")
    ap.add_argument("--gripper-base-link", default="rl_dg_mount",
                    help="Link inside the hand that bolts to the wrist: the nested "
                         "ArticulationRootAPI is stripped from it and the mount FixedJoint "
                         "attaches to it.")
    ap.add_argument("--strip-gripper-root-joint", default="root_joint",
                    help="Name of a joint-to-world inside the hand to delete. The DELTO source is "
                         "fixed to world by a free 6-DOF root joint that would fight the arm's "
                         "articulation. prepare_delto_hand.py already removes it, so this is "
                         "normally a no-op safety net; pass '' to skip.")
    ap.add_argument("--gripper-frame-links", default=",".join(_DELTO_TIP_LINKS),
                    help="Comma-separated hand links that may be massless, shapeless frames (the "
                         "DELTO's rl_dg_*_tip bodies). Any of them with no authored mass is given "
                         "--gripper-frame-mass and a small isotropic inertia; links that already "
                         "carry an authored mass are LEFT ALONE, so the hand total stays the "
                         "authored one that verify() checks.")
    ap.add_argument("--gripper-frame-mass", type=float, default=0.01,
                    help="Mass (kg) seeded onto massless --gripper-frame-links. Same 0.01 kg "
                         "hygiene value used for the UR10e's massless frame links.")
    ap.add_argument("--deinstance-links", default=",".join(_DELTO_LINKS),
                    help="Comma-separated hand links whose 'visuals' and 'collisions' prims are "
                         "de-instanced. Instance proxies are read-only: visuals must be writable "
                         "for per-env appearance DR and collision meshes for PhysX collider "
                         "visualization. De-instancing changes no geometry and no physics.")
    args = ap.parse_args()

    if args.standoff is None:
        args.standoff = 0.0 if args.gripper_base_link == "rl_dg_mount" else 0.049

    gripper_frame_links = _csv(args.gripper_frame_links)
    deinstance_links = _csv(args.deinstance_links)

    # Resolve the arm USD: explicit --arm-usd as given, else download the cloud asset to the cache.
    arm_usd = resolve_cloud_path(args.arm_usd) if args.arm_usd else resolve_cloud_path(_ARM_USD_URL)
    if not os.path.exists(arm_usd):
        raise SystemExit(f"arm USD not found: {arm_usd}\nDownload the calibrated USD from the cloud first.")
    if not os.path.exists(args.gripper_usd):
        raise SystemExit(f"hand USD not found: {args.gripper_usd}\nRun prepare_delto_hand.py first.")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    stage = Usd.Stage.Open(arm_usd)

    # 1) strip the 2F-85. The 2F-85's own mount FixedJoint is nested under robotiq_base_link, so it
    #    is removed along with that body. The 6 arm links/joints and root_joint are untouched, which
    #    is what preserves the UR5e sysid/FK calibration.
    stripped = 0
    for name in F85_BODIES + F85_JOINTS:
        p = f"{ROOT}/{name}"
        if stage.GetPrimAtPath(p):
            stage.RemovePrim(p)
            stripped += 1
    print(f"  stripped {stripped} 2F-85 prim(s) ({len(F85_BODIES)} bodies + {len(F85_JOINTS)} joints expected)")

    # 2) reference the hand under {ROOT}/gripper and place it at the flange. The prim name is fixed
    #    ('gripper') because dexlift's HAND_PRIM and its fingertip ContactSensorCfg paths address it.
    gpath = f"{ROOT}/{HAND_PRIM}"
    gprim = stage.DefinePrim(gpath, "Xform")
    if args.gripper_root_prim:
        gprim.GetReferences().AddReference(os.path.abspath(args.gripper_usd), Sdf.Path(args.gripper_root_prim))
    else:
        gprim.GetReferences().AddReference(os.path.abspath(args.gripper_usd))  # references its defaultPrim

    # world pose of the hand = wrist_3 world * mount(translate +standoff along wrist_3 +Z).
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    wrist = stage.GetPrimAtPath(f"{ROOT}/wrist_3_link")
    if not wrist or not wrist.IsValid():
        raise SystemExit(f"wrist_3_link not found under {ROOT}; is this the calibrated UR5e arm USD?")
    T_wrist_w = xc.GetLocalToWorldTransform(wrist)
    T_mount = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, 0.0, args.standoff))
    T_grip_w = T_mount * T_wrist_w  # pre-multiply: mount is in wrist frame
    # gripper prim local = (root world)^-1 * gripper world
    T_root_w = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(ROOT))
    grip_xform = UsdGeom.Xformable(gprim).MakeMatrixXform()
    grip_xform.Set(T_grip_w * T_root_w.GetInverse())

    # 2a) the target pose belongs to the hand's BASE LINK, not to the referenced subtree's root
    #     prim. Those coincide in the linear gripper (robotiq_base_link sits exactly on
    #     /linear_gripper) but not in general: a hand extracted out of a full-robot USD keeps
    #     whatever residual pose it had inside that robot -- the DELTO's rl_dg_mount was measured
    #     52 um and 0.036 deg off /DeltoHand in the variant this correction was written for. Left
    #     uncorrected, the base link misses the flange by that much while the mount FixedJoint's
    #     anchor still asserts exactly (0, 0, standoff), so the solver yanks the hand at the first
    #     step, and every frame derived from this transform (the wrist camera extrinsic) inherits
    #     the error. Re-place the subtree so the base link lands exactly on target. This is an
    #     identity correction when the base link already sits at the subtree root.
    base = stage.GetPrimAtPath(f"{gpath}/{args.gripper_base_link}")
    if not base or not base.IsValid():
        raise SystemExit(f"hand base link not found: {gpath}/{args.gripper_base_link} "
                         f"(pass --gripper-base-link)")
    xc.Clear()
    T_base_in_grip = xc.GetLocalToWorldTransform(base) * xc.GetLocalToWorldTransform(gprim).GetInverse()
    residual_t = T_base_in_grip.ExtractTranslation()
    grip_xform.Set(T_base_in_grip.GetInverse() * T_grip_w * T_root_w.GetInverse())
    xc.Clear()
    print(f"  base-link residual corrected: {tuple(round(float(v) * 1e6, 1) for v in residual_t)} um")

    # 3) the referenced hand carries its own ArticulationRootAPI (prepare_delto_hand.py applies it
    #    to rl_dg_mount so the standalone asset is spawnable); remove it so the hand joins the arm's
    #    single articulation and the root stays {ROOT}/root_joint. Search the WHOLE referenced
    #    subtree rather than a fixed prim -- which link carries it is an asset-authoring detail.
    #    A hand may also carry its own joint to world; that has to go for the same reason.
    for prim in Usd.PrimRange(gprim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            print(f"  removed nested ArticulationRootAPI from {prim.GetPath()}")
    if args.strip_gripper_root_joint:
        for parent in (gpath, f"{gpath}/joints"):
            rj = stage.GetPrimAtPath(f"{parent}/{args.strip_gripper_root_joint}")
            if rj and rj.IsValid():
                stage.RemovePrim(rj.GetPath())
                print(f"  removed the hand's own root joint {rj.GetPath()}")

    # 3a) seed any massless, shapeless frame body (the DELTO's fingertip links) with a small mass +
    #     inertia. PhysX cannot derive either for a body with no collider and no density, and a
    #     zero-mass dynamic body in an articulation is ill-conditioned. Links that already carry an
    #     authored mass are left exactly as authored -- this must not perturb the hand total that
    #     verify() checks against _EXPECTED_HAND_MASS.
    seeded, already_massed = 0, 0
    for name in gripper_frame_links:
        prim = stage.GetPrimAtPath(f"{gpath}/{name}")
        if not prim or not prim.IsValid():
            raise SystemExit(f"hand frame link not found: {gpath}/{name}")
        api = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) else UsdPhysics.MassAPI.Apply(prim)
        if api.GetMassAttr().Get():
            already_massed += 1
            continue
        api.CreateMassAttr().Set(args.gripper_frame_mass)
        # Point-like body: a 5 mm-radius solid sphere of this mass, well below any real link.
        i = 0.4 * args.gripper_frame_mass * 0.005**2
        api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(i, i, i))
        # Author the COM as well: unset, it resolves to UsdPhysics' (-inf, -inf, -inf) "derive it
        # from geometry" sentinel, and these bodies have no geometry.
        api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        seeded += 1
    if gripper_frame_links:
        print(f"  hand frame links: seeded {seeded} massless body(ies) with {args.gripper_frame_mass} kg, "
              f"left {already_massed} authored mass(es) untouched")

    # 4) mount FixedJoint wrist_3 -> the hand base link, authored with the same attribute set as the
    #    2F-85's own mount joint but with our mount transform: identity rotation (hand approach +Z
    #    aligned to wrist_3 +Z) and a +standoff offset along wrist_3 +Z.
    fj = UsdPhysics.FixedJoint.Define(stage, f"{gpath}/{args.gripper_base_link}/MountJoint")
    fp = fj.GetPrim()
    fj.CreateBody0Rel().SetTargets([Sdf.Path(f"{ROOT}/wrist_3_link")])
    fj.CreateBody1Rel().SetTargets([Sdf.Path(f"{gpath}/{args.gripper_base_link}")])
    fp.CreateAttribute("physics:localPos0", Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(0.0, 0.0, args.standoff))
    fp.CreateAttribute("physics:localRot0", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(1, 0, 0, 0))
    fp.CreateAttribute("physics:localPos1", Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fp.CreateAttribute("physics:localRot1", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(1, 0, 0, 0))
    fp.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    fp.CreateAttribute("physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool).Set(False)
    fp.CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    fp.CreateAttribute("physics:breakForce", Sdf.ValueTypeNames.Float).Set(float("inf"))
    fp.CreateAttribute("physics:breakTorque", Sdf.ValueTypeNames.Float).Set(float("inf"))

    # 5) flatten (inlines the hand + its meshes) and export a self-contained USD.
    flat = stage.Flatten()

    # 5a) de-instance the hand's visual AND collision prims. Instance proxies are read-only: visuals
    #     must be writable for per-env appearance DR, and collision meshes must be writable for
    #     PhysX collider visualization. This changes no collision geometry and no physics; it only
    #     makes each small mesh independently authorable.
    deinstanced_visuals = 0
    deinstanced_collisions = 0
    for link in deinstance_links:
        for subtree, kind in (("visuals", "visual"), ("collisions", "collision")):
            spec = flat.GetPrimAtPath(f"{gpath}/{link}/{subtree}")
            if spec is not None and spec.instanceable:
                spec.instanceable = False
                if kind == "visual":
                    deinstanced_visuals += 1
                else:
                    deinstanced_collisions += 1
    print(f"  hand meshes: de-instanced {deinstanced_visuals} visual and "
          f"{deinstanced_collisions} collision prim(s)")
    if deinstance_links and deinstanced_visuals == 0:
        print("  WARNING: no hand visuals were de-instanced -- if they are also not plain prims, "
              "the RGB collection's gripper-appearance DR will fail with 'No prims found "
              "matching ... visuals/.*'.")

    flat.Export(args.output)
    print(f"Wrote {args.output}")
    print(f"  standoff along wrist_3 +Z = {args.standoff} m (identity rotation; approach +Z = wrist_3 +Z)")

    failures = verify(args.output, args.gripper_base_link)
    if failures:
        print(f"\nFAILED -- {len(failures)} problem(s) with the written asset:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        # pure pxr, no kit: a plain non-zero exit is not swallowed the way it is under Isaac.
        sys.exit(1)
    print("\nOK -- all checks passed.")


if __name__ == "__main__":
    main()
