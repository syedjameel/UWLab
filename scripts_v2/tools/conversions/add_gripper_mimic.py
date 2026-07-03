# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare the standalone linear-gripper USD for grasp sampling: fix the jaw coupling, relocate
the collisions onto clean jaw-box proxies, and bake in finger friction.

Jaw coupling -- use --dual-drive (RECOMMENDED, and what the current pipeline uses): keep BOTH
jaws' position DriveAPI and author NO mimic, so the grasp-sampling actuator drives both jaws to
one binary target. The PhysX prismatic mimic is unreliable even standalone (the soft driver
outruns the free follower and the solver pins both jaws at 0 -> the gripper never grips, and the
recorder logs finger_joint=0). The legacy mimic path (default, no --dual-drive) authors a
physxMimicJoint on the follower matching the reference 2F-85 USD (gearing +1.0 for our frame; the
old -1.0 demanded q_right=-q_finger, below the follower's lower limit 0, which froze the driver).

Run in-app (server or local leisaac)::

    ./uwlab.sh -p scripts_v2/tools/conversions/add_gripper_mimic.py \
        --usd source/uwlab_assets/uwlab_assets/local/Robots/LinearGripper/linear_gripper.usd \
        --mimic-joint right_finger_joint --driver-joint finger_joint --dual-drive
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Apply a PhysX mimic joint to a gripper USD.")
parser.add_argument("--usd", type=str, required=True, help="Gripper USD to edit in place.")
parser.add_argument("--mimic-joint", type=str, default="right_finger_joint", help="Passive joint name.")
parser.add_argument("--driver-joint", type=str, default="finger_joint", help="Reference (driver) joint name.")
parser.add_argument("--axis", type=str, default="transX", help="Mimic axis token (e.g. transX, rotZ).")
parser.add_argument("--gearing", type=float, default=1.0,
                    help="Mimic gearing. +1.0: both jaws' joint coords increase together to close "
                    "(right_finger_joint is frame-flipped Rz180, so q_right=+q_finger is symmetric). "
                    "-1.0 demands q_right=-q_finger, which is below right's lower limit 0 -> the mimic "
                    "constraint pins the driver at 0 and the gripper never closes. Ignored with --dual-drive.")
parser.add_argument("--dual-drive", action="store_true",
                    help="Do NOT author a mimic; keep BOTH jaws' position DriveAPI so they are dual-driven "
                    "to the same binary target (the standalone grasp-sampling config drives both). The PhysX "
                    "prismatic mimic is unreliable even in the standalone: the soft driver outruns the free "
                    "follower, the mimic equality constraint accumulates error, and the solver pins both jaws "
                    "at 0 (recorded grasps then have finger_joint=0). Dual-drive matches the full robot.")
parser.add_argument("--offset", type=float, default=0.0, help="Mimic offset.")
parser.add_argument("--finger-friction", type=float, default=100.0,
                    help="Static/dynamic friction baked onto the gripping fingers. The reference 2F-85 binds "
                         "PhysicsMaterial (friction=100, combineMode=max) ON THE left/right_inner_finger LINK "
                         "prims -- NOT the 0.8 fingertip_material. 0.8 lets the object slip; 100 grips "
                         "(verified: grip holds under gravity only with the link-bound 100 material).")
parser.add_argument("--finger-collision", type=str, default="sdf",
                    help="Collision for the L-shaped fingers: 'sdf' (exact, recommended), or convexHull/"
                         "convexDecomposition (both fail to represent the concave jaw face).")
parser.add_argument("--max-joint-velocity", type=float, default=130.0,
                    help="maxJointVelocity for the MIMIC jaw (high so it follows the driver rigidly).")
parser.add_argument("--close-velocity", type=float, default=130.0,
                    help="maxJointVelocity for the DRIVER jaw. Reference finger_joint uses 130; 0.5 throttled the "
                         "driver (verified the grip holds at 130 in the configured 0.06-0.08 finger_offset band).")
parser.add_argument("--test", action="store_true", help="After authoring, drive the joint in sim to verify coupling.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os

from pxr import PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade


def _find_joint(stage, name):
    for p in stage.Traverse():
        if p.GetName() == name and "Joint" in str(p.GetTypeName()):
            return p
    raise RuntimeError(f"joint '{name}' not found")


def relocate_collision_to_mesh(usd_path: str, friction: float | None = None,
                               finger_approximation: str = "convexDecomposition") -> int:
    """Move CollisionAPI onto the child Mesh in the collider PROTOTYPE, preserving instancing.

    Isaac's URDF importer applies PhysicsCollisionAPI/MeshCollisionAPI on an Xform wrapper
    (``node_STL_BINARY_``) and makes each body's ``collisions`` subtree a USD instance. The
    OmniReset collision-point sampler (RigidObjectHasher) only accepts colliders whose prim is
    a Mesh (like the reference 2F-85's ``/collisions/mesh_1``) -- so it finds 0 colliders and
    grasp sampling crashes (``torch.cat([])``).

    The prototype lives in the ``configuration/<name>_physics.usd`` layer under ``/colliders``.
    Editing it there (move CollisionAPI onto the child Mesh) fixes every instance at once and
    KEEPS instancing -- important, because de-instancing duplicates the big collision meshes
    (26k+ pts) per env and makes grasp sampling extremely slow. Returns colliders moved.
    """
    cfg_dir = os.path.join(os.path.dirname(os.path.abspath(usd_path)), "configuration")
    base = os.path.splitext(os.path.basename(usd_path))[0]
    phys = os.path.join(cfg_dir, f"{base}_physics.usd")
    if not os.path.exists(phys):
        print(f"  WARNING: prototype layer {phys} not found; skipping collision relocation.")
        return 0
    ps = Usd.Stage.Open(phys)
    moved = 0
    meshes = []
    for prim in list(ps.Traverse()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetTypeName() == "Mesh":
            continue
        mesh = next((c for c in prim.GetChildren() if c.GetTypeName() == "Mesh"), None)
        if mesh is None:
            continue
        # The L-shaped FINGER mesh cannot be collided reliably (convexHull crushes at the
        # bracket, convexDecomposition drops the thin jaw, SDF on a dynamic articulation link
        # generates no contacts against the jaw). So we DISABLE the finger mesh collision here
        # and add a clean jaw BOX proxy on the body instead (add_jaw_box_colliders). The BASE
        # mesh keeps its convexHull collision (it doesn't grip).
        prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
        if "inner_finger" in str(mesh.GetPath()):
            continue  # finger -> no mesh collider (jaw box added separately)
        ce_attr = prim.GetAttribute("physics:collisionEnabled")
        approx_attr = prim.GetAttribute("physics:approximation")
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        if ce_attr and ce_attr.Get() is not None:
            mesh.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(ce_attr.Get())
        approx = approx_attr.Get() if (approx_attr and approx_attr.Get() is not None) else "convexHull"
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approx)
        meshes.append(mesh.GetPrim())
        moved += 1

    # Bake friction onto the colliders. The grasp-sampling env does NOT randomize friction --
    # it relies on the friction in the gripper USD. Without it the object slides out of the
    # jaws and falls. The reference 2F-85 fingertips use 0.8. IMPORTANT: the material must live
    # INSIDE each collider body's subtree (/colliders/<body>/...), because each body references
    # /colliders/<body> -- a material at /colliders root is "outside the reference scope" and
    # PhysX silently ignores the binding.
    if friction is not None and meshes:
        made = {}
        for m in meshes:
            parts = str(m.GetPath()).split("/")
            # path: /colliders/<body>/<shape>/node.../mesh  -> body root = /colliders/<body>
            body_root = "/".join(parts[:3])
            if body_root not in made:
                mat = UsdShade.Material.Define(ps, f"{body_root}/PhysicsMaterial")
                papi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
                papi.CreateStaticFrictionAttr(friction)
                papi.CreateDynamicFrictionAttr(friction)
                made[body_root] = mat
            binding = UsdShade.MaterialBindingAPI.Apply(m)
            binding.Bind(made[body_root], bindingStrength=UsdShade.Tokens.weakerThanDescendants,
                         materialPurpose="physics")
        print(f"  Bound friction={friction} physics material to {len(meshes)} collider mesh(es).")

    ps.GetRootLayer().Save()
    return moved


# Jaw gripping-pad boxes on each finger's inner gripping FACE (body-local x=+/-0.0235). These are
# clean box proxies that collide reliably where the convex/SDF mesh colliders failed. The FULL jaw
# HEIGHT (Z half 0.040) holds the object -- a short box lets it slide out.
#
# The box Y must match the finger's actual front GRIPPING FACE, which is a narrow tab Y +/-0.0065.
# The pad body FLARES to +/-0.019 BEHIND the face, but the object only ever touches the front tab.
# A wider box (Y +/-0.019, or the flare) is INVISIBLE yet pokes ~10-13mm past the visible tab, so a
# diagonally-held object rests on that invisible ledge and floats off the rendered fingertip (a
# flush/flat grasp hides it). Y +/-0.0065 sits flush with the visible tab (hidden inside the flare
# behind it) and still holds (validated: grips + holds under gravity across the grip zone). Body-
# local: x[-0.043,-0.0235], z[-0.0295,0.0505] -> base-z 0.064..0.144.
_JAW_BOXES = {
    "left_inner_finger": ((-0.0333, 0.0, 0.0105), (0.0098, 0.0065, 0.040)),
    "right_inner_finger": ((0.0333, 0.0, 0.0105), (0.0098, 0.0065, 0.040)),
}


_BOX_CORNERS = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
_BOX_FACES = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 3, 7, 6, 2, 0, 4, 7, 3, 1, 2, 6, 5]


def add_jaw_box_colliders(stage, friction: float | None) -> int:
    """Add a clean box collider at each finger's gripping pad (replaces the mesh collider).

    The collider is a box MESH with convexHull approximation -- the SAME way the base and the
    working object USDs are authored. (A UsdGeom.Cube with non-uniform xformOp:scale does NOT
    collide -- the scale is not propagated to the PhysX shape; verified: base mesh+convexHull
    ejects the slab, the scaled Cube generates no contact.) The box sits on the inner gripping
    face of the upper jaw so the two jaws pinch an object between them.
    """
    from pxr import Gf, UsdGeom, Vt

    mat = None
    if friction is not None:
        mat = UsdShade.Material.Define(stage, "/linear_gripper/JawPhysicsMaterial")
        papi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        papi.CreateStaticFrictionAttr(friction)
        papi.CreateDynamicFrictionAttr(friction)
        # CRITICAL: PhysX combines the two contacting materials' frictions, and here it resolves
        # to the OBJECT's (low) friction -- so a high jaw friction has NO effect and a light
        # object slides straight down out of the grip (verified: jaw 0.5 vs 2.0 -> identical
        # slip). frictionCombineMode="max" makes the contact use the JAW's high friction
        # regardless of the object's, so the grip holds against gravity.
        pxmat = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        pxmat.CreateFrictionCombineModeAttr("max")

    n = 0
    for body, (center, half) in _JAW_BOXES.items():
        body_prim = stage.GetPrimAtPath(f"/linear_gripper/{body}")
        if not body_prim or not body_prim.IsValid():
            print(f"  WARNING: finger body /linear_gripper/{body} not found; skipping jaw box.")
            continue
        # The collider must live INSIDE the link's /collisions subtree (where the base's working
        # collider is); de-instance it so we can add the box there. /collisions is at identity
        # relative to the body, so body-local box points are correct.
        coll = stage.GetPrimAtPath(f"/linear_gripper/{body}/collisions")
        if coll and coll.IsValid():
            coll.SetInstanceable(False)
        cx, cy, cz = center
        hx, hy, hz = half
        mesh = UsdGeom.Mesh.Define(stage, f"/linear_gripper/{body}/collisions/JawBox")
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(cx + sx * hx, cy + sy * hy, cz + sz * hz)
                                             for sx, sy, sz in _BOX_CORNERS]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * 6))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(_BOX_FACES))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(cx - hx, cy - hy, cz - hz),
                                             Gf.Vec3f(cx + hx, cy + hy, cz + hz)]))
        UsdGeom.Imageable(mesh).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        if mat is not None:
            # Bind on the jaw collider mesh AND on the finger LINK prim. The reference 2F-85 binds
            # its grip material on the left/right_inner_finger LINK (not the collision mesh); a
            # mesh-only binding inside the (previously instanced) /collisions subtree did not take
            # effect (object slipped regardless of friction value). Binding on the link is what
            # actually makes the high friction reach the contact.
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                mat, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics"
            )
            UsdShade.MaterialBindingAPI.Apply(body_prim).Bind(
                mat, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics"
            )
        n += 1
    print(f"Added {n} jaw box collider mesh(es) (convexHull, friction={friction}).")
    return n


def author_mimic() -> None:
    print("PhysxMimicJointAPI methods:", [m for m in dir(PhysxSchema.PhysxMimicJointAPI) if not m.startswith("_")])
    stage = Usd.Stage.Open(os.path.abspath(args.usd))
    mimic_prim = _find_joint(stage, args.mimic_joint)
    driver_prim = _find_joint(stage, args.driver_joint)

    if args.dual_drive:
        # Dual-drive: NO mimic; keep both jaws' DriveAPI (the converter gives both a linear drive).
        # The standalone grasp-sampling actuator/action drive both jaws to the same binary target.
        if not mimic_prim.HasAPI(UsdPhysics.DriveAPI, "linear"):
            UsdPhysics.DriveAPI.Apply(mimic_prim, "linear")
        print(f"Dual-drive: kept DriveAPI on both jaws ('{args.driver_joint}', '{args.mimic_joint}'); no mimic.")
    else:
        api = PhysxSchema.PhysxMimicJointAPI.Apply(mimic_prim, args.axis)
        api.CreateGearingAttr(args.gearing)
        api.CreateOffsetAttr(args.offset)
        ref_rel = api.CreateReferenceJointRel()
        ref_rel.SetTargets([driver_prim.GetPath()])
        # referenceJointAxis defaults to the same axis; set explicitly to match the reference USD.
        if hasattr(api, "CreateReferenceJointAxisAttr"):
            api.CreateReferenceJointAxisAttr(args.axis)

        # CRITICAL: the URDF converter gives BOTH prismatic joints a position DriveAPI. On the
        # passive (mimic) joint that drive pins it at its target (0) and overrides the mimic, so
        # the second jaw never moves (-> 0% grasp success). The reference 2F-85's passive joints
        # have NO drive -- the mimic controls them. Remove the drive from the mimic joint to match.
        removed_drive = False
        for inst in ("linear", "angular"):
            if mimic_prim.HasAPI(UsdPhysics.DriveAPI, inst):
                mimic_prim.RemoveAPI(UsdPhysics.DriveAPI, inst)
                removed_drive = True
        print(f"Removed DriveAPI from mimic joint '{args.mimic_joint}': {removed_drive}")

    # Per-jaw velocity caps decouple "gentle close" from "rigid mimic":
    #  * DRIVER jaw -> a LOW cap (--close-velocity) so the actuator closes gently and does not
    #    slam/FLING a light object out (a fast close ejects it even with the right grip force).
    #  * MIMIC jaw  -> a HIGH cap (--max-joint-velocity) so it can move fast enough to follow the
    #    driver rigidly (a low cap on the mimic makes it lag badly). The URDF's 0.05 throttled both.
    dmv = driver_prim.GetAttribute("physxJoint:maxJointVelocity")
    mmv = mimic_prim.GetAttribute("physxJoint:maxJointVelocity")
    if dmv:
        dmv.Set(args.close_velocity)
    if mmv:
        mmv.Set(args.max_joint_velocity)
    print(f"Set maxJointVelocity: driver={args.close_velocity} (gentle close), mimic={args.max_joint_velocity} (follow).")

    # Relocate CollisionAPI onto the Mesh prims (in the prototype layer) so the OmniReset
    # hasher detects the colliders -- without de-instancing (keeps it fast).
    n = relocate_collision_to_mesh(args.usd, friction=args.finger_friction,
                                   finger_approximation=args.finger_collision)
    print(f"Relocated CollisionAPI onto {n} mesh prim(s) (base only; finger mesh collision disabled).")

    # Add clean jaw box colliders on the fingers (the mesh colliders can't grip the L-shape).
    add_jaw_box_colliders(stage, args.finger_friction)

    stage.GetRootLayer().Save()
    if args.dual_drive:
        print(f"Dual-drive standalone: both jaws driven, no mimic ('{args.driver_joint}', '{args.mimic_joint}').")
    else:
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
