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
parser.add_argument("--finger-friction", type=float, default=0.8,
                    help="Static/dynamic friction baked onto the gripper colliders (2F-85 fingertips use 0.8).")
parser.add_argument("--finger-collision", type=str, default="sdf",
                    help="Collision for the L-shaped fingers: 'sdf' (exact, recommended), or convexHull/"
                         "convexDecomposition (both fail to represent the concave jaw face).")
parser.add_argument("--max-joint-velocity", type=float, default=130.0,
                    help="physxJoint:maxJointVelocity for both jaws (URDF default 0.05 throttles the mimic).")
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


# Jaw gripping-pad boxes, measured from the upper-jaw region of the finger collision mesh
# (body-local frame): inner gripping face at x=+/-0.0235, pad spans Y +/-0.0075, Z 0.023..0.051.
# These are clean box proxies that collide reliably where the convex/SDF mesh colliders failed.
_JAW_BOXES = {
    "left_inner_finger": ((-0.0332, 0.0, 0.0369), (0.0097, 0.0075, 0.0137)),
    "right_inner_finger": ((0.0332, 0.0, 0.0369), (0.0097, 0.0075, 0.0137)),
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
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
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

    # The URDF's velocity="0.05" m/s caps physxJoint:maxJointVelocity at 0.05 on BOTH jaws,
    # which throttles the mimic jaw so it can't keep up with the driver (lag). The reference
    # 2F-85 uses 130. Raise it on both jaws so the mimic follows rigidly (it's a safety cap;
    # the drive/actuator sets the real closing speed).
    for jp in (mimic_prim, driver_prim):
        mv = jp.GetAttribute("physxJoint:maxJointVelocity")
        if mv:
            mv.Set(args.max_joint_velocity)
    print(f"Set physxJoint:maxJointVelocity = {args.max_joint_velocity} on both jaw joints.")

    # Relocate CollisionAPI onto the Mesh prims (in the prototype layer) so the OmniReset
    # hasher detects the colliders -- without de-instancing (keeps it fast).
    n = relocate_collision_to_mesh(args.usd, friction=args.finger_friction,
                                   finger_approximation=args.finger_collision)
    print(f"Relocated CollisionAPI onto {n} mesh prim(s) (base only; finger mesh collision disabled).")

    # Add clean jaw box colliders on the fingers (the mesh colliders can't grip the L-shape).
    add_jaw_box_colliders(stage, args.finger_friction)

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
