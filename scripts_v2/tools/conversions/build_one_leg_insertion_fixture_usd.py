# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the RECEPTIVE one-leg insertion fixture USD (bead UWLab-3o5.3).

Converts play2perfect's ``one_leg_sdf_hybrid.urdf`` (one link: a slab + 4 rim
walls as boxes, 3 inactive-hole cylinders, and the active hole detail as a
mesh) through ``isaaclab.sim.converters.UrdfConverter``, then runs the
post-conversion SDF marker pass (``uwlab.sim.converters.sdf_markers``) to
promote the hole-detail mesh collider from the importer's default
``convexHull`` approximation to a PhysX ``sdf`` approximation, per the
``<sdf resolution="256"/>`` marker already present in that URDF's
``<collision>`` block.

A single link deliberately mixes collider types (box + cylinder analytic
colliders alongside one mesh/sdf collider) -- this is not a bug to fix.

Run (needs Isaac Sim; PYTHONPATH must include the four uwlab source dirs;
never launch through uwlab.sh, which wipes PYTHONPATH)::

    PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl" \\
        timeout -s KILL 300 \\
        /home/dom-iva/github.com/orel/lerobot/UWLab/env_uwlab/bin/python -u \\
        scripts_v2/tools/conversions/build_one_leg_insertion_fixture_usd.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

_DEFAULT_URDF = (
    "/tmp/claude-1000/-home-dom-iva-github-com-orel-lerobot-UWLab/"
    "12218aa8-bf8a-4a9b-9be5-88ff1de1adf3/scratchpad/p2p/assets/urdf/furniture_bench/"
    "square_table/insertion_fixtures/one_leg_sdf_hybrid.urdf"
)
_DEFAULT_OUT_DIR = "source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture"
_DEFAULT_USD_NAME = "one_leg_insertion_fixture.usd"

parser = argparse.ArgumentParser(description="Convert the one-leg receptive fixture URDF to USD, then stamp sdf.")
parser.add_argument("--urdf", type=str, default=_DEFAULT_URDF, help="Source URDF path.")
parser.add_argument("--usd-dir", type=str, default=_DEFAULT_OUT_DIR, help="Output USD directory.")
parser.add_argument("--usd-name", type=str, default=_DEFAULT_USD_NAME, help="Output USD file name.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

from uwlab.sim.converters import apply_urdf_sdf_collision_markers, parse_urdf_sdf_collision_markers  # noqa: E402


def _verify(usd_path: str, urdf_path: str) -> None:
    """Read the converted+stamped USD back and enforce the acceptance criteria."""
    from pathlib import Path

    from pxr import Usd, UsdPhysics

    raw = Path(usd_path)
    physics_usd = raw.parent / "configuration" / f"{raw.stem}_physics.usd"
    edit_path = physics_usd if physics_usd.exists() else raw
    print(f"[verify] reading back: {edit_path} (physics sublayer exists: {physics_usd.exists()})")

    stage = Usd.Stage.Open(str(edit_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"[verify] failed to open {edit_path}")
    stage.Load()

    rows: list[tuple[str, list[str], str, bool]] = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        has_mesh_collision = prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        if not (has_collision or has_mesh_collision):
            continue
        schemas = [s for s in prim.GetAppliedSchemas()]
        approx = "n/a (analytic, no MeshCollisionAPI)"
        if has_mesh_collision:
            mesh_api = UsdPhysics.MeshCollisionAPI(prim)
            attr = mesh_api.GetApproximationAttr()
            approx = attr.Get() if (attr and attr.HasAuthoredValue()) else "<UNAUTHORED>"
        rows.append((prim.GetPath().pathString, schemas, str(approx), prim.IsInstanceProxy()))

    print("\n[verify] collider table (path | applied schemas | physics:approximation | instance_proxy)")
    for path, schemas, approx, is_proxy in rows:
        print(f"  {path} | {schemas} | {approx} | instance_proxy={is_proxy}")

    # De-dupe: UrdfConverter authors each collider ONCE under /colliders/... and the link-local
    # hierarchy (e.g. /<robot>/receptive/collisions/...) is an INSTANCE PROXY view referencing that
    # same prim via a Reference arc (confirmed with Usd.PrimCompositionQuery: the link-local path's
    # prim stack resolves through a reference to the /colliders/ path). Usd.TraverseInstanceProxies()
    # surfaces both paths for the one physical collider, so "exactly one sdf collider" is evaluated
    # on the non-proxy occurrences only -- counting proxy aliases would over-count by however many
    # reference sites happen to exist, independent of how many colliders were actually stamped.
    canonical_rows = [r for r in rows if not r[3]]
    sdf_rows = [r for r in canonical_rows if r[2] == "sdf"]
    unauthored_mesh_rows = [r for r in canonical_rows if r[2] == "<UNAUTHORED>"]

    print(f"\n[verify] canonical (non-instance-proxy) prims reading approximation=='sdf': {[r[0] for r in sdf_rows]}")
    print(f"[verify] canonical mesh colliders with UNAUTHORED approximation: {[r[0] for r in unauthored_mesh_rows]}")

    assert len(sdf_rows) == 1, f"expected exactly one sdf collider, got {len(sdf_rows)}: {[r[0] for r in sdf_rows]}"
    assert "one_leg_hole_detail" in sdf_rows[0][0], f"sdf collider is not the hole patch: {sdf_rows[0][0]}"
    assert not unauthored_mesh_rows, (
        f"mesh collider(s) with a missing physics:approximation (silent-hull-fallback signature): "
        f"{[r[0] for r in unauthored_mesh_rows]}"
    )
    print("[verify] PASS: exactly one sdf collider (the hole patch), no unauthored mesh approximations.")

    # -- hole-mouth report: the hole-detail mesh prim's own transform/z-extent in the fixture frame.
    # The prim the sdf marker was stamped on (.../one_leg_hole_detail/World) is an Xform, not the
    # Mesh itself -- the URDF importer nests the actual UsdGeom.Mesh one level deeper
    # (.../one_leg_hole_detail/World/mesh). Search for that Mesh-typed prim specifically, under the
    # canonical (non-proxy) /colliders/ path.
    from pxr import UsdGeom

    hole_prim = next(
        (
            p
            for p in stage.Traverse()
            if "one_leg_hole_detail" in p.GetPath().pathString and p.IsA(UsdGeom.Mesh)
        ),
        None,
    )
    if hole_prim is not None:
        xcache = UsdGeom.XformCache()
        mtx = xcache.GetLocalToWorldTransform(hole_prim)
        mesh = UsdGeom.Mesh(hole_prim)
        if mesh is not None:
            pts = mesh.GetPointsAttr().Get()
            zs = [p[2] for p in pts]
            z_local_min, z_local_max = min(zs), max(zs)
            import numpy as _np

            m = _np.array(mtx).reshape(4, 4)
            p_min = _np.array([0, 0, z_local_min, 1.0]) @ m
            p_max = _np.array([0, 0, z_local_max, 1.0]) @ m
            print(f"\n[verify] hole-detail mesh prim: {hole_prim.GetPath()}")
            print(f"[verify] local z-extent: {z_local_min * 1000:.3f} .. {z_local_max * 1000:.3f} mm")
            print(f"[verify] world (fixture-frame) point at local z-min: {p_min[:3]}")
            print(f"[verify] world (fixture-frame) point at local z-max: {p_max[:3]}")
            print(
                "[verify] mouth-direction dispute: earlier geometry sweep (insert/sweep.py) treated the "
                "HIGH-z end (mesh z ~= +15.62mm, table-top / +Z push per assembly.json's y-up->z-up note) "
                "as the mouth and the LOW-z end (~-11.56mm) as the blind floor. Not re-resolved here -- "
                "a critic is adjudicating that separately."
            )
    else:
        print("[verify] WARNING: could not find a prim path containing 'one_leg_hole_detail' to report on.")


def main() -> None:
    os.makedirs(args_cli.usd_dir, exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=os.path.abspath(args_cli.urdf),
        usd_dir=os.path.abspath(args_cli.usd_dir),
        usd_file_name=args_cli.usd_name,
        force_usd_conversion=True,
        # fix_base=False (bead UWLab-3o5.10, was True): this fixture is only ever spawned as a
        # plain kinematic RigidObjectCfg (write_root_pose_to_sim sets its pose directly), never as
        # an IsaacLab Articulation asset -- fix_base=True's purpose (weld the base to world via a
        # fixed joint, for a robot base or similar) buys nothing here and instead bakes an
        # ArticulationRootAPI + fixed root_joint into a single-link fixture with no <joint> at all.
        # PhysX then logs "cannot create a joint between static bodies" at every spawn, because a
        # kinematic rigid body is already effectively static and the redundant fixed-to-world joint
        # is invalid. fix_base=False leaves the root link free (no joint authored at all), which is
        # what a kinematic-spawned rigid body actually needs. The config-side
        # disable_articulation_root escape hatch on make_receptive_object (reset_states_cfg.py)
        # stays either way -- it is correct and harmless if this asset no longer needs it.
        fix_base=False,
        merge_fixed_joints=True,
        make_instanceable=False,
        # collider_type intentionally left at its default (convex_hull): UrdfConverter's vocabulary
        # is convex_hull | convex_decomposition only. sdf is reachable only via the post-conversion
        # marker pass below, which reads the <sdf> tag already present in the source URDF.
        #
        # This fixture's single link has no <joint> at all, so joint_drive never actually drives
        # anything -- but UrdfConverterCfg._validate() requires joint_drive.gains.stiffness to be
        # authored regardless (a MISSING sentinel fails validation even for a joint-less URDF).
        # Values are placeholders that are never exercised.
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=200.0, damping=20.0),
        ),
    )

    print("-" * 80)
    print(f"Input  URDF: {cfg.asset_path}")
    print("Urdf importer config:")
    print_dict(cfg.to_dict(), nesting=0)  # type: ignore
    print("-" * 80)

    converter = UrdfConverter(cfg)
    usd_path = converter.usd_path
    print(f"Generated USD file: {usd_path}")
    print("-" * 80)

    markers = parse_urdf_sdf_collision_markers(args_cli.urdf)
    print(f"[sdf_markers] found {len(markers)} <sdf> marker(s) in {args_cli.urdf}: {markers}")
    if not markers:
        raise RuntimeError("expected at least one <sdf> marker (the hole-detail mesh) but found none")

    result = apply_urdf_sdf_collision_markers(usd_path, args_cli.urdf, markers)
    print(f"[sdf_markers] done: {result}")

    _verify(usd_path, args_cli.urdf)


if __name__ == "__main__":
    main()
    simulation_app.close()
