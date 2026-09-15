# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lift rigid bodies nested inside other rigid bodies up to siblings, in place. Pure pxr, no Isaac app.

WHY. The Isaac Sim 6 URDF importer nests each link under its parent link
(``Geometry/robotiq_base_link/left_inner_finger``). Isaac Lab applies every spawn-time body property
-- ``rigid_props`` (``disable_gravity``, damping, ``max_depenetration_velocity``), ``mass_props``,
``collision_props`` -- through ``isaaclab.sim.utils.prims.apply_nested``, which stops descending at
the first prim it succeeds on, "based on the physics behavior that nested schemas are not allowed".
So on the nested layout only the ROOT link receives them and every child link silently keeps the
USD defaults.

Measured on the linear gripper: the grasp sampler's gripper is spawned with ``disable_gravity=True``
and ``linear_damping=50``, yet once the sampler switched gravity on the whole gripper sank at a
steady 0.143 m/s -- a damped terminal velocity -- carrying a correctly held object past the 0.25 m
deviation gate in every attempt. The Isaac Sim 5.1 importer wrote the links as siblings, which is the
layout ``apply_nested`` assumes, and this restores it.

Each lifted body keeps its world transform, and every relationship in the layer that pointed into a
moved subtree (joint ``body0``/``body1``, material bindings, filtered pairs) is retargeted.

Usage::

    python scripts_v2/tools/conversions/unnest_rigid_bodies.py --usd path/to/robot.usd
"""

from __future__ import annotations

import argparse

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def _nested_bodies(stage: Usd.Stage) -> list[Usd.Prim]:
    """Rigid bodies with a rigid-body ancestor, deepest first so parents move after their children."""
    hits = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        parent = prim.GetParent()
        while parent and parent.GetPath() != Sdf.Path.absoluteRootPath:
            if parent.HasAPI(UsdPhysics.RigidBodyAPI):
                hits.append(prim)
                break
            parent = parent.GetParent()
    return sorted(hits, key=lambda p: -p.GetPath().pathElementCount)


def _top_body_parent(prim: Usd.Prim) -> Usd.Prim:
    """The parent of the outermost rigid-body ancestor: where the lifted body belongs."""
    top = prim.GetParent()
    cursor = prim.GetParent()
    while cursor and cursor.GetPath() != Sdf.Path.absoluteRootPath:
        if cursor.HasAPI(UsdPhysics.RigidBodyAPI):
            top = cursor
        cursor = cursor.GetParent()
    return top.GetParent()


def unnest(usd_path: str) -> int:
    stage = Usd.Stage.Open(usd_path)
    layer = stage.GetRootLayer()
    moves = 0
    while True:
        nested = _nested_bodies(stage)
        if not nested:
            break
        prim = nested[0]
        src = prim.GetPath()
        new_parent = _top_body_parent(prim)
        dst = new_parent.GetPath().AppendChild(src.name)
        if stage.GetPrimAtPath(dst):
            raise SystemExit(f"cannot lift {src}: {dst} already exists")

        xc = UsdGeom.XformCache(Usd.TimeCode.Default())
        world = xc.GetLocalToWorldTransform(prim)
        parent_world = xc.GetLocalToWorldTransform(new_parent)
        local = world * parent_world.GetInverse()

        edit = Sdf.BatchNamespaceEdit()
        edit.Add(Sdf.NamespaceEdit.Reparent(src, new_parent.GetPath(), -1))
        if not layer.CanApply(edit):
            raise SystemExit(f"layer refuses to move {src} -> {dst}")
        if not layer.Apply(edit):
            raise SystemExit(f"moving {src} -> {dst} failed")

        moved = stage.GetPrimAtPath(dst)
        xf = UsdGeom.Xformable(moved)
        xf.ClearXformOpOrder()
        for name in list(moved.GetPropertyNames()):
            if name.startswith("xformOp:"):
                moved.RemoveProperty(name)
        xf.AddTransformOp().Set(Gf.Matrix4d(local))

        for other in stage.Traverse():
            for rel in other.GetRelationships():
                targets = rel.GetTargets()
                fixed = [t.ReplacePrefix(src, dst) if t.HasPrefix(src) else t for t in targets]
                if fixed != list(targets):
                    rel.SetTargets(fixed)
        print(f"  lifted {src} -> {dst}")
        moves += 1

    left = _nested_bodies(stage)
    if left:
        raise SystemExit(f"still nested after lifting: {[str(p.GetPath()) for p in left]}")
    layer.Save()
    return moves


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--usd", required=True, help="USD edited in place.")
    args = ap.parse_args()
    n = unnest(args.usd)
    print(f"unnested {n} rigid bod{'y' if n == 1 else 'ies'} in {args.usd}")


if __name__ == "__main__":
    main()
