# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for programmatically authoring simple OmniReset task assets.

These utilities author USD assets using *only* ``pxr.UsdPhysics`` (no PhysX-specific
schemas), so they can run under a plain ``pxr`` install as well as Isaac Sim's python.

Every asset follows the standard OmniReset asset format expected by
``make_insertive_object`` / ``make_receptive_object`` (see
``omnireset/config/ur5e_robotiq_2f85/reset_states_cfg.py``):

* a single ``Xform`` root carrying ``UsdPhysics.RigidBodyAPI`` (the default prim),
* a ``visuals`` subtree of rendered meshes (no physics), and
* a ``collisions`` subtree of invisible meshes carrying collision APIs.

Rectangular geometry is represented exactly with axis-aligned boxes; collision uses a
``convexHull`` approximation per box, which is exact for a box and avoids any dependence
on PhysX SDF cooking. Mass / solver / kinematic flags are intentionally *not* baked here
because the ``make_*`` spawn configs set them at spawn time.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import yaml
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

# Box corner offsets (unit half-extents) and outward quad faces.
_CORNERS = [
    (-1, -1, -1),
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, 1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, 1, 1),
]
_FACE_COUNTS = [4, 4, 4, 4, 4, 4]
# Winding is CCW as seen from outside so face normals point OUTWARD. This matters because
# the grasp sampler reads the visual mesh's trimesh face_normals (process=False, i.e. winding
# is trusted) to bias toward top faces and cast grasp rays along -normal. Inward normals yield
# zero grasp candidates. (Verified: for a box centered on the prim, dot(normal, centroid) > 0.)
_FACE_INDICES = [
    0, 3, 2, 1,  # -Z
    4, 5, 6, 7,  # +Z
    0, 1, 5, 4,  # -Y
    3, 7, 6, 2,  # +Y
    0, 4, 7, 3,  # -X
    1, 2, 6, 5,  # +X
]


def create_stage(usd_path: str, root_name: str) -> tuple[Usd.Stage, UsdGeom.Xform]:
    """Create a Z-up, meter-scale stage with an Xform root that is a rigid body."""
    os.makedirs(os.path.dirname(os.path.abspath(usd_path)), exist_ok=True)
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, f"/{root_name}")
    stage.SetDefaultPrim(root.GetPrim())
    # Mark the root as a single rigid body; the spawn config overrides mass/kinematic flags.
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    # MassAPI must exist for the spawn config's mass_props to apply (isaaclab
    # modify_mass_properties returns False otherwise). The mass value is set at spawn time.
    UsdPhysics.MassAPI.Apply(root.GetPrim())

    UsdGeom.Scope.Define(stage, f"/{root_name}/visuals")
    UsdGeom.Scope.Define(stage, f"/{root_name}/collisions")
    return stage, root


def add_box(
    stage: Usd.Stage,
    prim_path: str,
    center: Sequence[float],
    half_extents: Sequence[float],
    *,
    collision: bool,
    color: Sequence[float] | None = None,
) -> UsdGeom.Mesh:
    """Author an axis-aligned box mesh.

    Args:
        center: box center in the root frame (meters).
        half_extents: box half-sizes along x, y, z (meters).
        collision: if True author an invisible collider; else a rendered visual mesh.
        color: optional RGB display color for visual meshes.
    """
    cx, cy, cz = center
    hx, hy, hz = half_extents
    points = Vt.Vec3fArray([Gf.Vec3f(cx + sx * hx, cy + sy * hy, cz + sz * hz) for sx, sy, sz in _CORNERS])

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(_FACE_COUNTS))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(_FACE_INDICES))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr(
        Vt.Vec3fArray([Gf.Vec3f(cx - hx, cy - hy, cz - hz), Gf.Vec3f(cx + hx, cy + hy, cz + hz)])
    )

    if collision:
        UsdGeom.Imageable(mesh).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        # convexHull is exact for a box and needs no PhysX SDF cooking.
        mesh_collision.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    elif color is not None:
        mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))

    return mesh


def write_metadata(usd_path: str, metadata: dict) -> str:
    """Write a ``metadata.yaml`` next to ``usd_path`` and return its path."""
    out = os.path.join(os.path.dirname(os.path.abspath(usd_path)), "metadata.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(metadata, f, default_flow_style=None, sort_keys=False)
    return out
