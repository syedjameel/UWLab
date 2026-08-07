# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read collider vertices for every rigid body of a gripper USD stage."""

from __future__ import annotations

import numpy as np

from pxr import Usd, UsdGeom, UsdPhysics

_PRED = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)


class NoColliderGeometry(RuntimeError):
    """Raised when a body, or the whole gripper, yields no collider points."""


def _geometry_points(prim) -> np.ndarray | None:
    """Return authored mesh points in the geometry prim's frame, if present."""
    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        return None
    points = mesh.GetPointsAttr().Get()
    if not points:
        return None
    return np.asarray(points, dtype=float)


def collider_points(stage, expect_bodies: int | None = None) -> dict[str, np.ndarray]:
    """Return collider points per rigid body, expressed in that body's frame.

    CollisionAPI may be authored on an Xform wrapper rather than on the mesh itself, so each
    collider prim is descended for geometry. Coverage is deliberately strict: a placement gate
    that silently checks only a subset of bodies recreates the failure this reader prevents.
    """
    xform_cache = UsdGeom.XformCache()
    points_by_body: dict[str, np.ndarray] = {}
    bodies = [prim for prim in Usd.PrimRange.Stage(stage, _PRED) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    for body in bodies:
        body_world = np.array(xform_cache.GetLocalToWorldTransform(body)).T
        chunks = []
        for collider in Usd.PrimRange(body, _PRED):
            if not collider.HasAPI(UsdPhysics.CollisionAPI):
                continue
            for geometry in Usd.PrimRange(collider, _PRED):
                points = _geometry_points(geometry)
                if points is None:
                    continue
                geometry_in_body = (
                    np.linalg.inv(body_world) @ np.array(xform_cache.GetLocalToWorldTransform(geometry)).T
                )
                chunks.append((geometry_in_body[:3, :3] @ points.T).T + geometry_in_body[:3, 3])
        if chunks:
            points_by_body[body.GetName()] = np.vstack(chunks)

    if not points_by_body:
        raise NoColliderGeometry(
            f"no collider geometry found on any of {len(bodies)} rigid bodies. Either the asset "
            "authors collision shapes (Cube/Capsule/Sphere) rather than meshes -- extend "
            "_geometry_points -- or CollisionAPI sits somewhere this traversal does not reach. "
            "Refusing rather than returning empty: an empty result makes a placement gate accept "
            "every candidate."
        )
    if expect_bodies is not None and len(points_by_body) < expect_bodies:
        missing = sorted({body.GetName() for body in bodies} - set(points_by_body))
        raise NoColliderGeometry(
            f"collider geometry on only {len(points_by_body)} of {expect_bodies} expected bodies; "
            f"missing {missing}. A placement gate over a subset silently ignores the bodies it "
            "cannot see."
        )
    return points_by_body


def subsample(points: np.ndarray, count: int) -> np.ndarray:
    """Farthest-point subsample so the retained points continue to bound the hull."""
    if len(points) <= count:
        return points
    indices = [int(np.argmax(np.linalg.norm(points - points.mean(0), axis=1)))]
    distances = np.linalg.norm(points - points[indices[0]], axis=1)
    for _ in range(count - 1):
        index = int(np.argmax(distances))
        indices.append(index)
        distances = np.minimum(distances, np.linalg.norm(points - points[index], axis=1))
    return points[indices]
