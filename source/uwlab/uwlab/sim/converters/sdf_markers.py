# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-conversion pass that stamps PhysX SDF mesh-collision approximation onto
USD prims produced by :class:`isaaclab.sim.converters.UrdfConverter`.

``<sdf>`` is **not** a URDF tag -- it is not part of the URDF spec, and
IsaacLab's ``UrdfConverter`` ignores it entirely. The converter's collider
vocabulary is limited to :attr:`~isaaclab.sim.converters.UrdfConverterCfg.collider_type`
``"convex_hull"`` (default) or ``"convex_decomposition"``; there is no
importer-level path to a PhysX SDF collider. The only route is the schema API
(:func:`isaaclab.sim.schemas.define_mesh_collision_properties`) applied as a
pass over the already-converted stage.

To keep the SDF intent alongside the geometry it targets, the source URDF is
annotated with a ``<sdf>`` marker -- a sibling of ``<geometry>`` inside
``<collision>``::

    <collision>
      <geometry><mesh filename="part.obj"/></geometry>
      <sdf resolution="256" margin="0.01" narrow_band_thickness="0.01" subgrid_resolution="6"/>
    </collision>

:func:`parse_urdf_sdf_collision_markers` reads these markers out of the URDF
(a pure XML scan; the URDF importer never sees the tag and does not choke on
it). :func:`apply_urdf_sdf_collision_markers` then walks the converted USD and
re-applies :class:`~isaaclab.sim.schemas.SDFMeshPropertiesCfg` to the prim(s)
whose path contains the marked mesh's file stem.

Three failure modes this pass has to get right, or it silently no-ops:

1. **Layer.** ``UrdfConverter`` writes physics onto a *separate* sublayer at
   ``<usd_dir>/configuration/<stem>_physics.usd`` when that layer exists.
   Editing the raw (geometry-only) USD in that case edits a layer physics
   does not read from -- the edit target must resolve to the physics
   sublayer first.
2. **Traversal.** A plain ``stage.Traverse()`` skips instanced collision
   prims. This pass walks
   ``Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())``
   and only considers prims that already carry ``UsdPhysics.MeshCollisionAPI``
   (i.e. prims the URDF importer already turned into colliders), matched by
   mesh stem against any path component. Matches under ``/colliders/`` are
   preferred over other matches *for the same marker* when both exist.
3. **Zero-match guard.** A marker that matches nothing means a renamed or
   moved mesh silently degraded from ``sdf`` to the importer's default
   ``convexHull`` -- exactly the class of silent fallback that is not
   acceptable here. Zero matches for any marker raises, listing the marker,
   the stem searched for, and every collision prim path the traversal did
   find (to make the mismatch obvious).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UrdfSdfCollisionMarker:
    """One ``<sdf>`` marker parsed out of a URDF ``<collision>`` block."""

    mesh_stem: str
    """USD-safe identifier derived from the marked mesh file's stem.

    This is what gets matched against USD prim path components after
    conversion, since the URDF importer names collision prims after mesh
    file stems.
    """

    mesh_filename: str
    """The mesh filename exactly as written in the URDF's ``<mesh filename=...>``
    (kept only for error messages)."""

    resolution: int | None = None
    """``resolution`` attribute of ``<sdf>``. Maps to ``sdf_resolution``."""

    margin: float | None = None
    """``margin`` attribute of ``<sdf>``. Maps to ``sdf_margin``."""

    narrow_band_thickness: float | None = None
    """``narrow_band_thickness`` / ``narrowBandThickness`` attribute of ``<sdf>``.
    Maps to ``sdf_narrow_band_thickness``."""

    subgrid_resolution: int | None = None
    """``subgrid_resolution`` / ``subgridResolution`` attribute of ``<sdf>``.
    Maps to ``sdf_subgrid_resolution``."""


def _usd_safe_identifier(name: str) -> str:
    """Mirror the conservative subset of USD identifier rules the URDF importer uses.

    USD prim names cannot start with a digit and may only contain
    alphanumerics and underscores. The importer falls back to a scheme like
    this when naming prims after mesh file stems, so marker stems must be
    computed the same way to match.
    """
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not safe or not (safe[0].isalpha() or safe[0] == "_"):
        safe = f"mesh_{safe}"
    return safe


def _resolve_urdf_mesh_path(urdf_path: Path, mesh_filename: str) -> Path:
    mesh_path = Path(mesh_filename)
    if mesh_path.is_absolute():
        return mesh_path
    return (urdf_path.parent / mesh_path).resolve()


def _parse_optional_int(value: str | None) -> int | None:
    return None if value is None else int(value)


def _parse_optional_float(value: str | None) -> float | None:
    return None if value is None else float(value)


def parse_urdf_sdf_collision_markers(urdf_path: str) -> list[UrdfSdfCollisionMarker]:
    """Scan a URDF for ``<sdf>`` markers nested under ``<collision>`` and return them.

    This function's own body is a plain XML scan with no Isaac Sim / USD
    dependency. Note that importing it via the ``uwlab.sim.converters``
    package still requires an Isaac Sim app to already be running, same as
    every other converter in this package (``mesh_converter.py`` imports
    ``isaacsim.core`` at module scope) -- only :func:`apply_urdf_sdf_collision_markers`
    needs the app for its own work, but the package as a whole does not
    support being imported before one exists.

    Args:
        urdf_path: Path to the source URDF that was (or will be) converted.

    Returns:
        One :class:`UrdfSdfCollisionMarker` per ``<collision>`` block that carries
        a ``<sdf>`` child with a resolvable ``<geometry/mesh filename=...>``.
        Empty if the URDF has no ``<sdf>`` markers.
    """
    path = Path(urdf_path)
    root = ET.parse(path).getroot()
    markers: list[UrdfSdfCollisionMarker] = []
    for collision in root.findall(".//collision"):
        sdf_tag = collision.find("sdf")
        if sdf_tag is None:
            continue
        mesh_tag = collision.find("geometry/mesh")
        if mesh_tag is None or not mesh_tag.get("filename"):
            continue
        mesh_filename = str(mesh_tag.get("filename"))
        mesh_path = _resolve_urdf_mesh_path(path, mesh_filename)
        markers.append(
            UrdfSdfCollisionMarker(
                mesh_stem=_usd_safe_identifier(mesh_path.stem),
                mesh_filename=mesh_filename,
                resolution=_parse_optional_int(sdf_tag.get("resolution")),
                margin=_parse_optional_float(sdf_tag.get("margin")),
                narrow_band_thickness=_parse_optional_float(
                    sdf_tag.get("narrow_band_thickness") or sdf_tag.get("narrowBandThickness")
                ),
                subgrid_resolution=_parse_optional_int(
                    sdf_tag.get("subgrid_resolution") or sdf_tag.get("subgridResolution")
                ),
            )
        )
    return markers


def apply_urdf_sdf_collision_markers(
    usd_path: str,
    source_urdf_path: str,
    markers: list[UrdfSdfCollisionMarker],
) -> dict[str, list[str]]:
    """Stamp PhysX SDF mesh-collision properties onto the USD prim(s) each marker targets.

    Requires an Isaac Sim app to already be running. ``pxr`` and
    ``isaaclab.sim.schemas`` are imported lazily inside this function rather
    than at module scope, so that a failure here is the first Isaac-specific
    import this module performs (mirroring the reference implementation).

    Args:
        usd_path: Path to the USD produced by ``UrdfConverter`` for ``source_urdf_path``.
            The actual edit target is resolved from this (see layer trap below).
        source_urdf_path: Path to the URDF ``markers`` were parsed from. Only used
            for error/log messages.
        markers: Markers from :func:`parse_urdf_sdf_collision_markers`. A no-op if empty.

    Returns:
        Mapping from mesh stem to the list of prim paths SDF properties were applied
        to, for callers that want to log or verify what happened.

    Raises:
        RuntimeError: If the target USD fails to open, or if any marker matches zero
            collision prims on the stage (see module docstring, trap 3). Nothing is
            written to disk when this is raised.
    """
    if not markers:
        return {}

    from pxr import Usd, UsdPhysics

    from isaaclab.sim.schemas import SDFMeshPropertiesCfg, define_mesh_collision_properties

    # --- Trap 1: physics for a URDF-converted asset commonly lives on a
    # separate sublayer, not the raw USD. Editing the raw USD when that
    # sublayer exists silently edits a layer nothing reads collision
    # approximation from.
    raw_usd_path = Path(usd_path)
    physics_usd_path = raw_usd_path.parent / "configuration" / f"{raw_usd_path.stem}_physics.usd"
    edit_usd_path = physics_usd_path if physics_usd_path.exists() else raw_usd_path

    stage = Usd.Stage.Open(str(edit_usd_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Failed to open USD while applying URDF SDF markers: {edit_usd_path}")
    stage.Load()

    marker_by_stem = {marker.mesh_stem: marker for marker in markers}

    # --- Trap 2: walk instance proxies (a plain stage.Traverse() misses
    # instanced colliders), restrict to prims that already carry
    # MeshCollisionAPI (i.e. already-recognized colliders), and match by
    # mesh stem against any path component. Track /colliders/ and
    # non-/colliders/ matches separately, per marker, so a /colliders/ match
    # for one marker never displaces a fallback match that is the only match
    # another marker has.
    collider_matches: dict[str, list[Usd.Prim]] = {stem: [] for stem in marker_by_stem}
    fallback_matches: dict[str, list[Usd.Prim]] = {stem: [] for stem in marker_by_stem}
    all_collision_prim_paths: list[str] = []

    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        path = prim.GetPath().pathString
        all_collision_prim_paths.append(path)
        path_parts = [part for part in path.split("/") if part]
        stem = next((part for part in path_parts if part in marker_by_stem), None)
        if stem is None:
            continue
        if path.startswith("/colliders/"):
            collider_matches[stem].append(prim)
        else:
            fallback_matches[stem].append(prim)

    applied: dict[str, list[str]] = {}
    missing: list[UrdfSdfCollisionMarker] = []
    for stem, marker in marker_by_stem.items():
        prims = collider_matches[stem] or fallback_matches[stem]
        if not prims:
            missing.append(marker)
            continue
        applied[stem] = []
        for prim in prims:
            define_mesh_collision_properties(
                str(prim.GetPath()),
                SDFMeshPropertiesCfg(
                    sdf_margin=marker.margin,
                    sdf_narrow_band_thickness=marker.narrow_band_thickness,
                    sdf_resolution=marker.resolution,
                    sdf_subgrid_resolution=marker.subgrid_resolution,
                ),
                stage=stage,
            )
            applied[stem].append(str(prim.GetPath()))

    # --- Trap 3: raise, don't warn. Nothing is saved if we get here.
    if missing:
        found_desc = "\n".join(f"  {p}" for p in all_collision_prim_paths) or "  (none)"
        missing_desc = ", ".join(f"{m.mesh_filename!r} (stem={m.mesh_stem!r})" for m in missing)
        raise RuntimeError(
            f"URDF SDF marker(s) {missing_desc} from {source_urdf_path!r} matched zero collision "
            f"prims in {edit_usd_path}. A renamed or moved mesh silently degrades its collider "
            "from sdf to the importer's default convexHull approximation -- refusing to continue "
            f"instead of falling back.\nPrims carrying UsdPhysics.MeshCollisionAPI found on stage:"
            f"\n{found_desc}"
        )

    stage.GetRootLayer().Save()

    for stem, prim_paths in applied.items():
        print(
            f"[sdf_markers] applied SDF collision properties to {len(prim_paths)} prim(s) for "
            f"mesh stem {stem!r}: {prim_paths}",
            flush=True,
        )
    return applied


if __name__ == "__main__":
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description=(
            "Stamp PhysX SDF mesh-collision approximation onto a USD already converted from a "
            "URDF containing <sdf> collision markers."
        )
    )
    parser.add_argument("urdf", type=str, help="Path to the source URDF (scanned for <sdf> markers).")
    parser.add_argument(
        "usd",
        type=str,
        help=(
            "Path to the converted USD to stamp. The actual edit target follows the "
            "configuration/<stem>_physics.usd sublayer convention if that layer exists."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    markers = parse_urdf_sdf_collision_markers(args_cli.urdf)
    if not markers:
        print(f"[sdf_markers] no <sdf> collision markers found in {args_cli.urdf}; nothing to do.")
    else:
        print(f"[sdf_markers] found {len(markers)} <sdf> marker(s) in {args_cli.urdf}: {markers}")
        result = apply_urdf_sdf_collision_markers(args_cli.usd, args_cli.urdf, markers)
        print(f"[sdf_markers] done: {result}")

    simulation_app.close()
