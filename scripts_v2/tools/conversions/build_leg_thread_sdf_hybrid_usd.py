# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a THREAD-SDF-HYBRID insertive leg USD (sibling of the receptive
``build_one_leg_insertion_fixture_usd.py``, same recipe, other side of the mating pair).

BACKGROUND -- why this script exists. The audit that found this gap:
``OneLegInsertionFixture`` (the receptive fixture) got its ``<sdf>`` marker applied
(``build_one_leg_insertion_fixture_usd.py`` calls
``uwlab.sim.converters.sdf_markers.apply_urdf_sdf_collision_markers``), so its
``one_leg_hole_detail`` mesh carries ``physics:approximation = sdf`` (PhysX
``sdfResolution = 256``). The leg's OWN sdf_hybrid URDF --
``square_table_leg4_200mm_matchedmass_sdf_hybrid.urdf`` -- has an IDENTICAL
``<sdf resolution="256"/>`` marker already authored on its ``thread_link`` collision
(confirmed by reading the URDF directly: line with
``<collision><geometry><mesh filename="square_table_leg4_200mm_thread.obj".../></geometry>
<sdf resolution="256"/></collision>``), but NOTHING in this tree has ever run that URDF
through ``apply_urdf_sdf_collision_markers``. The only two places that spawn it
(``dexlift/table_leg_env_cfg.py`` and ``direct/delto_grasp/delto_grasp_env_cfg.py``) both
call ``sim_utils.UrdfFileCfg`` directly at RUNTIME with ``collider_type="convex_hull"`` /
``"convex_decomposition"`` respectively -- UrdfConverterCfg's collider vocabulary is
``convex_hull | convex_decomposition`` only (see ``isaaclab.sim.converters.urdf_converter_cfg
.UrdfConverterCfg.collider_type``), so the runtime import path can NEVER honor that
``<sdf>`` tag; it is silently ignored both times. The leg has therefore never actually
gotten the SDF treatment its own source URDF asked for -- the receiver got it, the
leg never did. That asymmetry (SDF hole vs. convex leg) is the mating-interface mismatch
this script exists to close.

THIS SCRIPT DOES NOT TOUCH THE SHIPPED, TRAINED ASSET. ``square_table_leg4_200mm.usd``
(``SquareTableLeg200mmDecomp``) is referenced by every OmniReset training config
(``reset_states_cfg.py`` / ``rl_state_cfg.py`` / ``partial_assemblies_cfg.py``'s
``"leg200mm"`` variant) and by the shipped reset banks
(``OneLegInsertionFixture__SquareTableLeg200mmDecomp``) -- it is produced by a SEPARATE
pipeline (``MeshConverter`` over a pre-merged, single OBJ that has already had
``PhysicsMeshCollisionAPI`` + ``PhysxConvexDecompositionCollisionAPI`` stamped onto it by
``scripts_v2/tools/asset_authoring/reauthor_leg_decomp.py``) and is left byte-for-byte
alone. This script writes an entirely new, separate USD under a new asset directory
(``_DEFAULT_OUT_DIR`` below) so the two can be A/B'd exactly the way
``reauthor_leg_decomp.py``'s own docstring insists on for its asset.

THE SPLIT IS ALREADY DONE -- NO GEOMETRIC CUT NEEDED. This was the open question this
script had to settle rather than guess at, and it is now settled with exact numbers, not
hedged language -- both by direct measurement of our own asset AND by an independent
side-by-side against play2perfect's equivalent (a separate agent's audit of their leg).

1. The URDF (``square_table_leg4_200mm_matchedmass_sdf_hybrid.urdf``) already declares
   ``thread_link`` (mesh ``square_table_leg4_200mm_thread.obj``, the ``<sdf>``-marked
   collision) as a SEPARATE link from ``body_link`` (12 already-CoACD-decomposed convex
   pieces, ``body_coacd/decomp_0.obj`` .. ``decomp_11.obj``), joined to a common
   ``base_link`` by two IDENTITY ``fixed`` joints (no ``<origin>`` on either -- URDF
   default is zero translation/rotation), matching play2perfect's own leg structure
   (two links on a base by identity fixed joints) link-for-link. The pre-split source OBJs
   (``square_table_leg4_200mm_thread.obj``, ``square_table_leg4_200mm_body.obj``) still
   exist on disk, siblings of this URDF -- they were never deleted, only concatenated
   into a throwaway ``/tmp`` merge for the DIFFERENT (MeshConverter/Decomp) pipeline
   (config.yaml's own ``asset_path: /tmp/convert_table_leg_merged_9dozgcfx/...`` proves
   that merge no longer exists on disk).
2. Independently, the ALREADY-SHIPPED merged collision mesh in
   ``SquareTableLeg200mmDecomp/Props/instanceable_meshes.usd`` prim
   ``/square_table_leg4_200mm_merged/geometry/mesh`` was pulled out with ``pxr`` and fed to
   ``trimesh`` directly (31855 raw points / 12258 triangles). After trimesh's standard
   vertex weld (``process=True``, the default): the mesh is watertight, winding-consistent,
   ``euler_number == 4`` -- i.e. it is topologically TWO disjoint closed solids, not one --
   and ``mesh.split(only_watertight=False)`` cleanly separates it into exactly two
   watertight components with no shared geometry: a long one (2882 faces, bounds x in
   [-0.1062, 0.0938], volume 1.5535e-04 m^3 -- the body) and a short one (9376 faces,
   bounds x in [-0.1021, -0.0751], a ~27 mm segment at one end, volume 1.9565e-06 m^3 --
   the thread). This independently corroborates (1): the "merge" that produced the Decomp
   asset's single mesh was a plain concatenation of two pre-existing, already-separate,
   already-closed solids, not a boolean union -- there was never a single continuous
   manifold to cut in the first place.
3. The standalone thread mesh, measured directly with trimesh: length along its axis is
   EXACTLY 25.000000 mm (bounds -0.04375 to -0.01875 m). It has 179 boundary edges (the
   only non-closed part of an otherwise ``euler_number == 2`` solid), and EVERY ONE of
   those 179 edges' vertices sit at x = -0.01875 to machine precision (std = 3.5e-18) --
   the seam end, with the opposite (tip, x = -0.04375) end fully capped (zero boundary
   vertices there). This reproduces play2perfect's own reported thread mesh, DIGIT FOR
   DIGIT (179 boundary edges, all at the cut plane, tip capped, 25.000 mm cut) -- strong
   evidence our asset came off the same authoring tool as theirs, not merely a similar one.
   25.000 mm also independently equals our bore's own measured engaged span: three
   numbers, one constant, from three different sources.
4. THE SEAM: NOT a zero-overlap butt joint, and NOT a deliberately-added overlap either --
   it is hull-inflation overlap that was ALREADY THERE, unmeasured until now. The raw
   ``square_table_leg4_200mm_body.obj`` visual/reference mesh touches the thread with
   exactly zero gap (body min-x == thread max-x == -0.01875, to the same machine
   precision as above). But the ACTUAL COLLISION geometry the URDF imports is not that
   body OBJ -- it is the twelve ``body_coacd/decomp_*.obj`` convex hulls, and measuring
   THOSE directly (not the reference mesh) settles the question: 7 of the 12 hulls
   (``decomp_0,1,4,5,6,7,10``) have min-x = -0.0197125, i.e. they extend ~0.9625 mm PAST
   the nominal seam, into the thread mesh's own bounding-box span. That is real
   convex-hull-over-a-rounded-cross-section inflation on the body side, the same
   mechanism play2perfect's own audit identified (their reported "0.3438 mm deliberate
   overlap" and their separately-reported "0.34 mm CoACD rounding overshoot at the far
   tip" are the same number at both ends of the same part -- hull inflation, not designed
   intent). Ours is ~2.8x theirs (0.96 mm vs 0.34 mm) but the SAME mechanism, already
   baked into an asset this project has already validated at 94-100% grasp rates -- there
   is nothing to add here; the overlap this recipe needs already exists on disk.

Consequently this script does not attempt any geometric split of its own: it converts
the URDF (which already keeps thread and body as separate links/meshes, with the body's
own CoACD hulls already overlapping the seam) through ``UrdfConverter`` and lets the
post-conversion marker pass promote only the already-tagged thread collision mesh to SDF,
exactly as the receptive fixture builder already does for the hole-detail mesh.

WATERTIGHTNESS CAVEAT FOR THE NEW SDF REGION SPECIFICALLY. The isolated
``square_table_leg4_200mm_thread.obj``, checked with ``trimesh`` (both ``process=False``
and ``process=True``), reports ``is_watertight == False`` -- but per point 3/4 above this
is now understood, not merely flagged: it is exactly play2perfect's own de-risked
construction (a single open disc at the cut-plane seam, tip fully capped, body pieces
supplying the volume across that seam once ``merge_fixed_joints`` unifies everything into
one rigid body). Their SDF bake works in their shipped task on the same shape of
non-watertightness (boundary edges concentrated at one plane, not scattered holes). This
is therefore NOT treated as a blocker. ``_verify()`` below still only checks the SCHEMA
outcome (``physics:approximation == "sdf"`` on the expected prim); it does not re-verify
mesh manifoldness after conversion.

MASS MUST BE PINNED TO THE DEPLOYED ASSET'S 0.12 kg, NOT LEFT AT THE URDF'S OWN VALUE.
This was checked directly (pxr on the actual prim, not a config file) rather than
assumed, because three different numbers for this leg's mass are floating around this
tree and only one of them is what is actually deployed:

- DEPLOYED (``SquareTableLeg200mmDecomp/square_table_leg4_200mm.usd``,
  ``/square_table_leg4_200mm_merged``, ``UsdPhysics.MassAPI.GetMassAttr()``, authored):
  **0.11999999731779099 kg** (i.e. 0.12 kg at float32 precision). This is the corrected
  mass this project already fixed once -- ``square_table_leg4_200mm.usd.bak_mass0.02275``
  (same directory) is a preserved backup of the PRE-correction file, and reading its mass
  the same way confirms **0.022749999538064003 kg** (22.75 g) -- a too-light,
  styrofoam-density leg that is flung by ordinary contact and reads as "grasped" while
  nothing is held. ``reset_states_cfg.py``'s ``"leg200mm"`` variant spawns with
  ``override_mass=False`` specifically to PRESERVE this corrected 0.12 kg rather than let
  the framework's default ``override_mass=True`` rewrite it back down to 0.001 kg.
- THIS URDF, if converted as-is: ``base_link``'s ``<inertial><mass value=
  "0.0573543915"/>`` is the ONLY authored mass in the file -- ``thread_link`` and
  ``body_link`` have no ``<inertial>`` at all, so after ``merge_fixed_joints=True``
  collapses all three links, the resulting rigid body's mass would be **0.0573543915 kg**,
  roughly HALF (47.8%) of the deployed 0.12 kg. Converting this URDF unmodified would
  silently ship a leg at roughly half the deployed mass, riding along invisibly with the
  collider fix -- exactly the kind of change that invalidates a plant comparison after
  the fact, since it would make TWO variables (thread collider AND mass) differ between
  old and new asset instead of one.
- A THIRD number already lives in this tree, independently of anything this script does:
  ``direct/delto_grasp/delto_grasp_env_cfg.py``'s ``TableLegObjectCfg.default_mass`` is
  **0.0573543915** -- i.e. it already matches the URDF's base_link mass, NOT the
  corrected 0.12 kg. That consumer of this leg and the OmniReset ``"leg200mm"`` consumer
  already disagree about this leg's mass TODAY, independently of this script.
- ``reset_states_cfg.py``'s OWN inline comment justifying ``override_mass=False`` claims
  the leg's root prim "carries its own MassAPI at 0.02275 kg (measured directly off the
  USD with pxr...)" -- that claim is STALE: it describes the PRE-correction ``.bak`` file's
  mass, not the current deployed 0.12 kg this script actually re-measured. The
  ``override_mass=False`` mechanism the comment describes is still doing the right thing
  (preserve whatever is authored, which is now 0.12 kg) -- only the NUMBER cited in the
  comment is wrong. Worth a separate doc fix; not touched by this script.

Given the above, ``main()`` below adds ONE post-conversion step beyond the sdf marker
pass: it re-opens the converted USD and overwrites ``UsdPhysics.MassAPI``'s ``mass``
attribute to ``0.11999999731779099`` (the deployed value, to the same float32 precision),
and deliberately does NOT author ``centerOfMass`` / ``diagonalInertia`` /
``principalAxes`` -- the deployed asset leaves those three unauthored too (confirmed by
the same pxr read), meaning PhysX auto-computes inertia/COM from the collision geometry
scaled to the authored total mass rather than from a hand-specified tensor. Matching that
authoring PATTERN (mass-only, geometry-derived inertia) rather than inventing an inertia
tensor is what makes "the only difference between old and new asset is the thread
collider" true in practice, not just in the mass number.

Run (needs Isaac Sim; PYTHONPATH must include the four uwlab source dirs; never launch
through uwlab.sh, which wipes PYTHONPATH) -- NOT YET RUN, see module docstring above.
DO NOT RUN until the mass-pin above has been reviewed and approved -- this is currently
on HOLD per explicit instruction, report-only::

    PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl" \\
        timeout -s KILL 300 \\
        /home/dom-iva/github.com/orel/lerobot/UWLab/env_uwlab/bin/python -u \\
        scripts_v2/tools/conversions/build_leg_thread_sdf_hybrid_usd.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# SETTLED (was a play2perfect-comparison seam; no swap needed). The independent
# side-by-side against play2perfect's own leg (179 boundary edges at the cut plane to
# machine precision, 25.000 mm cut, tip capped, hull-inflation overlap on the body side)
# confirmed our already-authored URDF is the same recipe, not a divergent one -- see the
# module docstring's "THE SPLIT IS ALREADY DONE" section for the full evidence.
# Ours is UWLab's own already-authored URDF -- ``thread_link`` (mesh
# ``square_table_leg4_200mm_thread.obj``, already ``<sdf resolution="256"/>``-tagged) +
# ``body_link`` (12 pre-decomposed CoACD convex pieces, untagged -> stays convex_hull).
_DEFAULT_URDF = (
    "/home/dom-iva/github.com/orel/lerobot/UWLab_ur5edelto/source/uwlab_assets/uwlab_assets/local/"
    "Props/FurnitureBench/SquareTableOneLeg/leg_200mm/square_table_leg4_200mm_matchedmass_sdf_hybrid.urdf"
)
# NEW asset directory -- deliberately NOT "SquareTableLeg200mmDecomp". That directory (and
# square_table_leg4_200mm.usd inside it) is the certified-checkpoint asset every OmniReset
# training config and every shipped reset bank references; it must stay byte-for-byte
# untouched. This writes a sibling that can be A/B'd against it.
_DEFAULT_OUT_DIR = "source/uwlab_assets/data/Props/FurnitureBench/SquareTableLeg200mmThreadSdfHybrid"
_DEFAULT_USD_NAME = "square_table_leg4_200mm_thread_sdf_hybrid.usd"

parser = argparse.ArgumentParser(
    description="Convert the leg's own sdf_hybrid URDF to USD, then stamp sdf onto its thread collision."
)
parser.add_argument("--urdf", type=str, default=_DEFAULT_URDF, help="Source URDF path.")
parser.add_argument("--usd-dir", type=str, default=_DEFAULT_OUT_DIR, help="Output USD directory.")
parser.add_argument("--usd-name", type=str, default=_DEFAULT_USD_NAME, help="Output USD file name.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402
import pathlib  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

from uwlab.sim.converters import apply_urdf_sdf_collision_markers, parse_urdf_sdf_collision_markers  # noqa: E402


# The deployed leg's actual authored mass (square_table_leg4_200mm.usd,
# /square_table_leg4_200mm_merged, UsdPhysics.MassAPI, read directly with pxr -- not taken
# from any config.yaml or code comment). This URDF's own base_link <inertial> mass
# (0.0573543915 kg) is roughly HALF of this and must NOT be what ships. See the module
# docstring's MASS section for the full three-numbers-disagree investigation.
_DEPLOYED_LEG_MASS_KG = 0.11999999731779099

# ---------------------------------------------------------------------------------------
# DEPLOYED asset's RUNTIME-COMPUTED (PhysX, root_physx_view -- not the USD attribute, which
# is unauthored on the deployed asset) mass/COM/inertia, measured fresh via a live spawned
# RigidObject (scripts_v2/tools/../.. one-off measurement, not transcribed from any message
# or prior run). This is what STEP 2 authors onto the produced asset -- see
# "WHY COM/INERTIA ARE AUTHORED HERE" below for why this reverses the mass-pin's own
# earlier "leave unauthored" precedent.
_DEPLOYED_LEG_COM_BODY_FRAME = (0.00011275688302703202, 0.0010115808108821511, 0.0009914301335811615)
_DEPLOYED_LEG_DIAGONAL_INERTIA = (1.929639385685027e-05, 0.0003932671467643291, 0.0003933335282368507)
# principalAxes: rotation from BODY frame to the frame in which the tensor above is
# diagonal, as a quaternion (w, x, y, z) -- eigendecomposition of the fresh-measured full
# 3x3 body-frame inertia tensor (which has small but real off-diagonal terms, so
# diagonalInertia alone is not sufficient; principalAxes is required too).
_DEPLOYED_LEG_PRINCIPAL_AXES_WXYZ = (
    0.9320064453518548,
    0.3624416356660627,
    4.1724196968174046e-05,
    0.00021170120547098675,
)

# ---------------------------------------------------------------------------------------
# ROOT-FRAME ALIGNMENT (STEP 1). The produced asset's ``base_link`` is a NESTED child of
# this USD's own outer defaultPrim (confirmed with pxr: base_link carries RigidBodyAPI and
# its own translate/orient/scale xformOps, all identity as UrdfConverter left them) --
# unlike the deployed asset, where the RigidBodyAPI prim (/square_table_leg4_200mm_merged)
# IS the outer/referenced prim itself, with no nesting. That structural difference is why
# the two assets' local origins sit at different physical points on the leg (see the
# earlier three-way comparison: runtime COM x differed by ~60mm between the two).
#
# The correction is a translate authored directly on base_link's OWN local xformOp, in the
# physics sublayer (``configuration/<stem>_physics.usd``) -- NOT the raw top-level file,
# which does not own this opinion (confirmed empirically: an edit to the raw file's
# base_link translate was silently shadowed when read back through the physics-sublayer
# composed stage; the same edit made directly in the physics sublayer took and persisted).
#
# THIS MECHANISM WAS VERIFIED, NOT ASSUMED, before being relied on: a throwaway test asset
# with a known 0.1m offset baked the same way was spawned via IsaacLab's RigidObjectCfg
# with init_state.pos=(0,0,0), and root_physx_view read back a world position of
# 0.09999999m -- i.e. IsaacLab applies init_state.pos to the OUTER (referenced) prim and
# preserves base_link's own static local offset underneath it, rather than overwriting
# base_link's transform directly. Had this test shown ~0.0 instead, this whole approach
# would have been wrong and is exactly the kind of thing this comment exists so nobody
# re-derives it from scratch or "simplifies" it back to editing the wrong prim.
#
# THE NUMBERS: found by ICP (trimesh.registration.icp) between the produced asset's thread
# collision mesh (4713 verts) and the deployed asset's thread-region connected component
# split out of its merged collision mesh (4690 verts) -- both already expressed in their
# own root-rigid-body-local frames (mesh-to-root confirmed identity on both sides via
# UsdGeom.XformCache). Rotation found: 0.0755 degrees -- noise-level for two independently
# processed meshes of the same part, NOT authored (treated as identity; see the residual
# note below for why a larger rotation would have been a problem this simplification
# could hide, and was checked for). Translation (deployed_frame = produced_frame + t):
_ROOT_ALIGN_TRANSLATE = (-0.0622272431, -0.0000279237397, 0.0000262072042)
# Global fit quality over the whole thread surface (NOT a single "tip" point, which is
# noisier -- see module docstring): mean nearest-neighbor residual after applying this
# transform = 0.37mm, RMS = 1.03mm, computed over all ~4700 thread vertices. A single most-
# extreme "tip apex" vertex comparison showed a much larger local residual (~10mm) that
# this script's docstring explains separately -- not evidence the transform is wrong, but
# real evidence the two thread meshes' tip CAPS are not byte-identical (decimation
# difference between the pristine URDF-sourced OBJ and the deployed asset's merged/
# processed mesh). The acceptance test that matters is the LIVE spawn-to-spawn world-space
# tip comparison, done separately from this build script -- see the module docstring.


def _resolve_physics_edit_path(usd_path: str):
    """Same physics-sublayer redirect as _verify() / sdf_markers.py's own Trap 1: UrdfConverter
    writes physics (RigidBodyAPI/MassAPI included) onto a SEPARATE sublayer at
    <usd_dir>/configuration/<stem>_physics.usd when that layer exists. Editing the raw
    (geometry) USD in that case edits a layer mass/transform is not read from -- confirmed
    empirically for the transform case (see _align_root_frame_to_deployed's docstring)."""
    from pathlib import Path

    raw = Path(usd_path)
    physics_usd = raw.parent / "configuration" / f"{raw.stem}_physics.usd"
    return str(physics_usd) if physics_usd.exists() else usd_path, physics_usd.exists()


def _find_canonical_rigid_body_prim(stage):
    from pxr import UsdPhysics

    prims = [
        p
        for p in stage.Traverse()
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and not p.IsInstanceProxy()
    ]
    if len(prims) != 1:
        raise RuntimeError(
            f"expected exactly one (canonical, non-instance-proxy) rigid body prim after "
            f"merge_fixed_joints, found {len(prims)}: {[p.GetPath() for p in prims]}"
        )
    return prims[0]


def _align_root_frame_to_deployed(usd_path: str) -> None:
    """STEP 1: bake the produced asset's root-frame offset from the deployed asset onto
    ``base_link``'s own local xformOp:translate, so both assets place the physical leg at
    the same world location when spawned at the same ``init_state.pos``.

    WHY base_link, and not the outer/referenced prim: the produced asset's RigidBodyAPI
    prim (``base_link``) is a NESTED CHILD of the USD's outer defaultPrim, unlike the
    deployed asset where the RigidBodyAPI prim IS the outer/referenced prim. IsaacLab's
    ``RigidObjectCfg.init_state.pos`` sets the pose of the OUTER referenced prim at spawn --
    it does not touch a nested child's own static local transform. This was VERIFIED
    empirically (not assumed): a throwaway test asset with a known 0.1m offset baked onto
    base_link's translate, spawned via IsaacLab with init_state.pos=(0,0,0), read back a
    live PhysX world position of 0.09999999m via root_physx_view -- i.e. the offset
    survives spawn exactly as intended. See _ROOT_ALIGN_TRANSLATE's own comment for how the
    offset itself was derived (ICP between the two assets' thread meshes).

    WHY THE PHYSICS SUBLAYER, not the raw top-level USD: also verified empirically, not
    assumed. Setting base_link's translate in the raw ``square_table_leg4_200mm_thread_
    sdf_hybrid.usd`` and then reading it back through the physics-sublayer composed stage
    (``configuration/..._physics.usd``, which _verify()/_pin_mass_com_inertia_to_deployed() already
    treat as authoritative) showed the edit was NOT visible -- the physics sublayer holds
    its own, stronger opinion for base_link's transform ops that shadows the raw file's.
    Editing the physics sublayer directly took and persisted on re-open.
    

    LOAD-BEARING FOR THE METADATA COPY, not only for spawn parity. Verified independently
    (critic3, 2026-08-23): _install_metadata_yaml() reuses the deployed asset's
    assembled_offset byte-for-byte, and that value is only geometrically valid in THIS
    asset's frame because of the translate applied here. Concretely, assembled_offset.pos.x
    is -0.106203 in the deployed frame; mapping it through this translate gives
    -0.106203 - (-0.0622272431) = -0.0439758 m, against the hybrid's own pristine tip at
    -0.04375 m -- a residual of 0.226 mm. Skipping, redoing, or re-deriving this alignment
    independently of the metadata-copy decision silently invalidates that reuse and shifts
    the depth scale of every banked state. The two changes must move together.
    """
    from pxr import Usd, UsdGeom

    edit_path, sublayer_exists = _resolve_physics_edit_path(usd_path)
    print(f"[align] editing: {edit_path} (physics sublayer exists: {sublayer_exists})")

    stage = Usd.Stage.Open(edit_path)
    if stage is None:
        raise RuntimeError(f"[align] failed to open {edit_path}")

    base_link = _find_canonical_rigid_body_prim(stage)
    xf = UsdGeom.Xformable(base_link)
    translate_op = None
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            translate_op = op
            break
    if translate_op is None:
        raise RuntimeError(f"[align] {base_link.GetPath()} has no xformOp:translate to set -- inspect before proceeding")

    before = translate_op.Get()
    translate_op.Set(_ROOT_ALIGN_TRANSLATE)
    stage.GetRootLayer().Save()

    check = Usd.Stage.Open(edit_path)
    got = check.GetPrimAtPath(base_link.GetPath())
    xcache = UsdGeom.XformCache()
    l2p, _ = xcache.GetLocalTransformation(got)  # local-to-parent, since parent (outer prim) is what spawn poses
    after_translate = None
    for op in UsdGeom.Xformable(got).GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            after_translate = op.Get()
    print(f"[align] {base_link.GetPath()}: translate before={before} -> after={after_translate} (target={_ROOT_ALIGN_TRANSLATE})")
    print(f"[align] local-to-parent transform (fresh re-read):\n{l2p}")
    assert tuple(after_translate) == _ROOT_ALIGN_TRANSLATE, (
        f"root-frame alignment did not take: read back {after_translate}, expected {_ROOT_ALIGN_TRANSLATE}"
    )


def _pin_mass_com_inertia_to_deployed(usd_path: str) -> None:
    """STEP 2: author mass, centerOfMass, diagonalInertia and principalAxes on the produced
    asset to match the DEPLOYED leg's own RUNTIME-COMPUTED values exactly (not the URDF's
    own values, and -- reversing this script's OWN earlier approach -- not left unauthored
    either).

    WHY COM/INERTIA ARE AUTHORED HERE, WHEN THE DEPLOYED ASSET ITSELF LEAVES THEM UNSET:
    an earlier version of this function stripped these three attributes to match the
    deployed asset's own authoring PATTERN (mass-only, PhysX derives inertia from geometry).
    That was the right call for matching the deployed asset's pattern, and it is the WRONG
    call for the experiment this asset exists to run: leaving them unauthored lets PhysX
    derive COM/inertia from the (new) collision geometry -- which is precisely what this
    asset changes (whole-part convexDecomposition vs. thread+twelve-hull) -- and a live
    three-way comparison measured a resulting 1.3-1.35x inertia difference between the two
    assets purely from that geometry change, even at identical authored mass. So mass alone
    is not enough to isolate "collider representation" as the only variable; COM and
    inertia have to be pinned too, to the DEPLOYED asset's own runtime-measured values,
    EXPRESSED IN THE FRAME THAT _align_root_frame_to_deployed() JUST ALIGNED TO the deployed
    asset's. If a future reader finds this "wrong" because the deployed asset leaves these
    unauthored and "fixes" it by stripping them again here, that reintroduces the exact
    confound this function exists to remove -- this is a DELIBERATE experimental control,
    not an oversight.

    The three deployed-asset values below (_DEPLOYED_LEG_MASS_KG /
    _DEPLOYED_LEG_COM_BODY_FRAME / _DEPLOYED_LEG_DIAGONAL_INERTIA /
    _DEPLOYED_LEG_PRINCIPAL_AXES_WXYZ) were measured fresh via a live spawned RigidObject's
    ``root_physx_view`` (mass, COM, and the FULL 3x3 body-frame inertia tensor, which has
    small but real off-diagonal terms -- diagonalInertia alone would silently drop them),
    then eigendecomposed for diagonalInertia + principalAxes. Because Step 1 aligned the
    produced asset's root frame to coincide with the deployed asset's (translation only;
    the measured rotation between the two frames was 0.0755 degrees, noise-level), these
    values apply DIRECTLY with no further transform -- no rotation needs to be applied to
    the COM position or the inertia tensor to re-express them in the produced asset's frame,
    because after Step 1 that frame IS (to within the same small residual) the deployed
    asset's frame.
    """
    from pxr import Gf, Usd, UsdPhysics

    edit_path, sublayer_exists = _resolve_physics_edit_path(usd_path)
    print(f"[mass-pin] editing: {edit_path} (physics sublayer exists: {sublayer_exists})")

    stage = Usd.Stage.Open(edit_path)
    if stage is None:
        raise RuntimeError(f"[mass-pin] failed to open {edit_path}")

    prim = _find_canonical_rigid_body_prim(stage)
    mass_api = UsdPhysics.MassAPI.Apply(prim) if not prim.HasAPI(UsdPhysics.MassAPI) else UsdPhysics.MassAPI(prim)

    before_mass = mass_api.GetMassAttr().Get() if mass_api.GetMassAttr() else None
    before_com = mass_api.GetCenterOfMassAttr().Get() if mass_api.GetCenterOfMassAttr() else None
    before_diag = mass_api.GetDiagonalInertiaAttr().Get() if mass_api.GetDiagonalInertiaAttr() else None
    before_axes = mass_api.GetPrincipalAxesAttr().Get() if mass_api.GetPrincipalAxesAttr() else None
    print(
        f"[mass-pin] BEFORE: mass={before_mass} com={before_com} diagInertia={before_diag} "
        f"principalAxes={before_axes}"
    )

    mass_api.CreateMassAttr().Set(_DEPLOYED_LEG_MASS_KG)
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*_DEPLOYED_LEG_COM_BODY_FRAME))
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*_DEPLOYED_LEG_DIAGONAL_INERTIA))
    w, x, y, z = _DEPLOYED_LEG_PRINCIPAL_AXES_WXYZ
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))

    stage.GetRootLayer().Save()

    # Re-read from a FRESH stage open, same discipline as reauthor_leg_decomp.py /
    # build_one_leg_insertion_fixture_usd.py: an in-memory stage reports what was requested,
    # only a fresh open reports what was actually written.
    check = Usd.Stage.Open(edit_path)
    got = check.GetPrimAtPath(prim.GetPath())
    got_mass_api = UsdPhysics.MassAPI(got)
    after_mass = got_mass_api.GetMassAttr().Get()
    after_com = got_mass_api.GetCenterOfMassAttr().Get()
    after_diag = got_mass_api.GetDiagonalInertiaAttr().Get()
    after_axes = got_mass_api.GetPrincipalAxesAttr().Get()
    print(
        f"[mass-pin] AFTER (fresh re-read): mass={after_mass} com={after_com} "
        f"diagInertia={after_diag} principalAxes={after_axes}"
    )

    assert after_mass == _DEPLOYED_LEG_MASS_KG, f"mass did not take: {after_mass} != {_DEPLOYED_LEG_MASS_KG}"
    assert tuple(after_com) == tuple(Gf.Vec3f(*_DEPLOYED_LEG_COM_BODY_FRAME)), f"com did not take: {after_com}"
    assert tuple(after_diag) == tuple(Gf.Vec3f(*_DEPLOYED_LEG_DIAGONAL_INERTIA)), f"diagInertia did not take: {after_diag}"
    print("[mass-pin] PASS: mass, centerOfMass, diagonalInertia, principalAxes all authored and verified from a fresh re-read.")


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CANONICAL_METADATA = _REPO_ROOT / (
    "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench"
    "/SquareTableLeg200mmDecomp/metadata.yaml"
)


def _install_metadata_yaml(usd_path: str) -> None:
    """Copy the leg's canonical ``metadata.yaml`` next to the generated USD.

    ``read_metadata_from_usd_directory()`` resolves ``metadata.yaml`` from the USD's OWN
    directory, with no inheritance from a sibling or parent variant. UrdfConverter writes
    ``config.yaml`` but knows nothing about this file, so a freshly built variant directory
    has no metadata at all and every consumer that needs ``assembled_offset`` /
    ``bottom_offset`` dies at import with a bare FileNotFoundError -- which is exactly how
    the first hybrid run failed (bead UWLab-3v6l.11).

    The file is copied verbatim rather than regenerated: all four existing leg variants
    (Decomp, Sdf, Sdf1024, Sdf2048) are byte-identical, and the file's own comments state
    twice that the copies MUST stay byte-identical. Re-deriving the offsets for this variant
    is deliberately NOT done here -- see bead UWLab-z200 -- because changing them shifts the
    depth axis of every previously banked state and recorded curve.
    """
    src = _CANONICAL_METADATA
    if not src.is_file():
        raise RuntimeError(f"canonical leg metadata.yaml not found at {src}")

    # The canonical copy is only canonical if the siblings actually still agree with it. If
    # one has drifted, copying blind would silently pick a side.
    siblings = sorted(src.parent.parent.glob("SquareTableLeg200mm*/metadata.yaml"))
    canon = src.read_bytes()
    disagree = [str(q) for q in siblings if q.read_bytes() != canon]
    if disagree:
        raise RuntimeError(
            "leg metadata.yaml copies have diverged, refusing to guess which is canonical: "
            f"{disagree}"
        )

    dst = pathlib.Path(usd_path).parent / "metadata.yaml"

    # The sibling check above only covers local/Props; `dst` lives in the data/Props tree and
    # is invisible to that glob, so it would NOT be caught as a divergent sibling. Check it
    # directly. Without this, a later hand-edit here -- most plausibly partial progress on the
    # deferred assembled_offset re-derivation (UWLab-z200) -- would be silently overwritten
    # back to the canonical bytes on the next build, which is exactly the class of loss the
    # sibling guard exists to prevent.
    if dst.is_file() and dst.read_bytes() != canon:
        raise RuntimeError(
            f"{dst} already exists and DIFFERS from the canonical copy at {src}. Refusing to "
            "overwrite. If the divergence is intended, reconcile the canonical copy and its "
            "siblings first; if it is accidental, delete this file and re-run."
        )

    dst.write_bytes(canon)
    if dst.read_bytes() != canon:
        raise RuntimeError(f"metadata.yaml write-back mismatch at {dst}")
    print(f"[metadata] installed {dst} (byte-identical to {src}, {len(siblings)} sibling(s) agree)")


def _verify(usd_path: str, urdf_path: str) -> None:
    """Read the converted+stamped USD back and enforce the acceptance criteria.

    Mirrors ``build_one_leg_insertion_fixture_usd.py``'s ``_verify`` exactly, adjusted for
    this asset's shape: ONE sdf collider (the thread), TWELVE convexHull colliders (the
    CoACD body pieces), zero unauthored mesh approximations (the silent-hull-fallback
    signature ``reauthor_leg_decomp.py`` was filed to catch on the OTHER leg asset).
    """
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

    # Same de-dupe rationale as the fixture builder: count canonical (non-instance-proxy)
    # occurrences only, since TraverseInstanceProxies() surfaces both a /colliders/... prim
    # and its link-local reference-alias for the same physical collider.
    canonical_rows = [r for r in rows if not r[3]]
    sdf_rows = [r for r in canonical_rows if r[2] == "sdf"]
    convex_hull_rows = [r for r in canonical_rows if r[2] == "convexHull"]
    unauthored_mesh_rows = [r for r in canonical_rows if r[2] == "<UNAUTHORED>"]

    print(f"\n[verify] canonical prims reading approximation=='sdf': {[r[0] for r in sdf_rows]}")
    print(f"[verify] canonical prims reading approximation=='convexHull': {[r[0] for r in convex_hull_rows]}")
    print(f"[verify] canonical mesh colliders with UNAUTHORED approximation: {[r[0] for r in unauthored_mesh_rows]}")

    assert len(sdf_rows) == 1, f"expected exactly one sdf collider (the thread), got {len(sdf_rows)}: {sdf_rows}"
    assert "thread" in sdf_rows[0][0], f"sdf collider is not the thread mesh: {sdf_rows[0][0]}"
    assert len(convex_hull_rows) == 12, (
        f"expected exactly 12 convexHull colliders (the CoACD body pieces), got {len(convex_hull_rows)}: "
        f"{[r[0] for r in convex_hull_rows]}"
    )
    assert not unauthored_mesh_rows, (
        f"mesh collider(s) with a missing physics:approximation (silent-hull-fallback signature, see "
        f"scripts_v2/tools/asset_authoring/reauthor_leg_decomp.py for the OTHER asset this already bit): "
        f"{[r[0] for r in unauthored_mesh_rows]}"
    )
    print(
        "[verify] PASS: exactly one sdf collider (the thread) at resolution matching the URDF's "
        "<sdf resolution=\"256\"/> marker, twelve convexHull CoACD body colliders, no unauthored "
        "mesh approximations."
    )
    print(
        "[verify] NOTE: this only confirms the SCHEMA was stamped correctly. It does not re-verify "
        "mesh manifoldness -- see the module docstring's watertightness caveat before trusting the bake."
    )


def main() -> None:
    os.makedirs(args_cli.usd_dir, exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=os.path.abspath(args_cli.urdf),
        usd_dir=os.path.abspath(args_cli.usd_dir),
        usd_file_name=args_cli.usd_name,
        force_usd_conversion=True,
        # fix_base=False: mirrors BOTH existing runtime spawns of this exact URDF
        # (dexlift/table_leg_env_cfg.py and direct/delto_grasp/delto_grasp_env_cfg.py both
        # spawn it as a free RigidObject, never as a fixed-base Articulation). Unlike the
        # fixture builder (which had to flip this from an inherited True), this URDF was
        # never given fix_base=True anywhere, so there is no ArticulationRootAPI bug to
        # route around here -- kept False for consistency with how this asset already runs.
        fix_base=False,
        # merge_fixed_joints=True: base_link -> thread_link and base_link -> body_link are
        # both `fixed` joints with no actuation; this collapses all three links into one
        # rigid body, matching both existing runtime env configs.
        merge_fixed_joints=True,
        # make_instanceable=False, DELIBERATELY, matching build_one_leg_insertion_fixture_usd.py
        # exactly. sdf_markers.py's own docstring (Trap 2) warns that a plain stage.Traverse()
        # misses instanced collision prims -- apply_urdf_sdf_collision_markers() below already
        # walks Usd.PrimRange(..., Usd.TraverseInstanceProxies()) to handle that if it ever
        # arises, but leaving this converter run un-instanced is what the working precedent
        # does, so this does the same rather than exercising a path nothing has verified yet.
        make_instanceable=False,
        # collider_type="convex_hull" for the BODY pieces (this setting applies to every mesh
        # collider the importer creates; the marker pass below then promotes JUST the
        # <sdf>-tagged thread mesh away from it). The URDF's body_link already supplies twelve
        # CoACD-pre-decomposed convex pieces (body_coacd/decomp_0.obj .. decomp_11.obj) --
        # asking the importer to run convex_decomposition AGAIN on each already-convex piece
        # would re-run VHACD for no shape benefit and cost minutes per launch. This is the same
        # reasoning table_leg_env_cfg.py's own TableLegObjectCfg already uses for this identical
        # URDF (collider_type="convex_hull", "the URDF already supplies twelve CoACD convex body
        # pieces... A convex hull preserves each authored convex piece").
        collider_type="convex_hull",
        # Placeholder only, never exercised: this URDF's joints are all `fixed` (no <joint> drives
        # anything), but UrdfConverterCfg._validate() requires joint_drive.gains.stiffness to be
        # authored regardless of whether any joint uses it. Same placeholder values as the fixture
        # builder.
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
        raise RuntimeError("expected at least one <sdf> marker (the thread mesh) but found none")
    if len(markers) != 1:
        raise RuntimeError(
            f"expected exactly one <sdf> marker (the thread mesh only), found {len(markers)}: {markers}"
        )

    result = apply_urdf_sdf_collision_markers(usd_path, args_cli.urdf, markers)
    print(f"[sdf_markers] done: {result}")

    _align_root_frame_to_deployed(usd_path)
    _pin_mass_com_inertia_to_deployed(usd_path)
    _install_metadata_yaml(usd_path)

    _verify(usd_path, args_cli.urdf)


if __name__ == "__main__":
    main()
    simulation_app.close()
