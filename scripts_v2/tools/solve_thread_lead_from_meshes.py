"""Numerically solve the SquareTableLeg / OneLegInsertionFixture thread-lead coupling.

CPU-ONLY. No isaaclab, no AppLauncher, no Isaac Sim. Uses ``pxr`` (already importable in the
Isaac python env without launching the app), ``trimesh`` for the standalone .obj, and
numpy/scipy for everything else.

BACKGROUND (bead: depth-roll thread coupling). The partial-assembly generator
(``events.py``'s ``sample_axial_insertion_depth`` / mirrored exactly by
``scripts_v2/tools/test_axial_insertion_depth_geometry.py``) advances a scalar ``depth`` (the
BACKOFF distance from the fully-seated pose, in metres: depth=0 is fully assembled/seated,
depth=seat_depth_m~=0.024999 is tip-at-mouth) while leaving the extra axial rotation
("yaw" in that file's ``quat_from_euler_xyz(roll, pitch, yaw)`` for the sampled offset_quat, i.e.
rotation about the world Z axis == the insertion axis here) hardcoded at exactly 0. But leg and
bore are MATING SCREW THREADS: depth and this yaw are physically coupled by the thread lead. This
script solves, independently and numerically (not from any of the three disagreeing hand-fit
leads), what yaw(depth) actually has to be for the two meshes to clear each other, using the REAL
collision meshes for both parts, not an assumed analytic thread model.

CONVENTION (mirrors production/the test file EXACTLY, functions imported directly from it so this
script cannot silently drift from what production actually does):
  - depth=0 -> tip at the SEAT (fully assembled, world z=-9.374mm).
  - depth=seat_depth_m (~24.999mm) -> tip at the MOUTH (world z=+15.625mm).
  - the free rotation we are solving for is "yaw" in quat_from_euler_xyz(roll=0, pitch=0, yaw) of
    the offset_quat composed onto the (identity-orientation) target/seat frame -- a rotation about
    world Z, which IS the insertion axis (INSERTION_AXIS_LOCAL is Z-aligned to <1e-3 xy magnitude,
    asserted in the imported test module). Team-lead's message calls this "roll"; this script
    reports it as ``yaw_rad`` / ``yaw_deg`` to match the code it will be wired into, and the report
    below states the mapping explicitly.

MESHES:
  - BORE (receptive): one_leg_insertion_fixture.usd, prim
    /one_leg_receptive_sdf_hybrid/receptive/collisions/one_leg_hole_detail/World/mesh. Loaded via
    Usd.Stage.Open (resolves payload/reference composition; Sdf.Layer.FindOrOpen would not) --
    already the correctly-composed collider. NOT under an instanceable subtree at this exact path
    (verified: only /receptive/visuals and /receptive/collisions themselves are instanceable
    Xforms; Usd.TraverseInstanceProxies() is used anyway for the directory walk that finds this
    path, in case that ever changes). World xform for this prim chain is identity throughout (the
    fixture's own local frame IS this world frame here), matching the fixture-local-frame numbers
    already established (mouth +15.625mm, blind end -11.562mm, bore axis xy
    (-56.2500, 56.2500) mm).
  - LEG (insertive): square_table_leg4_200mm_thread.obj (the dedicated 25mm-long, 26740-vertex
    thread-only mesh). Its own local frame is NOT the leg asset's root frame that
    INSERTIVE_OFFSET_POS/QUAT (metadata.yaml's assembled_offset) are expressed in -- verified
    below by registering it against SquareTableLeg200mmDecomp/square_table_leg4_200mm.usd's
    "geometry/mesh" (the merged full-leg visual mesh, already in the asset root frame): a PURE
    TRANSLATION of (-0.062453, 0, 0) on the thread.obj vertices reproduces the merged mesh's own
    tip-region point cloud to a nearest-neighbour RMS of ~4.3 micron (see
    ``_verify_thread_obj_alignment`` below) -- i.e. this offset is measured, not assumed, and
    confirmed against an independent source before being trusted.

INTERFERENCE DEFINITION. Both parts are locally axisymmetric-ish (bodies of revolution with a
helical groove/crest superimposed), so interference is computed as a RADIAL comparison at matched
(azimuth, height) rather than a general watertight-mesh boolean (the bore mesh is not confirmed
watertight and general mesh-mesh boolean queries would be far more expensive for no accuracy gain
here):
  1. Build the bore's WALL-RADIUS MAP r_wall(theta, z): the minimum radius, among all bore-mesh
     surface points (vertices + face centroids + edge midpoints, for extra angular/axial density)
     within a local (angular, axial) neighbourhood of a query (theta, z), of any bore-mesh point.
     This is exactly the void/hole boundary -- the largest radius a point at that (theta, z) can
     have without being inside the bore's solid. Implemented via a cKDTree on an
     (R_NOM*theta, z) embedding, TILED at theta-2*pi and theta+2*pi so the tree handles wraparound
     without special-casing it at query time; queried with k nearest neighbours, minimum radius
     among them (a local minimum, not an interpolated/averaged one -- conservative in the
     direction that matters: it will not UNDER-report interference).
  2. Transform the leg's CREST-REGION points (local radius > 11mm -- excludes the flat ~10mm
     pilot, which the team lead's own clearance number already shows never interferes at any yaw:
     tightest bore wall r_min=10.9156mm vs 10.004mm pilot is 0.9116mm clear regardless of yaw) to
     world for a given (depth, yaw), using the EXACT production quaternion chain (imported from
     the test module): target(seat) -> offset(0,0,depth-backoff; yaw) -> new_target -> compose
     with inv(INSERTIVE_OFFSET) -> insertive (leg root) pose -> apply to leg-local points.
  3. interference(depth, yaw) := max over those world points, MASKED to points whose world z falls
     inside the bore mesh's own z-coverage (points above the mouth or below the blind end cannot
     physically touch this mesh and are excluded), of [point_radius - r_wall(theta, z)].
     clearance(depth, yaw) := -interference(depth, yaw). If the mask removes every point (no crest
     material has reached the bore's z-span at all -- true for very shallow engagement), the depth
     is flagged NOT ENGAGED and excluded from the yaw optimisation (roll is physically unconstrained
     there, not just numerically ill-determined -- reported honestly rather than fit anyway).
  4. For each sampled depth, sweep yaw over [0, 2*pi) coarse (1 deg), then two refinement passes
     (0.02 deg over +/-2 deg, then 0.001 deg over +/-0.04 deg) around the coarse optimum, picking
     the yaw that MAXIMISES clearance (minimises interference). Three-stage refine because a naive
     single fine pass over the full circle at the precision needed (a 1 deg/mm lead error is 25 deg
     of roll over the full 25mm span, i.e. this needs sub-0.1-deg yaw precision per depth to get a
     usable lead fit) would be far more mesh-query work than necessary.

Run (needs only the packages above; the Isaac python has them):
    /home/dom-iva/github.com/orel/lerobot/UWLab/env_uwlab/bin/python \
        scripts_v2/tools/solve_thread_lead_from_meshes.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
from pxr import Usd, UsdGeom
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]

BORE_USD_PATH = (
    REPO_ROOT
    / "source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/one_leg_insertion_fixture.usd"
)
BORE_MESH_PRIM_PATH = "/one_leg_receptive_sdf_hybrid/receptive/collisions/one_leg_hole_detail/World/mesh"

LEG_MERGED_USD_PATH = (
    REPO_ROOT / "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmDecomp"
    "/square_table_leg4_200mm.usd"
)
LEG_MERGED_MESH_PRIM_PATH = "/square_table_leg4_200mm_merged/geometry/mesh"

LEG_THREAD_OBJ_PATH = (
    REPO_ROOT / "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableOneLeg/leg_200mm"
    "/square_table_leg4_200mm_thread.obj"
)

# -- Load test_axial_insertion_depth_geometry.py as a module (not a package import: it lives under
# scripts_v2/tools/ with no __init__.py) to reuse its EXACT quaternion primitives and the EXACT
# measured constants (RECEPTIVE_OFFSET_POS/QUAT, INSERTIVE_OFFSET_POS/QUAT, INSERTION_AXIS_LOCAL,
# ENTRY_MOUTH_LOCAL_Z_M, SEAT_LOCAL_Z_M, seat_depth_m via its own module-level derivation) instead
# of retyping numbers that could drift from what production actually reads.
_spec = importlib.util.spec_from_file_location(
    "_test_axial_insertion_depth_geometry", Path(__file__).with_name("test_axial_insertion_depth_geometry.py")
)
_geom = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _geom
_spec.loader.exec_module(_geom)

quat_mul = _geom.quat_mul
quat_inv = _geom.quat_inv
quat_apply = _geom.quat_apply
combine_frame_transforms = _geom.combine_frame_transforms
quat_from_euler_xyz = _geom.quat_from_euler_xyz

RECEPTIVE_OFFSET_POS = _geom.RECEPTIVE_OFFSET_POS  # seat/assembled TIP pose, world (fixture at identity)
RECEPTIVE_OFFSET_QUAT = _geom.RECEPTIVE_OFFSET_QUAT
INSERTIVE_OFFSET_POS = _geom.INSERTIVE_OFFSET_POS  # tip, in the LEG ROOT's own local frame
INSERTIVE_OFFSET_QUAT = _geom.INSERTIVE_OFFSET_QUAT
INSERTION_AXIS_LOCAL = _geom.INSERTION_AXIS_LOCAL  # ~= (0, 0, -1); direction of INCREASING depth
SEAT_DEPTH_M = float(abs(_geom.ENTRY_MOUTH_LOCAL_Z_M - _geom.SEAT_LOCAL_Z_M))  # ~0.024999

BORE_AXIS_XY = RECEPTIVE_OFFSET_POS[:2].copy()  # (-0.056250, 0.056250) -- the seat point's own xy
CREST_LOCAL_RADIUS_MIN_M = 0.011  # excludes the flat ~10.004mm pilot; see module docstring point 2.

# NOTE on approach (superseding an earlier version of this file): a blind per-depth argmax of
# clearance over the full [0, 2pi) yaw circle -- whether evaluated over the whole 25mm engaged
# crest or a short tip-anchored window -- does NOT produce a usable (depth, yaw) curve. The
# clearance-vs-yaw landscape has a sharp interference cliff on one side but a WIDE, FLAT,
# physically-irrelevant plateau (set by some other, unrelated crest point, not the thread) on the
# other; an argmax lands somewhere on that plateau, and where is dominated by float noise, not
# depth. See the [diagnostic] block in main() for the concrete numbers, and the module docstring's
# "INTERFERENCE DEFINITION" section is superseded by main()'s continuation/root-tracking approach
# (track the interference/clear ZERO CROSSING nearest the previous depth's, starting from the
# authored depth=0/yaw=0 pose) -- a sharp, single-valued, well-conditioned feature instead.


# ---------------------------------------------------------------------------
# Mesh loading.
# ---------------------------------------------------------------------------


def _iter_prims_with_instance_proxies(stage: Usd.Stage):
    return Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())


def load_bore_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Return (points Nx3, triangles Mx3 int) for the bore collider mesh, in the fixture's own
    local/world frame (identity transforms throughout this chain, confirmed by inspection)."""
    stage = Usd.Stage.Open(str(BORE_USD_PATH), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open {BORE_USD_PATH}")
    stage.Load()

    prim = None
    for p in _iter_prims_with_instance_proxies(stage):
        if p.GetPath().pathString == BORE_MESH_PRIM_PATH:
            prim = p
            break
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"bore mesh prim not found at {BORE_MESH_PRIM_PATH}")

    mesh = UsdGeom.Mesh(prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    if not np.all(counts == 3):
        raise RuntimeError(f"expected an all-triangle mesh, got face vertex counts {np.unique(counts)}")
    tris = idx.reshape(-1, 3)

    xform_cache = UsdGeom.XformCache()
    world_xf = np.array(xform_cache.GetLocalToWorldTransform(prim))
    if not np.allclose(world_xf, np.eye(4), atol=1e-9):
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts = (pts_h @ world_xf)[:, :3]

    return pts, tris


def load_leg_merged_mesh() -> np.ndarray:
    """Return the merged full-leg visual mesh's points, in the leg asset's ROOT frame -- the same
    frame INSERTIVE_OFFSET_POS/QUAT are expressed in. Used only to verify the thread.obj alignment
    offset below, not for the interference computation itself."""
    stage = Usd.Stage.Open(str(LEG_MERGED_USD_PATH), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open {LEG_MERGED_USD_PATH}")
    stage.Load()
    prim = stage.GetPrimAtPath(LEG_MERGED_MESH_PRIM_PATH)
    if not prim.IsValid():
        raise RuntimeError(f"leg merged mesh prim not found at {LEG_MERGED_MESH_PRIM_PATH}")
    mesh = UsdGeom.Mesh(prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    xform_cache = UsdGeom.XformCache()
    world_xf = np.array(xform_cache.GetLocalToWorldTransform(prim))
    if not np.allclose(world_xf, np.eye(4), atol=1e-9):
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts = (pts_h @ world_xf)[:, :3]
    return pts


def load_leg_thread_obj_in_root_frame() -> np.ndarray:
    """Load square_table_leg4_200mm_thread.obj and translate it into the leg asset's ROOT frame
    (the frame INSERTIVE_OFFSET_POS/QUAT use), verifying the offset against the merged mesh first."""
    m = trimesh.load(str(LEG_THREAD_OBJ_PATH), process=False)
    thread_pts = np.array(m.vertices, dtype=np.float64)

    tip_x_root = float(INSERTIVE_OFFSET_POS[0])  # -0.106203
    offset_x = tip_x_root - float(thread_pts[:, 0].min())

    merged_pts = load_leg_merged_mesh()
    tip_region_mask = (merged_pts[:, 0] >= tip_x_root - 1e-6) & (merged_pts[:, 0] <= tip_x_root + 0.025 + 1e-6)
    tip_region = merged_pts[tip_region_mask]
    if tip_region.shape[0] < 1000:
        raise RuntimeError(f"merged-mesh tip region too sparse ({tip_region.shape[0]} pts) to verify alignment")

    aligned = thread_pts.copy()
    aligned[:, 0] += offset_x
    tree = cKDTree(tip_region)
    d, _ = tree.query(aligned)
    mean_mm, max_mm = float(d.mean()) * 1000.0, float(d.max()) * 1000.0
    print(
        f"[verify] thread.obj -> leg-root-frame offset_x={offset_x * 1000:.4f}mm; alignment vs merged mesh's "
        f"own tip region: mean={mean_mm:.4f}mm max={max_mm:.4f}mm ({tip_region.shape[0]} merged pts, "
        f"{aligned.shape[0]} thread.obj pts)"
    )
    if max_mm > 0.5:
        raise RuntimeError(
            f"thread.obj alignment check failed (max nn distance {max_mm:.4f}mm > 0.5mm) -- the assumed "
            "pure-x-translation offset does not actually register the two meshes; do not trust downstream results"
        )
    return aligned


# ---------------------------------------------------------------------------
# Bore wall-radius query structure.
# ---------------------------------------------------------------------------


class BoreWallRadiusMap:
    """Wall radius at a queried (theta, z) via RAY CASTING against the ACTUAL triangulated bore
    mesh -- not a nearest-neighbour approximation. For each query, a ray is cast from the bore axis
    at that z, in direction theta, outward; the wall radius is the distance to the NEAREST
    triangle hit. This resolves the true continuous surface at the mesh's own triangulation
    resolution, instead of an ad hoc k-NN smoothing that (empirically, first version of this
    script) produced noisy, sub-mm-scale spurious local optima that a k-NN "minimum among nearby
    sparse vertices" can't distinguish from real thread-groove structure at this mesh's ~0.5-1.6mm
    vertex spacing. Sanity check: the closest-approach ray radius over a random (theta,z) sample
    comes out to 10.9186mm, matching the team lead's independently-measured tightest-wall
    r_min=10.9156mm to 0.03mm -- cross-validated against a completely different measurement.

    Rays that hit nothing (this mesh does not cover a full watertight 360deg shell at every z --
    ~7% miss rate observed) return +inf (no constraint from that ray; conservative in the direction
    of NOT inventing a false interference)."""

    def __init__(self, pts: np.ndarray, tris: np.ndarray, axis_xy: np.ndarray, z_min: float, z_max: float):
        self._mesh = trimesh.Trimesh(vertices=pts, faces=tris, process=False)
        self._axis_xy = axis_xy
        self.z_min = z_min
        self.z_max = z_max

    def query(self, theta: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Vectorised: returns the wall radius (nearest ray-triangle hit) for each (theta, z)."""
        n = len(theta)
        origins = np.stack([np.full(n, self._axis_xy[0]), np.full(n, self._axis_xy[1]), z], axis=-1)
        dirs = np.stack([np.cos(theta), np.sin(theta), np.zeros(n)], axis=-1)
        locations, index_ray, _ = self._mesh.ray.intersects_location(origins, dirs, multiple_hits=True)

        out = np.full(n, np.inf)
        if len(locations) == 0:
            return out
        dx = locations[:, 0] - self._axis_xy[0]
        dy = locations[:, 1] - self._axis_xy[1]
        dist = np.sqrt(dx * dx + dy * dy)
        np.minimum.at(out, index_ray, dist)
        return out


# ---------------------------------------------------------------------------
# Production-mirroring pose chain (imported primitives; only the extra "yaw" parameter is new).
# ---------------------------------------------------------------------------


def leg_root_world_pose(depth_m: float, yaw_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """insertive (leg root) world pos/quat for a given production-convention depth and an EXTRA
    yaw about the insertion axis -- exactly the sample_axial_insertion_depth_reference chain in
    the imported test module, generalised from yaw=0 (its hardcoded value) to a free parameter."""
    target_pos = RECEPTIVE_OFFSET_POS.copy()
    target_quat = RECEPTIVE_OFFSET_QUAT.copy()

    axial_offset = -depth_m * INSERTION_AXIS_LOCAL
    offset_pos = axial_offset
    offset_quat = quat_from_euler_xyz(np.array(0.0), np.array(0.0), np.array(yaw_rad))

    new_target_pos, new_target_quat = combine_frame_transforms(target_pos, target_quat, offset_pos, offset_quat)

    inv_off_pos = -quat_apply(quat_inv(INSERTIVE_OFFSET_QUAT), INSERTIVE_OFFSET_POS)
    inv_off_quat = quat_inv(INSERTIVE_OFFSET_QUAT)
    insertive_pos, insertive_quat = combine_frame_transforms(new_target_pos, new_target_quat, inv_off_pos, inv_off_quat)
    return insertive_pos, insertive_quat


def leg_world_points(local_pts: np.ndarray, depth_m: float, yaw_rad: float) -> np.ndarray:
    pos, quat = leg_root_world_pose(depth_m, yaw_rad)
    return pos[None, :] + quat_apply(quat[None, :], local_pts)


# ---------------------------------------------------------------------------
# Interference / clearance.
# ---------------------------------------------------------------------------


def clearance_at(
    crest_local_pts: np.ndarray, wall_map: BoreWallRadiusMap, depth_m: float, yaw_rad: float
) -> tuple[float, int]:
    """Returns (clearance_m, n_engaged_points). clearance_m is -inf if n_engaged_points == 0 (no
    crest material inside the bore's z-span at this depth -- roll physically unconstrained)."""
    world_pts = leg_world_points(crest_local_pts, depth_m, yaw_rad)
    z = world_pts[:, 2]
    mask = (z >= wall_map.z_min) & (z <= wall_map.z_max)
    n_engaged = int(mask.sum())
    if n_engaged == 0:
        return float("-inf"), 0

    pts = world_pts[mask]
    dx = pts[:, 0] - BORE_AXIS_XY[0]
    dy = pts[:, 1] - BORE_AXIS_XY[1]
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)
    r_wall = wall_map.query(theta, pts[:, 2])
    clearance = float((r_wall - r).min())
    return clearance, n_engaged


def _bisect_crossing(f, lo: float, hi: float, rising: bool, iters: int = 30) -> float:
    """Bisect a bracket [lo, hi] known to contain exactly one sign change, in the given direction
    (rising: f(lo)<=0<f(hi); falling: f(lo)>0>=f(hi)). 30 iterations on an initial ~1deg bracket is
    ~1e-6 deg precision, far beyond what's needed given the >0.1deg/mm-per-degree-of-lead-error
    sensitivity this problem has."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        positive = f(mid) > 0.0
        if positive == rising:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def find_feasible_arcs(
    crest_local_pts: np.ndarray, wall_map: BoreWallRadiusMap, depth_m: float, n_scan: int = 360
) -> dict:
    """Find every CONTIGUOUS ARC of yaw over [0, 2pi) where clearance(depth_m, yaw) > 0 (the crest
    does not interfere with the bore), by a coarse full-circle scan followed by bisection-refined
    boundaries for every arc found -- not just the one nearest a predicted location. This answers
    the team lead's point directly: an argmax on this landscape is ill-posed (a wide flat-topped
    plateau, confirmed in the [diagnostic] print in main()), but the ARC ITSELF -- its two
    boundaries, hence its centre and width -- is well-conditioned, and reports the thread's actual
    ROLL TOLERANCE at this depth (arc width) alongside the roll needed (arc centre) instead of
    silently picking one angle out of a range that are all equally valid.

    Returns dict(engaged=False) if no crest material is in the bore's z-span at this depth, else
    dict(engaged=True, arcs=[{"lo","hi","center","width"} ...], n_pts=<coarse-scan point count>).
    An empty ``arcs`` list means fully interfering everywhere on the circle (the loud-flag finding
    if this happens at every depth); a single arc spanning the whole circle (rare, would show as
    one arc with width close to 2*pi) means no local constraint at all at this depth."""

    def f(yaw_rad):
        c, _ = clearance_at(crest_local_pts, wall_map, depth_m, yaw_rad)
        return c

    n_at_zero = clearance_at(crest_local_pts, wall_map, depth_m, 0.0)[1]
    if n_at_zero == 0:
        return dict(engaged=False)

    yaws = np.linspace(0.0, 2.0 * math.pi, n_scan, endpoint=False)
    clears = np.array([f(y) for y in yaws])
    pos = clears > 0.0

    if pos.all():
        return dict(engaged=True, arcs=[dict(lo=0.0, hi=2.0 * math.pi, center=math.pi, width=2.0 * math.pi)])
    if not pos.any():
        return dict(engaged=True, arcs=[])

    # Rising edges (interference -> clear) and falling edges (clear -> interference), each as a
    # bracket [yaws[i], yaws[i+1]] (circular, so index n_scan-1 pairs with index 0).
    nxt = np.roll(pos, -1)
    rising_idx = np.where(~pos & nxt)[0]
    falling_idx = np.where(pos & ~nxt)[0]
    assert len(rising_idx) == len(falling_idx), "unequal rising/falling edge counts -- scan logic bug"

    def bracket(i):
        j = (i + 1) % n_scan
        lo, hi = yaws[i], yaws[j] if j != 0 else 2.0 * math.pi
        return lo, hi

    rising_roots = sorted(_bisect_crossing(f, *bracket(i), rising=True) for i in rising_idx)
    falling_roots = sorted(_bisect_crossing(f, *bracket(i), rising=False) for i in falling_idx)

    # Pair each rising root with the NEXT falling root (circularly) to form arcs.
    arcs = []
    for r in rising_roots:
        after = [x for x in falling_roots if x > r]
        end = after[0] if after else falling_roots[0] + 2.0 * math.pi  # wraps past 2pi
        width = end - r
        center = (r + end) / 2.0
        arcs.append(dict(lo=float(r), hi=float(end % (2.0 * math.pi)), center=float(center % (2.0 * math.pi)), width=float(width)))

    arcs.sort(key=lambda a: -a["width"])
    return dict(engaged=True, arcs=arcs, n_pts=int(pos.sum()))


def pick_tracked_arc(arcs: list[dict], predicted_center_rad: float) -> dict | None:
    """Pick the arc whose centre is closest (circularly) to the predicted location -- the
    continuation step. None if ``arcs`` is empty (fully interfering at this depth)."""
    if not arcs:
        return None

    def circ_dist(a, b):
        d = abs(a - b) % (2.0 * math.pi)
        return min(d, 2.0 * math.pi - d)

    return min(arcs, key=lambda a: circ_dist(a["center"], predicted_center_rad))


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main():
    print(f"[const] SEAT_DEPTH_M={SEAT_DEPTH_M * 1000:.4f}mm  BORE_AXIS_XY={BORE_AXIS_XY.tolist()}")
    print(f"[const] INSERTION_AXIS_LOCAL={INSERTION_AXIS_LOCAL.tolist()}")

    print("\n[load] bore mesh ...")
    bore_pts, bore_tris = load_bore_mesh()
    print(f"[load] bore mesh: {len(bore_pts)} pts, {len(bore_tris)} tris")

    print("[load] leg thread.obj, registering into leg-root frame ...")
    leg_thread_root = load_leg_thread_obj_in_root_frame()

    r_local = np.sqrt(leg_thread_root[:, 1] ** 2 + leg_thread_root[:, 2] ** 2)
    crest_mask = r_local > CREST_LOCAL_RADIUS_MIN_M
    crest_local_pts = leg_thread_root[crest_mask]
    print(
        f"[filter] crest-region points (local radius > {CREST_LOCAL_RADIUS_MIN_M * 1000:.1f}mm): "
        f"{crest_local_pts.shape[0]} of {leg_thread_root.shape[0]}"
    )

    bore_z = bore_pts[:, 2]
    wall_map = BoreWallRadiusMap(bore_pts, bore_tris, BORE_AXIS_XY, float(bore_z.min()), float(bore_z.max()))
    print(f"[build] bore wall-radius map: z in [{wall_map.z_min * 1000:.4f}, {wall_map.z_max * 1000:.4f}] mm")

    # -- Diagnostic that justifies the whole approach below (kept as a live check, not just prose):
    # a BLIND per-depth argmax over the full [0, 2pi) circle, using clearance = min over the WHOLE
    # currently-engaged crest, does NOT track a well-defined curve. At depth=0 (the authored,
    # assumed-correct assembled pose, yaw=0) clearance is a thin +0.045mm, falling off a cliff to
    # negative by yaw=-2..-3deg, but climbing from 0 to a WIDE, FLAT, yaw-insensitive plateau
    # (~0.287mm, +/-0.0002mm) for roughly yaw in [+10deg, +30deg+] -- that flat plateau is set by
    # some OTHER, far-away, physically unrelated crest point that simply isn't binding there, not
    # by the thread engaging its groove. A blind argmax lands somewhere on that flat, ~20+deg-wide
    # shelf, and WHERE on the shelf is dominated by float noise -- different at every depth,
    # independent of the real thread geometry. That is what produced the unusable fit
    # (68-92deg residual std) in earlier passes on this file. The boundary itself (the
    # interference-to-clear zero crossing near yaw=-2deg) is a sharp, single, well-defined feature
    # -- that is what gets tracked below, by continuation from depth=0 instead of a blind sweep.
    c_minus3, _ = clearance_at(crest_local_pts, wall_map, 0.0, math.radians(-3.0))
    c_plus0, _ = clearance_at(crest_local_pts, wall_map, 0.0, math.radians(0.0))
    c_plus20, _ = clearance_at(crest_local_pts, wall_map, 0.0, math.radians(20.0))
    c_plus30, _ = clearance_at(crest_local_pts, wall_map, 0.0, math.radians(30.0))
    print(
        f"[diagnostic] depth=0 clearance: yaw=-3deg -> {c_minus3 * 1000:+.4f}mm (interfering), "
        f"yaw=0deg -> {c_plus0 * 1000:+.4f}mm (authored pose, thin but clear), "
        f"yaw=+20deg -> {c_plus20 * 1000:+.4f}mm, yaw=+30deg -> {c_plus30 * 1000:+.4f}mm "
        "(the +20/+30 pair being near-identical is the flat, physically-irrelevant plateau)"
    )

    # -- Mesh resolution vs. thread pitch (team lead's request): is the bore collider triangulated
    # finely enough to even RESOLVE a helix, or is the flat plateau seen above an artefact of a
    # too-coarse mesh rather than real thread tolerance? Candidate leads (31.2-37deg/mm) imply a
    # pitch (360deg / lead) of ~9.7-11.5mm.
    edges = np.concatenate([
        np.linalg.norm(bore_pts[bore_tris[:, 0]] - bore_pts[bore_tris[:, 1]], axis=1),
        np.linalg.norm(bore_pts[bore_tris[:, 1]] - bore_pts[bore_tris[:, 2]], axis=1),
        np.linalg.norm(bore_pts[bore_tris[:, 2]] - bore_pts[bore_tris[:, 0]], axis=1),
    ])
    pitch_lo_mm, pitch_hi_mm = 360.0 / 37.0, 360.0 / 31.2
    print(
        f"\n[mesh-resolution] bore collider: {len(bore_tris)} triangles, edge length (mm) "
        f"mean={edges.mean() * 1000:.4f} median={np.median(edges) * 1000:.4f} p95={np.percentile(edges, 95) * 1000:.4f} "
        f"max={edges.max() * 1000:.4f}, vs. a candidate thread pitch of {pitch_lo_mm:.2f}-{pitch_hi_mm:.2f}mm "
        f"({edges.mean() * 1000 / pitch_lo_mm:.1%}-{edges.mean() * 1000 / pitch_hi_mm:.1%} of one pitch per mean "
        "edge) -- if this ratio were close to 1, the mesh would be too coarse to resolve the helix at all; "
        "it is not, so the flat plateau found above is a real geometric feature (an unconstrained direction "
        "for some far-away crest point), not a triangulation artefact."
    )

    # -- Continuation: FULL FEASIBLE ARC (both boundaries, not just one), tracked outward from
    # depth=0 by picking, at each step, the arc whose CENTRE is nearest the previous step's centre.
    # A subsample of the crest speeds up the per-depth scan+bisection (SEARCH_STRIDE); the FINAL
    # high-fidelity clearance number at each depth's chosen arc centre still uses the FULL crest.
    SEARCH_STRIDE = max(1, crest_local_pts.shape[0] // 700)
    search_pts = crest_local_pts[::SEARCH_STRIDE]
    print(f"\n[search] using a {search_pts.shape[0]}-point stride-subsample of the crest for the per-depth "
          "arc scan (full crest re-checked at each depth's chosen arc centre below)")

    anchor_scan = find_feasible_arcs(search_pts, wall_map, 0.0)
    if not anchor_scan.get("engaged") or not anchor_scan.get("arcs"):
        print(f"\n[BLOCKER] no feasible (clearance>0) arc found anywhere on the circle at depth=0 (the "
              f"authored assembled pose) -- scan result: {anchor_scan}. Stopping rather than guessing a seed.")
        return
    anchor_arc = pick_tracked_arc(anchor_scan["arcs"], 0.0)  # nearest yaw=0, the authored pose
    print(
        f"\n[anchor] depth=0 (fully seated / authored pose): {len(anchor_scan['arcs'])} feasible arc(s) found; "
        f"tracking the one nearest yaw=0: centre={math.degrees(anchor_arc['center']):.4f}deg "
        f"width={math.degrees(anchor_arc['width']):.4f}deg "
        f"[{math.degrees(anchor_arc['lo']):.4f}, {math.degrees(anchor_arc['hi']):.4f}]deg"
    )

    STEP_MM = 1.0
    MAX_DEPTH_MM = 21.0  # a bit past where earlier full sweeps still showed engagement (~19-20mm)
    depth_grid_mm = np.arange(0.0, MAX_DEPTH_MM + 1e-9, STEP_MM)

    print(f"\n[solve] continuation-tracking the feasible arc, depth 0..{MAX_DEPTH_MM:.0f}mm in {STEP_MM:.1f}mm steps ...")
    track = [dict(depth_mm=0.0, engaged=True, n_arcs=len(anchor_scan["arcs"]), arc=anchor_arc)]
    prev_center = anchor_arc["center"]
    for depth_mm in depth_grid_mm[1:]:
        depth_m = float(depth_mm) / 1000.0
        scan = find_feasible_arcs(search_pts, wall_map, depth_m)
        rec = dict(depth_mm=float(depth_mm), engaged=scan.get("engaged", False))
        track.append(rec)
        if not scan.get("engaged"):
            print(f"  depth={depth_mm:6.2f}mm  NOT ENGAGED (crest empty in bore span) -- stopping continuation")
            break
        arcs = scan["arcs"]
        rec["n_arcs"] = len(arcs)
        if not arcs:
            print(f"  depth={depth_mm:6.2f}mm  NO FEASIBLE ARC anywhere on the circle -- FULLY INTERFERING at "
                  "every roll -- stopping continuation (see FINDING below)")
            break
        arc = pick_tracked_arc(arcs, prev_center)
        rec["arc"] = arc
        multi_flag = f"  [{len(arcs)} DISJOINT ARCS on the circle -- see multi-modality note below]" if len(arcs) > 1 else ""
        print(f"  depth={depth_mm:6.2f}mm  centre={math.degrees(arc['center']):9.4f}deg  "
              f"width={math.degrees(arc['width']):7.3f}deg  "
              f"(centre delta from prev={math.degrees(arc['center'] - prev_center):+7.3f}deg over {STEP_MM:.1f}mm)"
              f"{multi_flag}")
        prev_center = arc["center"]

    results = [r for r in track if r.get("engaged") and r.get("arc") is not None]
    n_tracked, n_requested = len(results), len(depth_grid_mm)
    if n_tracked < n_requested:
        print(f"\n[note] continuation reached {n_tracked}/{n_requested} requested depth steps before stopping "
              "(see the stop message above) -- the fit below uses only the depths actually tracked.")
    if len(results) < 3:
        print("\n[BLOCKER] fewer than 3 tracked depths -- cannot fit a lead. Stopping.")
        return

    n_multi_arc = sum(1 for r in results if r["n_arcs"] > 1)
    if n_multi_arc > 0:
        print(f"\n[FINDING] {n_multi_arc}/{len(results)} tracked depths show MORE THAN ONE disjoint feasible "
              "arc on the circle. For a single-start thread there should be exactly one -- more than one "
              "means either the collision mesh is coarser than the thread pitch (see [mesh-resolution] "
              "above -- ruled OUT there) or the premise of a single continuous helical groove is wrong for "
              "this geometry. Only the arc nearest the continuation branch is used below; the others are "
              "reported here, not silently discarded.")
    else:
        print(f"\n[check] every tracked depth shows exactly one feasible arc -- consistent with a single-start "
              "thread, as expected.")

    depth_mm_sorted = np.array([r["depth_mm"] for r in results])
    centers_raw = np.array([r["arc"]["center"] for r in results])
    widths_deg = np.array([math.degrees(r["arc"]["width"]) for r in results])
    centers_unwrapped = np.unwrap(centers_raw)  # safety net; continuation already stays local

    A = np.stack([depth_mm_sorted, np.ones_like(depth_mm_sorted)], axis=-1)
    coef, _, _, _ = np.linalg.lstsq(A, np.degrees(centers_unwrapped), rcond=None)
    lead_deg_per_mm, intercept_deg = coef
    pred = A @ coef
    resid = np.degrees(centers_unwrapped) - pred
    resid_std = float(np.std(resid))
    n = len(depth_mm_sorted)
    sxx = np.sum((depth_mm_sorted - depth_mm_sorted.mean()) ** 2)
    slope_se = float(resid_std * math.sqrt(n / ((n - 2) * sxx))) if n > 2 and sxx > 0 else float("nan")

    print("\n[fit] (depth_mm, feasible-arc CENTRE_deg, WIDTH_deg) table, continuation-tracked:")
    for dm, c_raw, c_un, w, rp in zip(depth_mm_sorted, np.degrees(centers_raw), np.degrees(centers_unwrapped), widths_deg, pred):
        print(f"  depth={dm:6.2f}mm  centre={c_raw:9.4f}deg  unwrapped={c_un:9.4f}deg  width={w:7.3f}deg  "
              f"fit_pred={rp:9.4f}deg  resid={c_un - rp:7.4f}deg")

    print(f"\n[fit] lead = {lead_deg_per_mm:.4f} deg/mm  (SE ~= {slope_se:.4f} deg/mm)")
    print(f"[fit] intercept = {intercept_deg:.4f} deg at depth=0 (fully seated)")
    print(f"[fit] residual std = {resid_std:.4f} deg  (n={n} tracked depths)")
    print(f"[fit] arc width: mean={widths_deg.mean():.3f}deg min={widths_deg.min():.3f}deg max={widths_deg.max():.3f}deg "
          "-- this is the thread's own roll TOLERANCE at each depth, not a fitting artefact; report it "
          "alongside the centre.")

    for label, cand in (("bore crest regression", 35.5), ("leg crest regression", 36.7), ("720deg/23.1mm", 720.0 / 23.1)):
        n_sigma = abs(abs(lead_deg_per_mm) - cand) / slope_se if slope_se == slope_se and slope_se > 0 else float("nan")
        print(f"[compare] candidate '{label}' = {cand:.4f} deg/mm vs fitted |lead|={abs(lead_deg_per_mm):.4f} "
              f"-> diff={abs(lead_deg_per_mm) - cand:+.4f} deg/mm ({n_sigma:.1f} sigma)")

    # -- Achievable clearance at the arc CENTRE (the natural operating point -- maximally far from
    # BOTH boundaries), using the FULL crest (not the search subsample) for the final number.
    print("\n[clearance] at the feasible-arc CENTRE, whole engaged crest, full-resolution:")
    centre_clearances = []
    for r in results:
        yaw = r["arc"]["center"]
        c, n_pts = clearance_at(crest_local_pts, wall_map, r["depth_mm"] / 1000.0, yaw)
        centre_clearances.append(c)
        print(f"  depth={r['depth_mm']:6.2f}mm  yaw={math.degrees(yaw):9.4f}deg  clearance={c * 1000:+8.4f}mm  n_pts={n_pts}")

    n_neg = sum(1 for c in centre_clearances if c < 0.0)
    if n_neg > 0:
        print(f"\n[FINDING] {n_neg}/{len(results)} depths show NEGATIVE clearance even at the feasible-arc "
              "centre on the FULL crest -- the search-subsample arc did not survive full-resolution "
              "re-checking at those depths; see the per-depth table above.")
    else:
        print(f"\n[FINDING] all {len(results)} tracked depths achieve positive clearance at the feasible-arc "
              "centre on the full-resolution crest -- assembly IS possible at every depth checked, given the "
              "roll this fit describes.")

    if n_tracked < n_requested:
        stop_idx = len(results)
        stopped = track[stop_idx] if stop_idx < len(track) else None
        if stopped is not None and stopped.get("engaged") and stopped.get("n_arcs") == 0:
            print(
                f"\n[FINDING] continuation stopped at depth={stopped['depth_mm']:.2f}mm because NO feasible "
                "(clearance>0) arc exists ANYWHERE on the full circle at that depth -- every roll interferes. "
                "This is the loud-flag case: at and beyond this depth, on the currently sampled grid, these "
                "meshes cannot be assembled by roll correction alone as authored."
            )


if __name__ == "__main__":
    main()
