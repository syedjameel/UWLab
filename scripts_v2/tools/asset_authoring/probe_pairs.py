"""Which collider pairs actually interpenetrate at the spawn posture?

The story so far has been "hulls of ADJACENT hand bodies overlap". That cannot be the mechanism:
PhysX always filters collision between a parent and child link joined by an articulation joint,
regardless of enabledSelfCollisions. So if hull+self-collisions diverges, the offending pairs must
be NON-ADJACENT -- link i vs link i+2 inside a curled finger, or a fingertip against the palm.

This measures it instead of arguing it:
  1. boot the env with the hull USD and self-collisions OFF, so the articulation does NOT diverge
     and we sample the authored spawn posture rather than the wreckage,
  2. read every body's world pose from the simulator,
  3. read each body's collision mesh from the USD, take its convex hull (which is exactly what
     PhysX collides when approximation="convexHull"),
  4. test every pair for interpenetration, excluding parent-child pairs read from the joint graph.

Output is the ranked list of offending pairs -- which is also the exact argument list for a
UsdPhysics.FilteredPairsAPI fix.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="DexLift-UR5eDelto-RelJointPos-Lift-v0")
parser.add_argument("--frames", type=int, default=5)
parser.add_argument("--stack", type=int, default=268435456)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import itertools  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402
from scipy.spatial import ConvexHull, Delaunay  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
env_cfg.sim.physx.gpu_collision_stack_size = args.stack
env = gym.make(args.task, cfg=env_cfg)
raw = env.unwrapped
robot = raw.scene["robot"]
usd_path = robot.cfg.spawn.usd_path
print(f"[pairs] usd={usd_path}", flush=True)

env.reset()
zero = torch.zeros((1, env.action_space.shape[-1]), device=raw.device)
for _ in range(args.frames):
    env.step(zero)

names = list(robot.data.body_names)
pos = robot.data.body_pos_w[0].cpu().numpy()
quat = robot.data.body_quat_w[0].cpu().numpy()  # wxyz
print(f"[pairs] {len(names)} bodies sampled after {args.frames} zero-action frames", flush=True)


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


stage = Usd.Stage.Open(usd_path)

# Parent-child pairs from the joint graph. PhysX filters these automatically, so an overlap here
# is harmless and must NOT be counted as a cause.
adjacent = set()
for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.Joint):
        continue
    j = UsdPhysics.Joint(prim)
    b0 = [p.name for p in j.GetBody0Rel().GetTargets()]
    b1 = [p.name for p in j.GetBody1Rel().GetTargets()]
    for a in b0:
        for b in b1:
            adjacent.add(frozenset((a, b)))
print(f"[pairs] {len(adjacent)} jointed (auto-filtered) pairs", flush=True)

# Convex hull of each body's collision geometry, in body-local coordinates.
#
# TRAVERSAL TRAP, and the first version of this probe fell straight into it: only 5 of the 28
# colliders are UsdGeom.Mesh prims. The other 23 carry CollisionAPI on an **Xform**
# (<body>/collisions/<body>_c/node_STL_BINARY_) with the mesh one level below. Requiring
# IsA(Mesh) AND CollisionAPI on the same prim therefore found the five fingertips and nothing
# else, and the probe cheerfully reported "0 overlapping pairs" having examined almost no
# geometry. Start from CollisionAPI, then descend for meshes.
#
# Body attribution follows the same path shape: .../<BODY>/collisions/<BODY>_c/node...
# so the body is the parent of the ancestor named "collisions".
def owning_body(prim):
    p = prim
    while p and p.IsValid():
        par = p.GetParent()
        if par and par.IsValid() and par.GetName() == "collisions":
            return par.GetParent().GetName()
        p = par
    return None


hulls: dict[str, np.ndarray] = {}
n_coll = 0
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    n_coll += 1
    body = owning_body(prim)
    if body is None or body not in names:
        print(f"[pairs] WARN unattributed collider {prim.GetPath()} -> {body}", flush=True)
        continue
    meshes = [prim] if prim.IsA(UsdGeom.Mesh) else [
        d for d in Usd.PrimRange(prim) if d.IsA(UsdGeom.Mesh)
    ]
    bprim = None
    for cand in Usd.PrimRange(stage.GetPseudoRoot()):
        if cand.GetName() == body:
            bprim = cand
            break
    if bprim is None:
        continue
    bxf = np.array(UsdGeom.Xformable(bprim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    for m in meshes:
        pts = np.asarray(UsdGeom.Mesh(m).GetPointsAttr().Get() or [], dtype=float)
        if len(pts) < 4:
            continue
        xf = np.array(UsdGeom.Xformable(m).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        rel = np.linalg.inv(bxf) @ xf
        local = (np.c_[pts, np.ones(len(pts))] @ rel.T)[:, :3]
        try:
            local = local[ConvexHull(local).vertices]
        except Exception:
            pass
        hulls[body] = np.vstack([hulls[body], local]) if body in hulls else local

print(f"[pairs] {n_coll} CollisionAPI prims -> {len(hulls)} bodies with hulls", flush=True)
if len(hulls) < 20:
    raise SystemExit(
        f"REFUSING: only {len(hulls)} bodies carry hulls, expected ~28. Traversal is wrong again "
        "-- a low count here is what made the first run report a meaningless zero."
    )

print(f"[pairs] {len(hulls)} bodies carry collision geometry", flush=True)

world = {}
for b, local in hulls.items():
    i = names.index(b)
    world[b] = local @ quat_to_R(quat[i]).T + pos[i]


def overlap(A, B):
    """Fraction of B's hull vertices strictly inside A's hull, and vice versa."""
    try:
        dA, dB = Delaunay(A), Delaunay(B)
    except Exception:
        return 0.0
    return max((dA.find_simplex(B) >= 0).mean(), (dB.find_simplex(A) >= 0).mean())


hits = []
for a, b in itertools.combinations(sorted(world), 2):
    if frozenset((a, b)) in adjacent:
        continue
    f = overlap(world[a], world[b])
    if f > 0:
        hits.append((f, a, b))

hits.sort(reverse=True)
print(f"\n=== NON-ADJACENT interpenetrating pairs at spawn: {len(hits)} ===", flush=True)
for f, a, b in hits[:40]:
    print(f"  {f * 100:6.1f}%  {a:22s} {b}", flush=True)
if not hits:
    print("  NONE -- the divergence is not static interpenetration at the spawn posture.", flush=True)

# Sanity: were any ADJACENT pairs overlapping? Harmless (PhysX filters them) but it tells us
# whether the geometry is generally inflated or only specific pairs are bad.
adj_hits = sum(
    1 for a, b in itertools.combinations(sorted(world), 2)
    if frozenset((a, b)) in adjacent and overlap(world[a], world[b]) > 0
)
print(f"\n[pairs] adjacent pairs also overlapping (auto-filtered, harmless): {adj_hits}", flush=True)
print("PROBE_OK", flush=True)
env.close()
simulation_app.close()
