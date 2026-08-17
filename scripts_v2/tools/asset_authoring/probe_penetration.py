"""Does the policy actually push its fingers through each other? The acceptance test for criterion 1.

The whole point of turning self-collisions on is that a policy trained with them OFF can learn a
grasp that is geometrically impossible -- fingers passing through one another -- and still be scored
a success, because nothing in the reward or the success predicate looks at interpenetration. That is
an argument, not a measurement. This measures it.

WHAT IT MEASURES, and why this exact set of bodies. Only the 25 convexHull bodies -- 20 phalanges and
5 fingertips -- are tested. rl_dg_mount / rl_dg_base / rl_dg_palm are EXCLUDED because in the hullfix
asset they are convexDecomposition, and taking their convex hull (which is what this script can
compute from the mesh) would report the concavity we deliberately preserved as a penetration. It
would score the fix as a failure by measuring the thing the fix removed. The 25 remaining bodies are
also exactly what "fingers interpenetration" means.

Adjacent pairs are excluded: PhysX always filters jointed parent-child pairs, so an overlap there is
expected and harmless -- 22 of them overlap in this hand at rest and always have.

Run it on two checkpoints and compare. A policy trained WITHOUT self-collisions is expected to show a
non-zero rate; one trained WITH them should be at or near zero, and a zero on both would mean the
grasp never needed the constraint and criterion 1 was never binding.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="DexLift-UR5eDelto-RelJointPos-Lift-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--every", type=int, default=5, help="evaluate geometry every Nth step")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--tag", required=True)
parser.add_argument("--stack", type=int, default=268435456)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import itertools  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402
from scipy.spatial import ConvexHull, Delaunay  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
env_cfg.sim.physx.gpu_collision_stack_size = args.stack
env_cfg.seed = args.seed
agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")

env = gym.make(args.task, cfg=env_cfg)
raw = env.unwrapped
robot = raw.scene["robot"]
usd_path = robot.cfg.spawn.usd_path
names = list(robot.data.body_names)
print(f"[pen] tag={args.tag} usd={usd_path}", flush=True)

# ---- geometry, read once from the USD --------------------------------------------------------
stage = Usd.Stage.Open(usd_path)

adjacent = set()
for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.Joint):
        continue
    j = UsdPhysics.Joint(prim)
    for a in [t.name for t in j.GetBody0Rel().GetTargets()]:
        for b in [t.name for t in j.GetBody1Rel().GetTargets()]:
            adjacent.add(frozenset((a, b)))

# Pairs the asset itself filters must not be counted as violations -- PhysX is not checking them by
# our own deliberate choice, so a penetration there is expected, not evidence of anything.
filtered = set()
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
        for t in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
            filtered.add(frozenset((prim.GetName(), t.name)))


def owning_body(prim):
    p = prim
    while p and p.IsValid():
        par = p.GetParent()
        if par and par.IsValid() and par.GetName() == "collisions":
            return par.GetParent().GetName()
        p = par
    return None


FINGER_RE = re.compile(r"rl_dg_[1-5]_([1-4]|tip)")  # 20 phalanges + 5 tips, in both assets

hulls: dict[str, np.ndarray] = {}
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    body = owning_body(prim)
    if body is None or body not in names:
        continue
    # SELECT THE COMPARISON SET BY NAME, NOT BY APPROXIMATION. Selecting on "convexHull" gives 25
    # bodies in the hullfix asset but 28 in the baseline convexhull asset, where mount/base/palm are
    # also hulls -- so the two runs would be measuring different body sets and the before/after
    # would be meaningless. The 20 phalanges and 5 tips are identical hulls in BOTH assets, which is
    # exactly what makes them comparable, and they are what "finger interpenetration" means.
    if not FINGER_RE.fullmatch(body):
        continue
    bprim = next((c for c in Usd.PrimRange(stage.GetPseudoRoot()) if c.GetName() == body), None)
    if bprim is None:
        continue
    bxf = np.array(UsdGeom.Xformable(bprim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    meshes = [prim] if prim.IsA(UsdGeom.Mesh) else [d for d in Usd.PrimRange(prim) if d.IsA(UsdGeom.Mesh)]
    for m in meshes:
        pts = np.asarray(UsdGeom.Mesh(m).GetPointsAttr().Get() or [], dtype=float)
        if len(pts) < 4:
            continue
        xf = np.array(UsdGeom.Xformable(m).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        local = (np.c_[pts, np.ones(len(pts))] @ (np.linalg.inv(bxf) @ xf).T)[:, :3]
        try:
            local = local[ConvexHull(local).vertices]
        except Exception:
            pass
        hulls[body] = np.vstack([hulls[body], local]) if body in hulls else local

if len(hulls) != 25:
    raise SystemExit(f"REFUSING: expected 25 convexHull finger bodies, found {len(hulls)}: {sorted(hulls)}")
radius = {b: float(np.linalg.norm(v - v.mean(0), axis=1).max()) for b, v in hulls.items()}
centre = {b: v.mean(0) for b, v in hulls.items()}
PAIRS = [
    (a, b)
    for a, b in itertools.combinations(sorted(hulls), 2)
    if frozenset((a, b)) not in adjacent and frozenset((a, b)) not in filtered
]
print(f"[pen] {len(hulls)} convexHull finger bodies, {len(PAIRS)} non-adjacent unfiltered pairs", flush=True)

# ---- policy ----------------------------------------------------------------------------------
# These five come from agent_cfg["params"]["env"], NOT ["config"] -- reading them from the wrong
# subtree silently yields the defaults and builds a differently-clipped observation than the policy
# was trained against, which would corrupt every action without erroring.
_e = agent_cfg["params"]["env"]
env = RlGamesVecEnvWrapper(
    env,
    agent_cfg["params"]["config"]["device"],
    _e.get("clip_observations", math.inf),
    _e.get("clip_actions", math.inf),
    _e.get("obs_groups"),
    _e.get("concate_obs_groups", True),
)
vecenv.register("IsaacRlgWrapper", lambda n, a, **kw: RlGamesGpuEnv(n, a, **kw))
env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
agent_cfg["params"]["load_checkpoint"] = True
agent_cfg["params"]["load_path"] = args.checkpoint
agent_cfg["params"]["config"]["num_actors"] = raw.num_envs
runner = Runner()
runner.load(agent_cfg)
agent: BasePlayer = runner.create_player()
agent.restore(args.checkpoint)
agent.reset()

obs = env.reset()
if isinstance(obs, dict):
    obs = obs["obs"]
_ = agent.get_batch_size(obs, 1)
if agent.is_rnn:
    agent.init_rnn()


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


checked = 0          # env-steps geometrically evaluated
violating = 0        # env-steps with at least one penetrating pair
pair_hits: dict[tuple[str, str], int] = {}
worst = 0.0
depths: list[float] = []   # mm, one entry per penetrating pair-instance
deepest = 0.0

with torch.inference_mode():
    for step in range(args.steps):
        actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
        obs, _, dones, _ = env.step(actions)
        if isinstance(obs, dict):
            obs = obs["obs"]
        if step % args.every:
            continue
        pos = robot.data.body_pos_w.cpu().numpy()
        quat = robot.data.body_quat_w.cpu().numpy()
        for e in range(raw.num_envs):
            checked += 1
            world, cen = {}, {}
            for b, local in hulls.items():
                i = names.index(b)
                R = quat_to_R(quat[e, i])
                world[b] = local @ R.T + pos[e, i]
                cen[b] = centre[b] @ R.T + pos[e, i]
            bad = False
            for a, b in PAIRS:
                # Bounding-sphere reject first; almost every pair fails it and Delaunay is the cost.
                if np.linalg.norm(cen[a] - cen[b]) > radius[a] + radius[b]:
                    continue
                A, B = world[a], world[b]
                try:
                    dA, dB = Delaunay(A), Delaunay(B)
                except Exception:
                    continue
                f = max((dA.find_simplex(B) >= 0).mean(), (dB.find_simplex(A) >= 0).mean())
                if f > 0:
                    bad = True
                    pair_hits[(a, b)] = pair_hits.get((a, b), 0) + 1
                    worst = max(worst, f)
                    # DEPTH IN MILLIMETRES, which is the only interpretable form of this. A vertex
                    # fraction cannot distinguish "fingers pass through each other" from "fingers
                    # touch and PhysX allows its contact/rest offset", because both put some vertex
                    # inside the other hull. Depth separates them: contact tolerance is tenths of a
                    # millimetre, a finger through a finger is millimetres.
                    # For a convex hull in halfspace form (n.x + d <= 0), a contained point's
                    # distance to the nearest face is -max_i(n_i.x + d_i). Exact, and cheap.
                    d_mm = 0.0
                    for pts, hull_pts in ((B, A), (A, B)):
                        try:
                            eq = ConvexHull(hull_pts).equations
                        except Exception:
                            continue
                        sd = pts @ eq[:, :3].T + eq[:, 3]      # (n_pts, n_faces)
                        inside = sd.max(axis=1)
                        if (inside < 0).any():
                            d_mm = max(d_mm, float(-inside.min()) * 1000.0)
                    depths.append(d_mm)
                    deepest = max(deepest, d_mm)
            violating += bool(bad)

rate = violating / checked if checked else float("nan")
print(f"\n=== {args.tag}: finger interpenetration over {checked} evaluated env-steps ===", flush=True)
print(f"  steps with >=1 penetrating non-adjacent finger pair: {violating}/{checked} = {rate * 100:.2f}%", flush=True)
print(f"  worst single-pair vertex overlap: {worst * 100:.1f}%", flush=True)
if depths:
    d = np.array(depths)
    print(f"  penetration DEPTH mm over {len(d)} penetrating pair-instances: "
          f"p50 {np.percentile(d, 50):.2f}  p90 {np.percentile(d, 90):.2f}  "
          f"p99 {np.percentile(d, 99):.2f}  max {d.max():.2f}", flush=True)
    for thr in (0.5, 1.0, 2.0, 5.0):
        print(f"    deeper than {thr:4.1f} mm: {int((d > thr).sum()):5d} / {len(d)} "
              f"({100.0 * (d > thr).mean():5.1f}%)", flush=True)
else:
    print("  penetration DEPTH: no penetrating pair-instances", flush=True)
for (a, b), n in sorted(pair_hits.items(), key=lambda kv: -kv[1])[:12]:
    print(f"    {n:5d}  {a:16s} {b}", flush=True)
if not pair_hits:
    print("    NONE", flush=True)
print(f"PEN_RESULT {args.tag} rate={rate:.4f} worst={worst:.4f} steps={checked} "
      f"deepest_mm={deepest:.3f} "
      f"p90_mm={np.percentile(depths, 90) if depths else 0.0:.3f} "
      f"frac_over_1mm={float((np.array(depths) > 1.0).mean()) if depths else 0.0:.4f}", flush=True)
print("PEN_OK", flush=True)
env.close()
simulation_app.close()
