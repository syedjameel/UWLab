"""Compare the HAND collider representation in our USD against the certified reference USD.

WHY THIS AND NOT THE AUDIT NOTE. An earlier audit reported "ours are SDF, the reference's are convex
hull" and that claim has never been checked against the assets themselves -- the same shape of error
as the self-collision claim, which was an inference from a throughput A/B and turned out backwards.
This reads the USDs.

WHAT MAKES IT DECISIVE. The two runs' serialized configs demand wildly different contact budgets for
the same 2048 envs and the same object set:
    ours  sim.physx.gpu_collision_stack_size = 4_026_531_840   (3.75 GiB)
    ref   sim.physx.gpu_collision_stack_size =    67_108_864   (64 MiB)
A 60x gap in contact-pair volume has to come from the collision geometry, and the hand is the only
part with 20+ moving bodies.
"""

from __future__ import annotations

import collections
import sys

from pxr import Usd, UsdPhysics

OURS = "/home/dom-iva/github.com/orel/lerobot/UWLab_ur5edelto/source/uwlab_assets/uwlab_assets/local/Robots/Ur5eDelto/ur5e_delto.usd"
# REF points into a SEPARATE repo (IsaacLabDexterous), a sibling of this one on the machine this
# script was written on -- it will NOT exist on DL_A6000 or any freshly provisioned host (e.g. the
# 5090) unless that repo is separately vendored there too. No CLI override; edit this constant or
# vendor the reference repo before running this script anywhere else.
REF = "/home/dom-iva/github.com/orel/IsaacLabDexterous/source/isaaclab_assets/data/robots/URTessoloAlik/ur10e_delto_optimized_separate_tips_limited_jnts_self_collision.usd"


def survey(path: str, label: str) -> None:
    stage = Usd.Stage.Open(path)
    if stage is None:
        print(f"{label}: CANNOT OPEN {path}")
        return

    approx = collections.Counter()
    hand_approx = collections.Counter()
    n_collision = 0
    tips = []

    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        n_collision += 1
        mesh = UsdPhysics.MeshCollisionAPI(prim)
        attr = mesh.GetApproximationAttr()
        kind = attr.Get() if attr and attr.HasAuthoredValue() else "(unauthored)"
        p = str(prim.GetPath())
        approx[kind] += 1
        # The DELTO hand's bodies are the rl_dg_* family in both trees.
        if "rl_dg" in p or "dg5f" in p:
            hand_approx[kind] += 1
            if "_tip" in p:
                tips.append((p.split("/")[-2] if "/" in p else p, kind))

    print(f"\n=== {label} ===\n{path}")
    print(f"collision prims total: {n_collision}")
    print(f"  approximation, ALL bodies : {dict(approx)}")
    print(f"  approximation, HAND bodies: {dict(hand_approx)}")
    for name, kind in tips[:8]:
        print(f"    tip {name:24s} {kind}")


survey(OURS, "OURS  ur5e_delto")
survey(REF, "REF   ur10e_delto_optimized_separate_tips_limited_jnts_self_collision")
print("\nNOTE: 'none'/'(unauthored)' on a Mesh means PhysX uses the TRIANGLE MESH as-is;")
print("'convexHull'/'convexDecomposition' are the cheap stable forms; 'sdf' is the expensive one.")
sys.stdout.flush()
