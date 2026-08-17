# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline analysis: does the converted grasp set hold the leg in an insertion-compatible
orientation? (bead: RestingEEGrasped/PartiallyAssembledEEGrasped, orientation-incompatibility check,
2026-08-17)

WHY THIS EXISTS. ObjectPartiallyAssembledEEGrasped fails on a DIFFERENT gate than
ObjectRestingEEGrasped: not_abnormal and coll_free stay healthy (no velocity explosion, nothing
interpenetrating), but orient_down collapses to 1-8/32 per batch. orient_down
(terminations.py::check_reset_state_success) asks whether the PALM's own approach direction points
within a 60-degree cone of world -Z. PartiallyAssembledEEGrasped chains TWO independent placements:
the object's pose comes from partial_assemblies.pt (fixed relative to the FIXTURE, presumably
already insertion-compatible -- tip-down -- by construction), and the palm's pose comes from
grasps.pt (fixed relative to the OBJECT, via the converter this bead built). Those two constraints
are only BOTH satisfiable if the grasp's own relative orientation is compatible with an
insertion-compatible object pose. Since the grasps came from a Stage-2 LIFT checkpoint that was never
rewarded for orientation (a separate Stage-3 pose/reorient checkpoint exists precisely because
orientation is a different objective), there is no a priori reason they should be.

THIS IS A PURE OFFLINE ANALYSIS over grasps.pt -- no Isaac, no GPU, torch only (quaternion ops are
hand-rolled below so this doesn't even need isaaclab.utils.math). For every recorded grasp, it asks:
if the palm were held in AN insertion-compatible orientation (palm's own approach direction pointing
exactly along world -Z -- the center of the orient_down gate's 60-degree cone, sourced from
Robots/DeltoHand/metadata.yaml's gripper_approach_direction, not guessed), where would the leg's long
axis end up pointing? The angle to -Z is invariant to the residual ROLL ambiguity about the approach
axis (any rotation about an axis that itself maps to -Z leaves every other vector's angle to -Z
unchanged), so a single representative "insertion posture" is enough to answer the question
regardless of which of the family of roll angles the real IK solver would have picked.

relative_orientation in grasps.pt is T_object_gripper (palm pose expressed in the OBJECT's frame --
see convert_policy_states_to_grasps.py's own docstring for why, confirmed from the consumer, not
assumed). To go from "palm orientation in world" to "object orientation in world" needs the INVERSE,
T_gripper_object = T_object_gripper^-1, then T_world_object = T_world_gripper (: insertion posture) o
T_gripper_object.

Run (plain python, no Isaac -- any interpreter with torch, e.g. the system python3 that already has
it, or env_uwlab's):
    python3 scripts_v2/tools/analyze_grasp_orientation_distribution.py \\
        --grasps_pt .../Grasps/SquareTableLeg200mmDecomp/grasps.pt
"""

from __future__ import annotations

import argparse
import math

import torch

# gripper_approach_direction from Robots/DeltoHand/metadata.yaml -- read directly, not guessed.
# Palm/mount-local unit vector (verified |.| ~= 1.0 in that file).
_GRIPPER_APPROACH_DIRECTION_LOCAL = torch.tensor([0.2582, 0.4717, 0.8431])

# Leg's long axis in its OWN local frame -- local X, per dexlift_ur5e_delto_tableleg_env_cfg.py's own
# measurement ("extent 0.200 x 0.030 x 0.030 m (long axis local X)").
_LEG_LONG_AXIS_LOCAL = torch.tensor([1.0, 0.0, 0.0])

_WORLD_INSERTION_AXIS = torch.tensor([0.0, 0.0, -1.0])


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a unit quaternion (w, x, y, z) -- its inverse."""
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product, (w, x, y, z) convention, batched over the leading dim(s)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v (..., 3) by unit quaternion q (..., 4), (w, x, y, z) convention."""
    qw = q[..., 0:1]
    qxyz = q[..., 1:4]
    t = 2.0 * torch.cross(qxyz, v, dim=-1)
    return v + qw * t + torch.cross(qxyz, t, dim=-1)


def quat_from_two_vectors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Shortest-arc unit quaternion rotating unit vector a onto unit vector b."""
    dot = (a * b).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    assert dot.min() > -0.999, "a and b are (near-)antiparallel -- degenerate case not handled here"
    axis = torch.cross(a.expand_as(b), b, dim=-1)
    m = torch.sqrt(2.0 + 2.0 * dot)
    w = m / 2.0
    xyz = axis / m
    return torch.cat([w, xyz], dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze grasp-set orientation compatibility with insertion.")
    parser.add_argument("--grasps_pt", type=str, required=True)
    args = parser.parse_args()

    data = torch.load(args.grasps_pt, map_location="cpu", weights_only=False)
    grasp_group = data["grasp_relative_pose"]
    rel_quat = torch.stack(list(grasp_group["relative_orientation"]))  # (G,4), T_object_gripper
    G = rel_quat.shape[0]
    print(f"[analyze] {G} grasps loaded from {args.grasps_pt}", flush=True)

    # -- sanity: quat_from_two_vectors actually maps approach_local -> -Z.
    palm_quat_insertion = quat_from_two_vectors(_GRIPPER_APPROACH_DIRECTION_LOCAL, _WORLD_INSERTION_AXIS)
    check = quat_apply(palm_quat_insertion, _GRIPPER_APPROACH_DIRECTION_LOCAL)
    assert torch.allclose(check, _WORLD_INSERTION_AXIS, atol=1e-4), f"self-check failed: {check}"
    print("[analyze] insertion-posture quaternion self-check passed (approach_local -> world -Z)", flush=True)

    # -- T_world_object = T_world_gripper(insertion posture) o T_gripper_object(= inverse of recorded)
    gripper_object = quat_inv(rel_quat)  # (G,4)
    leg_quat_world = quat_mul(palm_quat_insertion.expand(G, 4), gripper_object)  # (G,4)
    leg_long_axis_world = quat_apply(leg_quat_world, _LEG_LONG_AXIS_LOCAL.expand(G, 3))  # (G,3)

    cos_angle = (leg_long_axis_world * _WORLD_INSERTION_AXIS).sum(-1).clamp(-1.0, 1.0)
    angle_deg = torch.acos(cos_angle) * 180.0 / math.pi  # 0 = long axis points exactly -Z (tip down)
    # axis-alignment (rod is symmetric end-for-end for "vertical or not" purposes): 0 = perfectly
    # vertical (either end down), 90 = perfectly horizontal.
    axis_alignment_deg = torch.minimum(angle_deg, 180.0 - angle_deg)

    def stats(name: str, x: torch.Tensor) -> None:
        print(
            f"[analyze] {name}: min {x.min():.2f}  median {x.median():.2f}  mean {x.mean():.2f}"
            f"  max {x.max():.2f}  p5 {torch.quantile(x, 0.05):.2f}  p95 {torch.quantile(x, 0.95):.2f}",
            flush=True,
        )

    print("\n=== LEG LONG-AXIS ORIENTATION, IF THE PALM WERE HELD FOR INSERTION (approach -> world -Z) ===",
          flush=True)
    stats("angle to -Z, signed 0-180 deg (0 = tip-end points down)", angle_deg)
    stats("axis alignment, 0-90 deg (0 = vertical either way, 90 = horizontal)", axis_alignment_deg)

    for thresh in (15.0, 30.0, 45.0, 60.0):
        frac = (axis_alignment_deg < thresh).float().mean()
        print(f"[analyze] fraction within {thresh:.0f} deg of vertical (either end down): {frac:.3f}", flush=True)

    # the orient_down gate itself is a 60-degree cone on the PALM's approach direction, not directly
    # on the leg -- report both for completeness, but the leg-axis number is what determines whether
    # partial-assembly's OWN object orientation (tip-down, by construction) is reachable at all
    # while also satisfying this grasp.
    print(
        "\n[analyze] NOTE: orient_down itself gates the PALM's approach direction within 60 deg of -Z,"
        " not the leg's axis directly. But partial-assembly places the OBJECT (tip-down, by"
        " construction) FIRST; the grasp then fixes the palm RELATIVE TO THE OBJECT. If the leg's"
        " long axis (given an insertion-posture palm) is far from vertical here, then holding the"
        " object tip-down via this grasp requires the PALM to be far from the insertion posture --"
        " i.e. the two placements fight each other by construction, not by execution error.",
        flush=True,
    )


if __name__ == "__main__":
    main()
