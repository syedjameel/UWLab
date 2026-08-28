#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""IK-SOLVED deep C4 (ObjectPartiallyAssembledEEGrasped / "Near Goal" / task_3) reset bank.

WHY THIS SCRIPT EXISTS, AND WHY NOT THE OTHER TWO ROUTES ALREADY IN THIS DIRECTORY:

  POLICY route (record_reset_states.py rolling the trained policy forward from a deep spawn):
  measured incapable. The withdrawal policy converges to a constant 2-4mm final depth regardless
  of spawn depth (fit slope -0.210, R2 0.083 over spawns 10.4-20.8mm) -- it is not a rung, it is a
  different task the policy has already solved (backing out), so it cannot be used to MANUFACTURE
  the near-goal rung the architecture needs.

  COMPOSED route (gen_composed_c4_reset_bank.py, written, never run): nearest-neighbour match a
  required palm pose against arm configurations already recorded in the C2/C3 banks. Measured
  BLOCKED: position matches to 0.336mm but orientation does not -- rot_err min 18.07deg, p10 83deg,
  median 122deg, zero matches in all 12 (pos, rot) threshold cells swept. Root cause is structural,
  not a tuning problem: a seated leg is VERTICAL, TIP-DOWN (leg metadata.yaml's assembled_offset is
  quat=[0.70710678,0,0.70710678,0], i.e. root_quat = Ry(-90) once composed -- see that file's own
  derivation comment), while every C2/C3 state holds the leg HORIZONTAL. A grasp is stored relative
  to the leg (grasps.pt's relative_position/relative_orientation), so it rotates rigidly with the
  leg -- the C2/C3 arm-configuration library has literally never held this leg in the attitude
  insertion requires, at any recorded arm configuration, and no amount of matching can manufacture
  an orientation that was never recorded.

  THIS SCRIPT (IK route): the required palm pose is fully determined by (leg pose, grasp) --
  T_world_palm = T_world_leg . T_leg_palm -- so it must be SOLVED FOR with the arm's own inverse
  kinematics, not looked up in a library that never visited it. See "IS IK THE RIGHT TOOL" below
  for why this reframes the search as being over GRASPS, with IK as a per-(pose, grasp) feasibility
  test, not an optimizer with room to explore.

IS IK THE RIGHT TOOL, OR DOES THE 6-DOF ARM OVER-CONSTRAIN THIS AGAINST A FIXED GRASP? Yes, and
the over-constraint is real and shapes the loop below. The UR5e arm has 6 joints; a palm pose is a
6-DoF target (3 position + 3 orientation). Once (leg pose, grasp) picks ONE target pose, the
Jacobian null space at that target is 0-dimensional for a non-singular configuration -- there is no
continuous freedom left for IK to exploit, only a FINITE set of discrete IK branches (elbow-up vs
elbow-down, wrist-flipped vs not), reachable only by varying the SEED joint configuration IK starts
from, not by varying anything about the target itself. Differential IK converges to whichever
branch is nearest its seed and can miss an existing feasible branch entirely if seeded far from it
-- exactly the failure mode "joint limit lockout" describes below. So:
  - The real, continuous search is over GRASPS (3000 available per leg): each grasp choice moves
    the target palm pose to a materially different point in SE(3) relative to the leg, and IS
    the lever this script actually has for turning an infeasible pose into a feasible one.
  - The SEED configuration is a secondary, discrete lever for branch selection at a FIXED grasp --
    worth trying a small number of alternates (this script tries a home seed and one shoulder-pan-
    flipped alternate by default, --n-seeds) before giving up on that grasp, but not worth an
    exhaustive branch search: cheaper to move on to the next grasp candidate.
  - Consequently the loop below is a GRASP SWEEP with IK as a yes/no oracle per candidate, early-
    exiting per POSE on the first (grasp, seed) pair that clears the acceptance gate -- not a
    pose-by-pose numerical optimization and not an exhaustive pose x grasp x seed Cartesian product
    (4096 x 3000 x n_seeds IK solves would be enormous and pointless once a pose already has an
    accepted match). See "LOOP STRUCTURE" below.

JOINT LIMITS. The composed route's own scripted-replay ancestor (referenced in this project's
history, not reproduced here) is known to have produced solutions PINNED at joint limits that
looked superficially fine. This script does not clamp a limit-violating solution and keep it: after
the full-turn wrap that undoes any winding artifact (`_wrap_joints_into_limits`, copied verbatim
from omnireset/mdp/events.py -- see that file for why the wrap must come first, before any limit
judgement, since a wound angle is not actually at the limit), every ARM joint is checked against its
true USD limits with a margin (`--joint-limit-margin-deg`, default 1.0deg); a solution with any arm
joint inside that margin is REJECTED outright as a probable branch lockout, never silently clamped
to the boundary and accepted. A clamped-and-accepted state is a worse defect than a rejected one: it
writes a joint target the articulation cannot actually reach, and PhysX resolves the violation
during the episode by dragging the joint at its velocity cap -- this project has already measured
exactly that failure mode once (C1 98% / C3 73% spinning wrist_3, from the OLDER unwrapped IK
replay), and the wrap fixes the winding half of it, not the genuine-overreach half.

PHYSICS SETTLE: NOT PERFORMED, DELIBERATELY. Every pose this script writes is placed KINEMATICALLY
(root_pose writes, joint_position writes) with no `sim.step()` between placing the leg and reading
back the achieved palm pose -- FK only (`sim.forward()` + `Articulation.update(0.0)`, a kinematic
refresh, never a physics tick), exactly mirroring gen_composed_c4_reset_bank.py's own FK convention
and the production reset-event pipeline itself (every EventTerm in omnireset/mdp/events.py sets
state and calls `env.scene.write_data_to_sim()`; physics only starts stepping once the episode
proper begins, after every reset event has run). Settling INSIDE this generator would risk exactly
the failure the deep C4 band exists to avoid: the leg's placement is only "near goal" across a
~2.5mm slice of a 13mm spawn band (depth >= 22.5mm out of [12,25]mm, see partial_assemblies_deep_v2
provenance), and letting contact resolve BEFORE the state is written could push the leg axially out
of that slice -- silently converting an accepted, in-band state into one that fails the true
predicate for a reason this script would never see, since no depth check would run again after a
settle. If anyone wants to know whether the WRITTEN bank survives contact once training actually
uses it, that is a downstream, live-env measurement (a handful of `env.reset()` + a few
`env.step()`s under the real physics/friction/mass settings, checking whether the leg's mating-
frame depth stays in band and whether the object is still held after 1-2s) -- deliberately not
folded into this offline generator, the same way the production reset events never settle either.

HAND JOINTS ACROSS A GRAVITY-DIRECTION CHANGE THIS PROJECT HAS NOT TESTED: the closed hand posture
recorded in grasps.pt is defined in the LEG's own local frame (grasps.pt's relative_position/
relative_orientation, replayed via `combine_frame_transforms(object_pos_w, object_quat_w, rel_pos,
rel_quat)` -- see omnireset/mdp/events.py's reset_end_effector_from_grasp_dataset, the exact idiom
this script matches term-for-term), so the POSTURE transfers geometrically: the fingers close
around the same local cross-section of the leg regardless of the leg's world orientation, by
construction. That is NOT the same claim as "the grasp holds." Every one of those 3000 grasps was
presumably validated (shake-tested or similar) with the leg HORIZONTAL, i.e. with gravity acting
mostly ACROSS the fingers' closing plane. Here the leg is VERTICAL, TIP-DOWN, so gravity acts along
the leg's long axis -- directly along the one direction a radial finger-pinch resists only through
friction, not through any mechanical stop, and directly along the same axis this bank's whole depth
band is measured on. A grasp whose margin was validated against a side-load may have materially
less margin against an axial pull-out load it was never tested under. This is the single highest-
risk, LEAST-verified assumption this script makes, and it is a live-env question this script cannot
answer by construction (see the settle note above) -- flagged here as the first thing to check with
a real physics smoke test before trusting this bank's held-object rate, not something this script
silently assumes is fine.

LOOP STRUCTURE. For each of the n_poses partial-assembly poses (independent across poses -- no
information is shared pose-to-pose), try a per-pose-shuffled sequence of grasp candidates, each at
--n-seeds alternate arm seed configurations, STOPPING at the first (grasp, seed) pair whose IK
solution clears the acceptance gate (`_ik_attempt_batch` below). Attempts are batched ACROSS POSES,
not across grasps-per-pose: round r solves IK for every still-unresolved pose's r-th candidate in
one vectorized DifferentialIKController call (chunked into groups of --chunk-size spawned robot
clones, mirroring gen_composed_c4_reset_bank.py's `_fk_batch` clone-per-instance pattern, since
IsaacLab's IK controller is natively batched over "environments" and there is no cheaper way to get
a real per-instance Jacobian than one Articulation instance per candidate). This bounds total IK
solves to at most n_poses x grasps_per_pose x n_seeds, with per-pose early exit, rather than the
full (poses x 3000 grasps) Cartesian product -- consistent with the "IS IK THE RIGHT TOOL" answer
above: grasp choice is the actual search dimension, so the loop should give up on a pose's CURRENT
grasp quickly (few seed retries) and move on to the NEXT grasp, rather than spend the seed budget
exhaustively on one candidate.

NOTE ON WHAT THE WRITTEN YIELD ACTUALLY MEASURES: IK never touches the leg's pose -- only the arm.
The TRUE task_3 predicate (0.0025m / 0.025rad, both terms) is measured independently after writing
(see "INDEPENDENT VALIDATION" below) against the leg pose alone, which is exactly whatever
partial_assemblies_deep_v2.pt + the sampled fixture placement already produced in step 1-2, before
any grasp or IK is involved. This means the TRUE-predicate PASS FRACTION this script reports is a
property of the deep-band POSE SELECTION (what fraction of the 4096 poses already sit at depth
>=22.5mm, per the 25.0-depth_mm pos_err relation), not of IK/grasp yield -- IK/grasp yield only
determines how many of THOSE geometrically-passing poses actually get a written, hand-holding
state. Reported separately below (`geom_gate_fraction` vs `written_fraction_of_geom_passing`) so a
reader does not mistake one number for the other; multiplying them gives the number the whole
exercise exists to produce.

DOES NOT RUN ITSELF: needs a real Isaac boot for IK/FK and the resolved robot.joint_names/joint
limits. Written to be executed on an Isaac-capable box, not run here (no Isaac, no GPU available in
this session -- see the team message this script ships with).

Run (mirrors gen_composed_c4_reset_bank.py's env/path conventions):
    PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl" \\
        timeout -s KILL 3600 <python> -u scripts_v2/tools/gen_ik_c4_reset_bank.py \\
        --grasps-path Datasets_ur5e_delto/OmniReset/Grasps/SquareTableLeg200mmDecomp/grasps.pt \\
        --partial-assembly-path local_ckpts/deep_c4_partial_assemblies_v1/partial_assemblies_deep_v2.pt \\
        --out local_ckpts/deep_c4_partial_assemblies_v1/resets_ObjectPartiallyAssembledEEGrasped_ik_v1.pt \\
        --headless
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Solve IK to build a deep C4 reset bank from (leg pose, grasp) pairs.")
parser.add_argument("--grasps-path", type=str, required=True, help="Grasps/SquareTableLeg200mmDecomp/grasps.pt")
parser.add_argument(
    "--partial-assembly-path", type=str, required=True,
    help="Deep-band partial_assemblies.pt (e.g. partial_assemblies_deep_v2.pt, n=4096, depth band [12,25]mm).",
)
parser.add_argument("--out", type=str, required=True, help="Output .pt path. MUST NOT already exist.")

# --- fixture placement: SAMPLED per pose from the training scene's own distribution, never a
# single fixed pose (a fixed placement would be a new, unrequested degeneracy axis this bank does
# not need -- same reasoning gen_composed_c4_reset_bank.py's per-pose sampling already used).
# Defaults below are copied from reset_states_cfg.py's ResetStatesBaseEventCfg.
# reset_receptive_object_pose (the base class every *EEGrasped/*EEAnywhere reset-state event
# inherits), NOT re-derived -- verify against that file if it ever changes; do not trust these
# literals blindly.
parser.add_argument("--fixture-x-range", type=str, default="0.35,0.60", help="reset_states_cfg.py pose_range['x'].")
parser.add_argument("--fixture-y-range", type=str, default="-0.2,0.2", help="reset_states_cfg.py pose_range['y'].")
parser.add_argument(
    "--fixture-yaw-range-deg", type=str, default="-15.0,15.0",
    help="reset_states_cfg.py pose_range['yaw'] is (-pi/12, pi/12) = +-15deg.",
)
parser.add_argument(
    "--fixture-z-m", type=float, default=0.019625,
    help="reset_states_cfg.py's pose_range['z'] is (0.0, 0.0) with offset_asset_cfg=ur5_metal_support,"
    " use_bottom_offset=True -- the resolved constant is 0.004 (support root z) - (-0.015625)"
    " (fixture bottom_offset.pos.z, OneLegInsertionFixture/metadata.yaml) = 0.019625, matching"
    " C3_PAPER_SCALE_PROVENANCE.md's measured receptive_object z (mean 0.019625, std 0.0).",
)

# --- grasp/seed search budget (see "LOOP STRUCTURE" above).
parser.add_argument(
    "--grasps-per-pose", type=int, default=40,
    help="Max distinct grasp candidates tried per pose (from a per-pose random permutation of the"
    " full grasps.pt, so different poses do not all fail on the same bad candidate together)"
    " before giving up on that pose. 4096 poses x 3000 grasps is a large candidate pool; this caps"
    " the WORST-CASE per-pose cost, not the pool size actually available.",
)
parser.add_argument(
    "--n-seeds", type=int, default=2,
    help="Alternate arm seed configurations tried per grasp candidate before moving to the next"
    " grasp (see 'JOINT LIMITS'/'IS IK THE RIGHT TOOL' above: seed choice only selects an IK"
    " BRANCH at a fixed target, it does not change what is reachable). Seed 0 is the robot's own"
    " default/home joint_pos; seed 1+ perturb shoulder_pan by +180deg per seed index, a standard"
    " elbow/shoulder branch-flip heuristic for a 6-DOF serial arm.",
)
parser.add_argument("--max-poses", type=int, default=None, help="Cap on partial-assembly poses used. None = all.")
parser.add_argument(
    "--target-n", type=int, default=None,
    help="Stop once this many poses have an ACCEPTED state (early exit across the whole pose set,"
    " not per-pose). None = attempt every pose to its full --grasps-per-pose x --n-seeds budget.",
)

# --- IK mechanics (mirrors omnireset/mdp/events.py's reset_end_effector_from_grasp_dataset /
# reset_end_effector_round_fixed_asset exactly: same controller, same damped-update loop, same
# iteration count/step size, same post-hoc joint-limit wrap -- see module docstring for why this
# script does NOT invent a different integration scheme).
parser.add_argument("--ik-iterations", type=int, default=25, help="Matches delto_cfg.py's DELTO-local override (25, not the 2F-85 default 10) -- this hand's off-axis approach needs the same convergence budget the existing EEAnywhere event already required.")
parser.add_argument("--ik-step-size", type=float, default=0.25, help="Matches every existing IK reset event's damped-update fraction.")
parser.add_argument(
    "--pos-accept-mm", type=float, default=3.0,
    help="ACCEPT gate on achieved-vs-requested PALM position, measured by FK-ing the IK solution"
    " back and comparing to the requested target (never trusting the solver's own convergence"
    " claim). Justified against the leg's 15mm half-extent via the hand's own geometry: the"
    " DELTO's pinch point sits 169.7mm from the palm (DeltoHand/metadata.yaml grasp_center_offset,"
    " |.|=169.7mm) -- position error at the palm propagates ~1:1 to the pinch point, so 3mm here"
    " leaves ~12mm of the 15mm half-extent budget for the rotation term below plus the grasp's own"
    " inherent placement quality.",
)
parser.add_argument(
    "--rot-accept-deg", type=float, default=1.0,
    help="ACCEPT gate on achieved-vs-requested PALM orientation. Justified against the SAME 15mm"
    " half-extent, but via the pinch point's 169.7mm lever arm, not the palm: a rotation error at"
    " the palm induces an ADDITIONAL lateral displacement of L*sin(rot_err) at the pinch point"
    " (L=0.1697m). At 1deg that is 169.7*sin(1deg)=~3.0mm; combined with the 3mm position budget"
    " above, ~6mm total pinch-point displacement against a 15mm half-extent -- comfortably 'around'"
    " the leg, not 'beside' it (the old scripted-IK-replay route missed by median 19.72mm/11.12deg,"
    " which at this SAME lever arm is 11.12deg -> 169.7*sin(11.12deg)=~32.7mm of induced lateral"
    " error alone, a clean illustration of why that route produced zero-fingertip-contact banks)."
    " 1.0deg also matches this exact codebase's own established, ACHIEVED precedent: delto_cfg.py"
    " documents 25 IK iterations reducing this hand's off-axis-approach residual 'below the C7"
    " one-degree gate' on a similarly hard reach, so this is not an untested target.",
)
parser.add_argument(
    "--joint-limit-margin-deg", type=float, default=1.0,
    help="Reject (not clamp) a solution if any ARM joint, AFTER the full-turn wrap, is within this"
    " margin of its true USD limit -- treated as a probable branch lockout. See 'JOINT LIMITS' above.",
)
parser.add_argument("--chunk-size", type=int, default=512, help="Robot clones spawned per IK batch (mirrors gen_composed_c4_reset_bank.py's --chunk-size).")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

if os.path.exists(args_cli.out):
    raise FileExistsError(
        f"{args_cli.out} already exists -- refusing to overwrite (versioned output path only; this"
        " project has been bitten by two files sharing a name before)."
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below needs Isaac Sim modules, only importable after AppLauncher starts."""

import torch  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from uwlab_assets.robots.ur5e_delto import IMPLICIT_UR5E_DELTO  # noqa: E402

_ARM_JOINT_PATTERNS = ["shoulder.*", "elbow.*", "wrist.*"]  # ur5e_delto_cfg.py's robot_ik_cfg, unchanged for DELTO
_PALM_BODY_NAME = "rl_dg_mount"  # delto_cfg.py's DELTO_EE_BODY -- the DELTO's palm link
_HAND_JOINT_PATTERNS = [r"rj_dg_[1-5]_[1-4]"]  # delto_cfg.py's _DELTO_HAND_JOINTS, 20 joints


# ============================================================================================
# Copied from gen_composed_c4_reset_bank.py (NOT imported -- that module's sibling, this one,
# both boot their own AppLauncher at import time; importing one from a process that already
# booted the other would re-parse argv and attempt a second simulation app. See that file's own
# docstring for the identical reasoning.) Kept behaviourally identical.
# ============================================================================================

def _load_grasps(path: str, robot_joint_names: list[str], device: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"grasps.pt not found at {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    grasp_group = data.get("grasp_relative_pose", data)
    rel_pos_list = grasp_group.get("relative_position", [])
    rel_quat_list = grasp_group.get("relative_orientation", [])
    gripper_joint_positions_dict = grasp_group.get("gripper_joint_positions", {})
    num_grasps = len(rel_pos_list)
    if num_grasps == 0:
        raise ValueError(f"No grasp data found in {path}")

    rel_pos = torch.stack(
        [p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.float32) for p in rel_pos_list], dim=0
    ).to(device, dtype=torch.float32)
    rel_quat = torch.stack(
        [q if isinstance(q, torch.Tensor) else torch.as_tensor(q, dtype=torch.float32) for q in rel_quat_list], dim=0
    ).to(device, dtype=torch.float32)

    recorded = set(gripper_joint_positions_dict)
    print(f"[gen_ik] grasps.pt: {num_grasps} grasps, gripper joints recorded: {sorted(recorded)}", flush=True)
    overlap = recorded & set(robot_joint_names)
    if not overlap:
        raise ValueError(
            f"gen_ik_c4_reset_bank: zero overlap between grasps.pt's recorded joints ({sorted(recorded)})"
            f" and the robot's expected joints ({sorted(robot_joint_names)}) -- this grasps.pt was very"
            " likely recorded for a different gripper. This script writes gripper_joint_positions into"
            " the bank by joint NAME and must refuse outright if the name vocabulary does not match."
        )
    hand_joint_names = sorted(recorded & set(robot_joint_names))
    return rel_pos, rel_quat, gripper_joint_positions_dict, hand_joint_names


def _load_partial_assemblies(path: str, device: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"partial_assemblies.pt not found at {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    rel_pos, rel_quat = data.get("relative_position"), data.get("relative_orientation")
    if rel_pos is None or rel_quat is None or len(rel_pos) == 0:
        raise ValueError(f"No partial assembly data found in {path}")
    if not isinstance(rel_pos, torch.Tensor):
        rel_pos = torch.as_tensor(rel_pos, dtype=torch.float32)
    if not isinstance(rel_quat, torch.Tensor):
        rel_quat = torch.as_tensor(rel_quat, dtype=torch.float32)
    print(f"[gen_ik] partial_assemblies.pt: {rel_pos.shape[0]} poses", flush=True)
    return rel_pos.to(device, dtype=torch.float32), rel_quat.to(device, dtype=torch.float32)


def _atomic_torch_save(obj, out_path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    tmp_path = os.path.join(out_dir, f".tmp-{os.getpid()}-{os.path.basename(out_path)}")
    with open(tmp_path, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _wrap_joints_into_limits(q: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    """Standalone copy of omnireset/mdp/events.py's `_wrap_joints_into_limits` core math (that
    function operates on a live Articulation + env_ids; this one operates on plain tensors since
    this script does not carry an env). Same reasoning: differential IK is joint-limit-unaware and
    can wind a revolute joint past its USD limits by an exact multiple of 2*pi with no physical
    effect -- undo that winding BEFORE judging whether a solution is a genuine limit violation.
    """
    lo, hi = limits[..., 0], limits[..., 1]
    two_pi = 2.0 * math.pi
    q_wrapped = torch.where((q > hi) & (q - two_pi >= lo), q - two_pi, q)
    q_wrapped = torch.where((q_wrapped < lo) & (q_wrapped + two_pi <= hi), q_wrapped + two_pi, q_wrapped)
    return q_wrapped


# ============================================================================================
# NEW: IK feasibility oracle, grasp-sweep loop, schema construction, atomic write, independent
# reload+validation.
# ============================================================================================

def _ik_attempt_batch(
    robot_cfg,
    seed_joint_pos: torch.Tensor,     # [N, n_joints], FULL joint vector (arm+hand) to seed from
    target_pos_w: torch.Tensor,       # [N, 3] requested palm position, world
    target_quat_w: torch.Tensor,      # [N, 4] requested palm orientation, world (wxyz)
    arm_joint_names: list[str],
    hand_joint_names: list[str],
    sim: SimulationContext,
    device: str,
    ik_iterations: int,
    ik_step_size: float,
    joint_limit_margin_rad: float,
    chunk_size: int,
    chunk_prefix: str,
):
    """Batched differential-IK solve, chunked over spawned robot clones (mirrors
    gen_composed_c4_reset_bank.py's `_fk_batch` clone-per-instance pattern -- there is no cheaper
    way to get a real per-instance Jacobian than one Articulation per candidate).

    Deliberately builds `DifferentialIKController` directly rather than going through
    `DifferentialInverseKinematicsAction` (the ActionTerm every reset event in events.py uses):
    that class requires a live `ManagerBasedEnv` (env.scene, env.action_manager, ...), which this
    offline generator does not build. The Jacobian slicing / root-frame conversion / damped-update
    loop below are copied verbatim from that ActionTerm's own math (isaaclab/envs/mdp/actions/
    task_space_actions.py: `jacobian_b` property and `apply_actions`) and from events.py's
    `reset_end_effector_from_grasp_dataset.__call__` (the damped write loop, ik_step_size fraction
    per iteration, `_wrap_joints_into_limits` afterward) -- so this is the SAME controller class and
    the SAME surrounding numerical recipe the production reset events use, just re-wired to run
    without an env, exactly as directed.

    Returns a dict of per-instance results (all length N, N=seed_joint_pos.shape[0]):
      achieved_pos_w, achieved_quat_w, pos_err_mm, rot_err_deg, joint_position (physical, full
      vector), joint_position_target (IK-commanded arm target + seed hand values -- caller
      overwrites hand indices with the grasp's closed posture only for instances it keeps),
      pinned (bool, any arm joint within margin of its limit after the wrap), accepted (bool,
      caller fills in after applying the position/rotation gates -- this function only fills
      `pinned` and the raw errors).
    """
    n_total = seed_joint_pos.shape[0]
    n_joints = seed_joint_pos.shape[1]
    achieved_pos_w = torch.zeros((n_total, 3), device=device)
    achieved_quat_w = torch.zeros((n_total, 4), device=device)
    joint_position = torch.zeros((n_total, n_joints), device=device)
    joint_position_target_arm = None  # filled per-chunk once arm_joint_ids is known
    pinned = torch.zeros((n_total,), dtype=torch.bool, device=device)

    start = 0
    chunk_idx = 0
    while start < n_total:
        end = min(start + chunk_size, n_total)
        n = end - start
        chunk_idx += 1

        for i in range(n):
            sim_utils.create_prim(f"/World/{chunk_prefix}_{chunk_idx}_{i}", "Xform", translation=(3.0 * i, 0.0, 0.0))
        cfg = robot_cfg.replace(prim_path=f"/World/{chunk_prefix}_{chunk_idx}_.*/Robot")
        robot = Articulation(cfg=cfg)
        sim.reset()

        if not robot.is_fixed_base:
            raise RuntimeError("gen_ik_c4_reset_bank assumes a fixed-base arm (UR5e mounted); is_fixed_base=False.")

        arm_ids, _ = robot.find_joints(arm_joint_names, preserve_order=True)
        hand_ids, _ = robot.find_joints(hand_joint_names, preserve_order=True)
        palm_ids, palm_names = robot.find_bodies(_PALM_BODY_NAME)
        if len(palm_ids) != 1:
            raise RuntimeError(f"Expected exactly one {_PALM_BODY_NAME} body, found {palm_names}")
        palm_id = palm_ids[0]
        # fixed-base: jacobian excludes the (absent) floating-base rows, so body/joint indices for
        # the jacobian are the SAME as the articulation's own body/joint indices minus the base row
        # -- DifferentialInverseKinematicsAction.__init__'s exact rule (task_space_actions.py:73-78).
        jacobi_body_idx = palm_id - 1
        jacobi_joint_ids = arm_ids

        if joint_position_target_arm is None:
            joint_position_target_arm = torch.zeros((n_total, len(arm_ids)), device=device)

        root_pos = torch.zeros((n, 3), device=device)
        root_quat = torch.zeros((n, 4), device=device)
        root_quat[:, 0] = 1.0
        robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1))
        robot.write_root_velocity_to_sim(torch.zeros((n, 6), device=device))
        robot.write_joint_state_to_sim(
            position=seed_joint_pos[start:end], velocity=torch.zeros((n, n_joints), device=device)
        )
        robot.write_data_to_sim()
        sim.forward()
        robot.update(0.0)

        ik_ctrl = DifferentialIKController(
            cfg=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=n,
            device=device,
        )
        # target, in the robot ROOT frame (root is identity here, but go through the real
        # transform rather than assuming so -- robust if this convention ever changes).
        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            robot.data.root_link_pos_w, robot.data.root_link_quat_w,
            target_pos_w[start:end], target_quat_w[start:end],
        )
        ik_ctrl.set_command(torch.cat([target_pos_b, target_quat_b], dim=1))

        last_joint_pos_des_arm = None
        for _ in range(ik_iterations):
            ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
                robot.data.root_link_pos_w, robot.data.root_link_quat_w,
                robot.data.body_pos_w[:, palm_id], robot.data.body_quat_w[:, palm_id],
            )
            jacobian_w = robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, jacobi_joint_ids]
            base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(robot.data.root_quat_w))
            jacobian_b = jacobian_w.clone()
            jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_b[:, :3, :])
            jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_b[:, 3:, :])

            arm_joint_pos = robot.data.joint_pos[:, arm_ids]
            joint_pos_des_arm = ik_ctrl.compute(ee_pos_b, ee_quat_b, jacobian_b, arm_joint_pos)
            last_joint_pos_des_arm = joint_pos_des_arm

            # SAME damped-update recipe as every existing IK reset event (events.py's
            # reset_end_effector_from_grasp_dataset / reset_end_effector_round_fixed_asset): move
            # only ik_step_size of the way toward the controller's solved target per iteration,
            # rather than teleporting straight to it -- avoids overshoot on a near-singular
            # Jacobian, and is the numerically validated recipe this project already relies on.
            delta = ik_step_size * (joint_pos_des_arm - arm_joint_pos)
            robot.write_joint_state_to_sim(
                position=arm_joint_pos + delta,
                velocity=torch.zeros((n, len(arm_ids)), device=device),
                joint_ids=arm_ids,
                env_ids=torch.arange(n, device=device),
            )
            robot.write_data_to_sim()
            sim.forward()
            robot.update(0.0)

        # Full-turn wrap BEFORE judging limits (see _wrap_joints_into_limits docstring).
        limits = robot.data.joint_pos_limits[:, arm_ids]
        wrapped = _wrap_joints_into_limits(robot.data.joint_pos[:, arm_ids], limits)
        if not torch.equal(wrapped, robot.data.joint_pos[:, arm_ids]):
            robot.write_joint_state_to_sim(
                position=wrapped, velocity=torch.zeros((n, len(arm_ids)), device=device),
                joint_ids=arm_ids, env_ids=torch.arange(n, device=device),
            )
            robot.write_data_to_sim()
            sim.forward()
            robot.update(0.0)

        margin_lo = robot.data.joint_pos[:, arm_ids] - limits[..., 0]
        margin_hi = limits[..., 1] - robot.data.joint_pos[:, arm_ids]
        pinned[start:end] = (margin_lo.min(dim=1).values < joint_limit_margin_rad) | (
            margin_hi.min(dim=1).values < joint_limit_margin_rad
        )

        achieved_pos_w[start:end] = robot.data.body_pos_w[:, palm_id, :].clone()
        achieved_quat_w[start:end] = robot.data.body_quat_w[:, palm_id, :].clone()
        joint_position[start:end] = robot.data.joint_pos.clone()
        joint_position_target_arm[start:end] = last_joint_pos_des_arm.clone()

        sim_utils.delete_prim([f"/World/{chunk_prefix}_{chunk_idx}_{i}" for i in range(n)])
        del robot, ik_ctrl

        print(f"[gen_ik] {chunk_prefix} chunk {chunk_idx}: {start}:{end} of {n_total} IK'd", flush=True)
        start = end

    pos_err_mm = (achieved_pos_w - target_pos_w).norm(dim=1) * 1000.0
    rot_err_deg = math_utils.quat_error_magnitude(achieved_quat_w, target_quat_w) * 180.0 / np.pi

    return {
        "achieved_pos_w": achieved_pos_w,
        "achieved_quat_w": achieved_quat_w,
        "pos_err_mm": pos_err_mm,
        "rot_err_deg": rot_err_deg,
        "joint_position": joint_position,
        "joint_position_target_arm": joint_position_target_arm,
        "pinned": pinned,
        "arm_ids": arm_ids,
        "hand_ids": hand_ids,
    }


def _reproject_independent(insertive_root_pose: np.ndarray, receptive_root_pose: np.ndarray):
    """INDEPENDENT reprojection into the mating frame -- scipy Rotation matrices, NEVER
    combine_frame_transforms/subtract_frame_transforms (this generator's own composition tools),
    and never isaaclab.utils.math at all for this step. Copied from gen_composed_c4_reset_bank.py's
    function of the same name (see that file for the generalization-to-non-identity-receptive-pose
    note); constants below are the SAME ones, re-verified directly against the two assets'
    metadata.yaml `assembled_offset` blocks on 2026-08-22 (not merely copied without re-checking):
      leg (SquareTableLeg200mmDecomp/metadata.yaml):    pos=[-0.106203,0,0], quat=[0.70710678,0,0.70710678,0]
      fixture (OneLegInsertionFixture/metadata.yaml):   pos=[-0.056250,0.056250,-0.009374], quat=[1,0,0,0]
      success_thresholds (SAME file): position=0.0025, orientation=0.025 -- the TRUE task_3 gate.
    """
    leg_off_pos = np.array([-0.106203, 0.0, 0.0])
    leg_off_quat_wxyz = np.array([0.70710678, 0.0, 0.70710678, 0.0])
    recv_off_pos = np.array([-0.056250, 0.056250, -0.009374])
    recv_off_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

    def wxyz_to_xyzw(q):
        return q[..., [1, 2, 3, 0]]

    n = insertive_root_pose.shape[0]
    ins_pos, ins_quat_wxyz = insertive_root_pose[:, :3], insertive_root_pose[:, 3:7]
    rec_pos, rec_quat_wxyz = receptive_root_pose[:, :3], receptive_root_pose[:, 3:7]

    R_ins = Rotation.from_quat(wxyz_to_xyzw(ins_quat_wxyz))
    R_leg_off = Rotation.from_quat(wxyz_to_xyzw(np.tile(leg_off_quat_wxyz, (n, 1))))
    align_pos = ins_pos + R_ins.apply(np.tile(leg_off_pos, (n, 1)))
    R_align = R_ins * R_leg_off

    R_rec = Rotation.from_quat(wxyz_to_xyzw(rec_quat_wxyz))
    R_recv_off = Rotation.from_quat(wxyz_to_xyzw(np.tile(recv_off_quat_wxyz, (n, 1))))
    target_pos = rec_pos + R_rec.apply(np.tile(recv_off_pos, (n, 1)))
    R_target = R_rec * R_recv_off

    R_target_inv = R_target.inv()
    rel_pos = R_target_inv.apply(align_pos - target_pos)
    R_rel = R_target_inv * R_align

    depth_into_bore_mm = 25.0 - rel_pos[:, 2] * 1000.0
    lateral_mm = np.hypot(rel_pos[:, 0], rel_pos[:, 1]) * 1000.0
    pos_err_mm = np.linalg.norm(rel_pos, axis=1) * 1000.0
    tilt_deg = np.degrees(np.arccos(np.clip(R_rel.as_matrix()[:, 2, 2], -1.0, 1.0)))
    euler_xyz = R_rel.as_euler("xyz", degrees=True)

    def wrap(a):
        return (a + 180.0) % 360.0 - 180.0

    rot_err_deg = np.abs(wrap(euler_xyz[:, 0])) + np.abs(wrap(euler_xyz[:, 1]))  # e_x, e_y only; e_z (yaw) discarded
    return depth_into_bore_mm, lateral_mm, tilt_deg, pos_err_mm, rot_err_deg


def _true_predicate_torch_cross_check(insertive_root_pose: torch.Tensor, receptive_root_pose: torch.Tensor):
    """SECOND, isaaclab-math-based cross-check of the TRUE task_3 predicate, independent of the
    scipy path above in IMPLEMENTATION (not in constants -- same assembled_offset numbers, since
    those are asset properties, not a computation to duplicate two different ways). Exists because
    the team message flagged a specific, real footgun: `isaaclab.utils.math.compute_pose_error`
    defaults to `rot_error_type="axis_angle"` (a 3-vector); the training code's OWN predicate
    (omnireset/mdp/rewards.py's ProgressContext) needs the relative QUATERNION to feed
    `euler_xyz_from_quat`, so this function calls `compute_pose_error(..., rot_error_type="quat")`
    EXPLICITLY -- calling it without that argument returns axis-angle and `euler_xyz_from_quat`
    then either crashes on the shape mismatch or silently misreads a 3-vector as a quaternion,
    depending on isaaclab's version. Mirrors ProgressContext's own formula exactly (offset-apply,
    subtract-into-mating-frame, euler_xyz_from_quat, wrap_to_pi(e_x).abs()+wrap_to_pi(e_y).abs(),
    e_z discarded) rather than re-deriving it.
    """
    leg_off_pos = torch.tensor([-0.106203, 0.0, 0.0], device=insertive_root_pose.device)
    leg_off_quat = torch.tensor([0.70710678, 0.0, 0.70710678, 0.0], device=insertive_root_pose.device)
    recv_off_pos = torch.tensor([-0.056250, 0.056250, -0.009374], device=insertive_root_pose.device)
    recv_off_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=insertive_root_pose.device)

    n = insertive_root_pose.shape[0]
    ins_pos, ins_quat = insertive_root_pose[:, :3], insertive_root_pose[:, 3:7]
    rec_pos, rec_quat = receptive_root_pose[:, :3], receptive_root_pose[:, 3:7]

    align_pos, align_quat = math_utils.combine_frame_transforms(
        ins_pos, ins_quat, leg_off_pos.expand(n, -1), leg_off_quat.expand(n, -1)
    )
    target_pos, target_quat = math_utils.combine_frame_transforms(
        rec_pos, rec_quat, recv_off_pos.expand(n, -1), recv_off_quat.expand(n, -1)
    )
    rel_pos, rel_quat = math_utils.compute_pose_error(target_pos, target_quat, align_pos, align_quat, rot_error_type="quat")
    e_x, e_y, _ = math_utils.euler_xyz_from_quat(rel_quat)
    rot_err_rad = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
    pos_err_m = rel_pos.norm(dim=1)
    return pos_err_m, rot_err_rad


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    sim_cfg = sim_utils.SimulationCfg(device=device)
    sim = SimulationContext(sim_cfg)

    robot_cfg = IMPLICIT_UR5E_DELTO.copy()
    # GENERATION-PLANT ALIGNMENT (bead UWLab-snuv follow-on, ur5e_delto_cfg.py): IMPLICIT_UR5E_DELTO's
    # OWN spawn.usd_path/actuators are the ASSET DEFAULT plant (sdf hand colliders, 0.06-0.17 N.m /
    # 3.0 rad/s hand actuator) -- NOT what any held reset bank or the certified checkpoint was
    # produced under. Every OmniReset UR5eDelto task config calls _apply_ur5e_delto_generation_plant
    # right after binding this same robot object, swapping in the hullfix3 USD (25 convexHull + 3
    # convexDecomposition, zero sdf) and the reference hand actuator (30 N.m / 10000 rad/s) -- per
    # that function's own docstring, the asset default "cannot be used as-is (a plain convexHull
    # hand explodes the articulation at reset; the identified actuator caps below the closing speed
    # a policy commands and cannot close the hand at all)". Solving IK / recording joint targets
    # against the WRONG plant would silently make every number in this script's output describe a
    # robot no downstream consumer actually spawns. Reused via the real function (not re-derived)
    # so this cannot drift from what ur5e_delto_cfg.py's own classes do; wrapped in a bare
    # SimpleNamespace since that function only ever touches cfg.scene.robot's OWN attributes.
    from types import SimpleNamespace  # noqa: E402
    from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.ur5e_delto_cfg import (  # noqa: E402
        _apply_ur5e_delto_generation_plant,
        _assert_ur5e_delto_generation_plant,
    )
    _plant_holder = SimpleNamespace(scene=SimpleNamespace(robot=robot_cfg))
    _apply_ur5e_delto_generation_plant(_plant_holder)
    _assert_ur5e_delto_generation_plant(_plant_holder)
    robot_cfg = _plant_holder.scene.robot

    partial_rel_pos, partial_rel_quat = _load_partial_assemblies(args_cli.partial_assembly_path, device)
    if args_cli.max_poses is not None:
        partial_rel_pos, partial_rel_quat = partial_rel_pos[: args_cli.max_poses], partial_rel_quat[: args_cli.max_poses]
    n_poses = partial_rel_pos.shape[0]

    # ---- resolve robot.joint_names / joint limits / default seed pose (post-spawn only) ----
    for i in range(1):
        sim_utils.create_prim(f"/World/JointNameProbe_{i}", "Xform", translation=(0.0, 0.0, 0.0))
    probe_cfg = robot_cfg.replace(prim_path="/World/JointNameProbe_.*/Robot")
    probe_robot = Articulation(cfg=probe_cfg)
    sim.reset()
    robot_joint_names = list(probe_robot.joint_names)
    n_joints = len(robot_joint_names)
    name_to_idx = {name: i for i, name in enumerate(robot_joint_names)}
    default_joint_pos = probe_robot.data.default_joint_pos[0].clone()  # [n_joints], home seed 0
    print(f"[gen_ik] resolved robot.joint_names ({n_joints}): {robot_joint_names}", flush=True)

    grasp_rel_pos, grasp_rel_quat, gripper_joint_positions_dict, hand_joint_names = _load_grasps(
        args_cli.grasps_path, robot_joint_names, device
    )
    n_grasps_total = grasp_rel_pos.shape[0]
    arm_joint_names_resolved, _ = probe_robot.find_joints(_ARM_JOINT_PATTERNS, preserve_order=True)
    arm_joint_names_resolved = [robot_joint_names[i] for i in arm_joint_names_resolved]
    if len(arm_joint_names_resolved) != 6:
        raise RuntimeError(
            f"Expected 6 arm joints from {_ARM_JOINT_PATTERNS}, resolved {arm_joint_names_resolved}"
            " -- this script assumes a 6-DOF UR5e arm (see 'IS IK THE RIGHT TOOL' in the module"
            " docstring, which depends on the arm being exactly as constrained as its target)."
        )
    gjp_full = torch.stack(
        [torch.as_tensor([float(v) for v in gripper_joint_positions_dict[n]], dtype=torch.float32) for n in hand_joint_names],
        dim=1,
    ).to(device)  # [n_grasps_total, n_hand]
    hand_indices = [name_to_idx[n] for n in hand_joint_names]
    arm_indices = [name_to_idx[n] for n in arm_joint_names_resolved]

    # ---- per-pose fixture placement, sampled from the measured training-time distribution ----
    x_lo, x_hi = (float(v) for v in args_cli.fixture_x_range.split(","))
    y_lo, y_hi = (float(v) for v in args_cli.fixture_y_range.split(","))
    yaw_lo, yaw_hi = (math.radians(float(v)) for v in args_cli.fixture_yaw_range_deg.split(","))
    rec_x = math_utils.sample_uniform(x_lo, x_hi, (n_poses,), device=device)
    rec_y = math_utils.sample_uniform(y_lo, y_hi, (n_poses,), device=device)
    rec_yaw = math_utils.sample_uniform(yaw_lo, yaw_hi, (n_poses,), device=device)
    rec_pos_w = torch.stack([rec_x, rec_y, torch.full((n_poses,), args_cli.fixture_z_m, device=device)], dim=-1)
    rec_quat_w = math_utils.quat_from_euler_xyz(torch.zeros(n_poses, device=device), torch.zeros(n_poses, device=device), rec_yaw)

    # leg world pose per pose: combine_frame_transforms(fixture_world, partial_rel) -- events.py's
    # own idiom (reset_insertive_object_from_partial_assembly_dataset, events.py:1598-1600).
    leg_pos_w, leg_quat_w = math_utils.combine_frame_transforms(rec_pos_w, rec_quat_w, partial_rel_pos, partial_rel_quat)

    # ---- SELF-CHECK: composition round-trip on one synthetic pose, asserted before anything
    # else. Structurally the same guard gen_composed_c4_reset_bank.py runs against a REAL FK'd
    # state; here there is no recorded palm pose to round-trip against yet (this script generates
    # the arm config, it does not look one up), so instead this checks that combine then subtract
    # reconstructs the SAME leg pose from the SAME fixture pose + relative pose -- i.e. that
    # isaaclab's combine/subtract pair is still behaviourally inverse on this install, which every
    # downstream target-pose computation below assumes. ----
    chk_pos, chk_quat = math_utils.subtract_frame_transforms(
        rec_pos_w[:1], rec_quat_w[:1], leg_pos_w[:1], leg_quat_w[:1]
    )
    self_check_pos_err_mm = (chk_pos[0] - partial_rel_pos[0]).norm().item() * 1000.0
    self_check_rot_err_deg = math_utils.quat_error_magnitude(chk_quat, partial_rel_quat[:1])[0].item() * 180.0 / np.pi
    print(
        f"[gen_ik] SELF-CHECK round-trip residual: {self_check_pos_err_mm:.6f} mm, {self_check_rot_err_deg:.6f} deg",
        flush=True,
    )
    assert self_check_pos_err_mm < 0.01 and self_check_rot_err_deg < 0.01, (
        "SELF-CHECK FAILED: combine_frame_transforms/subtract_frame_transforms are not exact"
        " inverses on this install -- STOPPING, every target pose below would be built on it."
    )
    print("[gen_ik] SELF-CHECK PASSED. Proceeding.", flush=True)

    # ---- per-pose shuffled grasp-candidate order (bounded to --grasps-per-pose; different poses
    # get DIFFERENT candidate orders so a systematically-bad grasp does not fail every pose in the
    # same round together). ----
    n_try = min(args_cli.grasps_per_pose, n_grasps_total)
    candidate_order = torch.stack(
        [torch.randperm(n_grasps_total, device=device)[:n_try] for _ in range(n_poses)], dim=0
    )  # [n_poses, n_try]

    resolved = torch.zeros((n_poses,), dtype=torch.bool, device=device)
    kept_grasp_idx = torch.full((n_poses,), -1, dtype=torch.long, device=device)
    kept_joint_position = torch.zeros((n_poses, n_joints), device=device)
    kept_joint_position_target = torch.zeros((n_poses, n_joints), device=device)
    kept_pos_err_mm = torch.full((n_poses,), float("nan"), device=device)
    kept_rot_err_deg = torch.full((n_poses,), float("nan"), device=device)
    n_attempts_used = torch.zeros((n_poses,), dtype=torch.long, device=device)

    total_rounds = args_cli.n_seeds * n_try
    for round_idx in range(total_rounds):
        remaining = (~resolved).nonzero(as_tuple=True)[0]
        if remaining.numel() == 0:
            print("[gen_ik] every pose resolved -- stopping early.", flush=True)
            break
        if args_cli.target_n is not None and int(resolved.sum().item()) >= args_cli.target_n:
            print(f"[gen_ik] reached --target-n={args_cli.target_n} accepted poses -- stopping early.", flush=True)
            break

        seed_idx = round_idx // n_try
        grasp_col = round_idx % n_try
        grasp_idx_for_remaining = candidate_order[remaining, grasp_col]  # absolute grasp indices

        # target palm pose for this round's (pose, grasp) pairs
        leg_pos_r, leg_quat_r = leg_pos_w[remaining], leg_quat_w[remaining]
        grasp_pos_r, grasp_quat_r = grasp_rel_pos[grasp_idx_for_remaining], grasp_rel_quat[grasp_idx_for_remaining]
        target_pos_w, target_quat_w = math_utils.combine_frame_transforms(leg_pos_r, leg_quat_r, grasp_pos_r, grasp_quat_r)

        # seed joint vector: home for seed_idx==0, home with shoulder_pan flipped +180deg*seed_idx
        # otherwise (see --n-seeds help). Hand joints stay at the DEFAULT (open) posture throughout
        # IK -- they do not affect the palm's kinematics (the palm/rl_dg_mount is proximal to the
        # finger joints), and are overwritten with the grasp's own closed posture only for ACCEPTED
        # instances, below -- matching reset_end_effector_from_grasp_dataset's own two-step order
        # (IK first with the gripper untouched, gripper posture set afterward, separately).
        seed = default_joint_pos.unsqueeze(0).expand(remaining.numel(), -1).clone()
        if seed_idx > 0:
            shoulder_pan_idx = [i for i, nm in enumerate(arm_joint_names_resolved) if "shoulder_pan" in nm]
            if shoulder_pan_idx:
                seed[:, name_to_idx[arm_joint_names_resolved[shoulder_pan_idx[0]]]] += math.pi * seed_idx

        result = _ik_attempt_batch(
            robot_cfg=robot_cfg,
            seed_joint_pos=seed,
            target_pos_w=target_pos_w,
            target_quat_w=target_quat_w,
            arm_joint_names=arm_joint_names_resolved,
            hand_joint_names=hand_joint_names,
            sim=sim,
            device=device,
            ik_iterations=args_cli.ik_iterations,
            ik_step_size=args_cli.ik_step_size,
            joint_limit_margin_rad=math.radians(args_cli.joint_limit_margin_deg),
            chunk_size=args_cli.chunk_size,
            chunk_prefix=f"Ik_{round_idx}",
        )

        accept = (
            (result["pos_err_mm"] <= args_cli.pos_accept_mm)
            & (result["rot_err_deg"] <= args_cli.rot_accept_deg)
            & (~result["pinned"])
        )
        n_attempts_used[remaining] += 1
        newly_kept = remaining[accept]
        if newly_kept.numel() > 0:
            resolved[newly_kept] = True
            kept_grasp_idx[newly_kept] = grasp_idx_for_remaining[accept]
            kept_pos_err_mm[newly_kept] = result["pos_err_mm"][accept]
            kept_rot_err_deg[newly_kept] = result["rot_err_deg"][accept]
            jp = result["joint_position"][accept].clone()
            jpt = jp.clone()
            jpt[:, result["arm_ids"]] = result["joint_position_target_arm"][accept]
            kept_joint_position[newly_kept] = jp
            kept_joint_position_target[newly_kept] = jpt

        print(
            f"[gen_ik] round {round_idx + 1}/{total_rounds} (seed {seed_idx}, grasp-col {grasp_col}):"
            f" {int(accept.sum().item())}/{remaining.numel()} newly accepted;"
            f" {int(resolved.sum().item())}/{n_poses} total resolved",
            flush=True,
        )

    n_kept = int(resolved.sum().item())
    print(
        f"[gen_ik] {n_kept}/{n_poses} poses got an ACCEPTED IK match within the"
        f" grasps-per-pose={n_try} x n-seeds={args_cli.n_seeds} budget", flush=True,
    )
    if n_kept == 0:
        raise RuntimeError("gen_ik_c4_reset_bank: zero poses accepted -- nothing to write.")

    kept_idx = resolved.nonzero(as_tuple=True)[0]

    # ---- overwrite hand indices with the ACCEPTED grasp's own recorded closed posture (defect
    # class this project has already paid for once: hand posture must be the MATCHED grasp's
    # posture, not whatever the seed happened to carry) ----
    hand_posture = gjp_full[kept_grasp_idx[kept_idx]]  # [n_kept, n_hand]
    joint_position = kept_joint_position[kept_idx].clone()
    joint_position[:, hand_indices] = hand_posture
    joint_position_target = kept_joint_position_target[kept_idx].clone()
    joint_position_target[:, hand_indices] = hand_posture

    n_hand = len(hand_indices)
    joint_velocity = torch.zeros((n_kept, n_joints), device=device)
    joint_velocity_target = torch.zeros((n_kept, n_joints), device=device)

    robot_root_pose = torch.zeros((n_kept, 7), device=device)
    robot_root_pose[:, 3] = 1.0
    robot_root_velocity = torch.zeros((n_kept, 6), device=device)

    insertive_root_pose = torch.cat([leg_pos_w[kept_idx], leg_quat_w[kept_idx]], dim=-1)
    insertive_root_velocity = torch.zeros((n_kept, 6), device=device)
    receptive_root_pose = torch.cat([rec_pos_w[kept_idx], rec_quat_w[kept_idx]], dim=-1)
    receptive_root_velocity = torch.zeros((n_kept, 6), device=device)

    initial_state = {
        "articulation": {
            "robot": {
                "root_pose": list(robot_root_pose.cpu()),
                "root_velocity": list(robot_root_velocity.cpu()),
                "joint_position": list(joint_position.cpu()),
                "joint_velocity": list(joint_velocity.cpu()),
                "joint_position_target": list(joint_position_target.cpu()),
                "joint_velocity_target": list(joint_velocity_target.cpu()),
            }
        },
        # EXACTLY TWO KEYS (MultiResetManager's assumed_static_assets covers table/ur5_metal_support
        # -- see _assert_reset_file_covers_scene, omnireset/mdp/events.py).
        "rigid_object": {
            "insertive_object": {"root_pose": list(insertive_root_pose.cpu()), "root_velocity": list(insertive_root_velocity.cpu())},
            "receptive_object": {"root_pose": list(receptive_root_pose.cpu()), "root_velocity": list(receptive_root_velocity.cpu())},
        },
    }
    bank = {"initial_state": initial_state}

    _atomic_torch_save(bank, args_cli.out)
    out_sha = _sha256(args_cli.out)
    print(f"[gen_ik] wrote {args_cli.out}", flush=True)
    print(f"[gen_ik] n={n_kept}, sha256={out_sha}, size={os.path.getsize(args_cli.out)} bytes", flush=True)

    # ================= INDEPENDENT VALIDATION (re-load from disk; never trust in-memory tensors) =================
    reloaded = torch.load(args_cli.out, map_location="cpu", weights_only=False)
    ins_np = torch.stack(reloaded["initial_state"]["rigid_object"]["insertive_object"]["root_pose"]).numpy()
    rec_np = torch.stack(reloaded["initial_state"]["rigid_object"]["receptive_object"]["root_pose"]).numpy()
    depth_mm, lateral_mm, tilt_deg, pos_err_mm, rot_err_deg = _reproject_independent(ins_np, rec_np)

    pos_ok = pos_err_mm < 2.5
    rot_ok = rot_err_deg < math.degrees(0.025)
    both_ok = pos_ok & rot_ok

    # SECOND, isaaclab-math cross-check (see that function's docstring for the compute_pose_error
    # rot_error_type="quat" gotcha this guards against) -- must AGREE with the scipy path above
    # within float tolerance, or something is wrong with one of the two implementations.
    ins_t = torch.as_tensor(ins_np, dtype=torch.float32)
    rec_t = torch.as_tensor(rec_np, dtype=torch.float32)
    pos_err_m_torch, rot_err_rad_torch = _true_predicate_torch_cross_check(ins_t, rec_t)
    cross_check_pos_diff_mm = (pos_err_m_torch.numpy() * 1000.0 - pos_err_mm).__abs__().max()
    cross_check_rot_diff_deg = (np.degrees(rot_err_rad_torch.numpy()) - rot_err_deg).__abs__().max()
    print(
        f"[gen_ik] cross-check (scipy vs isaaclab compute_pose_error) max abs diff:"
        f" {cross_check_pos_diff_mm:.6f} mm, {cross_check_rot_diff_deg:.6f} deg (should be ~0)",
        flush=True,
    )

    def pct(a, ps=(0, 10, 25, 50, 75, 90, 100)):
        return {p: float(np.percentile(a, p)) for p in ps}

    # geom_gate_fraction: what fraction of the WRITTEN poses' leg placement alone (independent of
    # grasp/IK -- see module docstring's "NOTE ON WHAT THE WRITTEN YIELD ACTUALLY MEASURES")
    # already clears the true predicate. written_fraction_of_geom_passing separates IK/grasp yield
    # from pose-selection yield -- multiply the two to get the number this whole exercise exists
    # to produce (the predicted untrained task_3 baseline).
    geom_gate_fraction = float(both_ok.mean())
    kept_attempts = n_attempts_used[kept_idx].float().mean().item()
    all_attempts = n_attempts_used.float().mean().item()

    print("\n=== RE-LOADED, INDEPENDENTLY REPROJECTED (scipy Rotation, not this script's own combine_frame_transforms) ===", flush=True)
    print("depth_into_bore_mm:", pct(depth_mm), flush=True)
    print("lateral_mm:", pct(lateral_mm), flush=True)
    print("tilt_deg:", pct(tilt_deg), flush=True)
    print(f"fraction pos_ok (<2.5mm): {float(pos_ok.mean()):.4f}", flush=True)
    print(f"fraction rot_ok (<1.4324deg): {float(rot_ok.mean()):.4f}", flush=True)
    print(f"fraction BOTH -- TRUE task_3 predicate over WRITTEN states: {geom_gate_fraction:.4f}", flush=True)
    print(f"IK match-quality (accepted only) pos_err_mm: {pct(kept_pos_err_mm[kept_idx].cpu().numpy())}", flush=True)
    print(f"IK match-quality (accepted only) rot_err_deg: {pct(kept_rot_err_deg[kept_idx].cpu().numpy())}", flush=True)
    print(f"mean grasp-candidates tried, accepted poses: {kept_attempts:.2f} / all poses: {all_attempts:.2f}", flush=True)

    summary_path = os.path.splitext(args_cli.out)[0] + "_validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "n_kept": n_kept, "n_poses_input": n_poses, "sha256": out_sha,
            "match_pos_err_mm_at_accept": kept_pos_err_mm[kept_idx].cpu().tolist(),
            "match_rot_err_deg_at_accept": kept_rot_err_deg[kept_idx].cpu().tolist(),
            "depth_into_bore_mm_pct": pct(depth_mm), "lateral_mm_pct": pct(lateral_mm), "tilt_deg_pct": pct(tilt_deg),
            "pos_ok_fraction": float(pos_ok.mean()), "rot_ok_fraction": float(rot_ok.mean()),
            "true_predicate_fraction_written": geom_gate_fraction,
            "cross_check_max_abs_diff_mm": float(cross_check_pos_diff_mm),
            "cross_check_max_abs_diff_deg": float(cross_check_rot_diff_deg),
            "mean_attempts_accepted_poses": kept_attempts, "mean_attempts_all_poses": all_attempts,
        }, f, indent=2, default=float)
    print(f"[gen_ik] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
