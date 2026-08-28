# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FK-matching sweep for the COMPOSED-STATE C4 (Near Goal / ObjectPartiallyAssembledEEGrasped) design.

CONTEXT (do not re-derive, this is here so the script is self-contained). C4 states cannot currently
be produced by either existing route: the RECORDER route (reset_end_effector_from_grasp_dataset,
omnireset/mdp/events.py:1258) IK-solves an ARBITRARY Cartesian palm target and measurably misses it by
a median 19.72 mm / 11.12 deg on this plant; the POLICY route reproduces the grasp perfectly but has
no reward term that holds the leg seated once placed. This script tests a THIRD route: instead of
choosing a leg pose and solving IK for the arm, take an arm configuration ALREADY STORED in a
recorded C3 (ObjectAnywhereEEGrasped, "Stable Grasp") state -- which is forward-kinematically REAL,
never IK'd against an arbitrary target -- and ask whether it ALREADY lands the palm close to what some
partial-assembly leg pose + some recorded grasp demands. If so, that triple needs no IK at all, or at
most a short local correction, not a fresh 25-iteration damped solve from a cold start.

THIS SCRIPT MEASURES REACHABILITY ONLY. It does not write, settle, or validate a single composed
state -- it is the cheap go/no-go gate that decides whether a generation campaign is worth running.
It does not need the scene, the object props, or any manager -- see design note below.

WHY "ARTICULATION ONLY, NO SCENE, NO MANAGERS": going through gym.make(task) (the house convention,
see diagnose_ik_convergence.py) would construct the full ManagerBasedRLEnv, including
insertive_object/receptive_object with the DEFAULT variant (Peg/PegHole, both UWLAB_CLOUD_ASSETS_DIR)
unless a Hydra override is threaded through -- exactly the cloud-asset-fetch trap this project has
hit before. This script needs none of that: it is a pure forward-kinematics query on the robot alone,
so it constructs a bare isaaclab.assets.Articulation directly against a SimulationContext, mirroring
IsaacLab's own scripts/tutorials/01_assets/run_articulation.py, never a ManagerBasedEnv. Batched over
N env-clone origins in one process, one write, one sim.forward() (kinematic refresh, no physics step,
no dynamics) -- not a Python loop per state.

THE SELF-CHECK, RUN BEFORE THE SWEEP AND ASSERTED, PER "the arguments swapped silently compose the
inverse": this script does not have access to WHICH grasps.pt index produced any given recorded C3
state (that capture requires spying on the real reset event during a full env.reset(), which is the
expensive path this script exists to avoid). Instead it derives the relative grasp EMPIRICALLY from
one C3 state's own (recorded leg pose, FK'd palm pose) via subtract_frame_transforms(leg, palm), then
re-composes it via combine_frame_transforms(leg, relative) and asserts the result reconstructs the
FK'd palm pose to a tight tolerance. This exercises the EXACT same call (same argument order) the main
sweep uses; a frame swap in the real code would fail this round-trip, because combine/subtract are
asymmetric under argument order even though the round-trip is a tautology in exact math. If this
assertion fails, STOP -- every number after it is meaningless. See main()'s first block.

DATA CONTRACTS, read from source, not memory (cite is the file this script itself was written from):
  grasps.pt        -- omnireset/mdp/events.py:1330-1395 (_load_and_precompute_grasps). Top-level dict,
                       optionally nested under "grasp_relative_pose"; keys "relative_position",
                       "relative_orientation" (gripper-in-object, i.e. T_gripper_in_leg, WXYZ quat --
                       events.py:1420-1423's own comment: "T_gripper_world = T_object_world *
                       T_relative"), and "gripper_joint_positions" (dict keyed by ROBOT JOINT NAME).
  partial_assemblies.pt -- omnireset/mdp/events.py:1653-1666 (pose_logging_event, the writer) and
                       :1555-1568 (the reader). Keys "relative_position"/"relative_orientation" =
                       insertive-in-receptive frame (subtract_frame_transforms(receptive, insertive)).
  C3/C2 reset banks -- inventoried directly with torch.load in this campaign: top key "initial_state",
                       then "rigid_object"."insertive_object"."root_pose" (list of [7] pos3+quatWXYZ4
                       tensors) and "articulation"."robot"."joint_position" (list of [26] tensors,
                       6 arm + 20 DELTO hand, in robot.joint_names order).

MANDATORY, CARRIED FORWARD FOR WHOEVER WRITES THE GENERATION SCRIPT THIS SWEEP GATES (this sweep
itself never writes a state, so it does not need this -- but the next script does and must not skip
it): after writing ANY composed state's joint positions, you MUST reconcile the PD position AND
velocity TARGETS for BOTH the arm and the gripper joint groups (robot.set_joint_position_target +
set_joint_velocity_target, per group) and then call env.scene.write_data_to_sim(), mirroring
events.py:1500-1517 exactly. Skipping this reproduces an ALREADY-DIAGNOSED, ALREADY-FIXED bug in this
exact codebase: a stale PD target left at "open" drives the actuator on the very first physics step,
measured as a median ~11 rad summed gripper-joint gap, contact-correlated with a velocity explosion
(RestingEEGrasped incident, 2026-08-17).

Run (one Isaac process; never via uwlab.sh; articulation-only boot, expect well under the ~15-20 min
full-env boot cost budgeted for training rungs):

    PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl" \\
        timeout -s KILL 600 <python> -u scripts_v2/tools/fk_match_composed_c4.py \\
        --grasps-path <Datasets_ur5e_delto or Datasets_render>/OmniReset/Grasps/<leg-object-name>/grasps.pt \\
        --partial-assembly-path <...>/OmniReset/Resets/OneLegInsertionFixture__SquareTableLeg200mmDecomp/partial_assemblies.pt \\
        --c3-bank-path <...>/resets_ObjectAnywhereEEGrasped.pt \\
        --c2-bank-path <...>/resets_ObjectRestingEEGrasped.pt \\
        --out-dir /tmp/c4_fk_sweep

Exact paths were NOT hardcoded: the "2048 validated partial-assembly poses" and the specific
grasps.pt this campaign means were generated/validated after this script was written and I could not
locate them in my last inventory pass -- pass the real paths at the command line. The script fails
loudly (FileNotFoundError) rather than silently falling back to a stale bank if a path is wrong.
"""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FK-matching sweep: composed-state C4 reachability go/no-go.")
parser.add_argument("--grasps-path", type=str, required=True, help="Path to Grasps/<leg>/grasps.pt")
parser.add_argument(
    "--partial-assembly-path", type=str, required=True, help="Path to Resets/<pair>/partial_assemblies.pt"
)
parser.add_argument(
    "--c3-bank-path", type=str, required=True,
    help="Path to resets_ObjectAnywhereEEGrasped.pt (C3 Stable Grasp -- primary arm-config source, per"
    " team-lead direction: 'our best').",
)
parser.add_argument(
    "--c2-bank-path", type=str, default=None,
    help="Optional path to resets_ObjectRestingEEGrasped.pt (C2 Near Object -- secondary arm-config"
    " source; more candidates is better here). Report separates which bank supplies matches.",
)
parser.add_argument(
    "--max-arm-configs", type=int, default=None,
    help="Cap on total FK'd arm configs (C3 + C2 combined), for VRAM/time safety on a shared box."
    " None = use every state in both banks.",
)
parser.add_argument(
    "--max-grasps", type=int, default=300,
    help="Subsample of grasps.pt entries used to build targets. Full grasps.pt x full partial-assembly"
    " set is 2048 x (grasps.pt count), which is excessive for a go/no-go: grasp choice perturbs the"
    " palm target LOCALLY around the leg, pose choice moves it MACROSCOPICALLY -- a few hundred grasps"
    " already samples that local perturbation; raise this only after a first pass says it matters.",
)
parser.add_argument("--max-poses", type=int, default=None, help="Cap on partial-assembly poses used. None = all.")
parser.add_argument(
    "--pos-thresholds-mm", type=str, default="5,10,20,50",
    help="Comma-separated position thresholds (mm) for the go/no-go table.",
)
parser.add_argument(
    "--rot-thresholds-deg", type=str, default="5,15,30",
    help="Comma-separated orientation thresholds (deg) for the go/no-go table (reported as a"
    " thresholds x thresholds matrix, not one fixed pairing, since the right band is not yet known).",
)
parser.add_argument(
    "--receptive-world-pos", type=str, default="0,0,0",
    help="Assumed WORLD pose of receptive_object for this sweep, comma-separated xyz (m). "
    "partial_assemblies.pt only stores insertive-in-receptive RELATIVE poses; the real training env"
    " jitters receptive_object's world pose every episode (x in [0.35,0.60], y in [-0.2,0.2], yaw in"
    " [-15,15] deg -- reset_states_cfg.py:195-220). This sweep uses ONE representative placement as an"
    " approximation, documented here rather than silently assumed; sweeping the full jitter range is"
    " out of scope for a cheap go/no-go.",
)
parser.add_argument("--receptive-world-quat-wxyz", type=str, default="1,0,0,0", help="Assumed receptive_object world quat, WXYZ.")
parser.add_argument(
    "--chunk-size", type=int, default=512,
    help="Robot clones spawned per FK batch. Each clone is ~35 non-instanced bodies with full"
    " collision geometry -- this is the ONE mechanism in this script with no in-repo precedent (every"
    " existing tool goes through the full gym.make(task) path; this does not) and the one most likely"
    " to need tuning on first run. START SMALL: run once with --max-arm-configs 512 (a single chunk at"
    " the default size) as a sanity pass before scaling to the full C3(+C2) bank size, mirroring the"
    " VRAM-ladder protocol's own sanity-rung-before-large-rung logic. If chunk construction is slow or"
    " VRAM-heavy at 512, lower this before raising --max-arm-configs.",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out-dir", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows -- Isaac Sim modules are only importable after AppLauncher starts."""

import torch  # noqa: E402
import numpy as np  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

# The SAME robot object every OmniReset UR5eDelto config constructs from (ur5e_delto_cfg.py:94 etc.,
# via IMPLICIT_UR5E_DELTO). If the sdf-to-hullfix3 plant swap has landed by the time this runs, this
# import already resolves to the post-change USD/actuators -- nothing in this script needs to know
# which one it is, only the arm+hand JOINT TREE matters for FK, and the swap does not touch that
# (only colliders and actuator gains differ between the two; the skeleton is byte-identical).
from uwlab_assets.robots.ur5e_delto import IMPLICIT_UR5E_DELTO  # noqa: E402


def _load_grasps(path: str, robot_joint_names: list[str], device: str):
    """Mirrors omnireset/mdp/events.py:1330-1395 (_load_and_precompute_grasps) exactly."""
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
    print(f"[fk_match] grasps.pt: {num_grasps} grasps, gripper joints recorded: {sorted(recorded)}", flush=True)
    # Sanity only -- this sweep never writes gripper posture, so it does not need an exact
    # name-to-index map (unlike events.py:1359-1393's loader). It DOES need to catch "this
    # grasps.pt was recorded for the wrong gripper" early: if there is zero overlap with the
    # robot's expected joint-name vocabulary, something upstream pointed at the wrong dataset.
    overlap = recorded & set(robot_joint_names)
    if not overlap:
        print(
            f"[fk_match] WARNING: zero overlap between grasps.pt's recorded joints and the robot's"
            f" expected joints ({sorted(robot_joint_names)[:5]}...). This grasps.pt may have been"
            f" recorded for a different gripper -- relative_position/relative_orientation (what this"
            f" sweep actually uses) are gripper-geometry-independent, so the sweep can proceed, but"
            f" treat its results as suspect until this is confirmed intentional.",
            flush=True,
        )
    return rel_pos, rel_quat, gripper_joint_positions_dict


def _load_partial_assemblies(path: str, device: str):
    """Mirrors omnireset/mdp/events.py:1550-1568 (reset_insertive_object_from_partial_assembly_dataset)."""
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
    print(f"[fk_match] partial_assemblies.pt: {rel_pos.shape[0]} poses", flush=True)
    return rel_pos.to(device, dtype=torch.float32), rel_quat.to(device, dtype=torch.float32)


def _load_c_bank(path: str, device: str, label: str):
    """Reads a recorded *EEGrasped reset bank: leg pose + full robot joint_position per state."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} bank not found at {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    ro = data["initial_state"]["rigid_object"]
    if "insertive_object" not in ro:
        raise ValueError(f"{label} bank {path} has no insertive_object key -- not a 4-key/rekeyed bank")
    leg_root_pose = torch.stack(ro["insertive_object"]["root_pose"]).to(device, dtype=torch.float32)  # [N,7]
    joint_pos = torch.stack(data["initial_state"]["articulation"]["robot"]["joint_position"]).to(
        device, dtype=torch.float32
    )  # [N, n_joints]
    print(f"[fk_match] {label} bank: {joint_pos.shape[0]} states, {joint_pos.shape[1]} joints", flush=True)
    return leg_root_pose, joint_pos


def _fk_batch(robot_cfg, joint_positions: torch.Tensor, sim: SimulationContext, device: str, chunk_size: int = 2048):
    """Forward-kinematics a batch of stored joint configs to the rl_dg_mount (palm) world pose.

    Spawns `chunk_size` robot clones at a time (VRAM safety on a shared/rented box, not a Python loop
    over states -- each chunk is one batched write + one batched read), writes the FULL stored joint
    vector (arm + hand) via write_joint_state_to_sim, refreshes kinematics with sim.forward() (a
    kinematic-only fabric update -- no dynamics step, no time advance, same call
    diagnose_ik_convergence.py:172 uses for the identical purpose), and reads body_pos_w/body_quat_w
    at the palm body. Root pose is pinned at world origin, identity -- the SAME convention
    ur5e_delto.py:84-86 authors as this robot's init_state, and the same convention the real training
    scene places the robot+table assembly at (rl_state_cfg.py's table/robot origin comments).
    """
    n_total = joint_positions.shape[0]
    palm_pos_all = torch.zeros((n_total, 3), device=device)
    palm_quat_all = torch.zeros((n_total, 4), device=device)

    start = 0
    chunk_idx = 0
    while start < n_total:
        end = min(start + chunk_size, n_total)
        n = end - start
        chunk_idx += 1

        for i in range(n):
            sim_utils.create_prim(f"/World/FkOrigin_{chunk_idx}_{i}", "Xform", translation=(3.0 * i, 0.0, 0.0))
        cfg = robot_cfg.replace(prim_path=f"/World/FkOrigin_{chunk_idx}_.*/Robot")
        robot = Articulation(cfg=cfg)
        sim.reset()

        palm_ids, palm_names = robot.find_bodies("rl_dg_mount")
        if len(palm_ids) != 1:
            raise RuntimeError(f"Expected exactly one rl_dg_mount body, found {palm_names}")
        palm_id = palm_ids[0]

        root_pos = torch.zeros((n, 3), device=device)
        root_quat = torch.zeros((n, 4), device=device)
        root_quat[:, 0] = 1.0
        robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1))
        robot.write_root_velocity_to_sim(torch.zeros((n, 6), device=device))
        robot.write_joint_state_to_sim(
            position=joint_positions[start:end], velocity=torch.zeros((n, joint_positions.shape[1]), device=device)
        )
        robot.write_data_to_sim()
        sim.forward()  # kinematic refresh only -- no physics step, no time advance
        robot.update(0.0)

        palm_pos_all[start:end] = robot.data.body_pos_w[:, palm_id, :].clone()
        palm_quat_all[start:end] = robot.data.body_quat_w[:, palm_id, :].clone()

        # Free this chunk's prims before the next chunk to keep stage size bounded. One batched
        # DeletePrimsCommand (isaaclab.sim.utils.delete_prim accepts a path list) rather than a
        # per-prim raw stage.RemovePrim loop -- this is the library's own sanctioned deletion path
        # (isaaclab/sim/utils/prims.py:189), which goes through omni.kit.commands and the USD stage
        # cache rather than mutating the stage directly, so it is the safer choice for a prim that a
        # live PhysX articulation view is currently bound to.
        sim_utils.delete_prim([f"/World/FkOrigin_{chunk_idx}_{i}" for i in range(n)])
        del robot

        print(f"[fk_match] FK chunk {chunk_idx}: {start}:{end} of {n_total} done", flush=True)
        start = end

    return palm_pos_all, palm_quat_all


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    torch.manual_seed(args_cli.seed)
    os.makedirs(args_cli.out_dir, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(device=device)
    sim = SimulationContext(sim_cfg)

    robot_cfg = IMPLICIT_UR5E_DELTO.copy()
    joint_names = list(robot_cfg.init_state.joint_pos.keys())  # not the resolved order -- resolved after spawn

    # ---- load everything (CPU-cheap, no Isaac needed for this part) ----
    partial_rel_pos, partial_rel_quat = _load_partial_assemblies(args_cli.partial_assembly_path, device)
    if args_cli.max_poses is not None:
        partial_rel_pos, partial_rel_quat = partial_rel_pos[: args_cli.max_poses], partial_rel_quat[: args_cli.max_poses]
    n_poses = partial_rel_pos.shape[0]

    c3_leg_pose, c3_joint_pos = _load_c_bank(args_cli.c3_bank_path, device, "C3 (ObjectAnywhereEEGrasped)")
    sources = ["C3"] * c3_leg_pose.shape[0]
    all_leg_pose, all_joint_pos = c3_leg_pose, c3_joint_pos
    if args_cli.c2_bank_path is not None:
        c2_leg_pose, c2_joint_pos = _load_c_bank(args_cli.c2_bank_path, device, "C2 (ObjectRestingEEGrasped)")
        if c2_joint_pos.shape[1] != c3_joint_pos.shape[1]:
            raise ValueError("C2/C3 joint dimensionality mismatch -- banks are not from the same robot")
        sources += ["C2"] * c2_leg_pose.shape[0]
        all_leg_pose = torch.cat([all_leg_pose, c2_leg_pose], dim=0)
        all_joint_pos = torch.cat([all_joint_pos, c2_joint_pos], dim=0)

    if args_cli.max_arm_configs is not None and all_joint_pos.shape[0] > args_cli.max_arm_configs:
        idx = torch.randperm(all_joint_pos.shape[0])[: args_cli.max_arm_configs]
        all_leg_pose, all_joint_pos = all_leg_pose[idx], all_joint_pos[idx]
        sources = [sources[i] for i in idx.tolist()]

    n_arm = all_joint_pos.shape[0]
    print(f"[fk_match] total arm configs to FK: {n_arm} (from {set(sources)})", flush=True)

    # ---- FK all arm configs, batched ----
    fk_palm_pos, fk_palm_quat = _fk_batch(robot_cfg, all_joint_pos, sim, device, chunk_size=args_cli.chunk_size)

    # ---- SELF-CHECK: composition round-trip on ONE real arm-config state, asserted before anything
    # else ----
    # Derive the empirical relative grasp from state 0's own (recorded leg pose, FK'd palm pose),
    # THEN re-derive the palm pose from it via the SAME combine_frame_transforms call the main sweep
    # uses. A frame-order bug in that call fails this even though the math is a tautology in the
    # abstract -- see module docstring.
    #
    # MUST use all_leg_pose[0] here, NOT c3_leg_pose[0]: when --max-arm-configs subsamples, index 0 of
    # the (now shuffled) all_joint_pos/fk_palm_pos is NOT necessarily the C3 bank's original first
    # state -- all_leg_pose was permuted in lockstep with all_joint_pos at the subsampling step above,
    # so all_leg_pose[0] is the leg pose that actually PRODUCED fk_palm_pos[0], whichever state that
    # ended up being. Indexing the original c3_leg_pose[0] instead would silently compare two
    # unrelated states whenever subsampling is active, and the self-check would report a large,
    # confusing residual that has nothing to do with a frame-order bug.
    s0_leg_pos, s0_leg_quat = all_leg_pose[0, :3], all_leg_pose[0, 3:7]
    s0_palm_pos, s0_palm_quat = fk_palm_pos[0], fk_palm_quat[0]
    empirical_rel_pos, empirical_rel_quat = math_utils.subtract_frame_transforms(
        s0_leg_pos.unsqueeze(0), s0_leg_quat.unsqueeze(0), s0_palm_pos.unsqueeze(0), s0_palm_quat.unsqueeze(0)
    )
    recon_palm_pos, recon_palm_quat = math_utils.combine_frame_transforms(
        s0_leg_pos.unsqueeze(0), s0_leg_quat.unsqueeze(0), empirical_rel_pos, empirical_rel_quat
    )
    self_check_pos_err_mm = (recon_palm_pos[0] - s0_palm_pos).norm().item() * 1000.0
    self_check_rot_err_deg = (
        math_utils.quat_error_magnitude(recon_palm_quat, s0_palm_quat.unsqueeze(0))[0].item() * 180.0 / np.pi
    )
    print(
        f"[fk_match] SELF-CHECK round-trip residual: {self_check_pos_err_mm:.6f} mm,"
        f" {self_check_rot_err_deg:.6f} deg",
        flush=True,
    )
    assert self_check_pos_err_mm < 0.01 and self_check_rot_err_deg < 0.01, (
        "SELF-CHECK FAILED: composition round-trip did not reconstruct the FK'd palm pose. The"
        " composition is inverted (frame-order swap in combine_frame_transforms/subtract_frame_transforms"
        " arguments) and every number below is meaningless. STOPPING."
    )
    print("[fk_match] SELF-CHECK PASSED -- composition math verified against real FK. Proceeding.", flush=True)

    # ---- load grasps, build TARGET palm poses for (pose k, grasp g) pairs ----
    grasp_rel_pos, grasp_rel_quat, _ = _load_grasps(args_cli.grasps_path, joint_names, device)
    n_grasps_total = grasp_rel_pos.shape[0]
    if args_cli.max_grasps is not None and n_grasps_total > args_cli.max_grasps:
        g_idx = torch.randperm(n_grasps_total)[: args_cli.max_grasps]
        grasp_rel_pos, grasp_rel_quat = grasp_rel_pos[g_idx], grasp_rel_quat[g_idx]
    n_grasps = grasp_rel_pos.shape[0]
    print(f"[fk_match] using {n_grasps}/{n_grasps_total} grasps x {n_poses} poses = {n_grasps * n_poses} targets", flush=True)

    rec_pos_w = torch.tensor([float(x) for x in args_cli.receptive_world_pos.split(",")], device=device).unsqueeze(0)
    rec_quat_w = torch.tensor(
        [float(x) for x in args_cli.receptive_world_quat_wxyz.split(",")], device=device
    ).unsqueeze(0)

    # leg world pose for every pose k: combine_frame_transforms(receptive, partial_assembly_rel) --
    # events.py:1598-1600's exact idiom.
    leg_pos_w, leg_quat_w = math_utils.combine_frame_transforms(
        rec_pos_w.expand(n_poses, -1), rec_quat_w.expand(n_poses, -1), partial_rel_pos, partial_rel_quat
    )

    # target palm pose for every (k, g) pair -- broadcast leg pose over grasps.
    leg_pos_kg = leg_pos_w.unsqueeze(1).expand(-1, n_grasps, -1).reshape(-1, 3)  # [n_poses*n_grasps, 3]
    leg_quat_kg = leg_quat_w.unsqueeze(1).expand(-1, n_grasps, -1).reshape(-1, 4)
    grasp_pos_kg = grasp_rel_pos.unsqueeze(0).expand(n_poses, -1, -1).reshape(-1, 3)
    grasp_quat_kg = grasp_rel_quat.unsqueeze(0).expand(n_poses, -1, -1).reshape(-1, 4)
    target_pos, target_quat = math_utils.combine_frame_transforms(leg_pos_kg, leg_quat_kg, grasp_pos_kg, grasp_quat_kg)
    pose_index_kg = torch.arange(n_poses, device=device).unsqueeze(1).expand(-1, n_grasps).reshape(-1)

    # ---- nearest-neighbor match: each target against the FK'd arm-config palm poses. Position-only
    # KD-tree query (unchanged reasoning: mixing metres and radians into one unweighted Euclidean tree
    # metric would be a silent correctness bug -- position and rotation are kept as SEPARATE explicit
    # bounds throughout below, never folded into the tree itself).
    #
    # FIXED, was k=1: querying only the single position-nearest arm config meant every target's
    # reported orientation error came from THAT ONE candidate, even when a slightly farther-but-still-
    # close candidate (e.g. a different elbow-up/elbow-down IK branch landing at nearly the same palm
    # position) had a far better orientation match. That under-counted matches at every threshold cell
    # below -- an error that can only make the go/no-go verdict look worse than reality, never better.
    # Now each target keeps its k position-nearest candidates: the match-count matrix asks "does ANY
    # of the k clear both bounds" per cell (an explicit per-threshold existence check, not a single
    # fixed metric), and a separate normalized combined score (documented below, never fed into the
    # tree) picks ONE representative candidate per target only where a single number is needed -- the
    # residual-distribution percentiles. ----
    from scipy.spatial import cKDTree  # noqa: E402

    fk_pos_np = fk_palm_pos.cpu().numpy()
    tree = cKDTree(fk_pos_np)
    target_pos_np = target_pos.cpu().numpy()
    # k position-nearest candidates per target, guarded against fewer reference arm configs than k
    # (e.g. a tiny --max-arm-configs sanity run).
    _MATCH_KNN = 32
    k = min(_MATCH_KNN, fk_pos_np.shape[0])
    nn_dist_m, nn_idx = tree.query(target_pos_np, k=k)
    if k == 1:
        # cKDTree.query drops the trailing size-1 axis when k=1; restore it so every op below can
        # assume a [n_targets, k] shape unconditionally.
        nn_dist_m = nn_dist_m.reshape(-1, 1)
        nn_idx = nn_idx.reshape(-1, 1)
    nn_idx_t = torch.as_tensor(nn_idx, device=device, dtype=torch.long)  # [n_targets, k]
    nn_pos_err_mm = torch.as_tensor(nn_dist_m, device=device) * 1000.0  # [n_targets, k]
    n_targets = nn_pos_err_mm.shape[0]
    cand_quat = fk_palm_quat[nn_idx_t]  # [n_targets, k, 4]
    nn_rot_err_deg = (
        math_utils.quat_error_magnitude(
            target_quat.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4), cand_quat.reshape(-1, 4)
        ).reshape(n_targets, k)
        * 180.0
        / np.pi
    )  # [n_targets, k]

    # Combined score to pick ONE representative candidate per target, used ONLY for the residual-
    # distribution percentiles below -- never for the per-cell match matrix, which checks every
    # candidate against every cell's bounds separately (see loop below). Normalized by fixed mm/deg
    # scales rather than an unweighted position+quaternion Euclidean mix: this is a documented
    # post-hoc ranking over the k candidates the position-only tree already gathered, not a change to
    # the tree's own metric.
    _SCORE_POS_NORM_MM = 10.0
    _SCORE_ROT_NORM_DEG = 15.0
    combined_score = (nn_pos_err_mm / _SCORE_POS_NORM_MM) ** 2 + (nn_rot_err_deg / _SCORE_ROT_NORM_DEG) ** 2
    best_j = combined_score.argmin(dim=1)  # [n_targets]
    row_idx = torch.arange(n_targets, device=device)
    best_pos_err_mm = nn_pos_err_mm[row_idx, best_j]
    best_rot_err_deg = nn_rot_err_deg[row_idx, best_j]

    # ---- REPORT ----
    pos_thresholds = [float(x) for x in args_cli.pos_thresholds_mm.split(",")]
    rot_thresholds = [float(x) for x in args_cli.rot_thresholds_deg.split(",")]

    def pct(t):
        return [float(np.percentile(t, p)) for p in (0, 10, 25, 50, 75, 90, 100)]

    pos_np, rot_np = best_pos_err_mm.cpu().numpy(), best_rot_err_deg.cpu().numpy()
    summary = {
        "n_arm_configs_fk": n_arm,
        "n_poses": n_poses,
        "n_grasps": n_grasps,
        "n_targets": int(target_pos.shape[0]),
        "self_check_pos_err_mm": self_check_pos_err_mm,
        "self_check_rot_err_deg": self_check_rot_err_deg,
        "pos_err_mm_percentiles_[0,10,25,50,75,90,100]": pct(pos_np),
        "rot_err_deg_percentiles_[0,10,25,50,75,90,100]": pct(rot_np),
        "match_count_matrix": {},
        "matched_pose_coverage": {},
        "supplying_bank_at_thresholds": {},
    }

    print(
        "\n=== RESIDUAL DISTRIBUTION (best combined-score candidate among"
        f" {k} position-nearest arm configs per target) ===",
        flush=True,
    )
    print(f"position (mm):    {summary['pos_err_mm_percentiles_[0,10,25,50,75,90,100]']}", flush=True)
    print(f"orientation (deg): {summary['rot_err_deg_percentiles_[0,10,25,50,75,90,100]']}", flush=True)

    print("\n=== MATCH COUNT MATRIX (position threshold x orientation threshold) ===", flush=True)
    header = "pos<=mm \\ rot<=deg  " + "  ".join(f"{r:>8.1f}" for r in rot_thresholds)
    print(header, flush=True)
    for p in pos_thresholds:
        row_counts = []
        for r in rot_thresholds:
            # a target matches this cell if ANY of its k position-nearest candidates clears BOTH
            # bounds -- not just the single position-nearest one (see the fix note above).
            cell_hit = (nn_pos_err_mm <= p) & (nn_rot_err_deg <= r)  # [n_targets, k]
            mask = cell_hit.any(dim=1)
            n_match = int(mask.sum().item())
            row_counts.append(n_match)
            summary["match_count_matrix"][f"pos<={p}mm,rot<={r}deg"] = n_match
            # which bank supplied the matches at this cell, and how many DISTINCT poses got covered.
            # Per matched target, attribute to the source of its FIRST (position-nearest) qualifying
            # candidate -- cKDTree.query returns candidates sorted by increasing distance, so argmax
            # over the boolean row returns the index of the first True.
            if n_match > 0:
                matched_target_idx = mask.nonzero(as_tuple=True)[0]
                first_qualifying_j = cell_hit[matched_target_idx].to(torch.uint8).argmax(dim=1)
                qualifying_cand_idx = nn_idx_t[matched_target_idx, first_qualifying_j]
                srcs = [sources[i] for i in qualifying_cand_idx.cpu().tolist()]
                summary["supplying_bank_at_thresholds"][f"pos<={p}mm,rot<={r}deg"] = {
                    s: srcs.count(s) for s in set(srcs)
                }
        print(f"{p:>17.1f}  " + "  ".join(f"{c:>8d}" for c in row_counts), flush=True)

    # ---- pose coverage: at the MIDDLE threshold cell, which of the n_poses partial-assembly poses
    # actually got a match, and how are they distributed -- a healthy count concentrated on a narrow
    # slice of the 2048 is a degenerate bank even if the raw count looks fine. ----
    mid_p, mid_r = pos_thresholds[len(pos_thresholds) // 2], rot_thresholds[len(rot_thresholds) // 2]
    # same any-of-k-candidates rule as the match count matrix above, not just the single
    # position-nearest candidate.
    mid_mask = ((nn_pos_err_mm <= mid_p) & (nn_rot_err_deg <= mid_r)).any(dim=1)
    matched_pose_idx = pose_index_kg[mid_mask].cpu().numpy()
    n_distinct_poses_matched = len(set(matched_pose_idx.tolist()))
    summary["matched_pose_coverage"] = {
        "threshold_used": f"pos<={mid_p}mm,rot<={mid_r}deg",
        "n_distinct_poses_matched": n_distinct_poses_matched,
        "n_poses_total": n_poses,
        "fraction_of_poses_with_any_match": n_distinct_poses_matched / n_poses,
    }
    # cheap geometric proxies for coverage shape -- NOT a certified bore-axis definition (that would
    # need the same rigor as the DELIVER-2-class target-orientation derivation, not done here):
    # depth proxy = z of the insertive-in-receptive relative position; yaw proxy = yaw of its
    # relative orientation. Both computed only to characterize concentration, not to gate anything.
    depth_proxy = partial_rel_pos[:, 2].cpu().numpy()
    _, _, yaw_proxy = math_utils.euler_xyz_from_quat(partial_rel_quat)
    yaw_proxy = yaw_proxy.cpu().numpy()
    matched_set = set(matched_pose_idx.tolist())
    matched_mask_all_poses = np.array([i in matched_set for i in range(n_poses)])
    summary["matched_pose_coverage"]["depth_proxy_mm_matched_vs_all"] = {
        "matched_percentiles": pct(depth_proxy[matched_mask_all_poses] * 1000.0) if matched_mask_all_poses.any() else None,
        "all_percentiles": pct(depth_proxy * 1000.0),
    }
    summary["matched_pose_coverage"]["yaw_proxy_deg_matched_vs_all"] = {
        "matched_percentiles": pct(np.degrees(yaw_proxy[matched_mask_all_poses])) if matched_mask_all_poses.any() else None,
        "all_percentiles": pct(np.degrees(yaw_proxy)),
    }

    print(
        f"\n=== POSE COVERAGE at {summary['matched_pose_coverage']['threshold_used']} ===\n"
        f"{n_distinct_poses_matched}/{n_poses} distinct partial-assembly poses have >=1 matching arm"
        f" config ({100*n_distinct_poses_matched/n_poses:.1f}%).\n"
        f"depth proxy (mm), matched vs all poses: {summary['matched_pose_coverage']['depth_proxy_mm_matched_vs_all']}\n"
        f"yaw proxy (deg), matched vs all poses:  {summary['matched_pose_coverage']['yaw_proxy_deg_matched_vs_all']}\n"
        "If 'matched' concentrates on a narrow slice of either proxy relative to 'all', the resulting"
        " C4 bank would be degenerate even though the raw match count above may look healthy.",
        flush=True,
    )

    out_path = os.path.join(args_cli.out_dir, "fk_match_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[fk_match] wrote {out_path}", flush=True)

    print(
        "\n=== VERDICT (read, do not trust a single number) ===\n"
        "Dense enough to proceed: a nontrivial cluster of targets under the tightest threshold cell"
        " (5mm/5deg or 10mm/15deg), AND pose coverage above spread across depth/yaw rather than"
        " concentrated at one proxy value, AND both C2 and C3 (if both were passed) contributing"
        " matches rather than one bank alone.\n"
        "Clean no: near-zero matches even at the loosest cell (50mm/30deg), OR matches exist but"
        " collapse onto a narrow pose slice, OR matches vanish once C2 is excluded and only 3000 C3"
        " states remain -- either signals the composed-state route does not have enough raw material"
        " to be worth a generation campaign, and routes to reward shaping / staged task definition"
        " instead, per the design note.\n"
        "This script does not compute that verdict FOR you -- it is a judgment call over the"
        " distributions and coverage numbers above, not a threshold this script can pick on its own.",
        flush=True,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
