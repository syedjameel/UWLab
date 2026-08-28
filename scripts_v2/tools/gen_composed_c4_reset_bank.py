#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""THE MISSING SCRIPT: turns matched (leg pose, arm config, grasp) triples into an actual
StableState reset bank (resets_ObjectPartiallyAssembledEEGrasped.pt schema) for the C4 / task_3
"near goal" reset type, from a deep-band partial_assemblies.pt (e.g. partial_assemblies_deep_v2.pt)
that the normal policy-rollout route cannot produce (measured: the withdrawal policy always
converges to 1.6-3.8mm final depth regardless of spawn depth, slope -0.210, R2 0.083 -- capped, not
tunable by spawning deeper).

WHY THIS EXISTS SEPARATELY FROM fk_match_composed_c4.py: that script (already reviewed, never run)
answers "is there enough density/coverage in the C2/C3 arm-config libraries to be worth a
generation campaign" -- it computes match statistics and writes fk_match_summary.json, but by its
own docstring "does not write, settle, or validate a single composed state" and never persists the
actual per-pose winning (arm_config_idx, grasp_idx) pair, only aggregate percentiles. This script
redoes the same matching (deliberately re-implemented here, not imported -- see the note below) and
THEN, for every pose whose best match clears an accept threshold, actually builds and writes the
full scene state.

NOT IMPORTED FROM fk_match_composed_c4.py, DELIBERATELY: that module's top level (outside any
function) parses sys.argv with its OWN argparse and immediately constructs its OWN AppLauncher --
`import fk_match_composed_c4` from inside another already-launched Isaac process would silently
re-parse this script's unrelated CLI flags and attempt to boot a SECOND simulation app in the same
process, which is not a supported configuration. The loader/FK functions below
(_load_grasps/_load_partial_assemblies/_load_c_bank/_fk_batch) are therefore copied, not imported --
kept behaviourally identical (same docstrings, same tensor ops) so a future refactor that extracts
them into a shared importable module has a known-identical target to diff against.

DEFECTS THIS SCRIPT'S SCHEMA IS DESIGNED TO NOT REPRODUCE, each one this project has already paid
for once:
  1. joint_position_target / joint_velocity_target BOTH WRITTEN, ALWAYS, and the HAND portion is
     set to follow the newly-composed (closed, grasp-validated) hand joint_position, never left at
     whatever the source arm-config state's own hand target was and never left implicitly at a
     stale "open" default. A bank missing this measured 11/34 vs 34/34 still-held at 2s (the object
     released within milliseconds of reset because the PD target didn't match the commanded
     posture).
  2. EXACTLY TWO rigid_object keys for C4 -- insertive_object, receptive_object. NOT four. The
     training cfg's assumed_static_assets legitimately exempts table/ur5_metal_support for this
     reset type; writing them here would be an unrequested rekey, the exact defect class the
     "reset-file must never silently drop a needed rigid_object OR silently add an unneeded one"
     guard in MultiResetManager.__init__ (events.py, _assert_reset_file_covers_scene) exists to
     catch on the READ side -- this script avoids ever creating the WRITE-side version of it.
  3. ATOMIC WRITE. Writes to a temp file in the SAME directory as the destination, fsyncs it, then
     os.replace()s over the destination -- never truncates the destination path in place. Refuses
     to run at all if the destination already exists (versioned output path only; this project has
     twice been bitten by two files sharing a partial_assemblies.pt name where only one was good --
     the same discipline applies here).
  4. INDEPENDENT VALIDATION. After writing, RE-LOADS the file from disk (never trusts its own
     in-memory tensors) and reprojects insertive_object/receptive_object's root poses into the
     mating frame via the SAME scipy-Rotation-matrix path validate_deep_c4_partial_assemblies_v1.py
     already uses for the partial_assemblies.pt schema -- never the generator's own
     combine_frame_transforms call. Reports depth/lateral/tilt distributions and the fraction
     passing the TRUE 0.0025m/0.025rad gate, not just this script's own bookkeeping.
  5. GRIPPER-TARGET / HELD-OBJECT CONSISTENCY. Hand and object are placed by conceptually
     independent operations (the leg's pose comes from partial_assemblies_deep_v2.pt; the hand's
     posture comes from grasps.pt; only the ARM configuration linking them comes from a real FK'd
     C2/C3 state) -- exactly the class of bug this project has already seen once (gripper
     joint-target gap median ~11rad over 20 joints, correlated 0.5-0.6 with the object's post-reset
     velocity spike, from two independently-placed entities with no mutual settle). This script
     sets joint_position_target's hand indices to the SAME composed hand joint_position it writes,
     never to a value sourced from a different state -- so at write time there is no gap between
     "where the fingers are" and "where the PD controller thinks they should be", by construction.
     This does NOT by itself prove the hand is actually gripping the leg (see the settle question,
     answered in the team message this script ships with, not in code).

DOES NOT RUN ITSELF: needs a real Isaac boot (AppLauncher, SimulationContext, Articulation) for FK
and for the resolved robot.joint_names ordering. Written to be executed on an Isaac-capable box, not
run here.

Run (mirrors run_fk_match_composed_c4_deep_v2.sh's env/path conventions):
    PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl" \\
        timeout -s KILL 1800 <python> -u scripts_v2/tools/gen_composed_c4_reset_bank.py \\
        --grasps-path Datasets_ur5e_delto/OmniReset/Grasps/SquareTableLeg200mmDecomp/grasps.pt \\
        --partial-assembly-path local_ckpts/deep_c4_partial_assemblies_v1/partial_assemblies_deep_v2.pt \\
        --c3-bank-path <...>/resets_ObjectAnywhereEEGrasped.pt \\
        --c2-bank-path <...>/resets_ObjectRestingEEGrasped.pt \\
        --out local_ckpts/deep_c4_partial_assemblies_v1/resets_ObjectPartiallyAssembledEEGrasped_composed_v2.pt \\
        --headless
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compose a deep C4 reset bank from matched (leg pose, arm config, grasp) triples.")
parser.add_argument("--grasps-path", type=str, required=True)
parser.add_argument("--partial-assembly-path", type=str, required=True, help="Deep-band partial_assemblies.pt (e.g. deep_v2, [12,25]mm).")
parser.add_argument("--c3-bank-path", type=str, required=True, help="resets_ObjectAnywhereEEGrasped.pt")
parser.add_argument("--c2-bank-path", type=str, default=None, help="resets_ObjectRestingEEGrasped.pt (optional secondary arm-config source)")
parser.add_argument("--max-arm-configs", type=int, default=None)
parser.add_argument("--max-grasps", type=int, default=300, help="Same reasoning as fk_match_composed_c4.py's identical flag.")
parser.add_argument("--max-poses", type=int, default=None)
parser.add_argument("--pos-accept-mm", type=float, default=10.0, help="A pose is ACCEPTED into the bank only if its best match clears this position bound.")
parser.add_argument("--rot-accept-deg", type=float, default=15.0, help="...and this orientation bound, jointly (both must clear).")
parser.add_argument(
    "--receptive-x-range", type=str, default="0.35,0.60",
    help="Per-state receptive_object world x range (m) -- measured off the live resets_ObjectAnywhereEEGrasped.pt bank"
    " (std 0.0720, range [0.35,0.60]), NOT a fixed placement -- a fixed fixture pose would be a new, unrequested"
    " degeneracy axis this bank does not need.",
)
parser.add_argument("--receptive-y-range", type=str, default="-0.2,0.2", help="Measured range [-0.1999,0.2000].")
parser.add_argument("--receptive-yaw-range-deg", type=str, default="-15.0,15.0", help="Measured +-15.006deg.")
parser.add_argument("--receptive-z-m", type=float, default=0.0196, help="Measured constant height (std 0.0 across 10000 states).")
parser.add_argument("--chunk-size", type=int, default=512)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, required=True, help="Output .pt path. MUST NOT already exist.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

if os.path.exists(args_cli.out):
    raise FileExistsError(
        f"{args_cli.out} already exists -- refusing to overwrite (versioned output path only;"
        " this project has been bitten by two files sharing a name before)."
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
from isaaclab.sim import SimulationContext  # noqa: E402

from uwlab_assets.robots.ur5e_delto import IMPLICIT_UR5E_DELTO  # noqa: E402


# ============================================================================================
# Copied from fk_match_composed_c4.py (see module docstring above for why this is a copy, not an
# import). Kept behaviourally identical -- diff against that file if it changes.
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
    print(f"[gen_composed] grasps.pt: {num_grasps} grasps, gripper joints recorded: {sorted(recorded)}", flush=True)
    overlap = recorded & set(robot_joint_names)
    if not overlap:
        raise ValueError(
            f"gen_composed_c4_reset_bank: zero overlap between grasps.pt's recorded joints"
            f" ({sorted(recorded)}) and the robot's expected joints ({sorted(robot_joint_names)}) --"
            " this grasps.pt was very likely recorded for a different gripper. Unlike"
            " fk_match_composed_c4.py's reachability sweep (which only needs relative_position/"
            "relative_orientation and can proceed on a warning), THIS script writes"
            " gripper_joint_positions into the bank by joint NAME and must refuse outright if the"
            " name vocabulary does not match."
        )
    missing = set(robot_joint_names) - recorded
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
    print(f"[gen_composed] partial_assemblies.pt: {rel_pos.shape[0]} poses", flush=True)
    return rel_pos.to(device, dtype=torch.float32), rel_quat.to(device, dtype=torch.float32)


def _load_c_bank(path: str, device: str, label: str):
    """Reads a recorded *EEGrasped reset bank: leg pose + full robot state per state, INCLUDING the
    fields fk_match_composed_c4.py's own _load_c_bank never needed (joint_velocity,
    joint_position_target, joint_velocity_target, receptive_object's own root_pose) -- this script
    needs all of them to actually construct a state, not just to FK a palm pose."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} bank not found at {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    ro = data["initial_state"]["rigid_object"]
    if "insertive_object" not in ro:
        raise ValueError(f"{label} bank {path} has no insertive_object key -- not a 4-key/rekeyed bank")
    art = data["initial_state"]["articulation"]["robot"]

    # DEFENSIVE GUARD (critic-identified): this repo's own Datasets_ur5e_delto C2/C3 banks are an
    # OLDER GENERATION that predates joint_position_target/joint_velocity_target -- MultiResetManager
    # itself tolerates that (events.py:2328-2336, falls back to target := recorded joint_position/
    # joint_velocity, logging a warning once per file), but THIS script cannot: it inherits the arm
    # PORTION of joint_position_target verbatim from whichever C2/C3 state a pose matched against
    # (see main()), so a bank missing the field would silently propagate whatever the bare dict
    # index below produces -- and if that were allowed to fall back silently, the WRITTEN composed
    # bank would carry a stale/default target for the arm, or (worse, if the fallback masked this
    # entirely) leave the hand-joint target overwrite as the ONLY correct target in the file. Fail
    # up front, before any matching/FK work, naming the file, the missing fields, and why it matters
    # -- measured: a bank without correct PD targets replays with the gripper at its default OPEN
    # posture and releases the object almost immediately (34/34 vs 11/34 still-held at 2s).
    _required = ("joint_position_target", "joint_velocity_target")
    _missing = [k for k in _required if k not in art]
    if _missing:
        raise KeyError(
            f"{label} bank {path!r} is missing {_missing} under initial_state.articulation.robot -- "
            "this is an OLDER-GENERATION bank recorded before StableStateRecorder wrote PD targets. "
            "gen_composed_c4_reset_bank.py CANNOT fall back to target := joint_position the way "
            "MultiResetManager does at reset time (events.py's resolve_joint_targets), because this "
            "script inherits the ARM portion of joint_position_target verbatim from whichever C2/C3 "
            "state a leg pose matched against -- with no target field to inherit, the written bank "
            "would either crash later or (worse) silently carry a wrong/default target. A bank "
            "missing these fields replays with the gripper PD target at its default OPEN posture and "
            "releases the grasped object almost immediately after reset -- measured elsewhere in this "
            "project as 11/34 vs 34/34 still-held at 2 seconds. Point --c3-bank-path/--c2-bank-path "
            "at a bank that has these fields (the 2026-08-21 consolidated set on DL_H100 does; this "
            "repo's local Datasets_ur5e_delto copy of the same reset type may not -- check which one "
            "you are pointing at)."
        )

    out = {
        "leg_root_pose": torch.stack(ro["insertive_object"]["root_pose"]).to(device, dtype=torch.float32),
        "joint_position": torch.stack(art["joint_position"]).to(device, dtype=torch.float32),
        "joint_velocity": torch.stack(art["joint_velocity"]).to(device, dtype=torch.float32),
        "joint_position_target": torch.stack(art["joint_position_target"]).to(device, dtype=torch.float32),
        "joint_velocity_target": torch.stack(art["joint_velocity_target"]).to(device, dtype=torch.float32),
    }
    print(f"[gen_composed] {label} bank: {out['joint_position'].shape[0]} states, {out['joint_position'].shape[1]} joints", flush=True)
    return out


def _fk_batch(robot_cfg, joint_positions: torch.Tensor, sim: SimulationContext, device: str, chunk_size: int = 2048):
    """Identical to fk_match_composed_c4.py's _fk_batch -- see that file for the full comment on why
    this is chunked-clone FK rather than a per-state loop."""
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
        sim.forward()
        robot.update(0.0)

        palm_pos_all[start:end] = robot.data.body_pos_w[:, palm_id, :].clone()
        palm_quat_all[start:end] = robot.data.body_quat_w[:, palm_id, :].clone()

        sim_utils.delete_prim([f"/World/FkOrigin_{chunk_idx}_{i}" for i in range(n)])
        del robot

        print(f"[gen_composed] FK chunk {chunk_idx}: {start}:{end} of {n_total} done", flush=True)
        start = end

    return palm_pos_all, palm_quat_all


# ============================================================================================
# NEW: matching + state construction + atomic write + re-load validation.
# ============================================================================================

def _atomic_torch_save(obj, out_path: str) -> None:
    """Write to a temp file in the SAME directory, fsync, then os.replace -- never truncate the
    destination in place, and the destination is checked absent (above, before Isaac even booted)
    so this is the only write this process ever performs against out_path."""
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


def _reproject_independent(insertive_root_pose: np.ndarray, receptive_root_pose: np.ndarray):
    """Same independent (scipy Rotation matrix, NOT combine/subtract_frame_transforms) reprojection
    validate_deep_c4_partial_assemblies_v1.py already uses, generalized to a non-identity receptive
    root pose (that validator assumed receptive-at-world-origin, true for partial_assemblies.pt but
    NOT true for a reset bank, where receptive_object genuinely varies per state -- see the
    docstring above)."""
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

    rot_err_deg = np.abs(wrap(euler_xyz[:, 0])) + np.abs(wrap(euler_xyz[:, 1]))
    return depth_into_bore_mm, lateral_mm, tilt_deg, pos_err_mm, rot_err_deg


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    sim_cfg = sim_utils.SimulationCfg(device=device)
    sim = SimulationContext(sim_cfg)

    robot_cfg = IMPLICIT_UR5E_DELTO.copy()
    # NOTE: robot_cfg.init_state.joint_pos.keys() would give a name list too, but NOT the resolved
    # order (fk_match_composed_c4.py's own comment) -- this script waits for the post-spawn
    # probe_robot.joint_names below before calling _load_grasps, rather than using that coarse order.

    partial_rel_pos, partial_rel_quat = _load_partial_assemblies(args_cli.partial_assembly_path, device)
    if args_cli.max_poses is not None:
        partial_rel_pos, partial_rel_quat = partial_rel_pos[: args_cli.max_poses], partial_rel_quat[: args_cli.max_poses]
    n_poses = partial_rel_pos.shape[0]

    c3 = _load_c_bank(args_cli.c3_bank_path, device, "C3")
    sources = ["C3"] * c3["joint_position"].shape[0]
    all_c = {k: v for k, v in c3.items()}
    if args_cli.c2_bank_path is not None:
        c2 = _load_c_bank(args_cli.c2_bank_path, device, "C2")
        if c2["joint_position"].shape[1] != c3["joint_position"].shape[1]:
            raise ValueError("C2/C3 joint dimensionality mismatch")
        sources += ["C2"] * c2["joint_position"].shape[0]
        for k in all_c:
            all_c[k] = torch.cat([all_c[k], c2[k]], dim=0)

    if args_cli.max_arm_configs is not None and all_c["joint_position"].shape[0] > args_cli.max_arm_configs:
        idx = torch.randperm(all_c["joint_position"].shape[0])[: args_cli.max_arm_configs]
        for k in all_c:
            all_c[k] = all_c[k][idx]
        sources = [sources[i] for i in idx.tolist()]

    n_arm = all_c["joint_position"].shape[0]
    print(f"[gen_composed] total arm configs to FK: {n_arm} (from {set(sources)})", flush=True)

    fk_palm_pos, fk_palm_quat = _fk_batch(robot_cfg, all_c["joint_position"], sim, device, chunk_size=args_cli.chunk_size)

    # ---- SELF-CHECK, identical in spirit and tolerance to fk_match_composed_c4.py's: composition
    # round-trip on one real FK'd state, asserted before anything else. FAILS LOUDLY, no silent
    # fallback -- every number after this point is meaningless if it fails. ----
    s0_leg_pos, s0_leg_quat = all_c["leg_root_pose"][0, :3], all_c["leg_root_pose"][0, 3:7]
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
        f"[gen_composed] SELF-CHECK round-trip residual: {self_check_pos_err_mm:.6f} mm,"
        f" {self_check_rot_err_deg:.6f} deg",
        flush=True,
    )
    assert self_check_pos_err_mm < 0.01 and self_check_rot_err_deg < 0.01, (
        "SELF-CHECK FAILED: composition round-trip did not reconstruct the FK'd palm pose. STOPPING"
        " -- every match/state below would be built on an inverted composition."
    )
    print("[gen_composed] SELF-CHECK PASSED -- composition math verified against real FK. Proceeding.", flush=True)

    # ---- robot.joint_names, RESOLVED ORDER (only available post-spawn) ----
    # _fk_batch already spawned+deleted robots; spawn one more, un-deleted, purely to read the name
    # order (cheap, single instance).
    for i in range(1):
        sim_utils.create_prim(f"/World/JointNameProbe_{i}", "Xform", translation=(0.0, 0.0, 0.0))
    probe_cfg = robot_cfg.replace(prim_path="/World/JointNameProbe_.*/Robot")
    probe_robot = Articulation(cfg=probe_cfg)
    sim.reset()
    robot_joint_names = list(probe_robot.joint_names)
    print(f"[gen_composed] resolved robot.joint_names ({len(robot_joint_names)}): {robot_joint_names}", flush=True)

    grasp_rel_pos, grasp_rel_quat, gripper_joint_positions_dict, hand_joint_names = _load_grasps(
        args_cli.grasps_path, robot_joint_names, device
    )
    n_grasps_total = grasp_rel_pos.shape[0]
    grasp_idx_pool = torch.arange(n_grasps_total, device=device)
    if args_cli.max_grasps is not None and n_grasps_total > args_cli.max_grasps:
        grasp_idx_pool = torch.randperm(n_grasps_total, device=device)[: args_cli.max_grasps]
    grasp_rel_pos_sub = grasp_rel_pos[grasp_idx_pool]
    grasp_rel_quat_sub = grasp_rel_quat[grasp_idx_pool]
    n_grasps = grasp_rel_pos_sub.shape[0]

    name_to_idx = {name: i for i, name in enumerate(robot_joint_names)}
    hand_indices = [name_to_idx[n] for n in hand_joint_names]
    arm_indices = [i for i in range(len(robot_joint_names)) if i not in hand_indices]
    # gripper_joint_positions_dict[name] is a length-n_grasps_total list; stack into [n_grasps_total, n_hand]
    # in hand_joint_names order, then subsample the same way grasp_rel_pos/quat were.
    gjp_full = torch.stack(
        [torch.as_tensor([float(v) for v in gripper_joint_positions_dict[n]], dtype=torch.float32) for n in hand_joint_names],
        dim=1,
    ).to(device)  # [n_grasps_total, n_hand]
    gjp_sub = gjp_full[grasp_idx_pool]  # [n_grasps, n_hand]

    print(f"[gen_composed] using {n_grasps}/{n_grasps_total} grasps, {len(hand_indices)} hand joints, {len(arm_indices)} arm joints", flush=True)

    # ---- per-pose receptive_object world placement, sampled from the MEASURED training-time
    # jitter range (see argparse defaults' docstrings) -- NOT a single fixed pose. A fixed fixture
    # placement would be a new, unrequested degeneracy axis this bank does not need; the real
    # training scene randomizes x/y/yaw every episode and existing banks (resets_
    # ObjectAnywhereEEGrasped.pt) show it, so this bank should too. ----
    x_lo, x_hi = (float(v) for v in args_cli.receptive_x_range.split(","))
    y_lo, y_hi = (float(v) for v in args_cli.receptive_y_range.split(","))
    yaw_lo, yaw_hi = (math.radians(float(v)) for v in args_cli.receptive_yaw_range_deg.split(","))
    rec_x = math_utils.sample_uniform(x_lo, x_hi, (n_poses,), device=device)
    rec_y = math_utils.sample_uniform(y_lo, y_hi, (n_poses,), device=device)
    rec_yaw = math_utils.sample_uniform(yaw_lo, yaw_hi, (n_poses,), device=device)
    rec_pos_w = torch.stack([rec_x, rec_y, torch.full((n_poses,), args_cli.receptive_z_m, device=device)], dim=-1)
    rec_quat_w = math_utils.quat_from_euler_xyz(torch.zeros(n_poses, device=device), torch.zeros(n_poses, device=device), rec_yaw)

    # leg world pose for every pose k: combine_frame_transforms(receptive_world, partial_rel) --
    # events.py:1598-1600's own idiom (reset_insertive_object_from_partial_assembly_dataset).
    leg_pos_w, leg_quat_w = math_utils.combine_frame_transforms(rec_pos_w, rec_quat_w, partial_rel_pos, partial_rel_quat)

    # target palm pose for every (pose k, grasp g): broadcast leg pose over grasps -- same
    # broadcasting pattern as fk_match_composed_c4.py.
    leg_pos_kg = leg_pos_w.unsqueeze(1).expand(-1, n_grasps, -1).reshape(-1, 3)
    leg_quat_kg = leg_quat_w.unsqueeze(1).expand(-1, n_grasps, -1).reshape(-1, 4)
    grasp_pos_kg = grasp_rel_pos_sub.unsqueeze(0).expand(n_poses, -1, -1).reshape(-1, 3)
    grasp_quat_kg = grasp_rel_quat_sub.unsqueeze(0).expand(n_poses, -1, -1).reshape(-1, 4)
    target_pos, target_quat = math_utils.combine_frame_transforms(leg_pos_kg, leg_quat_kg, grasp_pos_kg, grasp_quat_kg)
    pose_index_kg = torch.arange(n_poses, device=device).unsqueeze(1).expand(-1, n_grasps).reshape(-1)
    grasp_index_kg = torch.arange(n_grasps, device=device).unsqueeze(0).expand(n_poses, -1).reshape(-1)

    from scipy.spatial import cKDTree  # noqa: E402

    fk_pos_np = fk_palm_pos.cpu().numpy()
    tree = cKDTree(fk_pos_np)
    target_pos_np = target_pos.cpu().numpy()
    _MATCH_KNN = 32
    k = min(_MATCH_KNN, fk_pos_np.shape[0])
    nn_dist_m, nn_idx = tree.query(target_pos_np, k=k)
    if k == 1:
        nn_dist_m, nn_idx = nn_dist_m.reshape(-1, 1), nn_idx.reshape(-1, 1)
    nn_idx_t = torch.as_tensor(nn_idx, device=device, dtype=torch.long)
    nn_pos_err_mm = torch.as_tensor(nn_dist_m, device=device) * 1000.0
    n_targets = nn_pos_err_mm.shape[0]
    cand_quat = fk_palm_quat[nn_idx_t]
    nn_rot_err_deg = (
        math_utils.quat_error_magnitude(
            target_quat.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4), cand_quat.reshape(-1, 4)
        ).reshape(n_targets, k)
        * 180.0 / np.pi
    )

    _SCORE_POS_NORM_MM, _SCORE_ROT_NORM_DEG = 10.0, 15.0
    combined_score = (nn_pos_err_mm / _SCORE_POS_NORM_MM) ** 2 + (nn_rot_err_deg / _SCORE_ROT_NORM_DEG) ** 2
    # Mask out candidates that fail the ACCEPT bound before picking a "best" -- unlike
    # fk_match_composed_c4.py's reporting-only script, this one must actually decide accept/reject.
    fails_accept = (nn_pos_err_mm > args_cli.pos_accept_mm) | (nn_rot_err_deg > args_cli.rot_accept_deg)
    combined_score_masked = torch.where(fails_accept, torch.full_like(combined_score, float("inf")), combined_score)
    best_j = combined_score_masked.argmin(dim=1)
    row_idx = torch.arange(n_targets, device=device)
    best_pos_err_mm = nn_pos_err_mm[row_idx, best_j]
    best_rot_err_deg = nn_rot_err_deg[row_idx, best_j]
    best_arm_idx = nn_idx_t[row_idx, best_j]  # index into all_c / sources
    target_accepted = ~fails_accept.gather(1, best_j.unsqueeze(1)).squeeze(1)

    # ---- per POSE (not per pose*grasp target): keep the single best-scoring ACCEPTED
    # (grasp, arm_config) pair, if any. ----
    n_hand = len(hand_indices)
    n_joints = len(robot_joint_names)
    kept_pose_idx, kept_arm_idx, kept_grasp_idx_local, kept_pos_err, kept_rot_err, kept_source = [], [], [], [], [], []
    combined_score_np = combined_score_masked[row_idx, best_j].cpu().numpy()
    pose_index_np = pose_index_kg.cpu().numpy()
    accepted_np = target_accepted.cpu().numpy()
    for k_pose in range(n_poses):
        rows = np.where(pose_index_np == k_pose)[0]
        rows = rows[accepted_np[rows]]
        if rows.size == 0:
            continue
        best_row = rows[np.argmin(combined_score_np[rows])]
        kept_pose_idx.append(k_pose)
        kept_arm_idx.append(int(best_arm_idx[best_row].item()))
        kept_grasp_idx_local.append(int(grasp_index_kg[best_row].item()))
        kept_pos_err.append(float(best_pos_err_mm[best_row].item()))
        kept_rot_err.append(float(best_rot_err_deg[best_row].item()))
        kept_source.append(sources[int(best_arm_idx[best_row].item())])

    n_kept = len(kept_pose_idx)
    print(
        f"[gen_composed] {n_kept}/{n_poses} poses have an ACCEPTED match"
        f" (pos<={args_cli.pos_accept_mm}mm, rot<={args_cli.rot_accept_deg}deg);"
        f" dropped {n_poses - n_kept}",
        flush=True,
    )
    if n_kept == 0:
        raise RuntimeError("gen_composed_c4_reset_bank: zero poses accepted -- nothing to write. Loosen --pos-accept-mm/--rot-accept-deg or investigate coverage via fk_match_composed_c4.py first.")

    kept_pose_idx_t = torch.as_tensor(kept_pose_idx, device=device, dtype=torch.long)
    kept_arm_idx_t = torch.as_tensor(kept_arm_idx, device=device, dtype=torch.long)
    kept_grasp_idx_t = torch.as_tensor(kept_grasp_idx_local, device=device, dtype=torch.long)

    # ---- construct the state, per kept pose ----
    ins_root_pos = leg_pos_w[kept_pose_idx_t]
    ins_root_quat = leg_quat_w[kept_pose_idx_t]
    rec_root_pos = rec_pos_w[kept_pose_idx_t]
    rec_root_quat = rec_quat_w[kept_pose_idx_t]

    arm_joint_position = all_c["joint_position"][kept_arm_idx_t].clone()  # [n_kept, n_joints], full vector
    arm_joint_position_target = all_c["joint_position_target"][kept_arm_idx_t].clone()

    # DEFECT (1)/(5): hand indices overwritten with the MATCHED GRASP's own recorded closed
    # posture, and the target for those SAME indices is set to follow -- never inherited from
    # whichever arbitrary posture the source C2/C3 state happened to have, and never left implicit.
    hand_posture = gjp_sub[kept_grasp_idx_t]  # [n_kept, n_hand]
    joint_position = arm_joint_position.clone()
    joint_position[:, hand_indices] = hand_posture
    joint_position_target = arm_joint_position_target.clone()
    joint_position_target[:, hand_indices] = hand_posture  # target := composed position, hand only

    joint_velocity = torch.zeros((n_kept, n_joints), device=device)
    joint_velocity_target = torch.zeros((n_kept, n_joints), device=device)

    robot_root_pose = torch.zeros((n_kept, 7), device=device)
    robot_root_pose[:, 3] = 1.0  # identity quat, world origin -- same convention as _fk_batch/ur5e_delto.py init_state
    robot_root_velocity = torch.zeros((n_kept, 6), device=device)

    insertive_root_pose = torch.cat([ins_root_pos, ins_root_quat], dim=-1)
    insertive_root_velocity = torch.zeros((n_kept, 6), device=device)
    receptive_root_pose = torch.cat([rec_root_pos, rec_root_quat], dim=-1)
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
        # EXACTLY TWO KEYS -- see module docstring, defect (2). Do not add table/ur5_metal_support.
        "rigid_object": {
            "insertive_object": {"root_pose": list(insertive_root_pose.cpu()), "root_velocity": list(insertive_root_velocity.cpu())},
            "receptive_object": {"root_pose": list(receptive_root_pose.cpu()), "root_velocity": list(receptive_root_velocity.cpu())},
        },
    }
    bank = {"initial_state": initial_state}

    _atomic_torch_save(bank, args_cli.out)
    out_sha = _sha256(args_cli.out)
    print(f"[gen_composed] wrote {args_cli.out}", flush=True)
    print(f"[gen_composed] n={n_kept}, sha256={out_sha}, size={os.path.getsize(args_cli.out)} bytes", flush=True)

    # ================= DEFECT (4): RE-LOAD and INDEPENDENTLY VALIDATE =================
    reloaded = torch.load(args_cli.out, map_location="cpu", weights_only=False)
    ins_np = torch.stack(reloaded["initial_state"]["rigid_object"]["insertive_object"]["root_pose"]).numpy()
    rec_np = torch.stack(reloaded["initial_state"]["rigid_object"]["receptive_object"]["root_pose"]).numpy()
    depth_mm, lateral_mm, tilt_deg, pos_err_mm, rot_err_deg = _reproject_independent(ins_np, rec_np)

    pos_ok = pos_err_mm < 2.5
    rot_ok = rot_err_deg < math.degrees(0.025)
    both_ok = pos_ok & rot_ok

    def pct(a, ps=(0, 10, 25, 50, 75, 90, 100)):
        return {p: float(np.percentile(a, p)) for p in ps}

    print("\n=== RE-LOADED, INDEPENDENTLY REPROJECTED (scipy Rotation, not this script's own combine_frame_transforms) ===", flush=True)
    print("depth_into_bore_mm:", pct(depth_mm), flush=True)
    print("lateral_mm:", pct(lateral_mm), flush=True)
    print("tilt_deg:", pct(tilt_deg), flush=True)
    print(f"fraction pos_ok (<2.5mm): {float(pos_ok.mean()):.4f}", flush=True)
    print(f"fraction rot_ok (<1.4324deg): {float(rot_ok.mean()):.4f}", flush=True)
    print(f"fraction BOTH (predicted baseline of the WRITTEN bank): {float(both_ok.mean()):.4f}", flush=True)

    summary_path = os.path.splitext(args_cli.out)[0] + "_validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "n_kept": n_kept, "n_poses_input": n_poses, "sha256": out_sha,
            "match_pos_err_mm_at_accept": kept_pos_err, "match_rot_err_deg_at_accept": kept_rot_err,
            "match_source_counts": {s: kept_source.count(s) for s in set(kept_source)},
            "depth_into_bore_mm_pct": pct(depth_mm), "lateral_mm_pct": pct(lateral_mm), "tilt_deg_pct": pct(tilt_deg),
            "pos_ok_fraction": float(pos_ok.mean()), "rot_ok_fraction": float(rot_ok.mean()), "both_ok_fraction": float(both_ok.mean()),
        }, f, indent=2, default=float)
    print(f"[gen_composed] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
