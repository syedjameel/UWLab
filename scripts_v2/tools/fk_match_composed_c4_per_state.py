# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FK-matching sweep for the COMPOSED-STATE C4 design -- PER-STATE FIXTURE variant.

THIS IS A VARIANT OF fk_match_composed_c4.py, not a bugfix to its composition math. That script's
frame algebra was audited term-for-term against the production consumer (omnireset/mdp/events.py:
1421-1423 and 1598-1600) and found CORRECT -- no argument-order inversion. The all-zero result it
produced (156 mm floor, 0/4096 pose coverage) traced instead to a WORKSPACE-PLACEMENT bug: that
script builds every partial-assembly TARGET using one CONSTANT --receptive-world-pos (default
(0,0,0), i.e. the robot's own base -- reset_states_cfg.py:200-202 documents x < 0.35 m as "the black
mat ... the base's zone", a physical dead-zone the arm cannot occupy), then compares that target
cloud against FK'd palm poses drawn from C2/C3 states whose *real* recorded receptive_object was
placed somewhere in x in [0.35,0.60], y in [-0.20,0.20], yaw +-15 deg (reset_states_cfg.py:195-220)
-- a DIFFERENT, RANDOM placement for every single state. One global constant cannot represent that;
matching against it injects up to +-125 mm in x and +-200 mm in y of pure placement noise on top of
whatever real reachability gap exists, easily enough to manufacture a second false negative.

THE FIX, PER-STATE: every C2/C3 bank state already carries its OWN recorded receptive_object
root_pose (initial_state.rigid_object.receptive_object.root_pose) -- the fixture placement that
state's arm configuration was ACTUALLY reached against. Instead of composing the target INTO world
frame with a guessed constant and comparing world-frame palm poses, this variant goes the other way:
it takes each FK'd palm pose OUT of world frame and INTO that state's OWN receptive_object frame via
subtract_frame_transforms(receptive_pos_i, receptive_quat_i, fk_palm_pos_i, fk_palm_quat_i). The
partial-assembly and grasp targets are ALREADY expressed relative to the receptive object (that is
the entire meaning of "insertive-in-receptive" / "gripper-in-object" relative poses -- see the parent
script's docstring, unchanged here), so composing them together (still via combine_frame_transforms,
same order, same convention, unchanged from the parent script) produces a target that lives in the
SAME per-state-relative frame as the transformed candidate palm pose -- with NO free constant, and NO
world-frame guess, anywhere in the comparison. The +-125/+-200 mm per-episode jitter cancels exactly,
by construction, rather than by hoping a --receptive-world-pos guess happens to average it out.

FALLBACK: not every bank this script might ever be pointed at is guaranteed to carry a
receptive_object key (the C4-type bank itself carries only insertive_object + insertive-relative-to-
receptive info, not a full 4-key initial_state -- do not assume key sets are uniform across bank
types). For any bank/state missing "receptive_object", this variant falls back to the ORIGINAL
global --receptive-world-pos world-frame approach for exactly those states, reported SEPARATELY so a
reader can tell which pool (per-state or fallback) supplied which matches. For the C2/C3 banks this
campaign actually uses, receptive_object is present for every state, so the fallback pool is expected
to be empty -- it exists for robustness, not because this run needs it.

QUATERNIONS remain (w, x, y, z) scalar-first throughout, exactly as in the parent script. All
composition goes through isaaclab.utils.math (combine_frame_transforms / subtract_frame_transforms),
never a hand-rolled or scipy-mediated reorder that could silently flip the convention.

STRENGTHENED SELF-CHECK: the parent script's round-trip self-check (subtract then combine on one real
(leg, FK'd palm) pair) is KEPT UNCHANGED below -- it is a valid regression guard on IsaacLab's own
combine/subtract pairing -- but it was, by construction, incapable of catching either an argument-order
inversion (there wasn't one) or the workspace-placement bug (there was). This variant adds a SECOND,
independent check that WOULD have caught the placement bug: it asserts the WORLD-frame centroid of
whichever target cloud is actually being compared in world coordinates lies inside the documented
reachable workspace (x > 0.35 m, reset_states_cfg.py:205), and fails loudly with the actual centroid
if not. Concretely: (a) the per-state pool's own recorded receptive_object positions (real, already-
reached placements) must themselves satisfy x > 0.35 m -- a sanity check that the right field was read,
not a check on this script's own math; (b) the fallback pool (if non-empty), which still relies on the
free --receptive-world-pos constant, has its resulting world-frame target centroid checked directly
against the same bound -- this is the literal check that fails on the exact misconfiguration that
produced the original "0/4096" result.

Run (one Isaac process; never via uwlab.sh; same CLI contract as the parent script, args unchanged):

    PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl" \\
        timeout -s KILL 600 <python> -u scripts_v2/tools/fk_match_composed_c4_per_state.py \\
        --grasps-path <Datasets_ur5e_delto or Datasets_render>/OmniReset/Grasps/<leg-object-name>/grasps.pt \\
        --partial-assembly-path <...>/partial_assemblies_deep_v2.pt \\
        --c3-bank-path <...>/resets_ObjectAnywhereEEGrasped.pt \\
        --c2-bank-path <...>/resets_ObjectRestingEEGrasped.pt \\
        --out-dir /tmp/c4_fk_sweep_per_state

    --receptive-world-pos / --receptive-world-quat-wxyz are now FALLBACK-ONLY: they are read but only
    ever applied to states whose bank lacks a receptive_object key. For C2/C3 they should have no
    effect on the result (a "fallback pool size" of 0 in the report confirms this).
"""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FK-matching sweep: composed-state C4 reachability go/no-go (per-state fixture).")
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
    help="FALLBACK ONLY (see module docstring): assumed WORLD pose of receptive_object, comma-separated"
    " xyz (m), used ONLY for arm-config states whose bank has no per-state receptive_object root_pose."
    " For C2/C3 banks (which do carry it) this is unused and should not affect the result.",
)
parser.add_argument(
    "--receptive-world-quat-wxyz", type=str, default="1,0,0,0",
    help="FALLBACK ONLY: assumed receptive_object world quat, WXYZ. See --receptive-world-pos.",
)
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
parser.add_argument(
    "--min-workspace-x-m", type=float, default=0.35,
    help="Reachable-workspace lower x bound (m), reset_states_cfg.py:205 -- the fixture-placement"
    " sanity check (see module docstring) fails loudly if a world-frame target/receptive centroid"
    " falls at or below this, since x <= this is documented as the robot's own base zone.",
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
# via IMPLICIT_UR5E_DELTO). Unchanged from the parent script -- see its docstring for why.
from uwlab_assets.robots.ur5e_delto import IMPLICIT_UR5E_DELTO  # noqa: E402


def _load_grasps(path: str, robot_joint_names: list[str], device: str):
    """Mirrors omnireset/mdp/events.py:1330-1395 (_load_and_precompute_grasps) exactly. Unchanged."""
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
    """Mirrors omnireset/mdp/events.py:1550-1568 (reset_insertive_object_from_partial_assembly_dataset).
    Unchanged: this dataset's relative_position/relative_orientation are already insertive-in-receptive,
    i.e. ALREADY expressed in the receptive object's own frame -- exactly the frame the per-state path
    below matches candidates into, with no extra composition needed.
    """
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
    """Reads a recorded *EEGrasped reset bank: leg pose + full robot joint_position per state, PLUS
    (new in this variant) the state's own recorded receptive_object root_pose when the bank carries
    one. Returns (leg_root_pose, joint_pos, receptive_root_pose, has_receptive) where
    receptive_root_pose is NaN-filled (never used) when has_receptive is False -- callers must gate on
    has_receptive/isnan, never on shape alone, since the tensor is always [N, 7].
    """
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
    n = joint_pos.shape[0]
    if "receptive_object" in ro:
        receptive_root_pose = torch.stack(ro["receptive_object"]["root_pose"]).to(device, dtype=torch.float32)
        has_receptive = True
    else:
        # Do NOT assume key sets are uniform across bank types (the C4-type bank itself only carries
        # 2 rigid_object keys) -- degrade to the fallback pool for this bank rather than crashing.
        receptive_root_pose = torch.full((n, 7), float("nan"), device=device)
        has_receptive = False
        print(
            f"[fk_match] WARNING: {label} bank has no receptive_object key -- its {n} states will use"
            f" the FALLBACK global --receptive-world-pos path, not per-state fixture matching.",
            flush=True,
        )
    print(
        f"[fk_match] {label} bank: {n} states, {joint_pos.shape[1]} joints,"
        f" receptive_object recorded: {has_receptive}",
        flush=True,
    )
    return leg_root_pose, joint_pos, receptive_root_pose, has_receptive


def _fk_batch(robot_cfg, joint_positions: torch.Tensor, sim: SimulationContext, device: str, chunk_size: int = 2048):
    """Forward-kinematics a batch of stored joint configs to the rl_dg_mount (palm) world pose.
    Unchanged from the parent script -- FK body, chunking, and deletion strategy are untouched.
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

        sim_utils.delete_prim([f"/World/FkOrigin_{chunk_idx}_{i}" for i in range(n)])
        del robot

        print(f"[fk_match] FK chunk {chunk_idx}: {start}:{end} of {n_total} done", flush=True)
        start = end

    return palm_pos_all, palm_quat_all


def _match_and_report(
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    pose_index_kg: torch.Tensor,
    fk_pos: torch.Tensor,
    fk_quat: torch.Tensor,
    cand_sources: list[str],
    pos_thresholds: list[float],
    rot_thresholds: list[float],
    device: str,
):
    """Shared KD-tree nearest-neighbor matching + go/no-go reporting, factored out of the parent
    script's main() so it can be run once per pool (per-state / fallback) with identical logic.
    Returns a dict populated the same way as the parent script's `summary`, scoped to this pool.
    Assumes target_pos/target_quat and fk_pos/fk_quat are ALREADY expressed in the SAME frame --
    callers are responsible for that (either both per-state-relative, or both world-frame).
    """
    from scipy.spatial import cKDTree

    n_targets = target_pos.shape[0]
    result = {
        "n_candidates": int(fk_pos.shape[0]),
        "n_targets": n_targets,
        "match_count_matrix": {},
        "matched_pose_coverage": {},
        "supplying_bank_at_thresholds": {},
    }
    if n_targets == 0 or fk_pos.shape[0] == 0:
        result["note"] = "empty pool -- skipped"
        return result, None, None

    fk_pos_np = fk_pos.cpu().numpy()
    tree = cKDTree(fk_pos_np)
    target_pos_np = target_pos.cpu().numpy()
    _MATCH_KNN = 32
    k = min(_MATCH_KNN, fk_pos_np.shape[0])
    nn_dist_m, nn_idx = tree.query(target_pos_np, k=k)
    if k == 1:
        nn_dist_m = nn_dist_m.reshape(-1, 1)
        nn_idx = nn_idx.reshape(-1, 1)
    nn_idx_t = torch.as_tensor(nn_idx, device=device, dtype=torch.long)
    nn_pos_err_mm = torch.as_tensor(nn_dist_m, device=device) * 1000.0
    cand_quat = fk_quat[nn_idx_t]
    nn_rot_err_deg = (
        math_utils.quat_error_magnitude(
            target_quat.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4), cand_quat.reshape(-1, 4)
        ).reshape(n_targets, k)
        * 180.0
        / np.pi
    )

    _SCORE_POS_NORM_MM = 10.0
    _SCORE_ROT_NORM_DEG = 15.0
    combined_score = (nn_pos_err_mm / _SCORE_POS_NORM_MM) ** 2 + (nn_rot_err_deg / _SCORE_ROT_NORM_DEG) ** 2
    best_j = combined_score.argmin(dim=1)
    row_idx = torch.arange(n_targets, device=device)
    best_pos_err_mm = nn_pos_err_mm[row_idx, best_j]
    best_rot_err_deg = nn_rot_err_deg[row_idx, best_j]

    def pct(t):
        return [float(np.percentile(t, p)) for p in (0, 10, 25, 50, 75, 90, 100)]

    pos_np, rot_np = best_pos_err_mm.cpu().numpy(), best_rot_err_deg.cpu().numpy()
    result["pos_err_mm_percentiles_[0,10,25,50,75,90,100]"] = pct(pos_np)
    result["rot_err_deg_percentiles_[0,10,25,50,75,90,100]"] = pct(rot_np)

    for p in pos_thresholds:
        for r in rot_thresholds:
            cell_hit = (nn_pos_err_mm <= p) & (nn_rot_err_deg <= r)
            mask = cell_hit.any(dim=1)
            n_match = int(mask.sum().item())
            result["match_count_matrix"][f"pos<={p}mm,rot<={r}deg"] = n_match
            if n_match > 0:
                matched_target_idx = mask.nonzero(as_tuple=True)[0]
                first_qualifying_j = cell_hit[matched_target_idx].to(torch.uint8).argmax(dim=1)
                qualifying_cand_idx = nn_idx_t[matched_target_idx, first_qualifying_j]
                srcs = [cand_sources[i] for i in qualifying_cand_idx.cpu().tolist()]
                result["supplying_bank_at_thresholds"][f"pos<={p}mm,rot<={r}deg"] = {
                    s: srcs.count(s) for s in set(srcs)
                }

    mid_p, mid_r = pos_thresholds[len(pos_thresholds) // 2], rot_thresholds[len(rot_thresholds) // 2]
    mid_mask = ((nn_pos_err_mm <= mid_p) & (nn_rot_err_deg <= mid_r)).any(dim=1)
    matched_pose_idx = pose_index_kg[mid_mask].cpu().numpy()
    result["matched_pose_coverage"] = {
        "threshold_used": f"pos<={mid_p}mm,rot<={mid_r}deg",
        "n_distinct_poses_matched": len(set(matched_pose_idx.tolist())),
    }
    return result, matched_pose_idx, mid_mask


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    torch.manual_seed(args_cli.seed)
    os.makedirs(args_cli.out_dir, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(device=device)
    sim = SimulationContext(sim_cfg)

    robot_cfg = IMPLICIT_UR5E_DELTO.copy()
    joint_names = list(robot_cfg.init_state.joint_pos.keys())

    # ---- load everything (CPU-cheap, no Isaac needed for this part) ----
    partial_rel_pos, partial_rel_quat = _load_partial_assemblies(args_cli.partial_assembly_path, device)
    if args_cli.max_poses is not None:
        partial_rel_pos, partial_rel_quat = partial_rel_pos[: args_cli.max_poses], partial_rel_quat[: args_cli.max_poses]
    n_poses = partial_rel_pos.shape[0]

    c3_leg_pose, c3_joint_pos, c3_receptive_pose, c3_has_receptive = _load_c_bank(
        args_cli.c3_bank_path, device, "C3 (ObjectAnywhereEEGrasped)"
    )
    sources = ["C3"] * c3_leg_pose.shape[0]
    all_leg_pose, all_joint_pos, all_receptive_pose = c3_leg_pose, c3_joint_pos, c3_receptive_pose
    if args_cli.c2_bank_path is not None:
        c2_leg_pose, c2_joint_pos, c2_receptive_pose, c2_has_receptive = _load_c_bank(
            args_cli.c2_bank_path, device, "C2 (ObjectRestingEEGrasped)"
        )
        if c2_joint_pos.shape[1] != c3_joint_pos.shape[1]:
            raise ValueError("C2/C3 joint dimensionality mismatch -- banks are not from the same robot")
        sources += ["C2"] * c2_leg_pose.shape[0]
        all_leg_pose = torch.cat([all_leg_pose, c2_leg_pose], dim=0)
        all_joint_pos = torch.cat([all_joint_pos, c2_joint_pos], dim=0)
        all_receptive_pose = torch.cat([all_receptive_pose, c2_receptive_pose], dim=0)

    if args_cli.max_arm_configs is not None and all_joint_pos.shape[0] > args_cli.max_arm_configs:
        idx = torch.randperm(all_joint_pos.shape[0])[: args_cli.max_arm_configs]
        all_leg_pose, all_joint_pos = all_leg_pose[idx], all_joint_pos[idx]
        all_receptive_pose = all_receptive_pose[idx]
        sources = [sources[i] for i in idx.tolist()]

    n_arm = all_joint_pos.shape[0]
    has_own_receptive = ~torch.isnan(all_receptive_pose[:, 0])
    print(
        f"[fk_match] total arm configs to FK: {n_arm} (from {set(sources)});"
        f" {int(has_own_receptive.sum().item())} carry a per-state receptive_object pose,"
        f" {int((~has_own_receptive).sum().item())} fall back to --receptive-world-pos",
        flush=True,
    )

    # ---- FK all arm configs, batched (unchanged from parent script) ----
    fk_palm_pos, fk_palm_quat = _fk_batch(robot_cfg, all_joint_pos, sim, device, chunk_size=args_cli.chunk_size)

    # ---- SELF-CHECK 1 (unchanged from parent script): composition round-trip on ONE real arm-config
    # state. Kept as a regression guard on IsaacLab's own combine/subtract pairing -- structurally
    # incapable of catching either an argument-order inversion or a workspace-placement bug (neither
    # of grasp_rel_pos nor receptive_world_pos is ever touched here), which is exactly why it did not
    # catch the placement bug the parent script shipped with. See SELF-CHECK 2 below. ----
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
        f"[fk_match] SELF-CHECK 1 (round-trip) residual: {self_check_pos_err_mm:.6f} mm,"
        f" {self_check_rot_err_deg:.6f} deg",
        flush=True,
    )
    assert self_check_pos_err_mm < 0.01 and self_check_rot_err_deg < 0.01, (
        "SELF-CHECK 1 FAILED: composition round-trip did not reconstruct the FK'd palm pose. STOPPING."
    )
    print("[fk_match] SELF-CHECK 1 PASSED. Proceeding.", flush=True)

    # ---- SELF-CHECK 2 (NEW): workspace-placement plausibility. This is the check that WOULD have
    # caught the parent script's actual bug -- it asserts that whichever target/receptive cloud is
    # being reasoned about in WORLD coordinates has its x-centroid past the documented base dead-zone
    # boundary (reset_states_cfg.py:205, "x < 0.35 m ... the base's zone"), and fails loudly with the
    # real centroid rather than silently reporting a spurious "0 matches". ----
    min_x = args_cli.min_workspace_x_m
    if bool(has_own_receptive.any()):
        own_idx = has_own_receptive.nonzero(as_tuple=True)[0]
        own_centroid_x = all_receptive_pose[own_idx, 0].mean().item()
        print(
            f"[fk_match] SELF-CHECK 2a: per-state receptive_object centroid x = {own_centroid_x*1000:.1f} mm"
            f" over {own_idx.numel()} states (must be > {min_x*1000:.0f} mm -- sanity that the right"
            f" recorded field was read, not a check on this script's own math).",
            flush=True,
        )
        assert own_centroid_x > min_x, (
            f"SELF-CHECK 2a FAILED: per-state receptive_object x-centroid ({own_centroid_x*1000:.1f} mm)"
            f" is at or inside the documented base dead-zone (x <= {min_x*1000:.0f} mm,"
            " reset_states_cfg.py:205). Either the wrong bank/field was loaded, or the bank's real"
            " placements are not where the docstring says they should be -- STOPPING rather than"
            " reporting a spurious go/no-go verdict."
        )
    print("[fk_match] SELF-CHECK 2 PASSED (or not applicable). Proceeding.", flush=True)

    # ---- load grasps ----
    grasp_rel_pos, grasp_rel_quat, _ = _load_grasps(args_cli.grasps_path, joint_names, device)
    n_grasps_total = grasp_rel_pos.shape[0]
    if args_cli.max_grasps is not None and n_grasps_total > args_cli.max_grasps:
        g_idx = torch.randperm(n_grasps_total)[: args_cli.max_grasps]
        grasp_rel_pos, grasp_rel_quat = grasp_rel_pos[g_idx], grasp_rel_quat[g_idx]
    n_grasps = grasp_rel_pos.shape[0]
    print(f"[fk_match] using {n_grasps}/{n_grasps_total} grasps x {n_poses} poses = {n_grasps * n_poses} targets", flush=True)

    grasp_pos_kg = grasp_rel_pos.unsqueeze(0).expand(n_poses, -1, -1).reshape(-1, 3)
    grasp_quat_kg = grasp_rel_quat.unsqueeze(0).expand(n_poses, -1, -1).reshape(-1, 4)
    pose_index_kg = torch.arange(n_poses, device=device).unsqueeze(1).expand(-1, n_grasps).reshape(-1)

    pos_thresholds = [float(x) for x in args_cli.pos_thresholds_mm.split(",")]
    rot_thresholds = [float(x) for x in args_cli.rot_thresholds_deg.split(",")]

    summary = {
        "n_arm_configs_fk": n_arm,
        "n_poses": n_poses,
        "n_grasps": n_grasps,
        "self_check_1_round_trip_pos_err_mm": self_check_pos_err_mm,
        "self_check_1_round_trip_rot_err_deg": self_check_rot_err_deg,
        "per_state_pool": {},
        "fallback_pool": {},
    }
    all_matched_pose_idx = []

    # ---- PER-STATE POOL (the fix): both sides expressed relative to the state's OWN receptive_object,
    # no free constant. leg pose IS partial_rel_pos/partial_rel_quat directly -- that dataset's
    # relative_position/relative_orientation is ALREADY insertive-in-receptive, i.e. already in this
    # frame; combining it with the grasp's gripper-in-object relative pose (same combine_frame_transforms
    # call, same order, verified against events.py -- see module docstring) gives the target
    # gripper-in-receptive pose. ----
    if bool(has_own_receptive.any()):
        own_idx = has_own_receptive.nonzero(as_tuple=True)[0]
        n_own = own_idx.numel()

        target_pos_local, target_quat_local = math_utils.combine_frame_transforms(
            partial_rel_pos.unsqueeze(1).expand(-1, n_grasps, -1).reshape(-1, 3),
            partial_rel_quat.unsqueeze(1).expand(-1, n_grasps, -1).reshape(-1, 4),
            grasp_pos_kg,
            grasp_quat_kg,
        )

        fk_palm_pos_local, fk_palm_quat_local = math_utils.subtract_frame_transforms(
            all_receptive_pose[own_idx, :3], all_receptive_pose[own_idx, 3:7],
            fk_palm_pos[own_idx], fk_palm_quat[own_idx],
        )
        own_sources = [sources[i] for i in own_idx.tolist()]

        print(f"\n=== PER-STATE POOL: {n_own} arm configs, {target_pos_local.shape[0]} targets ===", flush=True)
        result, matched_pose_idx, _ = _match_and_report(
            target_pos_local, target_quat_local, pose_index_kg,
            fk_palm_pos_local, fk_palm_quat_local, own_sources,
            pos_thresholds, rot_thresholds, device,
        )
        summary["per_state_pool"] = result
        print(f"pos_err_mm percentiles: {result.get('pos_err_mm_percentiles_[0,10,25,50,75,90,100]')}", flush=True)
        print(f"rot_err_deg percentiles: {result.get('rot_err_deg_percentiles_[0,10,25,50,75,90,100]')}", flush=True)
        print(f"match_count_matrix: {result['match_count_matrix']}", flush=True)
        if matched_pose_idx is not None:
            all_matched_pose_idx.append(matched_pose_idx)
    else:
        summary["per_state_pool"] = {"note": "no arm configs carried a per-state receptive_object pose"}
        print("[fk_match] PER-STATE POOL empty -- no bank supplied a receptive_object pose.", flush=True)

    # ---- FALLBACK POOL (parent-script behavior, unchanged, restricted to states lacking a per-state
    # receptive_object pose). Expected empty for C2/C3. ----
    fallback_idx = (~has_own_receptive).nonzero(as_tuple=True)[0]
    if fallback_idx.numel() > 0:
        rec_pos_w = torch.tensor([float(x) for x in args_cli.receptive_world_pos.split(",")], device=device).unsqueeze(0)
        rec_quat_w = torch.tensor(
            [float(x) for x in args_cli.receptive_world_quat_wxyz.split(",")], device=device
        ).unsqueeze(0)

        leg_pos_w, leg_quat_w = math_utils.combine_frame_transforms(
            rec_pos_w.expand(n_poses, -1), rec_quat_w.expand(n_poses, -1), partial_rel_pos, partial_rel_quat
        )
        # SELF-CHECK 2b: the literal check that would have caught the parent script's bug.
        leg_centroid_x = leg_pos_w[:, 0].mean().item()
        print(
            f"[fk_match] SELF-CHECK 2b: fallback-pool world-frame leg/target centroid x ="
            f" {leg_centroid_x*1000:.1f} mm (must be > {min_x*1000:.0f} mm, reset_states_cfg.py:205).",
            flush=True,
        )
        assert leg_centroid_x > min_x, (
            f"SELF-CHECK 2b FAILED: --receptive-world-pos ({args_cli.receptive_world_pos}) places the"
            f" fallback-pool target cloud's x-centroid at {leg_centroid_x*1000:.1f} mm, at or inside the"
            f" documented base dead-zone (x <= {min_x*1000:.0f} mm). This is the exact misconfiguration"
            " that produced the original script's spurious '0/4096' result -- pass a real workspace"
            " placement (e.g. the midpoint of reset_states_cfg.py's x in [0.35,0.60]) instead. STOPPING"
            " rather than reporting a meaningless verdict."
        )

        leg_pos_kg = leg_pos_w.unsqueeze(1).expand(-1, n_grasps, -1).reshape(-1, 3)
        leg_quat_kg = leg_quat_w.unsqueeze(1).expand(-1, n_grasps, -1).reshape(-1, 4)
        target_pos_world, target_quat_world = math_utils.combine_frame_transforms(
            leg_pos_kg, leg_quat_kg, grasp_pos_kg, grasp_quat_kg
        )
        fallback_sources = [sources[i] for i in fallback_idx.tolist()]

        print(f"\n=== FALLBACK POOL: {fallback_idx.numel()} arm configs, {target_pos_world.shape[0]} targets ===", flush=True)
        result, matched_pose_idx, _ = _match_and_report(
            target_pos_world, target_quat_world, pose_index_kg,
            fk_palm_pos[fallback_idx], fk_palm_quat[fallback_idx], fallback_sources,
            pos_thresholds, rot_thresholds, device,
        )
        summary["fallback_pool"] = result
        print(f"match_count_matrix: {result['match_count_matrix']}", flush=True)
        if matched_pose_idx is not None:
            all_matched_pose_idx.append(matched_pose_idx)
    else:
        summary["fallback_pool"] = {"note": "empty -- every arm config carried a per-state receptive_object pose"}
        print("[fk_match] FALLBACK POOL empty -- every arm config had its own receptive_object pose.", flush=True)

    # ---- combined pose coverage across both pools ----
    if all_matched_pose_idx:
        combined_matched = set(np.concatenate(all_matched_pose_idx).tolist())
    else:
        combined_matched = set()
    summary["combined_pose_coverage"] = {
        "n_distinct_poses_matched": len(combined_matched),
        "n_poses_total": n_poses,
        "fraction_of_poses_with_any_match": len(combined_matched) / n_poses if n_poses else 0.0,
    }
    print(
        f"\n=== COMBINED POSE COVERAGE (both pools) ===\n"
        f"{len(combined_matched)}/{n_poses} distinct partial-assembly poses matched"
        f" ({100*len(combined_matched)/n_poses:.1f}%).",
        flush=True,
    )

    out_path = os.path.join(args_cli.out_dir, "fk_match_summary_per_state.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[fk_match] wrote {out_path}", flush=True)

    print(
        "\n=== VERDICT (read, do not trust a single number) ===\n"
        "Same reading rules as the parent script apply to the PER-STATE POOL numbers above (this is"
        " the trustworthy pool for C2/C3 -- no free world-frame constant). The FALLBACK POOL, if"
        " non-empty, is a secondary signal only and inherits the parent script's world-frame caveats.\n"
        "This script does not compute the verdict FOR you.",
        flush=True,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
