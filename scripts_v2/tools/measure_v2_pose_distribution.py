"""T2 -- DexReset v2, measurement only (bead: first v2 experiment), REVISION 2.

Team-lead redirect (see run history): DEXLIFT_GOAL_VERTICAL_PROB is NOT part of production
staging (confirmed by T1's banner -- no "GOAL VERTICAL" line under plain production staging).
The production goal mechanism is the EPISODE MIXTURE (DEXLIFT_EPISODE_MIXTURE=1), which draws
each episode's goal from one of three branches (classic / low_goal / partial_assembly) per
mdp/episode_mixture.py's own EPISODE_KIND_* buffer. This revision:

  * runs under PLAIN production staging (DEXLIFT_REF_RESET=1 DEXLIFT_REF_ACTUATORS=1
    DEXLIFT_REF_HAND_ACT=1 DEXLIFT_REF_ARM_ACT=0 DEXLIFT_POSE_TILT=0.3
    DEXLIFT_EPISODE_MIXTURE=1), nothing forced;
  * reads the per-env episode kind off mdp.episode_mixture._get_episode_kind_buffer(env) after
    reset and splits every reported distribution by branch (classic/low_goal/partial_assembly);
  * reports goal tip_z in BOTH root frame and tip frame (tip = root - 0.106203 m along the
    leg's local -X, i.e. WORK_SURFACE_Z-referenced height), so the conversion is auditable;
  * reports orientation angle from tip-down using the AXIS-ONLY definition (angle between the
    rotated local tip axis and world -Z), which ignores spin/roll about the insertion axis --
    matches the orientation_tracking reward, which sums |e_x|+|e_y| and drops e_z;
  * also reconfirms the leg's own spawn/reset tip_z distribution (F37) from this same run, so
    F37 and the goal-band numbers come from one configuration.

Per RESET_SPEC_V2.md R5: every reported number states DEXLIFT_POSE_TILT and the episode-mixture
staging, READ BACK from the live [dexlift] banner (re-printed to this script's own log), never
assumed from the env vars this script exports. Measurement only; no sampling range, reward, or
env config is changed.
"""
import argparse
import hashlib
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--rounds", type=int, default=2, help="env.reset() calls; total samples = num_envs*rounds")
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply, combine_frame_transforms

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEG_METADATA_CANDIDATES = [
    os.path.join(
        REPO_ROOT,
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf/metadata.yaml",
    ),
    os.path.join(
        REPO_ROOT,
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmDecomp/metadata.yaml",
    ),
]
leg_metadata_path = next((p for p in LEG_METADATA_CANDIDATES if os.path.isfile(p)), None)
if leg_metadata_path is None:
    raise FileNotFoundError(f"none of the leg metadata.yaml candidates exist: {LEG_METADATA_CANDIDATES}")
with open(leg_metadata_path) as f:
    leg_metadata = yaml.safe_load(f)
ASSEMBLED_OFFSET_POS = leg_metadata["assembled_offset"]["pos"]  # local-frame tip position
TIP_ROOT_OFFSET_M = -ASSEMBLED_OFFSET_POS[0]  # 0.106203 m, root sits this far ABOVE tip when tip-down
print(f"[T2] leg metadata used: {leg_metadata_path}")
print(f"[T2] assembled_offset.pos (local tip position) = {ASSEMBLED_OFFSET_POS}")
print(f"[T2] tip-root offset (root above tip, tip-down) = {TIP_ROOT_OFFSET_M:.6f} m")

device = args_cli.device
env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs, use_fabric=True)
env = gym.make(args_cli.task, cfg=env_cfg)
unwrapped = env.unwrapped

from uwlab_tasks.manager_based.manipulation.dexlift.mdp import episode_mixture as em

KIND_NAMES = {
    em.EPISODE_KIND_CLASSIC: "classic",
    em.EPISODE_KIND_LOW_GOAL: "low_goal",
    em.EPISODE_KIND_PARTIAL_ASSEMBLY: "partial_assembly",
}


def quat_apply_local_tip_axis(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """World-frame unit vector along the local tip axis (local -X), rotated by quat_wxyz."""
    local_axis = torch.tensor([-1.0, 0.0, 0.0], device=quat_wxyz.device, dtype=quat_wxyz.dtype)
    local_axis = local_axis.unsqueeze(0).expand(quat_wxyz.shape[0], -1)
    v = quat_apply(quat_wxyz, local_axis)
    return v / v.norm(dim=-1, keepdim=True)


TARGET_DOWN = torch.tensor([0.0, 0.0, -1.0], device=device)


def axis_tilt_from_tipdown_deg(quat_wxyz: torch.Tensor) -> torch.Tensor:
    world_tip_dir = quat_apply_local_tip_axis(quat_wxyz)
    dot = (world_tip_dir * TARGET_DOWN.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


def tip_from_root(root_pos: torch.Tensor, root_quat_wxyz: torch.Tensor) -> torch.Tensor:
    offset = torch.tensor(ASSEMBLED_OFFSET_POS, device=root_pos.device, dtype=root_pos.dtype)
    offset = offset.unsqueeze(0).expand(root_pos.shape[0], -1)
    return root_pos + quat_apply(root_quat_wxyz, offset)


object_root_pos, object_root_quat = [], []
goal_root_pos_w, goal_root_quat_w = [], []
kind_all = []

cmd_term = unwrapped.command_manager.get_term("object_pose")
robot = unwrapped.scene["robot"]
obj = unwrapped.scene["object"]

for r in range(args_cli.rounds):
    env.reset()
    object_root_pos.append(obj.data.root_pos_w.detach().clone())
    object_root_quat.append(obj.data.root_quat_w.detach().clone())

    robot_pos_w = robot.data.root_pos_w.detach().clone()
    robot_quat_w = robot.data.root_quat_w.detach().clone()
    goal_pos_w, goal_quat_w = combine_frame_transforms(
        robot_pos_w,
        robot_quat_w,
        cmd_term.pose_command_b[:, :3].detach().clone(),
        cmd_term.pose_command_b[:, 3:7].detach().clone(),
    )
    goal_root_pos_w.append(goal_pos_w)
    goal_root_quat_w.append(goal_quat_w)

    kind = em._get_episode_kind_buffer(unwrapped).detach().clone()
    kind_all.append(kind)
    counts = {KIND_NAMES[k]: int((kind == k).sum()) for k in KIND_NAMES}
    print(f"[T2] round {r} done, {obj.data.root_pos_w.shape[0]} envs, kind counts={counts}")

object_root_pos = torch.cat(object_root_pos, dim=0)
object_root_quat = torch.cat(object_root_quat, dim=0)
goal_root_pos_w = torch.cat(goal_root_pos_w, dim=0)
goal_root_quat_w = torch.cat(goal_root_quat_w, dim=0)
kind_all = torch.cat(kind_all, dim=0)

n = object_root_pos.shape[0]
print(f"[T2] total samples n={n}")

object_tip_pos = tip_from_root(object_root_pos, object_root_quat)
goal_tip_pos = tip_from_root(goal_root_pos_w, goal_root_quat_w)

object_axis_tilt_deg = axis_tilt_from_tipdown_deg(object_root_quat)
goal_axis_tilt_deg = axis_tilt_from_tipdown_deg(goal_root_quat_w)

arrays = {
    "object_root_pos": object_root_pos.cpu().numpy(),
    "object_root_quat_wxyz": object_root_quat.cpu().numpy(),
    "object_tip_pos": object_tip_pos.cpu().numpy(),
    "goal_root_pos_w": goal_root_pos_w.cpu().numpy(),
    "goal_root_quat_w_wxyz": goal_root_quat_w.cpu().numpy(),
    "goal_tip_pos": goal_tip_pos.cpu().numpy(),
    "object_axis_tilt_from_tipdown_deg": object_axis_tilt_deg.cpu().numpy(),
    "goal_axis_tilt_from_tipdown_deg": goal_axis_tilt_deg.cpu().numpy(),
    "episode_kind": kind_all.cpu().numpy(),
}

np.savez(args_cli.out, **arrays)
print(f"[T2] wrote {args_cli.out}")

sha256 = hashlib.sha256()
with open(args_cli.out, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        sha256.update(chunk)
print(f"[T2] sha256={sha256.hexdigest()}")


def percentiles(x: np.ndarray):
    if x.size == 0:
        return None
    ps = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    vals = np.percentile(x, ps)
    return dict(zip(["min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max"], vals.tolist()))


def histogram(x: np.ndarray, bins=20):
    if x.size == 0:
        return None
    counts, edges = np.histogram(x, bins=bins)
    return {"counts": counts.tolist(), "edges": edges.tolist()}


report = {"n_samples": n}

# object (leg) own spawn distribution -- F37 reconfirmation, not split by branch (spawn is
# independent of goal branch)
for name, col in [
    ("object_root_z", arrays["object_root_pos"][:, 2]),
    ("object_tip_z", arrays["object_tip_pos"][:, 2]),
    ("object_axis_tilt_from_tipdown_deg", arrays["object_axis_tilt_from_tipdown_deg"]),
]:
    report[name] = {"percentiles": percentiles(col), "histogram": histogram(col)}

# goal distribution, split by episode-mixture branch
kind_np = arrays["episode_kind"]
report["goal_by_branch"] = {}
for k, name in KIND_NAMES.items():
    mask = kind_np == k
    n_branch = int(mask.sum())
    branch_report = {"n": n_branch}
    for col_name, col in [
        ("goal_root_z", arrays["goal_root_pos_w"][mask, 2]),
        ("goal_tip_z", arrays["goal_tip_pos"][mask, 2]),
        ("goal_axis_tilt_from_tipdown_deg", arrays["goal_axis_tilt_from_tipdown_deg"][mask]),
    ]:
        branch_report[col_name] = {"percentiles": percentiles(col), "histogram": histogram(col, bins=10)}
    report["goal_by_branch"][name] = branch_report

report["tip_root_offset_m"] = TIP_ROOT_OFFSET_M
report["assembled_offset_pos_local"] = ASSEMBLED_OFFSET_POS
report["leg_metadata_path"] = leg_metadata_path
report["WORK_SURFACE_Z"] = 0.0  # module constant, dexlift_ur5e_delto_env_cfg.py

print("[T2_REPORT_JSON]")
print(json.dumps(report, indent=2))
print("[T2_REPORT_JSON_END]")

env.close()
simulation_app.close()
