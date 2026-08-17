"""C3: DEXLIFT_PARTIAL_ASSEMBLY=1 against DexLift TableLeg-Reorient. Constructs, resets, steps,
then MEASURES:
  1. receptive_object (fixture) root pose in env frame vs RECEPTIVE_POSE_RANGE.
  2. leg pose relative to the fixture, decomposed and checked against the LIVE partial_assemblies.pt
     dataset (loaded independently, read-only) -- the compose/decompose round trip.
  3. the goal command converted to world frame vs the leg's actual spawn pose -- expect ~zero error.

dataset_dir overridden to the locally-transferred copy (DL_A6000 -> local -> this box), since the
module's cloud default 404s for this exact pair (confirmed before spending any GPU time)."""

from __future__ import annotations

import argparse
import os

os.environ["DEXLIFT_REF_RESET"] = "1"
os.environ["DEXLIFT_REF_ACTUATORS"] = "1"
os.environ["DEXLIFT_REF_HAND_ACT"] = "1"
os.environ["DEXLIFT_REF_ARM_ACT"] = "0"
os.environ["DEXLIFT_POSE_TILT"] = "0.3"
os.environ["DEXLIFT_PARTIAL_ASSEMBLY"] = "1"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0"
NUM_ENVS = 32
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=NUM_ENVS)
env_cfg.sim.physx.gpu_collision_stack_size = 2**24

DATASET_DIR = "/root/uwlab_ur5edelto/Datasets_ur5e_delto/OmniReset"
print(f"[dataset_dir override] using local transferred copy: {DATASET_DIR}", flush=True)
print(f"[dataset_dir default was] {env_cfg.events.reset_object.params.get('dataset_dir')}", flush=True)
env_cfg.events.reset_object.params["dataset_dir"] = DATASET_DIR

print(f"[verify] events.reset_object.func = {env_cfg.events.reset_object.func.__name__}", flush=True)
has_receptive = hasattr(env_cfg.scene, "receptive_object") and env_cfg.scene.receptive_object is not None
print(f"[verify] scene.receptive_object present = {has_receptive}", flush=True)
print(f"[verify] commands.object_pose class_type = {env_cfg.commands.object_pose.class_type}", flush=True)

env = gym.make(TASK, cfg=env_cfg)
raw = env.unwrapped
print("ENV_CONSTRUCTED_OK", flush=True)

env.reset()
print("RESET_OK", flush=True)

zero = torch.zeros((NUM_ENVS, env.action_space.shape[-1]), device=raw.device)
for i in range(5):
    env.step(zero)
print("C3_STEPPED_5_OK", flush=True)

# -- 1. receptive_object (fixture) pose -----------------------------------------------------
receptive = raw.scene["receptive_object"]
robot = raw.scene["robot"]
obj = raw.scene["object"]
env_origins = raw.scene.env_origins

fixture_pos_w = receptive.data.root_pos_w
fixture_quat_w = receptive.data.root_quat_w
fixture_pos_env = fixture_pos_w - env_origins
print(f"FIXTURE_Z_MIN={fixture_pos_env[:, 2].min().item():.6f}", flush=True)
print(f"FIXTURE_Z_MAX={fixture_pos_env[:, 2].max().item():.6f}", flush=True)
print(f"FIXTURE_X_MIN={fixture_pos_env[:, 0].min().item():.6f} FIXTURE_X_MAX={fixture_pos_env[:, 0].max().item():.6f}", flush=True)
print(f"FIXTURE_Y_MIN={fixture_pos_env[:, 1].min().item():.6f} FIXTURE_Y_MAX={fixture_pos_env[:, 1].max().item():.6f}", flush=True)
x_ok = bool((fixture_pos_env[:, 0] >= 0.35 - 1e-4).all() and (fixture_pos_env[:, 0] <= 0.60 + 1e-4).all())
y_ok = bool((fixture_pos_env[:, 1] >= -0.20 - 1e-4).all() and (fixture_pos_env[:, 1] <= 0.20 + 1e-4).all())
z_ok = bool((fixture_pos_env[:, 2] - 0.019625).abs().max().item() < 1e-4)
print(f"FIXTURE_RANGE_CHECK x_ok={x_ok} y_ok={y_ok} z_ok={z_ok}", flush=True)

# -- 2. leg pose relative to the fixture: compose/decompose against the LIVE dataset --------
object_pos_w = obj.data.root_pos_w
object_quat_w = obj.data.root_quat_w
rel_pos_live, rel_quat_live = math_utils.subtract_frame_transforms(
    fixture_pos_w, fixture_quat_w, object_pos_w, object_quat_w
)

dataset_path = f"{DATASET_DIR}/Resets/OneLegInsertionFixture__SquareTableLeg200mmDecomp/partial_assemblies.pt"
data = torch.load(dataset_path, map_location=raw.device)
stored_rel_pos = data["relative_position"].to(raw.device, dtype=torch.float32)
print(f"[dataset] loaded {len(stored_rel_pos)} stored relative positions from {dataset_path}", flush=True)

# For each env, nearest stored entry by position distance (pose_range_b={} means no extra jitter,
# so this should land within float32 round-trip precision of an EXACT stored entry).
dists = torch.cdist(rel_pos_live, stored_rel_pos)  # (NUM_ENVS, N_stored)
min_dist_per_env, nearest_idx = dists.min(dim=-1)
print(f"COMPOSE_DECOMPOSE_MAX_DIST={min_dist_per_env.max().item():.8f}", flush=True)
print(f"COMPOSE_DECOMPOSE_MEAN_DIST={min_dist_per_env.mean().item():.8f}", flush=True)
print(f"COMPOSE_DECOMPOSE_ALL_DISTS={[round(v, 8) for v in min_dist_per_env.tolist()]}", flush=True)

# -- 3. HEADLINE: goal command (pose_command_b, robot-root frame) vs the leg's actual world pose
command = raw.command_manager.get_command("object_pose")
cmd_pos_b = command[:, 0:3]
cmd_quat_b = command[:, 3:7]
robot_pos_w = robot.data.root_pos_w
robot_quat_w = robot.data.root_quat_w
cmd_pos_w, cmd_quat_w = math_utils.combine_frame_transforms(robot_pos_w, robot_quat_w, cmd_pos_b, cmd_quat_b)

pos_err = torch.linalg.vector_norm(cmd_pos_w - object_pos_w, dim=-1)
quat_err = math_utils.quat_error_magnitude(cmd_quat_w, object_quat_w)
print(f"GOAL_POS_ERROR_MAX={pos_err.max().item():.8f}", flush=True)
print(f"GOAL_POS_ERROR_MEAN={pos_err.mean().item():.8f}", flush=True)
print(f"GOAL_ORIENT_ERROR_MAX_RAD={quat_err.max().item():.8f}", flush=True)
print(f"GOAL_ORIENT_ERROR_MEAN_RAD={quat_err.mean().item():.8f}", flush=True)

print("C3_MEASURE_DONE", flush=True)

env.close()
simulation_app.close()
