"""E1: measure GOAL vs OBJECT pose IMMEDIATELY after env.reset(), ZERO sim steps in between.
This is the only moment the property "goal == spawn pose" (GoalAtSpawnPoseCommand) is actually
claimed to hold. If C3's 5-step-settled errors were physics settle-drift, this must be near float
noise. If not, that's a genuine bug in the command wiring, not a measurement artifact."""

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

import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import gymnasium as gym  # noqa: E402

TASK = "DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0"
NUM_ENVS = 32
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=NUM_ENVS)
env_cfg.sim.physx.gpu_collision_stack_size = 2**24

DATASET_DIR = "/root/uwlab_ur5edelto/Datasets_ur5e_delto/OmniReset"
env_cfg.events.reset_object.params["dataset_dir"] = DATASET_DIR

env = gym.make(TASK, cfg=env_cfg)
raw = env.unwrapped
print("ENV_CONSTRUCTED_OK", flush=True)

env.reset()
print("RESET_OK", flush=True)
# ZERO steps. Measure NOW, before physics has a chance to move anything.

receptive = raw.scene["receptive_object"]
robot = raw.scene["robot"]
obj = raw.scene["object"]

object_pos_w = obj.data.root_pos_w
object_quat_w = obj.data.root_quat_w
robot_pos_w = robot.data.root_pos_w
robot_quat_w = robot.data.root_quat_w

command = raw.command_manager.get_command("object_pose")
cmd_pos_b = command[:, 0:3]
cmd_quat_b = command[:, 3:7]
cmd_pos_w, cmd_quat_w = math_utils.combine_frame_transforms(robot_pos_w, robot_quat_w, cmd_pos_b, cmd_quat_b)

pos_err = torch.linalg.vector_norm(cmd_pos_w - object_pos_w, dim=-1)
quat_err = math_utils.quat_error_magnitude(cmd_quat_w, object_quat_w)

print(f"E1_T0_GOAL_POS_ERROR_MAX={pos_err.max().item():.10f}", flush=True)
print(f"E1_T0_GOAL_POS_ERROR_MEAN={pos_err.mean().item():.10f}", flush=True)
print(f"E1_T0_GOAL_ORIENT_ERROR_MAX_RAD={quat_err.max().item():.10f}", flush=True)
print(f"E1_T0_GOAL_ORIENT_ERROR_MEAN_RAD={quat_err.mean().item():.10f}", flush=True)
print(f"E1_T0_GOAL_POS_ERROR_ALL={[round(v, 8) for v in pos_err.tolist()]}", flush=True)
print(f"E1_T0_GOAL_ORIENT_ERROR_ALL={[round(v, 8) for v in quat_err.tolist()]}", flush=True)

print("E1_MEASURE_DONE", flush=True)

env.close()
simulation_app.close()
