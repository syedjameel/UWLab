"""C1 baseline: construct DexLift TableLeg-Reorient with the reference plant, both new toggles
UNSET. Control run -- if this fails, the transfer/box is the problem, not our modifications."""

from __future__ import annotations

import argparse
import os

os.environ["DEXLIFT_REF_RESET"] = "1"
os.environ["DEXLIFT_REF_ACTUATORS"] = "1"
os.environ["DEXLIFT_REF_HAND_ACT"] = "1"
os.environ["DEXLIFT_REF_ARM_ACT"] = "0"
os.environ["DEXLIFT_POSE_TILT"] = "0.3"
# both new toggles UNSET for this run -- the control

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

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0"
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=32)
env_cfg.sim.physx.gpu_collision_stack_size = 2**24

_hand_act = env_cfg.scene.robot.actuators["hand"]
_effort_vals = sorted(set(_hand_act.effort_limit_sim.values())) if isinstance(
    _hand_act.effort_limit_sim, dict
) else [_hand_act.effort_limit_sim]
_vel_vals = sorted(set(_hand_act.velocity_limit_sim.values())) if isinstance(
    _hand_act.velocity_limit_sim, dict
) else [_hand_act.velocity_limit_sim]
print(f"[verify] hand effort_limit_sim (distinct values): {_effort_vals}", flush=True)
print(f"[verify] hand velocity_limit_sim (distinct values): {_vel_vals}", flush=True)
print(f"[verify] events.reset_robot_joints.position_range = {env_cfg.events.reset_robot_joints.params['position_range']}", flush=True)
print(f"[verify] events.reset_finger_root_joints.position_range = {env_cfg.events.reset_finger_root_joints.params['position_range']}", flush=True)
print(f"[verify] events.reset_robot_elbow_joint.position_range = {env_cfg.events.reset_robot_elbow_joint.params['position_range']}", flush=True)

_reset_object_func = env_cfg.events.reset_object.func
print(f"[verify] events.reset_object.func = {_reset_object_func.__name__}", flush=True)

plant_ok = _effort_vals == [30.0] and _vel_vals == [10000.0]
print(f"PLANT_CHECK={'OK' if plant_ok else 'MISMATCH'} effort={_effort_vals} velocity={_vel_vals}", flush=True)

env = gym.make(TASK, cfg=env_cfg)
raw = env.unwrapped
print("ENV_CONSTRUCTED_OK", flush=True)

env.reset()
print("RESET_OK", flush=True)

zero = torch.zeros((32, env.action_space.shape[-1]), device=raw.device)
for i in range(5):
    env.step(zero)
print("C1_STEPPED_5_OK", flush=True)

env.close()
simulation_app.close()
