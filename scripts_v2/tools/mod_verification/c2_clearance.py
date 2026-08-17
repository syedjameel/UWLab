"""C2: DEXLIFT_SPAWN_CLEARANCE=1 against DexLift TableLeg-Reorient. Constructs, resets, steps,
then MEASURES the leg's actual achieved clearance above the table from its LIVE post-reset pose,
using the same corrected (origin-offset-aware) formula shipped in dexlift/mdp/spawn.py."""

from __future__ import annotations

import argparse
import os

os.environ["DEXLIFT_REF_RESET"] = "1"
os.environ["DEXLIFT_REF_ACTUATORS"] = "1"
os.environ["DEXLIFT_REF_HAND_ACT"] = "1"
os.environ["DEXLIFT_REF_ARM_ACT"] = "0"
os.environ["DEXLIFT_POSE_TILT"] = "0.3"
os.environ["DEXLIFT_SPAWN_CLEARANCE"] = "1"

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

_reset_object_func = env_cfg.events.reset_object.func
_reset_object_params = env_cfg.events.reset_object.params
_is_clearance_term = _reset_object_func.__name__ == "reset_object_pose_with_clearance"
print(f"[verify] events.reset_object.func = {_reset_object_func.__name__}", flush=True)
if _is_clearance_term:
    print(
        f"[verify] clearance_range={_reset_object_params['clearance_range']}"
        f" half_extents={_reset_object_params['half_extents']}"
        f" surface_z={_reset_object_params['surface_z']}"
        f" local_centre_offset={_reset_object_params.get('local_centre_offset')}",
        flush=True,
    )
else:
    print("SIXTH_VERIFY_LINE_FAILED: reset_object.func is not reset_object_pose_with_clearance", flush=True)

env = gym.make(TASK, cfg=env_cfg)
raw = env.unwrapped
print("ENV_CONSTRUCTED_OK", flush=True)

env.reset()
print("RESET_OK", flush=True)

zero = torch.zeros((NUM_ENVS, env.action_space.shape[-1]), device=raw.device)
for i in range(5):
    env.step(zero)
print("C2_STEPPED_5_OK", flush=True)

# -- MEASURE actual clearance from the LIVE post-reset pose ------------------------------------
obj = raw.scene["object"]
env_origins = raw.scene.env_origins
object_pos_w = obj.data.root_pos_w
object_quat_w = obj.data.root_quat_w
object_z_env_frame = object_pos_w[:, 2] - env_origins[:, 2]

HALF_EXTENTS = torch.tensor([0.100, 0.015, 0.015], device=raw.device)
CENTRE = torch.tensor([-0.0062028, 0.0, 0.0], device=raw.device)

rotation_matrix = math_utils.matrix_from_quat(object_quat_w)
row2 = rotation_matrix[:, 2, :]
lowest_offset = (row2.abs() * HALF_EXTENTS).sum(dim=-1) - (row2 * CENTRE).sum(dim=-1)

surface_z = 0.0
achieved_clearance = object_z_env_frame - lowest_offset - surface_z

print(f"CLEARANCE_MIN={achieved_clearance.min().item():.6f}", flush=True)
print(f"CLEARANCE_P50={achieved_clearance.median().item():.6f}", flush=True)
print(f"CLEARANCE_MAX={achieved_clearance.max().item():.6f}", flush=True)
below = (achieved_clearance < 0.01 - 1e-4).sum().item()
print(f"CLEARANCE_BELOW_1CM_COUNT={below} of {NUM_ENVS}", flush=True)
print(f"CLEARANCE_ALL_VALUES={[round(v, 5) for v in achieved_clearance.tolist()]}", flush=True)

print("C2_MEASURE_DONE", flush=True)

env.close()
simulation_app.close()
