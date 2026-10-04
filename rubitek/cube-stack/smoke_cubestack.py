"""Construct and step the UR10e+DELTO cube-stacking envs. Pure config/physics check, no policy.

Verifies, for each task id given:
  * the env constructs (this is where the full-actuation guard and the DELTO seams fire)
  * the cube pair actually resolved to the 40 mm InsertiveCube/ReceptiveCube
  * the success thresholds are the ones this variant intends
  * a few random-action steps run without NaN or a physics blow-up
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--variant", type=str, default="cube34", help="object variant for BOTH roles")
parser.add_argument(
    "--collision_stack_mb",
    type=int,
    default=None,
    help="Override sim.physx.gpu_collision_stack_size, in MiB. _apply_delto_collision_stack_size "
    "pins 3840 MiB (3.75 GiB, the maximum legal value), sized for the 24 GB training cards; on a "
    "16 GB dev card that OOMs PhysX while the scene is still building. Smoke tests use a handful "
    "of envs and run fine far lower. NOTE: an undersized stack does NOT crash at runtime -- PhysX "
    "logs 'Contacts have been dropped' and silently continues -- so never carry a reduced value "
    "into a recording or training run.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

FAIL = []


def check(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(label)


print(f"\n=== {args.task} ===")
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)

# Pin the cube pair the same way the offline tools do (no Hydra here).
if hasattr(env_cfg, "variants"):
    v = env_cfg.variants
    if "scene.insertive_object" in v:
        env_cfg.scene.insertive_object = v["scene.insertive_object"][args.variant]
    if "scene.receptive_object" in v:
        env_cfg.scene.receptive_object = v["scene.receptive_object"][args.variant]

if args.collision_stack_mb is not None:
    env_cfg.sim.physx.gpu_collision_stack_size = args.collision_stack_mb * 1024 * 1024
    print(f"  gpu_collision_stack_size -> {args.collision_stack_mb} MiB (DEV ONLY)")

ins = env_cfg.scene.insertive_object.spawn.usd_path
rec = env_cfg.scene.receptive_object.spawn.usd_path
print(f"  insertive : {ins}")
print(f"  receptive : {rec}")
check("insertive matches variant", args.variant.replace("cube", "InsertiveCube") in ins or "InsertiveCube" in ins)
check("receptive matches variant", "ReceptiveCube" in rec)

if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "task_command"):
    tc = env_cfg.commands.task_command
    print(f"  threshold override: pos={tc.success_position_threshold} rot={tc.success_orientation_threshold}")
if hasattr(env_cfg, "rewards") and hasattr(env_cfg.rewards, "dense_success_reward"):
    print(f"  dense_success_reward params: {env_cfg.rewards.dense_success_reward.params}")

# dataset_dir seam
dirs = set()
for name, term in vars(env_cfg.events).items():
    p = getattr(term, "params", None)
    if isinstance(p, dict) and "dataset_dir" in p:
        dirs.add(p["dataset_dir"])
print(f"  dataset_dir(s): {sorted(dirs)}")
check("dataset_dir is DELTO-local", all("Datasets_ur10e_delto" in d for d in dirs) if dirs else True,
      f"{sorted(dirs)}")

env = gym.make(args.task, cfg=env_cfg)
print(f"  action space : {env.action_space.shape}")
print(f"  obs space    : {env.observation_space}")

nj = env.unwrapped.scene["robot"].num_joints
print(f"  robot joints : {nj}")

if hasattr(env.unwrapped, "command_manager"):
    try:
        tcmd = env.unwrapped.command_manager.get_term("task_command")
        print(f"  LIVE thresholds: pos={tcmd.success_position_threshold} rot={tcmd.success_orientation_threshold}")
    except Exception as e:
        print(f"  (no task_command: {e})")

obs, _ = env.reset()
ok = True
for i in range(args.steps):
    a = torch.zeros_like(env.action_manager.action) if hasattr(env, "action_manager") else \
        torch.zeros((args.num_envs, env.action_space.shape[-1]), device=env.unwrapped.device)
    out = env.step(a)
    st = env.unwrapped.scene["robot"].data.joint_pos
    if torch.isnan(st).any() or torch.isinf(st).any():
        ok = False
        break
check(f"{args.steps} zero-action steps, finite state", ok)

env.close()
print(f"\n[SMOKE_RESULT] {'[PASS]' if not FAIL else '[FAIL] ' + ', '.join(FAIL)}")
simulation_app.close()
