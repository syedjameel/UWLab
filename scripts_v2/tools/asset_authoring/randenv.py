"""Environment inspection under a RANDOM policy, for one collider configuration.

Deliberately mirrors the earlier "corrected home posture" inspection so the two are comparable:
9 environments, 150 frames of ZERO actions so the spawn posture is visible and settled, then 750
frames of uniform random actions across all 26 joints. No policy, no checkpoint -- this shows the
ENVIRONMENT, not a behaviour.

Uses the -Play task exactly as that inspection did. Play pins init_difficulty to max_difficulty, so
gravity is on at reset; that is correct here and was called out there too -- the ADR curriculum is
irrelevant to a random policy, and objects falling is the honest picture of the scene.

Prints the spawn posture as the simulator reports it: requested vs actually spawned, per joint.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="DexLift-UR5eDelto-RelJointPos-Lift-Play-v0")
parser.add_argument("--num_envs", type=int, default=9)
parser.add_argument("--zero_frames", type=int, default=150)
parser.add_argument("--random_frames", type=int, default=750)
parser.add_argument("--tag", required=True)
parser.add_argument("--outdir", default="/tmp/randenv")
parser.add_argument("--stack", type=int, default=268435456, help="gpu_collision_stack_size bytes")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--video", action="store_true", help="record an mp4; needs the Omniverse ext cache")
args = parser.parse_args()
args.headless = True
# CAMERAS ONLY WHEN A VIDEO IS ASKED FOR. On DL_A6000 enable_cameras HANGS: that box cannot reach
# the Omniverse extension CDN and blocks until the exts/v2 cache has been warmed from a local copy
# (~5 GB). The zero-action gate needs only joint numbers, so it must be able to run without them.
args.enable_cameras = bool(args.video)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TOTAL = args.zero_frames + args.random_frames
env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
# THE 3.75 GiB CONTACT STACK IS SIZED FOR 2048 ENVS AND MUST BE SHRUNK HERE. Contact-pair volume
# scales with env count; at 9 envs with cameras enabled that reservation exhausts a 16 GB card and
# the run dies with 'CUDA driver error: out of memory' from an allocation as small as a quaternion.
# 256 MiB is ~4x what the SDF hand needed per-env at 2048, so it is generous at 9.
env_cfg.sim.physx.gpu_collision_stack_size = args.stack
print(f"[randenv] contact stack {args.stack / 2**20:.0f} MiB for {args.num_envs} envs", flush=True)
env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
if args.video:
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=os.path.join(args.outdir, args.tag),
        step_trigger=lambda s: s == 0,
        video_length=TOTAL,
        disable_logger=True,
    )

raw = env.unwrapped
robot = raw.scene["robot"]
print(f"[randenv] tag={args.tag}", flush=True)
print(f"[randenv] usd={robot.cfg.spawn.usd_path}", flush=True)
print(f"[randenv] convexhull={os.environ.get('DEXLIFT_CONVEXHULL', 'unset')}", flush=True)

env.reset()
dim = env.action_space.shape[-1]
zero = torch.zeros((args.num_envs, dim), device=raw.device)
for _ in range(args.zero_frames):
    env.step(zero)

# Posture AFTER the zero-action hold: what the simulator actually settled at, which is the number
# that matters -- a requested angle the solver immediately pushes out of is not the spawn pose.
q = robot.data.joint_pos.clone()
default = robot.data.default_joint_pos.clone()
jn = list(robot.data.joint_names)
ARM = [n for n in jn if not n.startswith("rj_dg")]
print(f"\n=== {args.tag}: spawn posture after {args.zero_frames} zero-action frames ===", flush=True)
print(f"{'joint':18s}{'requested':>11s}{'spawned(mean)':>15s}{'max|dev|':>10s}", flush=True)
for n in ARM:
    k = jn.index(n)
    req = float(default[0, k]) * 180 / math.pi
    got = float(q[:, k].mean()) * 180 / math.pi
    dev = float((q[:, k] - default[:, k]).abs().max()) * 180 / math.pi
    print(f"{n:18s}{req:11.2f}{got:15.2f}{dev:10.2f}", flush=True)

hand_idx = [jn.index(n) for n in jn if n.startswith("rj_dg")]
hd = (q[:, hand_idx] - default[:, hand_idx]).abs()
print(f"{'HAND (20 joints)':18s}{'':11s}{'':15s}{float(hd.max()) * 180 / math.pi:10.2f}", flush=True)

for _ in range(args.random_frames):
    env.step(torch.rand((args.num_envs, dim), device=raw.device) * 2.0 - 1.0)

# env.close() IS MANDATORY AND MUST COME BEFORE simulation_app.close(). RecordVideo finalises the
# mp4 in its close path; tearing the app down first destroys the wrapper without flushing and the
# run completes "successfully" having written no file at all -- which is exactly what happened the
# first time this script was run.
env.close()

import glob  # noqa: E402

if args.video:
    vids = glob.glob(os.path.join(args.outdir, args.tag, "*.mp4"))
    print(f"\n[randenv] {args.tag} videos written: {vids}", flush=True)
    if not vids:
        print("RANDENV_NO_VIDEO", flush=True)
        raise SystemExit(1)
print("RANDENV_OK", flush=True)
simulation_app.close()
