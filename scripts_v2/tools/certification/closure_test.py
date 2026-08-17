"""How far can the hand actually close, per collider configuration?

Run A's signature at epoch 409 is good_finger_contact 1.13 -- at or above the reference's own level --
with object_upward_motion on the floor at 0.0046. It reaches the object and cannot carry it. The
obvious mechanism is that self-collisions physically cap the closure: our hand has TWO SHELL BODIES
the reference does not have at all (rl_dg_base, rl_dg_palm), and palm-vs-distal-phalanx pairs are
deliberately left LIVE in ur5e_delto_hullfix.usd because a finger curling into the palm is a real
constraint. If those shells stand further out than the reference's single mount, the fingers hit them
earlier in the curl and the grasp is capped.

This tests that directly and needs no geometry at all: command maximum closure for a few hundred
steps with NO object in the way of the argument, and read back how far each joint actually got.
The action is RelativeJointPositionAction at scale 0.1 relative to MEASURED position, so a sustained
+1 drives every finger joint to its limit; whatever it settles at IS the reachable closure.

Interpretation:
  configurations agree      -> self-collisions do NOT cap closure, and Run A's stall is something else
  self-coll ON closes less  -> the cap is real, and the fix is the palm/base shells, not the policy
Report is per JOINT LEVEL (_1.._4), because the levels tell different stories: _1 is the spread/root
joint and _2.._4 are the curl.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="DexLift-UR5eDelto-RelJointPos-Lift-Play-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--settle", type=int, default=60, help="zero-action frames before commanding")
parser.add_argument("--close", type=int, default=250, help="frames of maximum closure command")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--tag", required=True)
parser.add_argument(
    "--levels",
    default="1234",
    help="which joint LEVELS to drive. '1234' drives everything including the _1 SPREAD joints, which"
    " splays the fingers sideways INTO each other -- that is the test jamming them together, not the"
    " curl being blocked, and it inflates any finger-vs-finger effect. '234' drives curl only and is"
    " the honest measurement of reachable closure.",
)
parser.add_argument("--stack", type=int, default=268435456)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
env_cfg.sim.physx.gpu_collision_stack_size = args.stack
env_cfg.seed = args.seed
env = gym.make(args.task, cfg=env_cfg)
raw = env.unwrapped
robot = raw.scene["robot"]
print(f"[close] tag={args.tag} usd={robot.cfg.spawn.usd_path}", flush=True)

torch.manual_seed(args.seed)
env.reset()
dim = env.action_space.shape[-1]
jn = list(robot.data.joint_names)
hand = [i for i, n in enumerate(jn) if n.startswith("rj_dg")]
# The action vector is 6 arm + 20 hand in joint order; drive ONLY the hand and hold the arm, so the
# arm does not wander into the table and contaminate the reading.
act = torch.zeros((args.num_envs, dim), device=raw.device)
lim = robot.data.joint_limits  # (envs, joints, 2)

for _ in range(args.settle):
    env.step(torch.zeros_like(act))

# Which action indices are the hand's? The action space is ordered like the joints for this action
# term, so the hand joint indices index the action vector too. Assert the count rather than trust it.
assert len(hand) == 20, f"expected 20 hand joints, got {len(hand)}"
driven = [i for i in hand if jn[i].rsplit("_", 1)[-1] in set(args.levels)]
print(f"[close] driving {len(driven)}/20 hand joints (levels {args.levels}); "
      f"undriven levels are commanded ZERO, not held at limit", flush=True)
act[:, driven] = 1.0
for _ in range(args.close):
    env.step(act)

q = robot.data.joint_pos.clone()
print(f"\n=== {args.tag}: reachable closure after {args.close} frames at max command ===", flush=True)
print(f"{'level':10s}{'n':>4s}{'mean deg':>11s}{'min deg':>10s}{'max deg':>10s}{'% of limit':>12s}", flush=True)
for lvl in ("1", "2", "3", "4"):
    idx = [i for i, n in enumerate(jn) if n.startswith("rj_dg") and n.endswith(f"_{lvl}")]
    if not idx:
        continue
    got = q[:, idx] * 180.0 / math.pi
    # Upper limit is the closed direction for these joints; report how much of it was reached.
    up = lim[:, idx, 1] * 180.0 / math.pi
    frac = (got / up.clamp(min=1e-6)).clamp(-2, 2)
    print(f"_{lvl:<9s}{len(idx):4d}{float(got.mean()):11.2f}{float(got.min()):10.2f}"
          f"{float(got.max()):10.2f}{100.0 * float(frac.mean()):11.1f}%", flush=True)

allh = q[:, hand] * 180.0 / math.pi
allu = lim[:, hand, 1] * 180.0 / math.pi
print(f"\nCLOSURE {args.tag} mean_deg={float(allh.mean()):.3f} "
      f"frac_of_limit={float((allh / allu.clamp(min=1e-6)).mean()):.4f}", flush=True)
print("CLOSE_OK", flush=True)
env.close()
simulation_app.close()
