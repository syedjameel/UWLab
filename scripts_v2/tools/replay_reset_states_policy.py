# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Write reset states back into the plant that produced them, then let the trained policy act.

THE QUESTION. A reset state stores every joint position and velocity plus every rigid-body pose
and velocity, and ``MultiResetManager._reset_to`` writes all of it AND sets the PD targets. That
looks like it should be a perfect restore, so why does a replayed state not simply continue? This
script answers it by measurement rather than argument, and it is also the closed-loop test that
every previous render of these states could only bound: an open-loop settle asks whether a state
survives with nobody driving, which is not the question anyone cares about.

THE DESIGN IS A 2x2, one variable per axis, same states in every cell:

                        no policy (zero action)     policy acting
    reference hand              A                        B
    identified hand             C                        D

  * REFERENCE vs IDENTIFIED is the hand actuator. The states were GENERATED under the reference
    block (effort 30 N*m, velocity 10000 rad/s) because that is what the checkpoint trained under;
    the OmniReset scene that will consume them builds the shared identified block instead
    (0.06-0.17 N*m, 3.0 rad/s). Cells A/B vs C/D price that seam.
  * ZERO ACTION vs POLICY is whether anything is closing the loop. A/C are the open-loop settle the
    earlier renders did; B/D are the real question.

Read as a 2x2 it separates two explanations that a single number confounds: if C collapses and D
does not, the weak hand is survivable because the policy compensates; if both collapse, it is not.

THE METRIC IS SUPPORT-INDEPENDENT, and this is deliberate. The previous render scored a state by
how far the OBJECT moved in the world, which a leg lying on the table satisfies whether or not the
hand is holding it -- every state that scored "held" there was simply resting on the surface. Here
the primary number is the object's pose IN THE PALM'S FRAME. If the hand keeps hold, that is
constant no matter where the arm carries the leg; if the leg is dropped, it diverges immediately
even if the leg lands on the table and stops moving. World displacement is reported beside it, not
instead of it.

WHAT THIS CANNOT SEE, stated so the numbers are not over-read:
  * A state is written mid-episode, so any observation term with history (previous action, and the
    running observation normalisation the checkpoint carries) is stale for a few steps. The policy
    is being asked to recover from a teleport, which is slightly harder than the reset it was
    trained on. --warmup-steps discards the first N steps from the drift baseline for this reason.
  * PhysX on GPU is unseeded here, and re-running the same state gives visibly different numbers.
    Every figure is over --num-states states; treat single-state values as draws.
  * Friction and mass are re-randomised at each env reset, so a replayed state does not necessarily
    get the material parameters it was recorded with. That is one of the reasons a "perfect"
    restore is not perfect, and it is measured rather than removed.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--reset-states", type=Path, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--agent-yaml", type=str, default=None)
parser.add_argument("--task", default="DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-Play-v0")
parser.add_argument("--num-states", type=int, default=32, help="states replayed in parallel = num_envs")
parser.add_argument("--rollout-steps", type=int, default=120, help="control steps after the write (60 Hz)")
parser.add_argument("--warmup-steps", type=int, default=5,
                    help="steps discarded before the in-hand baseline is taken, so the policy's"
                         " recovery from a mid-episode teleport is not scored as slippage.")
parser.add_argument("--policy", choices=["on", "off"], required=True,
                    help="'off' issues zero actions -- the open-loop settle the earlier renders did.")
parser.add_argument("--hand", choices=["reference", "identified"], required=True,
                    help="which DELTO actuator block to BUILD. Sets DEXLIFT_REF_HAND_ACT before the"
                         " env is constructed and then VERIFIES the built limits, because a silently"
                         " ineffective env var yields a plausible wrong number rather than an error.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--tag", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# THE PLANT VARS MUST BE SET BEFORE parse_env_cfg -- they are read at cfg construction.
os.environ["DEXLIFT_REF_RESET"] = "1"
os.environ["DEXLIFT_REF_ACTUATORS"] = "1"
os.environ["DEXLIFT_REF_ARM_ACT"] = "0"
os.environ["DEXLIFT_REF_HAND_ACT"] = "1" if args.hand == "reference" else "0"
os.environ["DEXLIFT_POSE_TILT"] = "0.3"
os.environ.setdefault("DEXLIFT_SPAWN_CLEARANCE", "1")

import torch  # noqa: E402
import yaml  # noqa: E402

raw = torch.load(args.reset_states, map_location="cpu")
state = raw["initial_state"]
n_total = len(state["articulation"]["robot"]["joint_position"])
g = torch.Generator().manual_seed(args.seed)
sel = torch.randperm(n_total, generator=g)[: args.num_states].tolist()
print(f"[replay] {n_total} states in bank; replaying {len(sel)} (seed {args.seed})", flush=True)

app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

cfg = parse_env_cfg(args.task, device=args.device, num_envs=len(sel))
cfg.seed = args.seed
cfg.sim.physx.gpu_collision_stack_size = 512 * 1024 * 1024
env = gym.make(args.task, cfg=cfg).unwrapped
robot = env.scene["robot"]
obj = env.scene["object"]  # dexlift's own name; the bank stores it as insertive_object

# VERIFY THE PLANT THAT WAS BUILT, never the env var that was requested. A silently ineffective
# DEXLIFT_REF_HAND_ACT is the exact failure that once made a generator run at 2.69 percent
# acceptance look like a policy problem.
hand_ids, _ = robot.find_joints([r"rj_dg_[1-5]_[1-4]"], preserve_order=False)
eff = robot.data.joint_effort_limits[0, hand_ids]
vel = robot.data.joint_velocity_limits[0, hand_ids]
print(f"[replay] hand AS BUILT: effort {eff.min():.3f}-{eff.max():.3f} N*m, "
      f"velocity {vel.min():.1f}-{vel.max():.1f} rad/s  (asked for {args.hand})", flush=True)
is_reference = float(eff.max()) > 1.0
if (args.hand == "reference") != is_reference:
    raise SystemExit(f"[replay] REFUSING: asked for the {args.hand} hand and the env built the other one")

def unwrap(o):
    """RlGamesVecEnvWrapper returns {"obs": tensor}; reset may also return (obs, info)."""
    if isinstance(o, tuple):
        o = o[0]
    if isinstance(o, dict):
        o = o["obs"]
    return o


player = None
if args.policy == "on":
    agent_yaml = args.agent_yaml or os.path.join(Path(args.checkpoint).parent.parent, "params", "agent.yaml")
    agent_cfg = yaml.safe_load(open(agent_yaml))
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_act = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    wrapped = RlGamesVecEnvWrapper(env, args.device, clip_obs, clip_act,
                                   agent_cfg["params"]["env"].get("obs_groups"),
                                   agent_cfg["params"]["env"].get("concate_obs_groups", True))
    vecenv.register("IsaacRlgWrapper",
                    lambda name, num_actors, **kw: RlGamesGpuEnv(name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: wrapped})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args.checkpoint
    agent_cfg["params"]["config"]["num_actors"] = env.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(args.checkpoint)
    player.reset()
    player.has_batch_dimension = True
    print("[replay] POLICY_LOADED", flush=True)
    obs = unwrap(wrapped.reset())
else:
    wrapped = None
    env.reset()

dev = env.device
EE_BODY = "rl_dg_mount"
ee_ids, _ = robot.find_bodies([EE_BODY])
ee_id = ee_ids[0]


def stack(key_path, idx_list):
    seq = state
    for k in key_path:
        seq = seq[k]
    return torch.stack([seq[i] for i in idx_list]).to(dev)


def to_world(root_pose: torch.Tensor) -> torch.Tensor:
    """Stored poses are RELATIVE to the env origin (``get_state(is_relative=True)``).

    With one env the origin is (0,0,0) and forgetting this is invisible; with several it teleports
    every env but the first by its grid offset. That is what a first run of this script did -- all
    eight states fired ``object_out_of_bound`` on step 0 with a 3.5 m world drift. Mirrors what
    ``MultiResetManager._reset_to`` does under ``is_relative``.
    """
    out = root_pose.clone()
    out[:, :3] += env.scene.env_origins
    return out


def write_states() -> None:
    """Write the whole bank slice at once: robot root, all 26 joints, PD targets, and the leg."""
    robot.write_root_pose_to_sim(to_world(stack(["articulation", "robot", "root_pose"], sel)))
    robot.write_root_velocity_to_sim(stack(["articulation", "robot", "root_velocity"], sel))
    q = stack(["articulation", "robot", "joint_position"], sel)
    qd = stack(["articulation", "robot", "joint_velocity"], sel)
    robot.write_joint_state_to_sim(q, qd)
    # The production loader does this too (MultiResetManager._reset_to). Without it the actuators
    # keep commanding whatever they last held -- for these envs, the default OPEN posture -- and the
    # hand drives itself off the object, which measures the controller and not the state.
    robot.set_joint_position_target(q)
    robot.set_joint_velocity_target(torch.zeros_like(qd))
    obj.write_root_pose_to_sim(to_world(stack(["rigid_object", "insertive_object", "root_pose"], sel)))
    obj.write_root_velocity_to_sim(stack(["rigid_object", "insertive_object", "root_velocity"], sel))
    env.scene.write_data_to_sim()
    env.sim.forward()


def in_hand() -> torch.Tensor:
    """Object position expressed in the palm frame. Constant while the grasp holds, whatever the arm does."""
    p, q = math_utils.subtract_frame_transforms(
        robot.data.body_pos_w[:, ee_id], robot.data.body_quat_w[:, ee_id],
        obj.data.root_pos_w, obj.data.root_quat_w,
    )
    return p


# WHAT ELSE THE ENV RE-DREW WHEN IT RESET, none of which the state file stores. This is the
# substantive answer to "the state records every position and velocity, so why is the replay not
# exact": the file records the CONFIGURATION, and the env re-samples the PLANT around it. In this
# task the reset-mode events are reset_object, reset_root, reset_robot_joints,
# reset_robot_wrist_joint, reset_robot_elbow_joint, reset_finger_root_joints (all overwritten by
# the write below), plus TWO that are not -- variable_gravity and randomize_arm_sysid.
grav = env.sim.get_physics_context().get_gravity()  # (direction, magnitude) on this Isaac build
print(f"[replay] gravity as reset drew it: {grav}", flush=True)
arm_ids, arm_names = robot.find_joints([r"shoulder.*", r"elbow.*", r"wrist.*"], preserve_order=False)
print(f"[replay] arm joint friction as reset drew it: "
      f"{[round(float(v), 3) for v in robot.data.joint_friction_coeff[0, arm_ids]]}", flush=True)
print(f"[replay] arm armature as reset drew it:       "
      f"{[round(float(v), 3) for v in robot.data.joint_armature[0, arm_ids]]}", flush=True)

write_states()
written_pos = obj.data.root_pos_w.clone()
target_gap = float((robot.data.joint_pos_target - robot.data.joint_pos).abs().max())
print(f"[replay] wrote {len(sel)} states; PD target gap {target_gap:.4f} rad", flush=True)

alive = torch.ones(len(sel), dtype=torch.bool, device=dev)
term_reported = False
# Baseline taken IMMEDIATELY, not only after the warmup. An env that dies inside the warmup window
# would otherwise never get a baseline at all, leave its drift at exactly 0.0, and be scored as a
# perfect hold -- the same shape of confound this script exists to remove.
baseline = in_hand().clone()
rel_drift = torch.zeros(len(sel), device=dev)
world_drift = torch.zeros(len(sel), device=dev)
z_end = obj.data.root_pos_w[:, 2].clone()
died_at = torch.full((len(sel),), -1, dtype=torch.long, device=dev)

for step in range(args.rollout_steps):
    with torch.inference_mode():
        if player is not None:
            act = player.get_action(obs, is_deterministic=True)
            # STEP THROUGH THE WRAPPER, not the raw env. RlGamesVecEnvWrapper is what assembles
            # and clips the observation groups the checkpoint was trained on; reading env.step's
            # raw dict instead handed the policy a 185-d tensor against the 1390-d running-mean
            # buffer and every policy arm of a first run died on that shape mismatch.
            ret = wrapped.step(act)
            obs = unwrap(ret[0])
            done = ret[2].to(dev).bool()
        else:
            act = torch.zeros((len(sel), env.action_manager.total_action_dim), device=dev)
            _, _, terminated, truncated, _ = env.step(act)
            done = (terminated | truncated).to(dev).bool()

        if step == args.warmup_steps:
            # Re-baseline AFTER the warmup for the envs still alive, so recovering from the
            # mid-episode teleport is not scored as slipping. Envs already dead keep the
            # write-time baseline they were measured against.
            fresh = in_hand()
            baseline = torch.where(alive.unsqueeze(1), fresh, baseline)
            written_pos = torch.where(alive.unsqueeze(1), obj.data.root_pos_w, written_pos)

        cur_rel = torch.linalg.norm(in_hand() - baseline, dim=1)
        cur_world = torch.linalg.norm(obj.data.root_pos_w - written_pos, dim=1)
        rel_drift = torch.where(alive, cur_rel, rel_drift)
        world_drift = torch.where(alive, cur_world, world_drift)
        z_end = torch.where(alive, obj.data.root_pos_w[:, 2], z_end)

        # An env that terminates is auto-reset INSIDE env.step, so its state is gone. Freeze its
        # metrics at the last valid step rather than letting a fresh episode overwrite them.
        if bool((done & alive).any()) and not term_reported:
            # WHICH term fired, not just that one did. A replay that dies instantly and a replay
            # that slips look identical in the summary; the term name separates them.
            term_reported = True
            fired = {n: int(env.termination_manager.get_term(n)[done & alive].sum())
                     for n in env.termination_manager.active_terms}
            print(f"[replay] first terminations at step {step}: {fired}", flush=True)
        newly = done & alive
        died_at = torch.where(newly, torch.full_like(died_at, step), died_at)
        alive = alive & ~done

HOLD_MM = 20.0
rel_mm = rel_drift * 1e3
world_mm = world_drift * 1e3
held = rel_mm < HOLD_MM


def q(t):
    return [round(float(v), 2) for v in torch.quantile(t, torch.tensor([0.5, 0.25, 0.75]).to(t.device))]


tag = args.tag or f"{args.hand}_{args.policy}"
print("\n[replay] ==================== RESULT ====================", flush=True)
print(f"[replay] arm={tag}  hand={args.hand}  policy={args.policy}  states={len(sel)}  steps={args.rollout_steps}", flush=True)
print(f"[replay] IN-HAND drift mm  median {q(rel_mm)[0]}  IQR [{q(rel_mm)[1]}, {q(rel_mm)[2]}]", flush=True)
print(f"[replay] WORLD   drift mm  median {q(world_mm)[0]}  IQR [{q(world_mm)[1]}, {q(world_mm)[2]}]", flush=True)
print(f"[replay] STILL HELD (in-hand drift < {HOLD_MM:.0f} mm): {int(held.sum())}/{len(sel)} = {float(held.float().mean()):.3f}", flush=True)
print(f"[replay] object z at end   median {float(z_end.median()):.4f} m", flush=True)
print(f"[replay] episodes terminated early: {int((died_at >= 0).sum())}/{len(sel)}", flush=True)
print("[replay] REPLAY_OK", flush=True)

import sys as _sys  # noqa: E402

_sys.stdout.flush()
_sys.stderr.flush()
os._exit(0)
