# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Closed-loop grasp-fidelity measurement for bead UWLab-algw.7's final follow-up: measure grip via
FINGERTIP CONTACT FORCE at the end of a real 2s policy rollout, with NOTHING else changed -- no
stopping the policy, no settle, no probe, no disturbance. A jog at the end of an active rollout was
tried earlier and is uninterpretable, because residual momentum makes object displacement exceed
gripper displacement in the same direction; reading force instead of motion sidesteps that.

LINEAGE. Promoted from a scratch script (/tmp/algw7_replay_contact.py, itself descended from
replay_reset_states_policy.py's 2x2 A/B harness in this directory) after it produced this project's
n=300 headline grasp-fidelity number. Reuses replay_reset_states_policy.py's write/rollout machinery
verbatim and reuses dexlift's own ``_sensor_force_magnitudes`` (mdp/rewards.py) for the contact
read -- same sensors, same 0.2 N gate, same thumb/other-tip opposition split ``held_with_probe``
uses -- so a passing grade here means the same thing a training-time success check would mean.

ONE FIX ON TOP OF THE SCRATCH VERSION, found by using it rather than by inspection: the scratch
script tracked `alive`/`done` only to freeze the in-hand drift baseline, and never checked whether
any env had fired a termination mid-rollout. ManagerBasedRLEnv auto-resets a done env INSIDE the
same env.step() call, so an env that terminates at step 20 has been silently teleported into a
BRAND NEW random episode by the time the final-step contact read happens at step 120 -- reading
"grasped" on that env describes an unrelated fresh reset, not the outcome of the state this script
is grading. Confirmed non-hypothetical on the real n=300 run: 2/300 envs terminated (steps 20, 29)
and were excluded from every aggregate below (n=298), not silently averaged in. The two are still
printed in the per-state table, flagged, so nothing is dropped from the record, only from the
headline statistics that would otherwise have been quietly wrong by two states' worth of noise.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--reset-states", type=Path, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--agent-yaml", type=str, default=None)
parser.add_argument("--task", default="DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-Play-v0")
parser.add_argument("--num-states", type=int, default=34)
parser.add_argument("--rollout-steps", type=int, default=120)
parser.add_argument("--warmup-steps", type=int, default=5)
parser.add_argument("--hand", choices=["reference", "identified"], default="reference")
parser.add_argument("--target", choices=["recorded", "q"], default="recorded")
parser.add_argument("--force-threshold", type=float, default=0.2,
                     help="N, held_with_probe's own default (matches good_finger_contact / "
                          "probe_grasp.py's calibration) -- reused, not re-picked.")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

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
print(f"[contact] {n_total} states in bank; replaying {len(sel)} (seed {args.seed}) target={args.target}", flush=True)

app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

# REUSED, not reimplemented -- the same function held_check.py calls, same thumb/tip name split
# held_with_probe defaults to (rl_dg_1_tip/rl_dg_5_tip = thumb side; rl_dg_2/3/4_tip = other tips).
from uwlab_tasks.manager_based.manipulation.dexlift.mdp.rewards import _sensor_force_magnitudes  # noqa: E402

THUMB_TIP_NAMES = ("rl_dg_1_tip", "rl_dg_5_tip")
OTHER_TIP_NAMES = ("rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip")
ALL_TIP_NAMES = THUMB_TIP_NAMES + OTHER_TIP_NAMES

cfg = parse_env_cfg(args.task, device=args.device, num_envs=len(sel))
cfg.seed = args.seed
cfg.sim.physx.gpu_collision_stack_size = 512 * 1024 * 1024
env = gym.make(args.task, cfg=cfg).unwrapped
robot = env.scene["robot"]
obj = env.scene["object"]

agent_yaml = args.agent_yaml or os.path.join(Path(args.checkpoint).parent.parent, "params", "agent.yaml")
agent_cfg = yaml.safe_load(open(agent_yaml))
clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
clip_act = agent_cfg["params"]["env"].get("clip_actions", math.inf)
wrapped = RlGamesVecEnvWrapper(env, args.device, clip_obs, clip_act,
                               agent_cfg["params"]["env"].get("obs_groups"),
                               agent_cfg["params"]["env"].get("concate_obs_groups", True))
vecenv.register("IsaacRlgWrapper", lambda name, num_actors, **kw: RlGamesGpuEnv(name, num_actors, **kw))
env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped})
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


def unwrap(o):
    if isinstance(o, tuple):
        o = o[0]
    if isinstance(o, dict):
        o = o["obs"]
    return o


obs = unwrap(wrapped.reset())
dev = env.device
EE_BODY = "rl_dg_mount"
ee_ids, _ = robot.find_bodies([EE_BODY])
ee_id = ee_ids[0]


def stack(key_path, idx_list):
    seq = state
    for k in key_path:
        seq = seq[k]
    return torch.stack([seq[i] for i in idx_list]).to(dev)


def to_world(root_pose):
    out = root_pose.clone()
    out[:, :3] += env.scene.env_origins
    return out


def in_hand():
    p, q = math_utils.subtract_frame_transforms(
        robot.data.body_pos_w[:, ee_id], robot.data.body_quat_w[:, ee_id],
        obj.data.root_pos_w, obj.data.root_quat_w,
    )
    return p


def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval (default z=1.96). No scipy needed."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - margin, center + margin)


DEXLIFT_TABLE_SURFACE_Z_LOCAL = 0.0
AIRBORNE_THRESHOLD_M = 0.08

robot.write_root_pose_to_sim(to_world(stack(["articulation", "robot", "root_pose"], sel)))
robot.write_root_velocity_to_sim(stack(["articulation", "robot", "root_velocity"], sel))
q0 = stack(["articulation", "robot", "joint_position"], sel)
qd0 = stack(["articulation", "robot", "joint_velocity"], sel)
robot.write_joint_state_to_sim(q0, qd0)
if args.target == "recorded":
    target_q = stack(["articulation", "robot", "joint_position_target"], sel)
    target_qd = stack(["articulation", "robot", "joint_velocity_target"], sel)
else:
    target_q = q0
    target_qd = torch.zeros_like(qd0)
robot.set_joint_position_target(target_q)
robot.set_joint_velocity_target(target_qd)
obj.write_root_pose_to_sim(to_world(stack(["rigid_object", "insertive_object", "root_pose"], sel)))
obj.write_root_velocity_to_sim(stack(["rigid_object", "insertive_object", "root_velocity"], sel))
env.scene.write_data_to_sim()
env.sim.forward()

height_start = (obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - DEXLIFT_TABLE_SURFACE_Z_LOCAL).clone()
airborne_mask = (height_start > AIRBORNE_THRESHOLD_M).cpu()

alive = torch.ones(len(sel), dtype=torch.bool, device=dev)
baseline = in_hand().clone()
rel_drift = torch.zeros(len(sel), device=dev)
# DIED-AT TRACKING, added on top of the inherited script. object_out_of_bound is an ACTIVE
# termination on this Play cfg (confirmed at gym.make: time_out / object_out_of_bound /
# success_rate_log). ManagerBasedRLEnv auto-resets any env that fires a termination INSIDE the
# same env.step() call -- so an env that terminates at, say, step 40 has been silently teleported
# into a BRAND NEW random episode by step 120, and a "passive final-step contact read" on that env
# would measure an unrelated fresh reset, not the outcome of the state this job is grading. The
# inherited script tracked `alive` only for the drift baseline and never checked or reported this;
# without ruling it out, "grasped_by_contact" could be silently contaminated by fresh-episode noise
# for exactly the states most likely to have been dropped early.
died_at = torch.full((len(sel),), -1, dtype=torch.long, device=dev)

# ==================== the SAME 2s closed-loop rollout, UNCHANGED except the died_at bookkeeping ====================
for step in range(args.rollout_steps):
    with torch.inference_mode():
        act = player.get_action(obs, is_deterministic=True)
        ret = wrapped.step(act)
        obs = unwrap(ret[0])
        done = ret[2].to(dev).bool()

        if step == args.warmup_steps:
            fresh = in_hand()
            baseline = torch.where(alive.unsqueeze(1), fresh, baseline)

        cur_rel = torch.linalg.norm(in_hand() - baseline, dim=1)
        rel_drift = torch.where(alive, cur_rel, rel_drift)
        newly = done & alive
        died_at = torch.where(newly, torch.full_like(died_at, step), died_at)
        alive = alive & ~done

HOLD_MM = 20.0
rel_mm = (rel_drift * 1e3).cpu()
held_strict = rel_mm < HOLD_MM
died_at_cpu = died_at.cpu()
n_early_reset = int((died_at_cpu >= 0).sum())
print(f"\n[contact] Phase 1 (2s closed-loop) done. held-by-strict-20mm: "
      f"{int(held_strict.sum())}/{len(sel)} = {float(held_strict.float().mean()):.3f}", flush=True)
print(f"[contact] envs that fired a termination mid-rollout (auto-reset into a NEW episode before "
      f"the final-step contact read): {n_early_reset}/{len(sel)}"
      + (f"  steps={sorted(int(v) for v in died_at_cpu[died_at_cpu >= 0].tolist())}" if n_early_reset else ""),
      flush=True)
if n_early_reset:
    fired = {name: int(env.termination_manager.get_term(name).cpu()[died_at_cpu >= 0].sum())
             for name in env.termination_manager.active_terms} if hasattr(env, "termination_manager") else {}
    print(f"[contact] WARNING: {n_early_reset} state(s) above will be EXCLUDED from grasped_by_contact "
          f"and held_strict -- their final-step read would describe an unrelated fresh episode, not "
          f"the state this job is grading. Term breakdown (last step's buffer, informational only): "
          f"{fired}", flush=True)

# ==================== PASSIVE CONTACT-FORCE READ AT THE FINAL STEP -- NOTHING CHANGED ====================
with torch.inference_mode():
    thumb_force = _sensor_force_magnitudes(env, THUMB_TIP_NAMES)   # (N, 2)
    other_force = _sensor_force_magnitudes(env, OTHER_TIP_NAMES)   # (N, 3)
    all_force = _sensor_force_magnitudes(env, ALL_TIP_NAMES)       # (N, 5)

thumb_force_cpu = thumb_force.cpu()
other_force_cpu = other_force.cpu()
all_force_cpu = all_force.cpu()

thumb_loaded = (thumb_force_cpu > args.force_threshold).any(dim=-1)
other_loaded = (other_force_cpu > args.force_threshold).any(dim=-1)
n_tips_loaded = (all_force_cpu > args.force_threshold).sum(dim=-1)
grasped_by_contact = thumb_loaded & other_loaded  # opposed: >=1 thumb-side + >=1 other tip loaded

print(f"\n[contact] force threshold = {args.force_threshold} N "
      f"(held_with_probe's own default -- not re-picked)", flush=True)
print(f"[contact] per-tip force distribution (N), all {len(sel)} requested states x 5 tips "
      f"({ALL_TIP_NAMES}):", flush=True)
flat = all_force_cpu.flatten()
print(f"[contact]   min={flat.min().item():.4f}  median={flat.median().item():.4f}  "
      f"p90={torch.quantile(flat, 0.9).item():.4f}  max={flat.max().item():.4f}", flush=True)
print(f"[contact] n_tips_loaded distribution over {len(sel)} requested states: "
      f"{[int((n_tips_loaded == k).sum()) for k in range(6)]}  (index = tip count 0..5)", flush=True)

# EXCLUDE any env that auto-reset mid-rollout from every aggregate below -- its final-step read is
# not about the state this job is grading (see the WARNING above). n is the count of VALID states,
# not len(sel); a state excluded here still gets its raw numbers printed in the per-state table,
# flagged, so nothing is silently dropped from the record, only from the headline statistics.
valid_mask = valid_mask_full = (died_at_cpu < 0)
n = int(valid_mask.sum())
grasped_v = grasped_by_contact[valid_mask]
held_v = held_strict[valid_mask]
airborne_v = airborne_mask[valid_mask]
n_grasped = int(grasped_v.sum())
n_held = int(held_v.sum())
ci_grasped = wilson_ci(n_grasped, n)
ci_held = wilson_ci(n_held, n)
print(f"\n[contact] === RESULT: GRASPED BY CONTACT (thumb-side AND other-side tip both loaded, "
      f">{args.force_threshold}N) ===  [n={n} valid states out of {len(sel)} requested; "
      f"{len(sel) - n} excluded for an early auto-reset]", flush=True)
print(f"[contact] grasped_by_contact: {n_grasped}/{n} = {n_grasped / n:.4f}  "
      f"Wilson 95% CI [{ci_grasped[0]:.4f}, {ci_grasped[1]:.4f}]", flush=True)
print(f"[contact] held-by-strict-20mm: {n_held}/{n} = {n_held / n:.4f}  "
      f"Wilson 95% CI [{ci_held[0]:.4f}, {ci_held[1]:.4f}]", flush=True)

print("\n[contact] === THREE-WAY CROSS-TAB (held-by-strict-20mm x grasped-by-contact) ===", flush=True)
both = held_v & grasped_v
only_strict = held_v & ~grasped_v
only_contact = ~held_v & grasped_v
neither = ~held_v & ~grasped_v
print(f"[contact] held AND grasped-by-contact  : {int(both.sum())}/{n} = {float(both.float().mean()):.3f}", flush=True)
print(f"[contact] held only (contact says no)  : {int(only_strict.sum())}/{n} = {float(only_strict.float().mean()):.3f}", flush=True)
print(f"[contact] contact only (not held)      : {int(only_contact.sum())}/{n} = {float(only_contact.float().mean()):.3f}", flush=True)
print(f"[contact] neither                      : {int(neither.sum())}/{n} = {float(neither.float().mean()):.3f}", flush=True)

n_airborne = int(airborne_v.sum())
n_supported = n - n_airborne
print(f"\n[contact] === AIRBORNE/SUPPORTED SPLIT (threshold={AIRBORNE_THRESHOLD_M}m) ===", flush=True)
print(f"[contact] n_airborne={n_airborne}/{n}  n_supported={n_supported}/{n}", flush=True)
for label, mask in (("AIRBORNE", airborne_v), ("SUPPORTED", ~airborne_v)):
    grp_n = int(mask.sum())
    if grp_n == 0:
        print(f"[contact] {label:10s} n=0 (empty group)", flush=True)
        continue
    grp_grasped = int(grasped_v[mask].sum())
    grp_held = int(held_v[mask].sum())
    ci_g = wilson_ci(grp_grasped, grp_n)
    print(f"[contact] {label:10s} n={grp_n:4d}  held-strict={grp_held}/{grp_n}={grp_held / grp_n:.3f}  "
          f"grasped-by-contact={grp_grasped}/{grp_n}={grp_grasped / grp_n:.3f}  "
          f"Wilson 95% CI [{ci_g[0]:.3f}, {ci_g[1]:.3f}]", flush=True)

print("\n[contact] === PER-STATE (ALL requested states, INCLUDING excluded ones, flagged) ===", flush=True)
print("[contact] idx  valid  held_strict  n_tips_loaded  thumb_loaded  other_loaded  grasped_by_contact  "
      "tip_forces(rl_dg_1,5,2,3,4)", flush=True)
for i in range(len(sel)):
    forces_str = ",".join(f"{all_force_cpu[i, j].item():.3f}" for j in range(5))
    print(
        f"[contact] {i:3d}  {bool(valid_mask_full[i].item())!s:5s}  "
        f"{bool(held_strict[i].item())!s:6s}  {int(n_tips_loaded[i].item()):3d}          "
        f"{bool(thumb_loaded[i].item())!s:6s}       {bool(other_loaded[i].item())!s:6s}       "
        f"{bool(grasped_by_contact[i].item())!s:6s}              [{forces_str}]",
        flush=True,
    )

print("\nCONTACT_DONE", flush=True)
import sys as _sys  # noqa: E402
_sys.stdout.flush()
_sys.stderr.flush()
os._exit(0)
