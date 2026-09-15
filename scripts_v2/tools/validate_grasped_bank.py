# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Replay EVERY state of a grasped reset bank in parallel and keep only the ones that still hold.

The recorder accepts a state after 5 stable control steps. That lets a marginal grasp through: on
the UR10e linear-gripper C2 bank one filmed state in six passed at record time and slid out of the
jaws during a 60-step settle (the insertive cube pressed against the receptive one, jaws closed on
the pair). Filming only samples the bank, so this replays all of it under the same conditions the
film uses -- arm written back to its recorded pose every step, hand holding with the recorder's own
scripted close, no episode time-out -- and filters.

Kept = after ``--settle-steps`` control steps the object is still inside the per-axis grasp-centre
box (bank max |deviation| per axis + ``--margin``, hand frame; the same test the recorder's
``check_reset_state_success`` applies with ``grasp_center_margin``) AND the gripper joints are within
``--joint-fraction`` of travel of the grasp bank median (jaws stopped on the object).

Writes ``--out`` in the bank's own format with only the kept states, and prints per-state numbers.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--reset-states", type=Path, required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--dataset-dir", required=True, help="dataset root holding Grasps/<object>/grasps.pt")
parser.add_argument("--insertive-variant", default="cube34")
parser.add_argument("--receptive-variant", default="cube34")
parser.add_argument("--settle-steps", type=int, default=60)
# Wider than the recorder's 6 mm: after a 60-step settle a held cube slides along the 13 mm-wide pad
# (filmed OPPOSED at y = 23 mm), while the jaw test still rejects jaws that shut past the object.
parser.add_argument("--margin", type=float, default=0.012)
parser.add_argument("--joint-fraction", type=float, default=0.15)
# CONTACT VERDICT, the same test as the film renderer: keep a state when every named fingertip presses
# the object above --contact-threshold after the settle. The box/joint geometry test disagreed with
# the films on 3/45 pinch states (fingers 18-19% of travel from the bank median, cube still held).
parser.add_argument("--tip-names", default=None,
                    help="comma-separated fingertip bodies that must all touch the object (default by task)")
parser.add_argument("--contact-threshold", type=float, default=0.2)
parser.add_argument("--out", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app = AppLauncher(args).app

import uwlab_is6_bridge  # noqa: E402

uwlab_is6_bridge.apply()

import gymnasium as gym  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import torch  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
import uwlab_tasks.manager_based.manipulation.omnireset.mdp.scripted_gripper as scripted_gripper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

raw = torch.load(args.reset_states, map_location="cpu", weights_only=False)
state = raw["initial_state"]
n = len(state["rigid_object"]["insertive_object"]["root_pose"])
print(f"[validate] {n} states in {args.reset_states}", flush=True)

cfg = parse_env_cfg(args.task, device=args.device, num_envs=n)
ee_body_name = cfg.terminations.success.params["ee_body_name"]
for name in list(vars(cfg.terminations)):
    if getattr(cfg.terminations, name, None) is not None:
        setattr(cfg.terminations, name, None)
cfg.episode_length_s = 1.0e6
cfg.scene.insertive_object = cfg.variants["scene.insertive_object"][args.insertive_variant]
cfg.scene.receptive_object = cfg.variants["scene.receptive_object"][args.receptive_variant]
for term in vars(cfg.events).values():
    if term is not None and getattr(term, "params", None) and "dataset_dir" in term.params:
        term.params["dataset_dir"] = args.dataset_dir
for an, at in list(vars(cfg.actions).items()):
    if at is not None and any(k in type(at).__name__ for k in ("InverseKinematics", "Cartesian", "OperationalSpace")):
        setattr(cfg.actions, an, None)
from isaaclab.sensors import ContactSensorCfg  # noqa: E402

if "Delto" in args.task:
    TIP_NAMES = (args.tip_names or "rl_dg_1_tip,rl_dg_2_tip").split(",")
    TIP_PREFIX = "{ENV_REGEX_NS}/Robot/gripper"
else:
    TIP_NAMES = (args.tip_names or "left_inner_finger,right_inner_finger").split(",")
    TIP_PREFIX = "{ENV_REGEX_NS}/Robot/gripper/Geometry"
for tip in TIP_NAMES:
    setattr(cfg.scene, f"{tip}_validate_s", ContactSensorCfg(
        prim_path=f"{TIP_PREFIX}/{tip}", filter_prim_paths_expr=["{ENV_REGEX_NS}/InsertiveObject"]))
cfg.scene.robot.spawn = cfg.scene.robot.spawn.replace(activate_contact_sensors=True)
cfg.scene.insertive_object.spawn = cfg.scene.insertive_object.spawn.replace(activate_contact_sensors=True)
if os.environ.get("UWLAB_GPU_COLLISION_STACK_SIZE"):
    (getattr(cfg.sim, "physics", None) or cfg.sim.physx).gpu_collision_stack_size = int(
        os.environ["UWLAB_GPU_COLLISION_STACK_SIZE"])

env = gym.make(args.task, cfg=cfg).unwrapped
robot = env.scene["robot"]
obj = env.scene["insertive_object"]
_t = lambda x: getattr(x, "torch", x)  # noqa: E731
arm_ids, _ = robot.find_joints(["shoulder.*", "elbow.*", "wrist.*"])
ee_idx = list(robot.body_names).index(ee_body_name)
# Same arm hold as the film renderer: stiff PD DURING the 12 substeps of each control step, plus the
# joint state written back after it. With the shipped 0/0 gains the arm sags inside every step and
# snaps back, shaking held cubes loose (dropped a state the film shows held).
robot.write_joint_stiffness_to_sim(torch.full((n, len(arm_ids)), 8000.0, device=env.device), joint_ids=arm_ids)
robot.write_joint_damping_to_sim(torch.full((n, len(arm_ids)), 400.0, device=env.device), joint_ids=arm_ids)

# Grasp-centre box and jaw reference, from the grasp bank the replay event loaded.
replay = env.event_manager.get_term_cfg("reset_end_effector_pose_from_grasp_dataset").func
points = math_utils.quat_apply_inverse(replay.rel_quaternions, -replay.rel_positions)
center = points.median(dim=0).values
half = (points - center).abs().max(dim=0).values + args.margin
grip_ids = torch.as_tensor(replay.gripper_joint_ids, device=env.device, dtype=torch.long)
grip_median = replay.gripper_joint_positions.median(dim=0).values
limits = _t(robot.data.joint_pos_limits)[0, grip_ids]
travel = (limits[:, 1] - limits[:, 0]).clamp(min=1e-6)
print(f"[validate] box +-{[round(float(v) * 1000, 1) for v in half]} mm around "
      f"{[round(float(v) * 1000, 1) for v in center]} mm; jaw median {[round(float(v), 4) for v in grip_median]}",
      flush=True)


def stacked(tree):
    if isinstance(tree, dict):
        return {k: stacked(v) for k, v in tree.items()}
    return torch.stack(list(tree)).to(env.device)


batch = stacked(state)
env.reset()
env.scene.reset_to(batch, is_relative=True)
rob = batch["articulation"]["robot"]
if "joint_position_target" in rob:
    robot.set_joint_position_target(rob["joint_position_target"])
    robot.set_joint_velocity_target(rob["joint_velocity_target"])
env.scene.write_data_to_sim()
env.sim.forward()
hold_q = rob["joint_position"][:, arm_ids].clone()


def in_hand_now():
    hand_pos = _t(robot.data.body_link_pos_w)[:, ee_idx]
    hand_quat = _t(robot.data.body_link_quat_w)[:, ee_idx]
    obj_in_hand, _ = math_utils.subtract_frame_transforms(hand_pos, hand_quat, _t(obj.data.root_pos_w))
    off = obj_in_hand - center
    inside = (off.abs() <= half).all(dim=1)
    jaw_err = (_t(robot.data.joint_pos)[:, grip_ids] - grip_median).abs() / travel
    jaws = (jaw_err <= args.joint_fraction).all(dim=1)
    return inside & jaws, off, jaw_err.max(dim=1).values


for _ in range(args.settle_steps):
    env.step(scripted_gripper.scripted_gripper_actions(env, close=True))
    # PD target only. Writing the arm's joint state every step teleports the articulation and resets
    # contact friction -- that is why this validator once disagreed with the films (and why the
    # pinch dropped every cube on the arm).
    robot.set_joint_position_target(hold_q, joint_ids=arm_ids)

geo_ok, off, jaw_err = in_hand_now()
forces = torch.stack([
    _t(env.scene.sensors[f"{tip}_validate_s"].data.force_matrix_w)[:, 0, 0, :].norm(dim=-1) for tip in TIP_NAMES
], dim=1)
keep = (forces > args.contact_threshold).all(dim=1)
for i in range(n):
    print(f"[validate] state {i:3d} {'KEEP' if keep[i] else 'DROP'} tip N "
          f"{[round(float(v), 2) for v in forces[i]]} | geometry {'ok' if geo_ok[i] else 'off'} offset mm "
          f"{[round(float(v) * 1000, 1) for v in off[i]]} jaw err {float(jaw_err[i]):.2f}", flush=True)
kept = torch.where(keep)[0].cpu().tolist()
print(f"[validate] KEPT {len(kept)}/{n} after {args.settle_steps} steps", flush=True)


def select(tree):
    if isinstance(tree, dict):
        return {k: select(v) for k, v in tree.items()}
    return [tree[i] for i in kept]


filtered = dict(raw)
filtered["initial_state"] = select(state)
args.out.parent.mkdir(parents=True, exist_ok=True)
torch.save(filtered, args.out)
print(f"[validate] wrote {args.out}", flush=True)
env.close()
app.close()
