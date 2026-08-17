# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Instrumented diagnostic for the stale-PD-target hypothesis in
``reset_end_effector_from_grasp_dataset`` (bead: RestingEEGrasped velocity-explosion, 2026-08-17).

BACKGROUND. ObjectRestingEEGrasped measured 0.83% acceptance for the leg pair (40/4799), against
57.21% for ObjectAnywhereEEAnywhere on the same pair. The binding rejection gate was ``not_far``
(6/32), and the robot's per-step joint-velocity-sum diagnostic showed median 3.452 against a
5.0 gate but a MAX of 9306.249 -- a tail three-plus orders of magnitude past the gate, while
``coll_free`` stayed high (30/32): not a collision problem, a dynamics explosion at reset.

HYPOTHESIS (established from source alone, not yet measured): ``reset_end_effector_from_grasp_dataset``
(events.py) runs 25 damped IK iterations, each of which calls the IK action's ``apply_actions()``,
which sets the ARM ACTUATOR'S PD POSITION TARGET (``joint_pos_target``) to a FRESH one-shot IK
solution every iteration -- but the manual loop only writes 25% of that gap to the RAW joint
position (with velocity explicitly zeroed each write). Nothing after the loop -- including
``_wrap_joints_into_limits``, which only touches position -- ever reconciles ``joint_pos_target``
back to the position actually written. Compare ``MultiResetManager._reset_to`` (events.py:1817-1820),
which explicitly calls ``set_joint_position_target``/``set_joint_velocity_target`` after writing state,
with its own FIXME flagging exactly this PD-controller assumption. If this is the mechanism, the
FIRST physics step after a grasp-dataset reset should show a velocity spike whose magnitude tracks
the (``joint_pos_target`` - ``joint_pos``) gap left over from the reset -- and that is exactly what
this script measures, without changing anything.

WHAT THIS SCRIPT DOES, per env, all on the ARM joints (``shoulder.*``, ``elbow.*``, ``wrist.*`` --
the same ``robot_ik_cfg`` pattern the grasp-dataset reset itself uses):
  1. Snapshot joint_pos BEFORE any reset (whatever the previous state/spawn defaults left it at).
  2. Call env.reset() -- runs the REAL, unmodified reset event chain for
     OmniReset-UR5eDelto-ObjectRestingEEGrasped-v0 (reset_everything, reset_robot_pose,
     reset_receptive_object_pose, reset_insertive_object_pose_from_reset_states,
     reset_end_effector_pose_from_grasp_dataset -- the LAST reset-mode term declared for this task,
     confirmed from the printed Event Manager table, so nothing after it can touch joint_pos_target).
  3. Snapshot joint_pos_target and joint_pos AFTER reset() returns -- since nothing after the grasp-
     dataset event's own loop+wrap touches either quantity, these ARE the "after the loop" /
     "after loop+wrap" values team asked for, read the honest way (from the real call, not a
     monkey-patched stand-in).
  4. Compute, per env, per arm joint: the JUMP (post-reset pos - pre-reset pos), the TARGET GAP
     (joint_pos_target - joint_pos after reset), and joint-limit margin (distance to nearest limit).
  5. Step ONE physics tick (zero action) and read joint_vel -- the SAME "sum of |v|" metric quoted
     against the 5.0 gate in ``terminations.py``'s ``stable`` condition, computed the identical way
     (``asset.data.joint_vel.abs().sum(dim=1)``).
  6. Report the joint-vel distribution (median/max, matching the numbers already measured) AND its
     correlation against the target-gap and the jump -- if the mechanism is right, the tail should be
     the envs with the largest gap, not a random subset.

Does NOT change reset_end_effector_from_grasp_dataset, does NOT apply the fix -- diagnostic only, per
instruction: confirm the mechanism empirically before changing anything.

Run (one Isaac process; never via uwlab.sh):
    PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl" \\
        timeout -s KILL 300 <python> -u scripts_v2/tools/diagnose_grasp_dataset_reset_velocity.py \\
        --task OmniReset-UR5eDelto-ObjectRestingEEGrasped-v0 --num_envs 256 \\
        env.scene.insertive_object=leg200mm env.scene.receptive_object=onelegfixture
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Instrument the grasp-dataset reset for a first-step velocity spike.")
parser.add_argument("--task", type=str, default="OmniReset-UR5eDelto-ObjectRestingEEGrasped-v0")
parser.add_argument("--num_envs", type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402

# Same arm-joint pattern reset_end_effector_from_grasp_dataset's own robot_ik_cfg uses
# (reset_states_cfg.py) -- arm joints only, not the 20 hand joints.
_ARM_JOINT_PATTERN = ["shoulder.*", "elbow.*", "wrist.*"]


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    obj = env.scene["insertive_object"]

    arm_ids, arm_names = robot.find_joints(_ARM_JOINT_PATTERN, preserve_order=True)
    hand_ids, hand_names = robot.find_joints(["rj_dg_[1-5]_[1-4]"], preserve_order=True)
    print(f"[diag] arm joints ({len(arm_names)}): {arm_names}", flush=True)
    print(f"[diag] hand joints ({len(hand_names)}): {hand_names}", flush=True)
    limits = robot.data.joint_pos_limits[:, arm_ids].clone()  # (N, 6, 2)

    # -- one settle reset so the FIRST measured reset isn't confounded by whatever arbitrary state
    # the scene starts construction in.
    env.reset()

    # -- (1) pre-reset snapshot.
    pos_before = robot.data.joint_pos[:, arm_ids].clone()
    hand_pos_before = robot.data.joint_pos[:, hand_ids].clone()

    # -- (2) the REAL, unmodified reset event chain -- includes reset_end_effector_pose_from_grasp_dataset.
    env.reset()

    # -- (3) post-reset snapshot: joint_pos_target and joint_pos are exactly what the grasp-dataset
    # event's loop+wrap left them at (see module docstring for why nothing after it touches either).
    pos_after = robot.data.joint_pos[:, arm_ids].clone()
    target_after = robot.data.joint_pos_target[:, arm_ids].clone()
    hand_pos_after = robot.data.joint_pos[:, hand_ids].clone()
    hand_target_after = robot.data.joint_pos_target[:, hand_ids].clone()

    # -- (4) derived per-env, per-joint quantities, ARM.
    jump = (pos_after - pos_before).abs()  # how far the IK replay moved each arm joint
    gap = (target_after - pos_after).abs()  # the unreconciled PD-target residual, per joint
    margin_lo = pos_after - limits[..., 0]
    margin_hi = limits[..., 1] - pos_after
    margin = torch.minimum(margin_lo, margin_hi)  # distance to nearest limit, per joint (negative = outside)

    jump_env = jump.sum(dim=1)
    gap_env = gap.sum(dim=1)
    margin_env = margin.min(dim=1).values  # worst (smallest) margin across this env's 6 arm joints

    # -- same, HAND joints -- the grasp-dataset event writes closed-posture positions with velocity
    # zeroed (events.py:1463-1468) but, like the arm, never calls set_joint_position_target for the
    # hand either. Whatever the hand's PD target was left at BEFORE this reset (e.g. an open-posture
    # default from a prior term) is a second, independent candidate stale-target gap.
    hand_jump_env = (hand_pos_after - hand_pos_before).abs().sum(dim=1)
    hand_gap_env = (hand_target_after - hand_pos_after).abs().sum(dim=1)

    # -- achieved grasp geometry at this reset (same subtract_frame_transforms this whole bridge
    # already uses): palm-object distance, to test whether a BAD (e.g. interpenetrating) placement --
    # not a stale PD target -- is what correlates with the violent tail.
    ee_cfg = SceneEntityCfg("robot", body_names="rl_dg_mount")
    ee_cfg.resolve(env.scene)
    palm_id = ee_cfg.body_ids[0]
    palm_pos_after = robot.data.body_pos_w[:, palm_id, :].clone()
    obj_pos_after = obj.data.root_pos_w.clone()
    palm_obj_dist_env = torch.linalg.norm(palm_pos_after - obj_pos_after, dim=-1) * 1000.0  # mm

    # -- (5) step ONE physics tick, zero action, and read the SAME metric terminations.py's `stable`
    # gate uses: joint_vel.abs().sum(dim=1), over ALL joints (arm + hand), not just the arm subset.
    # ALSO read the object's own linear/angular velocity after the same step -- if the robot's and
    # the object's velocity spikes co-occur, that points at a shared contact-impulse event (a
    # penetration the sim resolves violently) rather than an isolated robot-only PD-servo defect.
    zero_action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    env.step(zero_action)
    vel_sum_env = robot.data.joint_vel.abs().sum(dim=1)
    obj_lin_vel_env = obj.data.root_lin_vel_w.abs().sum(dim=1)
    obj_ang_vel_env = obj.data.root_ang_vel_w.abs().sum(dim=1)

    def stats(name: str, x: torch.Tensor) -> None:
        print(
            f"[diag] {name}: min {x.min():.4f}  median {x.median():.4f}  mean {x.mean():.4f}"
            f"  max {x.max():.4f}  p95 {torch.quantile(x, 0.95):.4f}",
            flush=True,
        )

    print("\n=== PRE/POST RESET, ARM JOINTS ===", flush=True)
    stats("jump (sum |pre-post| over 6 arm joints, rad)", jump_env)
    stats("target-gap (sum |target-actual| over 6 arm joints, rad)", gap_env)
    stats("limit margin (min over 6 arm joints, rad; negative = outside limit)", margin_env)
    n_outside = int((margin_env < 0).sum())
    n_near = int(((margin_env >= 0) & (margin_env < 0.01)).sum())
    print(f"[diag] envs with an arm joint AT/OUTSIDE its limit: {n_outside}/{env.num_envs}", flush=True)
    print(f"[diag] envs within 0.01 rad of a limit (not outside): {n_near}/{env.num_envs}", flush=True)

    print("\n=== PRE/POST RESET, HAND JOINTS ===", flush=True)
    stats("jump (sum |pre-post| over 20 hand joints, rad)", hand_jump_env)
    stats("target-gap (sum |target-actual| over 20 hand joints, rad)", hand_gap_env)

    print("\n=== ACHIEVED GRASP GEOMETRY AT RESET ===", flush=True)
    stats("palm-object distance (mm)", palm_obj_dist_env)

    print("\n=== POST-RESET, ONE PHYSICS STEP, joint_vel.abs().sum(dim=1) (ALL joints, terminations.py's own metric) ===",
          flush=True)
    stats("robot joint |v| sum", vel_sum_env)
    print(f"[diag] fraction below the 5.0 stability gate: {(vel_sum_env < 5.0).float().mean():.3f}", flush=True)
    stats("object lin |v| sum", obj_lin_vel_env)
    stats("object ang |v| sum", obj_ang_vel_env)

    # -- correlation: does the target-gap (or the jump) predict the velocity spike? Does the OBJECT's
    # velocity co-occur with the robot's (shared contact-impulse signature) or not (independent
    # mechanisms)?
    def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.float()
        b = b.float()
        return float(torch.corrcoef(torch.stack([a, b]))[0, 1])

    print("\n=== CORRELATIONS WITH FIRST-STEP robot joint |v| sum ===", flush=True)
    print(f"[diag] corr(arm target-gap, vel_sum)  = {pearson(gap_env, vel_sum_env):.4f}", flush=True)
    print(f"[diag] corr(arm jump, vel_sum)         = {pearson(jump_env, vel_sum_env):.4f}", flush=True)
    print(f"[diag] corr(hand target-gap, vel_sum)  = {pearson(hand_gap_env, vel_sum_env):.4f}", flush=True)
    print(f"[diag] corr(hand jump, vel_sum)         = {pearson(hand_jump_env, vel_sum_env):.4f}", flush=True)
    print(f"[diag] corr(palm-obj distance, vel_sum) = {pearson(palm_obj_dist_env, vel_sum_env):.4f}", flush=True)
    print(f"[diag] corr(object lin|v|, robot vel_sum) = {pearson(obj_lin_vel_env, vel_sum_env):.4f}", flush=True)
    print(f"[diag] corr(object ang|v|, robot vel_sum) = {pearson(obj_ang_vel_env, vel_sum_env):.4f}", flush=True)

    # -- top-20 worst envs by velocity, full row, to attribute the tail directly.
    worst = torch.argsort(vel_sum_env, descending=True)[:20]
    print("\n=== TOP 20 WORST ENVS BY POST-RESET robot joint |v| sum ===", flush=True)
    print(
        f"{'env':>5} {'vel_sum':>10} {'a_gap':>8} {'a_jump':>8} {'margin':>8}"
        f" {'h_gap':>8} {'h_jump':>8} {'palm_obj_mm':>12} {'obj_lin_v':>10} {'obj_ang_v':>10}",
        flush=True,
    )
    for i in worst.tolist():
        print(
            f"{i:5d} {vel_sum_env[i]:10.3f} {gap_env[i]:8.4f} {jump_env[i]:8.4f} {margin_env[i]:8.4f}"
            f" {hand_gap_env[i]:8.4f} {hand_jump_env[i]:8.4f} {palm_obj_dist_env[i]:12.2f}"
            f" {obj_lin_vel_env[i]:10.4f} {obj_ang_vel_env[i]:10.4f}",
            flush=True,
        )

    # -- best 10 too, for contrast.
    best = torch.argsort(vel_sum_env, descending=False)[:10]
    print("\n=== BEST 10 ENVS BY POST-RESET robot joint |v| sum ===", flush=True)
    print(
        f"{'env':>5} {'vel_sum':>10} {'a_gap':>8} {'a_jump':>8} {'margin':>8}"
        f" {'h_gap':>8} {'h_jump':>8} {'palm_obj_mm':>12} {'obj_lin_v':>10} {'obj_ang_v':>10}",
        flush=True,
    )
    for i in best.tolist():
        print(
            f"{i:5d} {vel_sum_env[i]:10.3f} {gap_env[i]:8.4f} {jump_env[i]:8.4f} {margin_env[i]:8.4f}"
            f" {hand_gap_env[i]:8.4f} {hand_jump_env[i]:8.4f} {palm_obj_dist_env[i]:12.2f}"
            f" {obj_lin_vel_env[i]:10.4f} {obj_ang_vel_env[i]:10.4f}",
            flush=True,
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
