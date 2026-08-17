# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test whether the grasp-dataset replay's 25-iteration damped IK is CONVERGING SLOWLY or hitting an
UNREACHABLE target (bead: RestingEEGrasped/PartiallyAssembledEEGrasped, IK-convergence check,
2026-08-17).

BACKGROUND. diagnose_grasp_orientation_error.py measured, with a ground-truth captured grasp index
(not a post-hoc guess -- see that script's own method-correction note), a median 19.7 mm
commanded-vs-achieved POSITION error and an 11.1 deg ORIENTATION error at the palm after the event's
normal 25 damped IK iterations. A ~20 mm median residual on a TYPICAL env, not a tail, reads as
"stopped short of its target" rather than "occasionally struggles" -- this script asks which.

METHOD: reset_end_effector_from_grasp_dataset's 25-iteration loop is a deterministic damped
fixed-point iteration toward a FIXED target (set once via self.solver.process_actions(...) before the
loop starts). Continuing that SAME recurrence for MORE iterations from wherever iteration 25 left off
is mathematically identical to having run that many iterations from the original start -- same
target, same solver state, same trajectory, just carried further. So this script does not need to
re-run the reset from scratch to test 100 iterations: it lets one REAL env.reset() run the normal 25
iterations (capturing the ground-truth grasp_indices exactly like diagnose_grasp_orientation_error.py
does), measures the error at that point, then manually continues the identical loop body for 75 more
iterations (mirroring events.py's own write pattern verbatim, including the post-loop limit wrap) and
re-measures. If the error drops sharply, the solver just needed more steps -- a "not converged" verdict,
fixed by turning a knob. If it does not drop, the target is unreachable from this IK's starting
configuration and this script also reports per-env joint-limit margins to say why (limits, or a
config close enough to singular that DLS stalls).

Does not touch events.py -- this is a controlled, parameterized replay of its exact math, not a
change to the shared consumer. No fix applied here; diagnostic only, per "confirm before changing."

Run (one Isaac process; never via uwlab.sh):
    PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl" \\
        timeout -s KILL 300 <python> -u scripts_v2/tools/diagnose_ik_convergence.py \\
        --task OmniReset-UR5eDelto-ObjectRestingEEGrasped-v0 --num_envs 512 --extra_iterations 75 \\
        env.scene.insertive_object=leg200mm env.scene.receptive_object=onelegfixture
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test IK convergence: more iterations vs. an unreachable target.")
parser.add_argument("--task", type=str, default="OmniReset-UR5eDelto-ObjectRestingEEGrasped-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument(
    "--extra_iterations", type=int, default=75,
    help="Additional damped-IK iterations beyond the event's own 25 (25+this = the 'more iterations' trial).",
)
parser.add_argument(
    "--step_factor", type=float, default=0.25,
    help="Damping factor for the EXTRA iterations only (the event's own first 25 always use its hardcoded"
    " 0.25). 1.0 = no damping, take the full one-shot IK solution each extra iteration.",
)
parser.add_argument("--finger_reach_m", type=float, default=0.10)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

from uwlab_tasks.manager_based.manipulation.omnireset.mdp import events as omnireset_events  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    obj = env.scene["insertive_object"]
    N = env.num_envs
    device = env.device
    env_ids = torch.arange(N, device=device)

    ee_cfg = SceneEntityCfg("robot", body_names="rl_dg_mount")
    ee_cfg.resolve(env.scene)
    palm_id = ee_cfg.body_ids[0]

    env.reset()  # untraced settle reset

    term = env.event_manager.get_term_cfg("reset_end_effector_pose_from_grasp_dataset").func
    G = term.rel_positions.shape[0]

    # -- capture the ACTUAL sampled grasp_indices during the real (25-iteration) reset, same
    # ground-truth method as diagnose_grasp_orientation_error.py: a runtime-only wrap of
    # torch.randint for the duration of this one call, filtered on high==G (distinguishes this
    # event's own sampling call from any other reset-mode event's), restored immediately after.
    _real_randint = torch.randint
    captured = {}

    def _spying_randint(*args, **kwargs):
        result = _real_randint(*args, **kwargs)
        high = args[1] if len(args) > 1 else kwargs.get("high")
        if high == G:
            captured["grasp_indices"] = result.clone()
        return result

    torch.randint = _spying_randint
    try:
        env.reset()  # the MEASURED reset: runs the REAL 25-iteration event, unmodified
    finally:
        torch.randint = _real_randint

    if "grasp_indices" not in captured:
        raise RuntimeError("Did not capture grasp_indices -- check reset_types/probs for this task.")
    grasp_indices = captured["grasp_indices"]

    object_pos_w = obj.data.root_pos_w.clone()
    object_quat_w = obj.data.root_quat_w.clone()
    true_target_pos_w, true_target_quat_w = math_utils.combine_frame_transforms(
        object_pos_w, object_quat_w, term.rel_positions[grasp_indices], term.rel_quaternions[grasp_indices]
    )

    def measure(label: str) -> tuple[torch.Tensor, torch.Tensor]:
        palm_pos_w = robot.data.body_pos_w[:, palm_id, :]
        palm_quat_w = robot.data.body_quat_w[:, palm_id, :]
        pos_err_mm = torch.linalg.norm(palm_pos_w - true_target_pos_w, dim=-1) * 1000.0
        quat_err = math_utils.quat_mul(math_utils.quat_inv(true_target_quat_w), palm_quat_w)
        angle_err_deg = 2.0 * torch.acos(quat_err[:, 0].abs().clamp(max=1.0)) * 180.0 / math.pi
        print(
            f"[diag] {label} position error (mm): min {pos_err_mm.min():.2f}  median {pos_err_mm.median():.2f}"
            f"  mean {pos_err_mm.mean():.2f}  max {pos_err_mm.max():.2f}  p95 {torch.quantile(pos_err_mm, 0.95):.2f}",
            flush=True,
        )
        print(
            f"[diag] {label} orientation error (deg): min {angle_err_deg.min():.2f}  median {angle_err_deg.median():.2f}"
            f"  mean {angle_err_deg.mean():.2f}  max {angle_err_deg.max():.2f}"
            f"  p95 {torch.quantile(angle_err_deg, 0.95):.2f}",
            flush=True,
        )
        return pos_err_mm, angle_err_deg

    print(f"\n=== AT 25 ITERATIONS (the event's own default, unmodified) ===", flush=True)
    pos_err_25, ang_err_25 = measure("25-iter")

    # -- CONTINUE the identical recurrence for extra_iterations more steps. Mirrors events.py's loop
    # body verbatim: apply_actions() re-solves the IK fresh from the CURRENT joint_pos toward the
    # SAME fixed target (process_actions was already called once, inside the real event, before its
    # loop started -- the IK controller's command is still set), then the 25%-damped write.
    arm_ids = term.joint_ids
    n_arm = term.n_joints
    for _ in range(args_cli.extra_iterations):
        term.solver.apply_actions()
        delta = args_cli.step_factor * (
            robot.data.joint_pos_target[env_ids][:, arm_ids] - robot.data.joint_pos[env_ids][:, arm_ids]
        )
        robot.write_joint_state_to_sim(
            position=(delta + robot.data.joint_pos[env_ids][:, arm_ids]),
            velocity=torch.zeros((N, n_arm), device=device),
            joint_ids=arm_ids,
            env_ids=env_ids,
        )
    omnireset_events._wrap_joints_into_limits(robot, arm_ids, env_ids)
    env.sim.forward()  # refresh body_pos_w/body_quat_w from the joint state just written -- no step

    total_iters = 25 + args_cli.extra_iterations
    print(f"\n=== AT {total_iters} ITERATIONS (continued from the same 25-iter state, same target) ===", flush=True)
    pos_err_total, ang_err_total = measure(f"{total_iters}-iter")

    print("\n=== IMPROVEMENT ===", flush=True)
    print(
        f"[diag] median position error: {pos_err_25.median():.2f} mm -> {pos_err_total.median():.2f} mm"
        f"  ({(1 - pos_err_total.median()/pos_err_25.median())*100:.1f}% reduction)",
        flush=True,
    )
    print(
        f"[diag] median orientation error: {ang_err_25.median():.2f} deg -> {ang_err_total.median():.2f} deg"
        f"  ({(1 - ang_err_total.median()/ang_err_25.median())*100:.1f}% reduction)",
        flush=True,
    )

    # -- if the error did NOT meaningfully fall, report per-env joint-limit margins to say why:
    # a target parked against a limit, or a configuration close enough to singular that DLS stalls
    # regardless of iteration count, would show up as a small margin here.
    limits = robot.data.joint_pos_limits[:, arm_ids]
    pos_after = robot.data.joint_pos[:, arm_ids]
    margin = torch.minimum(pos_after - limits[..., 0], limits[..., 1] - pos_after).min(dim=1).values
    print("\n=== ARM JOINT-LIMIT MARGIN AT FINAL (100-iter) STATE ===", flush=True)
    print(
        f"[diag] margin (rad, min over 6 joints): min {margin.min():.4f}  median {margin.median():.4f}"
        f"  mean {margin.mean():.4f}  p5 {torch.quantile(margin, 0.05):.4f}",
        flush=True,
    )
    n_at_limit = int((margin < 0.01).sum())
    print(f"[diag] envs within 0.01 rad of an arm joint limit: {n_at_limit}/{N}", flush=True)

    # -- per-env: did MORE iterations help, and does that correlate with anything about the target?
    improved = pos_err_25 - pos_err_total
    print("\n=== TOP 10 ENVS: LARGEST REMAINING ERROR AFTER 100 ITERATIONS ===", flush=True)
    worst = torch.argsort(pos_err_total, descending=True)[:10]
    print(f"{'env':>5} {'pos_err_25':>11} {'pos_err_100':>12} {'improved_mm':>12} {'margin_rad':>11}", flush=True)
    for i in worst.tolist():
        print(
            f"{i:5d} {pos_err_25[i]:11.2f} {pos_err_total[i]:12.2f} {improved[i]:12.2f} {margin[i]:11.4f}",
            flush=True,
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
