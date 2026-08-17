# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure achieved grasp ORIENTATION error at reset, converted to a fingertip-lever-arm displacement
(bead: RestingEEGrasped velocity-explosion, hypothesis (1) / "check B", 2026-08-17).

BACKGROUND. Every physics-config hypothesis for the 0.83% ObjectRestingEEGrasped acceptance (leg
pair) has been killed: identical USD, identical baked-in collider (convexDecomposition,
MeshCollisionAPI), identical mass (0.12 kg both scenes, confirmed via pxr), identical
collision_props effect, and the one real difference (solver_position_iteration_count 16 vs 4) is
shared by the HEALTHY ObjectAnywhereEEAnywhere path too (57.21%) -- so it cannot be the
differentiator. The PD-target reconciliation fix (already applied, events.py) closed a real defect
but left the velocity distribution statistically unchanged. What survived every kill: robot and
object post-reset velocities co-occur (Pearson ~0.5-0.6), the signature of a genuine contact event.

THIS SCRIPT tests the last standing hypothesis: palm-to-object CENTROID distance measured a healthy
~190 mm at every prior check, but that number is dominated by the fixed palm-to-centroid offset and
says nothing about ORIENTATION accuracy. A small angular error, carried out to the ~100 mm the
DELTO's fingers reach from the palm, can put a fingertip meaningfully inside the object mesh while
the centroid distance still reads fine. If the resulting displacement is on the order of the leg's
15 mm half-extent (0.200 x 0.030 x 0.030 m stock, see dexlift_ur5e_delto_tableleg_env_cfg.py), that
is a real, geometrically concrete penetration candidate.

METHOD, avoiding a source edit: the grasp-dataset event (events.py::reset_end_effector_from_grasp_
dataset) samples a random grasp index PER ENV internally (torch.randint) and does not expose which
index it picked. Rather than monkey-patch the event to leak that index, this script instead finds it
post-hoc: for the ACHIEVED palm pose after env.reset(), and for EVERY one of the (currently ~500+)
grasps in the loaded dataset, it recomputes what that grasp's COMMANDED world pose would have been
against this env's actual (already-teleported) object pose -- using the exact same
math_utils.combine_frame_transforms call the event itself uses -- and takes the candidate with the
SMALLEST position error as "the one that was sampled". This is self-checking: the best-match
position error is reported too, and if IK converged at all (already established true from the
distance measurements) it should be small and unambiguous. The best match's ORIENTATION error is
then the quantity of interest.

Also re-measures the SAME first-physics-step joint-velocity metric as
diagnose_grasp_dataset_reset_velocity.py and correlates it against the orientation-error-derived
displacement -- if this is the mechanism, that correlation should be the strong, positive one none
of the previous candidates produced.

Read-only with respect to the grasp dataset and the task config -- does not change anything, does
not step more than one physics tick, matches "confirm before changing" from the prior round.

PROVENANCE -- FOUR DEAD HYPOTHESES FOR ObjectRestingEEGrasped's ~20 mm / ~11 deg RESIDUAL, MEASURED
2026-08-17, DL_A6000, leg pair (OneLegInsertionFixture__SquareTableLeg200mmDecomp). Recorded here
because every one of these questions has already been re-asked once; the next person should find the
numbers rather than re-run the measurement.

1. STALE PD TARGET (the FIRST hypothesis, fixed regardless of outcome). apply_actions() inside the
   event's 25-iteration loop sets the arm's PD position target to a fresh one-shot IK solution every
   iteration; the manual loop only ever wrote 25% of that gap to the raw joint position, and the
   gripper write had NO target call at all -- both left stale (arm: tail-only, ~0.6 rad mean gap;
   gripper: UNIVERSAL, ~11 rad summed over 20 joints, present in literally every reset measured).
   FIXED in events.py (reset_end_effector_from_grasp_dataset, after the limit wrap and gripper
   write): set_joint_position_target to the final written positions + set_joint_velocity_target to
   zero, for both self.joint_ids and self.gripper_joint_ids, mirroring MultiResetManager._reset_to's
   own pattern. Confirmed working (target gap 0.0000 post-fix, both arm and gripper, 1024 envs).
   BEFORE/AFTER robot joint |v| sum (1 physics step, 1024 envs): mean 1.5713 -> 1.5181, max
   57.4232 -> 50.1425 -- statistically indistinguishable. KEPT ANYWAY: it closes a real, confirmed
   defect (the stale target genuinely reached PhysX every reset, via ManagerBasedEnv.reset()'s own
   post-event scene.write_data_to_sim() flush) and cannot regress any other *EEGrasped consumer,
   since every existing user of this shared event had the identical latent staleness and now none
   do. It simply is not what explains today's residual.

2. UNDER-SOLVED IK (more iterations should help if true). Continued the SAME real reset's recurrence
   (same fixed IK target, so continuing = mathematically identical to a fresh run at the higher
   count) from the event's own 25 iterations out to 100, 512 envs:
     position error (mm):     median 20.04 -> 20.13   (-0.4%, i.e. none)
     orientation error (deg): median 11.28 -> 11.26    (0.2%, i.e. none)
   Several envs got WORSE with more iterations (env 129: 104.5 -> 183.2 mm; env 316: 91.3 -> 143.2mm)
   -- the recurrence is not stalling cleanly, it is oscillating for a subset. Joint limits ruled out
   as the cause: margin at the 100-iteration state, median 1.22 rad, only 1/512 envs within 0.01 rad
   of a limit. DEAD: not a "needs more steps" problem.

3. TUNABLE DAMPING (a larger step factor should help if the 0.25 damping is the defect). Same
   continuation method, this time undamped (step_factor=1.0, full one-shot solution each extra
   iteration) for 75 more iterations, a FRESH 512-env draw:
     position error (mm): median 20.52 -> 20.65 (flat), but mean 25.95 -> 42.53, max 400.47 -> 1209.72
     orientation error (deg): median 11.03 -> 11.11 (flat)
   Median unchanged; TAIL dramatically worse. 9/512 envs pinned exactly at a joint limit
   (margin=0.0000) post-undamped vs 1/512 damped -- a MEANINGFUL MINORITY of the worst cases do hit
   hard limits when pushed harder, which the gentler damped test's own final-state margins had
   masked. DEAD as a fix for the median; partially explanatory for the tail's worst envs only.

4. SYSTEMATIC FRAME OFFSET (a converter/consumer frame mismatch -- e.g. an unaccounted
   DifferentialInverseKinematicsActionCfg.body_offset -- would produce a residual that looks like
   noise in world coordinates but clusters tightly around a non-zero mean once expressed in the
   commanded EE frame). BY SOURCE: neither DifferentialInverseKinematicsActionCfg construction in
   events.py (the ~line-1041 one and reset_end_effector_from_grasp_dataset's own at ~line-1291) sets
   body_offset -- both default to IsaacLab's own `None` ("no offset is applied"). Also checked
   root_pos_w (used by the IK action's own _compute_frame_pose) against root_link_pos_w (used by the
   consumer's robot-base conversion): root_pos_w's own docstring is "Same as root_link_pos_w" -- a
   literal alias, not the separate root_com_pos_w quantity. No offset anywhere in the code. BY
   MEASUREMENT (this script's own SYSTEMATIC-OFFSET TEST sections, 512 envs, ground-truth captured
   grasp_indices -- see the method note above):
     position residual in EE frame (mm):     |mean| 5.18  vs mean per-axis std 24.79  (run 2: 3.13 vs 21.11)
     orientation residual in EE frame (deg): |mean| 0.36  vs mean per-axis std 6.91
   |mean| << std on every axis, both quantities, both runs -- the noise signature, not the offset
   signature (offset would read |mean| >> std, a tight cluster away from zero). DEAD, both by source
   and by the measurement built specifically to catch what neither of us could name in advance.

WHAT SURVIVES, UNMEASURED: the source grasps (Stage-2 LIFT checkpoint, position-only reward, median
69.9 deg off vertical -- see analyze_grasp_orientation_distribution.py) may simply ask this arm for
configurations it cannot reach cleanly from a fresh table-height start, regardless of solver tuning
-- the same FAMILY of problem as ObjectPartiallyAssembledEEGrasped's confirmed orientation-source
incompatibility, manifesting here as solver instability instead of a geometric gate failure. Not
chased further: the remedy is the same either way (a different grasp source, e.g. the Stage-3
pose/reorient checkpoint), and that choice belongs to the user, not to further instrumentation.

Run (one Isaac process; never via uwlab.sh):
    PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl" \\
        timeout -s KILL 300 <python> -u scripts_v2/tools/diagnose_grasp_orientation_error.py \\
        --task OmniReset-UR5eDelto-ObjectRestingEEGrasped-v0 --num_envs 512 \\
        env.scene.insertive_object=leg200mm env.scene.receptive_object=onelegfixture
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure achieved grasp orientation error at reset.")
parser.add_argument("--task", type=str, default="OmniReset-UR5eDelto-ObjectRestingEEGrasped-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument(
    "--finger_reach_m", type=float, default=0.10,
    help="Approximate palm-to-fingertip lever arm (m) used to convert an angular error to a linear displacement.",
)
parser.add_argument(
    "--leg_half_extent_m", type=float, default=0.015,
    help="Leg cross-section half-extent (m) -- the displacement scale that would count as 'inside the mesh'.",
)
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

from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    obj = env.scene["insertive_object"]

    ee_cfg = SceneEntityCfg("robot", body_names="rl_dg_mount")
    ee_cfg.resolve(env.scene)
    palm_id = ee_cfg.body_ids[0]

    # -- settle reset (untraced) so the MEASURED reset below isn't confounded by construction-time state.
    env.reset()

    # -- the grasp-dataset term instance, already carrying the full precomputed grasp set.
    term = env.event_manager.get_term_cfg("reset_end_effector_pose_from_grasp_dataset").func
    G = term.rel_positions.shape[0]
    print(f"[diag] grasp dataset entries: {G}", flush=True)

    # -- CAPTURE THE ACTUAL SAMPLED grasp_indices directly, rather than trusting a post-hoc argmin
    # identification (which this script also computes below for cross-check -- and which, on a first
    # pass, showed enough position-match ambiguity, median ~14mm, to not be trusted alone). The event
    # samples via `torch.randint(0, len(self.rel_positions), (num_envs,), device=env.device)`
    # (events.py) -- a call distinguished from every OTHER reset-mode event's own randint calls by its
    # `high` argument, which equals G (the grasp count), a value no other term's sampling uses.
    # Purely a runtime wrap for the duration of ONE env.reset() call in THIS process; touches no file,
    # restored immediately after, and does not change what any event computes or writes.
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
        env.reset()  # the MEASURED reset
    finally:
        torch.randint = _real_randint

    if "grasp_indices" not in captured:
        raise RuntimeError(
            "Did not capture a torch.randint call with high==G during env.reset() -- either the event"
            " didn't fire this reset (check reset_types/probs) or its sampling call changed shape."
        )
    grasp_indices = captured["grasp_indices"]
    print(f"[diag] captured grasp_indices directly: shape {tuple(grasp_indices.shape)}", flush=True)

    object_pos_w = obj.data.root_pos_w.clone()  # (N,3), already teleported by the earlier reset event
    object_quat_w = obj.data.root_quat_w.clone()  # (N,4)
    palm_pos_w = robot.data.body_pos_w[:, palm_id, :].clone()  # (N,3), ACHIEVED
    palm_quat_w = robot.data.body_quat_w[:, palm_id, :].clone()  # (N,4), ACHIEVED

    N = env.num_envs
    device = env.device

    # -- recompute the COMMANDED world pose for every (env, candidate-grasp) pair, exactly the way
    # the event itself does it (combine_frame_transforms(object, relative) -> world gripper pose).
    obj_pos_exp = object_pos_w.unsqueeze(1).expand(N, G, 3).reshape(N * G, 3)
    obj_quat_exp = object_quat_w.unsqueeze(1).expand(N, G, 4).reshape(N * G, 4)
    rel_pos_exp = term.rel_positions.unsqueeze(0).expand(N, G, 3).reshape(N * G, 3)
    rel_quat_exp = term.rel_quaternions.unsqueeze(0).expand(N, G, 4).reshape(N * G, 4)
    cand_pos_w, cand_quat_w = math_utils.combine_frame_transforms(obj_pos_exp, obj_quat_exp, rel_pos_exp, rel_quat_exp)
    cand_pos_w = cand_pos_w.reshape(N, G, 3)
    cand_quat_w = cand_quat_w.reshape(N, G, 4)

    # -- PRIMARY, GROUND-TRUTH measurement: the ACTUAL sampled index, captured directly above.
    env_arange = torch.arange(N, device=device)
    true_pos_w = cand_pos_w[env_arange, grasp_indices]
    true_quat_w = cand_quat_w[env_arange, grasp_indices]
    true_pos_err_mm = torch.linalg.norm(true_pos_w - palm_pos_w, dim=-1) * 1000.0
    true_quat_err = math_utils.quat_mul(math_utils.quat_inv(true_quat_w), palm_quat_w)
    angle_err = 2.0 * torch.acos(true_quat_err[:, 0].abs().clamp(max=1.0))  # (N,) radians, GROUND TRUTH
    displacement_mm = angle_err * args_cli.finger_reach_m * 1000.0  # linear displacement at the lever arm

    # -- CROSS-CHECK ONLY: the post-hoc argmin-by-position identification this script used before the
    # direct capture was added. Reported to show how much the two methods agree (or don't) -- not
    # used for the headline numbers above.
    pos_err_all = torch.linalg.norm(cand_pos_w - palm_pos_w.unsqueeze(1), dim=-1)  # (N,G)
    best_idx = torch.argmin(pos_err_all, dim=1)  # (N,)
    best_pos_err = pos_err_all.gather(1, best_idx.unsqueeze(1)).squeeze(1) * 1000.0  # mm
    argmin_matches_truth = (best_idx == grasp_indices).float().mean()

    # -- SYSTEMATIC-OFFSET TEST: express the (achieved - commanded) position residual IN THE
    # COMMANDED EE FRAME, not world frame. A residual that is random noise scatters around zero
    # in ANY frame; a residual that is actually a fixed frame-definition mismatch (e.g. a body_offset
    # the converter didn't account for) shows up as a TIGHT CLUSTER around a non-zero mean once
    # expressed in the frame the offset would have been defined in -- even though it looks like
    # patternless noise in world coordinates, because the commanded orientation itself varies
    # env-to-env. quat_apply(quat_inv(commanded_quat), residual_world) rotates the world residual
    # vector into the commanded frame.
    residual_world_m = palm_pos_w - true_pos_w  # (N,3), meters
    residual_ee_frame_mm = math_utils.quat_apply(math_utils.quat_inv(true_quat_w), residual_world_m) * 1000.0

    # -- SAME TEST, ORIENTATION: true_quat_err (= quat_mul(quat_inv(commanded), achieved), already
    # computed above) is BY CONSTRUCTION the achieved-vs-commanded rotation expressed IN the
    # commanded (EE) frame -- no extra rotation needed, unlike the position residual. Convert it to
    # an axis-angle (rotation-vector) representation: a CONSTANT rotational offset shows as a tight
    # cluster of rotation vectors pointing the same way; solver noise shows as scattered directions
    # with no consistent axis. Normalize the quaternion's sign first (q and -q are the same rotation)
    # so an arbitrary sign flip doesn't fake a random axis for what is actually a constant offset.
    q_signed = true_quat_err.clone()
    flip = q_signed[:, 0] < 0
    q_signed[flip] = -q_signed[flip]
    qw_pos = q_signed[:, 0].clamp(-1.0, 1.0)
    qxyz = q_signed[:, 1:4]
    qxyz_norm = torch.linalg.norm(qxyz, dim=-1).clamp(min=1e-8)
    angle_signed = 2.0 * torch.atan2(qxyz_norm, qw_pos)  # (N,), matches angle_err numerically
    axis_unit = qxyz / qxyz_norm.unsqueeze(-1)
    rotvec_ee_frame_deg = axis_unit * angle_signed.unsqueeze(-1) * (180.0 / math.pi)  # (N,3)

    def stats(name: str, x: torch.Tensor) -> None:
        print(
            f"[diag] {name}: min {x.min():.4f}  median {x.median():.4f}  mean {x.mean():.4f}"
            f"  max {x.max():.4f}  p95 {torch.quantile(x, 0.95):.4f}",
            flush=True,
        )

    print("\n=== GROUND-TRUTH POSITION ERROR (captured grasp_indices, sanity check on IK convergence) ===",
          flush=True)
    stats("ground-truth commanded-vs-achieved position error (mm)", true_pos_err_mm)

    print("\n=== SYSTEMATIC-OFFSET TEST: residual expressed in the COMMANDED EE FRAME (mm) ===", flush=True)
    mean_ee = residual_ee_frame_mm.mean(dim=0)
    std_ee = residual_ee_frame_mm.std(dim=0)
    print(f"[diag] per-axis mean (x,y,z): [{mean_ee[0]:.3f}, {mean_ee[1]:.3f}, {mean_ee[2]:.3f}]", flush=True)
    print(f"[diag] per-axis std  (x,y,z): [{std_ee[0]:.3f}, {std_ee[1]:.3f}, {std_ee[2]:.3f}]", flush=True)
    mean_norm = torch.linalg.norm(mean_ee)
    print(f"[diag] |mean| = {mean_norm:.3f} mm  vs mean |std| axis = {std_ee.mean():.3f} mm", flush=True)
    print(
        "[diag] a systematic frame offset reads as |mean| >> std on one or more axes (a tight cluster"
        " away from zero); solver noise reads as |mean| << std (scattered around zero in this frame too).",
        flush=True,
    )

    print("\n=== SYSTEMATIC-OFFSET TEST: ORIENTATION residual, axis-angle in the COMMANDED EE FRAME (deg) ===",
          flush=True)
    mean_rot = rotvec_ee_frame_deg.mean(dim=0)
    std_rot = rotvec_ee_frame_deg.std(dim=0)
    print(f"[diag] per-axis mean (x,y,z): [{mean_rot[0]:.3f}, {mean_rot[1]:.3f}, {mean_rot[2]:.3f}] deg", flush=True)
    print(f"[diag] per-axis std  (x,y,z): [{std_rot[0]:.3f}, {std_rot[1]:.3f}, {std_rot[2]:.3f}] deg", flush=True)
    mean_rot_norm = torch.linalg.norm(mean_rot)
    print(f"[diag] |mean| = {mean_rot_norm:.3f} deg  vs mean |std| axis = {std_rot.mean():.3f} deg", flush=True)
    print(
        "[diag] a constant rotational offset reads as a consistent axis direction -- |mean| >> std;"
        " solver noise reads as scattered rotation vectors -- |mean| << std.",
        flush=True,
    )

    print("\n=== CROSS-CHECK: post-hoc argmin identification vs the captured ground truth ===", flush=True)
    print(f"[diag] argmin picked the SAME index as the ground truth in {argmin_matches_truth:.1%} of envs", flush=True)
    stats("argmin best-match position error (mm)", best_pos_err)
    n_ambiguous = int((best_pos_err > 5.0).sum())
    print(f"[diag] envs with argmin position error > 5 mm (ambiguous match): {n_ambiguous}/{N}", flush=True)

    print("\n=== ACHIEVED ORIENTATION ERROR AT RESET (ground truth) ===", flush=True)
    stats("orientation error (rad)", angle_err)
    stats("orientation error (deg)", angle_err * 180.0 / math.pi)
    stats(f"displacement at {args_cli.finger_reach_m*1000:.0f} mm lever arm (mm)", displacement_mm)
    frac_over_half_extent = (displacement_mm > args_cli.leg_half_extent_m * 1000.0).float().mean()
    print(
        f"[diag] fraction with lever-arm displacement > leg half-extent"
        f" ({args_cli.leg_half_extent_m*1000:.0f} mm): {frac_over_half_extent:.3f}",
        flush=True,
    )

    # -- step ONE physics tick and correlate against the SAME robot joint |v| sum metric used before.
    zero_action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    env.step(zero_action)
    vel_sum_env = robot.data.joint_vel.abs().sum(dim=1)
    stats("robot joint |v| sum (1 step post-reset)", vel_sum_env)

    def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.corrcoef(torch.stack([a.float(), b.float()]))[0, 1])

    print("\n=== CORRELATIONS WITH FIRST-STEP robot joint |v| sum ===", flush=True)
    print(f"[diag] corr(orientation error, vel_sum)              = {pearson(angle_err, vel_sum_env):.4f}", flush=True)
    print(f"[diag] corr(lever-arm displacement, vel_sum)          = {pearson(displacement_mm, vel_sum_env):.4f}", flush=True)
    print(f"[diag] corr(ground-truth position error, vel_sum)     = {pearson(true_pos_err_mm, vel_sum_env):.4f}", flush=True)
    print(f"[diag] corr(argmin best-match position error, vel_sum) = {pearson(best_pos_err, vel_sum_env):.4f}", flush=True)

    worst = torch.argsort(vel_sum_env, descending=True)[:15]
    print("\n=== TOP 15 WORST ENVS: vel_sum vs ground-truth orientation-derived displacement ===", flush=True)
    print(f"{'env':>5} {'vel_sum':>10} {'true_pos_mm':>12} {'orient_deg':>11} {'lever_disp_mm':>14}", flush=True)
    for i in worst.tolist():
        print(
            f"{i:5d} {vel_sum_env[i]:10.3f} {true_pos_err_mm[i]:12.3f}"
            f" {angle_err[i]*180.0/math.pi:11.3f} {displacement_mm[i]:14.3f}",
            flush=True,
        )

    best = torch.argsort(vel_sum_env, descending=False)[:10]
    print("\n=== BEST 10 ENVS: vel_sum vs ground-truth orientation-derived displacement ===", flush=True)
    print(f"{'env':>5} {'vel_sum':>10} {'true_pos_mm':>12} {'orient_deg':>11} {'lever_disp_mm':>14}", flush=True)
    for i in best.tolist():
        print(
            f"{i:5d} {vel_sum_env[i]:10.3f} {true_pos_err_mm[i]:12.3f}"
            f" {angle_err[i]*180.0/math.pi:11.3f} {displacement_mm[i]:14.3f}",
            flush=True,
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
