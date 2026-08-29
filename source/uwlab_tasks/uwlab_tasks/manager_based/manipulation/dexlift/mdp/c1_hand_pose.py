# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartesian palm-pose IK reset for the C1 (free, arbitrary) rung.

RESET_SPEC_V2.md sec 1 C1, sec 1a (frames), sec 2 R1. V2_POSE_FINDINGS.md F10. Off by default;
wired only when ``DEXRESET_C1_HAND=1`` is staged -- see
``dexlift_ur5e_delto_env_cfg._apply_c1_hand_pose_stage``, which is the only place that constructs
the :class:`EventTermCfg` this module's term runs under.

WHY A NEW EVENT, NOT ``reset_end_effector_round_fixed_asset`` AS-IS. That function
(``omnireset/mdp/events.py``) IS the general Cartesian-pose-to-joints IK reset already used across
this codebase -- ``ObjectAnywhereEEAnywhereEventCfg`` and
``ObjectPartiallyAssembledEEAnywhereEventCfg`` in
``omnireset/config/ur5e_robotiq_2f85/reset_states_cfg.py``, the factory-extension env, and the
sim2real fk-pair collector all drive it. Its MECHANICS are reused here verbatim: the same
``DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")``, the
same damped iterate-and-teleport loop (``ik_step_size`` fraction of the gap, ``ik_iterations``
times), the same ``_wrap_joints_into_limits`` post-processing (imported, not re-derived). What it
CANNOT express is C1's specific requirement: "palm pointing downwards, +-45 deg variation applied
PER AXIS ABOUT THE PALM-DOWN NOMINAL". ``reset_end_effector_round_fixed_asset`` samples roll/pitch/
yaw as an ABSOLUTE world-frame Euler target (``quat_from_euler_xyz(roll, pitch, yaw)`` with no
nominal to perturb around), which has no way to express "about a nominal" when that nominal is an
arbitrary, non-axis-aligned quaternion -- and the DELTO's palm-down posture is exactly that: its own
approach axis (``gripper_approach_direction`` = [0.2582, 0.4717, 0.8431], off every basis plane,
``Ur5eDelto/metadata.yaml``) needs a genuine 3D rotation to bring to world -Z, not a single-axis
pitch shift. So the POSE-SAMPLING half is new here; the SOLVE half is the same mechanics, reused.

R1 / IK-FOR-HELD-STATES. RESET_SPEC_V2.md sec 2 R1 forbids IK for the HELD reset states (C2/C3/C4)
because those must be reached by a policy and held against gravity, not placed by a solver. C1 is
explicitly exempt from that constraint: nothing is held in C1 (it is free-space sampling, the leg's
own pose is independent and untouched by this term), so an IK-placed hand does not fall under R1's
prohibition.

POST-SOLVE GATE (critic review of the first version of this file, commit 1654e2c). The original
event wrote whatever the damped IK converged to, unchecked. Measured on the H100 (DL_H100, 256
envs, 2 forced resets, n=512 each): 17-19% of resets landed with the ACHIEVED palm height outside
the commanded ``z_range`` -- including a minimum of -0.317 m, i.e. the palm ended up BELOW the
tabletop -- and 4-7% outside ``xy_half_width``, with the violation rate HIGHER at the tighter band
(6.6% at +-0.10 m vs 4.1% at +-0.15 m), ruling out a proportional-to-box-size explanation and
pointing at absolute-scale IK divergence (commanded-vs-achieved position residual up to 678-693mm
was observed). See ``c1_hand_pose_core.py``'s ``DEFAULT_MAX_POS_ERR_MM_RAW`` comment for exactly
where those thresholds were chosen from, and :func:`c1_hand_pose_core.ik_gate_pass` for the full
gate (IK-quality residual + joint-limit margin + the achieved-pose-in-band check that is what
actually drives violations to zero -- residual alone under-predicts band membership on this same
data, so it is not relied on by itself).

Retries are BOUNDED (``max_retries``) and exhaustion is never silent: an env that never passes the
gate keeps its BEST-OF-ATTEMPTS state (lowest commanded-vs-achieved position residual seen across
every attempt, not just the LAST one, so an unlucky final resample cannot make an exhausted env
worse than an earlier attempt already was) and the exhaustion count is printed every reset it
happens on -- a silent fallback to whatever the last attempt produced would reproduce the original
defect with extra steps in between.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

from uwlab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils as omnireset_utils
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import _wrap_joints_into_limits

from .c1_hand_pose_core import C1HandPoseStage, RetryAttemptCounter, ik_gate_pass, quat_from_two_vectors

_WORLD_DOWN = (0.0, 0.0, -1.0)


class reset_end_effector_c1_hand_pose(ManagerTermBase):
    """Sample a hand pose (XY about a fixed anchor, height above the work surface, palm-down +- a
    per-axis tilt) and solve it to arm/wrist joints with differential IK.

    Touches ONLY the joints ``robot_ik_cfg`` resolves (the arm + wrist, matching this env's own OSC
    action term's ``joint_names=["shoulder.*","elbow.*","wrist.*"]``) -- fingers are left to
    whatever ``reset``-mode term already jitters them (``reset_finger_root_joints`` /
    ``reset_robot_joints``), which is the DexSuite-style finger randomization RESET_SPEC_V2.md's C1
    asks for; nothing new is needed there.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        robot_ik_cfg: SceneEntityCfg = cfg.params["robot_ik_cfg"]

        self.robot: Articulation = env.scene[robot_ik_cfg.name]
        self.joint_ids: list[int] | slice = robot_ik_cfg.joint_ids
        self.n_joints: int = self.robot.num_joints if isinstance(self.joint_ids, slice) else len(self.joint_ids)
        # Resolved by the framework before __init__ runs (SceneEntityCfg params in a class-based
        # reset-mode event's cfg get body_names -> body_ids resolution the same way asset_cfg does
        # elsewhere in this codebase, e.g. terminations.py's ``ee_body_idx``); PALM_BODY names one
        # exact body, so this is always a length-1 list.
        self.palm_body_idx: int = robot_ik_cfg.body_ids[0]
        # Cumulative, printed alongside every per-call exhaustion count so a run that only ever
        # sees a HANDFUL of exhaustions scattered thinly across many resets is still visible in
        # the log without a reader having to grep-and-sum every line themselves (R5: read back
        # from the log, not assumed).
        self._cumulative_calls = 0
        self._cumulative_exhausted = 0
        # bead dr-sj6.21 -- R2 accounting (RESET_SPEC_V2.md R2-pinned: accepted/attempted, every
        # attempt including held-state/gate deaths counted in the denominator). The counting rule
        # itself lives in ``RetryAttemptCounter`` (``c1_hand_pose_core.py``), NOT restated here, so
        # it has one implementation and is unit-testable without Isaac -- see that class's own
        # docstring. This term only calls ``start_round``/``end_call`` at the right points in the
        # loop below and reads ``.attempted``/``.accepted`` for the printed cumulative totals.
        self._retry_counter = RetryAttemptCounter()

        robot_ik_solver_cfg = DifferentialInverseKinematicsActionCfg(
            asset_name=robot_ik_cfg.name,
            joint_names=robot_ik_cfg.joint_names,  # type: ignore
            body_name=robot_ik_cfg.body_names,  # type: ignore
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
        )
        self.solver: DifferentialInverseKinematicsAction = robot_ik_solver_cfg.class_type(robot_ik_solver_cfg, env)  # type: ignore

        # Palm-down nominal orientation, in the SAME body frame ("mount-local", i.e. the IK
        # target body's own frame -- see PALM_BODY = "rl_dg_mount") and computed with the SAME
        # formula the orient_down success gate already scores against
        # (``omnireset/mdp/terminations.py::check_reset_state_success``, 60-degree cone about world
        # -Z), read from the SAME metadata key
        # (``gripper_approach_direction``, duplicated onto ``Ur5eDelto/metadata.yaml`` for exactly
        # this reason -- see ``delto_cfg.py``'s own docstring on why the robot-level file, not the
        # hand-level one, is what every DELTO reset-state/termination event reads at runtime).
        usd_path = self.robot.cfg.spawn.usd_path
        metadata = omnireset_utils.read_metadata_from_usd_directory(usd_path)
        approach_local = torch.tensor(
            metadata.get("gripper_approach_direction"), dtype=torch.float32, device=env.device
        )
        approach_local = approach_local / torch.linalg.vector_norm(approach_local)
        world_down = torch.tensor(_WORLD_DOWN, dtype=torch.float32, device=env.device)
        self.nominal_quat: torch.Tensor = quat_from_two_vectors(approach_local, world_down)  # (4,) wxyz

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        robot_ik_cfg: SceneEntityCfg,
        anchor_xy_root: tuple[float, float],
        z_range: tuple[float, float],
        xy_half_width: float,
        tilt: float,
        max_pos_err_m: float,
        max_ori_err_rad: float,
        min_joint_margin_rad: float,
        max_retries: int,
        ik_iterations: int = 10,
        ik_step_size: float = 0.25,
    ) -> None:
        # ``robot_ik_cfg`` is consumed once in __init__ (matching
        # ``reset_end_effector_round_fixed_asset``'s own convention of taking it in both places);
        # it is not re-read here.
        n_total = env.num_envs
        device = env.device

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w

        # ANCHOR: a fixed point at (anchor_xy_root, WORK_SURFACE_Z) in the robot's OWN root frame,
        # transformed to world with the robot's LIVE root pose -- exactly the same
        # combine_frame_transforms idiom ``ObjectUniformPoseCommand`` and ``assembly_keypoints.Offset``
        # use, so this correctly incorporates BASE_LINK_AUTHORED_YAW without assuming it. Root-frame
        # z = 0 is the work surface (see WORK_SURFACE_Z's own docstring: stated in the robot's own
        # root frame), so this anchor's WORLD z is the root's own world height with no further
        # correction -- and NO leg-tip offset applies, because this is a HAND pose, not a leg pose
        # (RESET_SPEC_V2.md sec 1a trap 1's 106.203 mm is ``SquareTableLeg200mmDecomp``'s own
        # assembled-offset and has nothing to do with the hand).
        anchor_pos_b = torch.tensor([anchor_xy_root[0], anchor_xy_root[1], 0.0], device=device).expand(n_total, 3)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n_total, 4)
        anchor_pos_w, _ = math_utils.combine_frame_transforms(root_pos_w, root_quat_w, anchor_pos_b, identity_quat)

        stage = C1HandPoseStage(
            z_lo=z_range[0],
            z_hi=z_range[1],
            xy_half_width=xy_half_width,
            tilt=tilt,
            ik_iterations=ik_iterations,
            max_pos_err_m=max_pos_err_m,
            max_ori_err_rad=max_ori_err_rad,
            min_joint_margin_rad=min_joint_margin_rad,
            max_retries=max_retries,
        )

        # POST-SOLVE GATE, bounded retries. Bookkeeping lives in ``env_ids``' OWN index space
        # (0..m-1), not the full env batch -- ``pending_local`` narrows every attempt to the envs
        # that have not yet passed; ``best_*`` tracks the lowest-residual attempt seen so far PER
        # ENV so an exhausted env's final written state is its best attempt, not merely its last
        # one. See this class's module docstring, "POST-SOLVE GATE", for the measured defect this
        # repairs and why the gate also checks the achieved pose against the band directly, not
        # only the IK residual.
        m = env_ids.numel()
        best_pos_err = torch.full((m,), float("inf"), device=device)
        best_joint_pos = torch.zeros((m, self.n_joints), device=device)
        pending_local = torch.arange(m, device=device)

        self._cumulative_calls += 1

        for _attempt in range(max_retries + 1):
            if pending_local.numel() == 0:
                break
            # bead dr-sj6.21: count THIS round's attempts before any of them can resolve -- see
            # RetryAttemptCounter.start_round's own docstring for why this must happen here.
            self._retry_counter.start_round(int(pending_local.numel()))
            pending_global = env_ids[pending_local]

            # XY/Z/orientation sampled for the FULL batch every attempt -- matches
            # ``reset_end_effector_round_fixed_asset``'s own convention of driving the solver over
            # every env each call ("for those non_reset_id, we will let ik solve for its current
            # position") so a partially-shaped command tensor is never sent to the action term;
            # only ``pending_global`` ever gets a physical write below, so envs outside it are
            # unaffected regardless of what they were sampled toward this attempt.
            xy = math_utils.sample_uniform(-xy_half_width, xy_half_width, (n_total, 2), device=device)
            z = math_utils.sample_uniform(z_range[0], z_range[1], (n_total, 1), device=device)
            pos_w = anchor_pos_w + torch.cat([xy, z], dim=-1)

            # ORIENTATION: an independent per-axis (roll, pitch, yaw) perturbation, +-tilt each,
            # composed as an EXTRINSIC world-frame rotation ON TOP OF the palm-down nominal --
            # quat_mul(perturb, nominal) applies ``nominal`` first (approach axis -> world -Z) and
            # then rotates the RESULT by ``perturb`` about world axes, which is "+-45 deg variation
            # applied per-axis about the palm-down nominal" read literally. See
            # ``c1_hand_pose_core.worst_case_composed_angle_rad`` for why this composition's WORST
            # CASE exceeds ``tilt`` -- printed in the staging banner, never gated on here (that
            # would silently substitute a cone bound for the per-axis one the spec states).
            rpy = math_utils.sample_uniform(-tilt, tilt, (n_total, 3), device=device)
            perturb_quat = math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
            nominal_quat = self.nominal_quat.expand(n_total, 4)
            quat_w = math_utils.quat_mul(perturb_quat, nominal_quat)

            pos_b, quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, pos_w, quat_w)
            self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))

            # Damped iterate-and-teleport loop, copied verbatim from
            # ``reset_end_effector_round_fixed_asset`` (defaults preserve ITS shipped convergence:
            # 0.75^10 ~= 5.6% residual) -- restricted to ``pending_global``, not the full batch, so
            # an env that already passed on an earlier attempt is never touched again.
            for _ in range(ik_iterations):
                self.solver.apply_actions()
                delta_joint_pos = ik_step_size * (
                    self.robot.data.joint_pos_target[pending_global] - self.robot.data.joint_pos[pending_global]
                )
                self.robot.write_joint_state_to_sim(
                    position=(delta_joint_pos + self.robot.data.joint_pos[pending_global])[:, self.joint_ids],
                    velocity=torch.zeros((pending_global.numel(), self.n_joints), device=device),
                    joint_ids=self.joint_ids,
                    env_ids=pending_global,  # type: ignore
                )
            # The differential IK is limit-unaware; wrap any wound-past-the-limit joint back into
            # range before PhysX has to resolve a violated limit -- same reasoning and same helper
            # as ``reset_end_effector_round_fixed_asset``, imported rather than copied.
            _wrap_joints_into_limits(self.robot, self.joint_ids, pending_global)

            # -- measure ACHIEVED vs COMMANDED for this attempt, pending envs only.
            achieved_pos_w = self.robot.data.body_pos_w[pending_global, self.palm_body_idx]
            achieved_quat_w = self.robot.data.body_quat_w[pending_global, self.palm_body_idx]
            cmd_pos_w = pos_w[pending_global]
            cmd_quat_w = quat_w[pending_global]

            pos_err = torch.linalg.norm(achieved_pos_w - cmd_pos_w, dim=-1)
            quat_err = math_utils.quat_mul(math_utils.quat_inv(cmd_quat_w), achieved_quat_w)
            ori_err = 2.0 * torch.acos(quat_err[:, 0].abs().clamp(max=1.0))

            limits = self.robot.data.joint_pos_limits[pending_global][:, self.joint_ids]
            jp = self.robot.data.joint_pos[pending_global][:, self.joint_ids]
            joint_margin = torch.minimum(jp - limits[..., 0], limits[..., 1] - jp).min(dim=1).values

            height = achieved_pos_w[:, 2] - root_pos_w[pending_global, 2]
            dx = achieved_pos_w[:, 0] - anchor_pos_w[pending_global, 0]
            dy = achieved_pos_w[:, 1] - anchor_pos_w[pending_global, 1]

            ok = ik_gate_pass(pos_err, ori_err, joint_margin, height, dx, dy, stage)

            # -- best-of-attempts bookkeeping (lowest position residual wins), independent of
            # whether this attempt passed.
            improved = pos_err < best_pos_err[pending_local]
            improved_local = pending_local[improved]
            best_pos_err[improved_local] = pos_err[improved]
            best_joint_pos[improved_local] = self.robot.data.joint_pos[pending_global][improved][:, self.joint_ids]

            pending_local = pending_local[~ok]

        # bead dr-sj6.21: whatever is NOT in ``pending_local`` now passed ``ik_gate_pass`` within
        # budget -- accepted, by construction, before the exhaustion fallback below ever runs. See
        # RetryAttemptCounter.end_call's own docstring.
        self._retry_counter.end_call(m, int(pending_local.numel()))

        # -- retry budget exhausted for whatever remains in ``pending_local``: restore each one's
        # BEST attempt (may already equal its last, in which case this is a harmless re-write) and
        # report the count LOUDLY, every time it happens -- never a silent fallback to an ungated
        # write.
        self._cumulative_exhausted += pending_local.numel()
        if pending_local.numel() > 0:
            exhausted_global = env_ids[pending_local]
            self.robot.write_joint_state_to_sim(
                position=best_joint_pos[pending_local],
                velocity=torch.zeros((pending_local.numel(), self.n_joints), device=device),
                joint_ids=self.joint_ids,
                env_ids=exhausted_global,  # type: ignore
            )
            print(
                f"[dexlift] C1_HAND gate: {pending_local.numel()}/{m} envs exhausted"
                f" max_retries={max_retries} without meeting the post-solve gate this reset"
                f" (cumulative across {self._cumulative_calls} reset() calls on this term so far:"
                f" {self._cumulative_exhausted} exhausted envs total); using each one's"
                " best-of-attempts state (lowest commanded-vs-achieved position residual across all"
                " attempts) rather than an ungated write. This is a finding about the C1 workspace,"
                " not a bug: some sampled targets are genuinely unreachable within"
                " max_pos_err_m/max_ori_err_rad/min_joint_margin_rad of the exact commanded pose.",
                flush=True,
            )

        # bead dr-sj6.21 -- R2 ACCOUNTING, PRINTED UNCONDITIONALLY EVERY CALL (not behind a debug
        # flag): RESET_SPEC_V2.md's R2-pinned yield is accepted/attempted, denominator including
        # every attempt that consumed compute. ``attempted`` here is the sum of every retry-loop
        # iteration's pending-env count above, NOT ``_cumulative_calls`` (a call count) times
        # envs-per-call -- a caller does not need to know envs-per-call to recover the yield from
        # this line, only to read it. Cumulative and monotonic, so the LAST such line printed
        # before a generation run ends already states the run's own totals.
        attempted = self._retry_counter.attempted
        accepted = self._retry_counter.accepted
        yield_str = f"{accepted / attempted:.4f}" if attempted > 0 else "n/a (0 attempts)"
        print(
            f"[dexlift] C1_HAND R2: cumulative accepted={accepted} attempted={attempted}"
            f" (yield={yield_str})"
            " -- attempted counts every IK attempt including retries (RESET_SPEC_V2.md R2's"
            " pinned denominator); accepted counts envs that met ik_gate_pass within max_retries"
            " (excludes the exhausted-env best-of-attempts fallback above, which is not accepted)"
            f" -- across {self._cumulative_calls} reset() calls so far"
            f" ({self._cumulative_exhausted} envs exhausted cumulative).",
            flush=True,
        )
