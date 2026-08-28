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

from .c1_hand_pose_core import quat_from_two_vectors

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
        ik_iterations: int = 10,
        ik_step_size: float = 0.25,
    ) -> None:
        # ``robot_ik_cfg`` is consumed once in __init__ (matching
        # ``reset_end_effector_round_fixed_asset``'s own convention of taking it in both places);
        # it is not re-read here.
        n = env.num_envs
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
        anchor_pos_b = torch.tensor([anchor_xy_root[0], anchor_xy_root[1], 0.0], device=device).expand(n, 3)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)
        anchor_pos_w, _ = math_utils.combine_frame_transforms(root_pos_w, root_quat_w, anchor_pos_b, identity_quat)

        # XY/Z jitter is added directly in WORLD frame (axis-aligned box), same as
        # ``reset_end_effector_round_fixed_asset`` adds its own ``pose_range_b`` samples on top of
        # its anchor -- RESET_SPEC_V2.md does not require the box to be aligned with the robot's
        # own (yawed) root frame, only that it be +-xy_half_width in XY and z_range above the
        # surface, both of which this satisfies regardless of box orientation.
        xy = math_utils.sample_uniform(-xy_half_width, xy_half_width, (n, 2), device=device)
        z = math_utils.sample_uniform(z_range[0], z_range[1], (n, 1), device=device)
        pos_w = anchor_pos_w + torch.cat([xy, z], dim=-1)

        # ORIENTATION: an independent per-axis (roll, pitch, yaw) perturbation, +-tilt each, composed
        # as an EXTRINSIC world-frame rotation ON TOP OF the palm-down nominal --
        # quat_mul(perturb, nominal) applies ``nominal`` first (approach axis -> world -Z) and then
        # rotates the RESULT by ``perturb`` about world axes, which is "+-45 deg variation applied
        # per-axis about the palm-down nominal" read literally.
        rpy = math_utils.sample_uniform(-tilt, tilt, (n, 3), device=device)
        perturb_quat = math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
        nominal_quat = self.nominal_quat.expand(n, 4)
        quat_w = math_utils.quat_mul(perturb_quat, nominal_quat)

        pos_b, quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, pos_w, quat_w)
        self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))

        # Damped iterate-and-teleport loop, copied verbatim from
        # ``reset_end_effector_round_fixed_asset`` (defaults preserve ITS shipped convergence:
        # 0.75^10 ~= 5.6% residual).
        for _ in range(ik_iterations):
            self.solver.apply_actions()
            delta_joint_pos = ik_step_size * (
                self.robot.data.joint_pos_target[env_ids] - self.robot.data.joint_pos[env_ids]
            )
            self.robot.write_joint_state_to_sim(
                position=(delta_joint_pos + self.robot.data.joint_pos[env_ids])[:, self.joint_ids],
                velocity=torch.zeros((len(env_ids), self.n_joints), device=device),
                joint_ids=self.joint_ids,
                env_ids=env_ids,  # type: ignore
            )
        # The differential IK is limit-unaware; wrap any wound-past-the-limit joint back into range
        # before PhysX has to resolve a violated limit -- same reasoning and same helper as
        # ``reset_end_effector_round_fixed_asset``, imported rather than copied.
        _wrap_joints_into_limits(self.robot, self.joint_ids, env_ids)
