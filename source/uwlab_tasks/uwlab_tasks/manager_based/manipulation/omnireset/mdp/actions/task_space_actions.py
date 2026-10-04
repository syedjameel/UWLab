# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import torch
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm

from uwlab_assets.robots.ur5e_robotiq_gripper.kinematics import compute_jacobian_analytical

from . import actions_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RelCartesianOSCAction(ActionTerm):
    """Relative Cartesian OSC action term using analytical Jacobian and PD control.

    Matches the real robot's OSC implementation using calibrated analytical kinematics:
        tau = J^T @ (Kp * pose_error + Kd * vel_error)

    No inertial dynamics decoupling. Velocity is computed from J @ dq for consistency
    with the analytical Jacobian. Designed to work with DelayedDCMotor actuator.

    The flow per policy step:
        1. process_actions: scale raw 6-DOF delta, compute desired EE pose
        2. apply_actions (every physics step): compute current state, analytical J,
           PD torques, clamp, and apply as joint effort targets

    Frame convention: both EE pose and analytical Jacobian are in the robot's
    base_link frame (REP-103), consistent with the calibrated USD model.
    """

    cfg: actions_cfg.RelCartesianOSCActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: actions_cfg.RelCartesianOSCActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Resolve joints
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_dof = len(self._joint_ids)
        # Avoid slice-vs-list indexing overhead when all joints match
        if self._num_dof == self._asset.num_joints:
            self._joint_ids = slice(None)

        # Resolve EE body
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for body_name '{self.cfg.body_name}', got {len(body_ids)}: {body_names}"
            )
        self._ee_body_idx = body_ids[0]

        # Controller gains (per-env for domain randomization): Kd = 2 * sqrt(Kp) * damping_ratio
        kp = torch.tensor(cfg.motion_stiffness, device=self.device, dtype=torch.float32)
        damping_ratio = torch.tensor(cfg.motion_damping_ratio, device=self.device, dtype=torch.float32)
        kd = 2.0 * torch.sqrt(kp) * damping_ratio
        # Store defaults (1D) and expand to per-env (N, 6)
        self._kp_default = kp
        self._kd_default = kd
        self._damping_ratio_default = damping_ratio
        self._kp = kp.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._kd = kd.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._torque_max = torch.tensor(cfg.torque_limit, device=self.device, dtype=torch.float32)

        # Action scaling
        self._scale = torch.tensor(cfg.scale_xyz_axisangle, device=self.device, dtype=torch.float32)
        if cfg.input_clip is not None:
            self._input_clip = torch.tensor(cfg.input_clip, device=self.device, dtype=torch.float32)
        else:
            self._input_clip = None

        # Buffers
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._ee_pos_des = torch.zeros(self.num_envs, 3, device=self.device)
        self._ee_quat_des = torch.zeros(self.num_envs, 4, device=self.device)

        # Gravity compensation, off unless UWLAB_OSC_GRAVITY_COMP=1 -- see apply_actions for the
        # measurement that motivates it. `_gravity_fn` is a three-state cache: None = unresolved,
        # False = PhysX exposes no accessor (warn once, run without), otherwise the bound callable.
        self._gravity_comp = os.environ.get("UWLAB_OSC_GRAVITY_COMP", "0") not in ("0", "", "false", "False")
        self._gravity_fn: Callable[[], torch.Tensor] | bool | None = None
        # SIGN AND SCALE, both measured rather than inferred. A three-mode probe on the C3 bank with
        # a null action, fingertip sag at step 120 (negative = falling):
        #     off        -0.114 m      (the shipped defect)
        #     sign=+1    -0.903 m      (adding it DOUBLES the fall -- so the value is not a
        #                               correction to apply on top of what PhysX already does)
        #     sign=-1    +0.059 m      (right direction, but now the arm DRIFTS UPWARD)
        # So -1 over-compensates: the correct term is a fraction of it. UWLAB_OSC_GRAVITY_SIGN is
        # therefore a signed SCALE, not just a sign, and the value that nulls the drift is swept
        # empirically. Anything that leaves a residual drift is a bias the policy must cancel on
        # every step of every episode, which is the whole defect being fixed here.
        self._gravity_sign = float(os.environ.get("UWLAB_OSC_GRAVITY_SIGN", "1"))
        if self._gravity_comp:
            print(
                "[RelCartesianOSCAction] gravity compensation ENABLED "
                f"(UWLAB_OSC_GRAVITY_COMP, sign={self._gravity_sign:+g})",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor):
        """Scale raw 6-DOF deltas and compute desired EE pose for the PD tracker.

        Called once per policy step. The desired pose is held fixed while
        apply_actions recomputes torques at each physics step.
        """
        self._raw_actions[:] = actions
        scaled = actions * self._scale
        if self._input_clip is not None:
            scaled = torch.clamp(scaled, min=self._input_clip[0], max=self._input_clip[1])
        self._processed_actions[:] = scaled

        # Current EE pose in root (base_link) frame
        ee_pos_b, ee_quat_b = self._get_ee_pose_root_frame()

        # Desired position = current + delta
        self._ee_pos_des[:] = ee_pos_b + scaled[:, :3]

        # Desired orientation: axis-angle delta -> quaternion -> compose
        delta_rot = scaled[:, 3:6]
        angle = torch.norm(delta_rot, dim=-1, keepdim=True)
        safe_angle = torch.where(angle > 1e-6, angle, torch.ones_like(angle))
        axis = delta_rot / safe_angle
        axis = torch.where(angle > 1e-6, axis, torch.zeros_like(axis))
        half = angle / 2.0
        delta_quat = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)
        self._ee_quat_des[:] = math_utils.quat_mul(delta_quat, ee_quat_b)

    def apply_actions(self):
        """Compute PD torques using analytical Jacobian and apply as joint efforts.

        Called every physics step (decimation times per policy step).
        """
        # Current state
        ee_pos_b, ee_quat_b = self._get_ee_pose_root_frame()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        joint_vel = self._asset.data.joint_vel[:, self._joint_ids]

        # Analytical Jacobian (base_link frame, matching EE pose frame). calibration_dir=None
        # -> UR5e default; set on the cfg to drive a different arm (e.g. UR10e) from its own metadata.
        jacobian = compute_jacobian_analytical(
            joint_pos, device=str(self.device), usd_dir=self.cfg.calibration_dir
        )

        # EE velocity from J @ dq (consistent with analytical Jacobian)
        ee_vel = torch.bmm(jacobian, joint_vel.unsqueeze(-1)).squeeze(-1)  # (N, 6)

        # Pose error
        pos_error = self._ee_pos_des - ee_pos_b
        quat_error = math_utils.quat_mul(self._ee_quat_des, math_utils.quat_inv(ee_quat_b))
        axis_angle_error = math_utils.axis_angle_from_quat(quat_error)
        pose_error = torch.cat([pos_error, axis_angle_error], dim=-1)  # (N, 6)

        # PD control: tau = J^T @ (Kp * err + Kd * (-vel))
        vel_error = -ee_vel
        task_force = self._kp * pose_error + self._kd * vel_error
        joint_torques = torch.bmm(jacobian.transpose(-1, -2), task_force.unsqueeze(-1)).squeeze(-1)

        # GRAVITY COMPENSATION (opt-in via UWLAB_OSC_GRAVITY_COMP=1).
        # Without this term a zero action commands exactly zero torque: the arm joints carry
        # stiffness 0.0 / damping 0.0 (ur10e_delto.py IMPLICIT_UR10E_DELTO), so J^T @ f is the ONLY
        # torque acting on them. Measured consequence on the training task: with a null action the
        # fingertips sag 91 mm over 60 control steps and the grasped cube is left behind 0.26 m from
        # the tips -- identical at every gripper bias, i.e. the arm, not the hand. 17.4 N of DELTO
        # weight against Kp_z = 200 N/m predicts 87 mm, within 5% of what was measured.
        # `process_actions` also re-latches the desired pose to the MEASURED pose every control step,
        # so the sag accumulated so far never generates a restoring error -- no constant command can
        # mean "stay here". ur10e_delto.py:214-222 concluded "the fix belongs in the action term,
        # not here" after stiff joint gains proved unstable at dt = 1/120; this is that fix.
        # It also matches the deployed robot, whose OSC compensates gravity via setPayload
        # (UR10E_OMNIRESET_GUIDE.md:415), so leaving it off is the sim2real mismatch, not the reverse.
        if self._gravity_comp:
            g = self._generalized_gravity()
            if g is not None:
                joint_torques = joint_torques + self._gravity_sign * g

        joint_torques = torch.clamp(joint_torques, -self._torque_max, self._torque_max)

        self._asset.set_joint_effort_target(joint_torques, joint_ids=self._joint_ids)

    def _generalized_gravity(self) -> torch.Tensor | None:
        """Per-DOF torque that cancels gravity, or None if PhysX does not expose it.

        Resolved once and cached: on a miss the term degrades to the shipped no-compensation
        behaviour with one warning, rather than failing a long training run at step 1.
        """
        if self._gravity_fn is False:
            return None
        if self._gravity_fn is None:
            view = getattr(self._asset, "root_physx_view", None)
            fn = None
            # Measured on Isaac Sim 6 / omni.physics.tensors.api.ArticulationView: the accessor is
            # `get_gravity_compensation_forces`. `get_generalized_gravity_forces` does NOT exist
            # here, so it is tried second purely for forward/backward compatibility.
            for name in ("get_gravity_compensation_forces", "get_generalized_gravity_forces"):
                if view is not None and hasattr(view, name):
                    fn = getattr(view, name)
                    break
            if fn is None:
                print(
                    "[RelCartesianOSCAction] UWLAB_OSC_GRAVITY_COMP=1 but PhysX exposes no "
                    "generalized-gravity accessor on this articulation view; continuing WITHOUT "
                    "gravity compensation (the arm will sag under a null action).",
                    flush=True,
                )
                self._gravity_fn = False
                return None
            self._gravity_fn = fn
        fn = self._gravity_fn
        if not callable(fn):
            return None
        g = fn()
        # The view returns a RAW warp array (warp._src.types.array, shape (num_envs, num_dof)) with
        # NO `.torch` attribute -- unlike an IsaacLab ProxyArray. Relying on `getattr(g, "torch", g)`
        # alone therefore passed the warp array straight through, and indexing it with the list
        # [0..5] of arm joint ids raised "'<' not supported between instances of 'list' and 'int'".
        # Convert explicitly; wp.to_torch shares memory, so this costs no copy.
        if not isinstance(g, torch.Tensor):
            t = getattr(g, "torch", None)
            if t is not None:
                g = t
            else:
                import warp as wp

                g = wp.to_torch(g)
        return g[:, self._joint_ids]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset targets to current EE pose to avoid transients."""
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        ee_pos_b, ee_quat_b = self._get_ee_pose_root_frame()
        self._ee_pos_des[env_ids] = ee_pos_b[env_ids]
        self._ee_quat_des[env_ids] = ee_quat_b[env_ids]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_ee_pose_root_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get EE pose in root (base_link) frame from sim state."""
        ee_pos_w = self._asset.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._ee_body_idx]
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w,
            self._asset.data.root_quat_w,
            ee_pos_w,
            ee_quat_w,
        )
        return ee_pos_b, ee_quat_b
