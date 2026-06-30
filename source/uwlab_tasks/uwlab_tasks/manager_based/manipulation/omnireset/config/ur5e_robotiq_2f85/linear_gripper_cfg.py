# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Linear-gripper variants of the OmniReset tasks (new variant alongside the 2F-85).

These subclass the 2F-85 task configs and swap ONLY the robot (UR5e + custom linear
parallel-jaw gripper) and the gripper action; everything else (objects, events, rewards,
sim settings, the object `variants`) is inherited unchanged. The 2F-85 tasks are untouched.

What differs from the 2F-85 and why:
* robot: ``IMPLICIT_UR5E_LINEAR_GRIPPER`` (train/eval) or ``EXPLICIT_UR5E_LINEAR_GRIPPER``
  (finetune), in place of ``IMPLICIT/EXPLICIT_UR5E_ROBOTIQ_2F85``.
* action: ``Ur5eLinearGripperRelativeOSCAction`` -- the SAME arm OSC (same UR5e arm +
  ``wrist_3_link`` IK body), only the binary gripper sub-action differs.
* gripper joint regex on the grasp-dataset reset event: the 2F-85 uses
  ``["finger_joint", ".*right.*", ".*left.*"]`` (its many passive joints); the linear
  gripper has exactly ``finger_joint`` (driver) + ``right_finger_joint`` (PhysX mimic), and
  ``.*left.*`` matches NOTHING -> would raise. Replaced with the explicit two joint names.

The link names (``robotiq_base_link``, ``wrist_3_link``) are reused by the linear gripper
(renamed to the 2F-85 contract), so all body_name references are inherited unchanged.

Registered gym ids (mirroring the 2F-85 ones):
* ``OmniReset-LinearGripper-GraspSampling-v0``
* ``OmniReset-UR5eLinearGripper-ObjectAnywhereEEAnywhere-v0``
* ``OmniReset-UR5eLinearGripper-ObjectRestingEEGrasped-v0``
* ``OmniReset-UR5eLinearGripper-ObjectAnywhereEEGrasped-v0``
* ``OmniReset-UR5eLinearGripper-ObjectPartiallyAssembledEEAnywhere-v0``
* ``OmniReset-UR5eLinearGripper-ObjectPartiallyAssembledEEGrasped-v0``
* ``OmniReset-UR5eLinearGripper-RelCartesianOSC-State-v0`` (+ Finetune / Play / Finetune-Play)
"""

from __future__ import annotations

import uwlab_assets.robots.ur5e_linear_gripper as ur5e_linear_gripper

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .actions import Ur5eLinearGripperRelativeOSCAction, Ur5eLinearGripperRelativeOSCEvalAction
from .grasp_sampling_cfg import Robotiq2f85GraspSamplingCfg
from .reset_states_cfg import (
    ObjectAnywhereEEAnywhereResetStatesCfg,
    ObjectAnywhereEEGraspedResetStatesCfg,
    ObjectPartiallyAssembledEEAnywhereResetStatesCfg,
    ObjectPartiallyAssembledEEGraspedResetStatesCfg,
    ObjectRestingEEGraspedResetStatesCfg,
)
from .rl_state_cfg import (
    Ur5eRobotiq2f85RelCartesianOSCEvalCfg,
    Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg,
    Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg,
    Ur5eRobotiq2f85RelCartesianOSCTrainCfg,
)

# The linear gripper's actuated joints: finger_joint (driver) + right_finger_joint (PhysX mimic).
_LINEAR_GRIPPER_JOINTS = ["finger_joint", "right_finger_joint"]


def _apply_linear_gripper(cfg, robot, action) -> None:
    """Swap the 2F-85 robot/action for the linear gripper and fix the gripper joint regex.

    Call this AFTER ``super().__post_init__()`` -- the 2F-85 finetune configs set
    ``scene.robot`` inside their own ``__post_init__`` (to the EXPLICIT 2F-85), so we must
    override last to win.
    """
    cfg.scene.robot = robot.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.actions = action
    # EEGrasped reset variants set the gripper joints from the grasp dataset via this event;
    # the EEAnywhere / RL variants have no such event (getattr -> None, skipped).
    ev = getattr(cfg.events, "reset_end_effector_pose_from_grasp_dataset", None)
    if ev is not None:
        ev.params["gripper_cfg"] = SceneEntityCfg("robot", joint_names=_LINEAR_GRIPPER_JOINTS)


# ---------------------------------------------------------------------------------------
# Grasp sampling (gripper-only, like ROBOTIQ_2F85)
# ---------------------------------------------------------------------------------------
@configclass
class LinearGripperGraspSamplingCfg(Robotiq2f85GraspSamplingCfg):
    """Grasp sampling with the custom linear gripper (gripper-only, like ROBOTIQ_2F85)."""

    def __post_init__(self):
        # Swap the gripper-only robot and the binary action before the base configures sim.
        self.scene.robot = ur5e_linear_gripper.LINEAR_GRIPPER.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions = ur5e_linear_gripper.LinearGripperBinaryGripperAction()
        super().__post_init__()


# ---------------------------------------------------------------------------------------
# Reset states (full UR5e + linear gripper)
# ---------------------------------------------------------------------------------------
@configclass
class LinearGripperObjectAnywhereEEAnywhereResetStatesCfg(ObjectAnywhereEEAnywhereResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur5e_linear_gripper.IMPLICIT_UR5E_LINEAR_GRIPPER, Ur5eLinearGripperRelativeOSCAction()
        )


@configclass
class LinearGripperObjectRestingEEGraspedResetStatesCfg(ObjectRestingEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur5e_linear_gripper.IMPLICIT_UR5E_LINEAR_GRIPPER, Ur5eLinearGripperRelativeOSCAction()
        )


@configclass
class LinearGripperObjectAnywhereEEGraspedResetStatesCfg(ObjectAnywhereEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur5e_linear_gripper.IMPLICIT_UR5E_LINEAR_GRIPPER, Ur5eLinearGripperRelativeOSCAction()
        )


@configclass
class LinearGripperObjectPartiallyAssembledEEAnywhereResetStatesCfg(
    ObjectPartiallyAssembledEEAnywhereResetStatesCfg
):
    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur5e_linear_gripper.IMPLICIT_UR5E_LINEAR_GRIPPER, Ur5eLinearGripperRelativeOSCAction()
        )


@configclass
class LinearGripperObjectPartiallyAssembledEEGraspedResetStatesCfg(
    ObjectPartiallyAssembledEEGraspedResetStatesCfg
):
    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur5e_linear_gripper.IMPLICIT_UR5E_LINEAR_GRIPPER, Ur5eLinearGripperRelativeOSCAction()
        )


# ---------------------------------------------------------------------------------------
# RL state (training / finetune / eval)
# ---------------------------------------------------------------------------------------
@configclass
class Ur5eLinearGripperRelCartesianOSCTrainCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    """Stage 1 training: implicit actuator, no curriculum."""

    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur5e_linear_gripper.IMPLICIT_UR5E_LINEAR_GRIPPER, Ur5eLinearGripperRelativeOSCAction()
        )


@configclass
class Ur5eLinearGripperRelCartesianOSCFinetuneCfg(Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg):
    """Stage 2 finetune: explicit actuator + curriculum (base sets EXPLICIT 2F-85; we override last)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur5e_linear_gripper.EXPLICIT_UR5E_LINEAR_GRIPPER, Ur5eLinearGripperRelativeOSCAction()
        )


@configclass
class Ur5eLinearGripperRelCartesianOSCEvalCfg(Ur5eRobotiq2f85RelCartesianOSCEvalCfg):
    """Eval after Stage 1: implicit actuator, soft gains."""

    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur5e_linear_gripper.IMPLICIT_UR5E_LINEAR_GRIPPER, Ur5eLinearGripperRelativeOSCAction()
        )


@configclass
class Ur5eLinearGripperRelCartesianOSCFinetuneEvalCfg(Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg):
    """Eval after Stage 2: explicit actuator, stiff eval gains."""

    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur5e_linear_gripper.EXPLICIT_UR5E_LINEAR_GRIPPER, Ur5eLinearGripperRelativeOSCEvalAction()
        )
