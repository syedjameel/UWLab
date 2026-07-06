# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e + linear-gripper variants of the OmniReset tasks.

These mirror ``linear_gripper_cfg`` (the UR5e + linear-gripper variant) with the UR10e arm:
subclass the 2F-85 task configs and swap ONLY the robot and the action; everything else is
inherited unchanged. The UR5e/UR10e arms share the exact joint/link naming contract
(``shoulder_*``, ``elbow_joint``, ``wrist_*``, ``wrist_3_link``, ``robotiq_base_link``), so
all joint regexes and ``body_name`` references carry over as-is, and the gripper is the SAME
linear gripper mounted identically -- ``_apply_linear_gripper`` (gripper joint-regex fix +
EEAnywhere pitch shift) is reused verbatim with the UR10e robot/action passed in.

What is UR10e-specific:
* robot: ``IMPLICIT/EXPLICIT_UR10E_LINEAR_GRIPPER`` (local graft USD, UR10e effort/velocity
  limits).
* action: ``Ur10eLinearGripperRelativeOSCAction`` -- same OSC, but ``calibration_dir`` points
  the analytical kinematics at the UR10e's calibrated_joints/link_inertials.

There is NO UR10e grasp-sampling task: grasp sampling is gripper-only (arm-independent), so
``OmniReset-LinearGripper-GraspSampling-v0`` serves both arms.

NOTE (P6): the reset EE / object placement ranges are inherited from the 2F-85 tasks, i.e.
tuned for the UR5e's ~0.85 m reach. The UR10e reaches ~1.3 m; re-check the workspace before
large-scale data generation.

Registered gym ids (mirroring the UR5e linear-gripper ones):
* ``OmniReset-UR10eLinearGripper-ObjectAnywhereEEAnywhere-v0``
* ``OmniReset-UR10eLinearGripper-ObjectRestingEEGrasped-v0``
* ``OmniReset-UR10eLinearGripper-ObjectAnywhereEEGrasped-v0``
* ``OmniReset-UR10eLinearGripper-ObjectPartiallyAssembledEEAnywhere-v0``
* ``OmniReset-UR10eLinearGripper-ObjectPartiallyAssembledEEGrasped-v0``
* ``OmniReset-UR10eLinearGripper-RelCartesianOSC-State-v0`` (+ Finetune / Play / Finetune-Play)
"""

from __future__ import annotations

import uwlab_assets.robots.ur10e_linear_gripper as ur10e_linear_gripper

import uwlab_tasks.manager_based.manipulation.omnireset.mdp as task_mdp

from isaaclab.utils import configclass

from .actions import (
    Ur10eLinearGripperRelativeOSCAction,
    Ur10eLinearGripperRelativeOSCEvalAction,
    Ur10eLinearGripperSysidOSCAction,
)
from .linear_gripper_cfg import _apply_linear_gripper
from .reset_states_cfg import (
    ObjectAnywhereEEAnywhereResetStatesCfg,
    ObjectAnywhereEEGraspedResetStatesCfg,
    ObjectPartiallyAssembledEEAnywhereResetStatesCfg,
    ObjectPartiallyAssembledEEGraspedResetStatesCfg,
    ObjectRestingEEGraspedResetStatesCfg,
)
from .rl_state_cfg import (
    LOCAL_UR10E_DATASET,
    Ur5eRobotiq2f85BoxCenterPaperTrainCfg,
    Ur5eRobotiq2f85CoverCloseRimPaperTrainCfg,
    Ur5eRobotiq2f85ObjectInBoxPaperTrainCfg,
    Ur5eRobotiq2f85RelCartesianOSCEvalCfg,
    Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg,
    Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg,
    Ur5eRobotiq2f85RelCartesianOSCTrainCfg,
)
from .sysid_cfg import SysidEnvCfg


def _repoint_ur10e_resets(cfg) -> None:
    """Repoint the RL reset-state loader at the UR10e-linear reset datasets (separate from the 2F-85's,
    since reset states encode the robot). Re-filter the reset-type mix to the types present on disk."""
    import os as _os

    ev = getattr(cfg.events, "reset_from_reset_states", None)
    if ev is None:
        return
    ev.params["dataset_dir"] = LOCAL_UR10E_DATASET
    pair = task_mdp.utils.compute_pair_dir(
        cfg.scene.insertive_object.spawn.usd_path, cfg.scene.receptive_object.spawn.usd_path
    )
    keep_t, keep_p = [], []
    for rt, p in zip(ev.params["reset_types"], ev.params["probs"]):
        if _os.path.exists(f"{LOCAL_UR10E_DATASET}/Resets/{pair}/resets_{rt}.pt"):
            keep_t.append(rt)
            keep_p.append(p)
    if keep_t:
        s = sum(keep_p)
        ev.params["reset_types"] = keep_t
        ev.params["probs"] = [p / s for p in keep_p]


# ---------------------------------------------------------------------------------------
# System identification (P8 sim2real: CMA-ES closed-loop replay against real trajectories)
# ---------------------------------------------------------------------------------------
@configclass
class Ur10eLinearGripperSysidEnvCfg(SysidEnvCfg):
    """UR10e sysid env: same minimal scene/MDP as the UR5e one, UR10e robot + unscaled OSC.

    The robot swap uses the EXPLICIT (DelayedPD) articulation like the base cfg -- the sysid
    search includes motor delay. No gripper joint-regex fix is needed (the sysid env has no
    grasp events), so this swaps robot + action directly instead of _apply_linear_gripper.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot = ur10e_linear_gripper.EXPLICIT_UR10E_LINEAR_GRIPPER.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.actions = Ur10eLinearGripperSysidOSCAction()


# ---------------------------------------------------------------------------------------
# Reset states (full UR10e + linear gripper)
# ---------------------------------------------------------------------------------------
@configclass
class Ur10eLinearGripperObjectAnywhereEEAnywhereResetStatesCfg(ObjectAnywhereEEAnywhereResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur10e_linear_gripper.IMPLICIT_UR10E_LINEAR_GRIPPER, Ur10eLinearGripperRelativeOSCAction()
        )


@configclass
class Ur10eLinearGripperObjectRestingEEGraspedResetStatesCfg(ObjectRestingEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur10e_linear_gripper.IMPLICIT_UR10E_LINEAR_GRIPPER, Ur10eLinearGripperRelativeOSCAction()
        )


@configclass
class Ur10eLinearGripperObjectAnywhereEEGraspedResetStatesCfg(ObjectAnywhereEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur10e_linear_gripper.IMPLICIT_UR10E_LINEAR_GRIPPER, Ur10eLinearGripperRelativeOSCAction()
        )


@configclass
class Ur10eLinearGripperObjectPartiallyAssembledEEAnywhereResetStatesCfg(
    ObjectPartiallyAssembledEEAnywhereResetStatesCfg
):
    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur10e_linear_gripper.IMPLICIT_UR10E_LINEAR_GRIPPER, Ur10eLinearGripperRelativeOSCAction()
        )


@configclass
class Ur10eLinearGripperObjectPartiallyAssembledEEGraspedResetStatesCfg(
    ObjectPartiallyAssembledEEGraspedResetStatesCfg
):
    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur10e_linear_gripper.IMPLICIT_UR10E_LINEAR_GRIPPER, Ur10eLinearGripperRelativeOSCAction()
        )


# ---------------------------------------------------------------------------------------
# RL state (training / finetune / eval)
# ---------------------------------------------------------------------------------------
@configclass
class Ur10eLinearGripperRelCartesianOSCTrainCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    """Stage 1 training: implicit actuator, no curriculum."""

    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur10e_linear_gripper.IMPLICIT_UR10E_LINEAR_GRIPPER, Ur10eLinearGripperRelativeOSCAction()
        )


@configclass
class Ur10eLinearGripperRelCartesianOSCFinetuneCfg(Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg):
    """Stage 2 finetune: explicit actuator + curriculum (base sets EXPLICIT 2F-85; we override last).

    NOTE: the sysid block in Ur10eLinearGripper/metadata.yaml is a PLACEHOLDER (UR5e values)
    until a real UR10e calibration is run (P8) -- do not trust finetune for sim2real before that.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur10e_linear_gripper.EXPLICIT_UR10E_LINEAR_GRIPPER, Ur10eLinearGripperRelativeOSCAction()
        )


@configclass
class Ur10eLinearGripperRelCartesianOSCEvalCfg(Ur5eRobotiq2f85RelCartesianOSCEvalCfg):
    """Eval after Stage 1: implicit actuator, soft gains."""

    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur10e_linear_gripper.IMPLICIT_UR10E_LINEAR_GRIPPER, Ur10eLinearGripperRelativeOSCAction()
        )


@configclass
class Ur10eLinearGripperRelCartesianOSCFinetuneEvalCfg(Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg):
    """Eval after Stage 2: explicit actuator, stiff eval gains."""

    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur10e_linear_gripper.EXPLICIT_UR10E_LINEAR_GRIPPER, Ur10eLinearGripperRelativeOSCEvalAction()
        )


# ---------------------------------------------------------------------------------------
# Box-assembly PAPER stages (UR10e + linear gripper): the 3 end-to-end pipeline stages.
#   Stage A = box -> table-center target       (BoxCenterPaper)
#   Stage B = object -> box cavity             (ObjectInBoxPaper)
#   Stage C = caprim cover -> box (obj inside) (CoverCloseRimPaper; edge-rim knob-free lid)
# Each subclasses the 2F-85 Paper stage cfg and swaps ONLY the robot + action to the UR10e
# linear gripper via _apply_linear_gripper (which also fixes the gripper joint-regex on the
# grasp-dataset reset event and shifts the EE-orientation pitch band by +pi/2 for the +Z
# approach axis). Object pairs, rewards, success, datasets are inherited unchanged.
# NOTE (P6): reset EE/object placement ranges are UR5e-tuned (~0.85 m reach); the UR10e reaches
# ~1.3 m -- re-validate the reset ranges before large-scale reset-state generation.
# ---------------------------------------------------------------------------------------
@configclass
class Ur10eLinearGripperBoxCenterPaperTrainCfg(Ur5eRobotiq2f85BoxCenterPaperTrainCfg):
    """Stage A (UR10e + linear gripper): box -> table-center target."""

    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur10e_linear_gripper.IMPLICIT_UR10E_LINEAR_GRIPPER, Ur10eLinearGripperRelativeOSCAction()
        )
        _repoint_ur10e_resets(self)


@configclass
class Ur10eLinearGripperObjectInBoxPaperTrainCfg(Ur5eRobotiq2f85ObjectInBoxPaperTrainCfg):
    """Stage B (UR10e + linear gripper): object -> box cavity."""

    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur10e_linear_gripper.IMPLICIT_UR10E_LINEAR_GRIPPER, Ur10eLinearGripperRelativeOSCAction()
        )
        _repoint_ur10e_resets(self)


@configclass
class Ur10eLinearGripperCoverCloseRimPaperTrainCfg(Ur5eRobotiq2f85CoverCloseRimPaperTrainCfg):
    """Stage C (UR10e + linear gripper): edge-rim (caprim) cover -> box with object inside."""

    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur10e_linear_gripper.IMPLICIT_UR10E_LINEAR_GRIPPER, Ur10eLinearGripperRelativeOSCAction()
        )
        _repoint_ur10e_resets(self)
