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
    Ur5eRobotiq2f85RelCartesianOSCEvalCfg,
    Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg,
    Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg,
    Ur5eRobotiq2f85RelCartesianOSCTrainCfg,
)
from .sysid_cfg import SysidEnvCfg


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

    The sysid block in Ur10eLinearGripper/metadata.yaml holds the REAL identified UR10e
    values (chirp + CMA-ES, 2026-07-05; per-joint fit <2 deg) -- the ADR curriculum ramps
    the dynamics toward this robot's measured behavior.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur10e_linear_gripper.EXPLICIT_UR10E_LINEAR_GRIPPER, Ur10eLinearGripperRelativeOSCAction()
        )
        # Measured motor delay (sysid 2026-07-05) is 8 ms = 1 physics step at this env's
        # 120 Hz. The inherited delay_range (0, 1) puts reality at the range BOUNDARY;
        # (0, 2) brackets it (paper Table 2 uses {0,1,2}). UR10e-only override.
        self.events.randomize_arm_sysid.params["delay_range"] = (0, 2)
        # The actuator's max_delay sizes its DelayBuffers (history_length = max_delay);
        # set_time_lag(2) on the inherited max_delay=1 buffers raises ValueError the first
        # time the ADR curriculum reaches scale_progress >= 0.75 and an env draws delay 2
        # -- i.e. hours into the finetune. Must be >= delay_range[1]. (The sysid script
        # handles the same invariant by rebuilding the actuator with max_delay=--delay_max.)
        self.scene.robot.actuators["arm"].max_delay = 2


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
        # Eval at the MEASURED motor delay (sysid: 8 ms = 1 step @ 120 Hz), not a draw
        # from the inherited (0, 1) range -- eval should mirror the real robot.
        self.events.randomize_arm_sysid.params["delay_range"] = (1, 1)
