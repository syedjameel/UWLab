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
from .linear_gripper_cfg import _apply_linear_gripper, _enable_fingertip_floor, _enable_wrist_camera_anchor
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
        _enable_fingertip_floor(self)  # deviation A (thin-object fingertip-vs-table floor)


@configclass
class Ur10eLinearGripperObjectAnywhereEEGraspedResetStatesCfg(ObjectAnywhereEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur10e_linear_gripper.IMPLICIT_UR10E_LINEAR_GRIPPER, Ur10eLinearGripperRelativeOSCAction()
        )
        _enable_fingertip_floor(self)  # deviation A (thin-object fingertip-vs-table floor)


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
        _enable_fingertip_floor(self)  # deviation A (thin-object fingertip-vs-table floor)


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
        _enable_wrist_camera_anchor(self)  # wrist_3 window -> camera settles toward +X (operator)


# Real gripper jaw speed (m/s per jaw joint): the physical gripper takes ~1.0 s for the
# full 68 mm stroke (user-measured 2026-07-12) -> 0.068 m/s. The sim's implicit jaw PD
# does the same stroke in 0.2 s (measured; velocity effectively uncapped at 130), so a
# policy trained without this cap learns to grab-and-go before real jaws would have
# closed (~0.7 s to the 40 mm-cube grip point = ~7 policy steps @ 10 Hz). Applied as the
# jaw velocity_limit_sim on the DEPLOYMENT-MATCHED envs only (finetune, Finetune-Play,
# RGB collection/play) -- the same treatment as the measured motor delay: Stage-1
# train/eval keep the authors' idealized dynamics, the finetune absorbs the real ones.
# Re-tune with a precise stopwatch/frame-count measurement if 1.0 s was approximate.
REAL_GRIPPER_JAW_SPEED = 0.068


def _apply_real_gripper_speed(cfg) -> None:
    """Cap the jaw drives at the measured real jaw speed (copy-on-write: the actuator cfg
    object is shared module-level state; mutating it in place would leak into Stage-1)."""
    cfg.scene.robot.actuators["gripper"] = cfg.scene.robot.actuators["gripper"].replace(
        velocity_limit_sim=REAL_GRIPPER_JAW_SPEED
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
        _apply_real_gripper_speed(self)
        _enable_wrist_camera_anchor(self)  # wrist_3 window -> camera settles toward +X (operator)
        # Motor delay: the CMA-ES "identified delay=4" was never actually simulated (the
        # scripts reset AFTER applying it, re-randomizing the buffers; fixed 2026-07-06).
        # Re-measured by sweeping delay 0..8 over the frozen 24-param fit: RMSE rises
        # monotonically from delay 0 (total 1.02 deg) -> the residual delay PAIRED WITH
        # the metadata sysid params is 0 steps @ 500 Hz (< 2 ms). Reality is delay 0.
        #
        # (0, 2) -> (0, 1) (2026-07-13): the ADR ramps the delay ceiling as
        # round(scale_progress * delay_hi) (see randomize_arm_from_sysid). With delay_hi=2
        # that ceiling DISCRETELY jumps 1 -> 2 exactly at scale_progress 0.75 (round(1.5)=2),
        # which is the wall the finetune got stuck on: once the real gripper-speed cap
        # (1 s stroke) shaved ~2% off the grasp-from-open task, the aggregate could no longer
        # absorb the delay-2 step and success fell below the 0.95 advance threshold every
        # time p reached 0.75 (measured: earlier no-gripper run crossed at task0=0.921 /
        # mean=0.926; slow-gripper run stalled at task0=0.902 / same mean). delay_hi=1 makes
        # the ceiling round(p): 0 for p<0.5, 1 for p>=0.5 -- no jump at 0.75, and since the
        # real delay is 0, delay 1 still over-brackets it. Removes the wall without weakening
        # sim2real below reality. If a future rig genuinely shows >1-step latency, restore
        # (0, 2) and expect this wall back.
        self.events.randomize_arm_sysid.params["delay_range"] = (0, 1)
        # The actuator's max_delay sizes its DelayBuffers (history_length = max_delay) and must
        # be >= delay_range[1]. Kept at 2 (a harmless margin above the delay-1 range) so the
        # buffers stay valid if delay_range is bumped back without touching this line. (The
        # sysid script rebuilds the actuator with max_delay=--delay_max for the same invariant.)
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
        _apply_real_gripper_speed(self)
        # Eval at the MEASURED residual motor delay: the delay sweep over the frozen
        # sysid fit (2026-07-06) is monotonically best at 0 steps @ 500 Hz (< 2 ms), so
        # paired with the metadata sysid params the real arm is delay 0 at this env's
        # 120 Hz too. Pin 0 rather than draw from the inherited (0, 1) -- eval should
        # mirror the real robot. (The earlier (1, 1) pin was based on the unmeasured
        # CMA-ES delay=4 artifact; see the Finetune cfg note.)
        self.events.randomize_arm_sysid.params["delay_range"] = (0, 0)
