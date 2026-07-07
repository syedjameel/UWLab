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
    _paper_stage_box_center,
    _paper_stage_cover_close,
    _paper_stage_object_in_box,
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

    The sysid block in Ur10eLinearGripper/metadata.yaml holds the REAL identified UR10e
    values (chirp + CMA-ES, 2026-07-05; per-joint fit <2 deg) -- the ADR curriculum ramps
    the dynamics toward this robot's measured behavior.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_linear_gripper(
            self, ur10e_linear_gripper.EXPLICIT_UR10E_LINEAR_GRIPPER, Ur10eLinearGripperRelativeOSCAction()
        )
        # Motor delay: the CMA-ES "identified delay=4" was never actually simulated (the
        # scripts reset AFTER applying it, re-randomizing the buffers; fixed 2026-07-06).
        # Re-measured by sweeping delay 0..8 over the frozen 24-param fit: RMSE rises
        # monotonically from delay 0 (total 1.02 deg) -> the residual delay PAIRED WITH
        # the metadata sysid params is 0 steps @ 500 Hz (< 2 ms). (0, 2) here = the paper
        # Table 2 range {0,1,2} @ this env's 120 Hz; reality sits at the low end, which is
        # structurally unavoidable for a nonnegative quantity, and the range brackets any
        # real latency growth from above. UR10e-only override.
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
        # Eval at the MEASURED residual motor delay: the delay sweep over the frozen
        # sysid fit (2026-07-06) is monotonically best at 0 steps @ 500 Hz (< 2 ms), so
        # paired with the metadata sysid params the real arm is delay 0 at this env's
        # 120 Hz too. Pin 0 rather than draw from the inherited (0, 1) -- eval should
        # mirror the real robot. (The earlier (1, 1) pin was based on the unmeasured
        # CMA-ES delay=4 artifact; see the Finetune cfg note.)
        self.events.randomize_arm_sysid.params["delay_range"] = (0, 0)


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


# ---------------------------------------------------------------------------------------
# Box-assembly PAPER stages -- Stage 2 finetune + finetune-eval (UR10e + linear gripper).
# Unlike the Stage-1 Train cfgs above (which subclass the 2F-85 Paper train cfg, so super()
# already applies the pair), these subclass the GENERIC UR10e finetune cfgs to inherit the
# explicit actuator + ADR curriculum + sysid/OSC-gain DR (Finetune) or the fixed-DR stiff
# eval action (FinetuneEval). super() therefore does NOT apply the box-assembly pair, so we
# call the extracted _paper_stage_* helper after it (same order as the Train cfgs: robot swap
# first, then pair/success/scene), then repoint the reset loader at the UR10e datasets.
# The helpers only touch the pair, the reset event, the object-material DR and the
# progress_context reward -- all present in both FinetuneEventCfg and FinetuneEvalEventCfg --
# and augment_box_assembly(scene_only=True) only declares scene entities, so no extra guards
# are needed beyond the getattr guards the helpers already carry.
# ---------------------------------------------------------------------------------------
@configclass
class Ur10eLinearGripperBoxCenterPaperFinetuneCfg(Ur10eLinearGripperRelCartesianOSCFinetuneCfg):
    """Stage A finetune (UR10e + linear gripper): box -> table-center target."""

    def __post_init__(self):
        super().__post_init__()
        _paper_stage_box_center(self)
        _repoint_ur10e_resets(self)


@configclass
class Ur10eLinearGripperObjectInBoxPaperFinetuneCfg(Ur10eLinearGripperRelCartesianOSCFinetuneCfg):
    """Stage B finetune (UR10e + linear gripper): object -> box cavity."""

    def __post_init__(self):
        super().__post_init__()
        _paper_stage_object_in_box(self)
        _repoint_ur10e_resets(self)


@configclass
class Ur10eLinearGripperCoverCloseRimPaperFinetuneCfg(Ur10eLinearGripperRelCartesianOSCFinetuneCfg):
    """Stage C finetune (UR10e + linear gripper): edge-rim (caprim) cover -> box with object inside."""

    def __post_init__(self):
        super().__post_init__()
        _paper_stage_cover_close(self)
        _repoint_ur10e_resets(self)


@configclass
class Ur10eLinearGripperBoxCenterPaperFinetuneEvalCfg(Ur10eLinearGripperRelCartesianOSCFinetuneEvalCfg):
    """Eval after Stage A finetune (UR10e + linear gripper): box -> table-center target."""

    def __post_init__(self):
        super().__post_init__()
        _paper_stage_box_center(self)
        _repoint_ur10e_resets(self)


@configclass
class Ur10eLinearGripperObjectInBoxPaperFinetuneEvalCfg(Ur10eLinearGripperRelCartesianOSCFinetuneEvalCfg):
    """Eval after Stage B finetune (UR10e + linear gripper): object -> box cavity."""

    def __post_init__(self):
        super().__post_init__()
        _paper_stage_object_in_box(self)
        _repoint_ur10e_resets(self)


@configclass
class Ur10eLinearGripperCoverCloseRimPaperFinetuneEvalCfg(Ur10eLinearGripperRelCartesianOSCFinetuneEvalCfg):
    """Eval after Stage C finetune (UR10e + linear gripper): edge-rim (caprim) cover -> box with object inside."""

    def __post_init__(self):
        super().__post_init__()
        _paper_stage_cover_close(self)
        _repoint_ur10e_resets(self)
