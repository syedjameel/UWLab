# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e + Tesollo DELTO DG-5F variants of the OmniReset reset-state and RL-state tasks.

Structural twin of ``ur10e_linear_gripper_cfg.py``, with ``_apply_delto`` in place of
``_apply_linear_gripper``: each task config subclasses its 2F-85 base, calls
``super().__post_init__()`` first, then swaps in the DELTO robot + action and fixes everything
that assumed a parallel jaw. Nothing else changes -- objects, events, rewards, sim settings and
the object ``variants`` are all inherited, and the 2F-85 and linear-gripper paths are untouched.

THE ARM IS UNCHANGED. Same calibrated UR10e, same ``shoulder_*``/``elbow_joint``/``wrist_*``
joints, same ``wrist_3_link`` IK body, same ``Robots/UR10e`` calibration directory for the
analytical OSC. So every arm-side number carries over verbatim, and deliberately so:

* the ``sysid`` block in ``Ur10eDelto/metadata.yaml`` is a byte-identical copy of
  ``Ur10eLinearGripper/metadata.yaml`` (metadata.yaml is keyed by USD directory and does not
  inherit, so the numbers have to be repeated, not re-derived);
* ``delay_range`` and ``max_delay`` keep the linear-gripper finetune's treatment exactly. Those
  are properties of the ARM's motors, which this robot shares. The measured residual delay is 0
  steps at 500 Hz; ``(0, 1)`` over-brackets it while avoiding the ADR wall that ``(0, 2)``
  reintroduces at ``scale_progress = 0.75``.

What genuinely differs from the linear-gripper variant, beyond ``_apply_delto`` itself, is the
two UR10e-specific gripper helpers -- see :func:`_exclude_hand_from_abnormal` and the
NO-SPEED-CAP note below.

Registered gym ids (mirroring the UR10e linear-gripper ones):
* ``OmniReset-UR10eDelto-ObjectAnywhereEEAnywhere-v0``
* ``OmniReset-UR10eDelto-ObjectRestingEEGrasped-v0``
* ``OmniReset-UR10eDelto-ObjectAnywhereEEGrasped-v0``
* ``OmniReset-UR10eDelto-ObjectPartiallyAssembledEEAnywhere-v0``
* ``OmniReset-UR10eDelto-ObjectPartiallyAssembledEEGrasped-v0``
* ``OmniReset-UR10eDelto-RelCartesianOSC-State-v0`` (+ Finetune / Play / Finetune-Play)
"""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.ur10e_delto as ur10e_delto

from .actions import Ur10eDeltoRelativeOSCAction, Ur10eDeltoRelativeOSCEvalAction
from .delto_cfg import _apply_delto
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

# NO SPEED CAP, deliberately -- this is the one UR10e helper with no DELTO analogue.
#
# The linear gripper needs ``_apply_real_gripper_speed`` because its SIM jaws are ~5x faster than
# the real ones (0.2 s vs 1.0 s for the full 68 mm stroke), so an uncapped policy learns to
# grab-and-go before real jaws would have closed. It applies that cap HERE, at task-config level,
# and only to the deployment-matched envs.
#
# The DELTO is capped too -- but in the ASSET, not here. ``_DELTO_HAND_ACTUATOR`` sets
# ``velocity_limit_sim = 3.0`` rad/s, for a different reason than the jaw's:
# closing speed sets the kinetic energy delivered to a ~40 g part on first touch, which is the
# ejection mode directly. Being an actuator property it already applies to EVERY env, so there is
# nothing left for a task-config override to do -- and a second cap on top would be the same
# mistake as a second effort cap: a coarser number shadowing a derived one.
#
# For scale, the binary action is nowhere near that limit: at ``DELTO_CLOSE_FRACTION = 0.10`` the
# largest per-joint commanded stroke is 0.1411 rad (8.09 deg, ``rj_dg_1_4``), against a 3.0 rad/s
# cap and the USD's independently-enforced ``physxJoint:maxJointVelocity`` of 419 deg/s
# (7.31 rad/s). The deployed close does not approach either limit.
#
# (This comment previously argued for no cap at all, on the basis that a full sweep was ~1.4 rad
# per joint and the USD ceiling was never approached. Both halves went stale: A8 added the actuator
# cap, and the A7 re-sweep took the fraction from 0.75 to 0.10, shrinking the stroke tenfold.)


def _exclude_hand_from_abnormal(cfg) -> None:
    """Restrict the authors' ``abnormal_robot`` check (|joint_vel| > 2x limit) to the ARM joints.

    Same scoping as the linear gripper's ``_exclude_gripper_from_abnormal``, for the same reason:
    the check exists to catch runaway ARM dynamics, and a force-limited end effector cannot run
    away. The DELTO adds 20 joints to a check that fires an episode-ending -100 penalty if ANY
    selected joint trips, so leaving them in means twenty extra chances for a benign drive
    overshoot to be read as a robot fault.

    THIS IS LOAD-BEARING. DO NOT DELETE IT AS DEAD CODE. An earlier revision of this docstring said
    the opposite -- that the scoping was a no-op because ``_DELTO_HAND_ACTUATOR`` set
    ``velocity_limit_sim = 10000``, putting 2x it out of reach. The current 3.0 rad/s limit makes
    the hand-joint threshold
    now 2 x 3.0 = 6.0 rad/s -- BELOW the USD's independently-enforced ``physxJoint:maxJointVelocity``
    of 7.31 rad/s, and therefore reachable. The comment silently became false when a number moved
    under it, which is exactly how the reader most likely to delete this line would be misled.

    Reachable is not the same as measured: the binary close commands at most 0.1411 rad per joint at
    the current closure fraction, so a normal closure will not come near 6 rad/s. The exposure is
    reset teleports (``write_joint_state_to_sim`` writes a posture with no velocity budget) and hard
    contact during PPO exploration. Nobody has measured whether it trips in practice -- and the
    linear gripper is the precedent for why that matters: its equivalent check started firing on
    ~9% of episodes as soon as its jaws were speed-capped, freezing the ADR curriculum below its
    0.95 gate. Scoping the check to the arm removes that failure mode before it can appear.
    """
    arm = SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"])
    if getattr(cfg, "rewards", None) is not None and getattr(cfg.rewards, "abnormal_robot", None) is not None:
        cfg.rewards.abnormal_robot.params = {"asset_cfg": arm}
    if getattr(cfg, "terminations", None) is not None and getattr(cfg.terminations, "abnormal_robot", None) is not None:
        cfg.terminations.abnormal_robot.params = {"asset_cfg": arm}


def _set_arm_max_delay(cfg, max_delay: int) -> None:
    """Resize the arm actuator's delay buffers.

    Written as a rebind rather than ``cfg.scene.robot.actuators["arm"].max_delay = ...`` because
    the in-place form only happens to be safe. ``configclass.replace`` is ``dataclasses.replace``,
    which is shallow, so a naive reading says ``cfg.scene.robot.actuators`` is the same dict object
    as ``EXPLICIT_UR10E_DELTO.actuators``; what actually saves it is that ``replace`` re-runs
    ``__init__``, and configclass appends a ``_custom_post_init`` that deepcopies every non-callable
    member. Measured: constructing the UR10e linear gripper's RGB cfg, which writes its jaw speed
    cap in place, does NOT leak into ``IMPLICIT_UR10E_LINEAR_GRIPPER`` or into a Stage-1 cfg built
    afterwards in the same process. So this is defensive, not a bug fix -- but it depends on an
    IsaacLab implementation detail rather than on ``replace``'s documented semantics, and rebinding
    costs nothing.
    """
    robot = cfg.scene.robot
    robot.actuators = {**robot.actuators, "arm": robot.actuators["arm"].replace(max_delay=max_delay)}


# ---------------------------------------------------------------------------------------
# Reset states (full UR10e + DELTO hand)
# ---------------------------------------------------------------------------------------
@configclass
class Ur10eDeltoObjectAnywhereEEAnywhereResetStatesCfg(ObjectAnywhereEEAnywhereResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur10e_delto.IMPLICIT_UR10E_DELTO, Ur10eDeltoRelativeOSCAction())


@configclass
class Ur10eDeltoObjectRestingEEGraspedResetStatesCfg(ObjectRestingEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur10e_delto.IMPLICIT_UR10E_DELTO, Ur10eDeltoRelativeOSCAction())


@configclass
class Ur10eDeltoObjectAnywhereEEGraspedResetStatesCfg(ObjectAnywhereEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur10e_delto.IMPLICIT_UR10E_DELTO, Ur10eDeltoRelativeOSCAction())


@configclass
class Ur10eDeltoObjectPartiallyAssembledEEAnywhereResetStatesCfg(ObjectPartiallyAssembledEEAnywhereResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur10e_delto.IMPLICIT_UR10E_DELTO, Ur10eDeltoRelativeOSCAction())


@configclass
class Ur10eDeltoObjectPartiallyAssembledEEGraspedResetStatesCfg(ObjectPartiallyAssembledEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur10e_delto.IMPLICIT_UR10E_DELTO, Ur10eDeltoRelativeOSCAction())


# ---------------------------------------------------------------------------------------
# RL state (training / finetune / eval)
# ---------------------------------------------------------------------------------------
@configclass
class Ur10eDeltoRelCartesianOSCTrainCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    """Stage 1 training: implicit actuator, no curriculum."""

    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur10e_delto.IMPLICIT_UR10E_DELTO, Ur10eDeltoRelativeOSCAction())


@configclass
class Ur10eDeltoRelCartesianOSCFinetuneCfg(Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg):
    """Stage 2 finetune: explicit actuator + curriculum (base sets EXPLICIT 2F-85; we override last).

    The ADR ramps the arm dynamics toward the identified UR10e. Note the sysid identification was
    run with the 0.575 kg linear gripper mounted, not this 1.7735 kg hand, so the payload the fit
    saw is not the payload this robot carries -- unavoidable for any new end effector, and the
    reason the OSC rotational damping ratio is re-derived rather than inherited (see
    ``actions._UR10E_DELTO_ROT_DAMPING_RATIO``).
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur10e_delto.EXPLICIT_UR10E_DELTO, Ur10eDeltoRelativeOSCAction())
        _exclude_hand_from_abnormal(self)
        # Arm motor delay: identical to the linear-gripper finetune, because it is an ARM property
        # and the arm is the same one. Measured residual delay is 0 steps at 500 Hz; delay_hi=1
        # over-brackets it and, unlike delay_hi=2, does not make the ADR ceiling round(p*hi) jump
        # discretely at scale_progress 0.75 -- the wall the linear-gripper finetune stalled on.
        self.events.randomize_arm_sysid.params["delay_range"] = (0, 1)
        # max_delay sizes the DelayBuffers and must be >= delay_range[1]; kept at 2 as a harmless
        # margin so the buffers stay valid if delay_range is bumped back.
        _set_arm_max_delay(self, 2)


@configclass
class Ur10eDeltoRelCartesianOSCEvalCfg(Ur5eRobotiq2f85RelCartesianOSCEvalCfg):
    """Eval after Stage 1: implicit actuator, soft gains."""

    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur10e_delto.IMPLICIT_UR10E_DELTO, Ur10eDeltoRelativeOSCAction())


@configclass
class Ur10eDeltoRelCartesianOSCFinetuneEvalCfg(Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg):
    """Eval after Stage 2: explicit actuator, stiff eval gains."""

    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur10e_delto.EXPLICIT_UR10E_DELTO, Ur10eDeltoRelativeOSCEvalAction())
        _exclude_hand_from_abnormal(self)
        # Eval at the measured residual arm delay (0), mirroring the real robot rather than
        # drawing from the inherited range.
        self.events.randomize_arm_sysid.params["delay_range"] = (0, 0)
