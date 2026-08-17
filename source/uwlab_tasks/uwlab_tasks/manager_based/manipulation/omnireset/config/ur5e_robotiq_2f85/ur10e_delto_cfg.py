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
NO-SPEED-CAP note below. (The linear gripper's other helper, the ``abnormal_robot`` scoping, has a
DELTO twin -- ``delto_cfg._exclude_hand_from_abnormal`` -- but it is applied inside ``_apply_delto``
for every variant rather than named at each call site here.)

Registered gym ids (mirroring the UR10e linear-gripper ones):
* ``OmniReset-UR10eDelto-ObjectAnywhereEEAnywhere-v0``
* ``OmniReset-UR10eDelto-ObjectRestingEEGrasped-v0``
* ``OmniReset-UR10eDelto-ObjectAnywhereEEGrasped-v0``
* ``OmniReset-UR10eDelto-ObjectPartiallyAssembledEEAnywhere-v0``
* ``OmniReset-UR10eDelto-ObjectPartiallyAssembledEEGrasped-v0``
* ``OmniReset-UR10eDelto-RelCartesianOSC-State-v0`` (+ Finetune / Play / Finetune-Play)
"""

from __future__ import annotations

from isaaclab.utils import configclass

import uwlab_assets.robots.ur10e_delto as ur10e_delto

from .actions import Ur10eDeltoRelativeOSCAction, Ur10eDeltoRelativeOSCEvalAction
from .delto_cfg import _apply_delto, _apply_delto_collision_stack_size
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
# WHAT THE ACTUATOR CAP NOW HAS TO HOLD BACK IS DIFFERENT, AND UNMEASURED. This paragraph used to
# say the commanded stroke was nowhere near the cap, because the closure was one scalar selecting a
# fixed posture whose largest per-joint step was 0.1411 rad. That posture is gone: the hand is
# twenty independent relative joint actions at 0.1 rad per unit action, so the per-step commanded
# delta is whatever the policy asks for, every step, in any direction. The 3.0 rad/s actuator limit
# and the USD's ``physxJoint:maxJointVelocity`` of 419 deg/s (7.31 rad/s) are now the ONLY bounds on
# hand joint speed, where they were previously slack by an order of magnitude. Nobody has measured
# how often a trained policy sits against them.
#
# (This comment previously argued for no cap at all, on the basis that a full sweep was ~1.4 rad
# per joint and the USD ceiling was never approached. Both halves went stale: A8 added the actuator
# cap, and the A7 re-sweep took the fraction from 0.75 to 0.10, shrinking the stroke tenfold.)


# ``_exclude_hand_from_abnormal`` used to live here and was called from the two finetune configs
# only. It now lives in ``delto_cfg.py`` and is called from ``_apply_delto``, so every DELTO
# variant gets it -- including the Stage-1 TRAINING env, which is the one env the fully actuated
# hand newly exposed to that check and the one that was missing it.


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
        _apply_delto_collision_stack_size(self)


@configclass
class Ur10eDeltoObjectRestingEEGraspedResetStatesCfg(ObjectRestingEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur10e_delto.IMPLICIT_UR10E_DELTO, Ur10eDeltoRelativeOSCAction())
        _apply_delto_collision_stack_size(self)


@configclass
class Ur10eDeltoObjectAnywhereEEGraspedResetStatesCfg(ObjectAnywhereEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur10e_delto.IMPLICIT_UR10E_DELTO, Ur10eDeltoRelativeOSCAction())
        _apply_delto_collision_stack_size(self)


@configclass
class Ur10eDeltoObjectPartiallyAssembledEEAnywhereResetStatesCfg(ObjectPartiallyAssembledEEAnywhereResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur10e_delto.IMPLICIT_UR10E_DELTO, Ur10eDeltoRelativeOSCAction())
        _apply_delto_collision_stack_size(self)


@configclass
class Ur10eDeltoObjectPartiallyAssembledEEGraspedResetStatesCfg(ObjectPartiallyAssembledEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur10e_delto.IMPLICIT_UR10E_DELTO, Ur10eDeltoRelativeOSCAction())
        _apply_delto_collision_stack_size(self)


# ---------------------------------------------------------------------------------------
# RL state (training / finetune / eval)
# ---------------------------------------------------------------------------------------
@configclass
class Ur10eDeltoRelCartesianOSCTrainCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    """Stage 1 training: implicit actuator, no curriculum."""

    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur10e_delto.IMPLICIT_UR10E_DELTO, Ur10eDeltoRelativeOSCAction())
        _apply_delto_collision_stack_size(self)


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
        # Arm motor delay: identical to the linear-gripper finetune, because it is an ARM property
        # and the arm is the same one. Measured residual delay is 0 steps at 500 Hz; delay_hi=1
        # over-brackets it and, unlike delay_hi=2, does not make the ADR ceiling round(p*hi) jump
        # discretely at scale_progress 0.75 -- the wall the linear-gripper finetune stalled on.
        self.events.randomize_arm_sysid.params["delay_range"] = (0, 1)
        # max_delay sizes the DelayBuffers and must be >= delay_range[1]; kept at 2 as a harmless
        # margin so the buffers stay valid if delay_range is bumped back.
        _set_arm_max_delay(self, 2)
        _apply_delto_collision_stack_size(self)


@configclass
class Ur10eDeltoRelCartesianOSCEvalCfg(Ur5eRobotiq2f85RelCartesianOSCEvalCfg):
    """Eval after Stage 1: implicit actuator, soft gains."""

    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur10e_delto.IMPLICIT_UR10E_DELTO, Ur10eDeltoRelativeOSCAction())
        _apply_delto_collision_stack_size(self)


@configclass
class Ur10eDeltoRelCartesianOSCFinetuneEvalCfg(Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg):
    """Eval after Stage 2: explicit actuator, stiff eval gains."""

    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur10e_delto.EXPLICIT_UR10E_DELTO, Ur10eDeltoRelativeOSCEvalAction())
        # Eval at the measured residual arm delay (0), mirroring the real robot rather than
        # drawing from the inherited range.
        self.events.randomize_arm_sysid.params["delay_range"] = (0, 0)
        _apply_delto_collision_stack_size(self)
