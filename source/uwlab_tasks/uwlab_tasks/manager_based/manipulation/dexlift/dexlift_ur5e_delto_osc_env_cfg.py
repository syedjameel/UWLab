# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR5e + Tesollo DELTO DG-5F, dexterous lift/reorient, with an OPERATIONAL-SPACE arm.

This is the SECOND of two action-space variants of one task. Read
:mod:`.dexlift_ur5e_delto_env_cfg` first: the scene, the timing, the rewards, the ADR curriculum,
the event set, the scoped instability termination and the shared articulation all come from there
unchanged, and every decision behind them is documented at the constant that implements it.

WHAT CHANGES, AND ONLY THIS:

* the ARM half of the action becomes OmniReset's :class:`RelCartesianOSCActionCfg` -- a 6-D
  Cartesian pose delta driven through the arm's analytical Jacobian as
  ``tau = J^T (Kp e + Kd (-v))``, written with ``set_joint_effort_target``;
* the arm ACTUATOR therefore becomes the explicit ``DelayedPDActuatorCfg`` at stiffness 0 and
  damping 0, so every newton-metre the arm sees comes from that action term.

The HAND half is the same ``DELTO_FULL_HAND_ACTIONS`` object variant 1 mounts -- twenty independent
relative joint-position dimensions, same scale, same +-1 clip -- and an import-time check in
:mod:`.dexlift_ur5e_delto_osc_actions` compares the two constructed halves rather than leaving that
as prose. Total action dimension is 6 + 20 = 26 -- the SAME
WIDTH as variant 1 and a completely different manifold. A checkpoint, an ONNX export or a recorded
dataset from one variant will load into the other without complaint and mean nothing, which is why
the two are registered under separate gym ids rather than selected by a flag. The action group
itself, and what the pairing has to get right, are in :mod:`.dexlift_ur5e_delto_osc_actions`.

THREE CONSEQUENCES THAT ARE NOT OBVIOUS FROM THE DIFF:

1. The SHARED mixin installs the IMPLICIT articulation and writes PD gains (800/40) onto its arm,
   because variant 1's ``RelativeJointPositionAction`` writes position targets and a zero-gain
   actuator would simply not move. Here that would be actively wrong: the position channel would
   fight the OSC's torques. This class therefore swaps the whole robot back to the shared
   articulation's EXPLICIT, zero-gain form after calling ``super().__post_init__()``, and
   :func:`_assert_arm_is_torque_only` fails construction if anything -- that ordering, or an
   inherited randomization event -- puts a second controller back on the arm.
   (Those two lines are variant-1 policy sitting in the shared mixin. Moving them down into
   ``Ur5eDeltoRelJointPosMixinCfg``, where the action that needs them lives, would make this swap
   unnecessary; it is left alone here rather than edited from a sibling module, and the assertion
   below is what makes the current arrangement safe either way.)
2. The actuator DELAY in ``randomize_arm_sysid`` becomes LIVE. Variant 1 notes it is inert under
   the implicit actuator because there is no delay buffer to write; ``DelayedPDActuatorCfg`` has
   one, so ``delay_range = (0, 1)`` now really does hold the arm's commands back by 0 or 1 physics
   steps per episode, exactly as it does in the OmniReset tasks this action came from. That is a
   gain in sim2real fidelity, not an accident, but it IS a behavioural difference between the two
   variants beyond the action space.
3. The OSC's stability depends on the CONTROL RATE, and this env inherits dexsuite's ``sim.dt`` of
   1/120 s. ``apply_actions`` runs once per PHYSICS step, so that -- not the 60 Hz policy rate --
   is the ``dt`` in the damping derivation. Changing ``sim.dt`` invalidates the rotational damping
   ratio; see its derivation in ``omnireset/config/ur5e_robotiq_2f85/actions.py``.

WHY THE ANALYTICAL KINEMATICS ARE STILL THE UR5e'S, verified rather than assumed. The Jacobian is
built by ``compute_jacobian_analytical``, which reads ``calibrated_joints`` (the six arm joint
frames) out of a ``metadata.yaml`` and walks them; it never touches ``link_inertials``, masses or
anything else the graft changed -- the mass matrix is a separate function and this controller
famously does not use one. The graft replaced the PAYLOAD, not the arm: chaining the six
``calibrated_joints`` frames reproduces the grafted USD's own authored arm transforms to 0.025 um,
so the same chain describes both robots. What the action DOES name explicitly is
``calibration_dir``; see the comment on ``_UR5E_DELTO_CALIBRATION_DIR``.
"""

from __future__ import annotations

import re

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.dexsuite import dexsuite_env_cfg as dexsuite

from uwlab_assets.robots.ur10e_delto.actions import DELTO_HAND_JOINT_NAMES
from uwlab_assets.robots.ur5e_delto import EXPLICIT_UR5E_DELTO

from ..omnireset import mdp as omnireset_mdp
from .dexlift_ur5e_delto_actions import ARM_JOINT_NAMES
from .dexlift_ur5e_delto_env_cfg import Ur5eDeltoEventCfg, Ur5eDeltoMixinCfg
from .dexlift_ur5e_delto_osc_actions import Ur5eDeltoOscActionCfg

##
# Guards.
##


def _any_nonzero(gain) -> bool:
    """True if a gain -- a scalar, or a per-joint/per-pattern mapping -- is anywhere nonzero.

    Not ``bool(gain)``, which is what this check used to be. An actuator whose gains are spelled
    per joint carries a MAPPING, and ``{"shoulder_pan_joint": 0.0, ...}`` is truthy while being
    exactly the zero-gain arm this environment requires: the truthiness form raises on a correct
    config and reads the VALUES of no entry, so it is testing the wrong thing in both directions.
    Compare against zero, per element when there is more than one.
    """
    if gain is None:
        return False
    if isinstance(gain, dict):
        return any(float(value) != 0.0 for value in gain.values())
    return float(gain) != 0.0


def _assert_arm_is_torque_only(env_cfg) -> None:
    """Fail construction unless the arm is driven ONLY by the OSC action term's torques.

    Two independent ways this environment can silently stop being an operational-space task, both
    of which leave a perfectly valid-looking config:

    * the ARM ACTUATOR carrying nonzero stiffness or damping. Variant 1's mixin sets 800/40 on
      purpose, and this class inherits that mixin's ``__post_init__``; if the swap below is ever
      reordered or dropped, the arm gets a position controller tracking a target nobody writes,
      fighting the OSC's efforts. The result is a stiff, slow arm -- not a crash.
    * an inherited EVENT writing gains back at reset. dexsuite randomizes actuator gains over
      ``.*``; variant 1 narrows that to the hand joints, and this check re-asserts the narrowing
      from this end rather than trusting it, because the term lives in a base class two packages
      away.

    Both are checked against the ARM joint names, so a term that legitimately randomizes only the
    hand passes.
    """
    arm = env_cfg.scene.robot.actuators["arm"]
    if _any_nonzero(arm.stiffness) or _any_nonzero(arm.damping):
        raise ValueError(
            "The UR5e+DELTO OSC environment requires a TORQUE-ONLY arm actuator, but"
            f" scene.robot.actuators['arm'] has stiffness={arm.stiffness}, damping={arm.damping}."
            " RelCartesianOSCAction supplies joint efforts directly; a nonzero PD gain adds a"
            " second controller tracking a position target this task never writes."
        )
    for event_name in ("joint_stiffness_and_damping", "joint_friction"):
        term = getattr(env_cfg.events, event_name, None)
        if term is None:
            continue
        patterns = term.params["asset_cfg"].joint_names
        if patterns is None:
            raise ValueError(
                f"events.{event_name} selects EVERY joint of the articulation. On this robot that"
                " includes the six identified UR5e joints, whose dynamics the OSC and"
                " randomize_arm_sysid together define. Narrow it to the hand joints."
            )
        hit = [
            (pattern, joint)
            for pattern in ([patterns] if isinstance(patterns, str) else patterns)
            for joint in ARM_JOINT_NAMES
            if re.fullmatch(pattern, joint)
        ]
        if hit:
            raise ValueError(
                f"events.{event_name} matches arm joints {sorted({j for _, j in hit})} through"
                f" pattern(s) {sorted({p for p, _ in hit})}. It would overwrite the identified UR5e"
                " dynamics that randomize_arm_sysid exists to reproduce, and on this variant it"
                " would also re-arm the position channel the OSC assumes is dead."
            )


##
# MDP settings.
##


@configclass
class Ur5eDeltoOscEventCfg(Ur5eDeltoEventCfg):
    """Variant 1's events, plus the guard that the OSC's joint ordering is the Jacobian's.

    ``check_osc_arm_joint_order`` is the startup half of the kinematics argument in this module's
    docstring. The analytical Jacobian's columns follow the chain written in ``metadata.yaml``;
    the joint vector it multiplies follows whatever order the ARTICULATION enumerates its joints
    in, which ``find_joints`` reports and nobody has ever pinned. A permutation between the two is
    finite, correctly shaped, and simply drives the arm wrongly. This is a grafted robot, so the
    articulation's ordering is exactly the thing that could have moved.
    """

    check_hand_fully_actuated = EventTerm(
        func=omnireset_mdp.check_action_manager_fully_actuates,
        mode="startup",
        params={"required_joints": list(DELTO_HAND_JOINT_NAMES), "context": "dexlift UR5e+DELTO (OSC arm)"},
    )

    # NO ``expected_joint_names``. The guard defaults to
    # ``uwlab_assets.robots.ur5e_robotiq_gripper.kinematics.ARM_JOINT_NAMES``, which is the list
    # ``compute_jacobian_analytical`` actually writes its columns in -- the authority. Passing a
    # list explicitly would defeat that default and validate the articulation against whatever the
    # caller happened to hand over. (This task's ``ARM_JOINT_NAMES`` is now an alias of the same
    # object, so the two cannot disagree; not passing it is what keeps that true if the alias ever
    # becomes a copy again.)
    check_osc_arm_joints = EventTerm(
        func=omnireset_mdp.check_osc_arm_joint_order,
        mode="startup",
        params={"action_term_name": "arm"},
    )


##
# Environment configuration.
##


@configclass
class Ur5eDeltoOscMixinCfg(Ur5eDeltoMixinCfg):
    """Variant 1's environment with its arm half replaced by the operational-space controller."""

    actions: Ur5eDeltoOscActionCfg = Ur5eDeltoOscActionCfg()
    events: Ur5eDeltoOscEventCfg = Ur5eDeltoOscEventCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        # Everything variant 1 decides -- scene, timing, workspace shift, contact sensors, the
        # scoped instability termination, the narrowed hand randomization, the construction-time
        # full-actuation guard over ``self.actions`` (which is the OSC group by the time it runs).
        super().__post_init__()

        # -- robot: the EXPLICIT articulation, whose arm actuator is a DelayedPDActuatorCfg at
        # stiffness 0 / damping 0. This REPLACES what the parent installed: variant 1 starts from
        # the implicit articulation and writes 800/40 onto it because its action term commands
        # joint positions. Replaced wholesale rather than edited, so the parent's line and this
        # one cannot half-apply, and so the module-level EXPLICIT_UR5E_DELTO -- whose ``actuators``
        # dict is shared with every other environment in the process -- is never mutated.
        robot_cfg = EXPLICIT_UR5E_DELTO.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # the fingertip contact sensors the inherited rewards read need contact reporters
        robot_cfg.spawn = robot_cfg.spawn.replace(activate_contact_sensors=True)
        self.scene.robot = robot_cfg

        # -- and nothing may put a second controller back on the arm. See the function.
        _assert_arm_is_torque_only(self)


@configclass
class DexLiftUR5eDeltoOscLiftEnvCfg(Ur5eDeltoOscMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexLiftUR5eDeltoOscLiftEnvCfg_PLAY(Ur5eDeltoOscMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass


@configclass
class DexLiftUR5eDeltoOscReorientEnvCfg(Ur5eDeltoOscMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexLiftUR5eDeltoOscReorientEnvCfg_PLAY(Ur5eDeltoOscMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    pass
