# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""How an OFFLINE RECORDER tells an environment's gripper to open or close.

The two dataset recorders -- ``scripts_v2/tools/record_grasps.py`` and
``scripts_v2/tools/record_reset_states.py`` -- are scripted rollouts, not policies. Both need to
hold the gripper shut while the environment shakes the object and decides whether the grasp
survived. Neither trains anything, so neither is a "training path"; what they are is the PRODUCER
for the datasets a trainable environment resets from, which is why getting this wrong is silent
and expensive rather than loud.

WHAT WAS WRONG. Both scripts wrote the close command into the LAST action dimension
(``actions[:, -1] = -1.0``, and ``-torch.ones(...)`` respectively). That is correct for exactly one
kind of gripper: a ``BinaryJointAction``, which occupies one dimension and follows Isaac Lab's
convention of negative = close. On the fully actuated DELTO hand the last dimension is
``rj_dg_5_4`` alone, and -1 commands 0.1 rad of EXTENSION on it per step -- the opposite of
closing, on one joint of twenty, with the other nineteen left at zero. Nothing would have raised:
the buffer is sized from ``env.action_space.shape``, so the shape is right and the meaning is
wrong. Every regenerated DELTO ``*EEGrasped`` dataset would have been recorded with an open hand.

WHAT THIS MODULE DOES INSTEAD. It asks the environment's own action terms what "close" means for
them, and returns a FULL-WIDTH action vector -- zeros for the arm, the gripper's command in the
gripper's own slice, at the gripper's own indices:

* a ``BinaryJointAction`` term takes -1 to close and +1 to open (Isaac Lab's convention, read off
  the term's class, not assumed from the action width);
* a per-joint term (one action dimension per joint, e.g. the DELTO's twenty) is SERVOED, joint by
  joint and BY NAME, toward a target posture: ``clamp((target - measured) / scale, lo, hi)``. There
  is no scalar and no shared command anywhere in that -- all twenty dimensions are written
  independently, every step, which is also why it must be recomputed each step rather than filled
  into a buffer once.

The postures come from the gripper asset (``uwlab_assets.robots.ur10e_delto.actions``), keyed by
joint NAME. Position-keyed lookup is impossible here on purpose: this YAML-and-metadata family is
finger-major while the articulation reports joints level-major, and the two orders differ.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import torch

from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointAction
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

from uwlab_assets.robots.ur10e_delto.actions import (
    DELTO_HAND_OPEN_JOINT_POS,
    DELTO_HAND_PINCH_CLOSE_JOINT_POS,
    DELTO_HAND_PINCH_OPEN_JOINT_POS,
    DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS,
)

# Joint name -> (open angle, closed angle), for every gripper whose action term is per-joint and
# therefore has no open/close command of its own. Keyed by JOINT NAME, so a gripper is recognised
# by the joints its action term resolved to; ``rj_dg_*`` exists on no other robot in this tree.
#
# A gripper missing from here fails LOUDLY in :func:`scripted_gripper_actions` rather than being
# driven with zeros, which for a relative action term means "hold wherever you are" -- i.e. an
# open, non-grasping hand recorded as a grasp.
_PER_JOINT_POSTURES: dict[str, tuple[float, float]] = {
    name: (float(DELTO_HAND_OPEN_JOINT_POS[name]), float(DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS[name]))
    for name in DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS
}

# The same mapping for a TWO-FINGER PINCH: thumb pad against index pad, every other body clear of
# the object. Both columns differ from the table above, not just the closed one -- the pre-grasp
# has to be the pinch's own open posture, because the hand's default OPEN is a splay that puts
# fingers 3/4/5 straight into the workspace and does not sit on the pinch's approach at all.
#
# Selected per environment by ``env.cfg.scripted_gripper_pinch_posture``, following the same
# pattern as ``scripted_gripper_deadband_rad`` below: absent on every cfg that does not opt in, so
# every existing caller, gripper and task keeps exactly the postures it has today.
#
# ONLY FINGERS 1 AND 2 CLOSE. The optimised closed posture also moves fingers 3/4/5 (by up to
# 0.19 rad on rj_dg_4_2), so the pinch used to curl all twenty joints -- measured in the sampler
# with the cube pinned, that stops the jaw at 58 mm because 3/4/5 reach the cube first, against
# 35 mm with them held (cube-stack/pinch/pinchhold.py --pinch-fingers-only). A pinch is the thumb
# and index pads; the other three fingers hold their open targets for the whole close.
_PINCH_CLOSING_FINGERS = ("rj_dg_1_", "rj_dg_2_")
_PER_JOINT_PINCH_POSTURES: dict[str, tuple[float, float]] = {
    name: (
        float(DELTO_HAND_PINCH_OPEN_JOINT_POS[name]),
        float(DELTO_HAND_PINCH_CLOSE_JOINT_POS[name] if name.startswith(_PINCH_CLOSING_FINGERS)
              else DELTO_HAND_PINCH_OPEN_JOINT_POS[name]),
    )
    for name in DELTO_HAND_PINCH_CLOSE_JOINT_POS
}

# Position deadband for the per-joint servo below (bead UWLab-z9j.11 follow-on). Without one,
# _per_joint_command recomputes (target - measured) / scale EVERY step with no tolerance, so a
# joint that has already reached its target keeps issuing a nonzero correction -- measured pinned
# at exactly +-3.000 rad/s (the actuator's velocity_limit_sim ceiling) with the sign flipping every
# 5-10 steps, for the WHOLE 18 s episode, in free space, touching nothing.
#
# DERIVATION, from the same measurement, not guessed: the actuator can travel
# velocity_limit_sim * control_dt = 3.0 rad/s * 0.1 s = 0.3 rad within a single control step,
# against a 0.1 rad commanded delta (one action unit at this term's scale) -- a 3x overshoot ratio.
# Any error still eliciting a "small" correction near that 0.1 rad unit gets overshot by up to 0.3
# rad and lands on the opposite side next step, which is the cycle. The measured commanded-vs-
# measured gap WHILE oscillating sits in the 0.06-0.20 rad band -- i.e. errors up to 0.20 rad are
# still triggering full-throttle corrective moves that overshoot and re-trigger. Deadbanding at the
# TOP of that observed band (rather than its middle or bottom) guarantees the joint stops receiving
# any correction once it enters the region that was empirically sustaining the cycle.
#
# THE DEFAULT, not a global. scripted_gripper_actions is shared by grasp sampling, reset-state
# generation and reset-state visualisation, across every gripper -- see its own comment for why a
# caller-specific override (env.cfg.scripted_gripper_deadband_rad, set by DeltoGraspSamplingCfg to
# 0.0 once segment-4 stiffness made the deadband itself measurably unnecessary for the sampler)
# lives there rather than here. This constant stays the answer for every caller that doesn't opt out.
_POSITION_DEADBAND_RAD = 0.20


def _term_scale(term, num_joints: int, device) -> torch.Tensor:
    """Per-joint action scale of a joint action term, as a (num_envs, num_joints) tensor."""
    scale = getattr(term, "_scale", 1.0)
    if isinstance(scale, torch.Tensor):
        return scale
    return torch.full((term.num_envs, num_joints), float(scale), device=device)


def _clip_bounds(term, num_joints: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The term's own action clip, or +-inf when it declares none."""
    clip = getattr(term, "_clip", None)
    if isinstance(clip, torch.Tensor):
        return clip[..., 0], clip[..., 1]
    inf = torch.full((term.num_envs, num_joints), float("inf"), device=device)
    return -inf, inf


def _per_joint_command(term, close: bool, device, deadband_rad: float = _POSITION_DEADBAND_RAD,
                       pinch: bool = False, gain: float = 1.0, stalled_gain: float | None = None,
                       stall_velocity: float = 0.05) -> torch.Tensor:
    """Servo command driving every joint of ``term`` toward its open or closed posture.

    ``deadband_rad`` defaults to the shared ``_POSITION_DEADBAND_RAD`` -- every caller that doesn't
    pass its own value gets today's behaviour unchanged. See ``scripted_gripper_actions`` for where
    a caller-specific value comes from and why it must not be the module-level constant itself.
    """
    postures = _PER_JOINT_PINCH_POSTURES if pinch else _PER_JOINT_POSTURES
    joint_names: list[str] = list(getattr(term, "_joint_names", []))
    unknown = [name for name in joint_names if name not in postures]
    if unknown:
        raise ValueError(
            f"No scripted open/closed posture is declared for gripper joint(s) {unknown}, which the"
            f" per-joint action term '{type(term).__name__}' drives. The offline recorders cannot"
            " command this gripper shut. Declare the postures in"
            " omnireset/mdp/scripted_gripper.py::_PER_JOINT_POSTURES, keyed by joint name."
            "\nDriving the term with zeros instead is NOT a fallback: a relative joint action reads"
            " zero as 'hold the measured position', so every recorded grasp would be an open hand."
        )
    index = 1 if close else 0
    target = torch.tensor([postures[name][index] for name in joint_names], device=device)
    measured = term._asset.data.joint_pos[:, term._joint_ids]
    scale = _term_scale(term, len(joint_names), device)
    lo, hi = _clip_bounds(term, len(joint_names), device)
    error = target.unsqueeze(0) - measured
    # Recomputed from the MEASURED position every step, so the command saturates against contact
    # and keeps a standing position error -- a sustained squeeze rather than a one-shot target.
    # gain < 1 commands only part of the remaining error per step (see scripted_gripper_servo_gain
    # on the DELTO pinch cfg for the chatter it removes); 1.0 is the behaviour every caller had.
    per_joint_gain = torch.full_like(error, gain)
    if stalled_gain is not None:
        # A joint that has stopped moving while still short of its target is pressed against the
        # object: give it the full command so the squeeze is not divided along with the free motion.
        velocity = term._asset.data.joint_vel[:, term._joint_ids]
        velocity = getattr(velocity, "torch", velocity)
        per_joint_gain = torch.where(velocity.abs() < stall_velocity, torch.full_like(error, stalled_gain), per_joint_gain)
    command = torch.clamp(per_joint_gain * error / scale, lo, hi)
    # Deadband: a joint already within deadband_rad of its target holds (zero command) instead of
    # issuing a correction the actuator would overshoot -- see _POSITION_DEADBAND_RAD's derivation.
    # A relative joint action reads zero as "hold the measured position", so this is a genuine hold,
    # not a stall: it stops the perpetual oscillation without moving the joint away from its target.
    return torch.where(error.abs() < deadband_rad, torch.zeros_like(command), command)


def scripted_gripper_actions(env, close: bool) -> torch.Tensor:
    """Full-width action vector that opens or closes this environment's gripper.

    Args:
        env: The (unwrapped) manager-based env. Its action manager is read, not its config.
        close: ``True`` for the close command, ``False`` for open.

    Returns:
        ``(num_envs, total_action_dim)``. Every non-gripper term (an OSC arm, for instance) is left
        at zero.

    Raises:
        ValueError: if no gripper term is found, or a per-joint gripper has no declared posture.
    """
    # Per-joint deadband, resolved PER CALLER rather than as one shared module constant. This
    # helper is shared by all three offline recorder scripts (grasp sampling, reset-state
    # generation, reset-state visualisation) across every gripper, so a plain constant change here
    # would silently retune reset-state generation's servo behaviour along with the sampler's --
    # exactly the class of cross-registry defect this project keeps hitting. ``env.cfg`` is read
    # instead of ``env.cfg.events.*.params`` because this function is a plain helper called
    # directly by a script's main loop, not an EventTerm with its own params dict to read from.
    # Absent on every cfg except DeltoGraspSamplingCfg (which sets it explicitly), so every other
    # caller/gripper/task falls back to the same _POSITION_DEADBAND_RAD every prior caller used.
    deadband_rad = getattr(env.cfg, "scripted_gripper_deadband_rad", _POSITION_DEADBAND_RAD)
    # Opt-in, resolved the same way and for the same reason: this helper is shared by all three
    # offline recorders across every gripper, so the choice belongs to the caller's cfg rather
    # than to a module constant. Default False => today's five-finger postures, unchanged.
    pinch = bool(getattr(env.cfg, "scripted_gripper_pinch_posture", False))
    gain = float(getattr(env.cfg, "scripted_gripper_servo_gain", 1.0))
    # PER-SUBSTEP RELATIVE ACTIONS. Isaac Lab calls ActionManager.apply_action() once per PHYSICS
    # substep (decimation times per env.step), and RelativeJointPositionAction.apply_actions sets
    # target = current joint position + processed action EACH time -- so one control step's command
    # is re-added `decimation` times. The servo below sizes its command as a one-shot correction; at
    # decimation 12 it overshoots ~12x. Filmed on the DELTO pinch: rj_dg_1_2 / 2_2 / 2_1 flip between
    # two values every control step, commanded AWAY from their closed posture, unchanged by servo gain,
    # stiffness or damping. Opt-in, so every caller tuned against today's behaviour keeps it.
    stalled_gain = None
    if bool(getattr(env.cfg, "scripted_gripper_relative_per_substep", False)):
        if bool(getattr(env.cfg, "scripted_gripper_full_squeeze_when_stalled", False)):
            # Dividing by decimation makes free motion converge but also divides the squeeze: a
            # finger blocked by the object sees only 1/decimation of its error per substep. Filmed:
            # the pinch held a 30 g cube with gravity off and dropped it within 10 steps of gravity
            # on, at 1/12, 1/4 and 1/2 of the full command alike.
            stalled_gain = gain
        gain /= max(1, int(getattr(env.cfg, "decimation", 1)))
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    offset = 0
    found = False
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        width = term.action_dim
        joint_names = list(getattr(term, "_joint_names", []))
        if isinstance(term, BinaryJointAction):
            # Isaac Lab's convention, taken from the class rather than from the action width.
            actions[:, offset : offset + width] = -1.0 if close else 1.0
            found = True
        elif (
            isinstance(term, JointPositionAction)
            and joint_names
            and width == len(joint_names)
            and joint_names[0] in _PER_JOINT_POSTURES
        ):
            # ABSOLUTE per-joint term: command the posture itself as the PD target. No servo, no
            # per-substep re-application, and a finger blocked by the object squeezes with its full
            # stiffness * (posture - position). processed = raw * scale + offset, so invert that.
            postures = _PER_JOINT_PINCH_POSTURES if pinch else _PER_JOINT_POSTURES
            index = 1 if close else 0
            target = torch.tensor([postures[n][index] for n in joint_names], device=env.device)
            # SQUEEZE (opt-in, UWLAB_PINCH_SQUEEZE_RAD). The pinch close posture sits only 0.01-0.04 rad
            # past where the thumb joints stop on the cube in the grasp bank, so stiffness x position
            # error leaves almost no grip force. Pushing thumb/index targets further along their closing
            # direction makes them press up to the effort limit; a finger stopped by the cube cannot
            # overshoot, it only squeezes harder.
            squeeze = float(os.environ.get("UWLAB_PINCH_SQUEEZE_RAD", "0") or 0)
            if pinch and close and squeeze > 0:
                for i, n in enumerate(joint_names):
                    if n.startswith(("rj_dg_1_", "rj_dg_2_")):
                        target[i] += squeeze if postures[n][1] >= postures[n][0] else -squeeze
            scale = _term_scale(term, len(joint_names), env.device)
            term_offset = getattr(term, "_offset", 0.0)
            term_offset = getattr(term_offset, "torch", term_offset)
            actions[:, offset : offset + width] = (target.unsqueeze(0) - term_offset) / scale
            found = True
        elif joint_names and width == len(joint_names) and joint_names[0] in _PER_JOINT_POSTURES:
            actions[:, offset : offset + width] = _per_joint_command(
                term, close, env.device, deadband_rad, pinch=pinch, gain=gain, stalled_gain=stalled_gain
            )
            found = True
        offset += width
    if not found:
        raise ValueError(
            "No gripper action term was recognised on this environment, so there is no way to"
            f" command it {'shut' if close else 'open'}. Action terms:"
            f" {list(env.action_manager.active_terms)}. A binary gripper is recognised by its"
            " BinaryJointAction class; a fully actuated hand by its joint names appearing in"
            " omnireset/mdp/scripted_gripper.py::_PER_JOINT_POSTURES."
        )
    return actions


def scripted_gripper_actions_for(env, close_mask: torch.Tensor) -> torch.Tensor:
    """Per-environment mix of the open and close commands, selected by ``close_mask``.

    Used by the reset-state recorder's non-grasped types, which deliberately record both open and
    closed end-effector states.
    """
    close = scripted_gripper_actions(env, close=True)
    open_ = scripted_gripper_actions(env, close=False)
    return torch.where(close_mask.view(-1, 1).to(torch.bool), close, open_)


def is_gripper_posture_declared(joint_names: Mapping[str, object] | list[str]) -> bool:
    """Whether every named joint has a declared scripted posture. For tooling diagnostics."""
    return all(name in _PER_JOINT_POSTURES for name in joint_names)
