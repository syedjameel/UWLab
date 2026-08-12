# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VARIANT 1 of the UR5e + DELTO dexlift action space: DexSuite-style relative joint position.

One action term over all twenty-six joints -- the six UR5e arm joints and the twenty DELTO finger
joints -- as RELATIVE joint-position commands. Action dimension is 26, and every joint gets exactly
one dimension of it, which is the standing requirement for this hand. The guard is applied twice
over: once at the bottom of this module, so importing it is already a check, and again from the
``__post_init__`` of every environment that mounts this group.

This variant lives in its own module, under its own gym ids and its own PPO ``experiment_name``,
because the other action-space variant is ALSO twenty-six dimensional. Shape alone would therefore
not stop a checkpoint of one from resuming the other; the separate experiment name is what does.

THE LOAD-BEARING PROPERTY: RELATIVE TO THE **MEASURED** JOINT POSITION.
``isaaclab.envs.mdp.actions.joint_actions.RelativeJointPositionAction.apply_actions`` is, verbatim::

    current_actions = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
    self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)

``data.joint_pos`` is the articulation's MEASURED state, not a stored target. The commanded target
is re-based on where the joints actually ARE on every step and can never be more than
``scale * action`` ahead of them. That is what structurally prevents a target backlog from
accumulating: a force-limited finger stalled against an object simply stops moving, and the command
stops with it.

Contrast the direct port in ``uwlab_tasks/direct/delto_grasp`` (``task.py:process_actions``)::

    env.target_pos.add_(env.action_scale * actions)
    env.target_pos.clamp_(env.robot_lower_limits, env.robot_upper_limits)

That form integrates into a STORED target, bounded only by the joint LIMITS. A stalled finger there
accumulates up to ``limit - measured`` of position error, which discharges as an impulse the moment
the obstruction moves -- the failure mode that flung objects in earlier UWLab dexterous attempts.
The distinction is not academic here: it is what decides the scale below.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from uwlab.envs.mdp.full_actuation import assert_action_cfg_fully_actuates

from uwlab_assets.robots.ur10e_delto.actions import DELTO_HAND_JOINT_NAMES

from . import mdp

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
"""The six UR5e joints, in the UR naming convention.

Defined HERE, in the module that spends one action dimension on each of them, and imported by the
environment config -- which also needs them for the sysid event and for the arm-scoped instability
termination. One definition, three consumers. (Same six NAMES as the UR10e; the numbers behind
them -- effort, velocity, sysid -- are the UR5e's and are carried by the articulation config.)
"""

##
# THE SCALE. Two references disagree; this is the decision and its reasons.
##

# BOTH references run at the SAME control rate -- decimation 2 on sim.dt 1/120, i.e. 60 Hz -- so
# their numbers are directly comparable and the disagreement is real, not a units artifact:
#
#   * IsaacLab dexsuite, ``config/kuka_allegro/dexsuite_kuka_allegro_env_cfg.py``:
#         RelativeJointPositionActionCfg(joint_names=[".*"], scale=0.1)
#     0.1 uniformly across arm and hand. This repository's UR10e+DELTO dexlift sibling uses the
#     same 0.1 on all 26, and the certified 92.87% table-leg policy was trained with the hand at
#     0.1 (``table_leg_env_cfg.TableLegJointPositionActionCfg``).
#   * The UWLab direct delto_grasp port, ``docs/source/publications/omnireset/
#     delto_reference_grasp.md`` and ``delto_grasp_env_cfg.ActionCfg``:
#         scale = (0.05,) * 6 + (0.01,) * 20
#     six arm joints at 0.05 rad/step, twenty hand joints at 0.01 rad/step.
#
# CHOSEN: 0.1 on all twenty-six. In the order the reasons decide it:
#
# 1. THE SMALL NUMBERS ARE A MITIGATION FOR A DIFFERENT INTEGRATOR, and the reason for them does
#    not transfer. The direct port accumulates into ``target_pos`` and clamps only to the joint
#    limits, so its per-step size directly governs how fast a backlog can build against a stalled
#    finger; 0.01 rad/step buys a slow build. This term re-bases on the MEASURED position every
#    step (see the module docstring), so no backlog can accumulate at ANY step size. Importing
#    0.01 here would import a defence against a failure this action term cannot have, and pay its
#    cost -- see 3 -- for nothing.
# 2. VARIANT 1 IS BY DEFINITION THE DEXSUITE-STYLE BASELINE. 0.1 uniform is the configuration the
#    reference and this repository's own certified policy were validated at, and the one the
#    UR10e+DELTO sibling env runs. Changing it would make variant 1 not the baseline, and would
#    confound the comparison against the other variant with a second uncontrolled difference.
# 3. AT 0.1 THE PLANT, NOT THE SCALE, IS THE BINDING CONSTRAINT ON EVERY JOINT -- which is what a
#    baseline wants, because it leaves the identified actuator limits in charge of what the robot
#    can do. With |action| = 1 at 60 Hz, 0.1 rad/step is 6.0 rad/s of COMMANDED joint speed:
#      - hand: against ``DELTO_HAND_ACTUATOR.velocity_limit_sim`` of 3.0 rad/s -- 2x the cap, so
#        the actuator sets closing speed. At 0.01 the command would be 0.6 rad/s, ONE FIFTH of the
#        actuator's capability, and the scale would set closing speed instead. Inside dexsuite's
#        4-second episode that is a real reduction in reachable behaviour, not a safety margin.
#      - arm: against this arm's own ``UR5E_VELOCITY_LIMITS`` -- 1.5708 rad/s on shoulder pan/lift
#        and elbow, 3.1415 rad/s on the three wrists. 6.0 rad/s clears all six. At 0.05 the command
#        would be 3.0 rad/s, which sits BELOW the wrists' 3.1415 rad/s: the scale, not the robot,
#        would cap wrist authority.
# 4. IT KEEPS PAIRED CONSTANTS HONEST. ``DELTO_HAND_ACTION_SCALE`` in the asset package is 0.1 and
#    ``DELTO_HAND_ACTION_CLIP`` derives its +-1 bound explicitly from that value at 60 Hz. A hand
#    scale of 0.01 here would silently invalidate that derivation for anything that reads the pair.
#
# WHAT IS *NOT* CLAIMED: that 0.1 is optimal for THIS arm. It is the baseline's value, adopted
# deliberately and reproduced exactly. Retuning it is the other variant's business, or a sweep.
ARM_ACTION_SCALE = 0.1
"""Radians of arm-joint motion per unit action. See the block above for why this is not 0.05."""

HAND_ACTION_SCALE = 0.1
"""Radians of finger-joint motion per unit action. See the block above for why this is not 0.01."""

# The VALUE is authored here rather than imported from the asset package, following the convention
# stated in ``table_leg_env_cfg``: an action scale is only ever validated inside one environment's
# dynamics, so it belongs to the task. The KEYS are not authored here -- the hand's come from the
# asset's canonical name tuple and the arm's from ARM_JOINT_NAMES above, so a renamed joint moves
# one definition instead of leaving two copies silently disagreeing.
#
# Spelled out PER JOINT rather than given as one regex-wide value, which is the whole point: a
# joint that ``joint_names`` matches but this mapping does not silently falls back to scale 1.0 --
# ten times the intended stroke on the hand and, on the arm, 1 rad per control step. A name here
# that matches NO joint raises inside ``JointAction.__init__`` (via
# ``string_utils.resolve_matching_names_values``). Both mistakes are therefore loud instead of
# quiet. It is also what lets the full-actuation guard decide, with no articulation and no
# simulator, that this term spends exactly one action dimension per joint.
UR5E_DELTO_REL_JOINT_POS_SCALE = {
    **{name: ARM_ACTION_SCALE for name in ARM_JOINT_NAMES},
    **{name: HAND_ACTION_SCALE for name in DELTO_HAND_JOINT_NAMES},
}


@configclass
class Ur5eDeltoRelJointPosActionCfg:
    """VARIANT 1: relative joint-position control over all 26 joints at 0.1 rad per unit action."""

    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=dict(UR5E_DELTO_REL_JOINT_POS_SCALE),
        # Drop the articulation's own offset, so the command really is measured + scale * action
        # and nothing else. Already the default for this cfg class; stated because it is the
        # property the module docstring rests on, and a silent default is a bad place to keep one.
        use_zero_offset=True,
        # DELIBERATELY NO ``clip``. The reference this variant reproduces has none, and neither
        # does the UR10e+DELTO sibling; adding one would make this something other than the
        # DexSuite-style baseline. The command is not unbounded in practice: it is re-based on the
        # measured position every step, the actuators cap the achievable rate (3.0 rad/s hand,
        # 1.5708/3.1415 rad/s arm), and dexsuite's ``action_l2_clamped`` / ``action_rate_l2_clamped``
        # penalise magnitude in the objective. The per-joint +-1 clip used by the table-leg task
        # (``DELTO_HAND_ACTION_CLIP``) is an orthogonal knob and the first thing to add if the arm
        # is seen slewing at the start of training.
    )


# The construction-time full-actuation guard, run at IMPORT. The env configs call it too, on their
# own ``self.actions``; this call is the cheaper, earlier one -- it fails on ``import`` of this
# module, with no environment, no scene and no simulator, which is what makes it runnable as a
# plain test. See ``uwlab.envs.mdp.full_actuation`` for what the requirement is and why a config
# object is the wrong thing to trust unchecked.
assert_action_cfg_fully_actuates(
    Ur5eDeltoRelJointPosActionCfg(), DELTO_HAND_JOINT_NAMES, context=f"{__name__} (import-time)"
)

# The guard above covers the HAND, which is the standing requirement, and says nothing about the
# ARM. This covers the other half: the mapping must name the six arm joints and the twenty hand
# joints and NOTHING else. An arm joint absent from the mapping is still matched by
# ``joint_names=[".*"]`` and silently takes scale 1.0 -- one radian per control step on a real arm
# -- and no guard above would notice, because no arm joint is a required hand joint.
#
# It cannot fire as the mapping is written TODAY, because both sides are derived from the same two
# name lists. That is the point: this is what keeps it that way. The repository's own convention
# pushes toward spelling scales out literally (see the block above for why), and a literal mapping
# is one typo away from an arm joint at scale 1.0.
_EXPECTED_SCALE_KEYS = frozenset(ARM_JOINT_NAMES) | frozenset(DELTO_HAND_JOINT_NAMES)
if frozenset(UR5E_DELTO_REL_JOINT_POS_SCALE) != _EXPECTED_SCALE_KEYS:
    raise ValueError(
        f"{__name__}: UR5E_DELTO_REL_JOINT_POS_SCALE must name exactly the 6 arm joints and the 20"
        " DELTO joints.\n"
        f"  missing: {sorted(_EXPECTED_SCALE_KEYS - frozenset(UR5E_DELTO_REL_JOINT_POS_SCALE))}\n"
        f"  unexpected: {sorted(frozenset(UR5E_DELTO_REL_JOINT_POS_SCALE) - _EXPECTED_SCALE_KEYS)}\n"
        "A joint matched by joint_names=['.*'] but absent from the mapping silently takes scale"
        " 1.0; a key that matches no joint raises only later, inside JointAction.__init__."
    )
