# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VARIANT 1 of the UR5e + DELTO dexlift action space: DexSuite-style relative joint position.

Twenty-six RELATIVE joint-position dimensions -- six for the UR5e arm joints, twenty for the DELTO
finger joints -- and every joint gets exactly one of them, which is the standing requirement for
this hand. The guard is applied twice over: once at the bottom of this module, so importing it is
already a check, and again from the ``__post_init__`` of every environment that mounts this group.

TWO TERMS, NOT ONE, AND THE HAND TERM IS NOT AUTHORED HERE. The hand half is the asset package's
:data:`DELTO_FULL_HAND_ACTIONS` -- the SAME object variant 2 mounts (:mod:`.dexlift_ur5e_delto_osc_
actions`) -- and only the six-joint arm term is written in this file. That split is what makes the
two variants a controlled comparison: the hand is the experimental constant, so it has to be one
definition rather than two configurations that happen to agree today.

It did not agree. This module previously mounted its own single term over ``joint_names=[".*"]``
with ``clip`` deliberately omitted, against variant 2's hand half carrying
``DELTO_HAND_ACTION_CLIP`` (+-1 on all twenty joints). Same scale, same class, different bound --
so with ``init_noise_std`` 1.0 an early |action| of 3 commanded 0.3 rad/step (18 rad/s) here and
0.1 rad/step (6 rad/s) there, a 3x difference in the finger command tail inside the one half that
was supposed to be identical. The bound now comes from the shared object on both sides and the arm
term below carries the same +-1 for the same reason.

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

from uwlab_assets.robots.ur10e_delto.actions import (
    DELTO_FULL_HAND_ACTIONS,
    DELTO_HAND_ACTION_SCALE,
    DELTO_HAND_JOINT_NAMES,
)

# THE authoritative arm-joint order: the one the analytical Jacobian's columns are written in.
from uwlab_assets.robots.ur5e_robotiq_gripper.kinematics import ARM_JOINT_NAMES as _KINEMATICS_ARM_JOINT_NAMES

from . import mdp

ARM_JOINT_NAMES = list(_KINEMATICS_ARM_JOINT_NAMES)
"""The six UR5e joints, in the UR naming convention -- ALIASED, not restated.

The list itself belongs to ``uwlab_assets.robots.ur5e_robotiq_gripper.kinematics``, because that is
where its ORDER is load-bearing: ``compute_jacobian_analytical`` emits one column per entry, and
variant 2's OSC arm multiplies that Jacobian by the joint vector its own patterns resolved to. A
second literal copy here would be a list that could silently diverge from the Jacobian it has to
match -- and ``check_osc_arm_joint_order``, whose whole job is to catch that divergence, defaults
to the kinematics list precisely so it is validating against the authority rather than the copy.

Re-exported from here because this is the module that spends one action dimension on each of the
six, and because the env config reads them for the sysid event. (Same six NAMES as the UR10e; the
numbers behind them -- effort, velocity, sysid -- are the UR5e's and come from the articulation.)
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
"""The hand's scale, for the reasoning above only. The VALUE lives in the asset package.

This module no longer authors a hand scale: the hand term is :data:`DELTO_FULL_HAND_ACTIONS`, whose
``DELTO_HAND_ACTION_SCALE`` is the number both variants run. The constant is kept because the
argument above quotes it, and the assert below is what keeps the quote true.
"""

if set(DELTO_HAND_ACTION_SCALE.values()) != {HAND_ACTION_SCALE}:
    raise ValueError(
        f"{__name__}: the asset package's DELTO_HAND_ACTION_SCALE is no longer a uniform"
        f" {HAND_ACTION_SCALE} rad per unit action (values:"
        f" {sorted(set(DELTO_HAND_ACTION_SCALE.values()))}). The scale argument in this module --"
        " and DELTO_HAND_ACTION_CLIP's own derivation of its +-1 bound at 60 Hz -- are written"
        " against that value. Re-read both before changing it."
    )

# The arm's per-joint mapping. The KEYS come from ARM_JOINT_NAMES (aliased from the kinematics
# module) rather than being restated, so a renamed joint moves one definition.
#
# Spelled out PER JOINT rather than given as one regex-wide value, which is the whole point: a
# joint that ``joint_names`` matches but this mapping does not silently falls back to scale 1.0 --
# one radian per control step on a real arm. A name here that matches NO joint raises inside
# ``JointAction.__init__`` (via ``string_utils.resolve_matching_names_values``). Both mistakes are
# therefore loud instead of quiet. It is also what lets the full-actuation guard decide, with no
# articulation and no simulator, that this term spends exactly one action dimension per joint.
UR5E_ARM_REL_JOINT_POS_SCALE = {name: ARM_ACTION_SCALE for name in ARM_JOINT_NAMES}

# Per-step bound on the commanded arm delta, in units of action -- the SAME +-1 the hand half
# carries, and derived the same way (see ``DELTO_HAND_ACTION_CLIP``).
#
# WHY THIS EXISTS AT ALL, since the DexSuite reference has no clip and neither does the UR10e+DELTO
# sibling. This term is RELATIVE and re-bases on the measured position each step, which bounds the
# accumulated TARGET but not the per-step COMMAND; the policy's raw Gaussian output is unbounded and
# ``init_noise_std`` is 1.0, so |action| = 3 is ordinary in the first iterations. Unclipped that is
# 0.3 rad/step = 18 rad/s of commanded joint speed into an arm whose own velocity limits are 1.5708
# and 3.1415 rad/s. The evidence the previous "no clip, like the reference" note left out is that
# this repository's own certified 92.87% DELTO policy DOES clip -- ``table_leg_env_cfg.py``,
# ``clip={name: (-1.0, 1.0) for name in DELTO_HAND_JOINT_NAMES}`` -- and it is the same run that
# certifies the 0.1 scale the reference argument leans on.
#
# WHAT IT COSTS: nothing reachable. At +-1 the command is still 6.0 rad/s, 1.9x the wrists' limit
# and 3.8x the shoulder/elbow limit, so the identified actuator -- not this bound -- remains the
# binding constraint on every arm joint, which is exactly what the scale block above argues for.
UR5E_ARM_ACTION_CLIP = {name: (-1.0, 1.0) for name in ARM_JOINT_NAMES}


@configclass
class Ur5eDeltoRelJointPosActionCfg:
    """VARIANT 1: relative joint-position control, 6 arm + 20 hand = 26 dims at 0.1 rad per unit.

    Two terms. ``gripper`` is the asset package's object, shared verbatim with variant 2 so the
    hand half of the comparison is one definition; ``arm`` is the half this variant exists to vary.
    """

    arm = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        # The six literal names, NOT ``[".*"]``. With a second term on the hand, a wildcard here
        # would claim the twenty finger joints as well and the two terms would both drive them.
        joint_names=list(ARM_JOINT_NAMES),
        scale=dict(UR5E_ARM_REL_JOINT_POS_SCALE),
        clip=dict(UR5E_ARM_ACTION_CLIP),
        # Drop the articulation's own offset, so the command really is measured + scale * action
        # and nothing else. Already the default for this cfg class; stated because it is the
        # property the module docstring rests on, and a silent default is a bad place to keep one.
        use_zero_offset=True,
    )

    # THE HAND HALF, NOT RE-AUTHORED: the same object variant 2 mounts, carrying the same twenty
    # scales and the same twenty +-1 clips. ``configclass`` deep-copies a class attribute into each
    # instance, so the two variants hold equal copies rather than the same object -- which is why
    # the cross-variant check in ``dexlift_ur5e_delto_osc_actions`` compares EQUAL rather than IS.
    gripper = DELTO_FULL_HAND_ACTIONS


# The construction-time full-actuation guard, run at IMPORT. The env configs call it too, on their
# own ``self.actions``; this call is the cheaper, earlier one -- it fails on ``import`` of this
# module, with no environment, no scene and no simulator, which is what makes it runnable as a
# plain test. See ``uwlab.envs.mdp.full_actuation`` for what the requirement is and why a config
# object is the wrong thing to trust unchecked.
assert_action_cfg_fully_actuates(
    Ur5eDeltoRelJointPosActionCfg(), DELTO_HAND_JOINT_NAMES, context=f"{__name__} (import-time)"
)

# The guard above covers the HAND, which is the standing requirement, and says nothing about the
# ARM. This covers the other half: the arm term's scale and clip mappings must name the six arm
# joints and NOTHING else. An arm joint absent from the scale mapping but still matched by
# ``joint_names`` silently takes scale 1.0 -- one radian per control step on a real arm -- and no
# guard above would notice, because no arm joint is a required hand joint. A joint missing from the
# CLIP mapping is likewise silent: ``JointAction.__init__`` fills unmatched entries with +-inf.
#
# It cannot fire as the mappings are written TODAY, because all three sides are derived from
# ARM_JOINT_NAMES. That is the point: this is what keeps it that way.
_EXPECTED_ARM_KEYS = frozenset(ARM_JOINT_NAMES)
for _mapping_name, _mapping in (
    ("UR5E_ARM_REL_JOINT_POS_SCALE", UR5E_ARM_REL_JOINT_POS_SCALE),
    ("UR5E_ARM_ACTION_CLIP", UR5E_ARM_ACTION_CLIP),
):
    if frozenset(_mapping) != _EXPECTED_ARM_KEYS:
        raise ValueError(
            f"{__name__}: {_mapping_name} must name exactly the 6 arm joints.\n"
            f"  missing: {sorted(_EXPECTED_ARM_KEYS - frozenset(_mapping))}\n"
            f"  unexpected: {sorted(frozenset(_mapping) - _EXPECTED_ARM_KEYS)}\n"
            "A joint matched by the term's joint_names but absent from the scale mapping silently"
            " takes scale 1.0, and one absent from the clip mapping is left unbounded; a key that"
            " matches no joint raises only later, inside JointAction.__init__."
        )

# And the arm term must not reach a HAND joint. It names six literals today, so this cannot fire --
# but a future edit back to a wildcard (which is what this file used to carry) would put both terms
# on the twenty finger joints at once, and the full-actuation guard would still pass, because each
# term does spend one dimension per joint it names.
_ARM_TERM_JOINTS = frozenset(Ur5eDeltoRelJointPosActionCfg().arm.joint_names)
if _ARM_TERM_JOINTS != _EXPECTED_ARM_KEYS:
    raise ValueError(
        f"{__name__}: the arm term must name exactly the 6 arm joints as literals, got"
        f" {sorted(_ARM_TERM_JOINTS)}. A pattern that also matches"
        f" {DELTO_HAND_JOINT_NAMES[0]}-style names would drive the hand from both action terms."
    )
