# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VARIANT 2 of the UR5e + DELTO dexlift action space: OmniReset OSC arm + DexSuite 20-DOF hand.

Two halves, from two different places, and neither is re-authored here:

* the ARM half is OmniReset's :class:`RelCartesianOSCActionCfg`, six dimensions of Cartesian pose
  delta tracked by an analytical-Jacobian task-space PD -- ``tau = J^T (Kp e + Kd (-v))``, no mass
  matrix -- written to the articulation with ``set_joint_effort_target``. The term and, more
  importantly, the derivation of the one gain that is NOT shared with the other robots on this arm
  family live in ``omnireset/config/ur5e_robotiq_2f85/actions.py``, next to every sibling gain set
  they have to be read against;
* the HAND half is the identical ``DELTO_FULL_HAND_ACTIONS`` that variant 1 mounts: twenty
  independent relative joint-position dimensions, one per DELTO joint.

TOTAL ACTION DIMENSION 6 + 20 = 26 -- THE SAME WIDTH AS VARIANT 1, A DIFFERENT MANIFOLD. Six of
those dimensions mean "move the wrist this far in space" here and "move these six joints this far"
there. Nothing about the tensors distinguishes them, so nothing but the separate gym ids and the
separate PPO ``experiment_name`` stops a checkpoint of one from resuming the other. That is the
reason this variant is a sibling module rather than a flag.

WHAT THE PAIRING ITSELF HAS TO GET RIGHT, and what checks it:

* the arm term must not claim a hand joint and the hand term must not claim an arm joint. The
  patterns are disjoint by construction (``shoulder.*|elbow.*|wrist.*`` against
  ``rj_dg_[1-5]_[1-4]``) and :func:`assert_action_cfg_fully_actuates`, run at import below, is what
  says so without a simulator;
* the arm must be TORQUE-ONLY, since every newton-metre it sees comes from this term. That is a
  property of the ARTICULATION rather than of the action, so it is checked in the env config -- see
  ``_assert_arm_is_torque_only`` in :mod:`.dexlift_ur5e_delto_osc_env_cfg`;
* the six joints the arm term resolves must arrive in the order the analytical Jacobian's columns
  are written in. Nothing in Isaac Lab connects those two orderings; the startup guard
  ``check_osc_arm_joint_order`` is what does.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from uwlab.envs.mdp.full_actuation import assert_action_cfg_fully_actuates

from uwlab_assets.robots.ur10e_delto.actions import DELTO_FULL_HAND_ACTIONS, DELTO_HAND_JOINT_NAMES

from ..omnireset.config.ur5e_robotiq_2f85.actions import UR5E_DELTO_RELATIVE_OSC


@configclass
class Ur5eDeltoOscActionCfg:
    """VARIANT 2: 6-D operational-space arm + 20 independent DELTO joints. Action dimension 26.

    The attribute NAMES are the action-manager term names, and ``arm`` is load-bearing: the startup
    guard ``check_osc_arm_joint_order`` looks the OSC term up by it.
    """

    # Pre-train / train gains. The eval gain set (UR5E_DELTO_RELATIVE_OSC_EVAL) is deliberately not
    # mounted anywhere: its rotational pair is far over this controller's own stability bound on
    # this hand, with the arithmetic written at its definition. Do not wire it in without redoing
    # that derivation.
    arm = UR5E_DELTO_RELATIVE_OSC

    # Variant 1's hand half, named from the SAME asset-package object rather than restated. Note
    # that ``configclass`` turns a class attribute into a field whose default is DEEP-COPIED per
    # instance, so the constructed group holds an equal copy and not that object -- which is what
    # makes it safe for two environments to exist in one process, and why the check is that the two
    # variants' hand halves compare EQUAL rather than that they are identical. The hand is the
    # experimental constant across the two variants; the moment it stops being this one definition,
    # the comparison they exist to make stops being controlled.
    gripper = DELTO_FULL_HAND_ACTIONS


# The construction-time full-actuation guard, run at IMPORT: it fails on ``import`` of this module,
# with no environment, no scene and no simulator, which is what makes it runnable as a plain test.
# The env configs call it again on their own ``self.actions``. Note what it can and cannot see here:
# the OSC term carries no per-joint ``scale`` mapping, so the guard reports its dimension as
# undetermined -- and that is harmless precisely because the term names no hand joint. The twenty
# required joints are all covered by the hand term. If someone ever widened the arm term's patterns
# to reach a finger, this call is what would fail.
assert_action_cfg_fully_actuates(
    Ur5eDeltoOscActionCfg(), DELTO_HAND_JOINT_NAMES, context=f"{__name__} (import-time)"
)
