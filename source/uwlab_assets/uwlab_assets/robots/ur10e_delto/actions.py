# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fully actuated hand action for the UR10e + Tesollo DELTO DG-5F.

All twenty hand joints are independent policy actions. There is deliberately no scalar closure
here, and no open/closed posture PAIR: a one-parameter close fraction cannot produce a two-jaw
pinch on this hand, because the pads never lead the phalanges at any single fraction. Anything
that reintroduces a "close command" reintroduces that limit, so a policy must be free to command
each joint.

The action is RELATIVE and, per Isaac Lab's ``RelativeJointPositionAction``, resolves against the
MEASURED joint position each step (``current joint positions + processed actions``). That is the
load-bearing property: an absolute or target-relative form lets the commanded target run ahead of
where the force-limited fingers actually are, and the accumulated backlog discharges as an impulse
that flings the object.

The OPEN posture below stays: it is the articulation's reset posture, not a closure input.
"""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg

from .ur10e_delto import DELTO_HAND_DEFAULT_JOINT_POS

# The open posture IS the articulation's default posture and the metadata's
# ``finger_open_joint_angles``; aliased rather than restated so the three cannot drift apart.
DELTO_HAND_OPEN_JOINT_POS = dict(DELTO_HAND_DEFAULT_JOINT_POS)

# The hand's twenty actuated joints, as one regex -- the same expression the actuator group uses.
DELTO_HAND_JOINT_REGEX = r"rj_dg_[1-5]_[1-4]"

# Radians of joint motion per unit of action, per joint. Spelled out one joint at a time rather
# than given as a single regex-wide value, following the precedent in
# ``dexlift/dexlift_ur10e_delto_env_cfg.py``: a joint that the action term matches but this dict
# does not silently falls back to scale 1.0, whereas a name here that matches NO joint raises
# during term parsing. Twenty explicit keys therefore make a renamed or dropped joint visible.
# These names must remain exactly the keys of ``DELTO_HAND_OPEN_JOINT_POS``.
DELTO_HAND_ACTION_SCALE = {
    "rj_dg_1_1": 0.1,
    "rj_dg_1_2": 0.1,
    "rj_dg_1_3": 0.1,
    "rj_dg_1_4": 0.1,
    "rj_dg_2_1": 0.1,
    "rj_dg_2_2": 0.1,
    "rj_dg_2_3": 0.1,
    "rj_dg_2_4": 0.1,
    "rj_dg_3_1": 0.1,
    "rj_dg_3_2": 0.1,
    "rj_dg_3_3": 0.1,
    "rj_dg_3_4": 0.1,
    "rj_dg_4_1": 0.1,
    "rj_dg_4_2": 0.1,
    "rj_dg_4_3": 0.1,
    "rj_dg_4_4": 0.1,
    "rj_dg_5_1": 0.1,
    "rj_dg_5_2": 0.1,
    "rj_dg_5_3": 0.1,
    "rj_dg_5_4": 0.1,
}

# Twenty independent policy actions, one per hand joint. Action dimension is 20.
DELTO_FULL_HAND_ACTIONS = RelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=[DELTO_HAND_JOINT_REGEX],
    scale=DELTO_HAND_ACTION_SCALE,
    # Relative to the MEASURED joint position: the articulation's own offset is dropped so the
    # commanded target cannot accumulate ahead of the fingers. See the module docstring.
    use_zero_offset=True,
)
