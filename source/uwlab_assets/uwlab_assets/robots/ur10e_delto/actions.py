# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Binary hand action for the UR10e + Tesollo DELTO DG-5F.

OmniReset exposes one policy gripper scalar, so the 20-joint hand is represented by one measured
open/closed posture pair. Only flexion joints move; opposition and spread joints stay at their
open values. The closed posture uses one fraction per joint *level* instead of one uniform
fraction. A uniform closure makes the distal phalanges lead the pads and leaves no usable cuboid
contact window. The level-specific solution keeps the opposing pads in front of the other links.

The fractions are absolute fractions of each flexion joint's remaining travel. They were selected
with exact convex-collider FK, independently cross-checked by an LP support solver, and verified at
the deployed posture for joint limits, >=1.5 mm non-adjacent self-clearance, opposition, and stop
margin. They are inseparable from the jaw frame in ``local/Robots/DeltoHand/metadata.yaml`` and the
34 mm ``DeltoBlock``; update and validate the package together.
"""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.utils import configclass

from .ur10e_delto import DELTO_HAND_DEFAULT_JOINT_POS

# The open posture IS the articulation's default posture and the metadata's
# ``finger_open_joint_angles``; aliased rather than restated so the three cannot drift apart.
DELTO_HAND_OPEN_JOINT_POS = dict(DELTO_HAND_DEFAULT_JOINT_POS)

# Upper (flexion) limit of each joint that closes, in radians, read from the hand USD's
# ``physics:upperLimit``. Only the 15 flexion joints appear -- the five opposition/spread joints do
# not move during a closure and so need no limit here. This asset is the "limited_jnts" variant,
# whose limits are clamped to flexion-only; a different DELTO USD has DIFFERENT limits and this
# table must be re-read from it (``rj_dg_{2,3,4}_3`` and ``rj_dg_5_2`` in particular are clamped
# from 90 deg of travel down to 30).
_DELTO_FLEXION_UPPER_LIMIT = {
    "rj_dg_1_1": 1.3439,
    "rj_dg_1_3": 1.5708,
    "rj_dg_1_4": 1.5708,
    "rj_dg_2_2": 2.0071,
    "rj_dg_2_3": 1.5708,
    "rj_dg_2_4": 1.5708,
    "rj_dg_3_2": 2.0071,
    "rj_dg_3_3": 1.5708,
    "rj_dg_3_4": 1.5708,
    "rj_dg_4_2": 1.9199,
    "rj_dg_4_3": 1.5708,
    "rj_dg_4_4": 1.5708,
    "rj_dg_5_2": 1.5708,
    "rj_dg_5_3": 1.5708,
    "rj_dg_5_4": 1.5708,
}

# Fractions of remaining flexion travel keyed by joint-name suffix. The distal level performs most
# of the enclosure; the proximal levels stay nearly fixed so phalanges do not preempt pad contact.
DELTO_CLOSE_FRACTIONS = {1: 0.004, 2: 0.092, 3: 0.0, 4: 0.947}

# CLOSED posture: flexion joints advance by the fraction for their level. Every other joint is held
# at its open value. Values cannot exceed a joint limit by construction.
DELTO_HAND_CLOSED_JOINT_POS = {
    name: (
        value + DELTO_CLOSE_FRACTIONS[int(name.rsplit("_", 1)[1])] * (_DELTO_FLEXION_UPPER_LIMIT[name] - value)
        if name in _DELTO_FLEXION_UPPER_LIMIT
        else value
    )
    for name, value in DELTO_HAND_OPEN_JOINT_POS.items()
}

# Binary open/close over all 20 hand joints. ONE scalar action selects a whole posture; the joints
# are slaved to it, not independent policy DOFs.
DELTO_BINARY_ACTIONS = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=[r"rj_dg_[1-5]_[1-4]"],
    open_command_expr=dict(DELTO_HAND_OPEN_JOINT_POS),
    close_command_expr=dict(DELTO_HAND_CLOSED_JOINT_POS),
)


@configclass
class DeltoBinaryGripperAction:
    """Hand-only action group, for the grasp-sampling env (no arm in the scene)."""

    gripper = DELTO_BINARY_ACTIONS
