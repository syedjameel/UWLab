# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fully actuated hand action for the UR10e + Tesollo DELTO DG-5F.

All twenty hand joints are independent policy actions. There is deliberately no scalar closure
ACTION here: a one-parameter close fraction cannot produce a two-jaw pinch on this hand, because
the pads never lead the phalanges at any single fraction. Anything that reintroduces a "close
command" as an action reintroduces that limit, so a policy must be free to command each joint.
:func:`uwlab.envs.mdp.full_actuation.assert_action_cfg_fully_actuates` is what enforces that;
every DELTO env config calls it at construction.

Two fixed POSTURES do live here, and neither is a control model. The OPEN posture is the
articulation's reset posture. The CLOSED posture at the bottom is a measurement consumed only by
the offline grasp/reset recorders, which servo to it over the same twenty independent action
dimensions. Read the comment above it before touching it.

The action is RELATIVE and, per Isaac Lab's ``RelativeJointPositionAction``, resolves against the
MEASURED joint position each step (``current joint positions + processed actions``). That is the
load-bearing property: an absolute or target-relative form lets the commanded target run ahead of
where the force-limited fingers actually are, and the accumulated backlog discharges as an impulse
that flings the object.

The OPEN posture below stays: it is the articulation's reset posture, not a closure input.
"""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.utils import configclass

from .ur10e_delto import DELTO_HAND_DEFAULT_JOINT_POS

# The open posture IS the articulation's default posture and the metadata's
# ``finger_open_joint_angles``; aliased rather than restated so the three cannot drift apart.
DELTO_HAND_OPEN_JOINT_POS = dict(DELTO_HAND_DEFAULT_JOINT_POS)

# The hand's twenty actuated joints, as one regex -- the same expression the actuator group uses.
DELTO_HAND_JOINT_REGEX = r"rj_dg_[1-5]_[1-4]"

# The same twenty joints spelled out. This tuple is the CANONICAL name list: the action scale
# below, the table-leg task's own scale mapping, and the full-actuation guard all key off it, so a
# renamed or dropped joint moves one definition rather than four independent copies. Ordered
# finger-major to match the metadata YAML; the articulation reports them level-major, which is
# exactly why nothing here may ever index by position.
DELTO_HAND_JOINT_NAMES = tuple(f"rj_dg_{finger}_{joint}" for finger in range(1, 6) for joint in range(1, 5))

# Radians of joint motion per unit of action, per joint. Spelled out one joint at a time rather
# than given as a single regex-wide value, following the precedent in
# ``dexlift/dexlift_ur10e_delto_env_cfg.py``: a joint that the action term matches but this dict
# does not silently falls back to scale 1.0, whereas a name here that matches NO joint raises
# during term parsing. Twenty explicit keys therefore make a renamed or dropped joint visible.
# The KEYS come from :data:`DELTO_HAND_JOINT_NAMES` rather than being restated, so only the VALUE
# is authored here. That is also what the full-actuation guard reads to decide, without an
# articulation, that this term spends one action dimension per joint.
DELTO_HAND_ACTION_SCALE = {name: 0.1 for name in DELTO_HAND_JOINT_NAMES}

# Per-step bound on the commanded joint delta, in units of action. The action is RELATIVE and
# INTEGRATING (``target = measured + scale * action`` every step), and a Gaussian policy's raw
# output is unbounded, so without this the only ceiling is the plant: the actuator's 3.0 rad/s
# ``velocity_limit_sim``, which Isaac Lab writes straight into PhysX as the joint's maximum
# velocity and which therefore also overwrites whatever the USD authored (see the correction on
# ``_DELTO_HAND_ACTUATOR``; the "7.31 rad/s USD limit" this comment used to cite does not exist).
# The deleted binary action was bounded by construction (it selected
# one of two fixed postures); nothing replaced that bound when it went. +-1 x 0.1 rad per control
# step is the stroke the hand can actually track: at the 60 Hz control rate it is 6 rad/s of
# commanded joint speed, already twice the actuator cap, so this clip constrains the COMMAND
# without constraining the achievable motion.
DELTO_HAND_ACTION_CLIP = {name: (-1.0, 1.0) for name in DELTO_HAND_JOINT_NAMES}

# Twenty independent policy actions, one per hand joint. Action dimension is 20.
DELTO_FULL_HAND_ACTIONS = RelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=[DELTO_HAND_JOINT_REGEX],
    scale=DELTO_HAND_ACTION_SCALE,
    clip=DELTO_HAND_ACTION_CLIP,
    # Relative to the MEASURED joint position: the articulation's own offset is dropped so the
    # commanded target cannot accumulate ahead of the fingers. See the module docstring.
    use_zero_offset=True,
)

# ---------------------------------------------------------------------------------------
# A validated CLOSED posture -- data for the offline recorders, NOT a control model.
# ---------------------------------------------------------------------------------------
# READ THIS BEFORE REUSING THE DICT BELOW. It is not an action, it is not reachable from any
# policy, and re-wiring it into an action term is exactly the regression the full-actuation guard
# (``uwlab.envs.mdp.full_actuation``) exists to stop. It is a MEASUREMENT: one hand posture that
# was shown, by direct PhysX replay against the authored 0.03 kg DeltoBlock, to hold that part
# through the sampler's shake test.
#
# WHY IT HAD TO COME BACK. The grasp-sampling and reset-state recorders are scripted rollouts:
# something has to command the hand shut while the sampler shakes the object and decides whether
# the grasp survived. Deleting the number along with the action term did not remove a control
# model, it removed the ability to REGENERATE ``Grasps/<object>/grasps.pt``, which
# ``reset_end_effector_from_grasp_dataset`` replays into every ``*EEGrasped`` reset state, which in
# turn is 75% of the reset distribution of the trainable
# ``OmniReset-UR10eDelto-RelCartesianOSC-State-v0``. The branch was left running off an artifact
# that could not be rebuilt.
#
# HOW IT IS CONSUMED. Per joint, by name, as a servo TARGET for the same twenty independent action
# dimensions the policy uses: ``action_j = clamp((target_j - measured_j) / scale_j, -1, 1)``. There
# is no scalar anywhere in that, and the recorders drive all twenty dimensions explicitly -- see
# ``omnireset/mdp/scripted_gripper.py``.
#
# PROVENANCE OF THE NUMBERS. Recovered from commit ea540d0^ and RESOLVED here, deliberately. They
# were previously computed as ``open + fraction[level] * (flexion_upper_limit - joint)`` from a
# level-keyed fraction table; storing that expression would make the record decodable only through
# the deleted closure model and would re-import its "one fraction per level" idea. What the record
# actually is, is twenty angles. Any change to the hand's OPEN posture or its USD joint limits
# invalidates these and they must be re-measured, not recomputed.
DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS = {
    "rj_dg_1_1": 0.186527,
    "rj_dg_1_2": -0.373635,
    "rj_dg_1_3": 0.303144,
    "rj_dg_1_4": 1.495991,
    "rj_dg_2_1": -0.058794,
    "rj_dg_2_2": 1.001853,
    "rj_dg_2_3": 0.695646,
    "rj_dg_2_4": 1.517962,
    "rj_dg_3_1": -0.032855,
    "rj_dg_3_2": 1.001853,
    "rj_dg_3_3": 0.695646,
    "rj_dg_3_4": 1.517962,
    "rj_dg_4_1": 0.174352,
    "rj_dg_4_2": 0.993831,
    "rj_dg_4_3": 0.695646,
    "rj_dg_4_4": 1.517962,
    "rj_dg_5_1": 0.193785,
    "rj_dg_5_2": 0.961714,
    "rj_dg_5_3": 0.695646,
    "rj_dg_5_4": 1.517962,
}


@configclass
class DeltoFullHandAction:
    """Hand-only action group for the grasp-sampling env, which has no arm in the scene.

    Twenty independent dimensions, identical to the hand half of the full-robot groups. The
    sampler's scripted close is a per-joint servo built by
    ``omnireset.mdp.scripted_gripper.scripted_gripper_actions``; the env itself has no close
    command, and the guard is applied to this group like any other.
    """

    gripper = DELTO_FULL_HAND_ACTIONS


# ---------------------------------------------------------------------------------------
# TWO-FINGER SUBSET -- an experiment, not the default. Nothing above is modified.
# ---------------------------------------------------------------------------------------
# For a 40 mm cube the hand only needs its designed opposition, and the DG-5F's grasp geometry is
# already defined as exactly that: ``DeltoHand/metadata.yaml:8`` calls it "a virtual jaw of finger 1
# against finger 2", and ``grasp_center_offset`` is ``midpoint(rl_dg_1_tip, rl_dg_2_tip)``. The
# five-tip centroid was explicitly REJECTED as the grasp centre because finger 5 sits at y = -68 mm.
# So {1, 2} is the one subset that leaves the calibrated grasp centre meaningful; any other pair
# needs a fresh FK derivation and a re-sampled dataset.
#
# Actuating {1, 2} and holding {3, 4, 5} takes the policy's action dimension from 6 + 20 = 26 down
# to 6 + 8 = 14. For reference the paper's own robot acts in 7 dimensions (6 Cartesian + a binary
# gripper), so 26 is a large departure from the setting its hyperparameters were chosen in.
#
# THIS IS NOT A FREE WIN, and the cost is not the full-actuation guard (that is satisfied simply by
# narrowing what is REQUIRED). The cost is that unactuated joints hold
# ``DELTO_HAND_DEFAULT_JOINT_POS``, which is the OPEN SPLAY -- fingers 3/4/5 sit at ``_2 = 0.9``,
# out in the workspace, where they can foul both the cube and the receptive fixture. Hence the
# tucked posture below, and hence this whole path stays gated behind measured grasp yield rather
# than being adopted on the strength of the argument.
DELTO_TWO_FINGER_JOINT_NAMES = tuple(f"rj_dg_{finger}_{joint}" for finger in (1, 2) for joint in range(1, 5))
DELTO_TWO_FINGER_JOINT_REGEX = r"rj_dg_[1-2]_[1-4]"

# The three held fingers, and the posture they are held AT. Curled toward the palm and away from
# the finger-1/finger-2 working volume. The proximal ``_1`` joints keep their open-posture values
# (they set finger spread, and moving them would swing the fingers sideways rather than fold them);
# the ``_2``/``_3``/``_4`` chain is what curls.
DELTO_HELD_FINGER_JOINT_NAMES = tuple(f"rj_dg_{finger}_{joint}" for finger in (3, 4, 5) for joint in range(1, 5))

DELTO_TUCKED_HELD_FINGER_JOINT_POS = {
    "rj_dg_3_1": -0.032855,  # spread, unchanged from the open posture
    "rj_dg_3_2": 1.60,
    "rj_dg_3_3": 1.40,
    "rj_dg_3_4": 1.00,
    "rj_dg_4_1": 0.174352,
    "rj_dg_4_2": 1.60,
    "rj_dg_4_3": 1.40,
    "rj_dg_4_4": 1.00,
    "rj_dg_5_1": 0.193785,
    "rj_dg_5_2": 1.60,
    "rj_dg_5_3": 1.40,
    "rj_dg_5_4": 1.00,
}

DELTO_TWO_FINGER_ACTION_SCALE = {name: 0.1 for name in DELTO_TWO_FINGER_JOINT_NAMES}
DELTO_TWO_FINGER_ACTION_CLIP = {name: (-1.0, 1.0) for name in DELTO_TWO_FINGER_JOINT_NAMES}

# Eight independent policy actions, one per joint of fingers 1 and 2. Action dimension is 8.
DELTO_TWO_FINGER_ACTIONS = RelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=[DELTO_TWO_FINGER_JOINT_REGEX],
    scale=DELTO_TWO_FINGER_ACTION_SCALE,
    clip=DELTO_TWO_FINGER_ACTION_CLIP,
    use_zero_offset=True,
)
