# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dexterous lift and reorient environments for the UR10e + Tesollo DELTO DG-5F.

The scene, observations, events and ADR curriculum all come from the dexsuite base package
vendored in IsaacLab; this module supplies only what is robot specific -- the articulation, the
action term, the fingertip contact sensors, and the contact names that every gated reward needs.

Two things here are load bearing and easy to break:

* ``scene.robot`` is the very same :data:`IMPLICIT_UR10E_DELTO` object the OmniReset environments
  use. That shared articulation is the entire mechanism by which the identified UR10e dynamics are
  common to both task families -- a second, locally defined robot config would look correct and
  silently drift.
* The dexsuite base randomizes actuator gains and joint friction over ``.*``. Left alone that would
  scale the identified arm gains by anything from 0.5x to 2x. Both terms are narrowed to the hand
  joints here and the arm instead gets OmniReset's own ``randomize_arm_from_sysid_fixed``.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.dexsuite import dexsuite_env_cfg as dexsuite

import uwlab.envs.mdp as uwlab_mdp
from uwlab.envs.mdp.full_actuation import assert_action_cfg_fully_actuates

from uwlab_assets.robots.ur10e_delto import IMPLICIT_UR10E_DELTO
from uwlab_assets.robots.ur10e_delto.actions import DELTO_HAND_JOINT_NAMES

from ..omnireset import mdp as omnireset_mdp
from . import mdp

##
# Robot-specific names.
##

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
"""The six UR10e joints whose dynamics are identified and shared with OmniReset."""

HAND_JOINT_REGEX = r"rj_dg_(1|2|3|4|5)_(1|2|3|4)"
"""The twenty DELTO joints. Every joint of the articulation is either an arm joint or one of these."""

HAND_PRIM = "gripper"
"""Prim that the DELTO subtree is grafted under, i.e. ``{ENV_REGEX_NS}/Robot/gripper/<link>``.

The UR10e graft script nests the end effector under this name; the linear-gripper build does the
same (its wrist camera lives at ``Robot/gripper/robotiq_base_link``). Contact sensors address prim
paths rather than body names, so this has to match the USD exactly or the sensors silently find
nothing.
"""

PALM_BODY = "rl_dg_mount"
THUMB_TIP_NAMES = ("rl_dg_1_tip", "rl_dg_5_tip")
"""The DELTO has two opposable digits, so the contact gate takes a sequence of thumbs."""
TIP_NAMES = ("rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip")
ALL_TIP_NAMES = THUMB_TIP_NAMES + TIP_NAMES
TIP_BODY_REGEX = r"rl_dg_(1|2|3|4|5)_tip"

ARM_STIFFNESS = 800.0
ARM_DAMPING = 40.0
"""Position-control gains for the arm, matching the reference UR10 dexsuite config.

The shared articulation ships the arm at zero stiffness and damping because OmniReset drives it
through an operational-space controller that supplies joint torques directly. This task instead
uses ``RelativeJointPositionAction``, which writes joint position targets, so the arm needs real PD
gains or it simply will not move. Gains are the controller, not the plant: the identified
quantities -- armature, joint friction, link inertials -- still come from the shared config and its
``metadata.yaml``, untouched.
"""

FLIP_BASE_YAW = False
"""The robot base is never rotated. Kept as a named constant so the decision stays greppable.

Our UR10e natively reaches +x from ``base_link`` (FK puts ``wrist_3_link`` at x = +0.691 in that
frame), and every OmniReset reset range is already positive x with no base rotation anywhere. The
vendored dexsuite scene instead sits at x = -0.55, which can be reconciled either by yawing the
robot to face it or by mirroring the scene. This env mirrors the scene, per :data:`WORKSPACE_X`.

Do not set this True. It is not a free choice between two equivalent conventions: the OmniReset RGB
cameras are parented to the robot ROOT prim with offsets in the root frame, so a 180-degree root
yaw here and not there would make identical extrinsics resolve to different world poses -- camera
parity would hold on paper and fail in fact. One base orientation across both envs, matching the
single orientation of the physical rig, is what makes that parity real.
"""

WORKSPACE_X = 0.55
"""Workspace centre, mirroring the inherited dexsuite scene from -x to +x.

Note the frame subtlety this depends on: ``base`` and ``base_link_inertia`` are rotated 180 degrees
about z from ``base_link`` per the ROS convention. The reach direction above is stated in
``base_link``, which is the frame ``init_state.rot`` and the articulation root use, and which the
goal command composes against. Anything that later binds a goal frame or a camera to
``body_names="base"`` flips sign relative to this.
"""


##
# MDP settings.
##


@configclass
class UR10eDeltoRelJointPosActionCfg:
    """Relative joint-position control over all 26 joints at 0.1 rad per unit action."""

    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        # spelled out rather than given as a scalar so that an unmatched joint is visible here:
        # any joint missing from this dict would silently fall back to scale 1.0.
        #
        # The twenty hand entries were previously ONE regex key. Same 26 dimensions either way, but
        # the full-actuation guard's construction-time half reads this mapping to decide, without
        # an articulation, which joints get their own action dimension -- and a regex key names no
        # joint in particular. Spelling them out from the asset's canonical tuple states the
        # property the guard needs and removes the fourth independent copy of the hand regex.
        scale={
            "shoulder_pan_joint": 0.1,
            "shoulder_lift_joint": 0.1,
            "elbow_joint": 0.1,
            "wrist_1_joint": 0.1,
            "wrist_2_joint": 0.1,
            "wrist_3_joint": 0.1,
            **{name: 0.1 for name in DELTO_HAND_JOINT_NAMES},
        },
    )


@configclass
class UR10eDeltoRewardsCfg(dexsuite.RewardsCfg):
    """Base rewards with the contact-gated terms repointed at the dexlift implementations.

    ``position_tracking``, ``orientation_tracking`` and ``success`` are redeclared rather than
    patched because the upstream versions gate on hardcoded Kuka-Allegro sensor names (and
    ``success_reward`` does not gate at all). The four terms after them are the additions the
    reference makes on top of the base task.
    """

    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.2,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
            "thumb_contact_name": THUMB_TIP_NAMES,
            "tip_contact_names": TIP_NAMES,
        },
    )

    orientation_tracking = RewTerm(
        func=mdp.orientation_command_error_tanh,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 1.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
            "thumb_contact_name": THUMB_TIP_NAMES,
            "tip_contact_names": TIP_NAMES,
        },
    )

    success = RewTerm(
        func=mdp.success_reward,
        weight=10,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_std": 0.1,
            "rot_std": 0.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
            "thumb_contact_name": THUMB_TIP_NAMES,
            "tip_contact_names": TIP_NAMES,
        },
    )

    # at least one thumb and one other fingertip on the object
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=2.0,
        params={
            "threshold": 0.2,
            "thumb_contact_name": THUMB_TIP_NAMES,
            "tip_contact_names": TIP_NAMES,
        },
    )

    # any fingertip at all, a much easier signal that shapes the approach
    any_finger_contact = RewTerm(
        func=mdp.any_contact,
        weight=1.0,
        params={"threshold": 0.2, "contact_names": ALL_TIP_NAMES},
    )

    # discourage dragging: only fires while the object is gripped AND still loading the table
    table_contact_penalty = RewTerm(
        func=mdp.table_contact_penalty,
        weight=-0.5,
        params={
            "table_contact_name": "table_s",
            "threshold": 0.2,
            "thumb_contact_name": THUMB_TIP_NAMES,
            "tip_contact_names": TIP_NAMES,
        },
    )

    object_upward_motion = RewTerm(
        func=mdp.object_upward_velocity_bonus,
        weight=0.5,
        params={
            "std": 0.2,
            "threshold": 0.2,
            "thumb_contact_name": THUMB_TIP_NAMES,
            "tip_contact_names": TIP_NAMES,
        },
    )


@configclass
class UR10eDeltoEventCfg(dexsuite.EventCfg):
    """Base events plus the arm-dynamics terms that keep this env in step with OmniReset.

    ``randomize_arm_sysid`` is OmniReset's own event class, reading the ``sysid`` block of
    ``Ur10eDelto/metadata.yaml`` next to the robot USD. The ``_fixed`` variant pins the curriculum
    progress at 1.0, so the arm always carries the identified armature and friction scaled by
    +/-20% -- the same dynamics OmniReset converges to at the end of its finetune stage. With an
    implicit actuator there is no delay buffer to write, so ``delay_range`` is inert here; it is
    kept at OmniReset's value so the two configs read the same.
    """

    # startup: the full-actuation guard, checked against the joints the action terms RESOLVED to.
    #
    # Every dexlift DELTO env inherits this class, so every one of them carries the check. It is
    # the runtime half of the pair: the construction-time half runs in each env config's
    # ``__post_init__`` and reasons about action-cfg PATTERNS, while this one sees the twenty
    # joints the regex actually matched on the spawned articulation -- a joint renamed in the USD,
    # or a scale mapping that quietly stopped covering one, is only visible here.
    #
    # ``startup`` specifically: the env applies startup events directly, whereas anything resolved
    # during scene/manager construction runs inside a timeline PLAY callback whose exceptions
    # Omniverse prints and then swallows. Same reasoning as omnireset's ``check_gripper_joints``.
    check_hand_fully_actuated = EventTerm(
        func=uwlab_mdp.check_action_manager_fully_actuates,
        mode="startup",
        params={"required_joints": list(DELTO_HAND_JOINT_NAMES), "context": "dexlift UR10e+DELTO"},
    )

    reset_robot_elbow_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="elbow_joint"),
            "position_range": [-0.2, 0.2],
            "velocity_range": [0.0, 0.0],
        },
    )

    randomize_arm_sysid = EventTerm(
        func=omnireset_mdp.randomize_arm_from_sysid_fixed,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": ARM_JOINT_NAMES,
            "actuator_name": "arm",
            "scale_range": (0.8, 1.2),
            "delay_range": (0, 1),
        },
    )


##
# Environment configuration.
##


@configclass
class UR10eDeltoMixinCfg:
    """Everything the UR10e + DELTO changes about the base dexsuite task."""

    rewards: UR10eDeltoRewardsCfg = UR10eDeltoRewardsCfg()
    actions: UR10eDeltoRelJointPosActionCfg = UR10eDeltoRelJointPosActionCfg()
    events: UR10eDeltoEventCfg = UR10eDeltoEventCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()

        # THE FULL-ACTUATION GUARD, at construction time. Every dexlift DELTO env -- reorient,
        # lift, table leg and their play/curriculum variants -- inherits this mixin, so this one
        # line is the whole family's construction-time coverage. It raises from this frame; the
        # ``check_hand_fully_actuated`` startup term in UR10eDeltoEventCfg re-checks the same
        # property against the resolved articulation.
        assert_action_cfg_fully_actuates(self.actions, DELTO_HAND_JOINT_NAMES, context=type(self).__name__)

        # -- robot: the same articulation object OmniReset uses, hence the same sysid metadata.
        # ``replace`` is a shallow dataclass copy, so nested cfgs are still shared with the module
        # level object; the init state is replaced wholesale rather than mutated in place, which
        # would otherwise yaw the robot for every other environment built in the same process.
        robot_cfg = IMPLICIT_UR10E_DELTO.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # the fingertip contact sensors need contact reporters on the robot's bodies
        robot_cfg.spawn = robot_cfg.spawn.replace(activate_contact_sensors=True)
        # position targets need PD gains; see ARM_STIFFNESS
        robot_cfg.actuators = dict(robot_cfg.actuators)
        robot_cfg.actuators["arm"] = robot_cfg.actuators["arm"].replace(stiffness=ARM_STIFFNESS, damping=ARM_DAMPING)
        if FLIP_BASE_YAW:  # never; see the constant's docstring
            robot_cfg.init_state = robot_cfg.init_state.replace(rot=(0.0, 0.0, 0.0, 1.0))
        self.scene.robot = robot_cfg

        # -- mirror the inherited dexsuite workspace from -x to +x so the base stays unrotated
        self.scene.table.init_state.pos = (WORKSPACE_X, 0.0, 0.235)
        self.scene.object.init_state.pos = (WORKSPACE_X, 0.1, 0.35)
        # already correct in the base frame once the base is not rotated
        self.commands.object_pose.ranges.pos_x = (0.3, 0.7)
        # The inherited bound box is written for the -x workspace. Left alone, an object at +0.55
        # is out of bounds on the first frame and every episode terminates immediately.
        self.terminations.object_out_of_bound.params["in_bound_range"]["x"] = (-0.5, 1.5)
        # keep the debug camera looking at the workspace rather than away from it
        self.viewer.eye = (2.25, 0.0, 0.75)

        # -- contact sensors: one per fingertip, filtered to the object, plus the table
        self.scene.table.spawn.activate_contact_sensors = True
        for link_name in ALL_TIP_NAMES:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{HAND_PRIM}/{link_name}",
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        self.scene.table_s = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/table",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )

        # -- observations
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in ALL_TIP_NAMES]},
            clip=(-20.0, 20.0),  # fingertip contact forces stay well under 20 N
        )
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = [
            PALM_BODY,
            TIP_BODY_REGEX,
        ]

        # -- reaching reward looks at the fingertips only, not the palm
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=[TIP_BODY_REGEX])

        # -- events
        # Narrow the blanket actuator/friction randomization to the hand. On the arm it would
        # overwrite the identified dynamics that randomize_arm_sysid is there to reproduce; this
        # is also exactly how OmniReset splits it (gripper joints randomized, arm joints not).
        self.events.joint_stiffness_and_damping.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[HAND_JOINT_REGEX]
        )
        self.events.joint_friction.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[HAND_JOINT_REGEX])
        self.events.reset_robot_wrist_joint.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=["wrist_3_joint"])
        # The reference robot floats and needs its root pinned every reset. Ours is bolted to the
        # table through a fixed root joint, so there is nothing to pin.
        self.events.reset_root = None

        # -- terminations: the reference disables the abnormal-velocity cut and its penalty, which
        # otherwise ends episodes on transients the policy can recover from
        self.terminations.abnormal_robot = None
        self.rewards.early_termination = None


@configclass
class DexLiftUR10eDeltoReorientEnvCfg(UR10eDeltoMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexLiftUR10eDeltoReorientEnvCfg_PLAY(UR10eDeltoMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    pass


@configclass
class DexLiftUR10eDeltoLiftEnvCfg(UR10eDeltoMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexLiftUR10eDeltoLiftEnvCfg_PLAY(UR10eDeltoMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
