# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dexterous lift and reorient environments for the UR5e + Tesollo DELTO DG-5F.

Sibling of :mod:`.dexlift_ur10e_delto_env_cfg`; read that module's docstring first, because the two
invariants it names hold here identically:

* ``scene.robot`` is the shared :data:`IMPLICIT_UR5E_DELTO` articulation object, the same one every
  other UR5e+DELTO environment spawns. That sharing is the whole mechanism by which the identified
  UR5e dynamics -- and the ``Ur5eDelto/metadata.yaml`` next to its USD -- are common to both task
  families. A locally redefined robot config would look correct and silently drift.
* The dexsuite base randomizes actuator gains and joint friction over ``.*``. Left alone that scales
  the identified UR5e arm gains by anything from 0.5x to 2x. Both terms are narrowed to the hand
  joints here, and the arm instead gets OmniReset's ``randomize_arm_from_sysid_fixed``.

Three things are decided HERE rather than inherited, and each has its reason written at the
constant or the line that implements it:

* TIMING -- see :data:`CONTROL_RATE_NOTE`. This env inherits the dexsuite rate, not OmniReset's.
* The SCENE is OmniReset's UR5e rig (real table, mount plate, ground, sky) rather than dexsuite's
  invisible cuboid, so the work surface sits at :data:`WORK_SURFACE_Z` instead of dexsuite's
  0.255 m and every height in the task moves with it. See :data:`WORKSPACE_Z_SHIFT`.
* The abnormal-joint-velocity termination is SCOPED TO THE ARM instead of being deleted. See
  :func:`_scope_abnormal_robot_to_arm`.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.dexsuite import dexsuite_env_cfg as dexsuite
from isaaclab_tasks.manager_based.manipulation.dexsuite.adr_curriculum import CurriculumCfg as DexsuiteCurriculumCfg

import uwlab.envs.mdp as uwlab_mdp
from uwlab.envs.mdp.full_actuation import assert_action_cfg_fully_actuates

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
from uwlab_assets.robots.ur10e_delto.actions import DELTO_HAND_JOINT_NAMES
from uwlab_assets.robots.ur5e_delto import IMPLICIT_UR5E_DELTO

from ..omnireset import mdp as omnireset_mdp
from . import mdp

# HAND-side names, imported rather than restated. The hand is the same graft on both arms, the
# bodies keep their names through it, and these are read by the contact sensors, the contact-gated
# rewards and the actuator/friction narrowing -- four places that must agree. The rewards class is
# likewise hand-side only (it names fingertips and the table sensor, never an arm joint), so it is
# reused as-is instead of being copied under a UR5e name.
from .dexlift_ur10e_delto_env_cfg import (
    ALL_TIP_NAMES,
    HAND_JOINT_REGEX,
    HAND_PRIM,
    PALM_BODY,
    TIP_BODY_REGEX,
    UR10eDeltoRewardsCfg as DeltoHandRewardsCfg,
)

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
"""The six UR5e joints whose dynamics are identified and shared with OmniReset.

Same six names as the UR10e -- it is the UR joint naming convention, not a shared robot. The
NUMBERS behind them (effort 150/150/150/28/28/28 N*m, velocity 1.5708 x3 then 3.1415 x3, and the
``sysid`` block) come from the UR5e and are carried by the articulation config, not by this list.
"""

ARM_STIFFNESS = 800.0
ARM_DAMPING = 40.0
"""Position-control gains for the arm.

The shared articulation ships the arm at zero stiffness and damping because OmniReset drives it
through an operational-space controller that supplies joint torques directly. This task instead
uses ``RelativeJointPositionAction``, which writes joint position targets, so the arm needs real PD
gains or it simply will not move. Gains are the controller, not the plant: the identified
quantities -- armature, joint friction, link inertials -- still come from the shared config and its
``metadata.yaml``, untouched.

The PAIR is inherited from the UR10e+DELTO task rather than re-derived, and that is a known
approximation, not a measurement. Torque stays bounded by this arm's own ``effort_limit_sim``
(150/150/150/28/28/28 N*m, half the UR10e's wrist authority), so the gain sets a slew rate rather
than a force; but the UR5e's links are lighter, so the same gains give a higher natural frequency
and a lower damping ratio than they do on the UR10e. Retuning needs a running simulator and is
listed as open.
"""

CONTROL_RATE_NOTE = "dexsuite: decimation 2, sim.dt 1/120 -> 60 Hz policy, episode_length_s 4.0"
"""THE TIMING DECISION, made explicitly: this env inherits DEXSUITE's rate, not OmniReset's.

OmniReset RL state runs decimation 12 at ``sim.dt`` 1/120 with ``episode_length_s`` 16.0 -- a 10 Hz
policy and 160 steps per episode. Nothing here changes ``decimation``, ``sim.dt`` or
``episode_length_s``, so all three stay at the dexsuite values quoted above.

Why that way round. Everything this environment's behaviour is made of was tuned at the dexsuite
rate and is per-STEP, not per-second:

* the dexsuite reward weights, and in particular ``action_l2`` / ``action_rate_l2``, which are
  summed once per policy step -- at 10 Hz they would contribute a sixth as much against unchanged
  task rewards;
* the ADR curriculum's promotion and demotion counters, which count steps;
* this package's own hand constants: ``DELTO_HAND_ACTION_SCALE`` 0.1 rad per unit action and
  ``DELTO_HAND_ACTION_CLIP`` +-1, whose docstring derives its bound explicitly "at the 60 Hz control
  rate", against an actuator capped at 3.0 rad/s. At 10 Hz the same clip permits 1 rad/s of
  commanded joint motion and the fingers close six times more slowly per step.

What is being inherited by that choice, stated plainly: the dexsuite reward scales, the dexsuite
ADR schedule, dexsuite's 4-second episode, and dexsuite's 60 Hz action semantics.

What is NOT at risk: the UR5e arm identification. ``sysid`` (armature, static/dynamic friction,
viscous friction) and ``link_inertials`` are plant parameters with units, not per-step quantities;
they are applied to the articulation and are independent of how often the policy is asked for an
action. The one genuinely rate-sensitive arm quantity, the actuator DELAY of
``DelayedPDActuatorCfg`` (min_delay 0, max_delay 1, counted in sim steps), is inert here because
the implicit actuator has no delay buffer -- the same footnote the UR10e task carries.
"""

##
# Scene geometry.
##

FLIP_BASE_YAW = False
"""The robot base is never rotated. Kept as a named constant so the decision stays greppable.

Identical reasoning to the UR10e task: the OmniReset cameras are parented to the robot ROOT prim
with offsets in the root frame, so a 180-degree root yaw here and not there would make identical
extrinsics resolve to different world poses. The scene is mirrored instead; see
:data:`WORKSPACE_X`. One base orientation across both task families, matching the single
orientation of the physical rig.
"""

WORKSPACE_X = 0.55
"""Workspace centre, mirroring the inherited dexsuite scene from -x to +x.

Note the frame subtlety: ``base`` and ``base_link_inertia`` are rotated 180 degrees about z from
``base_link`` per the ROS convention. This value is stated in ``base_link``, the frame
``init_state.rot`` and the articulation root use.
"""

WORK_SURFACE_Z = -0.013
"""Height of the OmniReset UR5e work surface in the robot's own root frame.

Taken from the mount plate's root z below, which is that scene's object-placement datum: the
current OmniReset UR5e scene states the convention outright ("ROOT z = the WORK SURFACE ... the
object-reset placement datum"), and this is the value that plate had in the pat_vention rig.

UNVERIFIED WITHOUT ISAAC. Nothing in this repository can open ``pat_vention.usd`` offline, so the
tabletop's true z is inferred from the plate, not measured. If the objects rest visibly above or
below the table on the first launch, this single constant is the thing to correct -- every other
height in this module is derived from it through :data:`WORKSPACE_Z_SHIFT`.
"""

GROUND_Z = -0.868
"""Floor height, from the same OmniReset UR5e scene."""

DEXSUITE_TABLE_TOP_Z = 0.255
"""Where the INHERITED dexsuite cuboid table's top sits: centre 0.235 + half of its 0.04 thickness.

Not a choice, a reading of the base config. It is the datum every inherited height in the dexsuite
task -- object spawn, goal range -- is implicitly written against.
"""

WORKSPACE_Z_SHIFT = WORK_SURFACE_Z - DEXSUITE_TABLE_TOP_Z
"""Vertical offset applied to every inherited height, so the task keeps its shape on the real rig.

The dexsuite geometry is preserved RELATIVE to the work surface (object spawned 95 mm above it,
goals 0.295-0.695 m above it) rather than re-authored, so the reward stds and the ADR tolerances
still describe the same distances they were tuned against.
"""


def _omnireset_table_cfg() -> RigidObjectCfg:
    """The OmniReset UR5e work table, replacing dexsuite's invisible cuboid.

    Contact reporting is on because ``rewards.table_contact_penalty`` reads a contact sensor
    bound to this prim.

    NOTE, and this is the honest state of it: the CURRENT OmniReset UR5e scene has moved on to a
    procedurally generated ``CustomLabTable`` measured off the real rig -- but that USD is not
    committed (only ``table_dims.yaml`` is; the USD is produced by ``make_custom_table_usd.py``),
    so an environment pinned to it cannot spawn from a fresh clone. This is therefore the last
    scene OmniReset had that resolves from committed assets: the pat_vention table and the UR5
    mount plate, both cloud assets, at their pat_vention-era poses. When the custom table asset is
    published, this factory and :data:`WORK_SURFACE_Z` are what move.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, -0.881), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention/pat_vention.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )


def _omnireset_mount_plate_cfg() -> RigidObjectCfg:
    """The plate the UR5e is bolted to. Its root z is the work-surface datum; see WORK_SURFACE_Z."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/UR5MetalSupport",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, WORK_SURFACE_Z), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention2/Ur5MetalSupport/ur5plate.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )


##
# MDP settings.
##


@configclass
class Ur5eDeltoRelJointPosActionCfg:
    """Relative joint-position control over all 26 joints at 0.1 rad per unit action."""

    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        # Spelled out rather than given as a scalar so that an unmatched joint is visible here: any
        # joint missing from this dict would silently fall back to scale 1.0. The hand keys come
        # from the asset's canonical tuple, which is also what the full-actuation guard reads to
        # decide, without an articulation, that this term spends one action dimension per joint.
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
class Ur5eDeltoEventCfg(dexsuite.EventCfg):
    """Base events plus the arm-dynamics terms that keep this env in step with OmniReset.

    ``randomize_arm_sysid`` is OmniReset's own event class, reading the ``sysid`` block of
    ``Ur5eDelto/metadata.yaml`` next to the robot USD -- which the graft copied byte-equal from the
    calibrated UR5e's own metadata, so these are UR5e numbers and not the UR10e placeholder chain.
    The ``_fixed`` variant pins the curriculum progress at 1.0, so the arm always carries the
    identified armature and friction scaled by +/-20%. With an implicit actuator there is no delay
    buffer to write, so ``delay_range`` is inert here; it is kept at OmniReset's value so the two
    configs read the same.
    """

    # startup: the full-actuation guard, checked against the joints the action terms RESOLVED to.
    # The construction-time half runs in ``Ur5eDeltoMixinCfg.__post_init__`` and reasons about
    # action-cfg PATTERNS; this one sees the twenty joints the regex actually matched on the
    # spawned articulation. ``startup`` specifically, because the env applies startup events
    # directly, whereas anything resolved during scene/manager construction runs inside a timeline
    # PLAY callback whose exceptions Omniverse prints and then swallows.
    check_hand_fully_actuated = EventTerm(
        func=uwlab_mdp.check_action_manager_fully_actuates,
        mode="startup",
        params={"required_joints": list(DELTO_HAND_JOINT_NAMES), "context": "dexlift UR5e+DELTO"},
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


def _scope_abnormal_robot_to_arm(env_cfg) -> None:
    """Point the abnormal-joint-velocity termination at the ARM's six joints only.

    THE FAILURE THIS PREVENTS. ``abnormal_robot_state`` ends the episode when any joint exceeds
    twice its velocity limit, and the paired ``early_termination`` reward pays a large negative for
    it (-1 in the dexsuite base; the OmniReset state task, which this robot's other family uses,
    weights the same signal -100). Both upstream implementations accept a ``SceneEntityCfg`` and
    then ignore its joint selection, reducing over EVERY joint of the articulation. On a 6-joint
    arm that is a sane instability detector. On this articulation, twenty of the twenty-six joints
    are soft force-limited fingers whose actuator cap is 3.0 rad/s while the USD independently
    authors ``physxJoint:maxJointVelocity`` 7.31 rad/s -- so a finger snapping back as it releases
    an object legitimately crosses the 2x threshold, kills the episode, and charges the penalty to
    a policy that did nothing wrong. The gradient that produces is "never let go".

    WHY NOT JUST DELETE IT, which is what the UR10e task does (``terminations.abnormal_robot`` and
    ``rewards.early_termination`` both set to None). Deleting it also throws away the arm-side
    instability cut, and a UR5e whose solver has diverged then runs the episode out at nonsense
    velocities and feeds that data to PPO. Scoping keeps the detector where it means something.

    This is a function rather than three lines in ``__post_init__`` because it is the kind of thing
    that gets silently reverted by the next config that inherits the mixin; a named call is
    greppable and can be re-asserted.
    """
    env_cfg.terminations.abnormal_robot = DoneTerm(
        func=mdp.abnormal_robot_state_scoped,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)},
    )
    # The penalty term stays, and stays bound to this key: with the cut scoped, firing it really
    # does mean the arm went unstable, which is worth a negative.
    if env_cfg.rewards.early_termination is not None:
        env_cfg.rewards.early_termination.params["term_keys"] = "abnormal_robot"


##
# ADR curriculum: the upstream noise-width bug.
##


def _assert_adr_noise_bounds_are_ranges(curriculum) -> None:
    """Fail construction if any ADR noise term ramps ``n_min`` and ``n_max`` to the SAME value.

    ``AdditiveUniformNoiseCfg`` computes ``data + rand() * (n_max - n_min) + n_min``. Equal bounds
    therefore collapse the width to zero and the term stops being noise: it becomes a constant
    BIAS of that value, applied to every element of the observation, every step, forever. Nothing
    upstream checks this, and it is invisible in a config diff -- which is exactly how
    ``object_obs_unoise_max_adr`` came to carry ``final_value: -0.01``, a permanent -1 cm shift of
    the object point cloud at full difficulty rather than the intended +-1 cm jitter.

    The check pairs terms by name (``<x>_unoise_min_adr`` / ``<x>_unoise_max_adr``) and looks at
    the values they interpolate TOWARDS, since that is where a sign typo hides; a term set to None
    by a subclass is skipped. It runs at config construction, from the caller's own stack frame.
    """
    terms = {name: term for name, term in vars(curriculum).items() if not name.startswith("_") and term is not None}
    for name, min_term in terms.items():
        if not name.endswith("_unoise_min_adr"):
            continue
        max_name = name[: -len("_unoise_min_adr")] + "_unoise_max_adr"
        max_term = terms.get(max_name)
        if max_term is None:
            continue
        min_final = min_term.params["modify_params"]["final_value"]
        max_final = max_term.params["modify_params"]["final_value"]
        if min_final == max_final:
            raise ValueError(
                f"ADR noise term pair '{name}' / '{max_name}' ramps to the SAME final value"
                f" ({min_final}). uniform_noise is 'data + rand() * (n_max - n_min) + n_min', so"
                " equal bounds give a zero-width interval: this is a constant bias of"
                f" {min_final}, not noise. Fix the pair so that n_max > n_min at full difficulty."
                f"\nChecked by: {__name__}._assert_adr_noise_bounds_are_ranges"
            )
        if max_final < min_final:
            raise ValueError(
                f"ADR noise term pair '{name}' / '{max_name}' has n_max ({max_final}) below n_min"
                f" ({min_final}) at full difficulty; the sampled interval is inverted."
            )


@configclass
class Ur5eDeltoAdrCurriculumCfg(DexsuiteCurriculumCfg):
    """The dexsuite ADR curriculum with the point-cloud noise width repaired.

    Upstream (``isaaclab_tasks/.../dexsuite/adr_curriculum.py``) ramps BOTH bounds of the object
    point-cloud noise to -0.01: ``object_obs_unoise_min_adr`` and ``object_obs_unoise_max_adr``.
    The perception observation therefore gets a rigid -1 cm offset at full difficulty instead of
    +-1 cm of jitter -- a systematic error a policy simply learns to subtract, which is the
    opposite of what the term is for.

    Fixed HERE, in this package, by overriding the one term. The vendored IsaacLab tree is outside
    UWLab's git and ``uwlab.sh`` recreates it, so an edit there is undone by the next setup run.
    The sibling ``TableLegCurriculumCfg`` subclasses this same base config the same way.
    """

    object_obs_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.perception.object_point_cloud.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            # +0.01, against the -0.01 of the min term. Upstream has -0.01 here.
            "modify_params": {"initial_value": 0.0, "final_value": 0.01, "difficulty_term_str": "adr"},
        },
    )

    def __post_init__(self):
        # Construction-time, so a future edit that reintroduces the collapse cannot reach training.
        _assert_adr_noise_bounds_are_ranges(self)


##
# Environment configuration.
##


@configclass
class Ur5eDeltoMixinCfg:
    """Everything the UR5e + DELTO changes about the base dexsuite task."""

    rewards: DeltoHandRewardsCfg = DeltoHandRewardsCfg()
    actions: Ur5eDeltoRelJointPosActionCfg = Ur5eDeltoRelJointPosActionCfg()
    events: Ur5eDeltoEventCfg = Ur5eDeltoEventCfg()
    curriculum: Ur5eDeltoAdrCurriculumCfg = Ur5eDeltoAdrCurriculumCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()

        # THE FULL-ACTUATION GUARD, at construction time. Every dexlift UR5e+DELTO env inherits
        # this mixin, so this one line is the whole family's construction-time coverage. It raises
        # from this frame; the ``check_hand_fully_actuated`` startup term in Ur5eDeltoEventCfg
        # re-checks the same property against the resolved articulation.
        assert_action_cfg_fully_actuates(self.actions, DELTO_HAND_JOINT_NAMES, context=type(self).__name__)

        # -- TIMING: deliberately untouched. See CONTROL_RATE_NOTE for what that inherits.

        # -- robot: the same articulation object OmniReset uses, hence the same sysid metadata.
        # ``replace`` is a shallow dataclass copy, so nested cfgs are still shared with the module
        # level object; anything mutated below is replaced wholesale rather than edited in place,
        # which would otherwise reach every other environment built in the same process.
        robot_cfg = IMPLICIT_UR5E_DELTO.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # the fingertip contact sensors need contact reporters on the robot's bodies
        robot_cfg.spawn = robot_cfg.spawn.replace(activate_contact_sensors=True)
        # position targets need PD gains; see ARM_STIFFNESS
        robot_cfg.actuators = dict(robot_cfg.actuators)
        robot_cfg.actuators["arm"] = robot_cfg.actuators["arm"].replace(stiffness=ARM_STIFFNESS, damping=ARM_DAMPING)
        if FLIP_BASE_YAW:  # never; see the constant's docstring
            robot_cfg.init_state = robot_cfg.init_state.replace(rot=(0.0, 0.0, 0.0, 1.0))
        self.scene.robot = robot_cfg

        # -- scene: OmniReset's UR5e rig. The robot base stays at the origin; the table, the mount
        # plate, the floor and the sky are that scene's, so this task and the OmniReset ones look
        # at the same room. The dexsuite cuboid table is REPLACED (not added to) -- two work
        # surfaces would intersect -- and ``plane``/``sky_light`` are edited rather than duplicated,
        # since both already own their prim paths (/World/GroundPlane, /World/skyLight).
        self.scene.table = _omnireset_table_cfg()
        self.scene.ur5_metal_support = _omnireset_mount_plate_cfg()
        self.scene.plane.init_state.pos = (0.0, 0.0, GROUND_Z)
        self.scene.sky_light.spawn.intensity = 1000.0

        # -- workspace: mirror the inherited dexsuite layout from -x to +x so the base stays
        # unrotated, and drop every height onto the real work surface. The relative geometry is
        # preserved so the reward stds still mean what they meant; see WORKSPACE_Z_SHIFT.
        self.scene.object.init_state.pos = (WORKSPACE_X, 0.1, 0.35 + WORKSPACE_Z_SHIFT)
        self.commands.object_pose.ranges.pos_x = (0.3, 0.7)
        self.commands.object_pose.ranges.pos_z = (0.55 + WORKSPACE_Z_SHIFT, 0.95 + WORKSPACE_Z_SHIFT)
        # The inherited bound box is written for the -x, table-at-0.235 workspace. Left alone, an
        # object at +0.55 is out of bounds on the first frame and every episode terminates
        # immediately; and with the surface now below z=0, an object simply RESTING on the table
        # would read as out of bounds too.
        self.terminations.object_out_of_bound.params["in_bound_range"]["x"] = (-0.5, 1.5)
        self.terminations.object_out_of_bound.params["in_bound_range"]["z"] = (WORK_SURFACE_Z - 0.05, 2.0)
        # keep the debug camera looking at the workspace rather than away from it
        self.viewer.eye = (2.25, 0.0, 0.75)
        self.viewer.lookat = (WORKSPACE_X, 0.0, WORK_SURFACE_Z + 0.2)

        # -- contact sensors: one per fingertip, filtered to the object, plus the table.
        # These resolve PRIM PATHS, not body names. The hand is referenced under {ROOT}/gripper by
        # ``graft_delto_on_ur5e.py`` (the script fixes that prim name precisely because these paths
        # address it), so a fingertip sits at {ENV_REGEX_NS}/Robot/gripper/rl_dg_<n>_tip. A wrong
        # path yields silent zeros forever and every contact-gated reward pays nothing.
        for link_name in ALL_TIP_NAMES:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{HAND_PRIM}/{link_name}",
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        # NOTE the prim path moved with the table: OmniReset spawns it at {ENV_REGEX_NS}/Table,
        # where dexsuite's cuboid was at .../table.
        self.scene.table_s = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Table",
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
        # overwrite the identified UR5e dynamics that randomize_arm_sysid is there to reproduce --
        # a 0.5x-2x scale over gains that were fitted, not chosen. This is also exactly how
        # OmniReset splits it (gripper joints randomized, arm joints not).
        self.events.joint_stiffness_and_damping.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[HAND_JOINT_REGEX]
        )
        self.events.joint_friction.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[HAND_JOINT_REGEX])
        # The base term names the reference robot's own wrist joint (``iiwa7_joint_7``).
        self.events.reset_robot_wrist_joint.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=["wrist_3_joint"])
        # The reference robot floats and needs its root pinned every reset. Ours is bolted to the
        # table through a fixed root joint -- the graft keeps the arm's ``root_joint`` and asserts
        # it is still the one articulation root -- so there is nothing to pin.
        self.events.reset_root = None

        # -- terminations: keep the instability cut, but only over the arm's joints.
        _scope_abnormal_robot_to_arm(self)


@configclass
class DexLiftUR5eDeltoReorientEnvCfg(Ur5eDeltoMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexLiftUR5eDeltoReorientEnvCfg_PLAY(Ur5eDeltoMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    pass


@configclass
class DexLiftUR5eDeltoLiftEnvCfg(Ur5eDeltoMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexLiftUR5eDeltoLiftEnvCfg_PLAY(Ur5eDeltoMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
