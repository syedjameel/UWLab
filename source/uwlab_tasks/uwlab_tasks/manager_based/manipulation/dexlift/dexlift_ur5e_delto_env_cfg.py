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

Four things are decided HERE rather than inherited, and each has its reason written at the
constant or the line that implements it:

* TIMING -- see :data:`CONTROL_RATE_NOTE`. This env inherits the dexsuite rate, not OmniReset's.
* The SCENE is OmniReset's UR5e rig (real table, mount plate, ground, sky) rather than dexsuite's
  invisible cuboid, so the work surface sits at :data:`WORK_SURFACE_Z` instead of dexsuite's
  0.255 m and every height in the task moves with it. See :data:`WORKSPACE_Z_SHIFT`.
* The abnormal-joint-velocity termination and its penalty are DELETED, because on this
  articulation the test cannot fire. See :func:`_drop_unreachable_abnormal_robot_cut`.
* The GOAL and RED/GREEN TASK-STATE markers are rebound to this scene and to the scored success
  predicate. dexsuite owns the markers; what it cannot know is that its indicator geometry was
  cloned from a table this env then replaced. See :func:`_bind_task_state_visualization` and
  :data:`STATUS_PAD_TOP_Z`.

THE ACTION SPACE IS NOT DECIDED HERE. :class:`Ur5eDeltoMixinCfg` deliberately sets NO ``actions``
field: it carries the robot, the scene, the sensors and the events, and one subclass per action-
space variant supplies the action group. :class:`Ur5eDeltoRelJointPosMixinCfg` below is variant 1,
the DexSuite-style 26-DOF relative joint-position space, defined in
:mod:`.dexlift_ur5e_delto_actions`.

Forgetting that override is loud rather than quiet, and by construction: the base
``dexsuite.ActionsCfg`` is an empty ``pass``, so an env that does not supply an action group has no
action terms at all, and the full-actuation guard called from ``__post_init__`` reports every hand
joint as having NO action term that names it.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.dexsuite import dexsuite_env_cfg as dexsuite
from isaaclab_tasks.manager_based.manipulation.dexsuite.adr_curriculum import CurriculumCfg as DexsuiteCurriculumCfg

import uwlab.envs.mdp as uwlab_mdp
from uwlab.envs.mdp.full_actuation import assert_action_cfg_fully_actuates

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR
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

# ARM_JOINT_NAMES lives in the ACTIONS module, which is the one place that spends an action
# dimension on each of the six; the sysid event and the arm-scoped instability termination below
# read the same list rather than a second copy of it. Re-exported from here because that is where
# readers of this task look for it.
from .dexlift_ur5e_delto_actions import ARM_JOINT_NAMES, Ur5eDeltoRelJointPosActionCfg

##
# Robot-specific names.
##

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

WORK_SURFACE_Z = 0.004
"""Height of the OmniReset UR5e work surface in the robot's own root frame.

MEASURED, not inferred. The asset frame of ``custom_lab_table.usd`` IS the robot base frame --
origin at the base flange centre, z = 0 at the flange plane (the structural tabletop the base bolts
to), +x toward the workspace -- and the table therefore spawns AT the robot's default root rather
than at an offset. Two 4 mm mats lie ON that structural top, so the surface an object actually rests
on is 4 mm higher; ``table_dims.yaml`` states it directly ("their top = the WORK SURFACE at +0.004")
and the generated USD's world bbox confirms it (z max = +0.004, floor at -0.676).

That is also the mount plate's root z below, which is this scene's object-placement datum by the
authors' own convention (plate root z == mat-top height), so the two agree by construction rather
than by coincidence.

THIS REPLACES -0.013, which was inferred from the pat_vention-era plate and is wrong by 17 mm. Every
other height in this module derives from this constant through :data:`WORKSPACE_Z_SHIFT`.
"""

GROUND_Z = -0.676
"""Floor height: ``table.body.z_bottom`` from ``table_dims.yaml``, matching main's ground plane.

680 mm from the floor to the work surface, of which the mats are the last 4 mm. The table's legs
reach exactly this z, so the rig stands on the ground plane instead of hovering over it -- which the
previous -0.868 (a pat_vention-era number) did not do.
"""

TABLE_TOP_X = (-0.35, 1.05)
TABLE_TOP_Y_HALF = 0.35
"""Extent of the work surface in the robot base frame, from ``table_dims.yaml``.

Recorded here because this table was MEASURED AROUND A UR10e (1.30 m reach) and is being used under
a UR5e (0.85 m). See :data:`REACHABILITY_NOTE`; nothing in this module rescales it.
"""

REACHABILITY_NOTE = "UR5e reaches x <= 0.835 on the work surface; the table runs to x = 1.05"
"""The consequence of putting a UR5e on a table sized for a UR10e, stated rather than papered over.

Distances below are from the SHOULDER origin (0, 0, 0.1625 -- the UR5e's d1 above the base flange),
which is the datum UR's 0.850 m reach figure is quoted against.

* The FURNITURE overhangs the arm. The UR5e's reach circle crosses the work surface at x = 0.835 m
  (y = 0); the tabletop runs to x = 1.05 m and its front corners sit 1.118 m out. The front ~215 mm
  of the table is simply unreachable. This is cosmetic here -- nothing is ever spawned or commanded
  there -- but it is why the table looks oversized next to this arm.
* The OBJECT SPAWN is comfortably inside: (0.55, 0.1, 0.099) is 0.563 m out.
* The GOAL BOX is inside except for one corner region. Sampling the mirrored, shifted box
  x (0.3, 0.7) * y (-0.25, 0.25) * z (0.299, 0.699) uniformly, 98.6% of the volume lies within
  0.850 m; the excess is the far-and-high corner, (0.7, +-0.25, 0.699) at 0.917 m.

FLAGGED, NOT RESCALED. Two reasons for leaving the inherited ranges alone. First, the 0.850 m figure
is to the tool FLANGE, and the goal is a pose for the OBJECT, which the hand holds roughly a palm's
length beyond the flange -- so the arm does not have to put its flange on the goal point, and the
1.4% of the box that is nominally out very likely is not. Second, shrinking the command ranges would
silently change the task the inherited dexsuite reward stds and ADR tolerances were tuned against,
which is a bigger change than the one it would be fixing.

UNVERIFIED WITHOUT ISAAC: whether that corner is genuinely attainable depends on the DELTO's grasp
offset and on wrist orientation at the goal, neither of which is decidable from the config alone. If
training shows the goal command saturating unreached at high x AND high z, this is the note to
revisit, and ``commands.object_pose.ranges.pos_z`` is the thing to lower.
"""

STATUS_PAD_BORDER = 0.08
STATUS_PAD_THICKNESS = 0.012
STATUS_PAD_TOP_Z = -0.006
"""Geometry of the RED/GREEN TASK-STATE PAD -- the "is the object on target right now" indicator.

Stock dexsuite makes its whole table the indicator by spawning the table ``visible=False`` and
letting the success marker (a clone of the table's own spawn config, tinted) be the thing you see.
That trick is not available here: this scene's table is a real, visible USD of the measured lab
bench, and hiding it would cost the scene its rig.

So the indicator is a SEPARATE slab, sized and placed so that it is legible without touching
anything the task uses. It is the tabletop footprint (:data:`TABLE_TOP_X`, :data:`TABLE_TOP_Y_HALF`)
grown by ``STATUS_PAD_BORDER`` in every horizontal direction, and it is tucked so its top face lies
``STATUS_PAD_TOP_Z`` -- 6 mm BELOW the structural tabletop plane at z = 0, hence 10 mm below the
work surface. The USD's own collider extents (probed with pxr) show the structural top slab
occupying z in [-0.030, 0.000] across the FULL footprint, so a 12 mm pad centred at -0.012 spans
[-0.018, -0.006] and is completely inside that slab wherever the two overlap.

What that buys, and it is the whole reason for the offsets rather than simply laying a coloured mat
on the table:

* NOTHING IS OCCLUDED. The workspace, the mats, the mount plate and any object resting on them are
  all above the pad and unobstructed.
* NO Z-FIGHTING. The 6 mm gap to the structural top and the 10 mm gap to the mats are far larger
  than any depth-buffer tolerance at this scale.
* NOTHING CAN REST ON IT OR CLIP IT. The pad is under the tabletop; the only part you see is the
  80 mm border protruding past the table edge on all four sides, which reads as a coloured frame
  around the bench from the debug camera's ~18-degree elevation and from directly above.

The pad is a marker, not a scene entity: no rigid body, no collider, no contact pair. See
:mod:`.mdp.task_state_vis` for that argument and for the visibility/cost gating.
"""

STATUS_PAD_SIZE = (
    (TABLE_TOP_X[1] - TABLE_TOP_X[0]) + 2 * STATUS_PAD_BORDER,
    2 * (TABLE_TOP_Y_HALF + STATUS_PAD_BORDER),
    STATUS_PAD_THICKNESS,
)

STATUS_PAD_OFFSET = (
    0.5 * (TABLE_TOP_X[0] + TABLE_TOP_X[1]),
    0.0,
    STATUS_PAD_TOP_Z - 0.5 * STATUS_PAD_THICKNESS,
)
"""Pad centre RELATIVE TO THE TABLE'S ROOT, which is what the marker is drawn at.

Not zero, and that is the point. The table's asset frame is the robot BASE frame, so its root sits
at the base flange, 0.35 m behind the tabletop centre -- whereas dexsuite's cuboid table had its
root at its own centre and upstream therefore draws the indicator at ``root_pos_w`` with no offset.
Left uncorrected the pad would be centred on the robot base.
"""

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
    """The REAL lab table, replacing dexsuite's invisible cuboid. Main's definition.

    Adopted verbatim from ``omnireset/config/ur5e_robotiq_2f85/rl_state_cfg.py`` on
    ``syedjameel/main``: same prim path, same USD, same identity pose. It is a LOCAL asset now --
    procedurally generated from the measured ``table_dims.yaml`` by
    ``scripts_v2/tools/conversions/make_custom_table_usd.py`` and committed alongside it -- so a
    fresh clone spawns this scene without reaching the asset cloud. That was the one thing blocking
    this env from using it; the previous ``pat_vention`` cloud table is gone.

    THE POSE IS IDENTITY, and that is the correction that matters. The asset frame IS the robot base
    frame, so the table spawns AT the robot's default root rather than at the (0.4, 0, -0.881) plus
    90-degree yaw the pat_vention rig needed. Everything the task measures against the surface moves
    with :data:`WORK_SURFACE_Z`.

    ONE DELIBERATE DIFFERENCE FROM MAIN: ``activate_contact_sensors=True``. Main's UR5e scene has no
    table contact sensor; this task does -- ``rewards.table_contact_penalty`` reads ``scene.table_s``,
    which is bound to this prim -- and without the flag the spawner writes no contact reporter and
    that sensor returns zeros forever, so the penalty would silently never pay.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Mounts/CustomLabTable/custom_lab_table.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )


def _omnireset_mount_plate_cfg() -> RigidObjectCfg:
    """The flush proxy plate in the mat's base cutout. Main's definition.

    No physical plate exists on the real rig; the entity is kept because its ROOT z is the
    object-placement datum the reset events are written against -- which is exactly
    :data:`WORK_SURFACE_Z`, hence the constant rather than main's literal ``0.004``. The USD authors
    its geometry BELOW its own root by the mat thickness, so the disk fills the cutout floor while
    the root sits on the work surface.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/UR5MetalSupport",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, WORK_SURFACE_Z), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Mounts/CustomLabTable/custom_mount_plate.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )


##
# MDP settings.
##


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


def _bind_task_state_visualization(env_cfg) -> mdp.TaskStateVisPoseCommandCfg:
    """Rebuild ``commands.object_pose`` so the goal and the red/green task state are both legible.

    THE MARKERS THEMSELVES ARE UPSTREAM'S. Vendored dexsuite's ``ObjectUniformPoseCommand`` already
    creates a goal-pose marker, a current-object-pose marker and a red/green success marker, and
    already toggles the first two on ``debug_vis``. Nothing here re-implements any of that; see
    :mod:`.mdp.task_state_vis` for exactly what is upstream and what is added.

    What this function decides is the three things upstream cannot know:

    * WHERE the state indicator goes -- :data:`STATUS_PAD_SIZE` and :data:`STATUS_PAD_OFFSET`,
      derived from the measured table rather than from dexsuite's cuboid. Without this the pad is a
      slab of the OLD table's dimensions centred on the robot base; upstream builds those prototypes
      inside ``super().__post_init__()``, which ran before this env swapped the table in, and it
      fails silently.
    * WHICH PREDICATE colours it. Both tolerances are read out of ``curriculum.adr``, i.e. the
      thresholded success the ADR scheduler promotes on (``pos_dist < pos_tol``, and ``rot_dist <
      rot_tol`` when it has one). They are NOT restated. Upstream's marker instead hardcodes
      0.05/0.5, which on the reorientation task disagrees with the scored rotation tolerance of
      ``rot_std / 2`` = 0.25 -- a marker that goes green on runs the curriculum counts as failures.
      Binding here means the two cannot drift.
    * HOW BIG the goal ball is: the position tolerance itself, so "the object is inside the sphere"
      is the success condition rather than a hint at it.

    The tolerances are read AFTER ``super().__post_init__()``, which is where dexsuite sets
    ``pos_tol = rewards.success.params["pos_std"] / 2`` and where the Lift subclass then drops
    ``rot_tol`` to None. Reading them earlier would capture the un-derived defaults.
    """
    adr_params = env_cfg.curriculum.adr.params if env_cfg.curriculum is not None else {}
    # The dexsuite base always sets pos_tol when a curriculum exists; the fallback is for a config
    # that deliberately runs with curriculum=None, where upstream's own literal is the only datum.
    pos_tol = adr_params.get("pos_tol") or 0.05
    rot_tol = adr_params.get("rot_tol")
    if not isinstance(pos_tol, (int, float)) or pos_tol <= 0.0:
        raise ValueError(
            f"curriculum.adr.params['pos_tol'] must be a positive distance; got {pos_tol!r}. The"
            " task-state marker colours itself with the same threshold the ADR scheduler promotes"
            " on, so an unusable value here would be an unusable success test there too."
        )
    return mdp.upgrade_pose_command_to_task_state_vis(
        env_cfg.commands.object_pose,
        success_pos_tol=float(pos_tol),
        success_rot_tol=float(rot_tol) if rot_tol is not None else None,
        status_pad_size=STATUS_PAD_SIZE,
        status_vis_offset=STATUS_PAD_OFFSET,
    )


def _drop_unreachable_abnormal_robot_cut(env_cfg) -> None:
    """Delete the abnormal-joint-velocity termination and its penalty. It cannot fire.

    ``abnormal_robot_state`` is ``|joint_vel| > 2 * data.joint_vel_limits``, and the paired
    ``early_termination`` reward pays a negative when it does (-1 in the dexsuite base; the
    OmniReset state task weights the same signal -100). The test compares the measured velocity
    against THE SAME NUMBER THAT WAS WRITTEN INTO PHYSX AS THE JOINT'S MAXIMUM:

        ``Articulation._process_actuators_cfg`` calls ``write_joint_velocity_limit_to_sim(
        actuator.velocity_limit_sim, ...)`` for EVERY actuator, implicit or explicit
        (isaaclab/assets/articulation/articulation.py:1773), and that method both fills
        ``_data.joint_vel_limits`` and calls ``root_physx_view.set_dof_max_velocities`` (:801-803).

    So PhysX brakes each joint toward exactly the value the termination divides by: 3.0 rad/s on
    the twenty hand joints, 1.5708/3.1415 rad/s on the six arm joints (both articulation variants
    set them). Twice that is not a state the solver produces. The term never evaluates True, the
    early-termination penalty never pays, and any prose about what it protects is describing a
    branch that is never taken.

    THIS REPLACES a "scope it to the arm" helper whose stated premise was fiction. It argued that
    the hand's fingers legitimately exceed the threshold because "the USD independently authors
    ``physxJoint:maxJointVelocity`` 7.31 rad/s" against an actuator cap of 3.0. Both halves are
    wrong: ``Robots/Ur5eDelto/ur5e_delto.usd`` authors 180 deg/s = 3.1416 rad/s on all 26 joints
    (checked with pxr; 419 deg/s = 7.31 rad/s appears in no USD in this repository), and the USD
    value is not independent -- the write above overwrites it at startup.

    WHAT IS ACTUALLY LOST by deleting it: nothing that was working. The instability this term is
    named for shows up as diverging joint POSITIONS, not velocities, precisely because the velocity
    is clamped; what catches a runaway here is ``object_out_of_bound`` plus the episode timeout.
    The UR10e+DELTO sibling reached the same end state (``abnormal_robot``/``early_termination``
    both None) and its docstring's claim that this was a mistake was itself the mistake.

    IF A REAL ARM-INSTABILITY CUT IS WANTED, it needs a criterion that can fire -- a joint-position
    or effort test, or dropping ``velocity_limit_sim`` from the arm actuator so the comparison has
    headroom. That second option changes the PLANT (and the shared articulation, which OmniReset
    also spawns), so it is a deliberate decision and not a config tweak.

    A named function rather than two lines in ``__post_init__``, so the decision is greppable.
    """
    env_cfg.terminations.abnormal_robot = None
    env_cfg.rewards.early_termination = None


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
    """Everything the UR5e + DELTO changes about the base dexsuite task EXCEPT the action space.

    There is deliberately no ``actions`` field here; see the module docstring. One subclass per
    action-space variant supplies it, and the guard below is what makes a missing override fail.
    """

    rewards: DeltoHandRewardsCfg = DeltoHandRewardsCfg()
    events: Ur5eDeltoEventCfg = Ur5eDeltoEventCfg()
    curriculum: Ur5eDeltoAdrCurriculumCfg = Ur5eDeltoAdrCurriculumCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()

        # THE FULL-ACTUATION GUARD, at construction time. Every dexlift UR5e+DELTO env of every
        # action-space variant inherits this mixin, so this one line is the whole family's
        # construction-time coverage, and it stays correct as variants are added: it reads whatever
        # ``self.actions`` the variant subclass supplied, including the empty
        # ``dexsuite.ActionsCfg`` a variant that forgot to supply one would inherit. It raises from
        # this frame; the ``check_hand_fully_actuated`` startup term in Ur5eDeltoEventCfg re-checks
        # the same property against the resolved articulation.
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

        # -- scene: OmniReset's UR5e rig, as ``syedjameel/main`` defines it. The robot base stays at
        # the origin and the table spawns there too, because the table's asset frame IS the base
        # frame. The dexsuite cuboid table is REPLACED (not added to) -- two work surfaces would
        # intersect -- and ``plane``/``sky_light`` are edited rather than duplicated, since both
        # already own their prim paths (/World/GroundPlane, /World/skyLight), which are also the
        # paths main uses.
        self.scene.table = _omnireset_table_cfg()
        self.scene.ur5_metal_support = _omnireset_mount_plate_cfg()
        self.scene.plane.init_state.pos = (0.0, 0.0, GROUND_Z)
        # main's dome light is intensity 1000 over the kloofendal_43d_clear_puresky_4k HDR. Only the
        # intensity is written here: the inherited dexsuite ``sky_light`` already names that exact
        # texture under ISAAC_NUCLEUS_DIR and differs from main in the intensity alone (750).
        self.scene.sky_light.spawn.intensity = 1000.0

        # -- workspace: mirror the inherited dexsuite layout from -x to +x so the base stays
        # unrotated, and drop every height onto the real work surface. The relative geometry is
        # preserved so the reward stds still mean what they meant; see WORKSPACE_Z_SHIFT.
        self.scene.object.init_state.pos = (WORKSPACE_X, 0.1, 0.35 + WORKSPACE_Z_SHIFT)
        self.commands.object_pose.ranges.pos_x = (0.3, 0.7)
        self.commands.object_pose.ranges.pos_z = (0.55 + WORKSPACE_Z_SHIFT, 0.95 + WORKSPACE_Z_SHIFT)
        # The inherited bound box is written for the -x, table-at-0.235 workspace. Left alone, an
        # object at +0.55 is out of bounds on the first frame and every episode terminates
        # immediately.
        # The FLOOR of the z bound has to sit BELOW the work surface, or an object simply RESTING on
        # the table reads as out of bounds and the episode ends the moment it is put down. 50 mm of
        # margin under WORK_SURFACE_Z puts it at -0.046 -- under the surface, and still 630 mm above
        # the ground plane at GROUND_Z, so an object knocked off the table's edge still trips it.
        # x is left wider than the tabletop (which runs -0.35..1.05; see TABLE_TOP_X) on purpose: an
        # object pushed past the edge leaves through the z floor a frame later anyway.
        self.terminations.object_out_of_bound.params["in_bound_range"]["x"] = (-0.5, 1.5)
        self.terminations.object_out_of_bound.params["in_bound_range"]["z"] = (WORK_SURFACE_Z - 0.05, 2.0)
        # keep the debug camera looking at the workspace rather than away from it
        self.viewer.eye = (2.25, 0.0, 0.75)
        self.viewer.lookat = (WORKSPACE_X, 0.0, WORK_SURFACE_Z + 0.2)

        # -- GOAL + TASK-STATE VISUALIZATION. Runs AFTER the table swap and after the workspace
        # numbers above, because it measures the pad against them. See the function; the markers
        # are drawn only when ``commands.object_pose.debug_vis`` is True, which the _PLAY configs
        # set and the training configs leave False, so a headless training run pays nothing.
        self.commands.object_pose = _bind_task_state_visualization(self)

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
        # NOTE the prim path moved with the table: main spawns it at {ENV_REGEX_NS}/Table (capital
        # T), where dexsuite's cuboid was at .../table. It addresses the USD's default prim, which
        # carries the RigidBodyAPI and therefore the contact reporter that
        # ``_omnireset_table_cfg``'s ``activate_contact_sensors`` writes; the collider Cubes sit
        # under it and report through it.
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

        # -- terminations: drop the abnormal-velocity cut and its penalty. See the function.
        _drop_unreachable_abnormal_robot_cut(self)


@configclass
class Ur5eDeltoRelJointPosMixinCfg(Ur5eDeltoMixinCfg):
    """VARIANT 1: the DexSuite-style 26-DOF relative joint-position action space.

    The action group and every scale in it -- including the choice of 0.1 over the direct port's
    0.05/0.01, and the reason that choice is not a matter of taste -- live in
    :mod:`.dexlift_ur5e_delto_actions`. This class exists to bind it to the shared mixin, so that
    a second variant is a sibling of this class rather than an edit to the environments.
    """

    actions: Ur5eDeltoRelJointPosActionCfg = Ur5eDeltoRelJointPosActionCfg()


@configclass
class DexLiftUR5eDeltoRelJointPosReorientEnvCfg(Ur5eDeltoRelJointPosMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexLiftUR5eDeltoRelJointPosReorientEnvCfg_PLAY(
    Ur5eDeltoRelJointPosMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY
):
    pass


@configclass
class DexLiftUR5eDeltoRelJointPosLiftEnvCfg(Ur5eDeltoRelJointPosMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexLiftUR5eDeltoRelJointPosLiftEnvCfg_PLAY(Ur5eDeltoRelJointPosMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
