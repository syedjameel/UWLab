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
* The GOAL and RED/GREEN TASK-STATE markers are rebound to this scene and to the ADR SCHEDULER's
  success threshold -- the only thresholded success test in the task, and NOT the reward term that
  happens to be named ``success``. dexsuite owns the markers; what it cannot know is that its
  indicator geometry was cloned from a table this env then replaced. See
  :func:`_bind_task_state_visualization` and :data:`STATUS_PAD_TOP_Z`.
* The scene is a RIGID ASSEMBLY: the table is the robot's mount, so neither may be jittered without
  the other. Upstream's ``reset_table`` is nulled for that reason -- it has real +-50 mm ranges --
  while ``reset_root`` is KEPT, because every one of its ranges is zero and it is therefore not a
  jitter at all but the term that writes the robot's root pose to the pose this config declares.
  Nulling it too, as an earlier revision did, left the arm facing away from its own workspace; see
  :data:`BASE_LINK_AUTHORED_YAW` and the ``reset_root`` block in ``__post_init__``. See also
  :data:`MOUNT_PLATE_NOTE` for the one scene entity main spawns that this module deliberately does
  not.

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

import math
import os
import pathlib

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
from uwlab_assets.robots.ur5e_delto import (
    IMPLICIT_UR5E_DELTO,
    REFERENCE_HAND_ACTUATOR_KWARGS,
    UR5E_DELTO_HULLFIX3_USD_NAME,
)

from ..omnireset import mdp as omnireset_mdp
from . import mdp
from .mdp.c1_hand_pose_core import worst_case_composed_angle_rad as _c1_worst_case_composed_angle_rad

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

Identical reasoning to the UR10e task: the OmniReset cameras are parented to the robot ROOT PRIM
with offsets in the root frame, so a 180-degree root yaw here and not there would make identical
extrinsics resolve to different world poses. The scene is mirrored instead; see
:data:`WORKSPACE_X`. One base orientation across both task families, matching the single
orientation of the physical rig.

DO NOT SET THIS TRUE AS A WAY TO FIX THE BASE-FRAME BUG. Rotating the referencing prim by pi does
compose with :data:`BASE_LINK_AUTHORED_YAW` to put the root link at identity -- measured, it works
-- but it fixes the WORLD and breaks the CONFIG: ``init_state.rot`` is also
``default_root_state``, so the declared root orientation would then be a half turn away from the
real one, and the first event that resets the root to its default would undo the whole thing. The
root is PINNED instead; see the ``reset_root`` block in ``__post_init__``.
"""

BASE_LINK_AUTHORED_YAW = 3.141592653589793
"""``ur5e_delto.usd`` authors its ROOT BODY a half turn from the prim the config poses. MEASURED.

Probed on the spawned stage rather than reasoned about. With ``init_state`` identity:

* ``/World/envs/env_0/Robot`` -- the prim the spawner writes ``init_state.pos``/``rot`` onto --
  carries ``xformOp:translate (0,0,0)``, ``xformOp:orient (1,0,0,0)``: identity, as configured.
* ``/World/envs/env_0/Robot/base_link``, which ``Articulation`` reports as ``body_names[0]``, i.e.
  the ARTICULATION ROOT, carries its own ``xformOp:orient (-4.371e-08, 0, 0, 1)`` -- yaw -pi. So
  does ``shoulder_link``. That rotation is authored INSIDE the referenced USD and the referencing
  Xform's identity does nothing about it.

The consequence, and it is the whole reason this constant is written down: ``init_state.rot`` is
NOT the root link's pose on this asset. It is the pose of the prim the asset is referenced under,
and it is separately used as ``default_root_state``. Left alone the two disagree by
:data:`BASE_LINK_AUTHORED_YAW`, and since the pose COMMAND is sampled in the root frame
(``ObjectUniformPoseCommand`` composes with ``robot.data.root_quat_w``) while the object spawn, the
table and the viewer are all written in the ENV frame, the goal and the object end up in opposite
half-spaces. That is exactly what happened; see the ``reset_root`` block in ``__post_init__`` for
the measurement and the repair.

This is a property of the ASSET, shared with the calibrated UR5e it was grafted from, so it is not
something this task may fix by editing the USD: OmniReset's own configs are written against the
same asset and are correct because they reset the robot root every episode.
"""

WORKSPACE_X = 0.55
"""Workspace centre, mirroring the inherited dexsuite scene from -x to +x.

Stated in the ENV frame, which is the frame ``scene.object.init_state.pos``, ``scene.table``,
``viewer.lookat`` and ``terminations.object_out_of_bound`` are all resolved in.

It is ALSO the robot's root frame, and that is a fact this module has to buy rather than assume:
see :data:`BASE_LINK_AUTHORED_YAW` for what the asset authors, and the ``reset_root`` block in
``__post_init__`` for the term that pins the two together. The commanded goal is sampled in the
ROOT frame, so if that pin is ever dropped this constant and
``commands.object_pose.ranges.pos_x`` stop describing the same half of the table.

(The older note here -- that ``base`` and ``base_link_inertia`` are rotated 180 degrees from
``base_link`` per the ROS convention, so the value is "stated in base_link" -- named the right
subtlety and then drew the wrong conclusion: measured, it is ``base_link`` ITSELF that is the
rotated one relative to the prim the config poses.)
"""

WORK_SURFACE_Z = 0.0
"""Height of the work surface in the robot's own root frame: the STRUCTURAL tabletop plane.

Was 0.004 while this scene spawned ``custom_lab_table.usd``, whose two 4 mm mats lay ON the
structural top so that an object rested 4 mm above the flange plane (``table_dims.yaml``: "their top
= the WORK SURFACE at +0.004", confirmed against the generated USD's per-prim bbox --
``visuals/mat_black``/``mat_green`` at z [0.0000, 0.0040] over ``visuals/table_frame`` whose slab
face is z = 0.0000).

THE MATS WENT WITH THE MESH. The table is now a single collision box (see :func:`_lab_table_cfg`),
so the surface an object rests on IS the structural top and this constant is 0.0 rather than 0.004.
The 4 mm is not lost anywhere -- every height in this module derives from this constant through
:data:`WORKSPACE_Z_SHIFT`, so spawn, goal and out-of-bounds all descend by the same 4 mm together
and the geometry stays self-consistent.

That 4 mm is ALSO why the box top is at 0.0 rather than at 0.004: the robot's base occupies z >= 0
on this asset, and a slab whose top face were +0.004 would intersect the base by exactly the depth
the real rig's mat cutout exists to provide. The box has no cutout, so the surface is flush instead.

(An older revision had -0.013, inferred from the pat_vention-era plate and wrong by 17 mm.)
"""

GROUND_Z = -0.676
"""Floor height: ``table.body.z_bottom`` from ``table_dims.yaml``, matching main's ground plane.

680 mm from the floor to the work surface, of which the mats are the last 4 mm. The table's legs
reach exactly this z, so the rig stands on the ground plane instead of hovering over it -- which the
previous -0.868 (a pat_vention-era number) did not do.

EXACTLY coincident, deliberately noted: ``visuals/table_frame`` bottoms out at z = -0.6760 and
``collisions/cabinet`` at -0.6760, on a plane at -0.676. Two coplanar faces would normally be a
z-fighting risk, but this pair is safe on both counts -- the leg face points DOWN and is never in
frame, and both bodies are static (the plane is a collider, the table is ``kinematic_enabled``), so
there is no contact-resolution instability to be had either.
"""

TABLE_TOP_X = (-0.35, 1.05)
TABLE_TOP_Y_HALF = 0.35
"""Extent of the work surface in the robot base frame, from ``table_dims.yaml``.

Recorded here because this table was MEASURED AROUND A UR10e (1.30 m reach) and is being used under
a UR5e (0.85 m). See :data:`REACHABILITY_NOTE`; nothing in this module rescales it.

TWO CONSEQUENCES OF THAT CARRY-OVER, both measured, neither rescaled:

* REACH -- the front ~215 mm of the table is unreachable. :data:`REACHABILITY_NOTE`.
* THE BASE CUTOUT IS OVERSIZED. ``table_dims.yaml`` sizes the black mat's hole at diameter 0.200
  for a UR10e ("UR10e base is 0.190 -> 5 mm radial clearance"). The UR5e's ``base_link`` world bbox
  in ``ur5e_delto.usd`` is [-0.0755, -0.0755, 0.0000]..[+0.0755, +0.0755, +0.0991], so the actual
  radial clearance is 24.5 mm, not 5 mm: a 24.5 mm annulus of bare structural tabletop, 4 mm deep,
  rings the base. Purely cosmetic -- nothing is placed or commanded there -- but it is the second
  "measured for a UR10e, used under a UR5e" number in this scene and the reach note does not cover
  it. Fixing it means regenerating the USD with a UR5e hole diameter, not editing this module.
"""

REACHABILITY_NOTE = "measured palm reach 1.06 m (p99); spawn and goal are inside it, the table is not"
"""What this arm can actually touch, MEASURED on the articulation, not quoted from a datasheet.

256,000 samples drawn uniformly over the six arm joints' measured limits and pushed through the
spawned articulation's own forward kinematics (``scripts/tools/reach_measure2.py``). Radii are from
the BASE, which is the frame everything else in this module is written in:

* palm (``rl_dg_mount``) radius: median 0.600, p75 0.778, p99 1.064, max 1.129. Farthest fingertip
  1.361. The grafted DELTO adds roughly 0.25 m to the flange, so the datasheet's 0.850 m -- which is
  quoted to the FLANGE, from the shoulder -- is not the number that governs this task.
* the OBJECT SPAWN volume, x [0.35, 0.75] * y [-0.2, 0.2] * z [0.099, 0.499]: the palm reaches
  82.4% of it to within 2 cm and 100% to within 5 cm, worst nearest-sample distance 0.0394 m. The
  residual is sampling density (mean nearest-neighbour spacing ~1.4 cm), not unreachable volume.
* the GOAL BOX, x (0.3, 0.7) * y (-0.25, 0.25) * z (0.299, 0.699): measured live, the sampled goal
  sits 0.504 / 0.725 / 0.992 m (min/median/max) from the base -- the whole box inside the p99 palm
  radius, and the goal is a pose for the OBJECT, which the hand holds a further palm's length out.
  This RETIRES the older note's estimate that 1.4% of the goal box (the far-and-high corner at
  0.917 m) was out of reach: it was computed against the 0.850 m flange figure and the corner is
  reachable.
* the OmniReset object envelope proven on this table under a UR5e -- x (0.35, 0.60) * y (+-0.2) *
  z (0, 0.3) -- is 100% reachable at 3 cm, worst nearest-sample distance 0.0345 m. Our spawn box is
  that envelope extended to x = 0.75, and the extension is inside the measured reach set.

NOT RESCALED, and now for a measured reason rather than a cautious one. The spawn and the goal are
both inside the reach set as inherited, so narrowing either would change the task the dexsuite
reward stds and ADR tolerances were tuned against while fixing nothing. The defect that made the
hand miss the object was never the size of these boxes; it was which half-space the arm was in. See
BASE_LINK_AUTHORED_YAW.

THE FURNITURE STILL OVERHANGS THE ARM, unchanged and still cosmetic: the tabletop runs to x = 1.05
and its front corners sit 1.118 m out, past even the p99 palm radius. Nothing is spawned or
commanded there.

CAVEAT, stated because the sweep cannot see it: these are KINEMATIC reach sets. Collisions are
ignored (the sampler lets the palm pass through z = -0.79, i.e. under the work surface), and the
sweep says nothing about wrist orientation at the goal or about the DELTO's grasp offset. A
collision-free reach set needs a second pass that rejects samples with contact force or joint
residual.
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
``STATUS_PAD_TOP_Z`` -- 6 mm BELOW the tabletop plane, which is now the work surface itself.

The occlusion arithmetic survived the table becoming a box, which is the only reason this pad did
not need re-deriving: the slab spans z [-0.030, 0.000] over the full footprint
x [-0.350, 1.050] * y [-0.350, 0.350], and a 12 mm pad centred at -0.012 spans [-0.018, -0.006],
completely inside it wherever the two overlap. It is now ONE box that is both the visible surface
and the collider, so the older caveat here -- that the measurement had to come from
``visuals/table_frame`` and not from the invisible ``collisions/top_slab``, since a rendering claim
proved from a collider proves nothing -- no longer has two prims to distinguish between.

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

_ARM_START_DEG = (0.0, -90.0, 100.0, -10.0, 90.0, 270.0)
"""Arm home posture, in degrees, in UR joint order (pan, lift, elbow, wrist 1-3).

REPLACES the posture inherited from ``UR5E_DEFAULT_JOINT_POS``, which is the OmniReset UR5e home
pose for the Robotiq/linear grippers: wrists at (-90, -90, -90) leave the palm off vertical, while
DexSuite's task assumes a top-down approach and the reference (IsaacLabDexterous
``ur_tessolo.py:48-55``) holds all three wrists at exactly 0.0 so the palm looks at the table. The
constant was right where it was authored and wrong where this port consumed it.
"""

_HAND_START_RAD_BY_JOINT = (
    # One row per JOINT LEVEL, five fingers across. NOT the order of ``DELTO_HAND_JOINT_NAMES``,
    # which is finger-major -- the mapping is written out joint-major on purpose.
    #
    # THESE ARE THE REFERENCE'S OWN LITERALS, copied from the certified asset config
    # (IsaacLabDexterous ``ur_tessolo.py`` init_state.joint_pos at fork commit 6fee5b9f, which
    # predates every edit of mine in that tree), in RADIANS rather than degrees so no conversion sits
    # between the reference and what we spawn.
    #
    # WHY THIS REPLACED THE EARLIER HAND-AUTHORED POSE, which read
    #   _1 (30, 0, 0, 0, 0) / _2 (-50, 30, 30, 30, 0) / _3 (0, 30, 30, 30, 30) / _4 (0, 30, 30, 30, 30) deg:
    # that pose curls fingers 2-4 and the thumb TO THE SAME SIDE, so the hand can satisfy the contact
    # gate -- which needs thumb-group AND tip-group but never checks that they OPPOSE -- by laying
    # every fingertip on one face of the object. Contact was acquired ~4x faster than the reference
    # and the object was never lifted. The reference instead throws the two thumb-group members to
    # OPPOSITE sides (finger 1 to -1.5707, finger 5 to +1.0472 at the ``_2`` level) and splays 2 and 4
    # apart (-0.348, +0.261), which is a pre-grasp with force closure available from the first step.
    (0.0, -0.348, 0.0, 0.261, 0.5236),      # _1  splay / abduction
    (-1.5707, 0.0, 0.0, 0.0, 1.0472),       # _2  proximal flexion  <- the opposition
    (0.0, 1.0472, 1.0472, 1.0472, 0.5236),  # _3  middle flexion
    (0.0, 0.5236, 0.5236, 0.5236, 0.5236),  # _4  distal flexion
)

_ARM_START = tuple(math.radians(d) for d in _ARM_START_DEG)
"""The arm home posture this robot actually uses. One value, set here, no switch.

THE REFERENCE'S POSTURE IS NOT USED, and the reason is measured rather than aesthetic. The reference
(IsaacLabDexterous ``ur_tessolo.py`` at 6fee5b9f) holds shoulder_lift -1.712, elbow +1.712 and ALL
THREE WRISTS AT EXACTLY 0.0. On this UR5e that puts wrist_2 90 degrees and wrist_3 270 degrees away
from :data:`_ARM_START_DEG`, and the wrists are what set hand orientation -- so the palm no longer
looks at the table. An earlier environment inspection fixed exactly this: at the reference wrists the
hand does not spawn over the bench palm-down, which is the geometry DexSuite's task assumes.

The shoulder and elbow barely differ between the two (8.1 and 1.9 degrees). The whole disagreement is
the wrist, and this posture is the one that was chosen after looking at renders of both.
"""

DEXSUITE_START_JOINT_POS = {
    **{name: rad for name, rad in zip(ARM_JOINT_NAMES, _ARM_START)},
    **{
        f"rj_dg_{finger}_{joint}": row[finger - 1]
        for joint, row in enumerate(_HAND_START_RAD_BY_JOINT, start=1)
        for finger in range(1, 6)
    },
}
"""The full 26-joint home posture this task spawns in, keyed by joint NAME so the tuple ordering
above cannot silently mis-assign a value."""

if len(DEXSUITE_START_JOINT_POS) != 26:
    raise ValueError(
        f"{__name__}: DEXSUITE_START_JOINT_POS must cover 6 arm + 20 hand joints, got"
        f" {len(DEXSUITE_START_JOINT_POS)}. A missing key spawns that joint at the USD default"
        " rather than raising, so this is checked at import."
    )

START_JITTER_RAD = math.radians(10.0)
"""Spawn jitter applied to every one of the 26 joints, as an offset from
:data:`DEXSUITE_START_JOINT_POS`. Replaces dexsuite's +-0.50 rad (+-28.6 degrees) base range."""

SELF_COLLISIONS_ENABLED = True
"""Matches the certified reference, VERIFIED AGAINST THE ASSET rather than inferred.

Not a toggle. It was one (``DEXLIFT_NO_SELFCOLL=1``), and turning it off is how a 94.9 percent
policy was produced that drove finger 3 through finger 4 on 92.5 percent of steps -- median 6.18 mm
of non-adjacent interpenetration, scored a success by a predicate that cannot see geometry. Any
number measured with these off is a pipeline test, not a result, so the switch is gone.
Surveyed from the three assets:

    OURS  ur5e_delto.usd     23 sdf + 5 convexHull   selfCollisions True   28 bodies, min body 5 g
    REF   URTessoloAlik      25 hull + 1 cvxDecomp   selfCollisions True   26 bodies, min body 25 g
    VALIK ur10e_delto.usd    26 convexHull           selfCollisions FALSE  26 bodies

All twenty hand joint limits are IDENTICAL across ours and the reference (1717 degrees of total
travel, zero differences), so limits are not what lets the reference survive hulls. What differs is
that VALIK pairs hulls with self-collisions OFF, while the reference pairs hulls with a 26-body
decomposition authored for them. Our 28-body hand took hulls WITH self-collisions on and the
articulation exploded at reset -- joints to 1,062,181 degrees under zero actions.

``ur10e_delto_optimized_separate_tips_limited_jnts_self_collision.usd`` (10,925,903 bytes, dated
2026-01-05, i.e. before the fork at 6fee5b9f) authors ``physxArticulation:enabledSelfCollisions =
True`` on its ``/ur10`` articulation-root prim, and ``ur_tessolo.py``'s
``ArticulationRootPropertiesCfg`` never sets ``enabled_self_collisions``, so the field stays ``None``
and the USD's ``True`` stands.

This was ``False`` for ~2.2x throughput during pipeline testing, on the strength of a note claiming
the reference did not enable them either. That note was an INFERENCE from the throughput measurement
(if disabling is 2.2x faster, surely the reference paid no such cost) and it was wrong. With
self-collisions off the fingers pass through one another and through the palm, and nothing in the
reward or the success predicate looks at finger-finger interpenetration, so a policy can learn a
geometrically impossible grasp and still be scored a success.

The asset asks for self-collisions and means it: ``ur5e_delto.usd`` inherits them from the
reference's ``..._self_collision.usd`` variant, which authors
``physxArticulation:enabledSelfCollisions = True``, and ``ur5e_delto.py`` sets them on the
articulation for that reason. This constant does NOT change the asset; it overrides the spawn for
this task only, so the truth stays in one place and the override is greppable.

WHY IT IS OFF RIGHT NOW. Training is simulation-bound -- ~98.7% of wall clock in PhysX, measured
``Perf/collection time`` 21 s against ``Perf/learning_time`` 0.3 s -- and we are ~3x slower per
environment than the reference that reached 88%/92.87% with the same Tesollo hand. Two candidate
causes were tested and REFUTED: the modelled lab-table mesh (replaced with a box: no change,
6055 -> ~6100 fps) and the missing contact-patch budget. Self-collisions across a 20-joint
five-finger hand is the next candidate, and it is the one that cannot be settled by reading code.

WHAT TURNING THEM OFF ACTUALLY COSTS, so this is not mistaken for free speed: the fingers may pass
through one another and through the palm. A policy trained this way can learn a grasp that is
geometrically impossible on the real hand, and it will still be scored a success, because nothing
in the reward or the success predicate looks at finger-finger interpenetration.

THE PLAN, agreed with the user: run the setup fast while the pipeline is being debugged, then set
this back to ``True`` for fine-tuning and for anything reported as a result. A policy certified with
this ``False`` is a pipeline test, not a result.
"""

def ref_actuators(part: str) -> bool:
    """Is the reference actuator block used for this half of the robot?

    ``DEXLIFT_REF_ACTUATORS`` used to be one switch over FOUR separable things -- hand actuators, arm
    actuators, solver/rigid-body properties, and whether ``randomize_arm_sysid`` runs -- so the only
    two reachable configurations were "all of the reference's plant" and "all of ours". That is what
    made the ``alik_hull_ouract`` control uninterpretable: it was meant to ask whether our identified
    UR5e ARM dynamics can carry the task, and it silently also reverted the HAND to actuators that
    cannot perform the task at all. Our hand caps at ``velocity_limit_sim`` 3.0 rad/s while the policy
    commands 6 rad/s of closure at action scale 0.1 and 60 Hz, and at 0.06-0.17 N.m a distal joint
    delivers ~2.4 N to an object of up to 0.40 kg. A run cannot answer a question about the arm while
    its fingers are physically unable to close.

    ``DEXLIFT_REF_ARM_ACT`` and ``DEXLIFT_REF_HAND_ACT`` override the master per half. Unset means
    "follow the master", so every existing launch script keeps its exact meaning.
    """
    master = os.environ.get("DEXLIFT_REF_ACTUATORS") == "1"
    override = os.environ.get(f"DEXLIFT_REF_{part}_ACT")
    return master if override is None else override == "1"


TABLE_SLAB_THICKNESS = 0.030
"""Thickness of the collision slab that replaced the lab-table MESH. See :func:`_lab_table_cfg`."""

TABLE_ROOT_X = 0.5 * (TABLE_TOP_X[0] + TABLE_TOP_X[1])
TABLE_ROOT_Z = WORK_SURFACE_Z - 0.5 * TABLE_SLAB_THICKNESS
"""Where the table's ROOT sits, now that the table is a centred box rather than a modelled bench.

A ``CuboidCfg`` is centred on its prim, so a slab whose TOP face is the work surface must have its
root half a thickness below it. The USD it replaced had its root at the robot base flange and its
geometry authored around that, which is why this pair of constants did not previously exist.

Everything that is drawn RELATIVE to the table has to move with this, which is the whole reason it
is a named constant and not a literal: see :data:`STATUS_PAD_OFFSET`.
"""

STATUS_PAD_OFFSET = (
    0.5 * (TABLE_TOP_X[0] + TABLE_TOP_X[1]) - TABLE_ROOT_X,
    0.0,
    STATUS_PAD_TOP_Z - 0.5 * STATUS_PAD_THICKNESS - TABLE_ROOT_Z,
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

Two heights escape this rule and each says so where it is written: the object spawn's y, which is
re-centred for a narrower table, and the out-of-bounds z FLOOR, which is recomputed from the
surface rather than shifted.

WHAT THE 95 mm IS NOT. It is not "resting on the table" and it is not "clearing the table" -- it is
dexsuite's 0.350 - 0.255, carried over exactly. ``reset_object`` then adds 0..+0.4 m of drop and a
uniformly random SO(3) orientation, and the longest object's half-extent is 0.125 m, so at a
zero z-offset draw its lowest point is 26 mm inside the ``collisions/top_slab`` span of
[-0.030, 0.000] and PhysX depenetrates it with a pop on the first frame. Identical geometry
upstream, so this is inherited behaviour rather than something this rig introduced; recorded
because "spawned 95 mm above the surface" reads like a clearance guarantee and is not one.
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
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(TABLE_ROOT_X, 0.0, TABLE_ROOT_Z), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn=sim_utils.CuboidCfg(
            size=(TABLE_TOP_X[1] - TABLE_TOP_X[0], 2 * TABLE_TOP_Y_HALF, TABLE_SLAB_THICKNESS),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.36, 0.38)),
        ),
    )


MOUNT_PLATE_NOTE = "ur5_metal_support is NOT spawned here: coincident with the tabletop, and unread"
"""Why this scene has no ``ur5_metal_support`` entity, although main's UR5e scene does.

Main spawns ``custom_mount_plate.usd`` at ``{ENV_REGEX_NS}/UR5MetalSupport``, root z 0.004, and an
earlier revision of this module copied that over. It was REMOVED, for two independent reasons, both
probed with pxr rather than reasoned about:

* IT IS COINCIDENT WITH THE TABLE. The USD authors its disk at local z [-0.0080, -0.0040], so a root
  at 0.004 puts it at world z [-0.0040, 0.0000] -- its top face at EXACTLY z = 0.000. The visible
  mesh ``/custom_lab_table/visuals/table_frame`` is a solid 9-box frame whose top slab face is also
  exactly z = 0.000 and spans the full footprint (no cutout: its 72 points carry no ring
  coordinates). Two coplanar-coincident faces over the whole r = 0.098 disk is guaranteed
  z-fighting, in the one patch of the scene the debug camera is aimed at. Its collider Cube likewise
  sits fully inside ``/custom_lab_table/collisions/top_slab`` (z [-0.030, 0.000]).
* NOTHING IN THIS TASK READS IT. The plate's former docstring claimed its root z was "the
  object-placement datum the reset events are written against". That is true in
  ``omnireset/config/ur5e_robotiq_2f85/reset_states_cfg.py``, which passes it as
  ``offset_asset_cfg`` -- and false here: this task's ``reset_object`` is written against
  ``scene.object.init_state.pos``, and no term, sensor or reward in ``dexlift`` names the entity.

What is lost by dropping it: nothing visible. No physical plate exists on the real rig; the mat's
base cutout exposes the bare structural tabletop, which is exactly what ``table_frame`` renders
there. What is saved: one redundant kinematic rigid body and one embedded collider per environment.

:data:`WORK_SURFACE_Z` stands on its own measurement (``table_dims.yaml`` plus the generated USD's
world bbox) and no longer needs the plate's root z as a corroborating datum.
"""


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

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": [-START_JITTER_RAD, START_JITTER_RAD],
            "velocity_range": [0.0, 0.0],
        },
    )
    """Tighten the base class's spawn jitter from +-0.50 rad to +-10 degrees on every joint.

    dexsuite ships ``position_range`` [-0.50, 0.50] (``dexsuite_env_cfg.py:366-373``), i.e. +-28.6
    degrees on all 26 joints. That is wide enough to scatter the authored home posture -- which for
    this port is the whole point of DEXSUITE_START_JOINT_POS -- so the jitter is narrowed to a band
    around it rather than removed.
    """

    reset_robot_elbow_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="elbow_joint"),
            "position_range": [-START_JITTER_RAD, START_JITTER_RAD],
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

    # THE FRAME GUARD, and the reason it is a ``reset`` term rather than a ``startup`` one: at
    # startup the robot still stands where the USD authored it, half a turn from the pose this
    # config declares (see BASE_LINK_AUTHORED_YAW), and it is dexsuite's ``reset_root`` -- a reset
    # term -- that corrects it. Event terms of one mode run in config DECLARATION order and a
    # subclass's fields come after its base's, so this runs after ``reset_root`` by construction.
    # It checks once and then costs nothing; see the function.
    #
    # The construction-time half, ``_assert_root_pin_is_a_pin``, reads the CONFIG. This one reads
    # the spawned articulation, and so is the half that would catch a future asset revision that
    # moved the root body again.
    reset_finger_root_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=r"rj_dg_(1|2|3|4|5)_(1)"),
            # DELIBERATE DEVIATION FROM THE REFERENCE, on explicit instruction to randomize every
            # angle by +-10 degrees. The reference pins these five abduction joints at [0, 0]
            # because the BASE term's +-28.6 degrees of splay destroys the thumb-opposition geometry
            # that the contacts() gate scores. At +-10 degrees that objection is much weaker, but it
            # is not zero -- if thumb/tip contact regresses against the previous run, this is the
            # first term to put back to [0.0, 0.0].
            "position_range": [-START_JITTER_RAD, START_JITTER_RAD],
            "velocity_range": [0.0, 0.0],
        },
    )
    """Cancel the base class's joint jitter on the five ABDUCTION joints. Ported verbatim.

    Reference: ``dexsuite/config/ur10_tessolo/dexsuite_ur10_tessolo_env_cfg.py:84-92`` in
    IsaacLabDexterous -- the configuration that measurably reached 88% on this task. The joint
    names are identical on our hand, so this ports without translation.

    Not cosmetic. dexsuite's base ``reset_robot_joints`` applies +-0.50 rad to EVERY joint
    (``dexsuite_env_cfg.py:366-373``), which on ``rj_dg_*_1`` is +-28.6 degrees of splay across the
    five digits at every reset. That is precisely the thumb-opposition geometry the ``contacts()``
    gate scores, so the base term randomises away the pre-grasp the reference starts from. Later
    terms of the same mode win, so declaring this AFTER the base class's term is what makes it a
    cancellation; that ordering is by construction, since a subclass's fields follow its base's.

    This was missing from our port -- ``grep -rn reset_finger_root_joints`` over the whole package
    returned nothing -- and it is an omission, not a decision.
    """

    check_robot_root_pinned = EventTerm(
        func=mdp.check_root_matches_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "context": "dexlift UR5e+DELTO"},
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
      is the success condition rather than a hint at it. NOTE this only shows on a POSITION-ONLY
      task; see :meth:`.mdp.task_state_vis.TaskStateVisPoseCommand._debug_vis_callback`.

    The tolerances are read AFTER ``super().__post_init__()``, which is where dexsuite sets
    ``pos_tol = rewards.success.params["pos_std"] / 2`` and where the Lift subclass then drops
    ``rot_tol`` to None. Reading them earlier would capture the un-derived defaults.

    WHAT "ADR" MEANS HERE, because there is a same-named reward and it is NOT this. The colour
    tracks the only THRESHOLDED success test in the task, which is the ADR scheduler's
    (``dexsuite/mdp/curriculums.py``). The reward term literally named ``success`` is a different
    quantity: on this env it resolves to ``dexlift.mdp.rewards.success_reward``, a per-step tanh
    that is additionally multiplied by a fingertip-CONTACT gate. On the reorient task an object
    thrown to the goal and released is inside tolerance -- the pad goes green and ADR promotes --
    while that reward pays exactly 0. Binding to ADR is the deliberate choice; it is written down
    here so the pad is not misread as "the reward is paying".
    """
    adr_term = env_cfg.curriculum.adr if env_cfg.curriculum is not None else None
    # ``mdp.adr_param``, NOT ``params.get(key) or default``: a configured 0.0 is falsy, so ``or``
    # would swallow it into the fallback and the guard below could never fire on zero -- which is
    # precisely the failure this binding exists to prevent (ADR would never promote, because
    # ``pos_dist < 0.0`` is always False, while the pad went green at 0.05). Present-and-None is
    # meaningful too -- the Lift subclass sets ``rot_tol`` to None to drop the orientation gate --
    # and ``adr_param`` preserves both, falling back for an ABSENT key to the scheduler's own
    # signature default rather than to a literal typed here. That last part is the whole reason to
    # route through it: it is the same resolver the certification harness uses, so the colour, the
    # promotion threshold and the certified threshold cannot be three different numbers.
    #
    # The literals below are reached only when the config has NO ADR curriculum at all, i.e. when
    # there is no scheduler to agree with; they are then upstream's own marker defaults. Without the
    # 0.5 a curriculum-less REORIENT config would silently colour on position alone, i.e. be
    # strictly more permissive than the upstream marker it replaced.
    pos_tol = mdp.adr_param(adr_term, "pos_tol") if adr_term is not None else 0.05
    rot_tol = mdp.adr_param(adr_term, "rot_tol") if adr_term is not None else 0.5
    if not isinstance(pos_tol, (int, float)) or pos_tol <= 0.0:
        raise ValueError(
            f"curriculum.adr.params['pos_tol'] must be a positive distance; got {pos_tol!r}. The"
            " task-state marker colours itself with the same threshold the ADR scheduler promotes"
            " on, so an unusable value here would be an unusable success test there too."
        )
    if rot_tol is not None and (not isinstance(rot_tol, (int, float)) or rot_tol <= 0.0):
        raise ValueError(
            f"curriculum.adr.params['rot_tol'] must be a positive angle or None; got {rot_tol!r}."
            " None deliberately drops the orientation gate (the Lift task does exactly that); a"
            " zero or negative value instead makes the gate unsatisfiable, and ADR -- which tests"
            " `rot_dist < rot_tol` -- would never promote while the pad still turned green."
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

    THE PAIRED REWARD GOES WITH IT, AND MUST GO FROM HERE. ``rewards.early_termination`` is
    ``is_terminated_term(term_keys="abnormal_robot")``, i.e. a reward that names this termination by
    string. Removing the termination without removing the reward does not degrade gracefully: it
    fails construction outright, because ``is_terminated_term.__init__`` resolves the key through
    ``TerminationManager.find_terms`` and ``isaaclab/utils/string.py:267`` raises on a regex that
    matches nothing, and ``ManagerBasedRLEnv`` builds the termination manager BEFORE the reward
    manager. Measured, not argued -- ``DexLift-UR5eDelto-RelJointPos-Reorient-v0``::

        ValueError: Not all regular expressions are matched! ...
            abnormal_robot: []
            Available strings: ['time_out', 'object_out_of_bound']

    The nulling previously lived at the END of :func:`_attach_success_rate_metric`, behind
    that function's two early returns. So it ran only for configs whose scored tolerances happened
    to equal the certified recorder's 0.05 m / 0.5 rad -- true for Lift, false for Reorient, and
    false for ANY tightened tolerance. A coupling between a termination and the reward that names it
    was gated on a question about metric naming, which is why Reorient had never once constructed.
    """
    env_cfg.terminations.abnormal_robot = None
    env_cfg.rewards.early_termination = None


def _apply_pose_tilt_stage(env_cfg) -> None:
    """Optionally shrink BOTH sampled orientations, so the DEMANDED ROTATION is bounded.

    Off unless ``DEXLIFT_POSE_TILT`` is set to a value in radians. It is a task-definition change and
    it is printed, because any result obtained under it describes a narrower task than the one the
    class name implies.

    WHY IT EXISTS, measured rather than supposed. On the Reorient task the goal samples roll and pitch
    over +-pi (``dexsuite_env_cfg.py:111-113``) while ``reset_object`` samples the object's roll, pitch
    AND yaw over +-pi (``:268-270``). The demanded rotation is therefore random-to-random, whose mean
    magnitude is ~2.2 rad. Certifying a Reorient policy at epoch 1700 over 128 episodes at full
    difficulty measured the per-episode CLOSEST approach as::

        min orientation error   p50 1.824 rad (104 deg)   best of 128  0.307 rad (17.6 deg)
        min position error      p50 0.026 m               best of 128  0.002 m

    against a scored gate of 0.25 rad. Not one episode in 128 ever entered it, and the terminal error
    (p50 2.374) is WORSE than the mid-episode minimum, i.e. the object tumbles past and drifts on
    rather than being held. Position is close; orientation is not in the neighbourhood.

    WHY NARROWING THE GOAL ALONE WOULD NOT HAVE WORKED, which is the part that is easy to get wrong:
    the required rotation is the angle BETWEEN the object's reset orientation and the goal. With the
    object still reset over the full range, pinning the goal to identity leaves the demanded rotation
    at the magnitude of a uniformly random orientation -- still ~2.2 rad on average. Both ends have to
    move, so this helper narrows ``reset_object``'s roll/pitch/yaw and the goal's roll/pitch together,
    which bounds the demand at roughly twice the tilt.

    Yaw is included on the OBJECT because ``reset_object`` randomizes it, and excluded on the GOAL
    because dexsuite pins goal yaw to 0 already.
    """
    raw = os.environ.get("DEXLIFT_POSE_TILT")
    if raw is None:
        return
    tilt = float(raw)
    if not 0.0 <= tilt <= math.pi:
        raise ValueError(f"DEXLIFT_POSE_TILT must be in [0, pi] radians; got {tilt}")
    for axis in ("roll", "pitch", "yaw"):
        env_cfg.events.reset_object.params["pose_range"][axis] = [-tilt, tilt]
    env_cfg.commands.object_pose.ranges.roll = (-tilt, tilt)
    env_cfg.commands.object_pose.ranges.pitch = (-tilt, tilt)
    # -- S_t SPAWN RECENTRING (RESET_SPEC_V2.md sec 1, C3/S_t; measured gap V2_POSE_FINDINGS.md F37:
    # spawn tip never comes within 65 mm of the table). MEASUREMENT-ONLY TOGGLE, OFF BY DEFAULT --
    # enabling it for training is a separate decision from running an experiment under it. The
    # ``pose_range`` written just above centres the object's SPAWN orientation on the leg's
    # as-shipped NEAR-HORIZONTAL baseline (pitch centred on 0 rad). S_t needs the leg spawned
    # TIP-DOWN instead. This codebase already makes that exact move on the GOAL side -- see
    # ``c3_transport_core.transport_goal_ranges``, pitch centred on ``-pi/2`` (Ry(-90) is the leg's
    # assembled/tip-down root orientation, same convention this function's own ``pose_range``
    # dict uses). This flag makes the same recentring on the SPAWN side ONLY:
    # ``commands.object_pose.ranges.pitch`` set two lines above (the GOAL band) is left untouched.
    spawn_tipdown_raw = os.environ.get("DEXRESET_ST_SPAWN_TIPDOWN")
    if spawn_tipdown_raw is not None and spawn_tipdown_raw not in ("0", "1"):
        raise ValueError(f"DEXRESET_ST_SPAWN_TIPDOWN must be '0' or '1'; got {spawn_tipdown_raw!r}")
    spawn_tipdown = spawn_tipdown_raw == "1"
    if spawn_tipdown:
        env_cfg.events.reset_object.params["pose_range"]["pitch"] = [-math.pi / 2 - tilt, -math.pi / 2 + tilt]
    # THE DROP HAS TO BE CLAMPED TOO, OR THE ORIENTATION STAGING IS MOSTLY DECORATIVE. ``reset_object``
    # samples z over [0.0, 0.4], i.e. the object is released from up to 40 cm and TUMBLES ON LANDING,
    # so its resting orientation is whatever the bounce produced and not the roll/pitch/yaw just
    # sampled. Measured: with tilt staged to 0.3 rad but the drop left alone, the episode-mean
    # orientation error settled at 1.746 rad -- far above the ~0.4 rad the staging implies, though
    # still below the 2.16 of the unstaged control, so the staging was working and being partly undone.
    #
    # Separable on purpose: DEXLIFT_DROP_Z overrides the clamp, so "did the tilt help" and "did the
    # drop clamp help" can be asked as two experiments rather than one confounded change.
    drop = float(os.environ.get("DEXLIFT_DROP_Z", "0.05"))
    if drop < 0.0:
        raise ValueError(f"DEXLIFT_DROP_Z must be >= 0; got {drop}")
    env_cfg.events.reset_object.params["pose_range"]["z"] = [0.0, drop]
    if spawn_tipdown:
        _st_pitch_lo, _st_pitch_hi = -math.pi / 2 - tilt, -math.pi / 2 + tilt
        _st_banner = (
            f" DEXRESET_ST_SPAWN_TIPDOWN=1: object reset pitch RECENTRED to tip-down, band"
            f" [{_st_pitch_lo:.4f}, {_st_pitch_hi:.4f}] rad ({math.degrees(_st_pitch_lo):.1f} to"
            f" {math.degrees(_st_pitch_hi):.1f} deg) -- SPAWN ONLY, goal pitch band above is unchanged."
            " MEASUREMENT TOGGLE, not a default."
        )
    else:
        _st_banner = (
            " DEXRESET_ST_SPAWN_TIPDOWN not set: object reset pitch remains centred on the"
            " near-horizontal baseline (0 rad)."
        )
    print(
        f"[dexlift] POSE_TILT staged: object reset roll/pitch/yaw and goal roll/pitch limited to"
        f" +-{tilt:.4f} rad (+-{math.degrees(tilt):.1f} deg); reset drop height clamped to"
        f" [0, {drop:.3f}] m so the sampled orientation survives to the first policy step."
        " THIS IS A NARROWER TASK than the unstaged +-pi configuration and any number measured"
        " under it must say so." + _st_banner,
        flush=True,
    )


def _apply_c1_hand_pose_stage(env_cfg) -> None:
    """Optionally add a Cartesian palm-pose IK reset for the C1 rung (RESET_SPEC_V2.md sec 1).

    Off unless ``DEXRESET_C1_HAND=1``. Like :func:`_apply_pose_tilt_stage` this is a
    task-definition change and it is printed. It goes through this opt-in env-var staging function
    rather than a config field a hydra override could appear to control but not actually reach --
    RESET_SPEC_V2.md sec 1a trap 3 / this module's own F6 finding: a hydra override of
    ``commands.object_pose.ranges.pitch`` is silently overwritten by :func:`_apply_pose_tilt_stage`
    above, and nothing added here gets an exemption from that lesson.

    WHAT IT ADDS. A new ``events.reset_c1_hand_pose`` term, set as an attribute here rather than
    declared on :class:`Ur5eDeltoEventCfg`, so the UNSET path is byte-for-byte the existing
    behaviour -- nothing about ``Ur5eDeltoEventCfg`` changes when this function returns early.
    Event terms of one mode run in the order their attribute was set (class-declared fields first,
    in declaration order, then anything a subclass or ``__post_init__`` adds after), and LATER
    terms of the same mode WIN (see ``reset_finger_root_joints``'s own docstring for the same fact
    used the same way) -- so appending this term here, after every class-declared ``reset``-mode
    term already ran, is what lets it OVERRIDE the arm/wrist half of
    ``reset_robot_joints``/``reset_robot_elbow_joint``'s +-10 deg jitter with an IK-solved Cartesian
    palm pose while leaving those SAME terms' write to the FINGER joints alone (this term's
    ``robot_ik_cfg`` only names ``ARM_JOINT_NAMES`` + ``PALM_BODY``, never a hand joint). Fingers
    therefore stay on the existing DexSuite-style jitter exactly as before -- RESET_SPEC_V2.md's C1
    asks for "fingers randomized in the DexSuite style", which is what those terms already are;
    nothing new is added for them here.

    ANCHOR (XY). RESET_SPEC_V2.md wants the hand centred "above the insertion hole (goal/bore XY),
    not above the leg". The dexlift scene has NO physical bore
    (V2_POSE_FINDINGS.md F2: "the dexlift scene HAS NO FIXTURE"). The closest stand-in available is
    ``commands.object_pose``'s own goal-XY distribution, so this reads its RANGE MIDPOINT
    (``ranges.pos_x``/``pos_y``, robot ROOT frame -- read here, after ``_apply_goal_vertical_mixture``,
    so every stage that could touch those ranges has already run).

    THIS IS A STATIC ANCHOR, NOT THE LIVE PER-EPISODE GOAL DRAW, and that is deliberate, not an
    oversight: ``ManagerBasedRLEnv._reset_idx`` runs ``self.event_manager.apply(mode="reset", ...)``
    BEFORE ``self.command_manager.reset(env_ids)`` (which is what resamples the goal) -- measured
    directly against IsaacLabDexterous's ``manager_based_rl_env.py``. A ``reset``-mode event
    reading ``env.command_manager.get_command("object_pose")`` would therefore read the PREVIOUS
    episode's goal, not the one about to be sampled for this one: the exact class of silent
    staleness RESET_SPEC_V2.md sec 1a trap 3 warns about (an override that looks live but is not),
    one layer deeper than the hydra case it names. Anchoring on the RANGE's midpoint instead
    sidesteps that ordering hazard entirely, at the cost of not tracking this episode's specific
    goal draw -- acceptable because the goal itself is drawn from this same range every episode, so
    the hand ends up centred in the region the goal will occupy, just not glued to the exact point.

    HEIGHT (Z). ``WORK_SURFACE_Z`` = 0 IS the table surface, stated in the ROBOT'S OWN ROOT FRAME
    (see that constant's own docstring), and ``commands.object_pose.ranges`` is ALSO in the root
    frame (``ObjectUniformPoseCommand``'s own docstring: "Targets are defined in the robot's base
    frame"). So an anchor placed at root-frame z = 0 lands at world z = the robot root's own world
    height once ``events.reset_root`` has pinned it -- which it always has by the time any
    ``reset``-mode term declared after it runs. ``DEXRESET_C1_HAND_Z``'s two numbers are therefore
    literal metres ABOVE the real tabletop, with NO leg-tip correction: the 106.203 mm offset in
    RESET_SPEC_V2.md sec 1a trap 1 is ``SquareTableLeg200mmDecomp``'s own assembled-offset and has
    nothing to do with a HAND pose -- stated explicitly so the next reader does not go looking for
    it here.

    ORIENTATION. "Palm pointing downwards" is built in :mod:`.mdp.c1_hand_pose` from the SAME
    metadata key and the SAME formula the ``orient_down`` success gate already uses
    (``omnireset/mdp/terminations.py::check_reset_state_success``, 60-degree cone about world -Z;
    ``gripper_approach_direction`` = [0.2582, 0.4717, 0.8431], duplicated onto
    ``Ur5eDelto/metadata.yaml``). ``DEXRESET_C1_HAND_TILT`` is a per-axis (roll, pitch, yaw)
    half-angle, sampled independently per axis and composed as an EXTRINSIC world-frame rotation on
    top of that nominal -- see the event class for the exact composition.

    R1 / IK EXEMPTION. RESET_SPEC_V2.md sec 2 R1 forbids IK for the HELD reset states (C2/C3/C4)
    because those must be reached by a policy and held against gravity. C1 is free-space sampling
    with nothing held, so it is exempt, and an IK-placed hand here does not violate R1.
    """
    raw = os.environ.get("DEXRESET_C1_HAND")
    if raw != "1":
        return

    z_raw = os.environ.get("DEXRESET_C1_HAND_Z", "0.10,0.20")
    try:
        z_lo, z_hi = (float(part) for part in z_raw.split(","))
    except ValueError as exc:
        raise ValueError(
            f"DEXRESET_C1_HAND_Z must be two comma-separated metres, e.g. '0.10,0.20'; got {z_raw!r}"
        ) from exc
    if not z_lo < z_hi:
        raise ValueError(f"DEXRESET_C1_HAND_Z must satisfy lo < hi; got {z_lo} >= {z_hi}")

    xy_raw = os.environ.get("DEXRESET_C1_HAND_XY", "0.15")
    try:
        xy_half_width = float(xy_raw)
    except ValueError as exc:
        raise ValueError(
            f"DEXRESET_C1_HAND_XY must be a single metres value, e.g. '0.15'; got {xy_raw!r}"
        ) from exc
    if not xy_half_width > 0.0:
        raise ValueError(f"DEXRESET_C1_HAND_XY must be > 0; got {xy_half_width}")

    tilt_raw = os.environ.get("DEXRESET_C1_HAND_TILT", "0.7854")
    try:
        tilt = float(tilt_raw)
    except ValueError as exc:
        raise ValueError(
            f"DEXRESET_C1_HAND_TILT must be a single radians value, e.g. '0.7854'; got {tilt_raw!r}"
        ) from exc
    if not 0.0 <= tilt <= math.pi:
        raise ValueError(f"DEXRESET_C1_HAND_TILT must be in [0, pi] radians; got {tilt}")

    # POST-SOLVE GATE thresholds (critic review of the first version of this function, commit
    # 1654e2c): the original event wrote whatever the damped IK converged to, unchecked, and that
    # measured at 17-19% of resets landing with the ACHIEVED palm height outside [z_lo, z_hi] (H100,
    # n=512 x 2 runs). A follow-up, team-lead-requested controlled comparison (run BEFORE tuning any
    # threshold, on instruction) separated three possible causes directly rather than by inference:
    # DEXLIFT_REF_RESET's wider pre-IK starting joints (REJECTED -- dropping it barely moved the
    # residual distribution), an insufficient ik_iterations budget (LARGELY CONFIRMED for the bulk
    # of the tail -- 10/20/40-iteration sweep measured median residual 26mm/1.5mm/0.005mm), and
    # inherent unreachability of specific sampled targets (CONFIRMED for a small <1% population
    # whose residual does not improve AT ALL between 20 and 40 iterations). See
    # ``c1_hand_pose_core.py``'s ``DEFAULT_IK_ITERATIONS_RAW``/``DEFAULT_MAX_POS_ERR_MM_RAW``
    # comment for the full numbers and where every default below comes from, and
    # ``c1_hand_pose.py``'s module docstring, "POST-SOLVE GATE", for the original defect. Parsed
    # here in the SAME inline style as the four fields above rather than delegating to
    # ``c1_hand_pose_core.parse_c1_hand_pose_env`` -- that function exists and is unit-tested
    # (``test_c1_hand_pose_stage.py``), but this file already duplicates its own copy of the
    # z/xy/tilt parsing rather than importing it (no other env_cfg module in this package imports a
    # ``*_core`` module directly), and matching the file's EXISTING convention here is lower-risk
    # than introducing a new one mid-fix.
    ik_iterations_raw = os.environ.get("DEXRESET_C1_HAND_IK_ITERATIONS", "20")
    try:
        ik_iterations = int(ik_iterations_raw)
    except ValueError as exc:
        raise ValueError(
            f"DEXRESET_C1_HAND_IK_ITERATIONS must be a positive integer, e.g. '20'; got"
            f" {ik_iterations_raw!r}"
        ) from exc
    if ik_iterations <= 0:
        raise ValueError(f"DEXRESET_C1_HAND_IK_ITERATIONS must be > 0; got {ik_iterations_raw!r}")

    max_pos_err_mm_raw = os.environ.get("DEXRESET_C1_HAND_MAX_POS_ERR_MM", "10.0")
    try:
        max_pos_err_mm = float(max_pos_err_mm_raw)
    except ValueError as exc:
        raise ValueError(
            "DEXRESET_C1_HAND_MAX_POS_ERR_MM must be a single millimetres value, e.g. '10.0'; got"
            f" {max_pos_err_mm_raw!r}"
        ) from exc
    if not max_pos_err_mm > 0.0:
        raise ValueError(f"DEXRESET_C1_HAND_MAX_POS_ERR_MM must be > 0; got {max_pos_err_mm}")

    max_ori_err_deg_raw = os.environ.get("DEXRESET_C1_HAND_MAX_ORI_ERR_DEG", "5.0")
    try:
        max_ori_err_deg = float(max_ori_err_deg_raw)
    except ValueError as exc:
        raise ValueError(
            "DEXRESET_C1_HAND_MAX_ORI_ERR_DEG must be a single degrees value, e.g. '5.0'; got"
            f" {max_ori_err_deg_raw!r}"
        ) from exc
    if not 0.0 < max_ori_err_deg <= 180.0:
        raise ValueError(f"DEXRESET_C1_HAND_MAX_ORI_ERR_DEG must be in (0, 180] degrees; got {max_ori_err_deg}")

    min_joint_margin_deg_raw = os.environ.get("DEXRESET_C1_HAND_MIN_JOINT_MARGIN_DEG", "1.0")
    try:
        min_joint_margin_deg = float(min_joint_margin_deg_raw)
    except ValueError as exc:
        raise ValueError(
            "DEXRESET_C1_HAND_MIN_JOINT_MARGIN_DEG must be a single degrees value, e.g. '1.0'; got"
            f" {min_joint_margin_deg_raw!r}"
        ) from exc
    if not min_joint_margin_deg >= 0.0:
        raise ValueError(f"DEXRESET_C1_HAND_MIN_JOINT_MARGIN_DEG must be >= 0; got {min_joint_margin_deg}")

    max_retries_raw = os.environ.get("DEXRESET_C1_HAND_MAX_RETRIES", "5")
    try:
        max_retries = int(max_retries_raw)
    except ValueError as exc:
        raise ValueError(
            "DEXRESET_C1_HAND_MAX_RETRIES must be a non-negative integer, e.g. '5'; got"
            f" {max_retries_raw!r}"
        ) from exc
    if max_retries < 0:
        raise ValueError(f"DEXRESET_C1_HAND_MAX_RETRIES must be >= 0; got {max_retries_raw!r}")

    ranges = env_cfg.commands.object_pose.ranges
    pos_x, pos_y = tuple(ranges.pos_x), tuple(ranges.pos_y)
    anchor_x = 0.5 * (pos_x[0] + pos_x[1])
    anchor_y = 0.5 * (pos_y[0] + pos_y[1])

    env_cfg.events.reset_c1_hand_pose = EventTerm(
        func=mdp.reset_end_effector_c1_hand_pose,
        mode="reset",
        params={
            "robot_ik_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=PALM_BODY),
            "anchor_xy_root": (anchor_x, anchor_y),
            "z_range": (z_lo, z_hi),
            "xy_half_width": xy_half_width,
            "tilt": tilt,
            "ik_iterations": ik_iterations,
            "max_pos_err_m": max_pos_err_mm / 1000.0,
            "max_ori_err_rad": math.radians(max_ori_err_deg),
            "min_joint_margin_rad": math.radians(min_joint_margin_deg),
            "max_retries": max_retries,
        },
    )
    # Worst-case composed rotation at this tilt (critic review item 4: "state the composed-angle
    # fact in the banner ... per-axis +-tilt is what the user chose and it stays; do not
    # substitute a cone"). This is NOT a gate threshold -- see
    # ``c1_hand_pose_core.worst_case_composed_angle_rad``'s own docstring for why a cone bound
    # would silently redefine what the spec asked for -- it is printed so nobody later reads
    # "+-tilt deg per axis" as a bound on the achieved angle to world -Z.
    worst_case_deg = math.degrees(_c1_worst_case_composed_angle_rad(tilt))
    print(
        "[dexlift] C1_HAND staged: hand reset height"
        f" [{z_lo:.4f}, {z_hi:.4f}] m above WORK_SURFACE_Z (no leg-tip correction -- this is a hand"
        f" pose, see RESET_SPEC_V2.md sec 1a trap 1), XY +-{xy_half_width:.4f} m about the goal-XY"
        f" anchor ({anchor_x:.4f}, {anchor_y:.4f}) m [robot root frame, = commands.object_pose"
        f".ranges pos_x/pos_y midpoint -- pos_x={pos_x}, pos_y={pos_y}], palm pointing down"
        f" (gripper_approach_direction -> world -Z) +-{tilt:.4f} rad (+-{math.degrees(tilt):.2f} deg)"
        " per axis (roll, pitch, yaw independently sampled, composed as an extrinsic world-frame"
        f" rotation on top of the palm-down nominal -- WORST-CASE COMPOSED ANGLE {worst_case_deg:.2f}"
        " deg, NOT a cone bound, see worst_case_composed_angle_rad), fingers UNCHANGED (existing"
        " DexSuite-style reset_finger_root_joints / reset_robot_joints jitter still applies)."
        f" Arm/wrist IK solved against body 'rl_dg_mount' (PALM_BODY), joints ARM_JOINT_NAMES,"
        f" {ik_iterations} damped iterations at step 0.25 (see DEFAULT_IK_ITERATIONS_RAW's own"
        " comment: a team-lead-requested controlled comparison found the ORIGINAL unconditional"
        " 10-iteration budget, not DEXLIFT_REF_RESET's wider starting joints, was the main driver"
        " of the pre-fix residual tail), joint-limit-wrapped, gated post-solve: reject-and-resample"
        f" (max_retries={max_retries}) unless commanded-vs-achieved position error <="
        f" {max_pos_err_mm:.1f} mm AND orientation error <= {max_ori_err_deg:.1f} deg AND arm"
        f" joint-limit margin >= {min_joint_margin_deg:.2f} deg AND the achieved pose itself lands"
        " inside the height/XY band above -- an env that exhausts max_retries keeps its"
        " best-of-attempts state and is reported, never silently accepted ungated. THIS OVERRIDES"
        " the arm/wrist half of reset_robot_joints/reset_robot_elbow_joint's jitter for every reset"
        " (declared last, later reset-mode terms win) -- +-10 deg by default, but +-0.5 rad"
        " (reset_robot_joints) / +-0.2 rad (reset_robot_elbow_joint), plus the additionally-created"
        " reset_robot_wrist_joint at +-0.5 rad on wrist_3_joint, under DEXLIFT_REF_RESET=1 (the"
        " setting every production run uses -- see its own banner line for the numbers actually in"
        " effect this run); finger joints from those same terms are untouched.",
        flush=True,
    )


def _apply_goal_vertical_mixture(env_cfg) -> None:
    """Optionally MIX a vertical (tip-down) near-table goal into the commanded goal distribution.

    Off unless ``DEXLIFT_GOAL_VERTICAL_PROB`` is set. Like :func:`_apply_pose_tilt_stage` it is a
    task-definition change and it is printed.

    WHY (epic UWLab-nnlv). DexReset's C4 -> C3 gap needs two intermediate reset rungs in which the
    leg is held VERTICAL near the bore. Measured 2026-08-26/27, the shipped ep_3600 policy never
    produces such a state: from a partial-assembly spawn it flips the leg horizontal and carries it
    away, and the whole 1800-state C3 bank has a MINIMUM tilt of 43 deg from vertical. It can grasp
    but cannot be told where to hold, because ``_apply_pose_tilt_stage`` only ever showed it goals
    within +-0.3 rad of upright-as-spawned. Opening that clamp at inference alone reaches 14.5-20.8
    deg at a 1-3 percent yield; this trains for it instead.

    MIXTURE, NOT REPLACEMENT, and that is the whole design. A 100-percent finetune on a changed
    goal distribution destroyed this exact policy before (55 percent of the skill gone in 50 epochs,
    certified 0.0000 at 30 mm by 1550). ``vertical_prob`` therefore names a FRACTION and the
    remainder keeps the staged goal, so the original task stays rewarded throughout.

    IT DOES NOT GO THROUGH ``cfg.ranges``. A hydra override of ``commands.object_pose.ranges.pitch``
    is silently overwritten by the tilt staging above -- that defect already cost one invalid
    experiment. The vertical band is a separate field the staging cannot reach.

    Defaults, in the ROBOT ROOT frame with the tabletop at ``WORK_SURFACE_Z`` = 0:

    * pitch centred on ``-pi/2``, which is the leg's ASSEMBLED orientation: its ``assembled_offset``
      composes to a root quaternion of Ry(-90), whose local tip axis maps to world (0, 0, -1).
    * ``roll`` and ``pitch`` half-width ``DEXLIFT_GOAL_VERTICAL_TILT`` (default 0.35 rad = 20 deg),
      i.e. the band the untrained policy was already grazing.
    * ``yaw`` pinned to 0, matching dexsuite's own goal-yaw convention. Yaw about the insertion axis
      is discarded by the success metric anyway (it sums ``|e_x| + |e_y|`` and drops ``e_z``).
    * ``pos_z`` = ``DEXLIFT_GOAL_VERTICAL_Z``, default 0.13-0.27 m. The leg's root sits 106.2 mm
      ABOVE its tip when tip-down (``assembled_offset`` pos ``[-0.106203, 0, 0]``), so that band
      places the TIP 24-164 mm above the tabletop. A band chosen from the TIP's height directly
      would command the tip through the table.

    WHAT THIS BAND DOES AND DOES NOT COVER, because the two rungs are produced by different means
    and it would be easy to read this as training both. The dexlift scene HAS NO FIXTURE -- the leg
    rests on a table -- so no commanded goal here can place a tip inside a bore. What this trains is
    the one thing both rungs need and the parent policy lacks: HOLD THE LEG TIP-DOWN AND STEADY AT A
    COMMANDED LOW HEIGHT. Its tip range 24-164 mm above the surface is S2' (tip 20-120 mm above the
    mouth) directly. S1 (tip 0-10 mm INSIDE the mouth) is NOT reachable as a goal in this scene and
    is not meant to be: it is generated later from a partially-assembled SPAWN in the OmniReset
    scene, where the leg starts in the bore and the policy's job is to hold it there. Widening this
    band downward would only command the tip into the tabletop.
    """
    raw = os.environ.get("DEXLIFT_GOAL_VERTICAL_PROB")
    if raw is None:
        return
    prob = float(raw)
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"DEXLIFT_GOAL_VERTICAL_PROB must be in [0, 1]; got {prob}")
    tilt = float(os.environ.get("DEXLIFT_GOAL_VERTICAL_TILT", "0.35"))
    if not 0.0 <= tilt <= math.pi / 2:
        raise ValueError(f"DEXLIFT_GOAL_VERTICAL_TILT must be in [0, pi/2] radians; got {tilt}")
    z_raw = os.environ.get("DEXLIFT_GOAL_VERTICAL_Z", "0.13,0.27")
    try:
        z_lo, z_hi = (float(part) for part in z_raw.split(","))
    except ValueError as exc:
        raise ValueError(
            f"DEXLIFT_GOAL_VERTICAL_Z must be two comma-separated metres, e.g. '0.13,0.27'; got"
            f" {z_raw!r}"
        ) from exc
    if not z_lo < z_hi:
        raise ValueError(f"DEXLIFT_GOAL_VERTICAL_Z must satisfy lo < hi; got {z_lo} >= {z_hi}")

    command_cfg = env_cfg.commands.object_pose
    if not isinstance(command_cfg, mdp.TaskStateVisPoseCommandCfg):
        raise TypeError(
            "DEXLIFT_GOAL_VERTICAL_PROB requires commands.object_pose to already be the"
            f" task-state-vis command; got {type(command_cfg).__name__}. This helper must run"
            " AFTER _bind_task_state_visualization, whose upgrade it extends."
        )
    # DEXLIFT_GOAL_VERTICAL_ANCHOR_XY: 'inherit' (default, original behaviour) or 'object'.
    # 'object' pins the vertical goal's x/y to the object's own live position -- the difference
    # between teaching "hold it vertical" and "hold it vertical straight up from where it lies".
    # The first is what the original run taught, and it generated ZERO reset states because the
    # rungs accept only a tip within 5-20 mm of the bore axis (bead UWLab-nnlv.4).
    anchor_xy = os.environ.get("DEXLIFT_GOAL_VERTICAL_ANCHOR_XY", "inherit")
    if anchor_xy not in ("inherit", "object"):
        raise ValueError(
            f"DEXLIFT_GOAL_VERTICAL_ANCHOR_XY must be 'inherit' or 'object'; got {anchor_xy!r}"
        )
    env_cfg.commands.object_pose = mdp.upgrade_pose_command_to_mixed_goal(
        command_cfg,
        vertical_prob=prob,
        roll=(-tilt, tilt),
        pitch=(-math.pi / 2 - tilt, -math.pi / 2 + tilt),
        yaw=(0.0, 0.0),
        pos_z=(z_lo, z_hi),
        anchor_xy=anchor_xy,
    )
    print(
        f"[dexlift] GOAL VERTICAL ANCHOR_XY: {anchor_xy}"
        + (
            " -- vertical goals are placed directly ABOVE THE OBJECT's own live position"
            if anchor_xy == "object"
            else " -- vertical goals keep the staged draw's x/y (original behaviour; this is the"
            " setting that made reset-state generation impossible)"
        ),
        flush=True,
    )
    print(
        f"[dexlift] GOAL VERTICAL MIXTURE staged: {prob:.3f} of goals drawn tip-down"
        f" (pitch -pi/2 +- {tilt:.4f} rad = {math.degrees(tilt):.1f} deg, roll +- same, yaw 0)"
        f" at root height [{z_lo:.3f}, {z_hi:.3f}] m, i.e. leg TIP {z_lo - 0.106203:.3f} to"
        f" {z_hi - 0.106203:.3f} m above the work surface. The other {1.0 - prob:.3f} keep the"
        " staged goal, which is what prevents the catastrophic forgetting a 100-percent change"
        " caused before. THIS IS A DIFFERENT TASK from the unmixed configuration and any number"
        " measured under it must say so.",
        flush=True,
    )


def _attach_success_rate_metric(env_cfg) -> None:
    """Log an episode success rate, under the metric name the certified rl_games runs used.

    THE PROBLEM THIS SOLVES IS NOT A MISSING NUMBER, IT IS TWO INCOMPARABLE DASHBOARDS. The runs
    that reached 88% on this task and 92.87% on the table leg were trained with rl_games, which
    logged a real success rate through dexsuite's ``DexsuiteSuccessRecorder`` as
    ``Episode/Success_Rate/Episode/success_rate``. This port trains with rsl_rl, whose configs
    carry no recorder, so its runs logged NO success rate at all -- the nearest series,
    ``Curriculum/adr``, is a difficulty fraction that reads 0.0 both for "never succeeds" and for
    "curriculum stalled". Overlaying the two lineages in wandb therefore meant comparing a success
    rate against a curriculum counter, which is not a comparison.

    :class:`~.mdp.success.EpisodeSuccessRateLogger` publishes the same quantity, computed with the
    same sticky-OR-over-the-episode protocol, under the same string. It is the ONLY thing added
    here; nothing about the task, the rewards or the terminations changes, and the term cannot end
    an episode.

    THE NAME IS CONDITIONAL; THE METRIC IS NOT. The certified name means a specific test -- the
    reference recorder's ``pos_dist < 0.05`` with its orientation gate dropped by ``position_only``.
    On the Lift task this port's own scored tolerances reduce to exactly that (see
    :func:`~.mdp.success.matches_certified_recorder_predicate`, which checks it rather than assuming
    it). On the Reorient task they do not -- the scored rotation tolerance is ``rot_std / 2`` = 0.25
    against the recorder's 0.5 -- and a line drawn under the certified name would put two different
    tests on one axis.

    THIS USED TO BE RESOLVED BY WITHHOLDING THE METRIC, and that was the wrong half to give up.
    Reorient then logged no success rate at all, leaving ``Curriculum/adr`` as the only signal --
    and that reads 0.0 both for "never succeeds" and for "curriculum stalled", a conflation that has
    already cost this campaign real time. The term is now always attached and
    :func:`~.mdp.success.episode_success_rate_keys` decides the NAME from the tolerances resolved off
    the live curriculum term, so a non-certified test gets a self-describing series
    (``..._p50mm_r250mrad``) instead of no series, and the certified name still means exactly one
    thing. That also removes a trap in the other direction: the withholding decision was taken here,
    at ``__post_init__``, while the logger resolves its tolerances lazily at runtime, so a hydra
    override of ``pos_tol`` used to publish the new number under the old name.

    The tolerances are read the same way everything else in this module reads them -- out of the
    LIVE ``curriculum.adr`` term via ``mdp.adr_param``, after ``super().__post_init__()`` has
    derived them -- so this decision cannot drift from the one the scheduler, the certification
    harness and the red/green table pad all make.
    """
    adr_term = env_cfg.curriculum.adr if env_cfg.curriculum is not None else None
    if adr_term is None:
        # No ADR term means no thresholded success test to report; ``resolve_adr_success_spec``
        # would raise at the first step rather than invent one.
        env_cfg.terminations.success_rate_log = None
        return
    env_cfg.terminations.success_rate_log = mdp.success_rate_log_term_cfg()


def _attach_gate_proxy_metric(env_cfg) -> None:
    """Optionally log the TRAINING-TIME GATE PROXY (``V2_REPOSE_RECIPE.md`` sec 4, bead dr-tlx.2).

    Off unless ``DEXRESET_GATE_PROXY=1``. Like :func:`_apply_pose_tilt_stage` and
    :func:`_apply_c1_hand_pose_stage` it announces itself either way -- see "WHY IT PRINTS EVEN WHEN
    OFF" below, which is not the usual reason.

    === WHAT IT MEASURES, AND WHY THE CAMPAIGN NEEDS IT ===

    ``RESET_SPEC_V2.md`` R2 wants ``accepted / attempted > 0.50`` per rung. Measured, that is a
    requirement on the REPOSE POLICY and not on the pose sampler: in the v1 S1 run at least 90.8
    percent of generation attempts died in the held-state gate chain before the rung's geometry was
    ever evaluated, ``co_move`` alone rejecting 46.8 percent (``V2_POSE_FINDINGS.md`` F28, confirmed
    on a second rung at 53.2 percent by F30). Discovering whether a checkpoint clears R2 by running
    a generation pass costs roughly 25,000 attempts per 600-state chunk (F26/R6), which is far too
    expensive to do while a ramp is deciding whether to advance.

    So this publishes the three gates of that chain that need NO probe -- ``settled``,
    ``opposed_contact``, ``co_move`` -- per episode and per mixture branch. It is a STRICT UPPER
    BOUND on gate-chain pass rate, never a yield (R7), for the reason spelled out in
    ``mdp/gate_proxy_core.py``: the other three gates need the probe action bias the generator
    injects on top of the policy, and injecting that during training would perturb the thing being
    trained. What it buys is a cheap assumption-free NECESSARY condition -- accepted states are a
    subset of passive-three-passing states, so a run below 0.50 here provably cannot meet R2.

    === WHY IT IS OPT-IN ===

    The term costs a fingertip-sensor read and a palm/object velocity read every step, on every
    task in this family, and adds a dozen log series. Off by default keeps the UNSET path
    byte-for-byte the existing behaviour -- the same discipline :func:`_apply_c1_hand_pose_stage`
    follows, and for the same reason.

    === WHY IT PRINTS EVEN WHEN OFF ===

    NOT decoration. ``V2_POSE_FINDINGS.md`` F40/F41 catalogue four flags in this repository that
    were read by nothing, one of which reached a published model card; F42 records the in-tree
    documentation of one of them going stale. The metric this term publishes is what the v2 ramp's
    advance gate and its A4 no-progress abort are evaluated on, so "was it actually on?" has to be
    answerable from the run's own log rather than from the launcher that was supposed to export the
    variable -- ``RESET_SPEC_V2.md`` sec 1a trap 3 ("read the staged value back out of the run log
    rather than trusting the command line"), applied to the instrument itself.

    === PER-BRANCH SPLIT ===

    Both this term and the success-rate logger are handed ``mdp.EPISODE_KIND_NAMES`` and
    ``mdp.EPISODE_KIND_BUFFER_ATTR``, so each publishes ``.../classic``, ``.../low_goal``,
    ``.../partial_assembly`` and ``.../transport`` series. Load-bearing in both directions: the
    recipe's advance target is stated on TRANSPORT-kind episodes (the branch that actually carries
    the leg to tip-down, which is what S1 generation depends on) and its collapse tripwire on
    CLASSIC-kind episodes (the original task whose loss IS the collapse). An aggregate over four
    branches muffles both signals exactly while the ramp is raising the tip-down fraction.

    The mapping is passed in from here rather than looked up inside either term: this module already
    imports ``mdp`` wholesale, while ``mdp/success.py`` does not import the mixture and
    ``mdp/gate_proxy_core.py`` cannot (isaaclab at module scope). Two independently-written
    int-to-name tables is F27's defect class; there is one, in ``episode_mixture.py``, next to the
    integers it names.

    Both splits are inert when the mixture is not wired: the terms read the kind buffer with
    ``getattr(env, ..., None)`` and never CREATE it, so a run without the mixture reports
    un-split metrics rather than labelling every episode ``classic``.
    """
    raw = os.environ.get("DEXRESET_GATE_PROXY")
    if raw is not None and raw not in ("0", "1"):
        raise ValueError(f"DEXRESET_GATE_PROXY must be '0' or '1'; got {raw!r}")
    if raw != "1":
        print(
            "[dexreset] GATE PROXY not staged (DEXRESET_GATE_PROXY unset or 0): no GateProxy/*"
            " series will be published, and the v2 ramp's advance gate and A4 abort"
            " (V2_REPOSE_RECIPE.md sec 2.2, 3.3) have nothing to read. Expected for a run that is"
            " not the v2 repose retrain.",
            flush=True,
        )
        return

    env_cfg.terminations.gate_proxy_log = mdp.gate_proxy_log_term_cfg(
        kind_names=mdp.EPISODE_KIND_NAMES,
    )
    # The success rate gets the SAME split, from the same mapping, by re-attaching the term that
    # ``_attach_success_rate_metric`` already created. Re-attached rather than edited in place
    # because that function may have set the slot to ``None`` (no ADR term -> no thresholded
    # success test to report), and a split of a metric that is not being computed is nothing.
    if getattr(env_cfg.terminations, "success_rate_log", None) is not None:
        env_cfg.terminations.success_rate_log = mdp.success_rate_log_term_cfg(
            kind_names=mdp.EPISODE_KIND_NAMES,
            kind_buffer_attr=mdp.EPISODE_KIND_BUFFER_ATTR,
        )
    print(
        "[dexreset] GATE PROXY staged: publishing GateProxy/{settled,opposed_contact,co_move,"
        "passive_three}_{atend,ever}_frac plus priority-ordered reach and first-fail counts, split"
        f" by episode kind {sorted(mdp.EPISODE_KIND_NAMES.values())}. Thresholds are"
        " held_check_core's own (settled > 60 steps, relative speed < 0.05 m/s, |obj vz| < 0.1"
        " m/s), shared with the generator's held predicate via passive_gates -- not restated."
        " THESE ARE AN UPPER BOUND ON GATE-CHAIN PASS RATE, NOT A YIELD (RESET_SPEC_V2.md R7):"
        " the three PROBE gates cannot be measured without injecting the probe bias into training."
        + (
            " Success rate is split by the same mapping."
            if getattr(env_cfg.terminations, "success_rate_log", None) is not None
            else " Success rate is NOT split: no success_rate_log term exists on this config."
        ),
        flush=True,
    )


def _assert_root_pin_is_a_pin(env_cfg) -> None:
    """Fail construction unless ``events.reset_root`` is present AND writes a fixed pose.

    Two opposite regressions, one check, because the term has to be both things at once:

    * DELETED (or ``None``). Then nothing writes the robot's root pose and the arm stands at
      whatever the USD authored -- :data:`BASE_LINK_AUTHORED_YAW`, a half turn -- while the object
      spawn, the table and the out-of-bounds box are all written in the env frame. That is the
      defect this check exists for; the ``reset_root`` block in ``__post_init__`` has the numbers.
    * WIDENED. dexsuite ships every range at [0, 0] and this scene needs them to stay there: the
      robot base is bolted to the table (see the ``reset_table`` block for the three ways a moving
      base breaks this rig), so a nonzero range here is the same defect as jittering the table.

    Construction-time and cheap. The runtime half is ``check_robot_root_pinned`` in
    :class:`Ur5eDeltoEventCfg`, which measures the pose that actually resulted.
    """
    term = getattr(env_cfg.events, "reset_root", None)
    if term is None:
        raise ValueError(
            "events.reset_root is missing. It is NOT a randomization -- all of its pose ranges are"
            " zero -- it is the term that writes the robot's root pose to its configured"
            f" init_state. This asset authors its root body {BASE_LINK_AUTHORED_YAW:.6f} rad from"
            " the prim the spawner poses (see BASE_LINK_AUTHORED_YAW), so without this term the"
            " arm faces away from the table while the object spawns on it."
        )
    asset_name = term.params["asset_cfg"].name
    if asset_name != "robot":
        raise ValueError(f"events.reset_root must pin scene['robot']; it names scene['{asset_name}'].")
    for range_kind in ("pose_range", "velocity_range"):
        for axis, bounds in term.params.get(range_kind, {}).items():
            if float(bounds[0]) != 0.0 or float(bounds[1]) != 0.0:
                raise ValueError(
                    f"events.reset_root.{range_kind}['{axis}'] = {tuple(bounds)} is not zero. On"
                    " this rig the robot base is bolted to the table, whose own reset term is"
                    " nulled for that reason; a base that moves on its own breaks the mount, every"
                    " height measured from the work surface, and the status pad's fixed offset."
                )


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
        # THE SUCCESS PREDICATE MOVES OUT OF THE SCHEDULER, and only the predicate: this subclass
        # keeps upstream's promotion arithmetic verbatim and reads ``move_up`` from
        # ``mdp.within_success_tolerance``, which is also what the certification harness scores
        # episodes with. Without it the thresholded success test exists twice -- once here, once in
        # whatever evaluates a checkpoint -- with nothing tying the two together. Only ``func`` is
        # replaced; the inherited ``params`` (and the ``pos_tol``/``rot_tol`` dexsuite writes into
        # them in ``__post_init__``) are untouched. See ``mdp/success.py`` for the guard that fails
        # loudly if the vendored upstream method drifts from the copy.
        self.adr.func = mdp.SharedPredicateDifficultyScheduler


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
        # the fingertip contact sensors need contact reporters on the robot's bodies, and
        # self-collisions are toggled HERE rather than on the asset -- see SELF_COLLISIONS_ENABLED.
        robot_cfg.spawn = robot_cfg.spawn.replace(
            activate_contact_sensors=True,
            articulation_props=robot_cfg.spawn.articulation_props.replace(
                enabled_self_collisions=SELF_COLLISIONS_ENABLED
            ),
        )

        # -- THE HAND COLLIDER SET. Not a toggle: this is the only one that trains.
        #
        # ur5e_delto_hullfix3.usd is 25 convexHull finger bodies + convexDecomposition on the three
        # palm-shell bodies + FilteredPairsAPI on {rl_dg_base, rl_dg_palm} x the 25 finger bodies.
        #
        # WHY IT HAS TO BE THIS AND NOT A PLAIN convexHull. Our graft splits the DELTO palm assembly
        # into three bodies welded by fixed joints, where the reference USD has ONE:
        #
        #     wrist_3_link --fixed--> rl_dg_mount --fixed--> rl_dg_base --fixed--> rl_dg_palm
        #     rl_dg_mount  --revolute--> rl_dg_{1..5}_1
        #
        # PhysX auto-filters phalanx-vs-mount because they are jointed, but phalanx-vs-palm and
        # phalanx-vs-base are two joints away and actively collided -- pairs the reference cannot
        # even express. The palm shell is CONCAVE and the finger roots sit in recesses, so a convex
        # hull fills them and swallows the roots: scratchpad/probe_pairs.py measured 7 interpenetrating
        # non-adjacent pairs at the authored spawn posture (1_1/palm 38.5%, 1_1/base 28.4%, 4_1/palm
        # 23.6%, ...), and with self-collisions on the depenetration impulses blow the articulation
        # apart at reset -- 8045 deg of hand deviation against a +-28.6 deg reset jitter band.
        # convexDecomposition restores the concavity; the filtered pairs are reference-faithful rather
        # than a cheat, because base and palm are rigidly welded to the phalanx's own joint parent,
        # making those pairs kinematically identical to mount-vs-N_1 which PhysX already filters.
        # rl_dg_mount vs finger and finger vs finger both stay LIVE, exactly as the reference has them.
        #
        # The cost, measured curl-only (levels _2.._4, spread held at zero, scratchpad/closure_test.py):
        #   convexHull, self-collisions OFF  83.98 deg      this asset, self-collisions ON  73.64 deg
        # i.e. 17.8 percent of reachable curl, of which 31 percent is those two extra shells and 69
        # percent is fingers genuinely blocking fingers -- the constraint the whole exercise is about.
        #
        # Two earlier variants were deleted rather than kept as options: a plain convexHull hand
        # (untrainable -- the explosion above) and hullfix2, which additionally filtered rl_dg_mount
        # and existed only to attribute that 17.8 percent between the two mechanisms. Both have served
        # their purpose; leaving them selectable only widens the space of silently-wrong plants.
        _HULLFIX_USD = str(pathlib.Path(robot_cfg.spawn.usd_path).with_name(UR5E_DELTO_HULLFIX3_USD_NAME))
        if not pathlib.Path(_HULLFIX_USD).is_file():
            raise FileNotFoundError(
                f"{_HULLFIX_USD} does not exist. Generate it with scratchpad/reauthor_hullfix.py,"
                " which verifies the saved layer by re-reading it."
            )
        robot_cfg.spawn = robot_cfg.spawn.replace(usd_path=_HULLFIX_USD)
        print(
            "[dexlift] hand colliders: 25 convexHull + 3 convexDecomposition (mount/base/palm)"
            f" + 10 filtered welded pairs ({_HULLFIX_USD})",
            flush=True,
        )

        # -- REFERENCE ACTUATOR AND SOLVER PARITY, opt-in via ``DEXLIFT_REF_ACTUATORS=1``.
        #
        # An exhaustive 322-parameter diff against IsaacLabDexterous found the actuators to be the
        # largest un-audited divergence in the port, and the one whose predicted signature matches
        # what five training runs actually produced:
        #
        #   hand effort_limit_sim    0.06-0.17 N.m  vs  30.0     (176x-500x)
        #   hand velocity_limit_sim  3.0 rad/s      vs  10000.0
        #   hand stiffness (distal)  0.17-0.30      vs  3.0       (10x-18x on the joints that close)
        #   hand damping (distal)    0.0025         vs  0.1       (40x)
        #   arm  effort_limit_sim    150 / 28 N.m   vs  870.0     (5.8x / 31x)
        #   arm  velocity_limit_sim  1.5708 rad/s   vs  2.094 / 3.1416 (USD-authored on both sides)
        #
        # THE MECHANISM, which survives the fact that gravity is zero for most of the curriculum: at
        # action scale 0.1 and 60 Hz the policy commands 6 rad/s of finger closure against a 3.0 rad/s
        # ceiling, so the fingers cannot track a closing command at all. They still cross the 0.2 N
        # contact gate trivially -- which is why good_finger_contact saturates near the reference's own
        # value -- while a distal joint capped at 0.06 N.m delivers ~2.4 N to an object of up to 0.40
        # kg. Squeezing hard enough to move an object is not the same problem as holding one up, so
        # zero gravity does not excuse a 500x torque deficit.
        #
        # WHY THIS IS A SWITCH: these values are the identified DELTO hand, tuned for an OFFLINE grasp
        # recorder that holds a fixed posture on 30-52 g parts. Reference parity and sysid fidelity are
        # mutually exclusive here, and the honest framing is that this run measures the reference's
        # plant, not our robot's. Do NOT edit ur10e_delto.py -- that actuator object is shared with
        # IMPLICIT/EXPLICIT_UR10E_DELTO, DELTO_HAND and the grasp recorder.
        if ref_actuators("HAND") or ref_actuators("ARM"):
            # Solver and rigid-body properties are plant-wide, not per-half, so they follow either.
            robot_cfg.spawn = robot_cfg.spawn.replace(
                articulation_props=robot_cfg.spawn.articulation_props.replace(
                    enabled_self_collisions=SELF_COLLISIONS_ENABLED,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=1,
                ),
                rigid_props=robot_cfg.spawn.rigid_props.replace(
                    retain_accelerations=True,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_depenetration_velocity=1000.0,
                ),
            )
        if ref_actuators("HAND"):
            robot_cfg.actuators = dict(robot_cfg.actuators)
            robot_cfg.actuators["hand"] = robot_cfg.actuators["hand"].replace(**REFERENCE_HAND_ACTUATOR_KWARGS)
            print("[dexlift] reference HAND actuators (30 N.m / 10000 rad/s / kp 3.0 / kd 0.1)", flush=True)
        else:
            print("[dexlift] identified DELTO hand actuators (0.06-0.17 N.m / 3.0 rad/s)", flush=True)
        # The DexSuite home posture, overridden HERE rather than on the asset: IMPLICIT_UR5E_DELTO's
        # default comes from UR5E_DEFAULT_JOINT_POS, which is the OmniReset UR5e+Robotiq/linear
        # posture, and moving it on the asset would reach every OmniReset environment.
        robot_cfg.init_state = robot_cfg.init_state.replace(joint_pos=dict(DEXSUITE_START_JOINT_POS))
        # position targets need PD gains; see ARM_STIFFNESS
        robot_cfg.actuators = dict(robot_cfg.actuators)
        robot_cfg.actuators["arm"] = robot_cfg.actuators["arm"].replace(stiffness=ARM_STIFFNESS, damping=ARM_DAMPING)
        if ref_actuators("ARM"):
            # The reference's flat 870 N.m and the USD-authored arm velocity limits. 870 is NOT a
            # hardware rating on either side -- it is 15.5x the UR10e's own 56 N.m wrist rating,
            # inherited from IsaacLab's UR10_CFG. Copying it buys parity by making both sides equally
            # unphysical, and a policy trained under it will not transfer to a real UR5e without a
            # torque-limited fine-tune. That is the deliberate price of the comparison.
            robot_cfg.actuators["arm"] = robot_cfg.actuators["arm"].replace(
                effort_limit_sim=870.0,
                velocity_limit_sim={
                    "shoulder_pan_joint": 2.0940,
                    "shoulder_lift_joint": 2.0940,
                    "elbow_joint": 3.1416,
                    "wrist_1_joint": 3.1416,
                    "wrist_2_joint": 3.1416,
                    "wrist_3_joint": 3.1416,
                },
            )
        if FLIP_BASE_YAW:  # never; see the constant's docstring
            robot_cfg.init_state = robot_cfg.init_state.replace(rot=(0.0, 0.0, 0.0, 1.0))

        # ONE UNCONDITIONAL LINE NAMING EVERY AXIS OF THE PLANT, so a launcher can assert the run it
        # actually got rather than the run it asked for. Colliders and self-collisions are no longer
        # switchable -- only the two actuator halves are -- but the line still prints all four,
        # because a launcher assertion that stops naming a value stops noticing when it changes.
        print(
            "[dexlift] PLANT"
            " colliders=hullfix3"
            f" self_collisions={'ON' if SELF_COLLISIONS_ENABLED else 'OFF'}"
            f" hand_act={'reference' if ref_actuators('HAND') else 'identified'}"
            f" arm_act={'reference' if ref_actuators('ARM') else 'identified'}"
            f" usd={pathlib.Path(robot_cfg.spawn.usd_path).name}",
            flush=True,
        )
        self.scene.robot = robot_cfg

        # -- scene: OmniReset's UR5e rig, as ``syedjameel/main`` defines it. The robot base stays at
        # the origin and the table spawns there too, because the table's asset frame IS the base
        # frame. The dexsuite cuboid table is REPLACED (not added to) -- two work surfaces would
        # intersect -- and ``plane``/``sky_light`` are edited rather than duplicated, since both
        # already own their prim paths (/World/GroundPlane, /World/skyLight), which are also the
        # paths main uses.
        self.scene.table = _omnireset_table_cfg()
        # NO mount plate; see MOUNT_PLATE_NOTE for what main spawns here and why this does not.
        # The ground plane's z is the table's own leg bottom, so the rig stands on it rather than
        # hovering; the two faces are exactly coincident, which is fine because both bodies are
        # static and the leg face points down (see GROUND_Z).
        self.scene.plane.init_state.pos = (0.0, 0.0, GROUND_Z)
        # main's dome light is intensity 1000 over the kloofendal_43d_clear_puresky_4k HDR. Only the
        # intensity is written here: the inherited dexsuite ``sky_light`` already names that exact
        # texture under ISAAC_NUCLEUS_DIR and differs from main in the intensity alone (750).
        self.scene.sky_light.spawn.intensity = 1000.0

        # -- workspace: mirror the inherited dexsuite layout from -x to +x so the base stays
        # unrotated, and drop every height onto the real work surface. The relative geometry is
        # preserved so the reward stds still mean what they meant; see WORKSPACE_Z_SHIFT.
        # y is RE-CENTRED, not inherited, and that is a deliberate deviation from "shift, don't
        # re-author". dexsuite spawns at y = +0.1 on a table 1.5 m wide (y_half 0.75), so its
        # +-0.2 m ``reset_object`` jitter left 450 mm of margin at the near edge. This table is
        # y_half 0.35 (TABLE_TOP_Y_HALF, measured), so the same bias leaves 50 mm -- and the largest
        # object in the set, ``CapsuleCfg(radius=0.025, height=0.2)``, has a 0.125 m half-extent and
        # is dropped from up to +0.4 m with a uniformly random orientation. At y = +0.1 roughly a
        # fifth of resets put part of it past the +y edge. At y = 0 the worst case is
        # 0.2 + 0.125 = 0.325 < 0.35 and the object is always over the table. y = 0 is also what the
        # rest of the task already assumes: the goal box ``ranges.pos_y`` is (-0.25, +0.25), i.e.
        # symmetric about the robot's own plane of symmetry, and so is the table.
        self.scene.object.init_state.pos = (WORKSPACE_X, 0.0, 0.35 + WORKSPACE_Z_SHIFT)
        # -- SPAWN THE OBJECT IN THE FAR HALF OF THE JITTER BOX ONLY.
        #
        # A DELIBERATE DEVIATION FROM THE REFERENCE, requested after watching rollouts: the inherited
        # ``reset_object`` jitters x over [-0.2, +0.2] about WORKSPACE_X, and the near half puts the
        # object close enough to the base that the arm has to fold back on itself to reach it. +x is
        # away from the robot here (the base sits at x = 0 and TABLE_TOP_X runs to +1.05), so the far
        # half is [0.0, +0.2].
        #
        # WHY IT MATTERS MORE NOW THAN IT DID BEFORE: with SELF_COLLISIONS_ENABLED back to True a
        # folded-back approach is genuinely BLOCKED rather than silently interpenetrating, so the
        # near-half spawns become unreachable rather than merely ugly. y is left untouched -- it is
        # already symmetric about the robot's own plane and re-centred to keep objects over the table.
        #
        # REVISIT THIS if the run succeeds: it narrows the task relative to the certified 92.87 percent
        # configuration, so a result obtained with it is not strictly comparable to that number until
        # the full range is restored and re-measured.
        self.events.reset_object.params["pose_range"]["x"] = [0.0, 0.2]
        # THE TWO LINES BELOW ARE IN A DIFFERENT FRAME FROM THE ONE ABOVE, and that is worth one
        # sentence because it is what the base-frame defect was made of. ``init_state.pos`` is
        # resolved in the ENV frame; ``ObjectUniformPoseCommand`` samples in the ROBOT ROOT frame
        # and composes with ``robot.data.root_quat_w`` (dexsuite/mdp/commands/pose_commands.py).
        # The two agree only while the root really is at the identity pose this config declares,
        # which is what ``events.reset_root`` buys -- see BASE_LINK_AUTHORED_YAW.
        self.commands.object_pose.ranges.pos_x = (0.3, 0.7)
        self.commands.object_pose.ranges.pos_z = (0.55 + WORKSPACE_Z_SHIFT, 0.95 + WORKSPACE_Z_SHIFT)
        _apply_pose_tilt_stage(self)
        # The inherited bound box is written for the -x, table-at-0.235 workspace. Left alone, an
        # object at +0.55 is out of bounds on the first frame and every episode terminates
        # immediately.
        # RECOMPUTED against both sampled volumes, in the env frame, with the root pinned:
        #   object spawn  = init_state.pos +- reset_object's [-0.2, 0.2] x/y and [0.0, 0.4] z
        #                 -> x [0.35, 0.75]  y [-0.20, 0.20]  z [0.099, 0.499]
        #   goal          = ranges above, in a root frame that is now the env frame
        #                 -> x [0.30, 0.70]  y [-0.25, 0.25]  z [0.299, 0.699]
        # Both fit inside (x, y, z) = (-0.5, 1.5), (-2.0, 2.0), (-0.046, 2.0) with margins of
        # 0.80 / 0.75 in x, 1.75 in y, and 0.145 below the spawn floor / 1.30 above the goal
        # ceiling in z, so the inherited x and y and the recomputed z floor all stand as they are.
        # Both measured boxes match those predictions (reach_measure5): object x [0.352, 0.739],
        # goal x [0.307, 0.697]. Worth stating what the SAME arithmetic said before the root was
        # pinned: the goal box then ran to x = -0.699, i.e. half of every goal sampled sat OUTSIDE
        # this termination's x floor of -0.5 -- the object would have been killed on arrival.
        # The FLOOR of the z bound has to sit BELOW the work surface, or an object simply RESTING on
        # the table reads as out of bounds and the episode ends the moment it is put down. 50 mm of
        # margin under WORK_SURFACE_Z puts it at -0.046 -- under the surface, and still 630 mm above
        # the ground plane at GROUND_Z, so an object knocked off the table's edge still trips it.
        # THIS ONE HEIGHT IS RECOMPUTED, NOT SHIFTED, and that is the module's only deviation from
        # the preserve-relative-geometry rule stated at WORKSPACE_Z_SHIFT. Shifting the inherited
        # 0.0 would have given -0.251, which also works; recomputing tightens the fall distance
        # before termination from 255 mm to 50 mm, so an object knocked off the table ends its
        # episode sooner. Deliberate, and named here so it is not read as an oversight.
        # x is left wider than the tabletop (which runs -0.35..1.05; see TABLE_TOP_X) on purpose: an
        # object pushed past the edge leaves through the z floor a frame later anyway.
        self.terminations.object_out_of_bound.params["in_bound_range"]["x"] = (-0.5, 1.5)
        self.terminations.object_out_of_bound.params["in_bound_range"]["z"] = (
            WORK_SURFACE_Z + WORKSPACE_Z_SHIFT,
            2.0,
        )
        # REFERENCE PARITY, replacing the 50 mm floor the comment above argues for. dexsuite's own
        # floor is z = 0.0 in ITS frame (dexsuite_env_cfg.py:464-470); shifted into ours by
        # WORKSPACE_Z_SHIFT that is -0.251, i.e. a 251 mm fall rather than 50 mm.
        #
        # The tighter floor was reasoned, not measured, and it is the ONE divergence with a measured
        # consequence at matched training time: our Episode_Termination/object_out_of_bound reads
        # 0.417 where the certified run reads 0.258 at the same env-step count. Ending an episode
        # 200 ms sooner is worth little; ending 16% more episodes than the reference costs exactly
        # the experience the bootstrap needs.
        # Keep the debug camera looking at the workspace rather than away from it. BOTH ends are
        # derived from WORK_SURFACE_Z, so the ~18-degree elevation the scene was framed at survives
        # a future correction of that constant; an absolute literal for the eye height would let the
        # angle drift silently while the lookat moved.
        self.viewer.eye = (2.25, 0.0, WORK_SURFACE_Z + 0.746)
        self.viewer.lookat = (WORKSPACE_X, 0.0, WORK_SURFACE_Z + 0.2)

        # -- GOAL + TASK-STATE VISUALIZATION. Runs AFTER the table swap and after the workspace
        # numbers above, because it measures the pad against them. See the function; the markers
        # are drawn only when ``commands.object_pose.debug_vis`` is True, which the _PLAY configs
        # set and the training configs leave False, so a headless training run pays nothing.
        self.commands.object_pose = _bind_task_state_visualization(self)

        # -- OPTIONAL vertical-goal MIXTURE (epic UWLab-nnlv). Must run AFTER the upgrade above,
        # because it extends that command config rather than replacing it, and after
        # ``_apply_pose_tilt_stage``, whose staged ``ranges`` become the mixture's OTHER component.
        # Inert unless DEXLIFT_GOAL_VERTICAL_PROB is set.
        _apply_goal_vertical_mixture(self)

        # -- OPTIONAL C1 Cartesian hand-pose IK reset (RESET_SPEC_V2.md sec 1, bead F10). Must run
        # HERE, after every stage above that can touch ``commands.object_pose.ranges.pos_x``/
        # ``pos_y`` -- see the function's own docstring for why it reads that range's midpoint as a
        # static anchor rather than the live per-episode goal command. Inert unless
        # DEXRESET_C1_HAND=1.
        _apply_c1_hand_pose_stage(self)

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

        # -- observations: the object point cloud is sampled at the REFERENCE's density, not the
        # vendored tree's. IsaacLabDexterous @ 2208576f uses ``num_points: 32`` on every point-cloud
        # term (dexsuite_env_cfg.py:187, :205, :240); the IsaacLab 2.3.2 dexsuite we inherit from
        # ships 64 (:168). That single number is the whole difference between our policy input width
        # and the reference's -- measured at runtime, 1870 against 1390, and 1870 - 480 = 1390
        # exactly, so nothing else diverges.
        #
        # It matters beyond parity for one specific reason: the 88% and 92.87% results were measured
        # on a 1390-wide observation. A network trained on 1870 is not the same experiment, its
        # checkpoints are not interchangeable with that lineage, and any claim of "we reproduced the
        # reference" would be false at the first layer.
        self.observations.perception.object_point_cloud.params["num_points"] = 32

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
        # -- ``reset_robot_wrist_joint`` IS REMOVED, not re-targeted. dexsuite writes +-3 rad
        # (+-171.9 degrees) to its last wrist joint every reset to scramble tool roll, and this port
        # used to point that at ``wrist_3_joint``. Measured consequence: wrist_3 spawned at +271.76
        # degrees against a +180 target while every other joint sat inside the +-10 degree band, so
        # the authored home posture was true of 25 joints out of 26. The term is dropped so
        # ``reset_robot_joints`` is the ONLY thing writing wrist_3.
        #
        # This is a DELIBERATE DEVIATION from the reference, on instruction. If tool-roll invariance
        # turns out to matter, restore it here rather than widening START_JITTER_RAD -- the roll axis
        # is the only joint it was ever meant to cover.
        self.events.reset_robot_wrist_joint = None

        # -- FULL-PARITY RESET, opt-in via the environment.
        #
        # Set ``DEXLIFT_REF_RESET=1`` to restore dexsuite's OWN start distribution wholesale: base
        # +-0.50 rad, elbow +-0.20, finger roots pinned, the +-3 rad tool-roll scramble on wrist_3,
        # and the object jitter back to the full +-0.2 m in x.
        #
        # WHY THIS IS A SWITCH AND NOT AN EDIT: it is the last structural deviation from the config
        # that produced 92.87 percent, and it is the one variable that tracks whether a run on this
        # plant ever climbs. Three runs with the NARROW reset (+-10 deg, no roll scramble) plateaued
        # at object_upward_motion ~0.003 regardless of hand posture, self-collisions or optimizer
        # step rate; the single run that reached adr 3.96e-3 and success 0.082 (axn28939) had the
        # wide one. Keeping both reachable from one tree means the two can be run side by side
        # without maintaining a fork, and the launch command records which was used.
        if os.environ.get("DEXLIFT_REF_RESET") == "1":
            self.events.reset_robot_joints.params["position_range"] = [-0.5, 0.5]
            self.events.reset_robot_elbow_joint.params["position_range"] = [-0.2, 0.2]
            self.events.reset_finger_root_joints.params["position_range"] = [0.0, 0.0]
            # WRIST ROLL IS JITTERED LIKE ANY OTHER JOINT, NOT SCRAMBLED. dexsuite writes +-3.0 rad
            # (+-171.9 degrees) here to make the policy invariant to tool roll. On instruction this
            # port uses +-0.5 rad instead -- the same band ``reset_robot_joints`` gives every other
            # arm joint above -- because a near-uniform roll is a large extra invariance to learn and
            # nothing in this task rewards it: the goal is a POSITION (``position_only`` is true and
            # ``rot_tol`` is None), so tool roll never enters the success test.
            #
            # THIS IS A DELIBERATE DEVIATION from the config that produced 92.87 percent, and it is
            # the last one left in the reset distribution. If a run with the colliders fixed still
            # fails to lift, this is a candidate to restore -- put 3.0 back here, not elsewhere.
            self.events.reset_robot_wrist_joint = EventTerm(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names="wrist_3_joint"),
                    "position_range": [-0.5, 0.5],
                    "velocity_range": [0.0, 0.0],
                },
            )
            # OBJECT STAYS IN THE FAR HALF. dexsuite jitters x over the full [-0.2, 0.2]; on this
            # UR5e a near-half spawn puts the object where the arm cannot approach it without
            # folding into itself. Kept as a deliberate deviation, on instruction.
            self.events.reset_object.params["pose_range"]["x"] = [0.0, 0.2]
            print("[dexlift] DEXLIFT_REF_RESET=1: dexsuite start distribution restored"
                  " (base +-0.5 rad, elbow +-0.2, wrist_3 +-0.5 rad [NOT dexsuite's +-3.0],"
                  " object x [0.0, 0.2] far half [NOT dexsuite's +-0.2])", flush=True)

        # -- CLEARANCE-AWARE LOW SPAWN, opt-in via the environment (bead UWLab-qiao.1).
        #
        # Set ``DEXLIFT_SPAWN_CLEARANCE=1`` to let the leg spawn NEAR THE TABLE, down to 1 cm of
        # clearance, at ANY orientation, in ADDITION to the existing high spawns (clearance range
        # goes up to 0.40 m, matching the old z pose_range's ceiling).
        #
        # WHY z CANNOT BE A FIXED OFFSET HERE: the leg's lowest point depends on its orientation.
        # A fixed origin height that is "1 cm above the table" for a flat leg (half-thickness
        # 0.015 m) is 85 mm INSIDE the table for a vertical one (half-length 0.100 m). ``reset_object``
        # therefore switches to ``mdp.reset_object_pose_with_clearance``, which samples x/y/roll/
        # pitch/yaw exactly as ``reset_root_state_uniform`` does and DERIVES z from the sampled
        # orientation; see that function's docstring in ``mdp/spawn.py`` for the corner geometry.
        #
        # x/y/roll/pitch/yaw ranges are carried over UNCHANGED from the term this replaces, i.e.
        # dexsuite's y [-0.2, 0.2] / roll,pitch,yaw [-3.14, 3.14] plus the far-half x [0.0, 0.2]
        # override two blocks above (and, under DEXLIFT_REF_RESET=1, that override is already
        # applied by the time this line runs, so this reads it back rather than re-stating it).
        #
        # DEFAULT PATH IS BYTE-IDENTICAL WHEN THIS IS UNSET: nothing above this block touches
        # ``self.events.reset_object``, and this block does not execute unless the env var is "1".
        if os.environ.get("DEXLIFT_SPAWN_CLEARANCE") == "1":
            existing_pose_range = self.events.reset_object.params["pose_range"]
            clearance_range = (0.01, 0.40)
            half_extents = (0.100, 0.015, 0.015)
            surface_z = 0.0
            # LOCAL BOUNDING-BOX CENTRE, relative to the leg's own prim origin. The leg is NOT
            # centred on its origin -- half_extents alone assumes it is, and gets the lowest-corner
            # depth wrong by up to 6.2 mm depending on orientation (found by adversarial review,
            # bead UWLab-qiao.1 follow-on). MEASURED with pxr's ComputeLocalBound on the default
            # prim /square_table_leg4_200mm_merged of square_table_leg4_200mm.usd:
            #   size = (0.19999999, 0.03000100, 0.03000000) m -> half_extents above, confirmed
            #   local bbox centre offset from the prim origin = (-0.0062028, +0.0000042, -0.0000006) m
            # y and z are sub-micron (4.2 um, 0.6 um) and zeroed deliberately; only x is real.
            local_centre_offset = (-0.0062028, 0.0, 0.0)
            self.events.reset_object = EventTerm(
                func=mdp.reset_object_pose_with_clearance,
                mode="reset",
                params={
                    "pose_range": {
                        "x": existing_pose_range["x"],
                        "y": existing_pose_range["y"],
                        "roll": existing_pose_range["roll"],
                        "pitch": existing_pose_range["pitch"],
                        "yaw": existing_pose_range["yaw"],
                    },
                    "clearance_range": clearance_range,
                    "half_extents": half_extents,
                    "surface_z": surface_z,
                    "local_centre_offset": local_centre_offset,
                    "asset_cfg": SceneEntityCfg("object"),
                },
            )
            print(f"[dexlift] DEXLIFT_SPAWN_CLEARANCE=1: reset_object switched to clearance-aware"
                  f" low spawn (clearance_range={clearance_range} m, half_extents={half_extents} m,"
                  f" surface_z={surface_z} m, local_centre_offset={local_centre_offset} m)", flush=True)

        # The arm half of DEXLIFT_REF_ACTUATORS. ``randomize_arm_sysid`` writes, every reset, the
        # OmniReset UR5e identification: armature [3.00, 1.22, 1.39, 0.18, 0.08, 0.38] kg.m^2, static
        # friction [11.0, 9.4, 13.1, 3.0, 2.8, 3.5] N.m and viscous [32.3, 17.8, 21.8, 3.4, 2.9, 3.5]
        # N.m.s/rad. The reference arm has ZERO of all three -- confirmed by pxr probe of its USD,
        # which authors physxJoint:jointFriction = 0.0 on all six arm joints and no armature at all.
        # Shoulder stiction alone eats 7-9 percent of our torque ceiling before the joint moves, and
        # shoulder_pan viscous drag reaches 50.8 N.m at our own velocity cap.
        #
        # The identification was also run with the 2F-85 + D415 wrist assembly mounted, NOT the 1.77 kg
        # DELTO hand, so these are not validated numbers for this robot in the first place.
        if ref_actuators("ARM"):
            self.events.randomize_arm_sysid = None
            print("[dexlift] reference ARM plant: 870 N.m, randomize_arm_sysid DISABLED"
                  " (reference arm has zero armature and zero joint friction)", flush=True)
        else:
            print("[dexlift] identified UR5e ARM plant: 150/28 N.m, randomize_arm_sysid LIVE"
                  " (armature 3.00-0.08 kg.m^2, stiction 11.0-2.8 N.m, viscous 32.3-2.9 N.m.s/rad)",
                  flush=True)
        # -- ``events.reset_root`` IS KEPT, exactly as dexsuite writes it. An earlier revision of
        # this module nulled it, next to ``reset_table``, on the argument that "the reference robot
        # floats and needs its root pinned every reset; ours is bolted to the table through a fixed
        # root joint, so there is nothing to pin". Every clause of that is true and the conclusion
        # is wrong, because the term is not a randomization: ALL THREE of its pose ranges are
        # [0, 0]. It does not jitter the robot, it WRITES THE ROOT POSE TO ``default_root_state``,
        # and on this asset that is the only thing making ``init_state.rot`` true. See
        # BASE_LINK_AUTHORED_YAW: the USD authors the root BODY a half turn from the prim the
        # spawner poses, so with nothing writing the root the arm stands at yaw -pi in the env
        # frame while every config number in this module says identity.
        #
        # MEASURED, 64 envs, at reset, env frame (scripts/tools/reach_measure3.py, _4, _5):
        #   nulled  -- object x [+0.359, +0.745] (on the table), fingertips x [-0.771, -0.132],
        #              goal x [-0.690, -0.301]. Robot, reset posture and goal all in -x; the object
        #              alone in +x. Object to nearest fingertip 0.568 / 1.035 / 1.346 m
        #              (min/median/max), and the hand is not over the table at all -- the tabletop
        #              runs x [-0.35, +1.05] (measured world bbox). Half the goal box was also
        #              outside ``object_out_of_bound``'s x floor of -0.5, i.e. the object would be
        #              killed on arrival. Nothing can touch anything; the ungated
        #              ``any_finger_contact`` reward measured 0.0000 for a whole training run, so
        #              no success, so ADR never promotes, so ``gravity_adr`` never leaves zero
        #              gravity and the object never even falls.
        #   kept    -- object x [+0.352, +0.739], fingertips x [+0.140, +0.748], goal
        #              x [+0.307, +0.697]; root yaw exactly 0.0, root pos exactly (0, 0, 0).
        #              Object to nearest fingertip 0.076 / 0.298 / 0.604 m. 100% of object spawns
        #              and 100% of goals over the tabletop footprint.
        # Nothing else moved: the object spawn, the goal ranges and the reward stds are untouched,
        # and the distances above changed because the arm now faces the workspace, not because the
        # task was rescaled.
        #
        # The rigid-assembly argument is unaffected and in fact strengthened. ``reset_table`` stays
        # nulled (below) because it has real +-50 mm ranges; this term has none, and pinning the
        # robot to (0, 0, 0) identity every reset is what holds the robot-to-table pose fixed
        # rather than leaving it to whatever a future asset revision authors.
        _assert_root_pin_is_a_pin(self)
        # THE TABLE MUST NOT MOVE RELATIVE TO THE ROBOT, and upstream's ``reset_table`` moves it.
        # dexsuite's is a free cuboid standing under a floating arm, so jittering it +-50 mm in x/y
        # every reset randomizes the robot-to-table pose -- a legitimate randomization THERE. Here
        # the table IS the robot's mount: ``table_dims.yaml`` states the frame contract outright
        # ("the robot/support/table are jittered TOGETHER (rigid assembly -- the base is bolted to
        # this table)"), and the robot root is pinned by the line above. Left live, the term breaks
        # the assembly three ways, all measured rather than guessed:
        #   * the base_link world bbox of ``ur5e_delto.usd`` is +-0.0755 in x/y, and the black mat's
        #     cutout radius is 0.1000, so a 50 mm x-jitter drives the base 25 mm into the mat mesh
        #     and a corner jitter (0.0707 diagonal) 46 mm into it;
        #   * every height in this module is measured from a work surface that would no longer be
        #     under the arm, and TABLE_TOP_Y_HALF's 50 mm of spawn margin (see the object spawn
        #     above) would go to zero;
        #   * the red/green status pad is drawn at ``table.root_pos_w`` plus a FIXED offset (see
        #     STATUS_PAD_OFFSET), so the pad itself would slide +-50 mm under the robot each episode.
        # Nulled, not narrowed: there is no axis along which this rig's table may move on its own.
        self.events.reset_table = None

        # -- gravity: LEFT ALONE, i.e. the reference's curriculum, reverting an edit of mine.
        #
        # An earlier revision pinned ``events.variable_gravity`` to full gravity and nulled
        # ``curriculum.gravity_adr``, on the measurement that ADR sat at exactly 0 and gravity never
        # turned on. That measurement was real and the conclusion was wrong: the certified run
        # (wandb ``in3kt4m6``, which ends at adr 0.903 / success 0.891) ALSO reads exactly 0.0 for
        # 529 epochs, including 511 of the 1001 epochs between 100 and 1100, and its ADR does not
        # leave the floor until epoch 1155 -- our iteration 650. Our longest run had reached 66.
        # The stall signature was the reference's own bootstrap, observed 10% of the way in.
        #
        # Two further reasons not to reinstate the pin, both from the reference's source:
        #   * ``initial_final_interpolate_fn`` returns NO_CHANGE below frac 0.1, so pinning gravity
        #     does not merely change gravity -- it silently freezes ALL TEN ADR terms.
        #   * dexsuite_env_cfg.py:386-389 states the intent outright: the gravity curriculum exists
        #     "which removes the need for a special Lift reward". Pinning gravity deletes the
        #     shaping the reward set was tuned around and makes difficulty-0 strictly harder.
        #
        # -- PhysX contact buffer: sized for a five-fingered hand, not for a parallel jaw.
        # MEASURED on run vbhg1qwz (4096 envs): 2379 occurrences of
        #   "collisionStackSize buffer overflow detected, please increase its size to at least
        #    <N> in the scene desc! Contacts have been dropped."
        # The demand printed as -1280352880, i.e. it had wrapped signed 32-bit: the real figure is
        # 2**32 - 1280352880 = 3.01 GB against a 64 MB default, and it was still growing.
        #
        # WHY THIS IS NOT A PERFORMANCE NOTE. Dropped contacts are dropped SILENTLY as far as the
        # MDP is concerned, and every reward that matters here is contact-gated: ``contacts()``
        # reads per-fingertip normal force out of ``force_matrix_w``, and ``position_tracking`` and
        # ``success`` are multiplied by a hard 0/1 gate built from it. A dropped contact is
        # therefore indistinguishable from a finger that never touched -- the grasp happens in the
        # simulation and pays nothing, ADR sees a failure, and the curriculum stalls for a reason
        # that appears nowhere in the logs unless you read PhysX's own stderr.
        #
        # 3.75 GiB, against ~18 GB free at 4096 envs (14.4 of 32.6 GB used) and against the 3.01 GB
        # measured demand. PhysX reserves this up front.
        #
        # THE CEILING IS HARD AND IT IS NOT A MEMORY LIMIT. The USD attribute
        # ``physxScene:gpuCollisionStackSize`` is an ``unsigned int``, so 2**32 is not merely large,
        # it is unrepresentable:
        #   pxr.Tf.ErrorException: Type mismatch for </physicsScene.physxScene:gpuCollisionStackSize>:
        #   expected 'unsigned int', got 'long'
        # -- which is what a first attempt at exactly 4 GiB (4294967296) died with, at gym.make,
        # before any physics ran. Anything at or above 4294967296 fails the same way regardless of
        # how much VRAM the card has.
        #
        # So if the demand ever exceeds ~4 GiB, the buffer CANNOT absorb it and the only honest
        # remedies are fewer environments (the demand scales with them) or simpler hand colliders.
        # Shrinking this back and living with dropped contacts is not one of them.
        self.sim.physx.gpu_collision_stack_size = 4026531840

        # -- contact patch budget, taken from the ORIGINAL DexSuite rather than left at the default.
        # dexsuite_env_cfg.py:516 in IsaacLabDexterous (the codebase that measurably reached 88% on
        # this task and 92.87% on the table leg) sets exactly this, and our config never carried it.
        # It is the same class of omission as the collision stack above: a five-fingered hand
        # generates far more contact patches than the default was chosen for, and running out of
        # patches degrades contact reporting rather than raising an error.
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15

        # -- terminations: drop the abnormal-velocity cut and its penalty. See the function.
        _drop_unreachable_abnormal_robot_cut(self)

        # -- metrics: publish an episode success rate under the certified lineage's own metric
        # name, so the rl_games baseline and this rsl_rl port land on ONE wandb series instead of
        # two. Runs last because it reads the tolerances the ADR curriculum ended up with, which
        # ``super().__post_init__()`` derives and the Lift subclass then trims. See the function.
        _attach_success_rate_metric(self)

        # -- metrics: the v2 gate proxy (V2_REPOSE_RECIPE.md sec 4, bead dr-tlx.2). Off unless
        # DEXRESET_GATE_PROXY=1, and it prints either way. MUST run AFTER
        # _attach_success_rate_metric, whose term it re-attaches with a per-kind split -- running
        # before would have that call overwrite the split back off. See the function.
        _attach_gate_proxy_metric(self)


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
