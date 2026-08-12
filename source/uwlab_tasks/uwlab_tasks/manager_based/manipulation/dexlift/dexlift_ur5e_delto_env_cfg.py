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
    """
    env_cfg.terminations.abnormal_robot = None
    env_cfg.rewards.early_termination = None


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


def _assert_gravity_is_on_and_fixed(env_cfg) -> None:
    """Fail construction unless gravity is real AND nothing ramps it.

    Two edits in different files have to agree, and the failure mode of them disagreeing is silent:
    ``curriculum.gravity_adr`` is the ramp, ``events.variable_gravity`` holds the value it ramps,
    and dexsuite ships that value at zero. Null the curriculum without writing the event and the
    result is not "full gravity" but "no gravity, permanently" -- the same logs, the same stalled
    ADR, and a config that reads as though it had been fixed.

    This is the check that ties them together, so neither half can be edited alone. Both halves
    carry the measurement that motivated them; see ``Ur5eDeltoAdrCurriculumCfg.gravity_adr``.
    """
    term = getattr(env_cfg.events, "variable_gravity", None)
    if term is None:
        raise ValueError(
            "events.variable_gravity is missing. It is the term that SETS gravity each reset"
            " (operation 'abs'), so without it gravity is whatever the last episode left behind."
        )
    lo, hi = term.params["gravity_distribution_params"]
    if tuple(lo) != (0.0, 0.0, -9.81) or tuple(hi) != (0.0, 0.0, -9.81):
        raise ValueError(
            f"events.variable_gravity gravity_distribution_params = {(tuple(lo), tuple(hi))}, not"
            " full gravity. dexsuite's default is ((0,0,0), (0,0,0)) and its gravity_adr curriculum"
            " ramps it up with difficulty; this task nulls that ramp (see"
            " Ur5eDeltoAdrCurriculumCfg.gravity_adr) because the ramp cannot start -- contact at"
            " zero gravity bats the object out of bounds, so nothing succeeds, so difficulty never"
            " leaves 0, so gravity never turns on. Measured in run kkbmdt36: adr exactly 0.0 from"
            " iteration 22, peak gravity -0.004 m/s^2, object_out_of_bound 0.519. Nulling the ramp"
            " WITHOUT writing this parameter freezes gravity off instead of turning it on."
        )
    curriculum = getattr(env_cfg, "curriculum", None)
    if curriculum is not None and getattr(curriculum, "gravity_adr", None) is not None:
        raise ValueError(
            "curriculum.gravity_adr is live while events.variable_gravity is pinned at full"
            " gravity. The curriculum term would overwrite the pinned value from difficulty 0 on"
            " the first reset, i.e. back to zero gravity. Set gravity_adr = None, or drop the pin."
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

    gravity_adr = None
    """Gravity is FULL from the first step, not curriculum-ramped. MEASURED, not preferred.

    Upstream ramps gravity from ``(0, 0, 0)`` to ``(0, 0, -9.81)`` with ADR difficulty, so at
    difficulty 0 there is literally no gravity. That is fine while the hand never reaches the
    object and fatal the moment it does, which is exactly the order the two defects were fixed in:

    * Before ``c535b2f`` the arm faced away from the workspace, nothing was ever touched, and the
      object hung motionless (120 steps of zero action, object z 0.29591 -> 0.29591).
    * After it, run ``kkbmdt36`` made contact from iteration 0 -- and every touch then batted the
      object away with nothing to bring it back. ``position_error`` climbed 0.40 -> 1.20 m,
      ``Episode_Termination/object_out_of_bound`` climbed 0.003 -> **0.519**, and ``Curriculum/adr``
      peaked at 3.6e-4 by iteration 7 and sat at **exactly 0.0** from iteration 22 on. Peak gravity
      over the whole run was -0.004 m/s^2: it never turned on.

    The scheduler cannot climb out of that on its own -- no settling means no success, no success
    means the failure decrement pins difficulty at zero, and zero difficulty means no gravity. The
    curriculum is its own floor. ``table_leg_env_cfg.py:561`` nulls this term for the same reason
    ("Zero-gravity resets leave the leg hovering"); that config reached its result with it nulled.

    Every OTHER ADR term is kept. This is not a retreat from domain randomization -- gravity is the
    one dimension whose difficulty-0 value is not "easy" but "physically degenerate", because the
    task is defined as bringing a FALLING object to a goal.
    """

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
        # THE TWO LINES BELOW ARE IN A DIFFERENT FRAME FROM THE ONE ABOVE, and that is worth one
        # sentence because it is what the base-frame defect was made of. ``init_state.pos`` is
        # resolved in the ENV frame; ``ObjectUniformPoseCommand`` samples in the ROBOT ROOT frame
        # and composes with ``robot.data.root_quat_w`` (dexsuite/mdp/commands/pose_commands.py).
        # The two agree only while the root really is at the identity pose this config declares,
        # which is what ``events.reset_root`` buys -- see BASE_LINK_AUTHORED_YAW.
        self.commands.object_pose.ranges.pos_x = (0.3, 0.7)
        self.commands.object_pose.ranges.pos_z = (0.55 + WORKSPACE_Z_SHIFT, 0.95 + WORKSPACE_Z_SHIFT)
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
        self.terminations.object_out_of_bound.params["in_bound_range"]["z"] = (WORK_SURFACE_Z - 0.05, 2.0)
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

        # -- gravity: full from the first step. TWO EDITS, AND EITHER ALONE IS WORSE THAN NEITHER.
        # ``curriculum.gravity_adr`` (nulled there, with the measurements) is only the RAMP; the
        # quantity it ramps is this event's parameter, and dexsuite ships that parameter at
        # ((0,0,0), (0,0,0)) -- zero. So nulling the curriculum term on its own does not restore
        # gravity, it FREEZES it off permanently, which is the exact failure being fixed and would
        # look identical in the logs. ``table_leg_env_cfg.py`` makes both edits (:561 and :662-665);
        # only the first is famous.
        self.events.variable_gravity.params["gravity_distribution_params"] = (
            (0.0, 0.0, -9.81),
            (0.0, 0.0, -9.81),
        )
        _assert_gravity_is_on_and_fixed(self)

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
