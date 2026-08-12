# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal and task-state visualization for the dexsuite object-pose command.

WHAT IS ALREADY UPSTREAM, and is therefore configured here rather than rebuilt. Vendored
``isaaclab_tasks/.../dexsuite/mdp/commands/pose_commands.py`` already ships all three markers:

* ``goal_pose_visualizer`` at ``/Visuals/Command/goal_pose`` -- the commanded pose;
* ``curr_pose_visualizer`` at ``/Visuals/Command/body_pose`` -- the object's current pose;
* ``success_visualizer`` at ``/Visuals/SuccessMarkers`` -- the red/green state indicator, drawn at
  the root of whatever scene entity ``success_vis_asset_name`` names. In stock dexsuite that entity
  is the table, whose own ``CuboidCfg`` is spawned ``visible=False`` so that the marker clone of it
  IS the visible table: the whole tabletop turns red or green.

THREE THINGS THAT ARRANGEMENT CANNOT DO ON THIS RIG, which is what this module adds:

1. **The stock indicator geometry is wrong here, and silently so.** Upstream builds the two success
   prototypes in ``DexsuiteReorientEnvCfg.__post_init__`` by cloning ``self.scene.table.spawn`` --
   evaluated while the table is still dexsuite's 0.8 x 1.5 x 0.04 cuboid, because this task swaps in
   the real lab table only *after* calling ``super().__post_init__()``. The result is a slab of the
   OLD table's dimensions drawn at the NEW table's root, which on this scene is the robot base
   flange: it would sit through the arm and z-fight the tabletop. Nothing raises. This module
   replaces the prototypes with geometry authored against the real table (see
   :func:`make_status_pad_marker_cfg`), and the env config passes the measured size and offset.

2. **The indicator has to move off the table root.** ``VisualizationMarkers.visualize`` places every
   instance at the translation it is handed, and upstream hands it ``root_pos_w`` exactly. That was
   fine when the table's root was its own centre; the real table's asset frame is the ROBOT BASE
   frame, so the tabletop centre is 0.35 m in front of the root. :attr:`status_vis_offset` is added
   to the root position for that reason, which keeps this a config change rather than a second USD.

3. **The colour must be driven by the SCORED predicate, not a second copy of it.** Upstream hard
   codes ``position_error < 0.05`` in ``_update_metrics`` AND, separately, ``distance < 0.05`` in
   ``_debug_vis_callback`` -- two literals for one question. Here the threshold is a config field
   bound, in the env config, to the ADR curriculum's own ``pos_tol``/``rot_tol``, which is the
   thresholded success that decides curriculum promotion. One tensor, :attr:`success_ids`, is
   computed once per step and every marker reads it.

   THE HONEST LIMIT OF THAT CLAIM. Three markers now read one tensor, but the RULE that builds the
   tensor is still written twice in the tree: here, as ``not position_only and success_rot_tol is
   not None``, and in the ADR scheduler as ``move_up = (pos_dist < pos_tol) & (rot_dist < rot_tol)
   if rot_tol else pos_dist < pos_tol`` (``dexsuite/mdp/curriculums.py:106``), which tests
   ``rot_tol`` for TRUTHINESS and never consults ``position_only``. Two ways the copies could
   disagree, and where each is closed:

   * ``rot_tol == 0.0`` -- ADR drops the gate (0.0 is falsy), this drops nothing. Closed at bind
     time: ``_bind_task_state_visualization`` refuses a non-positive ``rot_tol``.
   * ``position_only=True`` with a non-None ``rot_tol`` -- this drops the gate, ADR keeps it. NOT
     closed, only inert: every config built today pairs ``position_only=True`` with
     ``rot_tol=None`` (lift) or ``position_only=False`` with ``rot_tol=0.25`` (reorient). Making it
     structurally impossible means moving the predicate itself into one shared callable, which the
     ADR term would have to import.

   AND THE COLOUR IS NOT THE REWARD, despite the reward term being named ``success``. On this env
   that term resolves to ``dexlift.mdp.rewards.success_reward``, a CONTACT-GATED shadow of the
   dexsuite function: with ``rot_std`` set it multiplies by a thumb/fingertip contact test, so on
   reorient an object thrown to the goal and released is green here and promotes ADR while that
   reward pays exactly 0. The colour tracks the only THRESHOLDED success test in the task, which is
   ADR's; the reward is a dense shaping term with no threshold to track.

SEMANTICS OF THE COLOUR, stated because the two readings differ and an unlabelled marker gets
misread: **GREEN MEANS CURRENTLY WITHIN TOLERANCE.** It is not sticky and not latched. The object
being inside the goal ball right now turns the pad green; the object drifting back out turns it red
again in the same episode. That matches the scored quantity: dexsuite has no success TERMINATION,
its ``success_reward`` is a per-step tanh on the current error, and the ADR scheduler tests the
error it sees at the moment it runs. A latched "achieved at some point this episode" marker would
report something no term in this task rewards.

COST. Every marker in this module is drawn from ``_debug_vis_callback``, which the command manager
subscribes only when ``debug_vis`` is True, and the success visualizer's own visibility is tied to
the same flag -- so with ``debug_vis`` False (the default on every training config)
``VisualizationMarkers.visualize`` is never called, and its internal ``if not self.is_visible()``
guard would return before any device transfer even if it were. What a headless run pays is one
``<`` on a length-``num_envs`` tensor per step, on device, which the upstream code already paid.

SAFETY. Markers are ``UsdGeom.PointInstancer`` prototypes, not scene entities. The prototype configs
below deliberately carry NO ``rigid_props`` and NO ``collision_props``, so nothing writes a
``CollisionAPI`` or ``RigidBodyAPI`` in the first place, and ``VisualizationMarkers`` additionally
strips physics from every prototype subtree it is given (``_process_prototype_prim`` removes
``UsdPhysics.RigidBodyAPI``/``ArticulationRootAPI``, disables joints, and calls
``physx_utils.removeRigidBodySubtree``). It also sets ``invisibleToSecondaryRays``, so the markers
stay out of depth and tiled-camera renders. They cannot enter a contact pair or perturb physics.
"""

from __future__ import annotations

import dataclasses
import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error
from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.commands.pose_commands import ObjectUniformPoseCommand
from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.commands.pose_commands_cfg import (
    ALIGN_MARKER_CFG,
    ObjectUniformPoseCommandCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "TaskStateVisPoseCommand",
    "TaskStateVisPoseCommandCfg",
    "make_status_pad_marker_cfg",
    "make_goal_marker_cfg",
    "upgrade_pose_command_to_task_state_vis",
]

# Colours. Saturated diffuse plus a little emissive, because this pad is read at a glance from the
# default debug camera and the scene's only light is a dome at intensity 1000 -- a purely diffuse
# dark red and dark green are hard to tell apart in the shadow under the tabletop lip.
FAILURE_COLOR = (0.55, 0.06, 0.06)
SUCCESS_COLOR = (0.06, 0.50, 0.10)
FAILURE_EMISSIVE = (0.20, 0.01, 0.01)
SUCCESS_EMISSIVE = (0.01, 0.18, 0.03)


def make_status_pad_marker_cfg(size: tuple[float, float, float]) -> VisualizationMarkersCfg:
    """Two-prototype red/green pad for ``/Visuals/SuccessMarkers``.

    PROTOTYPE ORDER IS THE CONTRACT: index 0 is failure, index 1 is success, because the command
    term hands ``VisualizationMarkers`` the boolean success flag cast with ``.int()``. Reordering
    this dict inverts the colours everywhere with nothing to catch it, which is why
    :class:`TaskStateVisPoseCommand` asserts the order at construction.

    Args:
        size: full extents (x, y, z) of the pad, in metres.
    """
    return VisualizationMarkersCfg(
        prim_path="/Visuals/SuccessMarkers",
        markers={
            "failure": sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=FAILURE_COLOR, emissive_color=FAILURE_EMISSIVE, roughness=0.6
                ),
            ),
            "success": sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=SUCCESS_COLOR, emissive_color=SUCCESS_EMISSIVE, roughness=0.6
                ),
            ),
        },
    )


def make_goal_marker_cfg(prim_path: str, ball_radius: float, opacity: float = 1.0) -> VisualizationMarkersCfg:
    """Upstream's three-prototype align marker with the spheres resized to the success tolerance.

    PROTOTYPE ORDER IS AGAIN THE CONTRACT and is upstream's: 0 ``frame``, 1 ``position_far``,
    2 ``position_near``. The position-only draw path selects ``success + 1``, so a reorder breaks
    both the colour and the orientation display.

    The ``frame`` prototype is upstream's object, reused by reference rather than restated -- it is
    the only marker here that reads a USD from the asset cloud, and it is only ever loaded on a
    reorientation task with ``debug_vis`` on.

    Args:
        prim_path: where the ``PointInstancer`` goes.
        ball_radius: sphere radius. Passing the SUCCESS POSITION TOLERANCE makes the goal marker the
            tolerance ball itself -- the object being inside the sphere is exactly the condition
            that turns it green -- instead of upstream's decorative 1 cm dot. SCOPE: this only
            reaches a POSITION-ONLY task. On a reorientation task the goal has an orientation to
            show, so upstream's draw path selects prototype 0, the ``frame``, and neither sphere is
            instanced; see :meth:`TaskStateVisPoseCommand._debug_vis_callback`. Sizing them is still
            worth doing -- the same factory builds the lift task's marker -- but "the goal ball IS
            the tolerance ball" is a statement about lift, not about the task family.
        opacity: sphere opacity; below 1.0 so a goal ball drawn around the object does not hide it.
    """
    material = {"roughness": 0.4, "opacity": opacity}
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "frame": ALIGN_MARKER_CFG.markers["frame"],
            "position_far": sim_utils.SphereCfg(
                radius=ball_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=FAILURE_COLOR, emissive_color=FAILURE_EMISSIVE, **material
                ),
            ),
            "position_near": sim_utils.SphereCfg(
                radius=ball_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=SUCCESS_COLOR, emissive_color=SUCCESS_EMISSIVE, **material
                ),
            ),
        },
    )



class TaskStateVisPoseCommand(ObjectUniformPoseCommand):
    """dexsuite's object-pose command with one success predicate feeding all three markers.

    Behaviour is upstream's except for the four points in the module docstring. In particular the
    sampling, the command buffer, the metrics and the observation the policy sees are untouched:
    this class adds no state the MDP can observe and changes no reward.
    """

    cfg: TaskStateVisPoseCommandCfg

    def __init__(self, cfg: TaskStateVisPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # THE one predicate. Everything that draws reads this tensor; nothing recomputes it.
        self.success_ids = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._status_offset = torch.tensor(cfg.status_vis_offset, device=self.device).unsqueeze(0)

        # Prototype order is load-bearing (see the marker factories) and is a dict literal in a
        # config, so it is checked rather than trusted.
        pad_protos = list(cfg.success_visualizer_cfg.markers.keys())
        if pad_protos[:2] != ["failure", "success"]:
            raise ValueError(
                "success_visualizer_cfg.markers must begin with 'failure' then 'success': the term"
                " selects a prototype with the success flag cast to int, so index 0 is the failure"
                f" colour and index 1 the success colour. Got {pad_protos}."
            )

        # Upstream turns the status pad on unconditionally in its __init__ and redraws it every
        # step, headless or not. Tie it to debug_vis instead, which is what makes a headless
        # training run pay nothing: VisualizationMarkers.visualize returns on !is_visible() before
        # it touches the device.
        self.success_visualizer.set_visibility(cfg.debug_vis)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # Deliberately does NOT call super(): the base method both hardcodes the tolerances and
        # draws the status pad from inside the metrics update (i.e. every physics-decimated step,
        # headless included). The metric computation itself is reproduced verbatim.
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.object.data.root_state_w[:, :3],
            self.object.data.root_state_w[:, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

        # INSTANTANEOUS, not latched: green means "within tolerance right now". See the module
        # docstring for why a sticky flag would not match anything this task scores.
        success = self.metrics["position_error"] < self.cfg.success_pos_tol
        if not self.cfg.position_only and self.cfg.success_rot_tol is not None:
            success = success & (self.metrics["orientation_error"] < self.cfg.success_rot_tol)
        self.success_ids = success

    def _set_debug_vis_impl(self, debug_vis: bool):
        # NOTE: the base CommandTerm calls this from ITS __init__, before ObjectUniformPoseCommand
        # has created the success visualizer, hence the hasattr.
        super()._set_debug_vis_impl(debug_vis)
        if hasattr(self, "success_visualizer"):
            self.success_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        # Same de-initialization guard as upstream, plus one for the first frames: this callback is
        # subscribed from CommandTerm.__init__, which runs before the buffers below exist.
        if not hasattr(self, "success_ids") or not self.robot.is_initialized:
            return
        marker_ids = self.success_ids.int()

        # -- TASK STATE: red pad / green pad, at the success-vis asset's root plus the offset.
        self.success_visualizer.visualize(
            translations=self.success_vis_asset.data.root_pos_w + self._status_offset,
            marker_indices=marker_ids,
        )

        # -- GOAL and CURRENT OBJECT POSE. Upstream's two draw paths, with the one difference that
        # the position-only path reads the success flag computed above instead of recomputing a
        # second `distance < 0.05`.
        #
        # NOTE which branch gets the tolerance ball. ``visualize`` with no ``marker_indices`` sets
        # every instance to prototype 0 (isaaclab/markers/visualization_markers.py:316-336), and
        # prototype 0 is the ``frame``. So on a REORIENTATION task the goal is drawn as an
        # orientation frame -- correct, since the orientation is half the goal -- and the resized
        # red/green spheres are simply not instanced. The "goal ball == tolerance ball" property
        # therefore applies to the LIFT task only. Colouring the reorient goal would need a fourth
        # prototype (an oriented frame in each colour), not an index change.
        if not self.cfg.position_only:
            self.goal_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
            self.curr_visualizer.visualize(self.object.data.root_pos_w, self.object.data.root_quat_w)
        else:
            # prototype 1 is 'position_far' and 2 is 'position_near', so the flag shifts by one.
            self.goal_visualizer.visualize(self.pose_command_w[:, :3], marker_indices=marker_ids + 1)
            self.curr_visualizer.visualize(self.object.data.root_pos_w, marker_indices=marker_ids + 1)


@configclass
class TaskStateVisPoseCommandCfg(ObjectUniformPoseCommandCfg):
    """dexsuite's object-pose command with the success predicate and the pad geometry configurable.

    Every inherited field keeps its meaning. The four added below exist because the stock ones are
    literals inside the term's methods; see this module's docstring.
    """

    class_type: type = TaskStateVisPoseCommand

    success_pos_tol: float = 0.05
    """Position error, in metres, at or below which the state reads as success.

    Upstream's literal is 0.05 in two places. The env config OVERWRITES this with the ADR
    curriculum's ``pos_tol``, so the colour and curriculum promotion cannot disagree.
    """

    success_rot_tol: float | None = 0.5
    """Orientation error, in radians, additionally required when :attr:`position_only` is False.

    Ignored entirely on a position-only task, whatever it holds, so a value left over from a
    reorientation config cannot leak into a lift. ``None`` drops the orientation gate.
    """

    status_vis_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation added to the success-vis asset's ROOT before the status pad is drawn there.

    Upstream implicitly assumes root == centre of the thing being coloured. On this rig the table's
    asset frame is the robot base frame, so the pad needs the offset to the tabletop centre.
    """

    success_visualizer_cfg: VisualizationMarkersCfg = make_status_pad_marker_cfg((0.8, 1.5, 0.04))
    """The red/green status pad, restated with a type annotation purely for readability.

    NO BUG IS BEING FIXED HERE, and an earlier revision of this docstring claimed there was. It said
    that upstream declares this same name without an annotation (true --
    ``dexsuite/mdp/commands/pose_commands_cfg.py:91``), that a ``configclass`` therefore leaves it a
    plain CLASS attribute rather than a dataclass field, and that every env config in the process
    consequently shares one marker dict. THE LAST TWO ARE FALSE. ``configclass`` calls
    ``_add_annotation_types`` (``isaaclab/utils/configclass.py:88``, defined at :182), which promotes
    every unannotated class member to an annotated dataclass field, and ``_process_mutable_types``
    (:303) then replaces its value with a ``default_factory`` that deep-copies it (:381, :480).
    Verified against the real upstream class with the offline stub harness:

    * ``'success_visualizer_cfg' in {f.name for f in dataclasses.fields(ObjectUniformPoseCommandCfg)}``
      -> True, with ``default=MISSING`` and ``default_factory=_return_f.<locals>._wrap``;
    * ``hasattr(ObjectUniformPoseCommandCfg, 'success_visualizer_cfg')`` -> False, i.e. no class
      attribute exists to be shared;
    * two freshly constructed instances give ``a.…markers is b.…markers`` -> False, and mutating
      ``a``'s dict leaves ``b``'s at ``{}``.

    So the described cross-config leak never happened, and restating the field changes nothing about
    instance isolation. It is kept because this class also changes the DEFAULT -- from upstream's
    empty ``markers={}`` (filled in by the env's ``__post_init__``) to a usable red/green pad -- and
    because :func:`upgrade_pose_command_to_task_state_vis` overwrites it per instance anyway. The
    default size is dexsuite's own cuboid, kept only so the class is usable unconfigured."""


def upgrade_pose_command_to_task_state_vis(
    command_cfg: ObjectUniformPoseCommandCfg,
    *,
    success_pos_tol: float,
    success_rot_tol: float | None,
    status_pad_size: tuple[float, float, float],
    status_vis_offset: tuple[float, float, float],
    goal_ball_opacity: float = 0.35,
) -> TaskStateVisPoseCommandCfg:
    """Rebuild an inherited pose-command config as a :class:`TaskStateVisPoseCommandCfg`.

    Every field of the inherited term is COPIED rather than restated, so the sampling ranges, the
    resampling window, the asset names and ``position_only`` keep whatever the dexsuite base and
    this task's own overrides put there. Restating them in a subclassed ``CommandsCfg`` would be a
    second copy of upstream's numbers, free to drift from the first.

    ``class_type`` is excluded from the copy for the obvious reason -- it names the upstream term.

    Args:
        command_cfg: the constructed dexsuite command term to upgrade.
        success_pos_tol: position tolerance the colour is bound to. Pass the ADR curriculum's
            ``pos_tol``, which is the thresholded success the curriculum promotes on.
        success_rot_tol: orientation tolerance, or None. Pass the ADR curriculum's ``rot_tol``.
        status_pad_size: full extents of the red/green pad.
        status_vis_offset: pad centre relative to the success-vis asset's root.
        goal_ball_opacity: opacity of the goal / current-pose spheres.
    """
    fields = {
        field.name: getattr(command_cfg, field.name)
        for field in dataclasses.fields(command_cfg)
        if field.name != "class_type"
    }
    upgraded = TaskStateVisPoseCommandCfg(**fields)
    upgraded.success_pos_tol = success_pos_tol
    upgraded.success_rot_tol = success_rot_tol
    upgraded.status_vis_offset = status_vis_offset
    upgraded.success_visualizer_cfg = make_status_pad_marker_cfg(status_pad_size)
    upgraded.goal_pose_visualizer_cfg = make_goal_marker_cfg(
        "/Visuals/Command/goal_pose", ball_radius=success_pos_tol, opacity=goal_ball_opacity
    )
    # The CURRENT-pose marker is a locator, not a tolerance: a second ball of tolerance radius
    # centred on the object would swallow the object itself and both balls would overlap at success.
    upgraded.curr_pose_visualizer_cfg = make_goal_marker_cfg(
        "/Visuals/Command/body_pose", ball_radius=0.015, opacity=1.0
    )
    return upgraded
