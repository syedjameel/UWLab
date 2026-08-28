# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Partially-assembled table-leg spawn for the DEXLIFT TABLE-LEG REORIENT task (bead UWLab-qiao.2 /
UWLab-qiao.6), gated behind ``DEXLIFT_PARTIAL_ASSEMBLY`` in
``dexlift_ur5e_delto_tableleg_env_cfg.DexLiftUR5eDeltoRelJointPosTableLegReorientEnvCfg``.

USER'S GOAL, verbatim: "Spawn partially assembled configuration, with table itself and table leg
partially screwed. Pose goal is set exactly where table leg spawns. The idea policy just needs to
grasp it." Three pieces, each answered by one thing in this module:

1. A RECEPTIVE FIXTURE in a scene that has never had one -- :func:`make_dexlift_receptive_object_cfg`
   -- transplanted from ``rl_state_cfg.py:654-684,747-750`` (``make_receptive_object`` /
   the ``"onelegfixture"`` variant entry).
2. THE LEG SPAWNED PARTIALLY SCREWED INTO IT, reusing the 525 poses already collected in
   ``partial_assemblies.pt`` for this exact pair -- :class:`SpawnPartialAssembly`.
3. THE GOAL PINNED TO THE SPAWN POSE -- :class:`GoalAtSpawnPoseCommand`.

=== Y1: WHERE THE FIXTURE SITS, ARGUED EXPLICITLY ===

The OmniReset generator (``reset_states_cfg.py:195-220``) places ``onelegfixture`` at
``z = ur5_metal_support's DEFAULT root z (0.004) - bottom_offset.pos.z (-0.015625) = 0.019625``,
where ``ur5_metal_support`` is a proxy plate for the physical rig's black mat -- an entity the
DEXLIFT scene has never had (its own table top is at ``WORK_SURFACE_Z = 0.0``, no mat, a 4 mm
difference; see ``dexlift_ur5e_delto_env_cfg.py:219`` and the UWLab-qiao.3 verdict).

TWO WAYS TO PLACE IT. (a) Rest it on DEXLIFT's own, lower table top: root z = 0.000 -
bottom_offset.pos.z (-0.015625) = 0.015625, visually correct on THIS table. (b) Use the SAME
constant OmniReset uses, 0.019625, regardless of what is or is not under it in this scene.

(b) IS THE CORRECT CHOICE, and it is not a compromise. ``robot`` sits at world (0,0,0) in BOTH the
DEXLIFT scene and the OmniReset training scene -- established in UWLab-qiao.3 and re-verified in
UWLab-qiao.7 (``robot_cfg.init_state`` is never given a ``pos=`` override in either scene's
``__post_init__``). A reset state recorded here transfers to the OmniReset training scene by a PURE
RENAME, no coordinate transform (``rekey_dexlift_reset_states.py``'s own documented contract, and
the fixed schema now written by that same script's UWLab-qiao.7 extension). If this fixture's
recorded pose is (a) instead of (b), every harvested state that carries a ``receptive_object`` entry
plants the fixture 4 mm LOW once replayed in the OmniReset scene -- the exact class of bug
UWLab-qiao.7 found and fixed for the schema gap, self-inflicted this time instead of found in
someone else's data. (b) costs a cosmetic 4 mm float above this scene's own tabletop (the fixture is
KINEMATIC -- nothing depenetrates it, nothing reacts to the gap, and the leg spawned relative to it
floats the same 4 mm, invisible at normal camera distance) in exchange for every downstream
consumer -- this task's own recorder (Y4), and any future OmniReset training run that loads a state
this task records -- reading a number that means what it already means everywhere else in the
pipeline. Cosmetic tradeoffs are cheap to accept; silent 4 mm data-consistency bugs are not.

``RECEPTIVE_POSE_RANGE`` below is therefore IDENTICAL to ``reset_states_cfg.py:199-214``'s
``reset_receptive_object_pose`` x/y/yaw ranges, with z COLLAPSED TO THE CONSTANT rather than derived
via an ``offset_asset_cfg``/``use_bottom_offset`` composition -- this scene has no
``ur5_metal_support`` to compose against, and the constant is exactly what that composition would
yield here regardless (0.004 - (-0.015625) does not change because DEXLIFT lacks the plate; it is
independent of what this scene's own table happens to be at).

=== Y2: WHY THE FIXTURE PLACEMENT AND THE LEG COMPOSITION ARE ONE EVENT TERM, NOT TWO ===

``EventManager`` runs ``mode="reset"`` terms in the order ``self.cfg.__dict__.items()`` yields
(``event_manager.py:337,204`` -- confirmed in UWLab-qiao.6/7), which for a ``@configclass`` instance
is FIELD DECLARATION order: base-class fields first, then whatever a subclass's ``__post_init__``
assigns as genuinely NEW attribute names, appended at the end regardless of what order the Python
statements that assign them run in. ``self.events.reset_object`` already exists as a declared field
(inherited from ``dexsuite.EventCfg``) and REASSIGNING it does not move it. A brand-new
``self.events.reset_receptive_object_pose`` -- introduced here for the first time -- would therefore
ALWAYS land after ``reset_object`` in iteration order, no matter which line of this file's
``__post_init__`` runs first. Two separate terms would make the leg-composition term read last
episode's fixture pose (or, on episode 0, its untouched ``init_state``), not this episode's freshly
sampled one -- silently, forever, exactly the class of bug this whole bead chain has been finding
and fixing in OTHER people's code. Folding both steps into one ``ManagerTermBase.__call__`` turns the
ordering into an ordinary Python statement sequence this file fully controls, instead of a manager
iteration guarantee it cannot.

The actual leg-placement math is NOT reimplemented: :class:`SpawnPartialAssembly` constructs and
DELEGATES to ``omnireset.mdp.events.reset_insertive_object_from_partial_assembly_dataset`` --
the exact same class the OmniReset training scene uses to consume ``partial_assemblies.pt``
(``events.py:1519-1622``, composed via ``combine_frame_transforms(receptive_pos_w, receptive_quat_w,
rel_pos, rel_quat)`` at ``:1596-1599``) -- imported, not edited, not copied.

=== Y3: WHY THE GOAL MUST BE A COMMAND SUBCLASS ===

Settled in UWLab-qiao.6 by reading ``manager_based_rl_env.py`` and ``command_manager.py`` directly.
``ManagerBasedRLEnv._reset_idx`` calls ``event_manager.apply(mode="reset", ...)``
(``manager_based_rl_env.py:359-362``) BEFORE ``command_manager.reset(env_ids)``
(``:381``), in the SAME call. ``CommandManager.reset`` unconditionally calls
``self._resample(env_ids)`` -> ``self._resample_command(env_ids)`` (``command_manager.py:120-185``,
specifically ``:147`` and ``:185``) for every env id passed in, regardless of
``resampling_time_range`` -- that range only gates the DIFFERENT resample path inside
``compute()``, which runs mid-episode, not at reset. An event term that wrote "goal = spawn pose"
into ``pose_command_b`` would therefore be silently overwritten by the ordinary uniform resample a
few lines later, in the SAME reset call. :class:`GoalAtSpawnPoseCommand` overrides
``_resample_command`` itself, which is the one hook ``CommandManager.reset`` cannot bypass.

REORIENT ONLY, NOT LIFT: this module is imported into the shared ``dexlift.mdp`` namespace like every
other submodule, but nothing here is WIRED into any environment config by default -- see
``dexlift_ur5e_delto_tableleg_env_cfg.py``'s ``DEXLIFT_PARTIAL_ASSEMBLY`` block, which touches only
``DexLiftUR5eDeltoRelJointPosTableLegReorientEnvCfg``. On Lift, ``rot_std`` is forced ``None``, which
makes ``dexlift.mdp.rewards.success_reward`` return before its contact gate
(``TABLE_LEG_MASS_KG``'s docstring in ``dexlift_ur5e_delto_tableleg_env_cfg.py`` documents this same
early-return for the identical reason) -- an idle policy would collect ``success_reward`` the instant
``goal == spawn`` with no grasp required. On Reorient, ``success_reward`` AND
``rewards.position_tracking`` are both multiplied by ``contacts()``
(``dexlift/mdp/rewards.py:113-165``), so "the policy just needs to grasp it" is enforced by the
existing reward structure, unmodified here.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs.mdp import reset_root_state_uniform
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv, subtract_frame_transforms

from uwlab_assets import UWLAB_ASSETS_DATA_DIR, UWLAB_CLOUD_ASSETS_DIR

# Imported, not reimplemented -- see the "Y2" section of the module docstring. This is the SAME
# class ``rl_state_cfg.py``'s ``TrainEventCfg`` uses for the ``ObjectPartiallyAssembledEEGrasped``
# reset type; the file that defines it is owned by another bead right now and is not touched here.
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import (
    reset_insertive_object_from_partial_assembly_dataset,
)
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import read_metadata_from_usd_directory

from .task_state_vis import TaskStateVisPoseCommand, TaskStateVisPoseCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

__all__ = [
    "DEXLIFT_ONELEGFIXTURE_USD_PATH",
    "DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR",
    "RECEPTIVE_POSE_RANGE",
    "make_dexlift_receptive_object_cfg",
    "SpawnPartialAssembly",
    "GoalAtSpawnPoseCommand",
    "GoalAtSpawnPoseCommandCfg",
    "upgrade_to_goal_at_spawn",
    "GoalBelowSpawnPoseCommand",
    "GoalBelowSpawnPoseCommandCfg",
    "upgrade_to_goal_below_spawn",
]

# -- Y1: the fixture asset and where it sits. See the module docstring's "Y1" section for the
# argument; do not change this z without re-reading it.
DEXLIFT_ONELEGFIXTURE_USD_PATH = (
    f"{UWLAB_ASSETS_DATA_DIR}/Props/FurnitureBench/OneLegInsertionFixture/one_leg_insertion_fixture.usd"
)
"""Byte-identical to ``rl_state_cfg.py:748`` / ``reset_states_cfg.py:637``'s ``"onelegfixture"`` path."""

RECEPTIVE_POSE_RANGE: dict[str, tuple[float, float]] = {
    # x/y/yaw ranges copied verbatim from reset_states_cfg.py:205,209,213 (reset_receptive_object_pose).
    "x": (0.35, 0.60),
    "y": (-0.20, 0.20),
    # NOT a range: the derived constant, collapsed to (z, z) so reset_root_state_uniform's uniform
    # sampler draws exactly this every time. See the module docstring's "Y1" section for the 0.019625
    # derivation (0.004 ur5_metal_support default z - (-0.015625) bottom_offset.pos.z).
    "z": (0.019625, 0.019625),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (-math.pi / 12, math.pi / 12),
}

DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR = f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset"
"""Same default every other consumer of ``partial_assemblies.pt`` uses (``rl_state_cfg.py``,
``reset_states_cfg.py``, ``partial_assemblies_cfg.py``) -- a Hugging Face URL,
``safe_retrieve_file_path`` handles both that and a plain local path transparently (checked: it
branches on ``url.startswith(("http://","https://")) or os.path.isfile(url)``, ``omnireset/mdp/
utils.py:344``). For LOCAL testing against a DL_A6000-style working tree before the pair is
published, override on the CLI, e.g.
``env.events.spawn_partial_assembly.params.dataset_dir=/path/to/Datasets_ur5e_delto/OmniReset``."""


def make_dexlift_receptive_object_cfg() -> RigidObjectCfg:
    """The ``"onelegfixture"`` receptive fixture, transplanted into the DEXLIFT scene.

    Field-for-field identical to ``rl_state_cfg.py``'s ``make_receptive_object(...,
    disable_articulation_root=True)`` call for this pair (``:654-684`` for the factory,
    ``:747-750`` for the call): same USD, same kinematic/mass/collider settings, same
    ``ArticulationRootPropertiesCfg(articulation_enabled=False)`` -- required because the asset was
    run through ``UrdfConverter`` with ``fix_base=True``, which bakes an ``ArticulationRootAPI`` +
    fixed ``root_joint`` into the USD even for a single-link fixture; without disabling it,
    ``RigidObjectCfg`` construction hard-fails ("Found an articulation root when resolving ... for
    rigid objects").

    ``init_state.pos`` is the placeholder ``(0,0,0)`` -- same convention as the OmniReset factory --
    because the REAL per-episode pose comes from :data:`RECEPTIVE_POSE_RANGE` via
    ``reset_root_state_uniform``, not from this static default.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DEXLIFT_ONELEGFIXTURE_USD_PATH,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


class SpawnPartialAssembly(ManagerTermBase):
    """Place the receptive fixture, THEN compose+place the leg against it -- one atomic term.

    See the module docstring's "Y2" section for why this is one term rather than two, and why the
    leg composition delegates to ``omnireset.mdp.events.reset_insertive_object_from_partial_
    assembly_dataset`` instead of reimplementing it.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.receptive_object_cfg: SceneEntityCfg = cfg.params["receptive_object_cfg"]
        self.receptive_object = env.scene[self.receptive_object_cfg.name]
        self.fixture_pose_range: dict[str, tuple[float, float]] = cfg.params["fixture_pose_range"]
        self._dataset_dir: str = cfg.params["dataset_dir"]

        # Delegated composer, constructed against the SAME cfg (its own __init__ reads only the
        # keys it knows about via .get()/[...] -- an extra "fixture_pose_range" key in cfg.params is
        # inert to it). This is the reuse point: nothing below reimplements combine_frame_transforms.
        self._compose = reset_insertive_object_from_partial_assembly_dataset(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        dataset_dir: str,
        insertive_object_cfg: SceneEntityCfg,
        receptive_object_cfg: SceneEntityCfg,
        fixture_pose_range: dict[str, tuple[float, float]],
        pose_range_b: dict[str, tuple[float, float]] = dict(),
    ) -> None:
        del dataset_dir, fixture_pose_range  # consumed at __init__ (self._dataset_dir / self.fixture_pose_range)
        # 1) place the fixture. write_root_pose_to_sim updates RigidObjectData's cached buffers
        #    immediately (isaaclab/assets/rigid_object/rigid_object.py:227, "tensors are not set
        #    into simulation until step ... set into internal buffers"), so step 2 below reads THIS
        #    episode's fixture pose, not last episode's or the init_state default.
        reset_root_state_uniform(env, env_ids, self.fixture_pose_range, {}, self.receptive_object_cfg)
        # 2) compose the leg's world pose against the just-written fixture pose and write it.
        #    dataset_dir is accepted by the delegate's __call__ signature but unused there (the path
        #    was already resolved once, at its own __init__) -- passed through anyway so nothing
        #    relies on that being true.
        self._compose(env, env_ids, self._dataset_dir, insertive_object_cfg, receptive_object_cfg, pose_range_b)


class GoalAtSpawnPoseCommand(TaskStateVisPoseCommand):
    """Goal pinned to the object's OWN spawn pose. See the module docstring's "Y3" section for why
    this has to be a command subclass and cannot be an event term.

    Behaviour is otherwise ``TaskStateVisPoseCommand``'s (metrics, markers, success colour) unchanged
    -- only WHERE the target comes from differs.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        # Read the object's CURRENT world pose. By the time CommandManager.reset() runs, this
        # episode's reset-mode events (including SpawnPartialAssembly, above) have already written
        # it -- see that class's __call__ for the immediacy argument.
        object_pos_w = self.object.data.root_pos_w[env_ids]
        object_quat_w = self.object.data.root_quat_w[env_ids]
        pos_b, quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            object_pos_w,
            object_quat_w,
        )
        self.pose_command_b[env_ids, 0:3] = pos_b
        self.pose_command_b[env_ids, 3:7] = quat_b


@configclass
class GoalAtSpawnPoseCommandCfg(TaskStateVisPoseCommandCfg):
    """Config for :class:`GoalAtSpawnPoseCommand`. Every field means what it means on
    ``TaskStateVisPoseCommandCfg``; only ``class_type`` differs."""

    class_type: type = GoalAtSpawnPoseCommand


def upgrade_to_goal_at_spawn(command_cfg: TaskStateVisPoseCommandCfg) -> GoalAtSpawnPoseCommandCfg:
    """Rebuild an already-configured ``TaskStateVisPoseCommandCfg`` as a goal-at-spawn command.

    Same field-copy idiom as ``task_state_vis.upgrade_pose_command_to_task_state_vis`` -- every
    field of the inherited term (sampling ranges, resampling window, asset names, tolerances,
    markers) is copied rather than restated, so nothing here can drift from what
    ``_bind_task_state_visualization`` already built. The sampling ranges specifically become inert
    once installed (``_resample_command`` no longer reads them), which is fine: they stay in the cfg
    as an honest record of what a non-partial-assembly run would have sampled, not a footgun -- no
    code path reads them anymore on this command instance.
    """
    fields = {
        field.name: getattr(command_cfg, field.name)
        for field in dataclasses.fields(command_cfg)
        if field.name != "class_type"
    }
    return GoalAtSpawnPoseCommandCfg(**fields)


# === Y5 (bead UWLab-xp05.3, "Arm 3"): GOAL BELOW SPAWN, a shaping device against withdrawal ===
#
# ``GoalAtSpawnPoseCommand`` above pins the goal to the leg's OWN spawn pose, which makes tracking
# satisfied at t=0 -- the recorded mechanism (epic UWLab-xp05) for why a C4-generating policy
# grasps the leg and then withdraws it ~14mm during the grasp, parking at a near-constant 2-4mm
# final depth regardless of spawn depth: nothing in the objective penalises leaving. Pinning the
# goal DEEPER than spawn, along the bore's own axis, inverts that sign -- withdrawal now INCREASES
# tracking error instead of leaving it at its already-satisfied minimum.
#
# THIS IS A SHAPING DEVICE, NOT A TARGET (bead UWLab-xp05.3's own framing, load-bearing): the point
# is not for the policy to reach the displaced goal, only to bias where it leaves the leg. Judge any
# run against this command by the BANKED DEPTH DISTRIBUTION it produces, never by command-tracking
# success.
#
# ``delta_m`` IS SIGNED (bead UWLab-nnlv.3). The paragraph above is the POSITIVE case, and it is the
# case this class was originally built for. A NEGATIVE delta displaces the goal along the SAME bore
# axis in the OPPOSITE direction -- OUT of the mouth, ABOVE it -- which shapes toward a hold above
# the bore instead of a seat inside it. That is the S2' rung (bead UWLab-nnlv, band 20-120mm above
# the mouth), which had no shaping knob at all while this value was asserted ">= 0": the only device
# in the tree could push the goal deeper, i.e. mildly OPPOSE the band it was meant to serve.
def live_bore_deep_axis(fixture, local_deep_axis, env_ids):
    """The bore's own "deep" axis for THIS episode, rotated into world by the fixture's LIVE
    orientation, unit-normalised, with the runtime sign guard applied.

    Shared by :class:`GoalBelowSpawnPoseCommand` and by the episode mixture's partial-assembly
    branch. Both displace a goal along this axis, so both must agree about which way "deeper" points
    and both must refuse under the same conditions -- one function, one guard, no second copy to
    drift (bead UWLab-nnlv.5).

    The axis is read off the FIXTURE, never the leg: the question is "which way is further INTO the
    bore", not "which way is this leg pointing".

    RUNTIME GUARD (team-lead review, bead UWLab-xp05.3): the construction-time sign checks
    # in __init__ read ONLY metadata.yaml -- they never see this episode's actual fixture
    # orientation, so they cannot detect a live orientation their "world-fixed at (0,0,-1)"
    # assumption does not hold for. THIS is the check that looks at fixture_quat_w. Because
    # local_axis is a unit vector, axis_world's z-component is exactly the cosine of the angle
    # between it and world -Z -- close to -1 whenever the fixture sits at (or near) the
    # yaw-only orientations RECEPTIVE_POSE_RANGE promises (yaw about world Z leaves local Z
    # invariant). If it is not, either that range has since been widened to sample roll/pitch,
    # or this "kinematic" fixture has been physically disturbed -- both worth failing loudly on,
    # not silently continuing under a stale assumption. NOTE this does not mean the goal
    # POSITION above is wrong: it is computed FROM this same live-rotated axis_world, so it
    # already tracks whatever the true live orientation is; this assert exists to catch the
    # case where that live orientation itself has drifted from what this class's construction-
    # time reasoning assumed, so a bank generated under it gets flagged rather than silently
    # trusted.
    axis_world_z = axis_world[:, 2]
    _bad = axis_world_z >= -0.9
    assert not bool(_bad.any()), (
        f"GoalBelowSpawnPoseCommand: the LIVE fixture orientation this reset rotates the deep "
        f"axis to a world z-component not close to -1 for {int(_bad.sum())}/{_bad.numel()} "
        f"env(s) this call (worst z={axis_world_z.max().item():.4f}, need < -0.9) -- see this "
        "class's docstring, 'WHAT THESE TWO CHECKS DO NOT COVER' section. Either "
        "RECEPTIVE_POSE_RANGE now samples nonzero roll/pitch, or the fixture has been "
        "disturbed; the construction-time sign check no longer covers this episode's actual "
        "geometry. Refusing to bank a state under an unverified axis assumption."
    )
    """
    fixture_quat_w = fixture.data.root_quat_w[env_ids]
    local_axis = local_deep_axis.expand(fixture_quat_w.shape[0], -1)
    axis_world = quat_apply(fixture_quat_w, local_axis)
    axis_world = axis_world / axis_world.norm(dim=-1, keepdim=True)

    axis_world_z = axis_world[:, 2]
    _bad = axis_world_z >= -0.9
    assert not bool(_bad.any()), (
        f"live_bore_deep_axis: the LIVE fixture orientation rotates the deep axis to a world "
        f"z-component not close to -1 for {int(_bad.sum())}/{_bad.numel()} env(s) this call "
        f"(worst z={axis_world_z.max().item():.4f}, need < -0.9). Either RECEPTIVE_POSE_RANGE now "
        "samples nonzero roll/pitch, or the fixture has been disturbed; the construction-time sign "
        "check no longer covers this episode's actual geometry. Refusing to displace a goal under "
        "an unverified axis assumption."
    )
    return axis_world


class GoalBelowSpawnPoseCommand(TaskStateVisPoseCommand):
    """Goal pinned ``delta_m`` away from the leg's own spawn pose, along the bore's own "deep" axis.
    ``delta_m`` is SIGNED: POSITIVE places the goal DEEPER into the bore (the original,
    withdrawal-opposing use -- hence the class name), NEGATIVE places it the opposite way along that
    same axis, OUT of the mouth and ABOVE it. See the module docstring's "Y5" section above for why
    (the withdrawal-incentive argument, and what the negative sign is for) and for the explicit
    warning against reading either sign as a target to reach.

    AXIS SOURCE, reused not re-derived (bead UWLab-xp05.3's own instruction: "source a frame
    convention from the code that CONSUMES it, never from a description of it"). The fixture-local
    ``-Z`` ("deep", mouth -> further in) axis, rotated into world by the FIXTURE's live orientation,
    is the SAME convention ``dexlift/mdp/rewards.py``'s ``axial_displacement_error_tanh`` and
    ``generate_reset_states_policy.py``'s ``SeatedHeldWithProbe._decompose`` both already use and
    have separately validated (the latter by reproducing the known partial-assembly spawn
    distribution) -- not re-derived from the LEG's own orientation, which would answer a different
    question (which way THIS leg happens to be pointing) than the one that matters here (which way
    is further INTO THE BORE). Because ``RECEPTIVE_POSE_RANGE`` only randomises the fixture's yaw
    (roll/pitch pinned to 0), rotating fixture-local ``-Z`` by any sampled fixture orientation
    leaves it at world ``(0,0,-1)`` regardless of the draw -- consistent with, and cross-checked
    against, the metadata-derived "leg enters from above, travelling -Z" argument recorded in
    ``SquareTableLeg200mmDecomp/metadata.yaml``'s own ``assembled_offset`` comment.

    SIGN, asserted at construction from metadata, not merely documented. Two independent,
    metadata-grounded checks, either of which failing means this axis convention does not hold for
    the configured pair and construction refuses. What they pin down is the AXIS -- which world
    direction is "deeper into the bore" for this fixture/leg pair -- NOT the sign of ``delta_m``.
    That makes them exactly as load-bearing under a signed delta as under the original unsigned one
    (bead UWLab-nnlv.3): a negative delta travels the SAME axis backwards, so an inverted axis
    constant or a wrong-signed metadata entry sends BOTH signs to the wrong place -- a positive
    delta out of the mouth, a negative one into the bore -- and each is as silently wrong as the
    other. Neither check is, or may be made, conditional on the sign of ``delta_m``:

    1. The fixture's own ``assembled_offset.quat`` must be identity (WXYZ ``[1,0,0,0]``) -- the same
       precondition ``SeatedHeldWithProbe`` asserts, for the same reason: the "deep" axis is defined
       as a pure translation in the fixture's ROOT frame, not a rotated target frame, and that is
       only valid when the two coincide.
    2. The LEG's own ``assembled_offset`` (its root-to-tip direction, at the fully assembled
       orientation ``root_quat = inverse(offset_quat)``) must point in the SAME world direction as
       the fixture's deep axis -- i.e. a leg placed exactly at assembly has its tip already lying
       along the direction a POSITIVE ``delta_m`` displaces the goal (and directly opposite the
       direction a negative one does). This is the cross-check that the "deeper" this class computes
       and the physical seat the leg is built to reach actually agree.

    WHAT THESE TWO CHECKS DO NOT COVER (team-lead review, bead UWLab-xp05.3): both run ONCE, at
    construction, and read ONLY the two ``metadata.yaml`` files -- neither ever touches
    ``self.fixture.data.root_quat_w``. They are therefore blind BY CONSTRUCTION to anything about
    THIS episode's actual live fixture orientation; they cannot "fail loudly" on a live condition
    they have no way to observe, they simply never look. What they DO catch: a wrong-signed
    hardcoded axis constant or a wrong-signed metadata entry (either flips the dot product from
    ~+1 to ~-1, failing loudly) -- but that is a check on the STATIC CONSTANTS' internal
    consistency, not on runtime behaviour. A THIRD, RUNTIME check in ``_resample_command`` below
    closes that gap: it reads the live ``fixture_quat_w`` every call and verifies the axis it
    actually rotates into world still points close to that same direction, so a widened
    ``RECEPTIVE_POSE_RANGE`` (roll/pitch no longer pinned to 0) or a physically-disturbed
    "kinematic" fixture is caught per-episode, not just algebraically at construction. Note the
    goal POSITION itself was never at risk from this gap -- ``_resample_command`` already rotates
    the local axis by the LIVE per-episode orientation, not a hardcoded world constant -- the gap
    was purely in this class's ability to notice when that live orientation stopped matching the
    assumption the docstring above states.
    """

    def __init__(self, cfg: GoalBelowSpawnPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.receptive_object_cfg: SceneEntityCfg = cfg.receptive_object_cfg
        self.receptive_object_cfg.resolve(env.scene)
        assert self.receptive_object_cfg.name in env.scene.rigid_objects, (
            f"GoalBelowSpawnPoseCommand: {self.receptive_object_cfg.name!r} did not resolve to a "
            "rigid object in the scene -- this command requires DEXLIFT_PARTIAL_ASSEMBLY=1 (the "
            "fixture/receptive_object entity) to already be present, so the bore's own 'deep' axis "
            "has something to be read off. Refusing to construct a displacement with nothing to "
            "measure against."
        )
        self.fixture = env.scene[self.receptive_object_cfg.name]

        assert cfg.leg_usd_path, (
            "GoalBelowSpawnPoseCommand: cfg.leg_usd_path is empty -- must be set by the caller "
            "(see upgrade_to_goal_below_spawn) to the leg USD whose metadata.yaml this class reads "
            "for the sign cross-check below."
        )
        leg_metadata = read_metadata_from_usd_directory(cfg.leg_usd_path)
        fixture_metadata = read_metadata_from_usd_directory(cfg.fixture_usd_path)
        for name, path, metadata in (
            ("leg", cfg.leg_usd_path, leg_metadata),
            ("fixture", cfg.fixture_usd_path, fixture_metadata),
        ):
            assert metadata.get("assembled_offset") is not None, (
                f"GoalBelowSpawnPoseCommand: {name} metadata.yaml (next to {path!r}) has no "
                "'assembled_offset' -- cannot run the sign cross-check without it."
            )

        device = env.device
        # -- SIGN CHECK 1: fixture assembled_offset.quat must be identity. Same precondition, same
        # reason, as SeatedHeldWithProbe's own assert (this class's docstring, "SIGN" section).
        fixture_offset_quat = torch.tensor(fixture_metadata["assembled_offset"]["quat"], dtype=torch.float32, device=device)
        assert torch.allclose(fixture_offset_quat.cpu(), torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-4), (
            f"GoalBelowSpawnPoseCommand: fixture ({cfg.fixture_usd_path!r}) assembled_offset.quat = "
            f"{fixture_metadata['assembled_offset']['quat']} is not identity (WXYZ [1,0,0,0]) -- the "
            "fixture-local -Z 'deep' axis convention this class reuses (see its docstring, 'AXIS "
            "SOURCE' section) is only valid when it is. Refusing to construct."
        )

        # This class's own "deep" axis, in the fixture's local frame -- the SAME constant
        # dexlift/mdp/rewards.py's axial_displacement_error_tanh and SeatedHeldWithProbe._decompose
        # already use (see this class's docstring, "AXIS SOURCE" section), reused here rather than
        # re-derived so all three stay in lockstep by construction, not by coincidence.
        self._fixture_local_deep_axis = torch.tensor([0.0, 0.0, -1.0], device=device)

        # -- SIGN CHECK 2: the leg's own tip, at the fully ASSEMBLED orientation, must point along
        # this SAME world direction. root_quat_at_assembly = inverse(offset_quat) because
        # target_quat (identity) = root_quat o offset_quat -- see the metadata.yaml comment beside
        # SquareTableLeg200mmDecomp's own assembled_offset for the full derivation this reproduces.
        leg_offset_pos = torch.tensor(leg_metadata["assembled_offset"]["pos"], dtype=torch.float32, device=device)
        leg_offset_quat = torch.tensor(leg_metadata["assembled_offset"]["quat"], dtype=torch.float32, device=device)
        leg_offset_norm = leg_offset_pos.norm()
        assert leg_offset_norm > 1e-6, (
            f"GoalBelowSpawnPoseCommand: leg ({cfg.leg_usd_path!r}) assembled_offset.pos = "
            f"{leg_metadata['assembled_offset']['pos']} is ~zero -- cannot form a root-to-tip "
            "direction to cross-check against."
        )
        leg_tip_local_axis = leg_offset_pos / leg_offset_norm
        root_quat_at_assembly = quat_inv(leg_offset_quat.unsqueeze(0))
        leg_tip_axis_world_at_assembly = quat_apply(root_quat_at_assembly, leg_tip_local_axis.unsqueeze(0))[0]
        agreement = torch.dot(leg_tip_axis_world_at_assembly, self._fixture_local_deep_axis).item()
        assert agreement > 0.9, (
            "GoalBelowSpawnPoseCommand: leg tip axis at the ASSEMBLED orientation resolves to "
            f"{leg_tip_axis_world_at_assembly.tolist()}, which does not agree (dot={agreement:.4f}, "
            "need > 0.9) with the fixture's own 'deep' axis "
            f"{self._fixture_local_deep_axis.tolist()} -- the axis this class displaces along is "
            "inverted for this pair, so BOTH signs of delta_m land backwards: a positive delta "
            "would move the goal OUT of the mouth instead of deeper into the bore, and a negative "
            "one into the bore instead of above the mouth. Refusing to construct rather than "
            "silently produce a backwards goal. See this class's docstring, 'SIGN' section, "
            "check 2."
        )

        self.delta_m = float(cfg.delta_m)

        # -- LOWER BOUND (bead UWLab-nnlv.3). Was `>= 0.0`: the value was UNSIGNED, so a goal could
        # only ever be pushed DEEPER, and the S2' rung (band 20-120mm ABOVE the mouth) had no
        # shaping device at all -- the only one in the tree mildly opposed it. A negative delta now
        # travels the SAME bore axis backwards (see `_resample_command`: goal = spawn + delta *
        # axis_world, so delta < 0 moves along -axis_world, out of the mouth).
        #
        # -0.200m is a POLICY bound, NOT a measured physical constant -- unlike _ENGAGED_SPAN_M
        # below, which is the bore's own geometry. Outside the mouth there is no bore feature to
        # measure against; this number is simply headroom around the band the sign exists to serve
        # (S2' tops out 120mm above the mouth), sized to catch a unit slip or a typo rather than to
        # describe anything about the hardware. Widening it costs nothing physical; it only widens
        # what a mistyped env var can silently do.
        _ABOVE_MOUTH_LIMIT_M = 0.200
        assert self.delta_m >= -_ABOVE_MOUTH_LIMIT_M, (
            f"GoalBelowSpawnPoseCommand: delta_m={self.delta_m * 1000.0:.2f}mm is below the signed "
            f"lower bound of -{_ABOVE_MOUTH_LIMIT_M * 1000.0:.1f}mm. delta_m displaces the "
            "COMMANDED goal from the leg's spawn pose along the bore's own axis: POSITIVE means "
            "DEEPER INTO the bore, NEGATIVE means the opposite way along that same axis -- OUT of "
            "the mouth, ABOVE it. So this value asks for a goal more than "
            f"{_ABOVE_MOUTH_LIMIT_M * 1000.0:.1f}mm above the mouth, past the headroom around the "
            "S2' rung's 20-120mm band that this floor exists to bound (a policy bound, not a "
            "physical one -- see this assert's comment). Pass a value in [-200mm, +25mm], in "
            "METRES here and in MILLIMETRES via DEXLIFT_GOAL_BELOW_SPAWN_MM."
        )

        # -- UPPER BOUND (critic3 review, bead UWLab-xp05.3): only ">= 0" was ever asserted before
        # this, so DEXLIFT_GOAL_BELOW_SPAWN_MM=50 (double the bore's own engaged span) passed
        # silently. 0.025m matches generate_reset_states_policy.py's own --c4_engaged_span_mm
        # default (25.0mm) -- the mouth-to-seat distance for THIS pair; see that flag's help text
        # for why it is a measured CLI constant, not something metadata.yaml carries. Past that
        # span the commanded goal sits BEYOND the blind end of the bore entirely, which is not "more
        # shaping", it is nonsense for this pair. This constant is a SAFETY CEILING only, not wired
        # to `--c4_engaged_span_mm` itself -- if that flag is ever pointed at a different pair with a
        # different span, this ceiling does not follow it automatically; re-check by hand.
        _ENGAGED_SPAN_M = 0.025
        assert self.delta_m <= _ENGAGED_SPAN_M, (
            f"GoalBelowSpawnPoseCommand: delta_m={self.delta_m * 1000.0:.2f}mm exceeds the bore's "
            f"own engaged span ({_ENGAGED_SPAN_M * 1000.0:.1f}mm) -- past this the goal sits beyond "
            "the blind end of the bore, not deeper inside it. Depth and roll are coupled at "
            "~38.39 deg/mm through the thread (bead UWLab-xp05.3's own physical-constraint note): a "
            "large offset does not buy more depth, it buys ejections. Refusing to construct above "
            "the physical ceiling."
        )
        if self.delta_m > 0.010:
            print(
                f"[dexlift] WARNING GoalBelowSpawnPoseCommand: delta_m={self.delta_m * 1000.0:.2f}mm"
                " exceeds 10mm. This is a SHAPING DEVICE, not a target -- the plan calls for 3-5mm;"
                " above roughly 10mm the thread coupling (38.39 deg/mm) starts trading depth for"
                " ejections, not more depth. Proceeding, but this is outside the planned range.",
                flush=True,
            )

        # -- THE NEGATIVE SIDE GETS ITS OWN THRESHOLD, NOT THE MIRROR OF 10mm (bead UWLab-nnlv.3).
        # Deliberately asymmetric, because the two numbers measure different things. 10mm above is
        # PHYSICS INSIDE THE BORE: past roughly there, thread coupling (38.39 deg/mm) starts trading
        # depth for ejections. Outside the mouth there is no thread, no engaged span and no such
        # trade, so that number describes nothing on this side -- and mirroring it would fire on
        # EVERY sanctioned negative run, since the rung this sign exists for (S2', bead UWLab-nnlv)
        # asks for 20-120mm above the mouth, all of which is past 10mm. A warning that fires on
        # every intended use is a warning nobody reads. So the threshold here is the FAR EDGE of
        # that rung's own band instead: past 120mm above the mouth, no rung in this epic is asking
        # for a goal, which makes such a value more likely a unit slip than a choice. Note this
        # bound and the -200mm floor asserted above are the same kind of number (a band edge and its
        # headroom), whereas 10mm and the +25mm ceiling are both bore geometry.
        if self.delta_m < -0.120:
            print(
                f"[dexlift] WARNING GoalBelowSpawnPoseCommand: delta_m={self.delta_m * 1000.0:.2f}mm"
                " places the goal more than 120mm ABOVE the bore mouth. This is a SHAPING DEVICE,"
                " not a target -- the S2' rung this sign exists for tops out at 120mm above the"
                " mouth, so nothing planned asks for a goal beyond here. Proceeding, but check this"
                " is not a unit slip.",
                flush=True,
            )

        # -- MID-EPISODE RESAMPLE GUARD (critic3, CONFIRMED via isaaclab source trace, bead
        # UWLab-xp05.3): CommandManager.compute() re-resamples via _resample_command whenever
        # time_left <= 0 (isaaclab command_manager.py:160-166), INDEPENDENT of episode reset. The
        # Reorient _PLAY classes' own resampling_time_range=(2.0,3.0) against episode_length_s=4.0
        # (dexsuite_env_cfg.py) means a SECOND resample fires mid-episode (~t=2-3s), rebasing the
        # goal onto wherever the leg has ALREADY withdrawn to by then -- silently absorbing exactly
        # the withdrawal this arm exists to penalise, for the back half of every episode. The train
        # class is unaffected (resample 10s > episode 4s); this only bites the Play/generation path,
        # which is the one this class runs under.
        #
        # _apply_partial_assembly_and_goal_toggles (dexlift_ur5e_delto_tableleg_env_cfg.py) already
        # forces resampling_time_range strictly past episode_length_s at __post_init__ time -- but
        # that is a SNAPSHOT: generate_reset_states_policy.py's own --episode_length_s override
        # mutates env_cfg.episode_length_s AFTER parse_env_cfg (i.e. after __post_init__ already
        # ran), so a resampling_time_range set only there can go stale the moment that flag is used
        # -- and the generator script's own comments say it will be. THIS check re-derives the same
        # relation at MANAGER-CONSTRUCTION time (inside gym.make, strictly after any such override
        # has landed on env.cfg), so it is correct regardless of ordering -- the same "read it off
        # env.cfg inside the term's __init__" pattern already used elsewhere in this file family for
        # exactly this class of override-ordering bug.
        _episode_length_s = float(env.max_episode_length_s)
        _resample_min_s = float(cfg.resampling_time_range[0])
        assert _resample_min_s > _episode_length_s, (
            f"GoalBelowSpawnPoseCommand: commands.object_pose.resampling_time_range="
            f"{tuple(cfg.resampling_time_range)} does not stay strictly past "
            f"env.max_episode_length_s={_episode_length_s}s -- CommandManager.compute() would "
            "re-resample this goal MID-EPISODE, rebasing it onto wherever the leg has already "
            "withdrawn to and silently defeating this class's entire premise (see this assert's "
            "own comment). Fix at the source (_apply_partial_assembly_and_goal_toggles) or, if "
            "episode_length_s was overridden after cfg construction, re-derive "
            "resampling_time_range immediately after that override."
        )

        # SIGNED (bead UWLab-nnlv.3) -- name the direction outright rather than leaving a reader of
        # the log to infer it from a minus sign in front of a variable called "below spawn".
        _direction = "DEEPER INTO the bore" if self.delta_m > 0.0 else "OUT OF the mouth, ABOVE it"
        print(
            f"[dexlift] GoalBelowSpawnPoseCommand ENABLED: delta_m={self.delta_m:.4f} "
            f"({self.delta_m * 1000.0:.2f}mm, {_direction}) from spawn along the fixture's own deep axis "
            f"{self._fixture_local_deep_axis.tolist()} (fixture-local; ROTATED BY THE LIVE FIXTURE "
            "ORIENTATION every reset, expected world-fixed only under the yaw-only "
            "RECEPTIVE_POSE_RANGE assumption -- re-verified per-episode in _resample_command, not "
            "just assumed) -- construction-time sign cross-check passed: leg-tip-at-assembly . "
            f"fixture-deep-axis = {agreement:.4f}. resampling_time_range="
            f"{tuple(cfg.resampling_time_range)} vs episode_length_s={_episode_length_s}s -- exactly "
            "one resample per episode confirmed. SHAPING DEVICE, NOT A TARGET -- judge by banked "
            "depth, not by command tracking (see this class's own docstring).",
            flush=True,
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # Read the object's CURRENT world pose -- by the time CommandManager.reset() runs, this
        # episode's reset-mode events (including SpawnPartialAssembly) have already written it, same
        # immediacy argument as GoalAtSpawnPoseCommand._resample_command above.
        object_pos_w = self.object.data.root_pos_w[env_ids]
        object_quat_w = self.object.data.root_quat_w[env_ids]

        # Extracted to a module function (bead UWLab-nnlv.5) so the episode mixture's
        # partial-assembly branch displaces along the SAME axis, computed by the SAME code, under
        # the SAME runtime guard. A second copy of this is exactly the kind of drifting duplicate
        # this project has been bitten by before.
        axis_world = live_bore_deep_axis(self.fixture, self._fixture_local_deep_axis, env_ids)

        # THE SIGN LIVES HERE, and nowhere else: `axis_world` is the bore's "deep" direction (into
        # the bore, ~world -Z, guarded just above), so delta_m > 0 adds along it -- deeper -- and
        # delta_m < 0 adds along -axis_world, i.e. back out of the mouth and above it. One
        # expression serves both signs; nothing downstream branches on the sign (bead UWLab-nnlv.3).
        goal_pos_w = object_pos_w + self.delta_m * axis_world
        pos_b, quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            goal_pos_w,
            object_quat_w,
        )
        self.pose_command_b[env_ids, 0:3] = pos_b
        self.pose_command_b[env_ids, 3:7] = quat_b


@configclass
class GoalBelowSpawnPoseCommandCfg(TaskStateVisPoseCommandCfg):
    """Config for :class:`GoalBelowSpawnPoseCommand`. Every field means what it means on
    ``TaskStateVisPoseCommandCfg``; ``class_type`` plus three new fields differ.

    ``leg_usd_path``/``fixture_usd_path`` are NOT read from a module constant here (unlike
    ``DEXLIFT_ONELEGFIXTURE_USD_PATH`` above) -- deliberately: the caller (this pair's
    ``__post_init__`` toggle function) already has ``TABLE_LEG_USD_PATH`` in scope, and passing it
    explicitly avoids this module carrying a second, independently-drifting copy of a constant
    another file already owns (the exact class of bug ``assembled_offset``'s own metadata.yaml
    comment records having happened once already).
    """

    class_type: type = GoalBelowSpawnPoseCommand
    leg_usd_path: str = ""
    fixture_usd_path: str = DEXLIFT_ONELEGFIXTURE_USD_PATH
    receptive_object_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object")
    # SIGNED, metres (bead UWLab-nnlv.3): positive = goal displaced DEEPER into the bore, negative =
    # the same bore axis the other way, OUT of the mouth and above it. Range [-0.200, +0.025],
    # asserted with its reasoning in GoalBelowSpawnPoseCommand.__init__.
    delta_m: float = 0.0


def upgrade_to_goal_below_spawn(
    command_cfg: TaskStateVisPoseCommandCfg, *, leg_usd_path: str, delta_m: float
) -> GoalBelowSpawnPoseCommandCfg:
    """Rebuild an already-configured ``TaskStateVisPoseCommandCfg`` as a goal-below-spawn command.

    Same field-copy idiom as ``upgrade_to_goal_at_spawn`` right above -- every field of the
    inherited term is copied rather than restated, so nothing here can drift from what
    ``_bind_task_state_visualization`` already built. ``leg_usd_path`` has no default (the caller
    must supply ``TABLE_LEG_USD_PATH`` or equivalent -- see ``GoalBelowSpawnPoseCommandCfg``'s own
    docstring for why); ``delta_m`` has no default either, to force the caller to make an explicit
    choice rather than silently constructing a zero-offset (degenerate to goal-AT-spawn) command.
    """
    fields = {
        field.name: getattr(command_cfg, field.name)
        for field in dataclasses.fields(command_cfg)
        if field.name != "class_type"
    }
    return GoalBelowSpawnPoseCommandCfg(**fields, leg_usd_path=leg_usd_path, delta_m=delta_m)
