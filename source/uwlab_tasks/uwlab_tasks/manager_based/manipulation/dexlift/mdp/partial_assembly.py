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

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs.mdp import reset_root_state_uniform
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from uwlab_assets import UWLAB_ASSETS_DATA_DIR, UWLAB_CLOUD_ASSETS_DIR

# Imported, not reimplemented -- see the "Y2" section of the module docstring. This is the SAME
# class ``rl_state_cfg.py``'s ``TrainEventCfg`` uses for the ``ObjectPartiallyAssembledEEGrasped``
# reset type; the file that defines it is owned by another bead right now and is not touched here.
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import (
    reset_insertive_object_from_partial_assembly_dataset,
)

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
