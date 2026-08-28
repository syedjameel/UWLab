# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-episode MIXTURE over three episode kinds for the DEXLIFT TABLE-LEG REORIENT finetune (beads
under UWLab-g3z4): CLASSIC (the dexsuite high goal band, unchanged), LOW GOAL (same spawn, a goal
near table height instead), and PARTIAL ASSEMBLY (grasp-only: leg spawns pre-inserted in the
fixture, goal pinned to that spawn pose). Each reset draws one kind per env from three probabilities
and commits to it for the whole episode.

WHY A MIXTURE, AND WHY THE CLASSIC FRACTION MAY NEVER HIT ZERO. Measured directly: driving 100% of
envs to goal-at-spawn (``DEXLIFT_GOAL_AT_SPAWN=1`` on every env, see ``partial_assembly.py``) loses
55% of the skill by epoch 50 and 89% by epoch 300, landing at pass@30mm 0.0000, because the objective
no longer contains the transport task -- nothing left rewards carrying the leg to a real goal.
Keeping most episodes on the ordinary sampled goal (``classic_goal_prob``, default 0.50) is the
documented surviving remedy; ``low_goal_prob`` (0.25) and ``partial_assembly_prob`` (0.25) are new
episode kinds layered alongside it, not replacements for it. :func:`assert_episode_mixture_is_sane`
enforces ``classic_goal_prob > 0`` for exactly this reason -- see its docstring for the full check.

WHY ONE SHARED DECISION, NOT TWO INDEPENDENT DRAWS. The spawn (an EVENT, ``mode="reset"``) and the
goal (a COMMAND, resampled by ``CommandManager.reset()``) have to agree on which kind an env got --
a partial-assembly SPAWN paired with a CLASSIC goal would ask the policy to both grasp AND transport
from a fixture-inserted pose no reward term anticipates, and a CLASSIC spawn paired with a
GOAL-AT-SPAWN target would pay an idle policy the instant the object is dropped, per
``partial_assembly.py``'s "REORIENT ONLY, NOT LIFT" argument (unaffected here: this module is wired
into Reorient only). Two independent ``torch.rand`` draws -- one in the event, one in the command --
would disagree on some fraction of envs by construction. Instead :class:`MixtureResetObject` draws
ONCE per reset and writes the result into a buffer on the env
(:func:`_get_episode_kind_buffer`); :class:`MixtureGoalPoseCommand` only ever READS that buffer.
This is safe by the exact ordering ``partial_assembly.py``'s "Y3" section already established from
reading ``manager_based_rl_env.py`` / ``command_manager.py`` directly:
``ManagerBasedRLEnv._reset_idx`` runs ``event_manager.apply(mode="reset", ...)`` before
``command_manager.reset(env_ids)``, in the same call, for the same ``env_ids`` -- so by the time the
command reads the buffer for a given env, THIS episode's event has already written it, never last
episode's.

WHY THE BUFFER LIVES ON ``env`` AND NOT ON EITHER TERM. ``ManagerTermBase`` instances do not have a
handle to each other -- the event term and the command term are constructed independently by their
own managers -- but both already receive ``env`` in ``__init__`` (stored as ``self._env``). A single
lazily-created attribute on ``env`` is therefore the only thing both sides can reach without either
manager needing to know the other exists. Lazy creation (``getattr`` / create-if-missing) makes this
independent of which manager happens to construct its terms first.

THE FIXTURE IS WRITTEN EVERY RESET, FOR EVERY ENV -- NEVER LEFT STALE. ``scene.receptive_object`` is
a real kinematic, COLLIDABLE prop with mass and baked collision geometry, unconditionally present the
moment ``partial_assembly_prob`` is nonzero (which it is by default). ``RigidObject.reset()`` clears
only external-wrench buffers, NOT root pose -- PhysX keeps whatever pose was last WRITTEN to it,
indefinitely, across resets. If :class:`MixtureResetObject` only repositioned the fixture for the
PARTIAL ids (as an early revision of this module did), then on any env whose kind ever draws PARTIAL,
the fixture's in-bore pose PERSISTS into every later CLASSIC/LOW episode on that same env -- an
unlogged, un-randomized obstacle left sitting inside the leg's own spawn workspace
(``RECEPTIVE_POSE_RANGE`` x (0.35, 0.60) / y (-0.20, 0.20) overlaps the leg spawn's x (0.55, 0.75) /
y (-0.20, 0.20) entirely in y and partly in x), silently corrupting roughly three quarters of all
episodes in a multi-hour run. Worse on an env's very first-ever reset, before it has drawn PARTIAL
even once: the fixture then sits at its literal ``init_state.pos == (0, 0, 0)`` -- the robot's own
origin. The fix is unconditional: :class:`MixtureResetObject` writes the fixture's pose on EVERY
call, for EVERY env_id in the batch -- PARTIAL ids get the composed in-bore pose (as before), CLASSIC
and LOW ids get :data:`PARKED_FIXTURE_POSE_RANGE`, a fixed pose chosen to sit clear of THIS env's own
table, ground plane and robot reach envelope (not merely clear of the leg workspace) so it cannot add
spurious PhysX contact-patch load either -- this codebase has independently documented that a
5-fingered hand's contact patches alone can overflow the shared GPU collision-stack budget (see
``dexlift_ur5e_delto_env_cfg.py``'s ``gpu_collision_stack_size`` comment), so an idle collider is not
a "harmless" place to be sloppy. THIS PARKING POSE IS NOT "EMPTY SPACE" IN AN ABSOLUTE SENSE, though
-- see the "THE PARKED POSE DEPENDS ON COLLISION FILTERING" section below, it reaches into the
neighbouring env's own tile and is safe only because of PhysX collision-group isolation between env
replicas. A clearance assert against the ACTUAL just-written leg position (not a hardcoded box) backs
the leg-overlap half of this at runtime -- see the assert inside :meth:`MixtureResetObject.__call__`.

THE MIXTURE PROBABILITIES ARE READ AT TERM-CONSTRUCTION TIME, NOT AT CFG-CONSTRUCTION TIME. An
earlier revision captured ``classic_goal_prob``/``low_goal_prob``/``partial_assembly_prob`` as literal
floats baked into the ``EventTerm.params`` dict inside ``_apply_episode_mixture``
(``dexlift_ur5e_delto_tableleg_env_cfg.py``), which runs inside ``__post_init__`` at CFG construction
-- ``load_cfg_from_registry`` in ``isaaclab_tasks.utils.hydra.register_task_to_hydra``. Hydra CLI
overrides land LATER: ``hydra_main`` calls ``env_cfg.from_dict(...)`` (-> ``update_class_from_dict``,
a plain recursive ``setattr`` that never re-runs ``__post_init__``) only after that cfg object already
exists. So a sweep such as ``env.classic_goal_prob=0.9`` silently sets an ORPHANED top-level field
that nothing downstream ever reads again -- the baked params dict still holds the ORIGINAL 0.50 /
0.25 / 0.25, the draw in ``__call__`` still uses them, and ``assert_episode_mixture_is_sane`` (if
called at cfg-construction time) already validated the pre-override defaults and therefore cannot
catch a bad override either. Managers, by contrast, are built during ``ManagerBasedEnv.__init__``
inside ``gym.make(task, cfg=env_cfg)``, which is called AFTER ``env_cfg.from_dict(...)`` has already
run -- and ``ManagerBasedEnv.__init__`` stores the fully-overridden cfg as ``self.cfg`` before any
manager (including the ``EventManager`` that constructs :class:`MixtureResetObject`) is built. So
:class:`MixtureResetObject.__init__` reads ``env.cfg.classic_goal_prob`` /
``env.cfg.low_goal_prob`` / ``env.cfg.partial_assembly_prob`` directly, at THAT point, and
:func:`assert_episode_mixture_is_sane` is called there too -- validating the values that will
actually be drawn from, not a snapshot of the pre-override defaults. ``_apply_episode_mixture`` no
longer bakes these two probabilities into ``params`` at all; see its own docstring.

THE PARKED POSE DEPENDS ON COLLISION FILTERING, NOT ON DISTANCE -- MADE LOUD, NOT SILENT. At this
scene's ``env_spacing == 3.0`` m, each env tile's own local frame extends the same
``TABLE_TOP_X == (-0.35, 1.05)`` as this one, and the boundary between adjacent tiles sits at
``+-1.5`` m from an env's own origin. :data:`PARKED_FIXTURE_POSE_RANGE`'s ``x == -2.0`` is therefore
NOT clear of the ``-x`` neighbour's own geometry -- it lands roughly 1.0 m into that neighbour's local
frame, i.e. inside where ITS OWN table/workspace sits. This pose is safe only because
``InteractiveSceneCfg.filter_collisions`` -- ``isaaclab.scene.interactive_scene_cfg``, default
``True`` -- puts every env replica in its own PhysX collision group, so geometry in one env can never
generate a contact with geometry in another regardless of world-frame proximity. That is an assumption
about a FLAG ELSEWHERE IN THE CFG, not a property of this module's own numbers, and an assumption
silently relied upon is exactly the class of bug this whole epic has been finding and closing. So
:class:`MixtureResetObject.__init__` asserts ``env.cfg.scene.filter_collisions is True`` at
construction, with a message pointing back here -- if anyone ever disables it (e.g. to inspect
cross-env geometry, or because a future task genuinely needs inter-env physics), they get a loud,
immediate error instead of a fixture silently sitting in the neighbour's workspace. DO NOT "fix" this
by moving :data:`PARKED_FIXTURE_POSE_RANGE` closer to this env's own origin -- pulling it in without
also shrinking below the leg's own spawn region (x up to 0.75) buys nothing, and this env's own
geometry (table, robot) fills essentially everything closer in.

REUSE, NOT REIMPLEMENTATION. The partial-assembly SPAWN half delegates to the same
``omnireset.mdp.events.reset_insertive_object_from_partial_assembly_dataset`` composer
``partial_assembly.SpawnPartialAssembly`` already uses (this module calls it directly on a subset of
``env_ids`` rather than the whole batch); the GOAL-AT-SPAWN half is the same three lines as
``partial_assembly.GoalAtSpawnPoseCommand._resample_command``; the CLASSIC half is
``ObjectUniformPoseCommand._resample_command`` via ``super()`` (nothing here restates the high-band
sampling). Only the LOW GOAL z band and the mixing logic are new.

SCOPE, mirroring ``partial_assembly.py``: wired into the Reorient table-leg classes only
(``dexlift_ur5e_delto_tableleg_env_cfg.DexLiftUR5eDeltoRelJointPosTableLegReorientEnvCfg`` and its
``_PLAY`` sibling), never Lift, for the identical contact-gate reason documented there.

DOES NOT REPLACE THE WHOLE-RUN ``DEXLIFT_PARTIAL_ASSEMBLY`` / ``DEXLIFT_GOAL_AT_SPAWN`` TOGGLES.
Several tools outside training (``generate_reset_states_policy.py``, ``render_partial_assemblies.py``,
``rekey_dexlift_reset_states.py``, ``cert_ft.sh``) depend on being able to force 100% of envs into the
partial-assembly/goal-at-spawn path for reset-state harvesting and certification, which a probabilistic
mixture cannot give them. The table-leg Reorient ``__post_init__`` therefore still calls
``_apply_partial_assembly_and_goal_toggles`` first; this module's mixture is applied only when that
call reports neither legacy env var fired.

THE MIXTURE IS OPT-IN (``DEXLIFT_EPISODE_MIXTURE=1``), NOT DEFAULT-ON -- CORRECTING THE CLAIM THIS
DOCSTRING USED TO MAKE HERE. An earlier revision wired ``_apply_episode_mixture`` unconditionally
whenever neither legacy toggle fired, and claimed that made every OTHER script's behaviour
"untouched". That was false for exactly the scripts this paragraph names: an ORDINARY
``generate_reset_states_policy.py`` run (any ``--reset_type`` other than
``ObjectPartiallyAssembledEEGrasped``) sets NEITHER legacy env var either -- it never wanted
partial-assembly at all -- so ``_apply_partial_assembly_and_goal_toggles`` reported "nothing fired"
and the mixture installed itself anyway, changing that script's `events.reset_object`/
`commands.object_pose` out from under it. Confirmed by running it: the crash surfaced as
``MixtureResetObject.__init__() got an unexpected keyword argument 'dataset_dir'`` inside
``wrapped_env.reset()`` -- the SAME swallow-then-TypeError chain a bad Hydra override produces (see
``dexlift_ur5e_delto_tableleg_env_cfg.py``'s gravity/mixture bugfix history), but this time the REAL,
swallowed error (visible only earlier in the log, inside Kit's deferred "at play" event-manager
callback) was an ``urllib.error.HTTPError: HTTP Error 404: Not Found`` fetching
``partial_assemblies.pt`` for this exact pair from the default (Hugging Face)
``DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR`` -- that file has never been published there (confirmed
against the HF repo's own directory listing, which does not contain an
``OneLegInsertionFixture__SquareTableLeg200mmDecomp`` entry at all). ``MixtureResetObject`` used to
construct that dataset-loading composer UNCONDITIONALLY at ``__init__`` regardless of
``partial_assembly_prob``, so ANY construction with a positive partial fraction -- the class DEFAULT
is 0.25 -- paid this network dependency whether or not the caller wanted it, including callers that
never draw a single partial-assembly episode in practice (``partial_assembly_prob`` too small,
short smoke runs, etc.) and, worse, callers like ordinary reset-generation that have no interest in
the mixture existing at all.

Both halves of the fix:

1. THE MECHANISM IS NOW GATED BEHIND ``DEXLIFT_EPISODE_MIXTURE=1``, checked in
   ``dexlift_ur5e_delto_tableleg_env_cfg.py``'s ``__post_init__`` alongside (and independently of)
   the legacy-toggle check -- ``_apply_episode_mixture`` now runs ONLY when that env var is "1" AND
   neither legacy toggle fired. Unset (the default for every existing caller, including
   ``generate_reset_states_policy.py``, ``render_partial_assemblies.py``,
   ``rekey_dexlift_reset_states.py`` and any hand-run smoke test), the Reorient table-leg classes
   build byte-identical to how they did before this module existed: plain ``reset_root_state_uniform``,
   plain ``TaskStateVisPoseCommandCfg``, no ``receptive_object`` in the scene at all. Training scripts
   that DO want the mixture (``launch_dexlift_tableleg_reorient_finetune.sh``) must export it
   explicitly, matching the existing ``DEXLIFT_REF_RESET``/``DEXLIFT_POSE_TILT`` idiom -- opt-in for
   training, not default-on for every construction of these classes.
2. :class:`MixtureResetObject` NOW CONSTRUCTS THE PARTIAL-ASSEMBLY COMPOSER LAZILY -- only when
   ``partial_assembly_prob > 0.0`` (see "THE COMPOSER IS CONSTRUCTED LAZILY" at its ``__init__``) --
   so a mixture with the partial fraction genuinely disabled (an explicit ``partial_assembly_prob=0``
   override, sum-preserved by raising ``classic_goal_prob``/``low_goal_prob``) never pays the network
   dependency either. This does NOT fix the underlying 404 -- a run that legitimately wants
   ``partial_assembly_prob > 0`` (the class default, and any real finetune) still needs
   ``partial_assemblies.pt`` for this pair to be reachable, either by publishing it to the
   ``DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR`` default or, more reliably given it is not there today, by
   overriding ``dataset_dir`` to a locally-vendored copy (``env.events.reset_object.params.
   dataset_dir=...`` under Hydra, exactly as ``DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR``'s own docstring
   in ``partial_assembly.py`` already documents doing for local testing).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp import reset_root_state_uniform
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_unique, subtract_frame_transforms

from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import (
    reset_insertive_object_from_partial_assembly_dataset,
)

from . import c3_transport_core
from .partial_assembly import DEXLIFT_ONELEGFIXTURE_USD_PATH, live_bore_deep_axis
from .task_state_vis import TaskStateVisPoseCommand, TaskStateVisPoseCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

__all__ = [
    "EPISODE_KIND_CLASSIC",
    "EPISODE_KIND_LOW_GOAL",
    "EPISODE_KIND_PARTIAL_ASSEMBLY",
    "EPISODE_KIND_TRANSPORT",
    "LOW_GOAL_POS_Z_RANGE",
    "PARKED_FIXTURE_POSE_RANGE",
    "assert_episode_mixture_is_sane",
    "MixtureResetObject",
    "MixtureGoalPoseCommand",
    "MixtureGoalPoseCommandCfg",
    "upgrade_to_episode_mixture",
]

# Episode-kind codes stored in the per-env buffer. Values, not an Enum, because the buffer is a
# plain torch.long tensor compared with ``==`` against these -- an IntEnum would work too but adds
# nothing a comparison against a literal doesn't already have.
EPISODE_KIND_CLASSIC = 0
EPISODE_KIND_LOW_GOAL = 1
EPISODE_KIND_PARTIAL_ASSEMBLY = 2
EPISODE_KIND_TRANSPORT = 3
"""TRANSPORT: ordinary (classic/low-goal-style) spawn, tip-down +- tilt goal ANCHORED to the
object's own spawn x/y. See ``c3_transport_core``'s module docstring (bead dr-ai1.13, F43) for the
full argument: this is the branch missing from the mixture that actually asks the policy to bring
the leg to a tip-down pose, as opposed to holding it near-horizontal (classic/low-goal) or already
being there with nothing to do (partial-assembly)."""

LOW_GOAL_POS_Z_RANGE: tuple[float, float] = (0.02, 0.15)
"""Absolute z range (metres, robot-root frame) for the LOW goal band -- roughly table height.

Same frame as ``commands.object_pose.ranges.pos_z``'s existing high band, which
``dexlift_ur5e_delto_env_cfg.py`` sets to ``(0.295 + WORK_SURFACE_Z, 0.695 + WORK_SURFACE_Z)`` ==
``(0.295, 0.695)`` since ``WORK_SURFACE_Z == 0.0`` -- i.e. the table top sits at z == 0.0 in this
same frame, so 0.02-0.15 m is 2-15 cm above it, well inside the reachable envelope the existing
object spawn (dropped from 0-0.4 m above the same surface) and out-of-bound floor already assume.
Never previously sampled: the high band's floor is 0.295, so nothing between 0 and 0.295 was ever a
goal before this module existed -- the airborne-median symptom this bead responds to.
"""

PARKED_FIXTURE_POSE_RANGE: dict[str, tuple[float, float]] = {
    # Collapsed-to-constant idiom, same as RECEPTIVE_POSE_RANGE's z (partial_assembly.py). x = -2.0 is
    # well clear of THIS env's own table (TABLE_TOP_X == (-0.35, 1.05)), ground plane (a thin plane at
    # GROUND_Z == -0.676; z = 0.02 sits well above it) and UR5e reach envelope -- but is NOT clear of
    # the NEIGHBOURING env's replica: at the scene's env_spacing == 3.0 m, the tile boundary sits at
    # +-1.5 m from this env's own origin, so x = -2.0 lands ~1.0 m INTO the -x neighbour's own local
    # frame -- well within where ITS table/workspace geometry sits (its own TABLE_TOP_X spans the same
    # (-0.35, 1.05) relative to ITS origin). This pose is safe ONLY because
    # ``InteractiveSceneCfg.filter_collisions`` (default True) puts every env replica in its own PhysX
    # collision group, so nothing here can physically touch the neighbour's geometry regardless of
    # world-frame proximity. See :class:`MixtureResetObject`'s ``filter_collisions`` assert, which
    # makes that dependency loud instead of silent, and do NOT read this pose as "empty space" in an
    # absolute sense -- it depends on collision filtering, not distance.
    "x": (-2.0, -2.0),
    "y": (0.0, 0.0),
    "z": (0.019625, 0.019625),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}
"""Where CLASSIC/LOW envs' fixture is parked every reset, so it never sits stale in the leg workspace.

See the module docstring's "THE FIXTURE IS WRITTEN EVERY RESET" section for the bug this closes, and
its "THE PARKED POSE DEPENDS ON COLLISION FILTERING" section for why this is safe by PhysX collision
group isolation between env replicas, not by raw distance from other geometry.
"""

_PARKED_FIXTURE_MIN_CLEARANCE_M = 1.0
"""Minimum required x/y distance (metres) between a non-partial env's fixture and its OWN leg's
just-written spawn position, checked every reset by :class:`MixtureResetObject`. 1.0 m is a large
margin against both the fixture's and the leg's own extents (each well under 0.3 m) and against
:data:`PARKED_FIXTURE_POSE_RANGE`'s 2.0 m parking offset -- a real regression would fail this by
metres, not centimetres, so there is no risk of the margin itself being the flaky part."""

_EPISODE_KIND_BUFFER_ATTR = "_dexlift_episode_kind"


def _get_episode_kind_buffer(env) -> torch.Tensor:
    """Get-or-create the per-env episode-kind buffer, keyed by a private attribute on ``env``.

    Lazy and order-independent: whichever of :class:`MixtureResetObject` /
    :class:`MixtureGoalPoseCommand` is constructed first creates it; the other reuses it. Reset to
    :data:`EPISODE_KIND_CLASSIC` on creation -- inert, since every env's real kind is written by
    :class:`MixtureResetObject` before :class:`MixtureGoalPoseCommand` ever reads it (see module
    docstring's ordering argument), so the initial value is never actually consulted.
    """
    buf = getattr(env, _EPISODE_KIND_BUFFER_ATTR, None)
    if buf is None or buf.shape[0] != env.num_envs:
        buf = torch.full((env.num_envs,), EPISODE_KIND_CLASSIC, dtype=torch.long, device=env.device)
        setattr(env, _EPISODE_KIND_BUFFER_ATTR, buf)
    return buf


def assert_episode_mixture_is_sane(
    classic_goal_prob: float,
    low_goal_prob: float,
    partial_assembly_prob: float,
    transport_goal_prob: float = 0.0,
) -> None:
    """Fail loudly rather than train silently on a broken mixture.

    CALL THIS WITH THE VALUES THAT WILL ACTUALLY BE DRAWN FROM -- i.e. ``env.cfg.classic_goal_prob``
    etc. read INSIDE :meth:`MixtureResetObject.__init__`, after Hydra CLI overrides have landed on
    ``env.cfg``. Calling it earlier, against the cfg object's fields at ``__post_init__`` time, only
    validates whatever the dataclass DEFAULTS were and provably cannot catch a bad CLI override --
    see the module docstring's "THE MIXTURE PROBABILITIES ARE READ AT TERM-CONSTRUCTION TIME" section
    for exactly why (the override lands on the cfg object between when ``__post_init__`` runs and
    when the manager that would read these values is constructed).

    ``transport_goal_prob`` defaults to 0.0 so every EXISTING call site (and every existing config,
    whose dataclass simply has no such field yet) keeps validating exactly as before -- see
    ``c3_transport_core`` (bead dr-ai1.13, F43) for what this fourth branch is.

    Two independent checks, delegated to :func:`c3_transport_core.validate_episode_mixture_fractions`
    so they are unit-testable without an Isaac Sim process (that module's own docstring explains the
    split):

    1. The four fractions must sum to 1.0 -- catches a CLI override of one field
       (``env.classic_goal_prob=0.7``) without updating the others, which ``update_class_from_dict``
       would otherwise apply silently since every field already exists with a default (see this
       module's callers for why the fields must exist as literals in the first place).
    2. ``classic_goal_prob`` must stay strictly positive -- see the module docstring's "WHY A
       MIXTURE" section for the measured 55%/89%/pass@30mm-0.0000 collapse this guards against.
    """
    c3_transport_core.validate_episode_mixture_fractions(
        classic_goal_prob, low_goal_prob, partial_assembly_prob, transport_goal_prob
    )


class MixtureResetObject(ManagerTermBase):
    """Reset event: draw this reset's per-env episode kind, then spawn accordingly.

    CLASSIC and LOW GOAL envs get the ordinary uniform object spawn (``pose_range`` -- whatever
    ``reset_object`` already carried before this term replaced it, narrowed x included) AND have the
    fixture parked clear of this env's own table/robot/ground (:data:`PARKED_FIXTURE_POSE_RANGE`,
    safe from the NEIGHBOURING env's geometry only via PhysX collision filtering -- see the module
    docstring's "THE PARKED POSE DEPENDS ON COLLISION FILTERING" section and this class's
    ``filter_collisions`` assert) -- every reset, not only the first one an env draws CLASSIC/LOW,
    because PhysX keeps a kinematic body's pose forever until something writes it again; see the
    module docstring's "THE FIXTURE IS WRITTEN EVERY RESET" section for the persistence bug this
    closes. PARTIAL ASSEMBLY envs get the fixture placed and the
    leg composed against it, delegating to
    ``omnireset.mdp.events.reset_insertive_object_from_partial_assembly_dataset`` exactly as
    ``partial_assembly.SpawnPartialAssembly`` does -- see that class's docstring ("Y2") for why the
    fixture-then-leg placement has to be one atomic term rather than two separately-ordered ones;
    the same argument applies here, restricted to a subset of ``env_ids``.

    See the module docstring for why the kind is drawn HERE (not in the command) and written into a
    shared buffer instead of being redrawn independently downstream, and for why the mixture
    fractions are read from ``env.cfg`` in ``__init__`` rather than accepted as ``__call__`` params.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.receptive_object_cfg: SceneEntityCfg = cfg.params["receptive_object_cfg"]
        self.receptive_object = env.scene[self.receptive_object_cfg.name]
        self.insertive_object_cfg: SceneEntityCfg = cfg.params["insertive_object_cfg"]
        self.insertive_object = env.scene[self.insertive_object_cfg.name]
        self.fixture_pose_range: dict[str, tuple[float, float]] = cfg.params["fixture_pose_range"]
        self._dataset_dir: str = cfg.params["dataset_dir"]

        # -- PARKED_FIXTURE_POSE_RANGE is safe by PhysX collision-GROUP isolation between env
        # replicas, not by distance -- see the module docstring's "THE PARKED POSE DEPENDS ON
        # COLLISION FILTERING" section: at this scene's env_spacing, the parking x reaches well into
        # the neighbouring env's own tile. Assert the dependency instead of relying on it silently, so
        # disabling filtering anywhere in the cfg fails loudly here instead of parking the fixture in
        # the neighbour's workspace.
        assert env.cfg.scene.filter_collisions, (
            "MixtureResetObject parks the fixture (PARKED_FIXTURE_POSE_RANGE) at a pose that is only"
            " clear of the neighbouring env's geometry because InteractiveSceneCfg.filter_collisions"
            " puts every env replica in its own PhysX collision group -- it is NOT clear by distance"
            f" (env_spacing={env.cfg.scene.env_spacing}). scene.filter_collisions is currently False;"
            " re-enable it, or give the parked fixture a pose that is actually clear of the"
            " neighbouring tile before disabling collision filtering."
        )

        # -- READ THE MIXTURE FRACTIONS HERE, OFF env.cfg, NOT FROM cfg.params. By the time this
        # __init__ runs (EventManager construction, inside ManagerBasedEnv.__init__, inside
        # gym.make(cfg=env_cfg)), any Hydra CLI override on env_cfg.{classic,low,partial}_goal_prob
        # has already landed on ``env.cfg`` -- ``update_class_from_dict`` runs before ``gym.make`` is
        # ever called. Capturing these as EventTermCfg.params instead (baked at __post_init__ time)
        # is exactly the bug this fixes: __post_init__ runs BEFORE the override lands, so a params
        # dict built there is a snapshot of the pre-override defaults forever. See the module
        # docstring's "THE MIXTURE PROBABILITIES ARE READ AT TERM-CONSTRUCTION TIME" section.
        #
        # READ BEFORE CONSTRUCTING THE COMPOSER BELOW, DELIBERATELY -- see "THE COMPOSER IS
        # CONSTRUCTED LAZILY" in the module docstring: partial_assembly_prob decides whether the
        # (network-dependent) composer is built at all.
        self.classic_goal_prob: float = env.cfg.classic_goal_prob
        self.low_goal_prob: float = env.cfg.low_goal_prob
        self.partial_assembly_prob: float = env.cfg.partial_assembly_prob
        # transport_goal_prob (bead dr-ai1.13): 0.0 unless the config's dataclass declares the field
        # AND a caller has set it positive -- getattr rather than a direct attribute read so this
        # stays inert (and does not AttributeError) against any config that predates this branch.
        self.transport_goal_prob: float = float(getattr(env.cfg, "transport_goal_prob", 0.0))
        assert_episode_mixture_is_sane(
            self.classic_goal_prob, self.low_goal_prob, self.partial_assembly_prob, self.transport_goal_prob
        )
        print(
            f"[dexlift] episode mixture (validated post-override): classic_goal_prob={self.classic_goal_prob:.3f}"
            f" low_goal_prob={self.low_goal_prob:.3f} partial_assembly_prob={self.partial_assembly_prob:.3f}"
            f" transport_goal_prob={self.transport_goal_prob:.3f}",
            flush=True,
        )

        # -- THE COMPOSER IS CONSTRUCTED LAZILY, ONLY WHEN partial_assembly_prob > 0. Its own
        # __init__ (reset_insertive_object_from_partial_assembly_dataset) downloads
        # partial_assemblies.pt over the network the moment it is constructed -- see
        # partial_assembly.SpawnPartialAssembly.__init__ for why constructing it against this same
        # cfg is otherwise safe (it reads only the keys it knows about). With
        # partial_assembly_prob == 0.0, the kind draw in __call__ below (``draw <
        # self.partial_assembly_prob``) can never select EPISODE_KIND_PARTIAL_ASSEMBLY -- draw is
        # sampled from [0, 1), so nothing is ever less than 0.0 -- so self._compose is never called
        # either; constructing it anyway would be a needless, possibly-failing network dependency
        # for a run that will never use it. Kept unconditional whenever the fraction IS positive:
        # that case genuinely needs the dataset, same as the legacy SpawnPartialAssembly always has.
        self._compose = (
            reset_insertive_object_from_partial_assembly_dataset(cfg, env) if self.partial_assembly_prob > 0.0 else None
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        dataset_dir: str,
        insertive_object_cfg: SceneEntityCfg,
        receptive_object_cfg: SceneEntityCfg,
        fixture_pose_range: dict[str, tuple[float, float]],
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        pose_range_b: dict[str, tuple[float, float]] = dict(),
    ) -> None:
        del dataset_dir, fixture_pose_range  # consumed at __init__, see above.
        env_ids_t = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, device=env.device)
        if env_ids_t.numel() == 0:
            return

        # -- draw the kind, ONE draw per env in this reset, and commit it to the shared buffer
        # immediately: MixtureGoalPoseCommand._resample_command reads exactly this, for exactly
        # these env_ids, later in the same CommandManager.reset() call.
        draw = torch.rand(env_ids_t.shape[0], device=env.device)
        kind = torch.full_like(draw, EPISODE_KIND_CLASSIC, dtype=torch.long)
        kind[draw < self.partial_assembly_prob] = EPISODE_KIND_PARTIAL_ASSEMBLY
        low_band = (draw >= self.partial_assembly_prob) & (draw < self.partial_assembly_prob + self.low_goal_prob)
        kind[low_band] = EPISODE_KIND_LOW_GOAL
        transport_lo = self.partial_assembly_prob + self.low_goal_prob
        transport_band = (draw >= transport_lo) & (draw < transport_lo + self.transport_goal_prob)
        kind[transport_band] = EPISODE_KIND_TRANSPORT
        # Whatever's left (draw >= transport_lo + transport_goal_prob) stays EPISODE_KIND_CLASSIC,
        # the fill value -- same convention the three-band version already used.
        _get_episode_kind_buffer(env)[env_ids_t] = kind

        partial_ids = env_ids_t[kind == EPISODE_KIND_PARTIAL_ASSEMBLY]
        normal_ids = env_ids_t[kind != EPISODE_KIND_PARTIAL_ASSEMBLY]

        # CLASSIC and LOW GOAL envs: identical leg spawn, same as before this term existed. Only the
        # GOAL differs between them, decided downstream by MixtureGoalPoseCommand. The fixture is
        # PARKED here, unconditionally, on every single one of these envs every reset -- never left
        # at whatever a PAST reset (on this same env, possibly episodes ago) wrote. See the module
        # docstring; skipping this write is Bug 1.
        if normal_ids.numel() > 0:
            reset_root_state_uniform(env, normal_ids, pose_range, velocity_range, insertive_object_cfg)
            reset_root_state_uniform(env, normal_ids, PARKED_FIXTURE_POSE_RANGE, {}, self.receptive_object_cfg)

            # -- Verify against the REAL, just-written poses, not a hardcoded workspace box: read
            # both bodies' live sim state back and require real clearance. Catches a future edit to
            # either PARKED_FIXTURE_POSE_RANGE or the leg's own pose_range that reintroduces overlap,
            # rather than trusting the two ranges stay disjoint by construction forever.
            leg_xy = self.insertive_object.data.root_pos_w[normal_ids, 0:2]
            fixture_xy = self.receptive_object.data.root_pos_w[normal_ids, 0:2]
            clearance = torch.linalg.vector_norm(leg_xy - fixture_xy, dim=-1)
            bad = clearance <= _PARKED_FIXTURE_MIN_CLEARANCE_M
            assert not bool(bad.any()), (
                f"parked fixture is not clear of the leg spawn region on {int(bad.sum())} env(s) of"
                f" {normal_ids.numel()}; min clearance {clearance.min().item():.3f} m, required >"
                f" {_PARKED_FIXTURE_MIN_CLEARANCE_M} m. env_ids={normal_ids[bad].tolist()}"
            )

        # PARTIAL ASSEMBLY envs: place the fixture, then compose the leg against it -- see
        # partial_assembly.SpawnPartialAssembly.__call__, this is the same two steps on a subset.
        if partial_ids.numel() > 0:
            # Cannot happen given partial_assembly_prob > 0.0 is required for any draw to land here
            # (see __init__'s lazy-construction comment) -- asserted anyway so a future edit that
            # breaks that invariant fails with a clear message instead of AttributeError: NoneType.
            assert self._compose is not None, (
                "partial_ids is non-empty but self._compose was never constructed (partial_assembly_prob"
                f" was {self.partial_assembly_prob} at __init__ time) -- the kind draw and the lazy"
                " composer construction have gone out of sync."
            )
            reset_root_state_uniform(env, partial_ids, self.fixture_pose_range, {}, self.receptive_object_cfg)
            self._compose(env, partial_ids, self._dataset_dir, insertive_object_cfg, receptive_object_cfg, pose_range_b)


class MixtureGoalPoseCommand(TaskStateVisPoseCommand):
    """Goal command that reads the per-env episode kind :class:`MixtureResetObject` just wrote, and
    samples (or pins) the goal accordingly. See the module docstring for the shared-buffer argument.

    Never draws its own randomness for WHICH kind an env is -- only for the CLASSIC/LOW GOAL pose
    values within a kind already decided. Behaviour per kind:

    * CLASSIC -- unmodified ``ObjectUniformPoseCommand._resample_command`` via ``super()``: the
      existing high pos_z band, untouched.
    * LOW GOAL -- the same x/y/roll/pitch/yaw sampling, with pos_z redrawn from
      :data:`LOW_GOAL_POS_Z_RANGE` instead of ``cfg.ranges.pos_z``.
    * PARTIAL ASSEMBLY -- goal pinned to the object's current (just-spawned) world pose, identical
      to ``partial_assembly.GoalAtSpawnPoseCommand._resample_command``.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        # -- PARTIAL-ASSEMBLY GOAL DISPLACEMENT (bead UWLab-nnlv.5). This lives on the COMMAND term,
        # not the event term: the two are different objects, and an earlier revision set it on the
        # event and crashed here with AttributeError the first time a partial episode resampled.
        # Default 0.0 reproduces goal-AT-spawn byte for byte, so no existing config changes.
        #
        # Read off env.cfg for the same reason the mixture probabilities are: by now a Hydra
        # override has landed on env.cfg, whereas anything captured at __post_init__ would be a
        # pre-override snapshot forever.
        self._partial_delta_m: float = float(getattr(env.cfg, "partial_goal_delta_m", 0.0))
        if not -0.200 <= self._partial_delta_m <= 0.025:
            raise ValueError(
                f"partial_goal_delta_m={self._partial_delta_m} is outside [-0.200, +0.025] m. "
                "Positive displaces the goal DEEPER into the bore (ceiling = the 25 mm engaged "
                "span); negative displaces it OUT of the mouth, above it (floor = headroom around "
                "the S2' rung's 20-120 mm band). Same bounds as GoalBelowSpawnPoseCommand."
            )
        self._fixture = None
        self._fixture_local_deep_axis = None
        if self._partial_delta_m != 0.0:
            if "receptive_object" not in env.scene.rigid_objects:
                raise ValueError(
                    "partial_goal_delta_m is set but 'receptive_object' (the fixture) is not in the "
                    "scene, so the bore's deep axis has nothing to be read off. This needs the "
                    "partial-assembly scene (DEXLIFT_PARTIAL_ASSEMBLY / the episode mixture)."
                )
            self._fixture = env.scene["receptive_object"]
            self._fixture_local_deep_axis = torch.tensor([0.0, 0.0, -1.0], device=self.device)
            print(
                f"[dexlift] MIXTURE PARTIAL GOAL DELTA: {self._partial_delta_m * 1000.0:+.2f} mm "
                + (
                    "DEEPER INTO the bore"
                    if self._partial_delta_m > 0
                    else "OUT OF the mouth (ABOVE it)"
                )
                + " from spawn, on the partial-assembly episodes only. delta 0 would pin the goal AT"
                " spawn, which satisfies tracking at t=0 and teaches nothing.",
                flush=True,
            )

        # -- TRANSPORT GOAL BRANCH (bead dr-ai1.13, F43). Read off env.cfg for the same
        # post-override reason as everything above. Default transport_goal_prob is 0.0 (validated
        # by assert_episode_mixture_is_sane above, called from MixtureResetObject.__init__, which
        # runs before this term is constructed -- see EventManager ordering in that module's
        # docstring), so this block is inert -- no banner, no ranges built -- unless a caller has
        # actually raised the fraction above 0. See c3_transport_core's module docstring for why
        # the ranges/tilt/z parsing lives there instead of being re-derived here.
        self._transport_goal_prob: float = float(getattr(env.cfg, "transport_goal_prob", 0.0))
        if self._transport_goal_prob > 0.0:
            tilt = float(getattr(env.cfg, "transport_goal_tilt", c3_transport_core.DEFAULT_TRANSPORT_GOAL_TILT_RAD))
            z_range = tuple(
                float(v)
                for v in getattr(env.cfg, "transport_goal_z", c3_transport_core.DEFAULT_TRANSPORT_GOAL_Z_RANGE_M)
            )
            c3_transport_core.validate_transport_goal_z(*z_range)
            self._transport_ranges = c3_transport_core.transport_goal_ranges(tilt)
            self._transport_z = z_range
            print(
                c3_transport_core.transport_goal_banner(self._transport_goal_prob, tilt, *z_range),
                flush=True,
            )
        else:
            self._transport_ranges = None
            self._transport_z = None

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, device=self.device)
        if env_ids_t.numel() == 0:
            return
        kind = _get_episode_kind_buffer(self._env)[env_ids_t]

        classic_ids = env_ids_t[kind == EPISODE_KIND_CLASSIC]
        low_ids = env_ids_t[kind == EPISODE_KIND_LOW_GOAL]
        partial_ids = env_ids_t[kind == EPISODE_KIND_PARTIAL_ASSEMBLY]
        transport_ids = env_ids_t[kind == EPISODE_KIND_TRANSPORT]

        if classic_ids.numel() > 0:
            super()._resample_command(classic_ids)
        if low_ids.numel() > 0:
            self._resample_low_goal(low_ids)
        if partial_ids.numel() > 0:
            self._resample_goal_at_spawn(partial_ids)
        if transport_ids.numel() > 0:
            self._resample_transport(transport_ids)

    def _resample_low_goal(self, env_ids: torch.Tensor):
        # Same shape as ObjectUniformPoseCommand._resample_command, pos_z swapped for the low band.
        r = torch.empty(env_ids.shape[0], device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*LOW_GOAL_POS_Z_RANGE)
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _resample_goal_at_spawn(self, env_ids: torch.Tensor):
        """Goal for a PARTIAL-ASSEMBLY episode: the leg's spawn pose, optionally DISPLACED along
        the bore's own axis by :attr:`partial_delta_m`.

        WHY THE DISPLACEMENT EXISTS (bead UWLab-nnlv.5). With delta 0 -- the original and still the
        default -- the goal is pinned exactly at the leg's spawn pose, so tracking is SATISFIED AT
        t=0 and the episode carries NO GRADIENT: the objective reduces to "stay where you already
        are". That is not a hypothesis; it is the mechanism behind this mixture's own measured
        result, a finetune that certified 20 POINTS WORSE while a quarter of its episodes were
        teaching nothing. A negative delta displaces the goal back OUT of the mouth, which is the
        S2' rung's actual target and gives those episodes something to learn.

        The sign convention, the axis and the runtime guard all come from
        :func:`~.partial_assembly.live_bore_deep_axis` -- the same function
        :class:`~.partial_assembly.GoalBelowSpawnPoseCommand` uses -- so the two cannot disagree
        about which way "deeper" points.
        """
        # Identical to partial_assembly.GoalAtSpawnPoseCommand._resample_command -- see that
        # class's docstring / the module's "Y3" section for why the goal must be read here (a
        # command hook) rather than written by the spawn event, which CommandManager.reset() would
        # silently overwrite a few lines later in the same reset call.
        object_pos_w = self.object.data.root_pos_w[env_ids]
        object_quat_w = self.object.data.root_quat_w[env_ids]

        goal_pos_w = object_pos_w
        if self._partial_delta_m != 0.0:
            axis_world = live_bore_deep_axis(self._fixture, self._fixture_local_deep_axis, env_ids)
            # Same single expression for both signs, exactly as GoalBelowSpawnPoseCommand does it:
            # axis_world points INTO the bore, so a negative delta adds along -axis_world, i.e. back
            # out of the mouth and above it. Nothing downstream branches on the sign.
            goal_pos_w = object_pos_w + self._partial_delta_m * axis_world

        pos_b, quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            goal_pos_w,
            object_quat_w,
        )
        self.pose_command_b[env_ids, 0:3] = pos_b
        self.pose_command_b[env_ids, 3:7] = quat_b

    def _resample_transport(self, env_ids: torch.Tensor):
        """Goal for a TRANSPORT episode (bead dr-ai1.13, F43): tip-down +- tilt, x/y ANCHORED to
        the object's own just-spawned position, z drawn from the transport height band.

        See ``c3_transport_core``'s module docstring for why the anchor is not a toggle (the
        148-attempts/0-states measurement an independent x/y produced for the near-identical
        GOAL_VERTICAL band) and for the orientation convention (same shape as
        ``goal_mixture.MixedGoalPoseCommand._resample_command``'s ``vertical_anchor_xy == "object"``
        branch, ported here rather than reused because that class upgrades the WHOLE command term
        and is incompatible with the episode mixture -- F38).

        Assumes ``self._transport_ranges`` / ``self._transport_z`` are set -- true whenever any
        env_id can carry :data:`EPISODE_KIND_TRANSPORT`, since both this command and
        :class:`MixtureResetObject` read the identical ``env.cfg.transport_goal_prob`` after the
        same Hydra override point.
        """
        assert self._transport_ranges is not None and self._transport_z is not None, (
            "transport_ids is non-empty but transport_goal_prob was 0.0 (or unset) at this command's"
            " own __init__ time -- the kind draw (MixtureResetObject) and this command have gone"
            " out of sync on env.cfg.transport_goal_prob."
        )
        obj_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            self.object.data.root_pos_w[env_ids],
        )
        self.pose_command_b[env_ids, 0] = obj_pos_b[:, 0]
        self.pose_command_b[env_ids, 1] = obj_pos_b[:, 1]

        r = torch.empty(env_ids.shape[0], device=self.device)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self._transport_z)

        euler_angles = torch.zeros(env_ids.shape[0], 3, device=self.device)
        euler_angles[:, 0].uniform_(*self._transport_ranges.roll)
        euler_angles[:, 1].uniform_(*self._transport_ranges.pitch)
        euler_angles[:, 2].uniform_(*self._transport_ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat


@configclass
class MixtureGoalPoseCommandCfg(TaskStateVisPoseCommandCfg):
    """Config for :class:`MixtureGoalPoseCommand`. Every field means what it means on
    ``TaskStateVisPoseCommandCfg``; only ``class_type`` differs."""

    class_type: type = MixtureGoalPoseCommand


def upgrade_to_episode_mixture(command_cfg: TaskStateVisPoseCommandCfg) -> MixtureGoalPoseCommandCfg:
    """Rebuild an already-configured ``TaskStateVisPoseCommandCfg`` as a mixture goal command.

    Same field-copy idiom as ``partial_assembly.upgrade_to_goal_at_spawn`` /
    ``task_state_vis.upgrade_pose_command_to_task_state_vis`` -- every field of the inherited term
    (sampling ranges, resampling window, asset names, tolerances, markers) is copied rather than
    restated. ``cfg.ranges.pos_z`` stays exactly what it was: it is still read for the CLASSIC
    fraction (via ``super()._resample_command``); only the LOW GOAL fraction ignores it in favour of
    :data:`LOW_GOAL_POS_Z_RANGE`.
    """
    fields = {
        field.name: getattr(command_cfg, field.name)
        for field in dataclasses.fields(command_cfg)
        if field.name != "class_type"
    }
    return MixtureGoalPoseCommandCfg(**fields)
