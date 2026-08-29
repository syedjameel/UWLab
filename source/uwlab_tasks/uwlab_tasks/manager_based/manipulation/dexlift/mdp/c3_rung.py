# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The ISAAC-TOUCHING half of the C3 RUNG stage -- ``RESET_SPEC_V2.md`` sec 1 C3, bead ``dr-ai1.4``:
**C3 = 50% S1 + 50% S_t**, drawn per env, in ONE run.

**Read ``c3_rung_core.py``'s module docstring first.** It carries the whole design argument -- what
S1 and S_t each are, why S_t's peg is HORIZONTAL and needs no spawn change (F51), why this is a
whole-run stage rather than a set of episode-mixture fractions, and the F49 frame rule. This file is
only the wiring, and it deliberately holds no arithmetic and no policy of its own: everything
decidable without a GPU lives in the core module and is unit-tested there
(``source/uwlab_tasks/test/test_c3_rung_stage.py``).

OFF BY DEFAULT. Nothing here runs unless ``DEXRESET_C3_RUNG=1``; see
``dexlift_ur5e_delto_tableleg_env_cfg._apply_c3_rung_stage``, which is the only caller.

WHAT IS REUSED RATHER THAN REIMPLEMENTED, and this is most of the file:

* **S1's spawn** -- ``omnireset.mdp.events.reset_insertive_object_from_partial_assembly_dataset``,
  the same delegate ``partial_assembly.SpawnPartialAssembly`` and ``episode_mixture.MixtureResetObject``
  both already use, preceded by the same fixture placement. The leg spawns pre-inserted, hence
  tip-down.
* **S_t's spawn** -- ``reset_root_state_uniform`` with WHATEVER ``reset_object`` already carried.
  **This is the point: S_t needs NO spawn change** (F51), so the ordinary draw is passed straight
  through and this file adds nothing to it.
* **S1's goal** -- the leg's spawn pose displaced along ``partial_assembly.live_bore_deep_axis``,
  the SAME function ``GoalBelowSpawnPoseCommand`` and the mixture's partial branch use, so all three
  agree about which way "deeper" points and refuse under the same runtime guard.
* **S_t's goal** -- the leg's own pose, pinned. The arithmetic is identical to
  ``GoalAtSpawnPoseCommand._resample_command``; WHEN it is read is not -- see
  :meth:`C3RungGoalPoseCommand._update_command` for the deferred re-pin at the SETTLED pose
  (bead ``dr-ai1.18``).
* **S_t's settle predicate** -- ``held_check_core.SETTLE_STEPS`` for the step floor, imported, plus
  the absolute resting-speed ceiling ``generate_reset_states_policy.py``'s ``--c2_max_resting_speed``
  already establishes. No third settle test is written here.
* **The fixture parking pose** -- ``episode_mixture.PARKED_FIXTURE_POSE_RANGE`` and its clearance
  margin, imported rather than re-picked, together with the ``filter_collisions`` assert that makes
  that pose's real dependency loud (it is clear of the NEIGHBOURING env's geometry by PhysX
  collision-group isolation, NOT by distance -- see that module's own docstring).

WHY THE GOAL IS A COMMAND SUBCLASS AND NOT AN EVENT (``partial_assembly.py``'s "Y3", restated
because it is the trap most likely to be re-fallen into): ``CommandManager.reset()`` resamples the
command AFTER reset-mode events have run, in the same reset call, regardless of
``resampling_time_range``. An event term that wrote the goal would be silently overwritten a few
lines later. Both halves' goals are therefore computed inside ``_resample_command``.

WHY ONE SHARED KIND DECISION, NOT TWO INDEPENDENT DRAWS (``episode_mixture.py``'s argument, and it
transfers exactly): the spawn is an EVENT and the goal is a COMMAND -- two different objects. If
each drew its own coin, an S1 SPAWN could be paired with an S_t GOAL, i.e. a leg seated in the bore
told to stay exactly where it is (a rung that is neither S1 nor S_t and that nothing downstream
expects). :class:`C3RungResetObject` draws once per env per reset and commits the result to a buffer
on ``env``; :class:`C3RungGoalPoseCommand` READS it. The ordering is guaranteed by
``ManagerBasedEnv``: all reset-mode events run before ``CommandManager.reset()``.

MID-EPISODE RESAMPLE IS FATAL TO BOTH HALVES, and the guard is in the staging function, not here.
``CommandManager.compute()`` re-resamples whenever ``time_left <= 0``, independent of episode reset
(the Y6 defect, confirmed in isaaclab source under bead UWLab-xp05.3). A second resample would
rebase EITHER goal onto wherever the leg has been carried to by then -- for S_t that means paying
the policy for holding the leg anywhere at all, which is precisely the rung's content destroyed.
``_apply_c3_rung_stage`` forces ``resampling_time_range`` strictly past ``episode_length_s`` for
exactly this reason, and :meth:`C3RungGoalPoseCommand.__init__` re-asserts it at
manager-construction time -- after any ``--episode_length_s`` override the generator script applies
post-``parse_env_cfg`` -- so a stale value fails loudly instead of silently mis-generating.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp import reset_root_state_uniform
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import (
    reset_insertive_object_from_partial_assembly_dataset,
)

from . import c3_rung_core
from .episode_mixture import _PARKED_FIXTURE_MIN_CLEARANCE_M, PARKED_FIXTURE_POSE_RANGE
from .held_check_core import SETTLE_STEPS
from .partial_assembly import live_bore_deep_axis
from .task_state_vis import TaskStateVisPoseCommand, TaskStateVisPoseCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

__all__ = [
    "C3RungResetObject",
    "C3RungGoalPoseCommand",
    "C3RungGoalPoseCommandCfg",
    "upgrade_to_c3_rung",
]

_C3_KIND_BUFFER_ATTR = "_dexreset_c3_kind"
"""Attribute name of the per-env C3 kind buffer hung off ``env``. DISTINCT from the episode
mixture's ``_dexlift_episode_kind`` on purpose: the two mechanisms are mutually exclusive (the
staging function refuses to install this stage when ``DEXLIFT_EPISODE_MIXTURE=1``), and a shared
attribute name would make a future relaxation of that refusal silently cross-wire two different
kind vocabularies."""


def _get_c3_kind_buffer(env) -> torch.Tensor:
    """Get-or-create the per-env C3 kind buffer.

    Lazy and order-independent, same idiom as ``episode_mixture._get_episode_kind_buffer``:
    whichever of :class:`C3RungResetObject` / :class:`C3RungGoalPoseCommand` is constructed first
    creates it. Initialised to :data:`~.c3_rung_core.C3_KIND_S1`, which is inert -- every env's real
    kind is written by the reset event before the command ever reads it.
    """
    buf = getattr(env, _C3_KIND_BUFFER_ATTR, None)
    if buf is None or buf.shape[0] != env.num_envs:
        buf = torch.full((env.num_envs,), c3_rung_core.C3_KIND_S1, dtype=torch.long, device=env.device)
        setattr(env, _C3_KIND_BUFFER_ATTR, buf)
    return buf


class C3RungResetObject(ManagerTermBase):
    """Reset event: draw each env's C3 half, then spawn accordingly.

    * **S1** -- place the fixture at ``fixture_pose_range``, then compose the leg against it (leg
      pre-inserted, tip-down). Same two steps, same delegate, same atomicity argument as
      ``partial_assembly.SpawnPartialAssembly.__call__`` ("Y2": two separately-ordered EventTerms
      would race; the composer must read THIS episode's fixture pose).
    * **S_t** -- the ORDINARY uniform spawn, i.e. whatever ``reset_object`` already carried, passed
      through untouched, plus the fixture PARKED clear of this env's own table/robot. **No spawn
      change is made for S_t and none should be**: the leg settles horizontal on the table by
      default (F51; measured baseline n=2048 settled, 99.02% lying flat with the tip within 20 mm),
      which IS the rung's precondition.

    The fixture is parked on every S_t env on EVERY reset, not only the first time an env draws S_t:
    PhysX keeps a kinematic body's pose forever until something writes it again, so skipping the
    write leaves a fixture from some past S1 episode sitting in the leg's workspace. That exact bug
    is documented in ``episode_mixture.py``'s "THE FIXTURE IS WRITTEN EVERY RESET" section; this
    term inherits the fix rather than rediscovering it.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.receptive_object_cfg: SceneEntityCfg = cfg.params["receptive_object_cfg"]
        self.receptive_object = env.scene[self.receptive_object_cfg.name]
        self.insertive_object_cfg: SceneEntityCfg = cfg.params["insertive_object_cfg"]
        self.insertive_object = env.scene[self.insertive_object_cfg.name]
        self.fixture_pose_range: dict[str, tuple[float, float]] = cfg.params["fixture_pose_range"]
        self._dataset_dir: str = cfg.params["dataset_dir"]

        # -- The split fraction is baked into cfg.params, NOT read off env.cfg. That is the opposite
        # of what the episode mixture does, and deliberately: the mixture's fractions are Hydra
        # DATACLASS FIELDS meant to be swept from the CLI, so reading them at __post_init__ time
        # would capture a pre-override snapshot (the bug that module's docstring documents at
        # length). This stage's fraction comes from an ENV VAR read directly out of os.environ
        # inside __post_init__, which has no override-ordering window at all -- the same reachability
        # argument every other whole-run toggle in dexlift_ur5e_delto_tableleg_env_cfg.py makes for
        # DEXLIFT_PARTIAL_ASSEMBLY / DEXLIFT_GOAL_BELOW_SPAWN_MM / DEXRESET_C1_HAND.
        self.s1_fraction: float = float(cfg.params["s1_fraction"])
        c3_rung_core.validate_s1_fraction(self.s1_fraction)

        # -- PARKED_FIXTURE_POSE_RANGE is safe by PhysX collision-GROUP isolation between env
        # replicas, not by distance -- at this scene's env_spacing the parking x reaches well into
        # the neighbouring env's own tile. Assert the dependency rather than rely on it silently,
        # exactly as MixtureResetObject does; the pose is imported from that module, so the
        # precondition must be imported with it.
        assert env.cfg.scene.filter_collisions, (
            "C3RungResetObject parks the fixture on S_t envs (episode_mixture.PARKED_FIXTURE_POSE_RANGE)"
            " at a pose that is only clear of the neighbouring env's geometry because"
            " InteractiveSceneCfg.filter_collisions puts every env replica in its own PhysX collision"
            f" group -- it is NOT clear by distance (env_spacing={env.cfg.scene.env_spacing})."
            " scene.filter_collisions is currently False; re-enable it, or give the parked fixture a"
            " pose that is actually clear of the neighbouring tile before disabling collision"
            " filtering."
        )

        # -- The composer downloads partial_assemblies.pt over the network the moment it is
        # constructed. Built unconditionally here, unlike the mixture's lazy construction: an
        # s1_fraction of 0 is a deliberate pure-S_t characterisation run and is legal
        # (validate_s1_fraction allows it), but the S1 half is this stage's reason to exist, so
        # paying the dataset dependency up front and failing loudly at construction beats failing at
        # the first reset that happens to draw S1.
        self._compose = reset_insertive_object_from_partial_assembly_dataset(cfg, env)

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
        s1_fraction: float,
        pose_range_b: dict[str, tuple[float, float]] = dict(),
    ) -> None:
        del dataset_dir, fixture_pose_range, s1_fraction  # consumed at __init__, see above.
        env_ids_t = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, device=env.device)
        if env_ids_t.numel() == 0:
            return

        # -- Draw the half, ONE draw per env, and commit it immediately: C3RungGoalPoseCommand reads
        # exactly this, for exactly these env_ids, later in the same reset. The comparison is the
        # tensor form of c3_rung_core.c3_kind_for_draw -- half-open the same way, so a fraction of
        # 0.0 yields no S1 envs (nothing is < 0.0) and a fraction of 1.0 yields no S_t envs.
        draw = torch.rand(env_ids_t.shape[0], device=env.device)
        kind = torch.full_like(draw, c3_rung_core.C3_KIND_ST, dtype=torch.long)
        kind[draw < self.s1_fraction] = c3_rung_core.C3_KIND_S1
        _get_c3_kind_buffer(env)[env_ids_t] = kind

        s1_ids = env_ids_t[kind == c3_rung_core.C3_KIND_S1]
        st_ids = env_ids_t[kind == c3_rung_core.C3_KIND_ST]

        # -- S_t: the ordinary spawn, untouched, plus the fixture parked out of the way.
        if st_ids.numel() > 0:
            reset_root_state_uniform(env, st_ids, pose_range, velocity_range, insertive_object_cfg)
            reset_root_state_uniform(env, st_ids, PARKED_FIXTURE_POSE_RANGE, {}, self.receptive_object_cfg)

            # Verify against the REAL, just-written poses rather than trusting two ranges to stay
            # disjoint by construction forever -- same check, same margin, as MixtureResetObject.
            leg_xy = self.insertive_object.data.root_pos_w[st_ids, 0:2]
            fixture_xy = self.receptive_object.data.root_pos_w[st_ids, 0:2]
            clearance = torch.linalg.vector_norm(leg_xy - fixture_xy, dim=-1)
            bad = clearance <= _PARKED_FIXTURE_MIN_CLEARANCE_M
            assert not bool(bad.any()), (
                f"parked fixture is not clear of the S_t leg spawn region on {int(bad.sum())} env(s)"
                f" of {st_ids.numel()}; min clearance {clearance.min().item():.3f} m, required >"
                f" {_PARKED_FIXTURE_MIN_CLEARANCE_M} m. env_ids={st_ids[bad].tolist()}"
            )

        # -- S1: place the fixture, then compose the leg against it. Order matters (see the class
        # docstring's "Y2" reference); write_root_pose_to_sim updates the cached buffers
        # immediately, so the composer reads THIS episode's fixture pose.
        if s1_ids.numel() > 0:
            reset_root_state_uniform(env, s1_ids, self.fixture_pose_range, {}, self.receptive_object_cfg)
            self._compose(env, s1_ids, self._dataset_dir, insertive_object_cfg, receptive_object_cfg, pose_range_b)


class C3RungGoalPoseCommand(TaskStateVisPoseCommand):
    """Goal command that reads the per-env C3 half :class:`C3RungResetObject` just wrote.

    * **S1** -- goal = the leg's spawn pose displaced ``s1_goal_delta_m`` along the bore's own deep
      axis, ORIENTATION UNCHANGED. Tip-down is inherited from the spawn rather than commanded, which
      is what makes the rung a depth task about the mating frame and never a reorientation task.
    * **S_t** -- goal = the leg's OWN pose, position and orientation, ZERO delta, re-pinned once
      at the SETTLED pose (:meth:`_update_command`) rather than left at the mid-air spawn pose.

    Both are computed by the same expression with a per-env delta (``0`` for S_t), which is the same
    single-expression form ``GoalBelowSpawnPoseCommand`` and
    ``MixtureGoalPoseCommand._resample_goal_at_spawn`` already use -- nothing branches on the sign,
    and S_t is not a special case of the arithmetic, only of the number.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self._s1_goal_delta_m: float = float(cfg.s1_goal_delta_m)
        c3_rung_core.validate_s1_goal_delta_mm(self._s1_goal_delta_m * 1000.0)

        # -- S_t's DEFERRED GOAL RE-PIN (bead dr-ai1.18, V2_C3_DESIGN.md sec 7). See
        # :meth:`_update_command`. SETTLE_STEPS is IMPORTED from held_check_core, never restated
        # here -- c3_rung_core.st_should_repin takes min_steps as a required argument precisely so
        # this file is the only place that names the source.
        self._st_settle_speed_mps: float = float(cfg.st_settle_speed_mps)
        c3_rung_core.validate_st_settle_speed(self._st_settle_speed_mps)
        self._st_settle_ang_speed_rad_s: float = float(cfg.st_settle_ang_speed_rad_s)
        c3_rung_core.validate_st_settle_ang_speed(self._st_settle_ang_speed_rad_s)
        self._st_settle_min_steps: int = int(
            SETTLE_STEPS if cfg.st_settle_min_steps is None else cfg.st_settle_min_steps
        )
        c3_rung_core.validate_st_settle_min_steps(self._st_settle_min_steps)
        # Per-env latch: True from the reset that drew S_t until the re-pin fires. S1 envs are
        # never armed, so their goal keeps the reset-time pin (their leg spawns already seated in
        # the fixture; there is no settling transient to wait out, and re-pinning them at rest
        # would silently absorb any slump into the target).
        self._st_awaiting_repin = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if "receptive_object" not in env.scene.rigid_objects:
            raise ValueError(
                "C3RungGoalPoseCommand needs 'receptive_object' (the fixture) in the scene: the S1"
                " half reads the bore's own deep axis off it. _apply_c3_rung_stage adds it; this"
                " command was installed some other way."
            )
        self._fixture = env.scene["receptive_object"]
        self._fixture_local_deep_axis = torch.tensor([0.0, 0.0, -1.0], device=self.device)

        # -- Y6 RE-ASSERT AT MANAGER-CONSTRUCTION TIME. _apply_c3_rung_stage already forced this at
        # __post_init__, but generate_reset_states_policy.py's --episode_length_s override lands
        # AFTER that and would go stale against a number fixed only there. A second resample would
        # rebase the goal onto wherever the leg has been carried to mid-episode, which destroys S_t
        # outright (it would reward holding the leg anywhere) and silently loosens S1. See the module
        # docstring.
        resample_lo = float(self.cfg.resampling_time_range[0])
        episode_length_s = float(env.cfg.episode_length_s)
        assert resample_lo > episode_length_s, (
            f"C3RungGoalPoseCommand: resampling_time_range={tuple(self.cfg.resampling_time_range)}"
            f" does not clear episode_length_s={episode_length_s}s, so the goal would be resampled"
            " MID-EPISODE and rebased onto wherever the leg has been carried to by then. That"
            " destroys S_t (the goal would follow the leg) and loosens S1. Raise"
            " resampling_time_range past episode_length_s -- _apply_c3_rung_stage does this, but an"
            " --episode_length_s override applied after parse_env_cfg can outrun it."
        )

    @property
    def goal_is_final(self) -> torch.Tensor:
        """Per-env bool: is this env's commanded goal FINAL, i.e. will it not change again this
        episode? ``(num_envs,)``, on this term's device. THE PUBLIC READ FOR EVERY OTHER LAYER --
        do not reach into ``_st_awaiting_repin``, and do not recompute
        ``c3_rung_core.st_should_repin``'s conditions to derive this.

        WHY THIS IS PUBLIC (team-lead decision 2026-08-29). The latch behind it grew three consumers
        in one day -- this command's own ``_update_command``, the generation-side acceptance addon,
        and the GPU smoke's phase 2 -- and two of them were either reaching into a private field or
        evaluating a SECOND COPY of the settle conditions. Two layers independently computing "is it
        settled yet" is the exact shape that produced the pre-settle capture bug twice today, in two
        different agents' code, from one spec. This is the same fix as importing
        ``held_check_core.SETTLE_STEPS`` instead of restating 60, one level up: the CONDITION becomes
        single-source, not just its constants.

        A DERIVED VIEW, NOT A SECOND BUFFER. It is exactly ``~self._st_awaiting_repin`` -- there is
        one piece of state and this is its negation, computed on read, so the two cannot drift. It
        is a property with no setter: callers cannot write it, and the returned tensor is freshly
        derived, so mutating it cannot corrupt the latch.

        SEMANTICS PER RUNG HALF -- READ THIS BEFORE USING IT AS AN ACCEPTANCE GATE, because a caller
        who gets it backwards silently rejects or accepts an ENTIRE rung:

        * **S_t** -- ``False`` from reset until the deferred re-pin fires, then ``True`` for the rest
          of the episode. While ``False`` the commanded goal is the PROVISIONAL mid-air spawn pose
          and must NOT be trusted: accepting against it scores gravity as policy error.
        * **S1** -- ``True`` from the moment the episode is armed, and never ``False``. S1 is never
          re-pinned because its goal is already correct at reset (the leg spawns pre-inserted, and
          the goal is that pose displaced along the bore axis), so there is nothing to wait for.
          **Do not read S1's ``True`` as "the re-pin has happened" -- read the whole flag as "the
          goal is trustworthy", which is why it is named for the settled state rather than for the
          re-pin event.** Conversely, never read ``False`` as "this env is S1".

        BEFORE THE FIRST RESET the buffer is all-``False``-awaiting, so this reads ``True`` for every
        env. That window is not meaningful -- no goal has been sampled yet -- and no consumer should
        read a command before the first reset anyway. The value is meaningful from the first reset
        onward.

        The kind itself is NOT available through this flag by design; a caller that needs to know
        which half an env drew should read the kind buffer rather than inferring it from timing.
        """
        return ~self._st_awaiting_repin

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, device=self.device)
        if env_ids_t.numel() == 0:
            return
        kind = _get_c3_kind_buffer(self._env)[env_ids_t]

        s1_ids = env_ids_t[kind == c3_rung_core.C3_KIND_S1]
        st_ids = env_ids_t[kind == c3_rung_core.C3_KIND_ST]

        if st_ids.numel() > 0:
            # PROVISIONAL for S_t -- the leg is still airborne at this instant. This write keeps the
            # command well-defined for the first ~SETTLE_STEPS steps (an env's goal is read every
            # step by observations and rewards, so it cannot be left unset); _update_command
            # replaces it with the SETTLED pose as soon as the leg comes to rest. See that method.
            self._pin_goal_at_object_pose(st_ids, delta_m=0.0)
        if s1_ids.numel() > 0:
            self._pin_goal_at_object_pose(s1_ids, delta_m=self._s1_goal_delta_m)

        # Arm the deferred re-pin for exactly the S_t envs resetting in THIS call, and disarm the
        # S1 ones. Written for all env_ids first so an env that switched S1 -> S_t (or back) between
        # episodes cannot inherit the previous episode's latch.
        self._st_awaiting_repin[env_ids_t] = False
        if st_ids.numel() > 0:
            self._st_awaiting_repin[st_ids] = True

    def _update_command(self):
        """Re-pin S_t's goal ONCE, at the settled pose (bead dr-ai1.18, V2_C3_DESIGN.md sec 7).

        THE HOOK ALREADY EXISTED; NOTHING WAS ADDED TO THE FRAMEWORK. ``CommandTerm.compute(dt)``
        calls ``_update_metrics()``, decrements ``time_left``, resamples if due, then calls
        ``_update_command()`` -- every step, for every env (isaaclab ``managers/command_manager.py``,
        ``compute``). ``ObjectUniformPoseCommand._update_command`` is a bare ``pass`` and
        ``TaskStateVisPoseCommand`` does not override it (it overrides only ``_update_metrics``), so
        this is an unused, already-wired, post-reset per-step hook on the command term this stage
        already owns. No new event term, no new manager, no config plumbing, no change to any file
        outside this stage.

        WHY THE GOAL MOVES AT ALL, restated because a reader will reasonably flinch at a command
        that rewrites itself mid-episode: S_t's rung definition is "the target is exactly where the
        leg is". At reset the leg is mid-air in a randomized orientation; where it IS, in the sense
        the rung means, is only determined once it has settled. Pinning at reset does not merely
        add position error (that is bounded by the ~50 mm drop clamp) -- it can place the goal up to
        ~90 deg from the leg's resting orientation and so command a REORIENTATION, which is exactly
        what S_t must never ask for. The re-pin fires at most once per episode and the goal is
        constant from then on, so nothing downstream sees a moving target after settle.

        ONE STEP OF SKEW, stated rather than hidden: ``compute`` calls ``_update_metrics()`` BEFORE
        ``_update_command()``, so on the single step the re-pin fires, that step's metrics were
        computed against the provisional goal. Every subsequent step uses the settled goal.

        S1 IS NEVER RE-PINNED -- see ``__init__``'s latch comment.
        """
        super()._update_command()

        # Cheap early-out: no S_t env is waiting, which is every step after each episode's re-pin
        # and every step of an all-S1 configuration.
        if not bool(self._st_awaiting_repin.any()):
            return

        # ABSOLUTE object linear speed in the world frame -- NOT held_check's relative
        # |v_obj - v_palm| (a leg carried steadily by the hand passes that and is not at rest) and
        # NOT held_check's own `settled`, which is a pure step count. Both halves are used here:
        # the step floor below and this speed ceiling. See c3_rung_core.DEFAULT_ST_SETTLE_SPEED_MPS
        # for the provenance of 0.05 m/s (--c2_max_resting_speed).
        speed = torch.linalg.vector_norm(self.object.data.root_lin_vel_w, dim=-1)
        # ANGULAR speed too (team-lead decision 2026-08-29). NOT redundant with the linear term: a
        # leg pivoting on a corner or spinning about a vertical axis has a near-zero LINEAR speed
        # while its orientation is still changing -- and orientation is the quantity this whole
        # re-pin exists to get right, so a linear-only predicate can pin a wrong orientation at a
        # moment of zero linear speed. See c3_rung_core.DEFAULT_ST_SETTLE_ANG_SPEED_RAD_S, including
        # why its 0.05 rad/s (F50/F51) and the linear 0.05 m/s (--c2_max_resting_speed) come from
        # DIFFERENT sources on purpose and must not be "unified".
        ang_speed = torch.linalg.vector_norm(self.object.data.root_ang_vel_w, dim=-1)
        steps = self._env.episode_length_buf

        # Tensor form of c3_rung_core.st_should_repin -- same three conditions, same order. The
        # scalar version is what the unit test proves; this is the batched expression of it.
        ready = (
            self._st_awaiting_repin
            & (steps > self._st_settle_min_steps)
            & (speed <= self._st_settle_speed_mps)
            & (ang_speed <= self._st_settle_ang_speed_rad_s)
        )
        ready_ids = ready.nonzero().flatten()
        if ready_ids.numel() == 0:
            return

        self._pin_goal_at_object_pose(ready_ids, delta_m=0.0)
        self._st_awaiting_repin[ready_ids] = False

    def _pin_goal_at_object_pose(self, env_ids: torch.Tensor, *, delta_m: float):
        """Goal = the object's CURRENT world pose, optionally displaced ``delta_m`` along the bore's
        deep axis. ``delta_m == 0.0`` is S_t (goal exactly at the leg's own pose, no reorientation);
        ``delta_m != 0.0`` is S1.

        ``delta_m`` is keyword-only so a call site cannot pass the displacement positionally and
        silently attach it to the wrong half -- the two halves differ by nothing else.

        Reads the object's pose HERE, in a command hook, not in the spawn event: by the time
        ``CommandManager.reset()`` runs, this episode's reset-mode events have already written it,
        and anything the event wrote to the command would be overwritten right here anyway (the "Y3"
        argument in ``partial_assembly.py``).
        """
        object_pos_w = self.object.data.root_pos_w[env_ids]
        object_quat_w = self.object.data.root_quat_w[env_ids]

        goal_pos_w = object_pos_w
        if delta_m != 0.0:
            # Same single expression for both signs, and the axis comes from the FIXTURE's live
            # orientation via the shared helper, so this cannot disagree with
            # GoalBelowSpawnPoseCommand about which way "deeper" points. The tensor form of
            # c3_rung_core.s1_goal_position.
            axis_world = live_bore_deep_axis(self._fixture, self._fixture_local_deep_axis, env_ids)
            goal_pos_w = object_pos_w + delta_m * axis_world

        # The goal QUATERNION is the object's own, for BOTH halves -- c3_rung_core.s1_goal_orientation
        # / st_goal_pose. Neither half commands a reorientation.
        pos_b, quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            goal_pos_w,
            object_quat_w,
        )
        self.pose_command_b[env_ids, 0:3] = pos_b
        self.pose_command_b[env_ids, 3:7] = quat_b


@configclass
class C3RungGoalPoseCommandCfg(TaskStateVisPoseCommandCfg):
    """Config for :class:`C3RungGoalPoseCommand`. Every field means what it means on
    ``TaskStateVisPoseCommandCfg``; ``class_type`` and ``s1_goal_delta_m`` are the only additions."""

    class_type: type = C3RungGoalPoseCommand

    s1_goal_delta_m: float = c3_rung_core.DEFAULT_S1_GOAL_DELTA_MM / 1000.0
    """Signed metres the S1 goal is displaced DEEPER along the bore axis from the leg's spawn pose.
    Ignored by the S_t half, which is pinned at zero delta by definition. See
    ``c3_rung_core.DEFAULT_S1_GOAL_DELTA_MM`` for where the default comes from and why it is a
    shaping device rather than a target."""

    st_settle_speed_mps: float = c3_rung_core.DEFAULT_ST_SETTLE_SPEED_MPS
    """m/s ceiling on the object's ABSOLUTE linear speed for S_t's deferred goal re-pin. See
    ``c3_rung_core.DEFAULT_ST_SETTLE_SPEED_MPS`` for the provenance of 0.05 and for why
    ``held_check``'s ``comove_speed_thresh`` is a different quantity that cannot be used here."""

    st_settle_ang_speed_rad_s: float = c3_rung_core.DEFAULT_ST_SETTLE_ANG_SPEED_RAD_S
    """rad/s ceiling on the object's ABSOLUTE angular speed for the same re-pin. Sourced from
    F50/F51's settled pair, unlike the linear bound -- see
    ``c3_rung_core.DEFAULT_ST_SETTLE_ANG_SPEED_RAD_S`` for why that divergence is deliberate."""

    st_settle_min_steps: int | None = None
    """Minimum steps since reset before S_t's re-pin may fire. ``None`` means
    ``held_check_core.SETTLE_STEPS``, resolved in :meth:`C3RungGoalPoseCommand.__init__` -- the
    number is not copied into this file."""


def upgrade_to_c3_rung(
    command_cfg: TaskStateVisPoseCommandCfg,
    s1_goal_delta_m: float,
    st_settle_speed_mps: float = c3_rung_core.DEFAULT_ST_SETTLE_SPEED_MPS,
    st_settle_ang_speed_rad_s: float = c3_rung_core.DEFAULT_ST_SETTLE_ANG_SPEED_RAD_S,
    st_settle_min_steps: int | None = None,
) -> C3RungGoalPoseCommandCfg:
    """Rebuild an already-configured ``TaskStateVisPoseCommandCfg`` as a C3 rung goal command.

    Same field-copy idiom as ``partial_assembly.upgrade_to_goal_at_spawn`` /
    ``episode_mixture.upgrade_to_episode_mixture`` -- every field of the inherited term (resampling
    window, asset names, tolerances, markers) is copied rather than restated, so nothing here can
    drift from what ``_bind_task_state_visualization`` already built. The sampling ranges come along
    and become INERT: neither half of C3 reads ``cfg.ranges``, because both goals are derived from
    the leg's own pose. They stay in the cfg as an honest record of what a non-C3 run would have
    sampled.
    """
    # ``s1_goal_delta_m`` is excluded alongside ``class_type`` so this is IDEMPOTENT: applied to a
    # cfg that is already a C3RungGoalPoseCommandCfg it re-stamps the delta instead of raising
    # "got multiple values for keyword argument". Nothing calls it twice today; this costs one
    # clause and removes a failure mode from a future edit.
    fields = {
        field.name: getattr(command_cfg, field.name)
        for field in dataclasses.fields(command_cfg)
        if field.name
        not in (
            "class_type",
            "s1_goal_delta_m",
            "st_settle_speed_mps",
            "st_settle_ang_speed_rad_s",
            "st_settle_min_steps",
        )
    }
    return C3RungGoalPoseCommandCfg(
        **fields,
        s1_goal_delta_m=s1_goal_delta_m,
        st_settle_speed_mps=st_settle_speed_mps,
        st_settle_ang_speed_rad_s=st_settle_ang_speed_rad_s,
        st_settle_min_steps=st_settle_min_steps,
    )
