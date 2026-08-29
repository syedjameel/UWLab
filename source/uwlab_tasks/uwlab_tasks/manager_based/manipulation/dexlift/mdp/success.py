# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The one THRESHOLDED success test of the dexsuite lift/reorient family, written once.

WHAT SUCCESS IS IN THIS TASK, and what it is not. There is no success TERMINATION term: the
dexsuite terminations are ``time_out``, ``object_out_of_bound`` and ``abnormal_robot`` (the last of
which this task deletes as unreachable). There IS a reward term literally named ``success``, and it
is NOT a success test -- on this env it resolves to :func:`..mdp.rewards.success_reward`, a per-step
tanh on the current pose error, additionally gated on fingertip contact. It has no threshold; a
number computed from it would look like a success rate and would not be one.

The only thresholded test is the one the ADR curriculum promotes on, in
``isaaclab_tasks/.../dexsuite/mdp/curriculums.py``::

    move_up = (pos_dist < pos_tol) & (rot_dist < rot_tol) if rot_tol else pos_dist < pos_tol

That rule and the distances it consumes are this module's :func:`goal_pose_error` and
:func:`within_success_tolerance`. Everything downstream -- curriculum promotion, the evaluation
harness -- calls those two functions rather than restating the rule.

WHY A FORK OF THE UPSTREAM SCHEDULER EXISTS HERE. The predicate above is not importable: it is four
lines in the middle of ``DifficultyScheduler.__call__``, interleaved with the promotion bookkeeping.
The vendored IsaacLab tree is outside UWLab's git and ``uwlab.sh`` recreates it, so an edit there is
undone by the next setup run (the same reasoning that put ``Ur5eDeltoAdrCurriculumCfg`` in this
package). :class:`SharedPredicateDifficultyScheduler` is therefore a subclass that keeps upstream's
bookkeeping verbatim and takes its predicate from here, and the dexlift UR5e+DELTO curriculum config
binds it in place of upstream's class.

THE COPY IS GUARDED, because a copy nobody checks is the defect this module exists to remove. The
subclass's constructor fingerprints upstream's ``__call__`` (its AST, so comments and formatting do
not matter) and raises if it no longer matches what was forked. An upstream revision that changes
either the predicate or the promotion arithmetic then fails loudly at environment construction,
before a training step runs, instead of leaving two rules that quietly disagree.

TOLERANCES ARE NEVER RESTATED EITHER. :func:`resolve_adr_success_spec` reads them out of the LIVE
curriculum term -- ``pos_tol``/``rot_tol`` as the scheduler will actually be called with them -- and
falls back, for any key the config leaves unset, to the scheduler's OWN signature default rather
than to a literal typed here. dexsuite derives ``pos_tol = rewards.success.params["pos_std"] / 2``
in ``DexsuiteReorientEnvCfg.__post_init__`` and the Lift subclass then sets ``rot_tol = None``, so
the resolved values follow whatever those configs do.

EPISODE-LEVEL SUCCESS IS AN EVALUATION PROTOCOL, NOT A PROPERTY OF THE PREDICATE. The predicate is
instantaneous: it answers "is the object within tolerance of the goal at this instant". The ADR
scheduler samples it exactly once per episode, at the terminal state, because
``ManagerBasedRLEnv._reset_idx`` calls ``curriculum_manager.compute`` before it resets the scene.
The certification harness instead samples it EVERY step and takes the sticky OR over the episode --
"reached the goal at any point" -- which is the protocol the reference figure was measured under.
:class:`EpisodeSuccessProbe` is what makes that sampling possible; see its docstring for why an
evaluator cannot simply read the state after ``env.step`` returns.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import pathlib
import torch
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error
from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.curriculums import DifficultyScheduler

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "ADR_TERM_NAME",
    "AdrSuccessSpec",
    "CUMULATIVE_SUCCESS_RATE_KEY",
    "DEXSUITE_RECORDER_ORIENTATION_THRESHOLD",
    "DEXSUITE_RECORDER_POSITION_THRESHOLD",
    "EPISODE_SUCCESS_RATE_KEY",
    "EpisodeSuccessProbe",
    "EpisodeSuccessRateLogger",
    "GOAL_COMMAND_NAME",
    "RL_GAMES_EPISODE_LOG_PREFIX",
    "SUCCESS_PROBE_TERM_NAME",
    "SUCCESS_RATE_LOG_TERM_NAME",
    "SharedPredicateDifficultyScheduler",
    "adr_param",
    "assert_upstream_predicate_unchanged",
    "goal_pose_error",
    "matches_certified_recorder_predicate",
    "resolve_adr_success_spec",
    "success_probe_term_cfg",
    "success_rate_log_term_cfg",
    "upstream_predicate_fingerprint",
    "within_success_tolerance",
]


GOAL_COMMAND_NAME = "object_pose"
"""Command term the goal pose comes from. A literal inside upstream's scheduler, named here once."""

ADR_TERM_NAME = "adr"
"""Curriculum term that owns the success tolerances. dexsuite's own name for it."""

SUCCESS_PROBE_TERM_NAME = "adr_success_probe"
"""Name the evaluation harness registers :class:`EpisodeSuccessProbe` under.

Deliberately not ``success``: nothing in this task family may end an episode on success, and a term
called ``success`` in a ``TerminationsCfg`` reads like one that does.
"""

SUCCESS_RATE_LOG_TERM_NAME = "success_rate_log"
"""Name a TRAINING config registers :class:`EpisodeSuccessRateLogger` under.

Also, unavoidably, the tail of one more logged series: ``TerminationManager.reset`` publishes
``Episode_Termination/<term name>`` for every term it owns, so this one contributes a flat 0.0
under ``Episode_Termination/success_rate_log``. That series is the term declaring, every
iteration, that it never ended an episode -- which is the property the class is built around.
"""

RL_GAMES_EPISODE_LOG_PREFIX = "Episode/"
"""Prefix rl_games' ``IsaacAlgoObserver`` puts in front of every ``extras["log"]`` key, and which
rsl_rl does NOT -- so this package has to add it back by hand to land on the same wandb tag.

The two trainers disagree about who owns the namespace, and the disagreement is the entire reason
a metric that is the same quantity shows up as two unrelated series in wandb:

* rl_games (``rl_games/common/algo_observer.py``, ``IsaacAlgoObserver.after_print_stats``)::

      self.writer.add_scalar("Episode/" + key, value, epoch_num)

  Every key the environment logs is re-namespaced under ``Episode/``. That is where the certified
  runs' ``Episode/Episode_Reward/success``, ``Episode/Curriculum/adr`` and
  ``Episode/Success_Rate/Episode/success_rate`` come from -- the environment emitted
  ``Success_Rate/Episode/success_rate`` and the observer prefixed it.

* rsl_rl (``rsl_rl/runners/on_policy_runner.py``, ``OnPolicyRunner.log``)::

      if "/" in key:
          self.writer.add_scalar(key, value, locs["it"])
      else:
          self.writer.add_scalar("Episode/" + key, value, locs["it"])

  A key that already contains a slash is passed through VERBATIM. Every IsaacLab log key contains
  one, so under rsl_rl the prefix is never applied and the same environment produces
  ``Episode_Reward/success`` and ``Curriculum/adr``.

So for an rsl_rl run to write the tag the rl_games lineage wrote, the environment must emit the
prefix itself. That is what :class:`EpisodeSuccessRateLogger` does, and it is why its
``log_key_prefix`` parameter exists rather than being hardcoded: a config that trains this task
with rl_games must set it to ``""``, or the observer above would double it into
``Episode/Episode/Success_Rate/...``. No dexlift task registers an rl_games entry point today
(``dexlift/__init__.py`` registers ``rsl_rl_cfg_entry_point`` only), which is why the default is
the rsl_rl-correct value.
"""

EPISODE_SUCCESS_RATE_KEY = "Success_Rate/Episode/success_rate"
"""Per-reset-batch success rate, under the name dexsuite's own recorder logged it as.

Verbatim from ``DexsuiteSuccessRecorder.record_pre_reset`` in the reference tree
(``IsaacLabDexterous/source/isaaclab_tasks/.../dexsuite/mdp/recorders.py``). Copying the string is
the point: a chart that plots this key shows the certified rl_games runs and our rsl_rl runs as
two lines of one series instead of two series.
"""

CUMULATIVE_SUCCESS_RATE_KEY = "Success_Rate/Cumulative/success_rate"
"""Run-to-date success rate over every episode the run has finished. Also the recorder's name."""

DEXSUITE_RECORDER_POSITION_THRESHOLD = 0.05
"""``DexsuiteSuccessRecorderCfg.position_threshold``. A LITERAL in the reference recorder."""

DEXSUITE_RECORDER_ORIENTATION_THRESHOLD = 0.5
"""``DexsuiteSuccessRecorderCfg.orientation_threshold``, applied only when the goal command is not
``position_only``. A literal there too, and on the reorientation task it disagrees with the scored
rotation tolerance of ``rot_std / 2`` = 0.25 -- see :func:`matches_certified_recorder_predicate`.
"""

UPSTREAM_PREDICATE_FINGERPRINT = "1e8216838f3d89614abddba21fec42e49d5f20b4067fa7dc66e1057776e1acd5"
"""AST fingerprint of the upstream ``DifficultyScheduler.__call__`` this package's subclass was forked from.

Recomputed by :func:`upstream_predicate_fingerprint`. It covers the WHOLE method -- predicate and
promotion arithmetic -- because the subclass reproduces the whole method. It is taken over the
unparsed AST, so upstream reformatting or a comment change does not trip it and a changed
expression does. Verified identical under CPython 3.11 and 3.12.
"""


def goal_pose_error(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | slice,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    command_name: str = GOAL_COMMAND_NAME,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distance of the object from its commanded pose, as the ADR scheduler measures it.

    The command is expressed in the ROBOT ROOT frame and is composed with the robot's measured root
    pose, not with the env origin -- which is why ``asset_cfg`` names the robot in a function about
    the object. See the base-frame note in ``dexlift_ur5e_delto_env_cfg``: the two frames agree only
    while ``events.reset_root`` pins the root to the pose the config declares.

    Args:
        env: the environment.
        env_ids: environments to measure. ``slice(None)`` for all of them.
        asset_cfg: the robot whose root frame the command is resolved in.
        object_cfg: the manipulated object.
        command_name: the pose command term.

    Returns:
        ``(pos_dist, rot_dist)`` in metres and radians, both of shape ``(len(env_ids),)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, des_quat_w = combine_frame_transforms(
        asset.data.root_pos_w[env_ids], asset.data.root_quat_w[env_ids], command[env_ids, :3], command[env_ids, 3:7]
    )
    pos_err, rot_err = compute_pose_error(
        des_pos_w, des_quat_w, obj.data.root_pos_w[env_ids], obj.data.root_quat_w[env_ids]
    )
    return torch.norm(pos_err, dim=1), torch.norm(rot_err, dim=1)


def within_success_tolerance(
    pos_dist: torch.Tensor, rot_dist: torch.Tensor, pos_tol: float, rot_tol: float | None
) -> torch.Tensor:
    """THE success predicate. Upstream's ``move_up`` expression, unchanged.

    ``rot_tol`` is tested for TRUTHINESS, exactly as upstream tests it, so ``None`` and ``0.0`` both
    drop the orientation gate. That is worth stating because 0.0 dropping the gate is the opposite
    of what "tolerance zero" reads like, and because the task-state marker's own copy of this rule
    tests ``position_only`` instead -- ``_bind_task_state_visualization`` rejects a non-positive
    ``rot_tol`` at construction time so the two cannot be handed a value they would disagree on.
    """
    return (pos_dist < pos_tol) & (rot_dist < rot_tol) if rot_tol else pos_dist < pos_tol


def upstream_predicate_fingerprint() -> str:
    """SHA-256 over the AST of upstream's ``DifficultyScheduler.__call__``.

    Reads the source file rather than the imported object so the hash is of what a maintainer would
    diff. ``ast.unparse`` normalizes formatting and drops comments; what remains is the code.
    """
    source_file = inspect.getsourcefile(DifficultyScheduler)
    if source_file is None:  # pragma: no cover - only if isaaclab ships without sources
        raise RuntimeError("Cannot locate the source of DifficultyScheduler to fingerprint it.")
    tree = ast.parse(pathlib.Path(source_file).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == DifficultyScheduler.__name__:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "__call__":
                    return hashlib.sha256(ast.unparse(member).encode("utf-8")).hexdigest()
    raise RuntimeError(f"No {DifficultyScheduler.__name__}.__call__ found in {source_file}.")


def assert_upstream_predicate_unchanged() -> None:
    """Fail loudly if the forked scheduler is no longer a faithful copy of upstream's.

    Called from :class:`SharedPredicateDifficultyScheduler`'s constructor, i.e. once per environment
    build, before any step runs.
    """
    actual = upstream_predicate_fingerprint()
    if actual != UPSTREAM_PREDICATE_FINGERPRINT:
        raise RuntimeError(
            "The vendored dexsuite DifficultyScheduler.__call__ has changed since"
            " SharedPredicateDifficultyScheduler was forked from it.\n"
            f"  expected AST fingerprint: {UPSTREAM_PREDICATE_FINGERPRINT}\n"
            f"  actual AST fingerprint:   {actual}\n"
            f"  source: {inspect.getsourcefile(DifficultyScheduler)}\n"
            "This guard exists because the certification harness scores episodes with THIS package's"
            " copy of the success predicate while the curriculum promotes on the upstream one; a"
            " silent divergence would produce a success rate that is not the task's. Re-read"
            " upstream's method, port any change into SharedPredicateDifficultyScheduler and"
            " within_success_tolerance, then update UPSTREAM_PREDICATE_FINGERPRINT in"
            " uwlab_tasks/manager_based/manipulation/dexlift/mdp/success.py."
        )


class SharedPredicateDifficultyScheduler(DifficultyScheduler):
    """dexsuite's ADR scheduler with its success test taken from :func:`within_success_tolerance`.

    NOTHING ELSE CHANGES. ``__init__``, ``get_state``, ``set_state`` and the promotion/demotion
    arithmetic are upstream's -- the arithmetic is reproduced verbatim below because there is no hook
    to override, and :func:`assert_upstream_predicate_unchanged` is what keeps the reproduction
    honest. The signature is upstream's too, defaults included, because
    :func:`resolve_adr_success_spec` reads tolerances that a config omits out of THIS signature.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        assert_upstream_predicate_unchanged()
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        pos_tol: float = 0.1,
        rot_tol: float | None = None,
        init_difficulty: int = 0,
        min_difficulty: int = 0,
        max_difficulty: int = 50,
        promotion_only: bool = False,
    ):
        del init_difficulty  # read from cfg.params by __init__; kept for signature parity
        pos_dist, rot_dist = goal_pose_error(env, env_ids, asset_cfg, object_cfg)
        move_up = within_success_tolerance(pos_dist, rot_dist, pos_tol, rot_tol)
        # -- upstream's promotion bookkeeping, verbatim (dexsuite/mdp/curriculums.py:107-114).
        demot = self.current_adr_difficulties[env_ids] if promotion_only else self.current_adr_difficulties[env_ids] - 1
        self.current_adr_difficulties[env_ids] = torch.where(
            move_up,
            self.current_adr_difficulties[env_ids] + 1,
            demot,
        ).clamp(min=min_difficulty, max=max_difficulty)
        self.difficulty_frac = torch.mean(self.current_adr_difficulties) / max(max_difficulty, 1)
        return self.difficulty_frac


def adr_param(term_cfg, name: str) -> Any:
    """Value the ADR scheduler will actually be called with for ``name``.

    Config first, then the scheduler's OWN signature default. The second half is the point: the
    config-time fallbacks for these keys are not free to invent (upstream's ``pos_tol`` default is
    0.1, and the task-state marker's fallback of 0.05 is a different number), so this reads the
    default off the callable the curriculum manager will invoke.
    """
    if name in term_cfg.params:
        return term_cfg.params[name]
    func = term_cfg.func
    target = func if inspect.isfunction(func) else func.__call__
    parameter = inspect.signature(target).parameters.get(name)
    if parameter is None or parameter.default is inspect.Parameter.empty:
        raise KeyError(f"'{name}' is neither configured on the ADR term nor defaulted by {func!r}.")
    return parameter.default


@dataclass(frozen=True)
class AdrSuccessSpec:
    """Everything needed to evaluate the task's success test, resolved from a live environment."""

    term_name: str
    scheduler_class: str
    pos_tol: float
    rot_tol: float | None
    asset_cfg: SceneEntityCfg
    object_cfg: SceneEntityCfg
    command_name: str = GOAL_COMMAND_NAME

    @property
    def rule(self) -> str:
        """The predicate as prose, for the record a certification run writes."""
        if self.rot_tol:
            return f"pos_dist < {self.pos_tol} and rot_dist < {self.rot_tol}"
        return f"pos_dist < {self.pos_tol}"


def resolve_adr_success_spec(env: ManagerBasedRLEnv, term_name: str = ADR_TERM_NAME) -> AdrSuccessSpec:
    """Read the success test out of the running environment's curriculum manager.

    Not out of a config file and not out of an argument: the tolerances that matter are the ones the
    scheduler is invoked with, and by the time an environment is running those live on the manager's
    term config.
    """
    curriculum_cfg = getattr(env, "curriculum_manager", None)
    term_cfg = None if curriculum_cfg is None else getattr(curriculum_cfg.cfg, term_name, None)
    if term_cfg is None:
        raise RuntimeError(
            f"No curriculum term '{term_name}' on this environment, so the task has no thresholded"
            " success test to evaluate. The reward term named 'success' is a dense shaping term, not"
            " a success test; scoring against it would report a number that is not a success rate."
        )
    return AdrSuccessSpec(
        term_name=term_name,
        scheduler_class=f"{type(term_cfg.func).__module__}.{type(term_cfg.func).__name__}",
        pos_tol=float(adr_param(term_cfg, "pos_tol")),
        rot_tol=None if adr_param(term_cfg, "rot_tol") is None else float(adr_param(term_cfg, "rot_tol")),
        asset_cfg=adr_param(term_cfg, "asset_cfg"),
        object_cfg=adr_param(term_cfg, "object_cfg"),
    )


class EpisodeSuccessProbe(ManagerTermBase):
    """A termination term that never terminates, and measures the success test before each reset.

    WHY THIS IS A TERMINATION TERM AT ALL. An evaluator outside the environment cannot sample the
    predicate on the last step of an episode: ``ManagerBasedRLEnv.step`` resets the terminated
    environments before it returns, so by the time ``env.step`` hands back an observation the object
    and the goal of a finished episode are already the NEXT episode's. Reading the command term's
    metrics has the same problem -- ``command_manager.compute`` runs after ``_reset_idx``. The
    termination manager, by contrast, runs BEFORE the reset, and ``TerminationManager.reset`` does
    not clear the per-term buffers, so a term evaluated there is readable afterwards and still holds
    the terminal state's answer. That is the only in-band hook with the right timing.

    IT CANNOT END AN EPISODE. ``__call__`` returns an all-False buffer, which the termination manager
    ORs into its truncated/terminated buffers, contributing nothing. Success does not truncate the
    episode here, deliberately: the reference protocol lets a successful episode run to its timeout,
    and cutting it short would change both the terminal breakdown and the object's opportunity to
    leave the goal again.

    EVALUATION ONLY. The harness adds this term to the environment config it builds; no training
    config carries it. It costs one pose error per step.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._spec: AdrSuccessSpec | None = None
        self._adr_term_name: str = cfg.params.get("adr_term_name", ADR_TERM_NAME)
        # Allocated once, outside any inference-mode block, and never written in place afterwards:
        # the per-step results below are rebound, not filled, so nothing here can be tagged an
        # inference tensor by a policy call and then fail on the next reset.
        self._never = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.pos_dist = torch.full((env.num_envs,), float("nan"), device=env.device)
        self.rot_dist = torch.full((env.num_envs,), float("nan"), device=env.device)

    @property
    def spec(self) -> AdrSuccessSpec:
        """The resolved success test. Lazy because the curriculum manager is built after this one."""
        if self._spec is None:
            self._spec = resolve_adr_success_spec(self._env, self._adr_term_name)
        return self._spec

    def __call__(self, env: ManagerBasedRLEnv, adr_term_name: str = ADR_TERM_NAME) -> torch.Tensor:
        self._adr_term_name = adr_term_name
        spec = self.spec
        self.pos_dist, self.rot_dist = goal_pose_error(
            env, slice(None), spec.asset_cfg, spec.object_cfg, spec.command_name
        )
        self.success = within_success_tolerance(self.pos_dist, self.rot_dist, spec.pos_tol, spec.rot_tol)
        return self._never


def success_probe_term_cfg(adr_term_name: str = ADR_TERM_NAME) -> TerminationTermCfg:
    """Term config an evaluator adds to ``terminations`` to make the success test observable."""
    return TerminationTermCfg(func=EpisodeSuccessProbe, params={"adr_term_name": adr_term_name}, time_out=False)


def matches_certified_recorder_predicate(pos_tol: float, rot_tol: float | None, position_only: bool) -> bool:
    """Is this task's scored success test the SAME TEST the certified rl_games runs measured?

    The two are written down in different places and neither cites the other, so this function is
    where they are compared instead of assumed equal:

    * The reference recorder (``DexsuiteSuccessRecorder.record_post_step``) tests
      ``pos_dist < position_threshold`` with a HARDCODED 0.05, and ANDs in
      ``rot_dist < orientation_threshold`` (0.5) unless the goal command declares ``position_only``.
    * This package tests :func:`within_success_tolerance` with the tolerances the ADR curriculum
      is actually invoked with, which dexsuite derives as ``pos_tol = rewards.success.params[
      "pos_std"] / 2`` and which the Lift subclass then strips the rotation half of.

    ON THE LIFT TASK THEY COINCIDE, and that is a measured fact rather than a design intent: the
    four certified runs (``in3kt4m6``, ``pe7mepo5``, ``us5i9luk``, ``lvjenlx7``) and every run of
    this port log the identical ``env_cfg`` fragment -- ``rewards.success.params.pos_std = 0.1``
    (hence ``pos_tol`` 0.05), ``curriculum.adr.params.rot_tol = null``, and
    ``commands.object_pose.position_only = true``. Both predicates therefore reduce to the same
    single comparison, ``pos_dist < 0.05``, and a chart may honestly draw them as one series.

    ON THE REORIENTATION TASK THEY DO NOT. There ``rot_tol`` is ``rot_std / 2`` = 0.25 while the
    recorder's constant is 0.5, so the recorder counts successes this package's predicate rejects.
    A number logged under :data:`EPISODE_SUCCESS_RATE_KEY` for that task would silently overlay two
    different tests, so the metric is published under a self-describing name instead -- see
    :func:`episode_success_rate_keys`, which is this function's only caller outside the env config.

    ``rot_tol`` is tested for TRUTHINESS because that is how :func:`within_success_tolerance` tests
    it: ``None`` and ``0.0`` both drop the orientation gate.
    """
    if float(pos_tol) != DEXSUITE_RECORDER_POSITION_THRESHOLD:
        return False
    if position_only:
        # The recorder drops its orientation gate here, so ours has to be dropped too.
        return not rot_tol
    # Otherwise the recorder applies its own 0.5 rad gate, and only that exact value agrees.
    return bool(rot_tol) and float(rot_tol) == DEXSUITE_RECORDER_ORIENTATION_THRESHOLD


def episode_success_rate_keys(pos_tol: float, rot_tol: float | None, position_only: bool) -> tuple[str, str]:
    """The log keys a success rate at THESE tolerances is allowed to be published under.

    A success rate is meaningless without the test that produced it, and the certified names carry
    that test implicitly -- :data:`EPISODE_SUCCESS_RATE_KEY` means ``pos_dist < 0.05`` (plus a 0.5
    rad gate when the goal is not ``position_only``) because that is what the reference recorder
    computed. So those names are reserved for exactly that test, checked by
    :func:`matches_certified_recorder_predicate` rather than assumed, and every other tolerance gets
    a name that states itself:

        Success_Rate/Episode/success_rate_p10mm            (Lift scored at 1 cm)
        Success_Rate/Episode/success_rate_p50mm_r250mrad   (Reorient at the dexsuite tolerances)

    THIS IS THE FIX FOR TWO SEPARATE FAILURES, and the reason the naming is computed rather than
    configured. Reorient previously logged NO success rate at all, because the only implementation
    was withheld to protect the certified name -- leaving ``Curriculum/adr`` as the sole signal, and
    that reads 0.0 both for "never succeeds" and for "curriculum stalled". And once ``pos_tol`` is
    moved to 0.01, the withholding decision no longer protects anything: it is taken once at
    ``__post_init__`` against the tolerance in force THEN, while
    :class:`EpisodeSuccessRateLogger` resolves the tolerance lazily from the LIVE curriculum term,
    so a hydra override would have published a 1 cm number under the 5 cm name. Deriving the key
    from the same lazily-resolved spec closes that gap by construction.

    Millimetres and milliradians because they make the common tolerances integers, and an integer in
    a wandb tag is a tag you can still read six weeks later.
    """
    if matches_certified_recorder_predicate(pos_tol, rot_tol, position_only):
        return EPISODE_SUCCESS_RATE_KEY, CUMULATIVE_SUCCESS_RATE_KEY
    stamp = f"_p{round(float(pos_tol) * 1000)}mm"
    if rot_tol:
        stamp += f"_r{round(float(rot_tol) * 1000)}mrad"
    return EPISODE_SUCCESS_RATE_KEY + stamp, CUMULATIVE_SUCCESS_RATE_KEY + stamp


class EpisodeSuccessRateLogger(EpisodeSuccessProbe):
    """Per-episode success rate, published into ``extras["log"]`` for whatever trainer is watching.

    WHAT IT MEASURES, and why it is not any of the numbers already in wandb. ``Curriculum/adr`` is
    the ADR difficulty fraction: it samples the success predicate ONCE per episode, at the terminal
    state, and then reports a promotion counter divided by ``max_difficulty`` -- a number that is
    zero both when the policy never succeeds and when the curriculum has stalled, and that is not
    on a 0..1 success scale in between. The reward term named ``success`` is a per-step tanh with a
    contact gate and no threshold at all. Neither is a success rate. THIS is: the fraction of
    finished episodes in which the object was inside tolerance of its goal AT ANY POINT, which is
    the sticky-OR protocol the certified runs were scored under and the one the certification
    harness re-implements with :class:`EpisodeSuccessProbe`.

    WHY IT IS A TERMINATION TERM AND NOT A ``RecorderTerm``. The reference implementation is a
    recorder, and this is deliberately not a port of the class -- only of its definition:

    * There is nothing to import. The dexsuite package vendored into THIS tree has no
      ``mdp/recorders.py``; the file exists only in the reference ``IsaacLabDexterous`` checkout,
      and the vendored tree is recreated by ``uwlab.sh``, so adding it there would not survive.
    * A recorder costs an extra observation pass PER STEP. ``ManagerBasedRLEnv.step`` guards the
      recorder hooks with ``if len(self.recorder_manager.active_terms) > 0:`` and, inside that
      guard, recomputes ``self.obs_buf = self.observation_manager.compute()`` before calling
      ``record_post_step``. Activating one recorder term to log one scalar would therefore double
      the observation cost of every step -- and this environment's ``perception`` group samples an
      object point cloud. The termination manager gives the same pre-reset timing for free.
    * The timing argument is already written down, on the parent class: ``TerminationManager``
      runs BEFORE ``_reset_idx``, which is the only in-band hook that still sees the terminal
      state. This subclass adds the sticky OR and the reset-time publication on top of it.

    WHERE THE NUMBER SURFACES. ``TerminationManager.reset(env_ids)`` is called from
    ``ManagerBasedRLEnv._reset_idx`` AFTER that method has replaced ``self.extras["log"]`` with a
    fresh dict and BEFORE it returns, so a write performed from :meth:`reset` lands in the same
    dict the reward and termination managers just filled and reaches the trainer with them.

    WHAT IS NOT PORTED: the recorder's per-asset-label breakdown
    (``Success_Rate/Episode/<label>/rate``). It buckets environments by the USD attribute
    ``isaaclab:spawn:asset_label``, which none of the objects in either tree authors -- in all four
    certified runs the breakdown collapsed to the single bucket ``Unknown``, i.e. it duplicated the
    aggregate. Reproducing a key whose only value is a copy of another key is not comparability.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._log_key_prefix: str = cfg.params.get("log_key_prefix", RL_GAMES_EPISODE_LOG_PREFIX)
        # Allocated here, outside any inference-mode block, and mutated IN PLACE afterwards -- the
        # opposite of the parent's rebinding discipline, and correct for the opposite reason. The
        # parent rebinds per-step RESULTS so they are never inference tensors held across a reset;
        # this is a persistent accumulator, so it must be a normal tensor that survives, and an
        # in-place bool update inside inference mode is exactly what the reference recorder does
        # (``self._episode_success |= success_now``).
        self._episode_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._total_success = 0
        self._total_attempts = 0
        self._keys: tuple[str, str] | None = None
        # PER-BRANCH SPLIT (V2_REPOSE_RECIPE.md sec 3.1). None -> aggregate only, which is the
        # pre-existing behaviour and stays the default. The mapping is NOT built here: it is
        # `episode_mixture.EPISODE_KIND_NAMES`, passed in by the cfg-time attach function, so this
        # module neither imports the mixture nor keeps a second copy of the kind integers (F27).
        self._kind_names: dict[int, str] | None = cfg.params.get("kind_names", None)
        self._kind_buffer_attr: str | None = cfg.params.get("kind_buffer_attr", None)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        adr_term_name: str = ADR_TERM_NAME,
        log_key_prefix: str = RL_GAMES_EPISODE_LOG_PREFIX,
        kind_names: dict[int, str] | None = None,
        kind_buffer_attr: str | None = None,
    ) -> torch.Tensor:
        # `kind_names` / `kind_buffer_attr` are consumed in __init__ and IGNORED here, but they MUST
        # still appear in this signature. IsaacLab's ManagerBase._resolve_common_term_cfg checks
        # `set(signature_args) == set(term_cfg.params)` statically at construction, and it does the
        # comparison on the SET, so a params key with no matching argument raises
        # "expects mandatory parameters ... but received ..." before the sim ever starts. A
        # `**kwargs` catch-all does not satisfy it either -- `kwargs` itself is then counted as a
        # parameter without a default and fails the same check.
        del kind_names, kind_buffer_attr
        self._log_key_prefix = log_key_prefix
        never = super().__call__(env, adr_term_name)
        # STICKY OR, the reference recorder's ``record_post_step`` in one line. "Reached the goal at
        # any point in the episode", not "was at the goal when the clock ran out" -- the object is
        # not held at the goal by this task, and the terminal-state reading is what ADR already
        # reports.
        self._episode_success |= self.success
        return never

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Publish the rate for the episodes ending now, then clear their flags.

        ``env_ids`` arrives as ``slice(None)`` for a whole-environment reset (``TerminationManager``
        substitutes it for ``None``), so it is used as an index and never as a length.
        """
        if env_ids is None:
            env_ids = slice(None)
        flags = self._episode_success[env_ids]
        # COUNT ONLY EPISODES THAT ACTUALLY RAN. ``_reset_idx`` zeroes ``episode_length_buf`` after
        # this call, so here it still holds the finished episode's length -- and it is 0 for the
        # construction-time reset, where every environment would otherwise be counted as a failed
        # episode that never happened. The reference recorder never saw that case: its
        # ``record_pre_reset`` hook is driven from ``step``, while this one is driven from
        # ``_reset_idx``, which ``env.reset()`` also calls.
        ran = self._env.episode_length_buf[env_ids] > 0
        attempts = int(ran.sum().item())
        if attempts == 0:
            return
        successes = int((flags & ran).sum().item())
        self._total_success += successes
        self._total_attempts += attempts
        # Resolved from the LIVE curriculum term, and only here -- after the ``attempts == 0`` return
        # above, so the construction-time reset never reaches a curriculum manager that does not
        # exist yet. The key therefore always names the tolerance the number was actually computed
        # at, including after a hydra override that ``__post_init__`` could not have seen.
        episode_key, cumulative_key = self._resolved_keys()
        log = self._env.extras.setdefault("log", {})
        log[self._log_key_prefix + episode_key] = successes / attempts
        log[self._log_key_prefix + cumulative_key] = self._total_success / self._total_attempts
        self._publish_per_kind(env_ids, flags, ran, episode_key, log)
        self._episode_success[env_ids] = False

    def _publish_per_kind(self, env_ids, flags, ran, episode_key: str, log: dict) -> None:
        """Split the SAME flags by episode kind. No second success computation, on purpose.

        WHY THIS IS HERE RATHER THAN IN ITS OWN TERM. ``V2_REPOSE_RECIPE.md`` sec 3.1's collapse
        tripwire (A1) is stated on CLASSIC-kind episodes specifically -- the aggregate is diluted by
        three other branches and muffles the signal exactly when the ramp is raising the tip-down
        fraction, which is the moment the signature is supposed to appear. A separate term would
        have had to recompute the success predicate, and a second independently-computed success
        rate that disagrees with the first is F27's defect class. These are the same ``flags`` and
        the same ``ran`` mask the aggregate above just used, indexed.

        Silently inert when the mixture is not wired (no buffer) or no mapping was configured.
        """
        if self._kind_names is None or self._kind_buffer_attr is None:
            return
        kind = getattr(self._env, self._kind_buffer_attr, None)
        if kind is None:
            return
        kind = kind[env_ids]
        for kind_value, kind_label in sorted(self._kind_names.items()):
            in_kind = ran & (kind == kind_value)
            n_kind = int(in_kind.sum().item())
            # Count published even at zero so "no episodes of this kind finished in this batch" is
            # distinguishable from "they all failed" -- the conflation F15 records for
            # ``Curriculum/adr``, which reads 0.0 both for "never succeeds" and "curriculum stalled".
            log[f"{self._log_key_prefix}{episode_key}/episodes/{kind_label}"] = float(n_kind)
            if n_kind == 0:
                continue
            log[f"{self._log_key_prefix}{episode_key}/{kind_label}"] = (
                int((flags & in_kind).sum().item()) / n_kind
            )

    def _resolved_keys(self) -> tuple[str, str]:
        """Cache the key pair; the tolerances cannot change within a run, but resolving is not free."""
        if self._keys is None:
            spec = self.spec
            position_only = bool(
                getattr(self._env.command_manager.cfg, spec.command_name, None)
                and getattr(getattr(self._env.command_manager.cfg, spec.command_name), "position_only", False)
            )
            self._keys = episode_success_rate_keys(spec.pos_tol, spec.rot_tol, position_only)
        return self._keys


def success_rate_log_term_cfg(
    adr_term_name: str = ADR_TERM_NAME,
    log_key_prefix: str = RL_GAMES_EPISODE_LOG_PREFIX,
    kind_names: dict[int, str] | None = None,
    kind_buffer_attr: str | None = None,
) -> TerminationTermCfg:
    """Term config a TRAINING config adds to ``terminations`` to log the episode success rate.

    See :data:`RL_GAMES_EPISODE_LOG_PREFIX` for what ``log_key_prefix`` is for and when to blank it.

    ``kind_names`` / ``kind_buffer_attr`` opt into the per-branch split
    (``episode_mixture.EPISODE_KIND_NAMES`` and ``EPISODE_KIND_BUFFER_ATTR``). Both default to
    ``None``, so an existing call site is byte-for-byte unchanged: the split is additive and no
    pre-existing series moves or changes meaning.
    """
    return TerminationTermCfg(
        func=EpisodeSuccessRateLogger,
        params={
            "adr_term_name": adr_term_name,
            "log_key_prefix": log_key_prefix,
            "kind_names": kind_names,
            "kind_buffer_attr": kind_buffer_attr,
        },
        time_out=False,
    )
