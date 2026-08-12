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
    "EpisodeSuccessProbe",
    "GOAL_COMMAND_NAME",
    "SUCCESS_PROBE_TERM_NAME",
    "SharedPredicateDifficultyScheduler",
    "adr_param",
    "assert_upstream_predicate_unchanged",
    "goal_pose_error",
    "resolve_adr_success_spec",
    "success_probe_term_cfg",
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
