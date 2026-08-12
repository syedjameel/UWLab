# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The full-actuation guard: every named joint must be its OWN policy action dimension.

WHY THIS EXISTS. A multi-fingered hand is repeatedly "simplified" into one scalar close command --
a binary open/closed action, or a continuous fraction interpolating between two calibrated
postures. That is not a reduced parameterisation of grasping, it is a different, smaller set of
reachable grasps, and for the Tesollo DELTO it provably excludes the two-jaw pinch (the pads never
lead the phalanges at any single fraction). The requirement this module enforces is therefore
structural, not stylistic: on every path a policy is TRAINED on, each hand joint is an independent
action dimension.

Deleting the scalar action class once is not enough. It came back the last time because nothing
failed when it was reintroduced: an action group is just a config object, and a wrong one produces
a correctly-shaped, silently smaller action space. This module is the thing that fails.

TWO ENTRY POINTS, AND THE REASON THERE ARE TWO:

* :func:`assert_action_cfg_fully_actuates` runs at CONFIG CONSTRUCTION TIME -- inside
  ``__post_init__``, before any simulator exists. It raises from the caller's own stack frame, so
  it cannot be swallowed. This matters here specifically: ``SceneEntityCfg.resolve`` failures and
  anything else that happens during scene/manager construction run inside a timeline PLAY
  callback, where Omniverse's dispatcher PRINTS the exception and continues. A guard that only
  ran there would be a log line, not a failure. It also means the guard is exercisable with no
  GPU and no Isaac app, which is how its negative control is run.

* :func:`check_action_manager_fully_actuates` is an event term for ``mode="startup"``. The config
  check reasons about PATTERNS; this one reasons about the joints an action term actually resolved
  to on the spawned articulation. It catches what the static form cannot: a regex that matched 19
  joints because one was renamed in the USD, or a term whose scale mapping silently fell back.
  ``startup`` events are applied by the env directly rather than through a callback, so this one
  also fails the process outright (same reason ``check_gripper_joint_selection`` uses that mode).

WHAT COUNTS AS INDEPENDENT. An action term actuates its joints independently iff it spends exactly
one action dimension per joint it drives. A ``BinaryJointPositionAction`` over twenty joints has
``action_dim == 1``: twenty joints, one dimension, not independent. A ``JointPositionAction`` or
``RelativeJointPositionAction`` over twenty joints has ``action_dim == 20``: independent.

STATIC ANALYSIS IS DELIBERATELY DUCK-TYPED, and does not import Isaac Lab. That keeps this module
importable in a bare interpreter (no ``omni``), which is what makes the construction-time check
runnable as a test. The rules it uses on an action-term CONFIG are:

* a config carrying ``open_command_expr`` / ``close_command_expr`` is the binary family: ONE
  dimension, however many joints it names;
* a config whose ``scale`` is a mapping drives exactly the joints named by that mapping's keys,
  one dimension each. This package's own convention already requires the per-joint spelling (an
  unmatched key raises during term parsing, whereas a matched joint absent from the mapping
  silently takes scale 1.0), so relying on it here adds no new obligation;
* anything else cannot be decided from the config alone and is reported as UNVERIFIABLE rather
  than assumed good.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

# A joint name is "literal" -- i.e. it names one joint rather than matching a family -- when it
# contains no regular-expression metacharacter. ``rj_dg_1_1`` is literal; ``rj_dg_[1-5]_[1-4]`` is
# not. Only literal names can be attributed to a specific joint without an articulation to resolve
# against.
_REGEX_METACHARACTERS = re.compile(r"[\[\]\(\)\{\}\.\*\+\?\|\^\$\\]")

# Config attribute names that mark the one-scalar family. ``BinaryJointPositionActionCfg`` and
# every subclass of it (including any local "synergy" variant) carry both.
_BINARY_MARKERS = ("open_command_expr", "close_command_expr")


@dataclass(frozen=True)
class ActionTermSpec:
    """What one action term contributes, reduced to the only two facts this guard needs."""

    name: str
    """The term's attribute name in the action group, or its manager term name."""

    kind: str
    """Class name of the term (or of its config), for the error message."""

    joint_names: tuple[str, ...]
    """Joints the term drives, as far as they are known. Empty when the term names none."""

    action_dim: int | None
    """Action dimensions the term spends. ``None`` when it could not be determined."""

    @property
    def per_joint(self) -> bool:
        """True iff this term gives each joint it drives its own action dimension."""
        return self.action_dim is not None and bool(self.joint_names) and self.action_dim == len(self.joint_names)


def _is_literal(name: str) -> bool:
    return _REGEX_METACHARACTERS.search(name) is None


def _as_name_list(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(str(key) for key in value)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return ()


def describe_action_cfg_terms(actions_cfg) -> list[ActionTermSpec]:
    """Reduce an action GROUP config to one :class:`ActionTermSpec` per term.

    ``actions_cfg`` is the ``actions`` member of an env config: a plain object whose attributes are
    action-term configs. Attributes that are not action-term configs (no ``joint_names``) are
    skipped -- an operational-space arm term is described by its own joint patterns and simply
    never claims a hand joint.
    """
    specs: list[ActionTermSpec] = []
    for name, term in sorted(vars(actions_cfg).items()):
        if name.startswith("_") or term is None:
            continue
        if not hasattr(term, "joint_names"):
            continue
        kind = type(term).__name__
        if all(hasattr(term, marker) for marker in _BINARY_MARKERS):
            # One scalar, whatever it drives. Prefer the command mapping's keys: they are literal
            # joint names, whereas ``joint_names`` is usually one regex.
            joints = _as_name_list(getattr(term, "open_command_expr", None)) or _as_name_list(term.joint_names)
            specs.append(ActionTermSpec(name=name, kind=kind, joint_names=joints, action_dim=1))
            continue
        scale = getattr(term, "scale", None)
        if isinstance(scale, Mapping):
            joints = _as_name_list(scale)
            specs.append(ActionTermSpec(name=name, kind=kind, joint_names=joints, action_dim=len(joints)))
            continue
        specs.append(ActionTermSpec(name=name, kind=kind, joint_names=_as_name_list(term.joint_names), action_dim=None))
    return specs


def describe_action_manager_terms(action_manager) -> list[ActionTermSpec]:
    """Reduce a LIVE :class:`~isaaclab.managers.ActionManager` to one spec per term.

    Reads the resolved joint names (``_joint_names``, set by ``JointAction.__init__`` after the
    patterns are matched against the articulation) rather than the configured patterns, which is
    the whole point of running this check again at startup.
    """
    specs: list[ActionTermSpec] = []
    for name in action_manager.active_terms:
        term = action_manager.get_term(name)
        joints = _as_name_list(getattr(term, "_joint_names", None))
        dim = getattr(term, "action_dim", None)
        specs.append(
            ActionTermSpec(
                name=name,
                kind=type(term).__name__,
                joint_names=joints,
                action_dim=int(dim) if isinstance(dim, int) else None,
            )
        )
    return specs


def _explain(spec: ActionTermSpec) -> str:
    if spec.action_dim is None:
        return (
            f"action term '{spec.name}' ({spec.kind}) names it, but how many action dimensions that"
            " term spends cannot be determined from the config. Give its 'scale' as a per-joint"
            " mapping (one key per joint) so the guard can decide."
        )
    return (
        f"action term '{spec.name}' ({spec.kind}) spends {spec.action_dim} action"
        f" dimension{'' if spec.action_dim == 1 else 's'} on {len(spec.joint_names)} joints -- the"
        " joints are slaved to a shared command, not independent policy actions."
    )


def assert_joints_independently_actuated(
    required_joints: Sequence[str],
    specs: Sequence[ActionTermSpec],
    context: str,
) -> None:
    """Raise unless every joint in ``required_joints`` gets its own action dimension.

    Args:
        required_joints: Literal joint names that must each be an independent policy action.
        specs: What every action term of the environment drives.
        context: Where this is being checked, quoted verbatim in the error.

    Raises:
        ValueError: naming every joint that is not independently actuated and what drives it.
    """
    independent: set[str] = set()
    for spec in specs:
        if spec.per_joint:
            independent.update(name for name in spec.joint_names if _is_literal(name))

    failures: list[str] = []
    for joint in required_joints:
        if joint in independent:
            continue
        owners = [spec for spec in specs if joint in spec.joint_names]
        if not owners:
            failures.append(f"  {joint}: NO action term names it -- it is not a policy action at all.")
        else:
            for owner in owners:
                failures.append(f"  {joint}: {_explain(owner)}")
    if not failures:
        return

    inventory = "\n".join(
        f"  '{spec.name}' ({spec.kind}): action_dim="
        f"{'unknown' if spec.action_dim is None else spec.action_dim},"
        f" joints={list(spec.joint_names)}"
        for spec in specs
    )
    raise ValueError(
        "FULL-ACTUATION GUARD FAILED for "
        f"{context}.\n"
        f"All {len(required_joints)} hand joints must be INDEPENDENT policy actions -- one action"
        " dimension per joint. These are not:\n" + "\n".join(failures) + "\n\nAction terms seen:\n" + inventory + "\n\n"
        "This is the one-scalar closure model (a binary open/closed command, or one fraction"
        " interpolating between two calibrated postures). It is banned from every environment a"
        " policy is trained in: a single close command cannot put this hand's opposing pads in"
        " front of its phalanges, so the pinch is unreachable at every value of that scalar."
        " Give the hand a per-joint action term (RelativeJointPositionActionCfg /"
        " JointPositionActionCfg with a per-joint scale mapping), or, if this environment is"
        " genuinely not a training path, do not call the guard on it.\n"
        "Guard: uwlab.envs.mdp.full_actuation."
    )


def assert_action_cfg_fully_actuates(actions_cfg, required_joints: Sequence[str], context: str) -> None:
    """Construction-time form. Call at the END of ``__post_init__``, after the action is assigned.

    Raises directly in the caller's stack frame -- no manager, no callback, nothing to swallow it.
    """
    assert_joints_independently_actuated(required_joints, describe_action_cfg_terms(actions_cfg), context)


def assert_direct_action_space_fully_actuates(
    action_space: int,
    driven_joints: Sequence[str],
    required_joints: Sequence[str],
    context: str,
) -> None:
    """Construction-time form for a DIRECT-workflow env, which has no action manager.

    A ``DirectRLEnvCfg`` declares its width as one integer and the env class decides what the
    dimensions mean, so there are no terms to inspect. Give it the joints the env writes targets
    for; the same rule then applies -- one dimension per joint, or the hand is sharing a command.
    """
    spec = ActionTermSpec(
        name="<DirectRLEnvCfg.action_space>",
        kind="direct workflow",
        joint_names=tuple(driven_joints),
        action_dim=int(action_space),
    )
    assert_joints_independently_actuated(required_joints, [spec], context)


def check_action_manager_fully_actuates(env, env_ids, required_joints: Sequence[str], context: str = "") -> None:
    """Startup-event form: verify against the joints the action terms RESOLVED to.

    Declared with ``mode="startup"`` so the env applies it directly. Note ``env_ids`` is unused and
    present only to satisfy the event-term signature; the check draws no randomness and is a pure
    assertion about the environment's structure.
    """
    del env_ids
    label = context or type(getattr(env, "cfg", env)).__name__
    assert_joints_independently_actuated(
        required_joints, describe_action_manager_terms(env.action_manager), f"{label} (resolved action manager)"
    )
