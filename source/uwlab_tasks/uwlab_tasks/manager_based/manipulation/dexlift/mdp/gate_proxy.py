# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stateful IsaacLab term that logs the TRAINING-TIME gate proxy (``V2_REPOSE_RECIPE.md`` sec 4,
bead ``dr-tlx.2``).

Thin wrapper. Every definition lives in :mod:`gate_proxy_core` (the reduction and the log keys) and
in ``held_check_core.passive_gates`` (the gates themselves); this file only reads dexlift's scene
entities each step and hands the tensors over. See :mod:`gate_proxy_core` for what the metric means,
why it is an upper bound, and why ``_atend`` rather than ``_ever`` is the primary variant.

WHY IT IS A TERMINATION TERM THAT NEVER TERMINATES. Same three reasons
``mdp.success.EpisodeSuccessRateLogger`` is one, and they transfer unchanged:

* ``TerminationManager`` runs BEFORE ``_reset_idx``, so it is the only in-band hook that still sees
  the TERMINAL state -- which is the entire point of the ``_atend`` variant.
* ``TerminationManager.reset(env_ids)`` is called from ``_reset_idx`` after that method has replaced
  ``extras["log"]`` with a fresh dict and before it returns, so a write from :meth:`reset` reaches
  the trainer with the reward and termination logs.
* a ``RecorderTerm`` would cost an extra full observation pass per step
  (``ManagerBasedRLEnv.step`` recomputes ``obs_buf`` inside its recorder guard), and this env's
  ``perception`` group samples an object point cloud.

**It returns all-False unconditionally and cannot end an episode**, and it returns a freshly
zeroed tensor rather than a stored one, so a caller that mutates the result cannot latch a
permanent True into this term's state. A logging term that silently acquired the power to
terminate would corrupt the very training it exists to observe.

WHY IT READS THE SAME TENSORS ``held_with_probe`` READS, RATHER THAN REUSING THAT TERM. The
generator wires ``held_with_probe`` as ``terminations.success``; it ends episodes and its fourth
gate needs the probe action bias the generator injects. Neither is acceptable during training. So
this term duplicates the *reads* (palm body, object, fingertip sensors) but NOT the *decision* --
the gate booleans come from the one shared ``passive_gates``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from . import gate_proxy_core
from .episode_mixture import EPISODE_KIND_BUFFER_ATTR, EPISODE_KIND_NAMES
from .held_check_core import COMOVE_SPEED_THRESH, COMOVE_VZ_THRESH, SETTLE_STEPS, passive_gates
from .rewards import _sensor_force_magnitudes  # reused, not reimplemented -- see rewards.py:40-76

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GateProxyLogger(ManagerTermBase):
    """Publishes the passive-gate proxy into ``extras["log"]`` every reset. Never terminates.

    Params (all optional; the defaults are ``held_with_probe``'s own, read from the same places):
        robot_cfg: ``SceneEntityCfg`` naming the palm body (must resolve to exactly one).
        object_cfg: ``SceneEntityCfg`` for the manipulated object.
        thumb_contact_names / tip_contact_names: fingertip sensor names.
        force_threshold: N, per-fingertip normal-force gate.
        comove_speed_thresh, comove_vz_thresh, settle_steps: see ``held_check_core.passive_gates``.
        kind_names: ``{int: str}`` episode-kind labels, or ``None`` to publish un-split metrics.
        log_key_prefix: series prefix, default ``GateProxy/``.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        # The three GATE thresholds are IMPORTED from held_check_core, never restated -- they have
        # a second consumer (c3_rung.py imports SETTLE_STEPS for S_t's goal re-pin step floor,
        # bead dr-ai1.18), so a local copy would silently describe another subsystem's timing.
        # The scene-entity and force defaults below ARE copied from held_with_probe's own param
        # block: they are `cfg.params.get` defaults on a term this file cannot import an instance
        # of (it may not even be constructed in a training run), so there is nothing to read them
        # off. Guarded instead -- `_assert_thresholds_match` fails construction if a
        # held_with_probe term IS present and disagrees.
        self.robot_cfg: SceneEntityCfg = p.get("robot_cfg", SceneEntityCfg("robot", body_names="rl_dg_mount"))
        self.object_cfg: SceneEntityCfg = p.get("object_cfg", SceneEntityCfg("object"))
        self.thumb_contact_names = p.get("thumb_contact_names", ("rl_dg_1_tip", "rl_dg_5_tip"))
        self.tip_contact_names = p.get("tip_contact_names", ("rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip"))
        self.force_threshold = p.get("force_threshold", 0.2)
        self.comove_speed_thresh = p.get("comove_speed_thresh", COMOVE_SPEED_THRESH)
        self.comove_vz_thresh = p.get("comove_vz_thresh", COMOVE_VZ_THRESH)
        self.settle_steps = p.get("settle_steps", SETTLE_STEPS)
        self.kind_names: dict[int, str] | None = p.get("kind_names", EPISODE_KIND_NAMES)
        self._log_key_prefix: str = p.get("log_key_prefix", gate_proxy_core.DEFAULT_LOG_PREFIX)

        self.robot_cfg.resolve(env.scene)
        self.object_cfg.resolve(env.scene)
        assert len(self.robot_cfg.body_ids) == 1, (
            f"GateProxyLogger.robot_cfg must resolve to exactly one body, got {self.robot_cfg.body_ids}"
        )
        self._palm_id = self.robot_cfg.body_ids[0]

        n = env.num_envs
        device = env.device
        names = (*gate_proxy_core.PASSIVE_GATE_NAMES, gate_proxy_core.PASSIVE_ALL_NAME)
        # `_atend` holds the MOST RECENT step's value and is overwritten every step; at
        # `reset(env_ids)` time it therefore still holds the terminal step's value for exactly the
        # episodes ending now. Same technique, and the same ordering argument, as
        # `held_check._last_breakdown` -- see the comment on that attribute.
        self._atend: dict[str, torch.Tensor] = {
            name: torch.zeros(n, dtype=torch.bool, device=device) for name in names
        }
        self._ever: dict[str, torch.Tensor] = {
            name: torch.zeros(n, dtype=torch.bool, device=device) for name in names
        }
        self._cumulative: dict[str, int] = {}
        self._never = torch.zeros(n, dtype=torch.bool, device=device)
        self._assert_thresholds_match(env)

    def _assert_thresholds_match(self, env: ManagerBasedRLEnv) -> None:
        """Fail construction if a ``held_with_probe`` term is also wired and disagrees with us.

        The proxy's whole claim is that it computes the SAME gates the generator computes. The gate
        LOGIC is shared by construction (``passive_gates``), but the three thresholds arrive as
        term params and could be overridden on one term and not the other -- which would produce a
        metric that tracks nothing, silently, and would read as a policy result. Eleven recorded
        instances of that defect class say check it rather than assume it (F27).

        Only checks when such a term exists; in a training run it usually does not.
        """
        terms = getattr(env.cfg, "terminations", None)
        if terms is None:
            return
        for attr in dir(terms):
            term = getattr(terms, attr, None)
            params = getattr(term, "params", None)
            func = getattr(term, "func", None)
            if params is None or getattr(func, "__name__", "") != "held_with_probe":
                continue
            for key, ours in (
                ("comove_speed_thresh", self.comove_speed_thresh),
                ("comove_vz_thresh", self.comove_vz_thresh),
                ("settle_steps", self.settle_steps),
            ):
                theirs = params.get(key, ours)
                if theirs != ours:
                    raise ValueError(
                        f"GateProxyLogger.{key}={ours} disagrees with terminations.{attr}"
                        f" (held_with_probe).{key}={theirs}. The proxy exists to predict THAT"
                        " term's gate-chain pass rate (V2_REPOSE_RECIPE.md sec 4); a threshold"
                        " that differs makes it predict nothing while still publishing a number."
                    )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        kind_names: dict[int, str] | None = None,
        log_key_prefix: str = gate_proxy_core.DEFAULT_LOG_PREFIX,
    ) -> torch.Tensor:
        # Both are consumed in __init__ and ignored here, and both MUST still be declared. IsaacLab
        # compares `set(__call__'s args)` against `set(term_cfg.params)` statically at construction
        # (ManagerBase._resolve_common_term_cfg), so a params key with no matching argument raises
        # before the sim starts -- and `**kwargs` does not satisfy the check, it fails it. ANY new
        # key added to `gate_proxy_log_term_cfg`'s params must be added here too.
        del kind_names, log_key_prefix
        robot = env.scene[self.robot_cfg.name]
        obj = env.scene[self.object_cfg.name]

        thumb_force = _sensor_force_magnitudes(env, self.thumb_contact_names)
        tip_force = _sensor_force_magnitudes(env, self.tip_contact_names)
        relative_speed = torch.linalg.vector_norm(
            obj.data.root_lin_vel_w - robot.data.body_lin_vel_w[:, self._palm_id, :], dim=-1
        )

        settled, opposed_contact, co_move = passive_gates(
            steps_since_reset=env.episode_length_buf,
            thumb_loaded=thumb_force.gt(self.force_threshold).any(dim=-1),
            tip_loaded=tip_force.gt(self.force_threshold).any(dim=-1),
            relative_speed=relative_speed,
            obj_vz=obj.data.root_lin_vel_w[:, 2],
            settle_steps=self.settle_steps,
            comove_speed_thresh=self.comove_speed_thresh,
            comove_vz_thresh=self.comove_vz_thresh,
        )
        step = {
            "settled": settled,
            "opposed_contact": opposed_contact,
            "co_move": co_move,
            gate_proxy_core.PASSIVE_ALL_NAME: settled & opposed_contact & co_move,
        }
        for name, value in step.items():
            # In-place, so these persist across the inference-mode boundary the way
            # EpisodeSuccessRateLogger's accumulator does (see its __init__ comment).
            self._atend[name][:] = value
            self._ever[name] |= value

        # A LOGGING TERM MAY NOT END AN EPISODE. Returned fresh-zeroed rather than by returning
        # `self._never` directly, so a caller that mutates the returned tensor cannot latch a
        # permanent True into this term's state.
        return torch.zeros_like(self._never)

    def reset(self, env_ids=None) -> None:
        """Publish for the episodes ending now, then clear their flags.

        ``env_ids`` arrives as ``slice(None)`` for a whole-environment reset (``TerminationManager``
        substitutes it for ``None``), so it is used as an index and never as a length -- same
        contract as ``EpisodeSuccessRateLogger.reset``.
        """
        if env_ids is None:
            env_ids = slice(None)
        # `episode_length_buf` is zeroed by `_reset_idx` AFTER this call, so here it still holds the
        # finished episode's length -- and 0 for the construction-time reset, where every env would
        # otherwise be counted as a failed episode that never happened.
        ran = self._env.episode_length_buf[env_ids] > 0
        if not bool(ran.any()):
            self._clear(env_ids)
            return

        atend = {name: value[env_ids] for name, value in self._atend.items()}
        ever = {name: value[env_ids] for name, value in self._ever.items()}
        chain = gate_proxy_core.evaluate_priority_chain(
            settled=atend["settled"],
            opposed_contact=atend["opposed_contact"],
            co_move=atend["co_move"],
            ran=ran,
        )
        for key, mask in chain.items():
            self._cumulative[key] = self._cumulative.get(key, 0) + int(mask.sum().item())
        self._cumulative["episodes"] = self._cumulative.get("episodes", 0) + int(ran.sum().item())

        kind_buf = getattr(self._env, EPISODE_KIND_BUFFER_ATTR, None)
        kind = None if kind_buf is None else kind_buf[env_ids]
        # Cumulative PER-BRANCH counts, on top of the per-batch fractions build_log_entries
        # publishes. A reset batch can be a handful of episodes wide, and a per-branch fraction over
        # a handful of episodes is noise -- R0's baseline measurement (V2_REPOSE_RECIPE.md O6) needs
        # something it can sum over a whole rollout and divide once. See per_kind_counts.
        for key, value in gate_proxy_core.per_kind_counts(
            chain, ran, kind, None if kind is None else self.kind_names
        ).items():
            self._cumulative[key] = self._cumulative.get(key, 0) + value
        self._env.extras.setdefault("log", {}).update(
            gate_proxy_core.build_log_entries(
                atend=atend,
                ever=ever,
                ran=ran,
                cumulative_counts=self._cumulative,
                kind=kind,
                kind_names=None if kind is None else self.kind_names,
                prefix=self._log_key_prefix,
            )
        )
        self._clear(env_ids)

    def _clear(self, env_ids) -> None:
        for name in self._ever:
            self._ever[name][env_ids] = False
            self._atend[name][env_ids] = False


# Defaults are declared ONCE, here, and both the term and the banner read them from this dict --
# so the number a run PRINTS and the number it USES cannot diverge. The three that belong to
# held_check_core are imported from it, never restated: SETTLE_STEPS has a second consumer in
# c3_rung.py (S_t's goal re-pin step floor, bead dr-ai1.18), so a copy here would silently describe
# a different subsystem's timing too.
GATE_PROXY_DEFAULTS: dict[str, float] = {
    "settle_steps": SETTLE_STEPS,
    "comove_speed_thresh": COMOVE_SPEED_THRESH,
    "comove_vz_thresh": COMOVE_VZ_THRESH,
}


def gate_proxy_banner(kind_labels: list[str], success_split: bool) -> str:
    """The exact banner text printed when the gate proxy is staged (R5).

    Returned as a string rather than printed, so a test can assert on it byte-for-byte -- the same
    technique ``c3_transport_core.transport_goal_banner`` uses, and for the reason F42 records: an
    earlier version of this banner stated "settled > 60 steps, relative speed < 0.05 m/s" as
    LITERALS. That is the F27 defect class in the one place it does most damage, because a stale
    banner is what a reader checks a run's staging AGAINST (RESET_SPEC_V2.md sec 1a trap 3). Every
    number below is interpolated from :data:`GATE_PROXY_DEFAULTS`, which reads held_check_core.
    """
    return (
        "[dexreset] GATE PROXY staged: publishing GateProxy/{settled,opposed_contact,co_move,"
        "passive_three}_{atend,ever}_frac plus priority-ordered reach and first-fail counts, split"
        f" by episode kind {sorted(kind_labels)}. Thresholds are held_check_core's own, imported"
        f" not restated: settled > {GATE_PROXY_DEFAULTS['settle_steps']} steps, relative speed <"
        f" {GATE_PROXY_DEFAULTS['comove_speed_thresh']} m/s, |obj vz| <"
        f" {GATE_PROXY_DEFAULTS['comove_vz_thresh']} m/s -- the same values, via the same"
        " passive_gates call, that the generator's held predicate uses."
        " THESE ARE AN UPPER BOUND ON GATE-CHAIN PASS RATE, NOT A YIELD (RESET_SPEC_V2.md R7):"
        " the three PROBE gates cannot be measured without injecting the probe bias into training."
        + (
            " Success rate is split by the same mapping."
            if success_split
            else " Success rate is NOT split: no success_rate_log term exists on this config."
        )
    )


def gate_proxy_log_term_cfg(
    kind_names: dict[int, str] | None = None,
    log_key_prefix: str = gate_proxy_core.DEFAULT_LOG_PREFIX,
) -> TerminationTermCfg:
    """Term config a TRAINING config adds to ``terminations`` to log the gate proxy.

    ``time_out=False`` and the term itself always returns False, so it contributes nothing to the
    done signal -- see :class:`GateProxyLogger`.
    """
    return TerminationTermCfg(
        func=GateProxyLogger,
        params={
            "kind_names": EPISODE_KIND_NAMES if kind_names is None else kind_names,
            "log_key_prefix": log_key_prefix,
        },
        time_out=False,
    )
