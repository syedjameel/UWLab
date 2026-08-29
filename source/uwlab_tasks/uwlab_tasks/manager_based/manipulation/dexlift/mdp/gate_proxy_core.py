# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-tensor core of the TRAINING-TIME gate proxy (``V2_REPOSE_RECIPE.md`` sec 4, bead
``dr-tlx.2``).

No ``isaaclab`` / ``omni`` import anywhere in this file, same split and same reason as
``held_check_core.py`` and ``c3_transport_core.py`` next to it: ``test_gate_proxy_core.py`` loads it
by file path and runs it with no Isaac process. :mod:`gate_proxy` is the thin ``ManagerTermBase``
wrapper; nothing decision-relevant lives there that is not here.

=== WHAT THIS MEASURES, AND WHY IT IS THE METRIC THE RETRAIN IS STEERED BY ===

``RESET_SPEC_V2.md`` R2 requires ``accepted / attempted > 0.50`` per rung. Measured, that is not a
requirement on the pose sampler: in the v1 S1 run **at least 90.8 percent of generation attempts
died in the held-state gate chain before the rung's geometry was ever evaluated**, ``co_move``
alone rejecting 46.8 percent (``V2_POSE_FINDINGS.md`` F28, confirmed on a second rung by F30 at
53.2 percent). R2 is therefore a requirement on the REPOSE POLICY's held-state pass rate.

Finding out whether a checkpoint clears it by running a generation pass is expensive -- a
600-state production chunk implies roughly 25,000 attempts (F26/R6). So the training run carries a
proxy instead: the three gates of the chain that can be evaluated WITHOUT the probe.

=== WHY ONLY THREE OF THE SIX GATES, STATED SO THE NUMBER IS NEVER OVER-READ ===

The chain is ``settled -> opposed_contact -> co_move -> probe_ready -> probe_gripper_moved ->
probe_tracks``. The last three need a probe, and the probe is an action bias the generator injects
on top of the policy's own actions (``held_check_core.PROBE_ARM_ACTION_BIAS`` = 0.15). Injecting
that during training would perturb the thing being trained, so it is deliberately not done.

**Every number this module produces is therefore a STRICT UPPER BOUND on gate-chain pass rate and
must be reported as one. It is not a yield and may never be quoted as one** (``RESET_SPEC_V2.md``
R7). What it is good for is exact, and worth stating positively:

* accepted states are a SUBSET of passive-three-passing states, so ``passive_three > 0.50`` is a
  HARD NECESSARY condition for R2 -- assumption-free, and cheap enough to check every iteration.
  A run that does not clear 0.50 here cannot clear R2, and no generation pass is needed to know it.
* on F28's own histogram the decomposition is exact: passive-three 43/141 = 0.305, probe-stage
  conditional 13/43 = 0.302, product 0.0922 = the reported 9.22 percent.

=== TWO VARIANTS, BECAUSE THE EVALUATION MOMENT IS THE WHOLE POINT ===

``V2_POSE_FINDINGS.md`` F29 is about exactly this: ``seated: 0`` read like a measurement and was an
artifact of WHEN and IN WHAT ORDER the counter was evaluated. So:

* ``_atend`` -- the gate's value at the episode's FINAL step. This matches the generator, which
  evaluates once at episode end (``held_check.gate_breakdown`` reads a cache written that step).
  **This is the primary variant and the one the recipe's targets are stated against.**
* ``_ever`` -- sticky OR over the episode, the protocol ``EpisodeSuccessRateLogger`` uses for the
  success rate. It reads HIGHER by construction and must never be substituted for ``_atend``.

And F29's own stated remedy -- "a priority-ordered failure counter must also report how many
episodes reached each gate, or its zeros are uninterpretable" -- is implemented here rather than
left for a reader to want: :func:`evaluate_priority_chain` returns ``reached_*`` alongside
``first_fail_*``, so a zero in a first-fail row can always be divided by the number of episodes
that got there.

=== THE THRESHOLDS ARE NOT IN THIS FILE, ON PURPOSE ===

There is no copy of ``settle_steps``, ``comove_speed_thresh`` or ``comove_vz_thresh`` here, and no
re-written comparison. The gate booleans arrive already computed by
``held_check_core.passive_gates`` -- the SAME function ``held_decision`` calls. Importing the
constants and re-writing the comparisons would still have left two implementations, which is F27's
defect class (eleven recorded instances, most recently in a summary statistic and in an
instrument). There is one.
"""

from __future__ import annotations

import torch

# The three probe-free gates, in the PRIORITY ORDER the generator's rejection histogram reports
# them (V2_POSE_FINDINGS.md F28/F30). Order is load-bearing: `first_fail_*` is defined by it.
PASSIVE_GATE_NAMES: tuple[str, str, str] = ("settled", "opposed_contact", "co_move")

# The AND of all three. Named separately because it, not any individual gate, is what the recipe's
# 0.50 hard floor and 0.71 working target are stated against (V2_REPOSE_RECIPE.md sec 4.2).
PASSIVE_ALL_NAME = "passive_three"

DEFAULT_LOG_PREFIX = "GateProxy/"


def evaluate_priority_chain(
    settled: torch.Tensor,
    opposed_contact: torch.Tensor,
    co_move: torch.Tensor,
    ran: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Reduce the three gate booleans to the same shape of table the generator prints.

    Args:
        settled, opposed_contact, co_move: ``(N,)`` bool, as returned by
            ``held_check_core.passive_gates``. Not recomputed here -- see the module docstring.
        ran: ``(N,)`` bool, whether this environment's episode actually ran. The
            construction-time reset produces episodes of length 0 for every environment, and
            counting those as failures would put a large fake spike at iteration 0 in every series
            (the same trap ``EpisodeSuccessRateLogger.reset`` guards with its own ``ran`` mask, for
            the same reason: its parent recorder was driven from ``step`` and never saw the case).

    Returns:
        A dict of ``(N,)`` bool tensors:

        * ``reached_<gate>`` -- the episode got as far as this gate in priority order, i.e. every
          EARLIER gate passed. ``reached_settled`` is just ``ran``. This is F29's remedy: without
          it, a zero in ``first_fail_co_move`` cannot be told apart from "nothing ever reached
          co_move".
        * ``first_fail_<gate>`` -- this gate is the FIRST one the episode failed. Exactly the
          quantity F28's table counts, so a training-time table can be read against it directly.
        * ``passive_three`` -- all three passed.

        The ``first_fail_*`` masks and ``passive_three`` partition ``ran`` exactly; that invariant
        is asserted in the tests rather than assumed.
    """
    reached_settled = ran
    reached_opposed = ran & settled
    reached_co_move = reached_opposed & opposed_contact
    passed_all = reached_co_move & co_move
    return {
        "reached_settled": reached_settled,
        "reached_opposed_contact": reached_opposed,
        "reached_co_move": reached_co_move,
        f"reached_{PASSIVE_ALL_NAME}": ran,
        "first_fail_settled": reached_settled & ~settled,
        "first_fail_opposed_contact": reached_opposed & ~opposed_contact,
        "first_fail_co_move": reached_co_move & ~co_move,
        PASSIVE_ALL_NAME: passed_all,
    }


def per_kind_counts(
    chain: dict[str, torch.Tensor],
    ran: torch.Tensor,
    kind: torch.Tensor | None,
    kind_names: dict[int, str] | None,
) -> dict[str, int]:
    """Counts, per branch, that a caller can ACCUMULATE across resets.

    :func:`build_log_entries` publishes per-branch FRACTIONS for the batch that just finished, which
    is the right thing for a training dashboard and the wrong thing for a baseline measurement: a
    reset batch can be three episodes wide, and a fraction over three episodes is noise. R0 -- the
    control that measures this project's never-yet-measured passive-three baseline for ``ep_3600``
    (``V2_REPOSE_RECIPE.md`` O6, the denominator of every improvement claim in that document) --
    needs counts it can add up over a whole rollout and divide once at the end.

    Returns ``{"episodes/<branch>": n, "passive_three/<branch>": n, ...}`` plus the same two keys
    for each priority-chain row, so the training-time table can be reconstructed per branch.
    Empty when the mixture is not wired.
    """
    if kind is None or kind_names is None:
        return {}
    counts: dict[str, int] = {}
    for kind_value, kind_label in sorted(kind_names.items()):
        in_kind = ran & (kind == kind_value)
        counts[f"episodes/{kind_label}"] = int(in_kind.sum().item())
        for key, mask in chain.items():
            counts[f"{key}/{kind_label}"] = int((mask & in_kind).sum().item())
    return counts


def build_log_entries(
    *,
    atend: dict[str, torch.Tensor],
    ever: dict[str, torch.Tensor],
    ran: torch.Tensor,
    cumulative_counts: dict[str, int],
    kind: torch.Tensor | None,
    kind_names: dict[int, str] | None,
    prefix: str = DEFAULT_LOG_PREFIX,
) -> dict[str, float]:
    """Build the flat ``extras["log"]`` dict for the episodes finishing now.

    Args:
        atend: gate name -> ``(N,)`` bool, value at the episode's FINAL step. Must contain the
            three :data:`PASSIVE_GATE_NAMES` and :data:`PASSIVE_ALL_NAME`.
        ever: gate name -> ``(N,)`` bool, sticky OR over the episode. Same keys.
        ran: ``(N,)`` bool, episodes that actually ran (see :func:`evaluate_priority_chain`).
        cumulative_counts: running totals the caller maintains across resets, already updated for
            this batch. Published as-is so a short reset batch does not produce a fraction computed
            from three episodes.
        kind: ``(N,)`` int episode-kind labels, or ``None`` when the episode mixture is not wired.
        kind_names: mapping from kind integer to series name. **REQUIRED whenever ``kind`` is
            given, and deliberately has no default in this module** -- the integers are defined by
            ``episode_mixture.EPISODE_KIND_*``, which cannot be imported here (it pulls isaaclab),
            and restating them is the F27 defect class. The caller that CAN import them passes them
            in. Same guard technique as ``c3_transport_core.tip_z_from_root_z``'s required
            ``tilt_rad``: a caller must name the convention it is using.

    Returns:
        ``{key: float}``, ready to merge into ``env.extras["log"]``.

    Raises:
        ValueError: if ``kind`` is given without ``kind_names``, or a required gate key is missing.
    """
    if kind is not None and kind_names is None:
        raise ValueError(
            "kind_names is required whenever kind is given -- the kind integers are defined by"
            " episode_mixture.EPISODE_KIND_*, and this module must not restate them (F27)."
        )
    required = (*PASSIVE_GATE_NAMES, PASSIVE_ALL_NAME)
    for name in required:
        for source, label in ((atend, "atend"), (ever, "ever")):
            if name not in source:
                raise ValueError(f"{label} is missing gate {name!r}; got {sorted(source)}")

    n_ran = int(ran.sum().item())
    log: dict[str, float] = {}
    if n_ran == 0:
        # Every environment reset with a zero-length episode -- the construction-time reset. There
        # is nothing to report and publishing zeros would be a lie in every series at once.
        return log

    for name in required:
        log[f"{prefix}{name}_atend_frac"] = float((atend[name] & ran).sum().item()) / n_ran
        log[f"{prefix}{name}_ever_frac"] = float((ever[name] & ran).sum().item()) / n_ran

    for key, value in cumulative_counts.items():
        log[f"{prefix}{key}"] = float(value)

    if kind is None or kind_names is None:
        return log

    # PER-BRANCH. The whole point of the split: the recipe's target is stated on TRANSPORT-kind
    # episodes (they are the ones that carry the leg to a tip-down pose, which is what S1
    # generation depends on), and A1's collapse tripwire is stated on CLASSIC-kind episodes (they
    # are the original task whose loss is the collapse). An aggregate over all four branches
    # muffles both signals exactly when they matter.
    for kind_value, kind_label in sorted(kind_names.items()):
        in_kind = ran & (kind == kind_value)
        n_kind = int(in_kind.sum().item())
        # Published even at zero, as a count, so "this branch produced no episodes this batch" and
        # "this branch produced episodes that all failed" are distinguishable in the dashboard --
        # the exact confusion F29 records for `seated: 0` and F15 for `Curriculum/adr`.
        log[f"{prefix}episodes/{kind_label}"] = float(n_kind)
        if n_kind == 0:
            continue
        for name in required:
            log[f"{prefix}{name}_atend_frac/{kind_label}"] = (
                float((atend[name] & in_kind).sum().item()) / n_kind
            )
    return log
