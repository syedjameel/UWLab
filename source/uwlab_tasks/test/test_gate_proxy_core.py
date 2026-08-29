# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit-proves the training-time gate proxy's core (``V2_REPOSE_RECIPE.md`` sec 4, bead
``dr-tlx.2``) and the shared ``passive_gates`` factoring, without touching Isaac at all.

Needs ``torch`` but no Isaac Sim, no GPU, no env construction -- same technique and same reason as
``test_c3_transport_stage.py`` and ``test_held_check_core.py`` next to this file: both
``gate_proxy_core.py`` and ``held_check_core.py`` have no ``isaaclab`` import by design, so they are
loaded here BY FILE PATH.

Run with ``/home/dom-iva/.cache/simdist-cpu-venv/bin/python3`` -- the plain ``python3`` on this host
has neither torch nor pytest.

WHAT THIS CANNOT COVER, said so it is not mistaken for full coverage: the Isaac-touching half
(``mdp/gate_proxy.py``'s scene reads, the ``TerminationManager`` reset timing that makes ``_atend``
mean "at the terminal step", and the config wiring) needs Isaac Sim and is out of scope. What is
covered is every number the metric actually reports, plus the F27 guarantee that the proxy and the
generator share one gate implementation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_MDP = Path(__file__).resolve().parents[1] / "uwlab_tasks/manager_based/manipulation/dexlift/mdp"


def _load_by_path(name: str, path: Path):
    """Load an Isaac-free module by FILE PATH, COMPILING THE SOURCE TEXT (bead dr-76w.22).

    NOT ``spec_from_file_location(...).loader.exec_module(...)``: that consults and writes
    ``__pycache__``, and CPython's staleness check compares source mtime at ONE-SECOND granularity
    against the ``.pyc`` header, so an edit/run/restore cycle completed inside one second leaves a
    ``.pyc`` that looks valid for the restored source. The negative controls below MUTATE THE SOURCE
    ON DISK and restore it, which is exactly that cycle -- see ``test_c3_transport_stage.py``'s copy
    of this helper for the four phantom failures it produced there.
    """
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)  # noqa: S102
    return module


_core = _load_by_path("gate_proxy_core", _MDP / "gate_proxy_core.py")
_held = _load_by_path("held_check_core", _MDP / "held_check_core.py")

PASSIVE_GATE_NAMES = _core.PASSIVE_GATE_NAMES
PASSIVE_ALL_NAME = _core.PASSIVE_ALL_NAME
evaluate_priority_chain = _core.evaluate_priority_chain
build_log_entries = _core.build_log_entries
passive_gates = _held.passive_gates
held_decision = _held.held_decision

KIND_NAMES = {0: "classic", 1: "low_goal", 2: "partial_assembly", 3: "transport"}


def _raises(exc_type, fn, *args, **kwargs):
    """``pytest.raises`` without pytest (bead dr-76w.22).

    A suite that ``import pytest`` under the plain interpreter ABORTS at the first such case and
    exits non-zero on ModuleNotFoundError -- which reads as an environment problem rather than as N
    unrun tests. This helper keeps the ``__main__`` runner below working end to end while still
    collecting normally under pytest.
    """
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"expected {exc_type.__name__} from {getattr(fn, '__name__', fn)}(...)")


def _b(*values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bool)


# ---------------------------------------------------------------------------------------------
# passive_gates -- the shared implementation, and the thresholds it must honour
# ---------------------------------------------------------------------------------------------


def test_passive_gates_thresholds_are_held_checks_own():
    # V2_REPOSE_RECIPE.md sec 4.1 / held_check.py:239: co_move is relative_speed < 0.05 m/s AND
    # |obj_vz| < 0.1 m/s, and settled is episode_length > SETTLE_STEPS = 60.
    assert _held.SETTLE_STEPS == 60
    steps = torch.tensor([61.0, 61.0, 61.0, 61.0])
    loaded = _b(True, True, True, True)
    settled, opposed, co_move = passive_gates(
        steps_since_reset=steps,
        thumb_loaded=loaded,
        tip_loaded=loaded,
        relative_speed=torch.tensor([0.049, 0.051, 0.0, 0.0]),
        obj_vz=torch.tensor([0.0, 0.0, 0.099, 0.101]),
    )
    assert settled.all() and opposed.all()
    assert co_move.tolist() == [True, False, True, False]


def test_settled_is_strictly_greater_than_settle_steps():
    zero = torch.zeros(2)
    settled, _, _ = passive_gates(
        steps_since_reset=torch.tensor([60.0, 61.0]),
        thumb_loaded=_b(True, True),
        tip_loaded=_b(True, True),
        relative_speed=zero,
        obj_vz=zero,
    )
    assert settled.tolist() == [False, True]


def test_opposed_contact_needs_both_a_thumb_and_a_non_thumb():
    zero = torch.zeros(4)
    _, opposed, _ = passive_gates(
        steps_since_reset=torch.full((4,), 61.0),
        thumb_loaded=_b(True, True, False, False),
        tip_loaded=_b(True, False, True, False),
        relative_speed=zero,
        obj_vz=zero,
    )
    assert opposed.tolist() == [True, False, False, False]


def test_held_decision_uses_the_same_passive_gates_this_module_reports():
    """THE F27 GUARANTEE, checked rather than asserted in a comment.

    The proxy's whole claim is that it computes the SAME three gates the generator's ``held``
    predicate computes. If ``held_decision`` ever stopped routing through ``passive_gates`` -- or
    routed through it with different thresholds -- the proxy would keep publishing a number that
    predicts nothing. So: wherever the passive three are False, ``held_decision`` must be False, no
    matter how perfect the probe evidence is.
    """
    n = 16
    torch.manual_seed(0)
    steps = torch.randint(0, 120, (n,)).float()
    thumb = torch.rand(n) > 0.4
    tip = torch.rand(n) > 0.4
    rel = torch.rand(n) * 0.1
    vz = (torch.rand(n) - 0.5) * 0.3

    settled, opposed, co_move = passive_gates(
        steps_since_reset=steps, thumb_loaded=thumb, tip_loaded=tip, relative_speed=rel, obj_vz=vz
    )
    perfect_disp = torch.ones(n, 3) * 0.02
    held = held_decision(
        steps_since_reset=steps,
        thumb_loaded=thumb,
        tip_loaded=tip,
        relative_speed=rel,
        obj_vz=vz,
        probe_ready=torch.ones(n, dtype=torch.bool),
        obj_disp_probe=perfect_disp,
        gripper_disp_probe=perfect_disp,
    )
    passive_all = settled & opposed & co_move
    # With the probe made perfect, held is EXACTLY the passive three.
    assert torch.equal(held, passive_all)
    assert not bool((held & ~passive_all).any())


# ---------------------------------------------------------------------------------------------
# evaluate_priority_chain -- F28's table shape, and F29's reach counts
# ---------------------------------------------------------------------------------------------


def test_first_fail_masks_and_pass_partition_ran_exactly():
    """The invariant that makes the histogram readable: every episode that RAN lands in exactly one
    of {first_fail_settled, first_fail_opposed_contact, first_fail_co_move, passive_three}."""
    torch.manual_seed(1)
    n = 200
    ran = torch.rand(n) > 0.1
    chain = evaluate_priority_chain(
        settled=torch.rand(n) > 0.3,
        opposed_contact=torch.rand(n) > 0.3,
        co_move=torch.rand(n) > 0.5,
        ran=ran,
    )
    buckets = (
        chain["first_fail_settled"].int()
        + chain["first_fail_opposed_contact"].int()
        + chain["first_fail_co_move"].int()
        + chain[PASSIVE_ALL_NAME].int()
    )
    assert torch.equal(buckets, ran.int()), "each ran episode must fall in exactly one bucket"


def test_priority_order_attributes_to_the_earliest_failing_gate():
    # An episode failing BOTH opposed_contact and co_move is counted against opposed_contact only
    # -- the generator's counter is first-failing-gate in priority order (F28), and a proxy that
    # counted it twice, or against the later gate, would not be comparable to that table.
    chain = evaluate_priority_chain(
        settled=_b(True), opposed_contact=_b(False), co_move=_b(False), ran=_b(True)
    )
    assert chain["first_fail_opposed_contact"].tolist() == [True]
    assert chain["first_fail_co_move"].tolist() == [False]
    assert chain["reached_co_move"].tolist() == [False]


def test_reach_counts_make_a_zero_interpretable_f29():
    """F29's stated remedy: "a priority-ordered failure counter must also report how many episodes
    reached each gate, or its zeros are uninterpretable" -- the defect that produced ``seated: 0``.

    Two populations with an IDENTICAL zero in ``first_fail_co_move`` and opposite meanings.
    """
    nothing_got_there = evaluate_priority_chain(
        settled=_b(False, False), opposed_contact=_b(True, True), co_move=_b(True, True), ran=_b(True, True)
    )
    everything_passed = evaluate_priority_chain(
        settled=_b(True, True), opposed_contact=_b(True, True), co_move=_b(True, True), ran=_b(True, True)
    )
    assert int(nothing_got_there["first_fail_co_move"].sum()) == 0
    assert int(everything_passed["first_fail_co_move"].sum()) == 0
    # The zeros are identical; the reach counts are what tell them apart.
    assert int(nothing_got_there["reached_co_move"].sum()) == 0
    assert int(everything_passed["reached_co_move"].sum()) == 2


def test_episodes_that_did_not_run_are_excluded_everywhere():
    # The construction-time reset produces zero-length episodes for EVERY env. Counting them would
    # put a large fake spike in every series at iteration 0.
    chain = evaluate_priority_chain(
        settled=_b(False, False), opposed_contact=_b(False, False), co_move=_b(False, False), ran=_b(False, False)
    )
    for key, mask in chain.items():
        assert int(mask.sum()) == 0, key


def test_f28_histogram_reproduces_its_own_reported_rate():
    """The recipe's decomposition, replayed through this code (V2_REPOSE_RECIPE.md sec 4.2).

    F28/F30's S1 gated run: 141 attempts, first-failing gate settled 8 / opposed_contact 24 /
    co_move 66, leaving 43 that clear the passive three, of which 13 were finally accepted.
    43/141 = 0.3050 and 13/43 = 0.3023, and 0.3050 * 0.3023 = 0.0922 = the reported 9.22%.
    """
    n, n_settled_fail, n_opposed_fail, n_comove_fail = 141, 8, 24, 66
    settled = torch.ones(n, dtype=torch.bool)
    opposed = torch.ones(n, dtype=torch.bool)
    co_move = torch.ones(n, dtype=torch.bool)
    settled[:n_settled_fail] = False
    opposed[n_settled_fail : n_settled_fail + n_opposed_fail] = False
    start = n_settled_fail + n_opposed_fail
    co_move[start : start + n_comove_fail] = False

    chain = evaluate_priority_chain(
        settled=settled, opposed_contact=opposed, co_move=co_move, ran=torch.ones(n, dtype=torch.bool)
    )
    assert int(chain["first_fail_settled"].sum()) == 8
    assert int(chain["first_fail_opposed_contact"].sum()) == 24
    assert int(chain["first_fail_co_move"].sum()) == 66
    passive_pass = int(chain[PASSIVE_ALL_NAME].sum())
    assert passive_pass == 43
    assert abs(passive_pass / n - 0.3050) < 5e-4
    assert abs((passive_pass / n) * (13 / passive_pass) - 0.0922) < 5e-4
    # And the hard bound the recipe leans on: 0.305 is far below 0.50, so this checkpoint provably
    # could not have met R2 -- readable WITHOUT running a generation pass, which is the point.
    assert passive_pass / n < 0.50


# ---------------------------------------------------------------------------------------------
# build_log_entries
# ---------------------------------------------------------------------------------------------


def _entries(atend_vals, ever_vals, ran, kind=None, kind_names=None, cumulative=None):
    names = (*PASSIVE_GATE_NAMES, PASSIVE_ALL_NAME)
    return build_log_entries(
        atend=dict(zip(names, atend_vals)),
        ever=dict(zip(names, ever_vals)),
        ran=ran,
        cumulative_counts={} if cumulative is None else cumulative,
        kind=kind,
        kind_names=kind_names,
    )


def test_atend_and_ever_are_reported_separately_and_ever_reads_higher():
    # F29's lesson applied to the instrument's own evaluation moment: the generator evaluates ONCE
    # at episode end, so _atend is the variant that predicts it; _ever is the sticky-OR protocol
    # and reads higher by construction. Substituting one for the other is the whole trap.
    atend = [_b(False, False), _b(True, True), _b(True, True), _b(False, False)]
    ever = [_b(True, True), _b(True, True), _b(True, True), _b(True, False)]
    log = _entries(atend, ever, ran=_b(True, True))
    assert log["GateProxy/settled_atend_frac"] == 0.0
    assert log["GateProxy/settled_ever_frac"] == 1.0
    assert log["GateProxy/passive_three_atend_frac"] == 0.0
    assert log["GateProxy/passive_three_ever_frac"] == 0.5


def test_no_series_is_published_when_no_episode_ran():
    # Publishing zeros for the construction-time reset would be a lie in every series at once.
    ones = [_b(True, True)] * 4
    assert _entries(ones, ones, ran=_b(False, False)) == {}


def test_fractions_use_the_ran_denominator_not_the_tensor_length():
    atend = [_b(True, False), _b(True, False), _b(True, False), _b(True, False)]
    log = _entries(atend, atend, ran=_b(True, False))
    # 1 of 1 episodes that ran, not 1 of 2 slots.
    assert log["GateProxy/passive_three_atend_frac"] == 1.0


def test_per_kind_split_reports_each_branch_separately():
    # The recipe's target is stated on TRANSPORT-kind episodes and A1's tripwire on CLASSIC-kind;
    # an aggregate muffles both exactly when they matter (V2_REPOSE_RECIPE.md sec 3.1, 4.2).
    kind = torch.tensor([0, 0, 3, 3])
    atend = [_b(True, True, True, True)] * 3 + [_b(False, False, True, True)]
    log = _entries(atend, atend, ran=_b(True, True, True, True), kind=kind, kind_names=KIND_NAMES)
    assert log["GateProxy/passive_three_atend_frac"] == 0.5
    assert log["GateProxy/passive_three_atend_frac/classic"] == 0.0
    assert log["GateProxy/passive_three_atend_frac/transport"] == 1.0
    assert log["GateProxy/episodes/classic"] == 2.0
    assert log["GateProxy/episodes/transport"] == 2.0


def test_a_branch_with_no_episodes_publishes_a_zero_count_and_no_rate():
    # "No episodes of this kind finished" and "they all failed" must be distinguishable -- the
    # conflation F15 records for Curriculum/adr, which reads 0.0 for both.
    kind = torch.tensor([0, 0])
    ones = [_b(True, True)] * 4
    log = _entries(ones, ones, ran=_b(True, True), kind=kind, kind_names=KIND_NAMES)
    assert log["GateProxy/episodes/transport"] == 0.0
    assert "GateProxy/passive_three_atend_frac/transport" not in log
    assert log["GateProxy/passive_three_atend_frac/classic"] == 1.0


def test_kind_without_kind_names_raises_rather_than_inventing_labels():
    # The kind integers are episode_mixture.EPISODE_KIND_*; this module may not restate them (F27),
    # so a caller must name the convention -- the same guard c3_transport_core applies to tilt_rad.
    ones = [_b(True)] * 4
    exc = _raises(ValueError, _entries, ones, ones, _b(True), torch.tensor([0]), None)
    assert "kind_names is required" in str(exc)


def test_a_missing_gate_key_raises_rather_than_publishing_a_partial_table():
    names = (*PASSIVE_GATE_NAMES, PASSIVE_ALL_NAME)
    full = {name: _b(True) for name in names}
    partial = {name: _b(True) for name in names if name != "co_move"}
    exc = _raises(
        ValueError,
        build_log_entries,
        atend=partial,
        ever=full,
        ran=_b(True),
        cumulative_counts={},
        kind=None,
        kind_names=None,
    )
    assert "co_move" in str(exc)


def test_cumulative_counts_are_passed_through_under_the_prefix():
    ones = [_b(True)] * 4
    log = _entries(ones, ones, ran=_b(True), cumulative={"reached_co_move": 7, "episodes": 11})
    assert log["GateProxy/reached_co_move"] == 7.0
    assert log["GateProxy/episodes"] == 11.0


def test_per_kind_counts_are_summable_across_batches():
    """R0's baseline needs COUNTS to add up over a rollout, not per-batch fractions over three
    episodes (V2_REPOSE_RECIPE.md O6). Two batches, summed, must equal the one-shot answer."""
    per_kind_counts = _core.per_kind_counts

    def counts(settled, opposed, co_move, ran, kind):
        chain = evaluate_priority_chain(
            settled=settled, opposed_contact=opposed, co_move=co_move, ran=ran
        )
        return per_kind_counts(chain, ran, kind, KIND_NAMES)

    a = counts(_b(True, True), _b(True, False), _b(True, True), _b(True, True), torch.tensor([3, 3]))
    b = counts(_b(True, False), _b(True, True), _b(True, True), _b(True, True), torch.tensor([3, 0]))
    total = {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}
    assert total["episodes/transport"] == 3
    assert total["passive_three/transport"] == 2
    assert total["first_fail_opposed_contact/transport"] == 1
    assert total["episodes/classic"] == 1
    assert total["first_fail_settled/classic"] == 1
    # The per-branch episode counts must partition the batch, or the baseline divides by the wrong
    # denominator without any error.
    assert sum(v for k, v in total.items() if k.startswith("episodes/")) == 4


def test_per_kind_counts_are_empty_without_a_mixture():
    # A run with no episode mixture must report NOTHING per branch rather than labelling every
    # episode `classic` -- the metric must not conjure the structure it is measuring.
    chain = evaluate_priority_chain(
        settled=_b(True), opposed_contact=_b(True), co_move=_b(True), ran=_b(True)
    )
    assert _core.per_kind_counts(chain, _b(True), None, KIND_NAMES) == {}
    assert _core.per_kind_counts(chain, _b(True), torch.tensor([0]), None) == {}


# ---------------------------------------------------------------------------------------------
# NEGATIVE CONTROLS -- each mutates the source on disk, re-loads, and asserts the suite NOTICES.
# A test that passes against broken code is worse than no test.
# ---------------------------------------------------------------------------------------------


def _with_mutated_source(path: Path, old: str, new: str, check, load: bool = True):
    """Swap a snippet in the source ON DISK, run ``check``, and always restore.

    ``load=False`` for a module that cannot be imported here at all (``gate_proxy.py`` imports
    isaaclab at module scope). Those checks read the mutated source with ast instead, so ``check``
    is called with ``None`` -- the mutation is still real and still on disk, which is the part that
    makes the control meaningful.
    """
    original = path.read_text()
    assert original.count(old) == 1, f"negative control anchor not unique in {path.name}: {old!r}"
    try:
        path.write_text(original.replace(old, new))
        check(_load_by_path(f"_mutated_{path.stem}", path) if load else None)
    finally:
        path.write_text(original)
        # Restore the real modules for any test that runs after this one.
        _load_by_path("gate_proxy_core", _MDP / "gate_proxy_core.py")
        _load_by_path("held_check_core", _MDP / "held_check_core.py")


def test_negative_control_comove_threshold_change_is_detected():
    """If someone retunes the co-move speed threshold in held_check_core, the proxy MUST move with
    it -- that is the entire value of sharing one implementation. Prove the test above would catch
    a divergence by making the threshold disagree with the generator's documented 0.05."""

    def check(mutated):
        _, _, co_move = mutated.passive_gates(
            steps_since_reset=torch.tensor([61.0]),
            thumb_loaded=_b(True),
            tip_loaded=_b(True),
            relative_speed=torch.tensor([0.051]),
            obj_vz=torch.tensor([0.0]),
        )
        # Under the mutation 0.051 now passes; under the real source it does not.
        assert bool(co_move[0]) is True, "mutation did not take effect -- the control proves nothing"

    # Anchored on the SINGLE named constant, which is a strictly better control than the old
    # per-signature default it replaced: it also proves the constant actually propagates to both
    # passive_gates and held_decision rather than each carrying its own copy.
    _with_mutated_source(
        _MDP / "held_check_core.py",
        "COMOVE_SPEED_THRESH = 0.05",
        "COMOVE_SPEED_THRESH = 0.5",
        check,
    )
    # And the real source rejects it again, so the mutation really was the cause.
    _, _, co_move = passive_gates(
        steps_since_reset=torch.tensor([61.0]),
        thumb_loaded=_b(True),
        tip_loaded=_b(True),
        relative_speed=torch.tensor([0.051]),
        obj_vz=torch.tensor([0.0]),
    )
    assert bool(co_move[0]) is False


def test_negative_control_dropping_the_ran_mask_is_detected():
    """The zero-length-episode guard is load-bearing (a fake spike in every series at iteration 0).
    Remove it and the suite must notice."""

    def check(mutated):
        chain = mutated.evaluate_priority_chain(
            settled=_b(True, True), opposed_contact=_b(True, True), co_move=_b(True, True), ran=_b(False, False)
        )
        assert int(chain[PASSIVE_ALL_NAME].sum()) == 2, "mutation did not take effect"

    _with_mutated_source(
        _MDP / "gate_proxy_core.py",
        "    passed_all = reached_co_move & co_move",
        "    passed_all = settled & opposed_contact & co_move",
        check,
    )
    chain = evaluate_priority_chain(
        settled=_b(True, True), opposed_contact=_b(True, True), co_move=_b(True, True), ran=_b(False, False)
    )
    assert int(chain[PASSIVE_ALL_NAME].sum()) == 0


def test_negative_control_priority_order_collapse_is_detected():
    """If ``first_fail_*`` stopped being priority-ordered, an episode failing two gates would be
    counted twice and the partition invariant would break."""

    def check(mutated):
        chain = mutated.evaluate_priority_chain(
            settled=_b(True), opposed_contact=_b(False), co_move=_b(False), ran=_b(True)
        )
        counted = (
            int(chain["first_fail_opposed_contact"].sum())
            + int(chain["first_fail_co_move"].sum())
            + int(chain[PASSIVE_ALL_NAME].sum())
        )
        assert counted == 2, "mutation did not take effect -- expected the double-count"

    _with_mutated_source(
        _MDP / "gate_proxy_core.py",
        '        "first_fail_co_move": reached_co_move & ~co_move,',
        '        "first_fail_co_move": ran & ~co_move,',
        check,
    )
    chain = evaluate_priority_chain(
        settled=_b(True), opposed_contact=_b(False), co_move=_b(False), ran=_b(True)
    )
    assert int(chain["first_fail_co_move"].sum()) == 0


# ---------------------------------------------------------------------------------------------
# THE BANNER MUST NOT GO STALE. gate_proxy.py imports isaaclab, so this reads it with ast.
# ---------------------------------------------------------------------------------------------


def _banner_code_without_docstring() -> str:
    """Source of ``gate_proxy_banner``'s executable body, docstring EXCLUDED.

    The docstring deliberately quotes the stale literals it exists to warn about
    ("settled > 60 steps ... as LITERALS"), so a naive scan of the whole function flags the very
    comment that documents the fix. Strip it with ast and scan only what runs.
    """
    import ast

    tree = ast.parse((_MDP / "gate_proxy.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "gate_proxy_banner":
            body = node.body[1:] if ast.get_docstring(node) is not None else node.body
            return "\n".join(ast.unparse(stmt) for stmt in body)
    raise AssertionError("gate_proxy_banner not found")


def test_banner_interpolates_thresholds_and_states_no_literal():
    """The banner must PRINT the thresholds it actually uses.

    An earlier revision spelled them out -- "settled > 60 steps, relative speed < 0.05 m/s" -- in
    the one place a stale number does most damage, because a banner is what a reader checks a run's
    staging against (RESET_SPEC_V2.md sec 1a trap 3) and the R0/ramp launchers grep for it.
    V2_POSE_FINDINGS.md F42 is this repo's own record of in-tree documentation going stale that way.
    """
    body = _banner_code_without_docstring()
    for literal in ("> 60 steps", "< 0.05 m/s", "< 0.1 m/s"):
        assert literal not in body, f"banner hardcodes {literal!r}; interpolate it instead"
    for key in ("settle_steps", "comove_speed_thresh", "comove_vz_thresh"):
        assert "GATE_PROXY_DEFAULTS" in body and key in body, f"banner does not interpolate {key}"


def test_gate_proxy_defaults_come_from_held_check_core_not_a_copy():
    """GATE_PROXY_DEFAULTS must reference held_check_core's names, never repeat their values.

    SETTLE_STEPS has a SECOND consumer -- c3_rung.py imports it as S_t's goal re-pin step floor
    (bead dr-ai1.18) -- so a local copy in the gate proxy would silently describe a different
    subsystem's timing as well as its own.
    """
    src = (_MDP / "gate_proxy.py").read_text()
    start = src.index("GATE_PROXY_DEFAULTS: dict[str, float] = {")
    body = src[start : src.index("}", start)]
    assert '"settle_steps": SETTLE_STEPS' in body
    assert '"comove_speed_thresh": COMOVE_SPEED_THRESH' in body
    assert '"comove_vz_thresh": COMOVE_VZ_THRESH' in body
    for numeral in ("60", "0.05", "0.1"):
        assert numeral not in body, f"GATE_PROXY_DEFAULTS restates the literal {numeral}"
    # And the names it references really are held_check_core's, with the documented values.
    assert _held.SETTLE_STEPS == 60
    assert _held.COMOVE_SPEED_THRESH == 0.05
    assert _held.COMOVE_VZ_THRESH == 0.1


def test_naming_the_comove_thresholds_did_not_change_their_values():
    """Naming COMOVE_SPEED_THRESH / COMOVE_VZ_THRESH must be a pure rename.

    held_check_core now has a second subsystem depending on it, so a "harmless" retune smuggled in
    with a refactor would change S_t's re-pin timing and the generator's held predicate at once.
    These are the values every v1 number in V2_POSE_FINDINGS.md F28/F30 was measured under.
    """
    import inspect

    for fn in (_held.held_decision, _held.passive_gates):
        params = inspect.signature(fn).parameters
        assert params["comove_speed_thresh"].default == 0.05, fn.__name__
        assert params["comove_vz_thresh"].default == 0.1, fn.__name__
        assert params["settle_steps"].default == 60, fn.__name__


def test_negative_control_a_stale_banner_literal_is_detected():
    """Prove the staleness check fires: put a literal back and confirm the suite objects."""

    def check(_module):
        try:
            test_banner_interpolates_thresholds_and_states_no_literal()
        except AssertionError as exc:
            assert "hardcodes" in str(exc)
            return
        raise AssertionError("the staleness check did not fire on a hardcoded literal")

    _with_mutated_source(
        _MDP / "gate_proxy.py",
        "f\" not restated: settled > {GATE_PROXY_DEFAULTS['settle_steps']} steps, relative speed <\"",
        '" not restated: settled > 60 steps, relative speed <"',
        check,
        load=False,
    )
    test_banner_interpolates_thresholds_and_states_no_literal()


# ---------------------------------------------------------------------------------------------
# The IsaacLab TERM-PARAM CONTRACT, checked statically (ast only -- these modules import isaaclab).
# ---------------------------------------------------------------------------------------------


def _call_signature_and_params(module_path: Path, class_name: str, factory_name: str):
    """Extract ``__call__``'s argument names and the factory's ``params`` keys with ast.

    ast rather than ``inspect``: ``gate_proxy.py`` and ``success.py`` import isaaclab at module
    scope, so they cannot be imported by this suite at all. The contract is still checkable from
    the source text, and it is worth checking -- see the test below for what it costs to get wrong.
    """
    import ast

    tree = ast.parse(module_path.read_text())
    call_args: list[str] = []
    with_defaults: list[str] = []
    params: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__call__":
                    names = [a.arg for a in item.args.args]
                    n_def = len(item.args.defaults)
                    call_args = names
                    with_defaults = names[len(names) - n_def :] if n_def else []
        if isinstance(node, ast.FunctionDef) and node.name == factory_name:
            for item in ast.walk(node):
                if isinstance(item, ast.keyword) and item.arg == "params" and isinstance(item.value, ast.Dict):
                    params = [k.value for k in item.value.keys if isinstance(k, ast.Constant)]
    assert call_args, f"no __call__ found on {class_name}"
    assert params, f"no params dict found in {factory_name}"
    return call_args, with_defaults, params


def _assert_isaaclab_param_contract(module_path: Path, class_name: str, factory_name: str):
    """Replicate ``ManagerBase._resolve_common_term_cfg``'s static check exactly.

    ``isaaclab/managers/manager_base.py:357-370``, for a ManagerTermBase subclass (``min_argc``
    forwarded by 1 to account for ``self``)::

        args = args_without_defaults + args_with_defaults
        if len(args) > min_argc:
            if set(args[min_argc:]) != set(term_params + args_with_defaults):
                raise ValueError(...)

    THIS IS WHY THE TEST EXISTS. A param consumed only in ``__init__`` still has to appear in
    ``__call__``'s signature, because the check compares SETS and a params key with no matching
    argument fails it -- at env construction, before the sim starts. A ``**kwargs`` catch-all does
    not rescue it either: ``kwargs`` is then itself counted as an argument without a default and
    fails the same comparison. Adding the per-kind split to these two terms hit exactly this, and
    the failure surfaces only when Isaac constructs the env -- i.e. ~92 s into a job, on the GPU
    host, which is the most expensive possible place to learn it.
    """
    call_args, with_defaults, params = _call_signature_and_params(module_path, class_name, factory_name)
    without_defaults = [a for a in call_args if a not in with_defaults]
    args = without_defaults + with_defaults
    min_argc = 2  # env + self
    if len(args) > min_argc:
        assert set(args[min_argc:]) == set(params + with_defaults), (
            f"{class_name}.__call__ args {sorted(set(args[min_argc:]))} do not match"
            f" {factory_name}'s params {sorted(set(params + with_defaults))} -- IsaacLab's"
            " ManagerBase._resolve_common_term_cfg would raise at env construction."
        )
    assert "kwargs" not in call_args, (
        f"{class_name}.__call__ must not use **kwargs: IsaacLab counts `kwargs` as an argument"
        " without a default and the same set comparison fails."
    )


def test_gate_proxy_term_satisfies_the_isaaclab_param_contract():
    _assert_isaaclab_param_contract(_MDP / "gate_proxy.py", "GateProxyLogger", "gate_proxy_log_term_cfg")


def test_success_rate_term_still_satisfies_the_isaaclab_param_contract():
    # The per-kind split added two params to this pre-existing term. Without the matching two
    # arguments on __call__, EVERY dexlift env in this family would fail to construct.
    _assert_isaaclab_param_contract(_MDP / "success.py", "EpisodeSuccessRateLogger", "success_rate_log_term_cfg")


def test_negative_control_a_params_key_with_no_call_argument_is_detected():
    """Prove the contract check above actually fires, by adding a params key and nothing else."""

    def check(_module):
        try:
            _assert_isaaclab_param_contract(
                _MDP / "gate_proxy.py", "GateProxyLogger", "gate_proxy_log_term_cfg"
            )
        except AssertionError as exc:
            assert "do not match" in str(exc)
            return
        raise AssertionError("the contract check did not fire on a params key with no argument")

    _with_mutated_source(
        _MDP / "gate_proxy.py",
        '            "log_key_prefix": log_key_prefix,\n        },\n        time_out=False,',
        '            "log_key_prefix": log_key_prefix,\n            "unmatched_key": None,\n'
        "        },\n        time_out=False,",
        check,
        load=False,
    )
    _assert_isaaclab_param_contract(_MDP / "gate_proxy.py", "GateProxyLogger", "gate_proxy_log_term_cfg")


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            passed += 1
            print(f"[gate_proxy] {name} OK", flush=True)
    print(f"[gate_proxy] all {passed} tests passed", flush=True)
