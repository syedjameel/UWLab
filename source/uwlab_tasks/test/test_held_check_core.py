# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit-proves the HELD check's decisive gate (bead UWLab-dwx.2).

Needs only torch -- no Isaac Sim, no GPU, no env construction. That is the point: the adversarial
case this check exists to reject (an object already at its target pose, hand resting nearby, not
actually gripping) is constructed here as plain tensors, and the test asserts the check rejects it.
If it did not, that would be reported as the headline finding rather than patched around -- see
``dexlift/mdp/held_check_core.py``'s module docstring for why gates 1-3 alone cannot catch this
case and only the probe-displacement gate can.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

# Loaded by file path, NOT via `import uwlab_tasks...` -- the `uwlab_tasks` package's own
# __init__.py transitively imports isaaclab_tasks -> isaaclab -> omni.kit.app, which requires a
# running Isaac Sim app. held_check_core.py has no such dependency (torch only, by design -- see
# its module docstring), and loading it this way is what lets this test run with plain python.
_CORE_PATH = (
    Path(__file__).resolve().parents[1]
    / "uwlab_tasks/manager_based/manipulation/dexlift/mdp/held_check_core.py"
)
_spec = importlib.util.spec_from_file_location("held_check_core", _CORE_PATH)
_held_check_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_held_check_core)
held_decision = _held_check_core.held_decision


def _base_kwargs(n: int = 1) -> dict:
    """Fully-passing defaults for every gate except the probe, which callers fill in."""
    return dict(
        steps_since_reset=torch.full((n,), 100.0),
        thumb_loaded=torch.ones(n, dtype=torch.bool),
        tip_loaded=torch.ones(n, dtype=torch.bool),
        relative_speed=torch.zeros(n),
        obj_vz=torch.zeros(n),
        probe_ready=torch.ones(n, dtype=torch.bool),
    )


def test_adversarial_case_rejected():
    """Option-B: object at rest at (near) its target pose, hand idle nearby with incidental
    thumb+tip contact -- gates 1-3 (settled, opposed-contact, co-move) all pass, exactly the
    situation the bead's brief describes. The generator jogs the gripper 10 mm during the probe;
    an object merely resting nearby does not move with it.
    """
    kw = _base_kwargs()
    gripper_disp = torch.tensor([[0.0, 0.0, 0.010]])  # 10 mm commanded jog
    obj_disp = torch.tensor([[0.0, 0.0, 0.0002]])  # object barely moves (not actually held)

    held = held_decision(obj_disp_probe=obj_disp, gripper_disp_probe=gripper_disp, **kw)

    assert not bool(held[0]), (
        "HELD CHECK FAILED TO REJECT THE ADVERSARIAL CASE. Gates 1-3 (settled, opposed contact, "
        "co-move) were all constructed to pass, exactly like a motionless correctly-posed object "
        "beside an idle hand. If this assertion fails, the check is wrong -- do not weaken this "
        "test to make it pass; fix held_decision (or its probe-displacement gate specifically) "
        "instead."
    )


def test_positive_case_accepted():
    """Same settled/contact/co-move state, but this time the object genuinely tracks the gripper's
    probe jog (rigidly held), within tolerance.
    """
    kw = _base_kwargs()
    gripper_disp = torch.tensor([[0.0, 0.0, 0.010]])
    obj_disp = torch.tensor([[0.0, 0.0, 0.0098]])  # tracks the jog closely

    held = held_decision(obj_disp_probe=obj_disp, gripper_disp_probe=gripper_disp, **kw)

    assert bool(held[0]), "A genuinely held object (displacement tracks the probe) must be accepted."


def test_probe_not_ready_is_undecided_not_true():
    """Before a probe window has completed, held must read False (undecided), never True, even if
    every other gate already passes -- otherwise an env could be accepted on its very first step.
    """
    kw = _base_kwargs()
    kw["probe_ready"] = torch.zeros(1, dtype=torch.bool)
    gripper_disp = torch.tensor([[0.0, 0.0, 0.010]])
    obj_disp = torch.tensor([[0.0, 0.0, 0.0098]])  # would pass if probe were ready

    held = held_decision(obj_disp_probe=obj_disp, gripper_disp_probe=gripper_disp, **kw)

    assert not bool(held[0])


def test_degenerate_zero_jog_does_not_trivially_pass():
    """If the gripper barely moves during the probe, obj_disp~=gripper_disp~=0 would trivially
    'track' -- the probe_min_disp floor exists to stop a near-zero jog from passing by default.
    """
    kw = _base_kwargs()
    gripper_disp = torch.tensor([[0.0, 0.0, 0.0005]])  # 0.5 mm, below probe_min_disp (3 mm default)
    obj_disp = torch.tensor([[0.0, 0.0, 0.0004]])

    held = held_decision(obj_disp_probe=obj_disp, gripper_disp_probe=gripper_disp, **kw)

    assert not bool(held[0])


def test_settle_gate_rejects_ballistic_post_reset_drop():
    """Mirrors probe_grasp.py's own reasoning: every early post-reset step reads as
    momentarily-plausible unless gated on time since reset, because a dropped/repositioned object
    is briefly in free fall/settling with nothing around it constrained yet.
    """
    kw = _base_kwargs()
    kw["steps_since_reset"] = torch.full((1,), 5.0)  # well under the 60-step settle threshold
    gripper_disp = torch.tensor([[0.0, 0.0, 0.010]])
    obj_disp = torch.tensor([[0.0, 0.0, 0.0098]])

    held = held_decision(obj_disp_probe=obj_disp, gripper_disp_probe=gripper_disp, **kw)

    assert not bool(held[0])


def test_missing_opposition_rejected():
    """Contact on only one side (e.g. resting on the palm, not pinched between opposing digits)
    must not pass -- this is dexlift's own opposition-gate semantics, reused not weakened.
    """
    kw = _base_kwargs()
    kw["tip_loaded"] = torch.zeros(1, dtype=torch.bool)
    gripper_disp = torch.tensor([[0.0, 0.0, 0.010]])
    obj_disp = torch.tensor([[0.0, 0.0, 0.0098]])

    held = held_decision(obj_disp_probe=obj_disp, gripper_disp_probe=gripper_disp, **kw)

    assert not bool(held[0])


def test_batched():
    """Vectorized over multiple envs at once: one adversarial, one positive, in the same call."""
    kw = _base_kwargs(n=2)
    gripper_disp = torch.tensor([[0.0, 0.0, 0.010], [0.0, 0.0, 0.010]])
    obj_disp = torch.tensor([[0.0, 0.0, 0.0002], [0.0, 0.0, 0.0098]])

    held = held_decision(obj_disp_probe=obj_disp, gripper_disp_probe=gripper_disp, **kw)

    assert held.tolist() == [False, True]



# ---------------------------------------------------------------------------------------------
# THE CHAIN AND ITS TRIGGER MUST AGREE (P0, 2026-08-29)
# ---------------------------------------------------------------------------------------------
#
# co_move was dropped from held_decision's AND and LEFT IN held_check.py's probe-arming condition.
# Because `_probe_ready` is cleared on every reset(), probe_ready==True within an episode implied
# co_move had been true at some step -- so acceptance still required a co_move-true instant. The
# gate had been moved from the chain to the trigger, not removed, and the population the change
# aimed at (a bore-constrained leg that never co_moves) was rejected one layer up without ever
# reaching `tracks` to be judged there.
#
# That is V2_POSE_FINDINGS.md F27 in a new shape: a gate's DEFINITION in held_check_core and its
# ENABLING CONDITION in held_check, each valid alone, checked against each other by nothing.
#
# STRUCTURAL, via ast, and it has to be: held_check.py imports isaaclab and cannot be imported in
# this environment at all. What can be proved without a GPU is that the trigger does not require a
# gate the chain has stopped requiring.

_HELD_CHECK_PATH = _CORE_PATH.parent / "held_check.py"
# held_check names the live values `<gate>_now` to distinguish them from the cached breakdown rows.
_NOW_SUFFIX = "_now"


def _names_in(node) -> set[str]:
    """Every bare identifier READ in an expression, with held_check's ``_now`` suffix normalised."""
    import ast

    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            name = n.id
            out.add(name[: -len(_NOW_SUFFIX)] if name.endswith(_NOW_SUFFIX) else name)
    return out


def _chain_gates() -> set[str]:
    """The identifiers held_decision's returned expression actually reads."""
    import ast

    for node in ast.walk(ast.parse(_CORE_PATH.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == "held_decision":
            returns = [n for n in ast.walk(node) if isinstance(n, ast.Return) and n.value is not None]
            assert len(returns) == 1, f"held_decision should have exactly one return; got {len(returns)}"
            return _names_in(returns[0].value)
    raise AssertionError("held_decision not found in held_check_core.py")


def _arming_gates(source: str | None = None) -> set[str]:
    """The identifiers held_check's probe-arming condition reads."""
    import ast

    src = source if source is not None else _HELD_CHECK_PATH.read_text()
    for node in ast.walk(ast.parse(src)):
        # Every assignment form, not just ast.Assign: an annotated `pre_probe_ok: Tensor = ...`
        # would otherwise slip past this walker silently.
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == "pre_probe_ok" for t in targets):
            return _names_in(node.value)
    raise AssertionError("pre_probe_ok assignment not found in held_check.py")


def test_probe_arming_does_not_reintroduce_a_dropped_gate():
    """THE ONE THAT MATTERS. Whatever the probe-arming condition requires, the chain must require
    too -- otherwise a gate removed from the AND still gates acceptance through the trigger, and
    the states the removal was meant to admit are rejected one layer earlier instead.
    """
    chain, arming = _chain_gates(), _arming_gates()
    extra = arming - chain
    assert not extra, (
        f"held_check.py's probe-arming condition requires {sorted(extra)}, which held_decision's"
        " chain does NOT. Because _probe_ready is cleared on every reset, probe_ready==True then"
        " implies those gates were true at some step -- so they still gate acceptance, one layer"
        " up, on the population the chain deliberately stopped rejecting. Remove them from"
        " pre_probe_ok, or put them back in the chain; do not leave the two disagreeing."
    )


def test_co_move_gates_neither_the_chain_nor_the_trigger():
    """The user's decision, stated as an assertion rather than left to the reader of two files."""
    assert "co_move" not in _chain_gates(), "co_move is back in held_decision's AND"
    assert "co_move" not in _arming_gates(), "co_move is back in held_check's probe-arming condition"


def test_negative_control_a_gate_left_only_in_the_trigger_is_detected():
    """Prove the guard above actually fires, by putting co_move back in the trigger and nothing
    else -- exactly the defect it was written for, which passed every other test in this file.
    """
    src = _HELD_CHECK_PATH.read_text()
    old = "pre_probe_ok = settled & opposed_contact_now"
    assert src.count(old) == 1, "negative control anchor is not unique; the control is not valid"
    mutated = src.replace(old, old + " & co_move_now")
    assert "co_move" in _arming_gates(mutated), "the mutation did not take"
    assert "co_move" not in _chain_gates()
    assert _arming_gates(mutated) - _chain_gates() == {"co_move"}, (
        "the guard would NOT have caught a gate left only in the trigger"
    )

if __name__ == "__main__":
    test_adversarial_case_rejected()
    print("PASS: test_adversarial_case_rejected")
    test_positive_case_accepted()
    print("PASS: test_positive_case_accepted")
    test_probe_not_ready_is_undecided_not_true()
    print("PASS: test_probe_not_ready_is_undecided_not_true")
    test_degenerate_zero_jog_does_not_trivially_pass()
    print("PASS: test_degenerate_zero_jog_does_not_trivially_pass")
    test_settle_gate_rejects_ballistic_post_reset_drop()
    print("PASS: test_settle_gate_rejects_ballistic_post_reset_drop")
    test_missing_opposition_rejected()
    print("PASS: test_missing_opposition_rejected")
    test_batched()
    print("PASS: test_batched")
    test_probe_arming_does_not_reintroduce_a_dropped_gate()
    print("PASS: test_probe_arming_does_not_reintroduce_a_dropped_gate")
    test_co_move_gates_neither_the_chain_nor_the_trigger()
    print("PASS: test_co_move_gates_neither_the_chain_nor_the_trigger")
    test_negative_control_a_gate_left_only_in_the_trigger_is_detected()
    print("PASS: test_negative_control_a_gate_left_only_in_the_trigger_is_detected")
    print("ALL PASS")
