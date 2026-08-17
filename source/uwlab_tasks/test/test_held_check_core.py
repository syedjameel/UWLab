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
    print("ALL PASS")
