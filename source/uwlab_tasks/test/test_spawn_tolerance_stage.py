# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit-proves the C3(S_t) spawn-pose-tolerance addon's core math (V2_C3_DESIGN.md sec 5,
V2_ACCEPTANCE_CRITERIA.md sec 4, bead dr-sj6.22) without touching Isaac at all.

Needs only ``torch`` -- no Isaac Sim, no GPU, no env construction. Same reason and same technique
as ``test_c1_hand_pose_stage.py``/``test_c3_transport_stage.py`` next to this file:
``spawn_tolerance_core.py`` has no ``isaaclab`` import by design, so it is loaded here BY FILE
PATH, not via ``import uwlab_tasks...`` (whose package ``__init__.py`` transitively pulls in
``isaaclab_tasks -> isaaclab -> omni.kit.app``).

This test covers ONLY the pure math/validation half: tolerance-config validation (no default,
raises loudly), the pos/rot distance formula, and the accept/reject predicate either side of a
given tolerance. It cannot prove the Isaac-touching half (GENERATION-side, per the layer split:
``scripts_v2/tools/generate_reset_states_policy.py``'s ``_SpawnPoseToleranceAddon``/
``SpawnToleranceHeldWithProbe``, living alongside that script's own ``_SeatingGateAddon`` -- the
lazy spawn-pose capture off ``env.scene``, and its composition with ``held_with_probe``) actually
wires correctly at runtime -- that needs Isaac Sim and is out of scope here. ``dexlift/mdp/c3_rung.py``/
``c3_rung_core.py`` (bead dr-ai1.4) are the separate ENV-side half (draws which half of C3 an
episode is, sets the spawn/goal) -- they define no acceptance predicate and are not exercised or
duplicated by anything in this file.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

_CORE_PATH = (
    Path(__file__).resolve().parents[1]
    / "uwlab_tasks/manager_based/manipulation/dexlift/mdp/spawn_tolerance_core.py"
)
_spec = importlib.util.spec_from_file_location("spawn_tolerance_core", _CORE_PATH)
_spawn_tolerance_core = importlib.util.module_from_spec(_spec)
sys.modules["spawn_tolerance_core"] = _spawn_tolerance_core
_spec.loader.exec_module(_spawn_tolerance_core)

SpawnToleranceConfig = _spawn_tolerance_core.SpawnToleranceConfig
SpawnPoseDisplacement = _spawn_tolerance_core.SpawnPoseDisplacement
pose_distance = _spawn_tolerance_core.pose_distance
within_spawn_tolerance = _spawn_tolerance_core.within_spawn_tolerance


def _quat_about_z(angle_rad: float) -> torch.Tensor:
    """(w, x, y, z) quaternion for a rotation of ``angle_rad`` about world +Z -- a simple, exactly
    computable test fixture (no dependency on this module's own _quat_mul/_quat_inv, so the
    rotation-distance test below is not self-confirming)."""
    half = angle_rad * 0.5
    return torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)])


_IDENTITY_QUAT = torch.tensor([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------------------------
# SpawnToleranceConfig: NO DEFAULT, raises loudly (bead dr-sj6.24 / V2_ACCEPTANCE_CRITERIA.md sec 4
# marks both tolerances OPEN -- this campaign has already shipped one invented constant, and this
# class exists specifically so a caller cannot repeat that by omission).
# ---------------------------------------------------------------------------------------------


def test_spawn_tolerance_config_raises_when_pos_tol_m_is_missing():
    import pytest

    with pytest.raises(ValueError):
        SpawnToleranceConfig(pos_tol_m=None)


def test_spawn_tolerance_config_raises_when_pos_tol_m_is_non_positive():
    import pytest

    with pytest.raises(ValueError):
        SpawnToleranceConfig(pos_tol_m=0.0)
    with pytest.raises(ValueError):
        SpawnToleranceConfig(pos_tol_m=-0.01)


def test_spawn_tolerance_config_raises_when_rot_tol_rad_is_non_positive_but_not_none():
    import pytest

    # None is the documented "disable the rotation gate" value and must NOT raise.
    SpawnToleranceConfig(pos_tol_m=0.01, rot_tol_rad=None)
    # An explicit non-positive value is a different claim ("no rotation allowed" or nonsensical)
    # and must raise rather than silently behave like disabled.
    with pytest.raises(ValueError):
        SpawnToleranceConfig(pos_tol_m=0.01, rot_tol_rad=0.0)
    with pytest.raises(ValueError):
        SpawnToleranceConfig(pos_tol_m=0.01, rot_tol_rad=-0.1)


def test_spawn_tolerance_config_accepts_explicit_positive_values():
    cfg = SpawnToleranceConfig(pos_tol_m=0.02, rot_tol_rad=0.1)
    assert cfg.pos_tol_m == 0.02
    assert cfg.rot_tol_rad == 0.1


# ---------------------------------------------------------------------------------------------
# pose_distance: the raw displacement the R4 measurement (bead dr-sj6.24) needs recorded.
# ---------------------------------------------------------------------------------------------


def test_pose_distance_is_zero_at_the_spawn_pose_itself():
    spawn_pos = torch.tensor([[0.5, 0.2, 0.1]])
    spawn_quat = _quat_about_z(0.3).unsqueeze(0)
    pos_dist, rot_dist = pose_distance(spawn_pos, spawn_quat, spawn_pos.clone(), spawn_quat.clone())
    assert torch.allclose(pos_dist, torch.zeros(1), atol=1e-6)
    assert torch.allclose(rot_dist, torch.zeros(1), atol=1e-6)


def test_pose_distance_position_matches_the_l2_displacement():
    spawn_pos = torch.zeros(1, 3)
    live_pos = torch.tensor([[0.003, 0.004, 0.0]])  # 3-4-5 triangle -> 5mm
    pos_dist, _ = pose_distance(spawn_pos, _IDENTITY_QUAT.unsqueeze(0), live_pos, _IDENTITY_QUAT.unsqueeze(0))
    assert math.isclose(float(pos_dist[0]), 0.005, abs_tol=1e-6)


def test_pose_distance_rotation_matches_a_known_angle_about_z():
    spawn_quat = _IDENTITY_QUAT.unsqueeze(0)
    for angle_deg in (0.0, 10.0, 45.0, 90.0, 179.0):
        live_quat = _quat_about_z(math.radians(angle_deg)).unsqueeze(0)
        pos = torch.zeros(1, 3)
        _, rot_dist = pose_distance(pos, spawn_quat, pos, live_quat)
        assert math.isclose(math.degrees(float(rot_dist[0])), angle_deg, abs_tol=0.05), (angle_deg, rot_dist)


def test_pose_distance_rotation_is_the_same_formula_c1_hand_pose_uses():
    # c1_hand_pose.py:266-268's own IK-residual measurement: quat_err = quat_mul(quat_inv(cmd),
    # achieved); ori_err = 2*acos(|quat_err.w|). Cross-check pose_distance's local quat_mul/quat_inv
    # against a hand-composed relative quaternion for two ARBITRARY (non-axis-aligned) quaternions,
    # so agreement is not an artifact of the single-axis fixture above.
    def _hamilton(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return torch.tensor(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    def _conj(q):
        w, x, y, z = q
        return torch.tensor([w, -x, -y, -z])

    spawn_quat = torch.tensor([0.8446, 0.1913, 0.4560, 0.2049])
    spawn_quat = spawn_quat / spawn_quat.norm()
    live_quat = torch.tensor([0.5253, -0.3600, 0.7602, 0.1132])
    live_quat = live_quat / live_quat.norm()

    rel = _hamilton(_conj(spawn_quat), live_quat)
    expected_rad = 2.0 * math.acos(min(1.0, abs(float(rel[0]))))

    pos = torch.zeros(1, 3)
    _, rot_dist = pose_distance(pos, spawn_quat.unsqueeze(0), pos, live_quat.unsqueeze(0))
    assert math.isclose(float(rot_dist[0]), expected_rad, abs_tol=1e-5)


def test_spawn_pose_displacement_is_a_plain_float_record():
    # What R4 accumulates (bead dr-sj6.24): plain floats, no torch dependency surviving the
    # measurement -- see this dataclass's own docstring.
    d = SpawnPoseDisplacement(pos_dist_m=0.012, rot_dist_rad=0.05)
    assert d.pos_dist_m == 0.012
    assert d.rot_dist_rad == 0.05


# ---------------------------------------------------------------------------------------------
# within_spawn_tolerance: accept/reject either side of a given tolerance.
# ---------------------------------------------------------------------------------------------


def test_within_spawn_tolerance_accepts_strictly_inside_position_band():
    cfg = SpawnToleranceConfig(pos_tol_m=0.01)  # 10mm, rotation gate disabled
    pos_dist = torch.tensor([0.005, 0.0099])
    rot_dist = torch.tensor([10.0, 10.0])  # irrelevant -- rot gate is disabled (rot_tol_rad=None)
    result = within_spawn_tolerance(pos_dist, rot_dist, cfg)
    assert bool(result[0]) is True
    assert bool(result[1]) is True


def test_within_spawn_tolerance_rejects_at_and_beyond_the_position_band():
    cfg = SpawnToleranceConfig(pos_tol_m=0.01)  # strict less-than, same convention as success.py
    pos_dist = torch.tensor([0.01, 0.011, 0.5])
    rot_dist = torch.tensor([0.0, 0.0, 0.0])
    result = within_spawn_tolerance(pos_dist, rot_dist, cfg)
    assert bool(result[0]) is False  # exactly at the boundary: NOT accepted (strict <)
    assert bool(result[1]) is False
    assert bool(result[2]) is False


def test_within_spawn_tolerance_with_rotation_gate_enabled_requires_both():
    cfg = SpawnToleranceConfig(pos_tol_m=0.01, rot_tol_rad=math.radians(5.0))
    # in-position, in-rotation -> accept
    assert bool(within_spawn_tolerance(torch.tensor([0.005]), torch.tensor([math.radians(2.0)]), cfg)[0]) is True
    # in-position, OUT-of-rotation -> reject (this is the case a position-only gate would miss)
    assert bool(within_spawn_tolerance(torch.tensor([0.005]), torch.tensor([math.radians(6.0)]), cfg)[0]) is False
    # out-of-position, in-rotation -> reject
    assert bool(within_spawn_tolerance(torch.tensor([0.02]), torch.tensor([math.radians(2.0)]), cfg)[0]) is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[spawn_tolerance] {name} OK", flush=True)
    print("[spawn_tolerance] all tests passed", flush=True)
