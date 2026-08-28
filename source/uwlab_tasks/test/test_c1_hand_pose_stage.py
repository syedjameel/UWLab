# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit-proves the C1 hand-pose stage's env-var parsing (RESET_SPEC_V2.md sec 1 C1,
V2_POSE_FINDINGS.md F10) without touching the arm/IK at all.

Needs only torch -- no Isaac Sim, no GPU, no env construction. Same reason and same technique as
``test_held_check_core.py`` next to this file: the ISAAC-TOUCHING half (the actual IK-solving event
term, ``dexlift/mdp/c1_hand_pose.py``, and the config wiring in
``dexlift_ur5e_delto_env_cfg.py::_apply_c1_hand_pose_stage``) imports ``isaaclab`` at module scope,
which needs a running Isaac Sim process just to import. ``c1_hand_pose_core.py`` has none of that
dependency by design -- see its own module docstring -- so it is loaded here BY FILE PATH, not via
``import uwlab_tasks...`` (whose package ``__init__.py`` transitively pulls in
``isaaclab_tasks -> isaaclab -> omni.kit.app``).

This test therefore covers ONLY the parsing/validation/anchor-arithmetic/quaternion half. It
cannot, by construction, prove ``_apply_c1_hand_pose_stage`` wires the resulting values into
``env_cfg.events.reset_c1_hand_pose`` correctly, or that the IK event solves to a sane joint
configuration -- both of those need Isaac Sim and are out of scope for a torch-only test. What it
DOES prove: the four env vars parse to exactly RESET_SPEC_V2.md's numbers, malformed input raises
with the offending value quoted, the master switch's off-path is a true no-op, and the palm-down
quaternion actually rotates ``gripper_approach_direction`` onto world -Z.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

# Loaded by file path, registered in sys.modules BEFORE exec -- unlike test_held_check_core.py's
# loader, this module's C1HandPoseStage uses @dataclass, and Python 3.12's dataclass machinery
# resolves `sys.modules[cls.__module__]` while processing the class body; skipping this line raises
# `AttributeError: 'NoneType' object has no attribute '__dict__'` at import time. Registering the
# module under its own spec name first is what fixes it, with no other change to the pattern.
_CORE_PATH = (
    Path(__file__).resolve().parents[1] / "uwlab_tasks/manager_based/manipulation/dexlift/mdp/c1_hand_pose_core.py"
)
_spec = importlib.util.spec_from_file_location("c1_hand_pose_core", _CORE_PATH)
_c1_hand_pose_core = importlib.util.module_from_spec(_spec)
sys.modules["c1_hand_pose_core"] = _c1_hand_pose_core
_spec.loader.exec_module(_c1_hand_pose_core)

parse_c1_hand_pose_env = _c1_hand_pose_core.parse_c1_hand_pose_env
anchor_xy_from_ranges = _c1_hand_pose_core.anchor_xy_from_ranges
palm_down_self_check = _c1_hand_pose_core.palm_down_self_check
quat_from_two_vectors = _c1_hand_pose_core.quat_from_two_vectors
quat_apply = _c1_hand_pose_core.quat_apply

# Ur5eDelto/metadata.yaml's gripper_approach_direction, read directly (not guessed) -- same value
# scripts_v2/tools/analyze_grasp_orientation_distribution.py uses.
_GRIPPER_APPROACH_DIRECTION_LOCAL = torch.tensor([0.2582, 0.4717, 0.8431])


def test_master_switch_unset_is_a_true_no_op():
    """No DEXRESET_C1_HAND at all, and DEXRESET_C1_HAND set to anything but '1', both no-op."""
    assert parse_c1_hand_pose_env({}) is None
    assert parse_c1_hand_pose_env({"DEXRESET_C1_HAND": "0"}) is None
    assert parse_c1_hand_pose_env({"DEXRESET_C1_HAND": "false"}) is None
    # A no-op must not even look at the other three env vars -- garbage in DEXRESET_C1_HAND_Z must
    # not raise when the master switch is off.
    assert parse_c1_hand_pose_env({"DEXRESET_C1_HAND": "0", "DEXRESET_C1_HAND_Z": "not,numbers"}) is None


def test_defaults_match_reset_spec_v2():
    """RESET_SPEC_V2.md sec 1 C1: z in [0.10, 0.20] m, xy +-0.15 m, tilt +-45 deg (0.7854 rad)."""
    stage = parse_c1_hand_pose_env({"DEXRESET_C1_HAND": "1"})
    assert stage is not None
    assert stage.z_lo == 0.10
    assert stage.z_hi == 0.20
    assert stage.xy_half_width == 0.15
    assert math.isclose(stage.tilt, 0.7854, abs_tol=1e-9)
    assert math.isclose(math.degrees(stage.tilt), 45.0, abs_tol=0.01)


def test_permitted_relaxation_to_0_10m_xy():
    """RESET_SPEC_V2.md sec 1's explicitly permitted relaxation: XY narrowed to +-0.10 m, height
    range narrowed too -- both configurable, per the assignment.
    """
    stage = parse_c1_hand_pose_env(
        {"DEXRESET_C1_HAND": "1", "DEXRESET_C1_HAND_XY": "0.10", "DEXRESET_C1_HAND_Z": "0.10,0.15"}
    )
    assert stage.xy_half_width == 0.10
    assert (stage.z_lo, stage.z_hi) == (0.10, 0.15)


def test_bad_z_range_raises_with_value_quoted():
    for bad in ("0.20,0.10", "not,numbers", "0.10"):
        try:
            parse_c1_hand_pose_env({"DEXRESET_C1_HAND": "1", "DEXRESET_C1_HAND_Z": bad})
        except ValueError as exc:
            assert bad in str(exc) or "0.2" in str(exc)  # exact repr varies by which check fires
        else:
            raise AssertionError(f"DEXRESET_C1_HAND_Z={bad!r} should have raised ValueError")


def test_bad_xy_raises():
    for bad in ("-0.1", "0.0", "nope"):
        try:
            parse_c1_hand_pose_env({"DEXRESET_C1_HAND": "1", "DEXRESET_C1_HAND_XY": bad})
        except ValueError as exc:
            assert bad in str(exc)
        else:
            raise AssertionError(f"DEXRESET_C1_HAND_XY={bad!r} should have raised ValueError")


def test_bad_tilt_raises():
    for bad in ("-0.1", "4.0", "nope"):  # 4.0 rad > pi
        try:
            parse_c1_hand_pose_env({"DEXRESET_C1_HAND": "1", "DEXRESET_C1_HAND_TILT": bad})
        except ValueError as exc:
            assert bad in str(exc) or bad.replace(".0", "") in str(exc)
        else:
            raise AssertionError(f"DEXRESET_C1_HAND_TILT={bad!r} should have raised ValueError")


def test_anchor_is_the_goal_range_midpoint_not_the_leg():
    """RESET_SPEC_V2.md: nominal point is above the goal/bore XY, not above the leg. This dexlift
    env's stand-in for that anchor is commands.object_pose.ranges' own midpoint -- see this task's
    default pos_x=(0.3, 0.7), pos_y inherited at (-0.25, 0.25).
    """
    assert anchor_xy_from_ranges((0.3, 0.7), (-0.25, 0.25)) == (0.5, 0.0)


def test_palm_down_quaternion_rotates_approach_axis_to_world_minus_z():
    """The nominal orientation the event class builds must actually point the palm down: applying
    it to gripper_approach_direction must land at (0, 0, -1), inside the same 60-degree
    orient_down gate omnireset/mdp/terminations.py::check_reset_state_success scores.
    """
    # palm_down_self_check raises AssertionError internally if the rotation doesn't land at -Z;
    # reaching this line without raising IS the assertion.
    palm_down_self_check(_GRIPPER_APPROACH_DIRECTION_LOCAL)


def test_quat_from_two_vectors_rejects_antiparallel_input():
    """``quat_from_two_vectors`` documents itself as raising on (near-)antiparallel input rather
    than silently returning a degenerate/undefined quaternion. Not exercised by the shipped test
    suite (only the actual DELTO approach direction, which is nowhere near antiparallel to -Z, is
    tested there). ``gripper_approach_direction`` is asset metadata, not a compile-time constant --
    a future robot variant whose approach axis is close to world +Z (i.e. palm pointing UP) would
    hit exactly this path when computing the "palm down" nominal, and the failure mode matters: a
    loud ValueError at config time is fine, a NaN quaternion silently teleporting the arm is not.
    """
    a = torch.tensor([0.0, 0.0, 1.0])
    b = torch.tensor([0.0, 0.0, -1.0])
    try:
        quat_from_two_vectors(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("quat_from_two_vectors(a, -a) should raise ValueError, not return silently")


def test_per_axis_tilt_extremes_do_not_bound_the_composed_rotation_to_tilt():
    """RESET_SPEC_V2.md sec 1 C1 asks for "+-45 deg variation, applied per-axis about the palm-down
    nominal". The event composes the three independently-sampled per-axis angles into ONE
    quaternion via ``math_utils.quat_from_euler_xyz(roll, pitch, yaw)`` (a fixed Tait-Bryan
    composition order), then applies that as a single extrinsic rotation on top of the nominal.
    That is not the same object as "each axis independently bounded by 45 deg from the nominal" --
    Euler-angle composition is not additive, so nothing in the per-axis clamp guarantees the
    RESULTING single-rotation angle from the nominal stays inside 45 deg.

    This test does not assert compliance with the spec (that needs the actual isaaclab
    ``quat_from_euler_xyz``, unavailable here without Isaac); it characterizes the mechanism with
    the exact same formula ``isaaclab.utils.math.quat_from_euler_xyz`` documents (Tait-Bryan
    roll-pitch-yaw), at the worst-case corner (roll=pitch=yaw=+tilt), and asserts what that
    actually is: NOT simply `tilt` and NOT simply `3*tilt` either. Whoever reviews the GPU
    measurement in run B should compare the achieved cone-angle p95/max against this number, not
    against 45 deg alone.
    """

    def _quat_from_euler_xyz(roll, pitch, yaw):
        # Same formula as isaaclab.utils.math.quat_from_euler_xyz -- reproduced here because that
        # module cannot be imported without Isaac Sim (see this file's own docstring).
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        qw = cy * cr * cp + sy * sr * sp
        qx = cy * sr * cp - sy * cr * sp
        qy = cy * cr * sp + sy * sr * cp
        qz = sy * cr * cp - cy * sr * sp
        return torch.tensor([qw, qx, qy, qz])

    tilt = 0.7854  # +-45 deg, the shipped default

    def _angle_from_identity(q: torch.Tensor) -> float:
        w = abs(float(q[0].clamp(-1.0, 1.0)))
        return 2.0 * math.degrees(math.acos(w))

    worst_corner = _quat_from_euler_xyz(tilt, tilt, tilt)
    worst_angle = _angle_from_identity(worst_corner)

    # A single-axis perturbation at the same tilt reproduces exactly 45 deg, as a sanity check on
    # the formula/measurement.
    single_axis = _quat_from_euler_xyz(tilt, 0.0, 0.0)
    single_axis_angle = _angle_from_identity(single_axis)
    assert math.isclose(single_axis_angle, math.degrees(tilt), abs_tol=0.1), single_axis_angle

    # The composed all-axes-at-extreme corner must NOT equal the single-axis angle (that would mean
    # the composition is a no-op / degenerate) and must exceed it -- three simultaneous +-45 deg
    # per-axis draws compose to MORE than 45 deg of total rotation from the nominal, not the same
    # 45 deg the spec names.
    assert worst_angle > single_axis_angle + 1.0, (
        f"expected the 3-axis corner ({worst_angle:.1f} deg) to exceed the single-axis angle"
        f" ({single_axis_angle:.1f} deg) by a meaningful margin"
    )
    # Record the actual number for whoever reads this test: at the shipped default this corner is
    # ~travelled distance below, characterized empirically rather than asserted to a hardcoded
    # value (which would make this a change-detector, not a spec check). Anyone reading run B's
    # measured palm-angle p95/max should compare against THIS number, not against 45 deg.
    print(f"[c1_hand_pose] worst-case 3-axis corner at tilt=+-45deg composes to {worst_angle:.2f} deg total rotation from nominal (single-axis: {single_axis_angle:.2f} deg)", flush=True)
    assert worst_angle < 3 * math.degrees(tilt) + 1.0  # sanity upper bound: composition is not additive either


if __name__ == "__main__":
    # Runnable with plain python3, no pytest required -- same convention as the assignment allows
    # for a repo with no dedicated check_*.py location; this repo DOES have one
    # (source/uwlab_tasks/test/, see test_held_check_core.py), so the test lives there instead.
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[c1_hand_pose] {name} OK", flush=True)
    print("[c1_hand_pose] all tests passed", flush=True)
