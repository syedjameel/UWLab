# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Composite inertia of ``wrist_3_link`` + the whole DELTO hand about the ``wrist_3_joint`` axis.

THE NUMBER THIS PRODUCES IS THE ONE INPUT to the rotational damping ratio of the operational-space
arm action (``omnireset/config/ur5e_robotiq_2f85/actions.py``,
``_UR5E_DELTO_ROT_DAMPING_RATIO``). That controller is a mass-less Cartesian PD whose damping term
is explicit velocity feedback evaluated once per physics step, so its stability bound is
``kd * dt / I < 2`` with ``I`` the rotational inertia of everything distal to the wrist. The ratio
has to be re-derived for every end effector on this arm family, and this script is what makes the
derivation reproducible instead of a number in a comment.

It reads the USD directly with ``pxr`` -- NO simulator, no GPU, no Isaac app -- so it runs on any
machine that can import ``pxr``::

    python scripts_v2/tools/compute_delto_wrist_inertia.py \
        --usd source/uwlab_assets/uwlab_assets/local/Robots/Ur5eDelto/ur5e_delto.usd

Method, per rigid body in the distal subtree: compose the joint chain from ``wrist_3_link`` at the
requested joint angles, rotate the authored ``diagonalInertia`` by its ``principalAxes`` quaternion
into the wrist frame, then parallel-axis it out to the wrist origin. The ``wrist_3_joint`` authors
``axis = Z``, ``localPos1 = (0,0,0)`` and identity ``localRot1``, so the axis is wrist_3_link's own
+Z through its origin and the reported quantity is the ``zz`` entry of the composite tensor.

WHY THE POSTURE MATTERS, and the trap this script exists to close. ``I`` at ZERO hand angles and
``I`` at the ARTICULATION'S RESET POSTURE differ by 1.83x on this hand (2.757e-3 vs 5.041e-3), and
the reset posture is NOT zero -- the open posture folds the fingers out. A derivation that labels
the zero-angle number "the reset posture" understates the reachable inertia by nearly a factor of
two, in the direction that makes an aggressive damping ratio look worse than it is; the argument
for the chosen ratio then rests on a row that does not describe the robot at reset.
"""

from __future__ import annotations

import argparse
import ast
import numpy as np
import pathlib
import re

from pxr import Usd, UsdPhysics

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_USD = _REPO_ROOT / "source/uwlab_assets/uwlab_assets/local/Robots/Ur5eDelto/ur5e_delto.usd"
_ASSET_MODULE = _REPO_ROOT / "source/uwlab_assets/uwlab_assets/robots/ur10e_delto/ur10e_delto.py"
_ACTIONS_MODULE = _REPO_ROOT / "source/uwlab_assets/uwlab_assets/robots/ur10e_delto/actions.py"
_WRIST_BODY = "wrist_3_link"


def _read_posture(module_path: pathlib.Path, name: str) -> dict[str, float]:
    """Read one ``{joint: angle}`` literal out of an asset module, without importing Isaac Lab.

    The asset modules import ``isaaclab``, which needs ``omni``; this script deliberately runs in a
    bare interpreter, so the dict is parsed out of the source instead. Parsed with ``ast``, so a
    non-literal (an expression, a call) fails here rather than being evaluated.
    """
    source = module_path.read_text()
    match = re.search(rf"^{name} = (\{{.*?^\}})", source, re.DOTALL | re.MULTILINE)
    if match is None:
        raise SystemExit(f"{module_path}: no literal dict named {name}")
    return ast.literal_eval(match.group(1))


def _quat_to_mat(quat) -> np.ndarray:
    w, x, y, z = quat
    norm = float(np.sqrt(w * w + x * x + y * y + z * z))
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _gf_quat(quat) -> np.ndarray:
    imaginary = quat.GetImaginary()
    return np.array([quat.GetReal(), imaginary[0], imaginary[1], imaginary[2]], dtype=float)


def _axis_mat(axis: str, angle: float) -> np.ndarray:
    unit = {"X": np.array([1.0, 0.0, 0.0]), "Y": np.array([0.0, 1.0, 0.0]), "Z": np.array([0.0, 0.0, 1.0])}[axis]
    skew = np.array([[0, -unit[2], unit[1]], [unit[2], 0, -unit[0]], [-unit[1], unit[0], 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


class WristInertia:
    """The USD's distal chain, evaluable at any hand posture."""

    def __init__(self, usd_path: str):
        self.stage = Usd.Stage.Open(usd_path)
        self.joints = []
        for prim in self.stage.Traverse():
            if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.FixedJoint)):
                continue
            joint = UsdPhysics.Joint(prim)
            body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not body0 or not body1:
                continue
            axis = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get() if prim.IsA(UsdPhysics.RevoluteJoint) else None
            self.joints.append(
                dict(
                    name=prim.GetName(),
                    parent=str(body0[0]).rsplit("/", 1)[-1],
                    child=str(body1[0]).rsplit("/", 1)[-1],
                    pos0=np.array(joint.GetLocalPos0Attr().Get(), dtype=float),
                    rot0=_quat_to_mat(_gf_quat(joint.GetLocalRot0Attr().Get())),
                    pos1=np.array(joint.GetLocalPos1Attr().Get(), dtype=float),
                    rot1=_quat_to_mat(_gf_quat(joint.GetLocalRot1Attr().Get())),
                    axis=axis,
                )
            )
        self.limits = {}
        for prim in self.stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName().startswith("rj_dg_"):
                joint = UsdPhysics.RevoluteJoint(prim)
                lower, upper = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
                if lower is not None and upper is not None:
                    self.limits[prim.GetName()] = (np.deg2rad(lower), np.deg2rad(upper))

    def _poses(self, joint_angles: dict[str, float]) -> dict[str, np.ndarray]:
        poses = {_WRIST_BODY: np.eye(4)}
        changed = True
        while changed:
            changed = False
            for joint in self.joints:
                if joint["parent"] in poses and joint["child"] not in poses:
                    angle = joint_angles.get(joint["name"], 0.0) if joint["axis"] else 0.0
                    rotation = _axis_mat(joint["axis"], angle) if joint["axis"] else np.eye(3)
                    local = (
                        _transform(joint["rot0"], joint["pos0"])
                        @ _transform(rotation, np.zeros(3))
                        @ np.linalg.inv(_transform(joint["rot1"], joint["pos1"]))
                    )
                    poses[joint["child"]] = poses[joint["parent"]] @ local
                    changed = True
        return poses

    def evaluate(self, joint_angles: dict[str, float]) -> tuple[float, float, int]:
        """Return ``(I_zz about the wrist axis, distal mass, body count)`` at this posture."""
        poses = self._poses(joint_angles)
        inertia = np.zeros((3, 3))
        mass_total = 0.0
        bodies = 0
        for prim in self.stage.Traverse():
            pose = poses.get(prim.GetName())
            if pose is None:
                continue
            mass_api = UsdPhysics.MassAPI(prim)
            mass_attr = mass_api.GetMassAttr()
            if not (mass_attr and mass_attr.HasAuthoredValue()):
                continue
            mass = float(mass_attr.Get())
            if mass <= 0.0:
                continue
            diagonal = np.array(mass_api.GetDiagonalInertiaAttr().Get(), dtype=float)
            axes_attr = mass_api.GetPrincipalAxesAttr()
            axes = _quat_to_mat(_gf_quat(axes_attr.Get())) if axes_attr and axes_attr.HasAuthoredValue() else np.eye(3)
            com_attr = mass_api.GetCenterOfMassAttr()
            com = np.array(com_attr.Get(), dtype=float) if com_attr and com_attr.HasAuthoredValue() else np.zeros(3)

            rotation, translation = pose[:3, :3], pose[:3, 3]
            about_com = rotation @ (axes @ np.diag(diagonal) @ axes.T) @ rotation.T
            offset = translation + rotation @ com
            inertia += about_com + mass * (float(offset @ offset) * np.eye(3) - np.outer(offset, offset))
            mass_total += mass
            bodies += 1
        return float(inertia[2, 2]), mass_total, bodies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--usd", default=str(_DEFAULT_USD), help="robot USD to read")
    parser.add_argument("--samples", type=int, default=4000, help="uniform postures drawn inside the USD limits")
    parser.add_argument("--kp", type=float, default=3.0, help="rotational stiffness of the OSC action")
    parser.add_argument("--dt", type=float, default=1.0 / 120.0, help="PHYSICS step; apply_actions runs at sim.dt")
    parser.add_argument("--damping-ratios", type=float, nargs="+", default=[0.1, 0.2])
    args = parser.parse_args()

    model = WristInertia(args.usd)
    open_posture = _read_posture(_ASSET_MODULE, "DELTO_HAND_DEFAULT_JOINT_POS")
    close_posture = _read_posture(_ACTIONS_MODULE, "DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS")

    rows: list[tuple[str, float]] = []
    for label, angles in (
        ("zero hand angles", {}),
        ("reset posture (DELTO_HAND_DEFAULT_JOINT_POS)", open_posture),
        ("validated scripted close", close_posture),
    ):
        value, mass, bodies = model.evaluate(angles)
        rows.append((label, value))
        print(f"{label:46s} I = {value:.6e} kg*m^2   (distal mass {mass:.4f} kg over {bodies} bodies)")

    rng = np.random.default_rng(0)
    sampled = np.array(
        [
            model.evaluate({name: rng.uniform(*bounds) for name, bounds in model.limits.items()})[0]
            for _ in range(args.samples)
        ]
    )
    print(
        f"{args.samples} uniform postures in the USD limits    "
        f"min {sampled.min():.6e}  median {np.median(sampled):.6e}  max {sampled.max():.6e}"
    )
    rows.append(("uniform-sweep minimum", float(sampled.min())))

    # Coordinate descent to the worst case: the posture bound is what the damping ratio has to
    # survive, and the uniform sweep does not find it (20 dimensions).
    angles = {name: 0.0 for name in model.limits}
    for _ in range(4):
        for name, (lower, upper) in model.limits.items():
            best_value, best_angle = None, angles[name]
            for candidate in np.linspace(lower, upper, 25):
                angles[name] = float(candidate)
                value = model.evaluate(angles)[0]
                if best_value is None or value < best_value:
                    best_value, best_angle = value, float(candidate)
            angles[name] = best_angle
    worst = model.evaluate(angles)[0]
    rows.append(("coordinate-descent minimum", worst))
    print(f"{'coordinate-descent minimum over the limits':46s} I = {worst:.6e} kg*m^2")

    print(f"\nkd * dt / I against the explicit-Euler bound of 2   (kp {args.kp}, dt {args.dt:.6f} s)")
    for ratio in args.damping_ratios:
        kd = 2.0 * np.sqrt(args.kp) * ratio
        print(f"  damping_ratio {ratio}  ->  kd = {kd:.4f} N*m*s/rad")
        for label, value in rows:
            criterion = kd * args.dt / value
            print(f"      {label:44s} {criterion:5.2f}{'   OVER THE BOUND' if criterion > 2.0 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
