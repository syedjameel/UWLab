# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The per-gripper seam: where a robot variant declares which joints ARE its gripper.

Every OmniReset task config in this package is written for the Robotiq 2F-85. The event that
replays recorded grasps and the gripper-gain randomization must be told which joints belong to
the gripper, and that answer is different for every gripper. A variant states
it once, after ``super().__post_init__()``, by calling :func:`override_gripper_joints` -- see
``linear_gripper_cfg.py::_apply_linear_gripper``.

Leaving the 2F-85 defaults in place on another gripper is a mistake, not a fallback: the 2F-85
selection is a regex list, and a pattern that matches nothing raises IsaacLab's generic "Not all
regular expressions are matched" deep inside term parsing, which says nothing about grippers.
:class:`GripperJointsCfg` wraps that failure with a message that names the selection and points
here.
"""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# The events whose gripper joint selection a new gripper must override.
GRASP_DATASET_EVENTS = ("reset_end_effector_pose_from_grasp_dataset",)
# ...plus the gripper-gain DR, which selects joints through ``asset_cfg`` instead.
GRIPPER_DR_EVENT = "randomize_gripper_actuator_parameters"
# ...plus the startup check that turns a forgotten override into an immediate, explicit failure
# (mdp.check_gripper_joint_selection). It takes plain joint names, not a SceneEntityCfg.
GRIPPER_CHECK_EVENT = "check_gripper_joints"

# The Robotiq 2F-85's own joints: the driver plus its many passive linkage joints, which is why
# this is a regex list rather than explicit names. Correct for the 2F-85 ONLY.
ROBOTIQ_2F85_GRIPPER_JOINTS = ["finger_joint", ".*right.*", ".*left.*"]

_HINT = (
    "This is the Robotiq 2F-85 default. A different gripper must declare its own joints after"
    " super().__post_init__() via"
    " gripper_seam.override_gripper_joints(cfg, [...]) -- see linear_gripper_cfg._apply_linear_gripper."
)


@configclass
class GripperJointsCfg(SceneEntityCfg):
    """A gripper joint selection that explains itself when it fails to resolve.

    Behaves exactly like :class:`SceneEntityCfg`; it only rewrites the resolution error.
    """

    hint: str = _HINT

    def resolve(self, scene):  # type: ignore[override]
        try:
            super().resolve(scene)
        except ValueError as exc:
            raise ValueError(
                f"Gripper joint selection {self.joint_names} does not resolve on scene entity"
                f" '{self.name}'.\n{self.hint}\nUnderlying error: {exc}"
            ) from exc


def robotiq_2f85_gripper_joints() -> GripperJointsCfg:
    """The 2F-85 gripper joint selection, as used by the base task configs."""
    return GripperJointsCfg("robot", joint_names=list(ROBOTIQ_2F85_GRIPPER_JOINTS))


def override_gripper_joints(
    cfg,
    joint_names: Sequence[str],
    require_independent_actuation: bool = False,
) -> None:
    """Point every gripper-joint-selecting event on ``cfg`` at ``joint_names``.

    Args:
        cfg: The env config, AFTER ``super().__post_init__()`` (the base configs build their
            event terms there).
        joint_names: The gripper's joints, as explicit names or regex patterns.
        require_independent_actuation: When True, the startup check additionally requires each of
            those joints to be its own policy action dimension -- the full-actuation guard. False
            for a parallel jaw, whose many linkage joints legitimately share one driver; True for
            a fully actuated hand, where a shared command is the banned one-scalar closure. It is
            set HERE rather than on the check term directly so a gripper declares its joints and
            what it expects of them in one call, and cannot set one without the other.
    Terms absent from a given task variant are skipped (``getattr`` returns ``None``).
    """
    joint_names = list(joint_names)
    hint = (
        "Declared by gripper_seam.override_gripper_joints(); the names must match the joints of the"
        " robot this variant spawns."
    )
    for term_name in GRASP_DATASET_EVENTS:
        term = getattr(cfg.events, term_name, None)
        if term is not None:
            term.params["gripper_cfg"] = GripperJointsCfg("robot", joint_names=joint_names, hint=hint)
    dr = getattr(cfg.events, GRIPPER_DR_EVENT, None)
    if dr is not None:
        dr.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=joint_names)
    check = getattr(cfg.events, GRIPPER_CHECK_EVENT, None)
    if check is not None:
        check.params["joint_names"] = joint_names
        check.params["require_independent_actuation"] = require_independent_actuation
