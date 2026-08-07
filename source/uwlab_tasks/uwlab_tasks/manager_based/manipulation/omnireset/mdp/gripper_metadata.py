# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gripper-metadata readers shared by the reset events.

Kept free of Isaac/torch imports on purpose: the readers are pure dict -> dict and can be
unit-tested against the checked-in ``metadata.yaml`` files without launching the simulator.
"""

from __future__ import annotations

from collections.abc import Mapping

# Legacy (parallel-jaw) key: ONE scalar, implicitly the driver joint's open value.
LEGACY_OPEN_ANGLE_KEY = "finger_open_joint_angle"
# The joint the legacy scalar refers to. Every parallel-jaw gripper in this repo renames its
# driver joint to the Robotiq 2F-85 contract, so the scalar has always meant this joint.
LEGACY_OPEN_ANGLE_JOINT = "finger_joint"
# General key: a joint name -> open value mapping, for grippers with more than one driver
# (a multi-finger hand has no single "open" number -- open is a posture).
OPEN_POSTURE_KEY = "finger_open_joint_angles"


def read_gripper_open_posture(metadata: Mapping) -> dict[str, float]:
    """Return the gripper's OPEN posture as a ``{joint name: value}`` mapping.

    Two metadata spellings are accepted, in this order:

    1. ``finger_open_joint_angles``: an explicit joint name -> value mapping. Required for any
       gripper whose open configuration is not a single number (e.g. a five-finger hand).
    2. ``finger_open_joint_angle``: the legacy scalar, meaning ``{"finger_joint": <scalar>}``.
       Every parallel-jaw metadata file in the repo uses this and keeps working unchanged.

    Args:
        metadata: The parsed ``metadata.yaml`` of the gripper's USD directory.

    Returns:
        The open posture. Values are floats; joint names are matched exactly (not as regex)
        against the articulation's joint names by the caller.

    Raises:
        ValueError: If the mapping is present but is not a non-empty mapping of numbers, if the
            legacy scalar is not a number, or if neither key is present.
    """
    posture = metadata.get(OPEN_POSTURE_KEY)
    if posture is not None:
        if not isinstance(posture, Mapping) or not posture:
            raise ValueError(
                f"Gripper metadata '{OPEN_POSTURE_KEY}' must be a non-empty mapping of joint name ->"
                f" open joint value, got {posture!r}."
            )
        out = {}
        for joint_name, value in posture.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Gripper metadata '{OPEN_POSTURE_KEY}' entry '{joint_name}' must be a number, got {value!r}."
                )
            out[str(joint_name)] = float(value)
        return out

    if LEGACY_OPEN_ANGLE_KEY in metadata:
        value = metadata[LEGACY_OPEN_ANGLE_KEY]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Gripper metadata '{LEGACY_OPEN_ANGLE_KEY}' must be a number, got {value!r}.")
        return {LEGACY_OPEN_ANGLE_JOINT: float(value)}

    raise ValueError(
        "Gripper metadata declares no open posture: expected either"
        f" '{OPEN_POSTURE_KEY}' (joint name -> value mapping) or the legacy scalar"
        f" '{LEGACY_OPEN_ANGLE_KEY}'."
    )
