# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Linear-gripper variants of the OmniReset tasks (new variant alongside the 2F-85).

These subclass the 2F-85 task configs and swap ONLY the robot (UR5e + custom linear
parallel-jaw gripper) and the gripper action; everything else (objects, events, rewards,
sim settings, the object `variants`) is inherited unchanged. The 2F-85 tasks are untouched.

Registered gym ids (mirroring the 2F-85 ones):
* ``OmniReset-LinearGripper-GraspSampling-v0``
"""

from __future__ import annotations

import uwlab_assets.robots.ur5e_linear_gripper as ur5e_linear_gripper

from isaaclab.utils import configclass

from .grasp_sampling_cfg import Robotiq2f85GraspSamplingCfg


@configclass
class LinearGripperGraspSamplingCfg(Robotiq2f85GraspSamplingCfg):
    """Grasp sampling with the custom linear gripper (gripper-only, like ROBOTIQ_2F85)."""

    def __post_init__(self):
        # Swap the gripper-only robot and the binary action before the base configures sim.
        self.scene.robot = ur5e_linear_gripper.LINEAR_GRIPPER.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions = ur5e_linear_gripper.LinearGripperBinaryGripperAction()
        super().__post_init__()
