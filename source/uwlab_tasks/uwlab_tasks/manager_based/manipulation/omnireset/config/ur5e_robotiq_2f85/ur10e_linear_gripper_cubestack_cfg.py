# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e + linear gripper on the cube34 stacking pair: the baseline the DELTO cube stack is judged against.

The UR10e + linear-gripper reset-state configs (``ur10e_linear_gripper_cfg.py``) are the ones this
repository took to a real robot. The DELTO cube stack (``ur10e_delto_cubestack_cfg.py``) subclasses
the same 2F-85 bases with a different end effector. Recording the same four families on the same
cube pair with the parallel jaw separates what belongs to the environment (table, arm, objects,
reset logic, the Isaac Lab 3.0 port) from what belongs to the hand: if these banks hold grasps and
the DELTO ones do not, the difference is the hand.

Deliberately minimal. Each class is its linear-gripper parent plus two edits: pin the cube34 pair
(``_apply_cube_objects``, shared with the DELTO stack so the two scenes cannot drift apart) and
write to a linear-gripper dataset tree. Datasets are keyed by object pair only, so without the
second edit these banks would overwrite the DELTO ones.
"""

from __future__ import annotations

import os

from isaaclab.utils import configclass

from .delto_cfg import _apply_delto_dataset_dir
from .ur10e_delto_cubestack_cfg import _apply_cube_objects
from .ur10e_linear_gripper_cfg import (
    Ur10eLinearGripperObjectAnywhereEEAnywhereResetStatesCfg,
    Ur10eLinearGripperObjectAnywhereEEGraspedResetStatesCfg,
    Ur10eLinearGripperObjectPartiallyAssembledEEAnywhereResetStatesCfg,
    Ur10eLinearGripperObjectPartiallyAssembledEEGraspedResetStatesCfg,
    Ur10eLinearGripperObjectRestingEEGraspedResetStatesCfg,
    Ur10eLinearGripperRelCartesianOSCEvalCfg,
    Ur10eLinearGripperRelCartesianOSCTrainCfg,
)

LINEAR_CUBE_DATASET_DIR = os.environ.get("UWLAB_LINEAR_CUBE_DATASET_DIR", "./Datasets_ur10e_linear/OmniReset")


def _apply_linear_cube(cfg) -> None:
    _apply_cube_objects(cfg)
    # Gripper-agnostic despite the name: it repoints every dataset_dir-bearing event term.
    _apply_delto_dataset_dir(cfg, LINEAR_CUBE_DATASET_DIR)


def _apply_jaw_in_hand(cfg) -> None:
    """Accept a grasped state on the grip itself: cube between the jaws, jaws stopped on it.

    The stored-grasp distance test cannot be used here: these families perturb the hand off the
    stored grasp by up to +-2 cm / +-11 deg by design, which put the placement 26 mm (median) from
    any grasp point and rejected ~97% of real holds.
    """
    params = cfg.terminations.success.params
    params["max_grasp_distance_error"] = None
    # Per-axis box around the bank grasp centre in the hand frame: bank max |dev| + this margin.
    # 6 mm -> +-(8.8, 18.8, 30.4) mm; separates the filmed holds from jaws closed beside the cube.
    params["grasp_center_margin"] = float(os.environ.get("UWLAB_GRASP_CENTER_MARGIN", "0.006"))


@configclass
class LinearCubeStackObjectAnywhereEEAnywhereCfg(Ur10eLinearGripperObjectAnywhereEEAnywhereResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_linear_cube(self)


@configclass
class LinearCubeStackObjectRestingEEGraspedCfg(Ur10eLinearGripperObjectRestingEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_linear_cube(self)
        _apply_jaw_in_hand(self)
        # The hand lands up to +-2 cm off the stored grasp, so closing the jaws legitimately slides
        # the resting cube into the grip by more than the shipped 10 mm. Same limit as C4.
        self.terminations.success.params["max_object_pos_deviation"] = float(
            os.environ.get("UWLAB_C2_MAX_OBJECT_DEVIATION", "0.025"))
        # Placement noise around the stored grasp. The shipped +-2 cm / +-pi/16 suits the 2F-85's
        # self-centring linkage; this rigid parallel jaw leaves most such placements off-centre, so
        # only ~3% passed the in-hand box (~3 min per state). +-5 mm / +-pi/32 keeps varied grasps.
        _pos = float(os.environ.get("UWLAB_C2_GRASP_POS_NOISE", "0.005"))
        _rot = float(os.environ.get("UWLAB_C2_GRASP_ROT_NOISE", str(3.14159265 / 32)))
        self.events.reset_end_effector_pose_from_grasp_dataset.params["pose_range_b"] = {
            "x": (-_pos, _pos), "y": (-_pos, _pos), "z": (-_pos, _pos),
            "roll": (-_rot, _rot), "pitch": (-_rot, _rot), "yaw": (-_rot, _rot),
        }


@configclass
class LinearCubeStackObjectAnywhereEEGraspedCfg(Ur10eLinearGripperObjectAnywhereEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_linear_cube(self)
        _apply_jaw_in_hand(self)


@configclass
class LinearCubeStackObjectPartiallyAssembledEEAnywhereCfg(
    Ur10eLinearGripperObjectPartiallyAssembledEEAnywhereResetStatesCfg
):
    def __post_init__(self):
        super().__post_init__()
        _apply_linear_cube(self)


@configclass
class LinearCubeStackObjectPartiallyAssembledEEGraspedCfg(
    Ur10eLinearGripperObjectPartiallyAssembledEEGraspedResetStatesCfg
):
    def __post_init__(self):
        super().__post_init__()
        _apply_linear_cube(self)
        _apply_jaw_in_hand(self)


# ---------------------------------------------------------------------------------------
# RL state: the proven UR10e + linear-gripper Stage-1 train / eval, on the cube pair
# ---------------------------------------------------------------------------------------
# Nothing but the scene and the bank location change. The reward, observation, action and the 4-way
# 25% reset mixture are the ones the linear gripper was trained with to ~90% real-robot success.
@configclass
class LinearCubeStackTrainCfg(Ur10eLinearGripperRelCartesianOSCTrainCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_linear_cube(self)


@configclass
class LinearCubeStackEvalCfg(Ur10eLinearGripperRelCartesianOSCEvalCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_linear_cube(self)
