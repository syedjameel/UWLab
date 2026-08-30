# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .commands import TaskCommand, TaskDependentCommand


@configclass
class TaskDependentCommandCfg(CommandTermCfg):
    class_type: type = TaskDependentCommand

    reset_terms_when_resample: dict[str, EventTerm] = {}


@configclass
class TaskCommandCfg(TaskDependentCommandCfg):
    class_type: type = TaskCommand

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")

    insertive_asset_cfg: SceneEntityCfg = MISSING

    receptive_asset_cfg: SceneEntityCfg = MISSING

    success_position_threshold: float | None = None
    """Override for the receptive object's ``success_thresholds.position``.

    ``None`` (the default) keeps the historical behaviour: read the value from the receptive
    object's ``metadata.yaml``. Every existing task is therefore byte-for-byte unaffected.

    An explicit value lets a task vary the success criterion without editing a shared asset's
    metadata -- which matters because ``metadata.yaml`` is keyed by USD directory and is shared by
    every task that spawns that object.
    """

    success_orientation_threshold: float | None = None
    """Override for the receptive object's ``success_thresholds.orientation``, in radians.

    Set to ``math.inf`` to make the task ORIENTATION-FREE: ``ProgressContext`` computes
    ``orientation_aligned = euler_xy_distance < threshold`` (rewards.py), so an infinite threshold
    makes that term unconditionally true and success collapses to the position criterion alone.
    This is the single-point implementation of the paper's Eq. 6 with the orientation clause
    dropped; note the codebase already ignores YAW (rewards.py, "yaw could be different"), so the
    orientation criterion is a roll/pitch constraint only.
    """
