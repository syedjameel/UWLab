# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the dexterous lifting task.

Everything the vendored dexsuite package already provides is re-exported here unchanged, so a
config module only ever imports from this one place. The local :mod:`.rewards` import comes last
and deliberately shadows the upstream ``contacts``, ``success_reward``,
``position_command_error_tanh`` and ``orientation_command_error_tanh``, whose upstream versions are
hardcoded to the Kuka-Allegro fingertip sensor names.
"""

from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp import *  # noqa: F401, F403

# NOTE: there is no local ``actions`` module. It held ContinuousSynergyJointPositionAction/Cfg --
# one scalar interpolating all twenty DELTO joints between two calibrated postures. It had no call
# sites left, but with no ``__all__`` anywhere in this package the star-import above put it in the
# ``mdp`` namespace that every dexlift env config already uses, so reinstating the banned
# one-scalar closure was a single line, ``mdp.ContinuousSynergyJointPositionActionCfg(...)``, in
# any of them. Deleting the class is what makes the ban structural instead of a convention.
# NOTE: there is no local ``terminations`` module either. It held ONE function,
# ``abnormal_robot_state_scoped`` -- the upstream velocity-limit cut with ``asset_cfg.joint_ids``
# honoured, so it could be pointed at the arm alone. It was deleted with its only call site: the
# test is ``|joint_vel| > 2 * data.joint_vel_limits``, and ``joint_vel_limits`` is the very number
# Isaac Lab writes into PhysX as each joint's maximum velocity
# (``articulation.py:1773`` -> ``write_joint_velocity_limit_to_sim`` -> ``set_dof_max_velocities``),
# so no scoping of it can fire. See ``_drop_unreachable_abnormal_robot_cut`` in
# ``dexlift_ur5e_delto_env_cfg``.
from .rewards import *  # noqa: F401, F403
from .table_leg import *  # noqa: F401, F403

# The goal / red-green task-state markers. This module SUBCLASSES the dexsuite pose command rather
# than replacing it -- upstream already owns all three visualizers -- so the star-import above must
# stay first: ``TaskStateVisPoseCommandCfg`` is a strict extension of the
# ``ObjectUniformPoseCommandCfg`` it re-exports, not a shadow of it.
from .task_state_vis import *  # noqa: F401, F403
