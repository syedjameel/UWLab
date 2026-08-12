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
from .rewards import *  # noqa: F401, F403
from .table_leg import *  # noqa: F401, F403
