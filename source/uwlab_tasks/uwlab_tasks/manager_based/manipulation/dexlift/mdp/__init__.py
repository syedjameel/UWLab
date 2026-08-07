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

from .rewards import *  # noqa: F401, F403
