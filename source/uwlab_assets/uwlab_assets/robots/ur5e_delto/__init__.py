# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# There is deliberately no ``actions`` module here. The DELTO's fully actuated twenty-joint action
# term lives in ``uwlab_assets.robots.ur10e_delto.actions`` and is a property of the HAND; a second
# copy keyed to this arm would be a second place for the joint names -- and for the ban on the
# one-scalar closure -- to drift.
from .ur5e_delto import *
