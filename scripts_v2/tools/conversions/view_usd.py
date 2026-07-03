# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open a USD in the Isaac Sim GUI for visual inspection (no physics -- safe on the laptop).

Just opens the stage and idles the app so you can orbit/zoom. Does NOT press Play, so it
won't trigger the physics-stepping hang. Close the window (or Ctrl+C) to exit. Run::

    ./uwlab.sh -p scripts_v2/tools/conversions/view_usd.py \
        --usd source/uwlab_assets/uwlab_assets/local/Robots/Ur5eLinearGripper/ur5e_linear_gripper.usd
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View a USD in the Isaac Sim GUI.")
parser.add_argument("--usd", type=str, required=True, help="USD file to open.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False  # GUI

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os

import omni.usd

omni.usd.get_context().open_stage(os.path.abspath(args.usd))
print(f"Opened {args.usd} — orbit/zoom to inspect the gripper mount. Close the window to exit.")

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
