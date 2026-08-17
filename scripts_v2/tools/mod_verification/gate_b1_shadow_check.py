"""Prove PYTHONPATH-prepended shadowing actually works (bead UWLab-qiao.1 5090-transfer follow-on).

Behavioural check, not a PYTHONPATH string: boot the Isaac app (uwlab_tasks cannot even be
imported before that -- it eagerly pulls in isaaclab -> omni.kit.app), then import uwlab_tasks and
print its __file__ (must resolve to OUR /root/uwlab_ur5edelto tree, not /root/simdist/repo, which
carries a conflicting editable install of the same package name), then list which DexLift task ids
actually registered in the gym registry.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402

print(f"UWLAB_TASKS_FILE={uwlab_tasks.__file__}", flush=True)
import uwlab_assets  # noqa: E402

print(f"UWLAB_ASSETS_FILE={uwlab_assets.__file__}", flush=True)

dexlift_ids = sorted(k for k in gym.registry.keys() if "DexLift" in k)
print(f"DEXLIFT_TASK_COUNT={len(dexlift_ids)}", flush=True)
for k in dexlift_ids:
    print(f"  {k}", flush=True)

target_ids = [
    "DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0",
    "DexLift-UR5eDelto-RelJointPos-TableLeg-Lift-v0",
]
missing = [t for t in target_ids if t not in gym.registry]
if missing:
    print(f"SHADOW_CHECK_FAILED missing={missing}", flush=True)
else:
    print("SHADOW_CHECK_OK both target task ids registered", flush=True)

print("SHADOW_CHECK_DONE", flush=True)
simulation_app.close()
