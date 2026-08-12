"""Compare the two UR5e+DELTO action-space variants field by field, without a simulator.

The scene, the commands and the terminations must be identical between
``DexLift-UR5eDelto-RelJointPos-Lift-v0`` and ``DexLift-UR5eDelto-Osc-Lift-v0``: they are one task
with two action spaces. Only ``actions``, the arm ACTUATOR the action space requires, and the OSC
variant's extra startup guard may differ.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--a", type=str, default="DexLift-UR5eDelto-RelJointPos-Lift-v0")
parser.add_argument("--b", type=str, default="DexLift-UR5eDelto-OSC-Lift-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import uwlab_tasks  # noqa: F401,E402

ALLOWED_PREFIXES = (
    "actions",
    # the action space dictates the arm actuator: implicit + PD gains for joint-position targets,
    # explicit + zero gains for the operational-space controller. Documented in both modules.
    "scene.robot.actuators",
    # the OSC variant adds one startup guard of its own
    "events.check_osc_arm_joints",
    # a diagnostic LABEL, not behaviour: the full-actuation guard names which variant it is
    # reporting for ("dexlift UR5e+DELTO" vs "... (OSC arm)"). Pre-existing.
    "events.check_hand_fully_actuated.params.context",
)


def flatten(obj, prefix=""):
    out = {}
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        items = vars(obj).items()
    elif isinstance(obj, dict):
        items = obj.items()
    else:
        return {prefix: repr(obj)}
    for k, v in items:
        p = f"{prefix}.{k}" if prefix else str(k)
        if hasattr(v, "__dict__") or isinstance(v, dict):
            out.update(flatten(v, p))
        else:
            out[p] = repr(v)
    return out


def main():
    a = flatten(parse_env_cfg(args_cli.a, device="cpu", num_envs=4))
    b = flatten(parse_env_cfg(args_cli.b, device="cpu", num_envs=4))
    keys = sorted(set(a) | set(b))
    diffs = []
    for k in keys:
        if k.startswith(ALLOWED_PREFIXES):
            continue
        if a.get(k) != b.get(k):
            diffs.append((k, a.get(k), b.get(k)))
    watched = [k for k in keys if k.startswith(("scene.", "commands.", "terminations."))]
    print(f"compared {len(keys)} leaf fields; {len(watched)} under scene/commands/terminations")
    for k, va, vb in diffs:
        print(f"DIFF {k}\n  A: {va}\n  B: {vb}")
    print("PARITY_OK" if not diffs else f"PARITY_FAIL ({len(diffs)})")
    sys.stdout.flush()


main()
print("SCRIPT_COMPLETE")
