# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert the linear parallel-jaw gripper URDF to a USD articulation (Isaac Sim).

Run on the server (needs Isaac Sim / Isaac Lab)::

    ./uwlab.sh -p scripts_v2/tools/conversions/convert_gripper_urdf.py \
        --input  source/uwlab_assets/uwlab_assets/local/Robots/LinearGripper/gripper.urdf \
        --output source/uwlab_assets/uwlab_assets/local/Robots/LinearGripper/linear_gripper.usd

Notes / things to verify after import (mirrors the asset-rigor lessons):
* The gripper has a ``<mimic>`` (right_finger_joint mimics finger_joint). Isaac's URDF
  importer should create a coupled joint; CONFIRM both jaws move from one command in sim.
  If only one jaw moves, we author the coupling in USD or add a driver.
* The root body must be ``robotiq_base_link`` and the single driver joint ``finger_joint``
  (renamed in the URDF). Verify the prim/joint names survived import unchanged.
* fix_base is False so the gripper is a free-floating articulation (the grasp sampler
  teleports its root) -- same as the reference gripper-only USD.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert the linear gripper URDF to USD.")
parser.add_argument("--input", type=str, required=True, help="Input .urdf path.")
parser.add_argument("--output", type=str, required=True, help="Output .usd path.")
parser.add_argument("--fix-base", action="store_true", help="Fix the base link (default: floating).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.dict import print_dict


def main() -> None:
    usd_dir = os.path.dirname(os.path.abspath(args.output))
    usd_name = os.path.basename(args.output)
    os.makedirs(usd_dir, exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=os.path.abspath(args.input),
        usd_dir=usd_dir,
        usd_file_name=usd_name,
        force_usd_conversion=True,
        fix_base=args.fix_base,
        merge_fixed_joints=False,
        # Position-controlled finger; gains are placeholders refined later by the actuator cfg.
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=200.0, damping=20.0),
        ),
    )

    print("-" * 80)
    print(f"Input  URDF: {cfg.asset_path}")
    print("Urdf importer config:")
    print_dict(cfg.to_dict(), nesting=0)  # type: ignore
    print("-" * 80)

    converter = UrdfConverter(cfg)
    print(f"Generated USD file: {converter.usd_path}")
    print("-" * 80)


if __name__ == "__main__":
    main()
    simulation_app.close()
