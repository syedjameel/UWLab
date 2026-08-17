# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Round-trip verification for a converted grasps.pt (bead: RestingEEGrasped bridge).

Companion to ``convert_policy_states_to_grasps.py``. That script WRITES a grasps.pt; this one
replays it through the ACTUAL consumer (``events.reset_end_effector_from_grasp_dataset``, the same
event term ``OmniReset-UR5eDelto-ObjectAnywhereEEGrasped-v0`` already has wired in) and checks the
end effector ends up genuinely near the object it was supposedly grasping, not flung or offset.

WHAT THIS DOES NOT CHECK. This confirms the FILE IS CONSUMABLE and the composed pose is
GEOMETRICALLY sane (palm lands close to the object, consistent with the distances measured at
conversion time). It does not step physics or check fingertip contact forces -- that would need
several sim steps and its own settle window, which is a stronger but separate claim than "the
schema round-trips correctly."

RESULTS FOR THE LEG (``Grasps/SquareTableLeg200mmDecomp/grasps.pt``, measured 2026-08-17,
DL_A6000): run once against ``OmniReset-UR5eDelto-ObjectAnywhereEEGrasped-v0`` (object airborne,
consumer's own dataset-load log confirmed all 532 entries, no gripper-joint fallback warning) and
once against ``OmniReset-UR5eDelto-ObjectRestingEEGrasped-v0`` (object at genuine table height --
see ``convert_policy_states_to_grasps.py``'s own PROVENANCE section for the full numbers and why the
airborne/resting gap mattered). Both landed within ~5 mm mean palm-object distance of the
conversion-time FK measurement and of each other; hand-joint posture means matched to three
decimals. Full three-way table lives in the converter script's docstring, not duplicated here --
read it there before re-deriving these numbers.

Run (one Isaac process; never via uwlab.sh). Hydra overrides select the object pair exactly like
record_reset_states.py -- pass them as extra CLI tokens, they flow straight to hydra_args:
    PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl" \\
        timeout -s KILL 300 <python> -u scripts_v2/tools/verify_grasp_dataset_roundtrip.py \\
        --task OmniReset-UR5eDelto-ObjectAnywhereEEGrasped-v0 --num_envs 64 \\
        env.scene.insertive_object=leg200mm env.scene.receptive_object=onelegfixture
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Round-trip a converted grasps.pt through its actual consumer.")
parser.add_argument("--task", type=str, default="OmniReset-UR5eDelto-ObjectAnywhereEEGrasped-v0")
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

import uwlab_tasks.manager_based.manipulation.omnireset.mdp as task_mdp  # noqa: F401,E402
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.delto_cfg import (  # noqa: E402
    DELTO_EE_BODY,
)
from uwlab_tasks.utils.hydra import hydra_task_compose


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    print(f"[verify] insertive object usd: {env.scene['insertive_object'].cfg.spawn.usd_path}", flush=True)

    ee_cfg = SceneEntityCfg("robot", body_names=DELTO_EE_BODY)
    ee_cfg.resolve(env.scene)
    assert len(ee_cfg.body_ids) == 1, f"{DELTO_EE_BODY!r} must resolve to exactly one body, got {ee_cfg.body_ids}"
    palm_id = ee_cfg.body_ids[0]

    # env.reset() runs the FULL 'reset' event chain, including
    # reset_end_effector_pose_from_grasp_dataset -- which loads and consumes the grasps.pt this
    # verifies, through 25 differential-IK iterations, exactly as it would during real training.
    obs, _ = env.reset()

    palm_pos_w = env.scene["robot"].data.body_pos_w[:, palm_id, :]
    obj_pos_w = env.scene["insertive_object"].data.root_pos_w
    dist_mm = torch.linalg.norm(palm_pos_w - obj_pos_w, dim=-1) * 1000.0

    gripper_pos = env.scene["robot"].data.joint_pos
    # Sanity range check: DELTO joints should be near a CLOSED posture (this dataset only ever
    # wrote closed-hand snapshots), not wide open (which 0.0-fallback would silently produce if
    # gripper_joint_positions failed to load for this gripper's joints -- see events.py's own
    # "replayed at 0.0" warning path).
    hand_joint_ids, hand_joint_names = env.scene["robot"].find_joints(["rj_dg_[1-5]_[1-4]"], preserve_order=True)
    hand_joint_pos = gripper_pos[:, hand_joint_ids]

    print("\n=== ROUND-TRIP RESULT ===", flush=True)
    print(f"num envs: {env.num_envs}", flush=True)
    print(
        f"palm-object distance post-reset (mm): min {dist_mm.min():.2f}  mean {dist_mm.mean():.2f}"
        f"  max {dist_mm.max():.2f}  median {dist_mm.median():.2f}",
        flush=True,
    )
    print(
        f"hand joint positions post-reset (rad): min {hand_joint_pos.min():.4f}"
        f"  mean {hand_joint_pos.mean():.4f}  max {hand_joint_pos.max():.4f}",
        flush=True,
    )
    print(f"any NaN in palm/obj pose: {bool(torch.isnan(palm_pos_w).any() or torch.isnan(obj_pos_w).any())}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
