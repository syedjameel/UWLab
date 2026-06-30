# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A/B compare grasp sampling between OUR linear gripper and the DEFAULT (working) 2F-85.

Runs the SAME grasp-sampling simulation (same object) for one gripper, logs the metrics
that discriminate a good grasp from a bad one, and writes a JSON summary. Run it once per
gripper (separate processes -- Isaac corrupts the GPU PhysX context if two envs are built in
one process), then use --compare to print them side by side.

WHY these metrics: a stable grasp = the jaws approach the object on a good line, close on it,
and hold it still. We log, per evaluated grasp episode (at timeout, the moment success is
judged):
  * approach_world_z : world Z component of the gripper's approach axis. ~ -1 = top-down
    (stable on a flat object), ~ 0 = horizontal/side (slips/pivots on a peg or slab edge).
  * grip_width       : driver finger_joint at timeout (0 = open/missed, mid = closed on object).
  * obj_disp         : |object_pos - object_pos_at_episode_start| (how far the object was
    knocked/dragged; the success check fails it past max_pos_deviation).
  * obj_linvel       : object linear speed at timeout (success needs it near 0 = settled).
  * success          : whether the env's recorder exported this episode as a successful grasp.

USAGE (laptop, 6 GB -- keep the PhysX buffer trims; they are passed through as hydra args):
  WORKER (one gripper):
    ./uwlab.sh -p scripts_v2/tools/conversions/compare_grippers_grasp.py \
        --gripper 2f85  --object pcb --num_grasps 25 --out /tmp/cmp_2f85.json \
        env.sim.physx.gpu_collision_stack_size=67108864 \
        env.sim.physx.gpu_max_rigid_contact_count=2097152 \
        env.sim.physx.gpu_max_rigid_patch_count=2097152 \
        env.sim.physx.gpu_total_aggregate_pairs_capacity=2097152 \
        env.sim.physx.gpu_found_lost_aggregate_pairs_capacity=2097152
    ./uwlab.sh -p scripts_v2/tools/conversions/compare_grippers_grasp.py \
        --gripper linear --object pcb --num_grasps 25 --out /tmp/cmp_linear.json  <same trims>
  COMPARE (no Isaac):
    ~/miniconda3/envs/leisaac/bin/python scripts_v2/tools/conversions/compare_grippers_grasp.py \
        --compare /tmp/cmp_2f85.json /tmp/cmp_linear.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------------------
# --compare mode: pure-python, NO Isaac. Handle it before importing the simulator.
# ---------------------------------------------------------------------------------------
TASKS = {
    "2f85": "OmniReset-Robotiq2f85-GraspSampling-v0",
    "linear": "OmniReset-LinearGripper-GraspSampling-v0",
}


def _fmt(xs):
    if not xs:
        return "  n=0"
    import statistics as st

    return f"  n={len(xs)} mean={st.mean(xs):+.4f} min={min(xs):+.4f} max={max(xs):+.4f}"


def _print_compare(path_a: str, path_b: str) -> None:
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    print("\n================== GRASP A/B COMPARISON ==================")
    for d in (a, b):
        ev = d["evaluated"]
        succ = d["succeeded"]
        rate = (succ / ev * 100.0) if ev else 0.0
        print(f"\n### {d['gripper'].upper()}  object={d['object']}  task={d['task']}")
        print(f"  success: {succ}/{ev}  ({rate:.1f}% )")
        print(f"  approach_world_z (-1=top-down, 0=side):{_fmt(d['approach_world_z'])}")
        print(f"  grip_width finger_joint:               {_fmt(d['grip_width'])}")
        print(f"  obj_disp from start (m):               {_fmt(d['obj_disp'])}")
        print(f"  obj_linvel at timeout (m/s):           {_fmt(d['obj_linvel'])}")
    print("\n---------------------------------------------------------")
    print("Read: if our approach_world_z ~ 0 while 2F-85 ~ -1, ours grasps side-on (unstable).")
    print("If our obj_disp >> 2F-85, ours knocks/drags the object (bad grasp line or scale).")
    print("=========================================================\n")


def _early_args():
    """Peel off our flags before AppLauncher sees argv; leave the rest for hydra."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gripper", choices=list(TASKS.keys()))
    p.add_argument("--object", type=str, default="pcb")
    p.add_argument("--num_grasps", type=int, default=25)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--compare", nargs=2, default=None)
    return p.parse_known_args()


_args, _remaining = _early_args()

if _args.compare is not None:
    _print_compare(_args.compare[0], _args.compare[1])
    sys.exit(0)

if _args.gripper is None:
    print("ERROR: pass --gripper {2f85|linear} (worker mode) or --compare A.json B.json.")
    sys.exit(2)

# ---------------------------------------------------------------------------------------
# WORKER mode: launch Isaac and run grasp sampling for the chosen gripper.
# ---------------------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

_parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_parser)
# Re-inject our flags so AppLauncher's parser ignores them (we already parsed them).
_app_args, _hydra_args = _parser.parse_known_args(_remaining)
_app_args.headless = True
app_launcher = AppLauncher(_app_args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import uwlab_tasks  # noqa: F401, E402
import uwlab_tasks.manager_based.manipulation.omnireset.mdp as task_mdp  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402

TASK = TASKS[_args.gripper]


@hydra_task_compose(TASK, "env_cfg_entry_point", hydra_args=_hydra_args)
def main(env_cfg, agent_cfg) -> None:
    from typing import cast

    from isaaclab.managers.recorder_manager import DatasetExportMode

    from uwlab.utils.datasets.torch_dataset_file_handler import TorchDatasetFileHandler

    # Select the object variant (pcb/peg/cube/...) exactly like record_grasps' hydra path.
    if _args.object and "scene.object" in getattr(env_cfg, "variants", {}):
        env_cfg.scene.object = env_cfg.variants["scene.object"][_args.object]

    # A recorder so we can read the env's own success verdict (export mode = succeeded only).
    env_cfg.recorders = task_mdp.GraspRelativePoseRecorderManagerCfg(
        robot_name="robot", object_name="object", gripper_body_name="robotiq_base_link"
    )
    env_cfg.recorders.dataset_export_dir_path = "/tmp/_cmp_rec"
    env_cfg.recorders.dataset_filename = f"{_args.gripper}.pt"
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = TorchDatasetFileHandler
    env_cfg.scene.num_envs = 1
    env_cfg.seed = None

    env = cast(ManagerBasedRLEnv, gym.make(TASK, cfg=env_cfg)).unwrapped

    # Gripper approach axis (gripper-local) from the gripper USD metadata.
    gripper = env.scene["robot"]
    meta = task_mdp.utils.read_metadata_from_usd_directory(gripper.cfg.spawn.usd_path)
    approach_local = torch.tensor(
        [float(v) for v in meta.get("gripper_approach_direction", [0, 0, 1])],
        device=env.device,
        dtype=torch.float32,
    )
    finger_idx = list(gripper.joint_names).index("finger_joint") if "finger_joint" in gripper.joint_names else 0

    env.reset()
    actions = -torch.ones(env.action_space.shape, device=env.device, dtype=torch.float32)  # close

    finger_offset = float(meta.get("finger_offset", 0.1))
    metrics = {
        k: []
        for k in ("approach_world_z", "grip_width", "obj_disp", "obj_disp_early", "obj_jaw_lat", "obj_linvel", "grip_disp", "success")
    }
    obj_init = None
    grip_init = None  # gripper base pose at the grasp (to measure drift over the episode)
    grip_disp = 0.0
    obj_disp_early = 0.0  # object displacement captured just before gravity turns on (~step 10)
    obj_jaw_lat = 0.0  # lateral distance object->gripper approach line (is it between the jaws?)
    prev_el = 0
    prev_snapshot = None
    succ_prev = 0
    evaluated = 0

    while evaluated < _args.num_grasps:
        obj = env.scene["object"]
        el = int(env.episode_length_buf[0].item())
        # Snapshot the CURRENT (post-previous-step) state; on the step that resets, this was the
        # timeout state. Capture the object's start pose at the first step of each episode.
        if el <= 1 or obj_init is None:
            obj_init = obj.data.root_pos_w[0].clone()
            grip_init = gripper.data.root_pos_w[0].clone()
        grip_disp = float((gripper.data.root_pos_w[0] - grip_init).norm().item())
        gq = gripper.data.root_quat_w[0:1]
        approach_w = math_utils.quat_apply(gq, approach_local.unsqueeze(0))[0]
        # Just before gravity (the grasp env turns gravity on at episode_length_buf>10): is the
        # object knocked away during the close (disp_early large) and is it between the jaws?
        if el == 10:
            obj_disp_early = float((obj.data.root_pos_w[0] - obj_init).norm().item())
            base = gripper.data.root_pos_w[0]
            ahat = approach_w / (approach_w.norm() + 1e-8)
            d = obj.data.root_pos_w[0] - base
            obj_jaw_lat = float((d - (d @ ahat) * ahat).norm().item())
        snapshot = {
            "approach_world_z": float(approach_w[2].item()),
            "grip_width": float(gripper.data.joint_pos[0, finger_idx].item()),
            "obj_disp": float((obj.data.root_pos_w[0] - obj_init).norm().item()),
            "obj_disp_early": obj_disp_early,
            "obj_jaw_lat": obj_jaw_lat,
            "obj_linvel": float(obj.data.root_lin_vel_w[0].norm().item()),
            "grip_disp": grip_disp,
        }

        _, _, terminated, truncated, _ = env.step(actions)
        done = bool((terminated | truncated)[0].item())
        new_el = int(env.episode_length_buf[0].item())

        if done or (prev_el > 0 and new_el == 0):
            # Episode just ended; prev/this snapshot is the timeout state. Success = recorder grew.
            succ_now = env.recorder_manager.exported_successful_episode_count
            was_success = succ_now > succ_prev
            succ_prev = succ_now
            snap = prev_snapshot if prev_snapshot is not None else snapshot
            for k in ("approach_world_z", "grip_width", "obj_disp", "obj_disp_early", "obj_jaw_lat", "obj_linvel", "grip_disp"):
                metrics[k].append(snap[k])
            metrics["success"].append(1 if was_success else 0)
            evaluated += 1
            obj_init = None
            obj_disp_early = 0.0
            obj_jaw_lat = 0.0
            if evaluated % 5 == 0:
                print(f"[{_args.gripper}] evaluated {evaluated}/{_args.num_grasps}  succeeded {succ_prev}", flush=True)

        prev_el = new_el
        prev_snapshot = snapshot
        if env.sim.is_stopped():
            break

    out = {
        "gripper": _args.gripper,
        "task": TASK,
        "object": _args.object,
        "evaluated": evaluated,
        "succeeded": int(sum(metrics["success"])),
        **{
            k: metrics[k]
            for k in ("approach_world_z", "grip_width", "obj_disp", "obj_disp_early", "obj_jaw_lat", "obj_linvel", "grip_disp")
        },
    }
    out_path = _args.out or f"/tmp/cmp_{_args.gripper}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[{_args.gripper}] DONE  evaluated={evaluated}  succeeded={out['succeeded']}  -> {out_path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    os._exit(0)  # dodge the Isaac shutdown deadlock (results already written)
