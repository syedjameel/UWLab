# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert policy-generated OmniReset reset states into a grasp dataset (bead: RestingEEGrasped
bridge, UR5e+DELTO leg campaign).

WHY THIS EXISTS. Two DELTO reset-state types -- ObjectRestingEEGrasped and
ObjectPartiallyAssembledEEGrasped -- both replay a grasp dataset through differential IK
(``events.reset_end_effector_from_grasp_dataset``) to place the end effector. For ``leg200mm``
there is no such dataset: the geometric grasp sampler this project otherwise uses rejects 100% of
candidate poses on this object (structural, not tunable), and ``record_grasps.py`` is banned by the
user. But a grasp dataset entry is just a HELD moment expressed as a coordinate change -- the
object's pose relative to the end effector, plus the hand's joint positions -- and
``generate_reset_states_policy.py`` already produced 300 such moments for this exact leg
(``resets_ObjectAnywhereEEGrasped.pt``), each one a full scene snapshot at a step
``held_with_probe`` accepted as a genuine, tracked hold. This script re-expresses those same
moments in the grasp_relative_pose schema; it invents nothing.

THE ONE HARD PART: FRAME. The recorded snapshot (``env.scene.get_state(is_relative=True)``, see
``recorders.py::StableStateRecorder``) has the robot's ROOT pose and JOINT POSITIONS, not the
palm's world pose -- getting that needs forward kinematics. The honest way to get it is to replay
each state in the SAME task the policy ran in (dexlift table-leg, RelJointPos, so joint order/count
match exactly what was recorded), write robot root pose + joint positions into a fresh env, refresh
kinematics with ``sim.forward()`` (updates the articulation's cached body poses from the just-written
joint state without integrating any physics -- no step, no time advance), and read the palm body's
now-correct ``body_pos_w``/``body_quat_w`` off the live scene. NEVER hand-roll this transform: the
consumer (``events.py::reset_end_effector_from_grasp_dataset``, read directly, not guessed) composes
``T_world_gripper = T_world_object * T_relative`` via ``math_utils.combine_frame_transforms``, i.e.
``relative_position``/``relative_orientation`` store the GRIPPER'S pose EXPRESSED IN THE OBJECT'S
FRAME -- exactly what ``GraspRelativePoseRecorder.record_pre_reset`` (recorders.py) already computes
via ``math_utils.subtract_frame_transforms(obj_pos, obj_quat, gripper_pos, gripper_quat)``. This
script calls that SAME function the SAME way, not ``assembly_keypoints.Offset`` -- Offset is a thin
wrapper over the identical ``combine_frame_transforms``/``quat_inv``/``quat_apply`` primitives, built
for a compile-time-fixed pose composed with ONE runtime tensor; both of our operands here are
runtime, per-episode batched tensors, so it adds a wrapper with no additional safety over calling the
library function directly the way the recorder already does. Quaternion convention: (w, x, y, z),
confirmed by ``recorders.py``'s own inline comment on ``obj_root_state[:, 3:7]``, and by inspecting
the existing DeltoBlock grasps.pt on disk (dtype/shape checked, not assumed).

EE FRAME: ``rl_dg_mount`` (``delto_cfg.DELTO_EE_BODY``), the SAME body ``held_with_probe`` itself
measures (``held_check.py``'s default ``robot_cfg``) and the SAME body
``events.reset_end_effector_pose_from_grasp_dataset.robot_ik_cfg`` is repointed to for every DELTO
variant (``delto_cfg._EE_BODY_TERMS``). Getting this body wrong would put every replayed grasp at a
fixed, silent offset from the truth -- it would still "look" almost right, hence reading it from the
seam module rather than assuming.

OUTPUT PATH: ``{dataset_dir}/Grasps/{object_name}/grasps.pt``, where ``object_name`` is derived from
the leg's own USD path the SAME way the consumer derives it
(``utils.object_name_from_usd``, ``events.py::_compute_grasp_dataset_path``) -- not hardcoded, so a
future object swap does not silently point this script at the wrong directory.

PROVENANCE -- MEASURED 2026-08-17, DL_A6000, GPU1 (record it here; a briefing that preceded this run
described the frame backwards -- see below -- and the next person to touch this script should
inherit the corrected, source-verified account, not a restated one):

* Input bank: ``resets_ObjectAnywhereEEGrasped.pt`` for
  ``OneLegInsertionFixture__SquareTableLeg200mmDecomp``, produced by
  ``generate_reset_states_policy.py``. Had grown to 532 accepted episodes by conversion time (300 at
  proposal time above; the generator kept running in between -- check the live count rather than
  trusting the number quoted earlier in this docstring).
* Converted: 532 / 532. Output ``Grasps/SquareTableLeg200mmDecomp/grasps.pt``, on-disk schema
  cross-checked against the existing production ``Grasps/DeltoBlock/grasps.pt`` before writing
  (top key ``grasp_relative_pose``; ``relative_position`` list of (3,) float32; ``relative_orientation``
  list of (4,) float32; ``gripper_joint_positions`` dict of 20 DELTO joint names -> list of 0-d
  float32). Matched exactly.
* Quaternion convention: **(w, x, y, z)**, scalar-first.
* EE frame: **``rl_dg_mount``** (resolved live to body id 7 in the conversion run).
* FRAME-DIRECTION CORRECTION (the thing to carry forward): an earlier verbal briefing for this
  bridge described ``relative_position``/``relative_orientation`` as "the object relative to the end
  effector." Reading the consumer directly (``events.py::reset_end_effector_from_grasp_dataset``,
  the ``combine_frame_transforms(object_pos_w, object_quat_w, sampled_rel_positions,
  sampled_rel_quaternions)`` call) shows the opposite: it composes
  ``T_world_gripper = T_world_object * T_relative``, so the stored quantity is the **GRIPPER's pose
  expressed in the OBJECT's frame** -- i.e. "gripper relative to object", matching
  ``GraspRelativePoseRecorder``'s own ``subtract_frame_transforms(obj_pos, obj_quat, gripper_pos,
  gripper_quat)`` (recorders.py) exactly. This script implements the corrected direction. Getting
  this backwards would silently place every replayed grasp on the wrong side of the object.
* Round-trip verification (``verify_grasp_dataset_roundtrip.py``), palm-object distance, three
  independent measurements that should agree if the schema and frame are both right -- they do:

  =====================================================  =======  =======  =======  =======
  measurement                                            min mm   mean mm  max mm   median mm
  =====================================================  =======  =======  =======  =======
  FK at conversion time (this script, source states)      126.26   194.35   267.60   196.80
  IK replay, ObjectAnywhereEEGrasped (object airborne)     150.77   192.08   234.51   195.44
  IK replay, ObjectRestingEEGrasped (object AT TABLE)      123.26   189.49   249.13   191.56
  =====================================================  =======  =======  =======  =======

  The resting-height run is the important one: ``reset_insertive_object_pose_from_reset_states``
  (which feeds ``ObjectRestingEEGrasped``) draws from ``resets_ObjectAnywhereEEAnywhere.pt``, whose
  measured object clearance for this pair is median 0.019 m / 91% below 0.05 m -- genuinely
  table-resting, versus the 0.296-0.704 m airborne band the source grasps were sampled from (also
  measured, not assumed: ``resets_ObjectAnywhereEEGrasped.pt``'s own object-vs-table clearance).
  Despite a ~300-700 mm gap between the height the grasps were RECORDED at and the height they are
  REPLAYED at, all three distance distributions land on top of each other (means within 5 mm of each
  other) and hand-joint posture means match to three decimals across the airborne/resting replays.
  No sign of IK contortion or degraded reachability at table height for this leg/DELTO combination.
  Not checked: per-env joint-limit margins (aggregate stats only) -- if real-generation acceptance
  rates for Resting/PartiallyAssembled come back poor, that per-env breakdown is the next diagnostic
  to run, not a re-litigation of this frame/schema verification.

Run (one Isaac process; never via uwlab.sh):
    PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl" \\
        timeout -s KILL 300 <python> -u scripts_v2/tools/convert_policy_states_to_grasps.py \\
        --states_pt ./Datasets_ur5e_delto/OmniReset/Resets/OneLegInsertionFixture__SquareTableLeg200mmDecomp/resets_ObjectAnywhereEEGrasped.pt \\
        --dataset_dir ./Datasets_ur5e_delto/OmniReset
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert policy-generated reset states into a grasp dataset.")
parser.add_argument(
    "--task", type=str, default="DexLift-UR5eDelto-RelJointPos-TableLeg-Lift-Play-v0",
    help="The dexlift task the states were RECORDED against (joint order/count must match exactly).",
)
parser.add_argument(
    "--states_pt", type=str, required=True,
    help="Path to the policy-generated resets_ObjectAnywhereEEGrasped.pt (or similar) to convert.",
)
parser.add_argument(
    "--dataset_dir", type=str, default="./Datasets_ur5e_delto/OmniReset",
    help="Root Datasets_ur5e_delto/OmniReset directory. Output is written under {this}/Grasps/<object>/grasps.pt.",
)
parser.add_argument(
    "--num_envs", type=int, default=300,
    help="Batch size for the FK replay (all states if <= this; otherwise processed in chunks of this size).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402

import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import uwlab_tasks  # noqa: F401,E402
import uwlab_tasks.manager_based.manipulation.omnireset.mdp as task_mdp  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import gymnasium as gym  # noqa: E402

# The DELTO's palm/mount body -- see module docstring. Read from the seam module rather than
# hardcoding a second copy of this string.
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.delto_cfg import (  # noqa: E402
    DELTO_EE_BODY,
    _DELTO_HAND_JOINTS,
)


def main() -> None:
    print(f"[convert] loading recorded states: {args_cli.states_pt}", flush=True)
    states = torch.load(args_cli.states_pt, map_location="cpu", weights_only=False)
    robot_state = states["initial_state"]["articulation"]["robot"]
    # Accept either key. "object" is the raw, never-rekeyed dexlift-scene name this script
    # originally assumed; every copy of resets_ObjectAnywhereEEGrasped.pt actually on disk as of
    # 2026-08-20 (live, staged, and the pre-Task-1 backup) has already been rekeyed to
    # "insertive_object" (rekey_dexlift_reset_states.py step 1) -- the raw "object"-keyed dump this
    # script's own docstring was verified against no longer survives anywhere in this tree. Per
    # that same rekey script's docstring, the rename is "pure key rename, no coordinate transform
    # -- both names refer to the same physical body in the same frame convention", so reading
    # either key here is safe and does not change any of the FRAME/PROVENANCE claims documented
    # above.
    rigid_object_state = states["initial_state"]["rigid_object"]
    object_state = rigid_object_state.get("object", rigid_object_state.get("insertive_object"))
    if object_state is None:
        raise KeyError(
            f"Neither 'object' nor 'insertive_object' found in rigid_object keys "
            f"{sorted(rigid_object_state.keys())} of {args_cli.states_pt!r}."
        )
    n_episodes = len(robot_state["root_pose"])
    print(f"[convert] {n_episodes} recorded episodes", flush=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=min(args_cli.num_envs, n_episodes))
    # Bumped 2**24 -> 2**28 (16 MiB -> 256 MiB) for local single-GPU runs (16 GB 4070 Ti SUPER, not
    # the A6000 this script was originally verified on): PxgCudaDeviceMemoryAllocator failed to
    # allocate exactly 268435456 bytes (256 MiB) at scene creation with the old 16 MiB stack and
    # --num_envs 300 on this card, matching the standing guidance for small-env inspection runs on
    # this box (the inherited-default 3.75 GiB sizing is for thousands of envs and also fails, just
    # the other direction -- too small vs too big, both wrong for a few-hundred-env batch here).
    env_cfg.sim.physx.gpu_collision_stack_size = 2**28
    env_cfg.seed = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    obj = env.scene["object"]
    batch_size = env.num_envs

    # -- EE body (rl_dg_mount) and DELTO hand joints (rj_dg_[1-5]_[1-4], all 20), resolved off the
    # LIVE articulation rather than assumed indices -- see module docstring for why these two
    # particular names, sourced from delto_cfg.py, not guessed.
    ee_cfg = SceneEntityCfg("robot", body_names=DELTO_EE_BODY)
    ee_cfg.resolve(env.scene)
    assert len(ee_cfg.body_ids) == 1, f"{DELTO_EE_BODY!r} must resolve to exactly one body, got {ee_cfg.body_ids}"
    palm_id = ee_cfg.body_ids[0]

    hand_joint_ids, hand_joint_names = robot.find_joints(_DELTO_HAND_JOINTS, preserve_order=True)
    print(f"[convert] EE body: {DELTO_EE_BODY} (id {palm_id})", flush=True)
    print(f"[convert] hand joints ({len(hand_joint_names)}): {hand_joint_names}", flush=True)

    out_rel_pos: list[torch.Tensor] = []
    out_rel_quat: list[torch.Tensor] = []
    out_gripper_joints: dict[str, list[torch.Tensor]] = {name: [] for name in hand_joint_names}
    palm_obj_dist_mm: list[float] = []

    for chunk_start in range(0, n_episodes, batch_size):
        chunk_end = min(chunk_start + batch_size, n_episodes)
        chunk = list(range(chunk_start, chunk_end))
        env_ids = torch.arange(len(chunk), device=env.device)

        # -- stack the recorded per-episode tensors for this chunk.
        robot_root_pose = torch.stack([robot_state["root_pose"][i] for i in chunk]).to(env.device)
        robot_root_vel = torch.stack([robot_state["root_velocity"][i] for i in chunk]).to(env.device)
        robot_joint_pos = torch.stack([robot_state["joint_position"][i] for i in chunk]).to(env.device)
        robot_joint_vel = torch.stack([robot_state["joint_velocity"][i] for i in chunk]).to(env.device)
        obj_root_pose = torch.stack([object_state["root_pose"][i] for i in chunk]).to(env.device)
        obj_root_vel = torch.stack([object_state["root_velocity"][i] for i in chunk]).to(env.device)

        # -- is_relative=True convention (recorders.py wrote relative-to-env-origin positions):
        # add env_origins back before writing into a LIVE scene, exactly mirroring
        # MultiResetManager._reset_to (events.py) -- same operation, same reason.
        robot_root_pose = robot_root_pose.clone()
        robot_root_pose[:, :3] += env.scene.env_origins[env_ids]
        obj_root_pose = obj_root_pose.clone()
        obj_root_pose[:, :3] += env.scene.env_origins[env_ids]

        robot.write_root_pose_to_sim(robot_root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim(robot_root_vel, env_ids=env_ids)
        robot.write_joint_state_to_sim(robot_joint_pos, robot_joint_vel, env_ids=env_ids)
        # Match _reset_to's own follow-up (events.py): keep the PD targets consistent with the
        # teleported state so nothing snaps on the next real step (irrelevant here since we never
        # step, but zero-cost and keeps this replay byte-identical to the established idiom).
        robot.set_joint_position_target(robot_joint_pos, env_ids=env_ids)
        robot.set_joint_velocity_target(robot_joint_vel, env_ids=env_ids)
        obj.write_root_pose_to_sim(obj_root_pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim(obj_root_vel, env_ids=env_ids)

        env.scene.write_data_to_sim()
        # Refresh the articulation's cached body poses (forward kinematics) from the DOF state we
        # just wrote, WITHOUT integrating any physics -- see module docstring. No env.step() call
        # anywhere in this script.
        env.sim.forward()

        palm_pos_w = robot.data.body_pos_w[env_ids, palm_id, :]
        palm_quat_w = robot.data.body_quat_w[env_ids, palm_id, :]
        obj_pos_w = obj.data.root_pos_w[env_ids]
        obj_quat_w = obj.data.root_quat_w[env_ids]

        # T_gripper_in_object -- the SAME call, in the SAME argument order, GraspRelativePoseRecorder
        # uses (recorders.py:75), which is what the consumer's combine_frame_transforms (events.py)
        # expects on the other end. See module docstring.
        rel_pos, rel_quat = math_utils.subtract_frame_transforms(obj_pos_w, obj_quat_w, palm_pos_w, palm_quat_w)

        # Read the hand joint positions back off the LIVE articulation post-write (round-trips
        # through the same write call the arm state did, rather than re-slicing the input tensor,
        # so a joint-order mismatch between the recorded state and this env would show up here too).
        gripper_joint_pos = robot.data.joint_pos[env_ids][:, hand_joint_ids]

        dist_mm = torch.linalg.norm(palm_pos_w - obj_pos_w, dim=-1) * 1000.0
        palm_obj_dist_mm.extend(dist_mm.tolist())

        for i in range(len(chunk)):
            out_rel_pos.append(rel_pos[i].cpu())
            out_rel_quat.append(rel_quat[i].cpu())
            for j, name in enumerate(hand_joint_names):
                out_gripper_joints[name].append(gripper_joint_pos[i, j].cpu())

        print(f"[convert] processed {chunk_end}/{n_episodes}", flush=True)

    # -- output path: derived off the leg's own USD path, the SAME way the consumer derives it
    # (events.py::_compute_grasp_dataset_path), not hardcoded.
    obj_name = task_mdp.utils.object_name_from_usd(env.scene["object"].cfg.spawn.usd_path)
    output_dir = os.path.join(args_cli.dataset_dir, "Grasps", obj_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.abspath(os.path.join(output_dir, "grasps.pt"))

    grasp_data = {
        "grasp_relative_pose": {
            "relative_position": out_rel_pos,
            "relative_orientation": out_rel_quat,
            "gripper_joint_positions": out_gripper_joints,
        }
    }
    torch.save(grasp_data, output_path)

    dist_t = torch.tensor(palm_obj_dist_mm)
    print("\n=== CONVERTER RESULT ===", flush=True)
    print(f"grasps written: {len(out_rel_pos)}", flush=True)
    print(f"output: {output_path}", flush=True)
    print(
        f"palm-object distance at FK (mm): min {dist_t.min():.2f}  mean {dist_t.mean():.2f}"
        f"  max {dist_t.max():.2f}  median {dist_t.median():.2f}",
        flush=True,
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
