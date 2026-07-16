# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Print a robot body's base-frame pose at given joint angles (sim convention).

Used to convert the wrist camera's ArUco calibration (base-frame, from
diffusion_policy 0_camera_calibrate.py + 2_get_isaacsim_extrinsics.py) into the
link-relative offset the sim wrist ``TiledCamera`` needs: the camera is parented to
``robotiq_base_link``, so offset = inv(T_base_link) @ T_base_cam. The robot spawns at
the origin with identity rotation, so world pose == base-frame pose.

Pendant joints must be converted to sim convention first (q1_sim = q1_real - 90 deg;
see diffusion_policy real_world/ur10e_kinematics.py).

    conda activate leisaac && ./uwlab.sh -p scripts_v2/tools/conversions/get_link_pose_ur10e.py \
        --joint_angles -20.42 -98.08 138.53 -130.43 -89.95 -20.42
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="OmniReset-UR10eLinearGripper-CameraAlign-v0")
parser.add_argument("--body", type=str, default="robotiq_base_link")
parser.add_argument(
    "--joint_angles", type=float, nargs=6, required=True,
    help="Arm joint angles in degrees, SIM convention (pendant q1 - 90).",
)
parser.add_argument("--no_trims", action="store_true", help="Skip the laptop PhysX buffer trims.")
parser.add_argument("--out", type=str, default=None,
                    help="Also hold the pose (zero OSC deltas) and save each camera's frame here.")
parser.add_argument("--fabric_test", action="store_true",
                    help="Probe whether update_world_xforms() fixes the stale wrist camera.")
parser.add_argument("--finger_pos", type=float, nargs="*", default=None,
                    help="Force the finger joints to these positions (meters; NEGATIVE = wider "
                    "than nominal open) and save a wrist blend per value -- tests whether the "
                    "modeled open jaw separation is narrower than the real gripper's.")
parser.add_argument("--no_fabric", action="store_true",
                    help="Disable PhysX fabric (physics writes USD) -- link-attached cameras track.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True  # the CameraAlign scene contains TiledCameras
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import uwlab_tasks  # noqa: F401

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)

    # Laptop PhysX buffer trims (6 GB GPU) -- the RGB cfgs request 2 GB collision stacks.
    if not args.no_trims:
        cfg.sim.physx.gpu_collision_stack_size = 67108864
        cfg.sim.physx.gpu_max_rigid_contact_count = 2097152
        cfg.sim.physx.gpu_max_rigid_patch_count = 2097152
        cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2097152
        cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2097152

    if args.no_fabric:
        cfg.sim.use_fabric = False

    for name, deg in zip(JOINT_NAMES, args.joint_angles):
        cfg.scene.robot.init_state.joint_pos[name] = float(np.deg2rad(deg))

    env = gym.make(args.task, cfg=cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    bi = robot.body_names.index(args.body)
    pos = robot.data.body_pos_w[0, bi].cpu().numpy()
    quat = robot.data.body_quat_w[0, bi].cpu().numpy()  # (w, x, y, z)
    print(f"[link-pose] body={args.body} joints_deg={list(args.joint_angles)}")
    print(f"[link-pose] pos={np.round(pos, 6).tolist()} quat_wxyz={np.round(quat, 6).tolist()}")

    # sanity: where did each camera actually end up (world == base frame, robot at origin)?
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    for cam_name in ("front_camera", "side_camera", "wrist_camera"):
        cam = env.unwrapped.scene.sensors.get(cam_name)
        if cam is None:
            continue
        cpos = cam.data.pos_w[0].cpu().numpy()
        cquat = cam.data.quat_w_opengl[0].cpu().numpy()
        print(f"[cam-pose] {cam_name} pos={np.round(cpos, 4).tolist()} quat_wxyz_opengl={np.round(cquat, 4).tolist()}")
        K = cam.data.intrinsic_matrices[0].cpu().numpy()
        h, w = cam.image_shape
        print(f"[cam-intr] {cam_name} {w}x{h} fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
        prim = UsdGeom.Camera(stage.GetPrimAtPath(cam.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")))
        if prim:
            fl = prim.GetFocalLengthAttr().Get()
            ha = prim.GetHorizontalApertureAttr().Get()
            va = prim.GetVerticalApertureAttr().Get()
            print(f"[cam-usd ] {cam_name} focalLength={fl} horizAperture={ha} vertAperture={va}")

    if args.out is not None:
        import os

        import matplotlib
        import torch

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(args.out, exist_ok=True)
        # hold the pose with zero OSC deltas; gripper channel +1 = OPEN (BinaryJointAction:
        # positive/zero = open command, NEGATIVE = close -- verified in the Isaac source
        # and empirically 2026-07-16: holding -1 marches the jaws shut at ~1.3 mm/step)
        zero = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        zero[..., -1] = 1.0
        obs = None
        for _ in range(4):
            obs, *_ = env.step(zero)
        for key, val in obs["policy"].items():
            if not (torch.is_tensor(val) and val.ndim == 4 and val.shape[-1] == 3):
                continue
            frame = val[0].detach().cpu().numpy().astype(np.uint8)
            path = os.path.join(args.out, f"{key}.png")
            plt.imsave(path, frame)
            print(f"[link-pose] saved {path}")

        # jaw-width probe: force fingers to each value (negative = wider than nominal open)
        # and save a wrist frame + blend per value
        if args.finger_pos:
            robot = env.unwrapped.scene["robot"]
            fids, _ = robot.find_joints(["finger_joint", "right_finger_joint"])
            fids_t = torch.tensor(fids, device=env.unwrapped.device)
            # widen the lower limit so negative (wider-than-open) positions are allowed
            limits = torch.tensor([[-0.02, 0.068], [-0.02, 0.068]], device=env.unwrapped.device)
            robot.write_joint_position_limit_to_sim(limits.unsqueeze(0), joint_ids=fids)
            real_dir = "/home/syed/work/repos/diffusion_policy/scripts/sim2real/perception/calibrations"
            real_wrist_path = os.path.join(real_dir, "real_rgb_wrist.png")
            real_wrist = None
            if os.path.exists(real_wrist_path):
                real_wrist = (plt.imread(real_wrist_path)[..., :3] * 255).astype(np.uint8)
            wrist = env.unwrapped.scene.sensors["wrist_camera"]
            for v in args.finger_pos:
                pos_cmd = torch.full((1, 2), float(v), device=env.unwrapped.device)
                vel_cmd = torch.zeros_like(pos_cmd)
                for _ in range(3):
                    # force-write after each step so the binary-action drive cannot pull back
                    robot.write_joint_state_to_sim(pos_cmd, vel_cmd, joint_ids=fids_t)
                    obs, *_ = env.step(zero)
                sim_img = wrist.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)[..., :3]
                tag = f"fingers_{v:+.4f}"
                plt.imsave(os.path.join(args.out, f"{tag}.png"), sim_img)
                if real_wrist is not None and real_wrist.shape == sim_img.shape:
                    blend = ((sim_img.astype(np.float32) + real_wrist.astype(np.float32)) / 2).astype(np.uint8)
                    plt.imsave(os.path.join(args.out, f"{tag}_blend.png"), blend)
                print(f"[fingers] {v:+.4f} m rendered (separation {0.068 - 2 * v:+.4f} m span change vs open: {-2 * v * 1000:+.1f} mm)")

        # raw full-res sensor frames + 50/50 blends against the archived real captures
        real_dir = "/home/syed/work/repos/diffusion_policy/scripts/sim2real/perception/calibrations"
        for cam_name, real_name in (
            ("front_camera", "real_rgb_front.png"),
            ("side_camera", "real_rgb_side.png"),
            ("wrist_camera", "real_rgb_wrist.png"),
        ):
            cam = env.unwrapped.scene.sensors.get(cam_name)
            if cam is None:
                continue
            sim_img = cam.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)[..., :3]
            path = os.path.join(args.out, f"raw_{cam_name}.png")
            plt.imsave(path, sim_img)
            real_path = os.path.join(real_dir, real_name)
            if os.path.exists(real_path):
                real_img = (plt.imread(real_path)[..., :3] * 255).astype(np.uint8)
                if real_img.shape == sim_img.shape:
                    blend = ((sim_img.astype(np.float32) + real_img.astype(np.float32)) / 2).astype(np.uint8)
                    bpath = os.path.join(args.out, f"blend_{cam_name}.png")
                    plt.imsave(bpath, blend)
                    print(f"[blend] saved {bpath}")
                else:
                    print(f"[blend] shape mismatch sim={sim_img.shape} real={real_img.shape} for {cam_name}")

        if args.fabric_test:
            # Diagnose where the wrist camera ACTUALLY is after the tracker's write:
            # fabric view read, USD-composed transform, then orientation-variant renders.
            from pxr import Gf, Usd, UsdGeom

            import omni.usd

            wrist = env.unwrapped.scene.sensors["wrist_camera"]
            p, q = wrist._view.get_world_poses()
            print(f"[diag] view(fabric) pos={np.round(p[0].cpu().numpy(), 4).tolist()} "
                  f"quat={np.round(q[0].cpu().numpy(), 4).tolist()}")
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/gripper/robotiq_base_link/rgb_wrist_camera")
            m = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            print(f"[diag] usd-composed world translation={np.round(list(m.ExtractTranslation()), 4).tolist()}")
            ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
            print(f"[diag] ops={[op.GetOpName() for op in ops]}")
            if ops:
                lm = ops[0].Get()
                print(f"[diag] authored op translation={np.round(list(Gf.Matrix4d(lm).ExtractTranslation()), 4).tolist()}")

            # target world pose (link * offset), OpenGL convention, known-good from calib
            tgt_pos = Gf.Vec3d(0.60867477, -0.00999651, 0.21197911)
            tgt_quat_opengl = Gf.Quatd(0.66088417, Gf.Vec3d(0.28300325, 0.27983088, 0.63626721))
            flip_x = Gf.Quatd(0.0, Gf.Vec3d(1.0, 0.0, 0.0))  # 180 deg about X: opengl<->ros-ish flip
            variants = {
                "asis": tgt_quat_opengl,
                "flipx": tgt_quat_opengl * flip_x,
            }
            for name, quat in variants.items():
                mtx = Gf.Matrix4d(1.0).SetTransform(Gf.Rotation(quat), tgt_pos)
                ops[0].Set(mtx)
                obs, *_ = env.step(zero)
                obs, *_ = env.step(zero)
                frame = obs["policy"]["wrist_rgb"][0].detach().cpu().numpy().astype(np.uint8)
                path = os.path.join(args.out, f"wrist_variant_{name}.png")
                plt.imsave(path, frame)
                pv, qv = wrist._view.get_world_poses()
                print(f"[diag] variant={name} view pos={np.round(pv[0].cpu().numpy(), 4).tolist()} "
                      f"quat={np.round(qv[0].cpu().numpy(), 4).tolist()} saved={path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
