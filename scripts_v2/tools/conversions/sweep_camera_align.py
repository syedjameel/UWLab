# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Automated sim2real camera-alignment sweep (front/side cameras).

Holds the CameraAlign env at the real capture joints, then coordinate-descends over the
camera parameters (focal, pitch, yaw, roll, x, y, z): for each candidate value it rewrites
the camera prim's USD attributes, re-renders one frame, scores edge-map alignment against
the real reference image, and saves the frame + a 50/50 blend. The best value per stage is
locked in before the next stage sweeps. All renders happen in ONE Isaac session.

    conda activate leisaac && ./uwlab.sh -p scripts_v2/tools/conversions/sweep_camera_align.py \
        --camera front_camera \
        --real_image <...>/calibrations/real_rgb_front.png \
        --joint_angles -20.42 -98.08 138.53 -130.43 -89.95 -20.42 \
        --out table_swap_snaps/sweep_front

Score: Pearson correlation of Gaussian-blurred Sobel edge maps (robust to the sim/real
appearance gap; the real-only objects add a constant bias that does not affect ranking).
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="OmniReset-UR10eLinearGripper-CameraAlign-v0")
parser.add_argument("--camera", type=str, default="front_camera",
                    choices=["front_camera", "side_camera", "wrist_camera"])
parser.add_argument("--real_image", type=str, required=True)
parser.add_argument("--joint_angles", type=float, nargs=6, required=True, help="SIM-convention degrees.")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--stages", type=str, default="focal,pitch,yaw,roll,x,y,z")
parser.add_argument("--mask", type=int, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"),
                    help="Pixel box zeroed in BOTH edge maps before scoring -- use to exclude "
                    "real-only content (e.g. the ArUco marker in the wrist view).")
parser.add_argument("--reset_each", action="store_true",
                    help="env.reset() before every render so each candidate sees the identical "
                    "arm state. REQUIRED for the wrist camera: the OSC hold drifts slightly over "
                    "accumulated steps, which is invisible far-field but dominates at close range.")
parser.add_argument("--no_trims", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os

import matplotlib

matplotlib.use("Agg")
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial.transform import Rotation as R

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import uwlab_tasks  # noqa: F401

import omni.usd
from pxr import Gf, UsdGeom

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# candidate DELTAS per stage (applied around the current best)
STAGE_DELTAS = {
    "focal": [-2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0],       # mm
    "pitch": [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0],                    # deg, camera x
    "yaw":   [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0],                    # deg, camera y
    "roll":  [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],                               # deg, camera z
    "x": [-0.03, -0.015, 0.0, 0.015, 0.03],                                        # m, world
    "y": [-0.03, -0.015, 0.0, 0.015, 0.03],
    "z": [-0.03, -0.015, 0.0, 0.015, 0.03],
}


def edge_map(img_gray):
    gx = ndimage.sobel(img_gray, axis=1)
    gy = ndimage.sobel(img_gray, axis=0)
    mag = np.hypot(gx, gy)
    m = mag.max()
    if m > 0:
        mag /= m
    return ndimage.gaussian_filter(mag, sigma=3.0)


def main():
    os.makedirs(args.out, exist_ok=True)
    real = plt.imread(args.real_image)[..., :3]
    if real.dtype != np.uint8:
        real = (real * 255).astype(np.uint8)

    def masked(edges):
        if args.mask is not None:
            x0, y0, x1, y1 = args.mask
            edges[y0:y1, x0:x1] = 0.0
        return edges

    real_edges = masked(edge_map(real.mean(axis=2)))

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    if not args.no_trims:
        cfg.sim.physx.gpu_collision_stack_size = 67108864
        cfg.sim.physx.gpu_max_rigid_contact_count = 2097152
        cfg.sim.physx.gpu_max_rigid_patch_count = 2097152
        cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2097152
        cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2097152
    for name, deg in zip(JOINT_NAMES, args.joint_angles):
        cfg.scene.robot.init_state.joint_pos[name] = float(np.deg2rad(deg))

    env = gym.make(args.task, cfg=cfg)
    env.reset()
    # hold: zero OSC deltas, gripper channel +1 = OPEN (BinaryJointAction: positive/zero =
    # open, NEGATIVE = close -- a held -1 marches the jaws shut at ~1.3 mm/step)
    hold = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    hold[..., -1] = 1.0
    for _ in range(6):
        env.step(hold)

    cam = env.unwrapped.scene.sensors[args.camera]
    stage = omni.usd.get_context().get_stage()
    prim_path = cam.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")
    prim = stage.GetPrimAtPath(prim_path)
    assert prim.IsValid(), f"camera prim not found at {prim_path}"

    # Take over the prim transform with a single matrix op.
    # front/side: static parent -> the op IS the world pose; seed from the composed pose,
    #   translation deltas in WORLD axes.
    # wrist: moving link parent -> the op is the LINK->CAMERA offset (authored by the
    #   track_link_mounted_camera reset event; Fabric composes it with the live link);
    #   seed from that op, translation deltas in CAMERA-LOCAL axes (z = optical axis).
    is_wrist = args.camera == "wrist_camera"
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    if is_wrist:
        assert ops and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform, \
            "wrist camera op not authored -- did the tracking event run?"
        op = ops[0]
        base_mtx = Gf.Matrix4d(op.Get())
    else:
        base_mtx = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        xform.ClearXformOpOrder()
        op = xform.AddTransformOp()
        op.Set(base_mtx)
    focal_attr = prim.GetAttribute("focalLength")

    # state: base pose (pos + rotation matrix, row-convention Gf) and focal
    base_pos = np.array(base_mtx.ExtractTranslation())
    q0 = base_mtx.ExtractRotationQuat()
    base_R = R.from_quat([*q0.GetImaginary(), q0.GetReal()])  # camera->parent, opengl frame
    base_focal = float(focal_attr.Get())
    state = {"focal": base_focal, "pitch": 0.0, "yaw": 0.0, "roll": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}

    def apply_state(s):
        rot = base_R * R.from_euler("XYZ", [s["pitch"], s["yaw"], s["roll"]], degrees=True)
        delta = np.array([s["x"], s["y"], s["z"]])
        pos = base_pos + (base_R.apply(delta) if is_wrist else delta)
        q = rot.as_quat()  # xyzw
        gf_rot = Gf.Rotation(Gf.Quatd(float(q[3]), Gf.Vec3d(float(q[0]), float(q[1]), float(q[2]))))
        op.Set(Gf.Matrix4d(1.0).SetTransform(gf_rot, Gf.Vec3d(*map(float, pos))))
        focal_attr.Set(float(s["focal"]))

    def evaluate(trial, tag):
        if args.reset_each:
            env.reset()  # re-fires the wrist tracker op write; apply_state overrides it next
            apply_state(trial)
            env.step(hold)
            env.step(hold)
        else:
            apply_state(trial)
            env.step(hold)
        sim_img = cam.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)[..., :3]
        score = float(np.corrcoef(masked(edge_map(sim_img.mean(axis=2))).ravel(), real_edges.ravel())[0, 1])
        plt.imsave(os.path.join(args.out, f"{tag}.png"), sim_img)
        blend = ((sim_img.astype(np.float32) + real.astype(np.float32)) / 2).astype(np.uint8)
        plt.imsave(os.path.join(args.out, f"{tag}_blend.png"), blend)
        return score

    base_score = evaluate(state, "baseline")
    print(f"[sweep] baseline score={base_score:.4f}  focal={base_focal:.2f}")

    for stage_name in args.stages.split(","):
        deltas = STAGE_DELTAS[stage_name]
        center = state[stage_name]
        results = []
        for d in deltas:
            trial = dict(state)
            trial[stage_name] = center + d
            tag = f"{stage_name}_{center + d:+.3f}"
            sc = evaluate(trial, tag)
            results.append((sc, center + d))
            print(f"[sweep] {stage_name}={center + d:+.3f} score={sc:.4f}")
        best_score, best_val = max(results)
        state[stage_name] = best_val
        print(f"[sweep] >>> stage {stage_name}: best {best_val:+.3f} (score {best_score:.4f})")

    final_score = evaluate(state, "final")
    print(f"[sweep] FINAL state={ {k: round(v, 4) for k, v in state.items()} } score={final_score:.4f}")
    # absolute pose for pasting into _UR10E_CAMERA_POSES
    rot = base_R * R.from_euler("XYZ", [state["pitch"], state["yaw"], state["roll"]], degrees=True)
    q = rot.as_quat()
    pos = base_pos + np.array([state["x"], state["y"], state["z"]])
    print(f"[sweep] pos=({pos[0]:.7f}, {pos[1]:.7f}, {pos[2]:.7f})")
    print(f"[sweep] rot=(w,x,y,z)=({q[3]:.7f}, {q[0]:.7f}, {q[1]:.7f}, {q[2]:.7f})")
    print(f"[sweep] focal={state['focal']:.2f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
