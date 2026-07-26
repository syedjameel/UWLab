# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the RECORDED linear-gripper grasps (adopted syedjameel model) — genuine clamp montage.

Replays each recorded grasp: place the standalone linear gripper at the grasp pose, set the recorded
jaw angles, hold closed under gravity, render a montage (placed + settled). Each grasp already passed
check_grasp_success, so the object should stay held off the table.

Body names auto-detected (syedjameel: robotiq_base_link + left/right_inner_finger; falls back by
substring) so it survives USD-import naming. Task = OmniReset-LinearGripper-GraspSampling-v0.

    LOCAL (cameras): ./uwlab.sh -p scripts_v2/tools/diag_render_recorded_grasps_linear.py \
        --object mid --dataset_dir <DS> --headless
"""

from __future__ import annotations

import argparse
from typing import cast

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--object", default="mid")
parser.add_argument("--dataset_dir", default="source/uwlab_assets/data/Datasets/OmniReset")
parser.add_argument("--num_envs", type=int, default=9)
parser.add_argument("--hold_steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------------------------------------------------------------------------------
import inspect
import os

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import ManagerTermBase  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402

import uwlab_tasks  # noqa: F401
# object scene variants are shared (the linear grasp cfg subclasses the Robotiq one).
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.grasp_sampling_cfg import variants as GV  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402

TASK = "OmniReset-LinearGripper-GraspSampling-v0"


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    fwd = target - eye; fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    z = -fwd; x = np.cross(up, z); x = x / (np.linalg.norm(x) + 1e-9); y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    q = math_utils.quat_from_matrix(torch.tensor(R, dtype=torch.float32).unsqueeze(0))[0]
    return tuple(q.tolist())


def load_grasps(dataset_dir, obj):
    gdir = os.path.join(dataset_dir, "Grasps")
    objn = obj.capitalize()
    if not os.path.isdir(os.path.join(gdir, objn)) and os.path.isdir(gdir):
        for d in os.listdir(gdir):  # case-insensitive (caprim -> CapRim from object_name_from_usd)
            if d.lower() == obj.lower():
                objn = d
                break
    path = os.path.join(gdir, objn, "grasps.pt")
    d = torch.load(path, map_location="cpu", weights_only=False)["grasp_relative_pose"]
    rp = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in d["relative_position"]])
    rq = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in d["relative_orientation"]])
    gjp = {k: torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in vals])
           for k, vals in d["gripper_joint_positions"].items()}
    return rp, rq, gjp, path


def _find_base(bn):
    for c in ("robotiq_base_link", "gripper_base_link", "base_link", "base"):
        if c in bn:
            return bn.index(c)
    for i, b in enumerate(bn):
        if "base" in b.lower():
            return i
    return 0


def _find_jaws(bn, base_idx):
    jaws = [i for i, b in enumerate(bn) if i != base_idx and ("finger" in b.lower() or "jaw" in b.lower())]
    if len(jaws) >= 2:
        return jaws[0], jaws[1]
    if len(jaws) == 1:
        return jaws[0], jaws[0]
    others = [i for i in range(len(bn)) if i != base_idx]
    return (others[0], others[-1]) if others else (base_idx, base_idx)


@hydra_task_compose(TASK, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg) -> None:
    rp, rq, gjp, gpath = load_grasps(args_cli.dataset_dir, args_cli.object)
    ng = len(rp)
    n = min(args_cli.num_envs, ng)
    env_cfg.scene.num_envs = n
    env_cfg.seed = None
    env_cfg.scene.object = GV["scene.object"][args_cli.object]
    try:
        env_cfg.scene.sky_light.spawn.intensity = 4000.0
    except Exception:  # noqa: BLE001
        pass
    EYE = (0.42, 0.32, 1.46); TGT = (0.0, 0.0, 1.27)
    env_cfg.scene.cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/cam", height=680, width=860,
        offset=TiledCameraCfg.OffsetCfg(pos=EYE, rot=look_at_quat(EYE, TGT), convention="opengl"),
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=26.0),
    )

    env = cast(ManagerBasedRLEnv, gym.make(TASK, cfg=env_cfg)).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)

    robot = env.scene["robot"]
    obj = env.scene["object"]
    bn = robot.data.body_names
    print(f"[replay] body_names={bn}")
    base_idx = _find_base(bn)
    li, ri = _find_jaws(bn, base_idx)
    print(f"[replay] base='{bn[base_idx]}' jaws=('{bn[li]}','{bn[ri]}')")
    jnames = list(robot.data.joint_names)
    print(f"[replay] object={args_cli.object} grasps={ng} rendering n={n}  src={gpath}")

    env.reset()
    eids = torch.arange(n, device=env.device)

    # recorded (closed) jaw angles first
    jpos = robot.data.joint_pos[:n].clone()
    for jn, vals in gjp.items():
        if jn in jnames:
            jpos[:, jnames.index(jn)] = vals[:n].to(env.device)
    robot.write_joint_state_to_sim(jpos, torch.zeros_like(jpos), env_ids=eids)
    env.sim.forward()

    op = obj.data.root_pos_w[:n].clone()
    oq = obj.data.root_quat_w[:n].clone()
    tgt_pos, tgt_quat = math_utils.combine_frame_transforms(op, oq, rp[:n].to(env.device), rq[:n].to(env.device))
    root_w = robot.data.root_state_w[:n]
    base_p = robot.data.body_link_pos_w[:n, base_idx]
    base_q = robot.data.body_link_quat_w[:n, base_idx]
    rib_p, rib_q = math_utils.subtract_frame_transforms(base_p, base_q, root_w[:, 0:3], root_w[:, 3:7])
    nroot_p, nroot_q = math_utils.combine_frame_transforms(tgt_pos, tgt_quat, rib_p, rib_q)
    robot.write_root_pose_to_sim(torch.cat([nroot_p, nroot_q], dim=-1), env_ids=eids)
    robot.write_root_velocity_to_sim(torch.zeros((n, 6), device=env.device), env_ids=eids)
    env.sim.forward()
    nb_p = robot.data.body_link_pos_w[:n, base_idx]
    base_err = (nb_p - tgt_pos).norm(dim=1).mean().item() * 1000
    print(f"[replay] base placement err={base_err:.0f}mm")

    def probe():
        op = obj.data.root_pos_w[:n]
        fb = robot.data.body_link_pos_w[:n][:, [li, ri]]
        d = (fb - op.unsqueeze(1)).norm(dim=2).min(dim=1).values
        return d, op[:, 2]

    rows = int(np.ceil(np.sqrt(n))); cols = int(np.ceil(n / rows))

    def render_montage(tag, subtitle):
        cam = env.scene["cam"]
        for _ in range(12):
            env.sim.render()
        rgb = cam.data.output["rgb"].detach().cpu().numpy()[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * (255.0 if rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.6), squeeze=False)
        for i, ax in enumerate(axes.reshape(-1)):
            ax.axis("off")
            if i < n:
                ax.imshow(rgb[i]); ax.set_title(f"grasp {i+1}", color="#9aa4b2", fontsize=9)
        fig.patch.set_facecolor("#0d1117")
        fig.suptitle(f"linear-gripper grasps — {args_cli.object}  ({subtitle})", color="#e6edf3")
        os.makedirs("reports/omnireset/figures/diag", exist_ok=True)
        out = f"reports/omnireset/figures/diag/linear_grasps_{args_cli.object}_{tag}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out, dpi=120, facecolor="#0d1117"); plt.close(fig)
        print(f"[replay] saved {out}")

    act = -torch.ones(env.action_space.shape, device=env.device, dtype=torch.float32)  # hold closed
    d0, z0 = probe()
    print(f"[replay] placed: finger->obj mean={d0.mean()*1000:.0f}mm obj_z mean={z0.mean():.3f}")
    render_montage("placed", f"{n} recorded grasps at their validated config")
    for s in range(args_cli.hold_steps):
        env.step(act)
    d, z = probe()
    held = (z > 0.2)
    print(f"[replay] after {args_cli.hold_steps} steps gravity: finger->obj={d.mean()*1000:.0f}mm "
          f"obj_z={z.mean():.3f}  HELD {held.float().mean()*100:.0f}% ({int(held.sum())}/{n})")
    render_montage("settled", f"after gravity — {int(held.sum())}/{n} still held")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
