# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Film a few grasp-sampling episodes before sampling at scale.

Runs the grasp-sampling task exactly as ``record_grasps.py`` does (same env, same scripted close
every step) on a handful of environments, with the viewport aimed at environment 0's object, and
writes an mp4 with the per-step state burned into each frame: episode, step, gravity phase, object
height and displacement, and -- for the DELTO pinch -- both pad distances and the jaw aperture.

Usage::

    UWLAB_DELTO_PINCH=1 ./run6.sh scripts_v2/tools/film_grasp_sampling.py \\
        --task OmniReset-Delto-GraspSampling-v0 --object cube34 --episodes 3 --out out/grasp_film.mp4
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", required=True)
parser.add_argument("--object", default="cube34", help="scene.object variant")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--out", required=True)
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--max-steps", type=int, default=0, help="stop after this many control steps (0 = run episodes)")
parser.add_argument("--hold-object-steps", type=int, default=0,
                    help="pin the object at its reset pose for the first N control steps of each episode, "
                         "so the grip can build force before gravity acts on the object")
parser.add_argument("--robot-friction", type=float, default=None,
                    help="override the reset env's robot material randomisation to this static=dynamic friction")
parser.add_argument("--object-friction", type=float, default=None,
                    help="override the reset env's insertive-object material randomisation likewise")
parser.add_argument("--world-camera", action="store_true",
                    help="--eye/--lookat are WORLD coordinates (the arm scenes), not env-0-relative")
parser.add_argument("--eye", type=float, nargs=3, default=(0.38, -0.38, 0.62), help="camera, env-0 frame")
parser.add_argument("--lookat", type=float, nargs=3, default=(0.0, 0.0, 0.48), help="target, env-0 frame")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import uwlab_is6_bridge  # noqa: E402

uwlab_is6_bridge.apply()

import gymnasium as gym  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import uwlab_tasks  # noqa: F401,E402
import uwlab_tasks.manager_based.manipulation.omnireset.mdp as task_mdp  # noqa: E402
import uwlab_tasks.manager_based.manipulation.omnireset.mdp.scripted_gripper as scripted_gripper  # noqa: E402


def t(x):
    return getattr(x, "torch", x)


def burn(img: np.ndarray, lines: list[str]) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    h = 15 * len(lines) + 8
    d.rectangle([0, 0, im.width, h], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        d.text((6, 4 + 15 * i), line, fill=(255, 235, 90))
    return np.asarray(im)


cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
# Grasp-sampling tasks pick the object by variant; the cube-stack reset tasks pin their pair in the
# config and have no scene.object variant, so the override is skipped there.
if "scene.object" in getattr(cfg, "variants", {}) and hasattr(cfg.scene, "object"):
    cfg.scene.object = cfg.variants["scene.object"][args.object]
cfg.sim.render_interval = 1
for _term_name, _mu in (("reset_robot_material", args.robot_friction), ("insertive_object_material", args.object_friction)):
    _term = getattr(cfg.events, _term_name, None)
    if _mu is not None and _term is not None:
        _term.params["static_friction_range"] = (_mu, _mu)
        _term.params["dynamic_friction_range"] = (_mu, _mu)
        print(f"[film] {_term_name}: friction forced to {_mu}", flush=True)
cfg.viewer.origin_type = "world" if args.world_camera else "env"
cfg.viewer.env_index = 0
cfg.viewer.asset_name = None
cfg.viewer.eye = tuple(args.eye)
cfg.viewer.lookat = tuple(args.lookat)
cfg.viewer.resolution = (960, 720)

env = gym.make(args.task, cfg=cfg, render_mode="rgb_array").unwrapped
env.reset()
robot = env.scene["robot"]
obj = env.scene["object"] if "object" in env.scene.keys() else env.scene["insertive_object"]
names = list(robot.body_names)
pads = [n for n in ("rl_dg_1_tip", "rl_dg_2_tip") if n in names]
jaws = [n for n in ("left_inner_finger", "right_inner_finger") if n in names]
pair = pads or jaws
gravity_on_step = None
term = getattr(env.event_manager, "get_term_cfg", None)
try:
    g_int = env.event_manager.get_term_cfg("global_physics_control_event").params["gravity_on_interval"]
    gravity_on_step = int(g_int[0] / env.step_dt)
except Exception:  # noqa: BLE001 -- the overlay just omits the phase
    pass

# SERVO WIRING CHECK. The filmed joint log shows several pinch joints commanded AWAY from their
# closed posture every step (a period-2 limit cycle), which a sign or ordering mismatch between the
# posture table, the measured joint positions and the action term would produce. Print the three
# orderings side by side and the per-joint command against the direction it should have.
for _name in env.action_manager.active_terms:
    _term = env.action_manager.get_term(_name)
    _names = list(getattr(_term, "_joint_names", []))
    if not _names or not _names[0].startswith("rj_dg_"):
        continue
    _ids = list(_term._joint_ids) if not isinstance(_term._joint_ids, slice) else list(range(robot.num_joints))
    _by_ids = [robot.joint_names[i] for i in _ids]
    print(f"[film] term '{_name}': {len(_names)} names; names==joint_names[ids]: {_names == _by_ids}", flush=True)
    if _names != _by_ids:
        print(f"[film]   term names   {_names}", flush=True)
        print(f"[film]   names by ids {_by_ids}", flush=True)
    _scale = getattr(_term, "_scale", None)
    print(f"[film]   scale: {_scale if not hasattr(_scale, 'shape') else t(_scale)[0].tolist()}", flush=True)
    _cmd = task_mdp.scripted_gripper_actions(env, close=True)[0]
    _post = scripted_gripper._PER_JOINT_PINCH_POSTURES if getattr(env.cfg, "scripted_gripper_pinch_posture", False) \
        else scripted_gripper._PER_JOINT_POSTURES
    _q = t(robot.data.joint_pos)[0]
    for _k, _n in enumerate(_names):
        if _n.startswith(("rj_dg_1_", "rj_dg_2_")):
            _need = _post[_n][1] - float(_q[robot.joint_names.index(_n)])
            print(f"[film]   {_n}: q {float(_q[robot.joint_names.index(_n)]):+.3f} closed {_post[_n][1]:+.3f} "
                  f"needs {'+' if _need > 0 else '-'}  command {float(_cmd[_k]):+.2f}", flush=True)

frames, episode, successes, total_steps = [], 0, 0, 0
spawn = t(obj.data.root_pos_w)[0].clone()
while episode < args.episodes and not (args.max_steps and total_steps >= args.max_steps):
    total_steps += 1
    actions = task_mdp.scripted_gripper_actions(env, close=True)
    if args.hold_object_steps and int(env.episode_length_buf[0]) == 0:
        _held_pose = torch.cat([t(obj.data.root_pos_w), t(obj.data.root_quat_w)], dim=-1).clone()
    _, _, terminated, truncated, _ = env.step(actions)
    step = int(env.episode_length_buf[0])
    if args.hold_object_steps and 0 < step <= args.hold_object_steps:
        obj.write_root_pose_to_sim(_held_pose)
        obj.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))
    if step <= 1:
        spawn = t(obj.data.root_pos_w)[0].clone()
    op = t(obj.data.root_pos_w)[0]
    # FIRST-STEPS TRACE: does the HAND move, or is the object pushed? Mount displacement from the
    # step-1 position, object height and speed, and the largest hand-joint change since step 1.
    _mount = "rl_dg_mount" if "rl_dg_mount" in names else None
    if _mount is not None and step <= 6:
        _mp = t(robot.data.body_pos_w)[0, names.index(_mount)]
        _jq = t(robot.data.joint_pos)[0]
        if step <= 1:
            _mount0, _jq0 = _mp.clone(), _jq.clone()
        _hand = [i for i, n in enumerate(robot.joint_names) if n.startswith("rj_dg_")]
        print(f"[film] trace step {step}: mount moved {float((_mp - _mount0).norm()) * 1000:.1f} mm, "
              f"object z {float(op[2]):.3f} m v {float(t(obj.data.root_lin_vel_w)[0].norm()):.3f} m/s, "
              f"hand joint max change {float((_jq[_hand] - _jq0[_hand]).abs().max()):.3f} rad", flush=True)
    lines = [
        f"{args.task}  object={args.object}  episode {episode + 1}/{args.episodes}  step {step}",
        f"gravity {'ON' if gravity_on_step is not None and step > gravity_on_step else 'off'}   "
        f"object z {float(op[2]):.3f} m   moved {float((op - spawn).norm()) * 1000:.0f} mm",
    ]
    if len(pair) == 2:
        bp = t(robot.data.body_pos_w)[0]
        p0, p1 = bp[names.index(pair[0])], bp[names.index(pair[1])]
        lines.append(
            f"{pair[0]} {float((p0 - op).norm()) * 1000:.0f} mm, {pair[1]} {float((p1 - op).norm()) * 1000:.0f} mm "
            f"from object   aperture {float((p0 - p1).norm()) * 1000:.0f} mm"
        )
        # WHERE THE OBJECT IS IN THE HAND FRAME, against where the pads actually are. The sampler
        # places the object at a configured grasp centre (mount frame); if the live pad midpoint is
        # somewhere else, the pads close on an edge and push the object away.
        mount = "rl_dg_mount" if "rl_dg_mount" in names else ("robotiq_base_link" if "robotiq_base_link" in names else None)
        if mount is not None:
            mi = names.index(mount)
            mp, mq = t(robot.data.body_pos_w)[0:1, mi], t(robot.data.body_quat_w)[0:1, mi]
            obj_m = math_utils.quat_apply_inverse(mq, (op.unsqueeze(0) - mp))[0] * 1000
            mid_m = math_utils.quat_apply_inverse(mq, ((p0 + p1) / 2).unsqueeze(0) - mp)[0] * 1000
            fmt = lambda v: "(" + ", ".join(f"{float(x):+.0f}" for x in v) + ")"  # noqa: E731
            lines.append(f"in {mount} frame, mm: object {fmt(obj_m)}   pad midpoint {fmt(mid_m)}   "
                         f"object - midpoint {fmt(obj_m - mid_m)}")
            if step <= 1:
                print(f"[film] step {step} {mount}-frame mm: object {fmt(obj_m)} pad midpoint {fmt(mid_m)} "
                      f"diff {fmt(obj_m - mid_m)} aperture {float((p0 - p1).norm()) * 1000:.1f}", flush=True)
        # WHY THE APERTURE MOVES: per-step joint state for the two pinch fingers. The PD target is
        # what the actuator drives toward; the scripted action is what this step asked for.
        if step <= 30:
            jn = list(robot.joint_names)
            q_all, tgt_all = t(robot.data.joint_pos)[0], t(robot.data.joint_pos_target)[0]
            ids = [i for i, n in enumerate(jn) if n.startswith(("rj_dg_1_", "rj_dg_2_"))]
            if ids:
                act = actions[0]
                gaps = ", ".join(f"{jn[i][6:]} q{float(q_all[i]):+.2f} tgt{float(tgt_all[i]):+.2f}" for i in ids)
                print(f"[film] step {step:2d} aperture {float((p0 - p1).norm()) * 1000:5.1f} mm  "
                      f"action |max| {float(act.abs().max()):.2f}  {gaps}", flush=True)
    frames.append(burn(np.asarray(env.render())[:, :, :3], lines))
    if bool(terminated[0] | truncated[0]):
        done_success = bool(terminated[0])
        successes += int(done_success)
        print(f"[film] episode {episode + 1} ended at step {step}: {'terminated' if done_success else 'truncated'}", flush=True)
        episode += 1

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
iio.imwrite(args.out, frames, fps=args.fps)
print(f"[film] WROTE {args.out} ({len(frames)} frames)", flush=True)
simulation_app.close()
