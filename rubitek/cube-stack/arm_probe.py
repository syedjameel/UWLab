"""How fast can this arm actually move, and what is holding it back?

WHY
---
Driving the hand at a full-scale Cartesian command -- 20 mm per control step -- the end effector
realises about 0.35 mm/step at the training OSC gains and 1.4 mm/step at the package's eval gains.
That is 2% and 7% of what the action space advertises, and it caps every family whose cube has to
travel: a 209 mm carry needs more displacement than a 160-step episode can deliver.

Three candidate causes, and they are distinguishable by direct measurement rather than argument:
joint friction (the sysid metadata carries static friction of 21-29 N*m on the first three joints,
which would dwarf the torque a soft OSC produces), the actuator's own effort limit, and the OSC
gains themselves. This reads all three off the CONSTRUCTED scene and then measures the achieved
speed under a constant full-scale command, with no object, no reward and no policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str,
                    default="OmniReset-UR10eDelto-CubeStackGrip-RelCartesianOSC-State-Play-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--axis", type=int, default=0, help="0=x, 1=y, 2=z")
parser.add_argument("--motion_stiffness", type=float, default=None)
parser.add_argument("--motion_damping", type=float, default=None)
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
v = cfg.variants
cfg.scene.insertive_object = v["scene.insertive_object"]["cube34"]
cfg.scene.receptive_object = v["scene.receptive_object"]["cube34"]
for _, term in vars(cfg.events).items():
    p = getattr(term, "params", None)
    if isinstance(p, dict) and "dataset_dir" in p:
        p["dataset_dir"] = "./Datasets_ur10e_delto/OmniReset"
if args.motion_stiffness is not None or args.motion_damping is not None:
    a = cfg.actions.arm
    k, d = list(a.motion_stiffness), list(a.motion_damping_ratio)
    if args.motion_stiffness is not None:
        k[:3] = [args.motion_stiffness] * 3
    if args.motion_damping is not None:
        d[:3] = [args.motion_damping] * 3
    cfg.actions.arm = a.replace(motion_stiffness=tuple(k), motion_damping_ratio=tuple(d))

print("=== configured arm action ===")
a = cfg.actions.arm
for f in ("scale_xyz_axisangle", "motion_stiffness", "motion_damping_ratio", "torque_limit"):
    print(f"  {f}: {getattr(a, f, None)}")
print("=== event terms that touch the arm's dynamics ===")
for name, term in vars(cfg.events).items():
    if "sysid" in name or "osc" in name:
        print(f"  {name}: {getattr(term, 'func', None)}")
if not any("sysid" in n or "osc" in n for n in vars(cfg.events)):
    print("  (none -- no sysid friction or gain randomization is configured on this task)")

env = gym.make(args.task, cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
arm_ids, arm_names = robot.find_joints(
    ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], preserve_order=True)

env.reset()
print("\n=== arm joint dynamics as constructed ===")
print(f"  joints: {arm_names}")
for label, buf in (("friction", getattr(robot.data, "joint_friction_coeff", None)),
                   ("armature", getattr(robot.data, "joint_armature", None)),
                   ("vel limit", getattr(robot.data, "joint_vel_limits", None)),
                   ("effort limit", getattr(robot.data, "joint_effort_limits", None))):
    if buf is None:
        print(f"  {label:12s}: (not exposed)")
        continue
    print(f"  {label:12s}: {[round(float(x), 4) for x in buf[0, arm_ids]]}")

palm = robot.find_bodies("rl_dg_mount")[0][0]
n_act = u.action_manager.total_action_dim
n_arm = u.action_manager.get_term("arm").action_dim
act = torch.zeros(args.num_envs, n_act, device=u.device)
act[:, args.axis] = 1.0  # a constant, full-scale command on one axis

scale = getattr(u.action_manager.get_term("arm").cfg, "scale_xyz_axisangle", (0.02,) * 6)
print(f"\n=== constant full-scale command on axis {args.axis} "
      f"({scale[args.axis] * 1000:.1f} mm/step commanded) ===")
p0 = robot.data.body_pos_w[:, palm].clone()
trace = []
with torch.inference_mode():
    for step in range(1, args.steps + 1):
        # Step the GYM env, not the unwrapped one. The scripted-carry oracle moves the arm through
        # `env.step`, and this probe reading exactly 0.0 mm of travel at two different gain settings
        # is a probe bug before it is a physics result -- a genuinely stalled arm still sags under
        # gravity.
        env.step(act)
        d = (robot.data.body_pos_w[:, palm] - p0)[:, args.axis]
        if step == 1:
            jp = robot.data.joint_pos[0, arm_ids]
            print(f"  [sanity] after one step: arm joints {[round(float(x), 4) for x in jp]}")
            print(f"  [sanity] palm now {[round(float(x), 4) for x in robot.data.body_pos_w[0, palm]]}"
                  f" (was {[round(float(x), 4) for x in p0[0]]})")
        if step % 10 == 0:
            per_step = float(d.mean()) * 1000.0 / step
            trace.append({"step": step, "travel_mm": float(d.mean()) * 1000.0,
                          "mm_per_step": per_step,
                          "tracking_pct": 100.0 * per_step / (scale[args.axis] * 1000.0)})
            print(f"  step {step:3d}  travelled {trace[-1]['travel_mm']:7.1f} mm  "
                  f"({per_step:5.2f} mm/step = {trace[-1]['tracking_pct']:4.1f}% of commanded)")

res = {"task": args.task, "axis": args.axis, "scale": list(scale),
       "motion_stiffness": args.motion_stiffness, "motion_damping": args.motion_damping,
       "joint_friction": [float(x) for x in robot.data.joint_friction_coeff[0, arm_ids]]
       if getattr(robot.data, "joint_friction_coeff", None) is not None else None,
       "trace": trace}
if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")
env.close()
simulation_app.close()
