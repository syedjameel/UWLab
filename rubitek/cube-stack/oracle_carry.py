"""Can ANY controller stack the cube from the Stable-Grasp reset? A scripted one, to find out.

WHY THIS EXISTS
---------------
Stable-Grasp (ObjectAnywhereEEGrasped) has scored exactly 0.0000 on every run, and there are now two
competing explanations: the task is hard to LEARN, or the task is not solvable as configured. Those
call for opposite responses -- more GPU-hours, or another defect hunt -- and no amount of training
distinguishes them, because both look like a flat zero.

A scripted controller settles it without a single training step. The cube starts held (with the grip
bias) a mean 209 mm from the goal; the action space is relative Cartesian OSC on the end effector, so
if the cube is genuinely in the hand, driving the end effector along the object-to-goal error carries
the cube along with it. That is a proportional controller in three lines, and it needs no policy, no
checkpoint and no reward.

If it stacks: the environment is sound, the success predicate is reachable from this family, and
what is missing is learning signal or time. If it does not: the geometry, the grip or the predicate
is wrong, and no amount of training was ever going to fix it.

WHAT THE TARGET IS
------------------
`ProgressContext` compares the insertive object's BOTTOM face to the receptive object's TOP face:
the metadata gives the insertive an assembled offset of [0, 0, -0.017] and the receptive
[0, 0, +0.017] on a 34 mm cube. So the seated object centre is the receptive centre plus 34 mm of z,
and the position error the predicate thresholds at 5 mm is the distance to exactly that point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str,
                    default="OmniReset-UR10eDelto-CubeStackGrip-RelCartesianOSC-State-Play-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--reset_type", type=str, default="ObjectAnywhereEEGrasped")
parser.add_argument("--insertive", type=str, default="cube34")
parser.add_argument("--receptive", type=str, default="cube34")
parser.add_argument("--dataset_dir", type=str, default="./Datasets_ur10e_delto/OmniReset")
parser.add_argument("--collision_stack_mb", type=int, default=1536)
parser.add_argument("--gain", type=float, default=1.0,
                    help="Proportional gain on the normalised position error. 1.0 means 'command "
                         "the full remaining error', which the action clip then bounds to one "
                         "scale_xyz of travel per step.")
parser.add_argument("--hand_action", type=float, default=1.0,
                    help="Constant command on every hand dimension. 1.0 is full closure, which the "
                         "null-action probe shows holds the cube at 0.48-2.56 N of fingertip force.")
parser.add_argument("--motion_stiffness", type=float, default=None,
                    help="Override the OSC's TRANSLATIONAL stiffness (rotational is left alone).\n\n"
                         "The train gains are 200 N/m at a damping ratio of 3.0 -- three times "
                         "critically damped on a UR10e carrying a 1.77 kg hand. Measured, that "
                         "realises about 1 mm of end-effector travel per step against a 19.5 mm "
                         "command, so a 209 mm carry cannot finish inside a 160-step episode. The "
                         "eval gains are 1000 N/m at ratio 1.0; this makes the pair sweepable "
                         "without committing either.")
parser.add_argument("--motion_damping", type=float, default=None,
                    help="Override the OSC's TRANSLATIONAL damping ratio. See --motion_stiffness.")
parser.add_argument("--hand_effort_scale", type=float, default=None,
                    help="Multiply every hand joint's effort_limit_sim.\n\n"
                         "The authored caps are ~2.2 N at the fingertip, which is ample to HOLD a "
                         "0.29 N cube and is measured to do so. Whether they are ample to hold it "
                         "through a carry is a different question -- the cube stops following the "
                         "end effector partway through every long run -- and this is the cheapest "
                         "way to ask whether the actuator is the binding constraint or something "
                         "else is.")
parser.add_argument("--ee_only", action="store_true",
                    help="Ignore the object entirely and ask ONE question: can the arm put its hand "
                         "where the cube would have to go?\n\n"
                         "At reset, record the hand-to-cube offset and add it to the seated cube "
                         "pose. That is where the hand must end up for a rigidly held cube to be "
                         "seated. Then drive the HAND there and measure how close it gets. No grip, "
                         "no slip, no waypoint, no reward -- if the hand cannot reach that point, "
                         "every other measurement in this investigation is downstream of a "
                         "workspace limit and none of them mean what they appear to.")
parser.add_argument("--waypoint", action="store_true",
                    help="Carry OVER the base cube and descend, instead of driving straight at the "
                         "goal.\n\n"
                         "WHY IT IS NEEDED. The straight-line controller asymptotes at a median "
                         "66 mm of error and stays there however long the episode runs and however "
                         "stiff the arm is -- 69.4 mm at 16 s, 80.5 mm at 32 s, 66.4 mm at double "
                         "the eval stiffness. A proportional controller that still has authority "
                         "does not stall at a fixed distance; one driving its payload into an "
                         "obstacle does. The goal sits 34 mm above the RECEPTIVE cube's centre, so "
                         "a straight line from anywhere lower or beside it runs the carried cube "
                         "into the side of the base cube. Near-Goal never sees this because it "
                         "starts above the base already, which is exactly the family that scores.\n\n"
                         "So: climb to a clearance height, close the horizontal gap, then descend. "
                         "Three lines, and it is the difference between 'the task is unreachable' "
                         "and 'the straight line is blocked'.")
parser.add_argument("--clearance_m", type=float, default=0.06,
                    help="How far above the seated pose to carry before descending.")
parser.add_argument("--descend_xy_m", type=float, default=0.02,
                    help="Horizontal error under which the controller stops climbing and descends.")
parser.add_argument("--episode_length_s", type=float, default=None,
                    help="Override the episode time limit. A 209 mm carry at the train gains needs "
                         "more seconds than the 16 the task allows; separating 'too slow' from "
                         "'cannot get there' means being able to buy time and see if it finishes.")
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
v = env_cfg.variants
env_cfg.scene.insertive_object = v["scene.insertive_object"][args.insertive]
env_cfg.scene.receptive_object = v["scene.receptive_object"][args.receptive]
for _, term in vars(env_cfg.events).items():
    p = getattr(term, "params", None)
    if isinstance(p, dict) and "dataset_dir" in p:
        p["dataset_dir"] = args.dataset_dir
env_cfg.sim.physx.gpu_collision_stack_size = args.collision_stack_mb * 1024 * 1024
if args.episode_length_s is not None:
    env_cfg.episode_length_s = args.episode_length_s
_rst = getattr(env_cfg.events, "reset_from_reset_states", None)
if _rst is not None:
    _rst.params["reset_types"] = [args.reset_type]
    _rst.params["probs"] = [1.0]

if args.hand_effort_scale is not None:
    _act = env_cfg.scene.robot.actuators
    for _name, _a in list(_act.items()):
        lim = getattr(_a, "effort_limit_sim", None)
        if isinstance(lim, dict):
            _act[_name] = _a.replace(
                effort_limit_sim={k: v * args.hand_effort_scale for k, v in lim.items()})
    print(f"[oracle] hand effort limits x{args.hand_effort_scale}")

if args.motion_stiffness is not None or args.motion_damping is not None:
    _arm = env_cfg.actions.arm
    k = list(_arm.motion_stiffness)
    d = list(_arm.motion_damping_ratio)
    if args.motion_stiffness is not None:
        k[:3] = [args.motion_stiffness] * 3
    if args.motion_damping is not None:
        d[:3] = [args.motion_damping] * 3
    env_cfg.actions.arm = _arm.replace(motion_stiffness=tuple(k), motion_damping_ratio=tuple(d))
    print(f"[oracle] OSC translational gains -> stiffness {k[:3]}, damping ratio {d[:3]}")

env = gym.make(args.task, cfg=env_cfg)
u = env.unwrapped
dev = u.device
N = args.num_envs

obj = u.scene["insertive_object"]
rec = u.scene["receptive_object"]

# Grade off the same object the reward reads, so a pass here means what a pass during training means.
ctx = u.reward_manager.get_term_cfg("progress_context").func
cmd = u.command_manager.get_term("task_command")
print(f"[oracle] thresholds: position={cmd.success_position_threshold} "
      f"orientation={cmd.success_orientation_threshold}")

# The arm's per-step travel per unit action, read off the installed term rather than assumed -- the
# train and eval action groups differ on exactly this and picking the wrong one silently rescales
# the whole controller.
arm_term = u.action_manager.get_term("arm")
scale = getattr(arm_term.cfg, "scale_xyz_axisangle", (0.02,) * 6)
print(f"[oracle] arm scale_xyz_axisangle={scale}")
n_act = u.action_manager.total_action_dim
n_arm = arm_term.action_dim
print(f"[oracle] action dims: arm {n_arm}, total {n_act}")

STACK_HEIGHT_M = 0.034  # a 34 mm cube seated on a 34 mm cube
# The assembled offsets the reward reads, from each prop's metadata.yaml.
_INS_OFFSET_M = (0.0, 0.0, -0.017)
_REC_OFFSET_M = (0.0, 0.0, 0.017)
_ins_off = torch.tensor(_INS_OFFSET_M, device=dev).unsqueeze(0)
_rec_off = torch.tensor(_REC_OFFSET_M, device=dev).unsqueeze(0)

# DOES THE ARM MOVE, AND DOES THE CUBE COME WITH IT? Two different failures produce the same flat
# success rate: a controller whose end effector never travels (an OSC problem) and one whose end
# effector travels while the cube stays behind (a grip problem). Tracking both displacements from
# the reset pose separates them in one run.
robot = u.scene["robot"]
_palm_ids, _ = robot.find_bodies("rl_dg_mount")
_palm = _palm_ids[0]
_tip_sensors = {t: u.scene.sensors[f"{t}_object_s"]
                for t in ("rl_dg_1_tip", "rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip", "rl_dg_5_tip")
                if f"{t}_object_s" in u.scene.sensors}
travel_log = []

ever = torch.zeros(N, dtype=torch.bool, device=dev)
ep_len = torch.zeros(N, dtype=torch.long, device=dev)
done_end = done_ever = n_done = 0
lens_success, pos_err = [], []

env.reset()
_ee0 = robot.data.body_pos_w[:, _palm].clone()
_obj0 = obj.data.root_pos_w.clone()
_err0 = None
_step = 0
_hand_offset = (robot.data.body_pos_w[:, _palm] - obj.data.root_pos_w).clone()
_prev_ee = robot.data.body_pos_w[:, _palm].clone()
_cos_hist: list[float] = []
with torch.inference_mode():
    while n_done < args.episodes:
        # THE TARGET IS NOT "the receptive centre plus a cube height".
        # ProgressContext compares the insertive object's BOTTOM FACE to the receptive object's TOP
        # FACE, and both offsets are rotated by their own body's orientation:
        #     insertive_align = obj_pos + R_obj @ [0, 0, -0.017]
        #     receptive_align = rec_pos + R_rec @ [0, 0, +0.017]
        # Driving the cube CENTRE to rec + [0,0,0.034] therefore only zeroes the predicate when the
        # carried cube is level. The Stable-Grasp resets start at a mean roll+pitch of 1.245 rad --
        # about 71 degrees -- and this controller never commands rotation, so the cube stays tilted
        # and the target is misplaced by up to a full 34 mm. Near-Goal starts at 0.31 rad, nearly
        # level, which is why the same controller reaches 11 mm there and floors at 79 mm here.
        seated = (
            rec.data.root_pos_w
            + math_utils.quat_apply(rec.data.root_quat_w, _rec_off.expand(N, 3))
            - math_utils.quat_apply(obj.data.root_quat_w, _ins_off.expand(N, 3))
        )
        if args.ee_only:
            # Where the HAND has to be for a rigidly held cube to sit at `seated`, using the
            # hand-to-cube offset as it was at the reset.
            hand_target = seated + _hand_offset
            hand_err = math_utils.quat_apply_inverse(
                robot.data.root_quat_w, hand_target - robot.data.body_pos_w[:, _palm])
            act = torch.zeros(N, n_act, device=dev)
            act[:, :3] = (hand_err / torch.tensor(scale[:3], device=dev)).clamp(-1.0, 1.0)
            act[:, n_arm:] = args.hand_action
            if _err0 is None:
                _err0 = hand_err.norm(dim=-1).clone()
            _, _, terminated, truncated, _ = env.step(act)
            dones = terminated | truncated
            ep_len += 1
            _step += 1
            if _step % 20 == 0:
                print("  step %3d  hand-to-its-target %6.1f mm  (started %6.1f)" % (
                    _step, float(hand_err.norm(dim=-1).mean()) * 1000.0,
                    float(_err0.mean()) * 1000.0))
            d = dones.bool().flatten()
            if d.any():
                n_done += int(d.sum())
                ep_len[d] = 0
            continue
        if args.waypoint:
            # Phase 1: get above the base cube and close the horizontal gap. Phase 2: descend.
            # Switching on the HORIZONTAL error alone means the descent only begins once the cube
            # is over the target, which is what keeps it out of the base cube's side.
            xy_err = (seated[:, :2] - obj.data.root_pos_w[:, :2]).norm(dim=-1)
            over = xy_err < args.descend_xy_m
            target = seated.clone()
            climb = seated[:, 2] + args.clearance_m
            target[:, 2] = torch.where(over, seated[:, 2], climb)
        else:
            target = seated
        # THE COMMAND FRAME IS THE ROBOT BASE, NOT THE WORLD.
        # RelCartesianOSCAction.process_actions does
        #     ee_pos_b, _ = self._get_ee_pose_root_frame()
        #     self._ee_pos_des[:] = ee_pos_b + scaled[:, :3]
        # so the Cartesian delta is added to the end-effector pose expressed in base_link. Feeding
        # it a world-frame error rotates every command by the base orientation -- which produces
        # exactly the signature that had me chasing the arm for hours: half a metre of end-effector
        # travel with the goal error flat, because a rotated direction still has SOME component
        # toward the goal and reads as slow progress rather than as a wrong heading.
        err = math_utils.quat_apply_inverse(
            robot.data.root_quat_w, target - obj.data.root_pos_w)
        act = torch.zeros(N, n_act, device=dev)
        act[:, :3] = (err / torch.tensor(scale[:3], device=dev) * args.gain).clamp(-1.0, 1.0)
        act[:, n_arm:] = args.hand_action
        # Gymnasium's five-tuple, not the RSL-RL wrapper's four: this script talks to the raw env
        # because it needs no policy and therefore no wrapper.
        if _err0 is None:
            _err0 = err.norm(dim=-1).clone()
        _, _, terminated, truncated, _ = env.step(act)
        dones = terminated | truncated
        ep_len += 1
        _step += 1
        # DOES THE ARM GO WHERE IT IS TOLD? The end effector travels 512 mm over 300 steps while the
        # goal error sits flat at ~135 mm -- a lot of motion that closes nothing. Either the arm is
        # wandering, or the commanded direction is being interpreted in a frame other than the one
        # the error was computed in, in which case every "the arm is slow" reading tonight is really
        # "the arm is going the wrong way". The cosine between the command and the achieved
        # displacement separates those: ~1 means correct direction and merely slow, ~0 means the
        # command is orthogonal to the motion it produces.
        _now = robot.data.body_pos_w[:, _palm]
        _moved = math_utils.quat_apply_inverse(robot.data.root_quat_w, _now - _prev_ee)
        _cmd = act[:, :3]
        _mn = _moved.norm(dim=-1)
        _cn = _cmd.norm(dim=-1)
        _ok = (_mn > 1e-6) & (_cn > 1e-6)
        if _ok.any():
            _cos = ((_moved[_ok] * _cmd[_ok]).sum(dim=-1) / (_mn[_ok] * _cn[_ok])).mean()
            _cos_hist.append(float(_cos))
        _prev_ee = _now.clone()
        if _step % 20 == 0:
            ee_d = (robot.data.body_pos_w[:, _palm] - _ee0).norm(dim=-1)
            obj_d = (obj.data.root_pos_w - _obj0).norm(dim=-1)
            row = {
                "step": _step,
                "ee_travel_mm": float(ee_d.mean()) * 1000.0,
                "object_travel_mm": float(obj_d.mean()) * 1000.0,
                "goal_error_mm": float(err.norm(dim=-1).mean()) * 1000.0,
                # "cube moved" is a DISPLACEMENT MAGNITUDE, so a cube that simply falls to the table
                # registers ~79 mm of travel and reads as if it were being carried. Height separates
                # the two: 21 mm is the table.
                "cube_z_mm": float(obj.data.root_pos_w[:, 2].mean()) * 1000.0,
                "cmd_vs_motion_cos": (sum(_cos_hist[-20:]) / max(1, len(_cos_hist[-20:]))),
                "commanded_mm": float(act[:, :3].abs().max(dim=1).values.mean()) * scale[0] * 1000.0,
                "peak_tip_force_N": max(
                    (float(sen.data.net_forces_w.norm(dim=-1).max(dim=1).values.mean())
                     for sen in _tip_sensors.values() if sen.data.net_forces_w is not None),
                    default=0.0),
            }
            travel_log.append(row)
            print("  step %3d  EE moved %6.1f mm | cube moved %6.1f mm | goal error %6.1f mm "
                  "| cube z %5.1f mm | tip %.2f N | cos(cmd,motion) %+.2f" % (
                      row["step"], row["ee_travel_mm"], row["object_travel_mm"],
                      row["goal_error_mm"], row["cube_z_mm"], row["peak_tip_force_N"],
                      row["cmd_vs_motion_cos"]))

        succ = ctx.success.clone()
        ever |= succ
        d = dones.bool().flatten()
        if d.any():
            done_end += int(succ[d].sum())
            done_ever += int(ever[d].sum())
            pos_err += ctx.xyz_distance[d].tolist()
            lens_success += ep_len[d & succ].tolist()
            n_done += int(d.sum())
            ever[d] = False
            ep_len[d] = 0

env.close()


def pct(a, b):
    return 100.0 * a / max(1, b)


def q(xs, p):
    t = torch.tensor(xs)
    return float(t.quantile(p)) if len(xs) else float("nan")


res = {
    "task": args.task,
    "reset_type": args.reset_type,
    "episodes": n_done,
    "gain": args.gain,
    "hand_action": args.hand_action,
    "arm_scale": list(scale),
    "motion_stiffness": args.motion_stiffness,
    "motion_damping": args.motion_damping,
    "episode_length_s": env_cfg.episode_length_s,
    "waypoint": args.waypoint,
    "clearance_m": args.clearance_m,
    "success_held_at_end_pct": pct(done_end, n_done),
    "success_ever_during_pct": pct(done_ever, n_done),
    "position_error_m": {"median": q(pos_err, 0.5), "p10": q(pos_err, 0.1), "p90": q(pos_err, 0.9)},
    "travel": travel_log,
    "episode_length_success": {
        "n": len(lens_success),
        "median": sorted(lens_success)[len(lens_success) // 2] if lens_success else None,
        # A success shorter than 11 steps is a reset that spawned inside tolerance, not a carry.
        "n_gt_10": sum(1 for x in lens_success if x > 10),
    },
}
print(f"\n[ORACLE] {args.reset_type}: held-at-end {res['success_held_at_end_pct']:.1f}% | "
      f"ever-during {res['success_ever_during_pct']:.1f}% | n={n_done}")
print(f"[ORACLE] final position error: median {res['position_error_m']['median']*1000:.1f} mm "
      f"(p10 {res['position_error_m']['p10']*1000:.1f}, p90 {res['position_error_m']['p90']*1000:.1f})")
print(f"[ORACLE] successes longer than 10 steps: {res['episode_length_success']['n_gt_10']}")
if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")
simulation_app.close()
