"""Do the GRASPED reset families actually hold the cube?

WHY THIS EXISTS
---------------
The five-finger curriculum sits at task_1_success_rate = 0.0000 on ObjectAnywhereEEGrasped after
~200 iterations of training on that family. C3 is supposed to be the EASY half of the task: the cube
starts already in the hand, in the air (bank mean z = 0.103 m against a 0.021 m table), and the only
thing left to do is carry it to the goal and let go. A policy that cannot score there will never
score on Reaching, so "is C3 learnable at all" has to be answered before any more GPU-hours go into
it.

The bank stores a pose, not a force. `record_reset_states` accepted a state where the cube was near
the fingers with the joints in a closed posture; nothing in that check guarantees the grip carries
the cube's weight once physics resumes. If it does not, every C3 episode begins with the cube in
free fall and the family is unsolvable by construction -- which looks identical, in the training
logs, to a policy that simply has not learned it yet.

WHAT IT MEASURES
----------------
Reset into one family, then apply the ZERO action for the whole episode -- for a relative OSC action
space that is "hold the current pose", the null policy. Then per step: object height, object speed,
object-to-goal distance, palm-to-object distance, and the per-fingertip contact forces. A grip that
holds shows a flat z and non-zero tip forces; a grip that does not shows z decaying to the table
within a few tenths of a second.

Zero action is the right probe precisely because it is not a policy: whatever the cube does here is
the RESET's doing, not the actor's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str,
                    default="OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-Play-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--insertive", type=str, default="cube34")
parser.add_argument("--receptive", type=str, default="cube34")
parser.add_argument("--dataset_dir", type=str, default="./Datasets_ur10e_delto/OmniReset")
parser.add_argument("--collision_stack_mb", type=int, default=1024)
parser.add_argument("--reset_type", type=str, default="ObjectAnywhereEEGrasped")
parser.add_argument("--hold_hand", action="store_true",
                    help="Also drive the hand action channels to +1 (full close) instead of 0, to "
                         "separate 'the stored posture does not hold' from 'the null action lets go'.")
parser.add_argument("--grip_bias", type=float, default=None,
                    help="Constant CLOSING offset, in radians per control step, added to the hand\n"
                         "action before it is integrated onto the measured joint position.\n\n"
                         "WHY THIS KNOB EXISTS. The hand action term is\n"
                         "RelativeJointPositionAction with use_zero_offset=True, i.e.\n"
                         "  target = measured_joint_pos + scale * action.\n"
                         "At action = 0 the PD set point equals the measurement, so the joint error\n"
                         "is zero and the commanded grip force is zero. That is applied on the FIRST\n"
                         "step after every reset, which overwrites the joint_position_target the\n"
                         "reset bank restored -- the squeeze the recorder captured. A grasp can only\n"
                         "survive if the policy re-commands it every single step.\n\n"
                         "A constant offset on the flexion joints restores exactly the condition the\n"
                         "bank recorded: a set point held a fixed angle ahead of the measurement,\n"
                         "which against a blocked finger is a constant force (stiffness * bias) and\n"
                         "against a free finger is a slow close. It is a bias, not a constraint --\n"
                         "the policy can still command opening.")
parser.add_argument("--grip_posture", action="store_true",
                    help="Replace the relative hand action with an ABSOLUTE one whose zero point is "
                         "the validated closed posture.\n\n"
                         "WHY. The constant closing offset fixes the release at reset, but it is an "
                         "INTEGRATING command: `target = measured + scale*action + bias` moves the "
                         "set point a further `bias` beyond wherever the fingers currently are, on "
                         "every single step, without bound. The fingers never settle at a squeeze -- "
                         "they keep tightening, and the cube is slowly extruded. Measured: it sinks "
                         "about 0.1 mm per step at rest and 0.25 mm per step during a carry, which "
                         "over a full episode is the whole grasp.\n\n"
                         "An absolute joint-position action with the closed posture as its offset "
                         "commands a POSE, not an increment. Action 0 is exactly the posture the "
                         "grasp recorder validated, the squeeze is bounded by how far the cube stops "
                         "the fingers short of it, and the policy still has its full range either "
                         "side.")
parser.add_argument("--video", type=str, default=None,
                    help="Write an mp4 of the probe here. The null-action rollout is the one worth "
                         "watching: whatever the cube does in it is the RESET's doing, so the frames "
                         "answer 'is the cube in the hand' directly instead of by inference from a "
                         "distance in millimetres.")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import subprocess  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

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
_rst = getattr(env_cfg.events, "reset_from_reset_states", None)
if _rst is not None:
    _rst.params["reset_types"] = [args.reset_type]
    _rst.params["probs"] = [1.0]

if args.grip_posture:
    from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
    from uwlab_assets.robots.ur10e_delto.actions import DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS

    g = env_cfg.actions.gripper
    names = list(g.scale) if isinstance(g.scale, dict) else []
    missing = [n for n in names if n not in DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS]
    if missing:
        raise ValueError(f"no validated closed angle for {missing}")
    env_cfg.actions.gripper = JointPositionActionCfg(
        asset_name=g.asset_name,
        joint_names=g.joint_names,
        scale={n: g.scale[n] for n in names},
        clip={n: (-1.0, 1.0) for n in names},
        offset={n: DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS[n] for n in names},
        use_default_offset=False,
    )
    print(f"[probe] absolute hand action at the validated closed posture, {len(names)} joints")

if args.grip_bias is not None:
    # EVERY hand joint, which is what the `--hold_hand` control demonstrates works: driving all
    # twenty at +0.1 rad/step takes the fingertip forces from 0.02-0.06 N to 0.48-2.56 N and holds
    # the C3 cube for the whole episode (z 91.3 -> 95.0 mm, i.e. it even lifts slightly). Restricting
    # the bias to the `_2`/`_4` flexion pair -- the joints that differ between the recorded open and
    # closed postures -- only got the drop from 83 mm down to 40 mm at a bias of 0.08. The `_3`
    # joints are identical in both stored postures but carry the largest stiffness (4.0 N*m/rad),
    # and the hold_hand forces show they contribute most of the grip.
    gripper = env_cfg.actions.gripper
    names = [n for n in gripper.scale] if isinstance(gripper.scale, dict) else []
    bias = {n: args.grip_bias for n in names}
    env_cfg.actions.gripper = gripper.replace(use_zero_offset=False, offset=bias)
    print(f"[probe] grip bias {args.grip_bias} rad on {sorted(bias)}")

if args.video:
    # Framed on the hand rather than on the whole cell: the question is whether five fingertips are
    # on a 34 mm cube, and at the wide preset that is a dozen pixels.
    env_cfg.viewer.eye = (0.78, -0.28, 0.16)
    env_cfg.viewer.lookat = (0.48, 0.01, 0.045)
    env_cfg.viewer.resolution = (args.width, args.height)
    env_cfg.viewer.origin_type = "world"

env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
u = env.unwrapped
dev = u.device

obj = u.scene["insertive_object"]
rec = u.scene["receptive_object"]
robot = u.scene["robot"]

# The palm body, the one `ee_asset_distance` reads.
palm_ids, _ = robot.find_bodies("rl_dg_mount")
tip_names = ["rl_dg_1_tip", "rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip", "rl_dg_5_tip"]
tip_ids, tip_found = robot.find_bodies(tip_names)
print(f"[probe] tip bodies: {tip_found}")
sensors = {t: u.scene.sensors[f"{t}_object_s"] for t in tip_names if f"{t}_object_s" in u.scene.sensors}
print(f"[probe] contact sensors: {sorted(sensors)}")

ff = None
if args.video:
    Path(args.video).parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}",
         "-r", f"{1.0 / (env_cfg.sim.dt * env_cfg.decimation):.4f}", "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
         "-movflags", "frag_keyframe+empty_moov+default_base_moof", "-g", "30", args.video],
        stdin=subprocess.PIPE)

expect_bytes = args.width * args.height * 3
skipped_frames = 0

n_act = u.action_manager.total_action_dim
zero = torch.zeros(args.num_envs, n_act, device=dev)
act = zero.clone()
if args.hold_hand:
    act[:, 6:] = 1.0

u.reset()

# THE STATE THE RESET ACTUALLY PRODUCED, read before a single control step runs. Everything logged
# inside the loop below is post-step: env.step() advances `decimation` physics substeps before
# anything can be read, which is already long enough for a cube that is not held to start leaving.
# Telling "the bank never put the cube in the hand" apart from "the first control step threw it out"
# needs this frame, and only this frame.
_p0 = obj.data.root_pos_w
_tips0 = robot.data.body_pos_w[:, tip_ids]
_d0 = (_tips0 - _p0.unsqueeze(1)).norm(dim=-1)
reset_frame = {
    "z_mm": float(_p0[:, 2].mean()) * 1000.0,
    "tip_to_obj_mean_mm": float(_d0.mean()) * 1000.0,
    "tip_to_obj_min_mm": float(_d0.min(dim=1).values.mean()) * 1000.0,
    "tip_to_obj_min_p10_mm": float(_d0.min(dim=1).values.quantile(0.10)) * 1000.0,
    "per_tip_mean_mm": {t: float(_d0[:, i].mean()) * 1000.0 for i, t in enumerate(tip_found)},
    "envs_with_a_tip_within_25mm": float((_d0 < 0.025).any(dim=1).float().mean()),
    "envs_with_two_tips_within_30mm": float(((_d0 < 0.030).sum(dim=1) >= 2).float().mean()),
    # CAN THIS GRASP EVER BE PUT DOWN ON SOMETHING? To seat a 34 mm cube on top of another, nothing
    # may be underneath it -- and a five-finger hand that wraps a small cube can easily have
    # fingertips below its bottom face, in which case the tips strike the base cube before the
    # carried cube ever touches it. That is invisible in every distance-to-centre statistic and it
    # would cap the achievable error at roughly a fingertip's depth, whatever the policy does.
    "cube_bottom_z_mm": float(_p0[:, 2].mean() - 0.017) * 1000.0,
    "lowest_tip_z_mm": float(_tips0[:, :, 2].min(dim=1).values.mean()) * 1000.0,
    # Positive means the lowest fingertip sits BELOW the cube's bottom face by this much.
    "tip_below_cube_bottom_mm": float(
        ((_p0[:, 2] - 0.017) - _tips0[:, :, 2].min(dim=1).values).mean()
    ) * 1000.0,
    "envs_with_a_tip_below_the_cube": float(
        (_tips0[:, :, 2].min(dim=1).values < (_p0[:, 2] - 0.017)).float().mean()
    ),
}
print("RESET_FRAME " + json.dumps(reset_frame))

if ff is not None:
    # One throwaway render before the loop, so the first real frame is not also the frame that
    # builds the render product.
    try:
        u.render()
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] warm-up render failed ({exc.__class__.__name__}: {exc})")

trace = []
with torch.inference_mode():
    for step in range(args.steps):
        u.step(act)
        if ff is not None:
            # RE-AIM SPARINGLY. `set_camera_view` rebuilds the viewport's render product, and
            # `ManagerBasedRLEnv.render()` re-attaches its rgb annotator to whatever product it
            # finds -- so re-aiming immediately before every render raced the rebuild and threw
            # "Unable to write from unknown dtype, kind=f, size=0" from inside
            # `_rgb_annotator.attach()`, on step 0, every time. Once every 10 control steps is
            # smooth enough for a cube falling at 6 mm/step and leaves the product settled.
            if step % 10 == 0:
                c = obj.data.root_pos_w[0].tolist()
                off = [e - l for e, l in zip(env_cfg.viewer.eye, env_cfg.viewer.lookat)]
                u.sim.set_camera_view([c[i] + off[i] for i in range(3)], c)
            # And render defensively regardless: a dropped frame is a cosmetic loss, while an
            # exception here loses the whole recording AND hangs Isaac's shutdown for ten minutes.
            try:
                frame = u.render()
            except Exception as exc:  # noqa: BLE001
                if skipped_frames == 0:
                    print(f"[probe] render unavailable ({exc.__class__.__name__}: {exc}); skipping frames")
                frame = None
            if frame is not None and getattr(frame, "size", 0) >= expect_bytes // 3:
                ff.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            else:
                skipped_frames += 1
        p = obj.data.root_pos_w
        speed = obj.data.root_lin_vel_w.norm(dim=-1)
        palm = robot.data.body_pos_w[:, palm_ids[0]]
        tips = robot.data.body_pos_w[:, tip_ids]                      # (N, n_tips, 3)
        tip_d = (tips - p.unsqueeze(1)).norm(dim=-1)                  # (N, n_tips)
        forces = {}
        for t, s in sensors.items():
            f = s.data.net_forces_w  # (N, B, 3)
            forces[t] = float(f.norm(dim=-1).max(dim=1).values.mean()) if f is not None else 0.0
        trace.append({
            "step": step,
            "z_mean": float(p[:, 2].mean()),
            "z_min": float(p[:, 2].min()),
            "z_p10": float(p[:, 2].quantile(0.10)),
            "speed_mean": float(speed.mean()),
            "palm_to_obj_mean": float((palm - p).norm(dim=-1).mean()),
            "tip_to_obj_mean_mm": float(tip_d.mean()) * 1000.0,
            "tip_to_obj_min_mm": float(tip_d.min(dim=1).values.mean()) * 1000.0,
            "tips_within_25mm": float((tip_d < 0.025).any(dim=1).float().mean()),
            "obj_to_rec_xy": float((obj.data.root_pos_w[:, :2] - rec.data.root_pos_w[:, :2]).norm(dim=-1).mean()),
            "tip_forces_N": forces,
        })

z0 = trace[0]["z_mean"]
# The episode is 160 control steps and env.step() auto-resets inside the same call, so a trace that
# reaches the limit ends on a FRESH episode's cube -- which reads as the cube having risen back into
# the hand. Cut at the first upward jump larger than any physical rise, and summarise before it.
_z = [r["z_mean"] for r in trace]
_cut = len(_z)
for _i in range(1, len(_z)):
    if _z[_i] - _z[_i - 1] > 0.020:
        _cut = _i
        break
zf = _z[_cut - 1]
# "Dropped" = ended within a cube-half-height of the table. The table top is at 0.021 - 0.017 for a
# 34 mm cube resting on it, so a held cube that never rose is not counted as a drop; a cube that
# started at 0.10 m and ended at 0.021 m unambiguously is.
final_z = torch.tensor([trace[-1]["z_min"]])
res = {
    "reset_frame": reset_frame,
    "reset_type": args.reset_type,
    "task": args.task,
    "hold_hand": args.hold_hand,
    "grip_bias": args.grip_bias,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "z_start_mean": z0,
    "z_end_mean": zf,
    "z_drop_mean_mm": (z0 - zf) * 1000.0,
    "steps_before_autoreset": _cut,
    "speed_end_mps": trace[_cut - 1]["speed_mean"],
    "tip_forces_end_N": trace[_cut - 1]["tip_forces_N"],
    "trace": trace,
}
print("GRIP_PROBE " + json.dumps({k: v for k, v in res.items() if k != "trace"}))
for r in trace[:: max(1, args.steps // 12)]:
    tf = " ".join(f"{k.split('_')[2]}:{v:.2f}" for k, v in r["tip_forces_N"].items())
    print(f"  step {r['step']:>3}  z {r['z_mean']*1000:7.1f} mm  (min {r['z_min']*1000:6.1f})  "
          f"speed {r['speed_mean']:.3f}  palm-obj {r['palm_to_obj_mean']*1000:6.1f}  "
          f"tip-obj {r['tip_to_obj_mean_mm']:6.1f}/min {r['tip_to_obj_min_mm']:5.1f} mm  "
          f"near {r['tips_within_25mm']:.2f}  tipN {tf}")
if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")
if ff is not None:
    ff.stdin.close()
    ff.wait()
    print(f"wrote {args.video} (skipped {skipped_frames} un-warmed frames)")
env.close()
simulation_app.close()
