"""Render a trained cube-stacking policy to video, and log where the successes actually are.

WHY NOT JUST `play.py --video`. That records a fixed window from whenever it happens to start, so
whether a success lands inside the clip is luck. This runs a chosen number of episodes, records the
whole thing, and writes a JSON sidecar giving the frame range of every episode together with whether
it succeeded — so the successful segments can be cut out afterwards instead of hoping.

The camera is placed explicitly rather than left at the task default (eye 2.0 m out, 0.75 m up),
which frames the whole cell and renders a 34 mm cube about ten pixels across.

RENDERING IS LOCAL-ONLY BY NECESSITY. The deployment guide requires an RTX-class GPU on driver
branch 580 for any camera-enabled run: A100/H100 have no graphics engine and segfault about a second
into startup, and driver 595 segfaults `rtx.scenedb` on Isaac 5.1. The rented A6000 is on 570 and is
used headless only.

THIS HOST CANNOT RENDER THROUGH ISAAC, AND THE LINE ABOVE USED TO CLAIM OTHERWISE. It said the dev
box was "a 4070 Ti SUPER on 580.159.04"; `nvidia-smi` reports **610.57.04**, and that stale comment
cost a round of re-deriving the contradiction. The GPU is right, the driver branch is not: 610 is
what is installed, no 580-branch package exists in the repos (`nvidia-open-lts` is also 610.57.04),
and IOMMU is active (36 groups in /sys/kernel/iommu_groups, no `iommu=` override on the cmdline).
Isaac logs `[omni.rtx] IOMMU is enabled`, a documented cause of broken Omniverse rendering.

Six genuine defects in this file were found and fixed before that was understood, and the camera is
now provably correct (exact look-at quaternion, `cam.data.pos_w` non-zero) -- yet frames stay a
smooth gradient, mean |Laplacian| 0.17-0.36 against ~4+ for real content. So do NOT debug this path
further without first proving the host renders one frame of anything. The remedies are all
host-level, not code-level: boot with `iommu=off` (bootloader edit + reboot), rent a display-capable
GPU on the 580 branch, or render the robot from its own USD meshes offscreen -- which is what
`render_meshes.py` does, driven by the same physics via `trace_full_poses.py`, and which works.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import json
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--checkpoint", default=None)
p.add_argument("--null_action", action="store_true",
               help="Roll out the ZERO action instead of a policy, and require no checkpoint.\n\n"
                    "This is how a RESET is filmed rather than a policy. For the relative OSC action\n"
                    "space the zero action is 'hold the current pose', so whatever the cube does in\n"
                    "the clip is the reset's doing and nothing else's -- which is the only way to\n"
                    "show that the EEGrasped families were not holding the cube without the\n"
                    "objection that the policy dropped it.")
p.add_argument("--motion_stiffness", type=float, default=None,
               help="Override the OSC's translational stiffness, to match how the policy was "
                    "trained. The curricula pass this to train.py as a hydra override and the Play "
                    "configs ship softer defaults, so a recording made without it shows the policy "
                    "driving a different robot from the one it learned.")
p.add_argument("--motion_damping", type=float, default=None,
               help="Override the OSC's translational damping ratio. See --motion_stiffness.")
p.add_argument("--grip_bias", type=float, default=None,
               help="Constant closing offset added to every hand action before it is integrated "
                    "onto the measured joint position. Filming the same reset with and without it "
                    "is the before/after pair.")
p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--episodes", type=int, default=12)
p.add_argument("--out", type=str, default="artifacts/rollout")
p.add_argument("--insertive", default="cube34")
p.add_argument("--receptive", default="cube34")
p.add_argument("--dataset_dir", default="./Datasets_ur10e_delto/OmniReset")
p.add_argument("--collision_stack_mb", type=int, default=None)
p.add_argument(
    "--view", choices=["wide", "closeup"], default="closeup",
    help="Camera preset. WIDE is the original framing (eye 0.75 -0.55 0.45): it shows the whole "
         "cell and is right for the full task, where the approach itself is the thing to watch. "
         "But a 34 mm cube is then a few dozen pixels and the upper arm links occlude much of the "
         "frame -- and on near-goal recordings the hand sits directly between that camera and the "
         "cube, hiding the thing being judged exactly when it matters. CLOSEUP is framed from the "
         "measured geometry instead: the stored reset banks put the receptive cube at "
         "(0.482, 0.012, 0.021) +-0.13 m and the insertive at z 0.056, so the camera looks at "
         "(0.48, 0.01, 0.045) from 0.42 m at ~15 degrees elevation, on the far side of the "
         "workspace from the arm base. The cube then subtends ~100 px instead of ~30.")
p.add_argument("--eye", type=float, nargs=3, default=None)
p.add_argument("--lookat", type=float, nargs=3, default=None)
p.add_argument(
    "--track", action="store_true", default=True,
    help="Re-aim the camera at the receptive cube at the start of every episode. The reset banks "
         "place that cube anywhere in x 0.35..0.60, y -0.20..+0.20, so a fixed close-up frames some "
         "episodes well and puts the cube half out of shot -- or a hand's width from the lens -- in "
         "others. Tracking keeps the same relative viewpoint for every episode.")
p.add_argument("--no-track", dest="track", action="store_false")
p.add_argument("--track_asset", choices=["receptive_object", "insertive_object"],
               default="receptive_object",
               help="Which body the camera follows. The receptive cube is the right subject for a "
                    "stack, because that is where the episode ends. It is the WRONG subject for the "
                    "grasped families: their insertive cube starts ~180 mm away in the hand, so the "
                    "hand spends the episode out of frame and the only thing the clip shows is a "
                    "cube dropping into view from above.")
p.add_argument("--width", type=int, default=1280)
p.add_argument("--height", type=int, default=720)
p.add_argument(
    "--reset_type",
    default="ObjectAnywhereEEAnywhere",
    choices=["ObjectAnywhereEEAnywhere", "ObjectRestingEEGrasped",
             "ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped"],
    help="Which reset family to roll out FROM. Rendering the family a policy is actually competent "
         "at is not cherry-picking as long as the family is named -- it is the same decomposition "
         "the paper's Fig. 8 reports. Rendering only the full task when the policy cannot yet do it "
         "produces ten identical failure clips, which show nothing.",
)
args, _ = p.parse_known_args()

# Cameras are required for any rendering; without this the env builds headless and the recorder
# silently produces nothing.
args.enable_cameras = True
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

# ISAAC SIM 6 COMPAT, and it must sit exactly here: after the AppLauncher has started and BEFORE the
# first `import uwlab_tasks`. Isaac Lab 3.0 dropped `isaacsim.core.utils` from the default extension
# search path, so importing uwlab_tasks first dies with "No module named 'isaacsim.core.utils'" --
# which is precisely how this script failed against .venv-is6 while every probe script worked. The
# bridge also rebinds Isaac Lab's jit-scripted math helpers to accept the warp-backed ProxyArray
# that asset data now returns. Guarded, so the script still runs under the old Isaac 5.1 .venv where
# the shim neither exists nor is needed.
try:
    import uwlab_is6_bridge  # noqa: E402

    uwlab_is6_bridge.apply()
except ImportError:
    pass

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import uwlab_tasks  # noqa: F401,E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

# The repo's own train.py/play.py run the agent cfg through cli_args.sanitize_rsl_rl_cfg before
# handing it to the runner; without it the runner forwards keys the fork's PPO does not accept and
# dies with "PPO.__init__() got an unexpected keyword argument 'optimizer'". Reuse the repo's
# sanitizer rather than reimplementing which keys are legal.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/mattephi/Work/Rubitek/UWLab-cubestack/scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa: E402

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
v = cfg.variants
cfg.scene.insertive_object = v["scene.insertive_object"][args.insertive]
cfg.scene.receptive_object = v["scene.receptive_object"][args.receptive]
# Re-apply what the variant swap just threw away. The task's __post_init__ rebuilds the insertive
# object's spawn with activate_contact_sensors=True (ur10e_delto_cubestack_cfg.py:356-358), because
# make_insertive_object never sets it; assigning the raw variant over the top drops that flag, and
# the object's ContactSensor then fails initialisation with "could not find any bodies with contact
# reporter API" -- which is how every render died while the identical task id ran fine in probes
# that leave the variants alone.
cfg.scene.insertive_object = cfg.scene.insertive_object.replace(
    spawn=cfg.scene.insertive_object.spawn.replace(activate_contact_sensors=True)
)
for _, term in vars(cfg.events).items():
    prm = getattr(term, "params", None)
    if isinstance(prm, dict) and "dataset_dir" in prm:
        prm["dataset_dir"] = args.dataset_dir
if args.collision_stack_mb is not None:
    # Isaac Lab 3.0 carries no `physx` section on SimulationCfg under the Newton backend, so this
    # attribute exists only on the PhysX path and the bare assignment raised AttributeError against
    # .venv-is6. Everything else in this repo sizes the collision stack through
    # UWLAB_GPU_COLLISION_STACK_SIZE, which is read during cfg construction -- already applied by
    # the time we reach here -- so skipping is correct rather than a silent downgrade.
    physx = getattr(cfg.sim, "physx", None)
    if physx is not None:
        physx.gpu_collision_stack_size = args.collision_stack_mb * 1024 * 1024
    else:
        print(
            "[render_rollout] cfg.sim has no physx section (Isaac Lab 3 / Newton); collision stack "
            "is set by UWLAB_GPU_COLLISION_STACK_SIZE instead",
            flush=True,
        )

_rst = getattr(cfg.events, "reset_from_reset_states", None)
if _rst is not None:
    _rst.params["reset_types"] = [args.reset_type]
    _rst.params["probs"] = [1.0]

VIEWS = {
    "wide": ([0.75, -0.55, 0.45], [0.45, 0.0, 0.12]),
    "closeup": ([0.78, -0.28, 0.16], [0.48, 0.01, 0.045]),
}
_eye, _lookat = VIEWS[args.view]
cfg.viewer.eye = tuple(args.eye if args.eye is not None else _eye)
cfg.viewer.lookat = tuple(args.lookat if args.lookat is not None else _lookat)
cfg.viewer.resolution = (args.width, args.height)
cfg.viewer.origin_type = "world"

if args.motion_stiffness is not None or args.motion_damping is not None:
    _arm = cfg.actions.arm
    _k, _d = list(_arm.motion_stiffness), list(_arm.motion_damping_ratio)
    if args.motion_stiffness is not None:
        _k[:3] = [args.motion_stiffness] * 3
    if args.motion_damping is not None:
        _d[:3] = [args.motion_damping] * 3
    cfg.actions.arm = _arm.replace(motion_stiffness=tuple(_k), motion_damping_ratio=tuple(_d))
    print(f"[render] OSC translational gains -> stiffness {_k[:3]}, damping ratio {_d[:3]}")

if args.grip_bias is not None:
    # Applied to whatever hand action term the task installed, by name, so this works for the
    # 20-joint and the 8-joint variants alike. See ur10e_delto_cubestack_cfg._apply_grip_bias.
    _g = cfg.actions.gripper
    _names = list(_g.scale) if isinstance(_g.scale, dict) else []
    cfg.actions.gripper = _g.replace(use_zero_offset=False, offset={n: args.grip_bias for n in _names})
    print(f"[render] grip bias {args.grip_bias} rad on {len(_names)} hand joints")

agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")

# Episode length in control steps -- record every step of every requested episode.
steps_per_ep = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
total_steps = steps_per_ep * args.episodes
print(f"[render] {steps_per_ep} steps/episode x {args.episodes} episodes = {total_steps} frames")

# FRAMES ARE STREAMED TO ffmpeg, NOT BUFFERED.
# gymnasium's RecordVideo appends every frame to a Python list and encodes once at close. At
# 1280x720x3 that is 2.76 MB per frame, so a 60-episode recording (9600 frames) needs 26 GB of RAM
# on a 30 GB machine -- it thrashes swap for an hour and then dies with nothing written, which is
# the worst possible failure mode for a job whose entire output arrives at the end. Piping raw
# frames straight into ffmpeg holds one frame at a time and puts no ceiling on recording length.
video_path = out_dir / "rollout.mp4"
ff = None


def start_ffmpeg(first_frame):
    """Start the encoder from the FIRST FRAME'S ACTUAL SHAPE, not from --width/--height.

    Declaring the geometry up front was wrong in two ways at once, and silently. `--width 960
    --height 540` produced `-s 960x540 -pix_fmt rgb24` (1555200 bytes/frame) while the viewport
    actually handed back 480x270 RGBA (518400 bytes/frame): Isaac's RTX path clamps
    `cfg.viewer.resolution` (the log says "Render resolution of (480, 270)") and the annotator
    returns 4 channels, not 3. ffmpeg then read misaligned chunks -- every frame sheared, the frame
    count inflated from 480 to 853, and an 85 s file that the player reports as 3 s. It never
    errored; it just wrote garbage, which is why this has to be derived rather than declared.
    """
    h, w = first_frame.shape[0], first_frame.shape[1]
    ch = first_frame.shape[2] if first_frame.ndim == 3 else 1
    pix = {1: "gray", 3: "rgb24", 4: "rgba"}.get(ch)
    if pix is None:
        raise RuntimeError(f"[render] unsupported frame with {ch} channels, shape {first_frame.shape}")
    if (w, h) != (args.width, args.height):
        print(f"[render] viewport returned {w}x{h}x{ch}, not the requested "
              f"{args.width}x{args.height}; encoding at the real size", flush=True)
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{w}x{h}",
         "-r", f"{1.0 / (cfg.sim.dt * cfg.decimation):.4f}", "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
         # frag_keyframe+empty_moov instead of +faststart: faststart writes the moov atom only at
         # close, so the file cannot be opened at all until the render finishes. A fragmented mp4 is
         # readable as it is written, which is what makes a mid-render frame check possible.
         "-movflags", "frag_keyframe+empty_moov+default_base_moof",
         "-g", "30", str(video_path)],
        stdin=subprocess.PIPE,
    )

env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

if args.null_action:
    _n = env.unwrapped.action_manager.total_action_dim
    _zero = torch.zeros(args.num_envs, _n, device=env.unwrapped.device)
    def policy(_obs):  # noqa: D401 -- same call signature as the inference policy
        return _zero
    print(f"[render] NULL ACTION over {_n} dimensions; no checkpoint loaded")
else:
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --null_action is given")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

u = env.unwrapped
ctx = u.reward_manager.get_term_cfg("progress_context").func
cmd = u.command_manager.get_term("task_command")
print(f"[render] thresholds pos={cmd.success_position_threshold} rot={cmd.success_orientation_threshold}")

obs = env.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]

# THE SUCCESS PREDICATE DOES NOT REQUIRE THE CUBE TO BE LET GO OF.
# ProgressContext tests one distance and nothing else: no contact check, no resting check, no
# release check. A cube held motionless in the fingers at the goal point scores identically to a
# cube placed there. That is the paper's criterion and it is not changed here -- but a clip captioned
# "stacked" should be one where the cube is actually standing on the other cube, so these two extra
# per-episode measurements are recorded and the clip cutter can prefer episodes that satisfy them.
#
#   settled  final |v| of the insertive cube, m/s. A placed cube is at rest.
#   released final distance from the DELTO mount to the cube. If the hand has withdrawn, the cube is
#            supported by the stack rather than by the fingers.
# Defensive: a rollout takes the better part of an hour and its entire output arrives at the end,
# so a wrong scene key or body name here must not be able to destroy the recording. If either lookup
# fails the two extra measurements are simply omitted and the clip cutter falls back to ranking by
# episode length, which is what it did before these existed.
try:
    insertive = u.scene["insertive_object"]
    _mount = u.scene["robot"].find_bodies("rl_dg_mount")[0][0]
except Exception as exc:  # noqa: BLE001
    print(f"[render] physical-state probes unavailable ({exc.__class__.__name__}: {exc}); "
          f"recording position only")
    insertive, _mount = None, None

def aim_camera() -> None:
    """Point the camera at THIS episode's receptive cube, keeping the preset's relative viewpoint.

    The OFFSET is preserved rather than the absolute lookat, so the framing -- 0.42 m away, ~15
    degrees elevation, on the far side of the workspace from the arm base -- is identical for every
    episode while the subject moves. Without this, the reset banks' 0.25 x 0.40 m spread put some
    episodes' cubes half out of shot and others a hand's width from the lens.
    """
    if not args.track:
        # RE-ASSERT the preset view rather than returning. Setting cfg.viewer.eye/lookat once before
        # gym.make is not enough: something in the env's viewport handling relocates the camera after
        # the first frame, and with tracking OFF the recording still went blank from frame 1 onward
        # (frame 0 correct, every frame after it empty sky). Re-applying the fixed view each frame
        # overrides whatever moves it, whether that is the ViewportCameraController or a viewport
        # rebuild. Cheap enough to do per frame.
        try:
            u.sim.set_camera_view(tuple(cfg.viewer.eye), tuple(cfg.viewer.lookat))
        except Exception as exc:  # noqa: BLE001
            print(f"[render] fixed camera set failed ({exc.__class__.__name__}: {exc})", flush=True)
        return
    try:
        pos = u.scene[args.track_asset].data.root_pos_w
        # Isaac Lab 3 returns asset data as a warp-backed ProxyArray, not a Tensor. `[0].tolist()`
        # on it does not reliably give a 3-vector, so the target was built from whatever numbers
        # came back and the camera was aimed into empty sky -- every recorded frame after the first
        # was blank pale blue, and the bare `except: pass` below swallowed it without a word. Coerce
        # to torch first, then take exactly three floats.
        pos = getattr(pos, "torch", pos)
        c = [float(v) for v in pos[0][:3]]
        target = [c[0], c[1], c[2] + 0.024]     # mid-height of a stacked pair above the base cube
        offset = [e - l for e, l in zip(cfg.viewer.eye, cfg.viewer.lookat)]
        eye = [target[i] + offset[i] for i in range(3)]
        # A camera aimed somewhere impossible is worse than no tracking: it silently destroys the
        # recording. Sanity-check against the cell the reset banks actually occupy before committing.
        if all(abs(v) < 5.0 for v in target + eye):
            u.sim.set_camera_view(eye, target)
        else:
            print(f"[render] refusing camera aim at {target} (outside the workspace)", flush=True)
    except Exception as exc:  # noqa: BLE001 -- aiming a camera must never kill an hour-long recording
        # Reported, not swallowed. Silence here cost a full set of blank clips.
        print(f"[render] camera aim failed ({exc.__class__.__name__}: {exc}); keeping preset view",
              flush=True)


episodes = []
ep_start, ever = 0, False
aim_camera()
# ONE THROWAWAY RENDER. The viewport's rgb annotator returns a black image the first time it is
# attached, so without this the recording's frame 0 is black -- which shows up as an empty first
# tile on every contact sheet cut from it, once per sheet, forever.
try:
    u.render()
except Exception as exc:  # noqa: BLE001 -- a warm-up failure must not lose the recording
    print(f"[render] warm-up render failed ({exc.__class__.__name__}: {exc})")
with torch.inference_mode():
    for step in range(total_steps):
        obs, _, dones, _ = env.step(policy(obs))
        if isinstance(obs, tuple):
            obs = obs[0]
        # Every frame, not just at episode boundaries: the camera does not stay where it is put.
        aim_camera()
        frame = u.render()
        if frame is not None:
            if ff is None:
                ff = start_ffmpeg(frame)
            ff.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        succ_now = bool(ctx.success[0])
        ever = ever or succ_now
        # OFF-BY-ONE, FOUND BY LOOKING AT FRAMES. ManagerBasedRLEnv.step() resets a done env inside
        # the same call, and the frame is rendered AFTER step(), so the frame captured on the done
        # step already shows the NEXT episode's reset state. Attributing it to the ending episode put
        # a frame of the wrong episode at the end of every clip -- invisible in a 160-step recording
        # with padding, glaring on a 10-step one, where it was one tile in six showing an unstacked
        # cube at the end of an episode that had scored a success.
        # So the ending episode owns frames [ep_start, step-1] and the next one starts at `step`.
        done_now = bool(dones.flatten()[0])
        if done_now or step == total_steps - 1:
            extra = {}
            if insertive is not None:
                try:
                    obj_p = insertive.data.root_pos_w[0]
                    hand_p = u.scene["robot"].data.body_pos_w[0, _mount]
                    extra = {
                        "final_object_speed_mps": float(
                            torch.linalg.norm(insertive.data.root_lin_vel_w[0])),
                        "final_hand_to_object_m": float(torch.linalg.norm(hand_p - obj_p)),
                        # Height is what separates "carried and placed" from "dropped and left".
                        # The table top sits at 0.021 m for a 34 mm cube, so a final z near 0.021
                        # means the cube is on the table whatever else the episode did, and a final
                        # z near 0.055 means it is on top of the base cube.
                        "final_object_z_m": float(obj_p[2]),
                    }
                except Exception:  # noqa: BLE001
                    extra = {}
            episodes.append({
                "index": len(episodes),
                "frame_start": ep_start,
                "frame_end": step - 1 if done_now else step,
                "held_at_end": succ_now,
                "ever_during": ever,
                "final_position_error_m": float(ctx.xyz_distance[0]),
                "final_euler_xy_error_rad": float(ctx.euler_xy_distance[0]),
                # The last episode of a recording is cut off by the frame budget, not ended by the
                # environment. Its `held_at_end` is the INSTANTANEOUS predicate on an arbitrary
                # frame, not a termination -- so it is not evidence of anything, and on the
                # near-goal family (where the predicate holds 10-17% of the time) it is also
                # usually the LONGEST episode, which put it first in the clip cutter's ranking and
                # made it the hero clip. Flagged, and skipped downstream.
                "truncated": not done_now,
                **extra,
            })
            ep_start, ever = (step if done_now else step + 1), False
            aim_camera()

if ff is None:
    # The encoder is started by the first frame, so None means the viewport never returned one --
    # no video exists and saying so beats an AttributeError on the last line of an hour's work.
    print(f"[render] WARNING: no frames were rendered; {video_path} was never created")
    raise SystemExit(1)
ff.stdin.close()
if ff.wait() != 0:
    # An hour of rollout is worth an explicit failure rather than a truncated mp4 sitting beside a
    # sidecar that claims it is complete.
    print(f"[render] WARNING: ffmpeg exited {ff.returncode}; {video_path} may be truncated")
env.close()

n_end = sum(e["held_at_end"] for e in episodes)
n_ever = sum(e["ever_during"] for e in episodes)
meta = {
    "task": args.task,
    "checkpoint": args.checkpoint,
    "null_action": args.null_action,
    "motion_stiffness": args.motion_stiffness,
    "motion_damping": args.motion_damping,
    "grip_bias": args.grip_bias,
    "reset_type": args.reset_type,
    "fps_steps_per_second": 1.0 / (cfg.sim.dt * cfg.decimation),
    "steps_per_episode": steps_per_ep,
    "episodes": episodes,
    "summary": {
        "n": len(episodes),
        "held_at_end": n_end,
        "ever_during": n_ever,
    },
}
(out_dir / "episodes.json").write_text(json.dumps(meta, indent=2))
print(f"\n[RENDER_RESULT] {len(episodes)} episodes | held-at-end {n_end} | ever-during {n_ever}")
print(f"[RENDER_RESULT] video + episodes.json in {out_dir}")
for e in episodes:
    mark = "SUCCESS" if e["held_at_end"] else ("touched" if e["ever_during"] else "-")
    print(f"  ep{e['index']:>3} frames {e['frame_start']:>5}-{e['frame_end']:<5} {mark:>8}"
          f"  pos_err {e['final_position_error_m']*1000:6.1f} mm")
app.close()
