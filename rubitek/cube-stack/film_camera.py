"""Record a rollout through a CameraCfg SENSOR instead of the viewport.

WHY A SECOND RENDERER EXISTS. `render_rollout.py` records `env.render()`, which reads the Kit
VIEWPORT. Under Isaac Sim 6 / Isaac Lab 3 that viewport produces exactly ONE good frame per run and
then returns empty sky for every frame after it. Measured, not guessed:

  * with camera tracking on  -> frame 0 correct, frames 1..159 blank
  * with tracking off        -> identical
  * re-asserting set_camera_view on EVERY frame -> identical, and the call reports no error

So it is not the camera pose. The scene simply stops being drawn into that viewport. The Isaac 5.1
environment, which rendered these clips correctly before, now dies in libomni.hydra.rtx on this
box's 610 driver, so downgrading is not available either.

A `CameraCfg` sensor is the supported headless capture path in Isaac Lab 3: it owns its own render
product and annotator rather than borrowing the interactive viewport, and its output arrives as a
tensor through `scene["film_cam"].data.output["rgb"]`.

Null action means "hold the current pose" for the arm and "hold the latched grasp" for the hand, so
a null-action clip shows what the RESET did, with no policy to blame for a dropped cube.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="OmniReset-UR10eDelto-CubeStackTwoFingerGripCarry-RelCartesianOSC-State-v0")
p.add_argument("--checkpoint", default=None, help="rsl_rl checkpoint; omit for --null_action")
p.add_argument("--null_action", action="store_true")
p.add_argument("--episodes", type=int, default=2)
p.add_argument("--reset_type", default="ObjectAnywhereEEGrasped")
p.add_argument("--dataset_dir", default="./Datasets_ur10e_delto/OmniReset_pinch2f")
p.add_argument("--out", default="artifacts/film")
p.add_argument("--width", type=int, default=1280)
p.add_argument("--height", type=int, default=720)
p.add_argument("--eye", type=float, nargs=3, default=[0.78, -0.28, 0.16])
p.add_argument("--lookat", type=float, nargs=3, default=[0.48, 0.01, 0.045])
p.add_argument("--track", action="store_true", default=True, help="re-aim at the carried cube each episode")
p.add_argument("--no-track", dest="track", action="store_false")
args, _ = p.parse_known_args()
args.enable_cameras = True
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

try:
    import uwlab_is6_bridge  # noqa: E402

    uwlab_is6_bridge.apply()
except ImportError:
    pass

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_t = lambda x: getattr(x, "torch", x)  # noqa: E731

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
v = cfg.variants
cfg.scene.insertive_object = v["scene.insertive_object"]["cube34"]
cfg.scene.receptive_object = v["scene.receptive_object"]["cube34"]
# The task's __post_init__ rebuilds this spawn with contact sensors on; assigning the raw variant
# over the top drops the flag and the object's ContactSensor then fails to initialise.
cfg.scene.insertive_object = cfg.scene.insertive_object.replace(
    spawn=cfg.scene.insertive_object.spawn.replace(activate_contact_sensors=True)
)
for _, term in vars(cfg.events).items():
    prm = getattr(term, "params", None)
    if isinstance(prm, dict) and "dataset_dir" in prm:
        prm["dataset_dir"] = args.dataset_dir
_rst = getattr(cfg.events, "reset_from_reset_states", None)
if _rst is not None:
    _rst.params["reset_types"] = [args.reset_type]
    _rst.params["probs"] = [1.0]

def lookat_quat(eye, target):
    """Rotation taking +X to the eye->target direction, +Z up, as (x, y, z, w).

    Needed because the pose is baked into OffsetCfg instead of applied at runtime. The runtime call
    (`set_world_poses_from_view`) silently wrote into nothing and left the camera at the world origin
    staring at sky for every frame of every clip rendered tonight -- it raises no error, and
    `cam.data.pos_w` simply keeps reading [0,0,0]. A pose fixed at spawn cannot fail that way.
    """
    e = np.asarray(eye, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    x = t - e
    n = np.linalg.norm(x)
    if n < 1e-9:
        raise ValueError("eye and lookat coincide")
    x = x / n
    up = np.array([0.0, 0.0, 1.0])
    z = up - np.dot(up, x) * x
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w < 1e-6:  # degenerate; fall back to identity rather than dividing by ~0
        return (0.0, 0.0, 0.0, 1.0)
    return (
        float((R[2, 1] - R[1, 2]) / (4 * w)),
        float((R[0, 2] - R[2, 0]) / (4 * w)),
        float((R[1, 0] - R[0, 1]) / (4 * w)),
        float(w),
    )


# prim_path under {ENV_REGEX_NS}, NOT /World: InteractiveScene resolves sensor prims inside the
# per-environment namespace, and a /World path yields a Camera with no backing prims -- writes go
# nowhere and reads return zeros, with nothing raised.
_rot = lookat_quat(args.eye, args.lookat)
print(f"[film] camera at {args.eye} looking at {args.lookat}, rot(xyzw)={[round(v, 4) for v in _rot]}")
cfg.scene.film_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/film_cam",
    update_period=0.0,
    height=args.height,
    width=args.width,
    data_types=["rgb"],
    offset=CameraCfg.OffsetCfg(pos=tuple(args.eye), rot=_rot, convention="world"),
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 30.0)
    ),
)

env = gym.make(args.task, cfg=cfg).unwrapped
cam = env.scene["film_cam"]
dev = env.device


def aim(target=None):
    """Point the sensor. Unlike the viewport camera, this one stays where it is put."""
    eye = torch.tensor([args.eye], device=dev, dtype=torch.float32)
    tgt = torch.tensor([args.lookat if target is None else target], device=dev, dtype=torch.float32)
    if target is not None:
        offset = [e - l for e, l in zip(args.eye, args.lookat)]
        eye = torch.tensor([[target[i] + offset[i] for i in range(3)]], device=dev, dtype=torch.float32)
    cam.set_world_poses_from_view(eye, tgt)


def track_target():
    if not args.track:
        return None
    try:
        pos = _t(env.scene["insertive_object"].data.root_pos_w)
        c = [float(x) for x in pos[0][:3]]
        return [c[0], c[1], c[2] + 0.024]
    except Exception as exc:  # noqa: BLE001
        print(f"[film] track failed ({exc.__class__.__name__}: {exc})", flush=True)
        return None


steps_per_ep = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
total = steps_per_ep * args.episodes
fps = 1.0 / (cfg.sim.dt * cfg.decimation)
print(f"[film] {steps_per_ep} steps/episode x {args.episodes} episodes = {total} frames @ {fps:.1f} fps")

n_act = env.action_manager.total_action_dim
zero = torch.zeros((1, n_act), device=dev)
policy = None
if not args.null_action:
    if not args.checkpoint:
        print("[film] no checkpoint and no --null_action; nothing to roll out")
        raise SystemExit(2)
    # Rebuild the actor MLP straight from the state dict. Going through OnPolicyRunner drags in the
    # fork's PPO signature and dies on keys it does not accept; the actor is all that is needed to
    # act, and this cannot break when the trainer's config schema moves.
    sd = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    msd = sd["model_state_dict"] if "model_state_dict" in sd else sd
    ws = [(k, x) for k, x in msd.items() if k.startswith("actor.") and k.endswith(".weight")]
    bs = {k.rsplit(".", 1)[0]: x for k, x in msd.items() if k.startswith("actor.") and k.endswith(".bias")}
    layers, prev = [], None
    for i, (k, w) in enumerate(ws):
        lin = torch.nn.Linear(w.shape[1], w.shape[0]).to(dev)
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(bs[k.rsplit(".", 1)[0]])
        if i:
            layers.append(torch.nn.ELU())
        layers.append(lin)
        prev = w.shape[0]
    actor = torch.nn.Sequential(*layers).eval()
    print(f"[film] actor rebuilt from {args.checkpoint}: {len(ws)} layers, out {prev}")

    def policy(obs):  # noqa: D401
        return actor(obs)[:, :n_act]

ff = None


def start_ffmpeg(frame):
    """Encoder geometry comes from the FIRST FRAME, never from --width/--height.

    Declaring it cost a full set of sheared, unusable clips once already: the viewport handed back
    480x270 RGBA while ffmpeg had been told 960x540 rgb24, and it silently wrote garbage.
    """
    h, w = frame.shape[0], frame.shape[1]
    ch = frame.shape[2] if frame.ndim == 3 else 1
    pix = {1: "gray", 3: "rgb24", 4: "rgba"}.get(ch)
    if pix is None:
        raise RuntimeError(f"[film] unsupported frame shape {frame.shape}")
    print(f"[film] encoding {w}x{h}x{ch} as {pix}", flush=True)
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", pix,
         "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
         "-movflags", "frag_keyframe+empty_moov+default_base_moof", "-g", "30",
         str(out_dir / "rollout.mp4")],
        stdin=subprocess.PIPE,
    )


obs, _ = env.reset()
aim(track_target())
nonblank = 0
with torch.inference_mode():
    for step in range(total):
        act = zero if policy is None else policy(obs["policy"] if isinstance(obs, dict) else obs)
        obs, _, term, trunc, _ = env.step(act)
        rgb = cam.data.output["rgb"]
        rgb = _t(rgb)
        if rgb is not None and rgb.shape[0] > 0:
            frame = rgb[0].detach().cpu().numpy().astype(np.uint8)
            if ff is None:
                ff = start_ffmpeg(frame)
            ff.stdin.write(np.ascontiguousarray(frame).tobytes())
            # Count frames that actually SHOW something. The previous test (std > 6) certified 160
            # consecutive blank frames as "real content", because a smooth sky gradient has plenty of
            # spread -- a check that passes when it shouldn't is worse than no check. A rendered
            # workspace has hard edges (table rim, cubes, fingers); sky has none. So measure local
            # contrast via the mean |Laplacian|, which is ~0.13 on the blank frames measured tonight.
            g = frame.astype(np.float32).mean(axis=2)
            lap = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
            if float(lap.mean()) > 1.5:
                nonblank += 1
        done = bool(_t(term).flatten()[0]) or bool(_t(trunc).flatten()[0])
        if done:
            aim(track_target())

if ff is None:
    print("[film] WARNING: camera never produced a frame; no video written")
    raise SystemExit(1)
ff.stdin.close()
ff.wait()
print(f"[film] wrote {out_dir / 'rollout.mp4'}; {nonblank}/{total} frames have real content")
env.close()
app.close()
