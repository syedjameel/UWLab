"""Find the exact moment rendering dies, instead of guessing at camera placement.

Evidence so far, all measured:
  * viewport (env.render): frame 0 correct, every later frame blank sky -- unchanged by tracking on,
    tracking off, or re-asserting set_camera_view every frame
  * CameraCfg sensor: blank from the very first frame
  * Isaac 5.1: cannot render at all on this box's 610 driver (crashes in libomni.hydra.rtx)

Both renderers are fine before stepping and dead after, which points at env.step() rather than at
where the camera is. So: dump the camera's real world pose, and save individual frames at reset and
after 1 / 10 / 50 steps. Whichever of those is the first blank one localises the failure.

The blankness test here is deliberately strict. The previous one (std > 6) passed a smooth sky
gradient and reported "160/160 frames have real content" for a clip that was empty end to end --
a false pass is worse than no check. A rendered workspace has strong local contrast (edges of the
table, the cubes, the hand); sky does not. So measure the mean absolute LAPLACIAN, not the spread.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="OmniReset-UR10eDelto-CubeStackTwoFingerGripCarry-RelCartesianOSC-State-v0")
p.add_argument("--out", default="/home/mattephi/Work/Rubitek/cube-stack/results/film_diag")
p.add_argument("--dataset_dir", default="./Datasets_ur10e_delto/OmniReset_pinch2f")
args, _ = p.parse_known_args()
args.enable_cameras = True
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

try:
    import uwlab_is6_bridge

    uwlab_is6_bridge.apply()
except ImportError:
    pass

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pathlib import Path  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_t = lambda x: getattr(x, "torch", x)  # noqa: E731
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
v = cfg.variants
cfg.scene.insertive_object = v["scene.insertive_object"]["cube34"]
cfg.scene.receptive_object = v["scene.receptive_object"]["cube34"]
cfg.scene.insertive_object = cfg.scene.insertive_object.replace(
    spawn=cfg.scene.insertive_object.spawn.replace(activate_contact_sensors=True)
)
for _, term in vars(cfg.events).items():
    prm = getattr(term, "params", None)
    if isinstance(prm, dict) and "dataset_dir" in prm:
        prm["dataset_dir"] = args.dataset_dir
_rst = getattr(cfg.events, "reset_from_reset_states", None)
if _rst is not None:
    _rst.params["reset_types"] = ["ObjectAnywhereEEGrasped"]
    _rst.params["probs"] = [1.0]

# prim_path under {ENV_REGEX_NS}, NOT /World. InteractiveScene resolves sensor prims inside the
# per-environment namespace; a /World path produced a Camera object with no backing prims, so
# set_world_poses_from_view wrote into nothing (raising nothing) and cam.data.pos_w read back
# [0,0,0] forever. That, not the viewport, is why every frame was empty sky.
#
# The pose is baked into OffsetCfg rather than applied at runtime, so it cannot depend on a call
# that silently no-ops. convention="world" is +X forward / +Z up, and rot is (x, y, z, w). The
# quaternion below is the look-at rotation from eye (0.78, -0.28, 0.16) to target (0.48, 0.01,
# 0.045) -- the measured closeup framing: ~0.42 m out, ~15 degrees elevation, on the far side of
# the workspace from the arm base.
cfg.scene.film_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/film_cam",
    update_period=0.0,
    height=720,
    width=1280,
    data_types=["rgb"],
    offset=CameraCfg.OffsetCfg(
        pos=(0.78, -0.28, 0.16),
        rot=(-0.1243, 0.0502, 0.9186, 0.3715),
        convention="world",
    ),
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 30.0)
    ),
)
print(f"DIAG sim.render_interval={getattr(cfg.sim, 'render_interval', 'n/a')} decimation={cfg.decimation}")

env = gym.make(args.task, cfg=cfg).unwrapped
cam = env.scene["film_cam"]
dev = env.device


def detail(img) -> float:
    """Mean |Laplacian| -- high on edges, ~0 on a smooth sky gradient."""
    g = img.astype(np.float32).mean(axis=2)
    lap = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    return float(lap.mean())


EYE = None
TGT = None


def grab(tag: str) -> None:
    # RE-ASSERT the pose on every grab. The single call after env.reset() left cam.data.pos_w at
    # [0,0,0] for the entire run: the camera was never aimed at all, which is the real reason every
    # frame came back blank sky rather than anything to do with the viewport. set_world_poses_from_view
    # documents that it needs the camera prim initialised, and at reset time it evidently is not --
    # yet it raises nothing. Calling it again once stepping has started shows whether it then sticks.
    if EYE is not None:
        try:
            cam.set_world_poses_from_view(EYE, TGT)
        except Exception as exc:  # noqa: BLE001
            print(f"DIAG aim failed at {tag}: {exc.__class__.__name__}: {exc}", flush=True)
    rgb = _t(cam.data.output["rgb"])
    if rgb is None or rgb.shape[0] == 0:
        print(f"DIAG {tag}: no rgb")
        return
    img = rgb[0].detach().cpu().numpy()
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = img.astype(np.uint8)
    pos = _t(cam.data.pos_w)[0].tolist()
    q = _t(cam.data.quat_w_world)[0].tolist() if hasattr(cam.data, "quat_w_world") else []
    print(f"DIAG {tag}: detail={detail(img):7.3f} mean={img.mean():6.1f} "
          f"cam_pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] quat={[round(x, 3) for x in q]}", flush=True)
    try:
        from PIL import Image

        Image.fromarray(img).save(out / f"{tag}.png")
    except Exception as exc:  # noqa: BLE001
        print(f"DIAG {tag}: save failed ({exc.__class__.__name__})")


env.reset()
EYE = torch.tensor([[0.78, -0.28, 0.16]], device=dev, dtype=torch.float32)
TGT = torch.tensor([[0.48, 0.01, 0.045]], device=dev, dtype=torch.float32)
cam.set_world_poses_from_view(EYE, TGT)
cube = _t(env.scene["insertive_object"].data.root_pos_w)[0].tolist()
print(f"DIAG cube_at=[{cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f}]")

# Render a few times with no stepping at all: the camera annotator famously returns an empty first
# frame, so "blank at reset" could just be that.
for i in range(3):
    env.sim.render()
    grab(f"reset_render{i}")

zero = torch.zeros((1, env.action_manager.total_action_dim), device=dev)
with torch.inference_mode():
    for step in range(1, 51):
        env.step(zero)
        if step in (1, 2, 10, 50):
            grab(f"step{step:03d}")
            # Does an explicit render revive it? If the post-step frame is blank but this one is not,
            # the env simply is not rendering on step and the fix is to render explicitly.
            env.sim.render()
            grab(f"step{step:03d}_afterrender")

print("DIAG done")
env.close()
app.close()
