"""Render N cube-stacking environments at once into a single TILED video.

WHY THIS EXISTS ALONGSIDE render_rollout.py. That script records one env, so a report that wants to
show four successes needs four separate runs and four separate clips, and each run pays the whole
scene-build cost again. Stepping N envs together and tiling their views gives one artifact where
several successes are visible side by side, which is what a reviewer actually wants to look at.

WHICH CAMERA MECHANISM, AND WHY NOT TiledCamera
-----------------------------------------------
IsaacLab 0.54.4 ships `isaaclab.sensors.TiledCameraCfg`, and on paper it is the exact fit: one
render product, N sub-viewports, one batched read. It is NOT used here, because it is documented as
non-functional on this box. scripts_v2/tools/render_reset_states_viewport.py's module docstring
records the measurement: on the UR5e+DELTO scene, two 640x480 TiledCameras made `env.reset()` not
return in 20 minutes, and a single 320x240 TiledCamera reset in 0.6 s but then hung on the FIRST
camera-output read -- both hangs sitting right after the same carb `IMemoryBudgetManagerFactory`
warning, with the process burning CPU and ~3.6 GiB of GPU rather than deadlocked. That script exists
precisely because the viewport path was the one that then worked.

So this uses the viewport path, which is also the path render_rollout.py already records with.
`ManagerBasedRLEnv.render()` (manager_based_rl_env.py:243) lazily creates ONE
`rep.create.render_product` on `cfg.viewer.cam_prim_path` at `cfg.viewer.resolution` and reads a
single `rgb` annotator off it. There is exactly one such camera, so the N tiles are produced by
MOVING it: per rendered frame the viewport camera is teleported to each env's origin in turn via
`sim.set_camera_view` (simulation_context.py:392) and read, giving N images that numpy composes into
the grid. `cfg.viewer.origin_type` stays "world" so the ViewportCameraController does not fight the
teleports -- it re-aims the camera every render step only for the "asset_root"/"asset_body" origins.

THE COST OF THAT CHOICE IS HONEST: it is N renders per recorded frame instead of one, and RTX render
is the wall-clock bottleneck of the whole script. `--render_every` exists to buy it back by
recording every Kth control step rather than every step; the video's fps is divided to match, so
the clip still plays at real speed.

Camera placement matches render_rollout.py's defaults (eye 0.75 -0.55 0.45, lookat 0.45 0.0 0.12),
but interpreted PER ENV RELATIVE TO THAT ENV'S ORIGIN, since with num_envs>1 the envs are spread
across the world by `scene.env_origins`. For env 0, whose origin is the world origin, the framing is
byte-identical to render_rollout.py's.

RENDERING IS LOCAL-ONLY BY NECESSITY. The deployment guide requires an RTX-class GPU on driver
branch 580 for any camera-enabled run: A100/H100 have no graphics engine and segfault about a second
into startup, and driver 595 segfaults `rtx.scenedb` on Isaac 5.1.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

# numpy is imported ABOVE the Isaac imports on purpose: everything from here to `main()` is pure
# array work, and test_render_grid.py imports this module to exercise it. Anything that reaches
# `isaaclab.app` at module scope would make the helpers untestable without launching a simulator.
import numpy as np

# ---------------------------------------------------------------------------------------------
# Pure-numpy helpers. No Isaac, no PIL.
# ---------------------------------------------------------------------------------------------

GLYPH_W, GLYPH_H = 3, 5

# A 3x5 block font, drawn by array slicing. PIL would do this in one line, but it is not currently a
# dependency of this project's Isaac python and adding one to burn four characters into a frame is a
# bad trade -- a missing font/freetype build would fail at the END of an hour-long render.
_GLYPHS: dict[str, tuple[str, ...]] = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "A": ("111", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("111", "100", "100", "100", "111"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "111", "100", "111"),
    "F": ("111", "100", "111", "100", "100"),
    "G": ("111", "100", "101", "101", "111"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "J": ("001", "001", "001", "101", "111"),
    "K": ("101", "101", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("111", "101", "101", "101", "111"),
    "P": ("111", "101", "111", "100", "100"),
    "Q": ("111", "101", "101", "111", "011"),
    "R": ("111", "101", "111", "110", "101"),
    "S": ("111", "100", "111", "001", "111"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"),
    "Z": ("111", "001", "010", "100", "111"),
    " ": ("000", "000", "000", "000", "000"),
    "-": ("000", "000", "111", "000", "000"),
    ".": ("000", "000", "000", "000", "010"),
    ":": ("000", "010", "000", "010", "000"),
    "*": ("101", "010", "101", "000", "000"),
    "/": ("001", "001", "010", "100", "100"),
}


def text_size(text: str, scale: int = 2) -> tuple[int, int]:
    """Pixel (width, height) that :func:`draw_text` will occupy. One blank column between glyphs."""
    return len(text) * (GLYPH_W + 1) * scale, GLYPH_H * scale


def draw_text(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: int = 2,
    color: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] | None = None,
    pad: int = 1,
) -> np.ndarray:
    """Burn ``text`` into ``img`` in place at pixel (x, y), top-left origin.

    Every write is clipped to the image, so a label placed off the edge degrades to a partial label
    rather than raising -- a caption is never worth losing a frame over.
    """
    h, w = img.shape[:2]
    text = text.upper()
    advance = (GLYPH_W + 1) * scale
    if bg is not None:
        # A plain dark plate behind the glyphs: the render is a lit scene and white-on-white is
        # unreadable exactly on the bright frames a report is most likely to use.
        bx0, by0 = max(0, x - pad * scale), max(0, y - pad * scale)
        bx1 = min(w, x + advance * len(text) + pad * scale)
        by1 = min(h, y + GLYPH_H * scale + pad * scale)
        if bx1 > bx0 and by1 > by0:
            img[by0:by1, bx0:bx1] = bg
    for k, ch in enumerate(text):
        glyph = _GLYPHS.get(ch)
        if glyph is None:
            continue
        ox = x + k * advance
        for r, row in enumerate(glyph):
            for c, bit in enumerate(row):
                if bit != "1":
                    continue
                y0, x0 = y + r * scale, ox + c * scale
                y1, x1 = min(h, y0 + scale), min(w, x0 + scale)
                y0, x0 = max(0, y0), max(0, x0)
                if y1 > y0 and x1 > x0:
                    img[y0:y1, x0:x1] = color
    return img


def grid_shape(n: int) -> tuple[int, int]:
    """(rows, cols) for n tiles: square where it can be (4 -> 2x2, 9 -> 3x3), else widest-first."""
    if n < 1:
        raise ValueError(f"grid_shape needs at least one tile, got {n}")
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def compose_grid(
    tiles: list[np.ndarray | None],
    rows: int,
    cols: int,
    tile_h: int,
    tile_w: int,
) -> np.ndarray:
    """Lay tiles left-to-right, top-to-bottom into one uint8 RGB frame.

    A tile that is None, the wrong rank, or the wrong size is not an error: it leaves its cell black
    or is cropped into it. The renderer returns a zero-size buffer while it warms up and can return
    nothing at all, and one bad tile must not end a recording that has been running for an hour.
    """
    out = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        if i >= rows * cols:
            break
        if tile is None:
            continue
        a = np.asarray(tile)
        if a.ndim != 3 or a.shape[0] == 0 or a.shape[1] == 0 or a.shape[2] < 3:
            continue
        a = a[:tile_h, :tile_w, :3]
        r, c = divmod(i, cols)
        y0, x0 = r * tile_h, c * tile_w
        out[y0 : y0 + a.shape[0], x0 : x0 + a.shape[1]] = a.astype(np.uint8, copy=False)
    return out


def outline_tile(
    frame: np.ndarray,
    index: int,
    cols: int,
    tile_h: int,
    tile_w: int,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> np.ndarray:
    """Draw a border inside tile ``index``'s cell, in place.

    This is what makes the tiling readable at all: envs reset independently, so at any instant some
    tiles are succeeding and some are half a second into a fresh episode, and without a per-tile
    state marker the viewer cannot tell which is which.
    """
    r, c = divmod(index, cols)
    y0, x0 = r * tile_h, c * tile_w
    y1, x1 = y0 + tile_h, x0 + tile_w
    t = max(1, min(thickness, tile_h // 2, tile_w // 2))
    frame[y0 : y0 + t, x0:x1] = color
    frame[y1 - t : y1, x0:x1] = color
    frame[y0:y1, x0 : x0 + t] = color
    frame[y0:y1, x1 - t : x1] = color
    return frame


def pad_to_even(img: np.ndarray) -> np.ndarray:
    """yuv420p subsamples chroma 2x2 and libx264 refuses odd dimensions; pad rather than crop."""
    h, w = img.shape[:2]
    dh, dw = h % 2, w % 2
    if not (dh or dw):
        return img
    out = np.zeros((h + dh, w + dw, img.shape[2]), dtype=img.dtype)
    out[:h, :w] = img
    return out


MASK_ENV_LIMIT = 62  # runs are JSON ints; 1<<62 is the last value that survives int64 arithmetic


def coverage_runs(
    episodes_by_env: list[list[dict]],
    n_envs: int,
    n_frames: int,
) -> list[list[int]]:
    """Run-length encode, over frame index, which envs are mid-episode and which are succeeding.

    WHY RUN-LENGTH. The tiles do not line up: each env resets on its own clock, so a window where
    three tiles are simultaneously in a successful episode has to be searched for rather than
    assumed. That search needs a per-frame answer -- but writing one record per env per frame is
    n_envs * n_frames objects, which for 4 envs over a 60-episode recording is ~38k records and
    several MB of JSON describing something that only changes a few hundred times. Episodes are
    contiguous intervals, so the per-frame signal is piecewise constant and the encoding is exact,
    not lossy: one run per change, roughly 2 * (total episodes) runs.

    Each run is ``[frame_start, frame_end, active_mask, held_mask, ever_mask]`` with bit i meaning
    env i. ``held_mask`` marks envs inside an episode that ENDED with the success predicate holding;
    ``ever_mask``, one that touched success at any point. Both frame bounds are inclusive.
    """
    if n_envs > MASK_ENV_LIMIT:
        raise ValueError(f"coverage bitmasks support at most {MASK_ENV_LIMIT} envs, got {n_envs}")
    if n_frames <= 0:
        return []
    active = np.zeros(n_frames, dtype=np.int64)
    held = np.zeros(n_frames, dtype=np.int64)
    ever = np.zeros(n_frames, dtype=np.int64)
    for i, eps in enumerate(episodes_by_env[:n_envs]):
        bit = np.int64(1) << np.int64(i)
        for e in eps:
            s = max(0, int(e["frame_start"]))
            t = min(n_frames - 1, int(e["frame_end"]))
            if t < s:
                continue  # episode shorter than one recorded frame at this --render_every
            active[s : t + 1] |= bit
            if e.get("held_at_end"):
                held[s : t + 1] |= bit
            if e.get("ever_during"):
                ever[s : t + 1] |= bit

    runs: list[list[int]] = []
    prev: tuple[int, int, int] | None = None
    for f in range(n_frames):
        key = (int(active[f]), int(held[f]), int(ever[f]))
        if key != prev:
            runs.append([f, f, key[0], key[1], key[2]])
            prev = key
        else:
            runs[-1][1] = f
    return runs


def best_simultaneous_window(runs: list[list[int]], mask_field: int = 3) -> dict:
    """Report the run in which the most tiles are simultaneously succeeding.

    Cheap summary so the run's stdout answers "is there a shot worth cutting?" without a second
    tool. ``mask_field`` selects held_mask (3) or ever_mask (4).
    """
    best = {"frame_start": 0, "frame_end": 0, "n_envs": 0, "mask": 0}
    for run in runs:
        mask = run[mask_field]
        n = int(bin(mask).count("1"))
        span = run[1] - run[0]
        if n > best["n_envs"] or (n == best["n_envs"] and n > 0 and span > best["frame_end"] - best["frame_start"]):
            best = {"frame_start": run[0], "frame_end": run[1], "n_envs": n, "mask": mask}
    return best


# ---------------------------------------------------------------------------------------------
# Everything below launches Isaac. Kept inside main() so the helpers above stay importable.
# ---------------------------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--episodes", type=int, default=12,
                   help="episodes PER ENV; the recording is this many episode-lengths long and every "
                        "env runs for all of it, so the run yields num_envs * episodes episodes.")
    p.add_argument("--out", type=str, default="artifacts/grid")
    p.add_argument("--insertive", default="cube34")
    p.add_argument("--receptive", default="cube34")
    p.add_argument("--dataset_dir", default="./Datasets_ur10e_delto/OmniReset")
    p.add_argument("--collision_stack_mb", type=int, default=None)
    p.add_argument("--eye", type=float, nargs=3, default=[0.75, -0.55, 0.45])
    p.add_argument("--lookat", type=float, nargs=3, default=[0.45, 0.0, 0.12])
    p.add_argument("--width", type=int, default=640, help="width of EACH TILE, not of the video")
    p.add_argument("--height", type=int, default=360, help="height of EACH TILE, not of the video")
    p.add_argument("--render_every", type=int, default=1,
                   help="record every Kth control step. The viewport camera is moved and read once "
                        "PER ENV per recorded frame, so render cost scales with num_envs and this is "
                        "the knob that buys it back; the video fps is divided to match so playback "
                        "stays real-time.")
    p.add_argument("--settle_renders", type=int, default=1,
                   help="extra discarded renders after each camera teleport. NOT MEASURED ON A GPU "
                        "YET: the rgb annotator is read straight after sim.render() and RTX carries "
                        "temporal history across frames, so a teleported camera can plausibly hand "
                        "back the previous env's image or a half-converged one. One throwaway render "
                        "is the cheap guard; set 0 if the frames prove clean without it.")
    p.add_argument("--label_scale", type=int, default=2, help="block-font pixel scale for tile labels")
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

    from isaaclab.app import AppLauncher  # noqa: E402  (import here so the helpers stay Isaac-free)

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    import gymnasium as gym  # noqa: E402
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

    n_envs = args.num_envs
    rows, cols = grid_shape(n_envs)
    tile_w, tile_h = args.width, args.height

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=n_envs)
    v = cfg.variants
    cfg.scene.insertive_object = v["scene.insertive_object"][args.insertive]
    cfg.scene.receptive_object = v["scene.receptive_object"][args.receptive]
    for _, term in vars(cfg.events).items():
        prm = getattr(term, "params", None)
        if isinstance(prm, dict) and "dataset_dir" in prm:
            prm["dataset_dir"] = args.dataset_dir
    if args.collision_stack_mb is not None:
        cfg.sim.physx.gpu_collision_stack_size = args.collision_stack_mb * 1024 * 1024

    _rst = getattr(cfg.events, "reset_from_reset_states", None)
    if _rst is not None:
        _rst.params["reset_types"] = [args.reset_type]
        _rst.params["probs"] = [1.0]

    # The viewport is the ONE camera every tile is rendered through, so its resolution is the TILE
    # size. origin_type stays "world" because the camera is re-aimed by hand every tile; the
    # "asset_root"/"asset_body" origins install a per-render callback that would undo the teleports.
    cfg.viewer.eye = tuple(args.eye)
    cfg.viewer.lookat = tuple(args.lookat)
    cfg.viewer.resolution = (tile_w, tile_h)
    cfg.viewer.origin_type = "world"

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")

    steps_per_ep = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
    total_steps = steps_per_ep * args.episodes
    control_hz = 1.0 / (cfg.sim.dt * cfg.decimation)
    video_fps = control_hz / max(1, args.render_every)
    grid_w, grid_h = cols * tile_w, rows * tile_h
    if grid_w % 2 or grid_h % 2:
        grid_w += grid_w % 2
        grid_h += grid_h % 2
    print(f"[grid] {n_envs} envs in {rows}x{cols} of {tile_w}x{tile_h} -> {grid_w}x{grid_h}")
    print(f"[grid] {steps_per_ep} steps/episode x {args.episodes} = {total_steps} steps, "
          f"1 frame per {args.render_every} step(s) @ {video_fps:.2f} fps")
    print(f"[grid] {n_envs * (1 + args.settle_renders)} viewport renders per recorded frame")

    # FRAMES ARE STREAMED TO ffmpeg, NOT BUFFERED.
    # gymnasium's RecordVideo appends every frame to a Python list and encodes once at close. At
    # 1280x720x3 that is 2.76 MB per frame, so a 60-episode recording (9600 frames) needs 26 GB of RAM
    # on a 30 GB machine -- it thrashes swap for an hour and then dies with nothing written, which is
    # the worst possible failure mode for a job whose entire output arrives at the end. Piping raw
    # frames straight into ffmpeg holds one frame at a time and puts no ceiling on recording length.
    # A 2x2 grid is four times the pixels of one tile, so the ceiling matters more here, not less.
    video_path = out_dir / "grid.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{grid_w}x{grid_h}",
         "-r", f"{video_fps:.4f}", "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(video_path)],
        stdin=subprocess.PIPE,
    )

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    ctx = u.reward_manager.get_term_cfg("progress_context").func
    cmd = u.command_manager.get_term("task_command")
    print(f"[grid] thresholds pos={cmd.success_position_threshold} rot={cmd.success_orientation_threshold}")

    # Per-env camera poses, resolved once. --eye/--lookat are RELATIVE TO THE ENV ORIGIN because
    # num_envs>1 spreads the envs across the world; env 0 sits at the world origin, so its framing is
    # identical to render_rollout.py's. Plain floats, not np.float64 -- set_camera_view crosses into
    # the USD bindings and is not worth handing numpy scalars.
    origins = u.scene.env_origins.detach().cpu().numpy()
    cam_poses = [
        (tuple(float(x) for x in origins[i] + np.asarray(args.eye, dtype=float)),
         tuple(float(x) for x in origins[i] + np.asarray(args.lookat, dtype=float)))
        for i in range(n_envs)
    ]
    cam_prim = cfg.viewer.cam_prim_path

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
        print(f"[grid] physical-state probes unavailable ({exc.__class__.__name__}: {exc}); "
              f"recording position only")
        insertive, _mount = None, None

    OK = (60, 220, 90)      # success predicate holds right now
    TOUCHED = (235, 175, 40)  # held earlier in this episode, not now
    IDLE = (70, 70, 78)

    episodes_by_env: list[list[dict]] = [[] for _ in range(n_envs)]
    ep_step_start = np.zeros(n_envs, dtype=np.int64)
    ep_frame_start = np.zeros(n_envs, dtype=np.int64)
    ever = np.zeros(n_envs, dtype=bool)
    frame_idx = 0
    aborted = None

    def close_episode(i: int, step: int, succ_now: bool) -> None:
        extra: dict[str, float] = {}
        if insertive is not None:
            try:
                obj_p = insertive.data.root_pos_w[i]
                hand_p = u.scene["robot"].data.body_pos_w[i, _mount]
                extra = {
                    "final_object_speed_mps": float(torch.linalg.norm(insertive.data.root_lin_vel_w[i])),
                    "final_hand_to_object_m": float(torch.linalg.norm(hand_p - obj_p)),
                }
            except Exception:  # noqa: BLE001
                extra = {}
        episodes_by_env[i].append({
            "env": i,
            "index": len(episodes_by_env[i]),
            "frame_start": int(ep_frame_start[i]),
            "frame_end": max(int(ep_frame_start[i]), frame_idx - 1),
            "step_start": int(ep_step_start[i]),
            "step_end": int(step),
            "held_at_end": bool(succ_now),
            "ever_during": bool(ever[i]),
            "final_position_error_m": float(ctx.xyz_distance[i]),
            "final_euler_xy_error_rad": float(ctx.euler_xy_distance[i]),
            **extra,
        })
        ep_step_start[i] = step + 1
        ep_frame_start[i] = frame_idx
        ever[i] = False

    # The whole loop is guarded: an hour of rendering must survive an exception by flushing ffmpeg and
    # writing the sidecar for what was already recorded, rather than leaving an unfinalised mp4.
    try:
        with torch.inference_mode():
            for step in range(total_steps):
                obs, _, dones, _ = env.step(policy(obs))
                if isinstance(obs, tuple):
                    obs = obs[0]

                succ = ctx.success.detach().cpu().numpy().astype(bool)
                ever |= succ
                done = dones.detach().cpu().numpy().reshape(-1).astype(bool)
                last = step == total_steps - 1

                if step % args.render_every == 0 or last:
                    tiles: list[np.ndarray | None] = []
                    for i in range(n_envs):
                        u.sim.set_camera_view(cam_poses[i][0], cam_poses[i][1], cam_prim)
                        for _ in range(args.settle_renders):
                            u.render()
                        tiles.append(u.render())
                    frame = compose_grid(tiles, rows, cols, tile_h, tile_w)
                    for i in range(n_envs):
                        color = OK if succ[i] else (TOUCHED if ever[i] else IDLE)
                        outline_tile(frame, i, cols, tile_h, tile_w, color)
                        mm = float(ctx.xyz_distance[i]) * 1000.0
                        label = f"E{i} {min(mm, 999):3.0f}MM" + (" OK" if succ[i] else "")
                        r, c = divmod(i, cols)
                        draw_text(frame, label, c * tile_w + 6, r * tile_h + 6,
                                  scale=args.label_scale, color=color, bg=(16, 16, 20))
                    ff.stdin.write(np.ascontiguousarray(pad_to_even(frame), dtype=np.uint8).tobytes())
                    frame_idx += 1

                for i in range(n_envs):
                    if done[i] or last:
                        close_episode(i, step, bool(succ[i]))
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
        aborted = f"{exc.__class__.__name__}: {exc}"
        print(f"\n[grid] ABORTED after {frame_idx} frames -- {aborted}\n"
              f"[grid] flushing what was recorded rather than discarding it")

    # If ffmpeg is what died, the write above already raised into the guard and these will raise
    # again -- the sidecar below is the surviving record and must still be written.
    try:
        ff.stdin.close()
        ff.wait()
    except Exception as exc:  # noqa: BLE001
        print(f"[grid] ffmpeg did not finalise cleanly ({exc.__class__.__name__}: {exc})")
    env.close()

    flat = sorted((e for eps in episodes_by_env for e in eps), key=lambda e: (e["frame_start"], e["env"]))
    runs = coverage_runs(episodes_by_env, n_envs, frame_idx)
    best_held = best_simultaneous_window(runs, mask_field=3)
    best_ever = best_simultaneous_window(runs, mask_field=4)

    n_end = sum(e["held_at_end"] for e in flat)
    n_ever = sum(e["ever_during"] for e in flat)
    meta = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "reset_type": args.reset_type,
        "num_envs": n_envs,
        "grid_rows_cols": [rows, cols],
        "tile_width_height": [tile_w, tile_h],
        "video_width_height": [grid_w, grid_h],
        "render_every": args.render_every,
        "control_hz": control_hz,
        # Named to match render_rollout.py's sidecar so cut_successes.py reads this file unchanged.
        # It is the VIDEO's fps, which is control_hz/render_every -- frame_start/frame_end below are
        # indices into the mp4, not into the control loop (step_start/step_end are those).
        "fps_steps_per_second": video_fps,
        "steps_per_episode": steps_per_ep,
        "n_frames": frame_idx,
        "aborted": aborted,
        "episodes": flat,
        "episodes_by_env": episodes_by_env,
        "coverage": {
            "encoding": "run-length over video frame index; run = "
                        "[frame_start, frame_end, active_mask, held_mask, ever_mask], "
                        "bit i = env i, both frame bounds inclusive",
            "runs": runs,
        },
        "best_simultaneous": {"held_at_end": best_held, "ever_during": best_ever},
        "summary": {"n": len(flat), "held_at_end": n_end, "ever_during": n_ever},
    }
    (out_dir / "episodes.json").write_text(json.dumps(meta, indent=2))

    print(f"\n[RENDER_RESULT] {len(flat)} episodes across {n_envs} envs | "
          f"held-at-end {n_end} | ever-during {n_ever} | {frame_idx} frames")
    print(f"[RENDER_RESULT] best simultaneous held-at-end: {best_held['n_envs']} envs "
          f"over frames {best_held['frame_start']}-{best_held['frame_end']}")
    print(f"[RENDER_RESULT] best simultaneous ever-during: {best_ever['n_envs']} envs "
          f"over frames {best_ever['frame_start']}-{best_ever['frame_end']}")
    print(f"[RENDER_RESULT] video + episodes.json in {out_dir}")
    for i in range(n_envs):
        eps = episodes_by_env[i]
        good = sum(e["held_at_end"] for e in eps)
        print(f"  env{i}: {len(eps):>3} episodes, {good:>3} held-at-end")
        for e in eps:
            mark = "SUCCESS" if e["held_at_end"] else ("touched" if e["ever_during"] else "-")
            print(f"    ep{e['index']:>3} frames {e['frame_start']:>5}-{e['frame_end']:<5} {mark:>8}"
                  f"  pos_err {e['final_position_error_m']*1000:6.1f} mm")
    app.close()


if __name__ == "__main__":
    main()
