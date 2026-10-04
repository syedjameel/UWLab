"""Lay one episode out as a contact sheet, so it can actually be LOOKED at.

WHY THIS EXISTS. Every number in this project so far is aggregate: a success rate, a median error, a
histogram. Aggregates are exactly the wrong instrument for the questions that matter most right now
-- is the hand gripping the cube or caging it? does the cube sit on the other cube or hover beside
it? does the arm approach or thrash? -- and those questions are answered in one glance at the
frames.

A video cannot be looked at in one glance. This samples an episode at even intervals, tiles the
frames into a single image with the step number burned into each, and writes a PNG. One file, one
look, the whole motion.

Frames are labelled with the step index and, when episodes.json is available, the episode's outcome,
so a sheet can never be mistaken for a different episode than the one it came from.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

# A 3x5 block font, drawn by array slicing. PIL would be one more dependency for something this
# small, and the label only ever needs digits and a few letters.
GLYPHS = {
    "0": ("111", "101", "101", "101", "111"), "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"), "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"), "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"), "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"), "9": ("111", "101", "111", "001", "111"),
    "S": ("111", "100", "111", "001", "111"), "T": ("111", "010", "010", "010", "010"),
    "E": ("111", "100", "110", "100", "111"), "P": ("111", "101", "111", "100", "100"),
    "O": ("111", "101", "101", "101", "111"), "K": ("101", "110", "100", "110", "101"),
    "F": ("111", "100", "110", "100", "100"), "A": ("111", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"), "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"), "N": ("101", "111", "111", "111", "101"),
    "R": ("111", "101", "111", "110", "101"), "D": ("110", "101", "101", "101", "110"),
    "G": ("111", "100", "101", "101", "111"), "H": ("101", "101", "111", "101", "101"),
    "U": ("101", "101", "101", "101", "111"), "C": ("111", "100", "100", "100", "111"),
    "V": ("101", "101", "101", "101", "010"), "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"), "Y": ("101", "101", "010", "010", "010"),
    "B": ("110", "101", "110", "101", "110"), "J": ("001", "001", "001", "101", "111"),
    "Q": ("111", "101", "101", "111", "001"), "Z": ("111", "001", "010", "100", "111"),
    ".": ("000", "000", "000", "000", "010"), "-": ("000", "000", "111", "000", "000"),
    " ": ("000", "000", "000", "000", "000"), "%": ("101", "001", "010", "100", "101"),
    "/": ("001", "001", "010", "100", "100"), "#": ("101", "111", "101", "111", "101"),
}


def draw_text(img: np.ndarray, text: str, x: int, y: int, scale: int = 2,
              color=(255, 255, 255), plate=(0, 0, 0)) -> None:
    """Burn `text` into `img` at (x, y). Clips rather than raising when it runs off the edge."""
    text = text.upper()
    w = (len(text) * 4) * scale
    h = 5 * scale
    if plate is not None:
        y0, y1 = max(0, y - scale), min(img.shape[0], y + h + scale)
        x0, x1 = max(0, x - scale), min(img.shape[1], x + w + scale)
        if y1 > y0 and x1 > x0:
            img[y0:y1, x0:x1] = plate
    cx = x
    for ch in text:
        g = GLYPHS.get(ch)
        if g is None:
            cx += 4 * scale
            continue
        for r, row in enumerate(g):
            for c, bit in enumerate(row):
                if bit != "1":
                    continue
                py, px = y + r * scale, cx + c * scale
                if 0 <= py < img.shape[0] - scale and 0 <= px < img.shape[1] - scale:
                    img[py:py + scale, px:px + scale] = color
        cx += 4 * scale


def grid_shape(n: int) -> tuple[int, int]:
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def extract(video: Path, times: list[float], width: int, crop: str | None = None) -> list[np.ndarray]:
    """Pull one RGB frame per timestamp, by seeking. One ffmpeg call per frame.

    Seeking per frame rather than decoding the whole file matters: these recordings run 16 minutes
    and the sheet wants a dozen frames out of them.
    """
    out = []
    for t in times:
        probe = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video),
             "-frames:v", "1", "-vf", (f"crop={crop},scale={width}:-2" if crop else f"scale={width}:-2"),
             "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-"],
            capture_output=True,
        )
        if probe.returncode != 0 or not probe.stdout:
            out.append(None)
            continue
        n = len(probe.stdout)
        # Height is whatever `scale=W:-2` produced; recover it from the byte count.
        h = n // (width * 3)
        out.append(np.frombuffer(probe.stdout, dtype=np.uint8)[: h * width * 3].reshape(h, width, 3))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes-json", default=None,
                   help="episodes.json from render_rollout.py; used to pick an episode by index "
                        "and to label the sheet with its recorded outcome")
    p.add_argument("--episode", type=int, default=None, help="episode index from episodes.json")
    p.add_argument("--start", type=float, default=None, help="seconds, when not using --episode")
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--tile-width", type=int, default=420)
    p.add_argument("--crop", default=None,
                   help="ffmpeg crop filter, w:h:x:y, applied BEFORE scaling. The recorded framing "
                        "puts a 34 mm cube at a few dozen pixels, so judging whether the hand is "
                        "gripping it or merely near it needs a crop, not a bigger tile.")
    args = p.parse_args()

    video = Path(args.video)
    label_extra = ""
    if args.episode is not None:
        meta = json.loads(Path(args.episodes_json).read_text())
        fps = meta["fps_steps_per_second"]
        ep = next(e for e in meta["episodes"] if e["index"] == args.episode)
        t0, t1 = ep["frame_start"] / fps, ep["frame_end"] / fps
        steps = ep["frame_end"] - ep["frame_start"] + 1
        outcome = "OK" if ep["held_at_end"] else ("TOUCHED" if ep["ever_during"] else "FAIL")
        label_extra = (f"EP{ep['index']} {outcome} {steps}STEPS "
                       f"{ep['final_position_error_m'] * 1000:.1f}MM")
        if ep.get("final_hand_to_object_m") is not None:
            label_extra += (f" HAND{ep['final_hand_to_object_m'] * 1000:.0f}MM"
                            f" V{ep.get('final_object_speed_mps', 0) * 1000:.0f}MMS")
        step_of = lambda i: ep["frame_start"] + int(round(i * (steps - 1) / max(1, args.frames - 1)))
    else:
        t0, t1 = args.start or 0.0, args.end
        if t1 is None:
            raise SystemExit("give --episode with --episodes-json, or --start and --end")
        fps = 10.0
        step_of = lambda i: int(round((t0 + i * (t1 - t0) / max(1, args.frames - 1)) * fps))

    times = [t0 + i * (t1 - t0) / max(1, args.frames - 1) for i in range(args.frames)]
    tiles = extract(video, times, args.tile_width, args.crop)
    good = [t for t in tiles if t is not None]
    if not good:
        raise SystemExit(f"no frames could be read from {video}")
    th, tw = good[0].shape[:2]

    rows, cols = grid_shape(len(tiles))
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        if tile is None:
            continue
        if tile.shape[:2] != (th, tw):
            tile = tile[:th, :tw]
        sheet[r * th:r * th + tile.shape[0], c * tw:c * tw + tile.shape[1]] = tile
        draw_text(sheet, f"STEP {step_of(i)}", c * tw + 6, r * th + 6, scale=2)
    if label_extra:
        draw_text(sheet, label_extra, 6, sheet.shape[0] - 18, scale=2, color=(255, 220, 120))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{sheet.shape[1]}x{sheet.shape[0]}", "-i", "-", str(out)],
        input=sheet.tobytes(), check=True,
    )
    print(f"[SHEET] {len(good)}/{len(tiles)} frames -> {out} ({out.stat().st_size / 1e3:.0f} kB)")
    print(f"        {label_extra or f'{t0:.1f}s - {t1:.1f}s'}")


if __name__ == "__main__":
    main()
