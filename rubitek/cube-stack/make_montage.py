"""Build ONE montage of the genuine successes, slowed and labelled, instead of a gallery of clips.

WHY. The gallery had five entries: four 4-to-6 second success clips and the full 58-episode
recording. The full recording is 46 MB of mostly nothing -- it exists as the evidence the clips were
cut from, not as something to watch -- and next to it the short clips read as more of the same. The
successes are also genuinely quick: a near-goal episode starts with the cube already in the hand
about 13 mm from the stack pose, so the whole visible event is a short corrective placement.

So: one file, each success at half speed, each labelled in-frame with what it is, and a hold on the
final frame long enough to see that the cube is standing on its own with the hand gone. Labels are
burned in rather than left to a caption because a video gets shared without its page.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

CS = Path("/home/mattephi/Work/Rubitek/cube-stack")


def esc(text: str) -> str:
    """Escape for ffmpeg's drawtext, which treats : and ' as syntax."""
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=str(CS / "results/video_neargoal"))
    p.add_argument("--out", default=str(CS / "report/media/genuine_stacks_montage.mp4"))
    p.add_argument("--min-steps", type=int, default=10)
    p.add_argument("--slow", type=float, default=2.0, help="playback slowdown factor")
    p.add_argument("--pad", type=float, default=0.0,
                   help="seconds of lead-in. Defaults to ZERO so every frame in a clip belongs to "
                        "the episode its label names -- a lead-in reaches back into the PREVIOUS "
                        "episode, which on the near-goal family means showing someone else's "
                        "finished stack just before this one starts. There is no trailing pad "
                        "either: frames "
                        "after an episode's last belong to the NEXT episode's reset, because "
                        "ManagerBasedRLEnv resets inside step() and the frame is rendered after. A "
                        "trailing pad therefore ended every clip on an unstacked cube, one second "
                        "after the label claimed a stack.")
    p.add_argument("--hold", type=float, default=1.8,
                   help="seconds to hold the episode's FINAL frame -- the finished stack with the "
                        "hand clear, which is the thing worth looking at")
    p.add_argument("--max-clips", type=int, default=6)
    args = p.parse_args()

    d = Path(args.dir)
    meta = json.loads((d / "episodes.json").read_text())
    fps = meta["fps_steps_per_second"]
    src = d / "rollout.mp4"

    eps = [e for e in meta["episodes"] if not e.get("truncated")]
    for e in eps:
        e["steps"] = e["frame_end"] - e["frame_start"] + 1
    wins = sorted((e for e in eps if e["held_at_end"] and e["steps"] > args.min_steps),
                  key=lambda e: -e["steps"])[: args.max_clips]
    if not wins:
        raise SystemExit("no genuine success to montage")

    tmp = d / "_montage"
    tmp.mkdir(exist_ok=True)
    parts = []
    for i, e in enumerate(wins):
        t0 = max(0.0, e["frame_start"] / fps - args.pad)
        # +1 frame so the episode's final frame is included; nothing beyond it.
        t1 = (e["frame_end"] + 1) / fps
        hand = e.get("final_hand_to_object_m")
        line1 = f"GENUINE STACK {i + 1} of {len(wins)}   episode {e['index']}   {e['steps']} steps"
        line2 = (f"final error {e['final_position_error_m'] * 1000:.1f} mm  (tolerance 5.0 mm)"
                 + (f"   hand withdrawn {hand * 1000:.0f} mm" if hand is not None else ""))
        part = tmp / f"part_{i:02d}.mp4"
        # setpts slows playback; tpad holds the last frame so the finished stack is readable.
        vf = (
            f"setpts={args.slow}*PTS,"
            f"tpad=stop_mode=clone:stop_duration={args.hold},"
            f"drawbox=x=0:y=ih-58:w=iw:h=58:color=black@0.62:t=fill,"
            f"drawtext=text='{esc(line1)}':x=14:y=h-50:fontsize=20:fontcolor=white,"
            f"drawtext=text='{esc(line2)}':x=14:y=h-26:fontsize=17:fontcolor=0xC8E6C9"
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.2f}", "-to", f"{t1:.2f}",
             "-i", str(src), "-vf", vf, "-an",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
             str(part)],
            check=True,
        )
        parts.append(part)
        print(f"  ep{e['index']:>3}  {e['steps']:>4} steps  {e['final_position_error_m'] * 1000:5.1f} mm")

    lst = tmp / "concat.txt"
    lst.write_text("".join(f"file '{q.name}'\n" for q in parts))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(out)],
        check=True, cwd=tmp,
    )
    for q in parts:
        q.unlink(missing_ok=True)
    lst.unlink(missing_ok=True)
    tmp.rmdir()

    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "title": f"{len(wins)} genuine stacks, half speed",
        "kind": "success",
        "hero": True,
        "loop": True,
        "reset_type": meta.get("reset_type"),
        "checkpoint": Path(meta.get("checkpoint", "")).name,
        "episodes": len(wins),
        "caption": (
            "Every success in the rollout that ran longer than the 10-step boundary, at half speed, "
            "each labelled in-frame. A near-goal episode starts with the cube already held about "
            "13 mm from the stack pose, so what is visible is a corrective placement and release, "
            "not a pick from the table -- the full task does not work yet."),
    }, indent=2))
    print(f"\n[MONTAGE] {len(wins)} clips -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
