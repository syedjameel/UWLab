"""Cut the GENUINE successes out of a rollout recording, and write the report's clip metadata.

WHAT "GENUINE" MEANS HERE, AND WHY THE DISTINCTION IS THE WHOLE POINT
--------------------------------------------------------------------
The Play config terminates on
``consecutive_success_state_with_min_length(num_consecutive_successes=5, min_episode_length=10)``,
so an episode can end in success at step 10 at the earliest -- and it does so exactly when the
predicate already held at reset. That is not hypothetical: 9 of the 125 stored near-goal reset
states (7.2%) spawn the cube already inside the 5 mm tolerance, and the untrained policy's
"successes" were 80 out of 82 in the 10-step bucket. Clips of those show a cube that was stacked
before the episode began.

So the default here selects successes that ran LONGER than ``--min-steps`` (10), which are episodes
where the policy moved the cube into tolerance and held it. Selection is still by measurement:
render_rollout.py writes episodes.json with every episode's frame range and success flag, read from
the same ProgressContext the reward uses. Nothing is chosen because it looked good.

Each clip gets a sidecar .json next to it so the report can label it with the reset family, the
checkpoint, the episode length and the final error -- the page never hard-codes a caption.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--dir", required=True, help="directory holding the mp4 and episodes.json")
p.add_argument("--out-dir", default="/home/mattephi/Work/Rubitek/cube-stack/report/media",
               help="where the cut clips and their sidecars are written")
p.add_argument("--prefix", default=None, help="clip filename prefix (default: from reset family)")
p.add_argument("--pad", type=float, default=0.8, help="seconds of context before/after each episode")
p.add_argument("--max-clips", type=int, default=4)
p.add_argument("--min-steps", type=int, default=10,
               help="an episode must run MORE than this many steps to count. 10 is the "
                    "min_episode_length of the termination rule, i.e. the pre-solved boundary.")
p.add_argument("--hero", action="store_true",
               help="mark the single longest selected clip as the page's hero video")
p.add_argument("--also-longest-attempt", action="store_true",
               help="additionally cut the closest-approaching non-success episode, labelled as an "
                    "attempt. Useful when there is exactly one success and the reel needs context.")
args = p.parse_args()

d = Path(args.dir)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
meta = json.loads((d / "episodes.json").read_text())
fps = meta["fps_steps_per_second"]
# The recording is named rollout.mp4 by render_rollout.py. Picking "the largest mp4 in the
# directory" instead was a latent mismatch: these directories accumulate derived clips, and the
# moment one of them exceeded the source, every frame range from episodes.json would have been
# applied to the wrong video and the captions would have described the wrong episodes.
src = d / "rollout.mp4"
if not src.exists():
    legacy = sorted(d.glob("rollout*.mp4"), key=lambda f: f.stat().st_size, reverse=True)
    if not legacy:
        raise SystemExit(f"no rollout mp4 in {d}")
    src = legacy[0]

for e in meta["episodes"]:
    e["steps"] = e["frame_end"] - e["frame_start"] + 1

# A truncated final episode was cut off by the frame budget rather than ended by the environment,
# so its success flag is an instantaneous read on an arbitrary frame. It is also typically the
# longest episode in the recording, which put it first in the ranking below.
# Recordings made before render_rollout.py started writing `truncated` do not carry the flag, so
# infer it: the final episode of a recording is cut off by the frame budget rather than ended by the
# environment unless it happens to coincide with a real termination.
_eps = meta["episodes"]
if _eps and "truncated" not in _eps[-1]:
    last = _eps[-1]
    last["truncated"] = last["frame_end"] >= max(e["frame_end"] for e in _eps)
episodes = [e for e in _eps if not e.get("truncated")]
genuine = [e for e in episodes if e["held_at_end"] and e["steps"] > args.min_steps]
presolved = [e for e in episodes if e["held_at_end"] and e["steps"] <= args.min_steps]
family = meta.get("reset_type", "unknown")
prefix = args.prefix or family
# Clear this prefix's previous clips. Names are {prefix}_{kind}_{NN}, so a rerun producing fewer
# clips left the old high-index files behind -- and with them a second sidecar carrying
# "hero": true, which the page resolves by filename order. The displayed hero could then be a clip
# from an older checkpoint entirely.
for stale in sorted(out_dir.glob(f"{prefix}_*.mp4")) + sorted(out_dir.glob(f"{prefix}_*.json")):
    stale.unlink(missing_ok=True)

print(f"source        : {src.name}  ({src.stat().st_size/1e6:.1f} MB)")
print(f"episodes      : {meta['summary']['n']}   family: {family}")
print(f"scored success: {meta['summary']['held_at_end']}")
print(f"  pre-solved  : {len(presolved)}  (ran <= {args.min_steps} steps -- NOT cut)")
print(f"  genuine     : {len(genuine)}  (ran >  {args.min_steps} steps)")

# RANKING. Two things make a clip worth showing, in this order.
#
# 1. Whether the cube is standing on the other cube rather than being held at the right point. The
#    success predicate tests one distance and requires neither contact nor release, so a held cube
#    scores the same as a placed one. `final_object_speed_mps` and `final_hand_to_object_m` (written
#    by render_rollout.py) settle it: a placed cube is at rest and the hand has withdrawn from it.
# 2. Episode length. A 12-step clip shows almost nothing even when it is genuine; the episodes where
#    the policy worked longest to get there are the ones with visible behaviour in them.
#
# Both are recorded per episode and reported below, so a clip's caption can state which it is rather
# than leaving the viewer to assume.
SETTLED_MPS = 0.02      # 20 mm/s -- a cube at rest on a fixed cube, not one still being carried
RELEASED_M = 0.075      # DELTO mount to cube; the grasp itself sits near 0.05 m


def placed(e):
    v = e.get("final_object_speed_mps")
    h = e.get("final_hand_to_object_m")
    if v is None or h is None:
        return None                      # recorded before these fields existed
    return v < SETTLED_MPS and h > RELEASED_M


sel = sorted(genuine, key=lambda e: (-(placed(e) is True), -e["steps"]))[: args.max_clips]
kinds = ["success"] * len(sel)
n_placed = sum(1 for e in genuine if placed(e) is True)
if genuine and placed(genuine[0]) is not None:
    print(f"  of the genuine successes, {n_placed}/{len(genuine)} end with the cube at rest "
          f"(<{SETTLED_MPS} m/s) and the hand withdrawn (>{RELEASED_M} m)")

if args.also_longest_attempt:
    misses = [e for e in episodes if not e["held_at_end"]]
    if misses:
        sel.append(min(misses, key=lambda e: e["final_position_error_m"]))
        kinds.append("attempt")

if not sel:
    print("\n[CUT] nothing to cut: no success ran longer than the pre-solved boundary.")
    raise SystemExit(0)

written = []
for i, (e, kind) in enumerate(zip(sel, kinds)):
    t0 = max(0.0, e["frame_start"] / fps - args.pad)
    t1 = e["frame_end"] / fps + args.pad
    name = f"{prefix}_{kind}_{i:02d}"
    out = out_dir / f"{name}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.2f}", "-to", f"{t1:.2f}",
         "-i", str(src), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-movflags", "+faststart", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )
    if kind == "success":
        title = "Cube placed and released" if placed(e) is True else "Cube brought inside tolerance"
    else:
        title = "Closest attempt"
    sidecar = {
        "title": title,
        "kind": kind,
        "reset_type": family,
        "checkpoint": Path(meta.get("checkpoint", "")).name,
        "task": meta.get("task"),
        "episode_index": e["index"],
        "steps": e["steps"],
        "final_position_error_m": e["final_position_error_m"],
        "final_object_speed_mps": e.get("final_object_speed_mps"),
        "final_hand_to_object_m": e.get("final_hand_to_object_m"),
        "placed": placed(e),
        "loop": True,
    }
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    written.append((out, e, kind))
    print(f"  ep{e['index']:>3}  {kind:<8} {e['steps']:>4} steps  "
          f"{e['final_position_error_m']*1000:5.1f} mm  ->  {out.name}")

if args.hero:
    # The hero is the longest genuine success -- the clip with the most policy behaviour in it.
    best = max((w for w in written if w[2] == "success"), key=lambda w: w[1]["steps"], default=None)
    if best:
        j = best[0].with_suffix(".json")
        s = json.loads(j.read_text())
        s["hero"] = True
        j.write_text(json.dumps(s, indent=2))
        print(f"  hero -> {best[0].name}")

# Keep the full uncut recording available too; it is the evidence the clips were cut from.
full = out_dir / f"{prefix}_full_rollout.mp4"
if not full.exists() or src.stat().st_mtime > full.stat().st_mtime:
    shutil.copy2(src, full)
    full.with_suffix(".json").write_text(json.dumps({
        "title": f"Full {meta['summary']['n']}-episode recording",
        "kind": "full",
        "reset_type": family,
        "checkpoint": Path(meta.get("checkpoint", "")).name,
        "episodes": meta["summary"]["n"],
    }, indent=2))
    print(f"  full recording -> {full.name}  ({full.stat().st_size/1e6:.1f} MB)")

print(f"\n[CUT] {len(written)} clip(s) written to {out_dir}")
