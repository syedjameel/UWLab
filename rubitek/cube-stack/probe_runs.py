"""Report where each remote training run actually is, by reading its TensorBoard event file.

WHY NOT THE JOB LOG. The job scripts pipe training stdout through `tr | grep`, and both stages
block-buffer, so the log on the box can sit several minutes behind the run -- long enough that a
watcher reading it reports a stale iteration and, worse, cannot tell a stalled run from a buffered
one. The event file is written by the trainer itself on its own schedule and is authoritative.

Pulls each box's newest event file (rsync, newer-only, a few hundred kB) and prints one line per
run. Exit is silent on unreachable boxes: a transient ssh failure is not news.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CS = Path("/home/mattephi/Work/Rubitek/cube-stack")
KEY = "/home/mattephi/Work/IRiS/ip/meaningful-latents/.secrets/id_ed25519_ml_vast"
BOXES = {
    "b1": ("40624", "root@38.29.145.10", "money 4-family"),
    "b3": ("17626", "root@89.221.67.158", "oriented, staged"),
    "b4": ("40523", "root@38.117.87.51", "two-finger curriculum"),
    "b5": ("45350", "root@115.73.217.180", "five-finger curriculum"),
}
WATCH = [
    "Metrics/task_command/end_of_episode_success_rate",
    "Metrics/task_command/end_of_episode_pos_align_error",
    "Policy/mean_noise_std",
    # The grasp terms are the point of the newest change, and they are self-verifying: if the
    # fingertip contact sensors are not reading, grasp_contact logs exactly 0.0000 forever, which is
    # indistinguishable from "the policy never grips" unless it is watched from the start.
    "Episode_Reward/grasp_contact",
    "Episode_Reward/object_lift",
    "Episode_Reward/fingertip_object_distance",
]


def pull(tag: str, port: str, host: str) -> Path | None:
    """rsync one box's event files. Returns None on any failure, including a timeout.

    The timeout is 90 s, not 300: the caller allows 200 s for all three boxes, so a single slow
    rsync used to blow the whole probe and make every box look unreachable at once.
    """
    dst = CS / "results" / f"remote_run_{tag}"
    dst.mkdir(parents=True, exist_ok=True)
    ssh = f"ssh -i {KEY} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o BatchMode=yes -p {port}"
    # --update so a shorter/older remote file never replaces a longer local one, and the run
    # directory glob so a restarted run's new directory is picked up without editing this file.
    try:
        r = subprocess.run(
        # Event files ONLY. Checkpoints are ~8.8 MB each and pulling them here turned a probe that
        # should take seconds into one that outran a two-minute timeout; they are fetched
        # deliberately, by pull_checkpoint.sh, when there is a reason to evaluate one.
        ["rsync", "-az", "--update", "-e", ssh, "--include=*/", "--include=events.out.tfevents.*",
         # params/*.yaml too: the report labels each curve by the reset families and
         # entropy_coef the trainer recorded there, and a legend of raw timestamps is
         # useless for the comparison these charts exist to make. Two small files.
         "--include=params/", "--include=params/*.yaml",
         "--exclude=*",
         f"{host}:/workspace/UWLab/logs/rsl_rl/ur10e_delto_omnireset_agent/", str(dst) + "/"],
        capture_output=True, text=True, timeout=90,
        )
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    return dst


def newest_event(root: Path) -> Path | None:
    ev = list(root.rglob("events.out.tfevents.*"))
    return max(ev, key=lambda f: f.stat().st_mtime) if ev else None


def read(ev: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(ev), size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags()["scalars"])
    out = {"run": ev.parent.name}
    for t in WATCH:
        if t not in have:
            continue
        s = ea.Scalars(t)
        # `iter` from the success-rate tag specifically, not from whichever tag happened to be last.
        if t.endswith("success_rate"):
            out["iter"] = s[-1].step
        out.setdefault("iter", s[-1].step)
        out[t.split("/")[-1]] = round(s[-1].value, 5)
        if t.endswith("success_rate"):
            out["best"] = round(max(p.value for p in s), 4)
            # Recent mean rather than the last point: this metric is noisy iteration to iteration.
            tail = [p.value for p in s[-25:]]
            out["recent25"] = round(sum(tail) / len(tail), 4)
    return out


if __name__ == "__main__":
    rows = []
    for tag, (port, host, what) in BOXES.items():
        d = pull(tag, port, host)
        if d is None:
            continue
        ev = newest_event(d)
        if ev is None:
            continue
        try:
            r = read(ev)
        except Exception as e:  # a half-written event file is normal mid-flush
            print(f"{tag} {what}: event file unreadable ({e.__class__.__name__})", file=sys.stderr)
            continue
        r["box"] = tag
        r["what"] = what
        rows.append(r)
        print(f"{tag} [{what}] iter={r.get('iter')} "
              f"success={r.get('end_of_episode_success_rate')} "
              f"recent25={r.get('recent25')} best={r.get('best')} "
              f"pos_err={r.get('end_of_episode_pos_align_error')} "
              f"noise={r.get('mean_noise_std')} "
              f"tips={r.get('fingertip_object_distance')} "
              f"grasp={r.get('grasp_contact')} lift={r.get('object_lift')}")
    if rows:
        (CS / "results" / "remote_progress.json").write_text(json.dumps(rows, indent=2))
    else:
        print("no box reported; keeping the previous snapshot", file=sys.stderr)
