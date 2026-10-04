"""Compare training runs against the measured baselines, over matched iteration windows.

WHY WINDOWS AND NOT LAST-VALUE. `end_of_episode_success_rate` swings several points between
consecutive iterations, so a single point proves nothing; and the runs being compared are rarely at
the same iteration, so "latest vs latest" compares different amounts of training. Windowed means
over a common iteration range fix both.

The baselines are the two near-goal seeds that established the plateau: 42 (local) and 7 (rented).
Anything claiming to beat the plateau has to beat those two over the same window.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

CS = pathlib.Path("/home/mattephi/Work/Rubitek/cube-stack")
REPO = pathlib.Path("/home/mattephi/Work/Rubitek/UWLab-cubestack")
SUCCESS = "Metrics/task_command/end_of_episode_success_rate"
ERROR = "Metrics/task_command/end_of_episode_pos_align_error"

BASELINES = {
    "baseline s42": REPO / "logs/rsl_rl/ur10e_delto_omnireset_agent/2026-08-31_11-16-08",
    "baseline s7": CS / "results/seed7_neargoal",
}


def newest_run(box: str) -> pathlib.Path | None:
    root = CS / "results" / f"remote_run_{box}"
    runs = [d for d in root.glob("*/") if any(d.glob("events.out.tfevents.*"))] if root.exists() else []
    return max(runs, key=lambda d: d.stat().st_mtime) if runs else None


FAMILY_ABBREV = {
    "ObjectAnywhereEEAnywhere": "reach",
    "ObjectRestingEEGrasped": "near-obj",
    "ObjectAnywhereEEGrasped": "grasped",
    "ObjectPartiallyAssembledEEGrasped": "near-goal",
}


def criterion(d: pathlib.Path) -> str:
    """Position-only or position+orientation.

    Without this the table invites a false comparison: an ORIENTED run's success rate is measured
    against a strictly harder gate (euler_xy < 0.05 rad as well as position < 5 mm), so matching a
    position-only run's number means it is doing better, not the same. And the gate is not free --
    a position-only policy ends a near-goal episode at a median 0.68 rad of roll/pitch, with only
    26 of 58 episodes inside 0.05 rad.
    """
    try:
        txt = (d / "params" / "env.yaml").read_text()
    except OSError:
        return "?"
    for line in txt.splitlines():
        if "success_orientation_threshold" in line:
            v = line.split(":", 1)[1].strip()
            try:
                return "pos-only" if float(v) > 1.0 else f"pos+ori<{float(v):g}"
            except ValueError:
                return "?"
    return "?"


def families(d: pathlib.Path) -> str:
    """Which reset families a run samples.

    Without this the table silently compares a four-family run against near-goal-only baselines and
    invites the reading that the four-family run is doing badly -- it is doing a different, much
    harder task, in which only a quarter of episodes even start near the goal.
    """
    try:
        txt = (d / "params" / "env.yaml").read_text()
    except OSError:
        return "?"
    block = txt.split("reset_types:", 1)
    if len(block) < 2:
        return "?"
    out = []
    for line in block[1].splitlines()[1:]:
        st = line.strip()
        if st.startswith("- ") and st[2:] in FAMILY_ABBREV:
            out.append(FAMILY_ABBREV[st[2:]])
        elif not st.startswith("- "):
            break
    return "4-family" if len(out) == 4 else "+".join(out) or "?"


def read(d: pathlib.Path) -> tuple[dict, dict]:
    ev = max(d.glob("events.out.tfevents.*"), key=lambda f: f.stat().st_mtime)
    ea = EventAccumulator(str(ev), size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags()["scalars"])
    s = {p.step: p.value for p in ea.Scalars(SUCCESS)} if SUCCESS in have else {}
    e = {p.step: p.value for p in ea.Scalars(ERROR)} if ERROR in have else {}
    return s, e


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("boxes", nargs="*", default=["b1", "b2", "b3"])
    ap.add_argument("--windows", type=int, default=4)
    args = ap.parse_args()

    runs = dict(BASELINES)
    for b in args.boxes:
        d = newest_run(b)
        if d is not None:
            runs[f"{b} {d.name[11:16]}"] = d

    data = {}
    for name, d in runs.items():
        try:
            s, e = read(pathlib.Path(d))
        except (OSError, ValueError, KeyError):
            continue
        if s:
            data[name] = (s, e)
    if not data:
        raise SystemExit("no runs with a success curve")

    nmax = min(max(s) for s, _ in data.values())
    width = max(1, (nmax + 1) // args.windows)
    bounds = [(i * width, min(nmax, (i + 1) * width - 1)) for i in range(args.windows)]

    print(f"matched window: iterations 0-{nmax}   ({len(data)} runs)\n")
    head = "".join(f"{f'{a}-{b}':>10}" for a, b in bounds)
    print(f"{'run':<26}{'families':<11}{'criterion':<15}{head}{'median err':>13}")
    for name, (s, e) in data.items():
        cells = ""
        for a, b in bounds:
            v = [s[k] for k in s if a <= k <= b]
            cells += f"{statistics.mean(v):>10.3f}" if v else f"{'-':>10}"
        tail = [e[k] for k in sorted(e)[-20:]]
        err = f"{statistics.median(tail) * 1000:.1f} mm" if tail else "-"
        dd = pathlib.Path(runs[name])
        print(f"{name:<26}{families(dd):<11}{criterion(dd):<15}{cells}{err:>13}")
    print("\nBaselines plateaued at ~0.20 with median end-of-episode error ~17 mm against a 5 mm "
          "tolerance.\nA change only counts if it beats BOTH over the same window -- AND is on the "
          "same family set.\nA 4-family run is not comparable to a near-goal-only one: only a "
          "quarter of its episodes start near the goal.")
