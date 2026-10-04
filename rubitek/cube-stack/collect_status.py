"""Gather every measurement into report/data.json, which the report page fetches.

The page holds no numbers of its own. Everything it draws is read here from the artefacts that
produced it -- TensorBoard event files written by the trainer, eval JSON written by eval_policy.py,
episodes.json written by render_rollout.py -- so a number can only appear on the page if a run
actually produced it. Re-run this and reload the page; nothing has to be edited by hand.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

CS = Path("/home/mattephi/Work/Rubitek/cube-stack")
REPO = Path("/home/mattephi/Work/Rubitek/UWLab-cubestack")
OUT = CS / "report" / "data.json"

SCALARS = [
    "Metrics/task_command/end_of_episode_success_rate",
    "Metrics/task_command/end_of_episode_pos_align_error",
    "Metrics/task_command/average_pos_align_error",
    "Policy/mean_noise_std",
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Episode_Reward/dense_success_reward",
    "Episode_Reward/success_reward",
    "Episode_Reward/grasp_contact",
    "Episode_Reward/any_finger_contact",
    "Episode_Reward/object_lift",
    "Episode_Reward/fingertip_object_distance",
]


def read_events(run_dir: Path) -> dict:
    """Pull the curves out of a TensorBoard event file, thinned to keep the payload small."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # Newest by mtime, not largest by size: a run restarted in place leaves several event files,
    # and the biggest one is not necessarily the current one -- picking it rendered one segment
    # of the curve and dropped the rest with no notice.
    ev = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda f: f.stat().st_mtime)
    if not ev:
        return {}
    ea = EventAccumulator(str(ev[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags()["scalars"])
    out = {}
    for tag in SCALARS:
        if tag not in have:
            continue
        pts = ea.Scalars(tag)
        # One point per iteration is more than any chart needs; thin to ~200 so the JSON stays small
        # and the page stays responsive on a phone over the tailnet.
        step = max(1, len(pts) // 200)
        out[tag.split("/")[-1]] = [[p.step, round(p.value, 6)] for p in pts[::step]]
    return out


FAMILY_ABBREV = {
    "ObjectAnywhereEEAnywhere": "reach",
    "ObjectRestingEEGrasped": "near-obj",
    "ObjectAnywhereEEGrasped": "grasped",
    "ObjectPartiallyAssembledEEGrasped": "near-goal",
}


def label(run_dir: Path, fallback: str) -> str:
    """Name a run by what makes it different, not by when it started.

    Every directory here is a timestamp, and a chart legend of nine timestamps is unreadable -- which
    matters because the whole point of these charts is comparing configurations. The two fields that
    actually separate these runs are the reset families sampled and entropy_coef, so read them back
    out of the params the trainer already wrote and build the label from those.
    """
    params = run_dir / "params"
    fams, ent, fine = [], None, False
    try:
        env_txt = (params / "env.yaml").read_text()
        block = env_txt.split("reset_types:", 1)
        if len(block) > 1:
            for line in block[1].splitlines()[1:]:
                st = line.strip()
                if st.startswith("- ") and st[2:] in FAMILY_ABBREV:
                    fams.append(FAMILY_ABBREV[st[2:]])
                elif not st.startswith("- "):
                    break
        fine = "dense_success_reward_fine" in env_txt
    except OSError:
        pass
    try:
        for line in (params / "agent.yaml").read_text().splitlines():
            if "entropy_coef:" in line:
                ent = float(line.split(":", 1)[1])
                break
    except OSError:
        pass

    bits = []
    if fams:
        bits.append("+".join(fams) if len(fams) < 4 else "4-family")
    if ent is not None:
        bits.append(f"ent {ent:g}")
    if fine:
        bits.append("fine")
    return " · ".join(bits) if bits else fallback


def stamp(name: str) -> str:
    """The HH-MM of a run directory, or nothing when the name is not a timestamp.

    Two runs can share a configuration, so the label needs a discriminator; but not every run
    directory here is a timestamp (pulled runs keep their own names), and blindly slicing one
    produced labels like "seed7_neargoal oal".
    """
    return name[11:16] if len(name) >= 16 and name[:4].isdigit() else ""


def _ck_iter(f: Path) -> int | None:
    """Iteration number from a checkpoint filename, or None when it is not numbered.

    `int(f.stem.split("_")[1])` raised on a single `model_best.pt`, and because this runs before
    anything is written, one such file meant NO data.json at all -- the page kept serving the
    previous build with no indication it was stale.
    """
    parts = f.stem.split("_")
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    return int(parts[1])


def _checkpoints(d: Path) -> list[Path]:
    return sorted((f for f in d.glob("model_*.pt") if _ck_iter(f) is not None), key=_ck_iter)


def runs() -> list[dict]:
    root = REPO / "logs" / "rsl_rl" / "ur10e_delto_omnireset_agent"
    found = []
    for d in sorted(root.glob("*/")) if root.exists() else []:
        params = d / "params" / "env.yaml"
        ck = _checkpoints(d)
        found.append(
            {
                "name": f"{label(d, d.name)} · local {d.name[11:16]}",
                "dir": str(d),
                "checkpoints": [c.stem for c in ck],
                "last_iter": _ck_iter(ck[-1]) if ck else 0,
                "has_params": params.exists(),
                "scalars": read_events(d),
            }
        )
    # Runs pulled off the rented boxes live outside that tree. probe_runs.py mirrors each box's
    # logs/rsl_rl/<experiment>/ into results/remote_run_<box>/, so the event files sit one level
    # down in a per-run directory -- glob at the top level finds nothing and the remote runs vanish
    # from the page while looking perfectly healthy on the boxes.
    remote_dirs = []
    for box in sorted((CS / "results").glob("remote_run_*")):
        for run in sorted(box.glob("*/")):
            if any(run.glob("events.out.tfevents.*")):
                remote_dirs.append((f"{box.name.replace('remote_run_', '')}/{run.name}", run))
    for name, extra in [("seed7_neargoal", CS / "results" / "seed7_neargoal")] + remote_dirs:
        if extra.exists() and any(extra.glob("events.out.tfevents.*")):
            ck = _checkpoints(extra)
            found.append(
                {
                    "name": f"{label(extra, name)} · {name.split('/')[0]} {stamp(extra.name)}",
                    "dir": str(extra),
                    "checkpoints": [c.stem for c in ck],
                    "last_iter": _ck_iter(ck[-1]) if ck else 0,
                    "has_params": False,
                    "scalars": read_events(extra),
                }
            )
    return found


def evals() -> list[dict]:
    out = []
    for f in sorted((CS / "results").rglob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if "success_held_at_end_pct" not in d:
            continue
        # A STRICTER CUT, derived from the histogram already stored -- no re-run needed.
        #
        # ">10 steps" is the boundary the termination rule sets, but it is generous: firing at step
        # 12 only requires the predicate to have held from step 7, and a reset that starts 6-8 mm out
        # needs one nudge to get there. Successes lasting more than 20 steps cannot be explained that
        # way at all. The two populations separate completely under it: untrained 2 of 840, trained
        # 47 of 821.
        el = d.get("episode_length_success")
        if el and d.get("episodes"):
            # Prefer the exact count eval_policy now records. The histogram fallback is for JSON
            # written before that existed, and it is an UPPER BOUND: its buckets are 10 wide, so
            # `int(k) >= 20` includes episodes of exactly 20 steps, which ">20" excludes.
            gt20 = el.get("n_gt_20")
            approx = gt20 is None
            if approx and el.get("histogram"):
                gt20 = sum(v for k, v in el["histogram"].items() if int(k) >= 20)
            if gt20 is not None:
                d["sustained_success_n"] = gt20
                d["sustained_success_pct"] = 100.0 * gt20 / d["episodes"]
                d["sustained_is_upper_bound"] = approx
        d["degraded"] = bool(
            d.get("action_noise") or (d.get("solver_iters") not in (None, 36))
        )
        d["_file"] = f.name
        d["_mtime"] = f.stat().st_mtime
        out.append(d)
    return out


def _safe_json(path: Path):
    """A malformed sidecar must not be able to stop the whole report from being written."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        print(f"  WARNING: ignoring unreadable sidecar {path.name} ({exc.__class__.__name__})")
        return None


def carry_progress() -> list[dict]:
    """The per-checkpoint carry trend, which is what shows learning while success is still zero.

    Success on Stable-Grasp is a step function -- flat until the policy can both carry AND place
    within 5 mm -- so it carries no information during the phase the runs are in. Progress is
    measured against each episode's OWN starting distance, so a policy that changes nothing scores
    zero however dramatic its endpoints look, and it moves continuously.
    """
    out = []
    for f in sorted((CS / "results").glob("track_b*_model_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        pr = d.get("goal_progress_m")
        if not pr:
            continue
        stem = f.stem  # track_<box>_model_<iter>
        box = stem.split("_")[1]
        try:
            it = int(stem.split("model_")[1])
        except (IndexError, ValueError):
            continue
        out.append({
            "box": box,
            "iter": it,
            "median_mm": (pr.get("median") or 0.0) * 1000.0,
            "p90_mm": (pr.get("p90") or 0.0) * 1000.0,
            "frac_closer": (pr.get("frac_closer") or 0.0) * 100.0,
            "success_pct": d.get("success_held_at_end_pct"),
            "abs_action_arm": d.get("mean_abs_action_arm"),
            "episodes": d.get("episodes"),
        })
    out.sort(key=lambda r: (r["box"], r["iter"]))
    return out


def videos() -> list[dict]:
    """Every clip under report/media, with whatever episodes.json says about what is in it."""
    media = CS / "report" / "media"
    out = []
    for f in sorted(media.glob("*.mp4")) if media.exists() else []:
        meta = f.with_suffix(".json")
        out.append(
            {
                "file": f"media/{f.name}",
                "size_mb": round(f.stat().st_size / 1e6, 1),
                "meta": _safe_json(meta),
            }
        )
    return out


def sheets() -> list[dict]:
    """Contact sheets: one image per episode, the whole motion in a single glance.

    A video answers "did it succeed"; a contact sheet answers "what did it actually do", which is a
    different and often more useful question. The sheet that showed the hand arriving at the cube
    and then never closing its fingers settled the reward question faster than any aggregate did.
    """
    d = CS / "report" / "media" / "frames"
    out = []
    for f in sorted(d.glob("*.png")) if d.exists() else []:
        meta = f.with_suffix(".json")
        out.append(
            {
                "file": f"media/frames/{f.name}",
                "size_mb": round(f.stat().st_size / 1e6, 2),
                "meta": _safe_json(meta),
            }
        )
    return out


def banks() -> list[dict]:
    root = REPO / "Datasets_ur10e_delto" / "OmniReset"
    out = []
    for f in sorted(root.rglob("*.pt")) if root.exists() else []:
        out.append({"path": str(f.relative_to(root)), "size_mb": round(f.stat().st_size / 1e6, 2)})
    return out


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "log", "-1", "--format=%h %s"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return ""


MIN_ITERS = 5


def useful(r: dict) -> bool:
    """Drop runs that never really started.

    Several directories here are aborted launches -- the gsde ValueError, a mis-set variant, a
    restart -- and each still writes an event file with one or two points. On the Learning tab they
    render as a stub at the origin that is indistinguishable from a run that learned nothing, which
    is the one reading the page must not invite.
    """
    curve = r["scalars"].get("end_of_episode_success_rate", [])
    return len(curve) >= MIN_ITERS


def _hold_summary(path: Path) -> dict | None:
    """One grip-probe run, summarised at the last step BEFORE the episode wraps.

    The episode is 160 control steps and ManagerBasedRLEnv.step() resets a done env inside the same
    call, so a trace that reaches the limit ends on a FRESH episode's cube -- which reads as the
    cube having risen back into the hand, and did, until it was cut here. Older probe JSON has no
    `steps_before_autoreset` field, so the cut is recomputed from the trace either way.
    """
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    trace = d.get("trace")
    if not trace:
        return None
    z = [r["z_mean"] * 1000.0 for r in trace]
    cut = len(z)
    for i in range(1, len(z)):
        if z[i] - z[i - 1] > 20.0:
            cut = i
            break
    end = trace[cut - 1]
    return {
        "file": path.name,
        "reset_type": d.get("reset_type"),
        "grip_bias": d.get("grip_bias"),
        "hold_hand": d.get("hold_hand", False),
        "steps": cut,
        "z_start_mm": round(z[0], 1),
        "z_end_mm": round(z[cut - 1], 1),
        "drop_mm": round(z[0] - z[cut - 1], 1),
        "nearest_tip_start_mm": round(trace[0].get("tip_to_obj_min_mm", float("nan")), 1),
        "nearest_tip_end_mm": round(end.get("tip_to_obj_min_mm", float("nan")), 1),
        "palm_start_mm": round(trace[0]["palm_to_obj_mean"] * 1000.0, 1),
        "palm_end_mm": round(end["palm_to_obj_mean"] * 1000.0, 1),
        "tip_forces_end_N": {k: round(v, 3) for k, v in end["tip_forces_N"].items()},
        "max_tip_force_N": round(max(end["tip_forces_N"].values()), 3),
        "reset_frame": d.get("reset_frame"),
    }


def physics() -> dict:
    """The null-action grip probes: does the reset actually hold the cube, and at what bias.

    Separate from `evals` because it grades a RESET, not a policy. Nothing here involves a
    checkpoint -- the action is identically zero -- so any motion in it belongs to the environment.
    """
    res = CS / "results"
    families, sweep = [], []
    for f in sorted(res.glob("grip_*.json")):
        if f.name.startswith(("grip_bias_", "gripb2_")):
            continue
        row = _hold_summary(f)
        if row:
            families.append(row)
    for pattern in ("grip_bias_*.json", "gripb2_*.json"):
        for f in sorted(res.glob(pattern)):
            row = _hold_summary(f)
            if row:
                # The joint set differs between the two rounds and the distinction is the whole
                # point of round two, so carry it rather than letting the bias values collide.
                row["joints"] = "flexion _2/_4" if pattern.startswith("grip_bias_") else "all hand joints"
                bias = f.stem.split("_")[-1]
                row["grip_bias"] = float(bias) if bias.replace(".", "").isdigit() else None
                sweep.append(row)
    # THE SAME PROBE RUN THROUGH THE REGISTERED TASK IDS, with no command-line flag. Every earlier
    # measurement patched the action term from the command line; this is the path training and eval
    # actually take, and it had never been measured. It matters because a bias that silently failed
    # to apply here would look exactly like "the fix does not help".
    tasks = []
    for f in sorted(res.glob("taskbias_*.json")):
        row = _hold_summary(f)
        if row:
            row["variant"] = f.stem.replace("taskbias_", "")
            tasks.append(row)
    # THE SCRIPTED CARRY. A controller with no policy, no reward and perfect knowledge of the goal,
    # used to ask what is ACHIEVABLE rather than what was learned. Near-goal is the positive
    # control: it scores, so the environment can produce a success.
    oracle = []
    for f in sorted(res.glob("fin_*.json")) + sorted(res.glob("rf_*.json")) + sorted(res.glob("ep_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if "success_held_at_end_pct" not in d or "gain" not in d:
            continue
        pe = d.get("position_error_m") or {}
        oracle.append({
            "run": f.stem,
            "reset_type": d.get("reset_type"),
            "episodes": d.get("episodes"),
            "stiffness": d.get("motion_stiffness"),
            "damping": d.get("motion_damping"),
            "episode_length_s": d.get("episode_length_s"),
            "waypoint": d.get("waypoint"),
            "success_pct": d.get("success_held_at_end_pct"),
            "median_err_mm": (pe.get("median") or float("nan")) * 1000.0,
            "n_gt_10": (d.get("episode_length_success") or {}).get("n_gt_10"),
        })
    return {"families": families, "bias_sweep": sweep, "task_ids": tasks, "oracle": oracle}


if __name__ == "__main__":
    import datetime

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git_head(),
        "runs": [r for r in runs() if useful(r)],
        "evals": evals(),
        "videos": videos(),
        "sheets": sheets(),
        "banks": banks(),
        "physics": physics(),
        "carry": carry_progress(),
    }
    # NaN IS NOT JSON. Python's json.dumps emits a bare `NaN` token by default, and the browser's
    # JSON.parse rejects it -- so a single missing quantile anywhere in this payload throws inside
    # fetch() and the ENTIRE page renders blank, with the failure visible only in the console. An
    # empty quantile is normal (a run with no successes has no median success length), so this has
    # to be handled rather than avoided.
    def _clean(x):
        if isinstance(x, dict):
            return {k: _clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_clean(v) for v in x]
        if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
            return None
        return x

    OUT.write_text(json.dumps(_clean(payload), allow_nan=False))
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e3:.0f} kB)")
    print(f"  runs={len(payload['runs'])} evals={len(payload['evals'])} "
          f"videos={len(payload['videos'])} sheets={len(payload['sheets'])} "
          f"banks={len(payload['banks'])} "
          f"physics={len(payload['physics']['families'])}+{len(payload['physics']['bias_sweep'])} "
          f"carry={len(payload['carry'])}")
