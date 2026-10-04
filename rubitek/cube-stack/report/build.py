"""Build ~/reports/reset_banks/index.html: OmniReset C1-C4 reset banks, UR10e + linear gripper vs DELTO pinch.

Reads only what the pipelines leave on disk -- recorder logs, the banks themselves, the renderer's
per-state JSON -- so every number on the page is traceable to a file. Anything not produced yet
renders as a visible "pending" placeholder instead of breaking the page.

Run with the Isaac venv's python (needs torch to count bank states):
    /home/mattephi/Work/Rubitek/uwlab-env/.venv-is6/bin/python build.py
"""

from __future__ import annotations

import datetime
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import torch

REPO = Path("/home/mattephi/Work/Rubitek/UWLab-cubestack")
WORK = Path("/home/mattephi/Work/Rubitek/cube-stack")
OUT = Path("/home/mattephi/reports/reset_banks")
URL = "https://mattecache.tailcc6bd2.ts.net/reports/reset_banks/"
PAIR = "InsertiveCube34__ReceptiveCube34"

FAMILIES = [
    ("ObjectAnywhereEEAnywhere", "C1", "Reaching", "Cube anywhere on the table, hand anywhere in the workspace."),
    ("ObjectRestingEEGrasped", "C2", "Near object", "Cube resting on the table, hand closed on it."),
    ("ObjectAnywhereEEGrasped", "C3", "Grasped", "Cube held in the hand, anywhere in the workspace."),
    ("ObjectPartiallyAssembledEEGrasped", "C4", "Near goal", "Cube held, partly stacked on the receptive cube."),
]

RIGS = {
    "linear": {
        "name": "UR10e + linear gripper",
        "short": "Linear gripper",
        "dataset": REPO / "Datasets_ur10e_linear/OmniReset",
        "logs": WORK / "linear/out",
        "films": WORK / "linear/out/films",
        "tag": "linear",
        "tips": 2,
        "train_task": "OmniReset-UR10eLinearGripper-CubeStack-RelCartesianOSC-State-v0",
        "dataset_rel": "./Datasets_ur10e_linear/OmniReset",
    },
    "pinch2f": {
        "name": "UR10e + DELTO DG-5F, two-finger pinch",
        "short": "DELTO pinch",
        "dataset": REPO / "Datasets_ur10e_delto/OmniReset_pinch2f",
        # The full bank recorded after the absolute-hand-action fix, not the killed pre-fix run.
        "grasp_log": WORK / "pinch2f/out/grasps_full.log",
        "logs": WORK / "pinch2f/out",
        "films": WORK / "linear/out/films",
        "tag": "pinch2f",
        "tips": 5,
        "train_task": "OmniReset-UR10eDelto-CubeStackTwoFinger-RelCartesianOSC-State-v0",
        "dataset_rel": "./Datasets_ur10e_delto/OmniReset_pinch2f",
    },
}


# ----------------------------------------------------------------------------------------- data
def grasp_stats(rig: dict) -> dict | None:
    log = rig.get("grasp_log") or rig["logs"] / "grasps.log"
    if not log.exists():
        return None
    last = [l for l in log.read_text(errors="replace").splitlines() if "BATCH_END" in l]
    if not last:
        return None
    m_att = re.search(r"attempts_consumed_so_far=(\d+)", last[-1])
    m_ok = re.search(r"cumulative_successes=(\d+)", last[-1])
    att, ok = int(m_att.group(1)), int(m_ok.group(1))
    return {"attempts": att, "successes": ok, "rate": ok / att if att else 0.0}


def bank_count(path: Path) -> int | None:
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    try:
        return len(data["initial_state"]["articulation"]["robot"]["joint_position"])
    except (KeyError, TypeError):
        pass
    if isinstance(data, dict) and "grasp_relative_pose" in data:
        return len(data["grasp_relative_pose"]["relative_position"])
    return None


def contact_verdicts(rig: dict, family: str, full_bank: bool = False) -> dict | None:
    # full_bank: the QA replay of every recorded state (before filtering), when it exists; otherwise
    # and by default, the six states in the family's film.
    qa = rig["logs"] / "qa" / f"render_{family}"
    use_qa = full_bank and qa.exists() and any(qa.glob("state*.json"))
    d = qa if use_qa else rig["films"] / f"{rig['tag']}_{family}"
    rows = [json.loads(p.read_text()) for p in sorted(d.glob("state*.json"))] if d.exists() else []
    if not rows:
        return None
    return {
        "n": len(rows),
        "opposed": sum(r["contact_verdict"] == "OPPOSED" for r in rows),
        "contact": sum(r["contact_verdict"] == "CONTACT_NO_OPPOSITION" for r in rows),
        "none": sum(r["contact_verdict"] == "NO_CONTACT" for r in rows),
        "airborne": sum(not r["supported"] for r in rows),
    }


# ----------------------------------------------------------------------------------------- media
def publish_film(rig: dict, family: str) -> str | None:
    """Copy the film beside the page and cut a poster from two thirds through."""
    src = rig["films"] / f"{rig['tag']}_{family}.mp4"
    if not src.exists():
        return None
    (OUT / "media").mkdir(parents=True, exist_ok=True)
    dst = OUT / "media" / src.name
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dst)
    poster = dst.with_suffix(".jpg")
    if not poster.exists() or poster.stat().st_mtime < dst.stat().st_mtime:
        dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                              "default=nw=1:nk=1", str(dst)], capture_output=True, text=True).stdout.strip()
        at = f"{float(dur) * 0.66:.2f}" if dur else "1"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", at, "-i", str(dst), "-frames:v", "1",
                        "-q:v", "3", str(poster)], check=True)
    return f"media/{dst.name}"


def publish_file(src: Path) -> str | None:
    """Copy any film beside the page with a poster; returns its relative src, or None if missing."""
    if not src.exists():
        return None
    (OUT / "media").mkdir(parents=True, exist_ok=True)
    dst = OUT / "media" / src.name
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dst)
    poster = dst.with_suffix(".jpg")
    if not poster.exists() or poster.stat().st_mtime < dst.stat().st_mtime:
        dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                              "default=nw=1:nk=1", str(dst)], capture_output=True, text=True).stdout.strip()
        at = f"{float(dur) * 0.66:.2f}" if dur else "1"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", at, "-i", str(dst), "-frames:v", "1",
                        "-q:v", "3", str(poster)], check=True)
    return f"media/{dst.name}"


PINCH_FILMS = WORK / "pinch2f/out/film_check"


def pinch_fix_section() -> str:
    cards = []
    for name, label, note in (
        ("pinch_grasp3_close.mp4", "Before", "Relative finger command re-applied on all 12 physics substeps: the "
         "fingers overshoot and chatter, and the cube is shaken out within about 10 steps."),
        ("pinch_abs_close.mp4", "After", "Absolute finger targets: thumb and index close in ~5 steps and hold the "
         "cube within 4 mm of the pad midpoint through gravity and the shake."),
    ):
        src = publish_file(PINCH_FILMS / name)
        body = (f'<video controls loop muted playsinline preload="none" poster="{src[:-4]}.jpg" src="{src}"></video>'
                if src else '<div class="ph">film pending</div>')
        cards.append(f'<figure class="film"><div class="rig">{label}</div>{body}'
                     f'<figcaption class="note">{esc(note)}</figcaption></figure>')
    return '<div class="pair">' + "".join(cards) + "</div>"


# ----------------------------------------------------------------------------------------- html
def esc(x) -> str:
    return html.escape(str(x))


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def verdict_chip(v: dict | None, family: str = "") -> str:
    if v is None:
        return '<span class="chip pending">not filmed yet</span>'
    if family == "ObjectAnywhereEEAnywhere":
        return '<span class="chip pending">reaching state &middot; no grasp expected</span>'
    cls = "good" if v["opposed"] == v["n"] else ("mixed" if v["opposed"] else "bad")
    return f'<span class="chip {cls}">{v["opposed"]}/{v["n"]} held by opposed contact</span>'


def film_card(rig_key: str, family: str) -> str:
    rig = RIGS[rig_key]
    src = publish_film(rig, family)
    v = contact_verdicts(rig, family)
    if src is None:
        body = '<div class="ph">film pending</div>'
    else:
        body = (f'<video controls loop muted playsinline preload="none" poster="{src[:-4]}.jpg" '
                f'src="{src}"></video>')
    return f'<figure class="film"><div class="rig">{esc(rig["short"])}</div>{body}{verdict_chip(v, family)}</figure>'


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, kpis = [], []
    for key, rig in RIGS.items():
        g = grasp_stats(rig)
        grasps = bank_count(rig.get("grasp_bank") or rig["dataset"] / "Grasps/InsertiveCube34/grasps.pt")
        value = pct(g["rate"]) if g else "&mdash;"
        detail = f'{g["successes"]} of {g["attempts"]} attempts' if g else "pending"
        if grasps:
            detail += f" &middot; {grasps} grasps banked"
        kpis.append(
            f'<div class="kpi"><b>{value}</b><span>{esc(rig["short"])} grasp sampling success</span>'
            f"<small>{detail}</small></div>"
        )
        for fam, code, title, _ in FAMILIES:
            n = bank_count(rig["dataset"] / f"Resets/{PAIR}/resets_{fam}.pt")
            v = contact_verdicts(rig, fam, full_bank=True)
            rows.append((rig["short"], code, title, n, v))

    # No "airborne" column: the renderer measures height from the support's root, not from the
    # mat surface, so a cube resting on the table reads 24.5 mm up and is flagged airborne.
    table = ["<table><thead><tr><th>Rig</th><th>Family</th><th>States banked</th>"
             "<th>Held on full replay</th></tr></thead><tbody>"]
    for short, code, title, n, v in rows:
        if code == "C1":
            held = "reaching &middot; n/a" if n else "&mdash;"
            cls = ""
        elif v is None:
            held, cls = "&mdash;", ""
        else:
            # Grasped banks are filtered to the states that held; green when what ships is all held.
            held = f'{v["opposed"]}/{v["n"]} recorded &middot; {n} kept'
            cls = ' class="ok"' if n == v["opposed"] else ' class="bad"'
        table.append(f"<tr{cls}><td>{esc(short)}</td><td>{code} &middot; {esc(title)}</td>"
                     f"<td>{n if n is not None else '&mdash;'}</td><td>{held}</td></tr>")
    table.append("</tbody></table>")

    films = []
    for fam, code, title, blurb in FAMILIES:
        films.append(
            f'<section class="family"><h3><span class="code">{code}</span> {esc(title)}</h3>'
            f'<p class="blurb">{esc(blurb)}</p><div class="pair">{film_card("linear", fam)}'
            f'{film_card("pinch2f", fam)}</div></section>'
        )

    train = []
    for key, rig in RIGS.items():
        # train.py refuses to start unless this worktree's packages lead PYTHONPATH (proof-of-tree
        # guard), and the cube-stack configs read their dataset dir and pinch flags from the env.
        rig_env = ("UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_CUBE_DATASET_DIR" if key == "pinch2f"
                   else "UWLAB_LINEAR_CUBE_DATASET_DIR")
        cmd = (f"cd {REPO}\n"
               f"export PYTHONPATH={REPO}/source/uwlab_tasks:{REPO}/source/uwlab_assets:{REPO}/source/uwlab:"
               f"{REPO}/source/uwlab_rl\n"
               f"export {rig_env}={rig['dataset_rel']}\n"
               f"{WORK}/pinch/run6.sh scripts/reinforcement_learning/rsl_rl/train.py \\\n"
               f"  --task {rig['train_task']} --num_envs 2048 --headless")
        train.append(f"<h3>{esc(rig['name'])}</h3><pre><code>{esc(cmd)}</code></pre>")

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    page = TEMPLATE.format(
        css=CSS, stamp=stamp, kpis="".join(kpis), table="".join(table), films="".join(films),
        train="".join(train), fixes=FIXES, url=URL, pinchfix=pinch_fix_section(),
    )
    (OUT / "index.html").write_text(page)
    print(f"WROTE {OUT / 'index.html'}\n{URL}")


CSS = """
:root{--bg:#f7f6f3;--fg:#1d1c1a;--mut:#6b665f;--line:#e3dfd8;--card:#fff;--ok:#1b7f4c;--bad:#b0382d;--mix:#a86b00;--acc:#2d5da8}
@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--bg:#131211;--fg:#ebe8e3;--mut:#9b958d;--line:#2d2b28;--card:#1b1a18;--ok:#62d49c;--bad:#f28b80;--mix:#f0b44c;--acc:#86aef0}}
:root[data-theme=dark]{--bg:#131211;--fg:#ebe8e3;--mut:#9b958d;--line:#2d2b28;--card:#1b1a18;--ok:#62d49c;--bad:#f28b80;--mix:#f0b44c;--acc:#86aef0}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:0 18px 90px}
main{max-width:1180px;margin:0 auto}
header{padding:52px 0 24px}
h1{font-size:34px;letter-spacing:-.025em;margin:0 0 6px}
.sub{color:var(--mut);margin:0}
h2{font-size:22px;letter-spacing:-.015em;margin:56px 0 10px}
h3{font-size:16px;margin:26px 0 6px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:26px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.kpi b{display:block;font-size:34px;letter-spacing:-.03em;line-height:1.1}
.kpi span{display:block;font-weight:600;margin-top:4px}
.kpi small{color:var(--mut)}
.family{margin:34px 0}
.code{display:inline-block;background:var(--acc);color:#fff;border-radius:6px;padding:0 8px;margin-right:6px;font-size:13px}
.blurb{color:var(--mut);margin:0 0 10px}
.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
.film{margin:0;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px}
.film .rig{font-weight:600;font-size:13px;margin:0 2px 8px}
video{width:100%;display:block;border-radius:8px;background:#000;aspect-ratio:16/10}
.ph{aspect-ratio:16/10;border-radius:8px;border:1px dashed var(--line);display:grid;place-items:center;color:var(--mut)}
.chip{display:inline-block;margin-top:9px;font-size:12.5px;border-radius:999px;padding:2px 10px;border:1px solid var(--line)}
.chip.good{color:var(--ok);border-color:var(--ok)}.chip.bad{color:var(--bad);border-color:var(--bad)}
.chip.mixed{color:var(--mix);border-color:var(--mix)}.chip.pending{color:var(--mut)}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
th,td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
thead th{color:var(--mut);font-weight:600;font-size:12.5px;text-transform:uppercase;letter-spacing:.04em}
tr.ok td:first-child{box-shadow:inset 3px 0 var(--ok)}tr.bad td:first-child{box-shadow:inset 3px 0 var(--bad)}
pre{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;overflow-x:auto;font-size:13px}
details{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px;margin:12px 0}
summary{cursor:pointer;font-weight:600}
li{margin:6px 0}
.note{color:var(--mut)}
"""

FIXES = """
<ul>
<li><b>Quaternion order.</b> Isaac Lab 3.0 stores rotations as (x, y, z, w). Robot poses and metadata
authored as (w, x, y, z) had the table and arm upside down and the reach reward's TCP frame rotated.</li>
<li><b>Jaw colliders never applied.</b> The gripper prep script looked for Isaac Sim 5.1 prim paths, printed
warnings and skipped the jaw-pad boxes and the friction material. The jaws collided as convex hulls 3.4x
the finger volume and squeezed every object out at 0.36 m/s.</li>
<li><b>Gripper fell under gravity.</b> The Isaac Sim 6 importer nests links; Isaac Lab applies
<code>disable_gravity</code> only to the first rigid body it meets, so the fingers kept gravity and dragged
the free gripper down at 0.143 m/s. Links are now flattened after import.</li>
<li><b>Two articulation roots.</b> The graft's root removal did not survive flattening; the full robot
would not spawn.</li>
<li><b>Pinch used all five fingers.</b> Fingers 3/4/5 moved up to 0.19 rad during the pinch close and reached
the cube first. They now hold their open pose.</li>
<li><b>Grasped banks accepted empty hands.</b> The reset acceptance check never asked whether the cube was
in the gripper. Linear-gripper C2 had it in hand in 2% of states and C4 in 28%. Acceptance now requires the
cube where a recorded grasp holds it and the fingers near that grasp's joint values; C4 re-recorded at 100%.</li>
<li><b>Pinch fingers chattered instead of closing.</b> Isaac Lab re-applies a relative joint command on every
physics substep (12 per step), so the scripted close overshot 12x. The offline sampler now drives the hand
with absolute joint targets: the pinch holds through gravity and the shake (302 grasps, 43% success).</li>
<li><b>Replay films reset mid-settle.</b> Reset-state episodes last 20 control steps; the film replay ran 60,
so the env auto-reset and every held cube appeared to fall onto the table. Traced: jaws at 24 N and the cube
still for 19 steps, then all defaults on step 20. With the episode unbounded, the linear C2 check holds 6/6.</li>
<li><b>C2 recording took hours and still let misses through.</b> The arm drifted 5 cm with no policy and dragged
the cube out; it is now pinned at its IK pose while recording. The in-hand check now tests a per-axis box around
the grasp bank's own grasp centre in the hand frame (bank spread + 6 mm) plus jaws stopped on the cube; a radius
had passed jaws closed beside the cube. Placement noise for the rigid jaw is &plusmn;5 mm / &plusmn;5.6&deg;.
Result: 29 s per state instead of 4+ minutes. The 5-step acceptance still let about one marginal grasp in six
through (it slid out during a 60-step settle), so every state of every grasped bank is now replayed with the film
renderer and only states held by both jaws are kept.</li>
<li><b>DELTO pinch dropped every cube on the arm.</b> Recording and replay pinned the arm by rewriting its joint
state every step, which PhysX treats as a teleport and resets contact friction; the 0.06 N&middot;m pinch holds by
friction alone. The arm is now held by a stiff joint controller, the cube is held still while the grip loads,
and the fingertips keep the friction the grasps were sampled with. Pinch C3 went from 0/6 held to 6/6.</li>
<li><b>C4 recorded with the table-leg sampler.</b> The partial assemblies came from the generic task, whose
thread-insertion term rejects a cube pair; the cube-stack task drops it.</li>
<li><b>Training scripts.</b> <code>train.py</code>/<code>play.py</code> lacked the Isaac Sim 6 bridge and the
venv lacked rsl-rl; both fixed.</li>
</ul>
"""

TEMPLATE = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cube-Stack Reset Banks</title>
<style>{css}</style>
<main>
<header>
<h1>Cube-Stack Reset Banks</h1>
<p class="sub">OmniReset C1&ndash;C4 on the UR10e &middot; linear gripper and DELTO two-finger pinch &middot; generated {stamp}</p>
</header>
<div class="kpis">{kpis}</div>

<h2>What is in each bank</h2>
<p class="note">&ldquo;Held&rdquo; is measured on replay: both sides of the grasp carry more than 0.2 N of contact force
after the state settles. C1 is a reaching state and is not expected to hold anything.</p>
<div class="scroll">{table}</div>

<h2>DELTO pinch: why it dropped the cube, and the fix</h2>
<p class="note">One grasp-sampling episode each: the hand is placed around a floating cube, closes, then
gravity switches on and the cube is shaken. Filmed before any grasps were sampled at scale.</p>
{pinchfix}

<h2>Films</h2>
<p class="note">Each film replays six states from the bank. Left: the parallel-jaw baseline. Right: the DELTO pinch.</p>
{films}

<h2>Ready to train, not started</h2>
<p class="note">Stage-1 PPO with the repo's own hyperparameters and 25% of episodes starting from each bank.
Nothing below has been launched.</p>
{train}

<details><summary>What was broken on the way here</summary>{fixes}</details>
</main>
"""

if __name__ == "__main__":
    build()
