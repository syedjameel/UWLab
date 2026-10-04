"""Regenerate the DELTO pinch report from logs and rendered clips.

A generator rather than a hand-written page, because the inputs keep moving all night: the probe is
final, training is not, and the videos arrive last. Re-running this at any point produces the page as
it stands, so the 06:00 version is a command rather than a scramble.

Reads:
  * cube-stack/logs/probe_local.log        -- the endurance sweep (final)
  * cube-stack/logs/train_local.log        -- the 4070 Ti hedge run
  * cube-stack/logs/train_h100.log         -- the H100 curriculum, if it exists
  * cube-stack/results/**/*.mp4            -- rendered clips

Writes a self-contained page to ~/reports/delto_pinch/. Videos are copied in beside it and
referenced relatively; everything else is inlined.
"""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

CS = Path("/home/mattephi/Work/Rubitek/cube-stack")
OUT = Path.home() / "reports" / "delto_pinch"
LOGS = CS / "logs"

COND_RE = re.compile(r"^=== COND (\S+) effort=x(\S+) tuck=(\S+) bias=(\S+)")
PROBE_RE = re.compile(
    r"PROBE (\S+) step=\s*(\d+) in_hand=([\d.]+) finger_dq_rad=([+-][\d.]+) "
    r"tip_force_N=([\d.]+|nan) cube_to_tip=([\d.]+) cube_dz=([+-][\d.]+)"
)
ITER_RE = re.compile(r"Learning iteration (\d+)/(\d+)")
REW_RE = re.compile(r"Mean reward:\s*([-\d.]+)")
SUCC_RE = re.compile(r"Metrics/task_(\d)_success_rate:\s*([\d.]+)")
STAGE_RE = re.compile(r"STAGE_(\w+)_START iters=(\d+) families=(\S+)")
EJECT_M = 0.5
# The effort scale the training runs actually use. x40 scores highest in the sweep but by less than
# the measured noise band, at a fingertip force past the DG-5F's rated grip.
CHOSEN_EFF = "20"


def read(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""


def parse_probe(text: str):
    conds, rows = {}, {}
    for line in text.splitlines():
        m = COND_RE.match(line)
        if m:
            conds[m.group(1)] = (m.group(2), m.group(3), m.group(4))
            continue
        m = PROBE_RE.search(line)
        if m:
            rows.setdefault(m.group(1), {})[int(m.group(2))] = (
                float(m.group(3)), float(m.group(5)) if m.group(5) != "nan" else float("nan"),
                float(m.group(6)), float(m.group(7)),
            )
    out = []
    for name, steps in rows.items():
        eff, tuck, bias = conds.get(name, ("?", "?", "?"))
        s20, s150 = steps.get(20), steps.get(150)
        if not s150:
            continue
        out.append({
            "name": name, "eff": eff, "tuck": tuck, "bias": bias,
            "force20": s20[1] if s20 else float("nan"),
            "hold20": s20[0] if s20 else float("nan"),
            "hold150": s150[0], "dz": s150[3],
            "ejected": s150[2] > EJECT_M,
            "curve": {k: v[0] for k, v in sorted(steps.items())},
        })
    return out


def parse_training(text: str):
    """Walk the log in order, pairing each iteration with the metrics printed after it."""
    series, stages = [], []
    cur = None
    for line in text.splitlines():
        m = STAGE_RE.search(line)
        if m:
            stages.append({"stage": m.group(1), "iters": int(m.group(2)), "families": m.group(3),
                           "at": len(series)})
        m = ITER_RE.search(line)
        if m:
            cur = {"it": int(m.group(1)), "total": int(m.group(2)), "reward": None, "succ": {}}
            series.append(cur)
            continue
        if cur is None:
            continue
        m = REW_RE.search(line)
        if m:
            cur["reward"] = float(m.group(1))
        m = SUCC_RE.search(line)
        if m:
            cur["succ"][int(m.group(1))] = float(m.group(2))
    return series, stages


def c3_milestone_html(series) -> str:
    """The C3 milestone paragraph, COMPUTED from the log rather than written by hand.

    These figures moved from 2 non-zero readings to 4 inside half an hour, and each time the
    hardcoded prose went stale and had to be hand-corrected against the log. Deriving them means a
    regeneration is correct without intervention -- which matters most for the unattended run.
    """
    vals = [s["succ"].get(1) for s in series if 1 in s["succ"]]
    nz = [(i + 1, v) for i, v in enumerate(vals) if v]
    if not nz or not vals:
        return ""
    clusters: list[list] = []
    for idx, v in nz:
        if clusters and idx == clusters[-1][-1][0] + 1:
            clusters[-1].append((idx, v))
        else:
            clusters.append([(idx, v)])
    parts = []
    for c in clusters:
        rng = str(c[0][0]) if len(c) == 1 else f"{c[0][0]}&ndash;{c[-1][0]}"
        vs = " / ".join(f"<strong>{v:.4f}</strong>" for _, v in c)
        parts.append(f"{vs} at reading{'' if len(c) == 1 else 's'} {rng}")
    multi = len(clusters) > 1
    return (
        f'<p class="lede">C3 left zero{", and has done so in more than one window" if multi else ""}. '
        f'It logged {"; then ".join(parts)} &mdash; {len(nz)} non-zero out of {len(vals)} readings '
        f'({100.0 * len(nz) / len(vals):.1f}%), peaking at {max(v for _, v in nz) * 100:.2f}% success. '
        f'Stated precisely, because the temptation is to overstate it: '
        f'{"separated clusters are <em>recurrence</em> rather than a single fluke, but " if multi else ""}'
        f'this is not a trend and C3 is not a solved family. What it overturns is the categorical claim '
        f'that C3 has never scored above zero, which had held for this task&rsquo;s entire history '
        f'before tonight&rsquo;s fixes.</p>'
    )


def svg_chart(series, key, label, color, width=760, height=220):
    """Inline SVG line chart. No JS, no CDN -- the page must stand on its own."""
    pts = [(i, s) for i, s in enumerate(series) if s is not None]
    if len(pts) < 2:
        return f'<p class="muted">Not enough data yet for {html.escape(label)}.</p>'
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    lo, hi = min(ys), max(ys)
    # A constant series drawn on a full-height 0..1 axis is mostly empty space and reads as "no data
    # recorded" -- when the truth is "recorded every iteration and it never moved". That is a
    # different and much stronger claim, so say it in words and give the panel less room.
    flat = (hi - lo) < 1e-9
    if flat:
        label = f"{label} — flat at {lo:.4f} across all {len(pts)} logged iterations"
        height = 96
        hi = lo + 1.0
    pad = 34
    def X(i): return pad + (i - min(xs)) / max(1, (max(xs) - min(xs))) * (width - pad - 12)
    def Y(v): return height - pad - (v - lo) / (hi - lo) * (height - pad - 14)
    path = " ".join(f"{'M' if k == 0 else 'L'}{X(i):.1f},{Y(v):.1f}" for k, (i, v) in enumerate(pts))
    grid = "".join(
        f'<line x1="{pad}" y1="{Y(lo + (hi - lo) * f):.1f}" x2="{width - 12}" '
        f'y2="{Y(lo + (hi - lo) * f):.1f}" class="grid"/>'
        f'<text x="4" y="{Y(lo + (hi - lo) * f) + 4:.1f}" class="tick">{lo + (hi - lo) * f:.3f}</text>'
        for f in (0.0, 0.5, 1.0)
    )
    return (f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
            f'aria-label="{html.escape(label)}">{grid}'
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>'
            # Title anchored to the RIGHT edge: left-anchored at x=pad it overprinted the topmost
            # y-axis tick label on every chart.
            f'<text x="{width - 12}" y="12" text-anchor="end" class="ttl">{html.escape(label)}</text></svg>')


# results/ holds months of earlier runs -- near-goal rollouts, failure reels, old checkpoint films.
# Globbing it wholesale pulled in 16 unrelated clips (~230 MB) and would have turned the report into
# a junk drawer of unlabelled video. Only tonight's artefacts belong here, named by the two scripts
# that produce them: render_reset_pair.sh -> reset_eff<N>, render_films_local.sh -> video_pinch_<RT>.
# An EXPLICIT allowlist of directories, not a prefix match. A prefix of "film_" silently swept in
# film_test/ -- a throwaway camera-sensor probe whose 160 frames are entirely blank sky -- and
# published it beside the real clip. Everything rendered through the Kit viewport (reset_eff*,
# video_*) is blank for the same reason. A clip earns its place here only after being looked at.
# film_mixed_c3 and film_long400 are produced by after_mixed.sh / harvest_ck.sh, which BOTH end by
# regenerating this report. collect_videos() walks this tuple explicitly, so a clip missing from it is
# silently absent from the page -- no error, no warning, just a finished-looking report without its
# newest evidence. Register the directory when the harvest is written, not after noticing the gap.
CLIP_DIRS = ("film_mixed2", "film_mixed_c3", "film_long400", "film_bonus_c3", "film_carry_c3",
             "film_robot_nullgrip", "film_robot_c3", "film_robot_c4link",
             "film_reset_ObjectAnywhereEEAnywhere", "film_reset_ObjectRestingEEGrasped",
             "film_reset_ObjectAnywhereEEGrasped", "film_reset_ObjectPartiallyAssembledEEGrasped",
             "film_policy_c3", "film_policy_c4", "film_trace", "film_eff1", "film_eff20",
             "film_h100_c4", "film_h100_c3")

CAPTIONS = {
    "film_mixed2": "THE BEST POLICY OF THE NIGHT ON C3 (model_999, mixed C4+C3 curriculum, OmniReset's "
                   "published objective with the added bonus switched off). It ends the episode a median "
                   "146.2 mm from the goal, against 237.9 mm for the untrained-objective baseline and "
                   "185-190 mm for the two intermediate policies -- the best endpoint measured on this "
                   "task. It holds the cube 96 of 150 control steps at ~10.9 N, gets 29/64 environments "
                   "within 100 mm and 7/64 within 50 mm, and its best single environment reaches "
                   "37.7 mm. IT STILL NEVER SEATS THE CUBE: success is 0/64, the gate is 5 mm, and this "
                   "clip ends with the cube on the table rather than on the stack.",
    "film_mixed_c3": "THE MIXED-CURRICULUM POLICY ON C3, resumed from the C3-only checkpoint onto the "
                     "50/50 C4+C3 mixture with OmniReset's published objective (bonus weight 0). The "
                     "mixture was tried because C3 alone never shows the agent a seated cube, so it "
                     "cannot learn what finishing looks like. Read the numbers in the section above "
                     "before reading anything into the motion: on C3 this policy is scored by "
                     "task_1_success_rate, which stayed at exactly 0.0000.",
    "film_long400": "THE BEST-GRIPPING POLICY OF THE NIGHT (long_carry model_400, C3-only, hold/carry "
                    "bonus at 0.5 before it was switched off). The two-finger gate closes in 64/64 "
                    "environments at ~11.6 N and holds ~111 of 150 control steps; the cube is lifted "
                    "and moved ~236 mm. It never reaches the goal: net progress is +9.6 mm out of "
                    "248 mm and success is 0/64. A solved grasp attached to an unsolved task.",
    # The clip that best represents where this actually got to, and it is not a success. Env 20 of 64
    # is the MEDIAN pick, not the best one -- choosing a flattering environment is how an earlier set
    # of clips came to misrepresent the policy.
    "film_bonus_c3": "THE BEST POLICY SO FAR, ON C3 (model_199, with the hold/carry bonus at weight 0.5). "
                     "A real grip: the two-finger gate closes in 64/64 environments at a median 11.6 N and "
                     "stays shut for 111 of 150 control steps, against 50/150 without the bonus and "
                     "0.0000 N for every policy before tonight. The arm lifts the cube and moves it "
                     "236 mm. WATCH WHERE IT ENDS: the cube is never brought to the blue goal cube, the "
                     "hand lets go, and the episode scores 0/64. Net progress toward the goal is "
                     "+9.6 mm out of 248 mm. This is a working GRASP, not a working task.",
    "film_eff1": "The SHIPPED effort cap (x1), null action: the hand cannot hold the cube and drops it.",
    "film_eff20": "Identical reset and identical null action at effort x20: the grasp survives the episode.",
    # Labelled for what it is. Photoreal capture was impossible on this machine tonight (Isaac Sim 6
    # aims the camera correctly and renders empty sky; Isaac Sim 5.1 crashes in libomni.hydra.rtx;
    # the renderer reports IOMMU enabled, a known cause and a BIOS setting). Every coordinate drawn
    # here was measured by the physics engine -- but it is a plot, and calling it a render would be
    # a lie about what you are looking at.
    "film_policy_c3": "THE TRAINED POLICY on C3 (ObjectAnywhereEEGrasped) as a state trace — the family "
                      "that must be carried ~210 mm. Read it against the null-action clip: doing nothing "
                      "keeps real fingertip contact (median 11.2 N) in 26/64 environments at step 150 — "
                      "the 72% once quoted here came from the in_hand distance proxy, which overstates "
                      "it; measured contact is 41%. Either way the environment can hold this grasp and "
                      "the loss is the policy's own doing. The "
                      "robot render above shows the mechanism this plot only implies — the hand opens "
                      "at median step 1 in 64/64 environments. Measured state trace, not a render.",
    "film_policy_c4": "THE TRAINED POLICY on C4 (ObjectPartiallyAssembledEEGrasped) as a state trace. "
                      "CAPTION CORRECTED: this previously said 'the carried cube against the goal' and "
                      "called the right panel the honest measure of whether the policy does the task. "
                      "Neither is true — the cube starts seated and never moves (median displacement "
                      "0.00000 m), so the flat line on the right is the reset, not an approach. Kept "
                      "because the flatness is exactly the evidence, once you know what it means.",
    # THE FOUR OMNIRESET FAMILIES, under a NULL ACTION -- what the ENVIRONMENT does before any policy
    # touches it. Recorded with no checkpoint at all, so nothing here is policy behaviour. These also
    # correct an earlier claim on this page: the reset grasp is NOT self-sustaining. Under no policy
    # the two-finger contact gate decays in every grasped family, so "the environment holds the cube
    # and the policy throws it away" overstated the environment's part.
    "film_reset_ObjectAnywhereEEAnywhere":
        "C1 ObjectAnywhereEEAnywhere — NULL ACTION, NO POLICY. Cube and hand both start anywhere: the "
        "hand is not holding anything and never closes on anything. Contact gate 0/64 at every step, "
        "0.00 N at the fingertips. This is the family that genuinely requires reach-then-grasp, and "
        "nothing in these recordings attempts it.",
    "film_reset_ObjectRestingEEGrasped":
        "C2 ObjectRestingEEGrasped — NULL ACTION, NO POLICY. The cube starts resting on the table with "
        "the hand closed on it: gate open 57/64 at step 0 at a median 13.75 N. By step 150 it is open "
        "in only 4/64 — the grasp decays on its own, with no policy acting.",
    "film_reset_ObjectAnywhereEEGrasped":
        "C3 ObjectAnywhereEEGrasped — NULL ACTION, NO POLICY. The real carry task: the cube starts "
        "pinched in mid-air and must be carried ~210 mm to the stack. The reset grasp is the strongest "
        "of the four (gate 64/64 at step 0, median 11.68 N) but still decays to 15/64 by step 150, and "
        "the cube falls a median 85.6 mm doing nothing. Compare the trained-policy clip above, where "
        "the gate is open in 0/64 environments at every step.",
    "film_reset_ObjectPartiallyAssembledEEGrasped":
        "C4 ObjectPartiallyAssembledEEGrasped — NULL ACTION, NO POLICY. The cube starts already seated "
        "on the stack and still nominally held: gate 49/64 at step 0, 2/64 by step 150, and the cube "
        "falls 34.0 mm. Note the success count goes 6/64 at step 0 to 0/64 at step 149 with NO policy "
        "at all — physics alone loses it. This is the family whose reset distribution was previously "
        "misreported on this page as policy performance.",
    "film_carry_c3": "THE POLICY THAT ACTUALLY GRIPS AND CARRIES (model_899, 900 iterations). "
                     "Measured with the scene's own contact sensor, not a proxy: the two-finger gate "
                     "is open in 64/64 environments at step 0 and 50/64 at step 150, held for a median "
                     "150 of 150 steps, at 10.8-13.0 N of fingertip force — against 0.0000 N and 0/64 "
                     "for every earlier policy. It carries the cube 234.6 mm → 146.2 mm toward the "
                     "goal. Then it stops: watch the hand hold the orange cube beside the blue one and "
                     "never open. The plateau is real (last 30 steps move 15.6 mm) and the cause is a "
                     "reward I added — holding pays 0.189/step at that standoff and 0.399/step even "
                     "seated, while releasing pays exactly 0 until the cube lands inside 5 mm. "
                     "success_reward earned 0.0007 against hold_bonus's 0.1522, a factor of 217.",
    "film_robot_nullgrip": "WHAT A REAL GRASP LOOKS LIKE — and this one is NOT the policy. A null "
                           "action at effort ×20: the fingers stay closed on the cube and the contact "
                           "sensor reads a median 11.2 N at the fingertips, with 26/64 environments "
                           "still in contact at step 150. Compare the two policy clips below, where "
                           "the fingers hang open and the sensor reads 0.0000 N. The environment can "
                           "hold this cube; the trained policy simply never grips it.",
    # THE ACTUAL ROBOT, rendered from recorded link poses. These two replace the scatter plots as the
    # lead media: Matte asked to see the robot perform the task and was given plots instead, and the
    # plots are precisely what let "the policy carries the cube" survive unchallenged for hours.
    "film_robot_c4link": "THE ROBOT ON C4, rendered from the recorded pose of every link. Watch what "
                         "does NOT happen: the orange cube is already stacked on the blue one in the "
                         "first frame and never moves, and the contact sensor reads 0.0000 N — the hand "
                         "is not touching it either, fingertips 56–62 mm from the centre of a 34 mm "
                         "cube. So the policy neither carries nor releases anything here; it withdraws "
                         "without toppling a stack it was never holding. That is still worth something: "
                         "a null action destroys the same stack in 64/64 environments. But it is a long "
                         "way from 'carries the cube to the goal'.",
    "film_robot_c3": "THE ROBOT ON C3, the family that genuinely requires carrying the cube ~210 mm. "
                     "The opening is played at 1/5 speed because the whole event lasts a third of a "
                     "second: the cube starts pinched between the fingers, the hand opens, and the cube "
                     "drops straight to the table — it never travels toward the blue goal cube, which "
                     "stays in frame throughout so you can see how far away it remains. And the contact "
                     "sensor is blunter than the geometry: measured fingertip force through this rollout "
                     "is 0.0000 N at every step, so the hand never grips the cube at all — 'opens' and "
                     "'lets go' are both too generous. Compare the null-action clip at the top of this "
                     "section, where the same hand holds the same cube at a median 11.2 N. 0/64 success, "
                     "every rollout.",
    # The SECOND run's policy, rolled out through the identical harness. Stated with the seed caveat
    # attached, because "a second run does it too" invites a stronger reading than was actually tested.
    "film_h100_c4": "THE SECOND TRAINING RUN on C4, rolled out locally from its H100 checkpoint "
                    "(model_225, about a quarter of the first run's training). Same 5 mm tolerance, "
                    "same harness, same statistic: 30/64 succeed at a median 5.4 mm. Both runs use "
                    "seed 42, so this is a different training trajectory — 2048 environments against "
                    "1024, a different stage-A budget, different hardware — and NOT a seed-independent "
                    "replication. Measured state trace, not a render.",
    "film_h100_c3": "THE SECOND TRAINING RUN on C3, and the same failure. The cube is gone from the "
                    "hand inside the first ~50 steps; the arm then withdraws (watch the tips leave "
                    "the frame) and the cube sits where it fell for the rest of the episode. 0/64, "
                    "flat at ~156 mm. The first run ends at 155.6 mm — two different training "
                    "trajectories failing at nearly the same distance. Measured state trace.",
    "film_trace": "MEASURED STATE TRACE, not a render: fingertip and cube positions through one "
                  "C3 episode under a null action, shipped effort cap vs x20, with the fraction of "
                  "64 environments still holding the cube.",
}


def collect_videos():
    # DECLARATION order, not alphabetical. Sorting the glob put the two robot renders fifth and
    # sixth, below the scatter plots -- the exact inversion this page was rewritten to correct, since
    # leading with plots is what let a false claim about C4 stand unchallenged for hours.
    vids = []
    for name in CLIP_DIRS:
        d = CS / "results" / name
        if not d.is_dir():
            continue
        for mp4 in sorted(d.glob("**/*.mp4")):
            if mp4.stat().st_size > 2048:
                vids.append(mp4)
    return vids


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    probe = parse_probe(read(LOGS / "probe_local.log"))
    probe.sort(key=lambda r: float(r["eff"]))
    local_series, local_stages = parse_training(read(LOGS / "train_local.log"))
    h100_series, h100_stages = parse_training(read(LOGS / "train_h100.log"))

    vids = []
    for src in collect_videos():
        dst = OUT / f"{src.parent.name}_{src.name}"
        try:
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                shutil.copy2(src, dst)
            vids.append((dst.name, src.parent.name))
        except OSError:
            pass

    rows = "".join(
        f'<tr class="{"eject" if r["ejected"] else ""}">'
        f'<td class="mono">{html.escape(r["name"])}</td><td class="num">&times;{html.escape(r["eff"])}</td>'
        f'<td class="num">{html.escape(r["tuck"])}</td><td class="num">{html.escape(r["bias"])}</td>'
        f'<td class="num">{r["force20"]:.2f}</td><td class="num">{r["hold20"]:.3f}</td>'
        f'<td class="num strong">{r["hold150"]:.3f}</td><td class="num">{r["dz"]:+.3f}</td>'
        f'<td>{"ejected cubes" if r["ejected"] else "ok"}</td></tr>'
        for r in probe
    )

    def train_block(name, series, stages):
        if not series:
            return f'<p class="muted">No {html.escape(name)} data yet.</p>'
        last = series[-1]
        sl = ", ".join(f"{s['stage']} ({s['families'].strip('[]')})" for s in stages)
        c0 = svg_chart([s["succ"].get(0) for s in series], "c0", "C4 success rate (task_0)", "#2f81f7")
        c1 = svg_chart([s["succ"].get(1) for s in series], "c1", "C3 success rate (task_1)", "#d29922")
        cr = svg_chart([s["reward"] for s in series], "r", "Mean reward", "#3fb950")
        return (f'<p><strong>{html.escape(name)}</strong> &mdash; {len(series)} iterations logged, '
                f'latest {last["it"]}/{last["total"]}. Stages: {html.escape(sl) or "n/a"}.</p>'
                f'{c0}{c1}{cr}')

    vid_html = "".join(
        f'<figure><video controls preload="metadata" src="{html.escape(fn)}"></video>'
        f'<figcaption>{html.escape(CAPTIONS.get(tag, tag))}</figcaption></figure>' for fn, tag in vids
    ) or '<p class="muted">No clips rendered yet.</p>'

    # Report the configuration actually TRAINED ON, not the sweep's top score. x40 scores highest
    # (0.625) but beats x20 by less than the measured noise band while pushing fingertip force past
    # the DG-5F's rated grip, so x20 is what the training runs use. Quoting the best number in the
    # table as though it were the chosen setting would be a small lie in the first sentence.
    chosen = next((r for r in probe if r["eff"] == CHOSEN_EFF and r["tuck"] in ("0", "0.0")), None)
    base = next((r for r in probe if r["eff"] == "1" and r["tuck"] in ("0", "0.0")), None)
    headline = (
        f'Grip retention at step 150 went from <strong>{base["hold150"]:.3f}</strong> to '
        f'<strong>{chosen["hold150"]:.3f}</strong> once the hand\'s effort cap was lifted '
        f'(&times;{chosen["eff"]}, the setting both training runs use) &mdash; and this now rests on '
        f'<strong>measured fingertip force</strong>, not a distance proxy: under a null action the '
        f'tips exert a median <strong>2.36&nbsp;N</strong> at &times;1 against '
        f'<strong>11.2&nbsp;N</strong> at &times;20, with envs still in contact at step&nbsp;150 going '
        f'<strong>1/64&nbsp;&rarr;&nbsp;26/64</strong> and episode impulse '
        f'<strong>45&nbsp;&rarr;&nbsp;416&nbsp;N&middot;steps</strong>.'
        if chosen and base else "Probe data pending."
    )
    # THE GRASP IS FIXED, AND THAT LEADS. Computed from the acceptance rollout rather than
    # transcribed: every hardcoded C3/C4 figure on this page went stale within the hour at least
    # once tonight, so anything quoted in the first paragraph is read from the .npz at build time.
    # Column order follows DELTO_ALL_TIP_NAMES -- (1_tip, 5_tip) + (2_tip, 3_tip, 4_tip) -- so the
    # thumb is column 0 and the INDEX is column 2. Column 1 is rl_dg_5_tip, which is undriven;
    # using it once made the two-finger gate look unreachable even under a null action.
    grasp_lede = ""
    try:
        import numpy as _np

        # Lead with the BEST policy, not the baseline. This block originally read poses_ablc3.npz --
        # the weight-0 C3-only baseline -- so the page opened on the worst trained result (237 -> 238 mm,
        # release at step 12) while the best one (146.2 mm) sat several sections further down. Falls
        # back to the baseline only if the final rollout is missing.
        _best = CS / "results" / "poses_mixed2.npz"
        _d = _np.load(_best if _best.exists() else CS / "results" / "poses_ablc3.npz", allow_pickle=True)
        _c, _cp, _gp, _su = _d["contact"], _d["cube_pos"], _d["goal_pos"], _d["success"]
        _E = _cp.shape[1]
        _tot = _c.sum(2)
        _gate = (_c[:, :, 0] > 0.2) & (_c[:, :, 2] > 0.2)
        _dist = _np.linalg.norm(_cp - _gp, axis=2)
        # Sample at 149, NEVER at or past 160: beyond the time-out the reset is what gets measured,
        # and that has corrupted four separate measurements on this task.
        _f0 = float(_np.median(_tot[0]))
        _g0, _g149 = int(_gate[0].sum()), int(_gate[149].sum())
        _rel = int(_np.median(_np.asarray(
            [_np.argmax(~_gate[:150, i]) if (~_gate[:150, i]).any() else 150 for i in range(_E)])))
        _z = _cp[:150, :, 2]
        _fell = int((_z.min(0) < 0.04).sum())
        _d0, _d149 = float(_np.median(_dist[0])) * 1000, float(_np.median(_dist[149])) * 1000
        _s149 = int(_su[149].sum())
        grasp_lede = (
            f'<strong>The hand now grips.</strong> On C3 &mdash; the family that actually requires '
            f'carrying the cube ~210&nbsp;mm &mdash; the policy closes the two-finger gate in '
            f'<strong>{_g0}/{_E}</strong> environments at a median <strong>{_f0:.1f}&nbsp;N</strong> '
            f'of fingertip force, measured by the scene&rsquo;s own contact sensor. Every earlier '
            f'policy on this task read <strong>0.0000&nbsp;N</strong> and <strong>0/64</strong>. '
            f'That came from the <em>control path</em>, not from a reward change: hand action scale '
            f'0.1&nbsp;&rarr;&nbsp;0.02&nbsp;rad/step and <code>entropy_coef</code> '
            f'0.006&nbsp;&rarr;&nbsp;0.001. '
            f'<strong>It carries the cube, and it still never finishes.</strong> The grip reopens at a median step '
            f'<strong>{_rel}</strong> and is still closed at step&nbsp;149 in only '
            f'<strong>{_g149}/{_E}</strong> environments, the cube falls to the table in '
            f'<strong>{_fell}/{_E}</strong> '
            f'environments, and cube-to-goal closes from '
            f'<strong>{_d0:.0f}&nbsp;mm to {_d149:.0f}&nbsp;mm</strong>. Success is '
            f'<strong>{_s149}/{_E}</strong> &mdash; the success gate is 5&nbsp;mm, so the best policy '
            f'of the night stops about <strong>{_d149 / 5:.0f}&times;</strong> short of it, opening its '
            f'hand and leaving the cube on the table beside the stack. Grasping is solved and transport '
            f'is real; placement is not.</p><p class="lede">'
        )
    except (OSError, ValueError, KeyError, IndexError, ZeroDivisionError, ImportError) as exc:
        # Loud, for the same reason as the mixed-run block below: this computes the page's HEADLINE.
        # If it blanks silently the report still builds and still looks finished, just without its
        # lead result -- which is the worst of both outcomes during an unattended regeneration.
        print(f"WARNING: grasp lede SKIPPED -- {type(exc).__name__}: {exc}", flush=True)

    # The mixed-curriculum resume, computed from its log at build time rather than transcribed, so
    # the automatic post-run regeneration renders whatever actually happened. Deliberately reports
    # task_1 (C3) as the headline: on C4 the cube starts already seated, so a C4 number says only
    # whether the policy avoided wrecking a finished stack.
    mixed_html = ""
    try:
        _mt = (CS / "logs" / "mixed_resume.log").read_text(errors="ignore")
        _mg = lambda k: [float(x) for x in re.findall(rf"{re.escape(k)}:\s+([\d.eE+-]+)", _mt)]
        _mit = [int(x) for x in re.findall(r"Learning iteration (\d+)/", _mt)]
        _gc, _s0, _s1 = _mg("Episode_Reward/grasp_contact"), _mg("Metrics/task_0_success_rate"), _mg("Metrics/task_1_success_rate")
        # rsl_rl prints the NEXT "Learning iteration N/1000" header before that iteration's metrics
        # block is written, so the iteration list runs one entry ahead of every metric array. Indexing
        # by len(_mit) therefore walks off the end of the metrics -- an IndexError that the except
        # below used to swallow, blanking this whole section while generation reported success.
        _n = min(len(_mit), len(_gc), len(_s0), len(_s1))
        _mit, _gc, _s0, _s1 = _mit[:_n], _gc[:_n], _s0[:_n], _s1[:_n]
        if _mit and _gc and _s1:
            # Skip index 0: the first iteration after a checkpoint load reports cold statistics
            # (grasp_contact 0.0029 against a steady 0.020), which briefly looked like the mixture
            # destroying the grasp. It was an artifact, and quoting it as a low would repeat that.
            _w = _gc[1:] or _gc
            _rows = "".join(
                f"<tr><td>iteration {_mit[i]}</td><td class='num'>{_gc[i]:.4f}</td>"
                f"<td class='num'>{_s0[i]:.4f}</td><td class='num'>{_s1[i]:.4f}</td></tr>"
                for i in [j for j in (5, len(_mit) // 4, len(_mit) // 2, len(_mit) - 1) if 0 < j < len(_mit)]
            )
            _c3nz = sum(1 for x in _s1 if x > 0)
            # Trend from FIRST-TEN vs LAST-TEN means, never from hand-picked rows. The four sampled
            # table rows above descend by chance, and reading a "slow erosion of the grasp" off them
            # put a false claim on this page once already -- with numbers printed beside it that said
            # the opposite. A band this noisy (spread ~0.016 on a ~0.018 mean) needs averaging.
            _f10 = sum(_w[:10]) / min(10, len(_w))
            _l10 = sum(_w[-10:]) / min(10, len(_w))
            _trend = ("essentially flat" if 0.85 <= (_l10 / _f10 if _f10 else 1) <= 1.15
                      else "declining" if _l10 < _f10 else "rising")
            _trend = f"{_trend} ({(_l10 / _f10 - 1) * 100:+.1f}% across {len(_w)} iterations)" if _f10 else _trend
            mixed_html = (
                '<h2>The mixed curriculum: resuming the solved grasp where seating is visible</h2>'
                f'<p>Running from iteration {_mit[0]}, resumed from the C3-only checkpoint onto the '
                f'50/50 C4+C3 mixture with the bonus switched off. {len(_mit)} iterations so far.</p>'
                '<table><thead><tr><th>mixed resume</th><th class="num">grasp_contact</th>'
                '<th class="num">C4 success</th><th class="num">C3 success</th></tr></thead>'
                f'<tbody>{_rows}</tbody></table>'
                f'<p>The grasp survived the switch. <code>grasp_contact</code> averages '
                f'<strong>{_f10:.4f}</strong> over the first ten iterations after the resume and '
                f'<strong>{_l10:.4f}</strong> over the last ten &mdash; {_trend} &mdash; against ~0.022 '
                f'on C3-only and the ~0.011 a healthy mixture would show if half the environments had '
                f'no cube to hold. An apparent collapse to 0.0029 at the first post-resume iteration '
                f'was cold statistics, not a real loss.</p>'
                f'<p class="muted">The four sampled rows above happen to descend, and an earlier '
                f'version of this page generalised a &ldquo;slow erosion of the grasp&rdquo; from them. '
                f'That was reading noise: across the full series the values sit in a '
                f'{min(_w):.4f}&ndash;{max(_w):.4f} band with no systematic direction. Four '
                f'hand-picked points are not a trend, which is the same mistake as comparing two '
                f'single iterations &mdash; committed twice on this page now.</p>'
                f'<p class="retract"><strong>THE MIXED CURRICULUM ALSO FAILED A PRE-COMMITTED TEST, and '
                f'this hypothesis is closed.</strong> The bar was fixed at iteration&nbsp;466, before '
                f'the deciding data: by iteration&nbsp;700, C4 success had to exceed <strong>0.30</strong> '
                f'and C3 had to log at least one non-zero reading. Actual at iteration&nbsp;700: C4 '
                f'rolling-50 <strong>0.0030</strong> &mdash; a hundredfold miss, falling monotonically '
                f'from 0.0168 &mdash; and C3 non-zero in <strong>0 of 301</strong> readings. Since ~0.61 '
                f'is available on C4 for doing nothing at all, the policy is <strong>203&times; below '
                f'the reset distribution</strong>: it is not failing to seat the cube, it is actively '
                f'destroying stacks that arrive intact. The run was stopped at iteration&nbsp;706 rather '
                f'than spend its last 300 iterations confirming a result already decided.</p>'
                f'<p class="retract"><strong>BUT THE BAR TESTED THE WRONG THING, AND THE ROLLOUTS SAY '
                f'SO.</strong> Scored on behaviour instead of on two success counters, <em>both</em> '
                f'trained policies land in the same place: final cube-to-goal '
                f'<strong>185.7&nbsp;mm</strong> for the mixed policy and <strong>190.1&nbsp;mm</strong> '
                f'for the best C3-only checkpoint, against <strong>237.9&nbsp;mm</strong> for the '
                f'weight-0 baseline. That is ~50&nbsp;mm of real improvement from training, and a '
                f'4.4&nbsp;mm gap between the two that is well inside the noise. The mixed policy also '
                f'holds the cube <strong>127 of 150</strong> control steps against 50 for the baseline '
                f'and first releases at step <strong>51</strong> rather than 12. Success is still 0/64 '
                f'everywhere, so nothing is solved.</p>'
                f'<p class="retract"><strong>CORRECTION &mdash; an earlier version of this paragraph '
                f'called the mixed policy &ldquo;the best transport result of the night&rdquo; on a net '
                f'figure of +68.4&nbsp;mm. That comparison was confounded and is withdrawn.</strong> '
                f'The three rollouts do not start from the same distance: median cube-to-goal at step 0 '
                f'is 254.0&nbsp;mm (mixed), 208.3&nbsp;mm (C3-only) and 237.3&nbsp;mm (baseline), with '
                f'an interquartile spread near 100&nbsp;mm in each. <em>Net</em> transport therefore '
                f'rewards whichever policy happened to start furthest out. Compare the endpoint, not '
                f'the distance travelled.</p>'
                f'<p>What survives that correction is the criticism of my own bar. I built it on C4 '
                f'success &mdash; the same reset-distribution trap this page documents twice elsewhere '
                f'&mdash; and on C3 success, a 5&nbsp;mm gate roughly 37&times; away. Both were nearly '
                f'guaranteed to read zero while ~50&nbsp;mm of genuine progress happened underneath '
                f'them. <strong>I stopped that run at iteration 709 of 1000 on those counters</strong>, '
                f'and a bar should measure the thing you actually care about, which here was distance, '
                f'not a binary gate the policy was never close to tripping.</p>'
                f'<p><strong>So the run was restarted and carried to iteration 1000, and it kept '
                f'improving &mdash; which settles the point.</strong> Measured through the same '
                f'contact harness, <code>model_999</code> ends the episode a median '
                f'<strong>146.2&nbsp;mm</strong> from the goal. Against 237.9&nbsp;mm for the '
                f'weight-0 baseline, 190.1&nbsp;mm for the best C3-only checkpoint and 185.7&nbsp;mm '
                f'for the 709-iteration mixed policy, that is the best endpoint this task has produced. '
                f'It holds the cube <strong>96 of 150</strong> steps at ~10.9&nbsp;N, puts '
                f'<strong>29/64</strong> environments within 100&nbsp;mm and <strong>7/64</strong> '
                f'within 50&nbsp;mm (against 19 and 3 for the baseline), and its best single environment '
                f'reaches <strong>37.7&nbsp;mm</strong>. Quote the endpoint, not the +109.5&nbsp;mm of '
                f'net transport it also shows &mdash; this rollout happened to start furthest out '
                f'(255.7&nbsp;mm), which is precisely the confound retracted above.</p>'
                f'<p class="muted"><strong>And it still never seats the cube: 0/64.</strong> The gate is '
                f'5&nbsp;mm and the policy is 29&times; away from it. The clip at the top of this page '
                f'shows what that looks like &mdash; the cube is carried to within a few centimetres of '
                f'the target, and then the hand opens and leaves it on the table beside the stack. '
                f'Every hypothesis tested tonight was refuted, and the task is unsolved; what improved '
                f'is measurable and real, and it is not the same thing as working.</p>'
                f'<p>So the honest tally: the added reward term failed its bar and deserved to. The '
                f'mixed curriculum failed a bar it should never have been given, while producing the '
                f'only real transport this task has shown. And the part that was never a reward change '
                f'at all still stands: the grasp, fixed by hand action scale '
                f'0.1&nbsp;&rarr;&nbsp;0.02&nbsp;rad/step and <code>entropy_coef</code> '
                f'0.006&nbsp;&rarr;&nbsp;0.001, holding at 10&ndash;12&nbsp;N in 64/64 environments '
                f'where every previous policy measured 0.0000&nbsp;N.</p>'
                f'<p class="muted"><strong>Do not read the C4 column as task performance.</strong> The '
                f'C4 reset starts the cube already seated within tolerance, so its success rate is '
                f'largely the reset distribution &mdash; measured at ~0.61 for doing nothing useful. A '
                f'C4 figure far <em>below</em> that means the policy is actively knocking over stacks '
                f'that arrived intact, which is what the early iterations here show. C3 '
                f'(<code>task_1</code>) is the real carry task, and it has logged a non-zero success '
                f'rate in <strong>{_c3nz} of {len(_s1)}</strong> readings.</p>'
            )
    except (OSError, ValueError, KeyError, IndexError, ZeroDivisionError) as exc:
        # NEVER fail silently here. This report is regenerated unattended by after_mixed.sh at the end
        # of a training run, and a silent `pass` already cost one full round: the section vanished
        # while the generator printed "wrote ... index.html" exactly as if it had worked.
        print(f"WARNING: mixed-run section SKIPPED -- {type(exc).__name__}: {exc}", flush=True)

    try:
        _c4t = json.loads((CS / "results" / "trace_policy_c4.json").read_text())
        _n = len(_c4t["cube_xyz"][0])
        _s = sum(_c4t["success_env"][min(149, len(_c4t["success_env"]) - 1)])
        # Lead with the ROBUST statistic. The success count swings 14 points between adjacent
        # checkpoints while the median error does not move, so quoting it alone in the first sentence
        # would assert a precision the section below explicitly disclaims.
        _xe = sorted(_c4t["xyz_dist_env"][min(149, len(_c4t["xyz_dist_env"]) - 1)])
        _med = _xe[len(_xe) // 2] * 1000
        headline += (
            f' <strong>On C4 the policy does not carry anything, and an earlier version of this page '
            f'said it did.</strong> The C4 reset hands it a cube that is <em>already seated</em>: '
            f'measured over 150 steps, the cube&rsquo;s median displacement is '
            f'<strong>0.00000&nbsp;m</strong>, only 3 of 64 environments move it as much as 1&nbsp;mm, '
            f'and success goes <strong>39/64 at step&nbsp;0 to 38/64 at step&nbsp;149</strong> &mdash; '
            f'no environment improves, one gets worse. The {_med:.1f}&nbsp;mm median and the '
            f'{_s}/{_n} count are the <em>reset distribution</em>, not an achievement.'
            f' What the policy does earn is narrower and real: a <strong>null action destroys the '
            f'stack in every environment (0/64, median error 0.015&nbsp;&rarr;&nbsp;0.089&nbsp;m)</strong>, '
            f'while the trained policy keeps <strong>34&ndash;38 of 64</strong>. And it does that '
            f'<em>without ever touching the cube</em>: the contact sensor reads <strong>0.0000&nbsp;N</strong> '
            f'through the whole rollout. So the honest description of C4 is narrower still &mdash; the '
            f'policy keeps the arm clear of a stack it never holds, where doing nothing knocks it over.'
        )
    except (OSError, ValueError, KeyError, IndexError, ZeroDivisionError):
        pass

    # Cross-validation against the independent trace runs. Computed here rather than transcribed, so
    # the page cannot drift from the data if either harness is re-run. The two harnesses share no
    # code path beyond the task config: the probe sweeps conditions in one process each, the traces
    # record every environment's trajectory for the animation.
    def trace_at_150(name: str):
        p = CS / "results" / f"trace_{name}.json"
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            return None
        ih = d.get("in_hand") or []
        return ih[min(149, len(ih) - 1)] if ih else None

    t1, t20 = trace_at_150("eff1"), trace_at_150(f"eff{CHOSEN_EFF}")
    if t1 is not None and t20 is not None and base and chosen:
        cross_html = (
            f'<p><strong>Independent cross-check.</strong> A separate harness &mdash; the trajectory '
            f'recorder built for the animation below, 64 environments instead of 128 &mdash; measured '
            f'the same two conditions at step 150: <strong>{t1:.3f}</strong> at &times;1 against the '
            f'sweep\'s {base["hold150"]:.3f}, and <strong>{t20:.3f}</strong> at '
            f'&times;{CHOSEN_EFF} against the sweep\'s {chosen["hold150"]:.3f}. Both differences sit '
            f'inside the 0.062&ndash;0.180 run-to-run band measured on identical configurations, so '
            f'the result reproduces across harnesses rather than resting on one script.</p>'
        )
    else:
        cross_html = ""

    c3_html = c3_milestone_html(local_series)
    # C4's peak, computed for the same reason the C3 figures are: it kept rising through the night
    # (0.45 -> 0.54 -> 0.60 -> 0.6155) and a hardcoded "above 0.5" was stale within the hour.
    _c4 = [s["succ"].get(0) for s in local_series if 0 in s["succ"]]
    c4_peak_txt = f"a peak of {max(_c4):.2f}" if _c4 else "above 0.5"
    # The second run's C3 readings, stated with its sample size attached. Both runs use SEED=42, so
    # this is NOT seed-independent replication -- what differs is env count (2048 vs 1024), stage-A
    # budget (80 vs 200 iterations) and hardware. Worth saying explicitly, because "a second run also
    # did it" invites the stronger reading, and the stronger reading is not what was tested.
    h100_c3 = [s["succ"].get(1) for s in h100_series if 1 in s["succ"]]
    h100_nz = [(i + 1, v) for i, v in enumerate(h100_c3) if v]
    if h100_nz:
        c3_html += (
            f'<p>The H100 run reproduces it. On a separate training run it logged '
            f'{", ".join(f"<strong>{v:.4f}</strong> at reading {i}" for i, v in h100_nz)} '
            f'&mdash; {len(h100_nz)} non-zero out of {len(h100_c3)} readings so far.</p>'
            f'<p class="muted">Carefully: both runs use <code>SEED=42</code>, so this is <em>not</em> '
            f'seed-independent replication. What differs is the environment count (2048 vs 1024), the '
            f'stage-A budget it resumed from (80 vs 200 iterations) and the hardware &mdash; a genuinely '
            f'different training trajectory, but not the stronger claim. And {len(h100_c3)} readings is '
            f'far too few to compare rates against the local run meaningfully; the useful statement is '
            f'only that C3 leaving zero has now happened in two runs rather than one.</p>'
        )

    # THE SECOND RUN, rolled out through the identical harness. Computed rather than written in, for
    # the same reason as every other figure here: that run is still training, so a hand-typed number
    # is stale within the hour. Sampled at step 150 -- at or after 160 the time-out reset puts the
    # cube back in the hand and reports a recovery that never happened.
    def rollout_stats(name: str):
        p = CS / "results" / f"trace_{name}.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        se, xe = d.get("success_env"), d.get("xyz_dist_env")
        if not se or not xe:
            return None
        i = min(149, len(se) - 1, len(xe) - 1)
        s, x, n = se[i], sorted(xe[i]), len(se[i])
        return {"n": n, "succ": sum(bool(v) for v in s), "best": x[0],
                "med": (x[(n - 1) // 2] + x[n // 2]) / 2,
                "w100": sum(1 for v in x if v < 0.100),
                "ck": str(d.get("meta", {}).get("checkpoint", "?")).split("/")[-1]}

    _rr = [("C4", "4070 Ti run", rollout_stats("policy_c4")), ("C4", "H100 run", rollout_stats("h100_c4")),
           ("C3", "4070 Ti run", rollout_stats("policy_c3")), ("C3", "H100 run", rollout_stats("h100_c3"))]
    _rr = [(f, w, r) for f, w, r in _rr if r]
    cross_run_html = ""
    if len(_rr) >= 2:
        _tr = "".join(
            f'<tr><td>{fam}</td><td>{who}</td><td class="mono">{html.escape(r["ck"])}</td>'
            f'<td class="num">{r["succ"]} / {r["n"]}</td>'
            f'<td class="num strong">{r["med"] * 1000:.1f}&nbsp;mm</td>'
            f'<td class="num">{r["best"] * 1000:.1f}&nbsp;mm</td>'
            f'<td class="num">{r["w100"]} / {r["n"]}</td></tr>' for fam, who, r in _rr)
        cross_run_html = (
            '<h2>The second run, measured the same way</h2>'
            "<p>The H100 run's checkpoint was pulled and rolled out <em>locally</em>, through the "
            'identical harness, at the identical sampling step, scored by the environment&rsquo;s own '
            '<code>ProgressContext.success</code> &mdash; so these rows are comparable to each other and '
            'to every other number on this page.</p>'
            '<p class="muted"><strong>Read the C4 rows with the retraction above in mind.</strong> They '
            'do not measure placement accuracy: on C4 the cube starts seated and is never moved, so '
            'these figures compare how much of the reset distribution each policy leaves undisturbed. '
            'The C3 rows are the ones measuring an actual attempt at the task &mdash; and both runs '
            'score 0/64.</p>'
            '<table><thead><tr><th>family</th><th>run</th><th>checkpoint</th><th class="num">success</th>'
            '<th class="num">median error</th><th class="num">best</th>'
            '<th class="num">within 100&nbsp;mm</th></tr></thead><tbody>' + _tr + '</tbody></table>'
            '<p class="muted"><strong>Read the seed line before reading the agreement.</strong> Both runs '
            'use <code>SEED=42</code>. What differs is the environment count (2048 against 1024), the '
            'stage-A budget resumed from, and the hardware &mdash; a genuinely different training '
            'trajectory, but <em>not</em> a seed-independent replication. This page should not be read as '
            'claiming one.</p>')
        _c4r = {w: r for f, w, r in _rr if f == "C4"}
        if len(_c4r) == 2:
            _h, _l = _c4r["H100 run"], _c4r["4070 Ti run"]
            cross_run_html += (
                f'<p>On C4 the second run already does the task at roughly a quarter of the training: '
                f'<strong>{_h["succ"]}/{_h["n"]}</strong> inside the 5&nbsp;mm tolerance at a median '
                f'<strong>{_h["med"] * 1000:.1f}&nbsp;mm</strong>, against {_l["succ"]}/{_l["n"]} at '
                f'{_l["med"] * 1000:.1f}&nbsp;mm for the finished run &mdash; with '
                f'<strong>{_h["w100"]}/{_h["n"]}</strong> environments within 100&nbsp;mm, the same '
                f'all-64 result. That run is still training.</p>'
                f'<p class="muted">Both C4 rows inherit the retraction above. The cube is seated at '
                f'reset and the contact sensor reads 0.0000&nbsp;N throughout, so these figures compare '
                f'how much of the reset distribution each policy leaves undisturbed &mdash; not '
                f'placement skill, and not a grasp.</p>')
        _c3r = {w: r for f, w, r in _rr if f == "C3"}
        if len(_c3r) == 2:
            _a, _b = _c3r["4070 Ti run"]["med"], _c3r["H100 run"]["med"]
            cross_run_html += (
                f'<p>On C3 the two runs fail at almost exactly the same distance: medians '
                f'<strong>{_a * 1000:.1f}&nbsp;mm</strong> and <strong>{_b * 1000:.1f}&nbsp;mm</strong>, '
                f'{abs(_a - _b) * 1000:.1f}&nbsp;mm apart, both scoring 0/64. Two different training '
                f'trajectories landing on the same failure distance points at something systematic in the '
                f'task or the objective rather than at one unlucky run. With a shared seed it is weaker '
                f'evidence than it first looks &mdash; but it is consistent with the untested hypothesis '
                f'above, and it is the reason to test that hypothesis next rather than to train longer.</p>')

    page = f"""<!doctype html>
<meta charset="utf-8">
<title>DELTO pinch &mdash; grip endurance and training</title>
<style>
:root {{ color-scheme: light dark; --fg:#1f2328; --bg:#fff; --mut:#656d76; --line:#d0d7de; --card:#f6f8fa; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --fg:#e6edf3; --bg:#0d1117; --mut:#8b949e; --line:#30363d; --card:#161b22; }}
}}
body {{ margin:0; padding:32px 20px 64px; background:var(--bg); color:var(--fg);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:900px; margin:0 auto; }}
h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:19px; margin:34px 0 10px; }}
.lede {{ font-size:17px; padding:14px 16px; background:var(--card); border:1px solid var(--line); border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th,td {{ padding:6px 9px; border-bottom:1px solid var(--line); text-align:left; }}
th {{ font-weight:600; color:var(--mut); }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
.strong {{ font-weight:700; }}
tr.eject td {{ color:#d1242f; }}
.muted {{ color:var(--mut); }}
/* A retraction has to be impossible to skim past -- it corrects something a reader may already
   have taken away from an earlier version of this page. */
.retract {{ border-left:3px solid #d1242f; background:var(--card); padding:11px 14px;
  border-radius:6px; margin:14px 0; }}
.chart {{ width:100%; height:auto; margin:8px 0 18px; }}
.grid {{ stroke:var(--line); stroke-width:1; }}
.tick,.ttl {{ fill:var(--mut); font-size:11px; font-family:ui-monospace,monospace; }}
figure {{ margin:0 0 18px; }}
video {{ width:100%; border:1px solid var(--line); border-radius:8px; background:#000; }}
figcaption {{ color:var(--mut); font-size:13px; padding-top:5px; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:18px; }}
code {{ background:var(--card); padding:1px 5px; border-radius:4px; font-size:13px; }}
@media (max-width:420px) {{ body {{ padding:20px 16px 48px; }} table {{ font-size:12.5px; }} }}
</style>
<main>
<h1>DELTO two-finger pinch &mdash; the hand grips; it does not yet carry</h1>
<p class="muted">UR10e + DELTO DG-5F, thumb <code>rl_dg_1_tip</code> + index <code>rl_dg_2_tip</code>, cube34.
Generated {html.escape(__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"))}.</p>

<p class="lede">{grasp_lede}{headline}</p>

<h2>What the policy <em>used to</em> do on C3: it opened its hand</h2>
<div class="retract">
<p><strong>SUPERSEDED &mdash; this section describes <code>model_898</code> and <code>model_398</code>,
not the current policy.</strong> Everything below was measured and is accurate for those checkpoints:
the hand opened on the first control step in 64/64 environments and the contact sensor read
<strong>0.0000&nbsp;N</strong> throughout. It is kept because the contrast is the result. The current
policy grips &mdash; the figures are in the opening paragraph, computed from the acceptance rollout at
build time rather than transcribed. What survives unchanged is the <em>failure to carry</em>: the grip
is now real and lasts about twelve control steps instead of one, and the cube still ends the episode
no closer to the goal.</p>
</div>
<p>C3 (<code>ObjectAnywhereEEGrasped</code>) is the family that genuinely requires transport &mdash; the
cube starts grasped in mid-air and must be carried to the stack. Recording every link&rsquo;s pose
through the rollout says exactly what goes wrong, and it is not a grasp that fails under load:</p>
<table><thead><tr><th>C3, <em>superseded</em> policy (<code>model_898</code>), 64 environments</th>
<th class="num">value</th></tr></thead><tbody>
<tr><td>fingertip separation, step 0 &rarr; step 20</td><td class="num strong">64&nbsp;mm &rarr; 135&nbsp;mm</td></tr>
<tr><td>environments where the fingers open &gt;5&nbsp;mm</td><td class="num strong">64 / 64</td></tr>
<tr><td>median step at which they open</td><td class="num strong">1</td></tr>
<tr><td>cube distance from pinch centre, step 0 &rarr; 5</td><td class="num">16&nbsp;mm &rarr; 201&nbsp;mm</td></tr>
<tr><td>cube height, start &rarr; end</td><td class="num">0.161&nbsp;m &rarr; 0.021&nbsp;m (the table)</td></tr>
<tr><td>success</td><td class="num strong">0 / 64</td></tr>
</tbody></table>
<p><strong>The hand opens on the first control step, and the contact sensor says it never gripped
the cube at all.</strong> Measured fingertip force through the policy rollout is <strong>0.0000&nbsp;N</strong>
at every sampled step. So &ldquo;drops it&rdquo; and even &ldquo;lets go&rdquo; are too generous: there
is no grasp to lose. The 0.148&nbsp;m of cube motion is the cube falling and rolling, not being
transported.</p>
<p>The comparison that makes this damning is the first clip on this page. Under a <em>null action</em>
at the same effort setting, the same hand closes on the same cube and the sensor reads a median
<strong>11.2&nbsp;N</strong>, with <strong>26/64</strong> environments still in contact at step&nbsp;150.
The environment can hold this cube; the trained policy simply never grips it.</p>
<p class="muted">That points at one lever rather than at more training. A policy that opens its hand
immediately, in every environment, on the family where holding on is the whole task, is behaving as
though releasing is not penalised &mdash; and the success bonus is not gated on still holding the
cube. The concrete next experiments: log per-term rewards <em>during training</em> (not in a replay
harness, which failed twice here); add a penalty or termination for losing the grasp mid-carry; gate
the success bonus on retention; or shorten the carry before lengthening it. This is a hypothesis the
data motivates, not a demonstrated cause.</p>
<p><strong>The second run does exactly the same thing.</strong> The H100 run&rsquo;s <em>final</em>
checkpoint (<code>model_398</code>, the completed 399-iteration run) was rolled out locally through the
identical harness:</p>
<table><thead><tr><th>C3, step 150</th><th class="num">4070 Ti <code>model_898</code></th>
<th class="num">H100 <code>model_398</code></th></tr></thead><tbody>
<tr><td>fingertip separation, step 0 &rarr; 20</td><td class="num">64&nbsp;&rarr;&nbsp;135&nbsp;mm</td>
<td class="num">66&nbsp;&rarr;&nbsp;126&nbsp;mm</td></tr>
<tr><td>environments opening &gt;5&nbsp;mm</td><td class="num strong">64 / 64</td><td class="num strong">64 / 64</td></tr>
<tr><td>median step at which they open</td><td class="num strong">1</td><td class="num strong">1</td></tr>
<tr><td>cube from pinch centre, step 0 &rarr; 5</td><td class="num">16&nbsp;&rarr;&nbsp;201&nbsp;mm</td>
<td class="num">37&nbsp;&rarr;&nbsp;186&nbsp;mm</td></tr>
<tr><td>cube height, start &rarr; end</td><td class="num">0.161&nbsp;&rarr;&nbsp;0.021&nbsp;m</td>
<td class="num">0.164&nbsp;&rarr;&nbsp;0.021&nbsp;m</td></tr>
<tr><td>success</td><td class="num strong">0 / 64</td><td class="num strong">0 / 64</td></tr>
</tbody></table>
<p class="muted">Two training trajectories &mdash; 900 iterations against 399, 1024 environments
against 2048, different hardware &mdash; land on the same behaviour line for line: open the hand on the
first control step, in every environment. Both runs use <code>SEED=42</code>, so this is not
seed-independent replication; but it does mean the behaviour is a stable outcome of this objective
rather than one run&rsquo;s quirk, which is the argument for changing the objective rather than
training longer.</p>
<p class="muted">Also withdrawn: an earlier version of this page described C3 as &ldquo;a carry attempt
that drops the cube&rdquo;. For these checkpoints there is no carry attempt &mdash; the release happens at
step&nbsp;1, with no contact force at any point.</p>
<p class="muted">And that withdrawal is now itself out of date, which is the good news on this page.
With the hand action scale and entropy fixed, the release no longer happens at step&nbsp;1 and the
grasp is no longer imaginary: the gate closes in every environment at about 10&nbsp;N and holds for
roughly a dozen control steps. &ldquo;A carry attempt that drops the cube&rdquo; is, for the first time,
close to an accurate description &mdash; the arm lifts the cube before letting go. It is still not a
success: the cube reaches the table in most environments and the episode ends with it no nearer the
goal than it began.</p>

<h2>Why C3 was stuck at exactly 0.0000</h2>
<p>Five independent defects produced one identical symptom. Four were fixed earlier: the hand went
limp at action 0 (<code>target := measurement</code>, so zero PD error and zero grip force); the arm
sagged 114&nbsp;mm because the OSC computes <code>J&#7488;f</code> with no gravity term against
zero-stiffness joints; the bank QA certified geometry rather than force; and a relative action term
re-added its grip bias on all 12 physics substeps, turning a 0.05&nbsp;rad set point into a ramp that
drove the fingers 0.45&nbsp;rad past closure and <em>ejected</em> the cube.</p>
<p>What remained was endurance. With all four fixed, the grasp still decayed: held at step 5, gone by
step 120 &mdash; and success is scored only at step 160. The cause was the hand's
<code>effort_limit_sim</code>, a 0.06&ndash;0.17&nbsp;N&middot;m backstop worth ~2.2&nbsp;N at the
fingertip. The measured pinch sat at <strong>2.07&nbsp;N</strong>: exactly on the cap. That is why
every other knob read flat &mdash; raising the grip bias 5&times; changed fingertip force
<em>not at all</em>.</p>
<p class="muted">The cap is not a hardware limit: the USD authors 30&nbsp;N&middot;m per joint (the
DG-5F's ~7&nbsp;kg rated grip). &times;20 is ~44&nbsp;N at the fingertip against a hand rated ~68&nbsp;N.</p>

<h2>Endurance sweep</h2>
<p class="muted">C3 bank (<code>ObjectAnywhereEEGrasped</code>), null action, 128 envs. <em>hold</em> is the
fraction of envs still holding the cube. Run-to-run variance on identical configs measures
0.062&ndash;0.180, so differences under ~0.15 are noise.</p>
<table><thead><tr><th>condition</th><th class="num">effort</th><th class="num">tuck</th>
<th class="num">bias</th><th class="num">force@20 (N)</th><th class="num">hold@20</th>
<th class="num">hold@150</th><th class="num">cube &Delta;z (m)</th><th>stability</th></tr></thead>
<tbody>{rows}</tbody></table>
<p>Two things the sweep settles. Effort is the only lever that matters, and it saturates: past
&times;20 the gain falls inside the noise band while fingertip force stops tracking the cap
(4.5&nbsp;&rarr;&nbsp;4.7&nbsp;&rarr;&nbsp;5.5&nbsp;N), meaning contact geometry, not torque, is now
the limit. And <strong>tucking the undriven fingers 3/4/5 does not help</strong> &mdash; on its own it
made retention worse (0.023 vs 0.047). The fingers were never in the way; the cap was.</p>
{cross_html}

<h2>Training</h2>
<p>Two families are being trained. <strong>C4</strong> (near-goal) learns readily on the fixed
configuration &mdash; from 0.02 to {c4_peak_txt} in a few hundred iterations from scratch, where the old
broken setup peaked at 0.38&ndash;0.48 after ~1,300 iterations <em>while resuming</em>.
<strong>C3</strong>, which must carry the cube ~210&nbsp;mm, is the family that has sat at exactly
0.0000 for its entire recorded history &mdash; until tonight, when it produced its first non-zero
readings. It is the family this night was aimed at.</p>
{c3_html}
<p>C3 has <strong>not</strong> been solved, and the traces say something more specific than the success
number does. Under a null action the environment holds this grasp in ~72% of environments at step 150
&mdash; that is the endurance fix working. Under the policy the grasp is gone within about five
control steps and the cube sits where it fell for the rest of the episode.</p>
<p>The first three checkpoints showed <strong>no resolvable trend</strong> &mdash; every column
non-monotone, i.e. run-to-run variation. The fourth breaks that pattern: at <code>model_575</code>
all three columns move the same way and each clears the earlier spread. Grasp retention at step 1
goes 0.203 / 0.094 / 0.359 &rarr; <strong>0.516</strong>; the median final distance 161 / 180 / 168
&rarr; <strong>118&nbsp;mm</strong>, which is 43&nbsp;mm below the lowest of the first three when
those three spanned only 19&nbsp;mm between them; and environments finishing within 100&nbsp;mm go
15 / 8 / 10 &rarr; <strong>24 of 64</strong>, more than double the previous best.</p>
<p class="muted">Held to the same standard as everything else here: that is <em>one</em> rollout, and
single measurements have been wrong repeatedly tonight. Three columns agreeing is more than any
earlier point offered, but it is suggestive rather than established &mdash; and it did <strong>not</strong> reproduce: a fifth
checkpoint 150 iterations later came back at 156&nbsp;mm and 16/64, squarely inside the original band.
On five points the sequence reads 161 / 180 / 168 / <em>118</em> / 156&nbsp;mm, so
<code>model_575</code> was an outlier rather than the start of anything. This is the pre-committed
test resolving against the optimistic reading, which is the outcome it existed to catch, and the honest test is whether a
later checkpoint repeats it. Note also what has <em>not</em> changed &mdash; success is still 0/64,
and the grasp is still gone by step 20. The cube is being carried nearer the goal, not delivered to
it. Four checkpoints spanning 275 iterations, rolled out on C3
through the trainer's own inference path and scored with the environment's own
<code>ProgressContext.success</code>:</p>
<table><thead><tr><th>checkpoint</th><th class="num">hold @1</th><th class="num">hold @20</th>
<th class="num">median distance @150</th><th class="num">within 100 mm</th><th class="num">success</th></tr></thead>
<tbody>
<tr><td class="mono">model_300</td><td class="num">0.203</td><td class="num">0.000</td>
<td class="num">161 mm</td><td class="num">15/64</td><td class="num">0/64</td></tr>
<tr><td class="mono">model_375</td><td class="num">0.094</td><td class="num">0.000</td>
<td class="num">180 mm</td><td class="num">8/64</td><td class="num">0/64</td></tr>
<tr><td class="mono">model_450</td><td class="num">0.359</td><td class="num">0.000</td>
<td class="num">168 mm</td><td class="num">10/64</td><td class="num">0/64</td></tr>
<tr><td class="mono">model_575</td><td class="num strong">0.516</td><td class="num">0.000</td>
<td class="num strong">118 mm</td><td class="num strong">24/64</td><td class="num">0/64</td></tr>
<tr><td class="mono">model_725</td><td class="num">0.344</td><td class="num">0.000</td>
<td class="num">156 mm</td><td class="num">16/64</td><td class="num">0/64</td></tr>
</tbody></table>
<p class="muted">Caveat on the table above: these rollouts came from the same script as the retracted
reward figures, before the missing-reset fix, so treat the individual numbers as indicative rather
than exact. The qualitative finding is independently corroborated &mdash; the C3 clip below shows the
cube dropped and stationary, and <code>task_1_success_rate</code> was exactly 0.0000 for every logged
iteration across the window these two checkpoints span, which is measured by the trainer and not by
this script.</p>
<p class="muted">These rollouts do not contradict the milestone above, and should not be read as doing
so. <strong>64 environments cannot resolve a 0.74% success rate</strong>: the expected number of
successes is 0.47, so <code>0/64</code> is the single most likely outcome even if the training-reported
0.0074 is entirely real. The last row (<code>model_575</code>, stage-B iteration ~375) sits well after the
first milestone at reading 241 and still reads 0/64 &mdash; which is what a rate that small looks like
at this sample size, not evidence against it. Settling it would need thousands of episodes, or simply
reading the trainer's own metric over its 1024 environments.</p>
<p>Every column moved the wrong way. Over the same span mean reward rose from 4.34 to 6.93, so the
policy is being paid by the shaping terms &mdash; fingertip proximity, contact, lift &mdash; without
that translating into transport. The remaining gap is not the hand, the effort cap, or the reset
bank; it is that nothing in the objective makes <em>keeping</em> the grasp worth more than the
shaped terms the policy can collect while dropping it.</p>
<p class="retract"><strong>RETRACTED &mdash; this section previously claimed the objective pays about
twice as much for dropping the cube as for holding it (0.99 vs 2.56, and a per-environment split of
0.968 vs 1.891). That claim was wrong and is withdrawn.</strong> Two measurements of the same
quantity disagree: summing the per-step reward over the window gives 1.227, while the environment's own
per-term episode sums give 2.992 &mdash; 3&times; apart <em>and opposite in sign</em> on the
held-versus-dropped split. Until that is explained neither number can be quoted. The likely cause is
that the rollout script called <code>get_observations()</code> without resetting, so environments kept
whatever episode phase they were in and their episode sums include reward accumulated <em>before</em>
the recording window &mdash; which matches the episode sums being the larger of the two. That was
fixed &mdash; and repeating the measurement still did not resolve it, so this is left open rather than
answered.</p>
<p class="muted">The full account, because the instability is the point. This quantity was computed
four ways and gave three different answers: dropping paying ~2&times; more; then, from the
environment's own per-term episode sums, holding paying more by 0.237; then, after the reset fix,
holding paying more by 0.702 &mdash; but that last one only over the 52 of 64 environments whose two
measurements agreed, and the 12 excluded were 92% <em>holding</em> against 56% in the kept group, so
the exclusion is self-serving and the figure fails its own bias check. A number that changes sign
depending on how it is computed is not a result. Nothing from this family is quoted here, and "the
objective rewards dropping" should be read as an untested hypothesis rather than a finding.</p>
<p class="muted">A false lead worth recording, because it nearly became the published explanation: the
first suspicion was mid-window resets, apparently evidenced by single-step cube jumps above 50&nbsp;mm
in 35 of 64 environments. That evidence does not hold. Those jumps track <em>dropping</em>, not
resetting &mdash; a cube in free fall covers ~49&nbsp;mm in one 0.1&nbsp;s control step and more once
moving &mdash; and the counts follow the drop rate exactly (57/64 where the shipped effort cap drops
nearly every cube, 0/64 where the cube stays seated on C4). The conclusion to retract was right; the
reason first given for it was not.</p>
<p class="muted">That first comparison is confounded, and saying so is the point: the two arms differ
in more than grasp retention, because a null action holds the cube but transports nothing while the
policy moves the arm and collects the transport shaping. So it was re-run in a form that removes the
confound.</p>
<p class="retract"><strong>ALSO RETRACTED.</strong> A follow-up "clean experiment" reported here
split one null-action rollout by whether the grasp survived and gave 0.968 for holding against 1.891
for dropping (35/64 vs 29/64). It came from the same rollout script and therefore rests on the same
unreset episodes, so it is withdrawn too. Retracting one figure while leaving its sibling standing
would be worse than retracting neither.</p>
<p class="muted">For completeness, the per-term breakdown taken alongside it pointed the
<em>opposite</em> way &mdash; <code>grasp_contact</code> (+0.207) and
<code>fingertip_object_distance</code> (+0.199) both paid <em>more</em> for holding, and summing all
fourteen terms favoured holding by 0.237. That disagreement with the per-step figure is precisely the
inconsistency that exposed the bug, and it is why no number from this family is quoted here until the
corrected run lands.</p>
<p class="muted">What follows from that, and what does not. Swapping effort condition would not help:
the sweep's next-best settings (&times;40 and &times;10+tuck) differ from &times;20 by less than the
measured noise, and none of them change what a policy does with a grasp it already holds. More
iterations of the same objective are not indicated either &mdash; that is what these two checkpoints
test, and it got worse. Beyond that, what is left is a <em>hypothesis</em> rather than a
demonstrated cause: that the objective does not make keeping the grasp worth more than the shaped
terms collectable while dropping it. The attempt to measure that failed (above), so it should be
treated as the first thing to <em>test</em>, not as the diagnosis. Concretely worth trying: log
per-term rewards during training rather than in a replay harness; add a penalty or termination for
losing the grasp mid-carry; gate the success bonus on still holding; or shorten the carry with
curriculum shaping before lengthening it.</p>
{train_block("H100 curriculum", h100_series, h100_stages)}
{train_block("4070 Ti hedge run", local_series, local_stages)}

<h2>C4: what the policy actually does</h2>
<div class="retract">
<p><strong>RETRACTED: &ldquo;the policy carries the cube to the seated stack, releases it and
withdraws&rdquo;.</strong> Earlier versions of this page led with that sentence. It is wrong. The C4
reset family (<code>ObjectPartiallyAssembledEEGrasped</code>) starts with the cube <em>already seated
within tolerance</em> &mdash; and, contrary to what this page said earlier, <em>not held by the hand at
all</em>. The scene&rsquo;s own <code>object_hand_s</code> contact sensor reads <strong>0.0000&nbsp;N
at every sampled step</strong>, with <strong>0/64</strong> environments in contact at steps 0, 1, 5,
20 or 149 and only 2/64 registering any contact at any point; geometrically the fingertips sit
<strong>56&ndash;62&nbsp;mm</strong> from the centre of a 34&nbsp;mm cube, <strong>114&nbsp;mm</strong>
apart. The hand never touches the cube, so &ldquo;releases it and withdraws&rdquo; described letting go
of something never held. Measured from full pose recordings over steps
0&ndash;149: median cube displacement <strong>0.00000&nbsp;m</strong>, only <strong>3 of 64</strong>
environments move it even 1&nbsp;mm, success <strong>39/64 at step 0</strong> and <strong>38/64 at
step 149</strong>, with <strong>zero</strong> environments going from failing to succeeding and one
going the other way. There is no carry, no approach and no placement on C4.</p>
<p>How it survived: every check was a <em>summary</em> &mdash; success rate, median error, within-100&nbsp;mm
count, retention &mdash; and not one was a <em>difference over time</em>.
<code>ProgressContext.success</code> answers &ldquo;is it seated now&rdquo;, never &ldquo;did you
seat it&rdquo;. A single rendered frame at step&nbsp;0 would have shown a finished stack and ended it
immediately, which is exactly why the robot videos above now lead this page instead of the plots.</p>
</div>
<p><strong>What the policy does earn.</strong> Run the identical harness with a null action and the
stack is destroyed in every environment: success <strong>0/64</strong> at step 149, median error
<strong>0.015&nbsp;&rarr;&nbsp;0.089&nbsp;m</strong>, and <code>in_hand</code> 0.453 &mdash; the limp
hand stays tangled in the cube and drags it off. The trained policy holds <strong>34&ndash;38 of
64</strong> with <code>in_hand</code> 0.000, a clean release and retreat. Training on C4 went
0.02&nbsp;&rarr;&nbsp;0.63, which is consistent: the early policy actively knocked the stack over and
learned not to. That is a real, measurable skill &mdash; <em>release and withdraw without disturbing
an assembled stack</em> &mdash; and it is the honest description of C4.</p>
<table><thead><tr><th>C4, same harness, step 149</th><th class="num">null action</th>
<th class="num">trained policy</th></tr></thead><tbody>
<tr><td>success</td><td class="num strong">0 / 64</td><td class="num strong">34&ndash;38 / 64</td></tr>
<tr><td>median error</td><td class="num">0.089&nbsp;m</td><td class="num">0.004&nbsp;m</td></tr>
<tr><td>in_hand at step 150</td><td class="num">0.453</td><td class="num">0.000</td></tr>
</tbody></table>
<p class="muted">The table below is retained for the record, but read it as the <em>reset
distribution plus whatever the policy fails to disturb</em>, not as placement accuracy:</p>
<table><thead><tr><th>metric</th><th class="num">model_725</th><th class="num">model_800</th>
<th class="num">model_898 (final)</th></tr></thead><tbody>
<tr><td>grasp retention at step 1</td><td class="num">1.000</td><td class="num">1.000</td>
<td class="num">1.000</td></tr>
<tr><td>median final cube-to-goal</td><td class="num strong">3.0&nbsp;mm</td>
<td class="num strong">3.1&nbsp;mm</td><td class="num strong">3.8&nbsp;mm</td></tr>
<tr><td>best environment</td><td class="num">0.4&nbsp;mm</td><td class="num">0.4&nbsp;mm</td>
<td class="num">0.4&nbsp;mm</td></tr>
<tr><td>within 100&nbsp;mm</td><td class="num strong">64 / 64</td><td class="num strong">64 / 64</td>
<td class="num strong">64 / 64</td></tr>
<tr><td>success (within the 5&nbsp;mm tolerance)</td><td class="num strong">38 / 64 = 59.4%</td>
<td class="num strong">47 / 64 = 73.4%</td><td class="num strong">38 / 64 = 59.4%</td></tr>
</tbody></table>
<p class="muted"><em>Everything from here to the end of this section was written before the retraction
above, and analyses the success count as though it measured placement. It does not &mdash; it measures
how much of the reset distribution survives the episode. Kept because the run-to-run variability it
documents is real and still worth knowing.</em></p>
<p class="muted"><strong>The success row is the fragile one, and a third rollout settled it.</strong>
Across three checkpoints the count reads 59.4% / 73.4% / 59.4%: the two ends agree <em>exactly</em>,
and the middle reading is the outlier &mdash; not a policy that improved and then regressed. Mean
64.1%, spread 14.1 points, which is ~2.4 standard errors at n&nbsp;=&nbsp;64. Everything robust stayed
put across all three: retention 1.000, best environment 0.4&nbsp;mm, and <strong>64/64</strong> within
100&nbsp;mm every single time, with the median drifting only 3.0&nbsp;&rarr;&nbsp;3.8&nbsp;mm.</p>
<p class="muted">Why the count moves while nothing else does: the error distribution piles up against
the tolerance, so with the median at ~3&nbsp;mm and success scored at 5&nbsp;mm, small perturbations
flip a dozen environments across the line at once. <em>Quote the median error and the 64/64 within
100&nbsp;mm; treat the success percentage as ~60% with occasional higher draws, not as a figure
precise to the point.</em></p>
<p class="muted">Grasp retention at step 20 is 0.000 on C4, and that is <em>correct</em> &mdash; the
hand is supposed to let go. Retention indicates failure only on C3, where the cube must still be held.
The same statistic means opposite things in the two families.</p>

{cross_run_html}

<h2>Did the reward term I added earn its place? A matched A/B</h2>
<p>Matte asked why the rewards were changed at all, and whether OmniReset&rsquo;s own objective should
not simply be used as published. That question was worth more than the answer I expected to give. It
forced the experiment below, and the result is not flattering to my addition.</p>
<p>Both arms come from <strong>one script</strong> (<code>ab_hold_bonus.sh</code>) whose only variable is
<code>UWLAB_HOLD_BONUS_WEIGHT</code>: C3-only, 512 environments, 200 iterations, seed&nbsp;42,
<code>entropy_coef</code>&nbsp;0.001, hand action scale 0.02&nbsp;rad/step, effort&nbsp;&times;20. They are
evaluated through one harness and compared by a script that prints deltas, because comparing two
printouts by eye is how the <em>previous</em> version of this comparison went wrong.</p>
<table><thead><tr><th>C3, steps 0&ndash;149, 64 environments</th>
<th class="num">w=0 (published objective)</th><th class="num">w=0.5 (my term)</th></tr></thead><tbody>
<tr><td>fingertip force, t=0</td><td class="num">10.4&nbsp;N</td><td class="num strong">11.6&nbsp;N</td></tr>
<tr><td>two-finger gate still shut at t=149</td><td class="num">7 / 64</td><td class="num strong">32 / 64</td></tr>
<tr><td>median steps the gate is shut</td><td class="num">50 / 150</td><td class="num strong">111 / 150</td></tr>
<tr><td>median step of release</td><td class="num">12</td><td class="num strong">16</td></tr>
<tr><td>cube ends on the table</td><td class="num">56 / 64</td><td class="num strong">36 / 64</td></tr>
<tr><td><strong>net transport toward goal</strong></td><td class="num">&minus;0.6&nbsp;mm</td><td class="num strong">+9.6&nbsp;mm</td></tr>
<tr><td>environments ever within 100&nbsp;mm</td><td class="num strong">19 / 64</td><td class="num">16 / 64</td></tr>
<tr><td>environments ever within 50&nbsp;mm</td><td class="num strong">3 / 64</td><td class="num">1 / 64</td></tr>
<tr><td>success at t=149</td><td class="num strong">0 / 64</td><td class="num strong">0 / 64</td></tr>
</tbody></table>
<p><strong>The verdict: it helps holding, and on the better-powered metric it helps reaching too
&mdash; but it solves nothing yet.</strong> The bonus more than doubles how long the grasp survives,
gives the first positive net transport measured on this task, and closes 24&nbsp;mm more of the
alignment error over 512 environments. Against that, it is <em>worse</em> on the rollout&rsquo;s two
64-environment approach counts, and neither arm seats a cube. Read the 64-environment rows as the
weaker evidence: they are a single rollout of 64 episodes against ~100,000 per arm below.</p>
<p class="retract"><strong>CORRECTION &mdash; an earlier version of this section said success was
&ldquo;0.0000 for all 200 iterations in both arms&rdquo;. That is wrong for the with-bonus arm and is
withdrawn.</strong> It was true of <code>Metrics/task_0_success_rate</code>, but
<code>Metrics/task_command/end_of_episode_success_rate</code> is a <em>separate</em> metric and it is
<strong>not</strong> zero with the bonus: peak <strong>0.0052</strong>, non-zero in 1 of 200 iterations,
against a genuine <strong>0 of 200</strong> at weight&nbsp;0. One reading at 0.5% is noise-adjacent and
nowhere near a solved task &mdash; but &ldquo;exactly zero in both arms&rdquo; was a categorical claim and
it did not survive reading the second metric.</p>
<p><strong>The better-powered transport signal, and it favours the bonus.</strong> The trainer logs
<code>end_of_episode_pos_align_error</code> every iteration over all 512 environments &mdash; a
continuous distance, so it moves long before the 5&nbsp;mm success gate ever trips, and it rests on
~100,000 episodes per arm rather than the rollout&rsquo;s 64:</p>
<table><thead><tr><th>end-of-episode position alignment error</th>
<th class="num">w=0</th><th class="num">w=0.5</th></tr></thead><tbody>
<tr><td>iteration 24</td><td class="num">0.249&nbsp;m</td><td class="num">0.261&nbsp;m</td></tr>
<tr><td>iteration 99</td><td class="num">0.247&nbsp;m</td><td class="num strong">0.211&nbsp;m</td></tr>
<tr><td>iteration 149</td><td class="num">0.234&nbsp;m</td><td class="num strong">0.195&nbsp;m</td></tr>
<tr><td>iteration 199</td><td class="num">0.206&nbsp;m</td><td class="num strong">0.182&nbsp;m</td></tr>
</tbody></table>
<p>The bonus arm is <strong>24&nbsp;mm closer</strong> by iteration 199 and <em>both</em> arms are still
descending when the budget runs out &mdash; which is the evidence that 200 iterations was simply too
short, rather than a wall. That is why a 900-iteration run is now going.</p>
<p class="muted">What does not change: the bonus lifted mean reward 2.24&nbsp;&rarr;&nbsp;3.02 and
<code>grasp_contact</code> 0.013&nbsp;&rarr;&nbsp;0.019, far more than it moved the task, so it is still
substantially paying itself; and it carries slightly more <code>abnormal_robot</code> terminations
(0.077 vs 0.063).</p>
<p>So the answer to the original question is mostly &ldquo;you are right&rdquo;. The contact terms were
already in this repo before I touched it. The two changes that demonstrably fixed the grasp &mdash; hand
action scale 0.1&nbsp;&rarr;&nbsp;0.02&nbsp;rad/step and <code>entropy_coef</code>
0.006&nbsp;&rarr;&nbsp;0.001 &mdash; are <em>not reward changes at all</em>. What the added term buys is
narrower than I first argued for it and broader than the rollout alone suggested: it roughly doubles
grip retention and closes 24&nbsp;mm more alignment error, but it has never seated a cube, and it
inflates its own reward far more than it moves the task.</p>
<p class="retract"><strong>THE TERM FAILED A PRE-COMMITTED TEST AND IS NOW SWITCHED OFF.</strong> The
bar was fixed at iteration&nbsp;203, <em>before</em> the deciding data existed: the 50-iteration rolling
mean of <code>end_of_episode_pos_align_error</code> had to fall below <strong>0.170&nbsp;m</strong> by
iteration&nbsp;400. It reached <strong>0.1869&nbsp;m</strong> &mdash; a miss of 17&nbsp;mm. Over the same
400 iterations <code>hold_bonus</code> <strong>doubled</strong> (rolling-50 0.0315&nbsp;&rarr;&nbsp;0.0649)
and mean reward rose 2.30&nbsp;&rarr;&nbsp;3.77, to buy 44&nbsp;mm of alignment. Progress per 100
iterations went +34.2&nbsp;mm, +1.3&nbsp;mm, +8.4&nbsp;mm &mdash; and the metric still sits
<strong>37&times; away from the 5&nbsp;mm success gate</strong>. A term that inflates its own payout
faster than it moves the task has not earned its place, so
<code>UWLAB_HOLD_BONUS_WEIGHT=0</code> from here and OmniReset&rsquo;s published objective is restored.
Matte&rsquo;s question was right.</p>
<p><strong>A design mistake in my own experiment, which matters more than the verdict.</strong> The A/B
trained <strong>C3-only</strong> so the comparison would be clean &mdash; but C3 starts the cube
210&nbsp;mm from the goal and the policy never once seats it, so a C3-only curriculum <em>never shows
the agent what seating looks like</em>. The only run that ever approached the goal (146&nbsp;mm) was
the one trained on the <strong>50/50 C4+C3 mixture</strong>, where C4 starts the cube already seated and
supplies exactly that near-goal release experience. Cleanliness engineered out the behaviour being
asked for. The current run therefore <em>resumes</em> the grasp-solved checkpoint on the mixture at
weight&nbsp;0 &mdash; resumed rather than restarted, because the grasp (64/64 environments at
10&ndash;12&nbsp;N) came from the control fixes and should not be paid for twice.</p>

{mixed_html}

<h2>Media: these are mesh renders, and the one kind of render still missing</h2>
<p>The clips below are <strong>renders of the robot&rsquo;s own geometry</strong> &mdash; all 39 links,
loaded from the same USD/URDF meshes the simulator uses, lit with directional shadows and PBR
materials, and posed frame by frame from <em>measured</em> link poses read out of the physics engine
every step. They are not plots and not illustrations: every position and orientation in them came from
the simulation. An earlier version of this page called them state traces, which undersold what they
are; that description is corrected here.</p>
<p>What they are <em>not</em> is Isaac&rsquo;s own RTX renderer, which was asked for and is still
missing. That path does not work on this host, and after a long search the cause is not code.</p>
<p class="muted">Six real defects were found and fixed in the render path first &mdash; a missing
Isaac&nbsp;6 compatibility shim, a <code>cfg.sim.physx</code> section that no longer exists, a contact
sensor silently disabled by the object-variant swap, an encoder told the wrong frame geometry
(960&times;540 RGB declared against 480&times;270 RGBA actual, which wrote sheared frames without ever
erroring), a camera <code>prim_path</code> outside <code>{{ENV_REGEX_NS}}</code> that the scene never
resolved, and a pose call that no-ops in silence. After all six the camera is verifiably in the right
place with the right orientation &mdash; and the renderer still draws nothing but sky (mean
|Laplacian| 0.17&ndash;0.36 against ~4+ for real content). Isaac&nbsp;Sim&nbsp;5.1 does not render here
either; it crashes in <code>libomni.hydra.rtx</code>.</p>
<p class="muted">The host conditions, verified tonight rather than recalled: <code>nvidia-smi</code>
reports driver <strong>610.57.04</strong>, and <strong>no 580-branch package exists in the repos</strong>
(<code>nvidia-open-lts</code> is also 610.57.04). IOMMU is active &mdash; 36 groups under
<code>/sys/kernel/iommu_groups</code>, no <code>iommu=</code> override on the kernel command line, and
the kernel log shows AMD-Vi up with <code>Default domain type: Translated</code>. Omniverse rendering is
documented to break under IOMMU, and this stack expects the 580 branch. <em>Correction:</em> an earlier
version of this paragraph said the renderer itself logs <code>IOMMU is enabled</code>. That line was
quoted from memory and could not be found in any log on this box tonight, so the claim now rests only
on the kernel evidence above.</p>
<p class="muted"><strong>Three ways to get Isaac-native renders, two of which need a decision that is
not mine to take.</strong> (1) Boot with <code>amd_iommu=off</code> &mdash; a one-line bootloader change
and a reboot of this server; reversible, but intrusive, and this box serves other things.
(2) Render on a rented display-capable GPU on the 580 branch &mdash; costs money; the H100 is not an
option, as it has no graphics engine and segfaults seconds into any camera-enabled run. (3) Keep the
mesh renders below, which are driven by the same physics and already show the robot doing the task.
Downgrading the driver in place was considered and rejected: it would mean the Arch archive or AUR plus
a DKMS rebuild against kernel 7.2.2, on a headless machine, with a real chance of leaving it without a
working GPU.</p>
<p class="muted">Physics never depended on the renderer, so every number on this page is unaffected.</p>

<h2>Clips</h2>
<div class="grid2">{vid_html}</div>
</main>
"""
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {OUT / 'index.html'}  ({len(page)} bytes, {len(probe)} probe rows, "
          f"{len(local_series)} local iters, {len(h100_series)} h100 iters, {len(vids)} clips)")


if __name__ == "__main__":
    main()
