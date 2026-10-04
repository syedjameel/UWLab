"""Content for the pinch-grasp report. Every section degrades gracefully if its media is absent."""

from __future__ import annotations

import html
from collections import defaultdict
from pathlib import Path


def kpi(value: str, label: str) -> str:
    return f'<span class="k"><b>{html.escape(value)}</b><span>{html.escape(label)}</span></span>'


def summarise(results: list[dict], by: str, key: str = "held") -> list[dict]:
    groups: dict[object, list[dict]] = defaultdict(list)
    for r in results:
        groups[r[by]].append(r)
    rows = []
    for k, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        n = len(rs)
        held = sum(r["held"] for r in rs)
        pinched = sum(r.get("pinched", False) for r in rs)
        rows.append({
            by: k,
            "n": n,
            "held %": f"{100 * held / n:.0f}",
            "true pinch %": f"{100 * pinched / n:.0f}",
            "median slip mm": f"{sorted(r['final_dist_mm'] for r in rs)[n // 2]:.1f}",
            "median thumb err rad": f"{sorted(r['thumb_err_rad'] for r in rs)[n // 2]:.3f}",
            "median grip thumb N": f"{sorted(r.get('grip_thumb_N', 0) for r in rs)[n // 2]:.2f}",
            "median grip index N": f"{sorted(r.get('grip_index_N', 0) for r in rs)[n // 2]:.2f}",
        })
    return rows


def build(out: Path, video, image, table, load):
    S: list[str] = []

    pose = load("pinchpose.json")
    jawcal = load("jawcal.json")
    diag = load("diag_v3.json") or load("diag.json")
    sweep = load("sweep_v3.json")
    banks = {t: load(f"bank_before_{t}.json") for t in (
        "ObjectAnywhereEEAnywhere", "ObjectRestingEEGrasped",
        "ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped")}

    lede = (
        "<b>The environment was broken, and it was one line of convention.</b> Isaac Lab 3.0 "
        "changed the quaternion order from (w,&nbsp;x,&nbsp;y,&nbsp;z) to "
        "(x,&nbsp;y,&nbsp;z,&nbsp;w). UWLab is written for the old one, so every "
        "<code>rot=(1,&nbsp;0,&nbsp;0,&nbsp;0)</code> meant as identity was read as "
        "<b>a 180&deg; rotation about X</b>. The lab table spawned upside down with its 676&nbsp;mm "
        "cabinet standing above the work surface, the two cubes poked out through its underside, "
        "and the UR10e was bolted on inverted &mdash; <code>shoulder_link</code> measured "
        "z&nbsp;=&nbsp;&minus;0.190&nbsp;m against a base at &minus;0.010&nbsp;m, where a UR10e's "
        "shoulder belongs 180&nbsp;mm <em>above</em> its base.</p>"
        "<p>Everything else here was downstream of that. The shipped banks' folded "
        "<code>shoulder_lift&nbsp;=&nbsp;&minus;1.527</code> pose is the arm contorting up out of "
        "the cabinet; the grasps that &ldquo;approached from below&rdquo; were approaching a table "
        "that was above them. The fix is twenty literals on the OmniReset UR10e-Delto path "
        "(<code>88c8e21</code>); the same defect remains in ~30 more literals and ~173 quaternion "
        "call sites elsewhere in the repo.</p>"
        "<p class='warn'>Corrections to earlier claims below. The flipped-table finding, which was "
        "withdrawn, was <b>correct</b> &mdash; it is restored. The grasp-bank reachability filter "
        "(321&nbsp;&rarr;&nbsp;137) was compensating for the flip rather than fixing anything. "
        "Still withdrawn: the 281/397&nbsp;mm hand-to-cube distances as proof the banks were not "
        "grasps, <code>max_object_pos_deviation</code>&nbsp;=&nbsp;10&nbsp;mm (it is 50), and the "
        "arm needing PD gains. Every bank and posture recorded before <code>88c8e21</code> "
        "encodes the inverted rig and is void, including the three-defect result below.</p>"
    )

    # ------------------------------------------------------------------ the rig
    S.append("<h2>The rig, corrected</h2>")
    if (out / "env_fixed.png").exists():
        S.append(image(out / "env_fixed.png",
                       "Rendered at the scene's configured default pose rather than from a bank: "
                       "the lab table with its feet on the floor, black mat over the robot's zone "
                       "and green mat over the workspace, and the UR10e standing on it with the "
                       "DELTO hand over the green. Before the fix this same viewpoint showed an "
                       "open-topped box with the hand outside it."))
    S.append(table([
        {"measurement": "Table AABB z", "before": "-0.004 .. +0.676 (cabinet ABOVE the top)",
         "after": "-0.676 .. +0.004"},
        {"measurement": "Robot AABB z", "before": "-0.275 .. 0", "after": "0 .. +0.275"},
        {"measurement": "shoulder_link z", "before": "-0.190 m (below its own base)", "after": "+0.173 m"},
        {"measurement": "Table / Robot / plate root_quat_w", "before": "(1,0,0,0) = 180 deg about X",
         "after": "(0,0,0,1) identity"},
    ]))
    S.append("<p class='cap'>Measured in the running sim, before and after the convention fix.</p>")

    # ------------------------------------------------------- bench vs sampler
    S.append("<h2>Four ways the bench and the grasp sampler had drifted apart</h2>")
    S.append(
        "<p>Rebuilding on the corrected rig exposed a chain of latent defects, each of which let a "
        "posture score well in the hand-only bench and fail in the grasp sampler. Each had been "
        "invisible while the rig was inverted, because gravity then pushed the cube <em>into</em> "
        "the fingers.</p>")
    S.append(table([
        {"#": 1, "mismatch": "graspability never scored",
         "signature": "closing axis 51-54 deg off the approach; gate rejected all 61440 candidates",
         "fix": "--min-align-deg hard filter"},
        {"#": 2, "mismatch": "pre-grasp never optimised",
         "signature": "one pad reaches 65 mm from the grasp centre, the other asymptotes at 27.8 "
                      "against the 33.5 mm required",
         "fix": "openpose.py derives it: both pads 48 mm, +14.6 mm clear"},
        {"#": 3, "mismatch": "hold scored, not capture",
         "signature": "cube weightless until after the hand had closed",
         "fix": "--close-under-gravity"},
        {"#": 4, "mismatch": "wrong orientation, wrong actuators",
         "signature": "bench used the hand's own pose and the shared actuator; the sampler points "
                      "the approach down and rebinds stiffness 10x at velocity 1.0 rad/s",
         "fix": "--approach-axes, --sampler-gains"},
    ]))
    S.append("<p class='cap'>Each row is a property the sampler requires that the bench was not "
             "measuring.</p>")
    S.append(table([
        {"what the bench was scoring": "hold a cube handed to it (original objective)", "held": "87.9%"},
        {"what the bench was scoring": "+ capture a free cube, hand pointing down", "held": "80.9%"},
        {"what the bench was scoring": "+ the sampler's own actuator gains", "held": "16.4%"},
    ]))
    S.append("<p class='cap'>Same hand, same 34&nbsp;mm cube, 256 runs each. Aligning the bench "
             "with the sampler moves it from disagreeing with it to reproducing it.</p>")
    S.append("<h2>The fist grasps; the pinch stalls part-closed</h2>")
    S.append(
        "<p>Run against the same sampler, the same corrected rig, the same 34&nbsp;mm cube and the "
        "same servo, switched only by <code>UWLAB_DELTO_PINCH</code>:</p>")
    S.append(table([
        {"posture": "five-finger fist (0)", "jaw over the episode": "96 -> 84 -> 67 mm, still closing",
         "result": "3 successes in batch 1"},
        {"posture": "two-finger pinch (1)", "jaw over the episode": "94 -> 66 -> 63 mm, frozen there",
         "result": "0 successes, every batch"},
    ]))
    S.append("<p class='cap'>The sampler, the placement gate, the rig, the object and the relative "
             "action term all work. Fingers 1 and 2 simply never reach the commanded "
             "30.8&nbsp;mm jaw.</p>")
    S.append(
        "<p>The hand stops 14.5&nbsp;mm clear of the cube on each side with torque still available, "
        "and nothing is touching it: the proximal links measure 78&ndash;137&nbsp;mm from the cube "
        "at the stall. Six explanations have been tested and rejected &mdash; asymmetric distal "
        "stiffness (20&times; thumb), <code>effort_limit_sim</code> raised to 0.3&nbsp;N&middot;m, "
        "joint limits (0 of 40 targets out of range, checked against the USD), episode length "
        "(already 18&nbsp;s / 180 control steps), fingers 3/4/5 blocking the pair (held open "
        "through the whole close), and the cube fouling the proximal links. The jaw sits at "
        "63&nbsp;mm in every one.</p>"
        "<p class='warn'>One correction: the 0.1&nbsp;rad joint errors reported earlier as a stall "
        "are not one. The fist shows the same 0.100 values <em>while closing normally</em> &mdash; "
        "<code>_per_joint_command</code> recomputes from the measured position every step by "
        "design, so a standing error is what a working servo looks like here. Jaw aperture is the "
        "only honest signal, which is why it is now instrumented.</p>")
    S.append(
        "<p><b>Since then the hand has been made to close.</b> The binding joints were the "
        "<code>_2</code> pair, not the <code>_4</code> pair everything had been tuned around. Under "
        "the relative action's 0.1&nbsp;rad/step cap, realised torque is bounded at "
        "<code>stiffness&nbsp;&times;&nbsp;0.1</code> = 0.063&nbsp;N&middot;m on "
        "<code>rj_dg_1_2</code> &mdash; not enough to hold the thumb chain against its own weight "
        "with the hand pointing down, so the joint <em>sagged away</em> from its target "
        "(&minus;1.162 &rarr; &minus;1.179 &rarr; &minus;1.219, commanded toward &minus;0.866) "
        "while the servo kept asking for the same 0.1&nbsp;rad. The PD error reads a healthy 0.100 "
        "either way, which is why it hid; only the joint position showed it. At 10&times; the jaw "
        "now travels 64&nbsp;&rarr;&nbsp;32&nbsp;mm against a commanded 30.8.</p>"
        "<p>The bank is still empty. Pinning the cube for 25 control steps while the hand closes "
        "&mdash; so the sampler asks which poses <em>hold</em> a part rather than whether the hand "
        "can catch one falling &mdash; does not change it: still 0 in 192 attempts. So the hand "
        "closes, the cube is stationary while it does, the gate passes with clearance, and the "
        "grasp still is not retained, while the same posture reads <b>69.9% held</b> in the "
        "hand-only bench under matched gravity, orientation, actuator gains and close schedule. "
        "That divergence is the open question.</p>"
        "<p class='warn'><b>The bench's gravity was not gravity.</b> Isaac Lab's "
        "<code>set_external_force_and_torque</code> defaults to <code>is_global=False</code> "
        "&mdash; the LOCAL body frame. The bench applies the cube's weight as an external wrench, "
        "so that weight rotated with the cube: at an arbitrary grasp orientation it pointed in an "
        "arbitrary world direction, and for many orientations it pushed the part <em>into</em> the "
        "pinch rather than out. Every held rate this bench has reported was measured against it "
        "&mdash; <b>83.6%, 87.9%, 80.9% and 69.9% are all void</b>. With "
        "<code>is_global=True</code> the same posture reads <b>0.0%</b>, which is exactly what the "
        "sampler had been saying all along. The sampler was right; the instrument was wrong.</p>"
        "<p>Re-optimising against real gravity, with the sampler's gains, orientation, step close "
        "and relative action all matched, finds a posture at <b>46.1% held / 35.2% true "
        "pinch</b> &mdash; so a viable pinch does exist; the earlier ones simply were not it.</p>"
        "<p>That in turn exposed a straight conflict between two requirements. The placement gate "
        "rejects a candidate if any gripper collider point touches the object, fingertips included, "
        "which forces the pre-grasp open to 93&nbsp;mm. The grasp needs it narrow:</p>")
    S.append(table([
        {"same closed posture, pre-grasp": "50.0 mm", "held": "15.2%"},
        {"same closed posture, pre-grasp": "93.4 mm", "held": "0.0%"},
    ]))
    S.append("<p class='cap'>The cube leaves the gap before the pads arrive. The gate exists to "
             "reject placements that drive the hand THROUGH the part on approach, not to forbid "
             "the two surfaces whose job is to touch it, so the pads are now exempt from it.</p>")
    S.append(
        "<p>The bank is still empty, and the next lead is a specific one: at the first step of an "
        "episode the pads measure <b>160&nbsp;mm from the cube</b> where half a 50&nbsp;mm "
        "pre-grasp is <b>25&nbsp;mm</b>. The cube is not being placed between them.</p>"
        "<p>The placement gate clears &mdash; <b>0 of 61440 rejected</b> with +3.74&nbsp;mm of "
        "fingertip clearance, from a standing start of 100% rejection. The bank is still empty: at "
        "the episode timeout the cube reads <code>not_far=0, above_ground=0</code> in all 64 "
        "environments. What it is not is the closing <em>schedule</em>, the "
        "object, the gains or the rig &mdash; the five-finger fist grasps on all of them.</p>")

    # ------------------------------------------------------------------ current state
    S.append("<h2>The fist banks grasps; the pinch does not</h2>")
    S.append(
        "<p>On the corrected rig the five-finger fist fills a grasp bank: <b>302 grasps from "
        "19008 attempts</b>, roughly 2&ndash;3 per 64-attempt batch. The two-finger pinch banks "
        "nothing in ~2000 attempts, at any pre-grasp width, with any gains, and with the cube "
        "pinned through the whole close. Everything upstream of the grasp is verified &mdash; rig "
        "upright, gravity real in the bench, hand closing, gate not vetoing the pads, placement "
        "measured correct at 28/32&nbsp;mm from the two pads &mdash; and a viable posture exists "
        "at <b>43.8% held / 41.4% true pinch</b>. What remains is that the sampler's servo will "
        "not drive the pinch pair onto the part.</p>"
        "<p>So the reset families here are recorded against the fist. That is a departure from the "
        "two-finger pinch this work set out to build, and it is recorded as one rather than "
        "presented as the plan.</p>")
    S.append("<h2>Where the banks stand</h2>")
    S.append(table([
        {"family": "C1 ObjectAnywhereEEAnywhere", "states": "reused",
         "status": "grasp-independent, copied from the shipped tree"},
        {"family": "C2 ObjectRestingEEGrasped", "states": "recording",
         "status": "producing states; not_far 128/128, drift 0.0 mm"},
        {"family": "C3 ObjectAnywhereEEGrasped", "states": "40 / 40",
         "status": "RECORDED and verified"},
        {"family": "C4 ObjectPartiallyAssembledEEGrasped", "states": 0,
         "status": "BLOCKED -- stale partial_assemblies.pt"},
    ], highlight=""))
    S.append(
        "<p>The C3 bank checks out as real grasps. Cubes are held between 47 and 310&nbsp;mm in "
        "the air, the robot base sits correctly at z&nbsp;=&nbsp;&minus;0.0002&nbsp;m, and the "
        "fingers are closed. The arm posture is the clearest tell: median "
        "<code>shoulder_lift</code> is <b>&minus;0.139&nbsp;rad</b>, against <b>&minus;1.527</b> "
        "in the shipped banks &mdash; the latter being the pose that folds the arm underneath the "
        "table.</p>"
        "<p>C4 fails structurally rather than marginally: <code>above_ground</code> passes in "
        "<b>0 of 128</b> environments every episode and <code>orient_down</code> in 4, with no "
        "environment clearing even five of the six conditions. It is the one family that consumes "
        "<code>partial_assemblies.pt</code>, which was copied from the old dataset tree and "
        "therefore encodes seated poses produced by the pre-fix pipeline. Regenerating that file "
        "is a separate job.</p>"
    )

    # ------------------------------------------------------------------ three defects
    S.append("<h2>Why grasped reset recording produced nothing &mdash; three defects</h2>")
    S.append(
        "<p>Recording accepted <b>zero</b> grasped reset states across thousands of "
        "environment-episodes, under every configuration tried. Three independent defects, each "
        "of which hid the other two:</p>"
    )
    S.append(table([
        {"#": 1, "defect": "grasp bank 53.6% unreachable -- approach from under the table",
         "signature": "orient_down 27%", "fix": "filter 321 -> 137 reachable"},
        {"#": 2, "defect": "no torque supplied to an effort-controlled arm during recording",
         "signature": "arm sags, dragging the cube", "fix": "tau = g(q) gravity compensation"},
        {"#": 3, "defect": "acceptance reference sampled before the teleport was applied",
         "signature": "'drift' of 212 mm that was the teleport", "fix": "defer capture one step"},
    ], highlight=""))
    S.append(
        "<p><b>Defect 3 was the keystone.</b> <code>check_reset_state_success.reset()</code> read "
        "its reference from <code>data.root_pos_w</code>, but <code>write_*_to_sim</code> only "
        "STAGES into PhysX &mdash; those buffers still hold the previous step's values, and "
        "refreshing the asset does not help because the simulation has not stepped. The reference "
        "was therefore the pose from BEFORE the reset teleport, so <code>not_far</code> measured "
        "the teleport for the whole episode. It sat at 5&ndash;11% under every configuration, "
        "which meant nothing done to the grasp or the arm could ever move it.</p>"
    )
    S.append(table([
        {"condition": "not_abnormal", "shipped": "25%", "after all three": "~40%"},
        {"condition": "orient_down", "shipped": "27%", "after all three": "~48%"},
        {"condition": "stable", "shipped": "23%", "after all three": "~36%"},
        {"condition": "not_far", "shipped": "11%", "after all three": "~72%"},
        {"condition": "object drift, median", "shipped": "212 mm", "after all three": "0.0 mm"},
        {"condition": "accepted states", "shipped": "0", "after all three": "filling"},
    ], highlight=""))
    S.append(
        "<p>The median object now drifts <b>0.0 mm</b>: the pinch holds the cube dead still in the "
        "median environment. What remains is real &mdash; the 75th percentile still runs to several "
        "hundred millimetres, so roughly a third of environments drop the cube, and that is a "
        "property of the grasp rather than of the harness.</p>"
        "<p class='warn'>Three claims made earlier in this investigation were withdrawn after "
        "measurement: that the shipped banks' 281/397&nbsp;mm hand-to-cube distances showed they "
        "were not grasps (measured in a replay scene that was itself wrong); that the table was "
        "flipped or floating (read from a stale USD transform); and that "
        "<code>max_object_pos_deviation</code> was 10&nbsp;mm, which framed a threshold question "
        "that did not exist &mdash; it is 50&nbsp;mm. Arm PD gains were also tried and reverted: "
        "stiff enough to hold a UR10e is unstable at dt&nbsp;=&nbsp;1/120, soft enough to be "
        "stable still sags.</p>"
    )

    # ------------------------------------------------------------------ the four substages
    S.append("<h2>The four reset families, on video</h2>")
    S.append(
        "<p>Each clip replays six states from that family in the recorder's own environment and "
        "steps real physics. Nothing is posed: if the hand is not holding the cube, the cube "
        "falls on camera.</p>"
    )
    FAMILIES = [
        ("ObjectAnywhereEEAnywhere",
         "C1 &mdash; Reaching (S<sup>R</sup>)",
         "Object anywhere on the table, end effector anywhere. The cube is not held and is not "
         "supposed to be."),
        ("ObjectRestingEEGrasped",
         "C2 &mdash; Near-Object (S<sup>NO</sup>)",
         "Object resting, end effector grasped. The cube should be in the hand, on or near the "
         "surface."),
        ("ObjectAnywhereEEGrasped",
         "C3 &mdash; Stable Grasp (S<sup>G</sup>)",
         "Object anywhere, end effector grasped. The cube should be held in mid-air."),
        ("ObjectPartiallyAssembledEEGrasped",
         "C4 &mdash; Near-Goal (S<sup>NG</sup>)",
         "Object partially assembled and still grasped, just above the receptive cube."),
    ]
    S.append(
        "<p>Two takes of each family. <b>Corrected rig</b> is the current recording, on the "
        "un-flipped table, against the fist grasp bank. <b>Inverted rig</b> is the same family "
        "before the quaternion fix and is kept for comparison &mdash; that is what a 676&nbsp;mm "
        "cabinet standing above the work surface looks like from a camera at table height.</p>"
        "<p class='warn'>Watch the hand rather than the table in the corrected takes. The scene is "
        "right, but the cube is not held: all 18 filmed states replay with <code>n_tips_loaded=0, "
        "NO_CONTACT</code>, and a representative C3 state drifts 259&nbsp;mm the moment physics "
        "runs. For C1 that is correct &mdash; it requires no grasp. For C2 and C3 it is the "
        "defect.</p>")
    for stem, title, blurb in FAMILIES:
        shown = False
        for tag, label in (("fist", "corrected rig"), ("shipped", "inverted rig, for comparison")):
            f = out / "films" / f"{tag}_{stem}.mp4"
            if f.exists():
                if not shown:
                    S.append(f"<h3>{title}</h3>")
                    shown = True
                # video() escapes its caption, so this must be plain text.
                S.append(video(f, f"{label.upper()} \u2014 {blurb}"))

    S.append("<h2>The pinch itself</h2>")
    for name, cap in [
        ("posture_compare.mp4",
         "Identical close, gravity and 3 Hz shake. Left: the repo's fist, 51 mm jaw around a "
         "34 mm cube, pads close either side of it, cube gone. Middle: the naive two-finger "
         "posture, same 51 mm jaw, cube ends up resting in the crook of the curled fingers with "
         "0 N on both pads -- a scoop, which scores as 'held' on any position-only criterion. "
         "Right: the hold-optimised posture, 24.5 mm jaw, 1.7 N and 1.5 N on the two pads with "
         "fingers 3/4/5 at exactly zero."),
        ("cu_opt_pinch.mp4", "The hold-optimised pinch alone, close up."),
        ("cu_repo_full.mp4", "The repo's shipped SCRIPTED_CLOSE fist, same conditions."),
    ]:
        if (out / name).exists():
            S.append(video(out / name, cap))

    # ------------------------------------------------------------------ what the banks contain
    S.append("<h2>What the grasped families actually contain</h2>")
    S.append(
        "<p>Measured by the repo's own <code>render_reset_states_viewport.py</code>, replaying "
        "each bank inside the recorder's own environment. Six states per family, real physics, "
        "no posing.</p>"
    )
    S.append(table([
        {"family": "C1 ObjectAnywhereEEAnywhere", "states": 6, "fingertips loaded": "0/5",
         "cube": "on the table", "verdict": "correct -- not a grasped family"},
        {"family": "C2 ObjectRestingEEGrasped", "states": 6, "fingertips loaded": "0/5",
         "cube": "on the table, no hand on it", "verdict": "NOT GRASPED"},
        {"family": "C3 ObjectAnywhereEEGrasped", "states": 6, "fingertips loaded": "0/5",
         "cube": "airborne, untouched", "verdict": "NOT GRASPED"},
        {"family": "C4 ObjectPartiallyAssembledEEGrasped", "states": 6, "fingertips loaded": "0/5",
         "cube": "seated on the base cube, untouched", "verdict": "NOT GRASPED"},
    ], highlight=""))
    S.append(
        "<p>The load-bearing number is <b>tips_loaded</b>, which is the renderer's own filtered "
        "contact reading, not an interpretation of a picture: in C2 and C3 no fingertip touches "
        "the cube in any state sampled. C1 is the control and behaves exactly as it should &mdash; "
        "cube supported on the table, 4&nbsp;mm of drift, no contact expected and none found. "
        "That C1 comes out right is what makes the C2/C3 readings trustworthy: the same replay "
        "path, the same instrumentation, a family that passes.</p>"
    )

    S.append("<h2>Correction &mdash; how this was measured the first time</h2>")
    S.append(
        "<p class='warn'>The first version of this finding was produced by <code>banklab.py</code>, "
        "a replay scene written for this investigation. That scene was wrong: the robot's own "
        "documented home pose put the hand at z&nbsp;=&nbsp;&minus;0.676&nbsp;m, exactly ground "
        "level. The numbers it produced (281&nbsp;mm and 397&nbsp;mm medians) were withdrawn, and "
        "the figures above replace them.</p>"
        "<p>The cause was duplication: the repo already ships a renderer that replays states "
        "inside the real task environment. It would not start under Isaac Sim 6 &mdash; "
        "<code>isaacsim.core.utils</code> is gone and <code>SimulationCfg.physx</code> is now "
        "<code>.physics</code> &mdash; so a parallel scene got written instead of the two-line "
        "port, and the parallel scene drifted. Both breaks are now fixed in the repo tool, along "
        "with video capture and a camera override.</p>"
        "<p>The conclusion survived re-measurement with the correct tool. The evidence for it "
        "did not, and the first set of numbers should not be cited.</p>"
    )

    # ------------------------------------------------------------------ hero
    hero = out / "hero_compare.mp4"
    if hero.exists():
        S.append("<h2>The result</h2>")
        S.append(video(hero,
                       "Left: the repo's closed posture. Right: the derived pinch. Same 34 mm cube "
                       "(39.3 g as simulated), same closing schedule, same joint friction and "
                       "stiffness — only the commanded posture differs. The cube is weightless "
                       "while the hand closes, is given its weight at release, then shaken at 3 Hz."))
    else:
        for name, cap in [("hero3_opt_pinch.mp4", "The derived pinch, closing on a free cube."),
                          ("hero3_repo_full.mp4", "The repo's SCRIPTED_CLOSE posture, same conditions.")]:
            if (out / name).exists():
                S.append(video(out / name, cap))

    if pose:
        S.append("<h3>Derived pinch posture &mdash; geometric, later superseded</h3>")
        S.append(
            "<p>These are the numbers the <em>geometric</em> derivation produced, kept here "
            "because the rest of this section reasons about them. The posture actually used now "
            "comes from the hold search further down and closes to 24.2 mm; do not read the two "
            "as competing measurements of the same thing.</p>"
        )
        S.append(
            kpi(f"{pose['close']['aperture_mm']:.1f} mm", "closed jaw (34 mm cube)")
            + kpi(f"{pose['open']['aperture_mm']:.1f} mm", "pre-grasp jaw")
            + kpi(f"{pose['jaw_drift_mm']:.2f} mm", "grasp-point drift while closing")
            + kpi(f"{pose['close']['clearance_mm']:.1f} mm", "nearest non-pad body")
        )

    # ------------------------------------------------------------------ cause 1
    S.append("<h2>Cause 1 &mdash; the commanded &ldquo;closed&rdquo; hand is a fist, not a jaw</h2>")
    S.append(
        "<p><code>DeltoHand/metadata.yaml</code> declares the hand's grasp geometry to be "
        "&ldquo;a virtual jaw of finger 1 against finger 2&rdquo;, and "
        "<code>grasp_center_offset</code> is the midpoint of those two fingertips evaluated at "
        "<code>DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS</code>. Measured in sim at that same posture "
        "the two pads are <b>51.03 mm</b> apart, against a 34 mm cube. The declared grasp centre "
        "is arithmetically correct and physically the middle of a fist &mdash; every finger curls "
        "toward it at once, so what the hand does to a cube placed there is enclose it, not pinch "
        "it, and in this bench it does not retain it.</p>"
    )
    if jawcal:
        q = jawcal["aperture_mm_quantiles"]
        S.append(
            f"<p>A sweep of <b>{jawcal['samples']:,}</b> random postures over the eight joints of "
            f"fingers 1 and 2 reaches pad separations down to <b>{q['min']:.2f} mm</b> "
            f"(p1 = {q['p1']:.1f} mm, median {q['p50']:.1f} mm), so opposing these two pads is well "
            "within the hand's reach. It is simply not what the closed posture asks for.</p>"
        )

    # ------------------------------------------------------------------ cause 2
    S.append("<h2>Cause 2 &mdash; the controller cannot reach the posture it commands</h2>")
    S.append(
        "<p>The offline recorders are scripted rollouts, not policies, but they drive the hand "
        "through the same <code>RelativeJointPositionAction</code> the policy uses "
        "(<code>scale=0.1</code>, <code>clip=(-1,1)</code>). That term computes "
        "<code>target = measured + clamp(action,&nbsp;-1,&nbsp;1)&nbsp;&times;&nbsp;0.1</code>, so the "
        "PD target is never more than 0.1 rad ahead of the joint and the realised torque is "
        "bounded by <code>stiffness&nbsp;&times;&nbsp;0.1</code>. On <code>rj_dg_1_4</code> that ceiling "
        "is <b>0.0170 N&middot;m</b>, against a joint friction of 0.01 and a 0.06 N&middot;m effort "
        "limit that is never reached.</p>"
        "<p>Measured, at identical friction and stiffness, closing from the open posture:</p>"
    )
    S.append(table([
        {"control mode": "relative action term (as shipped)", "friction": 0.01, "stiff4": 1,
         "thumb finishes short by": "0.844 rad"},
        {"control mode": "absolute position target", "friction": 0.01, "stiff4": 1,
         "thumb finishes short by": "0.060 rad"},
        {"control mode": "relative action term", "friction": 0.0003, "stiff4": 1,
         "thumb finishes short by": "0.001 rad"},
        {"control mode": "relative action term", "friction": 0.01, "stiff4": 10,
         "thumb finishes short by": "0.005 rad"},
    ], highlight=""))
    S.append(
        "<p>The stall is an interaction, not an independent fault: it appears only at the repo's "
        "own default joint friction <i>and</i> its default distal stiffness &mdash; which is what "
        "ships. Raising either one, or writing absolute targets, clears it. It is also why joint "
        "<code>friction</code> looked like the culprit for so long: it is a dimensionless "
        "coefficient, but the comment in <code>ur10e_delto.py</code> calls it "
        "&ldquo;0.01&nbsp;N&middot;m&rdquo;.</p>"
    )

    # ------------------------------------------------------------------ the pinch
    if pose:
        S.append("<h2>The derived pinch</h2>")
        S.append(
            "<p>The constraint that produces a pinch rather than a fist is not about aperture and "
            "not about fingers 3/4/5 specifically. It is that the cube must be reachable only by "
            "the two pads: <i>every</i> hand body other than <code>rl_dg_1_tip</code> and "
            "<code>rl_dg_2_tip</code> is required to stay outside a sphere around the grasp point "
            "wider than the cube's 29.4 mm half-diagonal. That single requirement forces the "
            "fingers to extend, which is what a pinch is.</p>"
        )
        S.append(table([
            {"posture": "derived pinch (closed)",
             "pad separation mm": pose["close"]["aperture_mm"],
             "nearest non-pad body mm": pose["close"]["clearance_mm"],
             "grasp point, mount frame mm": ", ".join(f"{v:.1f}" for v in pose["close"]["jaw_mount_mm"])},
            {"posture": "derived pinch (pre-grasp)",
             "pad separation mm": pose["open"]["aperture_mm"],
             "nearest non-pad body mm": pose["open"]["clearance_mm"],
             "grasp point, mount frame mm": ", ".join(f"{v:.1f}" for v in pose["open"]["jaw_mount_mm"])},
            {"posture": "SCRIPTED_CLOSE (repo)", "pad separation mm": 51.03,
             "nearest non-pad body mm": 40.73, "grasp point, mount frame mm": "63.7, 46.5, 150.0"},
            {"posture": "OPEN (repo)", "pad separation mm": 96.70,
             "nearest non-pad body mm": 52.44, "grasp point, mount frame mm": "81.6, 67.5, 153.0"},
        ], highlight=""))
        S.append(
            "<p class='warn'>The grasp point moves about 35 mm from the value in "
            "<code>metadata.yaml</code>. That is expected &mdash; the old value was the midpoint of "
            "a fist &mdash; but it means <code>grasp_center_offset</code> has to be re-anchored "
            "before the sampler is run with this posture, or it will place objects where the old "
            "closure used to grip. That failure has a precedent in this repo: a stale grasp centre "
            "cost 6144 sampling attempts and produced 0 successes.</p>"
        )

    # ------------------------------------------------------------------ measurement corrections
    S.append("<h2>Two measurements that were wrong, and what they were hiding</h2>")
    S.append(
        "<ul>"
        "<li><b>Peak contact force was not grip force.</b> Taking the maximum over the whole "
        "episode captured penetration transients at first touch and the release of the pin, and "
        "read 15&ndash;35 N on a hand whose distal joints are limited to 0.06 N&middot;m &mdash; "
        "impossible as a grip. Replaced with the mean over the hold window.</li>"
        "<li><b>Fingertip contact force included self-contact.</b> The articulation has "
        "<code>enabled_self_collisions=True</code>, so an unfiltered net force on a fingertip "
        "counts the hand pressing on itself. Tips 3/4/5 read 6.76 N while sitting 36 mm from the "
        "centre of a cube whose half-diagonal is 29.4 mm &mdash; they could not have been touching "
        "it. Contacts are now filtered to the cube, which changed a &ldquo;grasp carried by three "
        "fingers&rdquo; into a cube that had escaped the jaw entirely.</li>"
        "<li><b>The hold radius was too generous.</b> At 35 mm, a cube that had squirted out and "
        "finished 32 mm from both pads still scored as held. Tightened to 20 mm, and the decisive "
        "test is now contact on both pads rather than proximity.</li>"
        "</ul>"
    )

    solo = out / "hero3_opt_pinch.mp4"
    if solo.exists():
        S.append(video(solo, "The derived pinch alone, closer in. The cube is weightless while "
                             "the hand closes, gets its weight at release, then is shaken at "
                             "3 Hz. Contact force is filtered to the cube, so what is reported "
                             "as grip is the part pressing on a pad and not the hand pressing "
                             "on itself."))

    seq = out / "pinch_sequence.png"
    if seq.exists():
        S.append("<h2>The pinch, frame by frame</h2>")
        S.append(image(seq, "1. pre-grasp at 50 mm pad separation. 2. closing. 3. pinched at "
                            "30 mm on a 34 mm cube. 4. still held after 2 s of gravity and 1 s "
                            "of 3 Hz shake. Fingers 3, 4 and 5 stay extended and never touch the "
                            "cube — measured, not eyeballed: their contact force with the cube is "
                            "0.00 N throughout."))

    # ------------------------------------------------------------------ squeeze depth
    depths = []
    for mm in (18, 20, 22, 25, 28, 30):
        d = load(f"final_{mm}.json") or load(f"sweep_{mm}.json")
        if not d:
            continue
        R = d["results"]
        depths.append({
            "commanded pad separation mm": mm,
            "n": len(R),
            "held %": f"{100 * sum(r['held'] for r in R) / len(R):.0f}",
            "true pinch %": f"{100 * sum(r.get('pinched', False) for r in R) / len(R):.0f}",
            "median grip thumb N": f"{sorted(r['grip_thumb_N'] for r in R)[len(R) // 2]:.2f}",
            "median grip index N": f"{sorted(r['grip_index_N'] for r in R)[len(R) // 2]:.2f}",
        })
    if depths:
        S.append("<h2>How hard to squeeze</h2>")
        S.append("<p>The cube is 34 mm, so a commanded pad separation of 22 mm is 12 mm of "
                 "interference — 6 mm per side. Too little and only one pad ever loads; "
                 "&ldquo;true pinch&rdquo; below requires <i>both</i> pads carrying force from the "
                 "cube and the other three fingers carrying none.</p>")
        S.append(table(depths, highlight=""))

    if sweep:
        res = sweep["results"]
        n_h = sum(r["held"] for r in res)
        n_p = sum(r.get("pinched", False) for r in res)
        S.append("<h2>Full matrix, with the corrected instrumentation</h2>")
        S.append(f"<p>{len(res)} configurations, one per environment, in a single Isaac process "
                 f"(16 GB of VRAM fits exactly one). Held: <b>{n_h}/{len(res)}</b>; scored as a "
                 f"true two-pad pinch: <b>{n_p}/{len(res)}</b>.</p>")
        for a in ("posture", "ctrl", "ramp_s", "friction", "stiff4", "pin"):
            if a in res[0] and len({str(r[a]) for r in res}) > 1:
                S.append(f"<h3>by <code>{html.escape(a)}</code></h3>")
                S.append(table(summarise(res, a), highlight=""))

    # ------------------------------------------------------------------ reset families
    fam = out / "reset_families.png"
    if fam.exists():
        S.append("<h2>What the four reset banks contain today</h2>")
        S.append(image(fam, "Cube pose in each bank. Top row is the table from above, bottom row "
                            "from the side. Blue = hand open, red = hand closed, green x = the goal "
                            "cube, dashed = the configured sampling box."))
    rows = []
    meta = [
        ("C1 ObjectAnywhereEEAnywhere", "Reaching S^R", 4008,
         "cube resting anywhere in a 250x400 mm box; hand open, at an unrelated pose"),
        ("C2 ObjectRestingEEGrasped", "Near-Object S^NO", 103,
         "cube resting; hand closed on a bank grasp, +/-20 mm and +/-11 deg"),
        ("C3 ObjectAnywhereEEGrasped", "Stable Grasp S^G", 122,
         "cube in the air, 21-296 mm up; hand exactly on the grasp, no perturbation"),
        ("C4 ObjectPartiallyAssembledEEGrasped", "Near-Goal S^NG", 125,
         "cube seated on the goal cube; hand closed, +/-10 mm and +/-5.6 deg"),
    ]
    for (name, paper, n, what) in meta:
        key = name.split()[1]
        b = banks.get(key)
        rows.append({
            "bank": name, "paper": paper, "states": n, "what a state is": what,
            "replayed stable": f"{b['stable']}/{b['shown']}" if b else "—",
        })
    S.append(table(rows, highlight=""))

    for (name, _, _, _) in meta:
        key = name.split()[1]
        v = out / f"bank_before_{key}.mp4"
        if v.exists():
            b = banks.get(key)
            note = (f"{b['stable']}/{b['shown']} states held still for {b['settle_s']} s of physics."
                    if b else "")
            S.append(video(v, f"{name}: stored states replayed under real physics, "
                              f"{b['settle_s'] if b else 1.5} s each. Nothing is frozen — a state "
                              f"whose grasp does not hold drops the cube on camera. {note}"))

    # ------------------------------------------------------------------ bank defects
    S.append("<h2>Two defects in the existing banks, against the paper's own wording</h2>")
    S.append(
        "<ul>"
        "<li><b><code>partial_assemblies.pt</code> is degenerate.</b> 256 rows, <b>one</b> unique "
        "value: relative position exactly (0,&nbsp;0,&nbsp;34&nbsp;mm), identity rotation. The "
        "paper's &sect;3.2 specifies a <i>continuum</i>, produced by dislodging the object from the "
        "goal with small random forces. In this repo the dislodging term is set to "
        "<code>None</code> for the cube pair because the only surviving such term is a "
        "leg-thread-insertion sampler, and no generic replacement exists; every reset then writes "
        "the identical seated pose. The recorder's first batch also skips its own de-duplication, "
        "which is why there are exactly <code>num_envs</code>&nbsp;=&nbsp;256 copies. C4 is "
        "therefore built entirely from one already-solved state.</li>"
        "<li><b>C2's gripper is never opened.</b> The paper specifies the Near-Object family has "
        "the gripper &ldquo;randomly open or closed&rdquo;. "
        "<code>record_reset_states.py</code> computes "
        "<code>grasped = \"EEGrasped\" in task</code> and forces the close mask all-true for any "
        "matching task, so all 103 C2 states are closed-hand.</li>"
        "</ul>"
    )

    # ------------------------------------------------------------------ the pin artefact
    S.append("<h2>One protocol choice dominated everything</h2>")
    S.append(
        "<p>Holding the cube immovable while the hand closes, then releasing it, scored "
        "<b>0% true pinch in every single cell of the sweep</b>. Letting the cube float free and "
        "weightless during the close &mdash; which is what the repo's own sampler already does "
        "with its gravity schedule &mdash; scored 25&ndash;42%. The pin stores contact energy in "
        "the fingers and dumps it into the part the moment it is released. That is an artefact of "
        "the probe, not a property of the grasp, and it is worth stating because it is exactly the "
        "kind of thing that gets read as &ldquo;the hand ejects the cube&rdquo;.</p>"
    )

    # ------------------------------------------------------------------ optimising for hold
    S.append("<h2>The posture was optimised against the wrong objective</h2>")
    S.append(
        "<p>Every posture above was derived from <em>geometry</em>: pad separation, clearance of "
        "the non-pad links, drift of the grasp point while closing. All three are necessary and "
        "none of them is the objective. The objective is whether the cube is still in the hand "
        "after gravity and a shake &mdash; and nothing had ever been optimised for it. The "
        "postures say so plainly:</p>"
    )
    S.append(table([
        {"posture": "pinchpose (30 mm pads)", "geometry": "clean", "true pinch": "0%"},
        {"posture": "pinchpose_22 (22 mm pads)", "geometry": "clean", "true pinch": "23%"},
        {"posture": "pinchpose_seg (21.7 mm)", "geometry": "clears the sampler gate", "true pinch": "0%"},
    ], highlight=""))
    S.append(
        "<p>The third is the instructive one. Satisfying the grasp sampler's placement gate "
        "pushed the fingers off the cube, the pads stopped touching it entirely, and every "
        "geometric score still read as an improvement.</p>"
        "<p><code>pinchhold.py</code> closes that gap: a cross-entropy search over all twenty "
        "joints in which each candidate posture is scored by the full close &rarr; gravity "
        "&rarr; shake protocol rather than by a proxy. One candidate per <em>environment</em>, so "
        "a generation is a single Isaac run of a few hundred environments instead of hundreds of "
        "runs &mdash; the only reason this is affordable on one GPU. The clearance requirement "
        "stays as a hard filter, because a posture the placement gate rejects is unusable however "
        "well it grips.</p>"
        "<p>One detail matters for the result to mean anything: PhysX's batched solver is not "
        "bit-identical across environments, so a posture scored from a single environment is "
        "scored with a coin flip attached, and the search will happily chase a lucky draw. Each "
        "candidate is therefore averaged over four replicates before ranking.</p>"
    )

    compare = out / "posture_compare.mp4"
    if compare.exists():
        S.append(video(compare,
                       "Identical close, gravity and 3 Hz shake. Left: the repo's fist \u2014 51 mm "
                       "jaw around a 34 mm cube, the pads close either side of it, cube gone. "
                       "Middle: the naive two-finger posture, same 51 mm jaw \u2014 the cube ends "
                       "up resting in the crook of the curled fingers with 0 N on both pads. That "
                       "is the scoop this bench exists to tell apart from a grasp, and it scores "
                       "as \u2018held\u2019 on any position-only criterion. Right: the "
                       "hold-optimised posture \u2014 24.5 mm jaw, cube between thumb pad and "
                       "index tip clear of the palm, 1.7 N and 1.5 N on the two pads with fingers "
                       "3/4/5 at exactly zero."))

    hold = load("pinchhold.json")
    if hold:
        S.append("<h3>Result of the search</h3>")
        S.append(
            kpi(f"{hold['pinch_rate'] * 100:.0f}%", "true pinch (was 23.4%)")
            + kpi(f"{hold['held_rate'] * 100:.0f}%", "held (was 34.4%)")
            + kpi(f"{hold['close']['aperture_mm']:.1f} mm", "closed jaw")
            + kpi(f"{hold['single_best'].get('clearance_mm', 0):.0f} mm", "nearest non-pad body")
        )
        S.append(
            "<p>Re-evaluated across every environment rather than reported from the single "
            "generation that produced it &mdash; the per-generation best is the luckiest sample "
            "of its replicate set and is not the number to quote.</p>"
        )

    # ------------------------------------------------------------------ friction, refuted
    S.append("<h2>A finding I asserted and then had to withdraw</h2>")
    S.append(
        "<p>I presented surface friction as the binding constraint, with a safety-factor "
        "calculation putting the grasp at 0.98 &mdash; marginal by construction. Measuring it "
        "refuted that outright:</p>"
    )
    S.append(table([
        {"friction mu": 0.2, "held": "35.9%", "true pinch": "31.2%"},
        {"friction mu": 0.4, "held": "35.9%", "true pinch": "31.2%"},
        {"friction mu": 0.7, "held": "35.9%", "true pinch": "31.2%"},
        {"friction mu": 1.0, "held": "35.9%", "true pinch": "31.2%"},
    ], highlight=""))
    S.append(
        "<p class='warn'>Two process failures produced that claim and are worth recording. The "
        "first &ldquo;friction sweep&rdquo; changed the <em>posture</em> rather than the "
        "friction, so it measured nothing about friction at all. The second read stale "
        "<code>mu_*.json</code> files from a previous run &mdash; and the identical numbers "
        "across four values of mu, which should have been the tell that something was wrong, "
        "were instead read as a clean null result. The joint-friction change made to "
        "<code>ur10e_delto.py</code> on the strength of the original claim has been reverted.</p>"
    )

    # ------------------------------------------------------------------ the acceptance gate
    S.append("<h2>What blocks the pinch reset banks &mdash; measured</h2>")
    S.append(
        "<p>With the hold-optimised posture and its 321-grasp bank wired in, C3 recording still "
        "accepts nothing. <code>check_reset_state_success</code> ANDs six conditions; over "
        "<b>44 episodes at 128 environments</b> (5632 environment-episodes) with "
        "<code>UWLAB_GRASP_DEBUG=1</code>:</p>"
    )
    S.append(table([
        {"condition": "coll_free", "median /128": 124.0, "pass rate": "96.9%", "range": "118-128"},
        {"condition": "above_ground", "median /128": 70.5, "pass rate": "55.1%", "range": "58-85"},
        {"condition": "orient_down", "median /128": 35.0, "pass rate": "27.3%", "range": "27-44"},
        {"condition": "not_abnormal", "median /128": 32.0, "pass rate": "25.0%", "range": "26-42"},
        {"condition": "stable", "median /128": 29.0, "pass rate": "22.7%", "range": "21-37"},
        {"condition": "not_far", "median /128": 14.0, "pass rate": "10.9%", "range": "7-22"},
        {"condition": "success (all six)", "median /128": 0.0, "pass rate": "0.0%", "range": "0-0"},
    ], highlight=""))
    S.append(
        "<p>There is no single guilty threshold. Four conditions each pass only 11&ndash;27% and "
        "all six must hold in the same environment at the same instant; multiplied out that is "
        "<b>0.09%</b>, or 0.12 environments per 128. Measured, it is 0 in 5632 &mdash; so the "
        "conditions are positively correlated as well, and the true rate is lower still.</p>"
        "<p>Two of the four are velocity conditions: <code>not_abnormal</code> and "
        "<code>stable</code>. Three quarters of environments are in an abnormal robot state at "
        "the moment of evaluation. That is the gap between this and the bench, where the same "
        "posture holds 83.6% of the time &mdash; the bench has a fixed hand and no arm, and the "
        "arm is what is moving here.</p>"
        "<p class='warn'>So the pinch banks do not exist yet, and the reason is not the "
        "10&nbsp;mm <code>max_object_pos_deviation</code> that earlier looked like the culprit. "
        "Relaxing that alone would move <code>not_far</code> from 11% to perhaps 30% and leave "
        "the conjunction still near zero. The next thing to measure is why the arm is moving so "
        "much at reset, not which threshold to loosen.</p>"
    )

    # ------------------------------------------------------------------ what shipped
    S.append("<h2>What changed in the repo</h2>")
    S.append(
        "<p>One commit, everything opt-in and defaulted off, so no existing bank, gripper or task "
        "changes behaviour:</p>"
        "<ul>"
        "<li><code>DELTO_HAND_PINCH_{OPEN,CLOSE}_JOINT_POS</code> and "
        "<code>DELTO_PINCH_GRASP_CENTER_OFFSET</code> in <code>actions.py</code>.</li>"
        "<li><code>_PER_JOINT_PINCH_POSTURES</code> and an "
        "<code>env.cfg.scripted_gripper_pinch_posture</code> flag in "
        "<code>scripted_gripper.py</code>, resolved exactly like the existing "
        "<code>scripted_gripper_deadband_rad</code> &mdash; so neither recorder script needed "
        "editing.</li>"
        "<li>A cfg-level <code>grasp_center_offset</code> override on the grasp-sampling event. "
        "The pinch's grasp point is ~35 mm from the fist's, and <code>metadata.yaml</code> is "
        "shared by every object the sampler touches, so it cannot carry a posture-specific "
        "value.</li>"
        "<li><code>DeltoGraspSamplingCfg</code> opts in under <code>UWLAB_DELTO_PINCH=1</code>, "
        "setting the posture flag and the matching grasp centre <i>together</i> &mdash; they must "
        "move together, or objects get placed where the old closure used to grip.</li>"
        "</ul>"
        "<p>Also reverted: an earlier uncommitted change that lowered the hand's joint friction "
        "to 3e-4. Friction was a symptom, not the cause, and changing a calibrated constant to "
        "work around a controller bug is the wrong fix.</p>"
    )

    # ------------------------------------------------------------------ honest gap
    S.append("<h2>Not done, and why</h2>")
    S.append(
        "<ul>"
        "<li><b>The grasp sampler runs with the pinch but produces nothing yet.</b> Two of its "
        "three blockers are fixed and committed: the event manager rejected the new "
        "<code>grasp_center_offset</code> param until it was declared in the term's signature, "
        "the placement gate threw away all 61440 candidates because the grasp centre had moved to "
        "the pinch's while the pre-grasp posture had not, and the three direction vectors in "
        "metadata are fist-derived too &mdash; <code>grasp_align_axis</code> is the closing axis "
        "the sampler aligns with the object's width, so pointing it at the fist's jaw rotates "
        "the hand until a pad enters the cube. Deriving those axes by forward kinematics "
        "improved the deepest intersection from 14.44 mm to 12.73 mm. It still rejects all "
        "61440: "
        "<code>_filter_placement_collisions</code> samples the gripper's collision geometry at "
        "the articulation's <i>current</i> joint state, which is the spawn default splay, not "
        "the override. Fixing that means posing the hand before the gate runs rather than adding "
        "another cfg key. Everything writes to a separate dataset directory, so nothing existing "
        "is at risk.</li>"
        "<li><b>No bank has been regenerated with the pinch.</b> The plumbing is in and the "
        "posture is validated in a hand-only bench, but the grasp bank and C2/C3/C4 still contain "
        "five-finger enclosures. Regenerating them is the next step, and it needs the re-anchored "
        "grasp centre to be exercised end to end before the output is trusted.</li>"
        "<li><b>The four substage replay videos are not usable.</b> Replaying a bank works and its "
        "numbers are in the table above, but I could not aim a camera at that scene. Isaac Lab's "
        "<code>Camera</code> sensor cannot be pointed at all in this build &mdash; "
        "<code>set_world_poses_from_view</code> never moves the prim, <code>OffsetCfg</code> "
        "applies the translation but not the rotation, and writing the USD xform directly is "
        "overwritten by the sensor on its next update. All three report a correct pose on "
        "readback while rendering the sky. The replicator camera path does aim correctly but "
        "renders this scene 180&deg; rolled and I ran out of time framing it. The pinch videos "
        "above use a different scene whose camera happens to sit, by luck, exactly where it is "
        "needed &mdash; which is also how this went unnoticed for so long. A hand-frame fallback "
        "(replay each bank state on a floating hand, where rendering works) was built and hits "
        "the same wall: the camera cannot be aimed there either, and the recorded grasped states "
        "reconstruct with the cube interpenetrating the hand, which launches it. Both the "
        "extraction (<code>banklab.py --dump-relative</code>) and the replay "
        "(<code>handframe.py</code>) are written and working apart from the camera.</li>"
        "<li><b>The replayed cube jumps ~21 mm on the first physics step</b>, which means the "
        "replay scene does not match the one the banks were recorded in. The per-state drift "
        "numbers are reported separately from that pop, but the scene mismatch should be found "
        "before those numbers are quoted as bank quality.</li>"
        "</ul>"
    )

    return lede, S
