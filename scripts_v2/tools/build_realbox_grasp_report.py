# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the RealBox grasp report: an interactive, self-contained HTML page.

Contents:
  * a zero-dependency canvas 3D explorer (orbit/zoom) showing every recorded grasp pose on the
    object's collision geometry (pads drawn at the recorded jaw aperture),
  * the Isaac replay montages from diag_render_recorded_grasps_linear.py (placed + settled),
  * the reset-state dataset table (counts + diversity), read live from the dataset tree.

Run (plain python, no kit needed):
    python scripts_v2/tools/build_realbox_grasp_report.py \
        --out reports/realbox_grasp_report.html
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os

import torch

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(TOOLS))
OBJECTS = {"Bottom": "tray", "Mid": "circuit", "CapRim": "cap"}
NICE = {"Bottom": "Box (tray)", "Mid": "Object (circuit board)", "CapRim": "Cover (cap)"}
MAX_APERTURE = 0.137


def b64img(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def load_viewer_data(dataset_dir: str) -> dict:
    boxes = json.load(open(os.path.join(TOOLS, "realbox_boxassembly_boxes.json")))
    out = {}
    for obj, key in OBJECTS.items():
        p = os.path.join(dataset_dir, "Grasps", obj, "grasps.pt")
        d = torch.load(p, weights_only=False, map_location="cpu")
        g = d.get("grasp_relative_pose", d)
        rp = torch.stack([t.flatten().cpu() for t in g["relative_position"]]).numpy()
        rq = torch.stack([t.flatten().cpu() for t in g["relative_orientation"]]).numpy()
        jq = [torch.stack([t.flatten().cpu() for t in v]).numpy().ravel() for v in g["gripper_joint_positions"].values()]
        jaw = (jq[0] + jq[1]) / 2.0
        out[obj] = {
            "name": NICE[obj],
            "boxes": [{"min": b["min"], "max": b["max"]} for b in boxes[key]],
            "pos": rp.round(5).tolist(),
            "quat": rq.round(5).tolist(),
            "ap": [round(float(MAX_APERTURE - 2 * q), 5) for q in jaw],
        }
    return out


# Pair directories are named from the two USD folders sorted alphabetically, so the name alone
# does not say which side is the insertive (grasped) object. The three stages are fixed:
PAIR_INSERTIVE = {"Bottom__TableCenterTarget": "Bottom", "Bottom__Mid": "Mid", "Bottom__CapRim": "CapRim"}
FINGER_COLS = (6, 7)  # pan, lift, elbow, wrist1-3, finger, right_finger
GRIP_SLACK = 0.006  # m of aperture either side of the recorded grasp range still counts as a hold


def grasp_aperture_range(dataset_dir: str, obj: str) -> tuple[float, float]:
    d = torch.load(os.path.join(dataset_dir, "Grasps", obj, "grasps.pt"), weights_only=False, map_location="cpu")
    j = d.get("grasp_relative_pose", d)["gripper_joint_positions"]
    q = [torch.stack([t.flatten().cpu() for t in v]).numpy().ravel() for v in j.values()]
    ap = MAX_APERTURE - q[0] - q[1]
    return float(ap.min()), float(ap.max())


def reset_stats(ur10e_ds: str, dataset_dir: str) -> list[dict]:
    """Per-dataset counts, spawn diversity, and — for the EEGrasped types — how many states
    actually hold the object.

    The jaw aperture recorded in a reset state must land near one of the apertures in that
    object's grasp dataset. A state that settled fully closed clamped nothing (the object was
    ejected); one that settled fully open was never commanded shut. Both record as successes,
    because the reset-state success gate only checks object stability and collision-freedom, so
    without this column a broken dataset is indistinguishable from a good one.
    """
    rows = []
    windows: dict[str, tuple[float, float]] = {}
    for f in sorted(glob.glob(os.path.join(ur10e_ds, "Resets", "*", "resets_*.pt"))):
        pair = os.path.basename(os.path.dirname(f))
        rtype = os.path.basename(f)[len("resets_"):-3]
        try:
            d = torch.load(f, weights_only=False, map_location="cpu")
            ins = d["initial_state"]["rigid_object"]["insertive_object"]["root_pose"]
            z = torch.stack([t.flatten().cpu() for t in ins])[:, 2]
            grip = ""
            obj = PAIR_INSERTIVE.get(pair)
            if "EEGrasped" in rtype and obj:
                if obj not in windows:
                    windows[obj] = grasp_aperture_range(dataset_dir, obj)
                lo, hi = windows[obj]
                jp = torch.stack([t.flatten().cpu() for t in d["initial_state"]["articulation"]["robot"]["joint_position"]])
                ap = (MAX_APERTURE - jp[:, FINGER_COLS[0]] - jp[:, FINGER_COLS[1]]).numpy()
                held = ((ap >= lo - GRIP_SLACK) & (ap <= hi + GRIP_SLACK)).mean()
                grip = f"{100 * held:.1f}%"
            rows.append({"pair": pair, "type": rtype, "n": len(ins), "grip": grip,
                         "z_std": round(float(z.std()), 4), "size_mb": round(os.path.getsize(f) / 1e6, 1)})
        except Exception as e:  # noqa: BLE001
            rows.append({"pair": pair, "type": rtype, "n": -1, "grip": "", "z_std": float("nan"), "size_mb": 0.0,
                         "err": str(e)[:60]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default=os.path.join(ROOT, "Datasets/OmniReset"))
    ap.add_argument("--ur10e_ds", default=os.path.join(ROOT, "source/uwlab_assets/data/Datasets_ur10e/OmniReset"))
    ap.add_argument("--diag_dir", default=os.path.join(ROOT, "reports/omnireset/figures/diag"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reports/realbox_grasp_report.html"))
    ap.add_argument("--extra_note", default="")
    args = ap.parse_args()

    data = load_viewer_data(args.dataset_dir)
    resets = reset_stats(args.ur10e_ds, args.dataset_dir)

    montages = {}
    held = {}
    for obj in OBJECTS:
        low = obj.lower()
        montages[obj] = {
            "placed": b64img(os.path.join(args.diag_dir, f"linear_grasps_{low}_placed.png")),
            "settled": b64img(os.path.join(args.diag_dir, f"linear_grasps_{low}_settled.png")),
        }
        # HELD stats are printed by the diag tool; read the sidecar if present
        side = os.path.join(args.diag_dir, f"linear_grasps_{low}_held.txt")
        held[obj] = open(side).read().strip() if os.path.exists(side) else ""

    grasp_rows = "".join(
        f'<div class="lrow"><div class="lkey">{d["name"]}</div>'
        f'<div class="lval">{len(d["pos"])} grasps · aperture {min(d["ap"])*1000:.0f}–{max(d["ap"])*1000:.0f} mm</div></div>'
        for d in data.values()
    )
    reset_rows = "".join(
        f'<tr><td>{r["pair"]}</td><td>{r["type"]}</td><td class="num">{r["n"]}</td>'
        f'<td class="num">{r["grip"] or "&mdash;"}</td>'
        f'<td class="num">{r["z_std"]}</td><td class="num">{r["size_mb"]}</td></tr>'
        for r in resets
    )
    montage_html = ""
    for obj in OBJECTS:
        m = montages[obj]
        if not m["placed"]:
            continue
        cap = f'<p class="mheld">{held[obj]}</p>' if held[obj] else ""
        montage_html += f"""
      <h3>{NICE[obj]}</h3>
      <div class="mgrid">
        <figure><img src="{m['placed']}" alt="{obj} placed"><figcaption>placed — recorded pose + jaw angles</figcaption></figure>
        <figure><img src="{m['settled']}" alt="{obj} settled"><figcaption>after 30 steps of gravity, jaws commanded closed</figcaption></figure>
      </div>{cap}"""

    html = HTML_TEMPLATE
    html = html.replace("/*__DATA__*/", "const DATA = " + json.dumps(data, separators=(",", ":")) + ";")
    html = html.replace("<!--__GRASP_ROWS__-->", grasp_rows)
    html = html.replace("<!--__RESET_ROWS__-->", reset_rows)
    html = html.replace("<!--__MONTAGES__-->", montage_html)
    html = html.replace("<!--__NOTE__-->", args.extra_note)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


HTML_TEMPLATE = r"""<title>RealBox — recorded grasps on the training models</title>
<style>
:root{
  --paper:#f6f3ec; --ink:#1d1c1a; --ink-soft:#5a564e; --line:#d8d2c4; --card:#fffdf7;
  --accent:#8a4b2c; --good:#2c6e49; --warn:#a86212; --mono:ui-monospace,'SF Mono',Menlo,monospace;
}
@media (prefers-color-scheme: dark){
  :root{ --paper:#181614; --ink:#e8e4da; --ink-soft:#a49c8e; --line:#3a352d; --card:#211e1a;
         --accent:#d08a5e; --good:#7fbf9a; --warn:#d9a05b; }
}
:root[data-theme="dark"]{ --paper:#181614; --ink:#e8e4da; --ink-soft:#a49c8e; --line:#3a352d; --card:#211e1a;
         --accent:#d08a5e; --good:#7fbf9a; --warn:#d9a05b; }
:root[data-theme="light"]{ --paper:#f6f3ec; --ink:#1d1c1a; --ink-soft:#5a564e; --line:#d8d2c4; --card:#fffdf7;
         --accent:#8a4b2c; --good:#2c6e49; --warn:#a86212; }
html{background:var(--paper);color:var(--ink);font:16px/1.6 Georgia,'Times New Roman',serif}
body{max-width:1060px;margin:0 auto;padding:2.2rem 1.2rem 5rem}
h1{font-size:2rem;line-height:1.2;margin:.2rem 0 .4rem;text-wrap:balance}
h2{font-size:1.28rem;margin:0 0 .8rem;border-bottom:1px solid var(--line);padding-bottom:.45rem}
h3{font-size:1.02rem;margin:1.4rem 0 .5rem}
.eyebrow{font:600 .72rem/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}
.sub{color:var(--ink-soft);margin:.1rem 0 0}
section{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:1.5rem 1.7rem;margin:1.6rem 0}
.secnum{font:600 .72rem/1 var(--mono);color:var(--ink-soft);letter-spacing:.14em}
.lrow{display:flex;gap:1rem;border-bottom:1px dotted var(--line);padding:.42rem 0}
.lrow:last-child{border-bottom:0}
.lkey{flex:0 0 240px;font-weight:600}
.lval{color:var(--ink-soft);font-family:var(--mono);font-size:.85rem}
table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:.82rem}
th,td{padding:.35rem .6rem;border-bottom:1px solid var(--line);text-align:left}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
th{font-weight:600;color:var(--ink-soft)}
.mgrid{display:grid;grid-template-columns:1fr 1fr;gap:.9rem}
@media(max-width:760px){.mgrid{grid-template-columns:1fr}}
figure{margin:0}
figure img{max-width:100%;border:1px solid var(--line);border-radius:4px;display:block}
figcaption{font:italic .82rem/1.4 Georgia,serif;color:var(--ink-soft);margin-top:.3rem}
.mheld{font-family:var(--mono);font-size:.82rem;color:var(--good)}
#viewer-wrap{position:relative}
#viewer{width:100%;height:560px;border:1px solid var(--line);border-radius:4px;background:
  radial-gradient(ellipse at 50% 30%, rgba(128,128,128,.06), transparent 70%);touch-action:none;cursor:grab}
.vbar{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin:.7rem 0}
.vbar button{font:600 .78rem var(--mono);letter-spacing:.05em;padding:.38rem .8rem;border:1px solid var(--line);
  border-radius:4px;background:transparent;color:var(--ink);cursor:pointer}
.vbar button.on{background:var(--accent);border-color:var(--accent);color:var(--paper)}
.vbar label{font:.78rem var(--mono);color:var(--ink-soft);display:flex;align-items:center;gap:.4rem}
#ginfo{font:.8rem var(--mono);color:var(--ink-soft);min-height:1.3em}
.note{border-left:3px solid var(--warn);padding:.5rem .9rem;color:var(--ink-soft);font-size:.92rem}
a{color:var(--accent)}
</style>

<p class="eyebrow">OmniReset · RealBox training port</p>
<h1>Recorded grasps on the customer enclosure models</h1>
<p class="sub">UWLab_jameel · branch omnireset/realbox-boxassembly · linear gripper, top-down sampler</p>

<section>
  <div class="secnum">01 · GRASP DATASETS</div>
  <h2>What was recorded</h2>
  <p>One <span style="font-family:var(--mono)">grasps.pt</span> per manipulated part, generated by the
  gripper-only sampling task (<span style="font-family:var(--mono)">OmniReset-LinearGripper-GraspSampling-v0</span>,
  256 envs). Every stored grasp passed the in-sim shake validation. The gripper pose is stored in the
  object frame; jaw angles are the recorded closed configuration.</p>
  <!--__GRASP_ROWS__-->
  <!--__NOTE__-->
</section>

<section>
  <div class="secnum">02 · INTERACTIVE EXPLORER</div>
  <h2>Every grasp pose on the collision geometry</h2>
  <p>Drag to orbit, wheel to zoom. Each glyph is one recorded grasp: two pads at the recorded jaw
  aperture, a stem along the approach axis. Pads are coloured by aperture (dark = narrow pinch,
  bright = wide clamp). Hover a glyph for numbers; click to pin it.</p>
  <div class="vbar" id="objtabs"></div>
  <div class="vbar">
    <label><input type="checkbox" id="showall" checked> show all grasps</label>
    <label>sample <input type="range" id="count" min="1" max="256" value="256" style="width:140px"><span id="countv"></span></label>
    <label><input type="checkbox" id="showboxes" checked> collision boxes</label>
  </div>
  <div id="viewer-wrap"><canvas id="viewer"></canvas></div>
  <div id="ginfo">&nbsp;</div>
</section>

<section>
  <div class="secnum">03 · ISAAC REPLAY</div>
  <h2>Replayed in simulation: placed, then held under gravity</h2>
  <p>Independent replay of the recorded grasps with the real gripper articulation: teleport to the
  recorded pose, set the recorded jaw angles, command closed, run gravity. A grasp counts as held if
  the object stays off the floor.</p>
  <!--__MONTAGES__-->
</section>

<section>
  <div class="secnum">04 · RESET-STATE DATASETS</div>
  <h2>Downstream datasets these grasps feed</h2>
  <p>Reset states recorded per stage pair into the UR10e dataset tree (4-entity scenes: box, object,
  cover, target). <span style="font-family:var(--mono)">z_std</span> is the insertive object's spawn
  height spread — the diversity signal that made the original recipe trainable.
  <span style="font-family:var(--mono)">holding</span> is the share of an
  <span style="font-family:var(--mono)">EEGrasped</span> dataset whose recorded jaw aperture matches
  one this object was actually sampled at: the reset-state success gate checks only stability and
  collision-freedom, so a state that settled fully closed (object ejected) or fully open (jaws never
  commanded shut) is banked as a success and is invisible in the state count alone.</p>
  <table>
    <thead><tr><th>pair</th><th>reset type</th><th class="num">states</th><th class="num">holding</th><th class="num">insertive z-std [m]</th><th class="num">MB</th></tr></thead>
    <tbody><!--__RESET_ROWS__--></tbody>
  </table>
</section>

<script>
/*__DATA__*/
(function(){
"use strict";
const cv = document.getElementById("viewer");
const ctx = cv.getContext("2d");
const info = document.getElementById("ginfo");
const tabs = document.getElementById("objtabs");
const names = Object.keys(DATA);
let cur = names[0], yaw = 0.7, pitch = 0.42, dist = 0.34, pinned = -1, hovered = -1;
let showAll = true, sampleN = 256, showBoxes = true;

function css(v){ return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }

names.forEach(n => {
  const b = document.createElement("button");
  b.textContent = DATA[n].name;
  b.onclick = () => { cur = n; pinned = -1;
    document.querySelectorAll("#objtabs button").forEach(x => x.classList.remove("on"));
    b.classList.add("on"); syncSlider(); draw(); };
  tabs.appendChild(b);
});
tabs.firstChild.classList.add("on");

const showallEl = document.getElementById("showall");
const countEl = document.getElementById("count");
const countV = document.getElementById("countv");
const showboxEl = document.getElementById("showboxes");
function syncSlider(){
  countEl.max = DATA[cur].pos.length;
  if (+countEl.value > +countEl.max) countEl.value = countEl.max;
  countV.textContent = showAll ? DATA[cur].pos.length : countEl.value;
}
showallEl.onchange = () => { showAll = showallEl.checked; syncSlider(); draw(); };
countEl.oninput = () => { showAll = false; showallEl.checked = false; syncSlider(); draw(); };
showboxEl.onchange = () => { showBoxes = showboxEl.checked; draw(); };

function qrot(q, v){ // wxyz quat rotate
  const [w,x,y,z] = q, [vx,vy,vz] = v;
  const tx = 2*(y*vz - z*vy), ty = 2*(z*vx - x*vz), tz = 2*(x*vy - y*vx);
  return [vx + w*tx + (y*tz - z*ty), vy + w*ty + (z*tx - x*tz), vz + w*tz + (x*ty - y*tx)];
}
function add(a,b){ return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }

let W, H, DPR;
function resize(){
  DPR = window.devicePixelRatio || 1;
  W = cv.clientWidth; H = cv.clientHeight || 560;
  cv.width = W*DPR; cv.height = H*DPR; ctx.setTransform(DPR,0,0,DPR,0,0);
}
window.addEventListener("resize", () => { resize(); draw(); });

function project(p, c){
  // camera orbit around object centre c
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  let x = p[0]-c[0], y = p[1]-c[1], z = p[2]-c[2];
  let x1 = cy*x + sy*y, y1 = -sy*x + cy*y;          // yaw about z
  let y2 = cp*y1 - sp*z, z2 = sp*y1 + cp*z;         // pitch
  const depth = x1 + dist;                          // camera looks along -x1
  const f = 0.9*Math.min(W,H)/dist;
  return [W/2 + (-y2)*f*dist/Math.max(depth,1e-4), H/2 - z2*f*dist/Math.max(depth,1e-4), depth];
}

function centre(){
  const bs = DATA[cur].boxes;
  let lo=[9,9,9], hi=[-9,-9,-9];
  bs.forEach(b => { for(let i=0;i<3;i++){ lo[i]=Math.min(lo[i],b.min[i]); hi[i]=Math.max(hi[i],b.max[i]); } });
  return [(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,(lo[2]+hi[2])/2 + 0.02];
}

function boxFaces(b){
  const [x0,y0,z0]=b.min,[x1,y1,z1]=b.max;
  const v=[[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]];
  return [[v[0],v[1],v[2],v[3]],[v[4],v[5],v[6],v[7]],[v[0],v[1],v[5],v[4]],
          [v[2],v[3],v[7],v[6]],[v[0],v[3],v[7],v[4]],[v[1],v[2],v[6],v[5]]];
}

const PAD = {w:0.008, d:0.024, z0:0.114, z1:0.144};  // jaw pad geometry in the gripper frame

let glyphHits = [];
function draw(){
  const d = DATA[cur], c = centre();
  ctx.clearRect(0,0,W,H);
  const ink = css("--ink"), soft = css("--ink-soft"), line = css("--line"), accent = css("--accent");
  const polys = [];

  if (showBoxes) d.boxes.forEach(b => boxFaces(b).forEach(f => {
    const pp = f.map(p => project(p,c));
    polys.push({z: (pp[0][2]+pp[1][2]+pp[2][2]+pp[3][2])/4, pts: pp, fill: "rgba(128,120,100,0.10)", stroke: line, kind:"box"});
  }));

  const n = showAll ? d.pos.length : Math.min(+countEl.value, d.pos.length);
  const apMin = Math.min(...d.ap), apMax = Math.max(...d.ap);
  glyphHits = [];
  for (let i=0;i<n;i++){
    const q = d.quat[i], o = d.pos[i], ap = d.ap[i];
    const t = apMax>apMin ? (ap-apMin)/(apMax-apMin) : 0.5;
    const sel = (i===pinned || i===hovered);
    const col = sel ? accent : `rgba(${Math.round(120+110*t)},${Math.round(90+60*t)},${Math.round(60+30*(1-t))},0.85)`;
    for (const s of [-1,1]){
      const x = s*ap/2;
      const face = [[x, -PAD.d/2, PAD.z0],[x, PAD.d/2, PAD.z0],[x, PAD.d/2, PAD.z1],[x, -PAD.d/2, PAD.z1]]
        .map(v => add(qrot(q, v), o));
      const pp = face.map(p => project(p,c));
      polys.push({z:(pp[0][2]+pp[1][2]+pp[2][2]+pp[3][2])/4, pts: pp, fill: col, stroke: sel?accent:"rgba(0,0,0,0.18)", kind:"pad", idx:i});
    }
    const stem = [add(qrot(q,[0,0,PAD.z0]), o), add(qrot(q,[0,0,0.06]), o)].map(p => project(p,c));
    polys.push({z:(stem[0][2]+stem[1][2])/2, line: stem, stroke: sel?accent:"rgba(128,120,100,0.35)", kind:"stem", idx:i});
  }

  polys.sort((a,b) => b.z - a.z);
  for (const p of polys){
    if (p.line){
      ctx.strokeStyle = p.stroke; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(p.line[0][0], p.line[0][1]); ctx.lineTo(p.line[1][0], p.line[1][1]); ctx.stroke();
      continue;
    }
    ctx.beginPath();
    p.pts.forEach((pt,i) => i ? ctx.lineTo(pt[0],pt[1]) : ctx.moveTo(pt[0],pt[1]));
    ctx.closePath();
    if (p.fill){ ctx.fillStyle = p.fill; ctx.fill(); }
    ctx.strokeStyle = p.stroke; ctx.lineWidth = 0.7; ctx.stroke();
    if (p.kind === "pad") glyphHits.push({idx:p.idx, pts:p.pts});
  }
  const which = pinned>=0 ? pinned : hovered;
  if (which>=0){
    const o = d.pos[which];
    info.textContent = `grasp #${which} · aperture ${(d.ap[which]*1000).toFixed(1)} mm · pos [${o.map(v=>(v*1000).toFixed(1)).join(", ")}] mm (object frame)` + (pinned>=0 ? " · pinned (click empty space to release)" : "");
    info.style.color = css("--accent");
  } else {
    info.textContent = `${d.name} — ${n} of ${d.pos.length} grasps shown`;
    info.style.color = soft;
  }
}

function hitTest(mx,my){
  for (let i=glyphHits.length-1;i>=0;i--){
    const g = glyphHits[i], p = g.pts;
    let inside = false;
    for (let a=0,b=p.length-1;a<p.length;b=a++){
      if (((p[a][1]>my)!==(p[b][1]>my)) && (mx < (p[b][0]-p[a][0])*(my-p[a][1])/(p[b][1]-p[a][1])+p[a][0])) inside=!inside;
    }
    if (inside) return g.idx;
  }
  return -1;
}

let drag=false, moved=false, lx=0, ly=0;
cv.addEventListener("pointerdown", e => { drag=true; moved=false; lx=e.clientX; ly=e.clientY; cv.setPointerCapture(e.pointerId); });
cv.addEventListener("pointermove", e => {
  if (drag){
    const dx=e.clientX-lx, dy=e.clientY-ly;
    if (Math.abs(dx)+Math.abs(dy)>2) moved=true;
    yaw += dx*0.008; pitch = Math.max(-1.4, Math.min(1.4, pitch + dy*0.008));
    lx=e.clientX; ly=e.clientY; draw();
  } else {
    const r = cv.getBoundingClientRect();
    const h = hitTest(e.clientX-r.left, e.clientY-r.top);
    if (h!==hovered){ hovered=h; draw(); }
  }
});
cv.addEventListener("pointerup", e => {
  drag=false;
  if (!moved){
    const r = cv.getBoundingClientRect();
    const h = hitTest(e.clientX-r.left, e.clientY-r.top);
    pinned = (h===pinned) ? -1 : h;
    draw();
  }
});
cv.addEventListener("wheel", e => { e.preventDefault(); dist = Math.max(0.12, Math.min(1.2, dist*(1+Math.sign(e.deltaY)*0.08))); draw(); }, {passive:false});

resize(); syncSlider(); draw();
new MutationObserver(() => draw()).observe(document.documentElement, {attributes:true, attributeFilter:["data-theme"]});
if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => draw());
})();
</script>
"""


if __name__ == "__main__":
    main()
