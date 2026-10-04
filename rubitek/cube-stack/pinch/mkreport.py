"""Assemble the pinch-grasp report as one self-contained HTML file under ~/reports/.

Videos are embedded as base64 data URIs so the page is a single file that can be copied
anywhere. Sections are declared in SECTIONS below; each one is skipped silently if its media
is not on disk yet, so this can be run at any point during the night and always produces a
readable page from whatever exists.
"""

from __future__ import annotations

import base64
import re
import subprocess
import html
import json
import os
import sys
from pathlib import Path

OUT_DIR = Path("/home/mattephi/Work/Rubitek/cube-stack/pinch/out")
REPORT = Path("/home/mattephi/reports/pinch_grasp.html")
URL = "https://mattecache.tailcc6bd2.ts.net/reports/pinch_grasp.html"


def data_uri(path: Path) -> str:
    mime = {
        ".mp4": "video/mp4",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".webm": "video/webm",
    }[path.suffix]
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def poster(path: Path) -> str:
    """A representative frame, as a data URI, for the video's poster attribute.

    Without one the page is a column of black rectangles until the reader presses play on each --
    which in a report whose findings ARE the videos means the findings are invisible on arrival.
    The frame is taken from two thirds through, not from the start: at t=0 the hand is open and
    every clip looks identical, whereas by then each one is showing its outcome.
    """
    cache = path.with_suffix(".poster.jpg")
    if not cache.exists() or cache.stat().st_mtime < path.stat().st_mtime:
        dur = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True).stdout.strip()
        at = f"{float(dur) * 0.66:.2f}" if dur else "1"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", at, "-i", str(path),
                        "-frames:v", "1", "-q:v", "4", str(cache)], check=True)
    return data_uri(cache)


def video(path: Path, caption: str = "") -> str:
    if not path.exists():
        return ""
    cap = f'<figcaption>{html.escape(caption)}</figcaption>' if caption else ""
    return (
        f'<figure><video controls loop muted playsinline preload="metadata" '
        f'poster="{poster(path)}" src="{data_uri(path)}"></video>{cap}</figure>'
    )


def image(path: Path, caption: str = "") -> str:
    if not path.exists():
        return ""
    cap = f'<figcaption>{html.escape(caption)}</figcaption>' if caption else ""
    return f'<figure><img src="{data_uri(path)}" alt="{html.escape(caption)}">{cap}</figure>'


def table(rows: list[dict], columns: list[str] | None = None, highlight: str = "held") -> str:
    if not rows:
        return ""
    columns = columns or list(rows[0])
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = []
    for r in rows:
        cls = ""
        if highlight in r:
            cls = ' class="ok"' if r[highlight] else ' class="bad"'
        cells = "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in columns)
        body.append(f"<tr{cls}>{cells}</tr>")
    return (
        '<div class="scroll"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def load(name: str):
    p = OUT_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


CSS = """
:root{--bg:#fbfaf8;--fg:#1c1b19;--mut:#66625c;--line:#e2ded7;--card:#fff;--ok:#1a7f4b;--bad:#a8342a;--acc:#2f5fa8}
:root:not([data-theme=light]) {}
@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--bg:#141312;--fg:#eae7e2;--mut:#9a948c;--line:#2e2c29;--card:#1c1b19;--ok:#5fd39a;--bad:#f08b80;--acc:#7fa8e8}}
:root[data-theme=dark]{--bg:#141312;--fg:#eae7e2;--mut:#9a948c;--line:#2e2c29;--card:#1c1b19;--ok:#5fd39a;--bad:#f08b80;--acc:#7fa8e8}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;padding:0 20px 80px}
main{max-width:1080px;margin:0 auto}
header{padding:48px 0 20px;border-bottom:1px solid var(--line);margin-bottom:32px}
h1{font-size:30px;margin:0 0 8px;letter-spacing:-.02em}
h2{font-size:21px;margin:44px 0 4px;letter-spacing:-.01em}
h3{font-size:16px;margin:26px 0 4px;color:var(--mut);font-weight:600}
.sub{color:var(--mut);margin:0}
.lede{font-size:17px;line-height:1.55;border-left:3px solid var(--acc);padding-left:16px;margin:18px 0 28px}
figure{margin:18px 0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
video,img{width:100%;display:block;border-radius:6px;background:#000}
figcaption{color:var(--mut);font-size:13px;margin-top:8px;line-height:1.45}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.scroll{overflow-x:auto;margin:14px 0}
table{border-collapse:collapse;font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;width:100%}
th,td{border-bottom:1px solid var(--line);padding:5px 9px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--mut);font-weight:600;border-bottom:2px solid var(--line)}
tr.ok td:first-child{border-left:3px solid var(--ok);padding-left:6px}
tr.bad td:first-child{border-left:3px solid var(--bad);padding-left:6px}
code{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:.9em}
.k{display:inline-block;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 12px;margin:4px 6px 4px 0}
.k b{display:block;font-size:20px;letter-spacing:-.01em}
.k span{color:var(--mut);font-size:12px}
p{margin:10px 0}
ul{margin:10px 0;padding-left:22px}
li{margin:5px 0}
.warn{border-left:3px solid var(--bad);padding-left:14px;color:var(--fg)}
details{border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:16px 0;background:var(--card)}
details summary{cursor:pointer;font-weight:600;font-size:15px;color:var(--fg);list-style:none}
details summary::-webkit-details-marker{display:none}
details summary:before{content:"\25b8  ";color:var(--mut)}
details[open] summary:before{content:"\25be  "}
details[open] summary{margin-bottom:8px;border-bottom:1px solid var(--line);padding-bottom:8px}
details h2{display:none}
"""

# Sections whose h2 matches any of these are collapsed by default. The findings, the videos and
# the retraction stay open; the long-form derivations that support them fold away. They are kept
# rather than deleted -- the reasoning is why the numbers are believable -- but they are not what
# someone opening this page needs to read first.
FOLD = (
    "Cause 1", "Cause 2", "Two measurements that were wrong",
    "Two defects in the existing banks", "One protocol choice dominated",
    "The pinch, frame by frame", "How hard to squeeze",
    "A finding I asserted and then had to withdraw",
    "The posture was optimised against the wrong objective",
    "What changed in the repo", "Not done, and why",
    "What the four reset banks contain today",
)


def fold_sections(body: str) -> str:
    """Wrap long-form sections in <details> so the page opens on results, not on derivations."""
    parts = re.split(r"(?=<h2>)", body)
    out = []
    for part in parts:
        m = re.match(r"<h2>(.*?)</h2>", part, re.S)
        if not m:
            out.append(part)
            continue
        heading = re.sub(r"<[^>]+>", "", m.group(1))
        if any(k.lower() in heading.lower() for k in FOLD):
            out.append(f"<details><summary>{heading}</summary>{part}</details>")
        else:
            out.append(part)
    return "".join(out)


def render(sections: list[str], title: str, lede: str) -> str:
    import datetime

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    body = fold_sections("\n".join(s for s in sections if s.strip()))
    return f"""<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
<main>
<header>
<h1>{html.escape(title)}</h1>
<p class="sub">DELTO DG-5F pinch grasp &middot; UWLab OmniReset cube stacking &middot; generated {stamp}</p>
</header>
<div class="lede">{lede}</div>
{body}
</main>
"""


def main() -> None:
    sections: list[str] = []
    lede = "Report in progress."

    # Section builders live in report_sections.py so the layout above stays stable while the
    # content is rewritten as results land.
    try:
        import report_sections

        lede, sections = report_sections.build(
            OUT_DIR, video=video, image=image, table=table, load=load
        )
    except ImportError:
        sections = ["<p>No sections module yet.</p>"]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(sections, "Pinch Grasp Bench", lede))
    size = REPORT.stat().st_size / 1e6
    print(f"WROTE {REPORT} ({size:.1f} MB)")
    print(URL)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
