"""Turn the endurance probe log into a table that can actually be chosen from.

The chain script picks on in_hand@150 alone, which is not sufficient: at effort x10 without the tuck
one or more envs recorded cube_to_tip = 8.7 m and cube_dz = -8.7 m, i.e. the cube was flung metres
away -- the EJECTION failure mode, back again at high force. A condition that holds 57% of cubes and
launches a few is not obviously better than one that holds 52% and launches none.

So this prints both, and flags any condition whose mean cube_to_tip exceeds a sane workspace span.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/mattephi/Work/Rubitek/cube-stack/logs/probe_local.log")
COND = re.compile(r"^=== COND (\S+) effort=x(\S+) tuck=(\S+) bias=(\S+)")
ROW = re.compile(
    r"PROBE (\S+) step=\s*(\d+) in_hand=([\d.]+) finger_dq_rad=([+-][\d.]+) "
    r"tip_force_N=([\d.nan]+) cube_to_tip=([\d.]+) cube_dz=([+-][\d.]+)"
)
# The reset banks span x 0.35..0.60, y +-0.20, so any mean cube-to-tip past half a metre means at
# least one cube left the cell entirely rather than merely being dropped.
EJECT_M = 0.5

conds: dict[str, tuple[str, str, str]] = {}
rows: dict[str, dict[int, tuple[float, float, float, float]]] = {}
for line in LOG.read_text(errors="replace").splitlines():
    m = COND.match(line)
    if m:
        conds[m.group(1)] = (m.group(2), m.group(3), m.group(4))
        continue
    m = ROW.search(line)
    if m:
        name, step = m.group(1), int(m.group(2))
        rows.setdefault(name, {})[step] = (
            float(m.group(3)), float(m.group(5)), float(m.group(6)), float(m.group(7))
        )

print(f"{'condition':<14}{'effort':>7}{'tuck':>6}{'bias':>6}{'force@20':>10}"
      f"{'hold@20':>9}{'hold@150':>10}{'cube_dz':>9}  stability")
print("-" * 88)
ranked = []
for name, steps in rows.items():
    eff, tuck, bias = conds.get(name, ("?", "?", "?"))
    s20, s150 = steps.get(20), steps.get(150)
    if not s150:
        continue
    hold20 = s20[0] if s20 else float("nan")
    force20 = s20[1] if s20 else float("nan")
    hold150, _, tip150, dz150 = s150
    ejected = tip150 > EJECT_M
    flag = "EJECTED cubes" if ejected else "ok"
    print(f"{name:<14}{'x' + eff:>7}{tuck:>6}{bias:>6}{force20:>10.2f}"
          f"{hold20:>9.3f}{hold150:>10.3f}{dz150:>9.3f}  {flag}")
    ranked.append((hold150, not ejected, name, eff, tuck, bias))

print()
stable = [r for r in ranked if r[1]]
if stable:
    best = max(stable)
    print(f"BEST STABLE: {best[2]}  effort=x{best[3]} tuck={best[4]} bias={best[5]}  hold@150={best[0]:.3f}")
if ranked:
    top = max(ranked)
    if not top[1]:
        print(f"(top raw score {top[2]} at {top[0]:.3f} is EXCLUDED: it ejects cubes out of the cell)")
