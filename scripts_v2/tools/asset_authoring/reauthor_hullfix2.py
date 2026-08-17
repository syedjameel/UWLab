"""Diagnostic asset: self-collisions ON, but ONLY finger-vs-finger is checked.

Self-collisions ON cost 25 percent of reachable closure (71.96 -> 54.06 deg mean, and 83.77 -> 48.33
at the second phalanx). Two mechanisms produce that and they call for opposite fixes:

  (a) FINGERS BLOCK FINGERS. Physically correct, present in the reference too, and the policy simply
      has to learn a grasp that does not rely on fusing finger 3 through finger 4. Nothing to fix.
  (b) FINGERS HIT OUR TWO EXTRA SHELL BODIES. rl_dg_base and rl_dg_palm do not exist in the reference
      or in VALIK's hand -- both have a single rl_dg_mount. If our shells stand proud of where the
      reference's mount surface is, they cap the curl with geometry no reference robot has, and the
      cap is our graft's artifact rather than a real constraint.

This variant filters ALL THREE SHELL BODIES against ALL 25 finger bodies, leaving finger-vs-finger as
the only live self-collision. Run the closure test on it:

  closure back near 72 deg -> mechanism (b), the shells are the cap
  closure still near 54 deg -> mechanism (a), fingers block fingers and the constraint is real

IT IS A DIAGNOSTIC, NOT NECESSARILY THE SHIPPING ASSET. Filtering palm-vs-distal-phalanx removes a
constraint the reference genuinely enforces (its mount IS checked against N_2..N_4 and the tips), so
adopting this depends on the answer above.
"""

from __future__ import annotations

import shutil
import sys

from pxr import Usd, UsdPhysics

ROOT = "/home/dom-iva/github.com/orel/lerobot/UWLab_ur5edelto/source/uwlab_assets/uwlab_assets/local/Robots/Ur5eDelto"
SRC = f"{ROOT}/ur5e_delto_hullfix.usd"
DST = f"{ROOT}/ur5e_delto_hullfix2.usd"

SHELLS = ("rl_dg_mount", "rl_dg_base", "rl_dg_palm")
FINGERS = tuple(f"rl_dg_{i}_{j}" for i in range(1, 6) for j in range(1, 5)) + \
          tuple(f"rl_dg_{i}_tip" for i in range(1, 6))

shutil.copyfile(SRC, DST)
stage = Usd.Stage.Open(DST)
bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
missing = [n for n in SHELLS + FINGERS if n not in bodies]
if missing:
    raise SystemExit(f"REFUSING: bodies not found: {missing}")

n = 0
for s in SHELLS:
    api = UsdPhysics.FilteredPairsAPI.Apply(bodies[s])
    rel = api.CreateFilteredPairsRel()
    existing = {t.name for t in rel.GetTargets()}
    for f in FINGERS:
        if f not in existing:
            rel.AddTarget(bodies[f].GetPath())
        n += 1
stage.GetRootLayer().Save()

v = Usd.Stage.Open(DST)
got = {}
for prim in v.Traverse():
    if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
        got[prim.GetName()] = sorted(t.name for t in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets())
for k, vv in sorted(got.items()):
    print(f"  {k}: {len(vv)} filtered")
bad = [s for s in SHELLS if len(got.get(s, [])) != 25]
if bad:
    print(f"FAIL: expected 25 filtered targets on each shell, wrong on {bad}: "
          f"{ {s: len(got.get(s, [])) for s in bad} }")
    sys.exit(1)
print(f"wrote {DST}  ({n} shell-vs-finger pairs filtered; finger-vs-finger left LIVE)")
print("HULLFIX2_OK")
