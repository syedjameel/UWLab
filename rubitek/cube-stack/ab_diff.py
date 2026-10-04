"""Print two A/B arms' behaviour SIDE BY SIDE from their saved pose recordings.

WHY THIS EXISTS. accept_policy.sh prints one arm at a time, and comparing two separate printouts by
eye is how I misread a comparison earlier tonight. This puts the arms in one table with the delta
computed, so a difference has to be read off rather than spotted.

WHAT IT REFUSES TO DO. It does not report a success rate before reporting contact and transport.
Success on this task has been misleading three separate ways: the C4 reset starts already seated
(39/64 "successes" at step 0 with the cube never touched), `in_hand` is a distance-CHANGE proxy that
reported a flawless grasp of a cube 56 mm from open fingers, and anything sampled at or past step 160
reads the post-time-out reset rather than the episode.

Sampling is steps 0..149 ONLY, for that last reason.

Usage:  python3 ab_diff.py <labelA>=<a.npz> <labelB>=<b.npz>
"""

from __future__ import annotations

import sys

import numpy as np

# Column order follows DELTO_ALL_TIP_NAMES: (1_tip, 5_tip) + (2_tip, 3_tip, 4_tip). Thumb is column
# 0 and INDEX is column 2 -- column 1 is rl_dg_5_tip, which is undriven. Using column 1 once made the
# two-finger gate look unreachable even under a null action.
THUMB, INDEX = 0, 2
THRESH = 0.2
LAST = 149


def metrics(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    c, cp, gp, su = d["contact"], d["cube_pos"], d["goal_pos"], d["success"]
    E = cp.shape[1]
    if c.size == 0:
        raise SystemExit(f"{path}: no contact data -- cannot judge a grasp from silence")
    tot = c.sum(2)
    gate = (c[:, :, THUMB] > THRESH) & (c[:, :, INDEX] > THRESH)
    dist = np.linalg.norm(cp - gp, axis=2)
    z = cp[:150, :, 2]
    rel = np.asarray([np.argmax(~gate[:150, i]) if (~gate[:150, i]).any() else 150 for i in range(E)])
    mn = dist[:150].min(0)
    return {
        "envs": E,
        "force t=0 (N)": float(np.median(tot[0])),
        "force t=20 (N)": float(np.median(tot[20])),
        "gate t=0": f"{int(gate[0].sum())}/{E}",
        "gate t=149": f"{int(gate[LAST].sum())}/{E}",
        "steps gate open (median)": float(np.median(gate[:150].sum(0))),
        "release step (median)": float(np.median(rel)),
        "cube dropped to table": f"{int((z.min(0) < 0.04).sum())}/{E}",
        "cube-goal t=0 (mm)": float(np.median(dist[0])) * 1000,
        "cube-goal t=149 (mm)": float(np.median(dist[LAST])) * 1000,
        "closest approach (mm)": float(np.median(mn)) * 1000,
        "best single env (mm)": float(mn.min()) * 1000,
        "envs ever <100mm": f"{int((mn < 0.10).sum())}/{E}",
        "envs ever <50mm": f"{int((mn < 0.05).sum())}/{E}",
        "success t=149": f"{int(su[LAST].sum())}/{E}",
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    (la, pa), (lb, pb) = (a.split("=", 1) for a in sys.argv[1:3])
    ma, mb = metrics(pa), metrics(pb)

    w = max(len(k) for k in ma) + 2
    print(f"\n{'':<{w}}{la:>18}{lb:>18}{'delta':>12}")
    print("-" * (w + 48))
    for k in ma:
        va, vb = ma[k], mb[k]
        if isinstance(va, float):
            print(f"{k:<{w}}{va:>18.1f}{vb:>18.1f}{vb - va:>+12.1f}")
        else:
            print(f"{k:<{w}}{str(va):>18}{str(vb):>18}{'':>12}")

    # The transport question is the one the added reward term exists to answer, so it gets stated
    # explicitly rather than left for the reader to derive from the rows above.
    ta = ma["cube-goal t=0 (mm)"] - ma["cube-goal t=149 (mm)"]
    tb = mb["cube-goal t=0 (mm)"] - mb["cube-goal t=149 (mm)"]
    print(f"\n  NET TRANSPORT TOWARD GOAL (start - end, positive = progress)")
    print(f"    {la:>18}: {ta:+7.1f} mm")
    print(f"    {lb:>18}: {tb:+7.1f} mm")
    print(f"    difference        : {tb - ta:+7.1f} mm")
    print("\n  Neither arm has demonstrated a seated cube until 'success t=149' is non-zero.")


if __name__ == "__main__":
    main()
