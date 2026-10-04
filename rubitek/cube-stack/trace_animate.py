"""Animate the recorded traces into an mp4, with no Isaac and no RTX involved.

Honest framing matters here: this is a STATE TRACE, not a photoreal render. It plots the measured
positions of the two pinching fingertips and the cube through a full 160-step episode, beside the
fraction of environments still holding the cube. Everything it draws was measured by the physics
engine; nothing is illustrative.

Pairs two runs so the comparison is the point: the shipped effort cap against the raised one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CS = Path("/home/mattephi/Work/Rubitek/cube-stack")


def load(name: str) -> dict:
    return json.loads((CS / "results" / f"trace_{name}.json").read_text())


def _tip_gap(D: dict, frame: int, ei: int) -> float:
    """Distance from the cube to the midpoint of the two pinching fingertips, one env, one frame."""
    c = D["cube_xyz"][frame][ei]
    a = D["tip1_xyz"][frame][ei]
    b = D["tip2_xyz"][frame][ei]
    mx, my, mz = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2
    return ((c[0] - mx) ** 2 + (c[1] - my) ** 2 + (c[2] - mz) ** 2) ** 0.5


def held_fraction(D: dict, upto: int) -> tuple[int, int]:
    """How many environments still hold the cube at `upto`, computed from the recorded positions.

    Deliberately NOT the stored `held_at_end`: that is sampled after the loop finishes, i.e. after the
    step-160 time-out has reset every environment, so it reports 59/64 for a run whose grasp was
    measured gone (in_hand 0.031) ten steps earlier. Putting that number in the panel title would have
    printed "59/64 held" directly above a curve showing total collapse. Comparing the tip gap at
    `upto` against step 1 reproduces the in_hand curve instead.
    """
    n_env = len(D["cube_xyz"][0])
    held = sum(1 for ei in range(n_env) if abs(_tip_gap(D, upto - 1, ei) - _tip_gap(D, 0, ei)) < 0.02)
    return held, n_env


def representative_env(D: dict, upto: int) -> int:
    """Index of the MEDIAN environment by cube-to-tip distance at the end of the window.

    Chosen by median rather than by picking env 0, which produced an "after" panel showing the cube
    dropped while the aggregate said 80% were holding -- env 0 simply happened to be one of the
    failures. Median is representative by construction and can be described as such honestly; taking
    the best-behaved environment would be cherry-picking.
    """
    cubes = D["cube_xyz"][upto - 1]
    t1 = D["tip1_xyz"][upto - 1]
    t2 = D["tip2_xyz"][upto - 1]
    dists = []
    for i, c in enumerate(cubes):
        mx = (t1[i][0] + t2[i][0]) / 2
        my = (t1[i][1] + t2[i][1]) / 2
        mz = (t1[i][2] + t2[i][2]) / 2
        dists.append((((c[0] - mx) ** 2 + (c[1] - my) ** 2 + (c[2] - mz) ** 2) ** 0.5, i))
    dists.sort()
    return dists[len(dists) // 2][1]


def main(a_name: str, b_name: str, out: str) -> None:
    A, B = load(a_name), load(b_name)
    # STOP AT 150, NOT 160. The episode is exactly 160 control steps, so the time-out fires on the
    # last one and every environment is reset before the sample is taken -- in_hand jumps from 0.047
    # back to 0.906 and the cube reappears in the hand. Animating that frame would end the "before"
    # clip with the cube leaping back into the grasp, which is the opposite of what happened.
    n = min(len(A["step"]), len(B["step"]), 150)

    fig = plt.figure(figsize=(11, 5.2), dpi=120)
    fig.patch.set_facecolor("white")
    axes = [fig.add_subplot(2, 2, i) for i in (1, 2)]
    axh = fig.add_subplot(2, 1, 2)

    ea, eb = representative_env(A, n), representative_env(B, n)
    # Both panels share one x/z frame so the two conditions are visually comparable; coordinates are
    # already relative to each environment's origin, so x reads ~0.48 m rather than ~5.8 m.
    allx, allz = [], []
    for D, ei in ((A, ea), (B, eb)):
        allx += [f[ei][0] for f in D["cube_xyz"][:n]] + [f[ei][0] for f in D["tip1_xyz"][:n]]
        allz += [f[ei][2] for f in D["cube_xyz"][:n]] + [f[ei][2] for f in D["tip1_xyz"][:n]]
    xlim = (min(allx) - 0.06, max(allx) + 0.06)
    zlim = (min(allz) - 0.04, max(allz) + 0.06)

    for ax, D, ttl, ei in ((axes[0], A, a_name, ea), (axes[1], B, b_name, eb)):
        eff = D["meta"]["effort_scale"]
        held, tot = held_fraction(D, n)
        frac = f"{held}/{tot} still holding at step {n}"
        ax.set_title(f"{ttl}  (effort cap x{eff})  — median env #{ei}, {frac}", fontsize=9)
        ax.set_xlabel("x (m, relative to env origin)", fontsize=8)
        ax.set_ylabel("z (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*zlim)

    artists = []
    for ax, D, ei in ((axes[0], A, ea), (axes[1], B, eb)):
        (cube,) = ax.plot([], [], "s", ms=11, color="#2f81f7", label="cube")
        (t1,) = ax.plot([], [], "o", ms=7, color="#d29922", label="thumb tip")
        (t2,) = ax.plot([], [], "o", ms=7, color="#3fb950", label="index tip")
        (trail,) = ax.plot([], [], "-", lw=1.0, color="#2f81f7", alpha=0.35)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.9)
        artists.append((cube, t1, t2, trail, D, ei))

    axh.set_xlabel("control step (episode is 160; success is scored at 160)", fontsize=8)
    axh.set_ylabel("fraction still holding", fontsize=8)
    axh.set_xlim(0, n)
    axh.set_ylim(0, 1.0)
    axh.grid(alpha=0.25, linewidth=0.5)
    axh.tick_params(labelsize=7)
    (ha,) = axh.plot([], [], lw=2, color="#d1242f", label=f"{a_name} (x{A['meta']['effort_scale']})")
    (hb,) = axh.plot([], [], lw=2, color="#2f81f7", label=f"{b_name} (x{B['meta']['effort_scale']})")
    (vline,) = axh.plot([], [], lw=1, color="#888", ls="--")
    axh.legend(fontsize=8, loc="upper right")

    def update(i):
        out_art = []
        for cube, t1, t2, trail, D, ei in artists:
            c = D["cube_xyz"][i][ei]
            p1 = D["tip1_xyz"][i][ei]
            p2 = D["tip2_xyz"][i][ei]
            cube.set_data([c[0]], [c[2]])
            t1.set_data([p1[0]], [p1[2]])
            t2.set_data([p2[0]], [p2[2]])
            xs = [f[ei][0] for f in D["cube_xyz"][: i + 1]]
            zs = [f[ei][2] for f in D["cube_xyz"][: i + 1]]
            trail.set_data(xs, zs)
            out_art += [cube, t1, t2, trail]
        ha.set_data(A["step"][: i + 1], A["in_hand"][: i + 1])
        hb.set_data(B["step"][: i + 1], B["in_hand"][: i + 1])
        vline.set_data([A["step"][i], A["step"][i]], [0, 1])
        return out_art + [ha, hb, vline]

    fig.suptitle(
        "DELTO two-finger pinch, C3 reset, NULL ACTION — measured state trace (not a render)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    anim = animation.FuncAnimation(fig, update, frames=n, interval=100, blit=False)
    writer = animation.FFMpegWriter(fps=10, bitrate=2400)
    anim.save(out, writer=writer)
    print(f"TRACE_ANIM wrote {out} ({n} frames)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
