"""Animate a POLICY rollout: is the trained policy actually doing the task?

The paired animator (`trace_animate.py`) compares two null-action conditions and answers "does the
grasp survive?". That is the endurance result, not the task. This one answers the different and more
important question -- does the cube get to where it has to go -- by plotting the cube against the
GOAL, with the success tolerance drawn explicitly.

Success in this task is the insertive cube's centre within 5 mm of the seated-stack pose, sampled at
the timeout. So cube-to-goal distance is the honest measure of policy performance; "still holding the
cube" is a precondition, not an achievement.

Stops at step 150 rather than 160: the episode is exactly 160 control steps, so the final sample is
taken after the time-out has reset every environment, which teleports the cube back to its start and
would draw a fake jump at the end of the clip.
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
SUCCESS_M = 0.005  # the task's own tolerance


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def median_env(D, upto):
    """Env with the MEDIAN final cube-to-goal distance -- representative, not flattering.

    Picking the best environment would be cherry-picking; picking env 0 already produced a panel that
    contradicted its own aggregate earlier tonight.
    """
    if "xyz_dist_env" in D:
        # The env's own distance (insertive cube in the receptive asset's frame) -- the quantity
        # success is actually scored on. The centre-to-centre proxy below reads ~34 mm for a
        # perfectly seated cube and would pick a "median" env by the wrong criterion entirely.
        d = sorted((D["xyz_dist_env"][upto - 1][i], i) for i in range(len(D["xyz_dist_env"][0])))
    else:
        d = sorted((dist(D["cube_xyz"][upto - 1][i], D["goal_xyz"][upto - 1][i]), i)
                   for i in range(len(D["cube_xyz"][0])))
    return d[len(d) // 2][1]


def main(name: str, out: str) -> None:
    D = json.loads((CS / "results" / f"trace_{name}.json").read_text())
    n = min(len(D["step"]), 150)
    ei = median_env(D, n)
    meta = D["meta"]
    n_env = len(D["cube_xyz"][0])

    # Threshold from the env itself, not a constant that happens to match today.
    thr = float(meta.get("success_position_threshold", SUCCESS_M))
    if "xyz_dist_env" in D:
        finals = D["xyz_dist_env"][n - 1]
    else:
        finals = [dist(D["cube_xyz"][n - 1][i], D["goal_xyz"][n - 1][i]) for i in range(n_env)]
    within = sum(1 for v in finals if v <= thr)
    med = sorted(finals)[len(finals) // 2]
    # The authoritative success flag (ProgressContext.success) when present -- the same quantity
    # training reports as task_N_success_rate.
    succ = D.get("success_env", [None])[n - 1] if "success_env" in D else None
    n_succ = sum(succ) if succ is not None else within

    fig = plt.figure(figsize=(11, 4.8), dpi=120)
    fig.patch.set_facecolor("white")
    axl = fig.add_subplot(1, 2, 1)
    axr = fig.add_subplot(1, 2, 2)

    # Include the FINGERTIPS in the limits. Framing on the cube and marker alone cropped the hand out
    # after the first step, so the only real motion in the clip -- the release and withdrawal --
    # happened off-screen and the panel looked static.
    cx = ([D["cube_xyz"][i][ei][0] for i in range(n)] + [D["goal_xyz"][0][ei][0]]
          + [D["tip1_xyz"][i][ei][0] for i in range(n)] + [D["tip2_xyz"][i][ei][0] for i in range(n)])
    cz = ([D["cube_xyz"][i][ei][2] for i in range(n)] + [D["goal_xyz"][0][ei][2]]
          + [D["tip1_xyz"][i][ei][2] for i in range(n)] + [D["tip2_xyz"][i][ei][2] for i in range(n)])
    axl.set_xlim(min(cx) - 0.07, max(cx) + 0.07)
    axl.set_ylim(min(cz) - 0.04, max(cz) + 0.08)
    axl.set_xlabel("x (m, relative to env origin)", fontsize=8)
    axl.set_ylabel("z (m)", fontsize=8)
    axl.tick_params(labelsize=7)
    axl.grid(alpha=0.25, lw=0.5)
    axl.set_title(f"median env #{ei} of {n_env} — {n_succ}/{n_env} succeed ({n_succ / n_env:.0%})", fontsize=9)

    g = D["goal_xyz"][0][ei]
    # Labelled for what it IS: the receptive cube's centre. It is NOT the goal pose -- the seated
    # stack sits one cube-height (~34 mm) above it. Calling this "goal" drew a marker 34 mm from the
    # cube beside a curve correctly reporting 4.4 mm, so the picture contradicted its own caption.
    # The goal distance is the right-hand panel, measured in the receptive asset's frame.
    axl.plot([g[0]], [g[2]], "*", ms=16, color="#d1242f", label="receptive cube centre", zorder=5)
    (cube,) = axl.plot([], [], "s", ms=11, color="#2f81f7", label="cube")
    (t1,) = axl.plot([], [], "o", ms=6, color="#d29922", label="thumb tip")
    (t2,) = axl.plot([], [], "o", ms=6, color="#3fb950", label="index tip")
    (trail,) = axl.plot([], [], "-", lw=1.0, color="#2f81f7", alpha=0.35)
    axl.legend(fontsize=6.5, loc="upper right", framealpha=0.9)

    # Prefer the ENV'S OWN distance (insertive cube expressed in the receptive asset's frame, which
    # is what success is actually scored on) over the centre-to-centre proxy, which reads ~34 mm for
    # a perfectly seated cube and made a working policy look like it never got close.
    if "xyz_dist_env" in D:
        dser = [D["xyz_dist_env"][i][ei] for i in range(n)]
        mser = D.get("xyz_distance", [])[:n]
    else:
        dser = [dist(D["cube_xyz"][i][ei], D["goal_xyz"][i][ei]) for i in range(n)]
        mser = D.get("cube_to_goal", [])[:n]
    axr.set_xlim(0, n)
    axr.set_ylim(0, max(max(dser), max(mser) if mser else 0) * 1.1 + 0.01)
    axr.set_xlabel("control step (success scored at 160)", fontsize=8)
    axr.set_ylabel("cube to goal (m)", fontsize=8)
    axr.tick_params(labelsize=7)
    axr.grid(alpha=0.25, lw=0.5)
    axr.axhline(thr, color="#d1242f", ls="--", lw=1, label=f"success tolerance {thr*1000:.0f} mm")
    (dline,) = axr.plot([], [], lw=2, color="#2f81f7", label=f"median env #{ei}")
    (mline,) = axr.plot([], [], lw=1.5, color="#8b949e", label=f"mean of {n_env} envs")
    axr.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"DELTO pinch — TRAINED POLICY on {meta.get('reset_family', '?')} "
        f"(effort x{meta.get('effort_scale', '?')}) — measured state trace, not a render",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def update(i):
        c = D["cube_xyz"][i][ei]
        p1 = D["tip1_xyz"][i][ei]
        p2 = D["tip2_xyz"][i][ei]
        cube.set_data([c[0]], [c[2]])
        t1.set_data([p1[0]], [p1[2]])
        t2.set_data([p2[0]], [p2[2]])
        trail.set_data([D["cube_xyz"][k][ei][0] for k in range(i + 1)],
                       [D["cube_xyz"][k][ei][2] for k in range(i + 1)])
        dline.set_data(range(1, i + 2), dser[: i + 1])
        if mser:
            mline.set_data(range(1, i + 2), mser[: i + 1])
        return [cube, t1, t2, trail, dline, mline]

    anim = animation.FuncAnimation(fig, update, frames=n, interval=100, blit=False)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=animation.FFMpegWriter(fps=10, bitrate=2400))
    # thr, not SUCCESS_M: they coincide today (both 5 mm), so this printed the right number by luck.
    # A constant that happens to match the config is a latent lie the moment the config moves.
    print(f"POLICY_ANIM wrote {out} ({n} frames); median final cube-to-goal {med*1000:.1f} mm; "
          f"{within}/{n_env} within {thr*1000:.0f} mm; ctx.success {n_succ}/{n_env}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
