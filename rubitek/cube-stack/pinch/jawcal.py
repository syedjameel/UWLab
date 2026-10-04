"""Is there a DELTO posture in which the thumb and index fingertips actually oppose?

WHY
---
``DeltoHand/metadata.yaml`` declares the hand's grasp geometry to be "a virtual jaw of finger 1
against finger 2" and gives ``grasp_center_offset = midpoint(rl_dg_1_tip, rl_dg_2_tip)`` at the
scripted CLOSE posture. But measured in sim at that very posture, ``|rl_dg_1_tip - rl_dg_2_tip|``
is 77.5 mm -- wider than the hand's own declared 59 mm maximum aperture, and more than twice the
34 mm cube it is supposed to pinch. If that holds, the "closed" posture is not a jaw at all, and
no amount of friction, stiffness or control tuning will make it grip: the two pads never meet.

This script answers the prior question by search rather than by argument. It sweeps the eight
joints of fingers 1 and 2 and measures, for each sample, the tip-to-tip aperture and where the
midpoint sits in the MOUNT frame (the frame ``grasp_center_offset`` is expressed in).

WHAT IT REPORTS
  * the minimum achievable aperture, and the posture that achieves it
  * the postures whose aperture is a working pinch on a 34 mm cube -- pads just inside the cube
    so they squeeze it, i.e. aperture in [PINCH_LO, PINCH_HI]
  * for those, the jaw centre in mount-frame coordinates, to be compared against the metadata's
    (63.7, 46.5, 150.0) mm
  * a matching PRE-GRASP posture: the same joints backed off until the aperture clears the cube

Kinematics only. Joint states are written directly and the PD target is written to match, so
the sim is being used as a forward-kinematics oracle and nothing is being dynamically settled.
"""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--envs", type=int, default=1024)
parser.add_argument("--batches", type=int, default=160)
parser.add_argument("--out", type=str, default="/home/mattephi/Work/Rubitek/cube-stack/pinch/out/jawcal.json")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import uwlab_is6_bridge  # noqa: E402

uwlab_is6_bridge.apply()

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from uwlab_assets.robots.ur10e_delto.actions import (  # noqa: E402
    DELTO_HAND_OPEN_JOINT_POS,
    DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS,
    DELTO_TUCKED_HELD_FINGER_JOINT_POS,
)
from uwlab_assets.robots.ur10e_delto.ur10e_delto import DELTO_HAND

CUBE_MM = 34.0
# A pad pair that stops exactly at the cube's width only touches it; to actually load the cube
# the commanded aperture has to be INSIDE it. 6 mm of squeeze on a 34 mm part is the same
# proportion a parallel jaw is normally commanded with.
PINCH_LO, PINCH_HI = 22.0, 30.0
# A pre-grasp has to clear the cube with room for placement error. metadata's own replay
# tolerance is +-3 mm, so 46-56 mm leaves 6-11 mm of clearance per side.
PREGRASP_LO, PREGRASP_HI = 46.0, 56.0

PINCH_JOINTS = [f"rj_dg_{f}_{j}" for f in (1, 2) for j in range(1, 5)]


@configclass
class HandOnlySceneCfg(InteractiveSceneCfg):
    robot = DELTO_HAND.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _t(x):
    return getattr(x, "torch", x)


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device or "cuda:0"))
    scene = InteractiveScene(HandOnlySceneCfg(num_envs=args.envs, env_spacing=0.6))
    sim.reset()
    robot: Articulation = scene["robot"]

    names = list(robot.data.joint_names)
    idx = {n: i for i, n in enumerate(names)}
    dev = sim.device
    torch.manual_seed(args.seed)

    limits = _t(robot.data.joint_pos_limits)[0]  # (n_joints, 2), identical across envs
    pj = [idx[n] for n in PINCH_JOINTS]
    lo = limits[pj, 0].to(dev)
    hi = limits[pj, 1].to(dev)
    print("JOINT LIMITS (rad):")
    for n, a, b in zip(PINCH_JOINTS, lo.tolist(), hi.tolist()):
        print(f"  {n}: [{a:+.4f}, {b:+.4f}]")

    # Baseline: what the repo's own postures actually produce.
    base = torch.tensor([[DELTO_HAND_OPEN_JOINT_POS[n] for n in names]], device=dev)
    tucked = dict(DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS)
    tucked.update(DELTO_TUCKED_HELD_FINGER_JOINT_POS)
    for label, posture in [
        ("OPEN", DELTO_HAND_OPEN_JOINT_POS),
        ("SCRIPTED_CLOSE", DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS),
        ("PINCH(tucked 3/4/5)", tucked),
    ]:
        q = torch.tensor([[posture[n] for n in names]], device=dev).repeat(args.envs, 1)
        ap, mid, t1, t2 = measure(sim, scene, robot, q)
        print(
            f"BASELINE {label:22s} aperture={float(ap[0])*1000:7.2f} mm"
            f"  jaw_mount_frame={fmt(mid[0])}"
            f"  tip1={fmt(t1[0])} tip2={fmt(t2[0])}"
        )

    # Random search over the eight pinch joints.
    best = {"aperture": 1e9}
    pinch_hits: list[dict] = []
    pregrasp_hits: list[dict] = []
    apertures = []

    for b in range(args.batches):
        q = base.repeat(args.envs, 1).clone()
        for n in DELTO_TUCKED_HELD_FINGER_JOINT_POS:
            q[:, idx[n]] = DELTO_TUCKED_HELD_FINGER_JOINT_POS[n]
        u = torch.rand((args.envs, len(pj)), device=dev)
        q[:, pj] = lo + u * (hi - lo)

        ap, mid, _, _ = measure(sim, scene, robot, q)
        apertures.append(ap.clone())

        k = int(torch.argmin(ap))
        if float(ap[k]) < best["aperture"]:
            best = {
                "aperture": float(ap[k]) * 1000,
                "jaw_mount_mm": [float(v) * 1000 for v in mid[k]],
                "joints": {n: float(q[k, idx[n]]) for n in PINCH_JOINTS},
            }

        ap_mm = ap * 1000
        for lo_mm, hi_mm, sink in (
            (PINCH_LO, PINCH_HI, pinch_hits),
            (PREGRASP_LO, PREGRASP_HI, pregrasp_hits),
        ):
            sel = ((ap_mm >= lo_mm) & (ap_mm <= hi_mm)).nonzero(as_tuple=False).squeeze(-1)
            for e in sel[:24].tolist():
                sink.append(
                    {
                        "aperture_mm": round(float(ap_mm[e]), 2),
                        "jaw_mount_mm": [round(float(v) * 1000, 2) for v in mid[e]],
                        "joints": {n: round(float(q[e, idx[n]]), 5) for n in PINCH_JOINTS},
                    }
                )
        if b % 20 == 0:
            print(f"  batch {b}/{args.batches} best_aperture={best['aperture']:.2f} mm "
                  f"pinch_hits={len(pinch_hits)} pregrasp_hits={len(pregrasp_hits)}", flush=True)

    allap = torch.cat(apertures) * 1000
    qs = torch.quantile(allap, torch.tensor([0.0, 0.001, 0.01, 0.05, 0.5, 1.0], device=dev))
    print("APERTURE DISTRIBUTION (mm) over "
          f"{allap.numel()} random postures: min={float(qs[0]):.2f} "
          f"p0.1={float(qs[1]):.2f} p1={float(qs[2]):.2f} p5={float(qs[3]):.2f} "
          f"p50={float(qs[4]):.2f} max={float(qs[5]):.2f}")
    print(f"BEST aperture {best['aperture']:.2f} mm  jaw@{best['jaw_mount_mm']}")
    print(f"PINCH-WINDOW postures found: {len(pinch_hits)}")
    print(f"PREGRASP-WINDOW postures found: {len(pregrasp_hits)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(
            {
                "samples": int(allap.numel()),
                "aperture_mm_quantiles": {
                    "min": float(qs[0]), "p0.1": float(qs[1]), "p1": float(qs[2]),
                    "p5": float(qs[3]), "p50": float(qs[4]), "max": float(qs[5]),
                },
                "best": best,
                "pinch_window": [PINCH_LO, PINCH_HI],
                "pregrasp_window": [PREGRASP_LO, PREGRASP_HI],
                "pinch_hits": pinch_hits[:400],
                "pregrasp_hits": pregrasp_hits[:400],
            },
            fh,
            indent=1,
        )
    print(f"WROTE {args.out}")


def fmt(v) -> str:
    return "(" + ",".join(f"{float(x) * 1000:7.1f}" for x in v) + ")"


def measure(sim, scene, robot, q):
    """Write a posture, settle kinematics, and read the jaw in the MOUNT frame.

    The mount frame is what matters: ``grasp_center_offset`` is expressed there, and the whole
    reason the first placement attempt put the cube in mid-air is that the mount frame is
    rotated roughly 180 degrees about x relative to world, so a world-space application of that
    offset lands nowhere near the fingers.
    """
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(1.0 / 120.0)

    ids, _ = robot.find_bodies(["rl_dg_1_tip", "rl_dg_2_tip", "rl_dg_mount"], preserve_order=True)
    bp = _t(robot.data.body_pos_w)
    bq = _t(robot.data.body_quat_w)
    t1, t2 = bp[:, ids[0]], bp[:, ids[1]]
    mount_p, mount_q = bp[:, ids[2]], bq[:, ids[2]]

    aperture = (t1 - t2).norm(dim=1)
    mid_w = 0.5 * (t1 + t2)
    inv_q = math_utils.quat_conjugate(mount_q)
    to_mount = lambda p: math_utils.quat_apply(inv_q, p - mount_p)  # noqa: E731
    return aperture, to_mount(mid_w), to_mount(t1), to_mount(t2)


main()
simulation_app.close()
