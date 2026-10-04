"""Derive a PINCH posture pair for the DELTO hand: a jaw that actually closes on a 34 mm cube.

THE PROBLEM THIS SOLVES, measured not assumed (see out/jawcal.json):

    posture                       |rl_dg_1_tip - rl_dg_2_tip|
    OPEN                                 96.70 mm
    DELTO_HAND_SCRIPTED_CLOSE            51.03 mm     <-- the repo's "closed" hand
    PINCH (3/4/5 tucked)                 51.03 mm     <-- identical; tucking 3/4/5 changes nothing

A 34 mm cube centred at the declared grasp centre sits in a 51 mm gap. Neither pad touches it.
That is why grasp yield is zero, and it is upstream of every friction, stiffness, effort-limit
and control-mode question asked so far: there is nothing to apply force THROUGH.

A random sweep of 122 880 postures over the eight joints of fingers 1 and 2 found the minimum
achievable aperture is 1.35 mm and that 2 502 samples land in a 22-30 mm pinch window, so the
hand is mechanically capable of opposing these two fingertips. It simply is not commanded to.

WHAT THIS SCRIPT PRODUCES
-------------------------
A matched pair of postures, by cross-entropy optimisation over the same eight joints:

  q_close   aperture ~= CLOSE_TARGET_MM, i.e. INSIDE the cube so the pads load it
  q_open    aperture ~= OPEN_TARGET_MM, clear of the cube, jaw centre in the same place

with two regularisers that matter as much as the aperture itself:

  * stay near DELTO_HAND_SCRIPTED_CLOSE. Every other calibrated quantity in this repo --
    grasp_center_offset, the 24.67-41.48 mm grasp window, gripper_approach_direction -- was
    derived at that posture. The minimum-aperture posture in the sweep is 8.85 mm and nowhere
    near it; adopting that would invalidate the rest of the calibration for no benefit.
  * keep the two jaw centres together. If the jaw centre translates while the hand closes, the
    pads sweep sideways across the cube and knock it away instead of squeezing it -- which is
    the ejection mode observed earlier, and it would survive any amount of speed limiting.

Kinematics only; no contact, no dynamics. The output is a posture pair to be tested against a
real cube in pinchlab.py, not a result on its own.
"""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--envs", type=int, default=1024)
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--elite", type=float, default=0.06)
parser.add_argument("--close-mm", type=float, default=26.0)
parser.add_argument("--open-mm", type=float, default=52.0)
parser.add_argument("--out", type=str, default="/home/mattephi/Work/Rubitek/cube-stack/pinch/out/jawopt.json")
parser.add_argument("--open-mode", type=str, default="distal", choices=["distal", "free"],
                    help="distal: open by extending the flexion joints only (a real pinch);"
                         " free: let the optimiser open the jaw any way it likes")
parser.add_argument("--limit-margin", type=float, default=0.05,
                    help="rad of clearance to keep from every joint limit")
parser.add_argument("--tuck-clear-mm", type=float, default=45.0,
                    help="how far tips 3/4/5 must sit from the jaw centre")
parser.add_argument("--tuck-iters", type=int, default=50)
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

PINCH_JOINTS = [f"rj_dg_{f}_{j}" for f in (1, 2) for j in range(1, 5)]
TUCK_JOINTS = [f"rj_dg_{f}_{j}" for f in (3, 4, 5) for j in range(1, 5)]

# Cost weights. Aperture is in metres throughout, so a 1 mm aperture error costs 1.0 at
# W_APERTURE = 1e3 -- the other terms are scaled to be worth roughly a millimetre each at the
# magnitudes they actually reach, which keeps the optimiser from trading the jaw away for
# posture similarity.
W_APERTURE = 1.0e3
W_JAW = 4.0e2  # jaw centre drift from the reference / from q_close
W_POSTURE = 3.0e1  # rad of deviation from SCRIPTED_CLOSE
# A joint parked exactly on its stop is not a usable posture: the PD has no authority left in
# one direction, and any placement error pushes it into the limit rather than into the cube.
# The first solve put rj_dg_2_4 at 1.570796 rad against a 1.5708 limit, which is why this exists.
W_LIMIT = 6.0e2
# Clearance of the three non-pinching fingertips from the jaw. This is not a tidiness
# constraint. DELTO_TUCKED_HELD_FINGER_JOINT_POS curls fingers 3/4/5 directly UNDER the jaw,
# where they become a shelf: measured on the first working posture, the thumb carried 2.22 N,
# the index tip carried 0.00 N, and tips 3/4/5 carried 9.97 N. The cube was resting on them.
# That scores as "held" on any distance test and is not a pinch.
W_CLEAR = 8.0e2


@configclass
class HandOnlySceneCfg(InteractiveSceneCfg):
    robot = DELTO_HAND.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _t(x):
    return getattr(x, "torch", x)


class Jaw:
    """Batched forward kinematics of the finger-1 / finger-2 jaw, using the sim as the oracle."""

    def __init__(self, sim, scene, robot):
        self.sim, self.scene, self.robot = sim, scene, robot
        self.names = list(robot.data.joint_names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.dev = sim.device
        self.pj = [self.idx[n] for n in PINCH_JOINTS]
        limits = _t(robot.data.joint_pos_limits)[0]
        self.lo = limits[self.pj, 0].to(self.dev)
        self.hi = limits[self.pj, 1].to(self.dev)
        ids, _ = robot.find_bodies(
            ["rl_dg_1_tip", "rl_dg_2_tip", "rl_dg_mount",
             "rl_dg_3_tip", "rl_dg_4_tip", "rl_dg_5_tip"],
            preserve_order=True,
        )
        self.ids = ids
        self.tj = [self.idx[n] for n in TUCK_JOINTS]
        self.tlo = limits[self.tj, 0].to(self.dev)
        self.thi = limits[self.tj, 1].to(self.dev)

        base = dict(DELTO_HAND_OPEN_JOINT_POS)
        base.update(DELTO_TUCKED_HELD_FINGER_JOINT_POS)
        self.base = torch.tensor([[base[n] for n in self.names]], device=self.dev)
        self.seed = torch.tensor(
            [DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS[n] for n in PINCH_JOINTS], device=self.dev
        )

    def set_tuck(self, angles: torch.Tensor) -> None:
        """Adopt a tuck posture for fingers 3/4/5 as the base for every later evaluation."""
        self.base = self.base.clone()
        self.base[0, self.tj] = angles

    def eval_tuck(self, x: torch.Tensor):
        """x: (N, 12) tuck angles -> min distance from tips 3/4/5 to the jaw, and to tips 1/2."""
        n = x.shape[0]
        q = self.base.repeat(n, 1).clone()
        q[:, self.pj] = self.seed  # score the tuck against the reference jaw, not a moving one
        q[:, self.tj] = torch.clamp(x, self.tlo, self.thi)
        bp = self._fk(q)
        t1, t2 = bp[:, self.ids[0]], bp[:, self.ids[1]]
        jaw = 0.5 * (t1 + t2)
        others = torch.stack([bp[:, self.ids[i]] for i in (3, 4, 5)], dim=1)  # (N, 3, 3)
        d_jaw = (others - jaw.unsqueeze(1)).norm(dim=2).min(dim=1).values
        d_t1 = (others - t1.unsqueeze(1)).norm(dim=2).min(dim=1).values
        d_t2 = (others - t2.unsqueeze(1)).norm(dim=2).min(dim=1).values
        return d_jaw, torch.minimum(d_t1, d_t2)

    def _fk(self, q: torch.Tensor):
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.robot.set_joint_position_target(q)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(1.0 / 120.0)
        return _t(self.robot.data.body_pos_w)

    def eval(self, x: torch.Tensor):
        """x: (N, 8) pinch-joint angles -> (aperture (N,), jaw centre in mount frame (N,3))."""
        n = x.shape[0]
        q = self.base.repeat(n, 1).clone()
        q[:, self.pj] = torch.clamp(x, self.lo, self.hi)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.robot.set_joint_position_target(q)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(1.0 / 120.0)

        bp = _t(self.robot.data.body_pos_w)
        bq = _t(self.robot.data.body_quat_w)
        t1, t2 = bp[:, self.ids[0]], bp[:, self.ids[1]]
        mp, mq = bp[:, self.ids[2]], bq[:, self.ids[2]]
        inv = math_utils.quat_conjugate(mq)
        mid = math_utils.quat_apply(inv, 0.5 * (t1 + t2) - mp)
        return (t1 - t2).norm(dim=1), mid


def limit_penalty(jaw: Jaw, x: torch.Tensor) -> torch.Tensor:
    """How far each joint intrudes into the margin reserved at its limits."""
    m = args.limit_margin
    lo_slack = torch.clamp((jaw.lo + m) - x, min=0.0)
    hi_slack = torch.clamp(x - (jaw.hi - m), min=0.0)
    return (lo_slack + hi_slack).sum(dim=1)


def cem(jaw: Jaw, cost, x0: torch.Tensor, sigma0: float, iters: int, label: str,
        free_mask: torch.Tensor | None = None, bounds: tuple | None = None,
        report=None):
    """Cross-entropy method. Population is one environment per sample, so it is one sim step."""
    n = args.envs
    lo, hi = bounds if bounds is not None else (jaw.lo, jaw.hi)
    mu = x0.clone()
    sigma = torch.full_like(mu, sigma0)
    n_elite = max(8, int(n * args.elite))
    best = (float("inf"), mu.clone())

    for it in range(iters):
        noise = sigma.unsqueeze(0) * torch.randn((n, mu.numel()), device=jaw.dev)
        if free_mask is not None:
            noise = noise * free_mask.unsqueeze(0)
        x = mu.unsqueeze(0) + noise
        x[0] = mu  # keep the incumbent so the best-so-far can never get worse
        x = torch.clamp(x, lo, hi)
        c, extra = cost(x)
        order = torch.argsort(c)
        elite = x[order[:n_elite]]
        mu = elite.mean(dim=0)
        sigma = elite.std(dim=0).clamp(min=1e-4)
        if float(c[order[0]]) < best[0]:
            best = (float(c[order[0]]), x[order[0]].clone())
        if it % 10 == 0 or it == iters - 1:
            print(f"  [{label}] it {it:3d} cost {float(c[order[0]]):9.3f} "
                  f"{extra(order[0])} sigma {float(sigma.mean()):.4f}", flush=True)
    return best[1]


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device or "cuda:0"))
    scene = InteractiveScene(HandOnlySceneCfg(num_envs=args.envs, env_spacing=0.6))
    sim.reset()
    jaw = Jaw(sim, scene, scene["robot"])
    torch.manual_seed(0)

    # Where the calibrated grasp centre is today, and what aperture the repo's closed posture
    # actually reaches there. Everything below is anchored to this.
    ref_ap, ref_mid = jaw.eval(jaw.seed.unsqueeze(0).repeat(args.envs, 1))
    ref_ap, ref_mid = ref_ap[0], ref_mid[0]
    print(f"REFERENCE SCRIPTED_CLOSE aperture {float(ref_ap) * 1000:.2f} mm  "
          f"jaw {[round(float(v) * 1000, 2) for v in ref_mid]} mm (mount frame)")

    close_t = args.close_mm / 1000.0
    open_t = args.open_mm / 1000.0
    clear_t = args.tuck_clear_mm / 1000.0

    # ---- STAGE 0: get fingers 3, 4 and 5 out of the jaw -------------------------------
    # DELTO_TUCKED_HELD_FINGER_JOINT_POS curls them straight under the jaw, where they carry
    # the cube as a shelf. Measured on the first working pinch: thumb 2.22 N, index 0.00 N,
    # tips 3/4/5 9.97 N. That is a scoop, and it scores as "held" on any distance test.
    tuck_seed = torch.tensor(
        [DELTO_TUCKED_HELD_FINGER_JOINT_POS[n] for n in TUCK_JOINTS], device=jaw.dev
    )
    d0, s0 = jaw.eval_tuck(tuck_seed.unsqueeze(0).repeat(args.envs, 1))
    print(f"REFERENCE tuck: nearest of tips 3/4/5 sits {float(d0[0]) * 1000:.1f} mm from the jaw"
          f" (want >= {args.tuck_clear_mm:.0f} mm), {float(s0[0]) * 1000:.1f} mm from tips 1/2")

    def tuck_cost(x):
        d_jaw, d_tips = jaw.eval_tuck(x)
        c = (
            W_CLEAR * torch.clamp(clear_t - d_jaw, min=0.0)
            # Do not let a finger escape the jaw by burying itself in the thumb or index; the
            # articulation has enabled_self_collisions=True and would jam.
            + W_CLEAR * torch.clamp(0.030 - d_tips, min=0.0)
            + W_LIMIT * ((torch.clamp((jaw.tlo + args.limit_margin) - x, min=0.0)
                          + torch.clamp(x - (jaw.thi - args.limit_margin), min=0.0)).sum(dim=1))
        )
        return c, lambda i: (f"clear {float(d_jaw[i]) * 1000:5.1f} mm"
                             f" selfgap {float(d_tips[i]) * 1000:5.1f} mm")

    print("STAGE 0: tuck fingers 3/4/5 clear of the jaw")
    q_tuck = cem(jaw, tuck_cost, tuck_seed, 0.35, args.tuck_iters, "tuck",
                 bounds=(jaw.tlo, jaw.thi))
    d1, s1 = jaw.eval_tuck(q_tuck.unsqueeze(0).repeat(args.envs, 1))
    print(f"RESULT tuck clearance {float(d1[0]) * 1000:.1f} mm from the jaw,"
          f" {float(s1[0]) * 1000:.1f} mm from tips 1/2")
    jaw.set_tuck(q_tuck)

    # The reference jaw has to be re-read: the tuck changed the base posture the pinch joints
    # are evaluated against, and finger 2's proximal joint shares a mount with finger 3.
    ref_ap, ref_mid = jaw.eval(jaw.seed.unsqueeze(0).repeat(args.envs, 1))
    ref_ap, ref_mid = ref_ap[0], ref_mid[0]

    def close_cost(x):
        aperture, mid = jaw.eval(x)
        c = (
            W_APERTURE * (aperture - close_t).abs()
            + W_JAW * (mid - ref_mid).norm(dim=1)
            + W_POSTURE * (x - jaw.seed).abs().mean(dim=1)
            + W_LIMIT * limit_penalty(jaw, x)
        )
        return c, lambda i: f"aperture {float(aperture[i]) * 1000:6.2f} mm"

    print("STAGE 1: closed posture")
    q_close = cem(jaw, close_cost, jaw.seed, 0.25, args.iters, "close")
    ap_c, mid_c = jaw.eval(q_close.unsqueeze(0).repeat(args.envs, 1))
    ap_c, mid_c = ap_c[0], mid_c[0]

    def open_cost(x):
        aperture, mid = jaw.eval(x)
        c = (
            W_APERTURE * (aperture - open_t).abs()
            + W_JAW * (mid - mid_c).norm(dim=1)
            + W_POSTURE * (x - q_close).abs().mean(dim=1)
            + W_LIMIT * limit_penalty(jaw, x)
        )
        return c, lambda i: f"aperture {float(aperture[i]) * 1000:6.2f} mm"

    # A pinch should OPEN by extending the same joints it closes with, so the pads travel along
    # the line between them. Letting the optimiser open the jaw however it likes produced a
    # posture that opened by abducting finger 2 sideways -- numerically a wider aperture, but a
    # motion that sweeps across the cube's face rather than releasing it.
    free = torch.zeros_like(q_close)
    if args.open_mode == "distal":
        distal_slots = [PINCH_JOINTS.index(f"rj_dg_{f}_{j}") for f in (1, 2) for j in (3, 4)]
        free[distal_slots] = 1.0
        print(f"STAGE 2: pre-grasp by extending {[PINCH_JOINTS[i] for i in distal_slots]} only")
    else:
        free[:] = 1.0
        print("STAGE 2: pre-grasp posture, all eight joints free")
    q_open = cem(jaw, open_cost, q_close, 0.20, args.iters, "open", free_mask=free)
    ap_o, mid_o = jaw.eval(q_open.unsqueeze(0).repeat(args.envs, 1))
    ap_o, mid_o = ap_o[0], mid_o[0]

    drift = float((mid_o - mid_c).norm()) * 1000
    print(f"RESULT close aperture {float(ap_c) * 1000:6.2f} mm  jaw {fmt(mid_c)}")
    print(f"RESULT open  aperture {float(ap_o) * 1000:6.2f} mm  jaw {fmt(mid_o)}")
    print(f"RESULT jaw centre drift open->close: {drift:.2f} mm")
    print(f"RESULT max joint change from SCRIPTED_CLOSE: "
          f"{float((q_close - jaw.seed).abs().max()):.4f} rad")

    payload = {
        "reference": {
            "aperture_mm": round(float(ref_ap) * 1000, 3),
            "jaw_mount_mm": [round(float(v) * 1000, 3) for v in ref_mid],
        },
        "close": {
            "aperture_mm": round(float(ap_c) * 1000, 3),
            "jaw_mount_mm": [round(float(v) * 1000, 3) for v in mid_c],
            "joints": {n: round(float(v), 6) for n, v in zip(PINCH_JOINTS, q_close.tolist())},
        },
        "open": {
            "aperture_mm": round(float(ap_o) * 1000, 3),
            "jaw_mount_mm": [round(float(v) * 1000, 3) for v in mid_o],
            "joints": {n: round(float(v), 6) for n, v in zip(PINCH_JOINTS, q_open.tolist())},
        },
        "jaw_drift_mm": round(drift, 3),
        "tucked": {n: round(float(v), 6) for n, v in zip(TUCK_JOINTS, q_tuck.tolist())},
        "tuck_clearance_mm": {
            "before": round(float(d0[0]) * 1000, 2),
            "after": round(float(d1[0]) * 1000, 2),
            "self_gap_after": round(float(s1[0]) * 1000, 2),
        },
        "targets_mm": {"close": args.close_mm, "open": args.open_mm},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"WROTE {args.out}")


def fmt(v) -> str:
    return "(" + ",".join(f"{float(x) * 1000:7.2f}" for x in v) + ") mm"


main()
simulation_app.close()
