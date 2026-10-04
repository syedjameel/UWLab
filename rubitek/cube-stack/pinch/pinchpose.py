"""Derive a REAL two-finger pinch for the DELTO hand: two pads on the cube, nothing else near it.

WHAT THE PREVIOUS ATTEMPT GOT WRONG, and it is worth stating because the numbers looked fine.

jawopt.py optimised |rl_dg_1_tip - rl_dg_2_tip| toward the cube's width while regularising the
posture to stay near DELTO_HAND_SCRIPTED_CLOSE. It converged: 30.5 mm aperture, jaw centre
within 4 mm of the metadata grasp centre, 1.3 mm of drift while closing. In the bench it scored
90.8% "held".

It was not a pinch. SCRIPTED_CLOSE is a FIST, so the midpoint of the two fingertips at that
posture lies INSIDE the curled finger cluster. A 34 mm cube placed there is engulfed: the
phalanges reach it long before the pads do, and it ends up resting on the fingers. Measured at
the end of the hold window -- thumb tip 0.11 N, index tip 0.37 N, tips 3/4/5 6.76 N, with the
cube 32-36 mm from three different fingertips at once. The distance test called that a grasp.

So the constraint that was missing is not about fingers 3/4/5 specifically, and not about
aperture. It is that a pinch requires the cube to be reachable ONLY by the two pads:

    for every hand body that is not rl_dg_1_tip or rl_dg_2_tip,
        distance(body, jaw centre) > cube half-diagonal + margin

A 34 mm cube has a 29.4 mm half-diagonal, so nothing else may come within ~35 mm of the grasp
point. That single constraint is what forces the fingers to EXTEND rather than curl, which is
what a pinch actually looks like and what the fist posture structurally cannot do.

The posture-similarity regulariser is therefore dropped, not weakened: staying near the fist is
exactly what prevents a pinch. The consequence is that grasp_center_offset moves, and this
script reports the new value so the sampler can be re-anchored to it rather than to a centre
derived for a closure model that no longer applies.

OUTPUT
  q_close   the pinch: aperture ~ CLOSE_TARGET, everything else clear of the cube
  q_open    the pre-grasp: the SAME fingers extended further so the pads clear the cube, with
            the jaw centre held in place so closing squeezes rather than sweeps
"""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--envs", type=int, default=2048)
parser.add_argument("--iters", type=int, default=90)
parser.add_argument("--restarts", type=int, default=4)
parser.add_argument("--elite", type=float, default=0.05)
parser.add_argument("--close-mm", type=float, default=30.0)
parser.add_argument("--open-mm", type=float, default=50.0)
parser.add_argument("--clear-mm", type=float, default=36.0,
                    help="how far every non-pad body must stay from the grasp point")
parser.add_argument("--limit-margin", type=float, default=0.06)
parser.add_argument("--out", type=str,
                    default="/home/mattephi/Work/Rubitek/cube-stack/pinch/out/pinchpose.json")
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
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from uwlab_assets.robots.ur10e_delto.actions import (  # noqa: E402
    DELTO_HAND_OPEN_JOINT_POS,
    DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS,
)
from uwlab_assets.robots.ur10e_delto.ur10e_delto import DELTO_HAND

PAD_BODIES = ("rl_dg_1_tip", "rl_dg_2_tip")

# The hand as a set of LINE SEGMENTS, not a set of body origins.
#
# The first version of this script required every non-pad body ORIGIN to stay 36 mm from the
# grasp point, and reported a comfortable 50 mm. The sampler's placement gate -- which samples
# 26 points on each body's actual collision mesh -- then found rl_dg_2_3 sitting 8.4 mm INSIDE
# the cube and rejected all 61440 candidates. A phalanx is a long link: its origin can be 50 mm
# away while its geometry reaches 40 mm back toward the part.
#
# Approximating each link as the segment between consecutive joint frames captures that, and is
# the cheapest thing that does. Chains are palm -> _1 -> _2 -> _3 -> _4 -> tip per finger.
FINGER_CHAINS = [
    ["rl_dg_palm"] + [f"rl_dg_{f}_{j}" for j in range(1, 5)] + [f"rl_dg_{f}_tip"]
    for f in range(1, 6)
]

# What each segment must clear. The cube is 34 mm: 17 mm inscribed radius, 29.4 mm circumscribed.
# The distal links of the two pinching fingers HAVE to lie alongside a cube face, so demanding
# 36 mm of them is geometrically impossible -- they get the inscribed radius plus a margin. The
# pad segments themselves are the grasp and are exempt.
# Two tables, because the two postures are judged against different things. The sampler's
# placement gate only ever tests the PRE-GRASP posture -- that is what _open_gripper writes
# before the gate runs -- so at the open posture EVERY segment must clear the part, pads
# included. At the closed posture the pads are supposed to be inside the cube's width; that is
# what squeezing is, and constraining them there would forbid the grasp.
#
# Exempting the pad segment at BOTH postures is what left rl_dg_2_4's geometry unconstrained and
# kept the gate rejecting: 12.68 mm of penetration, then 7.13 mm once the phalanges were fixed.
DISTAL_CLEAR_CLOSE_M = 0.019
DISTAL_CLEAR_OPEN_M = 0.026
PAD_CLEAR_OPEN_M = 0.020
ALL_JOINTS = [f"rj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]

W_APERTURE = 2.0e3  # per metre of aperture error
W_CLEAR = 3.0e3  # per metre any non-pad body intrudes inside the clearance sphere
W_JAW = 1.0e3  # per metre the jaw centre moves between open and closed
W_LIMIT = 5.0e2  # per radian a joint intrudes into its limit margin


@configclass
class HandOnlySceneCfg(InteractiveSceneCfg):
    robot = DELTO_HAND.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _t(x):
    return getattr(x, "torch", x)


class Hand:
    def __init__(self, sim, scene):
        self.sim, self.scene = sim, scene
        self.robot = scene["robot"]
        self.dev = sim.device
        self.names = list(self.robot.data.joint_names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.aj = [self.idx[n] for n in ALL_JOINTS]
        limits = _t(self.robot.data.joint_pos_limits)[0]
        self.lo = limits[self.aj, 0].to(self.dev)
        self.hi = limits[self.aj, 1].to(self.dev)

        body_names = list(self.robot.data.body_names)
        self.pad_ids = [body_names.index(b) for b in PAD_BODIES]
        self.mount_id = body_names.index("rl_dg_mount")
        # Everything that must stay away from the cube: every body except the two pads. The
        # mount is included -- a "pinch" that buries the cube against the palm is the failure
        # this whole script exists to rule out.
        self.other_ids = [i for i in range(len(body_names)) if i not in self.pad_ids]
        self.body_names = body_names
        self._clear_t = args.clear_mm / 1000.0

        self.segments: dict[str, list[tuple[int, int, float | None]]] = {"close": [], "open": []}
        self.segment_names: dict[str, list[str]] = {"close": [], "open": []}
        for finger, chain in enumerate(FINGER_CHAINS, start=1):
            for a, b in zip(chain[:-1], chain[1:]):
                if a not in body_names or b not in body_names:
                    continue
                ia, ib = body_names.index(a), body_names.index(b)
                pinching = finger in (1, 2)
                is_pad = pinching and b.endswith("_tip")
                is_distal = pinching and a.endswith("_3")

                if not is_pad:
                    self.segments["close"].append(
                        (ia, ib, DISTAL_CLEAR_CLOSE_M if is_distal else None))
                    self.segment_names["close"].append(f"{a}->{b}")

                need_open = None
                if is_pad:
                    need_open = PAD_CLEAR_OPEN_M
                elif is_distal:
                    need_open = DISTAL_CLEAR_OPEN_M
                self.segments["open"].append((ia, ib, need_open))
                self.segment_names["open"].append(f"{a}->{b}")

    def fk(self, x: torch.Tensor, stage: str = "close"):
        """x: (N, 20) joint angles -> aperture, jaw centre (mount frame), min segment gap."""
        q = torch.zeros((x.shape[0], len(self.names)), device=self.dev)
        q[:, self.aj] = torch.clamp(x, self.lo, self.hi)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.robot.set_joint_position_target(q)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(1.0 / 120.0)

        bp = _t(self.robot.data.body_pos_w)
        bq = _t(self.robot.data.body_quat_w)
        p1, p2 = bp[:, self.pad_ids[0]], bp[:, self.pad_ids[1]]
        jaw = 0.5 * (p1 + p2)
        gaps = self._segment_gaps(bp, jaw, stage)
        mp, mq = bp[:, self.mount_id], bq[:, self.mount_id]
        jaw_local = math_utils.quat_apply(math_utils.quat_conjugate(mq), jaw - mp)
        return (p1 - p2).norm(dim=1), jaw_local, gaps.min(dim=1).values, gaps

    def _segment_gaps(self, bp: torch.Tensor, jaw: torch.Tensor, stage: str) -> torch.Tensor:
        """Distance from the grasp point to each link segment, as (N, num_segments).

        Each segment carries its own requirement, so the returned value is the SLACK against
        that requirement expressed as a distance: a segment needing only 19 mm reports its
        distance plus the difference, which keeps a single `min` meaningful downstream.
        """
        out = []
        for (a, b, need) in self.segments[stage]:
            pa, pb = bp[:, a], bp[:, b]
            ab = pb - pa
            denom = (ab * ab).sum(dim=1).clamp(min=1e-12)
            t = (((jaw - pa) * ab).sum(dim=1) / denom).clamp(0.0, 1.0)
            closest = pa + t.unsqueeze(1) * ab
            d = (jaw - closest).norm(dim=1)
            if need is not None:
                d = d + (self._clear_t - need)
            out.append(d)
        return torch.stack(out, dim=1)

    def limit_penalty(self, x):
        m = args.limit_margin
        return (torch.clamp((self.lo + m) - x, min=0.0)
                + torch.clamp(x - (self.hi - m), min=0.0)).sum(dim=1)


def cem(hand: Hand, cost, mu: torch.Tensor, sigma0: float, iters: int, label: str):
    n = args.envs
    sigma = torch.full_like(mu, sigma0)
    n_elite = max(16, int(n * args.elite))
    best = (float("inf"), mu.clone())
    for it in range(iters):
        x = torch.clamp(mu + sigma * torch.randn((n, mu.numel()), device=hand.dev), hand.lo, hand.hi)
        x[0] = mu
        c, note = cost(x)
        order = torch.argsort(c)
        elite = x[order[:n_elite]]
        mu = elite.mean(dim=0)
        sigma = elite.std(dim=0).clamp(min=5e-4)
        if float(c[order[0]]) < best[0]:
            best = (float(c[order[0]]), x[order[0]].clone())
        if it % 15 == 0 or it == iters - 1:
            print(f"  [{label}] it {it:3d} cost {float(c[order[0]]):9.2f} {note(order[0])}", flush=True)
    return best


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device or "cuda:0"))
    scene = InteractiveScene(HandOnlySceneCfg(num_envs=args.envs, env_spacing=0.6))
    sim.reset()
    hand = Hand(sim, scene)
    torch.manual_seed(0)

    open_v = torch.tensor([DELTO_HAND_OPEN_JOINT_POS[n] for n in ALL_JOINTS], device=hand.dev)
    fist_v = torch.tensor([DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS[n] for n in ALL_JOINTS], device=hand.dev)

    for label, v in [("OPEN", open_v), ("SCRIPTED_CLOSE", fist_v)]:
        ap, jaw, gap, _ = hand.fk(v.unsqueeze(0).repeat(args.envs, 1), "close")
        print(f"REFERENCE {label:15s} aperture {float(ap[0]) * 1000:6.2f} mm   "
              f"nearest non-pad body to the grasp point {float(gap[0]) * 1000:6.2f} mm   "
              f"jaw {[round(float(c) * 1000, 1) for c in jaw[0]]} mm")

    close_t, clear_t = args.close_mm / 1000.0, args.clear_mm / 1000.0

    def close_cost(x):
        ap, jaw, gap, _ = hand.fk(x, "close")
        c = (W_APERTURE * (ap - close_t).abs()
             + W_CLEAR * torch.clamp(clear_t - gap, min=0.0)
             + W_LIMIT * hand.limit_penalty(x))
        return c, lambda i: (f"aperture {float(ap[i]) * 1000:6.2f} mm  "
                             f"clear {float(gap[i]) * 1000:6.2f} mm")

    # Multi-start: the clearance constraint carves the reachable set into disconnected basins
    # (a pinch with the thumb above finger 2 and one with it below are both valid and are not
    # continuously connected), so a single descent from one seed is not enough.
    print("STAGE 1: the pinch")
    best = (float("inf"), None)
    for r in range(args.restarts):
        seed = open_v if r == 0 else torch.clamp(
            open_v + 0.5 * torch.randn_like(open_v), hand.lo, hand.hi)
        cand = cem(hand, close_cost, seed, 0.45, args.iters, f"pinch/{r}")
        print(f"  restart {r}: cost {cand[0]:.2f}")
        if cand[0] < best[0]:
            best = cand
    q_close = best[1]
    ap_c, jaw_c, gap_c, gaps_c = hand.fk(q_close.unsqueeze(0).repeat(args.envs, 1), "close")
    ap_c, jaw_c, gap_c = ap_c[0], jaw_c[0], gap_c[0]

    open_t = args.open_mm / 1000.0

    def open_cost(x):
        ap, jaw, gap, _ = hand.fk(x, "open")
        c = (W_APERTURE * (ap - open_t).abs()
             + W_CLEAR * torch.clamp(clear_t - gap, min=0.0)
             + W_JAW * (jaw - jaw_c).norm(dim=1)
             + W_LIMIT * hand.limit_penalty(x))
        return c, lambda i: (f"aperture {float(ap[i]) * 1000:6.2f} mm  "
                             f"clear {float(gap[i]) * 1000:6.2f} mm  "
                             f"jawdrift {float((jaw[i] - jaw_c).norm()) * 1000:5.2f} mm")

    print("STAGE 2: the pre-grasp, same grasp point, pads clear of the cube")
    q_open = cem(hand, open_cost, q_close, 0.25, args.iters, "open")[1]
    ap_o, jaw_o, gaps_o_min, gaps_o = hand.fk(q_open.unsqueeze(0).repeat(args.envs, 1), "open")
    gap_o = gaps_o_min
    ap_o, jaw_o, gap_o = ap_o[0], jaw_o[0], gap_o[0]

    drift = float((jaw_o - jaw_c).norm()) * 1000
    print(f"RESULT close aperture {float(ap_c) * 1000:6.2f} mm  clearance {float(gap_c) * 1000:6.2f} mm"
          f"  jaw {[round(float(c) * 1000, 2) for c in jaw_c]} mm")
    print(f"RESULT open  aperture {float(ap_o) * 1000:6.2f} mm  clearance {float(gap_o) * 1000:6.2f} mm")
    print(f"RESULT jaw centre drift open->close {drift:.2f} mm")

    for stage, gaps in (("pinch", gaps_c), ("pre-grasp", gaps_o)):
        order = torch.argsort(gaps[0])
        print(f"CLOSEST LINK SEGMENTS at the {stage} (slack-adjusted against each requirement):")
        for k in order[:5].tolist():
            print(f"    {hand.segment_names['close' if stage == 'pinch' else 'open'][k]:28s} "
                  f"{float(gaps[0, k]) * 1000:6.2f} mm")

    payload = {
        "targets_mm": {"close": args.close_mm, "open": args.open_mm, "clear": args.clear_mm},
        "reference_close_aperture_mm": None,
        "close": {"aperture_mm": round(float(ap_c) * 1000, 3),
                  "clearance_mm": round(float(gap_c) * 1000, 3),
                  "jaw_mount_mm": [round(float(c) * 1000, 3) for c in jaw_c],
                  "joints": {n: round(float(v), 6) for n, v in zip(ALL_JOINTS, q_close.tolist())}},
        "open": {"aperture_mm": round(float(ap_o) * 1000, 3),
                 "clearance_mm": round(float(gap_o) * 1000, 3),
                 "jaw_mount_mm": [round(float(c) * 1000, 3) for c in jaw_o],
                 "joints": {n: round(float(v), 6) for n, v in zip(ALL_JOINTS, q_open.tolist())}},
        "jaw_drift_mm": round(drift, 3),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"WROTE {args.out}")


main()
simulation_app.close()
