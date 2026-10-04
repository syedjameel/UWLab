"""Derive the PRE-GRASP posture, which nothing has ever optimised.

The closed posture has been optimised repeatedly -- against geometry, then against measured hold.
The open one has been carried forward from a hand-authored seed the whole time, and it is the half
that decides whether the grasp sampler can place the hand at all. The sampler's placement gate
demands that no gripper collider point intersect the object at placement, with the object sitting
at the CLOSED pad midpoint (it has to sit there: put it anywhere else and closing the hand grips
empty air -- delto_cfg.py:668 records that failure, 0 successes in 6144 attempts). A DELTO
fingertip's collider reaches 16.5 mm past its body origin, so against a 34 mm cube each pad origin
must stand at least 17.0 + 16.5 = 33.5 mm off that centre.

Scaling the seed's open direction does not get there. Measured, as lambda takes the pre-grasp from
50 mm to 91 mm, pad 1 travels 30.0 -> 65.1 mm from the closed centre while pad 2 moves only
20.0 -> 27.8 mm and asymptotes short: the seed's opening motion is almost entirely one finger. The
thumb has to be told to retract, which is what this optimises -- directly, on the quantity the gate
measures, with the closed posture held fixed so the grasp itself is untouched.

Pure kinematics: one physics step per candidate batch, no rollout.
"""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher

P = "/home/mattephi/Work/Rubitek/cube-stack/pinch"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pose", default=f"{P}/out/pinchhold_v3.json",
                    help="pose file whose CLOSED posture defines the grasp centre (and is kept)")
parser.add_argument("--out", default=f"{P}/out/openpose.json")
parser.add_argument("--required-mm", type=float, default=33.5,
                    help="object half-width + fingertip collider reach; the standoff the placement "
                         "gate demands of each pad body origin")
parser.add_argument("--margin-mm", type=float, default=3.0,
                    help="head-room beyond --required-mm, so a candidate that only just clears in "
                         "this static pose still clears once the sampler tilts it")
parser.add_argument("--envs", type=int, default=256)
parser.add_argument("--gens", type=int, default=25)
parser.add_argument("--elite", type=float, default=0.15)
parser.add_argument("--sigma", type=float, default=0.35)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import uwlab_is6_bridge  # noqa: E402

uwlab_is6_bridge.apply()

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from uwlab_assets.robots.ur10e_delto.ur10e_delto import DELTO_HAND  # noqa: E402

DT = 1.0 / 120.0
PADS = ("rl_dg_1_tip", "rl_dg_2_tip")


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = DELTO_HAND.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _t(x):
    return getattr(x, "torch", x)


def main() -> None:
    pose = json.load(open(args.pose))
    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device or "cuda:0"))
    scene = InteractiveScene(SceneCfg(num_envs=args.envs, env_spacing=0.6))
    sim.reset()
    robot = scene["robot"]
    names = list(robot.data.joint_names)
    lower = _t(robot.data.joint_pos_limits)[0, :, 0].clone()
    upper = _t(robot.data.joint_pos_limits)[0, :, 1].clone()

    q_close = torch.tensor([pose["close"]["joints"][n] for n in names], device=sim.device)
    q_open0 = torch.tensor([pose["open"]["joints"][n] for n in names], device=sim.device)
    pad_ids, _ = robot.find_bodies(list(PADS), preserve_order=True)
    mount_id = robot.find_bodies(["rl_dg_mount"], preserve_order=True)[0][0]

    def pads_and_centre(q: torch.Tensor):
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)
        bp = _t(robot.data.body_pos_w)
        bq = _t(robot.data.body_quat_w)
        mp, mq = bp[:, mount_id], bq[:, mount_id]
        inv = math_utils.quat_conjugate(mq)
        to_mount = lambda p: math_utils.quat_apply(inv, p - mp)  # noqa: E731
        return to_mount(bp[:, pad_ids[0]]), to_mount(bp[:, pad_ids[1]])

    # The grasp centre is a property of the CLOSED posture and is fixed for the whole search.
    p1c, p2c = pads_and_centre(q_close.unsqueeze(0).repeat(args.envs, 1))
    centre = (0.5 * (p1c + p2c))[0].clone()
    print(f"closed pad midpoint (mount frame): {[round(float(v), 5) for v in centre]}", flush=True)

    target = (args.required_mm + args.margin_mm) / 1000.0
    mu = q_open0.clone()
    sigma = torch.full_like(mu, args.sigma)
    n_elite = max(2, int(args.envs * args.elite))
    best = (-1e9, q_open0.clone())

    for gen in range(args.gens):
        cand = torch.normal(mu.unsqueeze(0).repeat(args.envs, 1), sigma.unsqueeze(0))
        cand[0] = mu  # keep the incumbent
        cand = torch.max(torch.min(cand, upper.unsqueeze(0)), lower.unsqueeze(0))
        p1, p2 = pads_and_centre(cand)
        d1 = (p1 - centre).norm(dim=1)
        d2 = (p2 - centre).norm(dim=1)
        worst = torch.minimum(d1, d2)
        # Reward the binding pad up to the target and stop paying past it, so the search spends its
        # remaining freedom on staying near the seed rather than flinging the hand open.
        score = torch.clamp(worst, max=target) * 1000.0 - 0.02 * (cand - q_open0).abs().sum(dim=1)
        order = torch.argsort(score, descending=True)
        elite = cand[order[:n_elite]]
        mu = elite.mean(dim=0)
        sigma = elite.std(dim=0).clamp(min=0.01)
        k = int(order[0])
        if float(score[k]) > best[0]:
            best = (float(score[k]), cand[k].clone())
        print(f"  gen {gen:2d}  worst pad {float(worst[k]) * 1000:5.1f} mm  "
              f"(d1 {float(d1[k]) * 1000:5.1f}, d2 {float(d2[k]) * 1000:5.1f})  "
              f"pop worst {float(worst.max()) * 1000:5.1f}  sigma {float(sigma.mean()):.3f}", flush=True)

    q_best = best[1]
    p1, p2 = pads_and_centre(q_best.unsqueeze(0).repeat(args.envs, 1))
    d1 = float((p1[0] - centre).norm()) * 1000.0
    d2 = float((p2[0] - centre).norm()) * 1000.0
    aperture = float((p1[0] - p2[0]).norm()) * 1000.0
    print(f"WINNER pads {d1:.1f} / {d2:.1f} mm from the closed centre (need {args.required_mm:.1f}), "
          f"aperture {aperture:.2f} mm, margin {min(d1, d2) - args.required_mm:+.1f} mm")

    out = json.loads(json.dumps(pose))
    out["open"]["joints"] = {n: round(float(v), 6) for n, v in zip(names, q_best.tolist())}
    out["open"]["aperture_mm"] = round(aperture, 2)
    out["open_derived"] = {
        "source_pose": args.pose,
        "pad_distances_mm": [round(d1, 2), round(d2, 2)],
        "required_mm": args.required_mm,
        "margin_mm": round(min(d1, d2) - args.required_mm, 2),
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"WROTE {args.out}")
    simulation_app.close()


main()
