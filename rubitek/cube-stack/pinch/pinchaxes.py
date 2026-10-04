"""Derive the pinch's approach and closing axes, which metadata still records for the fist.

WHY. With the pinch posture and its grasp centre wired in, the grasp sampler's placement gate
rejected all 61440 candidates, and its own diagnostic named the reason: ``rl_dg_2_tip`` sitting
**14.44 mm inside the cube**. The pads were not straddling the part, they were driving into a
face. That is not a grasp-centre error -- the centre is right and the jaw does not drift while
closing -- it is an ORIENTATION error.

``DeltoHand/metadata.yaml`` carries three direction vectors, all derived at
``DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS``, i.e. for the fist:

    gripper_approach_direction: [0.2582, 0.4717, 0.8431]
    grasp_align_axis:           [-0.7422, 0.6555, -0.1395]
    orientation_sample_axis:    [-0.7422, 0.6555, -0.1395]

``grasp_align_axis`` is the CLOSING axis: the sampler lines it up with the object's local width
so the pads approach opposite faces. For a posture whose pads meet somewhere else entirely,
that vector points the wrong way, and the hand is rotated so a pad enters the cube.

WHAT THIS COMPUTES, all in the ``rl_dg_mount`` frame at the pinch's closed posture:

    closing axis    normalize(rl_dg_2_tip - rl_dg_1_tip)  -- the direction the pads squeeze along
    approach        normalize(grasp centre - rl_dg_palm)  -- how the hand arrives at the part

Both are read off the same forward kinematics that produced the posture, so they cannot drift
from it the way a hand-copied constant can.
"""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pose", type=str,
                    default="/home/mattephi/Work/Rubitek/cube-stack/pinch/out/pinchpose.json")
parser.add_argument("--out", type=str,
                    default="/home/mattephi/Work/Rubitek/cube-stack/pinch/out/pinchaxes.json")
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

from uwlab_assets.robots.ur10e_delto.ur10e_delto import DELTO_HAND  # noqa: E402


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = DELTO_HAND.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _t(x):
    return getattr(x, "torch", x)


def axes_for(robot, scene, sim, joints: dict) -> dict:
    names = list(robot.data.joint_names)
    q = torch.tensor([[joints[n] for n in names]], device=sim.device)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(1.0 / 120.0)

    ids, _ = robot.find_bodies(
        ["rl_dg_1_tip", "rl_dg_2_tip", "rl_dg_mount", "rl_dg_palm"], preserve_order=True)
    bp = _t(robot.data.body_pos_w)
    bq = _t(robot.data.body_quat_w)
    t1, t2 = bp[:, ids[0]], bp[:, ids[1]]
    mp, mq = bp[:, ids[2]], bq[:, ids[2]]
    palm = bp[:, ids[3]]
    inv = math_utils.quat_conjugate(mq)

    to_mount = lambda p: math_utils.quat_apply(inv, p - mp)[0]  # noqa: E731
    centre = to_mount(0.5 * (t1 + t2))
    closing = to_mount(t2) - to_mount(t1)
    closing = closing / closing.norm()
    approach = centre - to_mount(palm)
    approach = approach / approach.norm()
    return {
        "grasp_center_offset": [round(float(v), 6) for v in centre],
        "grasp_align_axis": [round(float(v), 6) for v in closing],
        "gripper_approach_direction": [round(float(v), 6) for v in approach],
        "pad_separation_mm": round(float((t1 - t2).norm()) * 1000, 3),
    }


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device or "cuda:0"))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=1.0))
    sim.reset()
    robot = scene["robot"]

    payload = json.load(open(args.pose))
    out = {"closed": axes_for(robot, scene, sim, payload["close"]["joints"]),
           "open": axes_for(robot, scene, sim, payload["open"]["joints"])}

    from uwlab_assets.robots.ur10e_delto.actions import DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS

    out["fist_reference"] = axes_for(robot, scene, sim, dict(DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS))
    # Stamp the source. applypose.py refuses to write constants unless the axes it is given were
    # derived from the very posture it is writing -- comparing pad separations cannot do that
    # job, because this rig has no ground plane and no cube and the fingers settle about a
    # millimetre differently under gravity than they do on the bench.
    out["source_pose"] = os.path.abspath(args.pose)
    out["metadata_today"] = {
        "grasp_center_offset": [0.0637, 0.0465, 0.1500],
        "grasp_align_axis": [-0.7422, 0.6555, -0.1395],
        "gripper_approach_direction": [0.2582, 0.4717, 0.8431],
    }

    for k, v in out.items():
        print(f"{k}:")
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"    {v}")
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"WROTE {args.out}")


main()
simulation_app.close()
