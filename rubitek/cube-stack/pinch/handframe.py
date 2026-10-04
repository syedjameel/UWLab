"""Film what a reset-state bank looks like AT THE HAND, and what the pinch would look like instead.

WHY THIS EXISTS RATHER THAN FILMING THE BANK IN ITS OWN SCENE. Replaying a bank in the full
UR10e + table scene works and its numbers are trustworthy, but that scene cannot be
photographed on this box: Isaac Lab's Camera sensor cannot be aimed at all in this build
(``set_world_poses_from_view`` never moves the prim, ``CameraCfg.OffsetCfg`` applies the
translation but not the rotation, and writing the USD xform directly is overwritten again by
the sensor), and the replicator path that does aim renders that scene 180 degrees rolled and
resisted every framing attempt.

The hand-only scene renders fine, because its camera happens to sit where the hand is. So the
bank is replayed HERE: ``banklab.py --dump-relative`` records, per state, the twenty hand joint
angles and the cube's pose in the ``rl_dg_mount`` frame; this script reproduces exactly that
relative configuration on a floating hand. The arm is dropped, which loses where in the room
the grasp happened and keeps the only thing a grasp actually is -- the relationship between the
hand and the part.

Each state is shown for --settle-s of REAL physics with gravity on, so a grasp that does not
hold drops the cube on camera. Nothing is frozen.

With --pinch, the same cube poses are shown with the hand commanded to the derived pinch
posture instead of the recorded one -- the before/after for a single bank, in one shot.
"""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rel", type=str, required=True, help="a rel_<Type>.json from banklab")
parser.add_argument("--states", type=int, default=8)
parser.add_argument("--settle-s", type=float, default=1.2)
parser.add_argument("--video", type=str, default="")
parser.add_argument("--stills", type=str, default="")
parser.add_argument("--out", type=str, default="")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--pinch", type=str, default="", help="pinchpose json: re-pose with the pinch")
parser.add_argument("--label", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = bool(args.video or args.stills)
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import uwlab_is6_bridge  # noqa: E402

uwlab_is6_bridge.apply()

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR  # noqa: E402
from uwlab_assets.robots.ur10e_delto.ur10e_delto import DELTO_HAND  # noqa: E402

DT = 1.0 / 120.0
CUBE_USD = f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/InsertiveCube34/insertive_cube34.usd"
G = 9.81


def _tip(i: int) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rl_dg_%d_tip" % i,
        history_length=1, track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )


@configclass
class HandFrameSceneCfg(InteractiveSceneCfg):
    robot = DELTO_HAND.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=DELTO_HAND.spawn.replace(activate_contact_sensors=True),
    )
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CUBE_USD, scale=(0.85,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=0,
                disable_gravity=True, max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
            activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    tip1 = _tip(1)
    tip2 = _tip(2)
    tip3 = _tip(3)
    tip4 = _tip(4)
    tip5 = _tip(5)
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.45)),
        spawn=sim_utils.GroundPlaneCfg())
    dome = AssetBaseCfg(prim_path="/World/DomeLight",
                        spawn=sim_utils.DomeLightCfg(intensity=1400.0, color=(0.93, 0.95, 1.0)))
    key = AssetBaseCfg(prim_path="/World/KeyLight",
                       spawn=sim_utils.DistantLightCfg(intensity=2600.0, angle=1.0),
                       init_state=AssetBaseCfg.InitialStateCfg(rot=(0.87, 0.34, 0.34, 0.0)))


def _t(x):
    return getattr(x, "torch", x)


def main() -> None:
    payload = json.load(open(args.rel))
    states = payload["states"][: args.states]
    print(f"REL {os.path.basename(args.rel)}: {len(states)} states")

    pinch_close = None
    if args.pinch:
        pinch_close = json.load(open(args.pinch))["close"]["joints"]
        print(f"[pinch] re-posing with the derived pinch from {os.path.basename(args.pinch)}")

    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device or "cuda:0"))
    scene = InteractiveScene(HandFrameSceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=False))

    camera = None
    if args.enable_cameras:
        from isaaclab.sensors import Camera, CameraCfg

        camera = Camera(CameraCfg(
            prim_path="/World/HandCam", update_period=0.0, height=720, width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=26.0, clipping_range=(0.005, 20.0))))

    sim.reset()
    robot = scene["robot"]
    cube = scene["cube"]
    contacts = [scene[f"tip{i}"] for i in range(1, 6)]
    dev = sim.device
    names = list(robot.data.joint_names)
    idx = {n: i for i, n in enumerate(names)}

    ids, _ = robot.find_bodies(["rl_dg_mount"], preserve_order=True)
    steps = int(args.settle_s / DT)
    frame_every = max(1, int(round((1.0 / args.fps) / DT)))
    frames: list = []
    records: list[dict] = []
    aimed = False

    for st in states:
        q = torch.zeros((1, len(names)), device=dev)
        table = dict(st["hand_joints"])
        if pinch_close:
            table.update(pinch_close)
        for n in names:
            q[0, idx[n]] = table[n]
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)

        # Put the cube back where it sat relative to the hand in the recorded state.
        bp = _t(robot.data.body_pos_w)[:, ids[0]]
        bq = _t(robot.data.body_quat_w)[:, ids[0]]
        rel_p = torch.tensor([st["cube_pos_mount"]], device=dev)
        rel_q = torch.tensor([st["cube_quat_mount"]], device=dev)
        pos = bp + math_utils.quat_apply(bq, rel_p)
        quat = math_utils.quat_mul(bq, rel_q)
        cube.write_root_pose_to_sim(torch.cat([pos, quat], dim=1))
        cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)

        start = _t(cube.data.root_pos_w)[0].clone()
        if camera is not None and not aimed:
            # Frame on the CUBE, not on the mount: the whole question is what the hand is doing
            # to the part, and the first attempt cropped so tightly on the palm that the cube
            # was outside the frame entirely.
            _aim(camera, start)
            aimed = True
        mass = float(_t(cube.data.default_mass)[0].sum())
        wrench = torch.zeros((1, 1, 3), device=dev)
        wrench[0, 0, 2] = -mass * G
        grip = torch.zeros(5, device=dev)
        n_acc = 0

        for s in range(steps):
            robot.set_joint_position_target(q)
            cube.set_external_force_and_torque(wrench, torch.zeros_like(wrench))
            scene.write_data_to_sim()
            render_now = camera is not None and s % frame_every == 0
            sim.step(render=render_now)
            scene.update(DT)
            if s > steps // 3:
                mags = []
                for c in contacts:
                    fm = _t(c.data.force_matrix_w)
                    mags.append(fm.reshape(1, -1, 3).norm(dim=-1).sum(dim=-1)[0]
                                if fm is not None else torch.zeros((), device=dev))
                grip += torch.stack(mags)
                n_acc += 1
            if render_now:
                camera.update(DT)
                frames.append(_t(camera.data.output["rgb"])[0, ..., :3].clone().cpu().numpy().astype("uint8"))

        end = _t(cube.data.root_pos_w)[0]
        drop_mm = float((start[2] - end[2])) * 1000
        moved_mm = float((end - start).norm()) * 1000
        g = (grip / max(n_acc, 1)).tolist()
        # Distance moved, not signed drop. A cube that interpenetrates the hand at spawn is
        # launched UPWARD by depenetration -- one state flew 4.4 m -- and a signed drop test
        # scored that as "held".
        held = moved_mm < 25.0
        records.append({"index": st["index"], "drop_mm": round(drop_mm, 1),
                        "moved_mm": round(moved_mm, 1), "held": held,
                        "grip_thumb_N": round(g[0], 3), "grip_index_N": round(g[1], 3),
                        "grip_other_N": round(sum(g[2:]), 3)})
        print(f"  state {st['index']:5d} moved {moved_mm:8.1f} mm  grip T={g[0]:5.2f}"
              f" I={g[1]:5.2f} oth={sum(g[2:]):5.2f} N"
              f" {'HELD' if held else ('LAUNCHED' if drop_mm < -50 else 'DROPPED')}", flush=True)

    n_held = sum(r["held"] for r in records)
    print(f"SUMMARY {os.path.basename(args.rel)}: {n_held}/{len(records)} held for {args.settle_s} s")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"rel": args.rel, "label": args.label, "pinch": bool(pinch_close),
                   "held": n_held, "shown": len(records), "records": records},
                  open(args.out, "w"), indent=1)
        print(f"WROTE {args.out}")

    if frames:
        import imageio.v3 as iio

        if args.stills:
            d = os.path.abspath(args.stills)
            os.makedirs(d, exist_ok=True)
            per = max(1, len(frames) // max(len(records), 1))
            for k in range(min(6, len(records))):
                j = min(len(frames) - 1, k * per + per - 2)
                iio.imwrite(os.path.join(d, f"state{k:02d}.png"), frames[j])
            print(f"WROTE stills to {d}")
        if args.video:
            v = os.path.abspath(args.video)
            os.makedirs(os.path.dirname(v), exist_ok=True)
            iio.imwrite(v, frames, fps=args.fps)
            print(f"WROTE {v}  ({len(frames)} frames)")


def _aim(camera, focus) -> None:
    """Same framing the pinch bench uses, which is the one camera path that works here."""
    target = focus
    span = 0.34
    eye = target + torch.tensor([0.62 * span, -0.86 * span, 0.36 * span], device=target.device)
    camera.set_world_poses_from_view(eye.unsqueeze(0).cpu(), target.unsqueeze(0).cpu())


main()
simulation_app.close()
