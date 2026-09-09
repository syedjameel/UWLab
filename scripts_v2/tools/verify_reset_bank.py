"""Check that a *EEGrasped reset bank is actually made of grasps, and optionally film it.

WHY THIS EXISTS. A grasped reset state must place the object at the hand's grasp point -- for
the DELTO hand about 170 mm from ``rl_dg_mount``. The banks this repository shipped do not.
Measured over 40 states each:

    resets_ObjectRestingEEGrasped.pt      0/40 in range, median 281 mm, max 561 mm
    resets_ObjectAnywhereEEGrasped.pt     3/40 in range, median 397 mm, max 792 mm

The arm was up to three quarters of a metre from the cube it was nominally holding. Nothing
noticed: the states load, the physics runs, the training loop reports a success rate of zero and
looks like a hyperparameter problem. Three of the four families a policy resets from were not
grasps at all.

The check is arithmetic on data the bank already contains, so it belongs in front of every bank
that claims to be grasped rather than being rediscovered:

    verify_reset_bank.py --bank .../resets_ObjectAnywhereEEGrasped.pt --states 40 \
        --settle-s 0.02 --verify-grasped        # exit 0 pass, 2 fail

Run without --verify-grasped it replays and films the bank instead, which is the other half of
judging a substage: one state is written, real physics runs for --settle-s, and a grasp that
does not hold drops the cube on camera.

ORIGINAL DOCSTRING FOLLOWS.

Replay a reset-state bank and film it, so a substage can be judged by looking at it.

WHY NOT THE EXISTING VIEWPORT RENDERER. scripts_v2/tools/render_reset_states_viewport.py builds
the whole ManagerBasedRLEnv to replay a bank, which drags in the event managers, the reset-type
mixture loader (it loads ALL four banks, not the one being rendered) and a CollisionAnalyzer
that does not finish on this box. None of that is needed to look at stored states: a bank is
just poses and joint angles, and this scene can be written directly.

WHAT A FRAME SHOWS. One state is written into the scene, then REAL physics runs for
--settle-s before the next state is loaded. Nothing is frozen and nothing is posed to flatter
the grasp: if the hand is not actually holding the cube, the cube falls on camera. The
per-state displacement over that settle window is measured and reported alongside the video,
so the number and the picture come from the same run.

The camera is fixed on the workspace for the whole clip so states can be compared to each
other, which is the point of filming a distribution rather than a single state.
"""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bank", type=str, required=True, help="path to a resets_<Type>.pt")
parser.add_argument("--states", type=int, default=12, help="how many states to show")
parser.add_argument("--settle-s", type=float, default=1.5, help="physics seconds per state")
parser.add_argument("--stride", type=int, default=0, help="take every Nth state (0 = spread evenly)")
parser.add_argument("--video", type=str, default="")
parser.add_argument("--stills", type=str, default="")
parser.add_argument("--out", type=str, default="")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--focal", type=float, default=20.0)
parser.add_argument("--cam-eye", type=str, default="")
parser.add_argument("--cam-at", type=str, default="")
parser.add_argument("--label", type=str, default="")
parser.add_argument("--verify-grasped", action="store_true",
                    help="fail unless the states really are grasps (see _verify_grasped)")
parser.add_argument("--grasp-lo-mm", type=float, default=120.0)
parser.add_argument("--grasp-hi-mm", type=float, default=210.0)
parser.add_argument("--min-grasped-frac", type=float, default=0.80)
parser.add_argument("--dump-relative", type=str, default="",
                    help="write hand joint angles + cube pose in the hand frame, and stop")
parser.add_argument("--view-probe", type=str, default="",
                    help="render one frame from each of several candidate eyes and stop")
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
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR  # noqa: E402
from uwlab_assets.robots.ur10e_delto.ur10e_delto import IMPLICIT_UR10E_DELTO  # noqa: E402

DT = 1.0 / 120.0

# The five scene entities a StableState bank stores, and the assets the cube-stacking pair uses.
# Kept in the same order the recorder writes them so a missing key is loud rather than silent.
BANK_RIGID = ["insertive_object", "receptive_object", "table", "ur5_metal_support"]


def _cube(usd: str, mass: float | None, scale: float, kinematic: bool = False) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + os.path.basename(usd).split(".")[0],
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd,
            scale=(scale,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=kinematic,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass) if mass else None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.1)),
    )


@configclass
class BankSceneCfg(InteractiveSceneCfg):
    robot = IMPLICIT_UR10E_DELTO.replace(prim_path="{ENV_REGEX_NS}/Robot")

    insertive_object = _cube(
        f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/InsertiveCube34/insertive_cube34.usd", 0.03, 0.85
    )
    receptive_object = _cube(
        f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/ReceptiveCube34/receptive_cube34.usd",
        None, 0.85, kinematic=True,
    )
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Mounts/CustomLabTable/custom_lab_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    ur5_metal_support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/UR5MetalSupport",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.004)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Mounts/CustomLabTable/custom_mount_plate.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    sky = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1200.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                         "kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    key = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=1800.0, angle=1.2),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.87, 0.34, 0.34, 0.0)),
    )


def _t(x):
    return getattr(x, "torch", x)


def _write_dump(dump: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(args.dump_relative)), exist_ok=True)
    with open(args.dump_relative, "w") as fh:
        json.dump({"bank": args.bank, "states": dump}, fh, indent=1)
    print(f"WROTE {args.dump_relative} ({len(dump)} states)")


def _verify_grasped(dump: list) -> bool:
    """Is a *EEGrasped bank actually made of grasps? An exit code, not a footnote.

    A grasped reset state must place the object at the hand's grasp point -- for this hand about
    170 mm from rl_dg_mount, and the recorded pinch grasp bank spans 133-180 mm. The SHIPPED
    banks do not: measured over 40 states each, ObjectRestingEEGrasped had 0/40 in range with a
    median of 281 mm, and ObjectAnywhereEEGrasped 3/40 with a median of 397 mm and a worst case
    of 792 mm. The arm was up to three quarters of a metre from the cube it was nominally
    holding, and nothing in the pipeline noticed: the states load, the physics runs, and
    training simply never learns.

    This is three lines of arithmetic on data the bank already contains. It belongs in front of
    every bank that claims to be grasped, which is why it sets an exit code.
    """
    import math

    d = sorted(math.dist([v * 1000 for v in s["cube_pos_mount"]], [0, 0, 0]) for s in dump)
    if not d:
        print("VERIFY: no states"); return False
    inrange = [v for v in d if args.grasp_lo_mm <= v <= args.grasp_hi_mm]
    frac = len(inrange) / len(d)
    print(f"VERIFY grasped: {len(inrange)}/{len(d)} states have the object "
          f"{args.grasp_lo_mm:.0f}-{args.grasp_hi_mm:.0f} mm from rl_dg_mount "
          f"({frac * 100:.0f}%, need {args.min_grasped_frac * 100:.0f}%). "
          f"min {d[0]:.1f} median {d[len(d) // 2]:.1f} max {d[-1]:.1f} mm")
    if frac < args.min_grasped_frac:
        print("VERIFY FAILED: this bank is not made of grasps. The hand is not on the object.")
        return False
    print("VERIFY PASSED")
    return True


def _dump_relative(scene, bank, i: int, dump: list) -> None:
    """Cube pose expressed in the hand's mount frame, plus the twenty hand joint angles.

    This is what lets a bank state be replayed in the HAND-ONLY scene, whose camera can
    actually be aimed. The arm is dropped: what a grasped reset state looks like is a property
    of the hand and the part, and the arm only decides where in the room that happens.
    """
    import isaaclab.utils.math as mu

    robot = scene["robot"]
    ids, _ = robot.find_bodies(["rl_dg_mount"], preserve_order=True)
    bp = _t(robot.data.body_pos_w)[0, ids[0]]
    bq = _t(robot.data.body_quat_w)[0, ids[0]]
    cp = _t(scene["insertive_object"].data.root_pos_w)[0]
    cq = _t(scene["insertive_object"].data.root_quat_w)[0]
    inv_q = mu.quat_conjugate(bq.unsqueeze(0))
    rel_p = mu.quat_apply(inv_q, (cp - bp).unsqueeze(0))[0]
    rel_q = mu.quat_mul(inv_q, cq.unsqueeze(0))[0]
    names = list(robot.data.joint_names)
    q = _t(robot.data.joint_pos)[0]
    hand = {n: float(q[j]) for j, n in enumerate(names) if n.startswith("rj_dg_")}
    dump.append({"index": i,
                 "cube_pos_mount": [float(v) for v in rel_p],
                 "cube_quat_mount": [float(v) for v in rel_q],
                 "hand_joints": hand})


def _view_probe(sim, scene, cube, out_dir: str, focal: float) -> None:
    """Render one frame from each of several candidate viewpoints and write them side by side.

    Every analytic attempt at framing this scene has put the camera under the table or in the
    sky, and the pose readbacks agree with the intention while the pixels do not. So: try
    several, look at them, pick one.
    """
    import imageio.v3 as iio
    import numpy as np

    os.makedirs(os.path.abspath(out_dir), exist_ok=True)
    # Offsets from the cube. The renders come out 180 degrees rolled (see the frame append in
    # main), so candidates BELOW the cube are included deliberately: after the rotation they
    # read as views from above.
    cands = {
        "a_below_near": (0.30, -0.40, -0.35),
        "b_below_far": (0.50, -0.65, -0.55),
        "c_below_steep": (0.20, -0.30, -0.70),
        "d_above_near": (0.30, -0.40, 0.35),
        "e_level": (0.45, -0.60, 0.02),
        "f_below_wide": (0.70, -0.90, -0.45),
    }
    for name, off in cands.items():
        eye = f"{cube[0] + off[0]},{cube[1] + off[1]},{cube[2] + off[2]}"
        at = f"{cube[0]},{cube[1]},{cube[2]}"
        grab = _replicator_camera(sim, eye, at, focal)
        sim.step(render=True)
        img = grab()
        if img is not None:
            img = img[::-1, ::-1]
            iio.imwrite(os.path.join(os.path.abspath(out_dir), f"{name}.png"), img)
            a = img.astype("float32")
            flat = float(np.mean((a.std(axis=2) < 12.0)))
            print(f"  view {name}: eye offset {off}  flat {flat * 100:.0f}%")
    print(f"WROTE view probe to {os.path.abspath(out_dir)}")


def _replicator_camera(sim, eye_s: str, at_s: str, focal: float):
    """A camera that actually points where it is told, via omni.replicator.

    Isaac Lab's Camera sensor cannot be aimed in this build. ``set_world_poses_from_view``
    never moves the prim; ``CameraCfg.OffsetCfg`` applies the translation but not the rotation;
    and writing the prim's xform ops directly is clobbered again by the sensor, which rewrites
    the pose from its own buffers on every update. All three report a correct pose on readback
    while the renderer photographs the sky -- which is how four bank films came back as
    1280x720 pictures of nothing, with the right frame count and a plausible file size.

    ``rep.create.camera(position=..., look_at=...)`` bakes the orientation into the prim at
    creation and nothing else writes to it afterwards. Returns a callable that yields the
    current RGB frame.
    """
    import numpy as np
    import omni.replicator.core as rep

    eye = tuple(float(v) for v in eye_s.split(","))
    at = tuple(float(v) for v in at_s.split(","))
    cam = rep.create.camera(position=eye, look_at=at, focal_length=float(focal))
    rp = rep.create.render_product(cam, (1280, 720))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([rp])
    print(f"CAMERA (replicator) at {eye} looking at {at}, focal {focal} mm")

    def grab():
        # Two extra renders before reading. A single one returns a TORN frame -- the top band
        # from the current render and the bottom from the previous one, which looks like the
        # robot appearing twice at different heights. The annotator is double buffered and the
        # first render only starts filling it.
        sim.render()
        sim.render()
        data = np.asarray(ann.get_data())
        if data.size == 0:
            return None
        return data[..., :3].astype("uint8")

    return grab


def _place_camera_prim(prim_path: str, eye_s: str, at_s: str) -> None:
    """Set the camera prim's transform directly through USD.

    Isaac Lab's two documented ways of pointing a camera both fail silently in this build:
    ``set_world_poses_from_view`` never moves the prim at all, and ``CameraCfg.OffsetCfg``
    applies the translation but not the rotation. Both report a correct pose on readback while
    the renderer photographs the sky -- which is how four bank films came back as 1280x720
    pictures of nothing. Writing the xform ops is the one path that the renderer cannot
    disagree with, because it is the same prim it reads.

    USD cameras look down their own -Z with +Y up, so the basis is [right, up, -forward].
    """
    import math as _math

    from pxr import Gf, UsdGeom
    import omni.usd

    eye = [float(v) for v in eye_s.split(",")]
    at = [float(v) for v in at_s.split(",")]
    f = [a - e for a, e in zip(at, eye)]
    n = _math.sqrt(sum(v * v for v in f))
    f = [v / n for v in f]
    up_ref = (0.0, 0.0, 1.0)
    right = [f[1] * up_ref[2] - f[2] * up_ref[1],
             f[2] * up_ref[0] - f[0] * up_ref[2],
             f[0] * up_ref[1] - f[1] * up_ref[0]]
    n = _math.sqrt(sum(v * v for v in right))
    right = [v / n for v in right]
    # right x up must equal -forward for a right-handed camera basis (X right, Y up, Z back),
    # which gives up = right x forward. Negating it here put the camera on its back looking at
    # the sky, with the pose still reading as correct.
    up = [right[1] * f[2] - right[2] * f[1],
          right[2] * f[0] - right[0] * f[2],
          right[0] * f[1] - right[1] * f[0]]

    # Gf.Matrix3d rows are the basis vectors; Matrix4d gives us a rotation quaternion directly.
    rot4 = Gf.Matrix4d(
        right[0], right[1], right[2], 0.0,
        up[0], up[1], up[2], 0.0,
        -f[0], -f[1], -f[2], 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    quat = rot4.ExtractRotationQuat()
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"no camera prim at {prim_path}")
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*eye))
    # The prim already carries a double-precision orient op from the spawn, and AddXformOp
    # refuses to change precision -- so match it rather than fight it.
    x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(quat))
    print(f"CAMERA prim placed at {eye} looking {[round(v, 3) for v in f]}")


def _frame_is_mostly_sky(frame) -> float:
    """Fraction of the frame that is flat background.

    Verifying a camera by reading its pose back is not enough here: this scene reported a
    perfect position AND a perfect look direction and still rendered the sky, because only the
    position was ever applied to the prim the renderer uses. So the check is made on the
    PIXELS, which cannot lie about what was photographed.
    """
    import numpy as np

    a = frame.astype("float32")
    flat = (a.std(axis=2) < 12.0) & (a[:, :, 2] > a[:, :, 0])
    return float(np.mean(flat))


def _quat_apply(q, v):
    w, x, y, z = q[0], q[1], q[2], q[3]
    t = 2.0 * torch.stack([
        y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0]])
    return v + w * t + torch.stack([
        y * t[2] - z * t[1], z * t[0] - x * t[2], x * t[1] - y * t[0]])


def main() -> None:
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)["initial_state"]
    total = len(bank["articulation"]["robot"]["joint_position"])
    n = min(args.states, total)
    stride = args.stride or max(1, total // n)
    picks = list(range(0, total, stride))[:n]
    print(f"BANK {os.path.basename(args.bank)}: {total} states, showing {len(picks)} "
          f"(stride {stride})")

    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device or "cuda:0"))
    scene = InteractiveScene(BankSceneCfg(num_envs=1, env_spacing=2.0))

    camera = None
    if args.enable_cameras:
        from isaaclab.sensors import Camera, CameraCfg


        # The pose is given as the cfg's spawn OFFSET rather than written afterwards.
        # set_world_poses / set_world_poses_from_view both left this camera sitting at the
        # origin -- confirmed by reading the pose back, after four bank films came out as
        # 1280x720 pictures of the sky. An offset is applied when the prim is created, so
        # there is no window in which the camera exists at the wrong place.
        camera = Camera(CameraCfg(
            prim_path="/World/BankCam", update_period=0.0, height=720, width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=args.focal, clipping_range=(0.01, 30.0)),
        ))

    sim.reset()

    grab = None

    robot = scene["robot"]
    dev = sim.device
    names = list(robot.data.joint_names)
    # A bank stores a bare 26-vector with no joint names. Writing it positionally is only valid
    # if this scene orders its joints the same way the recording scene did -- and this repo's
    # own comments warn repeatedly that these tables are finger-major while the articulation
    # reports level-major. Print the order so a mismatch is visible rather than showing up as
    # an arm that is mysteriously nowhere near the object it is supposed to be holding.
    print("SCENE JOINT ORDER:", names)
    steps = int(args.settle_s / DT)
    frame_every = max(1, int(round((1.0 / args.fps) / DT)))

    frames: list = []
    records: list[dict] = []
    dump: list[dict] = []

    for k, i in enumerate(picks):
        # ---- write the stored state ---------------------------------------------------
        q = bank["articulation"]["robot"]["joint_position"][i].to(dev).unsqueeze(0)
        qd = bank["articulation"]["robot"]["joint_velocity"][i].to(dev).unsqueeze(0)
        if q.shape[1] != len(names):
            raise SystemExit(f"bank stores {q.shape[1]} joints, scene has {len(names)}")
        robot.write_root_pose_to_sim(
            bank["articulation"]["robot"]["root_pose"][i].to(dev).unsqueeze(0))
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
        robot.write_joint_state_to_sim(q, qd)
        # Hold the stored posture: with no target written, the PD drives every joint to its
        # default and the arm swings away from the state we are meant to be photographing.
        robot.set_joint_position_target(q)

        for name in BANK_RIGID:
            asset = scene[name]
            asset.write_root_pose_to_sim(
                bank["rigid_object"][name]["root_pose"][i].to(dev).unsqueeze(0))
            asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))

        stored = bank["rigid_object"]["insertive_object"]["root_pose"][i][:3].clone()
        if args.dump_relative or args.verify_grasped:
            _dump_relative(scene, bank, i, dump)
            continue
        if args.view_probe and k == 0:
            _view_probe(sim, scene, [float(v) for v in stored], args.view_probe, args.focal)
            return
        if grab is None and args.enable_cameras:
            # Framed from the scene's OWN measured geometry rather than from hand-picked world
            # coordinates. Every guess so far put the camera under the table or in the sky; the
            # cube's actual position is the one thing here that cannot be wrong about where the
            # workspace is.
            c = [float(v) for v in stored]
            # OBLIQUE, deliberately. rep.create.camera's look_at degenerates as the view
            # approaches vertical -- a camera 0.85 m straight above the cube came back looking
            # UP at the arm against the sky, while a 35-degree view of the same scene rendered
            # the tabletop and both cubes correctly. The "180 degree roll" earlier inferred from
            # those vertical frames was that degeneracy, not a roll, and rotating the image was
            # the wrong fix.
            # Above and outside the table, looking down at about 40 degrees. Level with the
            # work surface, the table's own front edge cuts a black band straight across the
            # frame; near-vertical, the replicator look_at degenerates. 40 degrees clears both.
            eye = args.cam_eye or f"{c[0] + 0.72},{c[1] - 0.62},{c[2] + 0.78}"
            at = args.cam_at or f"{c[0]},{c[1]},{c[2]}"
            grab = _replicator_camera(sim, eye, at, args.focal)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)
        start = _t(scene["insertive_object"].data.root_pos_w)[0].clone()
        # A cube that jumps on the very FIRST step is being depenetrated, which means this
        # replay scene does not match the one the bank was recorded in. Report that separately
        # rather than folding it into "drift", or the video quietly misrepresents the bank.
        pop_mm = float((start.cpu() - stored).norm()) * 1000

        # ---- settle under real physics ------------------------------------------------
        for s in range(steps):
            robot.set_joint_position_target(q)
            scene.write_data_to_sim()
            render_now = grab is not None and s % frame_every == 0
            sim.step(render=render_now)
            scene.update(DT)
            if render_now:
                img = grab()
                if img is not None:
                    frames.append(img)

        end = _t(scene["insertive_object"].data.root_pos_w)[0]
        drift_mm = float((end - start).norm()) * 1000
        fell = bool(end[2] < start[2] - 0.02)
        records.append({"index": i, "drift_mm": round(drift_mm, 2), "fell": fell,
                        "pop_mm": round(pop_mm, 2),
                        "z_stored_mm": round(float(stored[2]) * 1000, 1),
                        "z_start_mm": round(float(start[2]) * 1000, 1),
                        "z_end_mm": round(float(end[2]) * 1000, 1)})
        print(f"  state {i:5d}  pop {pop_mm:6.2f} mm  drift {drift_mm:7.2f} mm  z "
              f"{float(stored[2]) * 1000:6.1f}(stored) -> {float(start[2]) * 1000:6.1f}"
              f" -> {float(end[2]) * 1000:6.1f} mm{'  FELL' if fell else ''}", flush=True)

    if frames:
        sky = _frame_is_mostly_sky(frames[len(frames) // 2])
        print(f"FRAME CHECK: {sky * 100:.0f}% of the frame is flat background")
        if sky > 0.80:
            print("FRAME WARNING: the camera is photographing empty sky. The clip is unusable;"
                  " fix the framing before believing anything below.")

    if args.verify_grasped:
        ok = _verify_grasped(dump)
        if args.dump_relative:
            _write_dump(dump)
        simulation_app.close()
        raise SystemExit(0 if ok else 2)

    if args.dump_relative:
        _write_dump(dump)
        return

    stable = [r for r in records if not r["fell"] and r["drift_mm"] < 10.0]
    pops = sorted(r["pop_mm"] for r in records)
    print(f"SUMMARY {os.path.basename(args.bank)}: {len(stable)}/{len(records)} states stable "
          f"(<10 mm drift, no fall) over {args.settle_s} s of physics; "
          f"median first-step pop {pops[len(pops) // 2]:.2f} mm, max {pops[-1]:.2f} mm")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"bank": args.bank, "label": args.label, "settle_s": args.settle_s,
                       "total_states": total, "shown": len(picks),
                       "stable": len(stable), "records": records}, fh, indent=1)
        print(f"WROTE {args.out}")

    if frames:
        import imageio.v3 as iio

        if args.stills:
            d = os.path.abspath(args.stills)
            os.makedirs(d, exist_ok=True)
            per = max(1, len(frames) // len(picks))
            for k in range(min(6, len(picks))):
                j = min(len(frames) - 1, k * per + per - 2)
                iio.imwrite(os.path.join(d, f"state{k:02d}.png"), frames[j])
            print(f"WROTE stills to {d}")
        if args.video:
            v = os.path.abspath(args.video)
            os.makedirs(os.path.dirname(v), exist_ok=True)
            iio.imwrite(v, frames, fps=args.fps)
            print(f"WROTE {v}  ({len(frames)} frames)")


main()
simulation_app.close()
