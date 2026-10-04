"""Hand-only pinch bench for the Tesollo DELTO DG-5F.

WHY THIS EXISTS
---------------
Grasp yield on the DELTO path is 0. Three candidate causes were conflated:

  (a) the jaw never closes far enough to touch the cube,
  (b) the distal joints stall short of their commanded target, and
  (c) the hand closes so hard it ejects a ~40 g cube on first touch.

All three are real and independent. Measured through the full UR10e + hand + OSC +
reset-manager stack they cannot be separated, so this bench strips the problem to one hand,
one cube, one Isaac process -- which is also all this box has room for (16 GB of VRAM fits
exactly one Isaac Sim, so the parallelism here is one config per ENVIRONMENT, not per process).

WHAT WAS ALREADY ESTABLISHED (out/jawcal.json, out/jawopt.json, out/probe.log)

  (a) GEOMETRY. |rl_dg_1_tip - rl_dg_2_tip| at DELTO_HAND_SCRIPTED_CLOSE is 51.03 mm against a
      34 mm cube. Tucking fingers 3/4/5 away changes that by 0.00 mm -- it is fingers 1 and 2
      that are too far apart. A cube at the declared grasp centre is touched by neither pad,
      which sits upstream of every force question asked so far. A sweep of 122 880 postures
      found apertures down to 1.35 mm, so the hand can oppose these tips; jawopt.py derives a
      posture pair that does (26 mm closed / 52 mm open, jaw centre within 5 mm of metadata's
      and drifting only 2.6 mm while closing).

  (b) CONTROL. The recorders drive the hand through RelativeJointPositionAction with scale=0.1
      and clip=(-1,1), so the PD target is never more than 0.1 rad ahead of measured and the
      realised torque is bounded by stiffness * 0.1 -- 0.0170 N*m on rj_dg_1_4, against 0.01
      friction and a 0.06 N*m effort limit that is never reached. Measured here: rj_dg_1_4
      finishes 1.011 rad short through the action term and 0.060 rad short with absolute
      position targets, at identical friction and stiffness.

  (c) EJECTION is what is left, and it only becomes a meaningful question once (a) and (b) are
      fixed. Before that, "the cube left the hand" and "the hand never touched the cube" are
      indistinguishable in the yield number.

THE JAW CENTRE IS MEASURED, NOT ASSUMED. An earlier version placed the cube using metadata's
grasp_center_offset applied in WORLD coordinates and put it 318 mm away, floating in mid-air,
because the mount frame is rotated ~180 degrees about x. Rather than fix the rotation and trust
it, this bench calibrates: it commands each environment's own closed posture in free space,
reads midpoint(rl_dg_1_tip, rl_dg_2_tip), and places the cube there. Correct by construction,
and per-posture -- which matters, because the optimised pinch has a different jaw centre from
the repo's.

GRAVITY is disabled on the cube in USD and applied by hand as m*g after release, so "weightless
while the grasp is established" is exact and per-environment rather than a global schedule.

WHAT IT SWEEPS
  ctrl      direct | rel        absolute position target vs the 0.1 rad/step relative term
  posture   repo_full           SCRIPTED_CLOSE, all five fingers  (what the current banks are)
            repo_pinch          SCRIPTED_CLOSE + 3/4/5 tucked     (the naive pinch)
            opt_pinch           jawopt.py's derived pair          (a jaw that actually closes)
  ramp_s    how fast the target travels open -> closed
  friction  the joint friction COEFFICIENT (ur10e_delto.py's comment calls it "0.01 N*m"; it is
            dimensionless, and that conflation is its own bug)
  stiff4    multiplier on distal _4 joint stiffness
  pin       cube held immovable while closing, or free to be punted

pin=on asks "does the closed jaw HOLD?"; pin=off asks "can it close on a free cube without
ejecting it?". Those are different failures with different fixes, and conflating them is what
made every previous yield number unreadable.

PROTOCOL (identical timing for every env)
  calibrate     command the closed posture in free space, read the jaw centre
  0.0 - 1.0 s   open posture, cube parked at that jaw centre, weightless
  1.0 - 1.0+r   target ramps open -> closed
  + 1.0 s       settle
  release       cube gets its weight; the pin, if any, is dropped
  + 2.0 s       hold window
  + 1.0 s       mass-scaled shake, as the repo's own sampler validates with

SCORING is geometric and local to this file: held == the cube is still within HOLD_RADIUS of
the measured jaw centre after the shake. Nothing here reads a success flag from code that has
its own opinion about what a grasp is.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--grid", type=str, default="main")
parser.add_argument("--out", type=str, default="out/sweep.json")
parser.add_argument("--video", type=str, default="")
parser.add_argument("--video-envs", type=int, default=0)
parser.add_argument("--stills", type=str, default="")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--focal", type=float, default=24.0, help="camera focal length mm")
parser.add_argument("--cam-span", type=float, default=0.0, help="override framing distance (m)")
parser.add_argument("--label", type=str, default="", help="text burned into the result json")
parser.add_argument("--only", type=str, default="")
parser.add_argument("--reps", type=int, default=1)
parser.add_argument("--friction-mu", type=float, default=0.0,
                    help="surface friction for the cube and hand pads (0 = leave the asset default)")
parser.add_argument("--postures", type=str,
                    default="/home/mattephi/Work/Rubitek/cube-stack/pinch/out/jawopt.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = bool(args.video)
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import uwlab_is6_bridge  # noqa: E402

uwlab_is6_bridge.apply()

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR  # noqa: E402
from uwlab_assets.robots.ur10e_delto.actions import (  # noqa: E402
    DELTO_HAND_OPEN_JOINT_POS,
    DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS,
    DELTO_TUCKED_HELD_FINGER_JOINT_POS,
)
from uwlab_assets.robots.ur10e_delto.ur10e_delto import DELTO_HAND

CUBE_USD = f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/InsertiveCube34/insertive_cube34.usd"
CUBE_SCALE = 0.85  # 0.85 * 40 mm = 34 mm, the size this hand's grip budget was sized against
CUBE_MASS = 0.03
G = 9.81

# A cube whose centre has wandered further than this from the measured jaw is not held. This
# started at 35 mm -- half a cube -- and that was too generous: a cube that had squirted out of
# the jaw entirely, ending 32 mm from BOTH pads, still scored as held. 20 mm is a real slip
# tolerance for a 34 mm part, and the contact test below is what actually decides.
HOLD_RADIUS = 0.020

# A pinch means both pads loaded and the other three fingers essentially unloaded. Without
# this, a cube resting on curled fingers 3/4/5 -- measured at thumb 2.22 N, index 0.00 N,
# others 9.97 N -- counts as a successful grasp, which is how the first "working" posture
# passed while being a scoop.
PINCH_MIN_N = 0.15
PINCH_MAX_OTHER_N = 0.60

DT = 1.0 / 120.0
CONTROL_HZ = 10.0  # the policy's rate; the relative term's 0.1 rad budget is PER CONTROL STEP


def _load_postures(path: str) -> dict[str, tuple[dict, dict]]:
    """(open, close) joint-angle dicts for each posture family under test."""
    repo_open = dict(DELTO_HAND_OPEN_JOINT_POS)
    repo_close = dict(DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS)
    naive_pinch = dict(repo_close)
    naive_pinch.update(DELTO_TUCKED_HELD_FINGER_JOINT_POS)
    out = {"repo_full": (repo_open, repo_close), "repo_pinch": (repo_open, naive_pinch)}

    # The optimised pair only exists once jawopt.py has run. Its absence is not fatal -- the two
    # repo postures are worth measuring alone -- but it must be LOUD, because silently falling
    # back to a posture that cannot close is exactly how a null result gets misread as "the
    # pinch does not work".
    if not os.path.exists(path):
        print(f"[postures] NOTE: {path} absent; opt_pinch unavailable this run")
        return out

    with open(path) as fh:
        payload = json.load(fh)
    # The tuck comes from the same solve, not from the module constant: the constant curls
    # fingers 3/4/5 under the jaw where they carry the cube as a shelf, and a bench that
    # silently substituted it would keep scoring scoops as pinches.
    tuck = payload.get("tucked", DELTO_TUCKED_HELD_FINGER_JOINT_POS)
    opt_open = dict(repo_open)
    opt_close = dict(repo_close)
    if len(payload["close"]["joints"]) < 20:
        # jawopt.py solves only fingers 1 and 2 and carries a tuck for 3/4/5 alongside.
        opt_open.update(tuck)
        opt_close.update(tuck)
    opt_open.update(payload["open"]["joints"])
    opt_close.update(payload["close"]["joints"])
    out["opt_pinch"] = (opt_open, opt_close)
    print(f"[postures] opt_pinch: open {payload['open']['aperture_mm']:.2f} mm ->"
          f" close {payload['close']['aperture_mm']:.2f} mm,"
          f" jaw drift {payload['jaw_drift_mm']:.2f} mm,"
          f" non-pad clearance {payload['close'].get('clearance_mm', '?')} mm")
    return out


POSTURES = _load_postures(args.postures)
HAVE_OPT = "opt_pinch" in POSTURES
ALL_POSTURES = ["repo_full", "repo_pinch"] + (["opt_pinch"] if HAVE_OPT else [])


def build_grid(name: str) -> list[dict]:
    if name == "main":
        axes = {
            "ctrl": ["direct", "rel"],
            "posture": ALL_POSTURES,
            "ramp_s": [0.0, 0.5, 2.0],
            "friction": [0.01, 3e-4, 0.0],
            "stiff4": [1.0, 10.0],
            "pin": [True, False],
        }
    elif name == "ctrl":
        axes = {
            "ctrl": ["direct", "rel"],
            "posture": ALL_POSTURES,
            "ramp_s": [0.0],
            "friction": [0.01],
            "stiff4": [1.0],
            "pin": [True],
        }
    elif name == "final":
        # What the sweeps settled on. pin=False only: holding the cube immovable while the hand
        # closes and then releasing it scored 0% true pinch in every single cell, against 25-42%
        # when the cube is free and weightless during the close. The pin stores contact energy
        # that is dumped into the part the instant it is released -- an artefact of the probe,
        # not of the grasp, and the repo's own sampler already closes against a weightless part.
        axes = {
            "ctrl": ["direct", "rel"],
            "posture": ["opt_pinch"],
            "ramp_s": [0.25, 0.5, 1.0, 2.0],
            "friction": [0.01, 3e-4],
            "stiff4": [1.0, 3.0],
            "pin": [False],
        }
    elif name == "pinch":
        # Once the geometry closes and the controller has authority, this is what is left:
        # which closing schedule and joint friction hold the cube without punting it.
        axes = {
            "ctrl": ["direct"],
            "posture": ["opt_pinch"] if HAVE_OPT else ["repo_pinch"],
            "ramp_s": [0.25, 0.5, 1.0, 2.0, 4.0],
            "friction": [0.01, 3e-3, 3e-4, 0.0],
            "stiff4": [1.0, 3.0, 10.0],
            "pin": [True, False],
        }
    elif name == "demo":
        # One env per posture family, free cube, slow close: the side-by-side video.
        axes = {
            "ctrl": ["direct"],
            "posture": ALL_POSTURES,
            "ramp_s": [2.0],
            "friction": [3e-4],
            "stiff4": [1.0],
            "pin": [False],
        }
    else:
        raise SystemExit(f"unknown grid {name!r}")
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*(axes[k] for k in keys))]


GRID = build_grid(args.grid)
if args.only:
    wanted = json.loads(args.only)
    GRID = [c for c in GRID if all(c[k] == v for k, v in wanted.items())]
GRID = GRID * args.reps
NUM_ENVS = len(GRID)
if not NUM_ENVS:
    raise SystemExit("empty grid after filtering")
print(f"[grid] {args.grid}: {NUM_ENVS} configs")


def _tip_contact(i: int) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rl_dg_%d_tip" % i,
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )


@configclass
class PinchSceneCfg(InteractiveSceneCfg):
    # activate_contact_sensors must be set on the SPAWN cfg, not after the fact -- missing that
    # is what once produced 0.000 N on all 25 hand bodies for a whole campaign.
    robot = DELTO_HAND.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=DELTO_HAND.spawn.replace(activate_contact_sensors=True),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CUBE_USD,
            scale=(CUBE_SCALE,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                # Weight is applied by hand after release, so the closing phase is weightless
                # per-environment instead of through a global gravity schedule.
                disable_gravity=True,
                kinematic_enabled=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
            # SURFACE friction, not joint friction, and ONLY when asked for. Attaching a
            # material unconditionally is itself a change: without one the cube keeps whatever
            # its USD authored, and silently replacing that with the reset scene's 0.3/0.2 made
            # a posture that measured 20% true pinch score 0% -- which looked like a friction
            # result and was actually the control being moved.
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=args.friction_mu * 1.5,
                dynamic_friction=args.friction_mu,
                restitution=0.0,
            ) if args.friction_mu else None,
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # FILTERED to the cube, and ONE SENSOR PER FINGERTIP. An unfiltered net force on a
    # fingertip includes self-contact, and this articulation has enabled_self_collisions=True:
    # tips 3/4/5 read 6.76 N while sitting 36 mm from the centre of a cube whose half-diagonal
    # is 29.4 mm -- they were pressing on the HAND, not on the part, and reading that as grip
    # force made a cube that had escaped the jaw look like it was carried by three fingers.
    #
    # Five separate sensors rather than one regex over five bodies, because PhysX sizes the
    # filter list as (sensor bodies x envs): a five-body sensor with one cube filter asks for
    # 15 entries and finds 3, and the whole sensor then fails to initialise.
    tip1 = _tip_contact(1)
    tip2 = _tip_contact(2)
    tip3 = _tip_contact(3)
    tip4 = _tip_contact(4)
    tip5 = _tip_contact(5)

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.4)),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1400.0, color=(0.93, 0.95, 1.0)),
    )
    key = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=2600.0, angle=1.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.87, 0.34, 0.34, 0.0)),
    )


def _t(x):
    """Isaac Lab 3.0 hands back warp-backed ProxyArrays from asset data."""
    return getattr(x, "torch", x)


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device or "cuda:0"))
    # replicate_physics must be off for FILTERED contact sensors: with it on, the filtered contact
    # view reports filter_count on a None backend for any num_envs > 1 (it happens to work at 1).
    # Scene creation is slower; at a couple of hundred hands that is seconds, and the alternative
    # is unfiltered forces that count the hand pressing on itself as grip.
    scene = InteractiveScene(
        PinchSceneCfg(num_envs=NUM_ENVS, env_spacing=0.75, replicate_physics=False)
    )

    camera, n_show = None, 0
    if args.video:
        from isaaclab.sensors import Camera, CameraCfg

        n_show = args.video_envs or min(4, NUM_ENVS)
        camera = Camera(CameraCfg(
            prim_path="/World/PinchCam", update_period=0.0, height=720, width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=args.focal, clipping_range=(0.005, 20.0)),
        ))

    # A camera's buffers (and its _device) exist only after the first reset, so it can be
    # constructed before this call but not aimed until after it.
    sim.reset()
    run(sim, scene, scene["robot"], scene["cube"], [scene[f"tip{i}"] for i in range(1, 6)],
        camera, n_show)


def run(sim, scene, robot: Articulation, cube: RigidObject, contacts: list[ContactSensor],
        camera, n_show: int) -> None:
    dev = sim.device
    names = list(robot.data.joint_names)
    idx = {n: i for i, n in enumerate(names)}

    print("CONTACT SENSORS:", [c.body_names[0] for c in contacts])
    mass = float(_t(cube.data.default_mass)[0].sum())
    print(f"CUBE MASS (as simulated) {mass:.5f} kg  (requested {CUBE_MASS} kg)")

    def posture_tensor(which: int) -> torch.Tensor:
        q = torch.zeros((NUM_ENVS, len(names)), device=dev)
        for e, cfg in enumerate(GRID):
            table = POSTURES[cfg["posture"]][which]
            for n in names:
                q[e, idx[n]] = table[n]
        return q

    open_q, close_q = posture_tensor(0), posture_tensor(1)

    friction = torch.zeros((NUM_ENVS, len(names)), device=dev)
    stiffness = _t(robot.data.joint_stiffness).clone()
    distal = [idx[f"rj_dg_{f}_4"] for f in range(1, 6)]
    for e, cfg in enumerate(GRID):
        friction[e, :] = cfg["friction"]
        stiffness[e, distal] *= cfg["stiff4"]
    robot.write_joint_friction_coefficient_to_sim(friction)
    robot.write_joint_stiffness_to_sim(stiffness)

    # ---- calibration: where IS this env's jaw, for its own closed posture? ------------
    tip_ids, _ = robot.find_bodies(["rl_dg_1_tip", "rl_dg_2_tip"], preserve_order=True)
    all_tip_ids, all_tip_names = robot.find_bodies(
        [f"rl_dg_{i}_tip" for i in range(1, 6)], preserve_order=True)
    zero_vel = torch.zeros((NUM_ENVS, 6), device=dev)
    park = scene.env_origins + torch.tensor([0.0, 0.0, -0.35], device=dev)
    cube.write_root_pose_to_sim(torch.cat([park, _identity_quat(dev)], dim=1))
    cube.write_root_velocity_to_sim(zero_vel)
    robot.write_joint_state_to_sim(close_q, torch.zeros_like(close_q))
    robot.set_joint_position_target(close_q)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(DT)

    bp = _t(robot.data.body_pos_w)
    tip1, tip2 = bp[:, tip_ids[0]], bp[:, tip_ids[1]]
    jaw_w = 0.5 * (tip1 + tip2)
    aperture = (tip1 - tip2).norm(dim=1)
    for p in dict.fromkeys(c["posture"] for c in GRID):
        e = next(i for i, c in enumerate(GRID) if c["posture"] == p)
        j = (jaw_w[e] - scene.env_origins[e]) * 1000
        print(f"CALIBRATED {p:11s} aperture {float(aperture[e]) * 1000:6.2f} mm"
              f"   jaw ({j[0]:6.1f},{j[1]:6.1f},{j[2]:6.1f}) mm")

    # ---- schedule --------------------------------------------------------------------
    ramp_max = max(c["ramp_s"] for c in GRID)
    t_close, t_release = 1.0, 1.0 + ramp_max + 1.0
    t_hold_end, t_shake_end = t_release + 2.0, t_release + 3.0
    total_steps = int(t_shake_end / DT) + 1
    ctrl_every = max(1, int(round((1.0 / CONTROL_HZ) / DT)))

    ramp_s = torch.tensor([c["ramp_s"] for c in GRID], device=dev).unsqueeze(1)
    is_rel = torch.tensor([c["ctrl"] == "rel" for c in GRID], device=dev).unsqueeze(1)
    pinned = torch.tensor([c["pin"] for c in GRID], device=dev)
    pin_pose = torch.cat([jaw_w, _identity_quat(dev)], dim=1)

    robot.write_joint_state_to_sim(open_q, torch.zeros_like(open_q))
    robot.set_joint_position_target(open_q)
    cube.write_root_pose_to_sim(pin_pose)
    cube.write_root_velocity_to_sim(zero_vel)
    scene.write_data_to_sim()
    if camera is not None:
        _aim(camera, jaw_w, n_show)

    commanded = open_q.clone()
    frames: list = []
    frame_every = max(1, int(round((1.0 / args.fps) / DT)))
    peak_speed = torch.zeros(NUM_ENVS, device=dev)
    peak_tip = torch.zeros((NUM_ENVS, 5), device=dev)
    # Peak-over-the-episode contact force is dominated by penetration transients at first
    # touch and by the pin releasing, and reads 15-35 N on a hand whose distal joints are
    # limited to 0.06 N*m -- physically impossible as a grip force. What the grasp is actually
    # holding with is the STEADY force during the hold window, so accumulate that separately.
    hold_sum = torch.zeros((NUM_ENVS, 5), device=dev)
    hold_n = 0
    dist_at_release = torch.zeros(NUM_ENVS, device=dev)
    tip_to_cube = torch.zeros((NUM_ENVS, 5), device=dev)
    wrench = torch.zeros((NUM_ENVS, 1, 3), device=dev)

    for step in range(total_steps):
        t = step * DT

        if t < t_close:
            goal = open_q
        else:
            alpha = torch.clamp((t - t_close) / torch.clamp(ramp_s, min=1e-6), max=1.0)
            goal = open_q + alpha * (close_q - open_q)

        if step % ctrl_every == 0:
            measured = _t(robot.data.joint_pos)
            # Reproduce the relative action term exactly: whatever the distance to target, it
            # can only ask for 0.1 rad beyond where the joint already is.
            rel_cmd = measured + torch.clamp(goal - measured, -0.1, 0.1)
            commanded = torch.where(is_rel, rel_cmd, goal)
        robot.set_joint_position_target(commanded)

        holding = pinned & (t < t_release)
        if holding.any():
            keep = holding.nonzero(as_tuple=False).squeeze(-1)
            cube.write_root_pose_to_sim(pin_pose[keep], env_ids=keep)
            cube.write_root_velocity_to_sim(zero_vel[keep], env_ids=keep)

        if t >= t_release:
            wrench[:, 0, 2] = -mass * G
            if t >= t_hold_end:
                # Mass-scaled shake, mirroring how the repo's sampler validates a candidate.
                ph = 2.0 * math.pi * 3.0 * (t - t_hold_end)
                wrench[:, 0, 0] = mass * 6.0 * math.sin(ph)
                wrench[:, 0, 1] = mass * 6.0 * math.cos(ph)
            cube.set_external_force_and_torque(wrench, torch.zeros_like(wrench))

        scene.write_data_to_sim()
        render_now = camera is not None and step % frame_every == 0
        sim.step(render=render_now)
        scene.update(DT)

        peak_speed = torch.maximum(peak_speed, _t(cube.data.root_lin_vel_w).norm(dim=1))
        mags = []
        for c in contacts:
            fm = _t(c.data.force_matrix_w)
            mags.append(fm.reshape(NUM_ENVS, -1, 3).norm(dim=-1).sum(dim=-1)
                        if fm is not None else torch.zeros(NUM_ENVS, device=dev))
        mag = torch.stack(mags, dim=1)  # (num_envs, 5) force on each pad FROM THE CUBE
        peak_tip = torch.maximum(peak_tip, mag)
        if t_release + 0.5 <= t < t_hold_end:
            hold_sum += mag
            hold_n += 1
        if abs(t - t_release) < DT * 0.5:
            dist_at_release = (_t(cube.data.root_pos_w) - jaw_w).norm(dim=1)
        if abs(t - (t_hold_end - DT)) < DT * 0.5:
            # Which fingertips are actually near the cube at the end of the hold. Contact force
            # alone said tips 3/4/5 were carrying 9.97 N while kinematics said they sat 63 mm
            # from the jaw, and those cannot both be true of a 34 mm cube. Measure it.
            bp_now = _t(robot.data.body_pos_w)
            cp_now = _t(cube.data.root_pos_w)
            tip_to_cube = torch.stack(
                [(bp_now[:, j] - cp_now).norm(dim=1) for j in all_tip_ids], dim=1)

        if render_now:
            camera.update(DT)
            frames.append(_t(camera.data.output["rgb"])[0, ..., :3].clone().cpu().numpy().astype("uint8"))

    # ---- score -----------------------------------------------------------------------
    final_dist = (_t(cube.data.root_pos_w) - jaw_w).norm(dim=1)
    held = final_dist < HOLD_RADIUS
    err = (close_q - _t(robot.data.joint_pos)).abs()
    tips = [c.body_names[0] for c in contacts]
    i1, i2 = tips.index("rl_dg_1_tip"), tips.index("rl_dg_2_tip")
    hold_force = hold_sum / max(hold_n, 1)
    # "Held" by distance alone cannot tell a pinch from a scoop: the naive tuck curls fingers
    # 3/4/5 under the jaw and the cube rests ON them, which passes any distance test. A pinch
    # requires BOTH pads loaded and the other three carrying next to nothing.
    other_idx = [k for k in range(len(tips)) if k not in (i1, i2)]
    pinched = (
        held
        & (hold_force[:, i1] > PINCH_MIN_N)
        & (hold_force[:, i2] > PINCH_MIN_N)
        & (hold_force[:, other_idx].sum(dim=1) < PINCH_MAX_OTHER_N)
    )

    results = [{
        **cfg,
        "held": bool(held[e]),
        "pinched": bool(pinched[e]),
        "final_dist_mm": round(float(final_dist[e]) * 1000, 2),
        "dist_at_release_mm": round(float(dist_at_release[e]) * 1000, 2),
        "aperture_mm": round(float(aperture[e]) * 1000, 2),
        "peak_cube_speed": round(float(peak_speed[e]), 4),
        "thumb_err_rad": round(float(err[e, idx["rj_dg_1_4"]]), 4),
        "index_err_rad": round(float(err[e, idx["rj_dg_2_4"]]), 4),
        "max_distal_err_rad": round(float(err[e, distal].max()), 4),
        "grip_thumb_N": round(float(hold_force[e, i1]), 3),
        "grip_index_N": round(float(hold_force[e, i2]), 3),
        "grip_other_N": round(float(hold_force[e].sum() - hold_force[e, i1] - hold_force[e, i2]), 3),
        "peak_thumb_N": round(float(peak_tip[e, i1]), 3),
        "peak_index_N": round(float(peak_tip[e, i2]), 3),
        "tip_to_cube_mm": [round(float(v) * 1000, 1) for v in tip_to_cube[e]],
    } for e, cfg in enumerate(GRID)]

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"grid": args.grid, "label": args.label, "postures": args.postures,
                   "num_envs": NUM_ENVS, "cube_mass_kg": mass, "results": results}, fh, indent=1)
    n = sum(r["held"] for r in results)
    p = sum(r["pinched"] for r in results)
    print(f"WROTE {out}  ({NUM_ENVS} configs, {n} held = {100 * n / NUM_ENVS:.1f}%,"
          f" {p} true pinch = {100 * p / NUM_ENVS:.1f}%)")

    if frames:
        import imageio.v3 as iio

        if args.stills:
            d = os.path.abspath(args.stills)
            os.makedirs(d, exist_ok=True)
            k = len(frames)
            for label, i in [("a_open", 0), ("b_closing", k // 3), ("c_closed", k // 2), ("d_end", k - 1)]:
                iio.imwrite(os.path.join(d, f"{label}.png"), frames[i])
            print(f"WROTE stills to {d}")
        vid = os.path.abspath(args.video)
        os.makedirs(os.path.dirname(vid), exist_ok=True)
        iio.imwrite(vid, frames, fps=args.fps)
        print(f"WROTE {vid}  ({len(frames)} frames)")


def _identity_quat(dev):
    q = torch.zeros((NUM_ENVS, 4), device=dev)
    q[:, 0] = 1.0
    return q


def _aim(camera, jaw_w, n_show: int) -> None:
    """Frame the jaws being rendered, from their MEASURED positions rather than a guess."""
    target = jaw_w[:n_show].mean(dim=0)
    spread = float((jaw_w[:n_show] - target).norm(dim=1).max()) if n_show > 1 else 0.0
    # A 24 mm lens on Isaac's default 20.955 mm aperture is a ~47 degree horizontal field, so
    # the eye has to stand off about 1.2x the width it needs to cover. Getting this wrong is not
    # a cosmetic problem: the first render framed the mount and cropped the jaw entirely out of
    # shot, which is the one thing the video exists to show.
    span = args.cam_span or (0.115 + 2.6 * spread)
    eye = target + torch.tensor([0.62 * span, -0.86 * span, 0.36 * span], device=target.device)
    camera.set_world_poses_from_view(eye.unsqueeze(0).cpu(), target.unsqueeze(0).cpu())


main()
simulation_app.close()
