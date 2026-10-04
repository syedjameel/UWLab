"""Optimise the pinch posture against MEASURED HOLD, not against a kinematic proxy.

WHY THIS EXISTS. Every posture derived so far was scored on geometry: pad separation, clearance
of the other links, drift of the grasp point. Those are all necessary and none of them is the
objective. The objective is whether the cube is still in the hand after gravity and a shake, and
nothing has ever been optimised for it. The results say so:

    pinchpose.json      30.0 mm pads, clean geometry      0% true pinch
    pinchpose_22.json   22 mm pads                       23% true pinch
    pinchpose_seg.json  21.7 mm, clears the sampler gate  0% -- pads never touch

The gap between the second and third is the whole problem: satisfying the sampler's placement
gate pushed the fingers off the cube, and no geometric score noticed.

HOW. One candidate posture per ENVIRONMENT, evaluated by the same protocol pinchlab.py uses --
close onto a weightless cube, release, hold under gravity, shake at 3 Hz -- and scored on what
comes out: both pads loaded by the cube, the other three fingers not, and the cube still there.
A generation is therefore ONE Isaac run of a few hundred environments rather than hundreds of
runs, which is the only reason this is affordable on one GPU.

Cross-entropy over the twenty joints, seeded from the best posture found geometrically, with the
clearance requirement kept as a hard filter so the winner is still something the grasp sampler's
placement gate will accept.
"""

from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed-pose", type=str,
                    default="/home/mattephi/Work/Rubitek/cube-stack/pinch/out/pinchpose_22.json")
parser.add_argument("--envs", type=int, default=256, help="environments per generation")
parser.add_argument("--reps", type=int, default=4,
                    help="environments per CANDIDATE. The physics is stochastic -- PhysX's\nbatched solver does not give bit-identical results across envs -- so a posture scored from a\nsingle env is scored with a coin flip attached, and the CEM will happily chase a lucky draw.\nThe smoke run showed exactly that: elite mean FELL from gen 0 to gen 1. Averaging over\nreplicates costs candidates per generation and buys a fitness that means something.")
parser.add_argument("--gens", type=int, default=10)
parser.add_argument("--elite", type=float, default=0.15)
parser.add_argument("--sigma", type=float, default=0.06, help="initial rad of exploration")
parser.add_argument("--clear-mm", type=float, default=30.0,
                    help="hard filter: non-pad links must clear the grasp point by this much")
parser.add_argument("--pinch-fingers-only", action="store_true",
                    help="""close ONLY fingers 1 and 2; hold 3, 4 and 5 at their open pose. The
DG-5F's own metadata calls its grasp "a virtual jaw of finger 1 against finger 2", and the other
three have no part in it -- but the closed posture curls all twenty joints, and measured in the
sampler with the cube pinned through the whole close, that stops the jaw at 58 mm with the pads
never within 27 mm of the part. Holding 3/4/5 open takes the same jaw to 35 mm: they reach the cube
first and the pinch pair never gets there.""")
parser.add_argument("--eval-only", action="store_true",
                    help="""skip the search and re-evaluate --seed-pose's CLOSED posture as it is.
--gens 1 is NOT this: the winner is then the best of a random spread around the seed, so a sweep run
that way measures random postures. That mistake has been made twice in this investigation, once for
a velocity sweep and once for a hold-duration sweep; both results were void.""")
parser.add_argument("--approach-axes", type=str, default=None,
                    help="""pinchaxes JSON. When given, the hand is spawned rotated so that its
CLOSED approach direction points straight down, which is the only orientation the grasp sampler
ever uses (it builds every candidate by taking the approach axis to world -Z). Scored in the hand's
own default orientation instead, "capture" is a different mechanical problem: a posture reaching
77.3% held on a free cube that way still drops it on all 64 sampler environments.""")
parser.add_argument("--hand-velocity-limit", type=float, default=None,
                    help="override velocity_limit_sim (rad/s) on all twenty hand joints, to measure "
                         "how capture depends on closing speed alone")
parser.add_argument("--relative-action", action="store_true",
                    help="""command the hand the way the sampler's action term does: a RELATIVE
action whose PD error is capped at 0.1 rad per control step (decimation 12, so every 12th sim step
here), rather than an absolute joint target. delto_cfg.py:469 records what that cap means -- the
realised torque is bounded by stiffness * 0.1 regardless of how far the joint still has to travel,
so a distal joint at the authored stiffness cannot overcome its own 0.01 N.m friction and freezes.
Without this the bench lets the optimiser spend joint travel the sampler can never actually
deliver: measured, the jaw stalls at 63 mm against a commanded 30.8 mm.""")
parser.add_argument("--step-close", action="store_true",
                    help="""command the closed posture as a STEP rather than ramping to it over
RAMP seconds. scripted_gripper_actions rebuilds the same full-close command every control step, so
the sampler never ramps -- the joints slew at whatever velocity_limit_sim allows. A ramp and a
velocity-limited step close at different speeds and catch a free cube differently.""")
parser.add_argument("--sampler-gains", action="store_true",
                    help="""rebind the hand actuator exactly as DeltoGraspSamplingCfg does --
segment-4 stiffness 1.698/1.694 on rj_dg_1_4 and rj_dg_5_4 (10x authored) and velocity_limit_sim
1.0 rad/s on all twenty joints. The sampler scopes those overrides to itself, so a posture derived
here has always been optimised under DIFFERENT actuator dynamics than the ones it is deployed
under, and the closing speed is precisely what decides whether a free cube is captured or batted
away.""")
parser.add_argument("--close-under-gravity", action="store_true",
                    help="give the cube its weight from the first step, so the posture is scored "
                         "on CAPTURING a free cube rather than on holding a pinned one")
parser.add_argument("--min-align-deg", type=float, default=70.0,
                    help="""hard filter: angle between the pad-to-pad closing axis and the palm-to-jaw
approach direction. The grasp sampler builds every candidate by rotating the approach axis to point
straight down and then lining the CLOSING axis up with an object face, so a posture whose two axes
are not close to perpendicular cannot be placed on a flat-faced part at all -- its fingers arrive
through the face. Holding well is not sufficient: the first posture derived on the un-flipped rig
scored 76%% held and had a 51 degree angle, and the sampler's placement gate rejected all 61440
candidates with a knuckle 2.89 mm inside the cube. The postures derived on the FLIPPED rig came in
at 80 degrees by luck -- gravity pulled the cube into the fingers there, which happens to select
opposed pinches. Upright, nothing selects them unless it is asked for.""")
parser.add_argument("--out", type=str,
                    default="/home/mattephi/Work/Rubitek/cube-stack/pinch/out/pinchhold.json")
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
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR  # noqa: E402
from uwlab_assets.robots.ur10e_delto.ur10e_delto import DELTO_HAND  # noqa: E402

DT = 1.0 / 120.0
G = 9.81
CUBE_MASS = 0.03
CUBE_USD = f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/InsertiveCube34/insertive_cube34.usd"
ALL_JOINTS = [f"rj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]
PAD_BODIES = ("rl_dg_1_tip", "rl_dg_2_tip")
FINGER_CHAINS = [["rl_dg_palm"] + [f"rl_dg_{f}_{j}" for j in range(1, 5)] + [f"rl_dg_{f}_tip"]
                 for f in range(1, 6)]

# Protocol, matching pinchlab.py so the numbers are comparable.
T_CLOSE, RAMP, SETTLE, HOLD, SHAKE = 0.6, 1.0, 0.6, 1.6, 1.0
# The sampler judges at its episode timeout, 18 s in. This bench judged 2.6 s after release, so a
# grasp that slips slowly reads as held here and fails there. --hold-seconds matches the two.
HOLD = float(__import__("os").environ.get("UWLAB_BENCH_HOLD_S", HOLD))
HOLD_RADIUS = 0.020
PINCH_MIN_N = 0.15
PINCH_MAX_OTHER_N = 0.60
REPS = max(1, args.reps)
n_cand = max(2, args.envs // REPS)
NUM_ENVS = n_cand * REPS


def _tip(i: int) -> ContactSensorCfg:
    return ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/rl_dg_%d_tip" % i, history_length=1,
                            track_air_time=False,
                            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"])


def _approach_down_quat() -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) taking the closed posture's approach axis to world -Z."""
    if not args.approach_axes:
        return tuple(DELTO_HAND.init_state.rot)
    import numpy as _np
    a = _np.asarray(json.load(open(args.approach_axes))["closed"]["gripper_approach_direction"], dtype=float)
    a = a / _np.linalg.norm(a)
    b = _np.array([0.0, 0.0, -1.0])
    v = _np.cross(a, b)
    c = float(a @ b)
    if _np.linalg.norm(v) < 1e-9:                     # already parallel or antiparallel
        return (0.0, 0.0, 0.0, 1.0) if c > 0 else (1.0, 0.0, 0.0, 0.0)
    axis = v / _np.linalg.norm(v)
    ang = _np.arccos(_np.clip(c, -1.0, 1.0))
    s_, w_ = _np.sin(ang / 2.0), _np.cos(ang / 2.0)
    return (float(axis[0] * s_), float(axis[1] * s_), float(axis[2] * s_), float(w_))


_HAND_ROT = _approach_down_quat()


def _hand_actuators() -> dict:
    """The sampler's actuator rebind, so the bench closes the hand the way the sampler will."""
    if not args.sampler_gains and args.hand_velocity_limit is None:
        return DELTO_HAND.actuators
    if not args.sampler_gains:
        return {**DELTO_HAND.actuators,
                "hand": DELTO_HAND.actuators["hand"].replace(
                    velocity_limit_sim={r"rj_dg_[1-5]_[1-4]": args.hand_velocity_limit})}
    from uwlab_assets.robots.ur10e_delto.ur10e_delto import DELTO_HAND_ACTUATOR
    st = dict(DELTO_HAND_ACTUATOR.stiffness)
    st["rj_dg_1_4"] = 1.698
    st["rj_dg_5_4"] = 1.694
    v = 1.0 if args.hand_velocity_limit is None else args.hand_velocity_limit
    return {**DELTO_HAND.actuators,
            "hand": DELTO_HAND.actuators["hand"].replace(
                stiffness=st, velocity_limit_sim={r"rj_dg_[1-5]_[1-4]": v})}


_HAND_ACTUATORS = _hand_actuators()


@configclass
class HoldSceneCfg(InteractiveSceneCfg):
    robot = DELTO_HAND.replace(prim_path="{ENV_REGEX_NS}/Robot",
                               init_state=DELTO_HAND.init_state.replace(rot=_HAND_ROT),
                               spawn=DELTO_HAND.spawn.replace(activate_contact_sensors=True),
                               actuators=_HAND_ACTUATORS)
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CUBE_USD, scale=(0.85,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=0,
                disable_gravity=True, max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
            activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)))
    tip1 = _tip(1)
    tip2 = _tip(2)
    tip3 = _tip(3)
    tip4 = _tip(4)
    tip5 = _tip(5)
    ground = AssetBaseCfg(prim_path="/World/GroundPlane",
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.45)),
                          spawn=sim_utils.GroundPlaneCfg())
    dome = AssetBaseCfg(prim_path="/World/DomeLight",
                        spawn=sim_utils.DomeLightCfg(intensity=1200.0))


def _t(x):
    return getattr(x, "torch", x)


def main() -> None:
    seed = json.load(open(args.seed_pose))
    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device or "cuda:0"))
    scene = InteractiveScene(HoldSceneCfg(num_envs=NUM_ENVS, env_spacing=0.75,
                                          replicate_physics=False))
    sim.reset()
    robot, cube = scene["robot"], scene["cube"]
    contacts = [scene[f"tip{i}"] for i in range(1, 6)]
    dev = sim.device
    names = list(robot.data.joint_names)
    idx = {n: i for i, n in enumerate(names)}
    aj = [idx[n] for n in ALL_JOINTS]
    limits = _t(robot.data.joint_pos_limits)[0]
    lo, hi = limits[aj, 0].to(dev), limits[aj, 1].to(dev)

    body_names = list(robot.data.body_names)
    pad_ids = [body_names.index(b) for b in PAD_BODIES]
    palm_id = body_names.index("rl_dg_palm")
    segs = []
    for f, chain in enumerate(FINGER_CHAINS, start=1):
        for a, b in zip(chain[:-1], chain[1:]):
            if a in body_names and b in body_names and not (f in (1, 2) and b.endswith("_tip")):
                segs.append((body_names.index(a), body_names.index(b)))

    q_close0 = torch.tensor([seed["close"]["joints"][n] for n in ALL_JOINTS], device=dev)
    q_open0 = torch.tensor([seed["open"]["joints"][n] for n in ALL_JOINTS], device=dev)
    print(f"POPULATION {n_cand} candidates x {REPS} replicates = {NUM_ENVS} envs")
    print(f"SEED {os.path.basename(args.seed_pose)}: close {seed['close']['aperture_mm']:.2f} mm, "
          f"open {seed['open']['aperture_mm']:.2f} mm")

    mu = q_close0.clone()
    sigma = torch.full_like(mu, args.sigma)
    n_elite = max(4, int(n_cand * args.elite))
    best = (-1.0, mu.clone(), {})

    for gen in range(0 if args.eval_only else args.gens):
        cand = torch.clamp(mu + sigma * torch.randn((n_cand, mu.numel()), device=dev), lo, hi)
        cand[0] = mu  # always carry the incumbent so a generation cannot lose ground
        x = cand.repeat_interleave(REPS, dim=0)
        score, detail = evaluate(sim, scene, robot, cube, contacts, dev, names, idx,
                                 pad_ids, palm_id, segs, q_open0, x)
        # Average each candidate's replicates BEFORE ranking. This is the whole point of --reps:
        # rank postures by their expected behaviour, not by their luckiest env.
        per = score.view(n_cand, REPS)
        cscore = per.mean(dim=1)
        chold = detail["held"].view(n_cand, REPS).float().mean(dim=1)
        cpinch = detail["pinched"].view(n_cand, REPS).float().mean(dim=1)
        order = torch.argsort(cscore, descending=True)
        elite = cand[order[:n_elite]]
        mu = elite.mean(dim=0)
        sigma = elite.std(dim=0).clamp(min=0.008)
        k = int(order[0])
        top = float(cscore[k])
        print(f"  gen {gen:2d}  best {top:.3f} (held {float(chold[k]) * 100:5.1f}%, "
              f"pinch {float(cpinch[k]) * 100:5.1f}%)  pop held {float(chold.mean()) * 100:5.1f}%  "
              f"align {float(detail['align_deg'][k * REPS]):4.1f}deg  "
              f"short {float(detail['shortfall_mm'][k * REPS]):4.1f}mm  "
              f"elite mean {float(cscore[order[:n_elite]].mean()):.3f}  "
              f"sigma {float(sigma.mean()):.4f}", flush=True)
        if top > best[0]:
            e = k * REPS  # a representative env of the winning candidate, for its contact numbers
            best = (top, cand[k].clone(),
                    {"held_rate": round(float(chold[k]), 3),
                     "pinch_rate": round(float(cpinch[k]), 3),
                     "grip_thumb_N": round(float(detail["gt"][e]), 3),
                     "grip_index_N": round(float(detail["gi"][e]), 3),
                     "grip_other_N": round(float(detail["go"][e]), 3),
                     "aperture_mm": round(float(detail["ap"][e]) * 1000, 2),
                     "clearance_mm": round(float(detail["clear"][e]) * 1000, 2),
                     "align_deg": round(float(detail["align_deg"][e]), 2)})

    # Re-evaluate the winner across the whole population to get an honest hold rate rather than
    # the single lucky sample that topped one generation.
    if args.eval_only:
        best = (0.0, torch.tensor([seed["close"]["joints"][n] for n in ALL_JOINTS], device=dev),
                {"held_rate": None, "pinch_rate": None, "aperture_mm": seed["close"].get("aperture_mm")})
    x = best[1].unsqueeze(0).repeat(NUM_ENVS, 1)
    _, detail = evaluate(sim, scene, robot, cube, contacts, dev, names, idx,
                         pad_ids, palm_id, segs, q_open0, x)
    held = float(detail["held"].float().mean())
    pinched = float(detail["pinched"].float().mean())
    print(f"WINNER re-evaluated over {NUM_ENVS} runs: held {held * 100:.1f}%, "
          f"true pinch {pinched * 100:.1f}%, closing axis {float(detail['align_deg'].mean()):.1f} deg "
          f"off the approach")

    payload = {
        "seed_pose": args.seed_pose,
        "held_rate": round(held, 4),
        "pinch_rate": round(pinched, 4),
        "single_best": best[2],
        "close": {"joints": {n: round(float(v), 6) for n, v in zip(ALL_JOINTS, best[1].tolist())},
                  "aperture_mm": best[2].get("aperture_mm")},
        "open": {"joints": {n: round(float(v), 6) for n, v in zip(ALL_JOINTS, q_open0.tolist())},
                 "aperture_mm": seed["open"]["aperture_mm"]},
        "jaw_drift_mm": 0.0,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(payload, open(args.out, "w"), indent=1)
    print(f"WROTE {args.out}")


def evaluate(sim, scene, robot, cube, contacts, dev, names, idx, pad_ids, palm_id, segs,
             q_open0, x: torch.Tensor):
    """Run the full close/hold/shake protocol once, one candidate posture per environment."""
    n = x.shape[0]
    # Carry nothing over from the previous generation: the external force from the last shake is
    # still latched in the sim buffer, and the cube is wherever it ended up. Both would perturb
    # the one step used to calibrate the jaw.
    zero_w = torch.zeros((n, 1, 3), device=dev)
    cube.set_external_force_and_torque(zero_w, zero_w)
    park = torch.zeros((n, 7), device=dev)
    park[:, 2] = -0.30
    park[:, 3] = 1.0
    cube.write_root_pose_to_sim(park)
    cube.write_root_velocity_to_sim(torch.zeros((n, 6), device=dev))

    close_q = torch.zeros((n, len(names)), device=dev)
    open_q = torch.zeros((n, len(names)), device=dev)
    for slot, name in enumerate(ALL_JOINTS):
        close_q[:, idx[name]] = (
            q_open0[slot] if (args.pinch_fingers_only and not name.startswith(("rj_dg_1_", "rj_dg_2_")))
            else x[:, slot]
        )
        open_q[:, idx[name]] = q_open0[slot]

    # Calibrate: where is each candidate's jaw, and does it clear the cube?
    robot.write_joint_state_to_sim(close_q, torch.zeros_like(close_q))
    robot.set_joint_position_target(close_q)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(DT)
    bp = _t(robot.data.body_pos_w)
    p1, p2 = bp[:, pad_ids[0]], bp[:, pad_ids[1]]
    jaw = 0.5 * (p1 + p2)
    aperture = (p1 - p2).norm(dim=1)
    clear = _min_segment_gap(bp, jaw, segs)
    # The two axes the grasp sampler consumes, measured here so the optimiser can be held to them
    # (pinchaxes.py derives the same pair afterwards, from the winner, in the mount frame -- the
    # ANGLE between them is frame-independent, so measuring in world here is equivalent).
    _close_v = torch.nn.functional.normalize(p2 - p1, dim=1)
    _appr_v = torch.nn.functional.normalize(jaw - bp[:, palm_id], dim=1)
    align_deg = torch.rad2deg(torch.arccos(((_close_v * _appr_v).sum(dim=1)).abs().clamp(max=1.0)))

    realised_aperture = aperture.clone()   # overwritten after the close, below
    zero6 = torch.zeros((n, 6), device=dev)
    quat = torch.zeros((n, 4), device=dev)
    quat[:, 3] = 1.0  # identity is (x, y, z, w) under Isaac Lab 3.0; w-first here is a 180 deg X flip
    pin = torch.cat([jaw, quat], dim=1)

    robot.write_joint_state_to_sim(open_q, torch.zeros_like(open_q))
    robot.set_joint_position_target(open_q)
    cube.write_root_pose_to_sim(pin)
    cube.write_root_velocity_to_sim(zero6)
    scene.write_data_to_sim()

    _DECIMATION = 12          # grasp_sampling_cfg.py:293 -- one control step per 12 sim steps
    _MAX_PD_ERROR = 0.1       # rad, the relative action term's per-control-step cap
    _last_target = open_q.clone()

    t_rel = T_CLOSE + RAMP + SETTLE
    total = int((t_rel + HOLD + SHAKE) / DT)
    hold_sum = torch.zeros((n, 5), device=dev)
    hold_n = 0
    wrench = torch.zeros((n, 1, 3), device=dev)
    mass = float(_t(cube.data.default_mass)[0].sum())

    for step in range(total):
        t = step * DT
        if args.step_close:
            alpha = torch.tensor(1.0 if t >= T_CLOSE else 0.0, device=dev)
        else:
            alpha = torch.clamp(torch.tensor((t - T_CLOSE) / RAMP, device=dev), 0.0, 1.0)
        _target = open_q + alpha * (close_q - open_q)
        if args.relative_action:
            # Refresh only on a control step; in between, the previous command stands.
            if step % _DECIMATION == 0:
                _q_now = _t(robot.data.joint_pos)
                _target = _q_now + torch.clamp(_target - _q_now, -_MAX_PD_ERROR, _MAX_PD_ERROR)
                _last_target[:] = _target
            else:
                _target = _last_target
        robot.set_joint_position_target(_target)
        # CAPTURE vs HOLD. Weightless until t_rel, the cube cannot fall out from under a closing
        # hand, so this bench only ever measured whether a posture HOLDS a cube handed to it. The
        # grasp sampler asks the harder question -- close on a free cube and still have it -- and a
        # posture that scores 87.9% held here drops it there on every one of 64 environments
        # (not_far=0, above_ground=0 at the episode timeout, with the pre-grasp at both 50 mm and
        # 82 mm, so it is not a clearance problem). Giving the cube its weight from the first step
        # makes the two benches ask the same question.
        if step == int(t_rel / DT):
            _bp = _t(robot.data.body_pos_w)
            realised_aperture = (_bp[:, pad_ids[0]] - _bp[:, pad_ids[1]]).norm(dim=1)
        if args.close_under_gravity or t >= t_rel:
            wrench[:, 0, 2] = -mass * G
            if t >= t_rel + HOLD:
                ph = 2.0 * math.pi * 3.0 * (t - t_rel - HOLD)
                wrench[:, 0, 0] = mass * 6.0 * math.sin(ph)
                wrench[:, 0, 1] = mass * 6.0 * math.cos(ph)
            # is_global=True, because Isaac Lab defaults it to FALSE -- the LOCAL body frame. Left
            # at the default, this wrench is "gravity" expressed in the cube's own frame, so it
            # rotates with the cube: at an arbitrary grasp orientation it points in an arbitrary
            # world direction, and for many orientations it pushes the part INTO the pinch instead
            # of out of it. That is the whole bench-versus-sampler divergence. Every held rate this
            # bench has ever reported -- 83.6%, 87.9%, 80.9%, 69.9% -- was measured against it.
            cube.set_external_force_and_torque(wrench, torch.zeros_like(wrench), is_global=True)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)
        if t_rel + 0.4 <= t < t_rel + HOLD:
            mags = []
            for c in contacts:
                fm = _t(c.data.force_matrix_w)
                mags.append(fm.reshape(n, -1, 3).norm(dim=-1).sum(dim=-1)
                            if fm is not None else torch.zeros(n, device=dev))
            hold_sum += torch.stack(mags, dim=1)
            hold_n += 1

    grip = hold_sum / max(hold_n, 1)
    gt, gi = grip[:, 0], grip[:, 1]
    go = grip[:, 2:].sum(dim=1)
    slip = (_t(cube.data.root_pos_w) - jaw).norm(dim=1)
    held = slip < HOLD_RADIUS
    pinched = held & (gt > PINCH_MIN_N) & (gi > PINCH_MIN_N) & (go < PINCH_MAX_OTHER_N)

    # Score. A true pinch is the goal; a plain hold is worth something because it is on the way
    # there; and the clearance filter is hard, because a posture the sampler's placement gate
    # rejects is not a posture we can use however well it grips.
    score = (2.0 * pinched.float() + 1.0 * held.float()
             + 0.30 * torch.clamp(torch.minimum(gt, gi), max=1.0)
             - 0.50 * torch.clamp(go - PINCH_MAX_OTHER_N, min=0.0))
    # CLOSURE SHORTFALL. Under the relative action's 0.1 rad/step cap the hand often never reaches
    # the posture it was commanded -- the jaw stalls part-open and the score is a flat zero for
    # every candidate, which gives the search nothing to climb. Paying for the millimetres actually
    # closed turns that flat landscape into a gradient, and it is the quantity the sampler cares
    # about: a posture is only worth what the action term can deliver.
    shortfall = torch.clamp(realised_aperture - aperture, min=0.0)
    score += 0.60 * torch.clamp(1.0 - shortfall / 0.04, min=0.0)
    score += 0.40 * torch.clamp((align_deg - args.min_align_deg) / 20.0, max=1.0)
    score = torch.where(clear >= args.clear_mm / 1000.0, score, torch.full_like(score, -1.0))
    score = torch.where(align_deg >= args.min_align_deg, score, torch.full_like(score, -1.0))
    return score, {"held": held, "pinched": pinched, "gt": gt, "gi": gi, "go": go,
                   "ap": aperture, "clear": clear, "align_deg": align_deg,
                   "shortfall_mm": shortfall * 1000.0}


def _min_segment_gap(bp, jaw, segs) -> torch.Tensor:
    out = []
    for a, b in segs:
        pa, pb = bp[:, a], bp[:, b]
        ab = pb - pa
        t = (((jaw - pa) * ab).sum(dim=1) / (ab * ab).sum(dim=1).clamp(min=1e-12)).clamp(0.0, 1.0)
        out.append((jaw - (pa + t.unsqueeze(1) * ab)).norm(dim=1))
    return torch.stack(out, dim=1).min(dim=1).values


main()
simulation_app.close()
