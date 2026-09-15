# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Render StableState reset states by REPLAYING them in Isaac and photographing the viewport.

WHY THIS EXISTS RATHER THAN render_delto_grasps_raw.py's --reset-states path. That script attaches
its own ``TiledCamera`` sensors. On the UR5e+DELTO leg+fixture scene that path does not produce a
frame on this box: with two 640x480 TiledCameras ``env.reset()`` did not return in 20 minutes; with
one 320x240 TiledCamera the reset returns in 0.6 s and the FIRST CAMERA OUTPUT READ then hangs
instead. Both hangs sit right after the same carb ``IMemoryBudgetManagerFactory`` warning, and in
both the process is genuinely computing (263 percent CPU, ~3.6 GiB of GPU) rather than deadlocked --
active, but not finishing, the same shape as this project's CollisionAnalyzer pathology.

``ManagerBasedRLEnv.render()`` with ``render_mode="rgb_array"`` is a DIFFERENT code path: it lazily
creates ONE ``rep.create.render_product`` on the viewport camera and attaches a single ``rgb``
annotator, instead of instantiating TiledCamera sensors in the scene. Same replicator underneath, so
this is not guaranteed to work -- it is a different thing to try, and it is the documented one.

WHAT IS DELIBERATELY NOT DONE HERE: no pose, framing or lighting is chosen to flatter a grasp. The
settle loop steps REAL physics for --settle-steps and the object's displacement over that window is
measured and written into the frame's own filename, so a state whose grasp is slipping reads as
slipping. A state that drops the leg outright shows a large drift and is labelled DROPPED.

FOUR CONFIGURATION FACTS THIS SCENE NEEDS, each of which cost a debugging round to find:
  1. terminations.success builds a Warp CollisionAnalyzer over the leg+fixture pair that does not
     finish; a render never reads a termination, so it is dropped.
  2. The task defaults to the Peg__PegHole variant pair. Left alone it loads Peg assets and looks
     for Resets/Peg__PegHole/..., and the resulting FileNotFoundError is SWALLOWED inside lazy
     event-term instantiation, resurfacing much later as a bogus "MultiResetManager.__init__() got
     an unexpected keyword argument 'dataset_dir'".
  3. gpu_collision_stack_size is inherited at 3.75 GiB (sized for thousands of envs) and simply
     FAILS to allocate at num_envs=1 on a 16 GB card, after which every GPU collision kernel fails
     to launch.
  4. Every reset-type file for the pair must exist locally -- the event term loads the whole
     mixture, not just the type being rendered -- plus Grasps/<object>/grasps.pt.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--reset-states", type=Path, required=True)
parser.add_argument("--task", default="OmniReset-UR5eDelto-ObjectRestingEEGrasped-v0")
parser.add_argument("--dataset-dir", default="./Datasets_render/OmniReset")
parser.add_argument("--insertive-variant", default="leg200mm")
parser.add_argument("--receptive-variant", default="onelegfixture")
parser.add_argument("--out", type=Path, default=Path("artifacts/reset_viewport"))
parser.add_argument("--count", type=int, default=6)
parser.add_argument("--settle-steps", type=int, default=20)
parser.add_argument("--video", type=Path, default=None,
                    help="write an mp4 of the settle itself, one frame every --video-every physics"
                         " steps, concatenated over the selected states. Stills show where a state"
                         " ENDED; only the video shows whether the hand held the object or the"
                         " object fell out of an open hand, which is the distinction this whole"
                         " exercise turns on.")
parser.add_argument("--allow-no-dataset-dir", action="store_true",
                    help="permit a task whose events take no dataset_dir (ObjectAnywhereEEAnywhere)")
parser.add_argument("--video-freeze", type=int, default=0,
                    help="frames to hold each state AT t=0, re-writing it every frame so nothing "
                         "drifts. This is the reset state as recorded. The settle that follows is "
                         "a different question and is labelled as such in the frame.")
parser.add_argument("--no-hold-arm", action="store_true",
                    help="leave the arm at its shipped zero PD gains (it will free-fall)")
parser.add_argument("--arm-stiffness", type=float, default=8000.0)
parser.add_argument("--arm-damping", type=float, default=400.0)
parser.add_argument("--video-every", type=int, default=4)
parser.add_argument("--video-fps", type=int, default=25)
parser.add_argument("--video-settle-steps", type=int, default=0,
                    help="settle steps to film (0 = --settle-steps). The measurement window is"
                         " deliberately short; a watchable clip usually wants longer.")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--drop-terminations", nargs="*", default=["success"])
parser.add_argument("--repeats", type=int, default=3,
                    help="settle each state N times from the same stored state and report the SPREAD."
                         " PhysX GPU is non-deterministic and the seed is unset, so a single settle"
                         " gave 9.0mm on one run and 25.3mm on the next FOR THE SAME STATE -- the same"
                         " size as the HELD/SLIPPED thresholds. One sample cannot support a verdict.")
parser.add_argument("--no-hold-targets", action="store_true",
                    help="do NOT command the recorded joint posture during settle. write_joint_state_to_sim"
                         " sets joint POSITIONS but leaves the actuators' PD TARGETS at whatever they held"
                         " (the default posture), so the hand is physically closed while being commanded"
                         " toward open and the fingers drive themselves apart. Drift then measures the hand"
                         " letting go, not the grasp failing. On by default; this flag restores the old"
                         " behaviour for comparison.")
parser.add_argument("--light-intensity", type=float, default=1000.0,
                    help="dome light intensity. reset_states_cfg ships 10000 while the TRAINING scene"
                         " (rl_state_cfg) uses 1000; 10x is what washed the frames out. Default matches"
                         " training, so the picture is in-domain as well as legible.")
parser.add_argument("--json-out", type=Path, default=None,
                    help="directory for one JSON sidecar per rendered state (defaults to --out).")
parser.add_argument("--bore-seat", type=float, nargs=3, default=None,
                    help="fixture-LOCAL point to aim the tight camera at -- the bore MOUTH for a"
                         " miss-legibility view, or the fitted seat (render_partial_assemblies.py's"
                         " assembled_offset.pos) for a fully-assembled reference. When given, adds a"
                         " third TIGHT 'bore' view per state, aimed at this fixed point rather than"
                         " at the object, so an off-axis leg reads as off-axis instead of being"
                         " re-centred by the camera.")
parser.add_argument("--engagement-json", type=Path, default=None,
                    help="optional {state_idx: {lateral_mm, depth_mm, tilt_deg}} file of ALREADY-"
                         " MEASURED bore-engagement numbers (not re-derived here) to burn onto the"
                         " frames and copy into each state's JSON sidecar verbatim.")
parser.add_argument("--radial-clearance-mm", type=float, default=0.912,
                    help="bore radius minus leg pilot radius; lateral_mm below this (and depth in"
                         " [0, --engaged-span-mm)) is the only way the tip is actually IN the bore.")
parser.add_argument("--engaged-span-mm", type=float, default=25.0)
parser.add_argument("--assembled-control", action="store_true",
                    help="render ONE extra control frame first: the leg teleported to the pose the"
                         " metadata calls fully assembled, composed through the production Offset"
                         " class (never a hand-rolled inverse). Without this control a picture of a"
                         " leg near a hole is ambiguous between 'the dataset does not seat it',"
                         " 'the fixture has no usable hole', and 'the seat constant is wrong'; with"
                         " it those are distinguishable. Mirrors render_partial_assemblies.py's own"
                         " CONTROL 2. Requires --fixture-assembled-offset / --leg-tip-offset /"
                         " --leg-tip-quat (defaults match OneLegInsertionFixture / leg200mm).")
parser.add_argument("--fixture-assembled-offset", type=float, nargs=3, default=[-0.056250, 0.056250, -0.009374],
                    help="fixture assembled_offset.pos -- the fitted bore SEAT (bottom), fixture-local."
                         " Default is OneLegInsertionFixture/metadata.yaml's own value.")
parser.add_argument("--leg-tip-offset", type=float, nargs=3, default=[-0.106203, 0.0, 0.0],
                    help="leg assembled_offset.pos -- the TIP in the leg's own mesh frame.")
parser.add_argument("--leg-tip-quat", type=float, nargs=4, default=[0.70710678, 0.0, 0.70710678, 0.0],
                    help="leg assembled_offset.quat, (w,x,y,z). Only used by --assembled-control.")
parser.add_argument("--macro-near-miss-idx", type=int, nargs="*", default=[],
                    help="dataset indices (bank order) to render a 5th MACRO view for: top-down"
                         " along the bore axis at --macro-standoff-m, arm parked out of frame. For"
                         " the near-miss states specifically -- a 40mm miss is unambiguous in the"
                         " regular tight view, but at ~1mm against ~0.9mm of clearance the regular"
                         " tight view is legible only because of its burned-in caption, not on its"
                         " own; this view exists to make the gap itself visible. Requires --bore-seat.")
parser.add_argument("--macro-standoff-m", type=float, default=0.05)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
args.enable_cameras = True

import torch  # noqa: E402

# ---- everything that can be decided WITHOUT Isaac is decided before Isaac launches -------------
raw = torch.load(args.reset_states, map_location="cpu")
state = raw["initial_state"]
obj_pose = torch.stack(state["rigid_object"]["insertive_object"]["root_pose"])  # (N, 7)
n_total = obj_pose.shape[0]

# RESTORE THE STORED PD TARGETS, WHEN THE BANK HAS THEM. write_joint_state_to_sim (below) sets q and
# qdot only; it does not touch the actuators' PD targets, which is exactly the defect this re-render
# exists to fix -- a state replayed with the target left at its post-reset default commands the hand
# open regardless of what the state itself recorded. If the bank carries joint_position_target /
# joint_velocity_target (written once the recorder was fixed to capture them), those are commanded
# verbatim; otherwise this falls back to the recorded joint_position q, exactly like before, with a
# one-line notice so a state without them is never silently mistaken for one that has them.
_robot_state = state["articulation"]["robot"]
HAS_STORED_TARGETS = "joint_position_target" in _robot_state and "joint_velocity_target" in _robot_state
if HAS_STORED_TARGETS:
    print("[view] bank carries joint_position_target/joint_velocity_target -- commanding STORED targets", flush=True)
else:
    print("[view] NOTICE: bank has no joint_position_target/joint_velocity_target -- "
          "falling back to commanding the recorded joint_position q as the target", flush=True)

if args.count >= n_total:
    # RENDER ALL, IN BANK ORDER. Farthest-point sampling over a set no larger than the request would
    # still end up including every index -- it would just reorder them -- so there is nothing to
    # explain about selection bias here; skip it and keep the bank's own order legible in the output.
    sel = list(range(n_total))
    print(f"[view] {n_total} states; count >= n_total, rendering ALL in bank order (no selection)", flush=True)
else:
    # Farthest-point sampling over object position+orientation, so the selection spans the dataset
    # instead of showing near-identical poses (evenly spaced indices demonstrably do that here).
    feat = obj_pose[:, :7].clone()
    sel = [int(torch.argmin(obj_pose[:, 2]).item())]  # start from the LOWEST state: the resting end
    for _ in range(args.count - 1):
        d = torch.cdist(feat, feat[sel]).min(dim=1).values
        d[torch.tensor(sel)] = -1.0
        sel.append(int(torch.argmax(d).item()))
    print(f"[view] {n_total} states; selected {sel}", flush=True)
heights = obj_pose[sel, 2]
print(f"[view] selected heights (m): {[round(float(h), 4) for h in heights]}", flush=True)

# ALREADY-MEASURED bore-engagement numbers (lateral/depth/tilt), keyed by state idx as a string.
# NOT re-derived here -- see --engagement-json help. Burned onto frames and copied into the sidecar.
ENGAGEMENT: dict[str, dict] = {}
if args.engagement_json is not None:
    import json as _json
    ENGAGEMENT = _json.loads(args.engagement_json.read_text())
    print(f"[view] loaded {len(ENGAGEMENT)} pre-measured engagement entries from {args.engagement_json}", flush=True)

target = obj_pose[sel, :3].mean(dim=0).tolist()
eye = [target[0] - 0.55, target[1] + 0.45, target[2] + 0.40]
# The default standoff was chosen for the leg+fixture scene. A 34 mm cube at 0.81 m fills a few
# dozen pixels, and the frames come back showing bare tabletop -- numbers without a picture, which
# is how the last round of this went wrong. Allow an explicit override.
if os.environ.get("UWLAB_VIEW_EYE"):
    eye = [float(v) for v in os.environ["UWLAB_VIEW_EYE"].split(",")]
if os.environ.get("UWLAB_VIEW_AT"):
    target = [float(v) for v in os.environ["UWLAB_VIEW_AT"].split(",")]
print(f"[view] viewer eye={[round(v,3) for v in eye]} lookat={[round(v,3) for v in target]}", flush=True)

t0 = time.time()
app = AppLauncher(args).app
print(f"[view] app up {time.time()-t0:.1f}s", flush=True)

# Isaac Sim 6 shim: uwlab.envs.real_rl_env imports ``isaacsim.core.utils``, which does not exist
# in Isaac Sim 6, and uwlab_tasks' package walk imports it transitively. Must run after the app
# starts and before any uwlab import -- the same two lines record_grasps.py and
# record_reset_states.py already carry. Without it this script cannot start under Isaac Sim 6,
# which is why a hand-written replay scene got used in its place and quietly drifted from the
# real one.
import uwlab_is6_bridge  # noqa: E402

uwlab_is6_bridge.apply()

import re  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import numpy as np  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
import uwlab_tasks.manager_based.manipulation.omnireset.mdp.scripted_gripper as _scripted_gripper  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from PIL import Image  # noqa: E402

# REUSED, NOT REIMPLEMENTED: dexlift's own fingertip-force reducer and the exact gate
# held_with_probe uses (mdp/held_check.py's defaults), so a verdict computed here means the same
# thing a verdict computed by the training-side termination would mean.
from uwlab_tasks.manager_based.manipulation.dexlift.mdp.rewards import _sensor_force_magnitudes  # noqa: E402

FORCE_THRESHOLD = 0.2  # N, matches held_with_probe's default force_threshold
# GRIPPER PROFILE, chosen from the task id. "Opposed" means at least one body from EACH side of
# the grasp carries load. The DELTO sides are thumb-side tips vs finger tips; the linear gripper's are
# its two jaws. The linear gripper's bodies sit under gripper/Geometry/ because its USD comes from the
# Isaac Sim 6 importer; the DELTO USD predates it and keeps the flat gripper/<body> layout.
if "LinearGripper" in args.task:
    THUMB_TIP_NAMES = ("left_inner_finger",)
    TIP_NAMES = ("right_inner_finger",)
    TIP_PRIM_PREFIX = "{ENV_REGEX_NS}/Robot/gripper/Geometry"
    PALM_BODY = "robotiq_base_link"
else:
    THUMB_TIP_NAMES = ("rl_dg_1_tip", "rl_dg_5_tip")
    TIP_NAMES = ("rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip")
    TIP_PRIM_PREFIX = "{ENV_REGEX_NS}/Robot/gripper"
    PALM_BODY = "rl_dg_mount"
ALL_TIP_NAMES = THUMB_TIP_NAMES + TIP_NAMES

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)

for term in args.drop_terminations:
    if getattr(cfg.terminations, term, None) is not None:
        setattr(cfg.terminations, term, None)
        print(f"[view] dropped termination '{term}'", flush=True)
# Reset-state configs run 2.0 s episodes (20 control steps). Any settle longer than that hits
# time_out INSIDE env.step, the env auto-resets, and the replay "drops" the cube onto its default
# pose -- measured: jaws holding at 24 N and the cube still for 19 steps, then everything at
# defaults on step 20. A replay has no episode, so make it effectively unbounded.
cfg.episode_length_s = 1.0e6
print("[view] episode_length_s -> 1e6 (no time_out auto-reset during settle)", flush=True)

cfg.scene.insertive_object = cfg.variants["scene.insertive_object"][args.insertive_variant]
cfg.scene.receptive_object = cfg.variants["scene.receptive_object"][args.receptive_variant]
print(f"[view] variants: {args.insertive_variant} / {args.receptive_variant}", flush=True)

# ADD FINGERTIP CONTACT SENSORS. Verified empirically (probe of this exact task/variant pair) that
# the OmniReset scene registers ZERO ContactSensors -- env.scene.sensors is {} -- unlike dexlift,
# whose scene wires one `{tip}_object_s` ContactSensorCfg per fingertip so `_sensor_force_magnitudes`
# has something to read. This is not a scene rewrite: it is the SAME per-tip ContactSensorCfg pattern
# dexlift's own env cfg uses (prim_path "{ENV_REGEX_NS}/Robot/gripper/{tip}", filtered to the
# manipulated object), added here at render time so the reused reducer works unmodified. The robot
# prim path ("{ENV_REGEX_NS}/Robot/gripper/...") and the object prim path
# ("{ENV_REGEX_NS}/InsertiveObject") were both confirmed against the live USD stage for this task
# before wiring this in, not assumed by analogy with dexlift's own (differently-named) "Object".
for _tip in ALL_TIP_NAMES:
    setattr(
        cfg.scene,
        f"{_tip}_object_s",
        ContactSensorCfg(
            prim_path=f"{TIP_PRIM_PREFIX}/{_tip}",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/InsertiveObject"],
        ),
    )
print(f"[view] added {len(ALL_TIP_NAMES)} fingertip ContactSensors (scene shipped none)", flush=True)

# CONTACT REPORTER API. A ContactSensor needs PhysxContactReportAPI on the prims it reads, which
# the shared ur5e_delto asset ships OFF by default (ur5e_delto.py:61) -- dexlift only gets working
# fingertip sensors because its own env cfg flips this on for both the robot AND the manipulated
# object (dexlift_ur5e_delto_env_cfg.py ~1150, table_leg_env_cfg.py:227). Mirrored here for the
# same two bodies; without it every sensor added above raises "could not find any bodies with
# contact reporter API" at gym.make (confirmed empirically).
cfg.scene.robot.spawn = cfg.scene.robot.spawn.replace(activate_contact_sensors=True)
cfg.scene.insertive_object.spawn = cfg.scene.insertive_object.spawn.replace(activate_contact_sensors=True)
print("[view] activated contact-report API on robot + insertive_object spawns", flush=True)

patched = 0
for name, term in vars(cfg.events).items():
    if term is not None and getattr(term, "params", None) and "dataset_dir" in term.params:
        term.params["dataset_dir"] = args.dataset_dir
        patched += 1
if patched == 0 and not args.allow_no_dataset_dir:
    raise SystemExit("[view] REFUSING: no event term takes dataset_dir; the override would be a silent no-op")
if patched == 0:
    # ObjectAnywhereEEAnywhere reads no grasp or reset bank at all, so it legitimately has no
    # dataset_dir-bearing event term. The refusal above is right for every OTHER family, where a
    # silently ignored --dataset-dir would render one bank while claiming another.
    print("[view] no dataset_dir event term (expected for ObjectAnywhereEEAnywhere)", flush=True)
print(f"[view] dataset_dir -> {args.dataset_dir} ({patched} term(s))", flush=True)

cfg.scene.num_envs = 1
cfg.sim.render_interval = 1
# Isaac Sim 6 renamed this block: SimulationCfg.physx -> SimulationCfg.physics. This repo's own
# rl_state_cfg.py:893 already writes the new name; only the older tool scripts still say physx.
# Resolve it rather than assuming, so this keeps working on either.
_physics_cfg = getattr(cfg.sim, "physics", None) or cfg.sim.physx
_physics_cfg.gpu_collision_stack_size = 256 * 1024 * 1024
# IN-DOMAIN LIGHTING. reset_states_cfg's dome light is 10000.0; the training scene the states are
# consumed in (rl_state_cfg) uses 1000.0. Rendering at 10x training brightness is both unreadable
# and not what the policy ever sees.
if getattr(cfg.scene, "sky_light", None) is not None and hasattr(cfg.scene.sky_light.spawn, "intensity"):
    was = cfg.scene.sky_light.spawn.intensity
    cfg.scene.sky_light.spawn.intensity = args.light_intensity
    print(f"[view] dome light {was} -> {args.light_intensity} (training scene value)", flush=True)
else:
    print("[view] WARNING: no scene.sky_light to dim; frames may stay overexposed", flush=True)

# ViewerCfg.origin_type decides what eye/lookat are RELATIVE to: "world", "env", "asset_root" or
# "asset_body". Anything but "world" silently offsets every camera position, which is exactly the
# shape of the framing failures here -- each attempt landed a fixed distance off, under the table.
print(f"[view] viewer origin_type={cfg.viewer.origin_type} asset_name={cfg.viewer.asset_name} "
      f"env_index={cfg.viewer.env_index}", flush=True)
cfg.viewer.origin_type = "world"
cfg.viewer.asset_name = None
cfg.viewer.eye = tuple(eye)
cfg.viewer.lookat = tuple(target)
cfg.viewer.resolution = (args.width, args.height)
cfg.viewer.origin_type = "world"

t = time.time()
# DROP THE ARM ACTION TERM.
#
# Measured, replaying a written bank state three ways:
#   raw sim.step            arm holds (0.22 rad) but the HAND explodes (1.7-4.7 rad) -- nothing
#                           drives the fingers, whose scripted term lives in the action manager.
#   env.step(zero action)   hand holds beautifully (0.09 rad) but the ARM swings 2.8-3.2 rad --
#                           the IK term recomputes a target from a pose that is stale immediately
#                           after a teleport and overwrites the recorded joint targets.
# Neither is usable. Removing the arm's term gives both: the hand keeps its action term, and the
# arm keeps the joint position targets write_state() set, held by the PD gains the actuator now
# has. The arm is scenery in a reset-state gallery; the hand and the object are the subject.
if not args.no_hold_arm:
    for _an, _at in list(vars(cfg.actions).items()):
        if _at is None:
            continue
        _cn = type(_at).__name__
        if "InverseKinematics" in _cn or "Cartesian" in _cn or "OperationalSpace" in _cn:
            setattr(cfg.actions, _an, None)
            print(f"[view] dropped arm action term '{_an}' ({_cn}) -- see comment", flush=True)

env = gym.make(args.task, cfg=cfg, render_mode="rgb_array").unwrapped
# See settle_step(): the arm carries zero PD gains and is driven entirely by its OSC action term,
# so the replay has to go through env.step() with an action rather than through env.sim.step().
ZERO_ACTION = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)

# HOLD THE ARM. The arm is ImplicitActuatorCfg(stiffness=0, damping=0) -- deliberately, because
# the task drives it with an operational-space controller that applies EFFORTS. A replay has no
# policy, so nothing supplies those efforts and the arm free-falls: measured from a written bank
# state, the pose is exact at t=0 and 0.22-0.61 rad off five physics steps later, with the cube on
# the floor by step 80. Everything read out of such a replay -- "the hand is not holding the cube"
# included -- is a property of the falling arm, not of the bank.
#
# For a reset-state gallery the arm is scenery: the question is whether the HAND holds the object.
# So the arm is pinned with a stiff PD for replay only. This is not what training does, and it is
# stated in the frame captions rather than left implicit. Effort limits still apply, so it cannot
# produce torques the real arm could not.
# TABLE PLACEMENT. Measured world bounds of /World/envs/env_0/Table are z[-0.004, +0.676], i.e.
# the asset's origin is at its FEET and its top face is 676 mm ABOVE the origin. The scene configs
# spawn it at (0,0,0), which lifts the whole table 676 mm: the work surface ends up at +0.676, the
# cubes (stored at z = 0.021) sit INSIDE the table volume, and the robot -- whose own bounds are
# z[-0.275, 0.000] -- hangs underneath it. That is the "cubes inside the table, arm under the
# table" every render of this scene has shown.
#
# The intended geometry is stated by the configs themselves: work surface at +0.004, robot base
# bolted to it at z = 0, and a 34 mm cube resting on it at 0.004 + 0.017 = 0.021 -- exactly the z
# the banks store. Dropping the table so its TOP lands at +0.004 satisfies all three at once.
TABLE_DZ = float(os.environ.get("UWLAB_TABLE_Z", "0") or 0)
if TABLE_DZ:
    print(f"[view] table + support will be shifted {TABLE_DZ:+.3f} m in z after every state write",
          flush=True)

_arm_ids, _arm_names = env.scene["robot"].find_joints(["shoulder.*", "elbow.*", "wrist.*"])
if not args.no_hold_arm:
    _n_env, _n_j = env.num_envs, len(_arm_ids)
    env.scene["robot"].write_joint_stiffness_to_sim(
        torch.full((_n_env, _n_j), args.arm_stiffness, device=env.device), joint_ids=_arm_ids)
    env.scene["robot"].write_joint_damping_to_sim(
        torch.full((_n_env, _n_j), args.arm_damping, device=env.device), joint_ids=_arm_ids)
    print(f"[view] arm PD pinned for replay: {len(_arm_ids)} joints, "
          f"stiffness={args.arm_stiffness}, damping={args.arm_damping} "
          f"(shipped value is 0/0 -- the arm is OSC effort-driven and a replay has no policy)",
          flush=True)
print(f"[view] action dim {env.action_manager.total_action_dim}; stepping through the action "
      f"pipeline (the arm has zero PD gains and is OSC-driven)", flush=True)
print(f"[view] gym.make {time.time()-t:.1f}s", flush=True)
t = time.time()
env.reset()
print(f"[view] first reset {time.time()-t:.1f}s", flush=True)

# The FIRST render() is where the render product and annotator are created -- the step that has
# hung on this scene via the TiledCamera path. Timed and announced on its own so a hang here is
# unambiguous rather than looking like a hung settle loop.
t = time.time()
print("[view] first render(): creating render product + rgb annotator ...", flush=True)
frame = env.render()
print(f"[view] first render() {time.time()-t:.1f}s -> {None if frame is None else frame.shape}", flush=True)
if frame is None:
    raise SystemExit("[view] REFUSING: render() returned None -- render_mode is not rgb_array")

args.out.mkdir(parents=True, exist_ok=True)
scene_obj = env.scene["insertive_object"]
robot = env.scene["robot"]
rows = []

def _quat_apply_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(w,x,y,z) scalar-first, same helper and convention as render_partial_assemblies.py."""
    u = q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + q[..., 0:1] * t + torch.cross(u, t, dim=-1)

fixture = env.scene["receptive_object"] if "receptive_object" in env.scene.rigid_objects else None
fx_pos = fx_quat = None
if fixture is not None:
    fx_pos = fixture.data.root_pos_w[0:1].clone()
    fx_quat = fixture.data.root_quat_w[0:1].clone()

if os.environ.get("UWLAB_SCENE_DEBUG"):
    # Where things actually are in WORLD coordinates. The camera is placed with origin_type
    # "world", so aiming it from recorded root poses (which are env-local) points it at the wrong
    # place the moment the scene has a non-zero env origin or a table whose prim sits elsewhere.
    print("[scene] env_origins[0] =", [round(v, 4) for v in env.scene.env_origins[0].tolist()], flush=True)
    _rb = env.scene["robot"]
    _bp = _rb.data.body_pos_w
    _bp = getattr(_bp, "torch", _bp)
    for _i, _bn in enumerate(_rb.body_names):
        print(f"[scene] body {_bn} = {[round(v, 4) for v in _bp[0, _i].tolist()]}", flush=True)
    for _n in ("robot",):
        _a = env.scene[_n]
        print(f"[scene] {_n}.root_pos_w = {[round(v, 4) for v in _a.data.root_pos_w[0].tolist()]}"
              f"  quat = {[round(v, 4) for v in _a.data.root_quat_w[0].tolist()]}", flush=True)
    for _n in env.scene.rigid_objects.keys():
        _a = env.scene[_n]
        print(f"[scene] {_n}.root_pos_w = {[round(v, 4) for v in _a.data.root_pos_w[0].tolist()]}"
              f"  quat = {[round(v, 4) for v in _a.data.root_quat_w[0].tolist()]}", flush=True)
    try:
        import isaacsim.core.utils.bounds as _bu
        _cache = _bu.create_bbox_cache()
        for _path in ("/World/envs/env_0/Table", "/World/envs/env_0/table",
                      "/World/envs/env_0/Robot", "/World/envs/env_0/InsertiveObject"):
            try:
                _r = _bu.compute_aabb(_cache, prim_path=_path)
                print(f"[scene] aabb {_path} = {[round(float(v), 4) for v in _r]}", flush=True)
            except Exception as _e:
                print(f"[scene] aabb {_path} -> {_e}", flush=True)
    except Exception as _e:
        print("[scene] bbox cache unavailable:", _e, flush=True)
    from pxr import Usd as _Usd, UsdGeom as _UsdGeom
    _stage = env.sim.stage if hasattr(env.sim, "stage") else None
    if _stage is not None:
        for _path in ("/World/envs/env_0", "/World/envs/env_0/Table",
                      "/World/envs/env_0/Table/visuals", "/World/envs/env_0/Robot",
                      "/World/envs/env_0/UR5MetalSupport", "/World/GroundPlane"):
            _pr = _stage.GetPrimAtPath(_path)
            if not _pr:
                print(f"[scene] xform {_path} -> missing", flush=True)
                continue
            _x = _UsdGeom.Xformable(_pr)
            _m = _x.ComputeLocalToWorldTransform(_Usd.TimeCode.Default())
            _t = _m.ExtractTranslation()
            _r = _m.ExtractRotationQuat()
            print(f"[scene] xform {_path} t=({_t[0]:.4f},{_t[1]:.4f},{_t[2]:.4f}) "
                  f"q=({_r.GetReal():.4f},{_r.GetImaginary()[0]:.4f},{_r.GetImaginary()[1]:.4f},"
                  f"{_r.GetImaginary()[2]:.4f})", flush=True)
    if _stage is not None:
        for _pr in _Usd.PrimRange(_stage.GetPrimAtPath("/World/envs/env_0")):
            _d = str(_pr.GetPath()).count("/")
            if _d <= 5:
                print("[scene] prim", _pr.GetPath(), _pr.GetTypeName(), flush=True)

SEAT_W = None
if args.bore_seat is not None:
    assert fixture is not None, "[view] REFUSING: --bore-seat given but no receptive_object in scene"
    seat_local = torch.tensor(args.bore_seat, device=env.device).unsqueeze(0)
    SEAT_W = (fx_pos + _quat_apply_wxyz(fx_quat, seat_local))[0].tolist()
    print(f"[view] bore seat (fixture-local {args.bore_seat}) -> world {[round(v,4) for v in SEAT_W]}", flush=True)

try:
    from PIL import ImageDraw, ImageFont
    _FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
except Exception as _e:
    _FONT = None
    print(f"[view] WARNING: no truetype font for burn-in text ({_e}); frames will have no overlay", flush=True)


def _burn_text(img: np.ndarray, lines: list[str]) -> np.ndarray:
    """Burn text into the TOP-LEFT of a frame over a translucent bar, so a card viewer cannot miss
    the numbers even if the downstream card-builder drops the JSON sidecar on the floor."""
    if _FONT is None or not lines:
        return img
    pil = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(pil, "RGBA")
    pad = 6
    line_h = 26
    w = max(draw.textlength(line, font=_FONT) for line in lines) + 2 * pad
    h = line_h * len(lines) + 2 * pad
    draw.rectangle([0, 0, w, h], fill=(0, 0, 0, 170))
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * line_h), line, font=_FONT, fill=(255, 255, 80, 255))
    return np.asarray(pil)
HOLD_Q = None


def settle_step() -> None:
    """Advance one control step THROUGH THE ACTION PIPELINE, not through raw physics.

    This robot's arm actuator ships with stiffness = 0 and damping = 0 (measured: all six joints).
    That is deliberate -- the arm is torque-controlled by an OSC action term, which computes and
    applies the joint efforts every control step, gravity compensation included. Stepping
    ``env.sim`` directly bypasses the action manager, so nothing drives the arm at all and it free
    falls. Measured with a bank state written: the pose is exact at t=0, then within FIVE physics
    steps the arm is 0.22-0.61 rad off, the hand is 1.7-3.0 rad off, and by step 80 the cube has
    fallen to floor level. Every "the hand is not holding the cube" reading taken that way is an
    artefact of the replay, not a property of the bank.

    A zero action is the right one to send: both action terms in this task are RELATIVE, so zero
    means "hold the current target" rather than "go limp" or "go to some default".
    """
    # Step through the action manager either way. With the arm's OSC term dropped above, the
    # only term left is the hand's, which is exactly what has to run: the fingers are servoed by
    # it, and the arm keeps the joint targets write_state() wrote, held by the actuator's PD.
    # UWLAB_VIEW_SCRIPTED_HOLD=1: hold with the recorder's own scripted close instead of zeros. Needed
    # when the hand action is ABSOLUTE (the DELTO pinch recording configs), where zero means 'drive
    # every hand joint to 0 rad' rather than 'hold', and it matches how the bank was recorded.
    if os.environ.get("UWLAB_VIEW_SCRIPTED_HOLD"):
        env.step(_scripted_gripper.scripted_gripper_actions(env, close=True))
    else:
        env.step(ZERO_ACTION)
    if os.environ.get("UWLAB_VIEW_TRACE"):
        # One line per control step: jaw q / target / applied torque and the cube's height and
        # speed, to see whether a dropped grasp is a jaw that opens or a cube that slides.
        _t = lambda x: getattr(x, "torch", x)
        _d = robot.data
        _hand = [i for i, n in enumerate(robot.joint_names) if i not in list(_arm_ids)]
        _cube = env.scene["insertive_object"].data
        _p = _t(_cube.root_pos_w)[0] - env.scene.env_origins[0]
        print("[view trace] q=%s tgt=%s tau=%s cube_z=%.4f cube_v=%.3f ee_z=%.4f" % (
            [round(float(v), 4) for v in _t(_d.joint_pos)[0, _hand]],
            [round(float(v), 4) for v in _t(_d.joint_pos_target)[0, _hand]],
            [round(float(v), 2) for v in _t(_d.applied_torque)[0, _hand]],
            float(_p[2]), float(_t(_cube.root_lin_vel_w)[0].norm()),
            float(_t(_d.body_pos_w)[0, -1, 2] - env.scene.env_origins[0, 2])), flush=True)
    if not args.no_hold_arm and HOLD_Q is not None:
        # Pin the arm KINEMATICALLY at the recorded pose. PD targets alone lose: the hand's
        # scripted term writes a command spanning all 26 joints, so whatever the arm target is set
        # to gets overwritten before physics runs -- measured, the arm still drifted 2.1-2.3 rad
        # with the OSC term already removed and stiffness at 2000. Writing the arm's joint STATE
        # back each control step makes it an anchor, which is what it should be here: the question
        # this gallery answers is whether the HAND holds the object, and an arm wandering off
        # under no policy only obscures it. Stated in the caption, not hidden.
        robot.write_joint_state_to_sim(
            HOLD_Q[:, _arm_ids], torch.zeros_like(HOLD_Q[:, _arm_ids]), joint_ids=_arm_ids)
        robot.set_joint_position_target(HOLD_Q[:, _arm_ids], joint_ids=_arm_ids)


# UWLAB_DEFAULT_POSE ignores the bank's robot state and shows the scene AS CONFIGURED -- the
# default joint posture at the default root pose. Banks recorded before the (x,y,z,w) quaternion
# fix hold joint angles that only made sense with the arm mounted upside down; replaying those
# says nothing about whether the scene itself is right. This does.
DEFAULT_POSE = bool(os.environ.get("UWLAB_DEFAULT_POSE"))


def write_state(idx: int) -> None:
    """Write one stored state verbatim: robot root + all 26 joints + every rigid object pose."""
    if DEFAULT_POSE:
        robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7].clone())
        robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.default_root_state[:, 7:13]))
        _dq = robot.data.default_joint_pos.clone()
        _dq = getattr(_dq, "torch", _dq)
        robot.write_joint_state_to_sim(_dq, torch.zeros_like(_dq))
        robot.set_joint_position_target(_dq)
        for key in state["rigid_object"]:
            if key in env.scene.rigid_objects:
                _a = env.scene[key]
                _ds = _a.data.default_root_state.clone()
                _ds = getattr(_ds, "torch", _ds)
                _a.write_root_pose_to_sim(_ds[:, :7])
                _a.write_root_velocity_to_sim(torch.zeros_like(_ds[:, 7:13]))
        return
    robot.write_root_pose_to_sim(state["articulation"]["robot"]["root_pose"][idx].unsqueeze(0).to(env.device))
    robot.write_root_velocity_to_sim(state["articulation"]["robot"]["root_velocity"][idx].unsqueeze(0).to(env.device))
    robot.write_joint_state_to_sim(
        state["articulation"]["robot"]["joint_position"][idx].unsqueeze(0).to(env.device),
        state["articulation"]["robot"]["joint_velocity"][idx].unsqueeze(0).to(env.device),
    )
    for key in state["rigid_object"]:
        if key in env.scene.rigid_objects:
            asset = env.scene[key]
            asset.write_root_pose_to_sim(state["rigid_object"][key]["root_pose"][idx].unsqueeze(0).to(env.device))
            asset.write_root_velocity_to_sim(state["rigid_object"][key]["root_velocity"][idx].unsqueeze(0).to(env.device))
    # The table's own pose is stored in the bank, so it is re-written above on every state and any
    # correction applied once at startup is undone here. Re-apply after the bank write, not before.
    if TABLE_DZ:
        for _nm in ("table", "ur5_metal_support"):
            if _nm in env.scene.rigid_objects:
                _a = env.scene[_nm]
                _pq = _a.data.root_state_w[:, :7].clone()
                _pq[:, 2] += TABLE_DZ
                _a.write_root_pose_to_sim(_pq)
                _a.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))

    # COMMAND THE POSTURE, DO NOT ONLY PLACE IT.
    #
    # write_joint_state_to_sim writes q and qdot. It does NOT touch the actuators' position
    # targets, which keep whatever they last held -- for a fresh env, the default posture, whose
    # hand is open. PhysX then runs a PD controller driving the fingers from the recorded grasp
    # toward that default, so the hand actively releases the leg and the measured "drift" is the
    # release, not a property of the reset state. Setting the target to the recorded q makes the
    # controller HOLD the grasp, which is what the stored state actually represents.
    q = state["articulation"]["robot"]["joint_position"][idx].unsqueeze(0).to(env.device)
    global HOLD_Q
    HOLD_Q = q
    if not args.no_hold_targets:
        if HAS_STORED_TARGETS:
            pt = state["articulation"]["robot"]["joint_position_target"][idx].unsqueeze(0).to(env.device)
            vt = state["articulation"]["robot"]["joint_velocity_target"][idx].unsqueeze(0).to(env.device)
            robot.set_joint_position_target(pt)
            robot.set_joint_velocity_target(vt)
        else:
            robot.set_joint_position_target(q)
    if not args.no_hold_arm:
        # The OSC term overwrites arm targets every control step, so the arm target has to be
        # (re)asserted here AND after each action step -- see settle_step.
        robot.set_joint_position_target(q[:, _arm_ids], joint_ids=_arm_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()


# WHICH HAND IS ACTUALLY REPLAYING THESE GRASPS?
#
# The states were GENERATED in the dexlift scene with DEXLIFT_REF_HAND_ACT=1 -- the REFERENCE
# DELTO actuator, effort 30 N*m and velocity 10000 rad/s. The OmniReset scene builds the shared
# _DELTO_HAND_ACTUATOR instead: effort 0.06-0.17 N*m, velocity 3.0 rad/s, i.e. 200-500x weaker.
# That seam is not a rendering detail -- it is the difference between the environment that made
# these states and the environment that will train on them, so it is printed rather than assumed.
# Matched by hand rather than with find_joints, which raises when nothing matches -- and on the
# linear gripper nothing does, which is the case the warning below is for.
_hand_ids = [i for i, n in enumerate(robot.joint_names) if re.fullmatch(r"rj_dg_[1-5]_[1-4]", n)]
if len(_hand_ids):
    _eff = robot.data.joint_effort_limits[0, _hand_ids]
    _vel = robot.data.joint_velocity_limits[0, _hand_ids]
    print(f"[view] hand actuators AS BUILT: effort {_eff.min():.3f}-{_eff.max():.3f} N*m, "
          f"velocity {_vel.min():.1f}-{_vel.max():.1f} rad/s over {len(_hand_ids)} joints", flush=True)
    print("[view]   (states were generated under the REFERENCE hand: 30.0 N*m, 10000 rad/s)", flush=True)
else:
    print("[view] WARNING: no rj_dg_* hand joints found; cannot report actuator strength", flush=True)


def target_gap_rad() -> float:
    """Max |PD target - actual joint position| in rad, to prove the targets were really applied."""
    return float((robot.data.joint_pos_target - robot.data.joint_pos).abs().max().item())


# PALM BODY + TABLE DATUM, for the reference in-palm offset and the airborne/supported height.
_palm_ids, _ = robot.find_bodies([PALM_BODY])
assert len(_palm_ids) == 1, f"expected exactly one {PALM_BODY} body, got {_palm_ids}"
PALM_ID = _palm_ids[0]
_table_support = env.scene["ur5_metal_support"]

args.json_out = args.json_out or args.out
args.json_out.mkdir(parents=True, exist_ok=True)

import json  # noqa: E402

# CONTROL: FULLY ASSEMBLED, through the production Offset class, never a hand-rolled inverse. This
# is what makes the twelve dataset states readable -- a photo of a leg near a hole is ambiguous
# between "the dataset does not seat it", "the fixture has no usable hole" and "the seat constant is
# wrong"; this control collapses that ambiguity by showing the geometry DOES seat cleanly when
# composed correctly. Mirrors render_partial_assemblies.py's own CONTROL 2 verbatim (same library
# calls, same constants), rendered here so it sits on the same page as the twelve.
if args.assembled_control:
    assert fixture is not None, "[view] REFUSING: --assembled-control needs receptive_object in scene"
    from uwlab_tasks.manager_based.manipulation.omnireset.assembly_keypoints import Offset  # noqa: E402

    # PARK THE ARM. The subject of this control is the leg/fixture geometry alone; the robot is
    # still wherever env.reset() put it (an EE-anywhere-shaped pose here, since no write_state() has
    # run yet), and it lands squarely between the camera and the bore on this rig -- confirmed by a
    # first attempt at this frame, which rendered the hand fully occluding the leg. Fold to the
    # default posture and re-command the target so it does not un-fold before the shot (same fix
    # render_partial_assemblies.py's own --hide-robot applies for the identical reason).
    _q0 = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(_q0, torch.zeros_like(_q0))
    robot.set_joint_position_target(_q0)
    env.scene.write_data_to_sim()
    env.sim.forward()
    print("[view] parked the arm at its default posture for the control frame", flush=True)

    LEG_TIP = torch.tensor(args.leg_tip_offset, device=env.device)
    fixture_off = Offset(pos=tuple(args.fixture_assembled_offset), quat=(1.0, 0.0, 0.0, 0.0))
    leg_off = Offset(pos=tuple(args.leg_tip_offset), quat=tuple(args.leg_tip_quat))
    tgt_pos, tgt_quat = fixture_off.combine(fx_pos, fx_quat)
    asm_pos, asm_quat = leg_off.subtract(tgt_pos, tgt_quat)
    scene_obj.write_root_pose_to_sim(torch.cat([asm_pos, asm_quat], dim=-1))
    scene_obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
    env.scene.write_data_to_sim()
    env.sim.forward()

    # Re-derive from what the simulator now HOLDS, not from the pre-Isaac arithmetic -- if the two
    # disagree, the composition is wrong and the picture would be lying.
    tip_w = scene_obj.data.root_pos_w[0:1] + _quat_apply_wxyz(scene_obj.data.root_quat_w[0:1], LEG_TIP.unsqueeze(0))
    fseat_local = torch.tensor(args.fixture_assembled_offset, device=env.device).unsqueeze(0)
    fseat_w = fx_pos + _quat_apply_wxyz(fx_quat, fseat_local)
    c_dxy_mm = float(torch.linalg.norm((tip_w - fseat_w)[0, :2]).item()) * 1000.0
    c_dz_mm = float((tip_w - fseat_w)[0, 2].item()) * 1000.0  # height above the SEAT (0 = fully seated)
    c_axis = _quat_apply_wxyz(scene_obj.data.root_quat_w[0:1], torch.tensor([[-1.0, 0.0, 0.0]], device=env.device))
    c_tilt = float(torch.rad2deg(torch.arccos((-c_axis[0, 2]).clamp(-1.0, 1.0))).item())
    # depth-below-MOUTH convention, to match the twelve dataset cards: mouth is engaged_span_mm ABOVE
    # the seat, so a leg exactly at the seat (c_dz_mm == 0) sits engaged_span_mm below the mouth.
    c_depth_below_mouth_mm = args.engaged_span_mm - c_dz_mm

    obj_now = scene_obj.data.root_pos_w[0].tolist()
    control_views = {"front": ([1.0770121, -0.1679045, 0.4486344], obj_now)}
    if SEAT_W is not None:
        control_views["bore"] = ([SEAT_W[0] + 0.20, SEAT_W[1] - 0.17, SEAT_W[2] + 0.09], list(SEAT_W))
    burn = [
        "CONTROL: FULLY ASSEMBLED (production Offset composition)",
        f"lateral {c_dxy_mm:.4f} mm  above-seat {c_dz_mm:.4f} mm  tilt {c_tilt:.4f} deg",
        f"(depth below mouth {c_depth_below_mouth_mm:.2f} mm)",
    ]
    stem = f"control_assembled_dxy{c_dxy_mm:.4f}mm_dz{c_dz_mm:.4f}mm_tilt{c_tilt:.4f}deg"
    for vname, (veye, vtarget) in control_views.items():
        env.sim.set_camera_view(tuple(veye), tuple(vtarget))
        env.sim.render()
        img = np.asarray(env.render())[:, :, :3]
        Image.fromarray(_burn_text(img, burn)).save(args.out / f"{stem}_{vname}.png")
    with open(args.json_out / f"{stem}.json", "w") as f:
        json.dump(
            {
                "kind": "assembled_control",
                "lateral_mm": c_dxy_mm,
                "above_seat_mm": c_dz_mm,
                "depth_below_mouth_mm": c_depth_below_mouth_mm,
                "tilt_deg": c_tilt,
                "fixture_assembled_offset_local": args.fixture_assembled_offset,
                "leg_tip_offset_local": args.leg_tip_offset,
                "leg_tip_quat_wxyz": args.leg_tip_quat,
                "png_stem": stem,
            },
            f,
            indent=2,
        )
    print(f"[view] CONTROL fully-assembled: lateral={c_dxy_mm:.4f}mm above_seat={c_dz_mm:.4f}mm "
          f"tilt={c_tilt:.4f}deg (depth below mouth {c_depth_below_mouth_mm:.2f}mm)", flush=True)
    if c_dxy_mm > 0.1 or abs(c_dz_mm) > 0.1 or c_tilt > 0.05:
        print("[view] WARNING: the assembled control does NOT land on its own seat within 0.1mm/0.05deg"
              " -- the metadata offsets or the Offset algebra disagree with each other", flush=True)

VIDEO_FRAMES: list = []

for ordinal, idx in enumerate(sel, start=1):
    # REPEAT THE SETTLE. PhysX runs on GPU with no seed set, so one settle of one state is a single
    # draw from a distribution, not a measurement: the same six states re-run gave 9.0 -> 25.3 mm,
    # 26.8 -> 95.2 mm and 34.1 -> 14.4 mm. That spread is the same size as the HELD/SLIPPED
    # thresholds, so a one-shot verdict is a coin flip wearing a label. Report median and range.
    drifts = []
    gaps = []
    n_reps = max(1, args.repeats)
    for _rep in range(n_reps):
        write_state(idx)
        gaps.append(target_gap_rad())
        before = scene_obj.data.root_pos_w.clone()
        # Film the LAST repeat only. Filming every repeat would trip the drift measurement into a
        # different code path (render=True) for some reps and not others, and the repeats exist
        # precisely because this settle is stochastic -- they must all run identically.
        filming = args.video is not None and _rep == n_reps - 1
        if os.environ.get("UWLAB_STATE_PROBE") and _rep == n_reps - 1:
            # Does the state SURVIVE being written? Three separate questions, each a different
            # failure: (a) is the arm at the stored joint angles the instant after the write,
            # (b) does it stay there while physics runs, (c) is the cube intersecting the table
            # at t=0. A camera cannot distinguish these; a render that looks wrong could be any
            # of them, or none.
            _q0 = state["articulation"]["robot"]["joint_position"][idx].to(env.device)
            _rb = env.scene["robot"]
            _names = list(_rb.data.joint_names)
            _arm = [k for k, nm in enumerate(_names) if not nm.startswith("rj_dg_")]
            _hand = [k for k, nm in enumerate(_names) if nm.startswith("rj_dg_")]

            def _err(tag):
                _qn = getattr(_rb.data.joint_pos, "torch", _rb.data.joint_pos)[0]
                d = (_qn - _q0).abs()
                _op = scene_obj.data.root_pos_w[0]
                print(f"[probe] {tag:>10s}  arm |dq| max={d[_arm].max():.4f} mean={d[_arm].mean():.4f} rad"
                      f"   hand |dq| max={d[_hand].max():.4f} rad"
                      f"   cube z={_op[2]:+.4f} m", flush=True)

            _st = getattr(_rb.data.joint_stiffness, "torch", _rb.data.joint_stiffness)[0]
            _dp = getattr(_rb.data.joint_damping, "torch", _rb.data.joint_damping)[0]
            print(f"[probe] GAINS arm  stiffness={[round(float(_st[k]),2) for k in _arm]}", flush=True)
            print(f"[probe] GAINS arm  damping  ={[round(float(_dp[k]),2) for k in _arm]}", flush=True)
            print(f"[probe] GAINS hand stiffness min={_st[_hand].min():.3f} max={_st[_hand].max():.3f}"
                  f"  damping min={_dp[_hand].min():.4f} max={_dp[_hand].max():.4f}", flush=True)
            # WHERE IS THE TABLE, ACTUALLY? Every camera guess so far has been argued from
            # renders. This asks USD for the world-space bounds of the prims themselves, which
            # settles both "is the cube inside the table" and where a camera has to stand.
            try:
                from pxr import Usd, UsdGeom
                _stage = env.sim.stage
                _cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                           [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
                for _path in ("/World/envs/env_0/Table", "/World/envs/env_0/Robot",
                              "/World/envs/env_0/InsertiveObject", "/World/envs/env_0/ReceptiveObject",
                              "/World/GroundPlane"):
                    _pr = _stage.GetPrimAtPath(_path)
                    if not _pr or not _pr.IsValid():
                        print(f"[bbox] {_path}: MISSING", flush=True)
                        continue
                    _r = _cache.ComputeWorldBound(_pr).ComputeAlignedRange()
                    if _r.IsEmpty():
                        print(f"[bbox] {_path}: empty", flush=True)
                        continue
                    _mn, _mx = _r.GetMin(), _r.GetMax()
                    print(f"[bbox] {_path:36s} x[{_mn[0]:+.3f},{_mx[0]:+.3f}] "
                          f"y[{_mn[1]:+.3f},{_mx[1]:+.3f}] z[{_mn[2]:+.3f},{_mx[2]:+.3f}]", flush=True)
                # The repo's dexlift notes say this asset's top slab face is at z = 0.000 with
                # collisions at z[-0.030, 0.000]. The aggregate bound above says otherwise, so
                # ask the sub-prims and the transform directly before believing either.
                for _sub in ("/World/envs/env_0/Table/custom_lab_table/collisions/top_slab",
                             "/World/envs/env_0/Table/custom_lab_table/visuals/table_frame",
                             "/World/envs/env_0/Table/custom_lab_table"):
                    _sp = _stage.GetPrimAtPath(_sub)
                    if not _sp or not _sp.IsValid():
                        print(f"[bbox] sub {_sub}: MISSING", flush=True)
                        continue
                    _sr = _cache.ComputeWorldBound(_sp).ComputeAlignedRange()
                    if not _sr.IsEmpty():
                        _a, _b = _sr.GetMin(), _sr.GetMax()
                        print(f"[bbox] sub {_sub.split('Table/')[-1]:42s} "
                              f"z[{_a[2]:+.4f},{_b[2]:+.4f}]", flush=True)
                _tp = _stage.GetPrimAtPath("/World/envs/env_0/Table")
                _xf = UsdGeom.Xformable(_tp)
                print(f"[bbox] Table xform ops: "
                      f"{[(o.GetOpName(), str(o.Get())) for o in _xf.GetOrderedXformOps()]}",
                      flush=True)
                print(f"[bbox] stage upAxis = {UsdGeom.GetStageUpAxis(_stage)}", flush=True)
            except Exception as _e:
                print(f"[bbox] failed: {_e}", flush=True)
            _err("t=0")
            for _k in range(80):
                settle_step()
                if _k + 1 in (5, 20, 80):
                    _err(f"step {_k+1}")
            write_state(idx)   # restore, so the filmed/measured run below is unaffected

        if filming and ordinal == 1 and _rep == n_reps - 1:
            _rb = env.scene["robot"]
            _bn2 = list(_rb.data.body_names)
            _bp2 = getattr(_rb.data.body_pos_w, "torch", _rb.data.body_pos_w)
            for _b2 in ("base_link", "wrist_3_link", "rl_dg_mount"):
                if _b2 in _bn2:
                    _p2 = _bp2[0, _bn2.index(_b2)]
                    print(f"[view] BODY {_b2:14s} ({_p2[0]:+.3f}, {_p2[1]:+.3f}, {_p2[2]:+.3f}) m",
                          flush=True)
            _op = scene_obj.data.root_pos_w[0]
            print(f"[view] BODY insertive_object ({_op[0]:+.3f}, {_op[1]:+.3f}, {_op[2]:+.3f}) m",
                  flush=True)
        if filming and args.video_freeze:
            # THE RESET STATE ITSELF, before any physics. Re-written every frame, so what is on
            # screen is exactly what the bank stores -- no controller, no drift, no argument.
            # Everything after this in the clip is the settle, which is a separate claim and is
            # captioned separately.
            if ordinal == 1:
                # Articulation link poses only reach the renderer once physics has stepped at least
                # once; before that even render() after a write shows the default pose (seen: state
                # 1's freeze frames had the arm stretched out). One warm-up step, then the state is
                # re-written below every frame anyway, so nothing of the step stays on screen.
                write_state(idx)
                settle_step()
            for _f in range(args.video_freeze):
                write_state(idx)
                # Tick the viewport before reading it. env.render() returns the last rendered image,
                # so without this the first state's freeze frames showed the scene from BEFORE the
                # write (arm at its default pose, no cubes) -- later states only looked right
                # because the previous state's settle steps had rendered in between.
                env.sim.render()
                _fr = np.asarray(env.render())[:, :, :3]
                VIDEO_FRAMES.append(_burn_text(_fr, [
                    f"state {ordinal}  idx {idx}",
                    "RESET STATE AS RECORDED (physics not stepped)"]))
        if filming:
            # Aim BEFORE the settle, at where this state STARTS. The gallery's per-state aim runs
            # after the settle and so cannot frame the settle itself -- which is the whole point
            # of the video: a state that begins in the hand and ends on the table has to be in
            # frame at both ends. Standoff is generous for the same reason.
            _o = scene_obj.data.root_pos_w[0].tolist()
            # Direction matters more than distance here. The gallery's own 3q offset is
            # (-d, +0.8d, +0.55d), which on THIS scene points the camera straight into the table
            # and the robot column -- every frame came back as bare tabletop. Viewed from +x/-y
            # instead, the hand, both cubes and the work surface are all in shot.
            _d = 1.0 + 0.6 * max(0.0, _o[2] - 0.02)
            env.sim.set_camera_view(
                (_o[0] + 0.30 * _d, _o[1] - 0.46 * _d, _o[2] + 0.34 * _d),
                (_o[0] - 0.10, _o[1], max(0.02, _o[2] - 0.02)))
            # WARM-UP. set_camera_view does not land in the rendered image until the viewport has
            # ticked: the first frames after aiming come back from the PREVIOUS pose. The stills
            # path never noticed because it renders twice per state anyway. Two discarded renders
            # is what it takes here -- checked by looking at the frames, not by assuming.
            env.render()
            env.render()
        # The MEASUREMENT window stays exactly args.settle_steps, always. Filming may run longer,
        # but the drift is sampled at the same step count in every repeat -- otherwise the filmed
        # repeat would contribute a drift drawn from a different window and quietly bias the
        # median this script reports.
        n_steps = max(args.settle_steps, args.video_settle_steps if filming else 0)
        drift_this_rep = None
        for _st in range(n_steps):
            settle_step()
            if _st + 1 == args.settle_steps:
                drift_this_rep = float((scene_obj.data.root_pos_w - before).norm(dim=-1).item()) * 1000.0
            if filming and _st % max(1, args.video_every) == 0:
                _f = np.asarray(env.render())[:, :, :3]
                VIDEO_FRAMES.append(_burn_text(_f, [
                    f"state {ordinal}  idx {idx}",
                    f"SETTLE step {_st}/{n_steps} -- arm pinned, hand on its own gains"]))
        if drift_this_rep is None:
            drift_this_rep = float((scene_obj.data.root_pos_w - before).norm(dim=-1).item()) * 1000.0
        drifts.append(drift_this_rep)
    drifts_sorted = sorted(drifts)
    drift_mm = drifts_sorted[len(drifts_sorted) // 2]
    drift_lo, drift_hi = drifts_sorted[0], drifts_sorted[-1]
    z_after = float(scene_obj.data.root_pos_w[0, 2].item())
    z_stored = float(obj_pose[idx, 2])
    gap = max(gaps)

    # AIM PER STATE, AT THE OBJECT'S ACTUAL SETTLED POSITION.
    #
    # A single fixed camera cannot frame this set: the selected states span 0.015-0.38 m in object
    # height, and a camera aimed at their MEAN put the hand and the leg outside the frame entirely
    # on the airborne ones. Unlike TiledCamera -- whose runtime re-aim is a silent no-op in this
    # IsaacLab build, verified by reading pos_w back -- the VIEWPORT camera has a real runtime API,
    # which is one more reason this path is the right one for a per-state gallery.
    #
    # Two views per state, both aimed at the object rather than at the scene: a three-quarter view
    # for context (is it near the fixture, is it on the table) and a closer one along -Y for the
    # grip itself. Distance scales with height so a leg held high is not framed identically to one
    # lying on the table.
    obj_now = scene_obj.data.root_pos_w[0].tolist()
    # FRONT view uses the scene's OWN rgb_front_camera placement (data_collection_rgb_cfg.py:75,
    # pos (1.0770121, -0.1679045, 0.4486344) relative to the robot root, which sits at the world
    # origin here) rather than an angle invented for this script -- so the gallery shows the leg
    # from the viewpoint the RGB pipeline itself uses. Aimed at the object so each state is framed.
    dist = 0.42 + 0.35 * max(0.0, obj_now[2] - 0.02)
    views = {
        "front": ([1.0770121, -0.1679045, 0.4486344], obj_now),
        "3q": ([obj_now[0] - dist, obj_now[1] + dist * 0.8, obj_now[2] + dist * 0.55], obj_now),
    }
    seat_w_state = None
    if SEAT_W is not None:
        # RECOMPUTE THE SEAT PER STATE, FROM THE LIVE FIXTURE POSE. write_state(idx) just wrote
        # THIS state's own stored receptive_object pose (this bank's fixture placement varies
        # noticeably across states -- confirmed by inspecting the raw tensor, X spans roughly
        # 0.29-0.59 m), so the pre-loop SEAT_W computed once from the post-reset() fixture pose is
        # stale for every state whose fixture moved since then. At the wide ~0.27 m 'bore' standoff
        # a stale-by-a-few-cm seat still kept the tray in frame, which is exactly how this bug hid
        # through the first visual QA pass; at a tight macro standoff it missed the fixture entirely
        # (caught by looking at the frame, not by trusting the numbers -- same lesson as the control
        # frame's arm occlusion). Recomputed fresh here from env.scene["receptive_object"] every
        # iteration fixes both the 'bore' view and the macro view identically.
        fx_pos_state = fixture.data.root_pos_w[0:1].clone()
        fx_quat_state = fixture.data.root_quat_w[0:1].clone()
        seat_local = torch.tensor(args.bore_seat, device=env.device).unsqueeze(0)
        seat_w_state = (fx_pos_state + _quat_apply_wxyz(fx_quat_state, seat_local))[0].tolist()
        # TIGHT, aimed at the FITTED BORE AXIS, not at the object -- so a leg that misses the hole
        # reads as off-centre instead of being re-centred by the camera (same principle
        # render_partial_assemblies.py's own "bore" view uses). Distance is wider than that script's
        # tuned constants (~0.18 m) because this bank's worst misses run to 40 mm laterally and a
        # tighter frame would crop the leg tip out on exactly the states most worth seeing.
        views["bore"] = (
            [seat_w_state[0] + 0.20, seat_w_state[1] - 0.17, seat_w_state[2] + 0.09],
            list(seat_w_state),
        )
    frames = {}
    for vname, (veye, vtarget) in views.items():
        env.sim.set_camera_view(tuple(veye), tuple(vtarget))
        env.sim.render()
        frames[vname] = np.asarray(env.render())[:, :, :3]

    # VERDICT FROM FINGERTIP CONTACT NORMAL FORCE, NOT FROM DISPLACEMENT. A leg lying flat on the
    # table satisfies "didn't move much over a settle" whether or not the hand is gripping it, so
    # the old drift-based HELD/SLIPPED/DROPPED label carried no information about the grasp and is
    # not reproduced here. Read AFTER the last repeat's settle, at the same simulation instant drift
    # was measured at, using the SAME reducer and SAME 0.2 N gate held_with_probe itself uses.
    thumb_force = _sensor_force_magnitudes(env, THUMB_TIP_NAMES)[0]
    tip_force = _sensor_force_magnitudes(env, TIP_NAMES)[0]
    all_force = {**dict(zip(THUMB_TIP_NAMES, thumb_force.tolist())), **dict(zip(TIP_NAMES, tip_force.tolist()))}
    thumb_loaded = [n for n in THUMB_TIP_NAMES if all_force[n] > FORCE_THRESHOLD]
    tip_loaded = [n for n in TIP_NAMES if all_force[n] > FORCE_THRESHOLD]
    n_loaded = len(thumb_loaded) + len(tip_loaded)
    opposed = bool(thumb_loaded) and bool(tip_loaded)
    if opposed:
        contact_verdict = "OPPOSED"
    elif n_loaded > 0:
        contact_verdict = "CONTACT_NO_OPPOSITION"
    else:
        contact_verdict = "NO_CONTACT"

    # OBJECT HEIGHT ABOVE THE TABLE DATUM (ur5_metal_support root z), so airborne vs supported is
    # legible on the card without eyeballing the render.
    table_datum_z = float(_table_support.data.root_pos_w[0, 2].item())
    height_above_table_m = z_after - table_datum_z
    supported = height_above_table_m < 0.01  # resting on/near the support surface

    # REFERENCE-ONLY in-palm offset (object pose in the rl_dg_mount frame). NOT a hold verdict --
    # a leg can sit at a stable in-palm offset while resting on the table just as well as while
    # actually held; it is reported purely as a positional reference alongside the contact verdict.
    p_palm = robot.data.body_pos_w[0:1, PALM_ID]
    q_palm = robot.data.body_quat_w[0:1, PALM_ID]
    obj_in_palm, _ = math_utils.subtract_frame_transforms(
        p_palm, q_palm, scene_obj.data.root_pos_w[0:1], scene_obj.data.root_quat_w[0:1]
    )
    obj_in_palm = obj_in_palm[0].tolist()

    # ALREADY-MEASURED bore engagement (lateral/depth/tilt), when supplied -- copied verbatim, not
    # recomputed. IN/NOT-IN-bore is the same threshold render_partial_assemblies.py uses: lateral
    # under the radial clearance AND depth within the engaged span.
    eng = ENGAGEMENT.get(str(idx))
    in_bore = None
    eng_stem_bit = ""
    if eng is not None:
        in_bore = (eng["lateral_mm"] < args.radial_clearance_mm) and (0.0 <= eng["depth_mm"] < args.engaged_span_mm)
        eng_stem_bit = f"_dxy{eng['lateral_mm']:.1f}mm_dz{eng['depth_mm']:.1f}mm_tilt{eng['tilt_deg']:.1f}deg"

    stem = f"state{ordinal:02d}_idx{idx}_z{z_stored:.3f}m_{contact_verdict}{eng_stem_bit}"
    loaded_names = thumb_loaded + tip_loaded
    tips_line = ("tips>%.1fN: " % FORCE_THRESHOLD) + (
        ", ".join(f"{n}({all_force[n]:.2f}N)" for n in loaded_names) if loaded_names else "none"
    )
    # TARGET-RESTORATION STATUS, ON EVERY CARD. The difference between "this state has a weak grip"
    # and "we could not command its grip" is exactly whether the bank carried stored PD targets --
    # burned in rather than left to a JSON field nobody reads, because it changes how every other
    # number on the card should be interpreted.
    target_line = (
        "PD target: STORED (restored)" if HAS_STORED_TARGETS
        else "PD target: NOT STORED -- fell back to target:=q"
    )
    burn_lines = [
        f"idx {idx}  {contact_verdict}",
        tips_line,
        f"height above table {height_above_table_m * 1000:.1f} mm  "
        f"({'SUPPORTED' if supported else 'AIRBORNE'})",
        f"in-palm offset (REFERENCE ONLY): "
        f"[{obj_in_palm[0]*1000:.1f}, {obj_in_palm[1]*1000:.1f}, {obj_in_palm[2]*1000:.1f}] mm",
        target_line,
    ]
    if eng is not None:
        burn_lines += [
            f"lateral {eng['lateral_mm']:.2f} mm  (clearance {args.radial_clearance_mm:.3f} mm)",
            f"depth below mouth {eng['depth_mm']:.2f} mm   tilt {eng['tilt_deg']:.2f} deg",
            "IN BORE" if in_bore else "NOT IN BORE",
        ]
    for vname, img in frames.items():
        Image.fromarray(_burn_text(img, burn_lines)).save(args.out / f"{stem}_{vname}.png")

    # MACRO NEAR-MISS VIEW, for the states named explicitly. The regular 'bore' view is legible for
    # a 40 mm miss without help; at ~1 mm against ~0.9 mm of clearance it is legible only because of
    # the burned-in caption, which is a caption with a picture attached, not a picture. This view is
    # meant to make the gap itself visible: straight down the fitted bore axis at a ~5 cm standoff,
    # arm parked (a stray finger fills the frame at this range otherwise -- confirmed by the same
    # occlusion the control frame hit before it was parked).
    has_macro = idx in args.macro_near_miss_idx
    if has_macro:
        assert seat_w_state is not None, "[view] REFUSING: --macro-near-miss-idx given but no --bore-seat"
        _q0 = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(_q0, torch.zeros_like(_q0))
        robot.set_joint_position_target(_q0)
        env.scene.write_data_to_sim()
        env.sim.forward()
        # SAME OBLIQUE DIRECTION AS THE 'bore' VIEW, SCALED DOWN -- not pure top-down. A first
        # attempt straight down the axis found a genuinely dark, hard-to-read frame: a narrow bore
        # gets little direct light from a diffuse dome source when viewed along its own axis, so the
        # cavity the miss needs to be legible against is exactly the part that goes black. The
        # 'bore' view's oblique offset is already known-legible (checked visually on all 12 states
        # in the prior render); reusing its direction at macro-standoff magnitude keeps that
        # lighting behaviour while closing the distance.
        _bore_dir = torch.tensor([0.20, -0.17, 0.02])
        _bore_dir = (_bore_dir / _bore_dir.norm() * args.macro_standoff_m).tolist()
        macro_eye = (seat_w_state[0] + _bore_dir[0], seat_w_state[1] + _bore_dir[1], seat_w_state[2] + _bore_dir[2])
        env.sim.set_camera_view(macro_eye, tuple(seat_w_state))
        env.sim.render()
        macro_img = np.asarray(env.render())[:, :, :3]
        macro_stem = f"{stem}_macro"
        Image.fromarray(_burn_text(macro_img, burn_lines + ["MACRO: top-down on bore axis, arm parked"])).save(
            args.out / f"{macro_stem}.png"
        )
        print(f"[view]   + macro top-down view for idx={idx} (lateral {eng['lateral_mm'] if eng else '?'} mm)",
              flush=True)

    rows.append((ordinal, idx, z_stored, drift_mm, drift_lo, drift_hi, contact_verdict, n_loaded, opposed, supported))
    print(
        f"[view] state {ordinal}/{len(sel)} idx={idx} z_stored={z_stored:.4f} "
        f"drift median={drift_mm:.1f}mm range=[{drift_lo:.1f},{drift_hi:.1f}] n={len(drifts)} "
        f"target_gap={gap:.3f}rad tips_loaded={n_loaded}/{len(ALL_TIP_NAMES)} forces={{"
        + ", ".join(f"{k}:{v:.3f}N" for k, v in all_force.items())
        + f"}} {contact_verdict} height_above_table={height_above_table_m*1000:.1f}mm "
        f"{'SUPPORTED' if supported else 'AIRBORNE'}",
        flush=True,
    )

    with open(args.json_out / f"{stem}.json", "w") as f:
        json.dump(
            {
                "ordinal": ordinal,
                "idx": idx,
                "z_stored_m": z_stored,
                "z_after_settle_m": z_after,
                "drift_mm_median": drift_mm,
                "drift_mm_range": [drift_lo, drift_hi],
                "drift_n_repeats": len(drifts),
                "target_gap_rad_max": gap,
                "has_stored_targets": HAS_STORED_TARGETS,
                "fingertip_force_N": all_force,
                "force_threshold_N": FORCE_THRESHOLD,
                "thumb_tip_names": list(THUMB_TIP_NAMES),
                "tip_names": list(TIP_NAMES),
                "thumb_loaded": thumb_loaded,
                "tip_loaded": tip_loaded,
                "n_tips_loaded": n_loaded,
                "opposed": opposed,
                "contact_verdict": contact_verdict,
                "table_datum_z_m": table_datum_z,
                "height_above_table_m": height_above_table_m,
                "supported": supported,
                "in_palm_offset_m_REFERENCE_ONLY": obj_in_palm,
                "bore_engagement_PREMEASURED_NOT_RECOMPUTED": eng,
                "bore_in_bore": in_bore,
                "radial_clearance_mm": args.radial_clearance_mm,
                "engaged_span_mm": args.engaged_span_mm,
                "png_stem": stem,
                "has_macro_view": has_macro,
                "macro_png_stem": (f"{stem}_macro" if has_macro else None),
            },
            f,
            indent=2,
        )

print("\n[view] SUMMARY  (contact_verdict from fingertip normal force, NOT displacement)", flush=True)
for ordinal, idx, z, d, lo, hi, v, n_loaded, opposed, supported in rows:
    print(
        f"  state {ordinal:2d}  idx {idx:5d}  z {z:7.4f} m  drift {d:8.1f} mm  [{lo:7.1f},{hi:7.1f}]  "
        f"{v:22s}  tips_loaded={n_loaded}/{len(ALL_TIP_NAMES)}  opposed={opposed}  {'SUPPORTED' if supported else 'AIRBORNE'}",
        flush=True,
    )
opposed_n = sum(1 for r in rows if r[6] == "OPPOSED")
contact_n = sum(1 for r in rows if r[6] == "CONTACT_NO_OPPOSITION")
none_n = sum(1 for r in rows if r[6] == "NO_CONTACT")
airborne_n = sum(1 for r in rows if not r[9])
print(
    f"[view] OPPOSED {opposed_n}/{len(rows)}  CONTACT_NO_OPPOSITION {contact_n}/{len(rows)}  "
    f"NO_CONTACT {none_n}/{len(rows)}  AIRBORNE {airborne_n}/{len(rows)}  frames in {args.out}  "
    f"json in {args.json_out}",
    flush=True,
)
if ENGAGEMENT:
    n_in_bore = sum(
        1 for idx in [r[1] for r in rows]
        if str(idx) in ENGAGEMENT
        and ENGAGEMENT[str(idx)]["lateral_mm"] < args.radial_clearance_mm
        and 0.0 <= ENGAGEMENT[str(idx)]["depth_mm"] < args.engaged_span_mm
    )
    print(f"[view] IN BORE {n_in_bore}/{len(rows)}  (pre-measured, radial clearance "
          f"{args.radial_clearance_mm} mm, engaged span {args.engaged_span_mm} mm)", flush=True)
if args.video is not None:
    if VIDEO_FRAMES:
        import imageio.v3 as _iio

        args.video.parent.mkdir(parents=True, exist_ok=True)
        _iio.imwrite(str(args.video), VIDEO_FRAMES, fps=args.video_fps)
        print(f"[view] WROTE {args.video}  ({len(VIDEO_FRAMES)} frames)", flush=True)
    else:
        # Loud, not silent. An empty video file is indistinguishable from a state that never
        # moved, and this script exists to tell those apart.
        print("[view] WARNING: --video requested but no frames captured", flush=True)

print("[view] RENDER_OK", flush=True)

# EXIT HARD. Isaac's shutdown path hangs on this codebase routinely -- both processes in the
# first four-category batch printed RENDER_OK, wrote every frame, and then sat at ~120 percent
# CPU holding 3-4 GiB of GPU until an external SIGKILL, which made a finished render look like a
# stuck one and serialised the queue behind a 900 s timeout per category. Every artifact this
# script produces is already on disk and fsynced by PIL at this point, so there is nothing left
# to flush; a clean interpreter shutdown buys only the hang. os._exit skips atexit and the Kit
# teardown entirely.
import os as _os, sys as _sys
_sys.stdout.flush(); _sys.stderr.flush()
_os._exit(0)
