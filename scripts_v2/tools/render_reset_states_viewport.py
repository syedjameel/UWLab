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

# Farthest-point sampling over object position+orientation, so the selection spans the dataset
# instead of showing near-identical poses (evenly spaced indices demonstrably do that here).
feat = obj_pose[:, :7].clone()
sel = [int(torch.argmin(obj_pose[:, 2]).item())]  # start from the LOWEST state: the resting end
for _ in range(min(args.count, n_total) - 1):
    d = torch.cdist(feat, feat[sel]).min(dim=1).values
    d[torch.tensor(sel)] = -1.0
    sel.append(int(torch.argmax(d).item()))
heights = obj_pose[sel, 2]
print(f"[view] {n_total} states; selected {sel}", flush=True)
print(f"[view] selected heights (m): {[round(float(h), 4) for h in heights]}", flush=True)

target = obj_pose[sel, :3].mean(dim=0).tolist()
eye = [target[0] - 0.55, target[1] + 0.45, target[2] + 0.40]
print(f"[view] viewer eye={[round(v,3) for v in eye]} lookat={[round(v,3) for v in target]}", flush=True)

t0 = time.time()
app = AppLauncher(args).app
print(f"[view] app up {time.time()-t0:.1f}s", flush=True)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from PIL import Image  # noqa: E402

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)

for term in args.drop_terminations:
    if getattr(cfg.terminations, term, None) is not None:
        setattr(cfg.terminations, term, None)
        print(f"[view] dropped termination '{term}'", flush=True)

cfg.scene.insertive_object = cfg.variants["scene.insertive_object"][args.insertive_variant]
cfg.scene.receptive_object = cfg.variants["scene.receptive_object"][args.receptive_variant]
print(f"[view] variants: {args.insertive_variant} / {args.receptive_variant}", flush=True)

patched = 0
for name, term in vars(cfg.events).items():
    if term is not None and getattr(term, "params", None) and "dataset_dir" in term.params:
        term.params["dataset_dir"] = args.dataset_dir
        patched += 1
if patched == 0:
    raise SystemExit("[view] REFUSING: no event term takes dataset_dir; the override would be a silent no-op")
print(f"[view] dataset_dir -> {args.dataset_dir} ({patched} term(s))", flush=True)

cfg.scene.num_envs = 1
cfg.sim.render_interval = 1
cfg.sim.physx.gpu_collision_stack_size = 256 * 1024 * 1024
# IN-DOMAIN LIGHTING. reset_states_cfg's dome light is 10000.0; the training scene the states are
# consumed in (rl_state_cfg) uses 1000.0. Rendering at 10x training brightness is both unreadable
# and not what the policy ever sees.
if getattr(cfg.scene, "sky_light", None) is not None and hasattr(cfg.scene.sky_light.spawn, "intensity"):
    was = cfg.scene.sky_light.spawn.intensity
    cfg.scene.sky_light.spawn.intensity = args.light_intensity
    print(f"[view] dome light {was} -> {args.light_intensity} (training scene value)", flush=True)
else:
    print("[view] WARNING: no scene.sky_light to dim; frames may stay overexposed", flush=True)

cfg.viewer.eye = tuple(eye)
cfg.viewer.lookat = tuple(target)
cfg.viewer.resolution = (args.width, args.height)
cfg.viewer.origin_type = "world"

t = time.time()
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array").unwrapped
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
def write_state(idx: int) -> None:
    """Write one stored state verbatim: robot root + all 26 joints + every rigid object pose."""
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
    # COMMAND THE POSTURE, DO NOT ONLY PLACE IT.
    #
    # write_joint_state_to_sim writes q and qdot. It does NOT touch the actuators' position
    # targets, which keep whatever they last held -- for a fresh env, the default posture, whose
    # hand is open. PhysX then runs a PD controller driving the fingers from the recorded grasp
    # toward that default, so the hand actively releases the leg and the measured "drift" is the
    # release, not a property of the reset state. Setting the target to the recorded q makes the
    # controller HOLD the grasp, which is what the stored state actually represents.
    q = state["articulation"]["robot"]["joint_position"][idx].unsqueeze(0).to(env.device)
    if not args.no_hold_targets:
        robot.set_joint_position_target(q)
    env.scene.write_data_to_sim()
    env.sim.forward()


# WHICH HAND IS ACTUALLY REPLAYING THESE GRASPS?
#
# The states were GENERATED in the dexlift scene with DEXLIFT_REF_HAND_ACT=1 -- the REFERENCE
# DELTO actuator, effort 30 N*m and velocity 10000 rad/s. The OmniReset scene builds the shared
# _DELTO_HAND_ACTUATOR instead: effort 0.06-0.17 N*m, velocity 3.0 rad/s, i.e. 200-500x weaker.
# That seam is not a rendering detail -- it is the difference between the environment that made
# these states and the environment that will train on them, so it is printed rather than assumed.
_hand_ids, _ = robot.find_joints([r"rj_dg_[1-5]_[1-4]"], preserve_order=False)
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


for ordinal, idx in enumerate(sel, start=1):
    # REPEAT THE SETTLE. PhysX runs on GPU with no seed set, so one settle of one state is a single
    # draw from a distribution, not a measurement: the same six states re-run gave 9.0 -> 25.3 mm,
    # 26.8 -> 95.2 mm and 34.1 -> 14.4 mm. That spread is the same size as the HELD/SLIPPED
    # thresholds, so a one-shot verdict is a coin flip wearing a label. Report median and range.
    drifts = []
    gaps = []
    for _rep in range(max(1, args.repeats)):
        write_state(idx)
        gaps.append(target_gap_rad())
        before = scene_obj.data.root_pos_w.clone()
        for _ in range(args.settle_steps):
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
        drifts.append(float((scene_obj.data.root_pos_w - before).norm(dim=-1).item()) * 1000.0)
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
    frames = {}
    for vname, (veye, vtarget) in views.items():
        env.sim.set_camera_view(tuple(veye), tuple(vtarget))
        env.sim.render()
        frames[vname] = np.asarray(env.render())[:, :, :3]

    # Verdict from the MEDIAN of the repeats, and the range is printed beside it so a
    # borderline call is visibly borderline rather than silently rounded to a label.
    verdict = "HELD" if drift_mm < 10.0 else ("SLIPPED" if drift_mm < 50.0 else "DROPPED")
    stem = f"state{ordinal:02d}_idx{idx}_z{z_stored:.3f}m_drift{drift_mm:06.1f}mm_{verdict}"
    for vname, img in frames.items():
        Image.fromarray(img).save(args.out / f"{stem}_{vname}.png")
    fn = args.out / f"{stem}_3q.png"
    rows.append((ordinal, idx, z_stored, drift_mm, drift_lo, drift_hi, verdict))
    print(f"[view] state {ordinal}/{len(sel)} idx={idx} z_stored={z_stored:.4f} drift median={drift_mm:.1f}mm range=[{drift_lo:.1f},{drift_hi:.1f}] n={len(drifts)} target_gap={gap:.3f}rad {verdict}", flush=True)

print("\n[view] SUMMARY  (drift = object displacement over real physics settle)", flush=True)
for ordinal, idx, z, d, lo, hi, v in rows:
    print(f"  state {ordinal:2d}  idx {idx:5d}  z {z:7.4f} m  drift {d:8.1f} mm  [{lo:7.1f},{hi:7.1f}]  {v}", flush=True)
held = sum(1 for r in rows if r[6] == "HELD")
print(f"[view] HELD {held}/{len(rows)}  frames in {args.out}", flush=True)
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
