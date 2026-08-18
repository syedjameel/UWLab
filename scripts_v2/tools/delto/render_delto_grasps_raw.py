# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Render recorded DELTO grasps from either a raw ``grasps.pt`` (D2) or a StableState reset-state
``.pt`` (D3, e.g. from ``generate_reset_states_policy.py``).

TWO INPUT SCHEMAS, pick exactly one with ``--grasps`` or ``--reset-states``:

  --grasps <path/to/grasps.pt>
    The D2 grasp sampler's raw output: ``grasp_relative_pose`` -> ``relative_position`` /
    ``relative_orientation`` (the GRIPPER pose expressed in the OBJECT's local frame -- the same
    convention ``mdp/events.py``'s ``reset_end_effector_pose_from_grasp_dataset`` and
    ``grasp_sampling_event`` both use: ``gripper_world = compose(object_world, relative)``) and
    ``gripper_joint_positions`` (one series per recorded grasp). The arm is NEVER moved for this
    path -- moving it would need IK and risks a pose that is not in the dataset -- so only the
    object (placed by inverting the recorded transform against the palm's actual pose) and the 20
    hand joints are written.

  --reset-states <path/to/resets_*.pt>
    A full ``StableStateRecorder`` snapshot (``initial_state`` -> ``articulation``/``rigid_object``,
    i.e. ``env.scene.get_state()``) -- what ``generate_reset_states_policy.py`` and (for other
    tasks) ``record_reset_states.py`` both export, unchanged. This is SIMPLER than the grasps.pt
    path: the file already carries the full robot root pose + every joint + every rigid object's
    pose, so it is written back verbatim -- no inverse-transform derivation, and the arm moves to
    wherever the snapshot has it (that is the point: a StableState is a physically-realized moment,
    not a candidate to be validated by replay).

CAMERAS, either path: two FREE (unparented) cameras, static per-run world poses computed BEFORE
launching Isaac / building the env -- NOT re-aimed per grasp at runtime, and NOT parented to any
robot body. Both were tried and both failed, badly enough to be worth recording rather than
re-discovering:
  1. Re-aiming a free camera every grasp via ``TiledCamera.set_world_poses_from_view``: a silent
     no-op in this IsaacLab build (verified by reading ``camera.data.pos_w`` back -- it never
     moves off the cfg-time placeholder; the rendered frame confirmed it, empty floor/wall).
  2. A camera parented under ``Robot/rl_dg_mount`` (the same body the rig's wrist_camera hangs
     off), added dynamically via ``cfg.scene.debug_cam = TiledCameraCfg(...)``: scene construction
     completed, but the process died with no Python traceback and no error line in either the
     IsaacLab or Kit log during/just after the first ``env.reset()`` -- a native crash, not
     something to keep debugging under a report deadline.
Since the arm never moves in ``--grasps`` mode, the palm's world pose is a deterministic constant
of (task, num_envs=1, object variants) -- verified identical to ~1e-6 across independent runs --
so that path uses a fixed measured constant, aims from the hand's own metadata.yaml (approach
axis, closing axis, grasp centre), and checks the drift between that constant and the real env's
own post-reset reading, loudly, rather than trusting it silently. In ``--reset-states`` mode there
is no such constant (the arm moves to wherever the snapshot puts it, per state), so instead the
camera targets the MEAN position of the primary object across the selected states, read directly
from the .pt file in plain Python before Isaac ever launches -- no forward kinematics needed, and
no per-state re-aim, because that is exactly the runtime call that doesn't work (see above).

THIRD THING WORTH RECORDING, found rehearsing --reset-states against a real file (a --task/scene
mismatch, not a crash): ``gym.make``/``env.reset()`` for this path completed fine -- no repeat of
crash class #2 above. But a resets file recorded against a DIFFERENT task's scene (e.g. the smoke
file this was rehearsed against, from a table-leg/dexlift scene: rigid_object keys ``object``/
``table``) has object names that don't exist as scene entities in ``--task``'s scene (the DELTO
CameraAlign scene has ``insertive_object``, not ``object``). A plain ``KeyError``/``None`` there
LOOKS LIKE A HANG from the outside: this codebase's ``simulation_app.close()`` is known to hang on
an exception exit path, not just a clean one, so a 280s timeout-KILL with an empty output directory
is what a fast, ordinary exception actually looks like here, not evidence of a fourth crash class.
The fix is the explicit ``object_key not in env.scene.keys()`` check below, which raises before the
settle loop with a message naming the mismatch, rather than however far into the loop the object
was first dereferenced.

FOURTH THING WORTH RECORDING, and it is the one that actually explains the others: CAMERA-ENABLED
RENDERING DOES NOT CONSTRUCT ON DL_A6000, PERIOD. Five configurations tried there -- two task
families (dexlift Play and the OmniReset reset_states_cfg family), three different reset-state
files, both debug cameras and a single debug camera in isolation -- every one froze at the
IDENTICAL point, right after "Completed setting up the environment..." plus one carb memory-budget
warning, ~16.8s in, and never recovered (killed by timeout every time; the output directory stayed
empty in all five). That rules out task, reset-state schema, and camera COUNT as the variable --
one camera hangs exactly like two. The leading explanation (this project's own recorded finding,
not re-derived here): DL_A6000 cannot reach the Omniverse extension CDN, so anything that enables
cameras hangs fetching render extensions until the extension cache is warmed locally (an
un-attempted ~5 GB rsync on this box) -- consistent with every non-camera job on that machine
running fine throughout, including the reset-state generation that produced the very states this
script was trying to render. Not something this script can fix or work around; recorded here so
the next person who sees an identical freeze on this box checks the extension cache FIRST, not the
renderer.

FIFTH THING, and it closes out the "insertion reset-state family" investigation with a DIFFERENT
cause than the above: --reset-states ALSO DOES NOT PRODUCE A RENDER ON THE LOCAL BOX, for this
specific task (OmniReset-UR5eDelto-ObjectRestingEEGrasped-v0, and by extension its siblings), even
though the local box's cameras are proven fine elsewhere in this file (the --grasps path renders
cleanly here) and this task is NOT the dexlift/ADR family that hangs locally for an unrelated
reason. Two things had to be fixed first, and both were HONEST, VISIBLE Python tracebacks -- unlike
every DL_A6000 failure, which is silent:
  1. FileNotFoundError for Resets/<pair>/resets_ObjectAnywhereEEAnywhere.pt -- this task's own
     MultiResetManager event term loads a WIDER mixture of reset-type files than just the one
     being targeted, so every reset-type file for the pair needs to be present locally, not only
     the one named on the command line.
  2. FileNotFoundError for Grasps/<insertive object>/grasps.pt -- the reset_end_effector_pose_
     from_grasp_dataset event term (used by the *EEGrasped task variants) needs the OBJECT's own
     grasp dataset too, separate from any reset-state file.
After both were copied down, construction proceeds much further than any DL_A6000 attempt (past
scene/manager/action-term setup, into the termination manager) and then stalls at "Initializing
term 'success' with class 'check_reset_state_success'". That class's __init__ builds a Warp-based
CollisionAnalyzer over the 200 mm leg + fixture pair, heavier collider geometry than anything the
successful --grasps/CameraAlign renders in this file exercise, so the working theory was a slow
first-use CUDA/Warp JIT compile -- and that theory was TESTED, not assumed: a second attempt ran
with timeout -s KILL 900 (15 minutes) on this exclusively-owned, uncontended local GPU. Result: GPU
utilization stayed ACTIVELY HIGH (60-99%, fluctuating, not a flat idle reading) for the entire 900s
and STILL never completed -- no output files, no further log line. Active-the-whole-time rules out
a plain hang (nothing computing) but 15 minutes of continuous compute for one CollisionAnalyzer
construction is far beyond an ordinary cold JIT compile, which points to a real performance
pathology in that construction path for this mesh pair (e.g. O(n^2)-or-worse over the collider,
or Warp kernel thrashing) rather than "just wait longer." Not diagnosed further; recorded so the
next person doesn't reach for "maybe it just needs more time" without first checking whether GPU
utilization is genuinely active (it will be) and, if so, that more time alone did not resolve it
here even at 15 minutes.
NET: the insertion reset-state family has two DISTINCT, UNRELATED construction failures blocking a
render -- DL_A6000's silent extension-cache freeze, and (once past that machine's problem and past
two real missing-dependency errors) a local CollisionAnalyzer construction that runs but does not
finish. Neither is this script's bug to fix.

Grasp/state SELECTION, either path: farthest-point sampling over (position, orientation) of the
primary object, not even spacing by index -- evenly-spaced indices can land on near-identical
poses (verified: they did, on the first pass of this script).

No pose, lighting, or framing is adjusted to flatter a marginal grip: settle steps run real
physics, and the primary object's displacement during settle is measured and printed/labeled so a
grasp that is visibly slipping reads as slipping, not as held.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--grasps", type=Path, help="raw D2 grasps.pt (grasp_relative_pose schema)")
source.add_argument("--reset-states", type=Path, help="StableState .pt (initial_state schema)")
parser.add_argument("--out", type=Path, default=Path("artifacts/delto_grasps_raw_v2"))
parser.add_argument("--count", type=int, default=6, help="number of pose-diverse grasps/states to render")
parser.add_argument("--settle-steps", type=int, default=10)
parser.add_argument("--cam-distance-3q", type=float, default=0.30, help="grasps mode default; overridden below for reset-states mode unless passed explicitly")
parser.add_argument("--cam-distance-axis", type=float, default=0.25)
parser.add_argument("--focal-length", type=float, default=14.0, help="wider (smaller) for a long object like the table leg")
parser.add_argument(
    "--single-camera",
    action="store_true",
    help="add only the three-quarter debug camera, not the closing-axis one -- isolates whether a"
    " construction hang is sensitive to ADDED camera COUNT (2 vs 1), not just presence of any camera",
)
parser.add_argument("--task", default="OmniReset-UR10eDelto-CameraAlign-v0", help="env with calibrated-but-unused front/side/wrist cams; only its scene/robot/lighting are used")
parser.add_argument("--state-task", default="OmniReset-UR10eDelto-RelCartesianOSC-State-v0", help="source of the scene.insertive_object/receptive_object variant cfgs")
parser.add_argument("--insertive-variant", default="deltoblock")
parser.add_argument("--receptive-variant", default="deltoslot")
parser.add_argument("--object-key", default=None, help="reset-states mode: rigid_object name to treat as the grasped object (auto-detected if omitted)")
parser.add_argument("--cam-height", type=int, default=480)
parser.add_argument("--cam-width", type=int, default=640)
parser.add_argument("--dataset-dir", default=None, help="override every event term's dataset_dir (reset states + grasps). Task default points at a different pair.")
parser.add_argument("--dry-run", action="store_true", help="load + select indices + print the schema, no Isaac, no rendering, nothing published")
parser.add_argument(
    "--drop-terminations",
    nargs="*",
    default=[],
    metavar="TERM",
    help="termination terms to set to None before gym.make. A render never reads a termination, and"
    " on the insertion family 'success' builds a Warp CollisionAnalyzer over the leg+fixture pair"
    " that does not finish (see module docstring, FIFTH). Explicit and logged per term.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

if args.dry_run:
    import torch  # noqa: E402

    def summarize(path: Path, reset_states: bool):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if reset_states:
            init = data["initial_state"]
            print(f"[dry-run] {path}")
            print(f"[dry-run]   articulation keys: {list(init['articulation'].keys())}")
            for name, fields in init["articulation"].items():
                n = len(fields["root_pose"])
                print(f"[dry-run]     {name}: {n} state(s), joint_position dim={fields['joint_position'][0].shape}")
            print(f"[dry-run]   rigid_object keys: {list(init['rigid_object'].keys())}")
            for name, fields in init["rigid_object"].items():
                print(f"[dry-run]     {name}: {len(fields['root_pose'])} state(s)")
        else:
            group = data.get("grasp_relative_pose", data)
            print(f"[dry-run] {path}")
            print(f"[dry-run]   {len(group['relative_position'])} grasps, joints={sorted(group['gripper_joint_positions'].keys())}")

    summarize(args.grasps or args.reset_states, args.reset_states is not None)
    raise SystemExit(0)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import uwlab_assets.robots.ur10e_delto as ur10e_delto  # noqa: E402
import uwlab_tasks  # noqa: E402,F401
from uwlab_assets.robots.ur10e_delto.actions import DELTO_HAND_JOINT_NAMES  # noqa: E402
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import read_metadata_from_usd_directory  # noqa: E402

DELTO_EE_BODY = "rl_dg_mount"
FIXTURE_NAMES = {"table", "receptive_object", "ur5_metal_support", "mount", "ground"}


def rgb(sensor) -> np.ndarray:
    image = sensor.data.output["rgb"][0].detach().cpu().numpy()
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (18, 18, 18), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return out


def invert_relative_pose(gripper_pos_w, gripper_quat_w, rel_pos, rel_quat):
    """Solve for the object's world pose given the palm's world pose and the recorded relative pose.

    ``relative`` was recorded (and is replayed elsewhere in this codebase) as the palm's pose
    expressed in the object's local frame: ``gripper_world = object_world ∘ relative``, i.e.
        gripper_quat_w = object_quat_w * rel_quat
        gripper_pos_w  = object_pos_w + object_quat_w * rel_pos
    Solved for the object:
        object_quat_w = gripper_quat_w * rel_quat^-1
        object_pos_w  = gripper_pos_w - object_quat_w * rel_pos
    """
    object_quat_w = math_utils.quat_mul(gripper_quat_w, math_utils.quat_inv(rel_quat))
    object_pos_w = gripper_pos_w - math_utils.quat_apply(object_quat_w, rel_pos)
    return object_pos_w, object_quat_w


def farthest_point_indices(pos: torch.Tensor, quat: torch.Tensor, k: int) -> list[int]:
    """Greedy k-center sampling over (position, orientation) so selected items are visually spread out.

    Evenly-spaced-by-index is NOT the same as visually distinct (verified: it wasn't). Position
    distance and quaternion geodesic angle are each normalized to [0, 1] by their own max before
    summing, so the two contribute comparably even though they're in different units.
    """
    pos_np = pos.numpy().astype(np.float64)
    quat_np = quat.numpy().astype(np.float64)
    n = len(pos_np)
    pos_d = np.linalg.norm(pos_np[:, None, :] - pos_np[None, :, :], axis=-1)
    dot = np.abs(quat_np @ quat_np.T).clip(-1.0, 1.0)
    ang_d = 2.0 * np.arccos(dot)
    combined = pos_d / (pos_d.max() + 1e-9) + ang_d / (ang_d.max() + 1e-9)
    selected = [int(np.argmax(combined.sum(axis=1)))]
    for _ in range(min(k, n) - 1):
        remaining = combined[selected].min(axis=0)
        remaining[selected] = -1.0
        selected.append(int(np.argmax(remaining)))
    return sorted(selected)


def look_at_quat(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> tuple:
    """OpenGL camera (looks down -Z, up +Y) rotation pointing eye->target. Returns (w, x, y, z).

    Verbatim construction from the proven ``look_at_quat`` helper in
    ``diag_render_recorded_grasps.py`` / ``diag_grasp_sampling_render.py``.
    """
    fwd = target - eye
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    z = -fwd
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    rot = np.stack([x, y, z], axis=1)
    quat = math_utils.quat_from_matrix(torch.tensor(rot, dtype=torch.float32).unsqueeze(0))[0]
    return tuple(quat.tolist())


def horizontal(v: np.ndarray) -> np.ndarray:
    """Project onto the world XY plane and re-normalize; falls back to world +X if near-vertical."""
    flat = np.array([v[0], v[1], 0.0])
    n = np.linalg.norm(flat)
    return flat / n if n > 1e-4 else np.array([1.0, 0.0, 0.0])


def build_cfg():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    state_cfg = parse_env_cfg(args.state_task, device=args.device, num_envs=1)
    # Only the DELTO CameraAlign-family scenes carry this insertive/receptive-object variant seam
    # (grasps.pt mode's default --task). A --task pointed at a different task family (e.g. a
    # DexLift plant, which has no such seam and no "insertive_object"/"receptive_object" concept at
    # all -- it ships whatever rigid object that task trains against, e.g. the table leg) has no
    # such attribute; skip the override there and keep the task's own default scene object rather
    # than raising or silently corrupting an unrelated cfg.
    if hasattr(cfg.scene, "insertive_object"):
        # PREFER THE TASK'S OWN VARIANT TABLE over --state-task's.
        #
        # The reset-state task family (OmniReset-UR5eDelto-Object*-v0) defines `variants` itself,
        # with the pair actually under test (leg200mm / onelegfixture). Reading them from a
        # DIFFERENT task -- the UR10e CameraAlign-family --state-task, whose defaults are
        # deltoblock/deltoslot -- silently builds a scene out of the wrong assets. That is not a
        # cosmetic mismatch: those USDs are fetched from the cloud, and a cloud-USD fetch is a
        # known SILENT hang in this project. It is what made an earlier attempt sit for 25 minutes
        # after "Time taken for scene creation" with no further log line and no traceback, which
        # read as "the insertion scene cannot render" when the scene builds in 11 s with the right
        # variants (measured, cameras included).
        source = cfg if getattr(cfg, "variants", None) else None
        if source is None:
            try:
                source = parse_env_cfg(args.state_task, device=args.device, num_envs=1)
                print(f"[render] variant source: --state-task {args.state_task} (task has no variants of its own)")
            except Exception as exc:  # noqa: BLE001
                source = None
                print(f"[render] WARNING: no variant table available ({exc}); using --task's own default scene object(s)")
        else:
            print(f"[render] variant source: --task {args.task}'s own variants table")
        if source is not None:
            # Fail loudly on an unknown variant name rather than falling back to a default: a
            # scene quietly built from the wrong object is exactly the failure being fixed here.
            for field, name in (("insertive_object", args.insertive_variant), ("receptive_object", args.receptive_variant)):
                table = source.variants[f"scene.{field}"]
                if name not in table:
                    raise SystemExit(
                        f"[render] REFUSING: '{name}' is not a known scene.{field} variant. Available: {sorted(table)}"
                    )
                setattr(cfg.scene, field, table[name])
            print(f"[render] variants: insertive={args.insertive_variant} receptive={args.receptive_variant}")

    # Point every reset-state / grasp event term at the dataset that holds THIS pair's files. The
    # task default is ./Datasets_ur5e_delto/OmniReset with the Peg__PegHole pair; the resulting
    # FileNotFoundError is swallowed inside lazy event-term instantiation and resurfaces later as
    # a bogus "MultiResetManager.__init__() got an unexpected keyword argument 'dataset_dir'".
    if args.dataset_dir is not None:
        patched = 0
        for name, term in vars(cfg.events).items():
            if term is not None and getattr(term, "params", None) and "dataset_dir" in term.params:
                term.params["dataset_dir"] = args.dataset_dir
                patched += 1
                print(f"[render] dataset_dir -> {args.dataset_dir}  (event '{name}')")
        if patched == 0:
            raise SystemExit("[render] REFUSING: --dataset-dir given but no event term takes one -- it would be a silent no-op")
    # DROP TERMINATION TERMS THAT ARE POINTLESS FOR A RENDER AND EXPENSIVE TO CONSTRUCT.
    #
    # This is what has blocked the insertion reset-state family from EVER rendering (see the
    # module docstring, FIFTH). ``check_reset_state_success.__init__`` builds a Warp
    # ``CollisionAnalyzer`` over the 200 mm leg + fixture collider pair, and on this local box that
    # construction ran at 60-99 percent GPU for a full 900 s without finishing -- a real
    # performance pathology in that path, not a cold JIT compile (tested, not assumed).
    #
    # A render replays STORED states and never reads a termination: nothing here scores an episode,
    # and the settle loop below steps physics directly rather than going through the termination
    # manager. So the term is pure construction cost for this script's purposes. Dropping it is a
    # narrower intervention than switching to a task without the fixture, which would render a
    # DIFFERENT scene than the one these states are consumed in -- the whole point of an in-domain
    # picture.
    #
    # NOT a silent default: --drop-terminations must be passed explicitly, each dropped term is
    # named on stdout, and a name that is absent is reported rather than ignored, so a typo cannot
    # quietly leave the expensive term in place and be read as "the fix did not help".
    for term in args.drop_terminations:
        if not hasattr(cfg.terminations, term):
            print(f"[render] --drop-terminations: '{term}' is not a termination on {args.task}, nothing dropped")
            continue
        if getattr(cfg.terminations, term) is None:
            print(f"[render] --drop-terminations: '{term}' was already None")
            continue
        setattr(cfg.terminations, term, None)
        print(f"[render] dropped termination '{term}' (render replays stored states; terminations are never read)")

    cfg.scene.num_envs = 1
    cfg.sim.render_interval = 1
    # Render-only, num_envs=1 -- the inherited training default (2 GiB, sized for thousands of
    # envs) is needlessly large here even though the pool is fixed-size rather than per-env.
    # NOTE: an earlier version of this comment blamed a 280s "hang" on shrinking this value on a
    # collider-heavy scene. WRONG -- traced later to an unrelated object_key/--task mismatch plus
    # this codebase's simulation_app.close() hanging on an exception exit path, not this setting.
    # 512 MiB and 1 GiB both behaved identically once that was fixed. Left at 1 GiB (half the
    # training default) out of caution, not because a smaller value was shown to be unsafe.
    cfg.sim.physx.gpu_collision_stack_size = 1024 * 1024 * 1024
    return cfg


def add_debug_cameras(cfg, eye_3q_w, quat_3q_w, eye_axis_w, quat_axis_w) -> None:
    # Free (unparented) cameras, static cfg-time world offsets -- see the module docstring for why
    # this is the fix, not a runtime re-aim and not a Robot-parented sensor.
    cfg.scene.debug_cam_3q = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/DebugCam3Q",
        height=args.cam_height,
        width=args.cam_width,
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(eye_3q_w.tolist()), rot=quat_3q_w, convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=args.focal_length),
    )
    if args.single_camera:
        return
    cfg.scene.debug_cam_axis = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/DebugCamAxis",
        height=args.cam_height,
        width=args.cam_width,
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(eye_axis_w.tolist()), rot=quat_axis_w, convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=args.focal_length),
    )


def render_and_save(env, out_dir: Path, ordinal: int, total: int, source_tag: str, drift_mm: float, threeq_only: list) -> None:
    tag = f"DELTO grasp {ordinal}/{total}  {source_tag}  drift {drift_mm:.1f}mm during settle"
    panel_3q = label(rgb(env.scene["debug_cam_3q"]), f"{tag}  [3/4 view]")
    threeq_only.append(panel_3q)
    stem = f"grasp_{ordinal:02d}"
    cv2.imwrite(str(out_dir / f"{stem}_3q.png"), cv2.cvtColor(panel_3q, cv2.COLOR_RGB2BGR))
    if args.single_camera:
        return
    panel_axis = label(rgb(env.scene["debug_cam_axis"]), f"{tag}  [closing-axis view]")
    pair = np.concatenate([panel_3q, panel_axis], axis=1)
    cv2.imwrite(str(out_dir / f"{stem}_pair.png"), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{stem}_axis.png"), cv2.cvtColor(panel_axis, cv2.COLOR_RGB2BGR))


def write_contact_sheet(out_dir: Path, threeq_only: list) -> None:
    thumb_w = 300
    thumbs = [cv2.resize(im, (thumb_w, round(im.shape[0] * thumb_w / im.shape[1]))) for im in threeq_only]
    cols = min(3, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    blank = np.zeros_like(thumbs[0])
    sheet_rows = []
    for r in range(rows):
        row_imgs = [thumbs[r * cols + c] if r * cols + c < len(thumbs) else blank for c in range(cols)]
        sheet_rows.append(np.concatenate(row_imgs, axis=1))
    sheet = np.concatenate(sheet_rows, axis=0)
    cv2.imwrite(str(out_dir / "contact_sheet_3q.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"[render] contact sheet: {out_dir / 'contact_sheet_3q.png'}")


def main_grasps() -> None:
    path = args.grasps
    if not path.exists():
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu")
    group = data.get("grasp_relative_pose", data)
    rel_pos_all = torch.stack([torch.as_tensor(p, dtype=torch.float32) for p in group["relative_position"]])
    rel_quat_all = torch.stack([torch.as_tensor(q, dtype=torch.float32) for q in group["relative_orientation"]])
    joint_series = group["gripper_joint_positions"]
    n_grasps = len(rel_pos_all)
    print(f"[render] loaded {n_grasps} raw grasps from {path}")

    indices = farthest_point_indices(rel_pos_all, rel_quat_all, args.count)
    print(f"[render] pose-diverse indices selected: {indices}")

    # Hand-local grasp geometry from the HAND's own metadata (not the full robot's) -- the same
    # source DeltoGraspSamplingCfg reads (see delto_cfg.py::_hand_metadata). Unit-length vectors,
    # in the palm's (rl_dg_mount) own local frame.
    hand_metadata = read_metadata_from_usd_directory(ur10e_delto.DELTO_HAND.spawn.usd_path)
    approach_local = np.asarray(hand_metadata["gripper_approach_direction"], dtype=np.float64)
    align_local = np.asarray(hand_metadata["grasp_align_axis"], dtype=np.float64)  # closing axis
    grasp_center_local = np.asarray(hand_metadata["grasp_center_offset"], dtype=np.float64)

    # A two-env "probe first, then build cameras" approach (build a throwaway env, read the palm's
    # world pose, close it, build the real one) was tried and HUNG starting the second env in the
    # same process. Since the arm is never moved and this task's default reset carries no
    # randomization for it, the palm's world pose is a deterministic constant of (task, num_envs=1,
    # object variants) -- verified identical to ~1e-6 across independent runs. Used as a fixed
    # constant here; the drift check right after the real env's own reset (below) is the actual
    # safety net -- if this ever stops holding, that check fires loudly instead of silently
    # mis-aiming the cameras.
    palm_pos_w = np.array([0.6913982629776001, 0.17414991557598114, 0.6768502593040466])
    palm_quat_w = np.array([-1.5050152342155343e-06, -1.0, -1.7434360870538512e-06, -1.4901148688295507e-06])
    print(f"[render] using fixed palm pose: pos={palm_pos_w.tolist()} quat={palm_quat_w.tolist()}")

    def world_vec(local_vec: np.ndarray) -> np.ndarray:
        q = torch.tensor(palm_quat_w, dtype=torch.float32).unsqueeze(0)
        v = torch.tensor(local_vec, dtype=torch.float32).unsqueeze(0)
        out = math_utils.quat_apply(q, v)[0].numpy()
        return out / np.linalg.norm(out)

    forward_w = world_vec(approach_local)  # palm -> grasp centre, in world
    side_w = world_vec(align_local)  # closing axis, in world
    target_w = palm_pos_w + world_vec(grasp_center_local) * np.linalg.norm(grasp_center_local)

    horiz = horizontal(forward_w)
    up = np.array([0.0, 0.0, 1.0])
    perp_horiz = np.cross(up, horiz)
    perp_horiz = perp_horiz / (np.linalg.norm(perp_horiz) + 1e-9)
    eye_3q_dir = 0.35 * horiz + 0.85 * perp_horiz + 0.55 * up
    eye_3q_dir = eye_3q_dir / np.linalg.norm(eye_3q_dir)
    eye_3q_w = target_w + eye_3q_dir * args.cam_distance_3q
    quat_3q_w = look_at_quat(eye_3q_w, target_w)

    eye_axis_w = target_w + side_w * args.cam_distance_axis
    quat_axis_w = look_at_quat(eye_axis_w, target_w)

    print(f"[render] target (grasp centre, world): {target_w.tolist()}")
    print(f"[render] cam_3q world pose: pos={eye_3q_w.tolist()} rot={quat_3q_w}")
    print(f"[render] cam_axis world pose: pos={eye_axis_w.tolist()} rot={quat_axis_w}")

    cfg = build_cfg()
    add_debug_cameras(cfg, eye_3q_w, quat_3q_w, eye_axis_w, quat_axis_w)
    print("[render] gym.make: constructing env (cameras add render products here) ...", flush=True)
    _t = time.time()
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array").unwrapped
    print(f"[render] gym.make: done in {time.time()-_t:.1f}s", flush=True)
    # THE FIRST RESET IS WHERE A CAMERA-ENABLED RUN HANGS ON THIS SCENE. IsaacLab's own last line
    # is "Completed setting up the environment...", which is emitted at the END of gym.make -- so a
    # freeze after it reads as "construction hung" when construction already finished. Measured:
    # first reset takes 0.5s with cameras ENABLED but no TiledCamera added, and did not return in
    # 20 minutes with two 640x480 TiledCameras on this scene.
    print("[render] env.reset(): first reset, triggers first render ...", flush=True)
    _t = time.time()
    env.reset()
    print(f"[render] env.reset(): done in {time.time()-_t:.1f}s", flush=True)

    robot = env.scene["robot"]
    device = env.device
    palm_idx = robot.body_names.index(DELTO_EE_BODY)
    hand_joint_ids = [robot.joint_names.index(name) for name in DELTO_HAND_JOINT_NAMES]
    joint_matrix = torch.stack(
        [torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in joint_series[name]]) for name in DELTO_HAND_JOINT_NAMES],
        dim=1,
    ).to(device)  # (n_grasps, 20)

    # Sanity check: the whole point of the fixed constant above was that this pose doesn't change
    # once the real env is also reset. Confirm it, loudly, instead of silently trusting it.
    palm_pos_w2 = robot.data.body_link_pos_w[0, palm_idx].detach().cpu().numpy().astype(np.float64)
    drift = float(np.linalg.norm(palm_pos_w2 - palm_pos_w)) * 1000.0
    print(f"[render] palm pose fixed-constant-vs-actual drift: {drift:.2f} mm")
    if drift > 5.0:
        print("[render] WARNING: palm pose differs from the constant by >5mm -- camera framing below may be off target.")

    args.out.mkdir(parents=True, exist_ok=True)
    threeq_only: list[np.ndarray] = []

    for ordinal, index in enumerate(indices, start=1):
        # Palm stays wherever the arm's default reset put it; only object + hand joints move.
        palm_pos_t = robot.data.body_link_pos_w[0, palm_idx].unsqueeze(0)
        palm_quat_t = robot.data.body_link_quat_w[0, palm_idx].unsqueeze(0)
        rel_pos = rel_pos_all[index].unsqueeze(0).to(device)
        rel_quat = rel_quat_all[index].unsqueeze(0).to(device)
        object_pos_w, object_quat_w = invert_relative_pose(palm_pos_t, palm_quat_t, rel_pos, rel_quat)

        obj = env.scene["insertive_object"]
        obj.write_root_pose_to_sim(torch.cat([object_pos_w, object_quat_w], dim=-1))
        obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))

        joint_target = joint_matrix[index].unsqueeze(0)
        env_ids = torch.tensor([0], device=device)
        robot.write_joint_state_to_sim(
            position=joint_target, velocity=torch.zeros_like(joint_target), joint_ids=hand_joint_ids, env_ids=env_ids
        )
        robot.set_joint_position_target(joint_target, joint_ids=hand_joint_ids, env_ids=env_ids)

        pos_before = obj.data.root_pos_w[0].clone()
        for _ in range(args.settle_steps):
            env.scene.write_data_to_sim()
            env.sim.step(render=True)
            env.scene.update(env.physics_dt)
        pos_after = obj.data.root_pos_w[0].clone()
        drift_mm = (pos_after - pos_before).norm().item() * 1000.0

        render_and_save(env, args.out, ordinal, len(indices), f"grasps.pt index {index}", drift_mm, threeq_only)
        print(f"[render] grasp {ordinal}/{len(indices)} index {index}: object drift during settle = {drift_mm:.1f} mm")

    write_contact_sheet(args.out, threeq_only)
    print(f"[render] dataset: {path}")
    print(f"[render] selected indices: {indices}")
    env.close()


def main_reset_states() -> None:
    path = args.reset_states
    if not path.exists():
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    init = data["initial_state"]
    art = init["articulation"]
    if "robot" not in art:
        raise ValueError(f"expected an articulation named 'robot' in {path}; found {list(art.keys())}")
    robot_state = art["robot"]
    n_states = len(robot_state["root_pose"])
    print(f"[render] loaded {n_states} reset state(s) from {path}")

    rigid = init["rigid_object"]
    # FAIL LOUD, not silent-fallback: this key determines what gets photographed, and a wrong
    # guess here does not error, it just writes the WRONG rigid body's pose (or none) while the
    # render proceeds and produces a picture that looks like an answer. Found the hard way: a
    # resets file recorded against one scene's naming (e.g. "object") rendered into a differently
    # -named target scene ("insertive_object") would leave the real object at its default pose and
    # photograph the hand gripping empty air -- indistinguishable, by eye, from a genuinely failed
    # grasp. So: only auto-pick between the two conventions this codebase actually uses; anything
    # else raises with both the file's and the (eventual) scene's name sets printed, rather than
    # guessing a "first non-fixture key" that might just be wrong.
    object_key = args.object_key
    if object_key is None:
        for preferred in ("insertive_object", "object"):
            if preferred in rigid:
                object_key = preferred
                break
        if object_key is None:
            raise ValueError(
                f"--object-key not given and neither 'insertive_object' nor 'object' is a rigid_object "
                f"key in {path}; found {list(rigid)}. Pass --object-key explicitly if this file uses a "
                "different convention -- do not guess."
            )
    if object_key not in rigid:
        raise ValueError(f"--object-key {object_key!r} not in {path}'s rigid_object keys {list(rigid)}")
    print(f"[render] primary object key: {object_key!r} (rigid_object keys in {path}: {list(rigid)})")

    obj_pos_all = torch.stack([torch.as_tensor(p[:3], dtype=torch.float32) for p in rigid[object_key]["root_pose"]])
    obj_quat_all = torch.stack([torch.as_tensor(p[3:7], dtype=torch.float32) for p in rigid[object_key]["root_pose"]])
    indices = farthest_point_indices(obj_pos_all, obj_quat_all, args.count)
    print(f"[render] pose-diverse indices selected: {indices}")

    # No hand-frame metadata is meaningful here (the arm moves to wherever each snapshot has it,
    # so there is no single "palm frame" for the whole run) and no forward kinematics is available
    # without Isaac already running. Instead: aim at the MEAN position of the primary object across
    # the selected states, read straight from the file, from a generic elevated three-quarter
    # world direction. Good enough to frame "a hand near a small object"; not claiming to track the
    # hand's own orientation the way the grasps-mode camera does.
    target_w = obj_pos_all[indices].mean(dim=0).numpy().astype(np.float64)
    up = np.array([0.0, 0.0, 1.0])
    eye_3q_dir = np.array([-0.6, 0.6, 0.7])
    eye_3q_dir = eye_3q_dir / np.linalg.norm(eye_3q_dir)
    eye_3q_w = target_w + eye_3q_dir * args.cam_distance_3q
    quat_3q_w = look_at_quat(eye_3q_w, target_w)

    eye_axis_dir = np.array([0.7, -0.7, 0.15])
    eye_axis_dir = eye_axis_dir / np.linalg.norm(eye_axis_dir)
    eye_axis_w = target_w + eye_axis_dir * args.cam_distance_axis
    quat_axis_w = look_at_quat(eye_axis_w, target_w)

    print(f"[render] target (mean object position, world): {target_w.tolist()}")
    print(f"[render] cam_3q world pose: pos={eye_3q_w.tolist()} rot={quat_3q_w}")
    print(f"[render] cam_axis world pose: pos={eye_axis_w.tolist()} rot={quat_axis_w}")

    cfg = build_cfg()
    add_debug_cameras(cfg, eye_3q_w, quat_3q_w, eye_axis_w, quat_axis_w)
    print("[render] gym.make: constructing env (cameras add render products here) ...", flush=True)
    _t = time.time()
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array").unwrapped
    print(f"[render] gym.make: done in {time.time()-_t:.1f}s", flush=True)
    # THE FIRST RESET IS WHERE A CAMERA-ENABLED RUN HANGS ON THIS SCENE. IsaacLab's own last line
    # is "Completed setting up the environment...", which is emitted at the END of gym.make -- so a
    # freeze after it reads as "construction hung" when construction already finished. Measured:
    # first reset takes 0.5s with cameras ENABLED but no TiledCamera added, and did not return in
    # 20 minutes with two 640x480 TiledCameras on this scene.
    print("[render] env.reset(): first reset, triggers first render ...", flush=True)
    _t = time.time()
    env.reset()
    print(f"[render] env.reset(): done in {time.time()-_t:.1f}s", flush=True)

    robot = env.scene["robot"]
    device = env.device
    args.out.mkdir(parents=True, exist_ok=True)
    threeq_only: list[np.ndarray] = []

    # Fail FAST and LOUD, before the settle loop and before any of this run's own error-exit path
    # can be mistaken for a hang: this codebase's ``simulation_app.close()`` is known to hang on an
    # exception exit (see the module docstring's crash-class notes), so a plain ``raise`` deep in
    # the per-state loop below would look identical to a genuine hang from the outside -- a wrong
    # object_key (e.g. a resets file whose scene doesn't match --task, which the ENTIRE point of
    # this check is to catch) is a config/schema mismatch, not a crash, and should read as one.
    if object_key not in env.scene.keys():
        raise ValueError(
            f"rigid_object {object_key!r} (from {path}) is not a scene entity of --task {args.task!r}; "
            f"scene.keys()={list(env.scene.keys())}. The reset-states file's object/scene names must "
            "match the env being rendered into -- this file was very likely recorded against a "
            "DIFFERENT task's scene than --task points at."
        )

    def as_tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)

    for ordinal, index in enumerate(indices, start=1):
        q = as_tensor(robot_state["joint_position"][index])
        qd = as_tensor(robot_state["joint_velocity"][index])
        robot.write_root_pose_to_sim(as_tensor(robot_state["root_pose"][index]))
        robot.write_root_velocity_to_sim(as_tensor(robot_state["root_velocity"][index]))
        robot.write_joint_state_to_sim(q, qd)
        robot.set_joint_position_target(q)

        obj = env.scene[object_key]
        for name, saved in rigid.items():
            asset = env.scene.get(name)
            if asset is None:
                continue
            asset.write_root_pose_to_sim(as_tensor(saved["root_pose"][index]))
            asset.write_root_velocity_to_sim(as_tensor(saved["root_velocity"][index]))

        pos_before = obj.data.root_pos_w[0].clone()
        for _ in range(args.settle_steps):
            env.scene.write_data_to_sim()
            env.sim.step(render=True)
            env.scene.update(env.physics_dt)
        pos_after = obj.data.root_pos_w[0].clone()
        drift_mm = (pos_after - pos_before).norm().item() * 1000.0

        render_and_save(env, args.out, ordinal, len(indices), f"reset-state index {index}", drift_mm, threeq_only)
        print(f"[render] state {ordinal}/{len(indices)} index {index}: object drift during settle = {drift_mm:.1f} mm")

    write_contact_sheet(args.out, threeq_only)
    print(f"[render] dataset: {path}")
    print(f"[render] selected indices: {indices}")
    env.close()


def main() -> None:
    if args.count < 1:
        raise ValueError("--count must be positive")
    if args.grasps is not None:
        main_grasps()
    else:
        main_reset_states()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
