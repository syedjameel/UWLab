# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""FIVE labelled photographs of FOUR DexReset v2 rungs (bead ``dr-sj6.26``), direct user request,
verbatim: "when you would finalise resets generation, could you visualize 4 of them, so that I
could take a look." C1, C2, C3(S1), C3(S_t), C4 -- C3 gets TWO images (see below) -- rendered by
REPLAYING one stored state per rung from a FINAL PRODUCTION bank and photographing it. Never a
smoke: the whole point is to show the user what the campaign will actually train on.

SURVEY, BEFORE WRITING THIS FILE. Two tools already exist in this directory:

* ``visualize_reset_states.py`` -- an INTERACTIVE, non-headless GUI loop
  (``while True: ... env.reset(); time.sleep(...)``) over the OLD OmniReset ``MultiResetManager``
  reset-type vocabulary (``ObjectAnywhereEEAnywhere`` etc., consumed live via
  ``reset_from_reset_states``). It produces NO image file at all -- Ctrl+C is its only exit --
  and has no camera/photography code whatsoever. Does not fit; not adapted.

* ``render_reset_states_viewport.py`` (748 lines) -- DOES replay a bank and photograph it, and
  turned out, on reading it, to already be pointed at the SAME on-disk schema
  ``generate_reset_states_policy.py`` writes (confirmed by loading a real box bank with
  ``torch.load``: ``raw["initial_state"]["rigid_object"]["insertive_object"/"receptive_object"]``,
  ``["articulation"]["robot"]`` -- the DexLift-scene ``object`` key is renamed to
  ``insertive_object`` on write by ``_DexliftToTrainingSceneRecorder``, so the OmniReset-family
  replay task this script targets, ``OmniReset-UR5eDelto-ObjectRestingEEGrasped-v0``, is the
  correct shell scene to pose-and-photograph a DexLift-generated bank in). ITS HARD-WON FIXES ARE
  REUSED HERE, NOT REDISCOVERED: the insertive/receptive scene-variant wiring, the ``dataset_dir``
  patch loop (a term missing it is a silent no-op elsewhere), ``gpu_collision_stack_size`` sized
  down for ``num_envs=1``, dropping the ``success`` termination (builds a Warp analyzer that never
  finishes and a render never reads it anyway), the in-domain dome-light fix (10x overexposed
  otherwise), the STORED-PD-TARGET restoration in ``write_state`` (without it a replayed grasp
  visibly releases, because ``write_joint_state_to_sim`` sets q/qdot but not the actuator target),
  the oblique non-top-down camera formula, the ``_burn_text`` PIL label helper, and the
  ``os._exit(0)`` shutdown (this codebase's Isaac teardown hangs; a clean exit buys nothing once
  every frame is already fsynced to disk).

  NOT REUSED, DELIBERATELY: its settle-repeat/drift-spread loop, its fingertip-force
  OPPOSED/CONTACT/NO_CONTACT grasp verdict, its bore-seat "assembled control" frame, and its
  ``--engagement-json`` external-label injection. All four exist to answer "is this grasp any
  good", a QA question this file's own request never asks -- the user wants to SEE the states the
  campaign will train on, not a grasp-quality report. Threading a new "one median state per rung"
  selection mode through that 748-line QA tool would have made an already-large script harder to
  read for its EXISTING job and risked its documented flags for the sake of a much smaller one. A
  new, deliberately lean driver -- reusing the QA tool's proven machinery where it applies,
  skipping what does not -- is the "adapt, don't triplicate" call for this specific request.

  NEITHER EXISTING SCRIPT computes tip-frame height or per-state axis-tilt-from-tip-down at all
  (the QA tool only ACCEPTS pre-measured engagement numbers via ``--engagement-json``); neither
  does median-of-a-distribution state selection (the QA tool does farthest-point sampling, for
  visual diversity, which is the opposite of what "typical, not cherry-picked" means here); neither
  splits a single C3 bank into its S1 and S_t halves. All three are new in this file.

WHY FIVE IMAGES FOR FOUR RUNGS. C3 = 50% S1 + 50% S_t (``RESET_SPEC_V2.md`` sec 1), and the two
halves are visually OPPOSITE: S1 is tip-down at the bore mouth, S_t is horizontal on the table. A
single "representative" C3 image would show one half and silently misrepresent the other -- so C3
gets two frames, ``c3_s1.png`` and ``c3_st.png``, and this file prints that reasoning again at
runtime so nobody is surprised by a fifth picture in the output directory.

STATE SELECTION: THE MEDIAN, NOT THE PRETTIEST. For each rung (each half, for C3), every state in
the bank is scored by its TIP-frame height (``c3_transport_core.tip_z_from_root_z(root_z, tilt_rad=
<this state's own measured tilt>)`` -- imported, never reimplemented, and the ACTUAL per-state
tilt, never the nominal 0/pi-2 a kind implies, per this bead's own instruction: "never the nominal
tilt when the actual per-state tilt is available"). States are sorted by that score and the one at
rank ``n // 2`` is rendered -- the median, printed as the criterion, not chosen by eye. A
cherry-picked render is worse than no render: it is the visual form of quoting a best-case number.

C3's S1/S_t split, WITHIN ONE BANK. A C3 production bank is generated with ``DEXRESET_C3_RUNG=1``,
which draws S1-vs-S_t PER ENV within the same run (``c3_rung.py``) -- there is no separate S1 bank
and S_t bank, and the recorded state schema carries no explicit kind field (confirmed by inspecting
a real bank's keys). This file classifies each state itself, by the SAME swap-threshold convention
the C3 GPU smoke (bead ``dr-ai1.4``/``dr-ai1.20``) already uses and this project has already vetted
against a deliberately-swapped fixture: axis-tilt-from-tip-down < 45 deg (the midpoint of
:data:`~.c3_rung_core.S1_NOMINAL_TILT_RAD` 0 deg and :data:`~.c3_rung_core.ST_NOMINAL_TILT_RAD` 90
deg, imported, not restated) is S1, >= 45 deg is S_t.

THE RENDER ITSELF IS A SECOND, INDEPENDENT SWAP CHECK. An S_t image whose leg is not roughly
horizontal, or an S1 image whose leg is not roughly tip-down, is visual evidence of exactly the
routing defect the C3 smoke exists to catch -- this file prints the classification threshold and
the measured tilt for every frame specifically so that comparison is possible on sight, not only in
a log.

EVERY FRAME IS LABELLED, burned into the image itself (not only a JSON sidecar, so a viewer who
never opens the sidecar still sees the numbers): the rung, the bank file name, the state index
within that bank (and how many candidate states of that kind existed), the leg's TIP-frame height
in mm, and its tilt from tip-down in degrees.

CAMERA: the QA tool's oblique "3q" formula (elevated 3/4 view, distance scaled to object height),
reused because it is already known-legible for this exact scene and is NOT top-down -- a top-down
view makes a tip-down peg and a horizontal peg look the same width, which would defeat the entire
point of a second, independent swap check.

Modelled on ``render_reset_states_viewport.py`` (see above for exactly what was reused vs.
deliberately dropped). ``c3_transport_core`` / ``c3_rung_core`` are loaded by the SAME file-path
loader idiom ``analyze_c3_rung_smoke.py`` already uses (compile-and-exec the source text directly),
because the state-selection pass runs BEFORE ``AppLauncher`` -- these two modules have no
``isaaclab`` import at module scope and so can be, and must be, loaded before Isaac exists; a plain
package import would pull in sibling ``mdp`` submodules that DO need Isaac already running.

TWO MODES:

* Production (default): pass all four of ``--c1_bank --c2_bank --c3_bank --c4_bank``. Writes
  ``c1.png c2.png c3_s1.png c3_st.png c4.png`` to ``--out``.
* ``--dry_run_bank <path>``: renders ONE frame from ANY bank found on the box, loudly labelled
  DRY RUN / NOT FOR THE USER, to prove the replay-and-photograph path works end to end without
  needing the (not-yet-generated) production banks. Mutually exclusive with the four ``--cN_bank``
  flags.

Run (one Isaac process; never two on one GPU):
    <python> scripts_v2/tools/render_v2_rung_gallery.py \\
        --c1_bank <bank1.pt> --c2_bank <bank2.pt> --c3_bank <bank3.pt> --c4_bank <bank4.pt> \\
        --out artifacts/v2_rung_gallery --headless
    <python> scripts_v2/tools/render_v2_rung_gallery.py \\
        --dry_run_bank <any_existing_bank.pt> --out artifacts/v2_rung_gallery_dryrun --headless
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch  # CPU-only here; pre-launch selection never touches the GPU or Isaac.

from isaaclab.app import AppLauncher

# ==================================== ARGS ====================================
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--c1_bank", type=Path, default=None, help="resets_*.pt for C1 (free, arbitrary).")
parser.add_argument("--c2_bank", type=Path, default=None, help="resets_*.pt for C2 (pre-grasp).")
parser.add_argument(
    "--c3_bank", type=Path, default=None,
    help="resets_*.pt for C3 (DEXRESET_C3_RUNG=1 run) -- ONE bank, split into S1/S_t images here"
    " by measured tilt; see module docstring.",
)
parser.add_argument("--c4_bank", type=Path, default=None, help="resets_*.pt for C4 (partially inserted).")
parser.add_argument(
    "--dry_run_bank", type=Path, default=None,
    help="Render ONE frame from this bank, labelled DRY RUN / NOT FOR THE USER, to prove the"
    " pipeline works end to end. Mutually exclusive with --c1_bank..--c4_bank.",
)
parser.add_argument("--task", default="OmniReset-UR5eDelto-ObjectRestingEEGrasped-v0",
                     help="Replay shell scene -- see module docstring for why this task (not a DexLift"
                     " task) is the correct one to pose-and-photograph a DexLift-generated bank in.")
parser.add_argument("--dataset-dir", default="./Datasets_render/OmniReset")
parser.add_argument("--insertive-variant", default="leg200mm")
parser.add_argument("--receptive-variant", default="onelegfixture")
parser.add_argument("--out", type=Path, default=Path("artifacts/v2_rung_gallery"))
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--light-intensity", type=float, default=1000.0,
                     help="dome light -- reset_states_cfg ships 10000, the training scene uses 1000;"
                          " reused fix from render_reset_states_viewport.py, 10x washes frames out.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
args.enable_cameras = True

_PRODUCTION_BANKS = [args.c1_bank, args.c2_bank, args.c3_bank, args.c4_bank]
if args.dry_run_bank is not None:
    if any(_PRODUCTION_BANKS):
        raise SystemExit("[gallery] REFUSING: --dry_run_bank is mutually exclusive with --c1_bank..--c4_bank.")
    DRY_RUN = True
else:
    if not all(_PRODUCTION_BANKS):
        raise SystemExit(
            "[gallery] REFUSING: production mode needs all four of --c1_bank --c2_bank --c3_bank"
            " --c4_bank (or pass --dry_run_bank alone for a pipeline smoke against any old bank)."
        )
    DRY_RUN = False

# ==================================== PURE-PYTHON CORE, LOADED BEFORE ISAAC ====================================
_MDP_DIR = (
    Path(__file__).resolve().parents[2]
    / "source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/dexlift/mdp"
)


def _load(name: str):
    """Load an Isaac-free ``mdp`` module by FILE PATH, compiling the source text directly -- same
    idiom, same reason, as ``test_c3_rung_stage.py`` / ``analyze_c3_rung_smoke.py``'s own ``_load``:
    NOT ``spec_from_file_location(...).loader.exec_module(...)``, whose ``__pycache__`` staleness
    check is only one-second granular and has already produced one false pass in this campaign.
    """
    path = _MDP_DIR / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"expected an Isaac-free mdp module at {path}")
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)  # noqa: S102
    return module


_c3_transport_core = _load("c3_transport_core")
_c3_rung_core = _load("c3_rung_core")
tip_z_from_root_z = _c3_transport_core.tip_z_from_root_z  # (root_z_m, *, tilt_rad) -- ACTUAL tilt, never nominal
import math as _math  # noqa: E402

S1_NOMINAL_TILT_DEG = _math.degrees(_c3_rung_core.S1_NOMINAL_TILT_RAD)
ST_NOMINAL_TILT_DEG = _math.degrees(_c3_rung_core.ST_NOMINAL_TILT_RAD)
# The same swap threshold the C3 GPU smoke uses (dr-ai1.4/dr-ai1.20) -- the one point equidistant
# from both nominal tilts, so classifying a state on the correct side of it is exactly "not swapped."
SWAP_THRESHOLD_DEG = (S1_NOMINAL_TILT_DEG + ST_NOMINAL_TILT_DEG) / 2.0

# -- Hand-rolled (w, x, y, z) quat_apply on plain CPU tensors, same convention (and the same
# self-test discipline) as measure_vertical_hold.py's own -- deliberately NOT isaaclab.utils.math's,
# so tilt can be computed on the raw bank tensors before AppLauncher exists.
_TIP_AXIS_LOCAL = torch.tensor([-1.0, 0.0, 0.0])
_WORLD_DOWN = torch.tensor([0.0, 0.0, -1.0])


def _quat_apply_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w = q[..., 0:1]
    u = q[..., 1:4]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def axis_tilt_from_tipdown_deg(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Angle (deg) between the leg's world tip axis (local -X) and world down. Same definition as
    c3_rung_core's nominal tilts and the C3 smoke's own axis_tilt_from_tipdown_deg -- restated once
    here as pure CPU torch, not re-derived."""
    axis_local = _TIP_AXIS_LOCAL.expand(quat_wxyz.shape[:-1] + (3,))
    world_axis = _quat_apply_wxyz(quat_wxyz, axis_local)
    world_axis = world_axis / world_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    dot = (world_axis * _WORLD_DOWN).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


# SELF-TEST against the same worked example measure_vertical_hold.py's own quat_apply self-test
# uses: Ry(-90) applied to the tip axis must give world down, i.e. tilt 0 deg (tip-down).
_ry_neg90 = torch.tensor([[0.70710678, 0.0, -0.70710678, 0.0]])
_check_deg = axis_tilt_from_tipdown_deg(_ry_neg90)
assert torch.allclose(_check_deg, torch.zeros(1), atol=1e-2), (
    f"[gallery] REFUSING: quat_apply self-test failed -- Ry(-90) gave tilt {_check_deg.tolist()} deg,"
    " expected ~0 (tip-down). The tilt-from-tipdown definition below would be silently wrong."
)
print("[gallery] quat_apply self-test OK: Ry(-90) tip axis -> tilt 0 deg (tip-down)", flush=True)


def select_median_state(bank_path: Path, kind_filter: str | None) -> dict:
    """Load ``bank_path``, score every state by TIP-frame height (actual per-state tilt, never
    nominal), optionally restrict to the S1 or S_t half by measured tilt, and return the state at
    the MEDIAN rank of that score -- never the extremes, never hand-picked.
    """
    raw = torch.load(bank_path, map_location="cpu", weights_only=False)
    state = raw["initial_state"]
    if "insertive_object" not in state["rigid_object"]:
        raise SystemExit(
            f"[gallery] REFUSING: {bank_path} has no 'insertive_object' in rigid_object"
            f" (got {sorted(state['rigid_object'].keys())}) -- not a bank this tool knows how to read."
        )
    obj_pose = torch.stack(state["rigid_object"]["insertive_object"]["root_pose"])  # (N, 7)
    n_total = obj_pose.shape[0]
    root_z = obj_pose[:, 2]
    quat_wxyz = obj_pose[:, 3:7]
    tilt_deg = axis_tilt_from_tipdown_deg(quat_wxyz)
    tip_z = torch.tensor(
        [tip_z_from_root_z(float(root_z[i]), tilt_rad=float(torch.deg2rad(tilt_deg[i]))) for i in range(n_total)]
    )

    if kind_filter == "s1":
        mask = tilt_deg < SWAP_THRESHOLD_DEG
    elif kind_filter == "st":
        mask = tilt_deg >= SWAP_THRESHOLD_DEG
    else:
        mask = torch.ones(n_total, dtype=torch.bool)
    cand_idx = mask.nonzero().flatten()
    if cand_idx.numel() == 0:
        raise SystemExit(
            f"[gallery] REFUSING: no states in {bank_path} match kind_filter={kind_filter!r}"
            f" (swap threshold {SWAP_THRESHOLD_DEG} deg) out of {n_total} total -- either this bank"
            " is not a mixed C3 bank, or something upstream drew only one half."
        )
    cand_tip_z = tip_z[cand_idx]
    order = torch.argsort(cand_tip_z)
    median_rank = order.numel() // 2
    idx = int(cand_idx[order[median_rank]])

    return {
        "bank_path": bank_path,
        "state": state,
        "idx": idx,
        "n_total": n_total,
        "n_candidates": int(cand_idx.numel()),
        "tip_z_m": float(tip_z[idx]),
        "root_z_m": float(root_z[idx]),
        "tilt_deg": float(tilt_deg[idx]),
        "kind_filter": kind_filter,
    }


# ==================================== BUILD THE MANIFEST (still pre-Isaac) ====================================
if DRY_RUN:
    specs = [("DRY_RUN", args.dry_run_bank, None, "dryrun")]
else:
    specs = [
        ("C1", args.c1_bank, None, "c1"),
        ("C2", args.c2_bank, None, "c2"),
        ("C3(S1)", args.c3_bank, "s1", "c3_s1"),
        ("C3(S_t)", args.c3_bank, "st", "c3_st"),
        ("C4", args.c4_bank, None, "c4"),
    ]

print(
    "[gallery] selection criterion: TIP-frame height "
    "(c3_transport_core.tip_z_from_root_z(root_z, tilt_rad=THIS STATE'S OWN measured tilt)),"
    " ascending; picks the candidate at rank n_candidates // 2 -- the median, never hand-picked.",
    flush=True,
)
if not DRY_RUN:
    print(
        "[gallery] NOTE: 5 images for 4 rungs. C3 = 50% S1 + 50% S_t and the two halves are visually"
        " OPPOSITE (tip-down at the bore vs horizontal on the table); a single C3 image would show"
        " one half and misrepresent the other, so C3 gets two frames (c3_s1.png, c3_st.png), split"
        f" from the SAME bank by measured tilt against a {SWAP_THRESHOLD_DEG:.0f} deg threshold.",
        flush=True,
    )

manifest = []
for label, bank_path, kind_filter, stem in specs:
    sel = select_median_state(bank_path, kind_filter)
    manifest.append((label, stem, sel))
    print(
        f"[gallery] {label:9s} bank={bank_path.name} idx={sel['idx']} of {sel['n_total']}"
        f" (n_candidates={sel['n_candidates']}, kind_filter={kind_filter!r})"
        f" tip_z={sel['tip_z_m']*1000:.1f}mm tilt={sel['tilt_deg']:.1f}deg",
        flush=True,
    )
    if kind_filter == "s1" and sel["tilt_deg"] >= SWAP_THRESHOLD_DEG:
        print(
            f"[gallery] WARNING: {label} selected a state with tilt {sel['tilt_deg']:.1f} deg >="
            f" the {SWAP_THRESHOLD_DEG:.0f} deg S1/S_t threshold -- should be impossible given the"
            " filter above; investigate before trusting this image.",
            flush=True,
        )
    if kind_filter == "st" and sel["tilt_deg"] < SWAP_THRESHOLD_DEG:
        print(
            f"[gallery] WARNING: {label} selected a state with tilt {sel['tilt_deg']:.1f} deg <"
            f" the {SWAP_THRESHOLD_DEG:.0f} deg S1/S_t threshold -- should be impossible given the"
            " filter above; investigate before trusting this image.",
            flush=True,
        )

if DRY_RUN:
    print(
        "[gallery] *** DRY RUN: rendering from an OLD bank, NOT a final production bank. This image"
        " is to prove the replay-and-photograph path works end to end -- it is NOT for the user and"
        " must not be shown or referenced as a v2 render. ***",
        flush=True,
    )

# ==================================== LAUNCH ISAAC ====================================
t0 = time.time()
app = AppLauncher(args).app
print(f"[gallery] app up {time.time()-t0:.1f}s", flush=True)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from PIL import Image  # noqa: E402

# Now safe: Isaac is up, so the package __init__ chain (which pulls in Isaac-touching siblings like
# c3_rung.py) can resolve. PARKED_FIXTURE_POSE_RANGE only, reused rather than re-picked -- same
# constant episode_mixture.py uses to park the fixture clear of a C1/C2 frame that never recorded one.
from uwlab_tasks.manager_based.manipulation.dexlift.mdp.episode_mixture import (  # noqa: E402
    PARKED_FIXTURE_POSE_RANGE,
)

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)

# -- Reused fixes from render_reset_states_viewport.py's own "FOUR CONFIGURATION FACTS" -- see that
# script's module docstring for the debugging round each one cost.
if getattr(cfg.terminations, "success", None) is not None:
    cfg.terminations.success = None
    print("[gallery] dropped termination 'success' (builds a Warp analyzer a render never reads)", flush=True)

cfg.scene.insertive_object = cfg.variants["scene.insertive_object"][args.insertive_variant]
cfg.scene.receptive_object = cfg.variants["scene.receptive_object"][args.receptive_variant]
print(f"[gallery] variants: {args.insertive_variant} / {args.receptive_variant}", flush=True)

patched = 0
for _name, term in vars(cfg.events).items():
    if term is not None and getattr(term, "params", None) and "dataset_dir" in term.params:
        term.params["dataset_dir"] = args.dataset_dir
        patched += 1
if patched == 0:
    raise SystemExit("[gallery] REFUSING: no event term takes dataset_dir; the override would be a silent no-op")
print(f"[gallery] dataset_dir -> {args.dataset_dir} ({patched} term(s))", flush=True)

cfg.scene.num_envs = 1
cfg.sim.render_interval = 1
cfg.sim.physx.gpu_collision_stack_size = 256 * 1024 * 1024
if getattr(cfg.scene, "sky_light", None) is not None and hasattr(cfg.scene.sky_light.spawn, "intensity"):
    was = cfg.scene.sky_light.spawn.intensity
    cfg.scene.sky_light.spawn.intensity = args.light_intensity
    print(f"[gallery] dome light {was} -> {args.light_intensity} (training scene value)", flush=True)
else:
    print("[gallery] WARNING: no scene.sky_light to dim; frames may be overexposed", flush=True)

# First selected object's position, just to give the initial viewer SOME sane framing before the
# per-rung camera below takes over on the first write_state().
_first_pose = torch.stack(manifest[0][2]["state"]["rigid_object"]["insertive_object"]["root_pose"])
_first_target = _first_pose[manifest[0][2]["idx"], :3].tolist()
cfg.viewer.eye = (_first_target[0] - 0.6, _first_target[1] + 0.5, _first_target[2] + 0.45)
cfg.viewer.lookat = tuple(_first_target)
cfg.viewer.resolution = (args.width, args.height)
cfg.viewer.origin_type = "world"

t = time.time()
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array").unwrapped
print(f"[gallery] gym.make {time.time()-t:.1f}s", flush=True)
t = time.time()
env.reset()
print(f"[gallery] first reset {time.time()-t:.1f}s", flush=True)

t = time.time()
frame = env.render()
print(f"[gallery] first render() {time.time()-t:.1f}s -> {None if frame is None else frame.shape}", flush=True)
if frame is None:
    raise SystemExit("[gallery] REFUSING: render() returned None -- render_mode is not rgb_array")

args.out.mkdir(parents=True, exist_ok=True)
robot = env.scene["robot"]
insertive = env.scene["insertive_object"]
_PARKED_FIXTURE_POSE = torch.tensor(
    [[PARKED_FIXTURE_POSE_RANGE["x"][0], PARKED_FIXTURE_POSE_RANGE["y"][0], PARKED_FIXTURE_POSE_RANGE["z"][0],
      1.0, 0.0, 0.0, 0.0]]
)

try:
    from PIL import ImageDraw, ImageFont
    _FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
except Exception as _e:  # noqa: BLE001
    _FONT = None
    print(f"[gallery] WARNING: no truetype font for burn-in text ({_e}); frames will have no overlay", flush=True)


def _burn_text(img: np.ndarray, lines: list[str]) -> np.ndarray:
    """Reused near-verbatim from render_reset_states_viewport.py's own ``_burn_text``: a translucent
    top-left bar so the label survives even if a downstream viewer drops any JSON sidecar."""
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


def write_state(state: dict, idx: int) -> bool:
    """Write one stored state verbatim: robot root + all joints + every rigid object the bank has.
    Reused near-verbatim from render_reset_states_viewport.py's own ``write_state``, INCLUDING the
    stored-PD-target restoration (load-bearing: without it the hand visibly releases the leg on
    replay). Parks the fixture off-camera when this bank never recorded one (C1/C2). Returns
    whether stored PD targets were found, for the frame label.
    """
    robot_state = state["articulation"]["robot"]
    robot.write_root_pose_to_sim(robot_state["root_pose"][idx].unsqueeze(0).to(env.device))
    robot.write_root_velocity_to_sim(robot_state["root_velocity"][idx].unsqueeze(0).to(env.device))
    robot.write_joint_state_to_sim(
        robot_state["joint_position"][idx].unsqueeze(0).to(env.device),
        robot_state["joint_velocity"][idx].unsqueeze(0).to(env.device),
    )
    has_targets = "joint_position_target" in robot_state and "joint_velocity_target" in robot_state
    if has_targets:
        robot.set_joint_position_target(robot_state["joint_position_target"][idx].unsqueeze(0).to(env.device))
        robot.set_joint_velocity_target(robot_state["joint_velocity_target"][idx].unsqueeze(0).to(env.device))
    else:
        robot.set_joint_position_target(robot_state["joint_position"][idx].unsqueeze(0).to(env.device))

    for key in state["rigid_object"]:
        if key in env.scene.rigid_objects:
            asset = env.scene[key]
            asset.write_root_pose_to_sim(state["rigid_object"][key]["root_pose"][idx].unsqueeze(0).to(env.device))
            asset.write_root_velocity_to_sim(
                state["rigid_object"][key]["root_velocity"][idx].unsqueeze(0).to(env.device)
            )

    if "receptive_object" not in state["rigid_object"] and "receptive_object" in env.scene.rigid_objects:
        fixture = env.scene["receptive_object"]
        fixture.write_root_pose_to_sim(_PARKED_FIXTURE_POSE.to(env.device))
        fixture.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))

    env.scene.write_data_to_sim()
    env.sim.forward()
    return has_targets


def camera_view(obj_pos: list[float]) -> tuple[list[float], list[float]]:
    """Oblique, height-scaled 3/4 view -- reused formula from render_reset_states_viewport.py's own
    '3q' view, chosen there (and here) because it is NOT top-down: a top-down camera makes a
    tip-down peg and a horizontal peg look the same width and would defeat this render's second
    purpose as an independent swap check."""
    dist = 0.42 + 0.35 * max(0.0, obj_pos[2] - 0.02)
    eye = [obj_pos[0] - dist, obj_pos[1] + dist * 0.8, obj_pos[2] + dist * 0.55]
    return eye, obj_pos


results = []
for label, stem, sel in manifest:
    has_targets = write_state(sel["state"], sel["idx"])
    obj_pos = insertive.data.root_pos_w[0].tolist()
    eye, target = camera_view(obj_pos)
    env.sim.set_camera_view(tuple(eye), tuple(target))
    env.sim.render()
    img = np.asarray(env.render())[:, :, :3]

    lines = [
        f"{label}",
        f"bank: {sel['bank_path'].name}",
        f"state idx {sel['idx']} of {sel['n_total']} (n_candidates={sel['n_candidates']}, MEDIAN by tip-z)",
        f"tip height (TIP frame): {sel['tip_z_m']*1000:.1f} mm",
        f"tilt from tip-down: {sel['tilt_deg']:.1f} deg  (S1/S_t threshold {SWAP_THRESHOLD_DEG:.0f} deg)",
        "PD target: STORED (restored)" if has_targets else "PD target: NOT STORED -- fell back to target:=q",
    ]
    if DRY_RUN:
        lines = ["*** DRY RUN -- OLD BANK -- NOT FOR THE USER ***"] + lines

    out_path = args.out / f"{stem}.png"
    Image.fromarray(_burn_text(img, lines)).save(out_path)
    results.append((label, out_path, sel))
    print(f"[gallery] wrote {out_path}", flush=True)

print("\n[gallery] SUMMARY", flush=True)
for label, out_path, sel in results:
    print(
        f"  {label:9s} -> {out_path}  idx={sel['idx']:6d}/{sel['n_total']:<6d} "
        f"tip_z={sel['tip_z_m']*1000:7.1f}mm  tilt={sel['tilt_deg']:6.1f}deg  bank={sel['bank_path']}",
        flush=True,
    )
if DRY_RUN:
    print(
        "[gallery] *** REMINDER: the image(s) above are a DRY RUN against an OLD bank. Do NOT show"
        " or reference them as a v2 render -- rerun in production mode once real C1-C4 banks exist. ***",
        flush=True,
    )
print("[gallery] RENDER_OK", flush=True)

# EXIT HARD -- this codebase's Isaac teardown hangs (see render_reset_states_viewport.py's own note
# on this); every frame above is already fsynced by PIL, so a clean interpreter shutdown buys only
# the hang.
import os as _os  # noqa: E402

sys.stdout.flush()
sys.stderr.flush()
_os._exit(0)
