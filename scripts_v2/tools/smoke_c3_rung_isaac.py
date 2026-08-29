# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""GPU SMOKE for the C3 RUNG stage's ISAAC-TOUCHING half (bead ``dr-ai1.4``, commit ``922c3d3``).

WHY THIS EXISTS. ``mdp/c3_rung_core.py`` (the pure-Python scalar core: which half a draw maps to,
the goal arithmetic, the frame conversion) has 36 passing tests and needs no GPU
(``source/uwlab_tasks/test/test_c3_rung_stage.py``). ``mdp/c3_rung.py`` -- the per-env TENSOR draw
that routes each env to S1 or S_t (``C3RungResetObject.__call__``, the
``kind[draw < self.s1_fraction] = c3_rung_core.C3_KIND_S1`` line and the dispatch that follows it)
and the goal command that reads that draw back (``C3RungGoalPoseCommand``) -- has never been
imported by a Python interpreter, let alone run. A silent S1/S_t swap in that dispatch (e.g. the
comparison flipped, or ``s1_ids``/``st_ids`` swapped at a call site) would poison every reset bank
generated afterwards with legs labelled S1 that are actually lying flat, or legs labelled S_t that
are actually seated tip-down in the bore -- and nothing downstream would catch it, because the
bank's own metadata would agree with itself. This script is the first GPU exercise of that half.

WHAT THIS MEASURES, per env, at reset (see the module docstring of the paired analysis script,
``analyze_c3_rung_smoke.py``, for what is asserted from it):

  * the drawn kind (S1=0, S_t=1), read off ``c3_rung._get_c3_kind_buffer`` -- the SAME private
    buffer both ``C3RungResetObject`` and ``C3RungGoalPoseCommand`` share, not re-derived;
  * the leg's root pose (world frame) and its geometric TIP pose, the latter computed by rotating
    the leg's own ``assembled_offset`` local-frame vector through its actual measured quaternion
    (exact for ANY orientation, not the nominal-tilt approximation);
  * the commanded goal pose (world frame), reconstructed from ``pose_command_b`` via
    ``combine_frame_transforms`` exactly as ``measure_v2_pose_distribution.py`` (the model script,
    read at ``~/github.com/orel/UWLab_v2/scripts_v2/tools/`` on DL_H100) reads the goal;
  * the leg's axis-tilt-from-tip-down, in degrees -- same local ``-X``-axis-vs-world-``-Z``
    definition ``measure_v2_pose_distribution.py`` / ``measure_vertical_hold.py`` both already use,
    so this number is comparable to F43/F50/F51's measured baselines without a re-derivation;
  * whether the fixture (``scene.receptive_object``) is at its normal S1 placement
    (``mdp.RECEPTIVE_POSE_RANGE``, x in [0.35, 0.60] m) or PARKED
    (``episode_mixture.PARKED_FIXTURE_POSE_RANGE``, x = -2.0 m), read from the fixture's OWN
    measured world position relative to this env's origin -- not inferred from the kind label, so a
    kind/behaviour disagreement (the exact swap this script exists to catch) shows up as a
    contradiction between two independently-read quantities rather than being definitionally
    impossible to observe.

Measured AT RESET (immediately after ``env.reset()``, before any ``env.step()``), matching
``measure_v2_pose_distribution.py``'s own read timing -- deliberately NOT waiting for the leg to
settle. S1's tip-down spawn is composer-written directly (F43: 0.00-0.28 deg off tip-down at spawn,
no physics needed) and, with ``DEXLIFT_POSE_TILT`` staged, S_t's spawn ``pose_range`` is already
centred on the near-horizontal baseline (roll/pitch/yaw all within +-``DEXLIFT_POSE_TILT`` of 0, see
``_apply_pose_tilt_stage``), so both halves' AT-RESET tilt already brackets their nominal value
without requiring a settle wait -- this script does not claim to reproduce the F50/F51 SETTLED
baseline (99.02% within 5 deg), only to catch a routing swap, for which the wider at-reset spread is
adequate (see the analysis script's swap-threshold derivation).

NO POLICY, NO STEPPING. This script never calls ``env.step()`` -- only ``env.reset()``, in a loop,
matching ``measure_v2_pose_distribution.py``'s ``--rounds`` idiom (``total samples = num_envs *
rounds``, a fresh draw each round). Reading state before any step also sidesteps
``run_policy_goal_below_spawn.sh``'s "actions must come from somewhere" problem entirely.

FRAME RULES obeyed here, each already the cause of a defect in this campaign (see
``mdp/c3_rung_core.py``'s own "FRAMES" docstring section, F49):
  * Z reported below is ROOT-frame unless explicitly named ``tip``; ``leg_tip_pos_w`` /
    ``goal_tip_pos_w`` are the TIP-frame arrays, computed geometrically (quaternion-rotated
    ``assembled_offset``, exact), never a bare ``root_z - 0.106203``.
  * ``c3_rung_core.goal_tip_z_from_root_z`` (which delegates to ``c3_transport_core.tip_z_from_root_z``,
    ``root_z - ROOT_ABOVE_TIP_M * cos(tilt_rad)``) is imported and used by the PAIRED ANALYSIS
    SCRIPT to compute the NOMINAL tip z per kind and cross-check it against this script's geometric
    (exact) tip z -- this script does not reimplement that arithmetic itself.

MANDATORY ENVIRONMENT -- set HERE, unconditionally, before ``parse_env_cfg``, never trusted from the
caller's shell (same idiom ``measure_vertical_hold.py`` uses and states its reason for): a silently
wrong or unset toggle has repeatedly produced a plausible WRONG number in this project rather than an
error. ``DEXRESET_C3_RUNG=1`` is forced here rather than merely checked, because the entire point of
this script is to exercise that path -- if the launcher's export were ever dropped, this script must
not degrade into silently measuring the UN-staged default env instead of refusing.
``DEXRESET_ST_SPAWN_TIPDOWN`` is explicitly REFUSED if set to ``"1"``: it is surplus for C3 (F51,
``c3_rung_core.py``'s module docstring) and must stay off, so a caller who set it gets a loud refusal
here rather than a quietly different S_t spawn distribution.

Modelled on ``measure_v2_pose_distribution.py`` (arg handling via ``argparse`` +
``AppLauncher.add_app_launcher_args``, the boot order, ``env.reset()`` round loop, npz field naming,
sha256-of-output print, and the ``[..._REPORT_JSON]``/``[..._REPORT_JSON_END]`` bracketed summary
block) -- that script is UNTRACKED on DL_H100 at
``~/github.com/orel/UWLab_v2/scripts_v2/tools/measure_v2_pose_distribution.py`` and was read there
before writing this one. INVENTED HERE, because the model script has no analogue for it: the C3 kind
buffer read (model reads ``episode_mixture``'s kind buffer; this reads ``c3_rung``'s, a different
attribute), the fixture-present/parked classification, the geometric (exact, non-nominal) tip
computation, and the swap-oriented framing of the whole exercise -- the model script measures a
GOAL DISTRIBUTION, this script measures WHETHER TWO STATE MACHINES AGREE.

PHASE 2 (bead ``dr-ai1.20``, ``--mode settle``): the block above is reset-only and therefore exercises
exactly one of the three paths in ``c3_rung.py`` that had never executed -- the per-env draw. The
other two are the DEFERRED RE-PIN machinery added in commits ``9b51f56`` / ``4217ed8``: the
``_st_awaiting_repin`` latch surviving across resets (armed in ``_resample_command``, cleared the
step it fires) and ``C3RungGoalPoseCommand._update_command`` being CALLED AT ALL. Both are still
unrun by ``--mode reset``. This is not a refinement to skip: without the re-pin, S_t's goal sits at
its mid-air spawn pose, in a randomized orientation up to ~90 deg from where the leg actually comes
to rest, which inverts the rung. ``--mode settle`` therefore, per round: resets, then steps with
ZERO actions (``torch.zeros(num_envs, action_manager.total_action_dim)`` -- this measures the settle
PHYSICS and the re-pin STATE MACHINE, never a policy) for ``held_check_core.SETTLE_STEPS +
--settle_margin`` env-steps, watching ``cmd_term._st_awaiting_repin`` for the True->False edge on
every step. Per env this records: the goal at step 0 (the provisional pin), the goal at the end of
the window, the leg's pose at the end, the EXACT internal step index
(``unwrapped.episode_length_buf`` at the moment of the edge, not this script's own loop counter --
the same quantity ``C3RungGoalPoseCommand._update_command``'s own predicate reads, so there is no
off-by-one between what fired the re-pin and what this script reports) the re-pin fired at, and
whether an env's own ``episode_length_buf`` ever DECREASED mid-window (a stray termination
auto-resetting that env inside a plain ``env.step()`` call, IsaacLab's own documented behaviour --
see ``measure_vertical_hold.py``'s module docstring -- which would silently splice a second episode's
draw into this one's row; such envs are flagged ``contaminated`` and excluded from the analysis
script's settle-mode assertions rather than trusted). See ``analyze_c3_rung_smoke.py`` for what gets
asserted from the settle-mode npz -- most importantly that a re-pin never fires at a bounce apex.

Run (one Isaac process; never two on one GPU):
    <python> scripts_v2/tools/smoke_c3_rung_isaac.py \\
        --task DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-Play-v0 \\
        --num_envs 256 --rounds 4 --s1_fraction 0.5 --pose_tilt 0.3 \\
        --out /path/to/out.npz --headless
    <python> scripts_v2/tools/smoke_c3_rung_isaac.py --mode settle \\
        --task DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-Play-v0 \\
        --num_envs 256 --rounds 2 --s1_fraction 0.5 --pose_tilt 0.3 --settle_margin 60 \\
        --out /path/to/out_settle.npz --headless
See ``launch_c3_rung_smoke.sh`` in this directory for the full wrapped invocation (env vars,
UWLAB_TMP_ROOT/TMPDIR, GPU pin).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

from isaaclab.app import AppLauncher

# ==================================== ARGS ====================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-Play-v0",
    help="MUST be one of the two Reorient classes -- _apply_c3_rung_stage is wired into exactly"
    " DexLiftUR5eDeltoRelJointPosTableLegReorientEnvCfg and its _PLAY sibling, never Lift and never"
    " the OSC variant. Default is the _PLAY class (no curriculum), matching this project's other"
    " measurement-only scripts.",
)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--rounds", type=int, default=4, help="env.reset() calls; total samples = num_envs*rounds")
parser.add_argument(
    "--mode",
    choices=["reset", "settle"],
    default="reset",
    help="'reset' (default, bead dr-ai1.4): measure at reset only, never calls env.step() -- proves"
    " the per-env S1/S_t draw and dispatch. 'settle' (bead dr-ai1.20): also steps with zero actions"
    " past SETTLE_STEPS to exercise the deferred S_t goal re-pin (_st_awaiting_repin /"
    " _update_command, commits 9b51f56/4217ed8) -- see the module docstring's PHASE 2 section.",
)
parser.add_argument(
    "--settle_margin",
    type=int,
    default=60,
    help="--mode settle only: env-steps stepped PAST held_check_core.SETTLE_STEPS (60), so the total"
    " settle window is SETTLE_STEPS + this. 60 is team-lead's own suggested margin -- 'a bounce has"
    " time to die out' -- not re-derived here.",
)
parser.add_argument(
    "--s1_fraction",
    type=float,
    default=0.5,
    help="DEXRESET_C3_S1_FRACTION -- the spec's 50/50 default. c3_rung_core.validate_s1_fraction"
    " enforces [0, 1]; this script does not re-validate, the staging function will refuse loudly.",
)
parser.add_argument(
    "--pose_tilt",
    type=float,
    default=0.3,
    help="DEXLIFT_POSE_TILT (rad). 0.3 matches the value measure_vertical_hold.py and the certified"
    " training lineage both stage. Centres S_t's SPAWN roll/pitch/yaw on the near-horizontal"
    " baseline (pitch=0), which is why this script can read a meaningful at-reset S_t tilt without"
    " waiting for physics to settle -- see the module docstring.",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, required=True, help="Path to write the npz to.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# ==================================== MANDATORY ENVIRONMENT ====================================
# Set here, unconditionally, BEFORE AppLauncher/parse_env_cfg -- see module docstring.
_st_spawn_tipdown = os.environ.get("DEXRESET_ST_SPAWN_TIPDOWN")
if _st_spawn_tipdown == "1":
    raise SystemExit(
        "[smoke_c3_rung] REFUSING: DEXRESET_ST_SPAWN_TIPDOWN=1 is set. It is surplus for C3 (F51,"
        " c3_rung_core.py's module docstring: 'S_t therefore requires NO spawn change') and must"
        " stay OFF -- unset it. Continuing would measure a different (tip-down-spawned) S_t than"
        " the one C3 actually ships."
    )
if os.environ.get("DEXLIFT_EPISODE_MIXTURE") == "1" or os.environ.get("DEXLIFT_PARTIAL_ASSEMBLY") == "1":
    raise SystemExit(
        "[smoke_c3_rung] REFUSING: DEXLIFT_EPISODE_MIXTURE=1 or DEXLIFT_PARTIAL_ASSEMBLY=1 is set."
        " _apply_c3_rung_stage raises on this combination (both replace events.reset_object /"
        " commands.object_pose). Unset it -- this script wants DEXRESET_C3_RUNG alone."
    )
os.environ["DEXRESET_C3_RUNG"] = "1"
os.environ["DEXRESET_C3_S1_FRACTION"] = str(args_cli.s1_fraction)
os.environ["DEXLIFT_POSE_TILT"] = str(args_cli.pose_tilt)
# Reference-plant vars -- same production-staging idiom every measurement/generation script in this
# directory sets (bead UWLab-qiao.1 follow-on: missing these turned a 46.71% acceptance run into a
# 2.69% one). Orthogonal to C3 kind routing, but this script should measure the plant C3 actually
# ships against, not whatever the default happens to be.
os.environ["DEXLIFT_REF_RESET"] = "1"
os.environ["DEXLIFT_REF_ACTUATORS"] = "1"
os.environ["DEXLIFT_REF_HAND_ACT"] = "1"
os.environ["DEXLIFT_REF_ARM_ACT"] = "0"
# DL_H100: /tmp/uwlab and /tmp/isaaclab are owned by another uid; TMPDIR is NOT covered by
# UWLAB_TMP_ROOT (IsaacLab's logger calls tempfile.gettempdir() directly). setdefault, not forced --
# the launcher sets both explicitly; this is a fallback for a direct invocation.
_tmp_root = os.environ.get("UWLAB_TMP_ROOT", os.path.expanduser("~/tmp_uwlab"))
os.environ.setdefault("UWLAB_TMP_ROOT", _tmp_root)
os.environ.setdefault("TMPDIR", _tmp_root)
os.makedirs(os.environ["UWLAB_TMP_ROOT"], exist_ok=True)
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

print(f"[smoke_c3_rung] task={args_cli.task} num_envs={args_cli.num_envs} rounds={args_cli.rounds}", flush=True)
print(
    f"[smoke_c3_rung] DEXRESET_C3_RUNG=1 DEXRESET_C3_S1_FRACTION={os.environ['DEXRESET_C3_S1_FRACTION']}"
    f" DEXLIFT_POSE_TILT={os.environ['DEXLIFT_POSE_TILT']}",
    flush=True,
)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_apply, combine_frame_transforms  # noqa: E402

import yaml  # noqa: E402

from uwlab_tasks.manager_based.manipulation.dexlift.mdp import c3_rung  # noqa: E402
from uwlab_tasks.manager_based.manipulation.dexlift.mdp import c3_rung_core  # noqa: E402
from uwlab_tasks.manager_based.manipulation.dexlift.mdp import held_check_core  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEG_METADATA_CANDIDATES = [
    os.path.join(
        REPO_ROOT,
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf/metadata.yaml",
    ),
    os.path.join(
        REPO_ROOT,
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmDecomp/metadata.yaml",
    ),
]
leg_metadata_path = next((p for p in LEG_METADATA_CANDIDATES if os.path.isfile(p)), None)
if leg_metadata_path is None:
    raise FileNotFoundError(f"none of the leg metadata.yaml candidates exist: {LEG_METADATA_CANDIDATES}")
with open(leg_metadata_path) as f:
    leg_metadata = yaml.safe_load(f)
ASSEMBLED_OFFSET_POS = leg_metadata["assembled_offset"]["pos"]  # local-frame tip position, e.g. [-0.106203, 0, 0]
print(f"[smoke_c3_rung] leg metadata used: {leg_metadata_path}", flush=True)
print(f"[smoke_c3_rung] assembled_offset.pos (local tip position) = {ASSEMBLED_OFFSET_POS}", flush=True)

device = args_cli.device
env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs, use_fabric=True)
env_cfg.seed = args_cli.seed
env = gym.make(args_cli.task, cfg=env_cfg)
unwrapped = env.unwrapped

# -- R5-style refusal: confirm the staging actually swapped in the C3 terms, rather than trusting
# that DEXRESET_C3_RUNG=1 silently did what it claims. This is exactly Trap 3
# (RESET_SPEC_V2.md sec 1a: "an env toggle can silently override a hydra override ... more than one
# v1 conclusion turned out to concern a variable that never took effect") turned into a hard gate.
cmd_term = unwrapped.command_manager.get_term("object_pose")
reset_object_term_type = type(unwrapped.event_manager.get_term_cfg("reset_object").func).__name__
cmd_term_type = type(cmd_term).__name__
if cmd_term_type != "C3RungGoalPoseCommand" or reset_object_term_type != "C3RungResetObject":
    raise SystemExit(
        "[smoke_c3_rung] REFUSING: DEXRESET_C3_RUNG=1 did not install the C3 terms -- "
        f"commands.object_pose is {cmd_term_type} (expected C3RungGoalPoseCommand),"
        f" events.reset_object is {reset_object_term_type} (expected C3RungResetObject). Staging"
        " silently failed; nothing below would be measuring what this script claims to measure."
    )
print(
    f"[smoke_c3_rung] staging verified: events.reset_object={reset_object_term_type},"
    f" commands.object_pose={cmd_term_type}",
    flush=True,
)

robot = unwrapped.scene["robot"]
obj = unwrapped.scene["object"]
fixture = unwrapped.scene["receptive_object"]
env_origins = unwrapped.scene.env_origins  # (num_envs, 3), world-frame per-env origin


def tip_from_root(root_pos: torch.Tensor, root_quat_wxyz: torch.Tensor) -> torch.Tensor:
    """World-frame TIP position: EXACT geometric rotation of assembled_offset by the ACTUAL
    measured quaternion (never the nominal-tilt approximation) -- same function
    ``measure_v2_pose_distribution.py`` uses for the same reason."""
    offset = torch.tensor(ASSEMBLED_OFFSET_POS, device=root_pos.device, dtype=root_pos.dtype)
    offset = offset.unsqueeze(0).expand(root_pos.shape[0], -1)
    return root_pos + quat_apply(root_quat_wxyz, offset)


_TIP_AXIS_LOCAL = torch.tensor([-1.0, 0.0, 0.0], device=device)  # leg's local -X = tip direction
_WORLD_DOWN = torch.tensor([0.0, 0.0, -1.0], device=device)


def axis_tilt_from_tipdown_deg(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Angle (deg) between the leg's world tip axis (local -X) and world down. Same definition as
    c3_rung_core.S1_NOMINAL_TILT_RAD (0) / ST_NOMINAL_TILT_RAD (pi/2), F43, F50, F51, and
    measure_vertical_hold.py's leg_tilt_deg -- not re-derived, restated once for this script's own
    tensor ops."""
    local_axis = _TIP_AXIS_LOCAL.unsqueeze(0).expand(quat_wxyz.shape[0], -1)
    world_tip_dir = quat_apply(quat_wxyz, local_axis)
    world_tip_dir = world_tip_dir / world_tip_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    dot = (world_tip_dir * _WORLD_DOWN.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


# -- FIXTURE PRESENT-VS-PARKED THRESHOLD. mdp.RECEPTIVE_POSE_RANGE (S1's normal placement) is x in
# [0.35, 0.60] m, local to the env origin; episode_mixture.PARKED_FIXTURE_POSE_RANGE (S_t's parking
# pose) is x = -2.0 m, local to the env origin (see that module's own docstring for why -2.0 is safe
# only under scene.filter_collisions, not by absolute distance). The two clusters are >1.9 m apart in
# local x; -1.0 m is the bucket threshold (not a clearance claim, unlike
# episode_mixture._PARKED_FIXTURE_MIN_CLEARANCE_M, which is a different, leg-to-fixture check) --
# chosen with >0.9 m of margin on both sides so no plausible physics jitter crosses it.
_FIXTURE_PARKED_LOCAL_X_THRESHOLD = -1.0

if args_cli.mode == "reset":
    # ==================================== PHASE 1: RESET-ONLY ====================================
    kind_all, leg_pos_all, leg_quat_all = [], [], []
    goal_pos_all, goal_quat_all = [], []
    fixture_pos_all, fixture_local_x_all = [], []
    env_id_all, round_all = [], []

    for r in range(args_cli.rounds):
        env.reset()

        kind = c3_rung._get_c3_kind_buffer(unwrapped).detach().clone()
        kind_all.append(kind)

        leg_pos = obj.data.root_pos_w.detach().clone()
        leg_quat = obj.data.root_quat_w.detach().clone()
        leg_pos_all.append(leg_pos)
        leg_quat_all.append(leg_quat)

        robot_pos_w = robot.data.root_pos_w.detach().clone()
        robot_quat_w = robot.data.root_quat_w.detach().clone()
        goal_pos_w, goal_quat_w = combine_frame_transforms(
            robot_pos_w,
            robot_quat_w,
            cmd_term.pose_command_b[:, :3].detach().clone(),
            cmd_term.pose_command_b[:, 3:7].detach().clone(),
        )
        goal_pos_all.append(goal_pos_w)
        goal_quat_all.append(goal_quat_w)

        fixture_pos = fixture.data.root_pos_w.detach().clone()
        fixture_pos_all.append(fixture_pos)
        fixture_local_x_all.append((fixture_pos[:, 0] - env_origins[:, 0]).detach().clone())

        env_id_all.append(torch.arange(args_cli.num_envs, device=device))
        round_all.append(torch.full((args_cli.num_envs,), r, device=device, dtype=torch.long))

        counts = {
            "S1": int((kind == c3_rung_core.C3_KIND_S1).sum()),
            "S_t": int((kind == c3_rung_core.C3_KIND_ST).sum()),
        }
        print(f"[smoke_c3_rung] round {r} done, {args_cli.num_envs} envs, kind counts={counts}", flush=True)

    kind_all = torch.cat(kind_all, dim=0)
    leg_pos_all = torch.cat(leg_pos_all, dim=0)
    leg_quat_all = torch.cat(leg_quat_all, dim=0)
    goal_pos_all = torch.cat(goal_pos_all, dim=0)
    goal_quat_all = torch.cat(goal_quat_all, dim=0)
    fixture_pos_all = torch.cat(fixture_pos_all, dim=0)
    fixture_local_x_all = torch.cat(fixture_local_x_all, dim=0)
    env_id_all = torch.cat(env_id_all, dim=0)
    round_all = torch.cat(round_all, dim=0)

    n = kind_all.shape[0]
    print(f"[smoke_c3_rung] total samples n={n}", flush=True)

    leg_tip_pos_all = tip_from_root(leg_pos_all, leg_quat_all)
    goal_tip_pos_all = tip_from_root(goal_pos_all, goal_quat_all)
    leg_tilt_deg_all = axis_tilt_from_tipdown_deg(leg_quat_all)
    goal_tilt_deg_all = axis_tilt_from_tipdown_deg(goal_quat_all)
    fixture_parked_all = fixture_local_x_all < _FIXTURE_PARKED_LOCAL_X_THRESHOLD

    arrays = {
        "kind": kind_all.cpu().numpy(),  # C3_KIND_S1=0, C3_KIND_ST=1 (c3_rung_core)
        "env_id": env_id_all.cpu().numpy(),
        "round": round_all.cpu().numpy(),
        "leg_root_pos_w": leg_pos_all.cpu().numpy(),
        "leg_root_quat_w_wxyz": leg_quat_all.cpu().numpy(),
        "leg_tip_pos_w": leg_tip_pos_all.cpu().numpy(),  # geometric, exact -- not the nominal conversion
        "leg_tilt_from_tipdown_deg": leg_tilt_deg_all.cpu().numpy(),
        "goal_pos_w": goal_pos_all.cpu().numpy(),
        "goal_quat_w_wxyz": goal_quat_all.cpu().numpy(),
        "goal_tip_pos_w": goal_tip_pos_all.cpu().numpy(),
        "goal_tilt_from_tipdown_deg": goal_tilt_deg_all.cpu().numpy(),
        "fixture_root_pos_w": fixture_pos_all.cpu().numpy(),
        "fixture_local_x_m": fixture_local_x_all.cpu().numpy(),
        "fixture_parked": fixture_parked_all.cpu().numpy(),
    }
    # -- Recorded so the analysis script needs no second source of truth for what this run asked for.
    meta = {
        "mode": "reset",
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "rounds": args_cli.rounds,
        "requested_s1_fraction": args_cli.s1_fraction,
        "requested_pose_tilt": args_cli.pose_tilt,
        "seed": args_cli.seed,
        "s1_goal_delta_m": float(cmd_term.cfg.s1_goal_delta_m),
        "fixture_parked_local_x_threshold_m": _FIXTURE_PARKED_LOCAL_X_THRESHOLD,
        "assembled_offset_pos_local": list(ASSEMBLED_OFFSET_POS),
        "leg_metadata_path": leg_metadata_path,
        "n_samples": n,
    }

else:
    # ==================================== PHASE 2: SETTLE (bead dr-ai1.20) ====================
    # Exercises the deferred S_t re-pin (_st_awaiting_repin, _update_command) -- see the module
    # docstring's PHASE 2 section for why --mode reset cannot exercise this at all.
    SETTLE_STEPS = held_check_core.SETTLE_STEPS  # imported, never restated (60 today)
    total_window = SETTLE_STEPS + args_cli.settle_margin

    # -- Headroom refusal, not an assumption: episode_length_s must comfortably outlast the settle
    # window or a mid-window auto-reset (ManagerBasedRLEnv resets a done sub-env INSIDE the SAME
    # env.step() call -- measure_vertical_hold.py's own module docstring) would splice a second
    # episode's draw into this round's row. This is exactly "a value established under one condition
    # and consumed under another" -- refuse rather than silently trust it.
    step_dt = float(unwrapped.step_dt)
    episode_length_s = float(unwrapped.cfg.episode_length_s)
    usable_steps = int(episode_length_s / step_dt)
    _HEADROOM_STEPS = 20
    if usable_steps < total_window + _HEADROOM_STEPS:
        raise SystemExit(
            f"[smoke_c3_rung] REFUSING (--mode settle): episode_length_s={episode_length_s}s /"
            f" step_dt={step_dt}s = {usable_steps} usable control steps, which does not clear the"
            f" settle window ({total_window} = SETTLE_STEPS {SETTLE_STEPS} + settle_margin"
            f" {args_cli.settle_margin}) plus a {_HEADROOM_STEPS}-step safety margin. Stepping this"
            " far would risk a mid-window auto-reset silently contaminating the data. Reduce"
            " --settle_margin, or pick a task/override with a longer episode_length_s."
        )
    print(
        f"[smoke_c3_rung] settle window: SETTLE_STEPS={SETTLE_STEPS} + settle_margin"
        f"={args_cli.settle_margin} = {total_window} steps; episode budget {usable_steps} steps"
        f" (episode_length_s={episode_length_s}s, step_dt={step_dt}s) -- headroom"
        f" {usable_steps - total_window} steps.",
        flush=True,
    )

    zero_action = torch.zeros((args_cli.num_envs, unwrapped.action_manager.total_action_dim), device=device)

    kind_all = []
    goal_pos_t0_all, goal_quat_t0_all = [], []
    goal_pos_final_all, goal_quat_final_all = [], []
    leg_pos_final_all, leg_quat_final_all = [], []
    repin_step_all, ever_repinned_all, contaminated_all = [], [], []
    env_id_all, round_all = [], []

    for r in range(args_cli.rounds):
        env.reset()

        kind = c3_rung._get_c3_kind_buffer(unwrapped).detach().clone()

        robot_pos_w = robot.data.root_pos_w.detach().clone()
        robot_quat_w = robot.data.root_quat_w.detach().clone()
        goal_pos_t0, goal_quat_t0 = combine_frame_transforms(
            robot_pos_w,
            robot_quat_w,
            cmd_term.pose_command_b[:, :3].detach().clone(),
            cmd_term.pose_command_b[:, 3:7].detach().clone(),
        )

        # -- The latch this whole phase exists to watch. Read directly off the live command-term
        # instance, same idiom as c3_rung._get_c3_kind_buffer: the ground truth is the buffer the
        # code itself reads, not a re-derivation.
        awaiting = cmd_term._st_awaiting_repin.detach().clone()  # noqa: SLF001
        repin_step = torch.full((args_cli.num_envs,), -1, dtype=torch.long, device=device)
        contaminated = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
        prev_episode_len = unwrapped.episode_length_buf.detach().clone()

        for _ in range(total_window):
            env.step(zero_action)

            cur_episode_len = unwrapped.episode_length_buf.detach().clone()
            # A DECREASE means a done sub-env was auto-reset inside that env.step() call -- its row
            # for this round now describes a splice of two episodes, not one continuous settle.
            contaminated |= cur_episode_len < prev_episode_len
            prev_episode_len = cur_episode_len

            cur_awaiting = cmd_term._st_awaiting_repin.detach().clone()  # noqa: SLF001
            just_fired = awaiting & (~cur_awaiting) & (repin_step < 0)
            if bool(just_fired.any()):
                # The EXACT step count the predicate itself used (self._env.episode_length_buf at
                # the instant _update_command fired), not this loop's own counter -- no off-by-one
                # between what triggered the re-pin and what this script reports.
                repin_step[just_fired] = cur_episode_len[just_fired]
            awaiting = cur_awaiting

        goal_pos_final, goal_quat_final = combine_frame_transforms(
            robot.data.root_pos_w.detach().clone(),
            robot.data.root_quat_w.detach().clone(),
            cmd_term.pose_command_b[:, :3].detach().clone(),
            cmd_term.pose_command_b[:, 3:7].detach().clone(),
        )
        leg_pos_final = obj.data.root_pos_w.detach().clone()
        leg_quat_final = obj.data.root_quat_w.detach().clone()
        ever_repinned = repin_step >= 0

        kind_all.append(kind)
        goal_pos_t0_all.append(goal_pos_t0)
        goal_quat_t0_all.append(goal_quat_t0)
        goal_pos_final_all.append(goal_pos_final)
        goal_quat_final_all.append(goal_quat_final)
        leg_pos_final_all.append(leg_pos_final)
        leg_quat_final_all.append(leg_quat_final)
        repin_step_all.append(repin_step)
        ever_repinned_all.append(ever_repinned)
        contaminated_all.append(contaminated)
        env_id_all.append(torch.arange(args_cli.num_envs, device=device))
        round_all.append(torch.full((args_cli.num_envs,), r, device=device, dtype=torch.long))

        st_mask_r = kind == c3_rung_core.C3_KIND_ST
        n_st_r = int(st_mask_r.sum())
        n_st_repinned_r = int((ever_repinned & st_mask_r).sum())
        n_contam_r = int(contaminated.sum())
        print(
            f"[smoke_c3_rung] round {r} settle done: {args_cli.num_envs} envs stepped {total_window}x,"
            f" S_t envs repinned {n_st_repinned_r}/{n_st_r}, contaminated envs {n_contam_r}",
            flush=True,
        )

    kind_all = torch.cat(kind_all, dim=0)
    goal_pos_t0_all = torch.cat(goal_pos_t0_all, dim=0)
    goal_quat_t0_all = torch.cat(goal_quat_t0_all, dim=0)
    goal_pos_final_all = torch.cat(goal_pos_final_all, dim=0)
    goal_quat_final_all = torch.cat(goal_quat_final_all, dim=0)
    leg_pos_final_all = torch.cat(leg_pos_final_all, dim=0)
    leg_quat_final_all = torch.cat(leg_quat_final_all, dim=0)
    repin_step_all = torch.cat(repin_step_all, dim=0)
    ever_repinned_all = torch.cat(ever_repinned_all, dim=0)
    contaminated_all = torch.cat(contaminated_all, dim=0)
    env_id_all = torch.cat(env_id_all, dim=0)
    round_all = torch.cat(round_all, dim=0)

    n = kind_all.shape[0]
    print(f"[smoke_c3_rung] total samples n={n}", flush=True)

    leg_tilt_final_deg_all = axis_tilt_from_tipdown_deg(leg_quat_final_all)
    goal_tilt_t0_deg_all = axis_tilt_from_tipdown_deg(goal_quat_t0_all)
    goal_tilt_final_deg_all = axis_tilt_from_tipdown_deg(goal_quat_final_all)

    arrays = {
        "kind": kind_all.cpu().numpy(),
        "env_id": env_id_all.cpu().numpy(),
        "round": round_all.cpu().numpy(),
        "goal_pos_t0_w": goal_pos_t0_all.cpu().numpy(),  # provisional pin, read at reset
        "goal_quat_t0_w_wxyz": goal_quat_t0_all.cpu().numpy(),
        "goal_tilt_t0_deg": goal_tilt_t0_deg_all.cpu().numpy(),
        "goal_pos_final_w": goal_pos_final_all.cpu().numpy(),  # after total_window steps
        "goal_quat_final_w_wxyz": goal_quat_final_all.cpu().numpy(),
        "goal_tilt_final_deg": goal_tilt_final_deg_all.cpu().numpy(),
        "leg_pos_final_w": leg_pos_final_all.cpu().numpy(),
        "leg_quat_final_w_wxyz": leg_quat_final_all.cpu().numpy(),
        "leg_tilt_final_deg": leg_tilt_final_deg_all.cpu().numpy(),
        "repin_step": repin_step_all.cpu().numpy(),  # -1 = never repinned in the window
        "ever_repinned": ever_repinned_all.cpu().numpy(),
        "contaminated": contaminated_all.cpu().numpy(),  # mid-window auto-reset; exclude from gates
    }
    meta = {
        "mode": "settle",
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "rounds": args_cli.rounds,
        "requested_s1_fraction": args_cli.s1_fraction,
        "requested_pose_tilt": args_cli.pose_tilt,
        "seed": args_cli.seed,
        "settle_steps_source": "held_check_core.SETTLE_STEPS",
        "settle_steps": SETTLE_STEPS,
        "settle_margin": args_cli.settle_margin,
        "total_settle_window_steps": total_window,
        "episode_length_s": episode_length_s,
        "step_dt": step_dt,
        "usable_steps": usable_steps,
        "n_samples": n,
    }

arrays["meta_json"] = np.array(json.dumps(meta))

np.savez(args_cli.out, **arrays)
print(f"[smoke_c3_rung] wrote {args_cli.out}", flush=True)

sha256 = hashlib.sha256()
with open(args_cli.out, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        sha256.update(chunk)
print(f"[smoke_c3_rung] sha256={sha256.hexdigest()}", flush=True)

print("[SMOKE_C3_RUNG_REPORT_JSON]")
print(json.dumps(meta, indent=2))
print("[SMOKE_C3_RUNG_REPORT_JSON_END]")

env.close()
simulation_app.close()
print("[smoke_c3_rung] DONE", flush=True)
