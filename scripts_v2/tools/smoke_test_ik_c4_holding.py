#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Physics smoke test for a deep-C4 (ObjectPartiallyAssembledEEGrasped / task_3) bank produced by
gen_ik_c4_reset_bank.py: does the IK-solved grasp actually HOLD the leg once real physics runs, or
does it fall apart on first contact?

WHY THIS SCRIPT EXISTS. gen_ik_c4_reset_bank.py deliberately never settles physics (see that
script's module docstring) -- every state it writes is a kinematic snapshot, valid by construction
for the geometry checks it runs, but UNTESTED against contact response. The one assumption that
script flagged as its highest-risk, least-verified: every one of the 3000 grasps in grasps.pt was
presumably validated with the leg HORIZONTAL (gravity mostly across the fingers' closing plane); a
C4 state holds the leg VERTICAL, TIP-DOWN (leg metadata.yaml's assembled_offset composes to
root_quat = Ry(-90)), so gravity now pulls along the leg's long axis -- the one direction a radial
finger-pinch resists only through friction, and the SAME axis this bank's whole depth band (12-25mm,
gate at >=22.5mm) is measured on. This script is the live-env measurement that question needs.

WHAT "SAME PATH TRAINING USES" MEANS HERE, CONCRETELY: this does NOT hand-roll write_root_state_to_sim
calls against a bare scene. It builds the REAL RL-state env config every OmniReset UR5eDelto training
run uses (Ur5eDeltoRelCartesianOSCTrainCfg, rl_state_cfg.py's TrainEventCfg), overrides its
``reset_from_reset_states`` event term (task_mdp.MultiResetManager) to draw ONLY from the bank under
test, and then just... calls env.reset() and env.step() with the zero action, the same calls a real
rollout makes. Everything else -- material/mass domain randomization, the arm's sysid actuator, the
OSC gains, the DELTO hand actuator, gravity -- is left exactly as training leaves it. That is
deliberate: a bank that only holds with DR turned off is not a bank that holds during training either.

GAIN REGIME: defaults to the TRAIN action group (Ur5eDeltoRelativeOSCAction / soft, implicit-actuator
gains), NOT the Eval action group, even though "eval" sounds like the right word for a smoke test.
Reason: UR5E_DELTO_RELATIVE_OSC_EVAL's own docstring in actions.py states it is KNOWN to violate the
kd*dt/I<=2 rotational stability bound by 12-21x, everywhere in the workspace -- a real, documented
actuator confound unrelated to grasp quality. Using it here would risk reading "the leg fell out" when
the true cause is "the wrist rang." --gain-regime lets a caller opt into eval/finetune gains for a
follow-up run once the train-gain result is in hand, but train is the fair first measurement.

ZERO ACTION = HOLD, an ASSUMPTION THIS SCRIPT CHECKS RATHER THAN TRUSTS: Ur5eDeltoRelativeOSCAction's
arm term is a RELATIVE Cartesian OSC action (RelCartesianOSCActionCfg) and its hand term is
DELTO_FULL_HAND_ACTIONS, twenty INDEPENDENT RELATIVE joint commands (delto_cfg.py's own comment: "the
hand is twenty independent relative commands rather than one binary posture selection") -- both
integrate a DELTA onto a persistent target, so a zero action should not move either target away from
wherever MultiResetManager's ``_reset_to`` just set it (which is this bank's own recorded
joint_position_target, arm AND hand, written together -- see gen_ik_c4_reset_bank.py's schema notes).
This script does not merely assume that: right after reset it snapshots the articulation's own
``joint_pos_target`` and asserts it is unchanged (within a small tolerance) after the FIRST zero-action
step, and prints a loud warning (not a silent pass) if that assumption does not hold on this install.

CONTACT SENSORS: OMNIRESET WIRES ZERO. Confirmed by reading reset_states_cfg.py's ResetStatesSceneCfg
and rl_state_cfg.py's RlStateSceneCfg -- neither declares a single ContactSensorCfg; every fingertip
force reader in this project (``_sensor_force_magnitudes``, ``contacts``) lives in dexlift/mdp/rewards.py
and is wired by dexlift's own scene configs. Rather than inventing a third contact-sensor convention,
this script REUSES dexlift's exact one-to-many pattern (table_leg_env_cfg.py's ``object_hand_s``): one
ContactSensorCfg rooted at the HELD OBJECT (here, ``{ENV_REGEX_NS}/InsertiveObject`` -- confirmed via a
direct pxr read of square_table_leg4_200mm.usd that its default prim, and therefore the spawned root
prim, carries PhysicsRigidBodyAPI directly, no nested sub-prim needed), filtered against the five DELTO
fingertip prims. Naming it literally ``object_hand_s`` is what makes ``_sensor_force_magnitudes``
(imported, not reimplemented) recognize it and do the one-sensor-many-fingers force lookup automatically.
PALM_BODY/HAND_PRIM/ALL_TIP_NAMES/THUMB_TIP_NAMES/TIP_NAMES are imported from
dexlift_ur10e_delto_env_cfg.py, which states explicitly "the hand is the same graft on both arms, the
bodies keep their names through it" -- confirmed independently by traversing ur5e_delto.usd directly
with pxr: {ENV_REGEX_NS}/Robot/gripper/rl_dg_mount and .../gripper/rl_dg_{1..5}_tip all exist exactly as
named (the USD's own default-prim NAME happens to be a leftover "ur5e_robotiq_gripper_d415_mount" --
cosmetic, unrelated to the graft, verified by traversal that the DELTO subtree is really there under
that root).

GRADING: FINGERTIP CONTACT FORCE, NOT DISPLACEMENT -- per direct instruction, because this project has
a MEASURED trap here: at n=298, a strict 20mm in-palm-displacement tolerance scored 0.54 while
contact-based opposed-grip grading scored 0.96 on the SAME rollouts, and the disagreement was
one-sided (131 contact-only vs 5 held-only) -- displacement grading was punishing legitimate in-hand
repositioning as failure. "Opposed grip" here means: at least one of the two thumb-side tips
(rl_dg_1_tip, rl_dg_5_tip) AND at least one of the three finger tips (rl_dg_2/3/4_tip) reads a filtered
contact force above --contact-threshold-n (default 0.2N, per direct instruction), read through
dexlift's OWN ``_sensor_force_magnitudes`` -- reused, not reimplemented, so this script's contact
grading cannot silently drift from the reward/gate logic the rest of the project already trusts. A
STRICT displacement comparator (leg lateral/tilt reprojected into the mating frame vs. its t=0 value)
is ALSO reported alongside, specifically so a reader can see directly whether the same displacement-vs-
contact disagreement this project already measured once reappears here, rather than trusting either
number blind.

TWO NAMED TRAPS, AND HOW THIS SCRIPT HANDLES EACH:

  (1) ManagerBasedRLEnv AUTO-RESETS a done sub-environment INSIDE the SAME env.step() call that
  detects it is done -- by the time step() returns, a just-terminated env's SCENE STATE (root poses,
  and very likely its contact-sensor buffers, since sensor updates ride the same per-step data
  pipeline as everything else) already belongs to the NEW, unrelated episode MultiResetManager placed
  it into. Reading env.scene fresh after step() therefore risks grading a different episode entirely
  for any env that just died. This script NEVER re-reads env.scene state for a env whose ``died_at``
  is already set. Instead it maintains a per-env RUNNING buffer (depth/lateral/tilt/held), updated
  EVERY control step for envs that were still alive at the START of that step, and FROZEN the instant
  ``died_at`` is recorded for that env -- checkpoint reports read the frozen buffer, never the live
  scene, for anything already dead. The depth/lateral/tilt/success numbers this freezes on come from
  ``ProgressContext`` (omnireset/mdp/rewards.py's reward term already wired into this env, computed
  DURING reward evaluation, which happens BEFORE the internal auto-reset within the same step() call,
  and is not itself reset by ``ProgressContext.reset()`` -- that method only zeros
  ``continuous_success_counter``) -- so those specific numbers are SAFE to read immediately after
  step() returns even for a just-died env. Contact-sensor forces are read the same step, before the
  death is even known, for the SAME reason (batched read, gated into the buffer only for envs alive at
  step START) -- this script does not assume sensor-buffer ordering it hasn't verified; if the ordering
  assumption is wrong the failure mode is a stale-by-one-step contact reading at the exact death
  instant, not a wrong episode, which is a small, bounded error, called out here rather than hidden.
  ``died_reason`` is also recorded per env (via ``env.termination_manager.get_term(name)``, one check
  per configured termination term) so a "success" termination (a GOOD outcome -- the state was already
  stable and stayed there) is never confused with "abnormal_robot"/"object_out_of_bound"/"dropped"
  (real holding failures) in the report.

  (2) Stored bank poses are RELATIVE TO THE ENV ORIGIN, invisible at num_envs=1 (env origin (0,0,0))
  and silently wrong at num_envs>1 if a caller forgets to add it. This script does NOT hand-roll that
  addition -- it reuses ``MultiResetManager._reset_to`` verbatim (via the real event term, called by a
  real ``env.reset()``), which already does exactly ``root_pose[:, :3] += scene.env_origins[env_ids]``
  when ``is_relative=True`` (confirmed by reading events.py directly) -- the same reason this script
  builds a real multi-env ``ManagerBasedRLEnv`` rather than writing scene state by hand.

SAMPLE SELECTION: MultiResetManager draws ``state_indices = torch.randint(0, num_states, (num_envs,))``
-- i.e. num_envs i.i.d. draws WITH replacement from the ENTIRE bank, not a caller-controlled subset.
This script does not fight that: with num_envs (default 256) far smaller than a full generation run's
state count, this already gives an unbiased ~256-state sample without any extra bookkeeping, and this
script never needs to know which bank INDEX landed in which env -- every diagnostic below is read
directly from the live scene/reward term at t=0 (immediately after reset, before the first step), which
is that env's own ground truth for "requested," and compared against the SAME env's own value at each
later checkpoint.

DOES NOT RUN ITSELF: needs a real Isaac boot, exactly like its two siblings. Written to be scheduled
once a bank exists and a GPU frees; not run in this session (no Isaac/GPU available here).

Run:
    PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl" \\
        timeout -s KILL 900 <python> -u scripts_v2/tools/smoke_test_ik_c4_holding.py \\
        --bank-path local_ckpts/deep_c4_partial_assemblies_v1/resets_ObjectPartiallyAssembledEEGrasped_ik_v1.pt \\
        --n-envs 256 --hold-seconds 2.0 \\
        --out local_ckpts/deep_c4_partial_assemblies_v1/smoke_test_ik_c4_holding_summary.json \\
        --headless

GO / NO-GO CRITERION, STATED BEFORE ANY RESULT IS SEEN (per direct instruction not to argue this after
the fact): the IK route is NOT worth pursuing further as-is if EITHER holds:
  (a) the median DEPTH DELTA (depth_into_bore_mm at the 2s checkpoint minus at t=0, frozen at death for
      any env that dies first) is negative and its magnitude exceeds 5mm -- more than a fifth of the
      entire 12-25mm spawn band lost to axial slip under held targets alone, with zero policy
      compensation, before training even starts asking the arm to do anything;
  (b) the CONTACT-graded (opposed-grip) held fraction at the 2s checkpoint is below 0.70 -- i.e. losing
      the grip is closer to a coin flip than an edge case, again with no policy involved yet.
Both are physics facts about THIS gravity direction against THIS grasp population that no amount of
retrying more grasps or tightening the IK accept tolerance can fix -- the intervention at that point is
a different one (grasps selected/screened under axial load specifically, or an active-hold reset
strategy), not a bigger IK sweep. Conversely, a held fraction >=0.70-0.80 at 2s with a depth-delta
distribution centered near zero (not systematically negative) is ordinary reset-bank settle noise and
this route should proceed to a full run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Physics smoke test: does an IK-generated deep C4 bank actually hold?")
parser.add_argument("--bank-path", type=str, required=True, help="resets_ObjectPartiallyAssembledEEGrasped*.pt from gen_ik_c4_reset_bank.py")
parser.add_argument("--n-envs", type=int, default=256, help="Also the sample size (see module docstring's SAMPLE SELECTION note).")
parser.add_argument("--hold-seconds", type=float, default=2.0)
parser.add_argument(
    "--checkpoints-s", type=str, default="0.1,0.5,1.0,2.0",
    help="Wall-clock times (s, post-reset) at which frozen per-env diagnostics are snapshotted.",
)
parser.add_argument("--contact-threshold-n", type=float, default=0.2, help="Per direct instruction: >0.2N on a tip counts as loaded.")
parser.add_argument(
    "--gain-regime", type=str, default="train", choices=["train", "eval", "finetune_eval"],
    help="'train' (default, recommended -- see module docstring's GAIN REGIME note) uses the soft"
    " implicit-actuator OSC gains every Stage-1 rollout trains under. 'eval'/'finetune_eval' use the"
    " Eval action group, which actions.py's own docstring documents as violating the kd*dt/I<=2"
    " rotational stability bound by 12-21x -- a real actuator confound, not a grasp-quality signal;"
    " only use these for an explicit gain-sensitivity follow-up, not as the primary read.",
)
parser.add_argument("--scratch-dir", type=str, default=None, help="Where the Resets/<pair>/ layout pointing at --bank-path is built. Defaults to a dir next to --bank-path.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, required=True, help="Summary JSON path.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below needs Isaac Sim modules, only importable after AppLauncher starts."""

import math  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.delto_cfg import (  # noqa: E402
    _apply_delto_dataset_dir,
)
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.ur5e_delto_cfg import (  # noqa: E402
    Ur5eDeltoRelCartesianOSCTrainCfg,
    Ur5eDeltoRelCartesianOSCEvalCfg,
    Ur5eDeltoRelCartesianOSCFinetuneEvalCfg,
)
from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils as omnireset_utils  # noqa: E402

# Reused, not reimplemented -- see module docstring's CONTACT SENSORS note.
from uwlab_tasks.manager_based.manipulation.dexlift.dexlift_ur10e_delto_env_cfg import (  # noqa: E402
    ALL_TIP_NAMES,
    HAND_PRIM,
    THUMB_TIP_NAMES,
    TIP_NAMES,
)
from uwlab_tasks.manager_based.manipulation.dexlift.mdp.rewards import _sensor_force_magnitudes  # noqa: E402

_GAIN_REGIME_CFG = {
    "train": Ur5eDeltoRelCartesianOSCTrainCfg,
    "eval": Ur5eDeltoRelCartesianOSCEvalCfg,
    "finetune_eval": Ur5eDeltoRelCartesianOSCFinetuneEvalCfg,
}

# Same mating-frame constants gen_ik_c4_reset_bank.py's _reproject_independent uses, re-verified
# against metadata.yaml on 2026-08-22 -- see that script for the full provenance. NOT used to
# recompute the true predicate here (ProgressContext, already wired into this env, does that with
# the SAME numbers via its own Offset objects -- see module docstring); used only for the SEPARATE,
# explicitly-labelled STRICT displacement comparator this script reports alongside contact grading.
_LEG_OFF_POS = (-0.106203, 0.0, 0.0)
_LEG_OFF_QUAT = (0.70710678, 0.0, 0.70710678, 0.0)
_RECV_OFF_POS = (-0.056250, 0.056250, -0.009374)
_RECV_OFF_QUAT = (1.0, 0.0, 0.0, 0.0)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stage_bank_into_scratch_layout(bank_path: str, scratch_dir: str, insertive_usd: str, receptive_usd: str) -> str:
    """Build a <scratch_dir>/Resets/<pair>/resets_ObjectPartiallyAssembledEEGrasped.pt layout
    MultiResetManager expects, pointing at (a copy of) --bank-path, WITHOUT touching --bank-path
    itself. ``pair`` is computed with the project's OWN ``compute_pair_dir`` (imported, not
    re-derived) against the ACTUAL usd paths the constructed env resolved for insertive/receptive_object
    -- not assumed from a string -- so this cannot silently drift from what the live scene will spawn.

    Resolves symlinks before copying (the bank path may be a convenience symlink into a dated
    generation-run directory) so the staged file is the real bytes, and verifies the copy's sha256
    matches the source before returning -- refuses to stage a corrupted copy silently.
    """
    real_bank_path = os.path.realpath(bank_path)
    if not os.path.isfile(real_bank_path):
        raise FileNotFoundError(f"--bank-path {bank_path!r} (resolved: {real_bank_path!r}) does not exist.")
    src_sha = _sha256(real_bank_path)

    pair = omnireset_utils.compute_pair_dir(insertive_usd, receptive_usd)
    dest_dir = os.path.join(scratch_dir, "Resets", pair)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, "resets_ObjectPartiallyAssembledEEGrasped.pt")
    if os.path.exists(dest_path):
        raise FileExistsError(
            f"{dest_path} already exists in the scratch layout -- refusing to overwrite (pass a fresh"
            " --scratch-dir; this project has been bitten by two files sharing a name before)."
        )
    shutil.copyfile(real_bank_path, dest_path)
    dest_sha = _sha256(dest_path)
    if dest_sha != src_sha:
        raise RuntimeError(f"Staged copy sha256 ({dest_sha}) != source sha256 ({src_sha}) -- refusing to use it.")
    print(f"[smoke] staged {real_bank_path} (sha256={src_sha[:16]}...) -> {dest_path}", flush=True)
    # Return the scratch ROOT, not dest_dir. The event term itself appends "Resets/<pair>/" when
    # it builds the bank filename, so handing it dest_dir produces a DOUBLED path segment
    # (<scratch>/Resets/<pair>/Resets/<pair>/resets_*.pt) and a FileNotFoundError that surfaces
    # later, and deceptively, as "MultiResetManager.__init__() got an unexpected keyword argument
    # 'dataset_dir'" -- this codebase swallows exceptions inside lazy event-term instantiation and
    # leaves term_cfg.func as the raw class.
    return scratch_dir


def _reproject_strict(insertive_root_pose: torch.Tensor, receptive_root_pose: torch.Tensor, device: str):
    """Same mating-frame reprojection as gen_ik_c4_reset_bank.py's _reproject_independent, in torch
    (batched over envs) rather than scipy/numpy, for the STRICT displacement comparator only (see
    module docstring -- the TRUE predicate is read from ProgressContext instead, not from this).
    """
    leg_off_pos = torch.tensor(_LEG_OFF_POS, device=device).expand(insertive_root_pose.shape[0], -1)
    leg_off_quat = torch.tensor(_LEG_OFF_QUAT, device=device).expand(insertive_root_pose.shape[0], -1)
    recv_off_pos = torch.tensor(_RECV_OFF_POS, device=device).expand(insertive_root_pose.shape[0], -1)
    recv_off_quat = torch.tensor(_RECV_OFF_QUAT, device=device).expand(insertive_root_pose.shape[0], -1)

    ins_pos, ins_quat = insertive_root_pose[:, :3], insertive_root_pose[:, 3:7]
    rec_pos, rec_quat = receptive_root_pose[:, :3], receptive_root_pose[:, 3:7]

    align_pos, align_quat = math_utils.combine_frame_transforms(ins_pos, ins_quat, leg_off_pos, leg_off_quat)
    target_pos, target_quat = math_utils.combine_frame_transforms(rec_pos, rec_quat, recv_off_pos, recv_off_quat)
    rel_pos, rel_quat = math_utils.compute_pose_error(target_pos, target_quat, align_pos, align_quat, rot_error_type="quat")

    depth_mm = 25.0 - rel_pos[:, 2] * 1000.0
    lateral_mm = torch.hypot(rel_pos[:, 0], rel_pos[:, 1]) * 1000.0
    e_x, e_y, _ = math_utils.euler_xyz_from_quat(rel_quat)
    tilt_rad = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
    tilt_deg = tilt_rad * 180.0 / math.pi
    return depth_mm, lateral_mm, tilt_deg


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg_cls = _GAIN_REGIME_CFG[args_cli.gain_regime]
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args_cli.n_envs
    env_cfg.sim.device = device

    # ---- point the scene at the leg/fixture pair, not the default peg/peg-hole (rl_state_cfg.py's
    # own Hydra-override convention, env.scene.object=leg200mm -- applied directly here since this
    # is a Python-constructed cfg, not a Hydra CLI run). ----
    env_cfg.scene.insertive_object = env_cfg.variants["scene.insertive_object"]["leg200mm"]
    env_cfg.scene.receptive_object = env_cfg.variants["scene.receptive_object"]["onelegfixture"]

    # ---- generous episode length so a TIMEOUT truncation never cuts off the last checkpoint --
    # a timeout AT OR AFTER --hold-seconds is a non-event for this measurement (see module
    # docstring); one landing BEFORE it would silently truncate the very data this script exists
    # to collect. ----
    env_cfg.episode_length_s = max(args_cli.hold_seconds + 0.5, env_cfg.episode_length_s)

    # ---- point reset_from_reset_states at ONLY this bank, 100% probability (default TrainEventCfg
    # is a 4-way C1/C2/C3/C4 mixture -- see rl_state_cfg.py's own "Reset mixture hides difficulty"
    # precedent in this project; a mixed draw here would silently test the WRONG reset type most of
    # the time). ----
    reset_term = env_cfg.events.reset_from_reset_states
    reset_term.params["reset_types"] = ["ObjectPartiallyAssembledEEGrasped"]
    reset_term.params["probs"] = [1.0]

    # ---- one-to-many contact sensor, leg as source, five DELTO fingertips as filters -- see
    # module docstring's CONTACT SENSORS note. Named "object_hand_s" so
    # dexlift.mdp.rewards._sensor_force_magnitudes's special-cased branch picks it up unmodified. ----
    env_cfg.scene.object_hand_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/{HAND_PRIM}/{tip}" for tip in ALL_TIP_NAMES],
    )

    # ---- resolve the ACTUAL usd paths this cfg will spawn (post variant-swap), and stage the bank
    # under test at the path MultiResetManager will look for -- never assumed from a literal string. ----
    scratch_dir = args_cli.scratch_dir or (os.path.dirname(os.path.abspath(args_cli.bank_path)) + "_smoke_scratch")
    dataset_dir = _stage_bank_into_scratch_layout(
        args_cli.bank_path, scratch_dir,
        env_cfg.scene.insertive_object.spawn.usd_path, env_cfg.scene.receptive_object.spawn.usd_path,
    )
    _apply_delto_dataset_dir(env_cfg, dataset_dir)
    # _apply_delto_dataset_dir repoints EVERY dataset_dir-bearing term, including any C1/C2/C3
    # bootstrap terms this class might carry for OTHER reset types -- harmless here since probs=[1.0]
    # only ever draws reset_from_reset_states's own single ObjectPartiallyAssembledEEGrasped path,
    # but note it explicitly rather than assume it silently does nothing.
    print(f"[smoke] reset_from_reset_states now points at dataset_dir={dataset_dir}, reset_types=['ObjectPartiallyAssembledEEGrasped'], probs=[1.0]", flush=True)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    n = env.num_envs
    obs, extras = env.reset()

    robot = env.scene["robot"]
    pc = env.reward_manager.get_term_cfg("progress_context").func  # ProgressContext instance, live

    # ---- SELF-CHECK: "zero action holds", asserted rather than assumed (see module docstring's
    # ZERO ACTION note). Compares the articulation's OWN joint_pos_target immediately after reset
    # against its value after ONE zero-action step. A large residual here means the relative-action
    # terms are NOT preserving the bank's recorded PD target the way this script's whole measurement
    # design assumes, and every downstream number should be distrusted until that is understood. ----
    _hand_ids = [i for i, nm in enumerate(robot.data.joint_names) if nm.startswith("rj_dg_")]
    assert len(_hand_ids) == 20, f"expected 20 rj_dg_* hand joints, found {len(_hand_ids)}"

    target_at_reset = robot.data.joint_pos_target.clone()

    # ---- DIAGNOSTIC: state AT RESET, before a single step is taken. --------------------------
    # Three runs disagreed with my model of the self-check residual (identical to 6 dp under a
    # zero action and under a servo action, which is impossible if the action drives the joints
    # the residual is measured over). Rather than iterate on the harness in the dark, measure the
    # thing the whole test exists to decide: DOES THE BANK HAVE A GRIP AT t=0, before any policy,
    # any action, any physics step? If it does not, holding is moot and the IK route fails on
    # grasp reproduction, not on gravity. If it does, the failure is in what happens after.
    _gap0 = (target_at_reset[:, _hand_ids] - robot.data.joint_pos[:, _hand_ids]).abs()
    print(
        f"[smoke] AT-RESET hand |stored_target - q|: median {_gap0.median().item():.4f}"
        f" p90 {_gap0.flatten().kthvalue(max(1, int(0.9 * _gap0.numel()))).values.item():.4f}"
        f" max {_gap0.max().item():.4f} rad",
        flush=True,
    )
    try:
        _thumb0 = _sensor_force_magnitudes(env, THUMB_TIP_NAMES)
        _tip0 = _sensor_force_magnitudes(env, TIP_NAMES)
        _thumb_on = (_thumb0 > 0.2).any(dim=-1)
        _tip_on = (_tip0 > 0.2).any(dim=-1)
        print(
            f"[smoke] AT-RESET opposed grip {(_thumb_on & _tip_on).float().mean().item():.4f}"
            f" | thumb-side loaded {_thumb_on.float().mean().item():.4f}"
            f" | finger-side loaded {_tip_on.float().mean().item():.4f}"
            f" | max thumb force {_thumb0.max().item():.3f} N"
            f" | max finger force {_tip0.max().item():.3f} N",
            flush=True,
        )
    except Exception as _e:  # noqa: BLE001 -- diagnostic only, must never kill the run
        print(f"[smoke] AT-RESET contact read failed ({_e!r}) -- continuing", flush=True)

    # ---- HOLD ACTION, not a zero action. ----------------------------------------------------
    # The hand is driven by RelativeJointPositionActionCfg with use_zero_offset=True, i.e.
    #     commanded_target = action * scale + MEASURED q
    # so a ZERO action commands target == q, which is ZERO PD ERROR and therefore ZERO GRIP
    # FORCE. Stepping a bank of grasps with zero actions makes the hand go limp BY CONSTRUCTION
    # and measures the harness dropping the object, not whether the grasp holds. (Observed:
    # hand target residual 2.157 rad against a 0.01 gate, held(contact) 0.000, depth -9.67 mm --
    # all of it the release, none of it the bank.)
    # To hold the recorded grasp we must SERVO THE STORED TARGET: action = (stored - q)/scale,
    # clipped to the term's own +-1. scale is 0.1 rad, and the recorded target-vs-q gap runs to
    # ~0.8 rad, so the fingers close over several steps rather than instantly -- which is the
    # real closing behaviour, not an artefact.
    _HAND_SCALE = 0.1
    _hand_act_ids: list[int] = []
    _hand_term = None
    _ofs = 0
    for _tname in env.action_manager.active_terms:
        _term = env.action_manager.get_term(_tname)
        _dim = _term.action_dim
        if "grip" in _tname.lower() or "hand" in _tname.lower():
            _hand_act_ids = list(range(_ofs, _ofs + _dim))
            _hand_term = _term
        _ofs += _dim
    assert len(_hand_act_ids) == 20, (
        f"expected a 20-dim hand action term, found {len(_hand_act_ids)};"
        f" active_terms={list(env.action_manager.active_terms)}"
    )

    # THE ACTION TERM'S DIM ORDER IS NOT THE ARTICULATION'S JOINT ORDER. find_joints() returns
    # whatever order the regex resolved to, and action dim k drives _joint_ids[k] -- so indexing
    # the gap by a separately-built rj_dg_* list silently servos the WRONG joints. That produced
    # a self-check residual identical to 6 decimal places across a zero-action run and a servo
    # run: the max-gap joint was never touched, while other joints moved enough to lift
    # held(contact) off zero. Take the ordering FROM THE TERM ITSELF.
    _term_joint_ids = _hand_term._joint_ids  # noqa: SLF001 -- no public accessor exists
    if isinstance(_term_joint_ids, slice):
        _term_joint_ids = list(range(len(robot.data.joint_names)))[_term_joint_ids]
    _term_joint_ids = list(_term_joint_ids)
    assert len(_term_joint_ids) == 20, f"hand term drives {len(_term_joint_ids)} joints, expected 20"
    assert set(_term_joint_ids) == set(_hand_ids), (
        "hand action term drives a different joint set than the rj_dg_* name match:"
        f" term={sorted(_term_joint_ids)} names={sorted(_hand_ids)}"
    )
    print(
        f"[smoke] hand action dim->joint map: {[robot.data.joint_names[j] for j in _term_joint_ids]}",
        flush=True,
    )

    def _hold_action() -> torch.Tensor:
        """Action that servos the hand back to its stored PD target; arm stays at zero (OSC hold)."""
        a = torch.zeros((n, env.action_manager.total_action_dim), device=env.device)
        gap = target_at_reset[:, _term_joint_ids] - robot.data.joint_pos[:, _term_joint_ids]
        a[:, _hand_act_ids] = (gap / _HAND_SCALE).clamp(-1.0, 1.0)
        return a

    # Step the SERVO action, not a zero action: the question that matters is whether the hold
    # action preserves the recorded target, since that is what the rollout below commands. A
    # zero-action probe here would both measure the wrong thing AND burn one limp-handed step
    # before the measurement starts, letting the object begin falling before t=0.
    env.step(_hold_action())
    target_after_one_step = robot.data.joint_pos_target.clone()
    delta = (target_after_one_step - target_at_reset).abs()

    # GATE ON THE HAND JOINTS ONLY. The arm is driven by RelCartesianOSC, i.e. EFFORT control --
    # set_joint_position_target is never called for it, so robot.data.joint_pos_target for the six
    # arm DOFs is an unused buffer that drifts freely and means nothing. Including it in the max
    # produced a 5.2 rad "failure" on a run whose hand targets were in fact held. The grasp lives
    # entirely in the 20 rj_dg_* hand joints; those are what this script's design depends on.
    _arm_ids = [i for i in range(len(robot.data.joint_names)) if i not in set(_hand_ids)]
    target_residual = delta[:, _hand_ids].max().item()
    arm_residual = delta[:, _arm_ids].max().item() if _arm_ids else 0.0
    print(
        f"[smoke] SELF-CHECK hold-action joint_pos_target residual after 1 step:"
        f" HAND {target_residual:.6f} rad (gated) | ARM {arm_residual:.6f} rad"
        f" (informational -- OSC/effort, target unused)",
        flush=True,
    )
    if target_residual > 0.01:
        print(
            "[smoke] WARNING: hold action did NOT preserve the reset-time joint_pos_target within"
            " 0.01 rad -- the 'zero action = hold' assumption this script's design depends on may be"
            " WRONG on this install. Every result below should be read with that in mind, not trusted"
            " blind.",
            flush=True,
        )

    dt = env.step_dt  # seconds per env.step() control step (physics_dt * decimation)
    checkpoints_s = sorted(float(v) for v in args_cli.checkpoints_s.split(","))
    checkpoint_steps = [max(1, round(t / dt)) for t in checkpoints_s]
    n_steps_total = checkpoint_steps[-1]
    print(f"[smoke] step_dt={dt:.5f}s, checkpoints at steps {checkpoint_steps} (~{checkpoints_s}s)", flush=True)

    # ---- t=0 baseline, read directly off the reward term / live scene right after reset (this
    # env's own ground truth for "requested" -- see module docstring's SAMPLE SELECTION note). ----
    ins_pose0 = torch.cat([env.scene["insertive_object"].data.root_pos_w, env.scene["insertive_object"].data.root_quat_w], dim=-1)
    rec_pose0 = torch.cat([env.scene["receptive_object"].data.root_pos_w, env.scene["receptive_object"].data.root_quat_w], dim=-1)
    depth0_mm, lateral0_mm, tilt0_deg = _reproject_strict(ins_pose0, rec_pose0, env.device)

    # ---- running (frozen-on-death) per-env buffers ----
    died_at = torch.full((n,), -1, dtype=torch.long, device=env.device)
    died_reason = ["alive"] * n
    buf_depth_mm = depth0_mm.clone()
    buf_lateral_mm = lateral0_mm.clone()
    buf_tilt_deg = tilt0_deg.clone()
    buf_true_pos_err_m = pc.xyz_distance.clone()
    buf_true_rot_err_rad = pc.euler_xy_distance.clone()
    buf_true_success = pc.success.clone()
    buf_held_contact = torch.zeros((n,), dtype=torch.bool, device=env.device)

    checkpoint_records = {t: None for t in checkpoints_s}
    next_checkpoint_idx = 0

    for step_idx in range(1, n_steps_total + 1):
        alive_before = died_at < 0
        _obs, _reward, terminated, truncated, extras = env.step(_hold_action())
        done = terminated | truncated

        # -- diagnostics for THIS step, read immediately, before any later step() can overwrite a
        # just-reset env's scene state (trap (1) -- see module docstring). Computed for every env,
        # written into the running buffer ONLY for envs alive at the START of this step. --
        ins_pose = torch.cat([env.scene["insertive_object"].data.root_pos_w, env.scene["insertive_object"].data.root_quat_w], dim=-1)
        rec_pose = torch.cat([env.scene["receptive_object"].data.root_pos_w, env.scene["receptive_object"].data.root_quat_w], dim=-1)
        depth_mm, lateral_mm, tilt_deg = _reproject_strict(ins_pose, rec_pose, env.device)
        thumb_force = _sensor_force_magnitudes(env, THUMB_TIP_NAMES)
        tip_force = _sensor_force_magnitudes(env, TIP_NAMES)
        held_contact = (thumb_force > args_cli.contact_threshold_n).any(dim=-1) & (tip_force > args_cli.contact_threshold_n).any(dim=-1)

        buf_depth_mm = torch.where(alive_before, depth_mm, buf_depth_mm)
        buf_lateral_mm = torch.where(alive_before, lateral_mm, buf_lateral_mm)
        buf_tilt_deg = torch.where(alive_before, tilt_deg, buf_tilt_deg)
        buf_true_pos_err_m = torch.where(alive_before, pc.xyz_distance, buf_true_pos_err_m)
        buf_true_rot_err_rad = torch.where(alive_before, pc.euler_xy_distance, buf_true_rot_err_rad)
        buf_true_success = torch.where(alive_before, pc.success, buf_true_success)
        buf_held_contact = torch.where(alive_before, held_contact, buf_held_contact)

        newly_died = done & alive_before
        if newly_died.any():
            died_at[newly_died] = step_idx
            for env_id in newly_died.nonzero(as_tuple=True)[0].tolist():
                died_reason[env_id] = _classify_termination(env, env_id)

        if next_checkpoint_idx < len(checkpoint_steps) and step_idx == checkpoint_steps[next_checkpoint_idx]:
            t = checkpoints_s[next_checkpoint_idx]
            checkpoint_records[t] = {
                "depth_mm": buf_depth_mm.clone(),
                "lateral_mm": buf_lateral_mm.clone(),
                "tilt_deg": buf_tilt_deg.clone(),
                "true_pos_err_m": buf_true_pos_err_m.clone(),
                "true_rot_err_rad": buf_true_rot_err_rad.clone(),
                "true_success": buf_true_success.clone(),
                "held_contact": buf_held_contact.clone(),
                "died_at": died_at.clone(),
            }
            print(f"[smoke] checkpoint t={t}s (step {step_idx}): {int((died_at >= 0).sum().item())}/{n} envs already died", flush=True)
            next_checkpoint_idx += 1

    # ================= REPORT =================
    def pct(a, ps=(0, 10, 25, 50, 75, 90, 100)):
        a = a.detach().cpu().numpy()
        return {p: float(np.percentile(a, p)) for p in ps}

    summary = {
        "bank_path": os.path.realpath(args_cli.bank_path),
        "n_envs": n, "gain_regime": args_cli.gain_regime, "contact_threshold_n": args_cli.contact_threshold_n,
        "zero_action_target_residual_rad": target_residual,
        "checkpoints": {},
    }

    reason_counts_total = {}
    for env_id, r in enumerate(died_reason):
        reason_counts_total[r] = reason_counts_total.get(r, 0) + 1

    print("\n=== per-checkpoint report ===", flush=True)
    for t in checkpoints_s:
        rec = checkpoint_records[t]
        depth_delta_mm = (rec["depth_mm"] - depth0_mm)
        held_frac_contact = float(rec["held_contact"].float().mean().item())
        strict_ok = (rec["lateral_mm"] < 15.0) & (rec["tilt_deg"] < 10.0)  # illustrative strict displacement comparator, see module docstring
        held_frac_strict = float(strict_ok.float().mean().item())
        true_predicate_frac = float(rec["true_success"].float().mean().item())
        died_by_then = int((rec["died_at"] >= 0).sum().item())

        cp_summary = {
            "died_fraction": died_by_then / n,
            "depth_delta_mm_pct": pct(depth_delta_mm),
            "lateral_mm_pct": pct(rec["lateral_mm"]),
            "tilt_deg_pct": pct(rec["tilt_deg"]),
            "held_fraction_contact_graded": held_frac_contact,
            "held_fraction_strict_displacement_20mm_class": held_frac_strict,
            "true_task3_predicate_fraction": true_predicate_frac,
        }
        summary["checkpoints"][str(t)] = cp_summary
        print(
            f"t={t:>4}s: died={died_by_then}/{n} "
            f"depth_delta_mm(median)={cp_summary['depth_delta_mm_pct'][50]:+.2f} "
            f"held(contact)={held_frac_contact:.3f} held(strict)={held_frac_strict:.3f} "
            f"true_predicate={true_predicate_frac:.3f}",
            flush=True,
        )

    summary["died_reason_counts"] = reason_counts_total
    print(f"\ndied_reason counts: {reason_counts_total}", flush=True)

    # ---- GO/NO-GO, per the criterion fixed in this script's module docstring BEFORE any result
    # was seen. Printed as data for a human to act on, not silently decided by the script. ----
    last_t = checkpoints_s[-1]
    last = summary["checkpoints"][str(last_t)]
    depth_delta_median = last["depth_delta_mm_pct"][50]
    held_contact_last = last["held_fraction_contact_graded"]
    criterion_a_triggered = depth_delta_median < -5.0
    criterion_b_triggered = held_contact_last < 0.70
    summary["go_no_go"] = {
        "criterion_a_depth_slip_over_5mm_median": criterion_a_triggered,
        "criterion_b_held_contact_under_0.70": criterion_b_triggered,
        "verdict": "NOT WORTH PURSUING AS-IS" if (criterion_a_triggered or criterion_b_triggered) else "PROCEED",
    }
    print(f"\n=== GO/NO-GO (criterion fixed a priori, see module docstring) ===\n{summary['go_no_go']}", flush=True)

    with open(args_cli.out, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[smoke] wrote {args_cli.out}", flush=True)

    env.close()


def _classify_termination(env, env_id: int) -> str:
    """Which configured termination term fired for this env this step, for died_reason bookkeeping.
    Checks 'success' first (a GOOD outcome, not a failure) before any failure-class term, since a
    state that was already stable can legitimately terminate via its own success gate.
    """
    for name in env.termination_manager.active_terms:
        try:
            fired = env.termination_manager.get_term(name)
        except (KeyError, AttributeError):
            continue
        if bool(fired[env_id].item()):
            return name
    return "unknown"


if __name__ == "__main__":
    main()
    simulation_app.close()
