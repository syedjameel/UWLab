#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared C4 (ObjectPartiallyAssembledEEGrasped / "near goal" / task_3) reset-bank validation harness.
Bead UWLab-xp05.5: PREREQUISITE for the whole deep-C4 epic (UWLab-xp05) -- neither the depth
measurement nor the acceptance test can be arm-specific, or arms 1-3 are not comparable and none of
them can be judged. Takes ANY C4 bank .pt and answers two questions:

  PART A -- MATING-FRAME DECOMPOSITION (no Isaac boot needed; pure numpy/scipy/torch on the bank's
  own tensors). Projects the leg tip into the fixture mating frame and reports depth / lateral / tilt
  per state, plus a self-check and a cross-check invariant BEFORE any number from it is trusted.

  PART B -- DYNAMIC HOLD TEST (needs a real Isaac boot). Resets each of --n-envs banked states, holds
  the STORED PD targets, steps REAL physics, and reads fraction-still-held (contact-graded) and depth
  delta at 0.1/0.5/1.0/2.0s. PASS = held>=0.70 at 2s AND depth delta within 5mm.

WHY ROOT-TO-ROOT DISTANCE IS NEVER USED FOR SEATING (Part A's central design decision). It mixes
axial withdrawal with lateral displacement and ignores tilt entirely -- a leg nudged sideways while
seated and a leg backed half out of the bore can read the SAME root-to-root number. This project has
already been burned by it once: a run read median root-to-root 143.3mm against a 131-136mm spawn
baseline, looked seated, and the SAME states decomposed to depth MEDIAN -0.59mm with 60% of states at
NEGATIVE depth -- a false green that only the mating-frame projection caught. This script computes
root-to-root ONLY as a clearly-labelled INFORMATIONAL side-by-side column, never as a pass/fail input,
specifically so that contrast stays visible rather than being designed away.

PART A's SELF-CHECK, PRINTED BEFORE ANY OTHER PART-A NUMBER: synthesizes a leg pose placed EXACTLY at
the assembled pose (the fixture's mating frame and the leg's mating frame made to coincide exactly, in
closed form, not sampled) and asserts the SAME reprojection function used on real bank states scores
0.00e+00 (to float precision) on both position and rotation, and exactly 25.00mm depth. If this does
not hold, every depth/lateral/tilt number below is void and the script stops rather than printing them.

PART A's CROSS-CHECK INVARIANT, verified on real bank data, not just the self-check pose: since
>99.7% of the reprojected position error is axial for states near the mating axis, pos_err_mm ~=
25.0 - depth_mm should hold with R^2 close to 0.999999 on any bank whose states are actually near the
bore axis (a loosely-scattered bank, e.g. C1/C2, would legitimately show a lower R^2 here -- this is
reported as a diagnostic, not gated pass/fail, since it depends on how axis-aligned the bank already
is).

PART B REUSES, RATHER THAN REIMPLEMENTS, smoke_test_ik_c4_holding.py's hold-action servo logic and
at-reset diagnostics -- that script is ~90% of this part and is already patched through four harness
defects that each independently produced a confident FALSE "not viable" verdict on a real bank. Every
one is re-asserted here, not just carried over silently:

  1. SELF-CHECK GATES ON THE 20 rj_dg_* HAND JOINTS ONLY. The arm is RelCartesianOSC (EFFORT control)
     -- set_joint_position_target is never called for it, so robot.data.joint_pos_target for the six
     arm DOFs is an unused buffer that drifts freely and means nothing. Including it in a residual
     max produced a spurious 5.201 rad "failure" on a bank whose HAND targets were in fact held.
  2. NEVER STEP WITH A ZERO ACTION. The hand is RelativeJointPositionActionCfg with
     use_zero_offset=True: commanded_target = action*scale + MEASURED q. A zero action therefore
     commands target == q -- zero PD error, ZERO GRIP FORCE -- and reads held=0 for ANY bank,
     correct or not. This script servos the stored target every step: action = clamp((stored_target
     - q) / scale, -1, 1).
  3. ACTION DIM ORDER IS NOT ARTICULATION JOINT ORDER. find_joints() returns the regex's own
     ordering, interleaved by phalanx level (1_1, 2_1, 3_1, 4_1, 5_1, 1_2, ...) -- action dim k drives
     the action TERM's OWN _joint_ids[k], not a separately name-matched list. Indexing by the wrong
     list produced a self-check residual IDENTICAL TO SIX DECIMALS under a zero action and a servo
     action -- an action-independent residual that is impossible if the action actually drives the
     measured joints, which is exactly what exposed this bug. Asserted here: the action-term's own
     _joint_ids set must equal the rj_dg_* name-matched set.
  4. TMPDIR, NOT JUST UWLAB_TMP_ROOT. /tmp/isaaclab is owned by another uid like /tmp/uwlab, and
     UWLAB_TMP_ROOT does not cover it -- IsaacLab's own logger uses tempfile.gettempdir(). This is an
     operational (env-var) requirement on the CALLER, not something this script can enforce from
     inside Python once already dying; see the run command below.

UWLab-xp05.9 FOLLOW-UP, RESOLVED: the inherited defect-3 self-check (comparing the target BUFFER
before vs after one full env.step()) fires a warning on essentially every real bank, because
RelativeJointPositionAction.apply_actions() -- confirmed by reading joint_actions.py directly --
recomputes `processed_actions + LIVE q` on EVERY PHYSICS SUBSTEP, so that buffer is SUPPOSED to move
whenever q moves, which is not a defect. An A/B (same seed/bank/n-envs, shipped C4 bank, --self-check
-action zero vs servo) confirmed the residual is NOT action-independent (0.1725 rad zero vs 0.0786 rad
servo, median) -- ruling out defect 3 -- but also showed the check's threshold was simply wrong. Fixed
by replacing it with an ISOLATED, pre-physics check: read the action term's own `.processed_actions`
property immediately after `process_action()` (which never reads q -- see joint_actions.py's
`JointAction.process_actions`) and compare against this script's own closed-form
clamp(gap,-scale,scale) prediction. That check is a pure test of the action MECHANISM, decoupled from
the plant, and gates (asserts near-zero) rather than warns; the old full-step comparison is kept
alongside as an explicitly-labelled INFORMATIONAL diagnostic only, not a gate. See run_part_b's inline
comments for the exact code.

MANDATORY, ON EVERY BANK, BEFORE ANY HOLD NUMBER IS QUOTED: |stored_target - q| over the hand joints,
read immediately at reset, before a single step. A genuine policy-generated bank commands the target
AHEAD of q (C3 reference: 0.1138 rad median, 0.3067 p90). A bank reading 0.0000 median is commanding
ZERO SQUEEZE at reset and its geometry is irrelevant regardless of what Part A says -- this is what
finally explained why a geometrically perfect IK bank held nothing. Printed and stored in the JSON
BEFORE the checkpoint table, not buried after it.

GRADING IS BY CONTACT FORCE, NEVER IN-PALM DISPLACEMENT. Opposed grip = a thumb-side tip (rl_dg_1 or
rl_dg_5) AND at least one of rl_dg_2/3/4, each above --contact-threshold-n (default 0.2N), read
through dexlift's own ``_sensor_force_magnitudes`` (reused, not reimplemented, so this cannot drift
from the reward/gate logic the rest of the project trusts). Justification, measured at n=298: contact
grading read 0.9597 against a strict 20mm-displacement metric's 0.5369, and the disagreement was
ONE-SIDED (131 contact-only vs 5 held-only) -- displacement grading was punishing legitimate in-hand
repositioning as failure. A strict displacement comparator is reported alongside anyway, purely so a
reader can see directly whether that same disagreement reappears here.

Run (Part A only, no GPU/Isaac needed):
    /home/dom_iva/venv_uwlab/bin/python scripts_v2/tools/validate_c4_bank.py \\
        --bank-path <bank>.pt --part a --n-states-a 256 --out <out>.json

Run (Part A + Part B, needs Isaac; TMPDIR/UWLAB_TMP_ROOT and CUDA_VISIBLE_DEVICES must be exported
by the caller -- see module docstring's item 4 above and this project's DL_H100 operating rules):
    UWLAB_TMP_ROOT=/home/dom_iva/tmp TMPDIR=/home/dom_iva/tmp CUDA_VISIBLE_DEVICES=1 \\
        timeout -s KILL 900 /home/dom_iva/venv_uwlab/bin/python -u scripts_v2/tools/validate_c4_bank.py \\
        --bank-path <bank>.pt --part both --n-states-a 256 --n-envs 128 --hold-seconds 2.0 \\
        --out <out>.json --headless
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil

import numpy as np
import torch

# ============================================================================================
# Bank schema validation -- cheap, pure-python, runs unconditionally before either part. Mirrors
# the epic's (UWLab-xp05) "standard bank validation" bullet: rigid_object keys correct for C4 (exactly
# two: insertive_object, receptive_object -- never rekey, per MultiResetManager's
# assumed_static_assets convention), joint_position_target present, field lengths equal.
# ============================================================================================


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_bank_schema(raw: dict, bank_path: str) -> dict:
    report = {"ok": True, "errors": [], "n_states": None}

    def fail(msg):
        report["ok"] = False
        report["errors"].append(msg)

    if "initial_state" not in raw:
        fail("top-level dict has no 'initial_state' key")
        return report
    init = raw["initial_state"]

    if "rigid_object" not in init:
        fail("'initial_state' has no 'rigid_object' key")
    else:
        keys = set(init["rigid_object"].keys())
        if keys != {"insertive_object", "receptive_object"}:
            fail(f"rigid_object keys = {sorted(keys)}, expected exactly {{'insertive_object', 'receptive_object'}} (never rekey)")

    if "articulation" not in init or "robot" not in init.get("articulation", {}):
        fail("'initial_state.articulation.robot' missing")
    else:
        robot = init["articulation"]["robot"]
        if "joint_position_target" not in robot:
            fail("'initial_state.articulation.robot.joint_position_target' missing")

    lengths = {}
    for group_name, group in init.items():
        for obj_name, obj in group.items():
            for field_name, field in obj.items():
                lengths[f"{group_name}.{obj_name}.{field_name}"] = len(field)
    if lengths:
        n_vals = set(lengths.values())
        if len(n_vals) != 1:
            fail(f"field lengths disagree: {lengths}")
        else:
            report["n_states"] = next(iter(n_vals))

    print(f"[validate] schema check ({bank_path}): {'OK' if report['ok'] else 'FAILED'} -- n_states={report['n_states']}", flush=True)
    if not report["ok"]:
        for e in report["errors"]:
            print(f"[validate]   ERROR: {e}", flush=True)
    return report


# ============================================================================================
# PART A -- mating-frame decomposition. Pure numpy/scipy, NO isaaclab import, so this half of the
# tool runs with zero GPU/Isaac boot cost. Constants and the reprojection itself are copied verbatim
# from gen_ik_c4_reset_bank.py's `_reproject_independent` (re-verified there against metadata.yaml on
# 2026-08-22) -- NOT re-derived, so this cannot silently drift from the generator's own math.
# ============================================================================================

_LEG_OFF_POS = np.array([-0.106203, 0.0, 0.0])
_LEG_OFF_QUAT_WXYZ = np.array([0.70710678, 0.0, 0.70710678, 0.0])
_RECV_OFF_POS = np.array([-0.056250, 0.056250, -0.009374])
_RECV_OFF_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[..., [1, 2, 3, 0]]


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[..., [3, 0, 1, 2]]


def reproject_mating_frame(insertive_root_pose: np.ndarray, receptive_root_pose: np.ndarray) -> dict:
    """Project the leg tip into the fixture mating frame. Returns depth/lateral/tilt (mm/mm/deg)
    plus pos_err_mm/rot_err_deg (the TRUE task_3 predicate's own error terms) and root_to_root_mm
    (informational only -- see module docstring's ROOT-TO-ROOT note; never used for pass/fail).

    rot_err_deg uses e_x, e_y ONLY (e_z / yaw DISCARDED), matching ProgressContext's own predicate
    exactly (omnireset/mdp/rewards.py) -- this is deliberately NOT the naive full-quaternion angle.
    """
    from scipy.spatial.transform import Rotation

    n = insertive_root_pose.shape[0]
    ins_pos, ins_quat_wxyz = insertive_root_pose[:, :3], insertive_root_pose[:, 3:7]
    rec_pos, rec_quat_wxyz = receptive_root_pose[:, :3], receptive_root_pose[:, 3:7]

    R_ins = Rotation.from_quat(_wxyz_to_xyzw(ins_quat_wxyz))
    R_leg_off = Rotation.from_quat(_wxyz_to_xyzw(np.tile(_LEG_OFF_QUAT_WXYZ, (n, 1))))
    align_pos = ins_pos + R_ins.apply(np.tile(_LEG_OFF_POS, (n, 1)))
    R_align = R_ins * R_leg_off

    R_rec = Rotation.from_quat(_wxyz_to_xyzw(rec_quat_wxyz))
    R_recv_off = Rotation.from_quat(_wxyz_to_xyzw(np.tile(_RECV_OFF_QUAT_WXYZ, (n, 1))))
    target_pos = rec_pos + R_rec.apply(np.tile(_RECV_OFF_POS, (n, 1)))
    R_target = R_rec * R_recv_off

    R_target_inv = R_target.inv()
    rel_pos = R_target_inv.apply(align_pos - target_pos)
    R_rel = R_target_inv * R_align

    depth_mm = 25.0 - rel_pos[:, 2] * 1000.0
    lateral_mm = np.hypot(rel_pos[:, 0], rel_pos[:, 1]) * 1000.0
    pos_err_mm = np.linalg.norm(rel_pos, axis=1) * 1000.0
    euler_xyz = R_rel.as_euler("xyz", degrees=True)

    def wrap(a):
        return (a + 180.0) % 360.0 - 180.0

    rot_err_deg = np.abs(wrap(euler_xyz[:, 0])) + np.abs(wrap(euler_xyz[:, 1]))  # e_x, e_y only; e_z discarded
    tilt_deg = np.degrees(np.arccos(np.clip(R_rel.as_matrix()[:, 2, 2], -1.0, 1.0)))
    root_to_root_mm = np.linalg.norm(ins_pos - rec_pos, axis=1) * 1000.0  # INFORMATIONAL ONLY -- see docstring

    return {
        "depth_mm": depth_mm, "lateral_mm": lateral_mm, "tilt_deg": tilt_deg,
        "pos_err_mm": pos_err_mm, "rot_err_deg": rot_err_deg, "root_to_root_mm": root_to_root_mm,
    }


def _mating_frame_self_check() -> dict:
    """Synthesizes a leg pose placed EXACTLY at the assembled pose (closed-form, not sampled) against
    a NON-TRIVIAL receptive pose (not identity -- a self-check that only passes at the origin would be
    too weak), and asserts reproject_mating_frame scores 0.00e+00 on pos_err_mm/rot_err_deg and
    exactly 25.00mm depth. MUST be called, and must PASS, before any other Part-A number is trusted.
    """
    from scipy.spatial.transform import Rotation

    rec_pos = np.array([[0.05, -0.03, 0.02]])
    rec_quat_wxyz = _xyzw_to_wxyz(Rotation.from_euler("xyz", [11.0, -19.0, 34.0], degrees=True).as_quat())[None, :]

    R_rec = Rotation.from_quat(_wxyz_to_xyzw(rec_quat_wxyz[0]))
    R_recv_off = Rotation.from_quat(_wxyz_to_xyzw(_RECV_OFF_QUAT_WXYZ))
    target_pos = rec_pos[0] + R_rec.apply(_RECV_OFF_POS)
    R_target = R_rec * R_recv_off

    R_leg_off = Rotation.from_quat(_wxyz_to_xyzw(_LEG_OFF_QUAT_WXYZ))
    R_ins = R_target * R_leg_off.inv()  # so that R_ins * R_leg_off == R_target exactly
    ins_pos = target_pos - R_ins.apply(_LEG_OFF_POS)  # so that ins_pos + R_ins.apply(leg_off) == target_pos exactly
    ins_quat_wxyz = _xyzw_to_wxyz(R_ins.as_quat())[None, :]

    ins_pose = np.concatenate([ins_pos[None, :], ins_quat_wxyz], axis=1)
    rec_pose = np.concatenate([rec_pos, rec_quat_wxyz], axis=1)
    out = reproject_mating_frame(ins_pose, rec_pose)

    pos_err_mm = float(out["pos_err_mm"][0])
    rot_err_deg = float(out["rot_err_deg"][0])
    depth_mm = float(out["depth_mm"][0])
    print(
        f"[validate] PART-A SELF-CHECK (leg placed exactly at assembled pose): "
        f"pos_err={pos_err_mm:.2e} mm, rot_err={rot_err_deg:.2e} deg, depth={depth_mm:.6f} mm (expect 25.000000)",
        flush=True,
    )
    ok = pos_err_mm < 1e-3 and rot_err_deg < 1e-3 and abs(depth_mm - 25.0) < 1e-3
    if not ok:
        raise AssertionError(
            "PART-A SELF-CHECK FAILED: a leg placed exactly at the assembled pose did not score "
            "~0.00e+00 on position/rotation error and 25.00mm depth. STOPPING -- every depth/lateral/"
            "tilt number this script would otherwise print is void until this is understood."
        )
    print("[validate] PART-A SELF-CHECK PASSED. Proceeding to real bank states.", flush=True)
    return {"pos_err_mm": pos_err_mm, "rot_err_deg": rot_err_deg, "depth_mm": depth_mm, "passed": ok}


def _pct(a: np.ndarray, ps=(0, 25, 50, 75, 100)) -> dict:
    return {str(p): float(np.percentile(a, p)) for p in ps}


def run_part_a(bank_path: str, raw: dict, n_states_a: int | None, seed: int) -> dict:
    self_check = _mating_frame_self_check()

    init = raw["initial_state"]
    ins_all = torch.stack(init["rigid_object"]["insertive_object"]["root_pose"]).numpy()
    rec_all = torch.stack(init["rigid_object"]["receptive_object"]["root_pose"]).numpy()
    n_total = ins_all.shape[0]

    if n_states_a is not None and n_states_a < n_total:
        g = np.random.default_rng(seed)
        idx = g.choice(n_total, size=n_states_a, replace=False)
        idx.sort()
    else:
        idx = np.arange(n_total)
    ins, rec = ins_all[idx], rec_all[idx]
    n = ins.shape[0]

    out = reproject_mating_frame(ins, rec)
    depth_mm, lateral_mm, tilt_deg = out["depth_mm"], out["lateral_mm"], out["tilt_deg"]
    pos_err_mm, rot_err_deg, root_to_root_mm = out["pos_err_mm"], out["rot_err_deg"], out["root_to_root_mm"]

    # ---- cross-check invariant: pos_err_mm ~= 25.0 - depth_mm, R^2 close to 0.999999 on a bank
    # whose states are near the bore axis (see module docstring -- diagnostic, not gated). ----
    predicted = 25.0 - depth_mm
    ss_res = float(np.sum((pos_err_mm - predicted) ** 2))
    ss_tot = float(np.sum((pos_err_mm - pos_err_mm.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    band_frac = float(np.mean((depth_mm >= 12.0) & (depth_mm <= 25.0)))
    true_pred_frac = float(np.mean((pos_err_mm < 2.5) & (rot_err_deg < math.degrees(0.025))))

    result = {
        "self_check": self_check,
        "n_total_in_bank": int(n_total),
        "n_sampled": int(n),
        "sample_seed": seed,
        "depth_mm_pct": _pct(depth_mm),
        "lateral_mm_pct": {"max": float(lateral_mm.max()), **_pct(lateral_mm)},
        "tilt_deg_pct": {"max": float(tilt_deg.max()), **_pct(tilt_deg)},
        "root_to_root_mm_pct_INFORMATIONAL_ONLY": _pct(root_to_root_mm),
        "depth_band_12_25mm_fraction": band_frac,
        "true_task3_predicate_fraction_2p5mm_0p025rad": true_pred_frac,
        "cross_check_invariant_pos_err_eq_25_minus_depth": {"r2": r2, "n": int(n)},
    }

    print(f"\n=== PART A: mating-frame decomposition ({bank_path}) ===", flush=True)
    print(f"[validate] n_total_in_bank={n_total}  n_sampled={n} (seed={seed})", flush=True)
    print(f"[validate] depth_mm percentiles (0/25/50/75/100): {result['depth_mm_pct']}", flush=True)
    print(f"[validate] lateral_mm max={lateral_mm.max():.3f}  percentiles: {result['lateral_mm_pct']}", flush=True)
    print(f"[validate] tilt_deg max={tilt_deg.max():.3f}  percentiles: {result['tilt_deg_pct']}", flush=True)
    print(f"[validate] root_to_root_mm (INFORMATIONAL ONLY, NOT a seating measurement): {result['root_to_root_mm_pct_INFORMATIONAL_ONLY']}", flush=True)
    print(f"[validate] fraction in [12,25]mm depth band: {band_frac:.4f}", flush=True)
    print(f"[validate] fraction meeting TRUE task_3 predicate (2.5mm/0.025rad): {true_pred_frac:.4f}", flush=True)
    print(f"[validate] cross-check invariant pos_err_mm ~= 25.0-depth_mm: R^2={r2:.6f} (n={n}; expect close to 0.999999 for an axis-aligned bank)", flush=True)
    return result


# ============================================================================================
# PART B -- dynamic hold test. Needs a real Isaac boot; everything below this point only executes
# when --part is "b" or "both". Structurally follows smoke_test_ik_c4_holding.py's main() (see that
# script and this module's docstring for full provenance of every defect-guard here) generalized to
# take a --gain-regime-agnostic bank path and to also emit the mandatory at-reset gap check into JSON.
# ============================================================================================


def _stage_bank_into_scratch_layout(bank_path: str, scratch_dir: str, insertive_usd: str, receptive_usd: str, omnireset_utils) -> str:
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
            f"{dest_path} already exists in the scratch layout -- refusing to overwrite (pass a fresh --scratch-dir)."
        )
    shutil.copyfile(real_bank_path, dest_path)
    dest_sha = _sha256(dest_path)
    if dest_sha != src_sha:
        raise RuntimeError(f"Staged copy sha256 ({dest_sha}) != source sha256 ({src_sha}) -- refusing to use it.")
    print(f"[validate] staged {real_bank_path} (sha256={src_sha[:16]}...) -> {dest_path}", flush=True)
    return scratch_dir  # scratch ROOT, not dest_dir -- the event term appends "Resets/<pair>/" itself


def run_part_b(args_cli, env_module_globals: dict) -> dict:
    """env_module_globals carries the isaaclab/uwlab_tasks symbols imported after AppLauncher boots
    (torch, math_utils, ManagerBasedRLEnv, ContactSensorCfg, gain-regime cfg classes, dexlift names,
    _sensor_force_magnitudes, omnireset_utils, _apply_delto_dataset_dir) -- passed in rather than
    imported at module scope, since they are only importable once the Isaac app has started.
    """
    math_utils = env_module_globals["math_utils"]
    ManagerBasedRLEnv = env_module_globals["ManagerBasedRLEnv"]
    ContactSensorCfg = env_module_globals["ContactSensorCfg"]
    gain_regime_cfg = env_module_globals["gain_regime_cfg"]
    omnireset_utils = env_module_globals["omnireset_utils"]
    _apply_delto_dataset_dir = env_module_globals["_apply_delto_dataset_dir"]
    ALL_TIP_NAMES = env_module_globals["ALL_TIP_NAMES"]
    HAND_PRIM = env_module_globals["HAND_PRIM"]
    THUMB_TIP_NAMES = env_module_globals["THUMB_TIP_NAMES"]
    TIP_NAMES = env_module_globals["TIP_NAMES"]
    _sensor_force_magnitudes = env_module_globals["_sensor_force_magnitudes"]

    def _reproject_strict(insertive_root_pose, receptive_root_pose, device):
        # NOTE: _LEG_OFF_POS etc. are numpy float64 arrays (shared with Part A's numpy/scipy path) --
        # torch.tensor() on a tuple of numpy float64 scalars infers torch.float64 ("Double"), which
        # then crashes deep inside isaaclab's TorchScript quat_apply with "Found dtype Double but
        # expected Float" once combined with the (float32) live scene poses. dtype=torch.float32 is
        # explicit here so this torch path stays float32 regardless of the constants' numpy dtype.
        leg_off_pos = torch.tensor(tuple(_LEG_OFF_POS), device=device, dtype=torch.float32).expand(insertive_root_pose.shape[0], -1)
        leg_off_quat = torch.tensor(tuple(_LEG_OFF_QUAT_WXYZ), device=device, dtype=torch.float32).expand(insertive_root_pose.shape[0], -1)
        recv_off_pos = torch.tensor(tuple(_RECV_OFF_POS), device=device, dtype=torch.float32).expand(insertive_root_pose.shape[0], -1)
        recv_off_quat = torch.tensor(tuple(_RECV_OFF_QUAT_WXYZ), device=device, dtype=torch.float32).expand(insertive_root_pose.shape[0], -1)

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

    def _classify_termination(env, env_id: int) -> str:
        for name in env.termination_manager.active_terms:
            try:
                fired = env.termination_manager.get_term(name)
            except (KeyError, AttributeError):
                continue
            if bool(fired[env_id].item()):
                return name
        return "unknown"

    device = args_cli.device if args_cli.device is not None else "cuda:0"
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = gain_regime_cfg()
    env_cfg.scene.num_envs = args_cli.n_envs
    env_cfg.sim.device = device
    env_cfg.scene.insertive_object = env_cfg.variants["scene.insertive_object"]["leg200mm"]
    env_cfg.scene.receptive_object = env_cfg.variants["scene.receptive_object"]["onelegfixture"]
    env_cfg.episode_length_s = max(args_cli.hold_seconds + 0.5, env_cfg.episode_length_s)

    reset_term = env_cfg.events.reset_from_reset_states
    reset_term.params["reset_types"] = ["ObjectPartiallyAssembledEEGrasped"]
    reset_term.params["probs"] = [1.0]

    env_cfg.scene.object_hand_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/{HAND_PRIM}/{tip}" for tip in ALL_TIP_NAMES],
    )

    scratch_dir = args_cli.scratch_dir or (os.path.dirname(os.path.abspath(args_cli.bank_path)) + "_validate_scratch")
    dataset_dir = _stage_bank_into_scratch_layout(
        args_cli.bank_path, scratch_dir,
        env_cfg.scene.insertive_object.spawn.usd_path, env_cfg.scene.receptive_object.spawn.usd_path,
        omnireset_utils,
    )
    _apply_delto_dataset_dir(env_cfg, dataset_dir)
    print(f"[validate] reset_from_reset_states -> dataset_dir={dataset_dir}, reset_types=['ObjectPartiallyAssembledEEGrasped'], probs=[1.0]", flush=True)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    n = env.num_envs
    obs, extras = env.reset()

    robot = env.scene["robot"]
    pc = env.reward_manager.get_term_cfg("progress_context").func

    # ---- DEFECT 1 GUARD: gate the self-check on the 20 rj_dg_* hand joints only. ----
    _hand_ids = [i for i, nm in enumerate(robot.data.joint_names) if nm.startswith("rj_dg_")]
    assert len(_hand_ids) == 20, f"expected 20 rj_dg_* hand joints, found {len(_hand_ids)}"

    target_at_reset = robot.data.joint_pos_target.clone()

    # ---- MANDATORY CHECK, before any hold number: |stored_target - q| at reset, hand joints only. ----
    _gap0 = (target_at_reset[:, _hand_ids] - robot.data.joint_pos[:, _hand_ids]).abs()
    gap0_median = float(_gap0.median().item())
    gap0_p90 = float(_gap0.flatten().kthvalue(max(1, int(0.9 * _gap0.numel()))).values.item())
    gap0_max = float(_gap0.max().item())
    print(
        f"[validate] MANDATORY at-reset hand |stored_target - q|: median {gap0_median:.4f} p90 {gap0_p90:.4f} max {gap0_max:.4f} rad"
        f" (C3 reference: median 0.1138, p90 0.3067 -- a bank reading ~0.0000 median is commanding ZERO SQUEEZE at reset)",
        flush=True,
    )

    at_reset_contact = {}
    try:
        _thumb0 = _sensor_force_magnitudes(env, THUMB_TIP_NAMES)
        _tip0 = _sensor_force_magnitudes(env, TIP_NAMES)
        _thumb_on = (_thumb0 > args_cli.contact_threshold_n).any(dim=-1)
        _tip_on = (_tip0 > args_cli.contact_threshold_n).any(dim=-1)
        at_reset_contact = {
            "opposed_grip_fraction": float((_thumb_on & _tip_on).float().mean().item()),
            "thumb_side_loaded_fraction": float(_thumb_on.float().mean().item()),
            "finger_side_loaded_fraction": float(_tip_on.float().mean().item()),
        }
        print(f"[validate] AT-RESET contact: {at_reset_contact}", flush=True)
    except Exception as e:  # noqa: BLE001 -- diagnostic only
        print(f"[validate] AT-RESET contact read failed ({e!r}) -- continuing", flush=True)

    # ---- DEFECT 2/3 GUARD: hold action must servo the stored target using the ACTION TERM's own
    # _joint_ids, never a separately name-matched list. ----
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
    assert len(_hand_act_ids) == 20, f"expected a 20-dim hand action term, found {len(_hand_act_ids)}; active_terms={list(env.action_manager.active_terms)}"

    _term_joint_ids = _hand_term._joint_ids  # noqa: SLF001 -- no public accessor exists
    if isinstance(_term_joint_ids, slice):
        _term_joint_ids = list(range(len(robot.data.joint_names)))[_term_joint_ids]
    _term_joint_ids = list(_term_joint_ids)
    assert len(_term_joint_ids) == 20, f"hand term drives {len(_term_joint_ids)} joints, expected 20"
    assert set(_term_joint_ids) == set(_hand_ids), (
        f"hand action term drives a different joint set than the rj_dg_* name match: term={sorted(_term_joint_ids)} names={sorted(_hand_ids)}"
    )
    print(f"[validate] hand action dim->joint map: {[robot.data.joint_names[j] for j in _term_joint_ids]}", flush=True)

    def _hold_action():
        a = torch.zeros((n, env.action_manager.total_action_dim), device=env.device)
        gap = target_at_reset[:, _term_joint_ids] - robot.data.joint_pos[:, _term_joint_ids]
        a[:, _hand_act_ids] = (gap / _HAND_SCALE).clamp(-1.0, 1.0)
        return a

    def _zero_action():
        return torch.zeros((n, env.action_manager.total_action_dim), device=env.device)

    def _pct_stats(t: torch.Tensor) -> dict:
        flat = t.flatten()
        return {
            "median": float(flat.median().item()),
            "p90": float(flat.kthvalue(max(1, int(0.9 * flat.numel()))).values.item()),
            "max": float(flat.max().item()),
        }

    # ---- UWLab-xp05.9 ISOLATED SELF-CHECK (the real gate). Reads IsaacLab source directly rather
    # than re-deriving from memory: ActionManager.process_action() calls each term's own
    # process_actions(raw) -> processed = raw*scale + offset (clipped), a PURE function of the
    # submitted action with ZERO joint_pos read anywhere in it. RelativeJointPositionAction.apply_actions()
    # is what mixes in the plant -- it does `processed_actions + asset.data.joint_pos[:, joint_ids]`,
    # and IS called once per PHYSICS SUBSTEP (ActionManager.apply_action() docstring: "called at every
    # simulation step"), always re-reading the LIVE (moving) q. So `robot.data.joint_pos_target` after
    # a full env.step() is, by design, `processed_actions + q_at_the_LAST_substep`, not `+ q_at_reset`
    # -- it is supposed to move every substep as q drifts, which is exactly what the OLD self-check
    # below (comparing the target buffer before vs after a full step) was punishing as a "failure".
    # Confirmed via the UWLab-xp05.9 A/B (same seed/bank/n-envs, shipped C4 bank): full-step residual
    # median 0.0786 rad under a servo action vs 0.1725 rad under a zero action -- DIFFERENT, so this
    # was never defect 3 (an action-independent residual), it was the wrong invariant.
    # This isolated check instead reads the term's own PUBLIC `.processed_actions` property
    # IMMEDIATELY after process_action(), before apply_action()/any physics has run at all -- i.e.
    # before q can have moved -- and compares it to OUR OWN closed-form prediction of what that
    # pipeline should produce for the action we submitted. This is a pure test of the ACTION
    # MECHANISM (did our --HAND_SCALE/clamp assumption match the term's REAL cfg.scale/offset/clip),
    # with the plant held out of the loop entirely -- not a tautology: it WOULD fail if cfg.scale
    # were not the flat 0.1 float this script assumes, or if cfg.clip added an unaccounted-for bound.
    _actual_scale = _hand_term._scale  # noqa: SLF001 -- no public accessor exists
    if isinstance(_actual_scale, torch.Tensor):
        _scale_ok = bool((_actual_scale == _HAND_SCALE).all().item())
        _scale_repr = f"tensor min={_actual_scale.min().item():.6f} max={_actual_scale.max().item():.6f}"
    else:
        _scale_ok = abs(float(_actual_scale) - _HAND_SCALE) < 1e-9
        _scale_repr = f"{float(_actual_scale):.6f}"
    print(
        f"[validate] hand action term's ACTUAL cfg.scale = {_scale_repr} (this script assumes _HAND_SCALE={_HAND_SCALE})"
        f" -> {'MATCH' if _scale_ok else 'MISMATCH'}",
        flush=True,
    )
    assert _scale_ok, (
        f"hand action term's cfg.scale ({_scale_repr}) does not match this script's _HAND_SCALE={_HAND_SCALE} "
        "assumption -- the servo action's clamp/scale math above is wrong for this install; every hold "
        "number would be built on an incorrect step size. STOPPING."
    )

    _servo_action = _hold_action()
    env.action_manager.process_action(_servo_action)
    processed_hand = _hand_term.processed_actions.clone()  # PRE-physics: process_actions() never reads q
    gap_pre = target_at_reset[:, _term_joint_ids] - robot.data.joint_pos[:, _term_joint_ids]
    expected_processed = gap_pre.clamp(-_HAND_SCALE, _HAND_SCALE)  # == clamp(gap/scale,-1,1)*scale + 0 offset
    isolated_residual = (processed_hand - expected_processed).abs()
    isolated_stats = _pct_stats(isolated_residual)
    print(
        f"[validate] SELF-CHECK (ISOLATED, pre-physics, action-pathway-only) residual vs our own"
        f" closed-form prediction: median={isolated_stats['median']:.8f} p90={isolated_stats['p90']:.8f}"
        f" max={isolated_stats['max']:.8f} rad",
        flush=True,
    )
    assert isolated_stats["max"] < 1e-4, (
        f"ISOLATED SELF-CHECK FAILED: the action term's own processed_actions differ from our"
        f" clamp(gap,-scale,scale) formula by up to {isolated_stats['max']:.6f} rad BEFORE any physics"
        " ran -- our model of how RelativeJointPositionActionCfg computes its target does not match"
        " this install. STOPPING; every hold number below would be built on a wrong model of the"
        " action term (NOT a plant/physics issue -- q was never even read at this point)."
    )
    print("[validate] ISOLATED SELF-CHECK PASSED -- action mechanism verified correct, independent of any plant motion.", flush=True)

    # ---- OLD self-check, KEPT AS INFORMATIONAL ONLY (UWLab-xp05.9: demoted from a gate -- its
    # "residual should be ~0" premise was only ever valid for a ZERO action against an
    # otherwise-static target; under the REAL servo action it is supposed to move by design, per the
    # apply_actions() mechanism explained above, so a nonzero reading here is normal, not a defect).
    # Still useful as a coarse "does the plant respond at all" sanity signal, and it is what the
    # UWLab-xp05.9 A/B diffed to positively exclude defect 3 -- kept unmodified for that reason.
    self_check_action_fn = _zero_action if args_cli.self_check_action == "zero" else _hold_action
    env.step(self_check_action_fn())
    target_after_one_step = robot.data.joint_pos_target.clone()
    delta = (target_after_one_step - target_at_reset).abs()

    _arm_ids = [i for i in range(len(robot.data.joint_names)) if i not in set(_hand_ids)]
    hand_residual_stats = _pct_stats(delta[:, _hand_ids])
    target_residual = hand_residual_stats["max"]
    arm_residual = float(delta[:, _arm_ids].max().item()) if _arm_ids else 0.0
    print(
        f"[validate] INFORMATIONAL (action={args_cli.self_check_action}, includes physics substeps --"
        f" NOT a pass/fail gate, see ISOLATED check above for that) target-buffer residual after 1 full"
        f" step: HAND median={hand_residual_stats['median']:.8f} p90={hand_residual_stats['p90']:.8f}"
        f" max={hand_residual_stats['max']:.8f} rad | ARM max={arm_residual:.6f} rad",
        flush=True,
    )

    dt = env.step_dt
    checkpoints_s = sorted(float(v) for v in args_cli.checkpoints_s.split(","))
    checkpoint_steps = [max(1, round(t / dt)) for t in checkpoints_s]
    n_steps_total = checkpoint_steps[-1]
    print(f"[validate] step_dt={dt:.5f}s, checkpoints at steps {checkpoint_steps} (~{checkpoints_s}s)", flush=True)

    ins_pose0 = torch.cat([env.scene["insertive_object"].data.root_pos_w, env.scene["insertive_object"].data.root_quat_w], dim=-1)
    rec_pose0 = torch.cat([env.scene["receptive_object"].data.root_pos_w, env.scene["receptive_object"].data.root_quat_w], dim=-1)
    depth0_mm, lateral0_mm, tilt0_deg = _reproject_strict(ins_pose0, rec_pose0, env.device)

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
                "depth_mm": buf_depth_mm.clone(), "lateral_mm": buf_lateral_mm.clone(), "tilt_deg": buf_tilt_deg.clone(),
                "true_pos_err_m": buf_true_pos_err_m.clone(), "true_rot_err_rad": buf_true_rot_err_rad.clone(),
                "true_success": buf_true_success.clone(), "held_contact": buf_held_contact.clone(), "died_at": died_at.clone(),
            }
            print(f"[validate] checkpoint t={t}s (step {step_idx}): {int((died_at >= 0).sum().item())}/{n} envs already died", flush=True)
            next_checkpoint_idx += 1

    def pct(a, ps=(0, 10, 25, 50, 75, 90, 100)):
        a = a.detach().cpu().numpy()
        return {p: float(np.percentile(a, p)) for p in ps}

    result = {
        "bank_path": os.path.realpath(args_cli.bank_path),
        "n_envs": n, "gain_regime": args_cli.gain_regime, "contact_threshold_n": args_cli.contact_threshold_n,
        "at_reset_hand_target_minus_q_rad": {"median": gap0_median, "p90": gap0_p90, "max": gap0_max},
        "at_reset_contact": at_reset_contact,
        "hand_action_term_scale_match": {"actual": _scale_repr, "assumed": _HAND_SCALE, "match": _scale_ok},
        "self_check_isolated_pre_physics_action_pathway_residual_rad": isolated_stats,  # THE GATE (UWLab-xp05.9)
        "self_check_action_mode_informational": args_cli.self_check_action,
        "self_check_full_step_target_residual_rad_INFORMATIONAL_ONLY": hand_residual_stats,
        "self_check_full_step_arm_residual_rad_informational": arm_residual,
        "checkpoints": {},
    }

    reason_counts_total = {}
    for _env_id, r in enumerate(died_reason):
        reason_counts_total[r] = reason_counts_total.get(r, 0) + 1

    print("\n=== PART B: per-checkpoint report ===", flush=True)
    for t in checkpoints_s:
        rec = checkpoint_records[t]
        depth_delta_mm = rec["depth_mm"] - depth0_mm
        held_frac_contact = float(rec["held_contact"].float().mean().item())
        strict_ok = (rec["lateral_mm"] < 15.0) & (rec["tilt_deg"] < 10.0)
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
        result["checkpoints"][str(t)] = cp_summary
        print(
            f"t={t:>4}s: died={died_by_then}/{n} depth_delta_mm(median)={cp_summary['depth_delta_mm_pct'][50]:+.2f} "
            f"held(contact)={held_frac_contact:.3f} held(strict)={held_frac_strict:.3f} true_predicate={true_predicate_frac:.3f}",
            flush=True,
        )

    result["died_reason_counts"] = reason_counts_total
    print(f"\ndied_reason counts: {reason_counts_total}", flush=True)

    last_t = checkpoints_s[-1]
    last = result["checkpoints"][str(last_t)]
    depth_delta_median = last["depth_delta_mm_pct"][50]
    held_contact_last = last["held_fraction_contact_graded"]
    criterion_a = depth_delta_median < -5.0
    criterion_b = held_contact_last < 0.70
    result["go_no_go"] = {
        "criterion_a_depth_slip_over_5mm_median": criterion_a,
        "criterion_b_held_contact_under_0.70": criterion_b,
        "verdict": "FAIL (NOT VIABLE AS-IS)" if (criterion_a or criterion_b) else "PASS",
    }
    print(f"\n=== PART B GO/NO-GO ===\n{result['go_no_go']}", flush=True)

    env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="Shared C4 reset-bank validation harness (UWLab-xp05.5): mating-frame decomposition + contact-graded dynamic hold test.")
    parser.add_argument("--bank-path", type=str, required=True, help="resets_ObjectPartiallyAssembledEEGrasped*.pt")
    parser.add_argument("--part", type=str, default="both", choices=["a", "b", "both"], help="'a' needs no Isaac/GPU; 'b'/'both' need a real Isaac boot.")
    parser.add_argument("--label", type=str, default=None, help="Friendly name for this bank in the report (defaults to the filename).")
    parser.add_argument("--n-states-a", type=int, default=None, help="Cap on states for Part A (default: all states in the bank).")
    parser.add_argument("--n-envs", type=int, default=256, help="Part B sample size AND live env count -- see smoke_test_ik_c4_holding.py's SAMPLE SELECTION note (unbiased draw-with-replacement from the whole bank).")
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--checkpoints-s", type=str, default="0.1,0.5,1.0,2.0")
    parser.add_argument("--contact-threshold-n", type=float, default=0.2)
    parser.add_argument("--gain-regime", type=str, default="train", choices=["train", "eval", "finetune_eval"])
    parser.add_argument(
        "--self-check-action", type=str, default="servo", choices=["servo", "zero"],
        help="UWLab-xp05.9: which action drives the ONE self-check step (see run_part_b's inline docstring). "
        "'servo' (default) is the real hold action the main rollout also uses. 'zero' exists ONLY to "
        "positively exclude/confirm defect 3 -- run twice with identical --seed/--bank-path/--n-envs, "
        "one per mode, and diff the printed residual to several decimals. Does not affect the main rollout.",
    )
    parser.add_argument("--scratch-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)

    # AppLauncher args are always registered for CLI uniformity (--headless etc.); the app itself is
    # only INSTANTIATED (i.e. actually booted) below, gated on --part needing Isaac.
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    label = args_cli.label or os.path.basename(args_cli.bank_path)
    summary = {"bank_path": os.path.realpath(args_cli.bank_path), "label": label, "part": args_cli.part}
    summary["bank_sha256"] = _sha256(os.path.realpath(args_cli.bank_path))

    need_isaac = args_cli.part in ("b", "both")
    simulation_app = None
    if need_isaac:
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

    # ---- schema validation + Part A: pure torch/numpy/scipy, no isaaclab needed regardless of
    # whether Isaac already booted above. ----
    raw = torch.load(args_cli.bank_path, map_location="cpu", weights_only=False)
    summary["schema_validation"] = validate_bank_schema(raw, args_cli.bank_path)

    if args_cli.part in ("a", "both"):
        summary["part_a"] = run_part_a(args_cli.bank_path, raw, args_cli.n_states_a, args_cli.seed)

    if need_isaac:
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
        from uwlab_tasks.manager_based.manipulation.dexlift.dexlift_ur10e_delto_env_cfg import (  # noqa: E402
            ALL_TIP_NAMES, HAND_PRIM, THUMB_TIP_NAMES, TIP_NAMES,
        )
        from uwlab_tasks.manager_based.manipulation.dexlift.mdp.rewards import _sensor_force_magnitudes  # noqa: E402

        gain_regime_map = {
            "train": Ur5eDeltoRelCartesianOSCTrainCfg,
            "eval": Ur5eDeltoRelCartesianOSCEvalCfg,
            "finetune_eval": Ur5eDeltoRelCartesianOSCFinetuneEvalCfg,
        }
        env_module_globals = {
            "math_utils": math_utils, "ManagerBasedRLEnv": ManagerBasedRLEnv, "ContactSensorCfg": ContactSensorCfg,
            "gain_regime_cfg": gain_regime_map[args_cli.gain_regime], "omnireset_utils": omnireset_utils,
            "_apply_delto_dataset_dir": _apply_delto_dataset_dir, "ALL_TIP_NAMES": ALL_TIP_NAMES,
            "HAND_PRIM": HAND_PRIM, "THUMB_TIP_NAMES": THUMB_TIP_NAMES, "TIP_NAMES": TIP_NAMES,
            "_sensor_force_magnitudes": _sensor_force_magnitudes,
        }
        summary["part_b"] = run_part_b(args_cli, env_module_globals)

    with open(args_cli.out, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[validate] wrote {args_cli.out}", flush=True)

    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
