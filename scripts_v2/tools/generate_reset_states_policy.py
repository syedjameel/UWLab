# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy-driven OmniReset reset-state generator (bead UWLab-dwx.2).

Upstream's ``record_reset_states.py`` steps a SCRIPTED gripper action while a random grasp
candidate is shaken to see if it survives. That works for a two-jaw model with one hand-authored
grasp; it does not for our 200 mm table leg, which the DELTO's shipped grasp sampler was never
tuned against (all jitter at 0.0, sized for a 34 mm cube). This script rolls out a CERTIFIED,
already-trained rl_games checkpoint instead, in the EXACT plant it was trained on -- the dexlift
RelJointPos table-leg env, 60 Hz control (decimation=2, sim.dt=1/120), NOT an OmniReset env
(OmniReset's own reset-state envs run at 10 Hz -- decimation=12 -- which would silently step this
checkpoint at the wrong control rate; see the bead notes) -- and exports the resulting state
snapshots in OmniReset's own recorder schema, unchanged, so the existing loader consumes them with
zero changes.

FOUR THINGS THIS SCRIPT BUILDS, matching the bead's four numbered requirements:

1. POLICY ROLLOUT. play.py:142-215's rl_games idiom, verbatim: load_checkpoint/load_path set
   BEFORE runner.load(agent_cfg), then create_player/restore/reset, is_deterministic=True,
   player.has_batch_dimension = True after reset. Obs normalisation lives INSIDE the checkpoint
   (agent yaml's normalize_input: True); nothing here hand-normalises or hand-builds the model.
2. THE HELD-CHECK. Wired as env_cfg.terminations.success (dexlift.mdp.held_check.held_with_probe),
   so the recorder plumbing (StableStateRecorderManagerCfg, EXPORT_SUCCEEDED_ONLY,
   RecorderManager.record_pre_reset reading a term literally named "success") needs zero changes --
   this script only ever ADDS a termination field, never edits the recorder classes. See
   dexlift/mdp/held_check.py / held_check_core.py for the four gates and their unit tests.
3. SAMPLE-EFFICIENCY ACCOUNTING. Attempts vs accepted (mirrors record_reset_states.py:158's own
   success-rate print) PLUS a per-gate rejection breakdown, since a bare pass/fail number does not
   say what to fix against the standing 40% gate.
4. CHECKPOINTS. Loaded from wherever --checkpoint/--agent_yaml point (local paths -- pull them down
   from DL_A6000 with scp/rsync first; the remote tree is not a git checkout). The agent yaml is
   read directly from the checkpoint's own params/ dir rather than the task's registered
   rl_games_cfg_entry_point, so a generator run is pinned to the EXACT hyperparameters (including
   normalize_input's RunningMeanStd shape) the checkpoint was actually trained under, not whatever
   the local source tree's yaml currently says.

THE PROBE. held_with_probe only MEASURES palm/object displacement over a window; it cannot COMMAND
one (a termination term has no action-pipeline write access). This script injects the jog: a
constant bias (PROBE_ARM_ACTION_BIAS) is added to the six arm action dimensions, on top of whatever
the policy commands, for every env `success_term.probe_active` reports True this step. The window
is RE-ARMING and event-triggered (whenever settled & opposed_contact & co_move newly become true,
wherever that happens in the episode), not a fixed absolute window -- see held_check.py's module
docstring for the STEP-1 diagnostic (dwx.6/dwx.7) that found a fixed early window measured the
wrong moment on every episode of a 198-episode sample. Reading probe_active directly off the term
instance each step is what keeps the injected jog and the term's own measurement window from
drifting apart, now that there is no fixed schedule to share as constants instead.

Run (one Isaac process; never via uwlab.sh):
    PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl" \\
        timeout -s KILL 300 <python> -u scripts_v2/tools/generate_reset_states_policy.py \\
        --checkpoint <path>.pth --agent_yaml <path>/params/agent.yaml \\
        --num_envs 16 --smoke_steps 200
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Policy-driven OmniReset reset-state generator.")
parser.add_argument(
    "--task", type=str, default="DexLift-UR5eDelto-RelJointPos-TableLeg-Lift-Play-v0",
    help="The dexlift task id matching the checkpoint's OWN plant (RelJointPos, table-leg, 60Hz).",
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth checkpoint.")
parser.add_argument(
    "--agent_yaml", type=str, default=None,
    help="Path to the checkpoint's params/agent.yaml. Defaults to <checkpoint's params dir>/agent.yaml.",
)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument(
    "--receptive_usd_path", type=str, required=True,
    help=(
        "USD path of the receptive object (fixture) this reset-state pair is FOR. The dexlift table-leg"
        " plant this script rolls out has no fixture in its scene -- only `object` (the leg) -- so"
        " compute_pair_dir's other half cannot be read off the env and must be supplied here."
    ),
)
parser.add_argument(
    # Matches _UR5E_DELTO_RESET_DIR in ur5e_delto_cfg.py, the root the UR5e+DELTO training cfg
    # actually reads. The generic default record_reset_states.py uses ("./Datasets/OmniReset/") is
    # for the two-jaw family; this script is UR5e+DELTO-only (see module docstring), so it must
    # default to the DELTO-specific root, not the shared one.
    "--dataset_dir", type=str, default="./Datasets_ur5e_delto/OmniReset",
    help="Root Datasets_ur5e_delto/OmniReset directory (must match the training cfg's dataset root).",
)
_CANONICAL_RESET_TYPES = (
    "ObjectAnywhereEEAnywhere",
    "ObjectRestingEEGrasped",
    "ObjectAnywhereEEGrasped",
    "ObjectPartiallyAssembledEEGrasped",
)
parser.add_argument(
    "--reset_type", type=str, required=True, choices=_CANONICAL_RESET_TYPES,
    help=f"Reset type name for the output path. Must be one of: {', '.join(_CANONICAL_RESET_TYPES)}.",
)
parser.add_argument("--num_reset_states", type=int, default=0, help="Target accepted count; 0 = smoke mode only.")
parser.add_argument(
    "--smoke_steps", type=int, default=200,
    help="Smoke mode (num_reset_states=0): step this many env-steps, print the rejection breakdown, exit.",
)
parser.add_argument(
    "--zero_action", action="store_true",
    help=(
        "DIAGNOSTIC CONTROL (harness mandate, 2026-08-22): override the policy's OWN action with "
        "all-zeros every step, unconditionally (after the probe-bias injection too, so nothing "
        "overrides this override). The checkpoint is still loaded and player.get_action() is still "
        "called each step (an unmodified code path up to that point) -- only the resulting action "
        "tensor is zeroed before env.step(). Because the arm/hand actions are RelativeJointPosition"
        "Action with use_zero_offset=True, a zero action commands target == measured q, i.e. the "
        "robot HOLDS ITS CURRENT POSE and never moves. This isolates whether the leg's own "
        "spawn+physics trajectory (zero robot involvement) alone produces an observed depth/"
        "lateral drift -- separating a spawn-geometry defect (e.g. thread/bore solid-body overlap "
        "at spawn, PhysX depenetration ejecting the leg with no hand present) from a policy-caused "
        "disturbance. Meant to be paired with --c4_rewind_deepest's depth_vs_step/lateral_vs_step "
        "curves in smoke mode (--num_reset_states 0): same spawn path, same bins, robot frozen."
    ),
)
parser.add_argument(
    "--progress_every_episodes", type=int, default=50,
    help="Print an accepted/attempts progress line at least every N ended episodes (num_reset_states mode).",
)
parser.add_argument(
    "--progress_every_seconds", type=float, default=30.0,
    help="Print an accepted/attempts progress line at least every N wall-clock seconds (num_reset_states mode).",
)
parser.add_argument(
    "--episode_length_s", type=float, default=None,
    help=(
        "Override env_cfg.episode_length_s (seconds) on THIS run's env_cfg only, applied after "
        "parse_env_cfg and before gym.make. Default None leaves the task's registered value "
        "unchanged, so every past measurement stays reproducible from the same command line."
    ),
)
# -- C2-VIA-REWIND (bead UWLab-weyl, "C2 via rewind"). Gated behind this flag ONLY -- every line
# touched below is a no-op when it is absent, so the existing accept-time path is byte-identical
# to before. See _C2RewindBank's docstring for why the anchor is FIRST CONTACT OF ANY KIND (a
# revision from the original first-OPPOSED-contact anchor, after measurement showed that one is
# already downstream of real disturbance): held_with_probe's own accept gate only latches
# ~SETTLE_STEPS+PROBE_STEPS after (opposed) contact, by which point the leg is typically already
# lifted clear of the table -- rewinding a fixed offset from THAT step lands mid-hold, not
# near-object.
parser.add_argument(
    "--c2_rewind", action="store_true",
    help=(
        "Also bank NEAR-OBJECT (upstream C2) reset states by rewinding, on every ACCEPTED episode, "
        "from that env's first-contact-of-any-kind step to each of --c2_offsets_s seconds earlier. "
        "Off by default; existing accept-time behaviour is untouched when absent."
    ),
)
parser.add_argument(
    # LOCKED at 0.03/0.05/0.10/0.20 s (bead UWLab-weyl, measured 2026-08-20): at these offsets
    # object linear velocity is 0.000-0.049 m/s median and height sits at 15-18mm median -- the leg
    # is genuinely at rest. 0.35s/0.50s were dropped: they measurably reach back into this task's
    # scripted airborne free-fall spawn (median obj velocity 0.32-0.64 m/s, height 90-152mm at
    # those offsets) -- not "already grasped" as first guessed, but not resting either. Still a CLI
    # list, not a constant -- re-measure rather than assume if the anchor, task, or checkpoint ever
    # changes; see class docstring.
    "--c2_offsets_s", type=float, nargs="+", default=[0.03, 0.05, 0.10, 0.20],
    help=(
        "Seconds to rewind BEFORE first contact of any kind, one output file per offset (rounded "
        "to the nearest control step at this env's control rate). Only used with --c2_rewind."
    ),
)
parser.add_argument(
    "--c2_reset_type", type=str, default="ObjectAnywhereEENear",
    help=(
        "Filename stem for the C2 rewind output. One file per offset is written as "
        "resets_<c2_reset_type>_off<X.XX>s.pt in the SAME pair directory as the accept-time bank, "
        "so MultiResetManager consumes it unchanged given a matching reset_types entry."
    ),
)
parser.add_argument(
    "--c2_max_resting_speed", type=float, default=0.05,
    help=(
        "Hard filter, enforced not just documented: a rewound candidate whose object linear "
        "velocity magnitude exceeds this (m/s) at the REWOUND step is rejected at emit time, not "
        "written to disk. Rejected counts are reported per offset. Default 0.05 m/s -- measured "
        "medians at the locked offsets are 0.000-0.049 m/s, so this discards only the tail that "
        "was not actually resting."
    ),
)
# -- C4 SEATING GATE (DELIVERABLE 1, team-lead ask). Gated behind --c4_seating_gate ONLY -- every
# line touched below is a no-op when it is absent, so the existing accept-time path for every
# OTHER --reset_type (and for ObjectPartiallyAssembledEEGrasped runs that do not pass this flag)
# is byte-identical to before. See SeatedHeldWithProbe's own docstring for the full argument.
#
# MEASURED, not assumed: probing a 25%-partial-assembly finetune (n=100 accepted states) and
# decomposing into the mating frame found 0/100 states with the leg tip inside a seated depth
# band, 60% with NEGATIVE depth (tip at or above the bore mouth), median lateral miss 21.55mm and
# median tilt 12.32deg -- against a spawn distribution of depth 10.0-17.5mm / lateral 0.04mm /
# tilt 0.14deg. held_with_probe has NO spatial term at all: it only asks "is the object held", not
# "is it still where a partially-assembled state needs it to be" -- so a policy that grasps well
# and then withdraws the leg while holding it is accepted just as readily as one that holds it in
# place. This flag adds that missing spatial term as an independent, opt-in AND gate.
parser.add_argument(
    "--c4_seating_gate", action="store_true",
    help=(
        "OPT-IN spatial acceptance gate, only meaningful with --reset_type "
        "ObjectPartiallyAssembledEEGrasped (needs a receptive_object/fixture in the scene, i.e. "
        "DEXLIFT_PARTIAL_ASSEMBLY=1). ANDs held_with_probe's existing held-in-hand decision with a "
        "'is the leg tip still genuinely seated in the bore' check: tip depth inside "
        "[--c4_depth_min_mm, --c4_depth_max_mm], lateral miss <= --c4_lateral_max_mm, tilt <= "
        "--c4_tilt_max_deg. See SeatedHeldWithProbe's docstring for the measured defect this "
        "closes and the geometry this reuses. Off by default -- existing accept-time behaviour "
        "for every --reset_type, INCLUDING ObjectPartiallyAssembledEEGrasped, is untouched unless "
        "this flag is passed explicitly."
    ),
)
parser.add_argument(
    "--c4_engaged_span_mm", type=float, default=25.0,
    help=(
        "Physical engaged span of the bore (mouth to fully-seated tip), millimetres. NOT a "
        "metadata.yaml field -- only assembled_offset (the SEAT position/orientation) lives there; "
        "the mouth-to-seat distance is a separate mesh measurement (see "
        "test_axial_insertion_depth_geometry.py's ENTRY_MOUTH_LOCAL_Z_M, which agrees with this "
        "default to sub-mm precision) with no runtime source to read it from for an arbitrary pair. "
        "Only override this if --receptive_usd_path/the leg USD is not the "
        "OneLegInsertionFixture/SquareTableLeg200mmDecomp pair this default was measured for."
    ),
)
parser.add_argument(
    "--c4_depth_min_mm", type=float, default=5.0,
    help="Lower bound (mm) of the accepted tip-depth band below the bore mouth. Default leaves "
    "margin against the mouth (depth=0) so a state right at the entrance is not miscounted as seated.",
)
parser.add_argument(
    "--c4_depth_max_mm", type=float, default=20.0,
    help="Upper bound (mm) of the accepted tip-depth band. Default leaves margin against full seat "
    "(depth=--c4_engaged_span_mm) for the same reason as --c4_depth_min_mm.",
)
parser.add_argument(
    "--c4_lateral_max_mm", type=float, default=8.0,
    help="Max allowed lateral miss (mm) of the tip from the bore centreline. Default sits well "
    "inside the bore's own ~12.2-12.5mm crest/mouth radius (test_axial_insertion_depth_geometry.py's "
    "MOUTH_CROSSING_RADIUS_M/MOUTH_BORE_RADIUS_M), i.e. inside the physically possible range.",
)
parser.add_argument(
    "--c4_tilt_max_deg", type=float, default=20.0,
    help="Max allowed tilt (deg) of the leg's insertion axis vs. the bore's own deep axis. A "
    "generous default, not derived from the depth-dependent rim-cap bound "
    "test_axial_insertion_depth_geometry.py computes for its own sampler -- tune per the actual "
    "distribution wanted; this flag exists so that does not require an edit to this file.",
)
# -- ARM 1 (bead UWLab-xp05.1): rewind-to-deepest-grasp. Gated behind --c4_rewind_deepest ONLY --
# every line touched below is a no-op when it is absent. See _C4DeepestGraspBank's docstring for
# why this is a running per-env argmax snapshot, not a fixed-offset ring-buffer rewind like
# --c2_rewind (the anchor here is discovered dynamically, not at a known fixed backward offset).
parser.add_argument(
    "--c4_rewind_deepest", action="store_true",
    help=(
        "Also bank the DEEPEST earlier step (subject to opposed contact holding) of every episode "
        "the FULL settle+probe held-check (--reset_type ObjectPartiallyAssembledEEGrasped, with or "
        "without --c4_seating_gate) later accepts. Off by default; existing accept-time behaviour "
        "is untouched when absent. Requires --reset_type ObjectPartiallyAssembledEEGrasped (needs "
        "the receptive_object/fixture already in the scene, i.e. DEXLIFT_PARTIAL_ASSEMBLY=1) and is "
        "mutually exclusive with --c4_terminate_on_grasp (Arm 1 explicitly wants the FULL probe's "
        "own acceptance decision as the episode filter, not Arm 2's fast replacement for it)."
    ),
)
parser.add_argument(
    "--c4_rewind_reset_type", type=str, default="ObjectPartiallyAssembledEEGraspedDeep",
    help="Filename stem for the Arm-1 rewind output: resets_<this>.pt, written to the SAME pair "
    "directory as the accept-time bank (same convention as --c2_reset_type).",
)
parser.add_argument(
    "--c4_rewind_settle_steps", type=int, default=None,
    help="Episode steps before an opposed-contact step is even considered a rewind candidate -- "
    "guards against a spurious early depth/contact reading during the post-reset settle window "
    "(the SAME concern held_with_probe's own SETTLE_STEPS gate exists for). Default None -> reuses "
    "held_check_core.SETTLE_STEPS (60), the same constant the accept-time held-check itself uses. "
    "Only takes effect at all when --c4_rewind_require_settle is also passed -- see that flag.",
)
parser.add_argument(
    "--c4_rewind_require_settle", action="store_true",
    help=(
        "Arm-1 REAL BANK GATE (harness mandate, 2026-08-22, promoted from analysis to deployment "
        "on the strength of the settle x speed sweep): require settled (--c4_rewind_settle_steps) "
        "for a step to be considered a REAL-BANK candidate. OFF BY DEFAULT NOW -- the sweep found "
        "ZERO would-emit states with settle required, at EVERY speed threshold including ungated, "
        "and NONZERO the moment settle is lifted (a clean single-variable result across >1000 "
        "attempts) -- keeping settle required in the real path was measured to guarantee an empty "
        "bank. Pass this flag to restore the old (settle-required) real-bank behaviour if a "
        "different checkpoint/task ever needs it re-tested; the sweep's own settle_required=True "
        "cells remain available as an analysis-only cross-check regardless of this flag."
    ),
)
parser.add_argument(
    "--c4_rewind_depth_min_mm", type=float, default=12.0,
    help="Arm-1 BANK GATE (enforced at emit time, not just documented -- same discipline as "
    "--c2_max_resting_speed): lower bound (mm) of the accepted tip-depth band for the banked "
    "deepest-opposed-contact state. DELIBERATELY DIFFERENT from --c4_depth_min_mm/--c4_depth_max_mm "
    "(the --c4_seating_gate accept-time band [5,20]mm default): this is the bead's own measured "
    "target band for a genuinely deep C4 state, not the seating gate's accept-time band.",
)
parser.add_argument(
    "--c4_rewind_depth_max_mm", type=float, default=25.0,
    help="Arm-1 BANK GATE upper bound (mm) -- see --c4_rewind_depth_min_mm.",
)
parser.add_argument(
    "--c4_rewind_lateral_max_mm", type=float, default=1.0,
    help=(
        "Arm-1 BANK GATE: max allowed lateral miss (mm) of the tip from the bore centreline. "
        "TIGHTENED TO THE MEASURED BORE CLEARANCE (harness mandate, 2026-08-22, promoted from "
        "analysis to deployment): the bore's radial clearance is ~0.91mm (tightest wall 10.9156mm "
        "vs. the leg's 10.004mm flat pilot) -- a leg GENUINELY inside the bore cannot read much "
        "above ~1mm lateral, so a looser limit (this flag's old default, 5.0mm, and the unrelated "
        "--c4_lateral_max_mm seating-gate default, 8.0mm) admits poses that are not physically "
        "possible insertions: a large axial (depth) projection off an off-axis, near-the-mouth pose "
        "is not a deep insertion, and banking one is pointless -- it will not survive part B. This "
        "is a measured physical constant (a property of the fixture), not a tunable design choice, "
        "same discipline as --c4_engaged_span_mm's own default."
    ),
)
parser.add_argument(
    "--c4_rewind_tilt_max_deg", type=float, default=15.0,
    help="Arm-1 BANK GATE: max allowed tilt (deg) of the leg's insertion axis vs. the bore's deep axis.",
)
parser.add_argument(
    "--c4_rewind_max_speed", type=float, default=0.05,
    help=(
        "Arm-1 CANDIDATE-SELECTION GATE (harness review, UWLab-xp05.6): m/s ceiling on object "
        "linear speed for a step to even be CONSIDERED as a new best-so-far candidate in "
        "_C4DeepestGraspBank.step -- a step with opposed contact but a still-moving object is a "
        "transient (contact just made, leg decelerating into the seat), not a stable seat, and its "
        "depth reading can be spuriously the deepest one the episode ever produces. Mirrors C2's own "
        "hard resting-speed filter (default 0.05 m/s, --c2_max_resting_speed) but applied at "
        "CANDIDATE-SELECTION time, not just emit time -- see the class docstring: the emit-time "
        "depth/lateral/tilt band alone cannot catch this, because a transient-but-deep reading can "
        "still land inside that band. A bank with most of its yield discarded by this gate is itself "
        "a finding (most of the depth advantage was transient), not just a filter side effect."
    ),
)
parser.add_argument(
    "--c4_rewind_speed_sweep", type=float, nargs="+", default=[0.05, 0.10, 0.25, 0.50],
    help=(
        "Arm-1 SPEED-GATE SWEEP (harness mandate, 2026-08-22): additional thresholds (m/s), "
        "ANALYSIS-ONLY, to re-evaluate 'what would the emit yield have been' at, from this SAME "
        "run, without a second rollout -- every opposed+settled step is recorded unfiltered by "
        "speed, and each threshold's own argmax-depth-then-band-check is computed per accepted "
        "episode at report()/print_progress() time. Does NOT affect --c4_rewind_max_speed, which "
        "remains the only threshold that gates the REAL bank. An implicit ungated (inf) threshold "
        "is always included alongside these."
    ),
)
# -- ARM 2 (bead UWLab-xp05.2): terminate-on-grasp. Gated behind --c4_terminate_on_grasp ONLY --
# a no-op on every other --reset_type / flag combination when absent. See TerminateOnGraspSuccess's
# docstring for the full argument (replaces settle+probe with a fast N-consecutive-step
# opposed-contact-force check).
parser.add_argument(
    "--c4_terminate_on_grasp", action="store_true",
    help=(
        "Replace held_with_probe's settle+probe displacement test with a FAST N-consecutive-step "
        "opposed-contact-force check (plus a low object-velocity check), wired as terminations."
        "success in place of held_with_probe. Composes with --c4_seating_gate (ANDs the same "
        "spatial seating band on top, same as it does for the probe-based check). Off by default; "
        "existing accept-time behaviour is untouched when absent."
    ),
)
parser.add_argument(
    "--c4_terminate_consecutive_steps", type=int, default=8,
    help="N: consecutive control steps of (opposed contact AND low object speed) required before "
    "Arm 2 accepts. Bead-suggested start range 5-10 (0.08-0.17s at 60Hz) -- cheap to tune without "
    "an edit to this file. A single-instant contact reading is the transient-touch risk this "
    "mitigates; do NOT drop this to 1 (see class docstring).",
)
parser.add_argument(
    "--c4_terminate_obj_speed_thresh", type=float, default=0.05,
    help="m/s ceiling on object linear velocity magnitude for a step to count toward the "
    "consecutive-step counter -- the low-object-velocity mitigation the bead asks for, additional "
    "to (not a replacement for) the N-consecutive-step requirement.",
)
parser.add_argument(
    "--c4_terminate_force_threshold", type=float, default=0.2,
    help="N, per-fingertip normal-force gate for Arm 2's opposed-contact check. Same default as "
    "held_with_probe's own force_threshold (0.2N) -- not re-tuned, reused.",
)
parser.add_argument(
    "--c4_terminate_settle_steps", type=int, default=0,
    help="Episode steps before ANY step counts toward Arm 2's consecutive-step counter. DEFAULT 0, "
    "DELIBERATELY NOT held_with_probe's SETTLE_STEPS=60: the epic's own root-cause framing "
    "attributes the settle+probe route's ~70-step acceptance latency to settle(60)+probe(10) "
    "combined, and Arm 2 exists specifically to remove that latency -- reintroducing a 60-step "
    "settle floor here would defeat most of the point. The N-consecutive-step-of-stable-opposed-"
    "contact-and-low-velocity requirement is Arm 2's OWN replacement guard against a transient "
    "post-reset artifact (a still-settling object rarely holds low velocity AND opposed contact for "
    "N consecutive steps); this knob exists to tune that trade-off, not to silently restore it.",
)
# -- C3(S_t) SPAWN-TOLERANCE GATE (bead dr-sj6.22, RESET_SPEC_V2.md sec 1 C3 / V2_C3_DESIGN.md sec
# 5+7). Composes SpawnToleranceHeldWithProbe -- held_with_probe AND "is the leg still within
# tolerance of the COMMANDED GOAL" -- as terminations.success, the S_t analogue of --c4_seating_gate
# (never combined with it: S_t has no mating frame, see _SpawnPoseToleranceAddon's own docstring).
# Requires DEXRESET_C3_RUNG=1 already staged in the environment (this script does not set it -- same
# convention as DEXLIFT_PARTIAL_ASSEMBLY for the C4 flags above) and --reset_type
# ObjectRestingEEGrasped (team-lead decision, 2026-08-29 -- NOT ObjectPartiallyAssembledEEGrasped:
# that is the reset_type --c4_seating_gate couples to, and S_t must never be seating-gated or share
# a bank identity with S1/C4). DEXLIFT_PARTIAL_ASSEMBLY=1 must STILL be exported for this run --
# C3RungGoalPoseCommand needs 'receptive_object' in the scene for S_t too (it parks the fixture on
# every S_t env, c3_rung.py's own module docstring) -- --reset_type and the scene's actual contents
# are independent knobs here; see the carve-out on the bidirectional partial-assembly guard in main().
parser.add_argument(
    "--c3_st_spawn_tolerance", action="store_true",
    help=(
        "OPT-IN acceptance gate for C3(S_t) generation (bead dr-sj6.22): held_with_probe AND "
        "distance-from-the-COMMANDED-GOAL within tolerance -- the goal IS the leg's settled "
        "reference pose once C3RungGoalPoseCommand's own deferred re-pin fires (V2_C3_DESIGN.md "
        "sec 7); this gate never captures a reference pose of its own (team-lead correction -- see "
        "_SpawnPoseToleranceAddon's own docstring for the F27 defect that caused). Requires "
        "DEXRESET_C3_RUNG=1 already set in the environment, DEXLIFT_PARTIAL_ASSEMBLY=1 already set "
        "(the scene needs 'receptive_object' even for S_t -- it gets parked, not used as a mating "
        "frame), and --reset_type ObjectRestingEEGrasped (team-lead decision -- --reset_type is a "
        "naming/output-path selector here, decoupled from what the scene actually contains; NOT "
        "ObjectPartiallyAssembledEEGrasped, which --c4_seating_gate couples to and S_t must stay "
        "clear of) -- all asserted below, before gym.make(). Mutually exclusive with "
        "--c4_seating_gate/--c4_terminate_on_grasp/--c4_rewind_deepest (different success_func "
        "families; S_t is never seating-gated). Requires --c3_st_pos_tol_mm explicitly -- there is "
        "no default and the run refuses to start without one (V2_ACCEPTANCE_CRITERIA.md sec 4: this "
        "number is OPEN, not yet sourced -- bead dr-sj6.24 derives it from THIS flag's own R4 "
        "validation run, not from a guess)."
    ),
)
parser.add_argument(
    "--c3_st_pos_tol_mm", type=float, default=None,
    help=(
        "REQUIRED when --c3_st_spawn_tolerance is passed; no default. Max position drift (mm) of "
        "the leg from the commanded goal. OPEN per V2_ACCEPTANCE_CRITERIA.md sec 4 / bead "
        "dr-sj6.24 -- pass a value to run a measurement pass, not a value you believe is correct; "
        "the run's own SpawnToleranceConfig raises if this is missing or non-positive."
    ),
)
parser.add_argument(
    "--c3_st_rot_tol_deg", type=float, default=None,
    help=(
        "Optional max rotation drift (deg) of the leg from the commanded goal. Omitted (default) "
        "DISABLES the rotation gate entirely -- same 'tested for truthiness' convention as "
        "success.py's within_success_tolerance (0 would silently mean 'disabled', not 'no rotation "
        "allowed', so an explicit 0 is rejected rather than accepted as that). V2_C3_DESIGN.md sec "
        "7's own still-open item: the rotation METRIC here is the full quaternion angle (position "
        "AND orientation together), not an axis-tilt-only metric -- axis-tilt would ignore spin "
        "about the leg's own long axis, which for a horizontal S_t peg may be the more defensible "
        "choice (see that section) but is a SEPARATE decision from this tolerance NUMBER and has "
        "not been made; do not assume this flag is scoring axis-tilt."
    ),
)
parser.add_argument(
    "--c3_st_command_name", type=str, default=None,
    help="Override the command term name _SpawnPoseToleranceAddon reads the goal from. Default "
    "None -> dexlift_mdp.GOAL_COMMAND_NAME ('object_pose'), the term _apply_c3_rung_stage installs "
    "C3RungGoalPoseCommand under -- only override if the env config wires the C3 rung goal under a "
    "different command name.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import yaml  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import uwlab_tasks  # noqa: F401,E402
import uwlab_tasks.manager_based.manipulation.dexlift.mdp as dexlift_mdp  # noqa: E402
import uwlab_tasks.manager_based.manipulation.omnireset.mdp as task_mdp  # noqa: E402
from isaaclab.managers import ManagerTermBase  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.managers.recorder_manager import DatasetExportMode  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from uwlab.utils.datasets.torch_dataset_file_handler import (  # noqa: E402
    TorchDatasetFileHandler,
    atomic_torch_save,
)
from uwlab_tasks.manager_based.manipulation.dexlift.dexlift_ur5e_delto_actions import (  # noqa: E402
    ARM_JOINT_NAMES as DEXLIFT_ARM_JOINT_NAMES,
)
from uwlab_tasks.manager_based.manipulation.dexlift.mdp.held_check_core import (  # noqa: E402
    PROBE_ARM_ACTION_BIAS,
    SETTLE_STEPS,
)
from uwlab_tasks.manager_based.manipulation.dexlift.mdp.rewards import (  # noqa: E402
    _sensor_force_magnitudes,  # reused, not reimplemented -- see rewards.py:40-76 / held_check.py:58
)
from uwlab_tasks.manager_based.manipulation.dexlift.mdp.spawn_tolerance_core import (  # noqa: E402
    SpawnToleranceConfig,
    axis_tilt_rad,
    pose_distance,
    within_spawn_tolerance,
)
# NOTE: this file no longer imports c3_rung_core (bead dr-ai1.18, commit f1f3818). It previously
# pulled DEFAULT_ST_SETTLE_SPEED_MPS/DEFAULT_ST_SETTLE_ANG_SPEED_RAD_S/validate_st_settle_* to
# recompute the S_t settle predicate as a local trust latch; that whole duplication is gone now that
# _SpawnPoseToleranceAddon reads C3RungGoalPoseCommand.goal_is_final directly -- see that class's
# own docstring. Nothing in THIS file re-derives or duplicates c3_rung.py's goal-generation logic.
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.recorders.recorders import (  # noqa: E402
    StableStateRecorder,
)
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.recorders.recorders_cfg import (  # noqa: E402
    StableStateRecorderManagerCfg,
)


def atomic_json_save(obj, path: str) -> None:
    """``json.dump(obj, path)`` via a temp file + ``os.replace``, mirroring
    ``uwlab.utils.datasets.torch_dataset_file_handler.atomic_torch_save``'s own idiom exactly (same
    reason: a naive in-place write is 0 bytes or torn for most of the time it takes to serialize,
    and a kill landing in that window has already destroyed a whole file in production).

    STANDING RULE (harness mandate, 2026-08-22, after three complete measurements were lost the
    same day to a terminal-only report block): a long generation run must dump its accumulated
    diagnostics to disk PERIODICALLY, not only once at the very end -- a SIGKILL (timeout, OOM,
    contention) skips any Python cleanup entirely, so a report() that only ever prints at the last
    line of main() is a single point of failure. This helper is what main()'s periodic progress
    block calls every ~250 attempts to make that dump atomic.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = os.path.join(directory, f".{os.path.basename(path)}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w") as f:
            json.dump(obj, f, indent=2, default=float)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def unwrap(o):
    """RlGamesVecEnvWrapper returns {"obs": tensor}; reset may also return (obs, info)."""
    if isinstance(o, tuple):
        o = o[0]
    if isinstance(o, dict):
        o = o["obs"]
    return o


def _require(name: str, requested: bool, actual: bool, message: str) -> None:
    """Raise unless `requested` (what the launch command asked for) matches `actual` (what the
    constructed env_cfg actually built), in EITHER direction.

    A silently-unset toggle and a silently-still-active leftover from a previous invocation are
    the same failure mode -- a plausible-looking but WRONG distribution instead of an error -- so
    both directions raise, not just the "forgot to set it" one. This is the shape the
    DEXLIFT_SPAWN_CLEARANCE and DEXLIFT_GOAL_AT_SPAWN/DEXLIFT_PARTIAL_ASSEMBLY guards below already
    used before being extracted here (bead UWLab-algw.9); a fourth guard should be one line, not a
    copy-paste of this if/raise pair. `message` should read naturally as the sentence following
    "<name> ...; <message> Refusing to generate reset states ...".
    """
    if requested == actual:
        return
    if requested:
        verb = "was requested but the constructed env_cfg did not build it"
    else:
        verb = "was not requested but the constructed env_cfg built it anyway"
    raise RuntimeError(
        f"{name} {verb}. {message} Refusing to generate reset states under a distribution the "
        "launch command did not ask for."
    )


class _DexliftToTrainingSceneRecorder(StableStateRecorder):
    """``StableStateRecorder``, but re-keys THIS SCRIPT's dexlift-lift-scene rigid_object names to
    the OmniReset TRAINING scene's own names before a state ever reaches disk (bead: RestingEEGrasped
    bridge pre-flight catch, 2026-08-17).

    WHY THIS EXISTS AS A RECORDER, NOT A POST-PROCESSING PASS. The dexlift table-leg LIFT scene this
    script rolls out names its manipulated body ``object``; the OmniReset TRAINING scene that later
    consumes these states (via ``MultiResetManager._reset_to``, which matches rigid_object entries by
    NAME and SILENTLY SKIPS any absent from the state) names the same kind of body
    ``insertive_object``. For a KINEMATIC body that skip is harmless; ``insertive_object`` in the
    training scene is NOT kinematic (``rl_state_cfg.py::make_insertive_object``), so a state recorded
    under the wrong name silently reset the robot's hand into a "holding it" joint pose while leaving
    the actual object at its untouched spawn default -- no exception, no warning, just wrong training
    data. A one-time post-hoc re-key (``rekey_dexlift_reset_states.py``) fixed the 611 episodes already
    on disk, but a post-hoc pass only runs if someone remembers to run it, and this script's own
    documented invocation (``timeout -s KILL 300``) can and does kill a run mid-flight -- any state
    already flushed to disk before that kill would carry the wrong name with no later chance to fix
    it. Renaming HERE, inside the recorder term itself, means every single flush this script ever
    writes -- including the very last one before a SIGKILL -- already has the correct name. There is
    no window where a wrong-named state can reach disk.

    ``table`` is DROPPED, not renamed: it is a DIFFERENT USD asset in the two scenes (dexlift's own
    generic ``CuboidCfg`` slab vs. the training scene's ``custom_lab_table.usd``), and restoring one
    asset's recorded pose onto a different one is not something to do silently even though both
    happen to be kinematic. The training scene's own table is already correctly placed at spawn.
    ``ur5_metal_support`` is never invented -- kinematic in the training scene and correctly left at
    spawn when a state omits it; this scene has no support plate at all, so there is nothing to
    invent it from.

    ``receptive_object`` IS KEPT (renamed to itself, i.e. exported unchanged) WHEN PRESENT --
    extended for bead UWLab-qiao.2/.6, the ``DEXLIFT_PARTIAL_ASSEMBLY`` toggle
    (``dexlift_ur5e_delto_tableleg_env_cfg.py``). That toggle adds a real ``receptive_object`` entity
    to this scene (see ``dexlift.mdp.partial_assembly``), so a state recorded with it on now carries
    the fixture too and is schema-complete the moment it reaches disk -- exactly the gap
    UWLab-qiao.7 found and had to patch after the fact for the two files recorded before this entity
    existed. Nothing here re-derives the fixture's pose or the OmniReset training scene's z
    convention; this class only forwards whatever the scene already wrote.

    FAILS LOUDLY rather than guessing if the source keys are ever anything other than one of the two
    KNOWN schemas -- e.g. if this script is ever pointed at a different ``--task`` whose scene uses
    different names. A silent no-op remap would recreate exactly the defect this class exists to
    prevent, just relocated.
    """

    _RENAME = {"object": "insertive_object"}  # dexlift scene name -> OmniReset training scene name
    _DROP = {"table"}  # different asset in the training scene; do not carry its pose across
    # receptive_object is intentionally absent from both dicts above: absent from _RENAME because
    # its name already matches the training scene, absent from _DROP because (when present) it is
    # exactly the entity the training scene is missing when this file's rigid_object dict lacks it.
    _KNOWN_SCHEMAS = ({"object", "table"}, {"object", "table", "receptive_object"})

    def record_pre_reset(self, env_ids):
        key, state = super().record_pre_reset(env_ids)
        rigid_object = state["rigid_object"]
        keys = set(rigid_object.keys())
        if keys not in self._KNOWN_SCHEMAS:
            raise ValueError(
                f"_DexliftToTrainingSceneRecorder expected rigid_object keys {{'object', 'table'}}"
                f" (plain lift/reorient scene) or {{'object', 'table', 'receptive_object'}}"
                f" (DEXLIFT_PARTIAL_ASSEMBLY=1 scene), got {sorted(keys)}. This recorder's"
                f" rename/drop/passthrough is specific to those two scenes (see class docstring) --"
                f" refusing to silently mis-map (or silently pass through) an unexpected schema."
            )
        state["rigid_object"] = {
            self._RENAME.get(name, name): tensors for name, tensors in rigid_object.items() if name not in self._DROP
        }
        return key, state


class _C2RewindBank:
    """Bead UWLab-weyl "C2 via rewind": banks NEAR-OBJECT reset states by rewinding a rolling
    per-env scene-state ring buffer back from FIRST CONTACT OF ANY KIND, on every episode
    ``held_with_probe`` later accepts.

    THE ANCHOR, REVISED (team-lead correction after the first measured pass). The original anchor
    was first OPPOSED contact (thumb-side tip AND a non-thumb tip both loaded). Measured: at that
    anchor the object had typically ALREADY moved ~27 cm from its spawn pose even at the shortest
    (0.1 s) offset, while height stayed near-resting -- i.e. the hand had been sliding/jostling the
    leg with ONE-SIDED contact (single-finger force spikes up to 150 N while opposed_contact stayed
    False) before ever landing a clean two-sided pinch. Opposed contact is the first CLEAN pinch
    AFTER disturbance, not the first touch -- wrong anchor for "near object, untouched". This class
    now anchors on FIRST CONTACT OF ANY KIND: any single tip (thumb-side OR non-thumb) loaded above
    ``force_threshold``, whichever fires first. Opposed-contact detection is KEPT alongside it
    (``first_opposed_contact_step``), not for capture, but so ``report()`` can print how many steps
    earlier the any-contact anchor fires, per accepted episode -- see ``self._lead_deltas_s``.

    UNDER PRODUCTION SETTINGS (DEXLIFT_POSE_TILT set, plant+episode_length matched -- see the
    free-fall-confound note below) THE TWO ANCHORS NEARLY COINCIDE: any-contact and opposed-contact
    fired the SAME number of times (277 of 277) with median lead 1 step (0.017 s), down from ~8
    steps (~0.13 s) measured under the misconfigured runs that motivated this anchor change in the
    first place. A pre-oriented, cleanly-landed leg means first touch is usually already a clean
    two-sided pinch. THE ANY-CONTACT ANCHOR WAS STILL THE RIGHT CALL -- it is what made the short
    offsets meaningful under the WRONG configuration this project was actually running when the
    choice was made, and keeping it costs nothing now that the two rarely differ -- but do not read
    the any-vs-opposed distinction as load-bearing under production settings; it mostly is not.

    Anchoring on the ACCEPT step instead was considered and rejected: accept only latches
    ~SETTLE_STEPS+PROBE_STEPS after (opposed) contact, by which point the leg is typically already
    lifted clear of the table, so a fixed offset from accept lands mid-hold -- just another
    ObjectAnywhereEEGrasped, not near-object.

    WHY CAPTURE AT FIRST-CONTACT TIME, NOT AT ACCEPT TIME. The offsets look BACKWARD from first
    contact (approach and pre-contact rest), and first contact typically happens well before
    step SETTLE_STEPS(=60) -- long before accept is even possible (>= SETTLE_STEPS+PROBE_STEPS).
    A ring buffer sized for the largest offset does not reach that far back by the time an episode
    is known to be accepted. So this class extracts the pre-contact window from the ring buffer
    EAGERLY, the instant first (any) contact fires (while it is still in the buffer), and holds it
    per-env in ``self.pending`` until that episode's outcome (accept or not) is known -- discarding
    on either episode end via ``finalize_episodes``.

    THE DISTANCE METRIC, RECALIBRATED (team-lead correction). The palm frame's origin sits 15-20 cm
    from the natural grasp point, so a solidly-HELD object still reads ~150-200 mm from the palm --
    an absolute palm-object distance cannot be compared against a 2 cm jitter budget. Two fixes,
    both applied: (1) FINGERTIP-to-object distance is reported directly (``min_fingertip_obj_dist``
    -- the nearest of all 5 tip bodies to the object, read straight off ``body_pos_w`` at the SAME
    body names ``held_with_probe`` already uses for ``thumb_contact_names``/``tip_contact_names``,
    which double as body names here -- see ``dexlift_ur5e_delto_env_cfg.py``'s ``TIP_BODY_REGEX``
    for the ``rl_dg_<n>_tip`` naming this relies on); a fingertip's origin sits at the contact
    surface, so this number is meaningful in absolute mm. (2) Palm-object distance is kept but
    reported as a DELTA against its OWN value at the anchor step, same episode
    (``palm_obj_dist_delta_mm = d_palm(rewound) - d_palm(anchor)``) -- that removes the frame-offset
    constant and leaves only the actual approach gap.

    OBJECT-DISPLACEMENT-FROM-SPAWN IS UNRELIABLE FOR THIS TASK FAMILY; USE VELOCITY INSTEAD. The
    first measured pass also reported ``obj_disp_from_spawn_mm`` (rewound object position minus its
    pose at THIS episode's own reset) as evidence the leg had been disturbed -- median ~230-290 mm
    even at the shortest offsets, which read as "the hand has already jostled it." That reading was
    WRONG, or at least not the dominant effect, and self-corrected by checking rather than trusting
    it: this task's own ``events.reset_object`` (see this script's own printed ``[verify]`` line)
    draws ``pose_range.z`` in ``[0.0, 0.4]`` -- EVERY episode spawns the leg AIRBORNE and free-falls
    it onto the table before the policy ever gets near it (the sibling UR10e task's own
    ``table_leg_env_cfg.py`` names this explicitly: "The leg starts airborne and falls freely onto
    the table before acquisition"). A 25-30 cm displacement from the literal spawn pose is the
    expected SIZE of that scripted drop alone, and ``obj_disp_from_spawn_mm`` cannot distinguish
    "still free-falling" from "hand-caused disturbance". ``obj_lin_vel_mag`` (object linear velocity
    magnitude, near 0 once landed) is the metric that actually answers "is it still resting,
    untouched": measured medians are EXACTLY 0.000 m/s at 0.03/0.05/0.10 s and 0.049 m/s at 0.20 s,
    with height tight at 15-18 mm throughout that range -- genuinely at rest. At 0.35/0.50 s (since
    dropped from the locked default, see ``--c2_offsets_s``'s own comment) median velocity rises to
    0.32-0.64 m/s and height to 90-152 mm: THAT is reaching into the free-fall, not into an
    already-grasped state as first guessed when this anchor was still first-OPPOSED-contact -- the
    mechanism changed between the two measured passes, the SIZE of the problem ("0.5 s is too far")
    did not.

    THE [0.0, 0.4] RANGE ABOVE IS THE UNSTAGED DEFAULT, NOT A CONSTANT -- ``DEXLIFT_POSE_TILT`` (a
    task-definition env var, see ``_apply_pose_tilt_stage`` in ``dexlift_ur5e_delto_env_cfg.py``)
    clamps the SAME ``pose_range.z`` down to ``[0.0, DEXLIFT_DROP_Z]`` (default 0.05) when set --
    production's own certified invocation sets it. Measured under that staged config: median
    obj_disp_from_spawn_mm drops to ~105 mm (from ~230-290 mm unstaged) and acceptance rises to
    ~58% (from ~24% with the plant and episode length already correct but tilt unset) -- tilt was
    the third and largest of the three production-matching factors, ahead of episode_length_s. The
    "use velocity, not displacement" guidance above holds regardless of which range is active;
    ``report()``'s own caveat print reads the live ``[verify]`` line rather than hardcoding either
    number, for exactly this reason.

    HARD RESTING FILTER, ENFORCED NOT JUST DOCUMENTED (team-lead decision). Because a bank is
    consumed by filename with nothing else validating its contents, ``_emit`` rejects any candidate
    whose ``obj_lin_vel_mag`` at the rewound step exceeds ``max_resting_speed_m_s`` (default 0.05
    m/s) BEFORE it reaches ``self.accum`` -- see ``self.rejected_not_resting``, reported per offset
    by both ``write()`` and ``report()``. Given the medians above this is expected to discard only a
    tail, not to gut the yield.

    ON THE ~2 cm QUESTION (deliberate, not a bug to chase). Upstream's own C2 route jitters a
    recorded grasp pose by roughly +/-2 cm; measured fingertip-to-object distance here at the
    locked offsets spans ~70-160 mm median (0.03-0.20 s) -- farther than that. This is accepted
    deliberately: upstream's jitter route is the one THIS project already measured, at 0.83% probe
    acceptance, with the hand hovering BESIDE the leg and zero fingertip contact (see this script's
    own module docstring and the bead history). These states are instead ON-MANIFOLD -- real
    configurations a trained policy passed through and could recover from -- which is a different,
    better-founded distribution than upstream's jitter, not a reproduction of its 2 cm number. Do
    not silently narrow the offsets trying to hit 2 cm; report the achieved distances plainly.

    SCHEMA PARITY WITH ``StableStateRecorder``/``_DexliftToTrainingSceneRecorder``. Each ring-buffer
    slot is captured in the exact ``scene.get_state(is_relative=True)`` shape
    (``articulation``/``rigid_object`` -> per-asset dict) PLUS ``joint_position_target``/
    ``joint_velocity_target`` read straight off the live ``Articulation.data`` buffers, mirroring
    ``recorders.py``'s own ``add_joint_targets`` call -- see ``reset_state_schema.py``'s docstring
    for why a bank missing those two fields silently zeroes the commanded PD squeeze on replay (an
    entire 465-iteration run read exactly 0.0000 success from that exact defect). The SAME
    ``_RENAME``/``_DROP`` rigid_object rekey ``_DexliftToTrainingSceneRecorder`` applies is reused
    here (read off that class, not restated), so a state written by this bank loads through
    ``MultiResetManager`` identically to one written by the accept-time path.
    """

    def __init__(
        self,
        env,
        success_term,
        offsets_s: list[float],
        control_hz: float,
        output_dir: str,
        reset_type_stem: str,
        max_resting_speed_m_s: float = 0.05,
    ):
        self.env = env
        self.success_term = success_term
        self.output_dir = output_dir
        self.reset_type_stem = reset_type_stem
        self.max_resting_speed_m_s = max_resting_speed_m_s
        num_envs = env.num_envs
        device = env.device

        # (offset_seconds, offset_steps, output_path) -- offset_steps computed off the CONSTRUCTED
        # env_cfg's own control rate (sim.dt * decimation), not a hardcoded 60 Hz, so this stays
        # correct if the task's control rate ever changes.
        self.offsets: list[tuple[float, int, str]] = []
        for off_s in offsets_s:
            off_steps = max(1, round(off_s * control_hz))
            fname = f"resets_{reset_type_stem}_off{off_s:.2f}s.pt"
            path = os.path.abspath(os.path.join(output_dir, fname))
            self.offsets.append((off_s, off_steps, path))

        self.ring_depth = max(steps for _, steps, _ in self.offsets) + 1
        self.ring: list[dict | None] = [None] * self.ring_depth
        self.control_hz = control_hz

        robot = env.scene[success_term.robot_cfg.name]
        self._arm_joint_ids, _ = robot.find_joints(list(DEXLIFT_ARM_JOINT_NAMES), preserve_order=True)
        # rl_dg_<n>_tip doubles as BOTH the ContactSensor filter name (held_with_probe's
        # thumb_contact_names/tip_contact_names) AND an articulation body name -- see
        # dexlift_ur5e_delto_env_cfg.py's TIP_BODY_REGEX (`r"rl_dg_(1|2|3|4|5)_tip"`), used the same
        # way for the `fingers_to_object` reward. Resolved ONCE here, not re-derived per step.
        self._thumb_body_ids, _ = robot.find_bodies(list(success_term.thumb_contact_names), preserve_order=True)
        self._tip_body_ids, _ = robot.find_bodies(list(success_term.tip_contact_names), preserve_order=True)

        # TWO anchors tracked in parallel -- see class docstring. `_any` drives capture; `_opposed`
        # is kept only so report() can print how much earlier `_any` fires.
        self.contacted_any_this_episode = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.contacted_opposed_this_episode = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.first_any_contact_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.first_opposed_contact_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._lead_deltas_steps: list[int] = []  # first_opposed - first_any, accepted episodes only

        self.pending: dict[int, dict] = {}  # env_idx -> {"offsets": {off_steps: capture}, "anchor_diag": {...}}
        self.spawn_obj_pos = torch.zeros(num_envs, 3, device=device)

        # accum[offset_steps] mirrors TorchDatasetFileHandler's on-disk shape directly: a python
        # list of leaf tensors (no leading batch dim -- see the (7,)-shaped leaves this tool's own
        # `torch.load` probe found in an existing bank), one entry per accepted+captured episode.
        self.accum: dict[int, dict] = {
            steps: {"initial_state": {"articulation": {}, "rigid_object": {}}} for _, steps, _ in self.offsets
        }
        self.diagnostics: dict[int, list[dict[str, float]]] = {steps: [] for _, steps, _ in self.offsets}
        self.n_any_contact_events = 0
        self.n_opposed_contact_events = 0
        self.emitted_counts: dict[int, int] = {steps: 0 for _, steps, _ in self.offsets}
        self.rejected_not_resting: dict[int, int] = {steps: 0 for _, steps, _ in self.offsets}
        # Set every step by capture_step(), BEFORE check_first_contact() reads them the same
        # iteration -- see main()'s call order. None here only documents that dependency.
        self._last_thumb_force: torch.Tensor | None = None
        self._last_tip_force: torch.Tensor | None = None

    def seed_spawn_positions(self) -> None:
        """Call ONCE, right after the initial ``wrapped_env.reset()``, so ``self.spawn_obj_pos``
        is valid for every env's FIRST episode (which never passes through ``finalize_episodes``'s
        own post-reset seeding -- that only runs on episode END)."""
        obj = self.env.scene[self.success_term.object_cfg.name]
        self.spawn_obj_pos[:] = obj.data.root_pos_w

    def capture_step(self, global_step: int) -> None:
        """Snapshot the FULL batch (every env, un-indexed) into the ring buffer this control step.
        Unconditional -- run every step regardless of ``dones``, so the buffer is always warm by
        the time a first-contact event needs to read backward out of it."""
        env = self.env
        origins = env.scene.env_origins
        state = {"articulation": {}, "rigid_object": {}}
        for name, art in env.scene._articulations.items():
            root_pose = art.data.root_pose_w.clone()
            root_pose[:, :3] -= origins
            state["articulation"][name] = {
                "root_pose": root_pose,
                "root_velocity": art.data.root_vel_w.clone(),
                "joint_position": art.data.joint_pos.clone(),
                "joint_velocity": art.data.joint_vel.clone(),
                # PD SET POINT (bead UWLab-algw.7) -- read directly off the live buffer, AT THIS
                # REWOUND STEP, exactly like recorders.py's add_joint_targets does at accept time.
                "joint_position_target": art.data.joint_pos_target.clone(),
                "joint_velocity_target": art.data.joint_vel_target.clone(),
            }
        for name, obj in env.scene._rigid_objects.items():
            root_pose = obj.data.root_pose_w.clone()
            root_pose[:, :3] -= origins
            state["rigid_object"][name] = {"root_pose": root_pose, "root_velocity": obj.data.root_vel_w.clone()}

        robot = env.scene[self.success_term.robot_cfg.name]
        obj = env.scene[self.success_term.object_cfg.name]
        thumb_force = _sensor_force_magnitudes(env, self.success_term.thumb_contact_names)
        tip_force = _sensor_force_magnitudes(env, self.success_term.tip_contact_names)
        # Cached for check_first_contact (called right after this, same iteration) to reuse -- ONE
        # force read per step, not two independently-threshold-checked ones that could silently
        # diverge from each other.
        self._last_thumb_force = thumb_force
        self._last_tip_force = tip_force

        obj_pos_w = obj.data.root_pos_w.clone()
        thumb_tip_pos_w = robot.data.body_pos_w[:, self._thumb_body_ids, :]
        tip_tip_pos_w = robot.data.body_pos_w[:, self._tip_body_ids, :]
        all_tip_pos_w = torch.cat([thumb_tip_pos_w, tip_tip_pos_w], dim=1)  # (num_envs, 5, 3)
        # RECALIBRATED distance metric (team-lead correction): nearest FINGERTIP body to the
        # object, not the palm (whose origin sits 15-20 cm from the natural grasp point -- see
        # class docstring). Meaningful in absolute mm: near zero at genuine contact.
        min_fingertip_obj_dist = torch.linalg.norm(all_tip_pos_w - obj_pos_w.unsqueeze(1), dim=-1).amin(dim=-1)

        diag = {
            "palm_pos_w": robot.data.body_pos_w[:, self.success_term._palm_id, :].clone(),
            "obj_pos_w": obj_pos_w,
            "min_fingertip_obj_dist": min_fingertip_obj_dist,
            "thumb_force_max": thumb_force.amax(dim=-1),
            "tip_force_max": tip_force.amax(dim=-1),
            "arm_joint_vel_mag": torch.linalg.norm(robot.data.joint_vel[:, self._arm_joint_ids], dim=-1),
            # Cheap, direct "is it still moving" signal -- unlike displacement-from-spawn, NOT
            # confounded by this task family's scripted airborne spawn (dexsuite Lift's own
            # reset_object draws pose_range.z in [0.0, 0.4] -- the object free-falls onto the table
            # every episode, so a large raw spawn-to-rewound displacement is expected regardless of
            # any hand contact; see report()'s printed caveat).
            "obj_lin_vel_mag": torch.linalg.norm(obj.data.root_lin_vel_w, dim=-1),
        }
        self.ring[global_step % self.ring_depth] = {"global_step": global_step, "state": state, "diag": diag}

    def check_first_contact(self, global_step: int, dones: torch.Tensor) -> None:
        """Detect, per env, the step EITHER anchor newly becomes True THIS episode.

        ANY-CONTACT (``self._last_thumb_force``/``self._last_tip_force``, set by ``capture_step``
        the same iteration, just above) drives capture: extract+stash every requested offset that
        is (a) still inside this episode and (b) still inside the ring buffer. OPPOSED-CONTACT is
        tracked in parallel, purely for the ``report()`` lead-time comparison -- see class
        docstring. any_now is implied by opposed_now (opposed requires BOTH sides loaded, any
        requires EITHER), so first_any_contact_step is always already set by the time
        first_opposed_contact_step fires.

        Skips envs in ``dones`` -- an env auto-reset THIS step has already started its next episode
        by the time this runs (ManagerBasedRLEnv auto-resets inside the same step() call; see
        held_check.py's own note on this exact trap), so contact tracking for that new episode
        should start fresh on a LATER step, not this one."""
        env = self.env
        threshold = self.success_term.force_threshold
        thumb_loaded = self._last_thumb_force.gt(threshold)
        tip_loaded = self._last_tip_force.gt(threshold)
        any_now = thumb_loaded.any(dim=-1) | tip_loaded.any(dim=-1)
        opposed_now = thumb_loaded.any(dim=-1) & tip_loaded.any(dim=-1)
        not_done = ~dones

        newly_opposed = opposed_now & (~self.contacted_opposed_this_episode) & not_done
        if newly_opposed.any():
            for i in torch.nonzero(newly_opposed).flatten().tolist():
                self.contacted_opposed_this_episode[i] = True
                self.first_opposed_contact_step[i] = global_step
                self.n_opposed_contact_events += 1

        newly_any = any_now & (~self.contacted_any_this_episode) & not_done
        if not newly_any.any():
            return
        steps_since_reset = env.episode_length_buf
        anchor_slot = self.ring[global_step % self.ring_depth]
        for i in torch.nonzero(newly_any).flatten().tolist():
            self.contacted_any_this_episode[i] = True
            self.first_any_contact_step[i] = global_step
            self.n_any_contact_events += 1
            captured: dict[int, dict] = {}
            for _, off_steps, _ in self.offsets:
                if off_steps > int(steps_since_reset[i].item()):
                    continue  # would reach before this episode's own reset
                src_step = global_step - off_steps
                if src_step < 0:
                    continue
                slot = self.ring[src_step % self.ring_depth]
                if slot is None or slot["global_step"] != src_step:
                    continue  # buffer not yet warm that far back (only possible near run start)
                captured[off_steps] = self._slice_env(slot, i)
            if captured:
                self.pending[i] = {"offsets": captured, "anchor_diag": self._slice_env(anchor_slot, i)["diag"]}

    @staticmethod
    def _slice_env(slot: dict, i: int) -> dict:
        def rec(d: dict) -> dict:
            out = {}
            for k, v in d.items():
                out[k] = rec(v) if isinstance(v, dict) else v[i].clone()
            return out

        return {"state": rec(slot["state"]), "diag": rec(slot["diag"])}

    def finalize_episodes(self, done_idx: torch.Tensor, success_now: torch.Tensor) -> None:
        """Called once per step for envs in ``dones``. Emits any pending captures for ACCEPTED
        episodes, records the any-vs-opposed lead time for the SAME accepted episodes, then
        unconditionally clears this env's bookkeeping and reseeds its spawn marker off the state
        auto-reset already wrote (mirrors the existing spawn-trace code's own "read OLD holder
        value before overwrite" ordering further up this file)."""
        obj = self.env.scene[self.success_term.object_cfg.name]
        for i in done_idx.tolist():
            spawn_pos_this_episode = self.spawn_obj_pos[i].clone()
            accepted = bool(success_now[i])
            if accepted and i in self.pending:
                anchor_diag = self.pending[i]["anchor_diag"]
                for off_steps, captured in self.pending[i]["offsets"].items():
                    self._emit(off_steps, captured, spawn_pos_this_episode, anchor_diag)
                any_step = int(self.first_any_contact_step[i].item())
                opp_step = int(self.first_opposed_contact_step[i].item())
                if any_step >= 0 and opp_step >= 0:
                    self._lead_deltas_steps.append(opp_step - any_step)
            self.contacted_any_this_episode[i] = False
            self.contacted_opposed_this_episode[i] = False
            self.first_any_contact_step[i] = -1
            self.first_opposed_contact_step[i] = -1
            self.pending.pop(i, None)
            self.spawn_obj_pos[i] = obj.data.root_pos_w[i].clone()

    def _emit(
        self, off_steps: int, captured: dict, spawn_pos_this_episode: torch.Tensor, anchor_diag: dict
    ) -> None:
        state = captured["state"]
        diag = captured["diag"]

        # HARD RESTING FILTER (team-lead decision, not just documented): a rewound candidate whose
        # object is still visibly moving is not a "near object, resting, untouched" state no matter
        # how the anchor was chosen -- enforce the property at emit time rather than trusting the
        # anchor+offset combination to have gotten it right for every episode. Measured medians at
        # the locked offsets are 0.000-0.049 m/s, so this is expected to discard only a tail.
        obj_speed = diag["obj_lin_vel_mag"].item()
        if obj_speed > self.max_resting_speed_m_s:
            self.rejected_not_resting[off_steps] += 1
            return

        rigid_object = state["rigid_object"]
        keys = set(rigid_object.keys())
        if keys not in _DexliftToTrainingSceneRecorder._KNOWN_SCHEMAS:
            raise ValueError(
                f"_C2RewindBank expected rigid_object keys in {_DexliftToTrainingSceneRecorder._KNOWN_SCHEMAS}, "
                f"got {sorted(keys)} -- refusing to silently mis-map an unexpected schema (same guard as "
                "_DexliftToTrainingSceneRecorder)."
            )
        rekeyed_rigid = {
            _DexliftToTrainingSceneRecorder._RENAME.get(name, name): tensors
            for name, tensors in rigid_object.items()
            if name not in _DexliftToTrainingSceneRecorder._DROP
        }
        export_state = {"articulation": state["articulation"], "rigid_object": rekeyed_rigid}

        def append_rec(dest: dict, src: dict) -> None:
            for k, v in src.items():
                if isinstance(v, dict):
                    append_rec(dest.setdefault(k, {}), v)
                else:
                    dest.setdefault(k, []).append(v.cpu())

        append_rec(self.accum[off_steps]["initial_state"], export_state)
        self.emitted_counts[off_steps] += 1

        # RECALIBRATED metrics (team-lead correction) -- see class docstring:
        #  - fingertip_obj_dist_mm: RAW, meaningful in absolute mm (fingertip origin ~= contact
        #    surface, unlike the palm's 15-20 cm offset).
        #  - palm_obj_dist_delta_mm: palm distance MINUS its own value at the anchor step, same
        #    episode -- removes the frame-offset constant, leaves only the approach gap. Positive
        #    means farther from the object now than at first contact; ~0 means no real gap.
        fingertip_obj_dist_mm = diag["min_fingertip_obj_dist"].item() * 1000.0
        palm_obj_dist_now = torch.linalg.norm(diag["palm_pos_w"] - diag["obj_pos_w"]).item()
        palm_obj_dist_anchor = torch.linalg.norm(anchor_diag["palm_pos_w"] - anchor_diag["obj_pos_w"]).item()
        palm_obj_dist_delta_mm = (palm_obj_dist_now - palm_obj_dist_anchor) * 1000.0
        obj_disp_from_spawn_mm = torch.linalg.norm(diag["obj_pos_w"] - spawn_pos_this_episode).item() * 1000.0
        self.diagnostics[off_steps].append(
            {
                "fingertip_obj_dist_mm": fingertip_obj_dist_mm,
                "palm_obj_dist_delta_mm": palm_obj_dist_delta_mm,
                "obj_height_m": diag["obj_pos_w"][2].item(),
                "obj_disp_from_spawn_mm": obj_disp_from_spawn_mm,
                "obj_lin_vel_mag_m_s": diag["obj_lin_vel_mag"].item(),
                "thumb_force_max_n": diag["thumb_force_max"].item(),
                "tip_force_max_n": diag["tip_force_max"].item(),
                "arm_joint_vel_mag_rad_s": diag["arm_joint_vel_mag"].item(),
            }
        )

    def print_progress(self) -> None:
        """Unbuffered, one line, meant to share the SAME heartbeat cadence as the existing
        ``[progress]`` attempts/accepted line in ``main()`` -- ``self.accum``/``self.diagnostics``
        are only ever written to disk once, at the very end (``write()``), so without this there is
        NOTHING to watch mid-run for a caller who only cares about the C2 total (e.g. an isolated
        paper-scale run whose accept-time bank is a deliberate throwaway and whose real stopping
        criterion is the C2 total, not the accepted-episode count ``--num_reset_states`` actually
        gates on)."""
        total_emitted = sum(self.emitted_counts.values())
        total_rejected = sum(self.rejected_not_resting.values())
        per_offset = "  ".join(
            f"{off_s:.2f}s={self.emitted_counts[off_steps]}({self.rejected_not_resting[off_steps]}rej)"
            for off_s, off_steps, _ in self.offsets
        )
        print(f"[c2][progress] total_emitted={total_emitted}  total_rejected={total_rejected}  {per_offset}", flush=True)

    def write(self) -> None:
        """Assert requirement #1 (joint targets present) and flush one file per offset. A run that
        captured zero episodes for a given offset still writes nothing for it (there is nothing to
        assert or save) -- report() below says so explicitly rather than leaving a silent gap."""
        for off_s, off_steps, path in self.offsets:
            accum = self.accum[off_steps]
            n = self.emitted_counts[off_steps]
            rejected = self.rejected_not_resting[off_steps]
            if n == 0:
                print(
                    f"[c2] offset={off_s:.2f}s: 0 episodes captured ({rejected} rejected by the "
                    f"resting filter) -- NOT writing {path}",
                    flush=True,
                )
                continue
            for asset_name, asset_state in accum["initial_state"]["articulation"].items():
                missing = [k for k in ("joint_position_target", "joint_velocity_target") if k not in asset_state]
                assert not missing, (
                    f"[c2] offset={off_s:.2f}s articulation {asset_name!r} is missing {missing} -- "
                    "refusing to write a bank with a zeroed commanded PD squeeze on replay."
                )
            atomic_torch_save(accum, path)
            print(
                f"[c2] offset={off_s:.2f}s: wrote {n} episodes ({rejected} rejected by the resting "
                f"filter, |v|>{self.max_resting_speed_m_s} m/s) -> {path}",
                flush=True,
            )

    def report(self, n_attempts: int | None = None) -> None:
        print("\n=== C2-VIA-REWIND RESULT (bead UWLab-weyl) ===", flush=True)
        print(f"any-contact events observed:     {self.n_any_contact_events}", flush=True)
        print(f"opposed-contact events observed: {self.n_opposed_contact_events}", flush=True)

        # YIELD (team-lead ask): states banked per attempted episode, across ALL offsets combined
        # -- this is the multiplier that makes C2-via-rewind cheaper than one-state-per-episode
        # accept-time generation, and what paper-scale planning needs.
        total_emitted = sum(self.emitted_counts.values())
        total_rejected = sum(self.rejected_not_resting.values())
        print(f"\ntotal C2 states emitted (all offsets): {total_emitted}  (rejected by resting filter: {total_rejected})", flush=True)
        if n_attempts:
            print(f"states per attempted episode: {total_emitted / n_attempts:.3f}  (n_attempts={n_attempts})", flush=True)

        # ANY-vs-OPPOSED anchor lead time, side by side (team-lead ask): how much earlier does the
        # any-contact anchor fire, for the SAME accepted episodes, in both steps and seconds.
        if self._lead_deltas_steps:
            steps_t = torch.tensor(self._lead_deltas_steps, dtype=torch.float32)
            secs_t = steps_t / self.control_hz
            print(
                f"\nany-contact fires earlier than opposed-contact by (accepted episodes, n={len(self._lead_deltas_steps)}):",
                flush=True,
            )
            print(
                f"  steps: min={steps_t.min().item():.0f} median={steps_t.median().item():.0f} "
                f"max={steps_t.max().item():.0f} mean={steps_t.mean().item():.1f}",
                flush=True,
            )
            print(
                f"  secs:  min={secs_t.min().item():.3f} median={secs_t.median().item():.3f} "
                f"max={secs_t.max().item():.3f} mean={secs_t.mean().item():.3f}",
                flush=True,
            )
        else:
            print("\nany-contact fires earlier than opposed-contact by: no accepted episodes with both anchors set", flush=True)

        print(
            "\nCAVEAT on obj_disp_from_spawn_mm below: this task's own reset_object draws "
            "pose_range.z from SOME range around a free-fall drop -- SEE THIS RUN'S OWN [verify] "
            "events.reset_object.func / pose_range.z LINE ABOVE FOR THE ACTUAL VALUE, do not assume "
            "[0.0, 0.4] (the unstaged default) -- DEXLIFT_POSE_TILT, when set, clamps it down to "
            "[0.0, DEXLIFT_DROP_Z] (default 0.05) instead. Either way, every episode spawns the "
            "object AIRBORNE and free-falls it onto the table before acquisition, so a chunk of "
            "displacement from the literal spawn pose is expected from that scripted drop ALONE, "
            "regardless of any hand contact -- this metric cannot distinguish free-fall settling "
            "from hand-caused disturbance. obj_lin_vel_mag_m_s (near 0 once landed) and obj_height_m "
            "are the more trustworthy 'is it still resting, untouched' signals for this task family.",
            flush=True,
        )

        for off_s, off_steps, path in self.offsets:
            rows = self.diagnostics[off_steps]
            print(
                f"\n-- offset {off_s:.2f}s ({off_steps} control steps) -- emitted {len(rows)} "
                f"(rejected by resting filter: {self.rejected_not_resting[off_steps]}) -- {path}",
                flush=True,
            )
            if not rows:
                continue
            for key, label, unit in (
                ("fingertip_obj_dist_mm", "nearest fingertip-to-object distance (raw)", "mm"),
                ("palm_obj_dist_delta_mm", "palm-to-object distance, DELTA vs anchor step", "mm"),
                ("obj_height_m", "object height (world z)", "m"),
                ("obj_disp_from_spawn_mm", "object displacement from spawn (SEE CAVEAT ABOVE)", "mm"),
                ("obj_lin_vel_mag_m_s", "object linear velocity magnitude", "m/s"),
                ("thumb_force_max_n", "max thumb fingertip force", "N"),
                ("tip_force_max_n", "max non-thumb fingertip force", "N"),
                ("arm_joint_vel_mag_rad_s", "arm joint velocity magnitude", "rad/s"),
            ):
                vals = torch.tensor([r[key] for r in rows])
                print(
                    f"  {label:44s}: min={vals.min().item():8.3f} median={vals.median().item():8.3f} "
                    f"max={vals.max().item():8.3f} mean={vals.mean().item():8.3f}  ({unit})",
                    flush=True,
                )


def _opposed_contact(
    env,
    thumb_contact_names: tuple[str, ...],
    tip_contact_names: tuple[str, ...],
    force_threshold: float,
) -> torch.Tensor:
    """``(num_envs,)`` bool: at least one thumb-side tip (``rl_dg_1``/``rl_dg_5``) AND at least one
    non-thumb tip (``rl_dg_2``/``3``/``4``), each above ``force_threshold`` N, via
    ``dexlift.mdp.rewards._sensor_force_magnitudes``.

    SHARED BY ARM 1 (``_C4DeepestGraspBank.step``) AND ARM 2 (``TerminateOnGraspSuccess.__call__``)
    -- both a running-argmax anchor and a fast accept gate need "is this an opposed grasp right
    now", and both are genuinely pinned to this one function: neither can silently drift from the
    other on what "opposed" means.

    CORRECTED CLAIM (harness review): this is NOT true of all four places in this file that compute
    the same thumb-AND-tip AND. ``held_with_probe.__call__`` (``held_check.py``) and
    ``_C2RewindBank.check_first_contact`` STILL COMPUTE IT INLINE, not through this helper, and
    agree with it by CONSTRUCTION of the formula, not because they call it -- i.e. by coincidence of
    having been written the same way, not by a guarantee that prevents them drifting apart later.
    Left unrefactored deliberately, for different reasons each:
    - ``held_check.py``'s ``held_with_probe`` is the term that produced this project's certified
      baseline; refactoring it means re-earning the "byte-identical to before" property on the one
      class every existing measurement in this campaign is anchored to, for a benefit (avoiding
      drift on a formula that has been stable for the life of this file) that does not clearly
      outweigh that risk.
    - ``_C2RewindBank.check_first_contact`` reads ``self._last_thumb_force``/``_last_tip_force``,
      forces CACHED once per step by ``capture_step`` specifically so contact is read ONCE, not
      twice independently-thresholded (see that class's own comment, "ONE force read per step, not
      two ... that could silently diverge"). Calling this helper there would re-read
      ``_sensor_force_magnitudes`` a second time per step, undoing that.
    If either of those two ever needs to change what "opposed" means, update this function's
    formula AND those two call sites by hand -- they will not follow automatically.
    """
    thumb_force = _sensor_force_magnitudes(env, thumb_contact_names)
    tip_force = _sensor_force_magnitudes(env, tip_contact_names)
    thumb_loaded = thumb_force.gt(force_threshold).any(dim=-1)
    tip_loaded = tip_force.gt(force_threshold).any(dim=-1)
    return thumb_loaded & tip_loaded


def _quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` WXYZ quaternions -> ``(..., 3, 3)`` rotation matrices.

    Ported VERBATIM (formula, not re-derived) from the reference decomposition script
    (``c4_depth_decompose.py``, bead ask) that this whole gate reuses -- that script was validated
    by reproducing the KNOWN spawn distribution (depth 10.0-17.5mm, lateral 0.04mm, tilt 0.14deg)
    before being trusted. Deliberately NOT ``isaaclab.utils.math``'s equivalent: re-deriving the
    same result against a different implementation risks introducing a NEW sign bug in exactly the
    kind of geometry code this project has been burned by before, instead of reusing code already
    checked against ground truth.
    """
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    R = torch.empty(*q.shape[:-1], 3, 3, dtype=q.dtype, device=q.device)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _rotate(R: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``R: (...,3,3), v: (3,) or (...,3) -> (...,3)``. Same helper as the reference script."""
    if v.dim() == 1:
        v = v.expand(*R.shape[:-2], 3)
    return torch.einsum("...ij,...j->...i", R, v)


class _MatingFrameGeometry:
    """Pure geometry: decomposes a leg pose into the fixture's own mating ("target") frame -- tip
    depth below the bore mouth, lateral miss off the bore centreline, tilt of the leg's insertion
    axis vs. the bore's deep axis -- using the exact same math as ``c4_depth_decompose.py`` (the
    reference implementation the bead points at, validated there by reproducing the KNOWN
    partial_assemblies.pt spawn distribution before being trusted). ``_quat_wxyz_to_rotmat``/
    ``_rotate`` above are that script's own helpers, ported verbatim.

    EXTRACTED (team-lead ask, this file's own "traps already cost real time" section): this is now
    the ONE place the metadata-read + rotation math lives, instead of being duplicated between
    ``SeatedHeldWithProbe`` (the accept-time AND gate, DELIVERABLE 1) and Arm 1's
    ``_C4DeepestGraspBank`` (which needs the SAME decomposition, continuously, to find its
    argmax-depth anchor -- not just to gate at emit time). ``_SeatingGateAddon`` below wraps this
    with the receptive_object scene-entity resolution and the [min,max]/lateral/tilt band check;
    ``_C4DeepestGraspBank`` constructs this class directly (it needs the raw numbers, not a band
    decision against THIS band -- its own bank gate uses a deliberately different band, see
    ``--c4_rewind_depth_min_mm``'s help text).

    assembled_offset (each object's mating-feature position AND orientation) is READ FROM
    metadata.yaml AT RUNTIME, for both the leg and the fixture, never hardcoded -- this project has
    repeatedly been burned by a geometry constant quoted from memory going stale against the actual
    asset. The one number NOT read from metadata.yaml is the bore's ENGAGED SPAN
    (``--c4_engaged_span_mm``): metadata.yaml has no such field (only assembled_offset, which gives
    the SEAT, not the mouth), so it is a measured CLI constant instead.
    """

    def __init__(self, leg_usd_path: str, fixture_usd_path: str, engaged_span_mm: float, device):
        leg_metadata = task_mdp.utils.read_metadata_from_usd_directory(leg_usd_path)
        fixture_metadata = task_mdp.utils.read_metadata_from_usd_directory(fixture_usd_path)
        for name, path, metadata in (("leg", leg_usd_path, leg_metadata), ("fixture", fixture_usd_path, fixture_metadata)):
            assert metadata.get("assembled_offset") is not None, (
                f"_MatingFrameGeometry: {name} metadata.yaml (next to {path!r}) has no "
                "'assembled_offset' -- cannot decompose depth/lateral/tilt without it."
            )

        self.leg_offset_pos = torch.tensor(leg_metadata["assembled_offset"]["pos"], dtype=torch.float32, device=device)
        self.leg_offset_quat = torch.tensor(leg_metadata["assembled_offset"]["quat"], dtype=torch.float32, device=device)
        self.fixture_offset_pos = torch.tensor(
            fixture_metadata["assembled_offset"]["pos"], dtype=torch.float32, device=device
        )
        # fixture_offset_quat is read (for parity with the reference script) but UNUSED below: the
        # "target" frame this decomposition works in is built from R_fix alone (the fixture ROOT
        # orientation) plus a pure TRANSLATION by fixture_offset_pos -- neither this port nor the
        # reference c4_depth_decompose.py's analyze() ever ROTATES by fixture_offset_quat. That is
        # correct ONLY because OneLegInsertionFixture's own assembled_offset.quat happens to be
        # identity (target-frame axes == fixture-root-frame axes). It is NOT true in general: a
        # future fixture whose seat orientation differs from its root orientation would need the
        # target frame's axes rotated by fixture_offset_quat too (target_R = R_fix @ R(offset_quat)),
        # which this class does not implement -- a nonzero value would silently produce a
        # plausible-looking but WRONG lateral/tilt number, not an error. Asserted below instead of
        # merely documented, so a future pair fails loudly at construction rather than shipping a
        # quietly-wrong gate.
        self.fixture_offset_quat = torch.tensor(
            fixture_metadata["assembled_offset"]["quat"], dtype=torch.float32, device=device
        )
        assert torch.allclose(
            self.fixture_offset_quat.cpu(), torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-4
        ), (
            f"_MatingFrameGeometry: fixture ({fixture_usd_path!r}) assembled_offset.quat = "
            f"{fixture_metadata['assembled_offset']['quat']} is not identity (WXYZ [1,0,0,0]). This "
            "class's depth/lateral/tilt decomposition builds the mating 'target' frame from the "
            "fixture's ROOT orientation via a pure translation, never a rotation by "
            "fixture_offset_quat (see the comment immediately above) -- correct only when that quat "
            "IS identity. Refusing to silently produce a wrong lateral/tilt number for a fixture "
            "this geometry was never validated against."
        )

        self.engaged_span_m = float(engaged_span_mm) / 1000.0
        assert 0.0 < self.engaged_span_m, f"engaged_span_mm must be > 0, got {engaged_span_mm}"

        self._tip_local_axis = torch.tensor([-1.0, 0.0, 0.0], device=device)
        self._bore_deep_local_axis = torch.tensor([0.0, 0.0, -1.0], device=device)

    def decompose(
        self, leg_pos: torch.Tensor, leg_quat: torch.Tensor, fix_pos: torch.Tensor, fix_quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(depth_m, lateral_m, tilt_deg)``, each ``(num_envs,)`` -- same three
        quantities, same formula, as ``c4_depth_decompose.py``'s ``analyze()``."""
        R_leg = _quat_wxyz_to_rotmat(leg_quat)
        R_fix = _quat_wxyz_to_rotmat(fix_quat)

        leg_tip_world = leg_pos + _rotate(R_leg, self.leg_offset_pos)
        leg_tip_axis_world = _rotate(R_leg, self._tip_local_axis)
        leg_tip_axis_world = leg_tip_axis_world / leg_tip_axis_world.norm(dim=-1, keepdim=True)
        bore_deep_axis_world = _rotate(R_fix, self._bore_deep_local_axis)
        bore_deep_axis_world = bore_deep_axis_world / bore_deep_axis_world.norm(dim=-1, keepdim=True)

        R_fix_T = R_fix.transpose(-1, -2)
        leg_tip_in_fix_root = _rotate(R_fix_T, leg_tip_world - fix_pos)
        leg_tip_in_target = leg_tip_in_fix_root - self.fixture_offset_pos
        x_t, y_t, z_t = leg_tip_in_target.unbind(-1)

        depth_m = self.engaged_span_m - z_t
        lateral_m = torch.sqrt(x_t**2 + y_t**2)
        cosang = (leg_tip_axis_world * bore_deep_axis_world).sum(-1).clamp(-1.0, 1.0)
        tilt_deg = torch.rad2deg(torch.arccos(cosang))
        return depth_m, lateral_m, tilt_deg


class _SeatingGateAddon:
    """The "is the leg tip still genuinely seated in the bore" AND-term (DELIVERABLE 1), factored
    out as a plain composed object rather than a mixin -- so it can be attached to either
    ``held_with_probe`` (``SeatedHeldWithProbe``) or Arm 2's ``TerminateOnGraspSuccess``
    (``SeatedTerminateOnGrasp``) WITHOUT the two host classes' otherwise-unrelated ``__call__``
    bodies having to cooperate through multiple inheritance / MRO. Both host classes construct one
    of these in their own ``__init__`` and call ``.check(env)`` inside their own ``__call__``,
    ANDing the result onto whatever their own base decision already was -- exactly the shape
    ``SeatedHeldWithProbe`` used before this class existed.

    FAILS LOUDLY AT CONSTRUCTION, not inside a deferred callback: ``receptive_object_cfg.resolve()``
    is called here, synchronously, with an assert right after it -- the SAME defensive idiom
    ``held_with_probe`` already uses for ``robot_cfg``/``object_cfg`` (this codebase's own
    documented trap is a ``SceneEntityCfg.resolve()`` failure swallowed inside a deferred "at play"
    callback, surfacing only as a misleading ``TypeError`` on the NEXT reset).
    """

    def __init__(self, env, object_cfg: SceneEntityCfg, c4_cfg: dict):
        self.object_cfg = object_cfg
        self.receptive_object_cfg: SceneEntityCfg = SceneEntityCfg(
            c4_cfg.get("receptive_object_name", "receptive_object")
        )
        self.receptive_object_cfg.resolve(env.scene)
        assert self.receptive_object_cfg.name in env.scene.rigid_objects, (
            f"_SeatingGateAddon: {self.receptive_object_cfg.name!r} did not resolve to a rigid "
            "object in the scene -- --c4_seating_gate requires DEXLIFT_PARTIAL_ASSEMBLY=1 (the "
            "fixture/receptive_object entity) to already be present. Refusing to construct a gate "
            "that would silently have nothing to measure against."
        )

        leg_usd_path = c4_cfg["leg_usd_path"]
        fixture_usd_path = c4_cfg["fixture_usd_path"]
        self.geometry = _MatingFrameGeometry(leg_usd_path, fixture_usd_path, c4_cfg["c4_engaged_span_mm"], env.device)

        self.depth_min_m = float(c4_cfg["c4_depth_min_mm"]) / 1000.0
        self.depth_max_m = float(c4_cfg["c4_depth_max_mm"]) / 1000.0
        self.lateral_max_m = float(c4_cfg["c4_lateral_max_mm"]) / 1000.0
        self.tilt_max_deg = float(c4_cfg["c4_tilt_max_deg"])
        assert self.depth_min_m < self.depth_max_m, (
            f"c4_depth_min_mm ({c4_cfg['c4_depth_min_mm']}) must be < c4_depth_max_mm ({c4_cfg['c4_depth_max_mm']})"
        )

        n = env.num_envs
        device = env.device
        self.last_seated = torch.zeros(n, dtype=torch.bool, device=device)
        self.last_depth_mm = torch.zeros(n, device=device)
        self.last_lateral_mm = torch.zeros(n, device=device)
        self.last_tilt_deg = torch.zeros(n, device=device)

        print(
            f"[c4-seating-gate] ENABLED depth=[{c4_cfg['c4_depth_min_mm']:.2f},{c4_cfg['c4_depth_max_mm']:.2f}]mm "
            f"lateral<={c4_cfg['c4_lateral_max_mm']:.2f}mm tilt<={c4_cfg['c4_tilt_max_deg']:.2f}deg "
            f"engaged_span={c4_cfg['c4_engaged_span_mm']:.3f}mm  leg_usd={leg_usd_path}  fixture_usd={fixture_usd_path}",
            flush=True,
        )

    def check(self, env) -> torch.Tensor:
        obj = env.scene[self.object_cfg.name]
        fix = env.scene[self.receptive_object_cfg.name]
        depth_m, lateral_m, tilt_deg = self.geometry.decompose(
            obj.data.root_pos_w, obj.data.root_quat_w, fix.data.root_pos_w, fix.data.root_quat_w
        )
        self.last_depth_mm = depth_m * 1000.0
        self.last_lateral_mm = lateral_m * 1000.0
        self.last_tilt_deg = tilt_deg
        seated = (
            (depth_m >= self.depth_min_m)
            & (depth_m <= self.depth_max_m)
            & (lateral_m <= self.lateral_max_m)
            & (tilt_deg <= self.tilt_max_deg)
        )
        self.last_seated = seated
        return seated


class SeatedHeldWithProbe(dexlift_mdp.held_with_probe):
    """DELIVERABLE 1: ``held_with_probe`` AND a spatial "is the leg still genuinely seated in the
    bore" gate (``_SeatingGateAddon``). Constructed ONLY when ``--c4_seating_gate`` is passed -- see
    that flag's own help text in this file for the measured defect this closes (0/100 accepted
    states in a seated depth band, 60% with the tip already back at or above the mouth).

    THE GATE IS AN ADDITIONAL AND TERM, not a replacement: ``__call__`` still requires
    ``held_with_probe``'s own four gates (settled, opposed contact, co-move, probe-tracks) to pass
    -- this class only narrows an ALREADY-GRASPED state further, exactly like the existing
    ``abnormal_robot``/``object_out_of_bound`` AND at the end of the base class's own ``__call__``.

    CONFIGURATION COMES FROM ``env.cfg.c4_seating_gate_config``, NOT ``cfg.params`` (team-lead
    catch, third attempt). IsaacLab's manager construction (``manager_base.py``'s
    ``_resolve_common_term_cfg``) inspects ``__call__``'s SIGNATURE and requires it to match
    ``cfg.params``'s KEYS exactly, and ``TerminationManager.compute()`` re-passes that SAME dict as
    ``**kwargs`` on EVERY step (``termination_manager.py:168``), not only at construction -- so any
    non-empty ``cfg.params`` forces ``__call__`` to explicitly declare (and keep in sync with) every
    key, forever, on pain of a construction-time or step-time crash. ``held_with_probe`` sidesteps
    this by always being constructed with ``params={}``; this class does the same by threading its
    OWN configuration through a plain attribute on ``env.cfg`` instead, set by the generator's
    ``main()`` BEFORE ``gym.make()`` -- the identical mechanism ``MixtureResetObject`` already uses
    for ``classic_goal_prob``/``low_goal_prob``/``partial_assembly_prob``
    (``dexlift/mdp/episode_mixture.py``), for an unrelated reason (Hydra-override timing) but the
    same structural fix: read config off ``env.cfg`` directly, never explode it through
    ``cfg.params``. ``SeatedTerminateOnGrasp`` (Arm 2's seated variant) uses the identical mechanism
    off a DIFFERENT ``env.cfg`` attribute (``c4_terminate_on_grasp_config``), for the same reason.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        c4_cfg = getattr(env.cfg, "c4_seating_gate_config", None)
        assert c4_cfg is not None, (
            "SeatedHeldWithProbe constructed but env.cfg.c4_seating_gate_config is missing -- "
            "main() must set env_cfg.c4_seating_gate_config BEFORE gym.make() whenever this class "
            "is wired in as terminations.success. This is a construction-time wiring bug in THIS "
            "script, not a runtime one -- see this class's own docstring, 'CONFIGURATION COMES "
            "FROM' section, for why cfg.params is not used for this."
        )
        self._seating = _SeatingGateAddon(env, self.object_cfg, c4_cfg)

    def __call__(self, env) -> torch.Tensor:
        # SAME SIGNATURE AS THE BASE CLASS, deliberately -- (self, env), nothing else. See this
        # class's own docstring, "CONFIGURATION COMES FROM env.cfg.c4_seating_gate_config" section:
        # IsaacLab's manager construction validates __call__'s signature against cfg.params' keys,
        # and TerminationManager re-passes cfg.params as **kwargs on every step, not only at
        # construction -- so keeping params (and therefore this signature) at exactly what the base
        # class already uses is what makes that validation pass, the same way it already does for
        # held_with_probe itself.
        held = super().__call__(env)
        seated = self._seating.check(env)
        return held & seated

    def gate_breakdown(self, env) -> dict[str, torch.Tensor]:
        # Base class's own breakdown, cached THIS step by our super().__call__() above -- see that
        # method's own comment on why this must be a cache, not a live recompute, given
        # ManagerBasedRLEnv's auto-reset-inside-step() ordering. "seated" is appended the same way.
        # .copy() (harness review): super().gate_breakdown() returns the BASE instance's own
        # _last_breakdown dict, not a fresh one -- mutating it in place here would corrupt the base
        # class's own cache for anything else that reads it. Harmless today only because the base
        # reassigns _last_breakdown fresh every step; still one dict shared across a class boundary,
        # and this project has been bitten before by a value correct where produced, wrong where
        # consumed (see memory verified-here-applied-there).
        bd = super().gate_breakdown(env).copy()
        bd["seated"] = self._seating.last_seated.clone()
        return bd


class TerminateOnGraspSuccess(ManagerTermBase):
    """ARM 2 (bead UWLab-xp05.2): replaces ``held_with_probe``'s settle+probe displacement test
    with a FAST, N-consecutive-step opposed-contact-force check plus a low object-velocity check.
    The whole point is to let acceptance fire the moment a genuine grasp forms, instead of after
    the ~70-step settle+probe latency the epic's own root-cause analysis blames for the withdrawal
    this campaign exists to defeat -- see the module-level bead notes and this file's
    ``--c4_terminate_on_grasp`` help text.

    WHY CONTACT FORCE, NOT DISPLACEMENT (measured, not assumed): at n=298 contact-force grading
    agreed with the strict 20mm in-palm-displacement "held" metric 0.9597 of the time against that
    metric's own 0.5369 raw agreement rate, and the disagreement was ONE-SIDED (131 contact-only
    vs. 5 held-only) -- a policy legitimately repositioning the leg in-hand exceeds a 20mm
    displacement budget without ever losing the grip, so contact force is the better instrument
    here, not merely a faster one.

    THE RISK THIS MUST GUARD AGAINST: an instantaneous contact reading can catch a transient touch
    (a brush, a bounce) rather than a stable grasp -- that is exactly why ``held_with_probe``'s own
    probe exists. The mitigation here is DELIBERATELY NOT a displacement probe (that would
    reintroduce the exact latency this class exists to remove): instead, ``consecutive_steps_required``
    consecutive control steps of (opposed contact AND low object linear-velocity magnitude) must
    hold before ``__call__`` returns True. The counter resets to 0 on ANY step that fails either
    condition -- a single dropped step of contact is enough to require starting the count over, so
    a merely-touching-then-slipping episode cannot accumulate credit across separate brief contacts.

    CONFIGURATION COMES FROM ``env.cfg.c4_terminate_on_grasp_config``, NOT ``cfg.params`` -- same
    structural reason, and the SAME mechanism, as ``SeatedHeldWithProbe`` (see that class's own
    docstring): IsaacLab's manager construction validates ``__call__``'s signature against
    ``cfg.params``'s keys and ``TerminationManager.compute()`` re-passes that dict as ``**kwargs``
    every step, so ``__call__`` must stay ``(self, env)`` with ``params={}`` always.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        c4_cfg = getattr(env.cfg, "c4_terminate_on_grasp_config", None)
        assert c4_cfg is not None, (
            "TerminateOnGraspSuccess constructed but env.cfg.c4_terminate_on_grasp_config is "
            "missing -- main() must set env_cfg.c4_terminate_on_grasp_config BEFORE gym.make() "
            "whenever this class is wired in as terminations.success. See this class's own "
            "docstring, 'CONFIGURATION COMES FROM' section, for why cfg.params is not used."
        )

        self.robot_cfg: SceneEntityCfg = c4_cfg.get("robot_cfg", SceneEntityCfg("robot", body_names="rl_dg_mount"))
        self.object_cfg: SceneEntityCfg = c4_cfg.get("object_cfg", SceneEntityCfg("object"))
        self.thumb_contact_names = c4_cfg.get("thumb_contact_names", ("rl_dg_1_tip", "rl_dg_5_tip"))
        self.tip_contact_names = c4_cfg.get("tip_contact_names", ("rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip"))
        self.force_threshold = float(c4_cfg.get("force_threshold", 0.2))
        self.obj_speed_thresh = float(c4_cfg.get("obj_speed_thresh", 0.05))
        self.settle_steps = int(c4_cfg.get("settle_steps", 0))
        self.consecutive_steps_required = int(c4_cfg["consecutive_steps_required"])
        assert self.consecutive_steps_required >= 1, (
            f"consecutive_steps_required must be >= 1, got {self.consecutive_steps_required} -- a "
            "value of 0 would accept on a single instantaneous contact reading, exactly the "
            "transient-touch failure mode this class exists to reject (see docstring)."
        )

        self.robot_cfg.resolve(env.scene)
        self.object_cfg.resolve(env.scene)
        assert len(self.robot_cfg.body_ids) == 1, (
            f"TerminateOnGraspSuccess.robot_cfg must resolve to exactly one body, got {self.robot_cfg.body_ids}"
        )
        self._palm_id = self.robot_cfg.body_ids[0]

        n = env.num_envs
        device = env.device
        self._consecutive_count = torch.zeros(n, dtype=torch.long, device=device)
        self._last_breakdown: dict[str, torch.Tensor] = {}

        print(
            f"[c4-terminate-on-grasp] ENABLED consecutive_steps_required={self.consecutive_steps_required} "
            f"({self.consecutive_steps_required / 60.0:.3f}s @ 60Hz)  obj_speed_thresh={self.obj_speed_thresh}m/s "
            f"force_threshold={self.force_threshold}N  settle_steps={self.settle_steps}",
            flush=True,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._consecutive_count[env_ids] = 0

    def __call__(self, env) -> torch.Tensor:
        # SAME SIGNATURE AS held_with_probe -- (self, env). See docstring, 'CONFIGURATION COMES FROM'.
        obj = env.scene[self.object_cfg.name]
        steps = env.episode_length_buf

        opposed = _opposed_contact(env, self.thumb_contact_names, self.tip_contact_names, self.force_threshold)
        obj_speed = torch.linalg.norm(obj.data.root_lin_vel_w, dim=-1)
        low_obj_speed = obj_speed < self.obj_speed_thresh
        settled = steps > self.settle_steps

        instant_ok = opposed & low_obj_speed & settled
        # CONSECUTIVE-STEP counter, resets to 0 on ANY failing step (not merely "does not
        # increment") -- see docstring: a brief contact-then-slip must not accumulate credit across
        # separate touches.
        self._consecutive_count = torch.where(
            instant_ok, self._consecutive_count + 1, torch.zeros_like(self._consecutive_count)
        )
        stable_grasp = self._consecutive_count >= self.consecutive_steps_required

        self._last_breakdown = {
            "settled": settled,
            "opposed_contact": opposed,
            "low_obj_speed": low_obj_speed,
            "stable_grasp": stable_grasp.clone(),
        }

        held = stable_grasp
        # Same defensive AND as held_with_probe's own __call__ -- see that method's comment for why
        # this is read off the CURRENT-STEP termination buffers rather than trusted to coincidentally
        # not have fired.
        tm = env.termination_manager
        for name in ("abnormal_robot", "object_out_of_bound"):
            if name in tm.active_terms:
                held = held & ~tm.get_term(name)
        return held

    def gate_breakdown(self, env) -> dict[str, torch.Tensor]:
        # NOT a .copy() here: this IS self._last_breakdown, the same object __call__ just
        # reassigned fresh this step (see __call__ above: "self._last_breakdown = {...}", a new
        # dict literal every call, never mutated in place) -- so returning it directly is safe. A
        # caller that mutates the returned dict would be mutating THIS instance's current-step
        # cache, same as held_with_probe's own gate_breakdown does; SeatedTerminateOnGrasp.
        # gate_breakdown below is the one that must .copy() before adding "seated", since IT would
        # otherwise be the caller doing exactly that mutation.
        return self._last_breakdown


class SeatedTerminateOnGrasp(TerminateOnGraspSuccess):
    """``TerminateOnGraspSuccess`` AND the SAME spatial seating gate ``SeatedHeldWithProbe`` uses
    (``_SeatingGateAddon``), composed the identical way -- ``--c4_seating_gate`` composes with
    EITHER held-check mode, not just the probe-based one. See ``SeatedHeldWithProbe``'s docstring
    for why this is composition (a plain attribute, not multiple inheritance) and why configuration
    is threaded through ``env.cfg`` rather than ``cfg.params``.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        c4_cfg = getattr(env.cfg, "c4_seating_gate_config", None)
        assert c4_cfg is not None, (
            "SeatedTerminateOnGrasp constructed but env.cfg.c4_seating_gate_config is missing -- "
            "main() must set env_cfg.c4_seating_gate_config BEFORE gym.make() whenever this class "
            "is wired in as terminations.success."
        )
        self._seating = _SeatingGateAddon(env, self.object_cfg, c4_cfg)

    def __call__(self, env) -> torch.Tensor:
        held = super().__call__(env)
        seated = self._seating.check(env)
        return held & seated

    def gate_breakdown(self, env) -> dict[str, torch.Tensor]:
        # .copy() (harness review, same fix as SeatedHeldWithProbe): super().gate_breakdown()
        # returns TerminateOnGraspSuccess's own self._last_breakdown directly (see that method's
        # comment on why that one is safe on its own) -- mutating it here without copying would
        # corrupt the base instance's current-step cache for anything else reading it.
        bd = super().gate_breakdown(env).copy()
        bd["seated"] = self._seating.last_seated.clone()
        return bd


class _SpawnPoseToleranceAddon:
    """S_t's (bead dr-sj6.22) "is the leg still within tolerance of its OWN spawn pose" AND-term --
    the S_t analogue of ``_SeatingGateAddon``, factored out the SAME way (a plain composed object,
    not a mixin), for the SAME reason (attachable to either a probe-based or fast host without
    those hosts' otherwise-unrelated ``__call__`` bodies cooperating through MRO).

    ``V2_C3_DESIGN.md`` sec 5 / ``V2_ACCEPTANCE_CRITERIA.md`` sec 4: S_t is a horizontal peg with no
    mating frame, so ``_SeatingGateAddon`` would reject ~100% of valid S_t states -- the exact trap
    the retracted v1 ``stays_seated`` proposal fell into for S2'. This class is deliberately never
    composed with it. ``dexlift/mdp/c3_rung.py``/``c3_rung_core.py`` (bead dr-ai1.4, env-side) draw
    which half of C3 an episode is and set its GOAL accordingly; they define no acceptance predicate
    for either half (S1's own docstring says so explicitly: "what makes a state S1 is the ACCEPTANCE
    band applied downstream... not this displacement") -- this class is that downstream acceptance
    predicate for S_t, the sibling of ``_SeatingGateAddon`` for S1.

    THE MATH ITSELF lives in ``spawn_tolerance_core.py`` (pure torch, no Isaac, unit-tested in
    ``test_spawn_tolerance_stage.py`` without a GPU) -- this class is only the Isaac-touching half:
    resolving the object, reading its live pose and the COMMANDED GOAL off ``env``, and calling
    :func:`pose_distance`/:func:`within_spawn_tolerance`.

    TOLERANCES HAVE NO DEFAULT. ``pos_tol_m``/``rot_tol_rad`` are threaded straight into
    :class:`~uwlab_tasks.manager_based.manipulation.dexlift.mdp.spawn_tolerance_core.SpawnToleranceConfig`,
    whose own ``__post_init__`` raises if ``pos_tol_m`` is missing or non-positive, or if
    ``rot_tol_rad`` is given but non-positive -- see that class's own docstring
    ("TOLERANCES ARE OPEN, WITH NO DEFAULT"). Bead dr-sj6.24: these numbers are meant to be DERIVED
    from the R4 validation run's own measured grasp-induced displacement distribution, which is
    exactly what :attr:`last_pos_dist_m`/:attr:`last_rot_dist_rad` below (surfaced through
    ``SpawnToleranceHeldWithProbe.gate_breakdown``) exist to produce. Guessing a plausible-looking
    number here instead is exactly the failure R7 exists to prevent, and this campaign has already
    shipped one invented constant (``RESET_SPEC_V2.md`` sec 6 item 0, the withdrawn ``stays_seated``
    6.02%->43.19% pair).

    SECOND CORRECTION, 2026-08-29 (team-lead review, superseding the first). The first version of
    this class captured its OWN reference pose (either at the literal reset-time pose, then --
    after reading bead dr-ai1.18 -- at a self-detected "settled" moment). BOTH were wrong for the
    same reason team-lead named directly: a state this addon judges is ALSO the state
    ``C3RungGoalPoseCommand`` is separately, concurrently deciding is "the goal" -- two
    independently-computed notions of "the leg's reference pose" is the F27 defect class exactly,
    two individually-valid definitions consumed as if they were one. If this addon's own settle
    detection ever disagreed with c3_rung.py's by even one step, "acceptance" and "the goal" would
    silently describe two different poses.

    THE FIX: there is no self-captured reference pose left in this class at all. :meth:`check`
    reads the COMMANDED GOAL directly off ``env.command_manager.get_term(command_name).pose_command_w``
    (public API: ``CommandManager.get_term`` returns the live ``CommandTerm`` instance;
    ``pose_command_w`` is ``TaskStateVisPoseCommand``'s own already-computed WORLD-frame command
    buffer, ``task_state_vis.py:240-245``, the exact quantity ``success.py``'s ``goal_pose_error``
    recomputes independently by hand -- reading it here instead of recombining
    ``robot_root * pose_command_b`` a second time is the SAME reuse discipline applied one level
    deeper). ``command_name`` defaults to ``dexlift_mdp.GOAL_COMMAND_NAME`` (``"object_pose"``,
    ``success.py:100``), the term name ``_apply_c3_rung_stage`` installs
    ``C3RungGoalPoseCommand`` under (``env_cfg.commands.object_pose = mdp.upgrade_to_c3_rung(...)``)
    -- imported, not restated, so a rename on the env side cannot silently desync this addon.

    PRE-SETTLE WINDOW -- RESOLVED (bead dr-ai1.18, commit f1f3818, 2026-08-29). The prior version of
    this docstring flagged an open question here: no public accessor existed for "has this env's
    goal been repinned yet", so this class gated acceptance on an independently-evaluated copy of
    ``c3_rung_core.st_should_repin``'s three conditions, a narrower duplication of the same shape as
    the reference-pose bug two paragraphs up. c3-impl has since exposed
    ``C3RungGoalPoseCommand.goal_is_final`` (a property, ``return ~self._st_awaiting_repin`` --
    a derived view of the ONE latch, not a second buffer): ``False`` from reset until S_t's deferred
    re-pin fires, then ``True`` for the rest of the episode; always ``True`` for S1 (never re-pinned,
    goal is correct from arming). This class now reads that property directly -- :meth:`check` no
    longer recomputes any settle condition of its own, has no settle-related constructor parameters,
    and holds no local trust latch. There is exactly one place, ``c3_rung.py``, that decides whether
    a commanded goal is final; this class only reads it.

    An env whose goal is not yet final reads ``last_within_tolerance = False`` unconditionally
    (fails closed), same discipline ``_SeatingGateAddon``'s construction-time assert uses for
    "refuse to measure against nothing".
    """

    def __init__(
        self,
        env,
        object_cfg: SceneEntityCfg,
        pos_tol_m: float,
        rot_tol_rad: float | None = None,
        *,
        command_name: str = dexlift_mdp.GOAL_COMMAND_NAME,
    ) -> None:
        # NB: no default for pos_tol_m/rot_tol_rad at THIS signature either -- but a caller passing
        # an explicit ``None`` (e.g. an unset CLI flag threaded straight through) would not trip a
        # bare missing-argument TypeError, so the REAL validation lives in
        # SpawnToleranceConfig.__post_init__, invoked unconditionally right here.
        self.cfg = SpawnToleranceConfig(pos_tol_m=pos_tol_m, rot_tol_rad=rot_tol_rad)
        self.command_name = command_name

        self.object_cfg = object_cfg
        self.object_cfg.resolve(env.scene)
        assert self.object_cfg.name in env.scene.rigid_objects, (
            f"_SpawnPoseToleranceAddon: {self.object_cfg.name!r} did not resolve to a rigid object "
            "in the scene. Refusing to construct a gate that would silently have nothing to "
            "measure against."
        )
        # FAIL LOUDLY HERE, not on the first check() -- confirm the named command term actually
        # exists and is a C3RungGoalPoseCommand (has pose_command_w AND goal_is_final) before this
        # addon is trusted to read from it every step. Same "resolve at construction, not in a
        # deferred callback" idiom as object_cfg.resolve() above / _SeatingGateAddon's own
        # receptive_object_cfg check.
        goal_term = env.command_manager.get_term(self.command_name)
        assert hasattr(goal_term, "pose_command_w") and hasattr(goal_term, "goal_is_final"), (
            f"_SpawnPoseToleranceAddon: command term {self.command_name!r} "
            f"({type(goal_term).__name__}) is missing pose_command_w and/or goal_is_final -- this "
            "addon reads the COMMANDED GOAL (team-lead correction, dr-sj6.22) and whether it is "
            "FINAL yet (bead dr-ai1.18's public goal_is_final) and needs a C3RungGoalPoseCommand. "
            "Refusing to construct a gate that would read the wrong quantity or crash mid-run "
            "instead of at construction."
        )

        n = env.num_envs
        device = env.device
        self.last_pos_dist_m = torch.zeros(n, device=device)
        self.last_rot_dist_rad = torch.zeros(n, device=device)
        # SECOND rotation metric (team-lead ask, 2026-08-29): spin-invariant axis tilt, recorded
        # alongside last_rot_dist_rad's full quaternion angle, NEITHER used to gate acceptance below
        # -- see spawn_tolerance_core.py's own docstring, "TWO ROTATION METRICS". Both are surfaced
        # through gate_breakdown() so R4 (bead dr-sj6.24) collects them for every attempted state,
        # accepted or rejected, not only accepted ones.
        self.last_axis_tilt_rad = torch.zeros(n, device=device)
        self.last_within_tolerance = torch.zeros(n, dtype=torch.bool, device=device)

        rot_tol_str = "disabled" if self.cfg.rot_tol_rad is None else f"{math.degrees(self.cfg.rot_tol_rad):.2f}deg"
        print(
            f"[c3-st-spawn-tolerance-gate] ENABLED pos_tol={self.cfg.pos_tol_m * 1000.0:.2f}mm "
            f"rot_tol={rot_tol_str}  object={self.object_cfg.name}  command_name={self.command_name}"
            "  -- reference pose is the COMMANDED GOAL, gated on goal_is_final (bead dr-ai1.18),"
            " never a self-captured pose or a locally re-evaluated settle predicate",
            flush=True,
        )

    def reset(self, env_ids) -> None:
        """No-op: this class holds no per-env state of its own to clear any more -- goal_is_final
        and pose_command_w are both read fresh from the command term on every check(), and the
        command term owns its own reset() (called separately, by whatever wires terminations)."""
        del env_ids

    def check(self, env) -> torch.Tensor:
        obj = env.scene[self.object_cfg.name]
        live_pos_w = obj.data.root_pos_w
        live_quat_w = obj.data.root_quat_w

        goal_term = env.command_manager.get_term(self.command_name)
        goal_pos_w = goal_term.pose_command_w[:, :3]
        goal_quat_w = goal_term.pose_command_w[:, 3:]

        pos_dist_m, rot_dist_rad = pose_distance(goal_pos_w, goal_quat_w, live_pos_w, live_quat_w)
        self.last_pos_dist_m = pos_dist_m
        self.last_rot_dist_rad = rot_dist_rad
        # SECOND rotation metric, recorded not gated on -- see __init__'s comment on
        # last_axis_tilt_rad and spawn_tolerance_core.py's own "TWO ROTATION METRICS" docstring.
        self.last_axis_tilt_rad = axis_tilt_rad(goal_quat_w, live_quat_w)

        # goal_is_final: False while pose_command_w is still the provisional mid-air spawn pose
        # (S_t, pre-repin) -- fails closed rather than comparing against it. Always True for S1.
        within = within_spawn_tolerance(pos_dist_m, rot_dist_rad, self.cfg) & goal_term.goal_is_final
        self.last_within_tolerance = within
        return within


class SpawnToleranceHeldWithProbe(dexlift_mdp.held_with_probe):
    """S_t's acceptance criterion (bead dr-sj6.22): ``held_with_probe`` AND
    ``_SpawnPoseToleranceAddon.check()`` -- the S_t analogue of ``SeatedHeldWithProbe``. See
    ``_SpawnPoseToleranceAddon``'s own docstring for why S_t composes THIS addon and never
    ``_SeatingGateAddon``.

    CONFIGURATION COMES FROM ``env.cfg.c3_st_spawn_tolerance_config``, NOT ``cfg.params`` -- the
    SAME reason and mechanism ``SeatedHeldWithProbe`` uses ``env.cfg.c4_seating_gate_config`` for
    (see that class's own docstring, "CONFIGURATION COMES FROM"): ``TerminationManager.compute()``
    re-passes ``cfg.params`` as ``**kwargs`` on every step, so a non-empty ``cfg.params`` would
    force ``__call__`` to declare (and keep in sync with) every key forever.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        st_cfg = getattr(env.cfg, "c3_st_spawn_tolerance_config", None)
        assert st_cfg is not None, (
            "SpawnToleranceHeldWithProbe constructed but env.cfg.c3_st_spawn_tolerance_config is "
            "missing -- the caller must set env_cfg.c3_st_spawn_tolerance_config BEFORE gym.make() "
            "whenever this class is wired in as terminations.success, as a dict with an EXPLICIT "
            "'pos_tol_m' key (and optionally 'rot_tol_rad') -- see this class's own docstring, "
            "'CONFIGURATION COMES FROM', and _SpawnPoseToleranceAddon's own docstring for why there "
            "is no default to silently fall back to."
        )
        # command_name is the only OPTIONAL override left (bead dr-ai1.18 retired the
        # settle_min_steps/settle_speed_mps/settle_ang_speed_rad_s overrides along with the settle
        # predicate they tuned -- _SpawnPoseToleranceAddon now reads goal_is_final directly instead
        # of recomputing it, so there is nothing left here to override). Built as a kwargs dict so
        # an absent key truly means "use the addon's default", not "pass None and let SOMETHING ELSE
        # decide".
        addon_kwargs = {}
        if st_cfg.get("command_name") is not None:
            addon_kwargs["command_name"] = st_cfg["command_name"]
        self._spawn_tolerance = _SpawnPoseToleranceAddon(
            env, self.object_cfg, st_cfg.get("pos_tol_m"), st_cfg.get("rot_tol_rad"), **addon_kwargs
        )

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._spawn_tolerance.reset(env_ids)

    def __call__(self, env) -> torch.Tensor:
        # SAME SIGNATURE AS THE BASE CLASS, deliberately -- see SeatedHeldWithProbe's own docstring
        # on why (cfg.params must stay empty for the same reason there).
        held = super().__call__(env)
        within = self._spawn_tolerance.check(env)
        return held & within

    def gate_breakdown(self, env) -> dict[str, torch.Tensor]:
        # .copy() (same discipline as SeatedHeldWithProbe/SeatedTerminateOnGrasp): mutating the base
        # instance's own cached dict in place would corrupt it for anything else reading it.
        # "spawn_tolerance" plus the raw displacement are appended so a caller can both filter on
        # pass/fail AND collect the raw (pos_dist, rot_dist, axis_tilt) distribution R4 needs to
        # derive the tolerances themselves (bead dr-sj6.24) -- this is the "record the per-state
        # displacement into the output" requirement this class exists to satisfy. BOTH rotation
        # metrics are recorded (team-lead ask, 2026-08-29) for every attempted state, accepted or
        # rejected -- see _SpawnPoseToleranceAddon.check()/spawn_tolerance_core.py's own docstring,
        # "TWO ROTATION METRICS, DELIBERATELY BOTH KEPT, NEITHER CHOSEN".
        bd = super().gate_breakdown(env).copy()
        bd["spawn_tolerance"] = self._spawn_tolerance.last_within_tolerance.clone()
        bd["spawn_pos_dist_m"] = self._spawn_tolerance.last_pos_dist_m.clone()
        bd["spawn_rot_dist_rad"] = self._spawn_tolerance.last_rot_dist_rad.clone()
        bd["spawn_axis_tilt_rad"] = self._spawn_tolerance.last_axis_tilt_rad.clone()
        return bd


class _C4DeepestGraspBank:
    """ARM 1 (bead UWLab-xp05.1): banks the DEEPEST step (subject to opposed contact holding) of
    every episode the FULL settle+probe held-check (``held_with_probe`` / ``SeatedHeldWithProbe``,
    whichever is wired as ``terminations.success``) later accepts.

    MECHANISM DIFFERS FROM ``_C2RewindBank``'S FIXED-OFFSET RING BUFFER, DELIBERATELY. C2 rewinds a
    KNOWN, small, fixed number of steps (0.03-0.20s, <=12 control steps) backward from a known
    anchor event (first contact), so a small ring buffer sized to the largest offset suffices. Arm
    1's anchor -- argmax tip depth over every step opposed contact holds, for a window that can
    start any time after ``settle_steps`` and end wherever accept eventually fires (the epic's own
    measurement: accept does not latch before roughly step 70, and can latch considerably later) --
    is NOT a fixed offset from a known event; it can reach arbitrarily far back into an episode
    whose eventual length is not known in advance. A ring buffer wide enough to guarantee reaching
    it would have to be sized to the FULL episode length, which is both far more memory than C2's
    <=12-step ring and unnecessary: the identical information is available by tracking a single
    running best-so-far snapshot per env, updated online the instant a NEW deepest opposed-contact
    step is observed, and resolved (banked or discarded) the moment that episode's own accept
    decision is known. So this class captures the CANDIDATE directly at the step it becomes the new
    best -- no rewind, no ring -- and otherwise reuses every other piece of ``_C2RewindBank``'s
    design: the identical per-step state-capture schema (schema parity with ``StableStateRecorder``
    -- ``joint_position_target``/``joint_velocity_target`` present), the identical rigid_object
    rename/drop via ``_DexliftToTrainingSceneRecorder``, the identical hard emit-time-gate pattern
    (there: resting-speed; here: the depth/lateral/tilt band), the identical atomic write +
    ``report()`` diagnostics shape.

    CENTRAL RISK, stated here because it drives every design choice below: the state this class
    banks is NOT the state the probe validated -- it is an EARLIER step of the same episode, chosen
    purely by depth-while-opposed, that the probe never looked at. Only whether that episode's
    accept decision came back True is evidence the grasp was eventually genuine; nothing here
    verifies that THIS PARTICULAR earlier step was already a stable hold rather than a
    still-forming one. That is exactly why the dynamic hold test (``validate_c4_bank.py --part b``,
    NOT this script's own gate-rejection counters) is mandatory before this bank is trusted, not
    optional -- see the epic's own gate task (UWLab-xp05.6).

    ``settle_steps`` (default: ``held_check_core.SETTLE_STEPS``, i.e. the SAME 60-step floor
    ``held_with_probe`` itself uses) restricts candidate steps the same way ``held_with_probe``'s
    own settled gate does: a step before it is not trusted to be more than post-reset/post-spawn
    settling noise, even if opposed contact happens to read True that early. THIS IS A DESIGN
    CHOICE NOT EXPLICITLY REQUIRED BY THE BEAD TEXT (which says only "restricted to steps where
    opposed contact holds") -- flagged here and in the handoff report rather than silently added,
    since it is the one place this implementation narrows the bead's own anchor definition.
    """

    def __init__(
        self,
        env,
        success_term,
        geometry: _MatingFrameGeometry,
        receptive_object_name: str,
        output_path: str,
        settle_steps: int,
        depth_min_mm: float,
        depth_max_mm: float,
        lateral_max_mm: float,
        tilt_max_deg: float,
        max_speed_m_s: float = 0.05,
        speed_sweep_thresholds: list[float] | None = None,
        require_settle: bool = False,
    ):
        self.env = env
        self.success_term = success_term
        self.geometry = geometry
        self.output_path = output_path
        self.settle_steps = settle_steps
        self.depth_min_mm = depth_min_mm
        self.depth_max_mm = depth_max_mm
        self.lateral_max_mm = lateral_max_mm
        self.tilt_max_deg = tilt_max_deg
        self.max_speed_m_s = max_speed_m_s
        # REAL BANK GATE, PROMOTED FROM ANALYSIS (harness mandate, 2026-08-22): OFF by default now --
        # the settle x speed sweep measured ZERO would-emit states with settle required, at every
        # speed threshold including ungated, across >1000 attempts, and NONZERO the moment settle
        # is lifted. Keeping this True in the real path was measured to guarantee an empty bank
        # forever, not merely suspected -- see --c4_rewind_require_settle's own help text.
        self.require_settle = require_settle
        # SPEED-GATE SWEEP (harness mandate, 2026-08-22, after seeing speed_gated_steps=136233 vs.
        # candidate_updates=2693 -- roughly 50:1 -- on the FIRST relaunch's live progress). Root
        # concern: the leg spawns already deep (10-17.5mm, partial_assemblies.pt) and the policy
        # withdraws it to ~2-4mm, so a deep pose EXISTS in every episode by construction -- the open
        # question is whether a deep pose ever COINCIDES with a valid, non-transient grasp, and the
        # withdrawal window (object still deep AND still moving while the hand closes on it) is
        # exactly what a single fixed 0.05 m/s gate can silently discard wholesale if 0.05 turns out
        # to be below this plant's held-and-being-withdrawn noise floor. THESE THRESHOLDS DO NOT
        # GATE THE REAL BANK -- self.max_speed_m_s (unchanged, still the deployed gate) is what
        # step()'s own candidate_now/resting continues to use for actual banking. This sweep is
        # ANALYSIS-ONLY: every (opposed & settled & not-done) step this run sees is recorded into a
        # per-env, per-episode buffer regardless of its speed (collection time is unfiltered), and
        # at each episode's resolution this buffer is filtered AFTER THE FACT at each threshold in
        # this list to answer "what would the emit yield have been under a stricter/looser gate",
        # without needing a second rollout. See finalize_episodes/report() for the analysis itself.
        self.speed_sweep_thresholds = (
            list(speed_sweep_thresholds) if speed_sweep_thresholds else [0.05, 0.10, 0.25, 0.50, float("inf")]
        )

        self.receptive_object_cfg = SceneEntityCfg(receptive_object_name)
        self.receptive_object_cfg.resolve(env.scene)
        assert self.receptive_object_cfg.name in env.scene.rigid_objects, (
            f"_C4DeepestGraspBank: {receptive_object_name!r} did not resolve to a rigid object in "
            "the scene -- --c4_rewind_deepest requires DEXLIFT_PARTIAL_ASSEMBLY=1 (the "
            "fixture/receptive_object entity) to already be present."
        )

        num_envs = env.num_envs
        device = env.device
        self.best_depth_m = torch.zeros(num_envs, device=device)
        self.has_candidate = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.best_state: dict[int, dict] = {}  # env idx -> full per-env state snapshot
        self.best_diag: dict[int, dict] = {}  # env idx -> {"depth_mm", "lateral_mm", "tilt_deg"}

        self.accum: dict = {"initial_state": {"articulation": {}, "rigid_object": {}}}
        self.diagnostics: list[dict[str, float]] = []
        self.emitted_count = 0
        self.rejected_band_count = 0  # had a candidate, but it fell outside the bank gate band
        self.rejected_no_candidate_count = 0  # accepted episode, but opposed contact never held
        self.n_candidate_updates = 0
        # HARNESS ASK, 2026-08-22 (mid-run, "emitted=0 alone will not tell us WHY"): a bare
        # rejected_band COUNT does not distinguish "the anchor is marginal, the band is arguable"
        # from "the anchor is dead, no band choice rescues it" -- keep the actual depth/lateral/tilt
        # of every band-rejected candidate, not just how many there were. Appended in _emit(), same
        # place rejected_band_count already increments.
        self.rejected_band_diagnostics: list[dict[str, float]] = []
        # THE WHOLE THESIS OF ARM 1, made measurable (harness ask): for every ACCEPTED episode
        # (probe-validated, whether or not its rewind candidate then clears the emit band), the
        # deepest opposed-contact depth found DURING that episode -- i.e. self.best_diag[i]["depth_mm"]
        # at the moment finalize_episodes resolves it, before that per-env bookkeeping is cleared for
        # the next episode. None when the episode was accepted but never produced an opposed-contact
        # candidate at all (rejected_no_candidate). Appended in THE SAME per-step, ascending-env-index
        # order as IsaacLab's own RecorderManager.export_episodes writes the accept-time bank (verified
        # by reading recorder_manager.py: both iterate the same done env_ids list in the same order),
        # so index i here lines up 1:1 with the i-th state actually written to the accept-time bank --
        # main() uses that alignment to compute, post-hoc, the gap between "deepest moment reached"
        # and "depth at the accept-validated instant" without needing to read live scene state for an
        # env that has already auto-reset (impossible -- see step()'s own docstring on that trap).
        self.accepted_episode_deepest_candidate_depth_mm: list[float | None] = []
        self.n_speed_gated_steps = 0  # opposed & settled & not-done steps EXCLUDED for moving too fast

        # EARN THE THRESHOLD (team-lead ask, harness review): depth_mm/lateral_mm/tilt_deg/speed_m_s
        # for EVERY (opposed & settled & not-done) step this run sees, GATED OR NOT BY
        # max_speed_m_s -- so --c4_rewind_max_speed's 0.05 m/s (inherited from C2's untouched-
        # object-on-a-table context) is a MEASUREMENT against this plant's own held-in-hand noise
        # floor, and so the "speed at moments depth is already high" cross-tab (report()'s own
        # print) can be computed directly. One (k,4) tensor per step (k = qualifying envs that
        # step), concatenated once at report()/print_progress() time; fine at this run's scale,
        # revisit (e.g. streaming quantiles) before a paper-scale run if this grows unwieldy.
        self._candidate_samples: list[torch.Tensor] = []  # columns: depth_mm, lateral_mm, tilt_deg, speed_m_s

        # PER-EPISODE, EPHEMERAL: same steps as _candidate_samples above (opposed & not-done,
        # settled NOT required -- see step()'s own docstring on why), grouped per env and cleared
        # at that env's own episode boundary -- this is what the SWEEP needs (argmax depth among
        # steps satisfying a given (settle_required, speed<=T), WITHIN one episode), which a flat
        # run-wide accumulator cannot give (a later, slower, shallower step in a DIFFERENT episode
        # is not a valid substitute for "the deepest qualifying step in THIS episode"). Each tuple:
        # (depth_mm, lateral_mm, tilt_deg, speed_m_s, settled: bool, steps_since_reset: int).
        # buf[0] (the FIRST entry appended, chronologically) is by construction the FIRST step this
        # episode ever had opposed contact -- exactly what the harness's "depth/step at first
        # opposed contact" question needs, read for free off this same buffer in finalize_episodes.
        self.episode_candidate_buffer: dict[int, list[tuple[float, float, float, float, bool, int]]] = {}
        # THE WHOLE THESIS OF BOTH REMAINING ARMS (harness ask, 2026-08-22): per ATTEMPTED episode
        # (not just accepted ones -- this is about whether the hand ever touches the leg while deep
        # at all, a property of the policy's general behaviour, not of which episodes the probe
        # later validates), the depth and step index at the FIRST opposed-contact step. If this
        # depth distribution's median is ~2-3mm, contact never coincides with a deep pose and BOTH
        # Arm 1 and Arm 2 are dead (terminate-on-grasp only changes WHEN the snapshot is taken; it
        # cannot conjure contact that never happened). If it is 8-15mm, a real window exists.
        self.first_opposed_contact_depth_mm: list[float] = []
        self.first_opposed_contact_step: list[int] = []
        # GEOMETRIC HONESTY (harness ask, 2026-08-22): the bore's radial clearance is ~0.91mm
        # (tightest wall 10.9156mm vs. the leg's 10.004mm flat pilot) -- a leg GENUINELY inside the
        # bore cannot read much above ~1mm lateral. Depth alone cannot distinguish "inserted" from
        # "near the mouth, off-axis, whose axial projection happens to be large" -- lateral (and
        # tilt) are what actually detect insertion. Tracked alongside depth/step at the SAME
        # first-contact moment (buf[0]), not gated to a new pass over the buffer, so this is free.
        self.first_opposed_contact_lateral_mm: list[float] = []
        self.first_opposed_contact_tilt_deg: list[float] = []
        self.n_episodes_no_opposed_contact = 0  # attempted episodes with zero opposed-contact steps

        # WHICH BAND CONDITION IS BINDING (harness ask, 2026-08-22): for the "deep candidate" --
        # the single deepest buffered step of the WHOLE episode, no settle or speed restriction at
        # all (the widest possible anchor) -- computed for EVERY ATTEMPTED episode (not just
        # accepted ones), so this denominator is directly comparable to first_opposed_contact's own
        # (both are "out of every attempt", resolving the "why is the sweep's n so much smaller"
        # question: the SWEEP's own n is restricted to probe-accepted episodes by original design
        # -- Arm 1 only ever rewinds WITHIN an episode the full probe already validated -- but this
        # diagnostic classification is pure measurement, not banking, so it is not bound by that
        # restriction). Only episodes whose deepest point clears depth_min_mm (a "deep candidate"
        # in the harness's own words) are classified; a deep candidate that ALSO fails depth_max_mm
        # (over-inserted) counts as depth_only_fail, same category, since both are the same band
        # bound family.
        self.deep_candidate_band_breakdown = {
            "pass": 0, "depth_only_fail": 0, "lateral_only_fail": 0, "tilt_only_fail": 0, "multiple_fail": 0,
        }
        self.deep_candidate_diagnostics: list[dict[str, float]] = []

        # DEPTH-VS-STEP TRAJECTORY (harness mandate, TOP PRIORITY, 2026-08-22): median depth at
        # fixed step bins, aggregated across ALL episodes -- answers "does the leg spawn deep and
        # fall out on its own, before the hand ever touches it" (spawn/contact dynamics) vs. "is it
        # shallow from step 0" (a partial_assemblies.pt application defect, unrelated to any arm in
        # this epic). A running per-bin accumulator, not per-episode storage -- see step()'s own
        # comment for why one pass through the bins per step is exact, no ring buffer needed.
        self.depth_vs_step_bins = [0, 5, 10, 15, 20, 30, 45, 60]
        self.depth_vs_step_samples: dict[int, list[float]] = {b: [] for b in self.depth_vs_step_bins}
        # LATERAL-VS-STEP TRAJECTORY (harness mandate, 2026-08-22, SECOND priority curve -- added
        # after the band breakdown proved lateral, not depth, is the sole binding constraint on
        # "deep candidates"): same running per-bin accumulator shape as depth-vs-step, same bins,
        # so the two can be read paired -- do depth and lateral degrade TOGETHER (leg knocked
        # off-axis while also being withdrawn) or SEPARATELY (leg stays deep but drifts off-axis,
        # or goes off-axis but stays at depth)? Also answers, at bin 0 specifically, whether the
        # SPAWNED state itself is already off-axis (a spawn-path defect, not a policy behaviour --
        # partial_assemblies.pt's own file-level lateral miss (median 0.035mm) was verified on the
        # STORED poses, never on the REALISED state after physics settles at step 0).
        self.lateral_vs_step_samples: dict[int, list[float]] = {b: [] for b in self.depth_vs_step_bins}

        # SETTLED-DEPTH-VS-SPAWN-DEPTH (harness mandate, THE measurement, 2026-08-22): does settled
        # depth TRACK spawn depth (a deep C4 bank is reachable, and this curve says from which spawn
        # band) or does everything converge to one attractor regardless of spawn depth (a new,
        # better-than-2-4mm equilibrium, still short of the true 22.5mm gate, needing a further
        # mechanism)? The existing depth_vs_step accumulator is a FLAT, cross-episode aggregate at
        # fixed step numbers -- it cannot answer this, because it never pairs one episode's OWN
        # spawn depth with that SAME episode's OWN settled depth. This does: per env, spawn_depth_mm
        # is captured once at steps_since_reset==0 (the fresh-spawn step), and last_depth_mm is
        # updated EVERY step EXCEPT for envs in `dones` this step (auto-reset already overwrote their
        # live state before this runs -- the usual trap) -- so at the moment finalize_episodes sees an
        # env in done_idx, last_depth_mm[i] still holds the LAST live reading of the ENDING episode,
        # not the new one. One (spawn_depth_mm, settled_depth_mm) pair appended per completed episode.
        #
        # SAME-ITERATION COLLISION, caught before it became a silent corruption: an env whose OLD
        # episode ends (dones[i]=True) THIS iteration and whose auto-reset gives it a brand new
        # episode ALSO within this SAME wrapped_env.step() call reads steps_since_reset[i]==0 (the
        # NEW episode) at the exact same iteration finalize_episodes needs the OLD episode's own
        # spawn value. Writing spawn_depth_mm[i] unconditionally in step() would overwrite the OLD
        # value with the NEW one before finalize_episodes ever reads it. Fix: step() stashes a
        # fresh-spawn-that-coincides-with-a-done into _pending_new_spawn_depth_mm instead of writing
        # spawn_depth_mm directly; finalize_episodes reads+consumes the OLD spawn_depth_mm[i] FIRST,
        # THEN promotes the pending value (if any) into spawn_depth_mm[i] for the new episode.
        num_envs_local = env.num_envs
        self.spawn_depth_mm = torch.full((num_envs_local,), float("nan"), device=env.device)
        self.last_depth_mm = torch.full((num_envs_local,), float("nan"), device=env.device)
        self._pending_new_spawn_depth_mm: dict[int, float] = {}
        self.spawn_settled_pairs: list[tuple[float, float]] = []

        # SWEEP DIMENSIONS (harness mandate, 2nd catch, 2026-08-22): settled is swept the SAME way
        # speed is -- required vs. not-required -- crossed with every speed threshold, so "the deep
        # window is pre-settle" becomes a measured cell in this table rather than a second run.
        # settle_required=True reproduces the real bank's own settled gate exactly; False lifts it.
        self.settle_sweep_options = (True, False)
        # THIRD SWEEP DIMENSION (harness mandate, 2026-08-22, third catch on this same question):
        # Arm 1's own DESIGN restricts the rewind anchor to episodes the full settle+probe held-
        # check ALREADY validated (accepted=True) -- this bounds the arm's yield above by the
        # accept-time acceptance rate (~6%) BEFORE the emit band is even applied, and the probe
        # validates the EPISODE'S OWN TERMINAL state, ~70 steps after the mid-episode state being
        # rewound-to -- it says nothing about whether THAT EARLIER state was physically sound (only
        # part B, real physics, answers that). require_probe_accepted=True reproduces that original
        # restriction; False lifts it, evaluating every attempted episode's own buffered candidates
        # regardless of whether the probe later validated the episode. Real bank behaviour is
        # UNCHANGED either way -- this is the third analysis-only sweep axis, not a new gate.
        self.probe_accept_sweep_options = (True, False)
        # SWEEP RESULTS, keyed by (require_probe_accepted, settle_required, threshold) -- threshold
        # may be inf for "ungated". Populated in finalize_episodes for EVERY attempted episode
        # (accepted or not) -- mirrors the real bank's own accept-then-emit-band-check shape, just
        # re-run at analysis time against each (probe-accept, settle, speed) cell instead of only
        # the deployed (True, True, max_speed_m_s) one.
        self.sweep_stats: dict[tuple[bool, bool, float], dict] = {
            (require_probe_accepted, settle_required, T): {
                "n_would_have_candidate": 0, "n_would_emit": 0,
                "would_emit_depths_mm": [], "would_emit_lateral_mm": [], "would_emit_tilt_deg": [],
            }
            for require_probe_accepted in self.probe_accept_sweep_options
            for settle_required in self.settle_sweep_options
            for T in self.speed_sweep_thresholds
        }

        print(
            f"[c4-rewind-deepest] ENABLED require_settle={require_settle} (settle_steps={settle_steps})  "
            f"max_speed={max_speed_m_s}m/s  bank_gate depth=[{depth_min_mm:.2f},{depth_max_mm:.2f}]mm "
            f"lateral<={lateral_max_mm:.2f}mm tilt<={tilt_max_deg:.2f}deg  -> {output_path}",
            flush=True,
        )

    def _snapshot_env(self, i: int) -> dict:
        """Same per-asset fields as ``_C2RewindBank.capture_step``, for ONE env only (no ring
        buffer here -- see class docstring for why this is captured on demand, at the instant it
        becomes the new best, rather than read back out of a buffer)."""
        env = self.env
        origin = env.scene.env_origins[i]
        state = {"articulation": {}, "rigid_object": {}}
        for name, art in env.scene._articulations.items():
            root_pose = art.data.root_pose_w[i].clone()
            root_pose[:3] -= origin
            state["articulation"][name] = {
                "root_pose": root_pose,
                "root_velocity": art.data.root_vel_w[i].clone(),
                "joint_position": art.data.joint_pos[i].clone(),
                "joint_velocity": art.data.joint_vel[i].clone(),
                "joint_position_target": art.data.joint_pos_target[i].clone(),
                "joint_velocity_target": art.data.joint_vel_target[i].clone(),
            }
        for name, obj in env.scene._rigid_objects.items():
            root_pose = obj.data.root_pose_w[i].clone()
            root_pose[:3] -= origin
            state["rigid_object"][name] = {"root_pose": root_pose, "root_velocity": obj.data.root_vel_w[i].clone()}
        return state

    def step(self, dones: torch.Tensor) -> None:
        """Call every control step (unconditional, same discipline as ``_C2RewindBank.capture_step``
        -- the deepest opposed-contact step can occur on any step, not just when something ends).

        Skips envs in ``dones`` -- an env auto-reset THIS step has already started its next episode
        by the time this runs (see ``_C2RewindBank.check_first_contact``'s own note on this exact
        trap), so its running-best bookkeeping must not be updated from the NEW episode's first
        step here; ``finalize_episodes`` below resolves and clears the OLD episode's bookkeeping.

        CANDIDATE-SELECTION-TIME VELOCITY GATE (harness review, UWLab-xp05.6): a step with opposed
        contact but a still-moving object is not evidence of a stable seat -- it is as likely to be
        the instant contact was FIRST made, still decelerating into position, and the emit-time
        depth/lateral/tilt band cannot catch this because a transient-but-deep reading can still
        land inside that band (the band is a SPATIAL check, this is a MOTION check, orthogonal).
        Mirrors C2's own hard resting-speed filter (``--c2_max_resting_speed``, default 0.05 m/s),
        but applied HERE, at candidate-selection time, rather than at emit time only: a fast-moving
        step that would otherwise become the running best-so-far is simply never considered a
        candidate at all, so a genuinely-seated LATER (shallower-looking-but-actually-stable) step
        is free to become the best instead of being pre-empted by an earlier transient. C2's own
        filter stays emit-time-only because C2 has exactly one candidate per offset per episode (no
        argmax-over-time to protect); Arm 1's running-max needs the earlier gate for the reason
        above.

        SETTLED IS DROPPED FROM SWEEP/DIAGNOSTIC COLLECTION, DELIBERATELY (harness catch,
        2026-08-22, second one this file owes to the same reviewer): the REAL bank's own
        ``candidate_now`` below still requires ``settled`` (steps_since_reset > settle_steps,
        currently 60) -- that gate is UNCHANGED. But the leg SPAWNS deep (10-17.5mm,
        partial_assemblies.pt) and is withdrawn during the grasp, i.e. the deep window is the
        FIRST ~60 steps of every episode -- exactly what ``settled`` excludes. Requiring settled at
        SWEEP-collection time as well would have silently reproduced the epic's own root-cause
        blindness inside the very instrument built to diagnose it: a settled-only sweep can only
        ever answer "what if the speed gate were looser", never "what if the settle gate were
        looser", because it would never have SEEN a pre-settle deep step to sweep over in the first
        place. So the diagnostic collection mask below is ``opposed & not-done`` ONLY -- every
        step, pre- or post-settle -- and ``settled`` is carried as its own boolean alongside
        depth/lateral/tilt/speed, so the sweep can filter on it exactly like it filters on speed."""
        env = self.env
        steps_since_reset = env.episode_length_buf
        obj = env.scene[self.success_term.object_cfg.name]
        fix = env.scene[self.receptive_object_cfg.name]
        depth_m, lateral_m, tilt_deg = self.geometry.decompose(
            obj.data.root_pos_w, obj.data.root_quat_w, fix.data.root_pos_w, fix.data.root_quat_w
        )

        # DEPTH-VS-STEP TRAJECTORY (harness mandate, top priority, 2026-08-22): does the leg
        # withdraw from ITS OWN SPAWN DEPTH before the hand ever makes contact, or is depth already
        # shallow from step 0 (which would instead mean partial_assemblies.pt poses are not being
        # applied as intended -- a different, more serious defect upstream of every arm in this
        # epic)? UNCONDITIONAL on opposed/settled/dones -- this is a property of the object's own
        # trajectory, not of the grasp. steps_since_reset increments by exactly 1 per control step
        # and a just-reset env reads 0 at THIS point (auto-reset already happened inside
        # wrapped_env.step() before this runs), so every env passes through each integer bin value
        # exactly once per episode; no ring buffer or per-episode state needed, just bin by the
        # exact value seen this step.
        # LATERAL-VS-STEP TRAJECTORY, paired with depth-vs-step above (same bins, same mask, same
        # unconditional-on-opposed/settled/dones scope) -- see lateral_vs_step_samples' own comment
        # in __init__ for why this is now the SECOND curve, not an afterthought.
        for _bin in self.depth_vs_step_bins:
            _bin_mask = steps_since_reset == _bin
            if _bin_mask.any():
                self.depth_vs_step_samples[_bin].extend((depth_m[_bin_mask] * 1000.0).tolist())
                self.lateral_vs_step_samples[_bin].extend((lateral_m[_bin_mask] * 1000.0).tolist())

        # SETTLED-DEPTH-VS-SPAWN-DEPTH bookkeeping (see __init__'s own comment, including the
        # same-iteration-collision fix). spawn_depth_mm is captured once, at the fresh-spawn step
        # -- but ONLY for envs NOT also in `dones` this same iteration, since a fresh spawn
        # coinciding with a done means the OLD episode's own spawn_depth_mm[i] has not been
        # consumed by finalize_episodes yet; that case is stashed in _pending_new_spawn_depth_mm
        # instead, and finalize_episodes promotes it AFTER reading the old value out.
        # last_depth_mm updates EVERY OTHER step (masked out for envs in `dones` this step, so a
        # done env's own auto-reset -- already applied before this runs -- never overwrites the
        # ending episode's last live reading).
        _spawn_mask = steps_since_reset == 0
        if _spawn_mask.any():
            _spawn_and_not_done = _spawn_mask & (~dones)
            if _spawn_and_not_done.any():
                self.spawn_depth_mm[_spawn_and_not_done] = depth_m[_spawn_and_not_done] * 1000.0
            _spawn_and_done = _spawn_mask & dones
            if _spawn_and_done.any():
                for _i in torch.nonzero(_spawn_and_done).flatten().tolist():
                    self._pending_new_spawn_depth_mm[_i] = depth_m[_i].item() * 1000.0
        _not_done_mask = ~dones
        if _not_done_mask.any():
            self.last_depth_mm[_not_done_mask] = depth_m[_not_done_mask] * 1000.0

        opposed = _opposed_contact(
            env, self.success_term.thumb_contact_names, self.success_term.tip_contact_names,
            self.success_term.force_threshold,
        )
        settled = steps_since_reset > self.settle_steps
        obj_speed = torch.linalg.norm(obj.data.root_lin_vel_w, dim=-1)
        resting = obj_speed <= self.max_speed_m_s

        # DIAGNOSTIC/SWEEP COLLECTION MASK: opposed & not-done ONLY -- settled is NOT required
        # here (see docstring). Recorded UNFILTERED by speed as well, into the flat run-wide
        # accumulator (percentiles / depth-conditioned cross-tab) and the per-env, per-episode
        # buffer (the settle x speed sweep, which needs within-episode grouping a flat accumulator
        # cannot give). This is strictly BROADER than the real bank's own candidacy mask below.
        collect_mask = opposed & (~dones)
        if collect_mask.any():
            _sample_idx = torch.nonzero(collect_mask).flatten()
            self._candidate_samples.append(
                torch.stack(
                    [depth_m[_sample_idx] * 1000.0, lateral_m[_sample_idx] * 1000.0,
                     tilt_deg[_sample_idx], obj_speed[_sample_idx], settled[_sample_idx].float()],
                    dim=-1,
                ).detach().cpu()
            )
            for i in _sample_idx.tolist():
                self.episode_candidate_buffer.setdefault(i, []).append(
                    (
                        depth_m[i].item() * 1000.0, lateral_m[i].item() * 1000.0, tilt_deg[i].item(),
                        obj_speed[i].item(), bool(settled[i].item()), int(steps_since_reset[i].item()),
                    )
                )

        # REAL BANK CANDIDACY. ``require_settle`` gates whether ``settled`` is ANDed in here at all
        # -- OFF by default now (harness mandate, promoted from analysis to deployment: the sweep
        # measured zero would-emit states with settle required, at every speed threshold, across
        # >1000 attempts). The sweep above is analysis-only regardless and does not feed this.
        settled_opposed_mask = (opposed & settled & (~dones)) if self.require_settle else (opposed & (~dones))
        speed_gated = settled_opposed_mask & (~resting)
        if speed_gated.any():
            self.n_speed_gated_steps += int(speed_gated.sum())

        candidate_now = settled_opposed_mask & resting
        if not candidate_now.any():
            return
        for i in torch.nonzero(candidate_now).flatten().tolist():
            d = depth_m[i].item()
            if (not bool(self.has_candidate[i])) or d > self.best_depth_m[i].item():
                self.best_depth_m[i] = d
                self.has_candidate[i] = True
                self.best_state[i] = self._snapshot_env(i)
                self.best_diag[i] = {
                    "depth_mm": d * 1000.0,
                    "lateral_mm": lateral_m[i].item() * 1000.0,
                    "tilt_deg": tilt_deg[i].item(),
                }
                self.n_candidate_updates += 1

    def finalize_episodes(self, done_idx: torch.Tensor, success_now: torch.Tensor) -> None:
        """Call once per step for envs in ``dones`` (mirrors ``_C2RewindBank.finalize_episodes``'s
        own call site/ordering, and -- load-bearing for the ordinal alignment
        ``accepted_episode_deepest_candidate_depth_mm`` relies on -- the SAME per-step ``done_idx``
        RecorderManager.export_episodes iterates to write the accept-time bank). Emits the
        running-best candidate for every ACCEPTED episode that had one, then unconditionally clears
        bookkeeping for the env's next episode."""
        for i in done_idx.tolist():
            accepted = bool(success_now[i])
            # SETTLED-DEPTH-VS-SPAWN-DEPTH pair, unconditional on accepted (this measures the
            # ASSET/geometry, not the probe) -- read BEFORE any reset of this bookkeeping. NaN spawn
            # (env never got a step==0 reading this run, e.g. it was already mid-episode at start)
            # is skipped rather than recorded as a bogus pair.
            _spawn_val = float(self.spawn_depth_mm[i].item())
            _settled_val = float(self.last_depth_mm[i].item())
            if _spawn_val == _spawn_val:  # not NaN
                self.spawn_settled_pairs.append((_spawn_val, _settled_val))
            # Promote the NEW episode's own spawn reading (stashed by step() this SAME iteration
            # because it coincided with this done) now that the OLD value has been consumed above
            # -- otherwise fall back to NaN exactly as before.
            self.spawn_depth_mm[i] = self._pending_new_spawn_depth_mm.pop(i, float("nan"))
            # THE WHOLE THESIS OF BOTH REMAINING ARMS (harness ask): for EVERY attempted episode,
            # not just accepted ones -- this is about whether the hand ever touches the leg while
            # deep AT ALL, a property of the policy's general behaviour -- record the depth and
            # step index at the FIRST opposed-contact step. buf[0] is that step by construction
            # (episode_candidate_buffer is appended in chronological step order, only on opposed
            # contact). Read BEFORE the buffer is cleared below.
            _buf_for_first_contact = self.episode_candidate_buffer.get(i)
            if _buf_for_first_contact:
                self.first_opposed_contact_depth_mm.append(_buf_for_first_contact[0][0])
                self.first_opposed_contact_step.append(_buf_for_first_contact[0][5])
                self.first_opposed_contact_lateral_mm.append(_buf_for_first_contact[0][1])
                self.first_opposed_contact_tilt_deg.append(_buf_for_first_contact[0][2])
                self._classify_deep_candidate(_buf_for_first_contact)
            else:
                self.n_episodes_no_opposed_contact += 1
            if accepted:
                # Record BEFORE clearing has_candidate/best_diag below, and UNCONDITIONALLY on
                # acceptance (not gated on the emit band) -- this is the "how deep did this episode
                # ever get, whether or not that depth cleared the emit band" side of the harness's
                # gap question; the _emit() call right after may still separately reject-by-band.
                self.accepted_episode_deepest_candidate_depth_mm.append(
                    self.best_diag[i]["depth_mm"] if bool(self.has_candidate[i]) else None
                )
                if bool(self.has_candidate[i]):
                    self._emit(i)
                else:
                    self.rejected_no_candidate_count += 1
            # UNCONDITIONAL on `accepted` (harness mandate, 3rd sweep dimension): every attempted
            # episode's own buffered candidates get swept, not just probe-accepted ones -- see
            # _sweep_episode's own docstring for why require_probe_accepted is now itself a sweep
            # axis rather than a hard precondition on being swept at all.
            self._sweep_episode(i, accepted)
            self.has_candidate[i] = False
            self.best_state.pop(i, None)
            self.best_diag.pop(i, None)
            self.episode_candidate_buffer.pop(i, None)

    def _classify_deep_candidate(self, buf: list[tuple[float, float, float, float, bool, int]]) -> None:
        """WHICH BAND CONDITION IS BINDING (harness ask): take the single deepest buffered step of
        this episode -- no settle or speed restriction, the widest possible anchor, same population
        the depth-vs-step/first-contact stats use -- and, if its depth clears depth_min_mm at all
        (a "deep candidate" in the harness's own words), classify which of the three band
        conditions it fails. Called for EVERY attempted episode with any buffered candidate
        (unconditional on probe acceptance), unlike ``_sweep_episode`` which is intentionally
        accept-only (see that method's docstring / Arm 1's own original design) -- this is pure
        measurement, not banking, so it is not bound by that restriction, and using the same
        denominator as first_opposed_contact_depth_mm is what makes the two numbers comparable."""
        best = max(buf, key=lambda c: c[0])
        depth_mm, lateral_mm, tilt_deg = best[0], best[1], best[2]
        if depth_mm < self.depth_min_mm:
            return  # not a "deep candidate" -- outside what this breakdown is about
        self.deep_candidate_diagnostics.append({"depth_mm": depth_mm, "lateral_mm": lateral_mm, "tilt_deg": tilt_deg})
        depth_ok = depth_mm <= self.depth_max_mm  # lower bound already satisfied by the guard above
        lateral_ok = lateral_mm <= self.lateral_max_mm
        tilt_ok = tilt_deg <= self.tilt_max_deg
        n_fail = int(not depth_ok) + int(not lateral_ok) + int(not tilt_ok)
        if n_fail == 0:
            self.deep_candidate_band_breakdown["pass"] += 1
        elif n_fail >= 2:
            self.deep_candidate_band_breakdown["multiple_fail"] += 1
        elif not depth_ok:
            self.deep_candidate_band_breakdown["depth_only_fail"] += 1
        elif not lateral_ok:
            self.deep_candidate_band_breakdown["lateral_only_fail"] += 1
        else:
            self.deep_candidate_band_breakdown["tilt_only_fail"] += 1

    def _sweep_episode(self, i: int, accepted: bool) -> None:
        """PROBE-ACCEPT x SETTLE x SPEED SWEEP (harness mandate, three catches on the same question,
        2026-08-22): for THIS episode's own buffered candidates (every opposed & not-done step
        seen, settled or not, unfiltered by speed -- see step()'s docstring for why settled must
        NOT be required at collection time), re-run the same argmax-depth-then-emit-band-check the
        real bank does, independently at EACH (require_probe_accepted, settle_required,
        speed_threshold) cell -- answering "what would the emit yield have been under a
        stricter/looser probe-accept requirement, settle gate, and/or speed gate" without a second
        rollout. Analysis-only: does not touch the real bank, ``self.settle_steps``, or
        ``self.max_speed_m_s``. ``(True, True, self.max_speed_m_s)`` (if that value is in the swept
        list) reproduces the real bank's own candidacy exactly, as a sanity cross-check that the
        sweep and the real gate agree.

        CALLED FOR EVERY ATTEMPTED EPISODE, not just accepted ones -- ``accepted`` says whether
        THIS episode qualifies for the ``require_probe_accepted=True`` sub-population at all (an
        episode the probe never validated cannot possibly count toward "yield among accepted
        episodes", so that half of the table is skipped for it); ``require_probe_accepted=False``
        is evaluated regardless, since it is testing the hypothesis that the accept-time
        restriction itself -- not settle, not speed -- is the arm's real bottleneck."""
        buf = self.episode_candidate_buffer.get(i, [])
        for require_probe_accepted in self.probe_accept_sweep_options:
            if require_probe_accepted and not accepted:
                continue  # this episode was never probe-validated, so it cannot count here
            for settle_required in self.settle_sweep_options:
                pool = [c for c in buf if (not settle_required) or c[4]]
                for threshold in self.speed_sweep_thresholds:
                    survivors = [c for c in pool if c[3] <= threshold]
                    stats = self.sweep_stats[(require_probe_accepted, settle_required, threshold)]
                    if not survivors:
                        continue
                    stats["n_would_have_candidate"] += 1
                    best = max(survivors, key=lambda c: c[0])
                    depth_mm, lateral_mm, tilt_deg, _speed, _settled, _step = best
                    in_band = (
                        self.depth_min_mm <= depth_mm <= self.depth_max_mm
                        and lateral_mm <= self.lateral_max_mm
                        and tilt_deg <= self.tilt_max_deg
                    )
                    if in_band:
                        stats["n_would_emit"] += 1
                        stats["would_emit_depths_mm"].append(depth_mm)
                        stats["would_emit_lateral_mm"].append(lateral_mm)
                        stats["would_emit_tilt_deg"].append(tilt_deg)

    def _emit(self, i: int) -> None:
        diag = self.best_diag[i]
        depth_mm, lateral_mm, tilt_deg = diag["depth_mm"], diag["lateral_mm"], diag["tilt_deg"]
        # HARD BANK GATE, enforced not just documented -- same discipline as _C2RewindBank's
        # resting-speed filter: a candidate outside the target band is not "deep enough" no matter
        # how the argmax-over-opposed-contact anchor was chosen.
        in_band = (
            self.depth_min_mm <= depth_mm <= self.depth_max_mm
            and lateral_mm <= self.lateral_max_mm
            and tilt_deg <= self.tilt_max_deg
        )
        if not in_band:
            self.rejected_band_count += 1
            self.rejected_band_diagnostics.append(diag)
            return

        state = self.best_state[i]
        rigid_object = state["rigid_object"]
        keys = set(rigid_object.keys())
        if keys not in _DexliftToTrainingSceneRecorder._KNOWN_SCHEMAS:
            raise ValueError(
                f"_C4DeepestGraspBank expected rigid_object keys in "
                f"{_DexliftToTrainingSceneRecorder._KNOWN_SCHEMAS}, got {sorted(keys)} -- refusing "
                "to silently mis-map an unexpected schema (same guard as _DexliftToTrainingSceneRecorder)."
            )
        rekeyed_rigid = {
            _DexliftToTrainingSceneRecorder._RENAME.get(name, name): tensors
            for name, tensors in rigid_object.items()
            if name not in _DexliftToTrainingSceneRecorder._DROP
        }
        export_state = {"articulation": state["articulation"], "rigid_object": rekeyed_rigid}

        def append_rec(dest: dict, src: dict) -> None:
            for k, v in src.items():
                if isinstance(v, dict):
                    append_rec(dest.setdefault(k, {}), v)
                else:
                    dest.setdefault(k, []).append(v.cpu())

        append_rec(self.accum["initial_state"], export_state)
        self.emitted_count += 1
        self.diagnostics.append(diag)

    def print_progress(self, n_attempts: int | None = None) -> None:
        # HARNESS MANDATE (item c, 2026-08-22): the running four-way counters AND the current
        # percentile/sweep summary in EVERY periodic progress line, not only the final report() --
        # a truncated log (timeout, SIGKILL, crash) must still carry the answer. Reuses the exact
        # same printing code report() uses, just invoked mid-run instead of once at the end.
        print(
            f"[c4-rewind-deepest][progress] emitted={self.emitted_count}  "
            f"rejected_band={self.rejected_band_count}  rejected_no_candidate={self.rejected_no_candidate_count}  "
            f"candidate_updates={self.n_candidate_updates}  speed_gated_steps={self.n_speed_gated_steps}",
            flush=True,
        )
        self._print_diagnostics(n_attempts=n_attempts, header="[c4-rewind-deepest][progress]")

    def write(self) -> None:
        if self.emitted_count == 0:
            print(
                f"[c4-rewind-deepest] 0 episodes captured (rejected_band={self.rejected_band_count}, "
                f"rejected_no_candidate={self.rejected_no_candidate_count}) -- NOT writing {self.output_path}",
                flush=True,
            )
            return
        for asset_name, asset_state in self.accum["initial_state"]["articulation"].items():
            missing = [k for k in ("joint_position_target", "joint_velocity_target") if k not in asset_state]
            assert not missing, (
                f"[c4-rewind-deepest] articulation {asset_name!r} is missing {missing} -- refusing "
                "to write a bank with a zeroed commanded PD squeeze on replay."
            )
        atomic_torch_save(self.accum, self.output_path)
        print(
            f"[c4-rewind-deepest] wrote {self.emitted_count} episodes "
            f"(rejected_band={self.rejected_band_count}, rejected_no_candidate={self.rejected_no_candidate_count}) "
            f"-> {self.output_path}",
            flush=True,
        )

    def _print_diagnostics(self, n_attempts: int | None, header: str) -> None:
        """Shared by ``print_progress`` (mid-run, every periodic tick) and ``report`` (once, at the
        end) -- HARNESS MANDATE: a truncated log must carry the same answer a full one would, so
        this is not report()-only content. Every number here is recomputed from the CURRENT
        accumulators, cheap at this run's scale (torch.quantile over a few thousand floats)."""
        # DEPTH-VS-STEP TRAJECTORY (harness mandate, TOP PRIORITY -- printed FIRST, deliberately,
        # ahead of the speed/sweep material below): median depth at each fixed step bin, across
        # ALL episodes. READ IT THIS WAY: if depth is high (10-17.5mm) at bin 0 and falls toward
        # 2-3mm by bin 45-60 WITHOUT the hand's involvement being required for that fall, the leg is
        # exiting the bore on its own before any grasp -- a spawn/contact-dynamics problem upstream
        # of every arm in this epic (rewind, terminate-on-grasp, goal-below-spawn all assume
        # withdrawal is CAUSED by the grasp). If depth is already ~2-3mm at bin 0, partial_
        # assemblies.pt poses are not being applied as intended -- a different, more serious defect.
        # LATERAL is paired in here too (harness mandate, promoted to CO-EQUAL priority after the
        # band breakdown proved lateral, not depth, is the sole binding constraint): READ THE PAIR
        # THIS WAY -- if lateral is already > ~1mm (the measured bore clearance) at bin 0, the
        # SPAWNED state is already off-axis and this is a spawn-path defect, not a policy one (the
        # partial_assemblies.pt FILE was verified at lateral median 0.035mm, but that verified the
        # STORED poses, never the REALISED state after physics settles at step 0 -- nobody had
        # checked that until this curve). If lateral starts near 0.035mm and grows past ~1mm BEFORE
        # first contact, the leg is being knocked off-axis during the approach, before the hand ever
        # grips it -- a policy behaviour a retrain could target. Depth and lateral may degrade
        # TOGETHER (withdrawn while going off-axis) or SEPARATELY (stays deep but drifts off-axis,
        # or vice versa) -- printed side by side specifically so that distinction is visible.
        print(f"{header} depth+lateral-vs-step trajectory (median mm, across ALL episodes, n per bin):", flush=True)
        for _bin in self.depth_vs_step_bins:
            _dvals = self.depth_vs_step_samples[_bin]
            _lvals = self.lateral_vs_step_samples[_bin]
            if _dvals:
                _dvt = torch.tensor(_dvals)
                _lvt = torch.tensor(_lvals)
                print(
                    f"{header}   step={_bin:3d}: depth median={_dvt.median().item():7.3f} "
                    f"(p25={torch.quantile(_dvt, 0.25).item():7.3f} p75={torch.quantile(_dvt, 0.75).item():7.3f})  "
                    f"lateral median={_lvt.median().item():7.3f} "
                    f"(p25={torch.quantile(_lvt, 0.25).item():7.3f} p75={torch.quantile(_lvt, 0.75).item():7.3f})  "
                    f"n={_dvt.numel()}",
                    flush=True,
                )
            else:
                print(f"{header}   step={_bin:3d}: no samples yet (run hasn't reached this many concurrent episode-steps)", flush=True)

        # THE MEASUREMENT THAT MATTERS (harness ask): settled depth AS A FUNCTION OF SPAWN DEPTH,
        # binned -- not a single aggregate, because "settled depth tracks spawn depth" (a deep bank
        # is reachable, and this says from which spawn band) and "everything converges to one
        # attractor regardless of spawn depth" (a new, better equilibrium, still short of the true
        # gate) are indistinguishable in a mean and imply completely different next steps.
        # SUCCESS_DEPTH_MM_GATE=22.5 is the true task position-success threshold in this same depth
        # convention (pos_err_mm = 25.0 - depth_mm <= 2.5mm requires depth_mm >= 22.5mm).
        if self.spawn_settled_pairs:
            SUCCESS_DEPTH_MM_GATE = 22.5
            _spawn_arr = torch.tensor([p[0] for p in self.spawn_settled_pairs])
            _settled_arr = torch.tensor([p[1] for p in self.spawn_settled_pairs])
            _bin_edges = [12.0, 15.0, 18.0, 21.0, 22.5, 25.0]
            print(f"{header} settled depth vs spawn depth (n={_spawn_arr.numel()} completed episodes):", flush=True)
            for _lo, _hi in zip(_bin_edges[:-1], _bin_edges[1:]):
                _in_bin = (_spawn_arr >= _lo) & (_spawn_arr < _hi if _hi < _bin_edges[-1] else _spawn_arr <= _hi)
                _n_bin = int(_in_bin.sum())
                if _n_bin == 0:
                    print(f"{header}   spawn_depth in [{_lo:5.1f},{_hi:5.1f})mm: n=0 (no spawns in this band yet)", flush=True)
                    continue
                _s = _settled_arr[_in_bin]
                _sp = _spawn_arr[_in_bin]
                _frac_gate = float((_s >= SUCCESS_DEPTH_MM_GATE).float().mean())
                # ATTRACTOR-VS-TRACKING RATIO (harness ask, sharper than a bin median): within THIS
                # bin's own (narrow) spawn range, does settled depth vary PROPORTIONALLY (tracking,
                # ratio near 1) or collapse to one value regardless of where in the bin spawn
                # actually landed (attractor, ratio near 0)? IQR (p75-p25), not min/max, so one
                # outlier spawn/settle pair cannot swing the ratio.
                _settled_iqr = float(torch.quantile(_s, 0.75) - torch.quantile(_s, 0.25))
                _spawn_iqr = float(torch.quantile(_sp, 0.75) - torch.quantile(_sp, 0.25))
                _ratio = _settled_iqr / _spawn_iqr if _spawn_iqr > 1e-6 else float("nan")
                # CATASTROPHIC TAIL (harness ask): ejected/flung episodes are invisible in a median
                # or IQR -- a min/max of -1000+/+300mm next to a clean median is exactly that,
                # hiding in plain sight unless counted directly. Not assumed uniform across bins.
                _n_tail = int(((_s < 0.0) | (_s > 25.0)).sum())
                _frac_tail = _n_tail / _n_bin
                print(
                    f"{header}   spawn_depth in [{_lo:5.1f},{_hi:5.1f})mm: n={_n_bin:4d}  settled_depth "
                    f"median={_s.median().item():7.3f}  p25={torch.quantile(_s,0.25).item():7.3f}  "
                    f"p75={torch.quantile(_s,0.75).item():7.3f}  min={_s.min().item():7.3f}  "
                    f"max={_s.max().item():7.3f}  frac_settled>=22.5mm(true gate)={_frac_gate:.4f}  "
                    f"settled_IQR/spawn_IQR={_ratio:.4f} (spawn_IQR={_spawn_iqr:.3f}mm, near 1=tracking, near 0=attractor)  "
                    f"outside_0_25mm={_n_tail}/{_n_bin}({_frac_tail:.4f})",
                    flush=True,
                )
            _frac_gate_all = float((_settled_arr >= SUCCESS_DEPTH_MM_GATE).float().mean())
            print(
                f"{header}   ALL spawns pooled: fraction settling at or above the true {SUCCESS_DEPTH_MM_GATE}mm "
                f"gate: {_frac_gate_all:.4f} ({int((_settled_arr >= SUCCESS_DEPTH_MM_GATE).sum())}/{_settled_arr.numel()})",
                flush=True,
            )
            _spawn_iqr_all = float(torch.quantile(_spawn_arr, 0.75) - torch.quantile(_spawn_arr, 0.25))
            _settled_iqr_all = float(torch.quantile(_settled_arr, 0.75) - torch.quantile(_settled_arr, 0.25))
            _ratio_all = _settled_iqr_all / _spawn_iqr_all if _spawn_iqr_all > 1e-6 else float("nan")
            print(
                f"{header}   ALL spawns pooled: spawn_depth_IQR={_spawn_iqr_all:.3f}mm  "
                f"settled_depth_IQR={_settled_iqr_all:.3f}mm  ratio={_ratio_all:.4f}  -- the whole-"
                "population version of the same attractor-vs-tracking test, across the full spawn band.",
                flush=True,
            )
            # REGRESSION SLOPE + R^2 (harness ask): the binned table shows the SHAPE, this says how
            # much of settled-depth variance spawn depth explains AT ALL. Slope near 0 / low R^2 =
            # attractor (settled independent of spawn); slope near 1 / high R^2 = tracking. Ordinary
            # least squares, closed form (2 params, no library needed). BASELINE FOR COMPARISON (the
            # old broken-collider asset, same quantities): slope=-0.210, R^2=0.083 -- final depth
            # was essentially INDEPENDENT of spawn depth there, the signature of an attractor.
            _x = _spawn_arr.double()
            _y = _settled_arr.double()
            _x_mean, _y_mean = _x.mean(), _y.mean()
            _sxx = ((_x - _x_mean) ** 2).sum()
            _sxy = ((_x - _x_mean) * (_y - _y_mean)).sum()
            _slope = float(_sxy / _sxx) if _sxx > 1e-12 else float("nan")
            _intercept = float(_y_mean - _slope * _x_mean)
            _y_pred = _slope * _x + _intercept
            _ss_res = float(((_y - _y_pred) ** 2).sum())
            _ss_tot = float(((_y - _y_mean) ** 2).sum())
            _r2 = 1.0 - _ss_res / _ss_tot if _ss_tot > 1e-12 else float("nan")
            print(
                f"{header}   ALL spawns pooled: settled = {_slope:.4f} * spawn + {_intercept:.4f}  "
                f"(R^2={_r2:.4f}, n={_x.numel()})  -- baseline (old broken collider) was "
                "slope=-0.210 R^2=0.083 (attractor). Compare against that, not against 0.",
                flush=True,
            )
            # OUTLIER-DOMINATION CAVEAT ON THE LINE ABOVE (found the hard way, harness ask
            # follow-up): OLS's sum-of-squares is dominated by whatever catastrophic-tail episodes
            # exist (an ejected leg at -1680mm or flung to +379mm swamps hundreds of well-behaved
            # pairs), so the POOLED slope/R^2 above can read near-zero/meaningless even when the
            # non-catastrophic majority tracks cleanly -- print the SAME regression with the tail
            # (settled outside [0,25]mm) excluded, so "does it track" and "how bad is the tail" are
            # two separate numbers instead of one confounding each other.
            _n_tail_all = int(((_settled_arr < 0.0) | (_settled_arr > 25.0)).sum())
            print(
                f"{header}   ALL spawns pooled: outside_0_25mm={_n_tail_all}/{_settled_arr.numel()}"
                f"({_n_tail_all / _settled_arr.numel():.4f}) -- these dominate the pooled regression "
                "above; see the tail-excluded regression below for the real tracking signature.",
                flush=True,
            )
            _clean_mask = (_settled_arr >= 0.0) & (_settled_arr <= 25.0)
            if int(_clean_mask.sum()) >= 2:
                _xc = _spawn_arr[_clean_mask].double()
                _yc = _settled_arr[_clean_mask].double()
                _xc_mean, _yc_mean = _xc.mean(), _yc.mean()
                _sxxc = ((_xc - _xc_mean) ** 2).sum()
                _sxyc = ((_xc - _xc_mean) * (_yc - _yc_mean)).sum()
                _slopec = float(_sxyc / _sxxc) if _sxxc > 1e-12 else float("nan")
                _interceptc = float(_yc_mean - _slopec * _xc_mean)
                _yc_pred = _slopec * _xc + _interceptc
                _ss_resc = float(((_yc - _yc_pred) ** 2).sum())
                _ss_totc = float(((_yc - _yc_mean) ** 2).sum())
                _r2c = 1.0 - _ss_resc / _ss_totc if _ss_totc > 1e-12 else float("nan")
                print(
                    f"{header}   TAIL-EXCLUDED (settled in [0,25]mm only): settled = {_slopec:.4f} * "
                    f"spawn + {_interceptc:.4f}  (R^2={_r2c:.4f}, n={_xc.numel()} of {_settled_arr.numel()})",
                    flush=True,
                )

        print(f"{header} candidate updates observed (new best-so-far, across all envs/episodes): {self.n_candidate_updates}", flush=True)
        print(
            f"{header} speed-gated steps (opposed+settled but object speed > {self.max_speed_m_s}m/s, "
            f"excluded from REAL-BANK candidacy entirely): {self.n_speed_gated_steps}  -- see the sweep "
            "table below for whether this number means transients or a mis-set threshold",
            flush=True,
        )

        # EARN THE THRESHOLD (team-lead ask): the measured distribution of speed across every
        # opposed-and-settled step this run saw, GATED OR NOT -- turns --c4_rewind_max_speed's
        # 0.05 m/s (inherited from C2's untouched-object-on-a-table context) into a measurement
        # against THIS plant's own held-in-hand noise floor, instead of an assumption.
        if self._candidate_samples:
            all_samples = torch.cat(self._candidate_samples, dim=0)  # columns: depth, lateral, tilt, speed
            depth_col, speed_col = all_samples[:, 0], all_samples[:, 3]
            qs = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90])
            p10, p25, p50, p75, p90 = torch.quantile(speed_col, qs).tolist()
            print(
                f"{header} obj_speed distribution over opposed+settled steps (n={speed_col.numel()}, m/s, "
                f"deployed gate={self.max_speed_m_s}): p10={p10:.4f} p25={p25:.4f} p50={p50:.4f} "
                f"p75={p75:.4f} p90={p90:.4f} max={speed_col.max().item():.4f}",
                flush=True,
            )
            # THE SHARP HYPOTHESIS (harness, 2026-08-22): the leg spawns deep (10-17.5mm) and is
            # withdrawn to ~2-4mm, so a deep pose exists in every episode by construction -- the
            # open question is whether it ever coincides with a NON-transient grasp, and the
            # withdrawal window (still deep, still moving, hand closing) is exactly what a fixed
            # speed gate can discard wholesale. Condition the speed distribution on depth already
            # being in/near the target band (>= depth_min_mm) to test this directly.
            _high_depth_mask = depth_col >= self.depth_min_mm
            if _high_depth_mask.any():
                _hd_speed = speed_col[_high_depth_mask]
                hp10, hp25, hp50, hp75, hp90 = torch.quantile(_hd_speed, qs).tolist()
                print(
                    f"{header} obj_speed | depth>={self.depth_min_mm:.1f}mm (n={_hd_speed.numel()}, m/s): "
                    f"p10={hp10:.4f} p25={hp25:.4f} p50={hp50:.4f} p75={hp75:.4f} p90={hp90:.4f} "
                    f"max={_hd_speed.max().item():.4f}  -- if this population sits ABOVE the deployed "
                    "gate, the gate is discarding exactly the states Arm 1 exists to capture.",
                    flush=True,
                )
            else:
                print(f"{header} obj_speed | depth>={self.depth_min_mm:.1f}mm: no samples yet (depth never reached the band floor)", flush=True)
        else:
            print(f"{header} obj_speed distribution: none observed (no opposed contact this run yet)", flush=True)

        # THE WHOLE THESIS OF BOTH REMAINING ARMS (harness ask): depth and step index at the
        # moment opposed contact is FIRST achieved, per ATTEMPTED episode (not just accepted ones).
        # READ IT THIS WAY: if the depth median is ~2-3mm, contact never coincides with a deep pose
        # at all and BOTH Arm 1 and Arm 2 are dead (terminate-on-grasp only changes WHEN the
        # snapshot is taken; it cannot conjure contact that never happened). If it is 8-15mm, a
        # real window exists and the sweep table below is where to find it. The step-index side is
        # the quantitative version of the settled bug just fixed: how much of the deep window the
        # settle_steps={self.settle_steps} requirement was discarding before collection dropped it.
        n_first_contact_episodes = len(self.first_opposed_contact_depth_mm)
        print(
            f"{header} episodes with opposed contact at some point: {n_first_contact_episodes}  "
            f"(never contacted: {self.n_episodes_no_opposed_contact})",
            flush=True,
        )
        if self.first_opposed_contact_depth_mm:
            _fd = torch.tensor(self.first_opposed_contact_depth_mm)
            print(
                f"{header} depth AT FIRST opposed contact (mm, n={_fd.numel()}): min={_fd.min().item():.3f} "
                f"median={_fd.median().item():.3f} max={_fd.max().item():.3f} mean={_fd.mean().item():.3f}",
                flush=True,
            )
            # THE ONE NUMBER THAT DECIDES BOTH ARMS (harness ask, 2026-08-22): min/median/max cannot
            # answer "how often does first contact clear the emit band floor" -- report the fraction
            # directly, plus p75/p90 (the tail is exactly what matters when the question is "does a
            # non-trivial minority reach the band", not "what does the typical episode do"). Reported
            # BOTH raw and with the ejection tail (depth<=0) excluded, since the raw mean sits well
            # below the raw median -- the signature of a left tail (flung legs) dragging it, and the
            # honest read is over the genuinely-engaged population.
            _qs43 = torch.tensor([0.75, 0.90])
            _fd_p75, _fd_p90 = torch.quantile(_fd, _qs43).tolist()
            _frac_ge_floor = float((_fd >= self.depth_min_mm).float().mean())
            print(
                f"{header} fraction of first-contact events >= depth_min_mm({self.depth_min_mm:.1f}mm) "
                f"[RAW, n={_fd.numel()}]: {_frac_ge_floor:.4f}  p75={_fd_p75:.3f}  p90={_fd_p90:.3f}",
                flush=True,
            )
            _fd_engaged = _fd[_fd > 0.0]
            if _fd_engaged.numel() > 0:
                _fd_e_p75, _fd_e_p90 = torch.quantile(_fd_engaged, _qs43).tolist()
                _frac_ge_floor_e = float((_fd_engaged >= self.depth_min_mm).float().mean())
                _n_ejected = _fd.numel() - _fd_engaged.numel()
                print(
                    f"{header} fraction >= depth_min_mm [ENGAGED ONLY, depth>0, n={_fd_engaged.numel()}, "
                    f"excluded as ejection tail: {_n_ejected} ({_n_ejected / _fd.numel():.1%})]: "
                    f"{_frac_ge_floor_e:.4f}  p75={_fd_e_p75:.3f}  p90={_fd_e_p90:.3f}  "
                    f"median={_fd_engaged.median().item():.3f}  mean={_fd_engaged.mean().item():.3f}",
                    flush=True,
                )
            # GEOMETRIC HONESTY (harness ask, 2026-08-22): the bore's radial clearance is ~0.91mm
            # (tightest wall 10.9156mm vs. the leg's 10.004mm flat pilot) -- a leg GENUINELY inside
            # the bore cannot read much above ~1mm lateral. Depth alone cannot tell "inserted" from
            # "near the mouth, off-axis, whose axial projection happens to be large" -- this is the
            # PAIRED condition (depth AND lateral, not depth alone) that is geometrically honest.
            # 1.0mm is a measured physical constant (the clearance), not a tunable design choice --
            # hardcoded here for that reason, same discipline as this file's other measured geometry
            # constants (see e.g. --c4_engaged_span_mm's own help text).
            _GEOM_LATERAL_MM = 1.0
            _fl = torch.tensor(self.first_opposed_contact_lateral_mm)
            _frac_deep_and_inserted = float(((_fd >= self.depth_min_mm) & (_fl <= _GEOM_LATERAL_MM)).float().mean())
            _frac_lateral_alone = float((_fl <= _GEOM_LATERAL_MM).float().mean())
            print(
                f"{header} fraction of first-contact events with depth>={self.depth_min_mm:.1f}mm AND "
                f"lateral<={_GEOM_LATERAL_MM:.1f}mm (geometrically-honest 'deep AND actually inserted', "
                f"n={_fl.numel()}): {_frac_deep_and_inserted:.4f}  (lateral<={_GEOM_LATERAL_MM:.1f}mm alone: "
                f"{_frac_lateral_alone:.4f})  -- if this is far below the depth-alone fraction above, most "
                "'deep' first contacts are axial projections of an off-axis pose, not real insertions.",
                flush=True,
            )
            _fs = torch.tensor(self.first_opposed_contact_step, dtype=torch.float32)
            n_before_settle = int((_fs <= self.settle_steps).sum())
            print(
                f"{header} step index of FIRST opposed contact (n={_fs.numel()}): min={_fs.min().item():.0f} "
                f"median={_fs.median().item():.0f} max={_fs.max().item():.0f} mean={_fs.mean().item():.1f}  "
                f"-- {n_before_settle}/{_fs.numel()} ({n_before_settle / _fs.numel():.1%}) fired AT OR BEFORE "
                f"settle_steps={self.settle_steps} and would have been INVISIBLE to the old (settled-required-"
                "at-collection) design.",
                flush=True,
            )

        # WHICH BAND CONDITION IS BINDING (harness ask): for every ATTEMPTED episode's deepest
        # buffered step (no settle/speed restriction -- the SAME population first_opposed_contact
        # uses, deliberately NOT restricted to probe-accepted episodes, unlike the sweep table
        # below) that clears depth_min_mm at all, which of depth/lateral/tilt is actually failing.
        # READ IT THIS WAY: if lateral_only_fail or tilt_only_fail dominates, the band's OTHER two
        # conditions are what's actually screening out deep candidates, not depth -- worth asking
        # whether those limits are calibrated for this regime. If depth_only_fail dominates (an
        # over-inserted candidate past depth_max_mm) or "pass" is the rare case, depth itself is
        # doing the work and the route lives or dies on depth alone.
        _n_deep_candidates = sum(self.deep_candidate_band_breakdown.values())
        print(
            f"{header} deep-candidate (depth>={self.depth_min_mm:.1f}mm) band breakdown, ALL attempted "
            f"episodes (n={_n_deep_candidates}): {self.deep_candidate_band_breakdown}",
            flush=True,
        )
        if self.deep_candidate_diagnostics:
            for key, label, unit in (
                ("lateral_mm", "deep-candidate lateral miss", "mm"),
                ("tilt_deg", "deep-candidate tilt", "deg"),
            ):
                vals = torch.tensor([r[key] for r in self.deep_candidate_diagnostics])
                print(
                    f"{header}   {label:32s}: min={vals.min().item():8.3f} median={vals.median().item():8.3f} "
                    f"max={vals.max().item():8.3f} mean={vals.mean().item():8.3f}  ({unit}, n={vals.numel()})",
                    flush=True,
                )

        # THE SWEEP (harness mandate, three catches on the same question): what would the emit
        # yield have been at each (require_probe_accepted, settle_required, speed_threshold) cell,
        # analysis-only, from THIS one run -- the real bank above is untouched by this table.
        # require_probe_accepted=True/settle_required=True is the regime the real bank is currently
        # in; lifting either tests whether that restriction (not the other one, not speed) is the
        # arm's real bottleneck. require_probe_accepted=False evaluates EVERY attempted episode's
        # own buffered candidates, regardless of whether the probe later validated that episode --
        # testing whether the accept-time restriction itself (Arm 1's own original design) is
        # costing yield that a mid-episode dynamic-hold test (part B) could validate independently.
        print(f"{header} probe-accept x settle x speed-gate sweep (analysis-only, real bank unaffected):", flush=True)
        for require_probe_accepted in self.probe_accept_sweep_options:
            accept_label = (
                "require_probe_accepted=True (matches real bank population)" if require_probe_accepted
                else "require_probe_accepted=False (evaluated on EVERY attempted episode)"
            )
            print(f"{header} -- {accept_label} --", flush=True)
            for settle_required in self.settle_sweep_options:
                settle_label = "settle_required=True (matches real bank)" if settle_required else "settle_required=False (settle LIFTED)"
                print(f"{header}   -- {settle_label} --", flush=True)
                for threshold in self.speed_sweep_thresholds:
                    stats = self.sweep_stats[(require_probe_accepted, settle_required, threshold)]
                    label = "ungated" if threshold == float("inf") else f"{threshold:.2f}m/s"
                    depths = stats["would_emit_depths_mm"]
                    depth_summary = ""
                    if depths:
                        dvals = torch.tensor(depths)
                        lvals = torch.tensor(stats["would_emit_lateral_mm"])
                        tvals = torch.tensor(stats["would_emit_tilt_deg"])
                        # GEOMETRIC HONESTY (harness ask): the bore's radial clearance is ~0.91mm, so
                        # a genuinely-inserted leg reads well under ~1mm lateral -- print lateral/tilt
                        # for the would-emit population directly, not just depth, on EVERY cell, so
                        # "deep" is never read as "inserted" without checking the geometry permits it.
                        depth_summary = (
                            f"  would_emit_depth_mm(median={dvals.median().item():.2f},max={dvals.max().item():.2f}) "
                            f"lateral_mm(median={lvals.median().item():.2f},max={lvals.max().item():.2f}) "
                            f"tilt_deg(median={tvals.median().item():.2f},max={tvals.max().item():.2f})"
                        )
                    rate = f"{stats['n_would_emit'] / n_attempts:.4f}" if n_attempts else "n/a"
                    print(
                        f"{header}     speed_threshold={label:10s} n_would_have_candidate={stats['n_would_have_candidate']:5d}  "
                        f"n_would_emit={stats['n_would_emit']:5d}  would_emit_per_attempt={rate}{depth_summary}",
                        flush=True,
                    )

        print(f"{header} emitted={self.emitted_count}  rejected_band={self.rejected_band_count}  "
              f"rejected_no_candidate={self.rejected_no_candidate_count}", flush=True)
        if n_attempts:
            print(f"{header} states per attempted episode: {self.emitted_count / n_attempts:.3f}  (n_attempts={n_attempts})", flush=True)

        # HARNESS ASK: a bare rejected_band COUNT cannot distinguish "the anchor is marginal, the
        # band is arguable" from "the anchor is dead, no band choice rescues it" -- print the actual
        # depth/lateral/tilt distribution of every band-rejected candidate. READ IT THIS WAY: if
        # these depths cluster near depth_min_mm (the band floor), the anchor is marginal and a
        # looser band is a live option. If they cluster near the accept-time bank's own depth (~2-3mm
        # for this plant, see the shipped-baseline comparison run separately), the anchor found
        # nothing the accept-time gate did not already see, and no band choice rescues it.
        if self.rejected_band_diagnostics:
            for key, label, unit in (
                ("depth_mm", "rejected-candidate depth", "mm"),
                ("lateral_mm", "rejected-candidate lateral miss", "mm"),
                ("tilt_deg", "rejected-candidate tilt", "deg"),
            ):
                vals = torch.tensor([r[key] for r in self.rejected_band_diagnostics])
                print(
                    f"{header}   {label:32s}: min={vals.min().item():8.3f} median={vals.median().item():8.3f} "
                    f"max={vals.max().item():8.3f} mean={vals.mean().item():8.3f}  ({unit}, n={vals.numel()})",
                    flush=True,
                )
        else:
            print(f"{header}   no band-rejected candidates observed (either none had a candidate, or every candidate cleared the band)", flush=True)

        # THE WHOLE THESIS OF ARM 1, made measurable: the distribution of "deepest opposed-contact
        # depth reached during the episode" across EVERY accepted episode (not just Arm1-emitted
        # ones). Compare this against the accept-time bank's own depth distribution (main() prints
        # the gap directly, joining this list against the recorded bank) -- if this distribution
        # sits close to the accept-time depths, the deepest moment IS where the policy parks and
        # rewinding buys nothing; if it sits well above, a deeper state existed and was missed only
        # by anchoring on accept time.
        _valid_deepest = [d for d in self.accepted_episode_deepest_candidate_depth_mm if d is not None]
        if _valid_deepest:
            _dv = torch.tensor(_valid_deepest)
            print(
                f"{header} deepest-per-accepted-episode depth (mm, n={_dv.numel()} of "
                f"{len(self.accepted_episode_deepest_candidate_depth_mm)} accepted): "
                f"min={_dv.min().item():.3f} median={_dv.median().item():.3f} max={_dv.max().item():.3f} "
                f"mean={_dv.mean().item():.3f}  -- see main()'s own [verify] line for the GAP against "
                "each episode's own accept-time depth.",
                flush=True,
            )
        if self.diagnostics:
            for key, label, unit in (
                ("depth_mm", "banked tip depth", "mm"),
                ("lateral_mm", "banked lateral miss", "mm"),
                ("tilt_deg", "banked tilt", "deg"),
            ):
                vals = torch.tensor([r[key] for r in self.diagnostics])
                print(
                    f"{header}   {label:24s}: min={vals.min().item():8.3f} median={vals.median().item():8.3f} "
                    f"max={vals.max().item():8.3f} mean={vals.mean().item():8.3f}  ({unit})",
                    flush=True,
                )

    def report(self, n_attempts: int | None = None) -> None:
        print("\n=== ARM 1 (C4 REWIND-TO-DEEPEST-GRASP) RESULT (bead UWLab-xp05.1) ===", flush=True)
        self._print_diagnostics(n_attempts=n_attempts, header="")

    def as_json_dict(self, n_attempts: int | None = None) -> dict:
        """Plain-dict snapshot of every current accumulator, JSON-serializable, for the atomic
        incremental dump (harness mandate, item a: a terminal-only report is a single point of
        failure -- every long run today has proven it). Mirrors _print_diagnostics' own numbers so
        the dump and the log can never silently disagree."""
        all_samples = torch.cat(self._candidate_samples, dim=0) if self._candidate_samples else None
        speed_percentiles = None
        speed_percentiles_high_depth = None
        if all_samples is not None:
            depth_col, speed_col = all_samples[:, 0], all_samples[:, 3]
            qs = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90])
            speed_percentiles = dict(zip(("p10", "p25", "p50", "p75", "p90"), torch.quantile(speed_col, qs).tolist()))
            speed_percentiles["max"] = speed_col.max().item()
            speed_percentiles["n"] = speed_col.numel()
            _hd = speed_col[depth_col >= self.depth_min_mm]
            if _hd.numel() > 0:
                speed_percentiles_high_depth = dict(zip(("p10", "p25", "p50", "p75", "p90"), torch.quantile(_hd, qs).tolist()))
                speed_percentiles_high_depth["max"] = _hd.max().item()
                speed_percentiles_high_depth["n"] = _hd.numel()
        return {
            "n_attempts": n_attempts,
            "emitted_count": self.emitted_count,
            "rejected_band_count": self.rejected_band_count,
            "rejected_no_candidate_count": self.rejected_no_candidate_count,
            "n_candidate_updates": self.n_candidate_updates,
            "n_speed_gated_steps": self.n_speed_gated_steps,
            "max_speed_m_s_deployed": self.max_speed_m_s,
            "obj_speed_percentiles_m_s": speed_percentiles,
            "obj_speed_percentiles_m_s_given_depth_ge_band_floor": speed_percentiles_high_depth,
            "sweep": {
                f"require_probe_accepted={require_probe_accepted}|settle_required={settle_required}|"
                f"speed_threshold={'ungated' if T == float('inf') else T}": {
                    "n_would_have_candidate": s["n_would_have_candidate"],
                    "n_would_emit": s["n_would_emit"],
                    "would_emit_depth_mm_median": (
                        torch.tensor(s["would_emit_depths_mm"]).median().item() if s["would_emit_depths_mm"] else None
                    ),
                    "would_emit_lateral_mm_median": (
                        torch.tensor(s["would_emit_lateral_mm"]).median().item() if s["would_emit_lateral_mm"] else None
                    ),
                    "would_emit_lateral_mm": s["would_emit_lateral_mm"],
                    "would_emit_tilt_deg": s["would_emit_tilt_deg"],
                }
                for (require_probe_accepted, settle_required, T), s in self.sweep_stats.items()
            },
            "rejected_band_diagnostics": self.rejected_band_diagnostics,
            "accepted_episode_deepest_candidate_depth_mm": self.accepted_episode_deepest_candidate_depth_mm,
            "emitted_diagnostics": self.diagnostics,
            "n_episodes_no_opposed_contact": self.n_episodes_no_opposed_contact,
            "first_opposed_contact_depth_mm": self.first_opposed_contact_depth_mm,
            "first_opposed_contact_lateral_mm": self.first_opposed_contact_lateral_mm,
            "first_opposed_contact_tilt_deg": self.first_opposed_contact_tilt_deg,
            "first_opposed_contact_step": self.first_opposed_contact_step,
            "depth_vs_step_median_mm": {
                _bin: (torch.tensor(_vals).median().item() if _vals else None)
                for _bin, _vals in self.depth_vs_step_samples.items()
            },
            "depth_vs_step_n": {_bin: len(_vals) for _bin, _vals in self.depth_vs_step_samples.items()},
            "lateral_vs_step_median_mm": {
                _bin: (torch.tensor(_vals).median().item() if _vals else None)
                for _bin, _vals in self.lateral_vs_step_samples.items()
            },
            "lateral_vs_step_n": {_bin: len(_vals) for _bin, _vals in self.lateral_vs_step_samples.items()},
            "spawn_settled_depth_pairs_mm": self.spawn_settled_pairs,
            "deep_candidate_band_breakdown": self.deep_candidate_band_breakdown,
            "deep_candidate_diagnostics": self.deep_candidate_diagnostics,
        }


def main() -> None:
    agent_yaml = args_cli.agent_yaml
    if agent_yaml is None:
        agent_yaml = os.path.join(os.path.dirname(os.path.dirname(args_cli.checkpoint)), "params", "agent.yaml")
    with open(agent_yaml) as f:
        agent_cfg = yaml.safe_load(f)
    print(f"[generator] agent yaml: {agent_yaml}")
    print(f"[generator] normalize_input: {agent_cfg['params']['config'].get('normalize_input')}")
    # PRINT THE VALUE, NOT JUST THE PATH (bead UWLab-weyl, chunking follow-up #3): rl_games'
    # torch_runner.load_config reads agent_cfg["params"]["seed"] and calls torch.manual_seed +
    # np.random.seed with it GLOBALLY, before the env is ever constructed -- identical seeds
    # across chunk processes with an identical command line means identical env sampling, no
    # matter how many different agent_yaml PATHS point at files that turn out to carry the same
    # value. A caller relying on per-chunk seeded yaml copies (make_seeded_agent_yaml.py) should
    # grep THIS line to verify the intended seed actually loaded, rather than trusting the path.
    print(f"[generator] agent_cfg params.seed: {agent_cfg['params'].get('seed')}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # -- GOAL-BELOW-SPAWN SILENT-INERTNESS GUARD (critic3 review, bead UWLab-xp05.3).
    # DEXLIFT_GOAL_BELOW_SPAWN_MM only does anything because
    # dexlift_ur5e_delto_tableleg_env_cfg.py's _apply_partial_assembly_and_goal_toggles reads it --
    # and that function is wired into exactly the RelJointPos-Reorient{,_PLAY} __post_init__
    # methods. It is NOT wired into this script's own --task DEFAULT
    # (...-TableLeg-Lift-Play-v0 -- the Lift classes are bare `pass`) and NOT into the
    # OSC-Reorient sibling either (narrower than "Reorient only"). A run under either of those
    # would silently roll out the UNMODIFIED base command while the env var sits exported and
    # unconsumed -- reading as "the shaping idea does not work" rather than "the toggle was never
    # installed for this task", which is exactly backwards and exactly the failure mode this check
    # exists to convert into a one-line abort, before any Isaac/GPU time is spent. Checked against
    # the CONSTRUCTED cfg object itself (isinstance), not a string match on the task name, so it
    # also catches any future task variant that happens not to wire the toggle.
    _goal_below_spawn_mm_requested = float(os.environ.get("DEXLIFT_GOAL_BELOW_SPAWN_MM", "0") or "0")
    # `!= 0.0`, NOT `> 0.0` (bead UWLab-nnlv.3). The value used to be unsigned, so `> 0` and
    # "requested at all" were the same test; it is now SIGNED, and a NEGATIVE value -- the one the
    # S2' rung needs, placing the goal above the mouth -- is exactly the case that most needs this
    # check. Left at `> 0.0` this block would have skipped silently for every negative delta, which
    # is the precise failure its own comment above describes: the env var no-ops, the run completes,
    # and the result reads as "the shaping idea does not work" rather than "the toggle was never
    # installed". The sign of the request is not the subject of this check -- whether the command
    # was WIRED is.
    if _goal_below_spawn_mm_requested != 0.0:
        _object_pose_cfg = env_cfg.commands.object_pose
        if not isinstance(_object_pose_cfg, dexlift_mdp.GoalBelowSpawnPoseCommandCfg):
            raise SystemExit(
                f"REFUSING: DEXLIFT_GOAL_BELOW_SPAWN_MM={_goal_below_spawn_mm_requested} is "
                f"exported but --task {args_cli.task!r} constructed commands.object_pose as "
                f"{type(_object_pose_cfg).__name__}, not GoalBelowSpawnPoseCommandCfg -- this task "
                "variant never calls _apply_partial_assembly_and_goal_toggles (see "
                "dexlift_ur5e_delto_tableleg_env_cfg.py), so the env var would silently no-op. Use "
                "--task DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-Play-v0 (or the matching "
                "non-Play id) instead."
            )
        print(
            f"[generator] GOAL-BELOW-SPAWN WIRING CONFIRMED: --task {args_cli.task!r} constructed "
            f"commands.object_pose as {type(_object_pose_cfg).__name__} with delta_m="
            f"{_object_pose_cfg.delta_m:.4f} ({_goal_below_spawn_mm_requested:.2f}mm requested).",
            flush=True,
        )

    # Shrink the GPU collision stack for a small-env smoke/dev scene (the training default,
    # dexlift_ur5e_delto_env_cfg.py's own __post_init__, is 4026531840 bytes / 3.75 GiB -- sized for
    # thousands of parallel envs -- and parse_env_cfg above has already applied it to THIS env_cfg
    # before this line runs).
    #
    # NOT A SINGLE CONSTANT FOR EVERY --reset_type (team-lead catch, live-log comparison, 2026-08-21):
    # gen_c4_probe.log (ObjectPartiallyAssembledEEGrasped, 128 envs) hit 11 distinct
    # "PxGpuDynamicsMemoryConfig::collisionStackSize buffer overflow ... Contacts have been dropped"
    # errors, peak demand ~23.5 MB, against the OLD blanket 2**24 (16.0 MB) this line used to set
    # unconditionally -- i.e. the 100-state probe that this project's C4 seating-gate/retrain
    # decision rests on was measured with contacts silently dropped for at least part of the run.
    # gen_c2rewind_chunk1.log (ObjectAnywhereEEGrasped/Near via --c2_rewind, the SAME generator, the
    # SAME 128 envs) hit ZERO overflow events at that identical 16.0 MB budget -- so this is not a
    # num_envs effect (c4_train.log at 4096 envs is also clean, under the task's own 3.75 GiB) and
    # not a generator-wide undersizing, it is SPECIFIC to the C4 (ObjectPartiallyAssembledEEGrasped)
    # scene: the fixture is a live collider AND the leg spawns already inside its bore, so
    # leg-vs-bore contact volume exists from step 0 in a way no other --reset_type here produces.
    # Blanket-raising this for every reset type would be paying VRAM for a demand only C4 has ever
    # shown; narrowing the raise to C4 keeps every other --reset_type's resource footprint
    # byte-identical to before this fix.
    #
    # VALUE: 256 * 1024 * 1024 (256 MiB) -- not a fresh guess. It is the SAME constant
    # render_partial_assemblies.py already sets for this EXACT scene (OneLegInsertionFixture /
    # partial-assembly variants, --num_envs 1) -- reusing an already-battle-tested value for this
    # scene rather than inventing a new one. ~10.7x margin over the ~23.9 MB peak observed in the
    # LATEST (--c4_seating_gate-enabled) crash log, gen_c4prod_chunk1.log, at the same 128 envs.
    #
    # THE UINT32 CEILING (why this can't just always be the training value): USD's
    # physxScene:gpuCollisionStackSize is an unsigned int -- exactly 2**32 (4294967296) fails at
    # gym.make with a Tf.ErrorException type mismatch before any physics runs, so 4026531840
    # (3.75 GiB, the training value) is already the PRACTICAL ceiling for ANY --reset_type, not
    # merely a training-scale choice; see dexlift_ur5e_delto_env_cfg.py's own comment on this same
    # constant for the measurement this was originally sized from.
    _C4_RESET_TYPE = "ObjectPartiallyAssembledEEGrasped"
    if args_cli.reset_type == _C4_RESET_TYPE:
        env_cfg.sim.physx.gpu_collision_stack_size = 256 * 1024 * 1024
    else:
        env_cfg.sim.physx.gpu_collision_stack_size = 2**24
    print(
        f"[verify] gpu_collision_stack_size = {env_cfg.sim.physx.gpu_collision_stack_size} bytes "
        f"({env_cfg.sim.physx.gpu_collision_stack_size / (1024 * 1024):.1f} MiB) "
        f"-- {'C4 (partial-assembly scene) budget' if args_cli.reset_type == _C4_RESET_TYPE else 'default small-env budget'}, "
        f"reset_type={args_cli.reset_type!r}",
        flush=True,
    )

    # -- EPISODE LENGTH OVERRIDE, THIS run's env_cfg only, applied before gym.make so the max-step
    # count derived from it (episode_length_s / (decimation * sim.dt)) is recomputed at env
    # construction. Default None -> untouched, keeping the 47.22%-class baseline reproducible from
    # the same command line. Tests the probe_ready budget-exhaustion hypothesis: 238/238 rejections
    # were time_out at exactly step 240 (the registered 4.0s episode), zero variance -- see the
    # probe_ready diagnostic run this session.
    if args_cli.episode_length_s is not None:
        original_episode_length_s = env_cfg.episode_length_s
        env_cfg.episode_length_s = args_cli.episode_length_s
        print(
            f"[generator] episode_length_s OVERRIDE (this env_cfg copy only): "
            f"{original_episode_length_s} -> {env_cfg.episode_length_s}",
            flush=True,
        )
        # -- RE-DERIVE resampling_time_range if it goes stale (critic3 review, bead UWLab-xp05.3).
        # _apply_partial_assembly_and_goal_toggles forces commands.object_pose.resampling_time_range
        # strictly past episode_length_s at __post_init__ time -- i.e. BEFORE this override runs.
        # Without this block, --episode_length_s 8.0 on top of the toggle's 4.0s-based
        # resampling_time_range=(5.0,6.0) would silently reopen the exact mid-episode-resample bug
        # that fix exists to close (CommandManager.compute() resamples whenever time_left<=0,
        # independent of episode reset -- see GoalBelowSpawnPoseCommand's own construction-time
        # assert, which would now catch this at gym.make() rather than let it run silently, but
        # aborting a whole boot is worse than just keeping the invariant true here).
        if isinstance(env_cfg.commands.object_pose, dexlift_mdp.GoalBelowSpawnPoseCommandCfg):
            _min_resample_s = env_cfg.episode_length_s + 1.0
            env_cfg.commands.object_pose.resampling_time_range = (_min_resample_s, _min_resample_s + 1.0)
            print(
                f"[generator] GoalBelowSpawnPoseCommand resampling_time_range RE-DERIVED after "
                f"episode_length_s override: {env_cfg.commands.object_pose.resampling_time_range} "
                f"(episode_length_s={env_cfg.episode_length_s}s)",
                flush=True,
            )

    # -- PLANT/RESET VERIFICATION (diagnostic only, no config changes) -- reads the ALREADY-
    # CONSTRUCTED env_cfg's raw resolved values directly, rather than trusting the source's own
    # "reference"/"identified" text banner. A silently-unset DEXLIFT_REF_* env var gives a
    # plausible wrong number; this prints the five values needed to catch that before any run.
    _hand_act = env_cfg.scene.robot.actuators["hand"]
    _effort_vals = sorted(set(_hand_act.effort_limit_sim.values())) if isinstance(
        _hand_act.effort_limit_sim, dict
    ) else [_hand_act.effort_limit_sim]
    _vel_vals = sorted(set(_hand_act.velocity_limit_sim.values())) if isinstance(
        _hand_act.velocity_limit_sim, dict
    ) else [_hand_act.velocity_limit_sim]
    print(
        f"[verify] hand effort_limit_sim (distinct values): {_effort_vals}  "
        f"(expect [30.0] for reference, [0.06, 0.13, 0.14, 0.17] for identified)",
        flush=True,
    )
    print(
        f"[verify] hand velocity_limit_sim (distinct values): {_vel_vals}  "
        f"(expect [10000.0] for reference, [3.0] for identified)",
        flush=True,
    )
    print(
        f"[verify] events.reset_robot_joints.position_range = "
        f"{env_cfg.events.reset_robot_joints.params['position_range']}  (expect [-0.5, 0.5])",
        flush=True,
    )
    print(
        f"[verify] events.reset_finger_root_joints.position_range = "
        f"{env_cfg.events.reset_finger_root_joints.params['position_range']}  (expect [0.0, 0.0])",
        flush=True,
    )
    print(
        f"[verify] events.reset_robot_elbow_joint.position_range = "
        f"{env_cfg.events.reset_robot_elbow_joint.params['position_range']}  (expect [-0.2, 0.2])",
        flush=True,
    )
    # -- SIXTH verify line, same reasoning as the five above: which spawn term actually got built,
    # read off the CONSTRUCTED env_cfg rather than the DEXLIFT_SPAWN_CLEARANCE env var directly, so
    # a launcher that forgot to export it (or a toggle that silently failed to apply) is visible
    # here instead of producing a plausible-looking wrong reset-state distribution.
    #
    # THREE EXPLICIT CASES PLUS A SAFE FALLBACK, not a clearance-term/else split. The else-branch
    # used to assume "not the clearance term" meant reset_root_state_uniform, indexing
    # params['pose_range'] unconditionally -- true right up until DEXLIFT_PARTIAL_ASSEMBLY started
    # correctly routing reset_object to SpawnPartialAssembly (bead UWLab-qiao.1 5090-migration
    # follow-on, the Play-class fix), whose params have no 'pose_range' key at all. KeyError, on a
    # line whose whole job is to report what was built without crashing. The fallback below is the
    # fix that generalizes: ANY term this file doesn't know about prints its own name and params
    # keys rather than guessing a schema -- and deliberately does NOT use params.get() to paper over
    # an unknown field, because a verify line that prints None on a mismatch LOOKS like a passing
    # check instead of the "I don't recognise this" it should be.
    _reset_object_func = env_cfg.events.reset_object.func
    _reset_object_params = env_cfg.events.reset_object.params
    # Computed ONCE here, not re-tested inline in the branch below and not re-tested again at the
    # clearance hard guard further down -- both consume THIS variable. Recomputing it in more than
    # one place is exactly the coupling that let this refactor silently drop it the first time
    # (the guard kept referencing a name the branch rewrite no longer defined -- NameError, not a
    # False, so the crash was loud, but the discipline going forward is one source of truth).
    _reset_object_is_clearance_term = _reset_object_func is dexlift_mdp.reset_object_pose_with_clearance
    # Same "computed ONCE" discipline as the clearance boolean above -- both the print branch right
    # below and the --reset_type hard guard further down consume THIS variable rather than each
    # re-testing `_reset_object_func is dexlift_mdp.SpawnPartialAssembly` inline.
    _reset_object_is_partial_assembly_term = _reset_object_func is dexlift_mdp.SpawnPartialAssembly
    if _reset_object_is_clearance_term:
        print(
            f"[verify] events.reset_object.func = {_reset_object_func.__name__}  "
            f"clearance_range={_reset_object_params['clearance_range']} "
            f"half_extents={_reset_object_params['half_extents']} "
            f"surface_z={_reset_object_params['surface_z']}  (DEXLIFT_SPAWN_CLEARANCE=1)",
            flush=True,
        )
    elif _reset_object_is_partial_assembly_term:
        print(
            f"[verify] events.reset_object.func = {_reset_object_func.__name__}  "
            f"fixture_pose_range={_reset_object_params['fixture_pose_range']} "
            f"dataset_dir={_reset_object_params['dataset_dir']}  (DEXLIFT_PARTIAL_ASSEMBLY=1)",
            flush=True,
        )
    elif _reset_object_func.__name__ == "reset_root_state_uniform" and "pose_range" in _reset_object_params:
        print(
            f"[verify] events.reset_object.func = {_reset_object_func.__name__}  "
            f"pose_range.z = {_reset_object_params['pose_range'].get('z')}  "
            f"(DEXLIFT_SPAWN_CLEARANCE unset/not '1')",
            flush=True,
        )
    else:
        print(
            f"[verify] events.reset_object.func = {_reset_object_func.__name__}  "
            f"UNRECOGNISED TERM -- params keys: {sorted(_reset_object_params.keys())}",
            flush=True,
        )

    # -- SEVENTH verify line (bead UWLab-qiao.9): whether the pose GOAL is uniform-sampled or pinned
    # to the object's own spawn pose, read off the CONSTRUCTED command term's class -- not off
    # DEXLIFT_GOAL_AT_SPAWN / DEXLIFT_PARTIAL_ASSEMBLY directly -- same reasoning as the sixth line
    # above: C3's measurement showed the accepted-height floor tracks the GOAL range, not the spawn
    # distribution, so a silently-unset toggle here produces a plausible-looking but WRONG height
    # distribution rather than an error -- exactly the failure mode the sixth line already guards for
    # spawn, now covered for goal.
    # GoalBelowSpawnPoseCommand added to the accepted set (bead UWLab-xp05.3): a SECOND legitimate
    # "goal was not left at the uniform default" outcome, alongside GoalAtSpawnPoseCommand -- built
    # instead of it when DEXLIFT_GOAL_BELOW_SPAWN_MM is requested (which itself requires
    # DEXLIFT_PARTIAL_ASSEMBLY=1, so `_requested_goal_at_spawn` below is already correctly True in
    # that case; only the "what was actually built" side needed widening). Before this, the hard
    # guard just below correctly refused every DEXLIFT_GOAL_BELOW_SPAWN_MM run -- it did its job,
    # catching a real gap: this guard predates that command and had no way to know it existed.
    _object_pose_class = env_cfg.commands.object_pose.class_type
    _goal_is_pinned = _object_pose_class is dexlift_mdp.GoalAtSpawnPoseCommand
    _goal_is_below_spawn = _object_pose_class is dexlift_mdp.GoalBelowSpawnPoseCommand
    if _goal_is_pinned:
        print(
            f"[verify] commands.object_pose.class_type = {_object_pose_class.__name__}  "
            f"goal = PINNED to object spawn pose (DEXLIFT_GOAL_AT_SPAWN=1 or DEXLIFT_PARTIAL_ASSEMBLY=1)",
            flush=True,
        )
    elif _goal_is_below_spawn:
        print(
            f"[verify] commands.object_pose.class_type = {_object_pose_class.__name__}  "
            f"goal = spawn pose displaced {abs(env_cfg.commands.object_pose.delta_m) * 1000.0:.2f}mm "
            # The delta is SIGNED (bead UWLab-nnlv.3). Printing "deeper" unconditionally would state
            # the OPPOSITE of what a negative delta does, in the one line an operator reads to
            # confirm the shaping went in the direction they asked for -- which is the whole job of
            # this verification block.
            f"{'DEEPER INTO the bore' if env_cfg.commands.object_pose.delta_m > 0.0 else 'OUT OF the mouth (ABOVE it)'} "
            "along the bore's own axis (DEXLIFT_GOAL_BELOW_SPAWN_MM, bead UWLab-xp05.3 -- SHAPING "
            "DEVICE, not a target)",
            flush=True,
        )
    else:
        _pos_z_range = env_cfg.commands.object_pose.ranges.pos_z
        print(
            f"[verify] commands.object_pose.class_type = {_object_pose_class.__name__}  "
            f"goal = UNIFORM-SAMPLED, ranges.pos_z = {_pos_z_range}  "
            f"(DEXLIFT_GOAL_AT_SPAWN and DEXLIFT_PARTIAL_ASSEMBLY both unset/not '1')",
            flush=True,
        )

    # -- HARD GUARD, not a print. A silently-unset (or silently-ineffective) DEXLIFT_SPAWN_CLEARANCE
    # yields a plausible WRONG spawn distribution rather than an error -- the exact failure mode
    # that turned a 46.71% acceptance run into a 2.69% one when DEXLIFT_REF_* went unexported (see
    # this file's plant-verification block above). Fail loudly at startup instead of silently
    # generating reset states under the wrong distribution.
    _requested_clearance = os.environ.get("DEXLIFT_SPAWN_CLEARANCE") == "1"
    _require(
        "DEXLIFT_SPAWN_CLEARANCE",
        requested=_requested_clearance,
        actual=_reset_object_is_clearance_term,
        message=(
            f"events.reset_object.func is {_reset_object_func.__name__!r}, not "
            "reset_object_pose_with_clearance."
        ),
    )

    # -- HARD GUARD, not a print (bead UWLab-qiao.9/H). Same shape as the DEXLIFT_SPAWN_CLEARANCE
    # guard immediately above, for the same reason: a silently-unset (or silently-ineffective)
    # DEXLIFT_GOAL_AT_SPAWN / DEXLIFT_PARTIAL_ASSEMBLY yields a plausible WRONG goal distribution --
    # a full generation run of high-altitude grasps indistinguishable from "the mechanism does not
    # work" -- rather than an error. "What was requested" is read from os.environ (mirroring the
    # __post_init__ implication: DEXLIFT_PARTIAL_ASSEMBLY=1 implies goal-at-spawn without needing
    # DEXLIFT_GOAL_AT_SPAWN exported too); "what was built" is read from the CONSTRUCTED cfg, never
    # os.environ, same as the clearance guard.
    _requested_goal_at_spawn = (
        os.environ.get("DEXLIFT_GOAL_AT_SPAWN") == "1" or os.environ.get("DEXLIFT_PARTIAL_ASSEMBLY") == "1"
    )
    # `actual` WIDENED to accept EITHER legitimate class (bead UWLab-xp05.3) -- `requested` is
    # UNCHANGED, and the refusal path is UNCHANGED: if NEITHER class was built while either env var
    # was requested, this still raises exactly as before. Widening only the accepted set, never the
    # refusal, is deliberate -- see this guard's own comment above for why silently letting the
    # empty case through would be a worse bug than the one this fixes.
    _require(
        "DEXLIFT_GOAL_AT_SPAWN/DEXLIFT_PARTIAL_ASSEMBLY",
        requested=_requested_goal_at_spawn,
        actual=_goal_is_pinned or _goal_is_below_spawn,
        message=(
            f"commands.object_pose.class_type is {_object_pose_class.__name__!r}, not "
            "GoalAtSpawnPoseCommand or GoalBelowSpawnPoseCommand."
        ),
    )

    # -- HARD GUARD, not a print (bead UWLab-algw.9). Same shape as the two guards immediately
    # above, for the same reason -- but this time on --reset_type ITSELF, not an env var. Before
    # this guard, --reset_type was consumed at exactly ONE place (env_cfg.recorders.dataset_filename
    # below): it is a FILENAME and nothing else. That is how a Resting run (DEXLIFT_PARTIAL_ASSEMBLY
    # left unset, so events.reset_object.func built as the ordinary resting spawn term) shipped on
    # disk as "resets_ObjectPartiallyAssembledEEGrasped.pt" -- a plausible-looking file containing
    # the wrong distribution, not an error. `actual` requires BOTH the env var and the constructed
    # cfg to agree, per the bead: an env var is a request, the constructed cfg is the fact, and this
    # guard must not trust either one alone.
    _reset_type_requests_partial_assembly = args_cli.reset_type == "ObjectPartiallyAssembledEEGrasped"
    _dexlift_partial_env_set = os.environ.get("DEXLIFT_PARTIAL_ASSEMBLY") == "1"
    _built_partial_assembly = _dexlift_partial_env_set and _reset_object_is_partial_assembly_term
    # -- C3(S_t) CARVE-OUT (bead dr-sj6.22, team-lead's --reset_type decision, 2026-08-29). Team-lead:
    # --reset_type is a NAMING/output-path selector, not a machinery selector (its own help string:
    # "Reset type name for the output path"; v1 precedent, launch_dexreset_s1_s2_bank_gen.sh:27,
    # generates BOTH its rungs under one --reset_type and renames the bank downstream). DEXRESET_C3_RUNG=1
    # draws S1 vs S_t per episode from ONE scene, and C3RungGoalPoseCommand requires 'receptive_object'
    # in the scene for BOTH branches -- it parks the fixture on every S_t env too (c3_rung.py's own
    # module docstring: "The fixture is parked on every S_t env on EVERY reset"). So a CORRECT C3(S_t)
    # run legitimately sets DEXLIFT_PARTIAL_ASSEMBLY=1 (receptive_object built) while --reset_type is
    # DELIBERATELY ObjectRestingEEGrasped, not ObjectPartiallyAssembledEEGrasped: S_t must not share
    # --c4_seating_gate's reset_type, or MultiResetManager (which matches banks by reset_types entries,
    # not filenames) could silently consume S_t's bank as an S1/C4 one -- the worst failure shape
    # available, per team-lead. Without this carve-out the bidirectional guard below would reject every
    # correct C3(S_t) invocation as the exact naming mismatch it exists to catch, which it is not.
    _requested_partial_assembly_scene = _reset_type_requests_partial_assembly or args_cli.c3_st_spawn_tolerance
    _require(
        "--reset_type=ObjectPartiallyAssembledEEGrasped (or --c3_st_spawn_tolerance, which requests the "
        "same receptive_object scene under a different --reset_type)",
        requested=_requested_partial_assembly_scene,
        actual=_built_partial_assembly,
        message=(
            f"requires DEXLIFT_PARTIAL_ASSEMBLY=1 exported AND events.reset_object.func built as "
            f"SpawnPartialAssembly; got DEXLIFT_PARTIAL_ASSEMBLY={os.environ.get('DEXLIFT_PARTIAL_ASSEMBLY')!r} "
            f"and events.reset_object.func={_reset_object_func.__name__!r}. (--c3_st_spawn_tolerance="
            f"{args_cli.c3_st_spawn_tolerance} also requests this scene, independent of --reset_type; "
            "see the carve-out comment above this guard.)"
        ),
    )

    # -- (2) THE HELD-CHECK, wired as terminations.success. Set as a plain instance attribute on
    # the ALREADY-FULLY-CONSTRUCTED env_cfg (parse_env_cfg has already run every __post_init__,
    # including whatever the dexlift table-leg mixin tuned on top of dexsuite's generic
    # object_out_of_bound / abnormal_robot bounds) -- replacing env_cfg.terminations wholesale with
    # a freshly constructed class would silently discard those tunings and was tried first; it
    # produced spurious immediate terminations on every step. TerminationManager discovers terms via
    # `self.cfg.__dict__.items()` (termination_manager.py:257), not dataclass field introspection,
    # so a dynamically added instance attribute is picked up identically to a declared field.
    # -- DELIVERABLE 1: --c4_seating_gate swaps in a Seated* variant (base held-check AND the
    # spatial seating decomposition) instead of the plain term; params={} (byte-identical to
    # before this flag existed) whenever it is absent. See that flag's own help text / the class
    # docstrings for the full argument.
    # -- ARM 2 (bead UWLab-xp05.2): --c4_terminate_on_grasp swaps the BASE held-check itself from
    # held_with_probe to TerminateOnGraspSuccess. The two flags are ORTHOGONAL: --c4_seating_gate
    # composes with EITHER base (SeatedHeldWithProbe wraps held_with_probe, SeatedTerminateOnGrasp
    # wraps TerminateOnGraspSuccess) -- see _SeatingGateAddon's docstring for how that composition
    # is shared without multiple inheritance.
    if args_cli.c4_seating_gate or args_cli.c4_terminate_on_grasp or args_cli.c4_rewind_deepest:
        assert args_cli.reset_type == "ObjectPartiallyAssembledEEGrasped", (
            "--c4_seating_gate/--c4_terminate_on_grasp/--c4_rewind_deepest only make sense for "
            "--reset_type ObjectPartiallyAssembledEEGrasped (all three need a receptive_object/"
            f"fixture already in the scene); got --reset_type={args_cli.reset_type!r}. Refusing to "
            "construct a gate with nothing to measure."
        )
    assert not (args_cli.c4_rewind_deepest and args_cli.c4_terminate_on_grasp), (
        "--c4_rewind_deepest and --c4_terminate_on_grasp are mutually exclusive (see "
        "--c4_rewind_deepest's own help text): Arm 1 explicitly anchors on the episode outcome of "
        "the FULL settle+probe held-check, while --c4_terminate_on_grasp replaces that check with "
        "Arm 2's own fast one -- running Arm 1 against Arm 2's acceptance decision would not be "
        "measuring what Arm 1's own design (rewinding to before a genuinely PROBE-validated "
        "acceptance) is supposed to measure."
    )
    # -- C3(S_t) SPAWN-TOLERANCE GATE (bead dr-sj6.22). Same "assert before Isaac starts" discipline
    # as the C4 flags immediately above, not folded into their assert: S_t is a DIFFERENT rung with
    # a different success_func family, and conflating the messages would blur which gate a given
    # failure is about.
    if args_cli.c3_st_spawn_tolerance:
        assert not (args_cli.c4_seating_gate or args_cli.c4_terminate_on_grasp or args_cli.c4_rewind_deepest), (
            "--c3_st_spawn_tolerance is mutually exclusive with --c4_seating_gate/"
            "--c4_terminate_on_grasp/--c4_rewind_deepest -- different success_func families, and "
            "S_t is never seating-gated (_SpawnPoseToleranceAddon's own docstring: S_t has no "
            "mating frame, _SeatingGateAddon would reject ~100% of valid S_t states)."
        )
        assert args_cli.reset_type == "ObjectRestingEEGrasped", (
            "--c3_st_spawn_tolerance requires --reset_type ObjectRestingEEGrasped (team-lead decision, "
            "2026-08-29): S_t IS 'object resting on the table, end-effector grasped' -- the reset_type "
            "name literally describes the rung -- and it must NOT be ObjectPartiallyAssembledEEGrasped, "
            "the reset_type --c4_seating_gate couples to, because S_t is never seating-gated (see the "
            "mutual-exclusion assert just above) and its bank must not collide with an S1/C4 bank under "
            "MultiResetManager's reset_types matching. The receptive_object/fixture C3RungGoalPoseCommand "
            "needs is governed by DEXLIFT_PARTIAL_ASSEMBLY=1 SEPARATELY from --reset_type (see the carve-"
            "out on the bidirectional partial-assembly guard above in main()) -- --reset_type here is "
            f"purely the output-path/bank-identity name. Got --reset_type={args_cli.reset_type!r}."
        )
        assert os.environ.get("DEXRESET_C3_RUNG") == "1", (
            "--c3_st_spawn_tolerance requires DEXRESET_C3_RUNG=1 already staged in the environment "
            "(this script does not set it -- same convention as DEXLIFT_PARTIAL_ASSEMBLY for the C4 "
            f"flags). Got DEXRESET_C3_RUNG={os.environ.get('DEXRESET_C3_RUNG')!r}. Without it, "
            "env_cfg.commands.object_pose is never upgraded to C3RungGoalPoseCommand and this gate "
            "would construct against the wrong command term (or crash trying)."
        )
        # EXISTENCE check only, at the CLI boundary, for a fast/clear failure before Isaac starts --
        # the RANGE check (must be > 0) is SpawnToleranceConfig's own job (spawn_tolerance_core.py),
        # not restated here, so there is exactly one place that decides what a valid tolerance is.
        assert args_cli.c3_st_pos_tol_mm is not None, (
            "--c3_st_spawn_tolerance requires --c3_st_pos_tol_mm explicitly -- there is no default "
            "(V2_ACCEPTANCE_CRITERIA.md sec 4 / bead dr-sj6.24: this number is OPEN, meant to be "
            "DERIVED from this flag's own R4 validation run, not guessed). Refusing to start rather "
            "than falling back to a plausible-looking value."
        )
        assert args_cli.c3_st_rot_tol_deg is None or args_cli.c3_st_rot_tol_deg > 0.0, (
            f"--c3_st_rot_tol_deg must be > 0 or omitted (omitted disables the rotation gate; 0 "
            f"would silently mean the same thing, which is not what an explicit 0 should mean); got "
            f"{args_cli.c3_st_rot_tol_deg}."
        )
    if args_cli.c4_terminate_on_grasp:
        success_func = SeatedTerminateOnGrasp if args_cli.c4_seating_gate else TerminateOnGraspSuccess
    elif args_cli.c3_st_spawn_tolerance:
        success_func = SpawnToleranceHeldWithProbe
    else:
        success_func = SeatedHeldWithProbe if args_cli.c4_seating_gate else dexlift_mdp.held_with_probe
    # -- params STAYS EMPTY for every one of the four combinations (team-lead catch, third attempt,
    # the real fix). IsaacLab's manager construction (manager_base.py's _resolve_common_term_cfg)
    # inspects term_cfg.func's __call__ SIGNATURE and requires it to match term_cfg.params' KEYS
    # exactly -- it does not understand **kwargs (a parameter literally named "kwargs" is treated
    # as one more mandatory param the cfg failed to supply) and TerminationManager.compute()
    # re-passes the SAME params dict as **kwargs on every single step, not only at construction
    # (termination_manager.py:168). held_with_probe works ONLY because it is always constructed
    # with params={} -- __call__(self, env) then has ZERO extra parameters, matching params={}
    # exactly. Every class above keeps that SAME (self, env) shape; all per-run configuration is
    # instead threaded through plain attributes on env_cfg, set BELOW, BEFORE gym.make() -- the
    # SAME mechanism this codebase's own MixtureResetObject already uses for classic_goal_prob/
    # low_goal_prob/partial_assembly_prob (dexlift/mdp/episode_mixture.py), for an unrelated reason
    # (Hydra-override timing) but the identical structural fix. This removes the whole class of
    # failure rather than keeping a dict and a __call__ signature in sync by hand.
    success_params = {}
    if args_cli.c4_seating_gate:
        env_cfg.c4_seating_gate_config = {
            "leg_usd_path": env_cfg.scene.object.spawn.usd_path,
            "fixture_usd_path": args_cli.receptive_usd_path,
            "c4_engaged_span_mm": args_cli.c4_engaged_span_mm,
            "c4_depth_min_mm": args_cli.c4_depth_min_mm,
            "c4_depth_max_mm": args_cli.c4_depth_max_mm,
            "c4_lateral_max_mm": args_cli.c4_lateral_max_mm,
            "c4_tilt_max_deg": args_cli.c4_tilt_max_deg,
        }
    if args_cli.c4_terminate_on_grasp:
        env_cfg.c4_terminate_on_grasp_config = {
            "consecutive_steps_required": args_cli.c4_terminate_consecutive_steps,
            "obj_speed_thresh": args_cli.c4_terminate_obj_speed_thresh,
            "force_threshold": args_cli.c4_terminate_force_threshold,
            "settle_steps": args_cli.c4_terminate_settle_steps,
        }
    if args_cli.c3_st_spawn_tolerance:
        # command_name is the only key the CLI can override -- an absent value means
        # _SpawnPoseToleranceAddon's own signature default (dexlift_mdp.GOAL_COMMAND_NAME). No
        # settle_* keys any more (bead dr-ai1.18: the addon reads goal_is_final off the command term
        # directly instead of recomputing a settle predicate, so there is nothing left to override).
        # pos_tol_m / rot_tol_rad are the two exceptions -- pos_tol_m is asserted present above;
        # rot_tol_deg -> rad conversion happens here, once, rather than inside the addon, so the
        # addon's own unit is always radians regardless of caller.
        env_cfg.c3_st_spawn_tolerance_config = {
            "pos_tol_m": args_cli.c3_st_pos_tol_mm / 1000.0,
            "rot_tol_rad": math.radians(args_cli.c3_st_rot_tol_deg) if args_cli.c3_st_rot_tol_deg is not None else None,
            "command_name": args_cli.c3_st_command_name,
        }
    print(
        f"[verify] c4_seating_gate enabled={args_cli.c4_seating_gate}  "
        f"c4_terminate_on_grasp enabled={args_cli.c4_terminate_on_grasp}  "
        f"c3_st_spawn_tolerance enabled={args_cli.c3_st_spawn_tolerance}  "
        f"success.func={success_func.__name__}",
        flush=True,
    )
    env_cfg.terminations.success = DoneTerm(func=success_func, time_out=True, params=success_params)

    # -- recorder plumbing, otherwise UNCHANGED from record_reset_states.py (requirement #2's whole
    # point) -- MUST be set on env_cfg before gym.make(), same as record_reset_states.py:132-136,
    # since RecorderManager is built from cfg at env construction time. The pair dir alone must be
    # COMPUTED, matching record_reset_states.py:114-116/128, not hardcoded: this plant's scene has no
    # fixture (only `object`, the leg -- see module docstring), so the receptive half of the pair is
    # supplied on the CLI instead of read off a second scene entity. Read off env_cfg (pre-
    # construction), matching record_reset_states.py's own env_cfg.scene.insertive_object.spawn.
    # usd_path pattern, rather than env.scene (post-construction) -- the object's usd_path is already
    # fully resolved on env_cfg and does not require a live env to read.
    # compute_pair_dir sorts its arguments (utils.py:391-400), so passing (object, receptive) vs.
    # (receptive, object) here produces the identical directory name as record_reset_states.py would.
    pair = task_mdp.utils.compute_pair_dir(
        env_cfg.scene.object.spawn.usd_path, args_cli.receptive_usd_path
    )
    output_dir = os.path.join(args_cli.dataset_dir, "Resets", pair)
    os.makedirs(output_dir, exist_ok=True)
    env_cfg.recorders = StableStateRecorderManagerCfg()
    # Re-key rigid_object names to the OmniReset TRAINING scene's own naming AT RECORD TIME -- see
    # _DexliftToTrainingSceneRecorder's docstring for why this must live here rather than in a
    # post-processing pass (a killed run, this script's own documented usage, would skip a pass that
    # only runs after the loop exits; it cannot skip a rename that happens inside every flush).
    env_cfg.recorders.record_pre_reset_states.class_type = _DexliftToTrainingSceneRecorder
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = f"resets_{args_cli.reset_type}.pt"
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = TorchDatasetFileHandler

    # The whole defect this script is being fixed for is that ITS root diverged from the trainer's
    # root UNNOTICED, because the mismatch is only visible in a relative path -- print the fully
    # RESOLVED, ABSOLUTE path so a wrong --dataset_dir (or a script launched from the wrong CWD) is
    # obvious at startup instead of silently writing states the trainer will never find.
    resolved_output_file = os.path.abspath(os.path.join(output_dir, f"resets_{args_cli.reset_type}.pt"))
    print(f"[generator] OUTPUT_PATH (resolved, absolute): {resolved_output_file}", flush=True)

    env_cfg.seed = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    print(f"[generator] robot DOF: {env.scene['robot'].num_joints}, action dim: {env.action_manager.total_action_dim}")

    # -- (1) POLICY ROLLOUT: play.py:142-215's idiom, verbatim.
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_act = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    wrapped_env = RlGamesVecEnvWrapper(env, args_cli.device, clip_obs, clip_act, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: wrapped_env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    agent_cfg["params"]["config"]["num_actors"] = env.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(args_cli.checkpoint)
    player.reset()
    player.has_batch_dimension = True  # see play.py's own comment on get_action's unsqueeze bug
    print("[generator] POLICY_LOADED", flush=True)

    obs = unwrap(wrapped_env.reset())

    # SPAWN-VS-ACCEPTANCE TRACE (bead: C4 policy-route diagnostic, temporary instrumentation --
    # team-lead's ask: does the leg drift out of the bore slowly, or get carried out almost
    # immediately). Tracks, per env, the leg-to-fixture distance measured at THAT env's most recent
    # spawn (auto-reset happens INSIDE wrapped_env.step(), so the freshly-spawned pose is readable
    # right after step() returns for any env whose `dones` just fired -- see the SNAPSHOT BEFORE
    # step() comment below for the same auto-reset trap in this file). Printed only for the first
    # handful of ACCEPTED episodes, paired against that same episode's recorded (pre-reset) pose.
    _SPAWN_TRACE_ENABLED = os.environ.get("DEXLIFT_SPAWN_TRACE") == "1"
    _spawn_trace_count = 0
    _spawn_trace_max = 15
    if _SPAWN_TRACE_ENABLED:
        _obj = env.scene["object"]
        _fix = env.scene["receptive_object"]
        _spawn_dist_mm = torch.linalg.norm(
            _obj.data.root_pos_w - _fix.data.root_pos_w, dim=-1
        ) * 1000.0

    success_term = env.termination_manager.get_term_cfg("success").func  # the held_with_probe instance

    # -- C2-VIA-REWIND (bead UWLab-weyl), gated behind --c2_rewind ONLY -- see _C2RewindBank's
    # docstring. Constructed AFTER the initial reset (obs = unwrap(wrapped_env.reset()) above), so
    # seed_spawn_positions() below reads each env's real first-episode spawn pose.
    c2 = None
    if args_cli.c2_rewind:
        control_hz = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
        c2 = _C2RewindBank(
            env=env,
            success_term=success_term,
            offsets_s=args_cli.c2_offsets_s,
            control_hz=control_hz,
            output_dir=output_dir,
            reset_type_stem=args_cli.c2_reset_type,
            max_resting_speed_m_s=args_cli.c2_max_resting_speed,
        )
        c2.seed_spawn_positions()
        print(f"[c2] control_hz={control_hz:.2f}  ring_depth={c2.ring_depth} control steps", flush=True)
        for off_s, off_steps, path in c2.offsets:
            print(
                f"[c2] OUTPUT_PATH offset={off_s:.2f}s ({off_steps} steps): {path}  -- BACK UP THIS "
                "FILE BEFORE RUNNING if it already exists (this run truncates it on write()).",
                flush=True,
            )

    # -- ARM 1 (bead UWLab-xp05.1), gated behind --c4_rewind_deepest ONLY -- see
    # _C4DeepestGraspBank's docstring. Requires the FULL settle+probe success_term (asserted
    # mutually exclusive with --c4_terminate_on_grasp above), so success_term.thumb_contact_names/
    # tip_contact_names/force_threshold/object_cfg here are always held_with_probe's or
    # SeatedHeldWithProbe's own attributes.
    c4rewind = None
    if args_cli.c4_rewind_deepest:
        rewind_geometry = _MatingFrameGeometry(
            leg_usd_path=env_cfg.scene.object.spawn.usd_path,
            fixture_usd_path=args_cli.receptive_usd_path,
            engaged_span_mm=args_cli.c4_engaged_span_mm,
            device=env.device,
        )
        rewind_settle_steps = (
            args_cli.c4_rewind_settle_steps if args_cli.c4_rewind_settle_steps is not None else SETTLE_STEPS
        )
        rewind_output_path = os.path.abspath(os.path.join(output_dir, f"resets_{args_cli.c4_rewind_reset_type}.pt"))
        c4rewind = _C4DeepestGraspBank(
            env=env,
            success_term=success_term,
            geometry=rewind_geometry,
            receptive_object_name="receptive_object",
            output_path=rewind_output_path,
            settle_steps=rewind_settle_steps,
            depth_min_mm=args_cli.c4_rewind_depth_min_mm,
            depth_max_mm=args_cli.c4_rewind_depth_max_mm,
            lateral_max_mm=args_cli.c4_rewind_lateral_max_mm,
            tilt_max_deg=args_cli.c4_rewind_tilt_max_deg,
            max_speed_m_s=args_cli.c4_rewind_max_speed,
            speed_sweep_thresholds=list(args_cli.c4_rewind_speed_sweep) + [float("inf")],
            require_settle=args_cli.c4_rewind_require_settle,
        )
        print(
            f"[c4-rewind-deepest] OUTPUT_PATH (resolved, absolute): {rewind_output_path}  -- BACK UP "
            "THIS FILE BEFORE RUNNING if it already exists (this run truncates it on write()).",
            flush=True,
        )

    n_attempts = 0
    n_accepted_at_reset = 0
    if args_cli.c4_terminate_on_grasp:
        gate_names = ["settled", "opposed_contact", "low_obj_speed", "stable_grasp"]
    else:
        gate_names = ["settled", "opposed_contact", "co_move", "probe_ready", "probe_gripper_moved", "probe_tracks"]
    if args_cli.c4_seating_gate:
        # Seated*.gate_breakdown() adds this key -- see that class's own gate_breakdown. Appended
        # last: an env that fails BOTH the base held-check's own last gate AND seated is attributed
        # to the base gate (first-failing-gate priority order), same convention every other gate
        # here already follows -- and the SAME structural blind spot the team-lead flagged for
        # "seated" under the probe-based route applies here too (see this file's own top-level
        # warning about monitoring via this counter; use validate_c4_bank.py on the BANKED states
        # instead of trusting this breakdown for either arm).
        gate_names = gate_names + ["seated"]
    if args_cli.c3_st_spawn_tolerance:
        # SpawnToleranceHeldWithProbe.gate_breakdown() adds this key -- see that class's own
        # gate_breakdown / _SpawnPoseToleranceAddon.check(). Appended last, same convention as
        # "seated" immediately above (mutually exclusive with it, asserted before Isaac starts, so
        # this and the "seated" branch never both fire).
        gate_names = gate_names + ["spawn_tolerance"]
    rejection_counts = {g: 0 for g in gate_names}
    rejection_counts["accepted"] = 0
    # -- PER-GATE REACH COUNTS (repose-recipe requirement, team-lead 2026-08-29, this file named as
    # owner). rejection_counts[g] (first-failing-gate, priority order = gate_names) is a NUMERATOR
    # with no denominator: a gate reading zero first-failing-gate hits could mean "this gate almost
    # never rejects anything" or "almost nothing survives long enough to reach it" -- the exact
    # ambiguity that made C4's seated count read as a flat zero (this file's own top-level warning).
    # reach_counts[g] is that denominator: how many of this run's DONE episodes had every gate
    # BEFORE g (in gate_names priority order) evaluate True, i.e. how many episodes g's own result
    # was actually the deciding one for. reach_counts[gate_names[0]] == n_attempts always (every
    # episode reaches the first gate trivially); local_fail_rate[g] = rejection_counts[g] /
    # reach_counts[g] is the read repose-recipe's gate decomposition needs and this breakdown could
    # not previously produce.
    reach_counts = {g: 0 for g in gate_names}

    # -- DIAGNOSTIC ONLY, no gate/threshold touched: for every episode whose FIRST failing gate is
    # probe_ready specifically, record (a) which OTHER termination fired this same step and (b) the
    # episode_length_buf value at that step. Answers whether probe_ready rejections are budget
    # exhaustion (mostly time_out, steps near max) vs a live fault (abnormal_robot) vs a lost/flung
    # object (object_out_of_bound).
    other_term_names = [n for n in env.termination_manager.active_terms if n != "success"]
    probe_ready_term_histogram = {n: 0 for n in other_term_names}
    probe_ready_term_histogram["none_of_the_above"] = 0
    probe_ready_episode_lengths: list[int] = []

    # -- PROGRESS HEARTBEAT (num_reset_states mode only; smoke mode is short enough to not need it).
    # A genuinely-progressing multi-hour run and a hung one are otherwise indistinguishable from
    # outside the process: this loop prints nothing between the POLICY_LOADED line and the final
    # GENERATOR RESULT block. Unconditional (no verbosity flag) and unbuffered -- both are the point.
    n_attempts_at_last_progress = 0
    n_attempts_at_last_dump = 0
    last_progress_time = time.monotonic()

    n_steps = args_cli.smoke_steps if args_cli.num_reset_states == 0 else 10**9
    for step in range(n_steps):
        with torch.inference_mode():
            act = player.get_action(obs, is_deterministic=True)

            # -- probe: constant bias on the 6 arm action dims, on top of the policy's own command,
            # for every env held_with_probe currently has a probe window OPEN for. This is a
            # RE-ARMING, per-env, event-triggered window now (not a fixed global one -- see
            # held_check.py's module docstring for why: a fixed steps 60-70 window measured the
            # wrong moment on every episode in the STEP-1 diagnostic). success_term.probe_active is
            # the SAME state the term itself uses to decide when to finalize a probe, so the action
            # bias and the measurement window cannot drift apart.
            # ARM 2 (--c4_terminate_on_grasp) HAS NO PROBE -- TerminateOnGraspSuccess/
            # SeatedTerminateOnGrasp deliberately do not define probe_active (see those classes'
            # docstrings: a displacement probe is exactly the latency Arm 2 removes), so this whole
            # injection is skipped for that success_term rather than crashing on a missing attribute.
            in_probe = getattr(success_term, "probe_active", None)
            if in_probe is not None and in_probe.any():
                act[in_probe, 0:6] = act[in_probe, 0:6] + PROBE_ARM_ACTION_BIAS

            # DIAGNOSTIC CONTROL (--zero_action): unconditional, LAST, overriding anything above
            # (including the probe bias) -- a true "robot never moves" control run. See the flag's
            # own help text for why zero commands target==q (holds current pose) rather than
            # "no-op" in some other sense, given RelativeJointPositionAction's use_zero_offset=True.
            if args_cli.zero_action:
                act = torch.zeros_like(act)

            # SNAPSHOT BEFORE step(): ManagerBasedRLEnv auto-resets done envs INSIDE the same
            # step() call (episode_length_buf zeroed for them before this call returns) -- see
            # held_check.py's own comment on this exact trap. episode_length_buf increments once
            # per step before terminations are evaluated, so pre-step value + 1 is what it was AT
            # the moment termination fired, before any reset zeroed it.
            episode_length_buf_pre_step = env.episode_length_buf.clone()
            ret = wrapped_env.step(act)
            obs = unwrap(ret[0])
            dones = ret[2].flatten().to(torch.bool)

        # -- C2-VIA-REWIND: unconditional every step (not just when something's done), since first
        # contact can happen on any step and the ring buffer must stay warm for it. See
        # _C2RewindBank.capture_step/check_first_contact docstrings.
        if c2 is not None:
            c2.capture_step(step)
            c2.check_first_contact(step, dones)

        # -- ARM 1 (--c4_rewind_deepest): unconditional every step, same discipline as C2 above --
        # the deepest opposed-contact step can occur on any step, not just when something ends. See
        # _C4DeepestGraspBank.step's own docstring.
        if c4rewind is not None:
            c4rewind.step(dones)

        if dones.any():
            breakdown = success_term.gate_breakdown(env)
            success_now = env.termination_manager.get_term("success")
            done_idx = torch.nonzero(dones).flatten()
            n_attempts += done_idx.numel()

            # -- PER-GATE REACH COUNTS, vectorized over this step's done batch. `still_reaching`
            # starts all-True (every done episode reaches gate_names[0]) and is ANDed down through
            # the SAME priority order the first-failing-gate loop below walks -- reach_counts[g]
            # accumulates BEFORE ANDing in gate g's own result, so it counts "survived every gate
            # strictly before g", not "and also passed g". See the dict's own comment above for why
            # this denominator is needed.
            still_reaching = torch.ones(done_idx.numel(), dtype=torch.bool, device=dones.device)
            for g in gate_names:
                reach_counts[g] += int(still_reaching.sum().item())
                still_reaching = still_reaching & breakdown[g][done_idx]

            if c2 is not None:
                c2.finalize_episodes(done_idx, success_now)

            if c4rewind is not None:
                c4rewind.finalize_episodes(done_idx, success_now)

            if _SPAWN_TRACE_ENABLED and _spawn_trace_count < _spawn_trace_max:
                # OLD holder values (read BEFORE overwrite) = this env's distance at the START of
                # the episode that just ended -- i.e. "at spawn" for the episode success_now is
                # about to judge. The auto-reset already happened inside wrapped_env.step() above,
                # so root_pos_w right now is already each done env's NEXT spawn, not the accepted
                # episode's own pose -- that pose is only recoverable from the output .pt
                # (record_pre_reset_states captured it before this reset overwrote it), so this
                # trace prints ONLY the spawn side; cross-reference the .pt's recorded distance
                # (already measured separately) for the "at acceptance" side of the comparison.
                _spawn_at_episode_start_mm = _spawn_dist_mm[done_idx].clone()
                for _j, idx in enumerate(done_idx.tolist()):
                    if bool(success_now[idx]) and _spawn_trace_count < _spawn_trace_max:
                        print(
                            f"[spawn-trace] env={idx} accepted_episode's spawn_dist_mm="
                            f"{_spawn_at_episode_start_mm[_j].item():.1f}  (compare against this same"
                            " episode's RECORDED distance in the output .pt)",
                            flush=True,
                        )
                        _spawn_trace_count += 1
                # Seed the holder with each done env's NEW spawn, for whichever episode starts next.
                _spawn_dist_mm[done_idx] = torch.linalg.norm(
                    _obj.data.root_pos_w[done_idx] - _fix.data.root_pos_w[done_idx], dim=-1
                ) * 1000.0

            for idx in done_idx.tolist():
                if bool(success_now[idx]):
                    rejection_counts["accepted"] += 1
                    continue
                # attribute to the first gate (in a fixed priority order) that failed for this env.
                for g in gate_names:
                    if not bool(breakdown[g][idx]):
                        rejection_counts[g] += 1
                        if g == "probe_ready":
                            probe_ready_episode_lengths.append(int(episode_length_buf_pre_step[idx].item()) + 1)
                            fired_other = False
                            for n in other_term_names:
                                if bool(env.termination_manager.get_term(n)[idx]):
                                    probe_ready_term_histogram[n] += 1
                                    fired_other = True
                            if not fired_other:
                                probe_ready_term_histogram["none_of_the_above"] += 1
                        break

        new_accepted = env.recorder_manager.exported_successful_episode_count
        if new_accepted > n_accepted_at_reset:
            n_accepted_at_reset = new_accepted

        if args_cli.num_reset_states > 0:
            episodes_since_progress = n_attempts - n_attempts_at_last_progress
            seconds_since_progress = time.monotonic() - last_progress_time
            if (
                episodes_since_progress >= args_cli.progress_every_episodes
                or seconds_since_progress >= args_cli.progress_every_seconds
            ) and n_attempts > 0:
                rate = n_accepted_at_reset / n_attempts
                print(
                    f"[progress] attempts={n_attempts}  accepted={n_accepted_at_reset}"
                    f"/{args_cli.num_reset_states}  acceptance_rate={rate:.2%}",
                    flush=True,
                )
                if c2 is not None:
                    c2.print_progress()
                if c4rewind is not None:
                    c4rewind.print_progress(n_attempts=n_attempts)
                n_attempts_at_last_progress = n_attempts
                last_progress_time = time.monotonic()

            # -- STANDING RULE (harness mandate, item a): dump accumulated diagnostics to disk
            # every ~250 attempts, atomically. A SIGKILL skips report() entirely; this is what
            # survives that. Written next to the rewind bank's own output dir so it travels with
            # the run's other artifacts. Cadence is on n_attempts (not wall-clock), matching the
            # mandate's own "every ~250 attempts" wording.
            if c4rewind is not None and n_attempts - n_attempts_at_last_dump >= 250:
                _dump_path = os.path.join(output_dir, "arm1_progress_diagnostics.json")
                atomic_json_save(c4rewind.as_json_dict(n_attempts=n_attempts), _dump_path)
                print(f"[c4-rewind-deepest] progress diagnostics dumped -> {_dump_path}", flush=True)
                n_attempts_at_last_dump = n_attempts

        if args_cli.num_reset_states > 0 and n_accepted_at_reset >= args_cli.num_reset_states:
            break
        if env.sim.is_stopped():
            break

    print("\n=== GENERATOR RESULT ===", flush=True)
    print(f"attempts (episodes ended): {n_attempts}", flush=True)
    print(f"accepted (success_term True at reset, AND recorder-exported): {n_accepted_at_reset}", flush=True)
    # ARM 2 (--c4_terminate_on_grasp) HAS NO PROBE -- TerminateOnGraspSuccess/SeatedTerminateOnGrasp
    # do not define n_probes_armed/finalized/tracked (see docstring: no displacement probe at all),
    # so this print is skipped rather than crashing on a missing attribute for that success_term.
    if hasattr(success_term, "n_probes_armed"):
        print(f"probes armed: {success_term.n_probes_armed}  finalized: {success_term.n_probes_finalized}  "
              f"tracked (displacement matched): {success_term.n_probes_tracked}", flush=True)
    if n_attempts > 0:
        print(f"acceptance rate: {n_accepted_at_reset / n_attempts:.2%}", flush=True)
    print("rejection breakdown (first failing gate, priority order = gate_names above):", flush=True)
    for g in gate_names + ["accepted"]:
        print(f"  {g:22s}: {rejection_counts[g]}", flush=True)
    # -- PER-GATE REACH COUNTS + local fail rate (repose-recipe requirement, team-lead 2026-08-29).
    # reach_counts[g] is the denominator rejection_counts[g] never had: episodes for which g's
    # result was actually the deciding one (every gate before it, in this SAME priority order,
    # already passed). local_fail_rate=rejection_counts[g]/reach_counts[g] answers "of the states
    # that got this far, how many did THIS gate reject" -- distinct from
    # rejection_counts[g]/n_attempts, which conflates "this gate rejects a lot of what reaches it"
    # with "almost nothing reaches it at all". reach_counts[gate_names[0]] == n_attempts always.
    print("per-gate reach counts (episodes for which this gate's result was the deciding one):", flush=True)
    for g in gate_names:
        local_rate_str = f"{rejection_counts[g] / reach_counts[g]:.2%}" if reach_counts[g] > 0 else "n/a (0 reached)"
        print(
            f"  {g:22s}: reached={reach_counts[g]:<8d} rejected_here={rejection_counts[g]:<8d} "
            f"local_fail_rate={local_rate_str}",
            flush=True,
        )

    # PROBE-ROUTE-ONLY: "probe_ready" only appears in gate_names for the settle+probe route
    # (held_with_probe / SeatedHeldWithProbe) -- --c4_terminate_on_grasp has no probe at all, so
    # this whole diagnostic would just print zeros for that route; skip it rather than print a
    # misleading "zero rejections" for a gate that was never in play.
    if not args_cli.c4_terminate_on_grasp:
        print("\n=== probe_ready REJECTION DIAGNOSTIC (diagnostic only) ===", flush=True)
        print(f"n probe_ready-rejected episodes: {len(probe_ready_episode_lengths)}", flush=True)
        print("which OTHER termination fired the same step (a rejected episode can fire more than one):", flush=True)
        for n, c in probe_ready_term_histogram.items():
            print(f"  {n:22s}: {c}", flush=True)
        if probe_ready_episode_lengths:
            el = torch.tensor(probe_ready_episode_lengths, dtype=torch.float32)
            print(
                f"episode_length_buf at rejection: min={el.min().item():.0f} max={el.max().item():.0f} "
                f"mean={el.mean().item():.1f} median={el.median().item():.0f}",
                flush=True,
            )
            print(f"raw values: {probe_ready_episode_lengths}", flush=True)

    if c2 is not None:
        c2.write()
        c2.report(n_attempts=n_attempts)

    if c4rewind is not None:
        c4rewind.write()
        c4rewind.report(n_attempts=n_attempts)
        # FINAL dump too, not just the every-250-attempts cadence -- covers the tail end of a run
        # that finished (or was killed) between two periodic dumps.
        atomic_json_save(
            c4rewind.as_json_dict(n_attempts=n_attempts),
            os.path.join(output_dir, "arm1_progress_diagnostics.json"),
        )

    # -- POST-RUN BANK ASSERTION (team-lead mandate, 2026-08-22 incident). EXIT_CODE=0 ON THIS
    # HARNESS PROVES NOTHING: an exception raised inside a LAZILY-CONSTRUCTED event term (measured
    # cause -- SpawnPartialAssembly's dataset_dir download 404ing, then TypeError one level up
    # when the raw class is handed the event params as kwargs) can be swallowed well before it
    # ever reaches this process's own exit code, so a run that "completed" having accepted ZERO
    # states must be treated as a FAILURE here, not silently reported as done by virtue of reaching
    # this line. Checks the BANK ITSELF via torch.load, not just the in-memory counter -- a mismatch
    # between what this process COUNTED and what actually landed on disk is exactly as bad as zero
    # states and must fail exactly as loudly (a truncated/partial write is a silent-corruption mode
    # this project has already been burned by once, see _emit's/write()'s own comments elsewhere in
    # this file). Gated on --num_reset_states > 0: a --smoke_steps-only diagnostic run (num_reset_
    # states == 0) legitimately expects zero or few accepted states and is not what this guards.
    def _assert_bank_matches(path: str, expected_n: int, label: str) -> dict:
        assert os.path.exists(path), (
            f"POST-RUN ASSERTION FAILED ({label}): {expected_n} states were counted as accepted/"
            f"emitted this run but {path} does not exist on disk. Check the log above for a "
            "swallowed exception -- a lazily-constructed event term failing (e.g. "
            "SpawnPartialAssembly's dataset_dir 404ing) is a known cause; see "
            "DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR."
        )
        raw = torch.load(path, map_location="cpu", weights_only=False)
        rigid_object = raw["initial_state"]["rigid_object"]
        any_key = next(iter(rigid_object))
        n_written = len(rigid_object[any_key]["root_pose"])
        assert n_written == expected_n, (
            f"POST-RUN ASSERTION FAILED ({label}): in-memory count ({expected_n}) does not match "
            f"what was actually written to {path} ({n_written} states) -- a silent partial/"
            "mismatched write is worse than a crash."
        )
        print(f"[verify] POST-RUN BANK CHECK PASSED ({label}): {path} contains {n_written} states.", flush=True)
        return raw

    _main_bank_raw = None
    if args_cli.num_reset_states > 0:
        if n_accepted_at_reset == 0:
            raise RuntimeError(
                f"POST-RUN ASSERTION FAILED: --num_reset_states={args_cli.num_reset_states} was "
                f"requested but 0 states were accepted (n_attempts={n_attempts}). Refusing to exit "
                "as if this run succeeded -- check the log above for a swallowed exception."
            )
        _main_bank_raw = _assert_bank_matches(resolved_output_file, n_accepted_at_reset, "accept-time bank")

    if c4rewind is not None and c4rewind.emitted_count > 0:
        # NOT gated on num_reset_states>0 only, and NOT treating emitted_count==0 as fatal on its
        # own: an Arm-1 run can legitimately emit zero rewind states (every candidate rejected by
        # the depth/lateral/tilt band or the speed gate) while the accept-time bank above is
        # perfectly healthy -- that is a yield finding about the rewind anchor, not a crash. What
        # this DOES check is the same "write() claimed N states, are N states actually on disk"
        # consistency the accept-time bank gets above, whenever there is anything to check.
        _assert_bank_matches(rewind_output_path, c4rewind.emitted_count, "Arm 1 rewind bank")

    # -- THE WHOLE THESIS OF ARM 1, made measurable (harness ask, 2026-08-22): how far the deepest
    # opposed-contact moment of each ACCEPTED episode sits above that SAME episode's accept-time
    # depth. Computed here, not in _C4DeepestGraspBank, because it needs the accept-time BANK'S OWN
    # recorded pose (captured by RecorderManager.record_pre_reset BEFORE reset -- the true terminal,
    # probe-validated state) joined against c4rewind's own per-episode list by ORDINAL POSITION (see
    # that list's own docstring for why the two orders are verified to match). Nothing here reads
    # LIVE scene state for a done env -- that data is already gone by the time this runs, same trap
    # as everywhere else in this file.
    if c4rewind is not None and _main_bank_raw is not None:
        _deepest_list = c4rewind.accepted_episode_deepest_candidate_depth_mm
        _ins_poses = _main_bank_raw["initial_state"]["rigid_object"]["insertive_object"]["root_pose"]
        _rec_poses = _main_bank_raw["initial_state"]["rigid_object"]["receptive_object"]["root_pose"]
        # HARD ASSERTION, not a warn-and-truncate (harness mandate, 2026-08-22): if these two counts
        # ever diverge, the ordinal-alignment invariant this whole computation depends on (verified
        # by reading recorder_manager.py -- both finalize_episodes and export_episodes iterate the
        # SAME done env_ids list in the SAME order) has broken, and EVERY gap number below would be
        # a coincidentally-plausible pairing between the wrong episodes, not an approximation of the
        # right one. Fail loudly instead of silently joining a truncated, misaligned prefix.
        assert len(_deepest_list) == len(_ins_poses) == len(_rec_poses), (
            f"POST-RUN ASSERTION FAILED: accepted_episode_deepest_candidate_depth_mm has "
            f"{len(_deepest_list)} entries but the accept-time bank has {len(_ins_poses)} "
            f"insertive_object / {len(_rec_poses)} receptive_object poses -- the ordinal-alignment "
            "invariant the gap computation depends on has broken this run. Every gap number would "
            "be garbage (a plausible-looking pairing between the wrong episodes), so refusing to "
            "compute it rather than reporting one."
        )
        _n_join = len(_deepest_list)
        _ins_t = torch.stack(_ins_poses[:_n_join])
        _rec_t = torch.stack(_rec_poses[:_n_join])
        _accept_depth_m, _, _ = rewind_geometry.decompose(
            _ins_t[:, :3], _ins_t[:, 3:7], _rec_t[:, :3], _rec_t[:, 3:7]
        )
        _accept_depth_mm = (_accept_depth_m * 1000.0).tolist()
        _gaps = [
            _deepest_list[j] - _accept_depth_mm[j] for j in range(_n_join) if _deepest_list[j] is not None
        ]
        if _gaps:
            _gaps_t = torch.tensor(_gaps)
            print(
                f"[verify] ARM 1 DEEPEST-CANDIDATE-MINUS-ACCEPT-TIME-DEPTH GAP (mm), n={_gaps_t.numel()} "
                f"of {_n_join} joined accepted episodes: min={_gaps_t.min().item():.3f} "
                f"median={_gaps_t.median().item():.3f} max={_gaps_t.max().item():.3f} "
                f"mean={_gaps_t.mean().item():.3f}  -- THIS IS THE WHOLE THESIS OF ARM 1: near 0 means "
                "the deepest moment IS where the policy parks and rewinding buys nothing (refutation, "
                "same shape as Arm 3); well above 0 means a deeper state existed mid-episode and was "
                "missed only by anchoring acceptance on the terminal step.",
                flush=True,
            )
        else:
            print("[verify] ARM 1 GAP: no joinable accepted episodes with a candidate -- cannot compute.", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
