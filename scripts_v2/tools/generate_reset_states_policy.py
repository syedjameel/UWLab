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
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

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
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.managers.recorder_manager import DatasetExportMode  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from uwlab.utils.datasets.torch_dataset_file_handler import TorchDatasetFileHandler  # noqa: E402
from uwlab_tasks.manager_based.manipulation.dexlift.dexlift_ur5e_delto_actions import (  # noqa: E402
    ARM_JOINT_NAMES as DEXLIFT_ARM_JOINT_NAMES,
)
from uwlab_tasks.manager_based.manipulation.dexlift.mdp.held_check_core import (  # noqa: E402
    PROBE_ARM_ACTION_BIAS,
)
from uwlab_tasks.manager_based.manipulation.dexlift.mdp.rewards import (  # noqa: E402
    _sensor_force_magnitudes,  # reused, not reimplemented -- see rewards.py:40-76 / held_check.py:58
)
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.recorders.recorders import (  # noqa: E402
    StableStateRecorder,
)
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.recorders.recorders_cfg import (  # noqa: E402
    StableStateRecorderManagerCfg,
)


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
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(accum, path)
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


def main() -> None:
    agent_yaml = args_cli.agent_yaml
    if agent_yaml is None:
        agent_yaml = os.path.join(os.path.dirname(os.path.dirname(args_cli.checkpoint)), "params", "agent.yaml")
    with open(agent_yaml) as f:
        agent_cfg = yaml.safe_load(f)
    print(f"[generator] agent yaml: {agent_yaml}")
    print(f"[generator] normalize_input: {agent_cfg['params']['config'].get('normalize_input')}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # Shrink the GPU collision stack for a small-env smoke/dev scene (the training default is sized
    # for thousands of parallel envs).
    env_cfg.sim.physx.gpu_collision_stack_size = 2**24

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
    _object_pose_class = env_cfg.commands.object_pose.class_type
    _goal_is_pinned = _object_pose_class is dexlift_mdp.GoalAtSpawnPoseCommand
    if _goal_is_pinned:
        print(
            f"[verify] commands.object_pose.class_type = {_object_pose_class.__name__}  "
            f"goal = PINNED to object spawn pose (DEXLIFT_GOAL_AT_SPAWN=1 or DEXLIFT_PARTIAL_ASSEMBLY=1)",
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
    _require(
        "DEXLIFT_GOAL_AT_SPAWN/DEXLIFT_PARTIAL_ASSEMBLY",
        requested=_requested_goal_at_spawn,
        actual=_goal_is_pinned,
        message=(
            f"commands.object_pose.class_type is {_object_pose_class.__name__!r}, not "
            "GoalAtSpawnPoseCommand."
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
    _require(
        "--reset_type=ObjectPartiallyAssembledEEGrasped",
        requested=_reset_type_requests_partial_assembly,
        actual=_built_partial_assembly,
        message=(
            f"requires DEXLIFT_PARTIAL_ASSEMBLY=1 exported AND events.reset_object.func built as "
            f"SpawnPartialAssembly; got DEXLIFT_PARTIAL_ASSEMBLY={os.environ.get('DEXLIFT_PARTIAL_ASSEMBLY')!r} "
            f"and events.reset_object.func={_reset_object_func.__name__!r}."
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
    env_cfg.terminations.success = DoneTerm(func=dexlift_mdp.held_with_probe, time_out=True, params={})

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

    n_attempts = 0
    n_accepted_at_reset = 0
    gate_names = ["settled", "opposed_contact", "co_move", "probe_ready", "probe_gripper_moved", "probe_tracks"]
    rejection_counts = {g: 0 for g in gate_names}
    rejection_counts["accepted"] = 0

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
            in_probe = success_term.probe_active
            if in_probe.any():
                act[in_probe, 0:6] = act[in_probe, 0:6] + PROBE_ARM_ACTION_BIAS

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

        if dones.any():
            breakdown = success_term.gate_breakdown(env)
            success_now = env.termination_manager.get_term("success")
            done_idx = torch.nonzero(dones).flatten()
            n_attempts += done_idx.numel()

            if c2 is not None:
                c2.finalize_episodes(done_idx, success_now)

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
                n_attempts_at_last_progress = n_attempts
                last_progress_time = time.monotonic()

        if args_cli.num_reset_states > 0 and n_accepted_at_reset >= args_cli.num_reset_states:
            break
        if env.sim.is_stopped():
            break

    print("\n=== GENERATOR RESULT ===", flush=True)
    print(f"attempts (episodes ended): {n_attempts}", flush=True)
    print(f"accepted (held_with_probe True at reset, AND recorder-exported): {n_accepted_at_reset}", flush=True)
    print(f"probes armed: {success_term.n_probes_armed}  finalized: {success_term.n_probes_finalized}  "
          f"tracked (displacement matched): {success_term.n_probes_tracked}", flush=True)
    if n_attempts > 0:
        print(f"acceptance rate: {n_accepted_at_reset / n_attempts:.2%}", flush=True)
    print("rejection breakdown (first failing gate, priority order = gate_names above):", flush=True)
    for g in gate_names + ["accepted"]:
        print(f"  {g:22s}: {rejection_counts[g]}", flush=True)

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

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
