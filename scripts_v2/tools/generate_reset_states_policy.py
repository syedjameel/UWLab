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
from uwlab_tasks.manager_based.manipulation.dexlift.mdp.held_check_core import (  # noqa: E402
    PROBE_ARM_ACTION_BIAS,
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
    _reset_object_func = env_cfg.events.reset_object.func
    _reset_object_is_clearance_term = _reset_object_func is dexlift_mdp.reset_object_pose_with_clearance
    if _reset_object_is_clearance_term:
        _reset_object_params = env_cfg.events.reset_object.params
        print(
            f"[verify] events.reset_object.func = {_reset_object_func.__name__}  "
            f"clearance_range={_reset_object_params['clearance_range']} "
            f"half_extents={_reset_object_params['half_extents']} "
            f"surface_z={_reset_object_params['surface_z']}  (DEXLIFT_SPAWN_CLEARANCE=1)",
            flush=True,
        )
    else:
        print(
            f"[verify] events.reset_object.func = {_reset_object_func.__name__}  "
            f"pose_range.z = {env_cfg.events.reset_object.params['pose_range'].get('z')}  "
            f"(DEXLIFT_SPAWN_CLEARANCE unset/not '1')",
            flush=True,
        )

    # -- HARD GUARD, not a print. A silently-unset (or silently-ineffective) DEXLIFT_SPAWN_CLEARANCE
    # yields a plausible WRONG spawn distribution rather than an error -- the exact failure mode
    # that turned a 46.71% acceptance run into a 2.69% one when DEXLIFT_REF_* went unexported (see
    # this file's plant-verification block above). Fail loudly at startup instead of silently
    # generating reset states under the wrong distribution.
    _requested_clearance = os.environ.get("DEXLIFT_SPAWN_CLEARANCE") == "1"
    if _requested_clearance and not _reset_object_is_clearance_term:
        raise RuntimeError(
            f"DEXLIFT_SPAWN_CLEARANCE=1 was requested but the constructed env_cfg's "
            f"events.reset_object.func is {_reset_object_func.__name__!r}, not "
            f"reset_object_pose_with_clearance. The toggle did not take effect -- refusing to "
            f"generate reset states under a silently-wrong spawn distribution."
        )
    if not _requested_clearance and _reset_object_is_clearance_term:
        raise RuntimeError(
            f"DEXLIFT_SPAWN_CLEARANCE was unset (or not '1') but the constructed env_cfg's "
            f"events.reset_object.func is {_reset_object_func.__name__!r} anyway -- something else "
            f"switched the spawn term. Refusing to generate reset states under a distribution the "
            f"launch command did not ask for."
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

    success_term = env.termination_manager.get_term_cfg("success").func  # the held_with_probe instance
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

        if dones.any():
            breakdown = success_term.gate_breakdown(env)
            success_now = env.termination_manager.get_term("success")
            done_idx = torch.nonzero(dones).flatten()
            n_attempts += done_idx.numel()
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

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
