#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Per-reset-type (C1-C4) success-rate evaluation for a SINGLE OmniReset UR5eDelto checkpoint.

NOT chain_eval.py's kind of chain. chain_eval.py (this project's OTHER "chain" script, from the
earlier box-assembly campaign) solves a DIFFERENT problem: THREE DIFFERENT insertive/receptive
pairs (box->target, object->box, cover->box), THREE SEPARATELY-TRAINED policies, run in SEQUENCE
in one physical scene with state carried forward, each stage REBINDING the SAME observation
manager's term params to swap which entity pair it points at. That machinery does not apply here:
there is exactly ONE insertive/receptive pair (leg/fixture) and ONE policy. What "C4 -> C1" means
for THIS campaign is that a single trained policy must be evaluated SEPARATELY against each of the
FOUR RESET-TYPE STARTING DISTRIBUTIONS training already mixes uniformly (TrainEventCfg's own 4-way
reset_from_reset_states) -- i.e. four independent success rates for one policy on one task, not a
sequential handoff between different objects/policies. chain_eval.py is reusable for exactly one
thing here: its "report every stage separately, never collapse to one scalar" convention and its
single-observation-manager discipline, both kept below.

task_N INDEXING, CONFIRMED IN CODE, NOT ASSUMED (team-lead ask): MultiResetManager.__call__ (
omnireset/mdp/events.py) logs ``Metrics/task_{task_idx}_success_rate`` where ``task_idx`` is the
POSITIONAL index into whatever ``reset_types`` list the term's ``params`` were constructed with
(events.py:2423-2428, and ``self.task_id`` set at events.py:2436). TrainEventCfg (rl_state_cfg.py:
278-283) constructs it with:
    reset_types = ["ObjectAnywhereEEAnywhere", "ObjectRestingEEGrasped",
                   "ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped"]
so, for that ordering:
    task_0 = ObjectAnywhereEEAnywhere       (C1)
    task_1 = ObjectRestingEEGrasped          (C2)
    task_2 = ObjectAnywhereEEGrasped         (C3)
    task_3 = ObjectPartiallyAssembledEEGrasped (C4)
confirming task_3 = C4. This script REUSES that exact list/order (not a re-invented one) when it
overrides reset_from_reset_states below, so its own task_N mapping is identical to training's.

GATE VERIFIED, NOT ASSUMED (team-lead ask -- two candidates differ ~10x: true 0.0025m/0.025rad vs
loose/curriculum-parked 0.022m/0.15rad). Predicate: omnireset/mdp/rewards.py, ProgressContext.
__call__ (~lines 143-147): euler_xy_distance = |wrap_to_pi(e_x)| + |wrap_to_pi(e_y)| (e_z
DISCARDED), xyz_distance = norm(relative_pos), position_aligned = xyz_distance <
success_position_threshold, orientation_aligned = euler_xy_distance < success_orientation_threshold,
success = position_aligned & orientation_aligned. Those two thresholds are READ, PER STEP, off
``TaskCommand.success_position_threshold``/``.success_orientation_threshold`` (commands.py:152-153)
-- a single LIVE mutable attribute set ONCE at ``TaskCommand.__init__`` from the receptive object's
OWN metadata.yaml ``success_thresholds`` block (commands.py:100-101; confirmed 0.0025m/0.025rad for
the leg/fixture pair by ``threshold_curriculum``'s own docstring, events.py:3578). THE LOOSE GATE
IS NOT A SEPARATE CONSTANT SOMEWHERE -- it only exists because ``threshold_curriculum``
(events.py:3520+, wired only on ``Ur5eDeltoRelCartesianOSCTrainCfg`` via ``curriculum:
ThresholdCurriculumCfg`` in ur5e_delto_cfg.py:280) OVERWRITES those same two attributes, starting
at ``initial_position_threshold``/``initial_orientation_threshold`` (0.022/0.15 in armC/armG/armH's
own launch commands) and ramping toward the true value -- and is currently PARKED at that loose
initial value for every arm observed so far (rate never crossed ``success_threshold_up``). This
script deliberately builds its env from ``Ur5eDeltoRelCartesianOSCEvalCfg``
(``OmniReset-UR5eDelto-RelCartesianOSC-State-Play-v0``), which does NOT declare its own
``curriculum`` field and therefore inherits ``Ur5eRobotiq2f85RlStateCfg``'s base default,
``curriculum: NoCurriculumsCfg`` (rl_state_cfg.py:849) -- NO threshold_curriculum term AT ALL, so
``success_position_threshold``/``.orientation_threshold`` are NEVER mutated after construction and
stay exactly at the metadata-derived TRUE gate for the whole eval run. Printed and asserted below,
not just claimed: this script reads ``task_command.success_position_threshold`` /
``.orientation_threshold`` directly off the live env after construction and refuses to proceed if
either is not the metadata value (i.e. if some override snuck a curriculum term back in).

DEFAULT EvalCfg's OWN reset mixture IS C1-ONLY (``TrainEvalEventCfg``: reset_types=
["ObjectAnywhereEEAnywhere"], probs=[1.0], rl_state_cfg.py:295-305) -- NOT the 4-way mixture. This
script explicitly overrides ``env_cfg.events.reset_from_reset_states.params`` BEFORE gym.make() to
TrainEventCfg's own 4-item list, uniform 0.25 each, precisely so all four reset types actually get
sampled during this eval (otherwise every episode would silently be C1 and three of the four
per-task counters would sit at zero forever).

NEVER A SINGLE AVERAGED SCALAR, AND NEVER A SILENT ZERO FOR AN UNSAMPLED TASK. Per the epic's own
precedent (OmniReset success rates once averaged over the four reset types and reversed a
conclusion because one type starts already solved), this script's ONLY reported numbers are the
four PER-TASK rates; there is no "overall" figure computed anywhere, so there is nothing to collapse
to. SEPARATELY, and just as load-bearing: ``SuccessMonitor.get_success_rate()`` (success_monitor.py)
computes ``success_buf.sum(dim=1) / success_size.clamp(min=1)`` -- for a task that has NEVER been
sampled, ``success_size`` is 0, the clamp forces the divisor to 1, and the reported rate is a bare
0.0, INDISTINGUISHABLE from "sampled many times, zero successes". This script therefore tracks each
task's raw ``success_size`` itself and refuses to print a rate for any task below
--min_episodes_per_task, printing "INSUFFICIENT_DATA" instead of a number that looks real but is not.

``success_size`` COUNTS RESET EVENTS, NOT COMPLETED EPISODES -- a field-naming trap this script's
own field names must not repeat (team-lead finding, 2026-08-22). ``MultiResetManager.success_update``
is called on EVERY reset, and at environment CONSTRUCTION time (``RslRlVecEnvWrapper.__init__``'s own
``env.reset()``, before this script's rollout loop runs a single step) ``env_ids`` defaults to ALL
envs -- so a single startup reset alone pushes ``num_envs`` entries into the 100-slot buffer
(``success_monitor.py``'s ``success_update``, increment then clamp) and can SATURATE
``success_size`` to 100 before one real episode has run. Those entries are ``eval(success)``
evaluated AT RESET TIME -- i.e. whether the freshly-loaded RESET STATE itself already happens to
satisfy the gate, not "did an episode succeed". At any ``num_envs >= 100`` the entire window starts
as this one-time reset-time snapshot and is not flushed by genuine episode outcomes until ~100 real
terminations have occurred per task. On a bank with LOW at-reset success density (this script's own
C4 smoke test: 1/256) that reads as a deceptively-plausible near-zero; on a bank with HIGH at-reset
density it would read as an inflated near-one, in BOTH cases before this script's OWN
``--min_episodes_per_task`` guard (built on the SEPARATE, genuine ``cumulative_episodes`` counter
below, never on ``success_size``) would let a reader trust it. This is why every field this script
derives from ``success_size`` is named and printed as a RESET-EVENT count, never as an episode count.

A ROLLING WINDOW IS NOT A LONGER-RUN AVERAGE (team-lead correction, the neighbouring case to the
zero-episode trap above): ``SuccessMonitor``'s window is CAPPED at 100 (events.py:2382-2384), so its
reported rate at the end of a 1500-step run is over the SAME n<=100 as at the end of a 200-step run
-- just a later 100-episode slice, not an average over the whole run. Running longer does not add
precision to that number. This script therefore ALSO accumulates its OWN cumulative, non-windowed
per-task counters (every episode termination over the WHOLE run, attributed via ``task_id``
snapshotted before each step -- see the rollout loop's own comment for why the snapshot must happen
BEFORE ``wrapped.step()``, not after) and reports THAT rate, with raw counts and a Wilson 95% CI, as
the primary number -- INSUFFICIENT_DATA is gated on the cumulative count, not the windowed one. The
windowed rate is kept alongside, clearly tagged, so a material disagreement between the two (drift
during a nominally frozen-policy run) stays visible instead of being averaged away.

NO SECOND OBSERVATION MANAGER. Exactly one ``gym.make()`` call, one env, one
``env.observation_manager`` -- this script never constructs a second one (the concrete trap
chain_eval.py's own code comment names: a second manager corrupts obs history/axis-angle state).
Nothing here needs the REBIND chain_eval.py performs either: the insertive/receptive entity NAMES
are identical across all four reset types (only the reset EVENT differs, never the pair), so there
is no pair-swap to rebind in the first place.

Run (dry-run, any checkpoint, --num_envs modest, GPU1 only). --dataset_dir is REQUIRED and NAMED
EXPLICITLY -- see that flag's own help text for why the EvalCfg's own inherited default
(the shipped ./Datasets_ur5e_delto/OmniReset bank) must never be silently trusted:
    UWLAB_TMP_ROOT=/home/dom_iva/tmp TMPDIR=/home/dom_iva/tmp CUDA_VISIBLE_DEVICES=1 \\
        timeout -s KILL 900 /home/dom_iva/venv_uwlab/bin/python -u \\
        scripts_v2/tools/eval_reset_type_success_rates.py \\
        --checkpoint <path/to/model_*.pt> --dataset_dir <path/to/Resets/parent/dir> \\
        --num_envs 256 --num_steps 3000 --min_episodes_per_task 20 --out <out>.json --headless \\
        env.scene.insertive_object=leg200mm env.scene.receptive_object=onelegfixture
"""

from __future__ import annotations

import argparse
import json
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, __file__.rsplit("/scripts_v2/", 1)[0] + "/scripts/reinforcement_learning/rsl_rl")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="OmniReset-UR5eDelto-RelCartesianOSC-State-Play-v0",
                     help="Eval-cfg task id -- see module docstring for why this (not the Train id) is what keeps the TRUE gate active.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument(
    "--dataset_dir", type=str, required=True,
    help=(
        "REQUIRED, no inherited default (team-lead catch): the EvalCfg's own __post_init__ calls "
        "_apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR) with _UR5E_DELTO_RESET_DIR="
        "'./Datasets_ur5e_delto/OmniReset' (ur5e_delto_cfg.py:89) -- the SHIPPED/production bank, "
        "generated against the REJECTED convex-decomposition/hybrid leg collider (both eject the "
        "leg on contact per 2026-08-22 asset-registry fix). A run that silently inherits that "
        "default would score a policy against a broken bank and misread a bank defect as a policy "
        "failure. This script REFUSES to fall back to that default -- the caller must name the "
        "bank explicitly, e.g. ./Datasets_ur5e_delto/OmniReset_band1821_bank for anchors' new one."
    ),
)
parser.add_argument(
    "--reset_types", type=str, default=None,
    help=(
        "Comma-separated reset-type names to evaluate, in the SAME order/positions training itself "
        "uses (see module docstring's task_N INDEXING). Default (unset) = the full 4-way "
        "TrainEventCfg list. NARROW THIS when the pair directory does not yet have all four banks -- "
        "MultiResetManager loads every listed type's .pt file EAGERLY at construction (events.py: "
        "2303-2316) and raises FileNotFoundError immediately if even one is missing, before a single "
        "step runs. Confirmed 2026-08-22: after the leg-asset repoint to SquareTableLeg200mmSdf, the "
        "pair directory MultiResetManager resolves for ANY --dataset_dir is now "
        "OneLegInsertionFixture__SquareTableLeg200mmSdf (pair is derived from the ASSET path, not "
        "from --dataset_dir), and only C4 exists there so far -- pass "
        "--reset_types ObjectPartiallyAssembledEEGrasped to evaluate what actually exists today."
    ),
)
parser.add_argument(
    "--probs", type=str, default=None,
    help="Comma-separated probs matching --reset_types (default: uniform). Only meaningful together with --reset_types.",
)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_steps", type=int, default=3000, help="Control steps to roll out (real physics, deterministic policy).")
parser.add_argument("--min_episodes_per_task", type=int, default=20,
                     help="A per-task rate below this raw episode count is reported as INSUFFICIENT_DATA, never a bare number.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================================
import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

import cli_args  # type: ignore  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import uwlab_tasks  # noqa: F401,E402
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402

# Same 4-way list/order TrainEventCfg itself constructs MultiResetManager with (rl_state_cfg.py:
# 278-283) -- reused verbatim, not re-invented, so this script's task_N mapping is IDENTICAL to
# training's own. See module docstring's "task_N INDEXING" section. This is the DEFAULT only --
# --reset_types can narrow it (see that flag's own help text for why: a bank directory can
# legitimately have fewer than all four types present, and MultiResetManager fails LOUDLY and
# EAGERLY, at construction, on any missing one).
_DEFAULT_RESET_TYPES = [
    "ObjectAnywhereEEAnywhere",          # task_0 = C1
    "ObjectRestingEEGrasped",            # task_1 = C2
    "ObjectAnywhereEEGrasped",           # task_2 = C3
    "ObjectPartiallyAssembledEEGrasped", # task_3 = C4
]
_LABEL_BY_RESET_TYPE = {
    "ObjectAnywhereEEAnywhere": "C1",
    "ObjectRestingEEGrasped": "C2",
    "ObjectAnywhereEEGrasped": "C3",
    "ObjectPartiallyAssembledEEGrasped": "C4",
}
_RESET_TYPES = args_cli.reset_types.split(",") if args_cli.reset_types else list(_DEFAULT_RESET_TYPES)
_RESET_TYPE_LABELS = [_LABEL_BY_RESET_TYPE.get(rt, rt) for rt in _RESET_TYPES]
if args_cli.probs:
    _PROBS = [float(p) for p in args_cli.probs.split(",")]
    assert len(_PROBS) == len(_RESET_TYPES), f"--probs has {len(_PROBS)} values, --reset_types has {len(_RESET_TYPES)}"
else:
    _PROBS = [1.0 / len(_RESET_TYPES)] * len(_RESET_TYPES)


def main() -> None:  # noqa: C901
    @hydra_task_config(args_cli.task, args_cli.agent)
    def _inner(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
        device = args_cli.device or env_cfg.sim.device
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)

        # ---- OVERRIDE THE DEFAULT C1-ONLY EVAL MIXTURE -- see module docstring's "DEFAULT EvalCfg's
        # OWN reset mixture IS C1-ONLY" section. Uses _RESET_TYPES/_PROBS (full 4-way by default,
        # narrowable via --reset_types/--probs -- see that flag's help text). ----
        rt = env_cfg.events.reset_from_reset_states
        rt.params["reset_types"] = list(_RESET_TYPES)
        rt.params["probs"] = list(_PROBS)
        # ---- BANK PATH, EXPLICIT, NEVER THE INHERITED SHIPPED DEFAULT (team-lead catch). See
        # --dataset_dir's own help text for why the EvalCfg's own __post_init__ default
        # (_UR5E_DELTO_RESET_DIR) is not trustworthy here. Printed prominently, same discipline as
        # the GATE CHECK block below -- this is the same "seam" class as the asset registry: a
        # correct-looking config can still silently point at the wrong bank. ----
        rt.params["dataset_dir"] = args_cli.dataset_dir
        print(
            f"[eval] reset_from_reset_states OVERRIDDEN ({len(_RESET_TYPES)}-way): {rt.params['reset_types']} probs={rt.params['probs']}\n"
            f"[eval] BANK PATH (explicit, --dataset_dir): {args_cli.dataset_dir}"
            f"  (EvalCfg's own inherited default would have been ./Datasets_ur5e_delto/OmniReset -- "
            f"NOT used unless that is literally what --dataset_dir names)",
            flush=True,
        )

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        u = env.unwrapped

        # ---- GATE VERIFICATION, printed and asserted before any rollout step -- see module
        # docstring's "GATE VERIFIED, NOT ASSUMED" section. ----
        task_command = u.command_manager.get_term("task_command")
        pos_thr = float(task_command.success_position_threshold)
        rot_thr = float(task_command.success_orientation_threshold)
        curriculum_terms = list(vars(env_cfg.curriculum).keys()) if hasattr(env_cfg, "curriculum") else []
        print(
            f"[eval] GATE CHECK: task_command.success_position_threshold={pos_thr:.6f} m, "
            f".success_orientation_threshold={rot_thr:.6f} rad; env_cfg.curriculum terms={curriculum_terms}",
            flush=True,
        )
        assert not curriculum_terms, (
            f"REFUSING: env_cfg.curriculum declares terms {curriculum_terms} -- this eval task is "
            "supposed to carry NO curriculum terms (NoCurriculumsCfg) so the TRUE metadata gate is "
            "never overwritten. If this fires, --task no longer resolves to a curriculum-free EvalCfg."
        )
        assert pos_thr < 0.01 and rot_thr < 0.10, (
            f"REFUSING: success_position_threshold={pos_thr} / success_orientation_threshold={rot_thr} "
            "look like the LOOSE curriculum-parked gate (~0.022/0.15), not the TRUE metadata gate "
            "(~0.0025/0.025) this script's whole design assumes. Something is overriding the gate."
        )
        print("[eval] GATE CHECK PASSED -- TRUE metadata gate confirmed active, no curriculum term present.", flush=True)

        # ---- MultiResetManager instance itself, for per-env task_id (which reset type this env's
        # CURRENT episode was drawn from) and the SuccessMonitor's raw per-task episode counts. ----
        reset_term = u.event_manager.get_term_cfg("reset_from_reset_states").func
        assert hasattr(reset_term, "task_id") and hasattr(reset_term, "success_monitor"), (
            "REFUSING: reset_from_reset_states term does not expose .task_id/.success_monitor -- "
            "not the MultiResetManager this script's per-task accounting depends on."
        )
        # ProgressContext instance -- team-lead catch: SuccessMonitor's window is ROLLING (capped at
        # 100, events.py:2382-2384 / success_monitor.py), so the rate it reports at the END of a long
        # run is over the LAST <=100 episodes per task, NOT a longer-run average -- running 1500
        # steps instead of 200 slides the same-size window forward, it does not add precision. This
        # script therefore ALSO accumulates its own CUMULATIVE per-task counters (independent of
        # SuccessMonitor entirely), reading pc.success directly. Safe to read AFTER wrapped.step()
        # returns for a just-terminated env: ProgressContext.reset() (rewards.py:113-115) only zeros
        # continuous_success_counter, never self.success, so self.success[i] still holds the value
        # computed during that env's LAST reward-evaluation pass (i.e. the ending episode's own
        # outcome) even though the env has already been auto-reset into its NEXT episode by the time
        # this script reads it -- the SAME success signal MultiResetManager's own SuccessMonitor
        # update uses (identical `eval(success)` string), just accumulated without the 100-cap discard.
        pc = u.reward_manager.get_term_cfg("progress_context").func
        assert hasattr(pc, "success"), (
            "REFUSING: progress_context term does not expose .success -- cumulative counting depends on it."
        )
        num_tasks = len(_RESET_TYPES)
        cum_episodes = torch.zeros(num_tasks, dtype=torch.long)
        cum_successes = torch.zeros(num_tasks, dtype=torch.long)

        def _wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
            """95% Wilson score interval -- same formula measure_contact_grasp_closedloop.py already
            uses for this project's other per-state certification numbers, reused not re-derived."""
            if n == 0:
                return (float("nan"), float("nan"))
            p = x / n
            denom = 1 + z * z / n
            center = (p + z * z / (2 * n)) / denom
            margin = (z / denom) * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
            return (center - margin, center + margin)

        # ---- load the checkpoint, ACTOR-ONLY -- chain_eval.py's convention (its load_policy()),
        # NOT play.py's runner.load() (full actor+critic strict state-dict load). CORRECTED after a
        # smoke-test failure exposed the assumption behind the original choice was wrong: this
        # script deliberately builds its env from Ur5eDeltoRelCartesianOSCEvalCfg, NOT the TrainCfg
        # the checkpoint was actually trained under (see module docstring's "GATE VERIFIED" section
        # for why -- Eval carries no threshold_curriculum term). Eval's critic observation (privileged
        # DR/sysid terms) is NOT the same width as Train's -- confirmed empirically: armH's model_200
        # checkpoint has critic.0.weight shaped for 623 obs dims, this env's critic group resolves to
        # 530. runner.load()'s strict load fails on that mismatch even though the CRITIC is never used
        # for inference at all -- only actor.* and actor_obs_normalizer.* matter for get_inference_policy().
        # This is exactly the divergence chain_eval.py's own load_policy() docstring names (there,
        # caused by swapped-in ctx entities; here, by deliberately using Eval instead of Train cfg) --
        # same shape of problem, same fix, reused rather than re-discovered a second time.
        resume_path = retrieve_file_path(args_cli.checkpoint)
        print(f"[eval] loading checkpoint (ACTOR ONLY -- see code comment for why): {resume_path}", flush=True)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic
        _blob = torch.load(resume_path, map_location=device)
        _sd = _blob.get("model_state_dict", _blob.get("model", _blob))
        _actor_sd = {k: v for k, v in _sd.items() if k.split(".")[0] in ("actor", "actor_obs_normalizer", "std")}
        _load_result = policy_nn.load_state_dict(_actor_sd, strict=False)
        print(
            f"[eval] actor-only load: {len(_actor_sd)}/{len(_sd)} checkpoint keys used (critic/critic_obs_normalizer "
            f"skipped -- never used for inference); load_state_dict result: {_load_result}",
            flush=True,
        )
        policy = runner.get_inference_policy(device=device)

        obs = wrapped.get_observations()
        with torch.inference_mode():
            for step in range(args_cli.num_steps):
                # SNAPSHOT task_id BEFORE stepping -- MultiResetManager overwrites self.task_id[env_ids]
                # for the NEXT episode INSIDE this same wrapped.step() call, for any env that
                # terminates this step (ManagerBasedRLEnv auto-resets done envs before step() returns
                # -- the same trap this project's own harnesses have hit repeatedly). Reading task_id
                # AFTER step() would attribute an ending episode to whatever task its NEXT episode
                # happened to draw, not the one that just ended.
                task_id_before = reset_term.task_id.clone()
                actions = policy(obs)
                obs, _reward, dones, _extras = wrapped.step(actions)
                policy_nn.reset(dones)

                # CUMULATIVE, non-windowed accounting -- team-lead ask. pc.success is safe to read
                # here (see the note where `pc` was fetched above) for exactly the envs marked done.
                done_idx = dones.nonzero(as_tuple=True)[0]
                if done_idx.numel() > 0:
                    t_ids = task_id_before[done_idx].to(dtype=torch.long, device="cpu")
                    succ = pc.success[done_idx].to(dtype=torch.long, device="cpu")
                    cum_episodes += torch.bincount(t_ids, minlength=num_tasks)
                    cum_successes += torch.bincount(t_ids, weights=succ.float(), minlength=num_tasks).long()

                if (step + 1) % 200 == 0 or step == args_cli.num_steps - 1:
                    reset_event_counts = reset_term.success_monitor.success_size.tolist()  # RESET-EVENT counts, NOT episode counts -- see module docstring
                    rates = reset_term.success_monitor.get_success_rate().tolist()
                    print(
                        f"[eval] step {step + 1}/{args_cli.num_steps}  "
                        f"windowed_reset_event_count(per task, NOT an episode count)={reset_event_counts}  "
                        f"windowed_rates={['%.3f' % r for r in rates]}  "
                        f"cumulative_episodes(per task, GENUINE completed-episode count)={cum_episodes.tolist()}  "
                        f"cumulative_successes(per task)={cum_successes.tolist()}",
                        flush=True,
                    )

        reset_event_counts = reset_term.success_monitor.success_size.tolist()  # RESET-EVENT counts, NOT episode counts -- see module docstring
        rates = reset_term.success_monitor.get_success_rate().tolist()
        assert len(reset_event_counts) == num_tasks and len(rates) == num_tasks, (
            f"REFUSING: success_monitor tracks {len(reset_event_counts)} tasks, expected {num_tasks} -- "
            "reset_types override above did not take, or MultiResetManager resolved a different num_tasks."
        )

        # SuccessMonitorCfg's own hardcoded window (MultiResetManager.__init__, events.py:2382-2384:
        # `SuccessMonitorCfg(monitored_history_len=100, ...)`) -- read here as a literal constant
        # matching that call, not re-derived, so this label can never silently drift from the real
        # window if that constant ever changes.
        #
        # TEAM-LEAD CORRECTION, load-bearing, RESOLVED 2026-08-22 (was briefly mis-attributed to a
        # "rolling window read as a longer average" issue alone -- the deeper mechanism is worse than
        # that and is why every field below is named "reset_event", never "episode"):
        # SuccessMonitor.success_size counts RESET EVENTS, not completed episodes.
        # MultiResetManager.success_update() runs on EVERY reset (events.py:2417), and at environment
        # CONSTRUCTION (RslRlVecEnvWrapper.__init__'s own env.reset(), BEFORE this script's rollout
        # loop runs a single step) env_ids defaults to ALL envs -- so ONE startup reset alone pushes
        # num_envs entries into the 100-slot buffer (success_monitor.py: increment then clamp) and
        # can saturate success_size to 100 before a single real episode has run. Those 100 entries
        # are eval(success) evaluated AT RESET TIME (did the freshly-loaded state already pass the
        # gate?), not "did an episode succeed" -- on a low-at-reset-density bank they read as a
        # deceptively plausible near-zero, on a high-density bank they would read as an inflated
        # near-one, in BOTH cases before a single real episode. The CUMULATIVE count this script
        # tracks itself (cum_episodes/cum_successes, above, entirely independent of SuccessMonitor,
        # counting only GENUINE post-construction episode terminations) is therefore the primary,
        # GATING number below -- INSUFFICIENT_DATA is keyed off cum_episodes, never off
        # success_size. The windowed rate is still reported alongside, clearly labelled as a
        # RESET-EVENT-windowed rate (not an episode-windowed one), so a reader can compare the two;
        # a material disagreement is itself informative (drift, or a bank whose at-reset states
        # already skew toward/away from the gate).
        _SUCCESS_MONITOR_WINDOW_CAP = 100
        print("\n================ PER-RESET-TYPE RESULT (never an average) ================", flush=True)
        print(
            f"[eval] NOTE: 'cumulative' below is this script's OWN tally of GENUINE completed episodes "
            f"(every real episode termination over the whole {args_cli.num_steps}-step run, attributed "
            f"via task_id snapshotted BEFORE each step -- see code comment). 'windowed' is "
            f"SuccessMonitor's own rate over its last <=100 RESET EVENTS per task -- NOT 100 episodes: "
            f"success_size counts resets (including the one mass reset at environment construction, "
            f"which alone can saturate it to {_SUCCESS_MONITOR_WINDOW_CAP} before any real episode runs) "
            "and is reported for comparison ONLY, never as the gate. The cumulative rate + its Wilson "
            "95% CI is the number to certify against; INSUFFICIENT_DATA below is gated on the "
            "cumulative (genuine-episode) count, never on the reset-event count.",
            flush=True,
        )
        result = {
            "task": args_cli.task, "checkpoint": resume_path, "num_envs": args_cli.num_envs,
            "num_steps": args_cli.num_steps, "min_episodes_per_task": args_cli.min_episodes_per_task,
            "dataset_dir": args_cli.dataset_dir,
            "gate_pos_threshold_m": pos_thr, "gate_rot_threshold_rad": rot_thr,
            "success_monitor_window_cap_reset_events": _SUCCESS_MONITOR_WINDOW_CAP,
            "reset_types": _RESET_TYPES, "per_task": {},
        }
        any_insufficient = False
        for i, (label, rt_name) in enumerate(zip(_RESET_TYPE_LABELS, _RESET_TYPES)):
            n_cum = int(cum_episodes[i].item())
            n_succ = int(cum_successes[i].item())
            n_reset_events = int(reset_event_counts[i])  # RESET-EVENT count, NOT an episode count
            sufficient = n_cum >= args_cli.min_episodes_per_task  # GATED ON CUMULATIVE, not on reset events
            if not sufficient:
                any_insufficient = True

            if sufficient:
                cum_rate = n_succ / n_cum
                ci_lo, ci_hi = _wilson_ci(n_succ, n_cum)
                cum_str = f"{cum_rate:.4f} (95% CI [{ci_lo:.4f}, {ci_hi:.4f}])"
            else:
                cum_rate, ci_lo, ci_hi = None, None, None
                cum_str = "INSUFFICIENT_DATA"
            windowed_flag = (
                f"[WINDOW-CAPPED at {_SUCCESS_MONITOR_WINDOW_CAP} reset events]"
                if n_reset_events >= _SUCCESS_MONITOR_WINDOW_CAP else "[not yet capped]"
            )

            print(
                f"  task_{i} ({label}, {rt_name:32s}): "
                f"cumulative(genuine episodes)={n_succ}/{n_cum}={cum_str}  |  "
                f"windowed(n={n_reset_events} RESET EVENTS, not episodes, {windowed_flag})={rates[i]:.4f}",
                flush=True,
            )
            result["per_task"][f"task_{i}"] = {
                "label": label, "reset_type": rt_name,
                "cumulative_episodes": n_cum, "cumulative_successes": n_succ,
                "cumulative_success_rate": cum_rate, "cumulative_success_rate_wilson_95ci": [ci_lo, ci_hi],
                "windowed_success_rate_over_reset_events": rates[i],
                "n_reset_events_in_rolling_window": n_reset_events,
                "window_cap_reset_events": _SUCCESS_MONITOR_WINDOW_CAP, "sufficient_data": sufficient,
            }
        print("============================================================================\n", flush=True)
        if any_insufficient:
            print(
                "[eval] WARNING: at least one task has fewer than --min_episodes_per_task CUMULATIVE "
                "episodes -- its rate above is INSUFFICIENT_DATA, not a real number. Increase "
                "--num_steps or --num_envs and rerun before trusting a full table.",
                flush=True,
            )

        with open(args_cli.out, "w") as f:
            json.dump(result, f, indent=2, default=float)
        print(f"[eval] wrote {args_cli.out}", flush=True)
        print("EVAL_RUN_COMPLETE", flush=True)  # positive result marker -- see module docstring / caller contract

        env.close()

    _inner()


if __name__ == "__main__":
    main()
    simulation_app.close()
