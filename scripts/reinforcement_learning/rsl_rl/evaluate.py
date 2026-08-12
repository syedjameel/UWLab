# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Certify an RSL-RL checkpoint: deterministic policy, held-out seeds, one recorded threshold.

WHAT THIS SCRIPT SCORES. Two kinds of task appear in this repository and they are scored
differently, so the criterion is resolved rather than assumed and is always written into the output:

* ``--success_source termination`` -- the task has a THRESHOLDED success TERMINATION term (the table
  leg grasp-lift task's ``success``); an episode succeeds when that term fires.
* ``--success_source adr`` -- the dexsuite lift/reorient family, which has no such termination. Its
  only thresholded test is the one the ADR curriculum promotes on, ``pos_dist < pos_tol`` (plus
  ``rot_dist < rot_tol`` on the reorient half). The predicate and the tolerances come from
  ``dexlift.mdp.success``, resolved from the LIVE curriculum term of the environment being scored --
  not restated here. See that module for why the predicate lives there and what guards the copy.

  BEWARE THE REWARD TERM NAMED ``success``. On the dexsuite family it is a dense per-step tanh, not
  a thresholded outcome. Certifying against it would produce a number that looks like a success rate
  and is not one, which is why ``--success_source`` never resolves to a reward term.

``auto`` (the default) picks the termination term when the task has one and the ADR predicate
otherwise, and records which it used.

EPISODE SUCCESS IS STICKY-OR WITHIN THE EPISODE: an episode counts as a success once the criterion
holds at any step, even if the object is later dropped. For a success termination that is the same
thing as "the term fired". For the ADR predicate it is a real choice, and it is the protocol the
reference numbers for this task were measured under. Note it differs from what the ADR scheduler
itself does -- the scheduler samples the same predicate once, at the terminal state.

WHY MULTIPLE SEEDS IN ONE PROCESS. Held-out seeds (none of them the training seed) are the point of
a certification, but running them in one process is also what exercises the reset path a second
time. Tensors allocated under ``torch.inference_mode()`` are permanently tagged, and IsaacLab's
reset writes to its buffers in place, so a version of this loop that steps the environment inside
the inference-mode block passes the first seed and dies on the second with "Inplace update to
inference tensor outside InferenceMode is not allowed". Only the policy call is inside the block.

DOMAIN CONDITIONS ARE PART OF THE CLAIM. On a dexsuite ADR task the curriculum owns gravity: the
``variable_gravity`` event is authored at ZERO and ramped to -9.81 only as ADR difficulty rises, so
an evaluation that starts a fresh environment at ``init_difficulty = 0`` scores a lift task in
weightlessness -- and, because the difficulty then climbs during the run, scores later seeds under
different physics from earlier ones. ``--adr_difficulty`` therefore pins the difficulty for the
whole run, and defaults (``auto``) to pinning it at the configured maximum whenever the ADR
predicate is the criterion. The resolved difficulty, the resolved gravity and the observation-noise
state are all recorded in the output.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True)
parser.add_argument("--episodes", type=int, default=1024, help="Episodes PER SEED. Certification uses >= 1024.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument(
    "--seeds",
    type=int,
    nargs="+",
    default=[101, 202, 303, 404],
    help="Held-out evaluation seeds, run in one process. Must not include the training seed.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Single-seed shorthand; replaces --seeds. Kept for older invocations."
)
parser.add_argument(
    "--success_source",
    choices=("auto", "termination", "adr"),
    default="auto",
    help="Where success comes from. 'auto' uses --success_term if the task has it, else the ADR predicate.",
)
parser.add_argument(
    "--success_term", default="success", help="Termination term to score, for --success_source termination."
)
parser.add_argument(
    "--adr_difficulty",
    choices=("auto", "max", "configured"),
    default="auto",
    help=(
        "ADR curriculum difficulty for the whole run. 'max' pins it at the configured maximum (full domain"
        " randomization, real gravity); 'configured' leaves the task config alone, which on a fresh environment"
        " means difficulty 0 and, on dexsuite tasks, zero gravity; 'auto' is 'max' when the ADR predicate is the"
        " success criterion and 'configured' otherwise."
    ),
)
parser.add_argument("--output", type=Path)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.checkpoint is None:
    parser.error("--checkpoint is required")
if args_cli.seed is not None:
    args_cli.seeds = [args_cli.seed]
if len(set(args_cli.seeds)) != len(args_cli.seeds):
    parser.error(f"--seeds must be distinct; got {args_cli.seeds}")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.dexlift.mdp import success as adr_success
from uwlab_tasks.utils.hydra import hydra_task_config

# The inference policy is ``PPO``'s ``act_inference``, i.e. the action distribution's MEAN with no
# exploration noise sampled. Stated in the output so the number is not read as an on-policy average.
POLICY_DESCRIPTION = "rsl_rl inference policy (act_inference: distribution mean, no exploration noise)"


def _wilson(successes: int, episodes: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (p + z * z / (2.0 * episodes)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / episodes + z * z / (4.0 * episodes**2)) / denominator
    return center - radius, center + radius


def _quantiles(values: torch.Tensor) -> dict:
    """Spread of a per-episode diagnostic, and how much of it is finite.

    Reported so an all-zero or all-infinite predicate input cannot hide behind a plausible-looking
    success rate: a healthy untrained run shows a spread of real distances, not a degenerate one.
    """
    finite = values[torch.isfinite(values)]
    summary = {"episodes": int(values.numel()), "finite": int(finite.numel())}
    if finite.numel() == 0:
        return summary
    quantiles = torch.tensor([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0], dtype=torch.float64)
    keys = ["min", "p05", "p25", "median", "p75", "p95", "max"]
    computed = torch.quantile(finite.double(), quantiles)
    summary.update({key: float(value) for key, value in zip(keys, computed)})
    summary["mean"] = float(finite.double().mean())
    return summary


def _resolve_success_source(env_cfg) -> str:
    """Decide what a success IS for this task, before the simulator is built."""
    has_termination = getattr(env_cfg.terminations, args_cli.success_term, None) is not None
    has_adr = getattr(getattr(env_cfg, "curriculum", None), adr_success.ADR_TERM_NAME, None) is not None
    if args_cli.success_source == "termination":
        if not has_termination:
            raise ValueError(
                f"--success_source termination needs a termination term named '{args_cli.success_term}'; this task"
                f" has {sorted(name for name, term in vars(env_cfg.terminations).items() if term is not None)}."
            )
        return "termination"
    if args_cli.success_source == "adr":
        if not has_adr:
            raise ValueError("--success_source adr needs a curriculum term named 'adr'; this task has none.")
        return "adr"
    if has_termination:
        return "termination"
    if has_adr:
        return "adr"
    raise ValueError(
        f"This task has neither a termination term named '{args_cli.success_term}' nor an ADR curriculum, so it"
        " defines no thresholded success test to certify against."
    )


def _pin_adr_difficulty(env_cfg) -> dict:
    """Hold the ADR curriculum at its configured maximum for the whole run.

    Both bounds are moved, not just the initial value: with ``min_difficulty`` left at 0 a failing
    episode would demote and the domain would soften mid-certification.
    """
    term_cfg = getattr(getattr(env_cfg, "curriculum", None), adr_success.ADR_TERM_NAME, None)
    if term_cfg is None:
        raise ValueError(
            f"--adr_difficulty max needs a curriculum term named '{adr_success.ADR_TERM_NAME}'; this task has none."
            " Pass --adr_difficulty configured."
        )
    try:
        max_difficulty = adr_success.adr_param(term_cfg, "max_difficulty")
        adr_success.adr_param(term_cfg, "min_difficulty")
        adr_success.adr_param(term_cfg, "init_difficulty")
    except KeyError as error:
        raise ValueError(
            f"--adr_difficulty max cannot pin the '{adr_success.ADR_TERM_NAME}' term: {error}. Its scheduler"
            f" ({term_cfg.func}) does not use the dexsuite difficulty parameters. Pass --adr_difficulty configured"
            " and state the domain conditions of the run separately."
        ) from error
    term_cfg.params["init_difficulty"] = max_difficulty
    term_cfg.params["min_difficulty"] = max_difficulty
    return {"pinned_at": max_difficulty}


def _difficulty_frac(env) -> float | None:
    term_cfg = getattr(env.curriculum_manager.cfg, adr_success.ADR_TERM_NAME, None)
    frac = getattr(getattr(term_cfg, "func", None), "difficulty_frac", None)
    return None if frac is None else float(frac)


def _domain_record(env) -> dict:
    """The physical conditions the number was measured under, as the running environment has them."""
    record = {"adr_difficulty_frac": _difficulty_frac(env)}
    try:
        gravity_cfg = env.event_manager.get_term_cfg("variable_gravity")
        bounds = gravity_cfg.params["gravity_distribution_params"]
        record["gravity_distribution_params"] = [list(bound) for bound in bounds]
    except (ValueError, KeyError, AttributeError):
        record["gravity_distribution_params"] = None
    record["observation_corruption"] = {
        name: bool(getattr(group, "enable_corruption", False))
        for name, group in vars(env.cfg.observations).items()
        if group is not None and not name.startswith("_")
    }
    return record


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    if args_cli.episodes <= 0 or args_cli.num_envs <= 0:
        raise ValueError("--episodes and --num_envs must be positive")

    success_source = _resolve_success_source(env_cfg)
    adr_difficulty_mode = args_cli.adr_difficulty
    if adr_difficulty_mode == "auto":
        adr_difficulty_mode = "max" if success_source == "adr" else "configured"
    difficulty_record = {"mode": adr_difficulty_mode}
    if adr_difficulty_mode == "max":
        difficulty_record.update(_pin_adr_difficulty(env_cfg))
    if success_source == "adr":
        # Eval-only: the success test has to be sampled BEFORE the environment resets a finished
        # episode, and a termination term is the only in-band hook that runs there. It never
        # terminates anything; see EpisodeSuccessProbe.
        setattr(env_cfg.terminations, adr_success.SUCCESS_PROBE_TERM_NAME, adr_success.success_probe_term_cfg())

    # ``update_rsl_rl_cfg`` reads ``args_cli.seed``; the loop below re-seeds per block anyway.
    args_cli.seed = args_cli.seeds[0]
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli))
    env_cfg.scene.num_envs = min(args_cli.num_envs, args_cli.episodes)
    env_cfg.seed = args_cli.seeds[0]
    agent_cfg.seed = args_cli.seeds[0]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    checkpoint = Path(retrieve_file_path(args_cli.checkpoint)).resolve()
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy

    unwrapped = env.unwrapped
    device = unwrapped.device
    probe = None
    if success_source == "adr":
        probe = unwrapped.termination_manager.get_term_cfg(adr_success.SUCCESS_PROBE_TERM_NAME).func

        def success_now() -> torch.Tensor:
            return probe.success

    else:

        def success_now() -> torch.Tensor:
            return unwrapped.termination_manager.get_term(args_cli.success_term)

    # The probe is a measurement, not an outcome: it must not appear as a way an episode ended.
    terminal_names = [
        name for name in unwrapped.termination_manager.active_terms if name != adr_success.SUCCESS_PROBE_TERM_NAME
    ]

    per_seed = []
    all_dones = torch.ones(env.num_envs, device=device, dtype=torch.bool)
    # Progress goes to stderr, never stdout: stdout carries the JSON and nothing else. A
    # certification run is tens of minutes of otherwise silent stepping, and on this stack a silent
    # process is indistinguishable from one hung in ``app.close()``.
    stepped = 0
    for seed in args_cli.seeds:
        unwrapped.seed(seed)
        obs, _ = env.reset()
        policy_nn.reset(all_dones)

        completed = 0
        successes = 0
        episode_steps = 0
        terminal_counts = {name: 0 for name in terminal_names}
        min_pos_dists = []
        while completed < args_cli.episodes:
            batch_size = min(env.num_envs, args_cli.episodes - completed)
            active = torch.arange(env.num_envs, device=device) < batch_size
            finished = torch.zeros(env.num_envs, device=device, dtype=torch.bool)
            reached = torch.zeros(env.num_envs, device=device, dtype=torch.bool)
            steps = torch.zeros(env.num_envs, device=device, dtype=torch.long)
            closest = torch.full((env.num_envs,), float("inf"), device=device)
            while not bool(finished[active].all()):
                with torch.inference_mode():
                    actions = policy(obs)
                # Keep simulator state out of inference mode.  Otherwise tensors created by
                # ``env.step`` cannot be updated in-place by the reset between batches and seeds.
                obs, _, dones, _ = env.step(actions)
                steps[active & ~finished] += 1
                # Sticky-OR, and gated on ``~finished`` so a slot that has already produced its
                # outcome cannot contribute a second one from the episode it is now running.
                # Both are read after ``env.step`` but were computed before its internal reset.
                reached |= success_now() & ~finished
                if probe is not None:
                    closest = torch.where(finished, closest, torch.minimum(closest, probe.pos_dist))
                done = dones.bool()
                newly_finished = done & active & ~finished
                successes += int(reached[newly_finished].sum().item())
                for name in terminal_names:
                    term = unwrapped.termination_manager.get_term(name)
                    terminal_counts[name] += int(term[newly_finished].sum().item())
                episode_steps += int(steps[newly_finished].sum().item())
                if probe is not None:
                    min_pos_dists.append(closest[newly_finished].detach().float().cpu())
                finished |= newly_finished
                policy_nn.reset(dones)
                stepped += 1
                if stepped % 200 == 0:
                    print(
                        f"[certify] seed {seed}: {completed}/{args_cli.episodes} episodes done,"
                        f" {int(finished.sum().item())}/{batch_size} in flight finished,"
                        f" {stepped} steps",
                        file=sys.stderr,
                        flush=True,
                    )
            completed += batch_size
            if completed < args_cli.episodes:
                obs, _ = env.reset()
                policy_nn.reset(all_dones)

        low, high = _wilson(successes, completed)
        block = {
            "seed": seed,
            "episodes": completed,
            "successes": successes,
            "success_rate": successes / completed,
            "wilson_95": [low, high],
            "terminal_counts": terminal_counts,
            "mean_episode_steps": episode_steps / completed,
            "adr_difficulty_frac_after": _difficulty_frac(unwrapped),
        }
        if probe is not None:
            block["episode_min_pos_dist_m"] = _quantiles(torch.cat(min_pos_dists))
        per_seed.append(block)
        print(
            f"[certify] seed {seed}: {successes}/{completed} = {successes / completed:.4f}",
            file=sys.stderr,
            flush=True,
        )

    pooled_episodes = sum(block["episodes"] for block in per_seed)
    pooled_successes = sum(block["successes"] for block in per_seed)
    pooled_low, pooled_high = _wilson(pooled_successes, pooled_episodes)
    result = {
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "protocol": {
            "outcomes": "one_terminal_outcome_per_environment_slot_per_batch",
            "episode_success": "sticky_or_within_episode",
            "policy": POLICY_DESCRIPTION,
            "seeds": list(args_cli.seeds),
            "episodes_per_seed": args_cli.episodes,
            "parallel_envs": env.num_envs,
        },
        "success_criterion": _criterion_record(success_source, probe),
        "domain": {**_domain_record(unwrapped), "adr_difficulty": difficulty_record},
        "per_seed": per_seed,
        "pooled": {
            "episodes": pooled_episodes,
            "successes": pooled_successes,
            "success_rate": pooled_successes / pooled_episodes,
            "wilson_95": [pooled_low, pooled_high],
            "terminal_counts": {
                name: sum(block["terminal_counts"][name] for block in per_seed) for name in terminal_names
            },
            "mean_episode_steps": sum(
                block["mean_episode_steps"] * block["episodes"] for block in per_seed
            )
            / pooled_episodes,
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args_cli.output is not None:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(encoded + "\n", encoding="utf-8")
    env.close()


def _criterion_record(success_source: str, probe) -> dict:
    """Exactly what was certified, so a reported rate carries its own threshold."""
    if success_source == "termination":
        return {
            "source": "termination_term",
            "term": args_cli.success_term,
            "rule": f"termination term '{args_cli.success_term}' fired",
        }
    spec = probe.spec
    return {
        "source": "adr_curriculum_predicate",
        "term": spec.term_name,
        "rule": spec.rule,
        "pos_tol_m": spec.pos_tol,
        "rot_tol_rad": spec.rot_tol,
        "scheduler_class": spec.scheduler_class,
        "asset": spec.asset_cfg.name,
        "object": spec.object_cfg.name,
        "command": spec.command_name,
        "upstream_predicate_fingerprint": adr_success.upstream_predicate_fingerprint(),
    }


if __name__ == "__main__":
    main()
    simulation_app.close()
