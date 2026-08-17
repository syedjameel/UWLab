# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Certify an rl_games checkpoint of the UR5e+DELTO dexlift family: how close does the object get?

This is the rl_games twin of ``scripts/reinforcement_learning/rsl_rl/evaluate.py``. Same protocol,
same JSON shape, same Wilson interval; the only differences are the trainer-side plumbing (an
rl_games ``BasePlayer`` instead of an rsl_rl ``OnPolicyRunner``) and one addition -- the whole
tolerance ladder is scored in a single pass instead of one tolerance per run.

WHAT IT SCORES. The dexlift lift/reorient family has no success TERMINATION term. Its only
thresholded test is the one the ADR curriculum promotes on, ``pos_dist < pos_tol`` (plus
``rot_dist < rot_tol`` when there is one), and that predicate lives in ``dexlift.mdp.success``
rather than being restated here. Beware the REWARD term named ``success``: it is a dense per-step
tanh with no threshold, and a number computed from it would look like a success rate without being
one. Nothing here reads it.

EPISODE SUCCESS IS STICKY-OR: an episode counts once the predicate holds at ANY step, even if the
object is dropped afterwards. That is the protocol the reference numbers for this task were measured
under, and it is why the per-episode MINIMUM position error is the headline diagnostic -- with the
orientation gate off (Lift sets ``rot_tol = None``), "sticky-OR success at 1 cm" and "minimum
position error < 1 cm" are the SAME event, so publishing the distribution of minima makes the
reported pass fraction auditable instead of asserted. Where a rotation gate IS active the two stop
being the same event -- the position minimum and the rotation minimum can occur at different steps
-- so the pass fractions below are always accumulated from the per-step conjunction, never
reconstructed from the two minima.

THE EPISODE BUDGET IS FIXED BEFORE THE RUN AND NEVER CONSULTS AN OUTCOME. ``--episodes`` is divided
across ``--seeds`` up front and every seed runs exactly that many episodes. There is no
"stop when the interval is tight enough", no "stop after N successes", no discarding of an episode
for what it did: a stopping rule that depends on the outcome makes ``successes/episodes`` something
other than a success rate, and that exact bias has already invalidated results in this project.
Within a batch each environment slot contributes exactly ONE episode, chosen by position (the first
one it finishes), never by what happened in it.

DOMAIN CONDITIONS ARE PART OF THE CLAIM. The ADR curriculum owns gravity here: ``variable_gravity``
is authored at ZERO and ramped to -9.81 only as difficulty rises, so a fresh environment left at
``init_difficulty = 0`` would score a LIFT task in weightlessness and would then drift as the
curriculum promoted mid-run. ``--adr_difficulty max`` (the default) pins the difficulty at the
configured maximum for the whole run, so the policy is scored at full domain randomization and full
gravity. The resolved difficulty, gravity bounds and observation-noise state are recorded in the
output.

INFERENCE-MODE BOUNDARIES, and why the comment is here. Tensors allocated inside
``torch.inference_mode()`` are permanently tagged, and IsaacLab's reset writes to its buffers IN
PLACE, so a loop that steps the environment inside the block passes the first seed and dies on the
second with "Inplace update to inference tensor outside InferenceMode is not allowed". A single-seed
smoke test cannot see that. The rule obeyed below: ONLY the policy call (and the RNN-state zeroing
that mutates tensors the policy call produced) is inside the block; ``env.reset`` and ``env.step``
are always outside it, so nothing the environment will later write in place was ever allocated
under inference mode. Passing the resulting actions OUT of the block is safe -- the rl_games
wrapper's ``step`` does ``actions.detach().clone()`` before touching the simulator.

AGGREGATES ARE COMPUTED ONCE, FROM ``per_episode``. Every number in the summary -- per-seed,
pooled, per-tolerance, quantiles -- is derived by :func:`summarize` from the same serialized list,
so no aggregate can disagree with its own confidence interval. A previous harness in this project
rebound its loop variables and wrote a summary describing only the last block while its interval
described the total; deriving everything from the parts, and shipping the parts, removes that class
of defect rather than re-checking for it.

THE JSON IS WRITTEN BEFORE THE APP IS CLOSED. Isaac's ``simulation_app.close()`` hangs routinely on
this stack; a missing completion line means "hung", not "crashed". The result file and the RESULT
line are therefore both produced before anything is torn down.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# Tolerances every run reports, in metres. Scoring the ladder costs nothing -- the predicate is a
# comparison on a distance that has already been computed -- and it turns a pass/fail claim into a
# curve, which is what tells "just over the line" apart from "comfortably inside it".
TOLERANCE_LADDER_M = (0.05, 0.02, 0.01, 0.005)


def _seed_list(text: str) -> list[int]:
    seeds = [int(part) for part in text.replace(" ", "").split(",") if part]
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds needs at least one integer")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(f"--seeds must be distinct; got {seeds}")
    return seeds


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True, help="Gym id, e.g. DexLift-UR5eDelto-RelJointPos-Lift-Play-v0")
parser.add_argument("--checkpoint", required=True, help="rl_games .pth checkpoint")
parser.add_argument("--num_envs", type=int, default=256, help="Parallel environment slots.")
parser.add_argument(
    "--episodes",
    type=int,
    default=4096,
    help="TOTAL episodes over all seeds, fixed before the run and split evenly across --seeds.",
)
parser.add_argument(
    "--seeds",
    type=_seed_list,
    default=[101, 202, 303, 404],
    help="Comma-separated held-out evaluation seeds, run in one process. Must not include the training seed.",
)
parser.add_argument(
    "--pos_tol",
    type=float,
    default=0.01,
    help="Headline position tolerance in metres. Added to the reported ladder; does NOT replace it.",
)
parser.add_argument(
    "--rot_tol",
    type=float,
    default=None,
    help="Orientation tolerance in radians. Omitted: use the environment's own (None on Lift, i.e. no gate).",
)
parser.add_argument("--out", required=True, help="Path the JSON summary is written to, before shutdown.")
parser.add_argument(
    "--adr_difficulty",
    choices=("max", "configured"),
    default="max",
    help=(
        "'max' pins the ADR curriculum at its configured maximum for the whole run (full domain randomization,"
        " full gravity); 'configured' leaves the task config alone, which on a fresh environment means difficulty"
        " 0 and, on this family, ZERO GRAVITY. Use 'configured' only to reproduce a difficulty-0 measurement."
    ),
)
parser.add_argument(
    "--gpu_collision_stack_size",
    type=int,
    default=None,
    help="Override PhysX gpu_collision_stack_size. Omitted: whatever the task config declares.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# A certification never needs a viewer or a sensor, and both change what the run costs.
args.headless = True
args.enable_cameras = False
if args.episodes <= 0 or args.num_envs <= 0:
    parser.error("--episodes and --num_envs must be positive")
if args.pos_tol <= 0.0:
    parser.error("--pos_tol must be a positive distance in metres")
if args.rot_tol is not None and args.rot_tol <= 0.0:
    # 0.0 would not mean "exact": within_success_tolerance tests rot_tol for TRUTHINESS, so 0.0
    # silently DROPS the orientation gate instead of tightening it. Refuse the ambiguity.
    parser.error("--rot_tol must be a positive angle in radians; omit it to use the environment's own")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.utils.assets import retrieve_file_path  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from uwlab_tasks.manager_based.manipulation.dexlift.mdp import success as adr_success  # noqa: E402

# The scored action is the action distribution's MEAN (rl_games' player returns ``mus`` when
# ``is_deterministic``), with no exploration noise. Stated in the output so the number is not read
# as an on-policy average.
POLICY_DESCRIPTION = "rl_games player, is_deterministic=True (action distribution mean, no exploration noise)"
Z_95 = 1.959963984540054


def wilson(successes: int, episodes: int, z: float = Z_95) -> list[float]:
    """Wilson score interval -- the same one the rsl_rl harness reports, so the two are comparable."""
    if episodes <= 0:
        return [float("nan"), float("nan")]
    p = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (p + z * z / (2.0 * episodes)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / episodes + z * z / (4.0 * episodes**2)) / denominator
    return [center - radius, center + radius]


def quantiles(values: list[float]) -> dict:
    """Spread of a per-episode diagnostic, and how much of it is finite.

    ``finite`` is reported because a degenerate input -- all zeros, all infinities, a NaN that
    poisoned an episode's running minimum -- would otherwise hide behind a plausible-looking
    pass fraction.
    """
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    summary = {"episodes": int(array.size), "finite": int(finite.size)}
    if finite.size == 0:
        return summary
    for key, q in (("min", 0.0), ("p50", 50.0), ("p90", 90.0), ("p99", 99.0)):
        summary[key] = float(np.percentile(finite, q))
    summary["max"] = float(finite.max())
    summary["mean"] = float(finite.mean())
    return summary


def resolve_tolerances(env_pos_tol: float) -> list[float]:
    """The ladder actually scored: the published rungs, the CLI headline, and the env's own.

    The environment's own tolerance is included so the JSON always carries the number the ADR
    curriculum promotes on, next to the number the deliverable asks about.
    """
    values = set(TOLERANCE_LADDER_M) | {float(args.pos_tol), float(env_pos_tol)}
    return sorted(values, reverse=True)


def pin_adr_difficulty(env_cfg) -> dict:
    """Hold the ADR curriculum at its configured maximum for the whole run.

    BOTH bounds move, not just the initial value: with ``min_difficulty`` left at 0 a failing
    episode would demote the curriculum and the domain would soften part-way through the
    certification, so later episodes would be scored under easier physics than earlier ones.
    """
    term_cfg = getattr(getattr(env_cfg, "curriculum", None), adr_success.ADR_TERM_NAME, None)
    if term_cfg is None:
        raise ValueError(
            f"--adr_difficulty max needs a curriculum term named '{adr_success.ADR_TERM_NAME}'; this task has none."
            " Pass --adr_difficulty configured and state the domain conditions of the run separately."
        )
    try:
        max_difficulty = adr_success.adr_param(term_cfg, "max_difficulty")
        adr_success.adr_param(term_cfg, "min_difficulty")
        adr_success.adr_param(term_cfg, "init_difficulty")
    except KeyError as error:
        raise ValueError(
            f"--adr_difficulty max cannot pin the '{adr_success.ADR_TERM_NAME}' term: {error}. Its scheduler"
            f" ({term_cfg.func}) does not use the dexsuite difficulty parameters."
        ) from error
    term_cfg.params["init_difficulty"] = max_difficulty
    term_cfg.params["min_difficulty"] = max_difficulty
    return {"mode": "max", "pinned_at": max_difficulty}


def difficulty_frac(unwrapped) -> float | None:
    term_cfg = getattr(unwrapped.curriculum_manager.cfg, adr_success.ADR_TERM_NAME, None)
    frac = getattr(getattr(term_cfg, "func", None), "difficulty_frac", None)
    return None if frac is None else float(frac)


def domain_record(unwrapped) -> dict:
    """The physical conditions the number was measured under, as the RUNNING environment has them."""
    record = {"adr_difficulty_frac": difficulty_frac(unwrapped)}
    try:
        gravity_cfg = unwrapped.event_manager.get_term_cfg("variable_gravity")
        bounds = gravity_cfg.params["gravity_distribution_params"]
        record["gravity_distribution_params"] = [list(bound) for bound in bounds]
    except (ValueError, KeyError, AttributeError):
        record["gravity_distribution_params"] = None
    record["observation_corruption"] = {
        name: bool(getattr(group, "enable_corruption", False))
        for name, group in vars(unwrapped.cfg.observations).items()
        if group is not None and not name.startswith("_")
    }
    return record


def plant_record(unwrapped) -> dict:
    """WHICH ROBOT was measured.

    This task family selects its colliders, its self-collision setting and its actuator set from
    ``DEXLIFT_*`` environment variables, so two runs of the same command line on the same checkpoint
    can measure different plants. A certification that does not name its plant is a number that
    cannot be reproduced -- and probing a policy in a plant it was not trained in says nothing about
    the policy. The USD path is recorded next to the flags because it is the thing they resolve to.
    """
    try:
        usd_path = unwrapped.scene["robot"].cfg.spawn.usd_path
    except (KeyError, AttributeError):
        usd_path = None
    return {
        "robot_usd": usd_path,
        "dexlift_env": {key: value for key, value in sorted(os.environ.items()) if key.startswith("DEXLIFT_")},
    }


def unpack_obs(obs):
    """rl_games hands back either a tensor or a dict with an ``obs`` key, depending on the config."""
    return obs["obs"] if isinstance(obs, dict) else obs


def tolerance_key(pos_tol: float, rot_tol: float | None, position_only: bool) -> str:
    """The log key a rate at THESE tolerances is allowed to carry.

    Never a hardcoded string: ``episode_success_rate_keys`` reserves the certified name for the
    certified predicate and stamps every other tolerance into its own name, so a 1 cm number can no
    longer be published under the 5 cm key.
    """
    return adr_success.episode_success_rate_keys(pos_tol, rot_tol, position_only)[0]


def summarize(per_episode: list[dict], tolerances: list[float], keys: dict[float, str], env_pos_tol: float) -> dict:
    """Every aggregate this script reports, derived in ONE place from the serialized per-episode rows.

    Nothing above this function accumulates a running total that ends up in the output. If a
    consumer disagrees with a number here, they can recompute it from ``per_episode`` and the two
    can only differ by this function being wrong -- not by two accumulators having drifted apart.
    """
    seeds = sorted({row["seed"] for row in per_episode})
    per_tolerance = []
    for tol in tolerances:
        label = f"{tol:g}"
        hits = [bool(row["success"][label]) for row in per_episode]
        blocks = []
        for seed in seeds:
            seed_hits = [bool(row["success"][label]) for row in per_episode if row["seed"] == seed]
            blocks.append({
                "seed": seed,
                "episodes": len(seed_hits),
                "successes": int(sum(seed_hits)),
                "pass_fraction": (sum(seed_hits) / len(seed_hits)) if seed_hits else float("nan"),
                "wilson_95": wilson(int(sum(seed_hits)), len(seed_hits)),
            })
        per_tolerance.append({
            "pos_tol_m": tol,
            "log_key": keys[tol],
            "episodes": len(hits),
            "successes": int(sum(hits)),
            "pass_fraction": (sum(hits) / len(hits)) if hits else float("nan"),
            "wilson_95": wilson(int(sum(hits)), len(hits)),
            "is_headline": tol == float(args.pos_tol),
            "is_environment_native": tol == float(env_pos_tol),
            "per_seed": blocks,
        })

    terminal_counts: dict[str, int] = {}
    for row in per_episode:
        for name in row["terminal_terms"]:
            terminal_counts[name] = terminal_counts.get(name, 0) + 1

    return {
        "episodes": len(per_episode),
        "per_tolerance": per_tolerance,
        "min_pos_error_m": quantiles([row["min_pos_err_m"] for row in per_episode]),
        "min_rot_error_rad": quantiles([row["min_rot_err_rad"] for row in per_episode]),
        "terminal_pos_error_m": quantiles([row["terminal_pos_err_m"] for row in per_episode]),
        "terminal_rot_error_rad": quantiles([row["terminal_rot_err_rad"] for row in per_episode]),
        "mean_episode_steps": float(np.mean([row["steps"] for row in per_episode])) if per_episode else float("nan"),
        "terminal_counts": terminal_counts,
        "per_seed_episodes": {str(seed): sum(1 for row in per_episode if row["seed"] == seed) for seed in seeds},
    }


def main() -> None:
    seeds = list(args.seeds)
    # THE BUDGET IS DECIDED HERE, before a single step runs, and nothing below can change it.
    episodes_per_seed = math.ceil(args.episodes / len(seeds))
    episodes_total = episodes_per_seed * len(seeds)
    if episodes_total != args.episodes:
        print(
            f"[certify] --episodes {args.episodes} is not divisible by {len(seeds)} seeds;"
            f" rounding UP to {episodes_per_seed} per seed = {episodes_total} total.",
            file=sys.stderr,
            flush=True,
        )

    device = args.device if getattr(args, "device", None) else "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=min(args.num_envs, episodes_per_seed))
    if args.gpu_collision_stack_size is not None:
        env_cfg.sim.physx.gpu_collision_stack_size = args.gpu_collision_stack_size
    env_cfg.seed = seeds[0]

    difficulty_record = {"mode": "configured"}
    if args.adr_difficulty == "max":
        difficulty_record = pin_adr_difficulty(env_cfg)

    # EVAL-ONLY TERM. The success test must be sampled BEFORE the environment resets a finished
    # episode -- ``ManagerBasedRLEnv.step`` resets terminated envs before it returns, so by the time
    # an observation comes back the object and goal of a finished episode are already the NEXT
    # one's. The termination manager runs before the reset and does not clear its per-term buffers,
    # which makes a termination term the only in-band hook with the right timing. This one never
    # terminates anything; see EpisodeSuccessProbe.
    setattr(env_cfg.terminations, adr_success.SUCCESS_PROBE_TERM_NAME, adr_success.success_probe_term_cfg())

    agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
    checkpoint = Path(retrieve_file_path(args.checkpoint)).resolve()

    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    # These five come from agent_cfg["params"]["env"], NOT ["config"] -- reading them from the wrong
    # subtree silently yields the defaults and builds a differently-clipped observation than the
    # policy was trained against, which would corrupt every action without erroring.
    _e = agent_cfg["params"]["env"]
    # Keep the agent on the same device the simulation is on; otherwise the wrapper ferries
    # observations to a device the environment is not using.
    agent_cfg["params"]["config"]["device"] = device
    agent_cfg["params"]["config"]["device_name"] = device
    env = RlGamesVecEnvWrapper(
        env,
        agent_cfg["params"]["config"]["device"],
        _e.get("clip_observations", math.inf),
        _e.get("clip_actions", math.inf),
        _e.get("obs_groups"),
        _e.get("concate_obs_groups", True),
    )
    vecenv.register("IsaacRlgWrapper", lambda n, a, **kw: RlGamesGpuEnv(n, a, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
    agent_cfg["params"]["seed"] = seeds[0]
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(checkpoint)
    agent_cfg["params"]["config"]["num_actors"] = unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(str(checkpoint))
    agent.reset()

    sim_device = unwrapped.device
    num_envs = unwrapped.num_envs
    probe = unwrapped.termination_manager.get_term_cfg(adr_success.SUCCESS_PROBE_TERM_NAME).func
    spec = probe.spec
    rot_tol = float(args.rot_tol) if args.rot_tol is not None else spec.rot_tol
    tolerances = resolve_tolerances(spec.pos_tol)
    command_cfg = getattr(unwrapped.command_manager.cfg, spec.command_name, None)
    position_only = bool(command_cfg is not None and getattr(command_cfg, "position_only", False))
    keys = {tol: tolerance_key(tol, rot_tol, position_only) for tol in tolerances}
    labels = {tol: f"{tol:g}" for tol in tolerances}
    # The probe is a MEASUREMENT, not an outcome: it must never appear as a way an episode ended.
    terminal_names = [
        name for name in unwrapped.termination_manager.active_terms if name != adr_success.SUCCESS_PROBE_TERM_NAME
    ]
    # A slot that never terminates would spin here forever; every episode in this family ends at
    # ``time_out`` at the latest, so anything past a few horizons is a broken environment, not a
    # long episode. Failing loudly beats emitting a silently truncated sample.
    step_ceiling = int(math.ceil(float(unwrapped.max_episode_length))) * 4

    print(
        f"[certify] task={args.task} envs={num_envs} seeds={seeds} episodes/seed={episodes_per_seed}"
        f" rule='{spec.rule}' scored_pos_tols={tolerances} scored_rot_tol={rot_tol}",
        file=sys.stderr,
        flush=True,
    )

    def reset_env():
        """Full reset, always OUTSIDE inference mode -- see the module docstring."""
        obs = unpack_obs(env.reset())
        agent.get_batch_size(obs, 1)
        if agent.is_rnn:
            agent.init_rnn()
        return obs

    def zero_rnn_states(done: torch.Tensor) -> None:
        """rl_games' own done-handling, in the only place it is legal to do it.

        The states were produced by the policy call inside ``inference_mode``, so they are inference
        tensors and can only be written in place from inside such a block.
        """
        if not agent.is_rnn or agent.states is None:
            return
        indices = done.nonzero(as_tuple=False)
        if indices.numel() == 0:
            return
        with torch.inference_mode():
            for state in agent.states:
                state[:, indices, :] = state[:, indices, :] * 0.0

    per_episode: list[dict] = []
    stepped = 0
    for seed in seeds:
        unwrapped.seed(seed)
        obs = reset_env()
        completed = 0
        while completed < episodes_per_seed:
            batch = min(num_envs, episodes_per_seed - completed)
            active = torch.arange(num_envs, device=sim_device) < batch
            finished = torch.zeros(num_envs, dtype=torch.bool, device=sim_device)
            steps = torch.zeros(num_envs, dtype=torch.long, device=sim_device)
            min_pos = torch.full((num_envs,), float("inf"), device=sim_device)
            min_rot = torch.full((num_envs,), float("inf"), device=sim_device)
            terminal_pos = torch.full((num_envs,), float("nan"), device=sim_device)
            terminal_rot = torch.full((num_envs,), float("nan"), device=sim_device)
            sticky = {tol: torch.zeros(num_envs, dtype=torch.bool, device=sim_device) for tol in tolerances}
            term_flags = {
                name: torch.zeros(num_envs, dtype=torch.bool, device=sim_device) for name in terminal_names
            }
            inner = 0
            while not bool(finished[active].all()):
                # ONLY the policy call is inside inference mode. Everything the environment will
                # later write in place is allocated outside it.
                with torch.inference_mode():
                    actions = agent.get_action(obs, is_deterministic=True)
                obs, _, dones, _ = env.step(actions)
                obs = unpack_obs(obs)
                inner += 1
                stepped += 1
                if inner > step_ceiling:
                    raise RuntimeError(
                        f"seed {seed}: {int(finished[active].sum())}/{batch} slots finished after {inner} steps"
                        f" (> {step_ceiling}). An episode in this family always ends at time_out; a slot that"
                        " does not is a broken environment, and continuing would sample it non-uniformly."
                    )

                # ``probe.pos_dist`` / ``probe.rot_dist`` were computed by the termination manager
                # during this step, i.e. BEFORE the internal reset -- so on the step a slot finishes
                # they are that episode's terminal values, not the next episode's first ones.
                pos_dist = probe.pos_dist
                rot_dist = probe.rot_dist
                live = active & ~finished
                steps[live] += 1
                for tol in tolerances:
                    # STICKY OR of the FULL predicate, evaluated per step. Not reconstructed from
                    # the minima: with a rotation gate active the position minimum and the rotation
                    # minimum need not happen at the same step, and their conjunction would count
                    # successes that never occurred.
                    hit = adr_success.within_success_tolerance(pos_dist, rot_dist, tol, rot_tol)
                    sticky[tol] |= hit & live
                min_pos = torch.where(live, torch.minimum(min_pos, pos_dist), min_pos)
                min_rot = torch.where(live, torch.minimum(min_rot, rot_dist), min_rot)

                done = dones.to(sim_device).bool()
                newly = done & live
                terminal_pos = torch.where(newly, pos_dist, terminal_pos)
                terminal_rot = torch.where(newly, rot_dist, terminal_rot)
                for name in terminal_names:
                    term_flags[name] |= unwrapped.termination_manager.get_term(name) & newly
                finished |= newly
                zero_rnn_states(done)
                if stepped % 200 == 0:
                    # Progress on stderr: stdout carries the summary and nothing else, and on this
                    # stack a silent process is indistinguishable from one hung in app.close().
                    print(
                        f"[certify] seed {seed}: {completed}/{episodes_per_seed} episodes,"
                        f" {int(finished[active].sum())}/{batch} in flight finished, {stepped} steps",
                        file=sys.stderr,
                        flush=True,
                    )

            sticky_cpu = {tol: sticky[tol].cpu().tolist() for tol in tolerances}
            flags_cpu = {name: term_flags[name].cpu().tolist() for name in terminal_names}
            steps_cpu = steps.cpu().tolist()
            min_pos_cpu = min_pos.cpu().tolist()
            min_rot_cpu = min_rot.cpu().tolist()
            terminal_pos_cpu = terminal_pos.cpu().tolist()
            terminal_rot_cpu = terminal_rot.cpu().tolist()
            for slot in range(batch):
                per_episode.append({
                    "seed": seed,
                    "env_slot": slot,
                    "steps": int(steps_cpu[slot]),
                    "min_pos_err_m": float(min_pos_cpu[slot]),
                    "min_rot_err_rad": float(min_rot_cpu[slot]),
                    "terminal_pos_err_m": float(terminal_pos_cpu[slot]),
                    "terminal_rot_err_rad": float(terminal_rot_cpu[slot]),
                    "success": {labels[tol]: bool(sticky_cpu[tol][slot]) for tol in tolerances},
                    "terminal_terms": [name for name in terminal_names if flags_cpu[name][slot]],
                })
            completed += batch
            if completed < episodes_per_seed:
                obs = reset_env()
        done_here = sum(1 for row in per_episode if row["seed"] == seed)
        print(f"[certify] seed {seed}: {done_here} episodes recorded", file=sys.stderr, flush=True)

    summary = summarize(per_episode, tolerances, keys, spec.pos_tol)
    result = {
        "task": args.task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "protocol": {
            "trainer": "rl_games",
            "policy": POLICY_DESCRIPTION,
            "episode_success": "sticky_or_within_episode",
            "outcomes": "one_terminal_outcome_per_environment_slot_per_batch",
            "stopping_rule": "fixed episode budget decided before the run; no outcome-dependent stopping",
            "seeds": seeds,
            "episodes_requested_total": args.episodes,
            "episodes_per_seed": episodes_per_seed,
            "episodes_total": episodes_total,
            "parallel_envs": num_envs,
        },
        "success_criterion": {
            "source": "adr_curriculum_predicate",
            "term": spec.term_name,
            "rule_as_scored": (
                "pos_dist < <tolerance> at any step, i.e. min-over-episode pos_dist < <tolerance>"
                if not rot_tol
                else f"(pos_dist < <tolerance>) and (rot_dist < {rot_tol}) at the SAME step, at any step"
            ),
            "environment_rule": spec.rule,
            "environment_pos_tol_m": spec.pos_tol,
            "environment_rot_tol_rad": spec.rot_tol,
            "scored_pos_tol_m": float(args.pos_tol),
            "scored_rot_tol_rad": rot_tol,
            "rot_tol_source": "cli" if args.rot_tol is not None else "environment",
            "scored_tolerance_ladder_m": tolerances,
            "scheduler_class": spec.scheduler_class,
            "asset": spec.asset_cfg.name,
            "object": spec.object_cfg.name,
            "command": spec.command_name,
            "command_position_only": position_only,
            "upstream_predicate_fingerprint": adr_success.upstream_predicate_fingerprint(),
        },
        "domain": {**domain_record(unwrapped), "adr_difficulty": difficulty_record},
        "plant": plant_record(unwrapped),
        "summary": summary,
        "per_episode": per_episode,
    }

    # WRITTEN BEFORE ANYTHING IS TORN DOWN: env.close() and simulation_app.close() both hang
    # routinely on this stack, and a result that only exists after them is a result that can be lost.
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    headline = next(block for block in summary["per_tolerance"] if block["is_headline"])
    ladder = " ".join(
        f"pass@{block['pos_tol_m'] * 1000:g}mm={block['pass_fraction']:.4f}" for block in summary["per_tolerance"]
    )
    mins = summary["min_pos_error_m"]
    print(
        f"CERTIFY_RESULT task={args.task} episodes={summary['episodes']} seeds={','.join(str(s) for s in seeds)}"
        f" {ladder}"
        f" headline@{headline['pos_tol_m'] * 1000:g}mm={headline['pass_fraction']:.4f}"
        f" wilson95=[{headline['wilson_95'][0]:.4f},{headline['wilson_95'][1]:.4f}]"
        f" min_pos_p50={mins.get('p50', float('nan')):.4f} p90={mins.get('p90', float('nan')):.4f}"
        f" p99={mins.get('p99', float('nan')):.4f} max={mins.get('max', float('nan')):.4f}"
        f" adr_difficulty_frac={result['domain']['adr_difficulty_frac']} json={out_path}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
