# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure a checkpoint's TRAINING-TIME GATE PROXY on a frozen policy (``V2_REPOSE_RECIPE.md``
sec 4, bead ``dr-tlx.2``; this is the R0 control of that recipe's sec 6).

=== WHY THIS SCRIPT EXISTS ===

``V2_REPOSE_RECIPE.md``'s open item O6: **the passive-three baseline under ``ep_3600`` in the v2
env has never been measured**, and it is the denominator of every improvement claim the ramp will
make. Until it exists, "the retrain raised the passive-three pass rate to X" is a number against
nothing.

The only comparable figure the project has is F28/F30's S1 gated histogram -- 43/141 = 0.305 -- and
that was measured on ``ep_4300`` (the bore-mixture finetune), not on ``ep_3600``, through the
generator rather than the env, on one rung, n=141. It is a reference point, not a baseline.

=== WHY IT IS NOT A TRAINING JOB, AND NOT certify_pose.py ===

* Not a training job: a baseline must be measured on the FROZEN checkpoint. Handing ``ep_3600`` to
  ``train.py`` for "a few iterations" updates the weights on the first iteration, so the number
  would describe a policy that no longer exists.
* Not ``certify_pose.py``: that script scores the POSE-tracking success predicate at a tolerance
  ladder. This measures a different quantity entirely -- the held-state gate chain's probe-free
  prefix -- and the two answer different questions. Both are needed at R0 (``pass@30mm`` gives A2's
  ``P``; this gives the O6 baseline), and they are deliberately separate runs so neither number can
  be quoted as the other. R7.

=== HOW IT WORKS ===

``mdp.GateProxyLogger`` already maintains cumulative per-branch counters and republishes them into
``env.extras["log"]`` at every reset. So this script does not re-implement any episode bookkeeping:
it rolls the frozen policy out for a fixed step budget and harvests the LATEST value of each
cumulative counter. The counters are monotonic, so the last publication is the whole rollout's
answer.

=== WHAT IT REFUSES TO DO ===

**It fails closed if the ``GateProxy/*`` series never appear.** ``DEXRESET_GATE_PROXY=1`` is an
environment variable, and this repository has four documented cases of a flag that was read by
nothing while every launcher kept exporting it, one of which reached a published model card
(``V2_POSE_FINDINGS.md`` F40/F41, with F42 recording the in-tree documentation going stale). A
measurement script for a metric gated behind such a flag must not be able to report "0 episodes
passed" when the truth is "the metric was never attached". Same rule as
``RESET_SPEC_V2.md`` sec 1a trap 3: read the staged value back out of the run, do not trust the
command line.

Usage (see ``scripts_v2/tools/r0_control.sh``, which sets the whole environment):

    python measure_gate_proxy.py --task <gym id> --checkpoint <rl_games .pth> \\
        --num_envs 256 --steps 4000 --seed 12345 --out r0_gate_proxy.json --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True, help="Gym id, e.g. DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0")
parser.add_argument("--checkpoint", required=True, help="rl_games .pth checkpoint, measured FROZEN")
parser.add_argument("--num_envs", type=int, default=256, help="Parallel environment slots.")
parser.add_argument(
    "--steps",
    type=int,
    default=4000,
    help=(
        "Environment steps to roll out. The budget is in STEPS rather than episodes because the"
        " quantity being measured is published per RESET and the counters are cumulative -- there"
        " is no per-episode bookkeeping here to align a budget to. At the default 240-step episode"
        " length and 256 envs, 4000 steps is roughly 4200 episodes."
    ),
)
parser.add_argument("--seed", type=int, default=12345, help="Evaluation seed. Must not be the training seed.")
parser.add_argument("--out", required=True, help="Path the JSON summary is written to, before shutdown.")
parser.add_argument(
    "--min_episodes",
    type=int,
    default=256,
    help=(
        "Refuse to write a summary built on fewer finished episodes than this. A rollout that"
        " crashed early would otherwise produce a plausible-looking JSON with a wide, unstated"
        " interval, and the recipe's whole point is that this number is a denominator."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.steps <= 0 or args.num_envs <= 0 or args.min_episodes <= 0:
    parser.error("--steps, --num_envs and --min_episodes must be positive")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.utils.assets import retrieve_file_path  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from uwlab_tasks.manager_based.manipulation.dexlift.mdp import gate_proxy_core  # noqa: E402

PREFIX = gate_proxy_core.DEFAULT_LOG_PREFIX
POLICY_DESCRIPTION = "rl_games player, is_deterministic=True (action distribution mean, no exploration noise)"


def unpack_obs(obs):
    """rl_games hands back either a tensor or a dict with an ``obs`` key, depending on the config."""
    return obs["obs"] if isinstance(obs, dict) else obs


def derive_rates(counters: dict[str, float]) -> dict[str, dict]:
    """Turn the cumulative counters into the table ``V2_REPOSE_RECIPE.md`` sec 4.2 is written in.

    Every rate names its denominator explicitly -- ``reached`` for a conditional pass rate,
    ``episodes`` for an unconditional one. F29: a priority-ordered counter whose zeros cannot be
    divided by the number of episodes that reached that gate is uninterpretable, and this project
    has already published one such zero (``seated: 0``) as if it were a measurement.
    """
    episodes = counters.get("episodes", 0.0)
    table: dict[str, dict] = {}
    if episodes <= 0:
        return table
    for gate in gate_proxy_core.PASSIVE_GATE_NAMES:
        reached = counters.get(f"reached_{gate}", 0.0)
        first_fail = counters.get(f"first_fail_{gate}", 0.0)
        table[gate] = {
            "reached": reached,
            "first_fail": first_fail,
            "first_fail_frac_of_all_episodes": first_fail / episodes,
            "first_fail_frac_of_reached": (first_fail / reached) if reached > 0 else None,
        }
    passed = counters.get(gate_proxy_core.PASSIVE_ALL_NAME, 0.0)
    table[gate_proxy_core.PASSIVE_ALL_NAME] = {
        "passed": passed,
        "episodes": episodes,
        "rate": passed / episodes,
        # The two numbers the recipe steers on, carried next to the rate so nobody has to remember
        # them. Sec 4.2: 0.50 is a HARD necessary condition for R2 (accepted states are a subset of
        # passive-three-passing states, so this is assumption-free); 0.71 is the working target and
        # DOES rest on an assumption -- that the probe stage improves in proportion to this one.
        "recipe_hard_floor": 0.50,
        "recipe_working_target": 0.71,
        "clears_hard_floor": (passed / episodes) > 0.50,
    }
    return table


def main() -> None:
    device = args.device if getattr(args, "device", None) else "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.seed = args.seed

    # FAIL BEFORE THE 92-SECOND STARTUP IS SPENT, not after. `_attach_gate_proxy_metric` has already
    # run by now (it is called from the cfg's own __post_init__), so whether the flag was honoured
    # is knowable here -- and a run that cannot produce the metric should not roll out at all.
    if getattr(env_cfg.terminations, "gate_proxy_log", None) is None:
        print(
            "REFUSING: env_cfg.terminations.gate_proxy_log is not attached, so no GateProxy/*"
            " series can ever be published. Set DEXRESET_GATE_PROXY=1 in the environment of THIS"
            " process (it is read at config construction). See V2_POSE_FINDINGS.md F40/F41 for why"
            " this is checked rather than assumed.",
            file=sys.stderr,
            flush=True,
        )
        simulation_app.close()
        raise SystemExit(2)

    agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
    checkpoint = Path(retrieve_file_path(args.checkpoint)).resolve()

    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    # These come from agent_cfg["params"]["env"], NOT ["config"] -- reading them from the wrong
    # subtree silently yields the defaults and builds a differently-clipped observation than the
    # policy was trained against, which corrupts every action without erroring. Copied from
    # certify_pose.py, where the same trap is documented.
    _e = agent_cfg["params"]["env"]
    agent_cfg["params"]["config"]["device"] = device
    agent_cfg["params"]["config"]["device_name"] = device
    env = RlGamesVecEnvWrapper(
        env,
        device,
        _e.get("clip_observations", math.inf),
        _e.get("clip_actions", math.inf),
        _e.get("obs_groups"),
        _e.get("concate_obs_groups", True),
    )
    vecenv.register("IsaacRlgWrapper", lambda n, a, **kw: RlGamesGpuEnv(n, a, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
    agent_cfg["params"]["seed"] = args.seed
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(checkpoint)
    agent_cfg["params"]["config"]["num_actors"] = unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(str(checkpoint))
    agent.reset()

    unwrapped.seed(args.seed)
    obs = unpack_obs(env.reset())
    agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    counters: dict[str, float] = {}
    latest_batch: dict[str, float] = {}
    for step in range(args.steps):
        # ONLY the policy call is inside inference mode -- everything the environment later writes
        # in place must be allocated outside it, or the second seed dies with "Inplace update to
        # inference tensor outside InferenceMode". certify_pose.py's module docstring has the full
        # account; this loop follows the same rule.
        with torch.inference_mode():
            actions = agent.get_action(obs, is_deterministic=True)
        obs, _, _, _ = env.step(actions)
        obs = unpack_obs(obs)
        log = unwrapped.extras.get("log", {})
        for key, value in log.items():
            if not key.startswith(PREFIX):
                continue
            short = key[len(PREFIX) :]
            # Cumulative counters are monotonic, so LAST wins and is the whole rollout's answer.
            # Per-batch fractions are kept only as a last-batch diagnostic and are labelled as such
            # in the JSON -- they are NOT the measurement.
            if short.endswith("_frac") or "_frac/" in short:
                latest_batch[short] = float(value)
            else:
                counters[short] = float(value)
        if (step + 1) % 500 == 0:
            print(
                f"[gate-proxy] step {step + 1}/{args.steps}"
                f" episodes={counters.get('episodes', 0):.0f}"
                f" passive_three={counters.get(gate_proxy_core.PASSIVE_ALL_NAME, 0):.0f}",
                flush=True,
            )

    episodes = counters.get("episodes", 0.0)
    if episodes < args.min_episodes:
        print(
            f"REFUSING to write a summary: {episodes:.0f} finished episodes is below --min_episodes"
            f" {args.min_episodes}. Either the rollout ended early or the metric was not attached"
            " for most of it; either way this number is not a baseline.",
            file=sys.stderr,
            flush=True,
        )
        env.close()
        simulation_app.close()
        raise SystemExit(3)

    summary = {
        "schema": "dexreset.gate_proxy.v1",
        "task": args.task,
        "checkpoint": str(checkpoint),
        "policy": POLICY_DESCRIPTION,
        "seed": args.seed,
        "num_envs": unwrapped.num_envs,
        "steps": args.steps,
        # R5/R7: the staging that produced these numbers, read back from the process's OWN
        # environment rather than restated from the launcher. A gate-proxy number measured under a
        # different tilt or a different mixture is a number about a different task.
        "staging": {
            key: os.environ.get(key)
            for key in (
                "DEXRESET_GATE_PROXY",
                "DEXLIFT_EPISODE_MIXTURE",
                "DEXLIFT_POSE_TILT",
                "DEXLIFT_DROP_Z",
                "DEXLIFT_REF_RESET",
                "DEXLIFT_GOAL_VERTICAL_PROB",
                "DEXRESET_C3_RUNG",
                "DEXRESET_ST_SPAWN_TIPDOWN",
            )
        },
        "mixture_fractions": {
            name: getattr(env_cfg, name, None)
            for name in ("classic_goal_prob", "low_goal_prob", "partial_assembly_prob", "transport_goal_prob")
        },
        "transport_goal": {
            "tilt_rad": getattr(env_cfg, "transport_goal_tilt", None),
            "z_root_m": getattr(env_cfg, "transport_goal_z", None),
        },
        "cumulative_counters": counters,
        "rates": derive_rates(counters),
        "last_batch_fractions_DIAGNOSTIC_ONLY": latest_batch,
        "caveats": [
            "UPPER BOUND, NOT A YIELD (RESET_SPEC_V2.md R7). Only the three probe-free gates are"
            " measured; the three probe gates cannot be evaluated without injecting the probe"
            " action bias, which would perturb training. Accepted states are a SUBSET of these.",
            "Measured in the ENV, not through generate_reset_states_policy.py. Comparable to F28's"
            " histogram in definition (same passive_gates function) but not in path.",
            "A single seed. Widen with --seed on a second run before treating the number as tight.",
        ],
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True))
    rate = summary["rates"].get(gate_proxy_core.PASSIVE_ALL_NAME, {})
    print(
        f"[gate-proxy] RESULT episodes={episodes:.0f}"
        f" passive_three_rate={rate.get('rate', float('nan')):.4f}"
        f" (hard floor 0.50 -> {'CLEARS' if rate.get('clears_hard_floor') else 'BELOW'};"
        f" working target 0.71) -> {args.out}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
