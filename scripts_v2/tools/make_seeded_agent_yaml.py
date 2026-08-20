#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write a copy of an rl_games agent.yaml with params.seed overridden (bead UWLab-weyl, chunking
diversity fix).

WHY THIS EXISTS. Chunking a paper-scale generate_reset_states_policy.py run into several
independent Isaac processes (to avoid the generator's quadratic accept-time-bank write cost, and
to bound a crash's blast radius -- see merge_c2_chunks.py's own docstring) does nothing for
DIVERSITY if every chunk launches with an identical command line. Traced (team-lead): rl_games'
torch_runner.load_config reads agent_cfg["params"]["seed"] and calls torch.manual_seed(seed),
torch.cuda.manual_seed_all(seed) and np.random.seed(seed) GLOBALLY, before the env is ever
constructed. Every chunk process therefore starts from the SAME RNG state, and the env's own
sampling draws from that same global RNG -- env_cfg.seed=None and the generator's own "Seed not
set for the environment" print are a red herring; by the time the env samples anything, the seed
that matters has already been pinned by rl_games. Identical seed + identical command line means
the only thing differentiating chunks is uncontrolled GPU/PhysX non-determinism (contact-solver
ordering, non-associative float reduction, kernel scheduling) -- real on this hardware, but not
something to design a dataset's diversity around, and states can still be highly CORRELATED (same
spawn/goal draws, slowly diverging) even where they are not byte-identical.

THE FIX: give each chunk its OWN yaml with a distinct params.seed, keeping everything else --
checkpoint, plant, task -- identical. Nothing else needs to change; the seed is exactly the thing
that should vary per chunk and nothing more.

VERIFIES ITS OWN WRITE by reloading the file and checking the value round-tripped, rather than
trusting yaml.safe_dump silently. Does not verify what the GENERATOR actually loads at runtime --
that is generate_reset_states_policy.py's own "[generator] agent_cfg params.seed: N" print line
(added alongside this script), which a caller should grep from each chunk's own log.
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source_yaml", required=True, help="The original agent.yaml (never modified).")
    parser.add_argument("--seed", type=int, required=True, help="New params.seed value for the copy.")
    parser.add_argument("--output_yaml", required=True, help="Path to write the seeded copy to.")
    args = parser.parse_args()

    with open(args.source_yaml) as f:
        data = yaml.safe_load(f)

    # FAIL LOUDLY if the assumed schema (params.seed) is not there, rather than silently writing a
    # file whose seed override never takes effect because it landed at the wrong key.
    if not isinstance(data, dict) or "params" not in data or "seed" not in data.get("params", {}):
        print(
            f"FATAL: {args.source_yaml} has no params.seed key -- refusing to write a copy whose "
            "seed override would silently do nothing. Confirm this file's schema matches what "
            "rl_games' torch_runner.load_config actually reads.",
            file=sys.stderr,
        )
        sys.exit(1)

    old_seed = data["params"]["seed"]
    data["params"]["seed"] = args.seed

    out_dir = os.path.dirname(args.output_yaml)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_yaml, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    # VERIFY THE WRITE by reloading, rather than trusting yaml.safe_dump silently -- cheap, and
    # this is exactly the kind of thing a serialization library CAN silently get wrong (e.g. a
    # custom !!python tag on some other field breaking the round trip for the whole file).
    with open(args.output_yaml) as f:
        reloaded = yaml.safe_load(f)
    if reloaded.get("params", {}).get("seed") != args.seed:
        print(
            f"FATAL: wrote {args.output_yaml} but reloading it gives params.seed="
            f"{reloaded.get('params', {}).get('seed')!r}, not {args.seed!r} -- refusing to hand "
            "back a path whose own content does not match what was asked for.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"[make_seeded_agent_yaml] {args.source_yaml} (seed={old_seed}) -> {args.output_yaml} "
        f"(seed={args.seed}), verified by reload"
    )


if __name__ == "__main__":
    main()
