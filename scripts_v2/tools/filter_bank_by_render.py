# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Keep only the bank states the film renderer judged OPPOSED (held by both sides of the gripper).

render_reset_states_viewport.py writes one ``state<NN>_idx<I>_..._<VERDICT>.json`` per rendered state.
With ``--count`` >= the bank size it renders every state, so its verdicts are a full-bank replay test
under exactly the conditions the published films use. A batch replay written separately disagreed
with those films, so the renderer's own verdicts are the filter.

    python filter_bank_by_render.py --bank resets_X.pt --render-dir films/all_X --out resets_X.pt
"""

import argparse
import json
import re
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bank", type=Path, required=True)
parser.add_argument("--render-dir", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

raw = torch.load(args.bank, map_location="cpu", weights_only=False)
state = raw["initial_state"]
n = len(state["rigid_object"]["insertive_object"]["root_pose"])

verdicts = {}
for path in args.render_dir.glob("state*_idx*.json"):
    idx = int(re.search(r"_idx(\d+)_", path.name).group(1))
    verdicts[idx] = json.loads(path.read_text()).get("contact_verdict")
missing = sorted(set(range(n)) - set(verdicts))
if missing:
    raise SystemExit(f"render verdicts missing for {len(missing)} of {n} states: {missing[:10]} -- render with --count >= {n}")

kept = [i for i in range(n) if verdicts[i] == "OPPOSED"]
dropped = {i: verdicts[i] for i in range(n) if verdicts[i] != "OPPOSED"}
print(f"[filter] {args.bank.name}: kept {len(kept)}/{n}; dropped {dropped}")


def select(tree):
    if isinstance(tree, dict):
        return {k: select(v) for k, v in tree.items()}
    return [tree[i] for i in kept]


filtered = dict(raw)
filtered["initial_state"] = select(state)
args.out.parent.mkdir(parents=True, exist_ok=True)
torch.save(filtered, args.out)
print(f"[filter] wrote {args.out}")
