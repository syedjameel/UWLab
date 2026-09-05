# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Merge the SEATED deployment states into C1, keeping the authors' FOUR reset types.

The jig-removal deployment start is "jig seated on the enclosure+PCB stack". No stock recorder
produces a dependent placement like that, so it is recorded separately via the partial-assembly
event (as ``ObjectPartiallyAssembledEEAnywhere``) and then folded into
``resets_ObjectAnywhereEEAnywhere.pt`` -- which is what the jig-removal v1 line did.

WHY THE MERGE AND NOT A FIFTH TASK. v2 kept the seated set as its own reset type and paid for it:
``ObjectPartiallyAssembledEEAnywhere`` is the one EEAnywhere config with no
``reset_end_effector_pregrasp_seeds`` term, so with the interior blocker denying the one-sided rim
pinch there was no route at all to the straddle grasp. Measured: task_0 reached 0.0061 in 1708
iterations while every already-grasped task trained normally (0.72 / 0.77 / 0.52). Merging into C1
puts the deployment states on the config that CAN be seeded, and keeps
``reset_from_reset_states`` on its authors' default of four types at 0.25 each -- no override.

After merging, DELETE ``resets_ObjectPartiallyAssembledEEAnywhere.pt`` so nothing can load it as a
fifth task.
"""

from __future__ import annotations

import argparse
import os
import shutil

import torch

C1 = "resets_ObjectAnywhereEEAnywhere.pt"
SEATED = "resets_ObjectPartiallyAssembledEEAnywhere.pt"


def _n(state) -> int:
    return len(state["initial_state"]["rigid_object"]["insertive_object"]["root_pose"])


def _cat(a, b):
    """Recursively concatenate the two nested state dicts (lists of per-state tensors)."""
    if isinstance(a, dict):
        assert set(a) == set(b), f"key mismatch: {set(a) ^ set(b)}"
        return {k: _cat(a[k], b[k]) for k in a}
    if isinstance(a, list):
        return a + b
    if torch.is_tensor(a):
        return torch.cat([a, b], dim=0)
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help=".../OmniReset")
    ap.add_argument("--pair", default=None, help="default: the single dir under Resets/")
    ap.add_argument("--keep-source", action="store_true",
                    help="do NOT delete the seated file after merging (default deletes it, so it "
                         "cannot be picked up as a fifth reset type)")
    args = ap.parse_args()

    root = os.path.join(args.dataset_dir, "Resets")
    pair = args.pair or next(d for d in sorted(os.listdir(root))
                             if os.path.isdir(os.path.join(root, d)))
    d = os.path.join(root, pair)
    c1_path, seated_path = os.path.join(d, C1), os.path.join(d, SEATED)
    for p in (c1_path, seated_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")

    c1 = torch.load(c1_path, map_location="cpu", weights_only=False)
    seated = torch.load(seated_path, map_location="cpu", weights_only=False)
    n_c1, n_se = _n(c1), _n(seated)

    backup = c1_path + ".premerge"
    if not os.path.exists(backup):
        shutil.copy2(c1_path, backup)

    merged = _cat(c1, seated)
    assert _n(merged) == n_c1 + n_se, "merge lost states"
    torch.save(merged, c1_path)

    print(f"pair: {pair}")
    print(f"  C1 (mat, seeded)      {n_c1:7d}")
    print(f"  seated (deployment)   {n_se:7d}")
    print(f"  -> merged C1          {_n(merged):7d}   ({n_se / _n(merged) * 100:.1f}% seated)")
    print(f"  backup: {backup}")

    if not args.keep_source:
        os.remove(seated_path)
        print(f"  removed {SEATED} (so it cannot load as a 5th task)")
    print("\nreset_from_reset_states now sees the authors' FOUR types at 0.25 each -- pass no override.")


if __name__ == "__main__":
    main()
