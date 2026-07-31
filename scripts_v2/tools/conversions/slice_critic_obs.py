# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drop a contiguous slice of CRITIC observation columns from an rsl_rl checkpoint.

Why this exists
---------------
The critic's privileged observations include ``insertive_object_material_properties``, which is
``PhysX shape count x 3`` wide -- so it is a function of the ASSET's collider, not just the task.
The jig-v2 asset carries 2 extra interior "blocker" collision boxes on top of v1's 46, giving
48 x 3 = 144 columns where v1's jig gives 46 x 3 = 138. A Stage-1 policy trained on ``jigv2``
therefore has a 325-wide critic, while the same env on collider-free ``jig`` builds a 319-wide
one, and ``--resume_path`` dies with a size mismatch on ``critic.0.weight``.

The ACTOR is unaffected (its observation group carries no material properties -- 195 either way),
which is why the policy itself transfers fine. Only the critic needs surgery.

The blocker boxes are authored AFTER the 46 body boxes (verified in jig_v2.usd: shape order is
box_00..box_45, interior_00, interior_01), and ``get_material_properties()`` returns
(num_envs, num_shapes, 3) flattened shape-major, so the 6 dead columns are the LAST 6 of the
material block -- global indices [222:228] of the 325-wide critic input.

Slicing rather than re-initializing keeps the learned value function; a fresh critic would feed
garbage advantages into the first PPO updates and can damage a converged actor.

The optimizer state must be sliced too: Adam's ``exp_avg``/``exp_avg_sq`` carry the parameter's
shape, and PyTorch does not shape-check on ``load_state_dict`` -- it would only explode on the
first step.

    ./uwlab.sh -p scripts_v2/tools/conversions/slice_critic_obs.py \
      --input  logs/.../model_2200.pt \
      --output logs/.../model_2200_noblocker.pt \
      --drop 222:228
"""

from __future__ import annotations

import argparse
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--drop", required=True, help="half-open column range to remove, e.g. 222:228")
    ap.add_argument("--expect_in", type=int, default=None, help="assert the critic width before slicing")
    ap.add_argument("--expect_out", type=int, default=None, help="assert the critic width after slicing")
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.drop.split(":"))
    assert hi > lo, f"empty drop range {args.drop}"

    ck = torch.load(args.input, map_location="cpu", weights_only=False)

    def cut(t: torch.Tensor) -> torch.Tensor:
        assert t.shape[1] > hi - 1, f"slice {lo}:{hi} outside tensor of width {t.shape[1]}"
        return torch.cat([t[:, :lo], t[:, hi:]], dim=1)

    targets = ["critic.0.weight", "critic_obs_normalizer._mean",
               "critic_obs_normalizer._var", "critic_obs_normalizer._std"]
    msd = ck["model_state_dict"]
    n = 0
    for k in targets:
        assert k in msd, f"checkpoint has no {k}"
        before = tuple(msd[k].shape)
        if args.expect_in is not None:
            assert before[1] == args.expect_in, f"{k}: width {before[1]} != --expect_in {args.expect_in}"
        msd[k] = cut(msd[k])
        after = tuple(msd[k].shape)
        if args.expect_out is not None:
            assert after[1] == args.expect_out, f"{k}: width {after[1]} != --expect_out {args.expect_out}"
        print(f"  model  {k:34s} {before} -> {after}")
        n += 1

    # Adam moments for the same parameter. Find them by shape rather than by a hardcoded param id,
    # so this keeps working if the network layout ever changes.
    osd = ck.get("optimizer_state_dict")
    if osd is not None:
        width_in = args.expect_in if args.expect_in is not None else None
        for pid, st in osd["state"].items():
            for kk, vv in list(st.items()):
                if torch.is_tensor(vv) and vv.dim() == 2 and vv.shape[1] > hi - 1 and (
                    width_in is None or vv.shape[1] == width_in
                ):
                    # only touch moments that match the critic layer we just cut
                    if vv.shape[0] == msd["critic.0.weight"].shape[0]:
                        before = tuple(vv.shape)
                        st[kk] = cut(vv)
                        print(f"  optim  param {pid} {kk:24s} {before} -> {tuple(st[kk].shape)}")
                        n += 1

    torch.save(ck, args.output)
    print(f"\nsliced {n} tensors, dropped columns [{lo}:{hi}) -> {args.output}")


if __name__ == "__main__":
    main()
