# Certification harness

These scripts produce **every certified number quoted for this project's UR5e+DELTO policies**.

## Why this directory exists

Until 2026-08-17 all of it lived loose in `/home/dom_iva` on DL_A6000 — a tree that is an
`rsync`, not a git checkout (`git log` there returns "fatal: not a git repository"). There was
no version control on the only copy. `certify_pose.py` alone is 32 KB and is invoked by
`run_certify.sh` as `$HOME/certify_pose.py`, so the two were coupled by an absolute path into a
home directory rather than by anything in a repository.

If that box had been rebuilt, no certified figure in this project would have been reproducible:
not the Stage-2 lift `pass@30mm 0.9219`, not the Stage-3 pose `pass@30mm 0.6953`, none of the
23 stored cert log/json pairs. Copied here verbatim with original mtimes preserved.

## What each file is

| File | Role |
|---|---|
| `certify_pose.py` | the evaluator. 128 episodes, held-out seeds, deterministic, ADR pinned to `max_difficulty`. Writes a JSON summary plus a per-episode record |
| `run_certify.sh` | the wrapper that selects a plant and **asserts it** before believing any number. See below |
| `cert30.sh` | the batch that re-scored the campaign at the 3 cm bar after the user raised it from 1 cm |
| `cert_both.sh`, `eval_pair.sh` | earlier paired-run drivers |
| `certsum.py` | summarises a cert JSON |
| `checkrew.py`, `extract_ft_metrics.py` | reward / metric readers over training logs |
| `closure_test.py` | measures reachable finger curl; **drive only the flexion joints** — driving the `_1` spread joints splays the fingers into each other, which inflated a measured cost 5x and flipped a shipping decision |
| `cert_ft.sh` | 2026-08-17. Certifies the goal-at-spawn finetune against the base reorient task, with its parent re-certified in the same batch as an in-batch control |

## The two things that make a certified number mean anything

**1. The plant is asserted, not assumed.** `run_certify.sh` takes `COLLIDERS` / `SELFCOLL` /
`HAND` / `ARM`, then greps the run's own `[dexlift] PLANT ...` banner and exits non-zero on any
mismatch. Probing a policy under a different arm or a different collider set measures a
different robot. The certified configuration is
`colliders=hullfix3 self_collisions=ON hand_act=reference arm_act=identified`.

**2. `DROP_Z` must match the run being certified.** `_apply_pose_tilt_stage` clamps the reset
drop to 0.05 m by default whenever a tilt is staged, so certifying a policy that trained with
the unclamped 0.4 m drop under a bare `TILT=` scores it on a task it never saw.

## Traps

* **`GPU` defaults to 3.** On a shared box that card is often the busy one. Pass `GPU=` explicitly
  after checking `nvidia-smi` — never launch onto a GPU that already has another user's process.
* **Training reward is not evidence about certified quality.** Reward is read at whatever ADR the
  curriculum happened to reach; certification pins `init_difficulty` to `max_difficulty`. They are
  two different tasks. In this project the checkpoint with the *higher* training reward (43.36 vs
  38.39) certified **24 points worse** at 30 mm. Disambiguate checkpoints by the certifier's own
  `CKPT=` line and `checkpoint_sha256`, never by epoch number, reward, or curve shape.
* **Reproducibility is ±2 points at 10 mm and ±3.5 at 30 mm** across bitwise-identical replays —
  roughly three episodes in 128 sitting on a decision boundary. Treat anything smaller as no
  difference, and prefer an in-batch control over comparing to a stored number.
* **`launch_task.sh` does not exist.** `docs/dexlift_ur5e_delto_RECIPE.md` calls both it and
  `run_certify.sh` in its code blocks. Only `run_certify.sh` was ever found, and it is now here;
  the training-launch half of that recipe is still unreproducible and the recipe's commands cannot
  be run as written.

## Provenance

Copied from `/home/dom_iva` on DL_A6000 on 2026-08-17 by `tar` over ssh, mtimes preserved
(2026-07-08 to 2026-08-18). **No edits.** Every absolute path they contain — `$HOME/certify_pose.py`,
`$HOME/UWLab_ur5edelto`, `$HOME/UWLab/env_uwlab/bin/python` — is the one they were written with and
is **unverified against a fresh checkout**. They are vendored here so the logic survives; making
them path-portable is separate work.
