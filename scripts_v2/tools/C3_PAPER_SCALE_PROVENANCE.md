# C3 (ObjectAnywhereEEGrasped) paper-scale bank — provenance

Bead: UWLab-weyl. Bank: `Datasets_ur5e_delto/OmniReset/Resets/OneLegInsertionFixture__SquareTableLeg200mmDecomp/resets_ObjectAnywhereEEGrasped.pt`
(30,204,159 bytes, n=10000, generated + merged + rekeyed + validated 2026-08-20 on vast.ai instance 48131918).

## Checkpoint

- File: `last_dexlift_ur5e_delto_reljointpos_tableleg_reorient_ep_3600_rew_38.38917.pth` (21,132,213 bytes)
- sha256: `9534e102a64fc06a3588c5102fc3421e69ef1b18fe30e7ffb40f7df63b5d76af` — independently re-verified against the copy on the generation box (48131918).

The following provenance is **not verifiable from this box or repo** — both candidate checkpoints and their certification artifacts live on a separate lab machine, DL_A6000 (`/home/dom_iva/UWLab_ur5edelto`, not a git checkout; local rescue copy at `/home/dom-iva/a6000_rescue_20260819/`). The rented generation box only ever received the winning `.pth` + its `agent.yaml`. Sourced from team-lead relaying DL_A6000's certification logs, not independently re-run by this session:

Two candidates, both from 2026-08-16 runs, both warm-started from the same parent (`09-33-54`'s `ep_3350`, rew 37.49787, certified pass@30mm 0.6562):

| | path (under `logs/rl_games/dexlift_ur5e_delto_reljointpos_tableleg_reorient/`) | pass@50mm | pass@30mm (Wilson95) | pass@20mm | pass@10mm | min_pos p50 |
|---|---|---|---|---|---|---|
| **CHOSEN** | `2026-08-16_09-33-54/nn/last_..._ep_3600_rew_38.38917.pth` (sha256 matches above) | 0.8828 | **0.6953** [0.6108, 0.7684] | 0.1406 | 0.0078 | 0.0264 m |
| REJECTED | `2026-08-16_10-36-54/nn/last_..._ep_3550_rew_43.362007.pth` | 0.8672 | **0.4531** [0.3695, 0.5395] | 0.1406 | — | 0.0304 m |

Reason: a certified 24-point gap at 30 mm on non-overlapping Wilson 95% intervals (0.6953 vs 0.4531). The single variable separating the two lineages is an override unique to the rejected branch, `env.rewards.position_tracking.weight=4.0` — that change is what made it worse and must not be carried into any finetune.

Two traps worth recording alongside this, since they are the reason it needed writing down at all:
1. The rejected lineage **also** has a file named `ep_3600` (rew 42.78791), and it was never certified. Both lineages contain a file called "ep 3600" — disambiguate by sha256 or by the certifier's own `CKPT=` line, never by epoch number.
2. The worse policy has the **higher** training reward (42.788 vs 38.389) and a smoother reward curve. Training reward is read at whatever ADR the curriculum reached; certification pins `init_difficulty` to max — they are two different tasks, so a reward comparison is not weak evidence about certified quality, it is not evidence at all. Reaching for the higher-reward file picks the one that certified 24 points worse.

## Production configuration (identical across every chunk)

```
DEXLIFT_REF_RESET=1 DEXLIFT_REF_ACTUATORS=1 DEXLIFT_REF_HAND_ACT=1 DEXLIFT_REF_ARM_ACT=0
DEXLIFT_POSE_TILT=0.3
--episode_length_s 8.0 --num_envs 128 --reset_type ObjectAnywhereEEGrasped
--receptive_usd_path .../OneLegInsertionFixture/one_leg_insertion_fixture.usd
--agent_yaml <per-chunk copy of agent.yaml, differing ONLY in params.seed>
```

Generator: `scripts_v2/tools/generate_reset_states_policy.py`. Rekey/merge: `scripts_v2/tools/rekey_dexlift_reset_states.py`, `scripts_v2/tools/merge_c2_chunks.py` (`--filename resets_ObjectAnywhereEEGrasped.pt`).

## Why chunked at all: O(n²) rewrite cost

The generator rewrites its ENTIRE accept-time bank, non-atomically (truncate + full rewrite in place, no write-temp-then-rename), on every single accepted state. Per-state write cost is therefore linear in the bank's current size, so total cost for one run is quadratic in its target count. Measured directly on the original single 10,000-state attempt:

| n (accepted) | windowed accepted/s |
|---|---|
| ~600  | 0.973 (measured this session) |
| ~1170 | 0.645 |
| ~2000 | 0.395 |
| ~2700 | 0.327 |

Fit: rate ≈ 1034/(466+n). Projected finish for a single 10,000-state run: ~13 h, against a ~2–3 h projection made from the early rate alone (which is why the plan changed mid-run). A **fresh** small-n chunk (chunk 2, restarted at n=0) measured 1.26 accepted/s (569 accepted in 453 s) — a 3.8x throughput recovery purchased entirely by keeping each chunk's own n small. This is why chunks were kept to ~1700–2000 states each rather than consolidated to save an Isaac boot.

The same non-atomic rewrite also means the canonical output path is briefly **0 bytes** on disk multiple times per minute throughout a run, not just at start/end — confirmed by direct 0.2–0.5 s-interval sampling of the live file during generation.

## Chunk 1 rescue

The original single-run attempt (target 10,000, no chunking) was killed at n≈2,972 once the quadratic cost was diagnosed (its wrapper process was killed to remove a 4 h SIGKILL deadline that would otherwise have destroyed the whole run with zero output). Because of the torn-write behavior above, chunk 1's data was recovered from a **load-gated periodic snapshot**, not from the canonical file directly — a snapshot was copied and `torch.load`-verified every ~100 s during generation, and only kept if it loaded cleanly; several capture attempts landed in a torn window and were correctly discarded (one such attempt produced an `EOFError` on the copy, empirically confirming the hazard). The canonical path itself was briefly 0 bytes (`resets_ObjectAnywhereEEGrasped.pt.ZEROBYTE_torn_1787208048` preserved as evidence) before being restored from the same snapshot.

## Per-chunk table

| chunk | n accepted | seed | acceptance rate (attempts/accepted) | `--dataset_dir` | source |
|---|---|---|---|---|---|
| 1 | 2,972 | 42 | 60.46% (4916/2972) | n/a — rescued from the original unchunked run's accept-time snapshot | load-gated snapshot, independently re-verified (`torch.load`, n=2972, correct raw schema) |
| 2 | 2,000 | 42 | 59.88% (3340/2000) | `Datasets_ur5e_delto_c3chunk2/OmniReset` | direct chunk run |
| 3 | 1,700 | 43 | 68.08% (2497/1700) | `Datasets_ur5e_delto_c3chunk3/OmniReset` | direct chunk run |
| 4 | 1,700 | 44 | 63.03% (2697/1700) | `Datasets_ur5e_delto_c3chunk4/OmniReset_chunk4` | direct chunk run |
| 5 | 1,628 | 45 | 58.08% (2803/1628) | `Datasets_ur5e_delto_c3chunk5/OmniReset_chunk5` | direct chunk run |
| **total** | **10,000** | | | | |

Chunks 4–5 use a distinct final `--dataset_dir` path component (`OmniReset_chunk<N>`, not the shared default `OmniReset`) — added at the source once the dup-check labelling bug below was found, as defence in depth alongside the tool fix.

Seed 42 was the checkpoint's own default (`agent.yaml`'s `params.seed`, read by rl_games' `torch_runner.load_config` and applied via `torch.manual_seed`/`torch.cuda.manual_seed_all`/`np.random.seed` — a GLOBAL seed, identical across every process launch regardless of the generator's own `env_cfg.seed = None`, which only skips an *additional* env-level reseed and is not itself a source of non-determinism here). Chunks 3–5 used distinct per-chunk copies of `agent.yaml` (seeds 43/44/45) so that trajectory diversity is not accidentally dependent on uncontrolled GPU/PhysX floating-point non-determinism, even though that non-determinism was empirically found to be sufficient on its own (see below).

## Cross-chunk duplicate check — including the tool bug

`merge_c2_chunks.py`'s cross-chunk duplicate check compares every state's manipulated-object `root_pose` (position + orientation) against every OTHER chunk's states, in the same discovery pass as the merge, and reports near-duplicates (<1 mm, <0.5°) and exact duplicates (bit-identical) as a percentage — a large overlap would mean the chunks share RNG state and are not adding real diversity despite a correct total count.

**The first three-way run of this check silently compared only two of the three chunks.** Root cause: the chunk label used to key duplicate-detection results was `os.path.basename(chunk_dir)`, and two of the three chunk dataset dirs (`.../c3chunk2/OmniReset` and `.../c3chunk3/OmniReset`) share the identical basename `"OmniReset"` — one chunk's poses silently overwrote the other's in the label dict. The merge itself was unaffected (it works off the path list, not the label dict), so the reported state COUNT was correct while the duplicate REPORT was silently wrong (comparing chunk 1 against only one of chunk 2/3, not both). Fixed by index-prefixing every chunk's label (`"1:chunk1"`, `"2:OmniReset"`, `"3:OmniReset"`, ...), which is unique by construction regardless of basename collisions, plus a hard assertion that the label count always matches the path count.

Final five-way check, on the fixed tool, across the full bank:

```
1:chunk1=2972 + 2:OmniReset=2000 + 3:OmniReset=1700 + 4:OmniReset_chunk4=1700 + 5:OmniReset_chunk5=1628
near-dup (<1.0mm, <0.5deg, cross-chunk) = 0/10000 (0.0%)
exact-dup                               = 0/10000 (0.0%)
```

Zero cross-chunk duplication was also confirmed under the same-seed pair (chunks 1 vs 2, both seed 42) before the seed-varying change took effect — GPU/PhysX non-determinism alone fully decorrelated those two chunks' 8 s-episode rollouts on this hardware. The seed-varying change for chunks 3–5 was kept anyway, since relying on uncontrolled float non-associativity for dataset diversity is not a property this pipeline should depend on even though it happened to hold here.

## Final validation (at the canonical path, after rekey + swap)

```
rigid_object keys: [insertive_object, receptive_object, table, ur5_metal_support]
articulation.robot: joint_position_target and joint_velocity_target present
n_states = 10000, all field lengths consistent
receptive_object z mean=0.019625 std=0.0; x in [0.35,0.60]; y in [-0.20,0.20]
ur5_metal_support z mean ≈ 0.004020
table pinned at identity pose
insertive_object z quantiles: p10=0.3127 p50=0.4905 p90=0.6353 (matches historical C3 airborne signature, ~0.486 median)
```

Independently re-verified in this session (not just taken from the merge/rekey tool's own output). The prior file was backed up alongside the canonical path before the swap.
