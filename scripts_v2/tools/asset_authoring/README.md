# Asset authoring and verification scripts

These scripts **regenerate the USD assets this project trains on**, and the gates that
decide whether a regenerated asset is safe to train with.

## Why this directory exists

Until 2026-08-17 every file here lived only in an ephemeral `/tmp` scratchpad belonging to a
single agent session. Nothing in the repository could rebuild the assets, even though the code
that loads them names these scripts in its own error messages. Two of the assets they produce
are caught by the blanket `**/*.usd` rule in `.gitignore` and were never added to git, so a
fresh clone fetches nothing for those paths — not even a stub.

That meant: base asset in git, derived asset untracked, and the derivation itself in `/tmp`.
A cleared temp directory would have made `ur5e_delto_hullfix3.usd` — the hand every certified
policy in this project was trained on — permanently unreproducible.

That risk is now closed, not just mitigated: see "Verified reproducibility" below.
`reauthor_hullfix3.py` was rebuilt from the git-tracked base into an isolated scratch location
(same day) and came back byte-identical to the live asset, sha256 confirmed both directions.

## What produces what

| Script | Produces | Consumed by |
|---|---|---|
| `reauthor_hullfix.py` | `ur5e_delto_hullfix.usd` | superseded by hullfix3 |
| `reauthor_hullfix2.py` | `ur5e_delto_hullfix2.usd`, chained off `reauthor_hullfix.py`'s output | superseded by hullfix3 |
| `reauthor_hullfix3.py` | `ur5e_delto_hullfix3.usd` | **the shipped hand**, loaded at `dexlift_ur5e_delto_env_cfg.py:1197` |
| `reauthor_leg_decomp.py` | `SquareTableLeg200mmDecomp/Props/instanceable_meshes.usd` (the referenced payload the top-level `square_table_leg4_200mm.usd` points at by relative path; that top-level file itself is copied through unchanged) | **the shipped leg** |
| `reauthor_colliders.py` | `ur5e_delto_convexhull.usd` | investigation only — this configuration is untrainable, see below |

**CORRECTED 2026-08-17, after actually reading the scripts and running the one that could be run
safely (see "Verified reproducibility" below) — the line that used to be here ("inputs are the
git-tracked `ur5e_delto.usd` and the FurnitureBench leg mesh, so the chain is reproducible from
committed sources") was half right and half wrong:**

* **`reauthor_hullfix3.py` really does take a ONE-STEP path from a git-tracked input.** Its `SRC`
  is `Robots/Ur5eDelto/ur5e_delto.usd` directly (21,918,375 B, tracked, confirmed a real 21.9 MB
  git blob, not an LFS pointer) — it does NOT chain through `hullfix.usd`/`hullfix2.usd` despite
  the similar names; those two are earlier, superseded, dead-end diagnostics.
* **`reauthor_leg_decomp.py`'s input is NOT git-tracked.** Its `SRC` default is
  `Props/FurnitureBench/SquareTableLeg200mm/` (the pre-decomposition source), and that whole
  directory — including `Props/instanceable_meshes.usd`, the file the script actually mutates —
  is untracked by git exactly like its `...Decomp` output is. One file inside it
  (`instanceable_meshes.usd`) has a `.gitignore` negation (`!...instanceable_meshes.usd`) that
  would let it be added, but `git ls-files` shows it never was. So this half of the claim was
  false: the leg chain is reproducible from what's IN THIS DIRECTORY plus that source directory,
  not from git alone. Both `SquareTableLeg200mm/` and `SquareTableLeg200mmDecomp/` need to travel
  as bytes to any new host, same as the two `.usd` files themselves.

## Verified reproducibility (2026-08-17, bead UWLab-qiao.1 follow-on audit)

`reauthor_leg_decomp.py` already had, and `reauthor_hullfix3.py` was given (see below), the two
properties needed to safely rebuild-and-compare: a `sys.argv`-based `SRC`/`DST` override (no edit
needed) and running under bare `pxr` (no Isaac app boot). `reauthor_leg_decomp.py` first: redirected
to an isolated scratch copy of `SquareTableLeg200mm/`
(never touching the live asset), it reproduced `Props/instanceable_meshes.usd` **BYTE-IDENTICAL**
to the live asset: sha256 `0bbd17310f505857a52dbdaf976a6d7c94f61efd1e1f7e17cb6f1e5d5f3471c8` both
times, same 431,129 B. This is despite the script's own honest caveat that `PhysxSchema` is
unavailable under bare `pxr` and the decomposition-tuning attributes therefore don't survive as a
named API on re-read (`physx tuning: False -> decomposition runs at PhysX DEFAULT tuning`) — the
match proves the LIVE asset was itself produced the same way, under the same bare-`pxr`
limitation, not via a full Isaac Sim boot with real `PhysxSchema` tuning.

**`reauthor_hullfix3.py` is now PROVEN, not just structurally argued.** It originally had the same
hardcoded, non-redirectable `SRC`/`DST` as the other four scripts, so it could not be safely run
against a scratch location. Fixed 2026-08-17 by adding optional positional CLI overrides
(`sys.argv[1]` for `SRC`, `sys.argv[2]` for `DST`, independent of each other — `DST` is NOT derived
from `SRC`, unlike `reauthor_leg_decomp.py`'s pattern, because redirecting the output independently
of the input is the whole point when the output path is the dangerous one). With zero args, `SRC`
and `DST` are byte-identical to what the script always defaulted to — verified by direct
comparison, not just reasoning about the code.

Redirected to an isolated scratch copy of `ur5e_delto.usd` (the git-tracked base, never touching
the live paths at any point — SRC and DST both pointed into `/tmp`), the rebuilt
`ur5e_delto_hullfix3.usd` came back **BYTE-IDENTICAL** to the live asset:

    sha256  77fb40f3e42d12a2450a30cff3a6ecea859df6d20113dd31861a22227a564582
    size    21,923,172 B

matching in both fields, on both sides (scratch rebuild and live file re-verified after the run).
**The hand every certified policy in this project trained on is now proven reproducible from
committed sources (`ur5e_delto.usd`, git-tracked, confirmed a real 21.9 MB blob) plus this
repository alone.** No PhysxSchema availability issue surfaced for this asset the way it did for
the leg (this script only ever writes `physics:approximation` tokens and `FilteredPairsAPI`
relationships, both plain-`pxr`-native — it never touches `PhysxConvexDecompositionCollisionAPI`,
which is what triggered the leg's PhysxSchema caveat).

`reauthor_hullfix.py`, `reauthor_hullfix2.py`, and `reauthor_colliders.py` still have the
no-override problem `reauthor_hullfix3.py` used to have, but their outputs are dead-end
diagnostics, not the shipped asset — lower stakes; same fix (positional `sys.argv` overrides) if
anyone wants to verify them the same way.

## Verification gates — run these before any regenerated asset reaches training

`randenv.py` is the cheap one and it is not optional. Nine envs, zero actions, print
`|q - default|` per joint. About ten minutes including Isaac boot. **If any joint deviates
beyond the reset jitter band with zero actions applied, the asset is broken.** A blanket
`sdf -> convexHull` rewrite once passed casual inspection and made the articulation diverge by
thousands of degrees at reset; the resulting policy metrics were meaningless because the robot
was never intact. That cost roughly twelve hours of training and a rented box, against ten
minutes for this gate.

`probe_penetration.py` measures non-adjacent finger-pair interpenetration **depth** — not rate.
Rate fires on ordinary contact, because PhysX permits penetration up to its rest offset, and
reading it as the headline nearly buried a real 25x improvement. Depth is what separates
fingers touching from fingers passing through each other.

`probe_pairs.py` lists interpenetrating pairs at the authored spawn posture.
`closure_test.py` measures reachable finger curl; drive only the flexion joints — driving the
`_1` spread joints splays the fingers into each other and the test jams them itself, which
inflated a measured cost by 5x and flipped a shipping decision.

`probe_colliders.py` and `probe_usd.py` are read-only USD inspection helpers. `probe_colliders.py`
also hardcodes a `REF` path into a SEPARATE repository (`IsaacLabDexterous` -- see the comment at
its `REF` constant) that will NOT exist on DL_A6000 or a freshly provisioned host such as the
5090 unless that other repo is vendored there too; there is no CLI override for it.

## Provenance

Copied verbatim from the agent scratchpad on 2026-08-17 with original mtimes preserved
(2026-08-14 to 2026-08-16). No edits were made during vendoring.

Two edits were made AFTER vendoring, same day, both narrow and both documented at the point of
change:
* `reauthor_hullfix3.py` gained optional positional `sys.argv` overrides for `SRC` and `DST`
  (default behaviour, zero args, is byte-identical to before the edit -- verified, not assumed)
  so it could be rebuilt-and-compared without ever writing to its live output path. See
  "Verified reproducibility" above.
* `probe_colliders.py` gained a comment on its `REF` constant noting the separate-repo dependency
  above. No behaviour change.

Every other script, and every other line of these two, is unedited from the scratchpad copy --
any path assumptions they still carry are the ones they were written with, and are unverified
against a fresh checkout.
