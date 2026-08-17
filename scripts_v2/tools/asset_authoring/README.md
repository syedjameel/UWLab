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

## What produces what

| Script | Produces | Consumed by |
|---|---|---|
| `reauthor_hullfix.py` | `ur5e_delto_hullfix.usd` | superseded by hullfix3 |
| `reauthor_hullfix2.py` | `ur5e_delto_hullfix2.usd` | superseded by hullfix3 |
| `reauthor_hullfix3.py` | `ur5e_delto_hullfix3.usd` | **the shipped hand**, loaded at `dexlift_ur5e_delto_env_cfg.py:1197` |
| `reauthor_leg_decomp.py` | `SquareTableLeg200mmDecomp/square_table_leg4_200mm.usd` | **the shipped leg** |
| `reauthor_colliders.py` | `ur5e_delto_convexhull.usd` | investigation only — this configuration is untrainable, see below |

Inputs are the git-tracked `ur5e_delto.usd` and the FurnitureBench leg mesh, so the chain is
reproducible from committed sources once these scripts exist.

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

`probe_colliders.py` and `probe_usd.py` are read-only USD inspection helpers.

## Provenance

Copied verbatim from the agent scratchpad on 2026-08-17 with original mtimes preserved
(2026-08-14 to 2026-08-16). No edits were made during vendoring — any path assumptions they
carry are the ones they were written with, and are unverified against a fresh checkout.
