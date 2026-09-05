# Jig-Removal v2 — Deviation Ledger

Branch `omnireset/jig-removal-v2`, cut from `omnireset/jig-v2-stable-grasp` (`5d78368`).

**Task.** The jig starts SEATED on the enclosure+PCB stack; the manipulator must LIFT it off and
place it on an elevated plate at the far end of the table.

**Why v2 exists.** Removal-v1 failed: the policy learned to SLIDE the jig to the target instead of
lifting it. The target there was a flat invisible marker with a 50 mm success threshold, so a jig
shoved across the mat satisfied every condition. v2 replaces it with a real elevated plate whose
`assembled_offset` is on its TOP face, so success requires the jig's bottom to be at plate-top
height — a slid jig is a full plate-thickness low and fails on geometry, not on a penalty.

**Net effect on deviations: v2 has FEWER than v1.** Three of v1's deviations disappear because the
target is now a genuine receptive object rather than a marker success was retargeted at.

| v1 deviation | v2 |
|---|---|
| R3 loose 50 mm position threshold | **gone** — stock 5 mm |
| R4 world-fixed invisible marker | **gone** — a real plate |
| R5 success/command/reward retargeted off the receptive | **gone** — the plate IS the receptive |
| R8 authored near-goal file | stays (D2 below) |
| authored seated-start file | stays (D1 below) |

---

## Bug fixes — defects, not deviations

**B1. Quad-face triangulation** (`mdp/utils.py`, commit `bc39976`, carried from jig-removal).
`prim_to_warp_mesh` passed `faceVertexIndices` to Warp as a triangle list. Correct for STL meshes,
wrong for hand-built box colliders, which are QUADS (6 faces x 4 indices) and became 8 nonsense
triangles. Routed through `prim_to_trimesh`.

**B2. Collider quaternion convention** (`mdp/collision_analyzer.py`, same commit).
`RigidObjectHasher` stores rel quats WXYZ; the Warp kernel consumed them as `wp.quat` (XYZW), so
identity became a 180 deg rotation about X and every obstacle was queried MIRRORED.
Evidence: seated jig -4.25 mm "inside" 0/16 accepted -> -0.00 mm 16/16; seat lowered 6 mm still
correctly rejected. These two shared-code fixes affect every task; both were proven with positive
and negative controls, and they measurably sped up reset recording.

**B3. Pedestal collider approximation** (`build_jig_enclosure_usds.py`).
`add_box` defaults to `convexHull`. PhysX shrinks convex hulls inward when cooking, and the
180 x 180 x 10 mm plate (18:1 aspect) degenerated to **no contacts at all** — a jig placed on it
free-fell through the plate, the mat and the table at exactly g. Fixed with `boundingCube`, which
skips hull cooking. Measured, 100 C4 states:

| | convexHull | boundingCube |
|---|---|---|
| C4 acceptance | 0.05% | **24.07%** |
| wall time | 86 min | **1 min 49 s** |
| `coll_free` pass | 29.8% | 99.8% |
| `stable` pass | 39.1% | 95.6% |

Nothing else in the project hit this: every load-bearing collider is chunky (the jig rests on the
enclosure's ~7 mm cubic pillars). The enclosure has thin boxes too (`box_13` is 3 mm) but nothing
ever rests on them. **Any future wide thin collider needs `boundingCube`.**

**B4. v2c blocker was stale against jig2.** The blocker's high slabs covered only |x| >= 30,
because the OLD jig's long walls dropped to 9 mm height for |x| <= 28 and material above 8.5 mm
there would have stood in open air (the v2 failure: the jig was pickable BY its blocker).
jig2 closed those slots, so the exclusion had nothing left to justify — it was a 60 mm wide open
channel into the jig's middle, i.e. exactly the volume the blocker exists to deny, and the
one-sided rim pinch would have returned through it. High slab is now FULL WIDTH; interior sealed
(0 open vertical channels), still clear of the enclosure at the seat.

**B5. `enclosure_pcb` init_state.** `reset_root_states_uniform` treats `pose_range` as a DELTA from
the asset's default root state, so a non-zero `init_state.pos` silently shifts the whole band
(measured: `pos.x=0.55` pushed the 0.46-0.70 band to 1.04-1.22, off the table). Pinned to origin.

**B6. `variants` missing on `CameraAlignEnvCfg`.** Without it, `env.scene.insertive_object=<x>` is
assigned as a raw STRING and the env dies with "Incorrect type under namespace". Means no
object-pair override has ever worked on the camera-align env, including guide §7's calibration
path. Added; defaults unchanged.

---

## Additive config changes — no existing behaviour altered

**A1. `pedestal` receptive variant** in `rl_state_cfg`, `reset_states_cfg`,
`partial_assemblies_cfg`. Plain `make_receptive_object` — no special collision props needed once
B3 is fixed (verified by drop test).

**A2. `enclosure_pcb` scene field** in `rl_state_cfg` and `reset_states_cfg`. Kinematic prop, the
PICK site. Not the receptive object; carries no observation term. Inert for every other task.

**A3. `reset_enclosure_pcb_pose` event** in `ResetStatesBaseEventCfg`. Stock
`reset_root_states_uniform`, band x 0.46-0.65 / y +-0.24 / free yaw. No-op for other tasks, which
never reference the asset.

No existing term, threshold, function or default was modified. Every other task loads identically.

---

## Deviations proper — two, both DATA-ONLY

**D1. Authored seated-start `partial_assemblies.pt`** (`EnclosurePcb__JigV2c`, 2 entries).
The deployment start is "jig seated on the enclosure". The stock C1 recorder samples the jig
INDEPENDENTLY of anything, so it can never produce a dependent placement; the stock mechanism for
that is the partial-assembly event, which composes a stored relative pose with the target's current
randomised pose. Retargeting `receptive_object_cfg` at `enclosure_pcb` makes the loader resolve the
pair against the enclosure. rel z = 0.011563, DERIVED from the two metadata files
(`receptive.assembled_z - insertive.assembled_z`), not hardcoded. Verified: recorded states came
back at 11.51 mm, spread 0.000.

**D2. Authored C4 near-goal `partial_assemblies.pt`** (`JigV2c__Pedestal`, 256 entries).
`record_partial_assemblies.py` walks the insertive out of a SOCKET, recording the continuum of
partial insertions. **A flat plate has no such continuum** — the jig is either on it or off it.
Measured: the stock recorder returned 11 identical assembled poses plus 1 that slid off the plate.
Authored instead: +-35 mm lateral, 0-70 mm above the plate top, yaw within +-0.6 rad of each twin
(the gate is 0.35 rad, symmetry 2, so ~53% start inside it and the rest must be corrected).

Both are `.pt` files in the authors' own format. No code or config semantics changed.

---

## Measured facts worth keeping

| fact | value |
|---|---|
| jig collider | 46 -> **20** boxes (removal never inserts, so the pillar sockets above z=5 collapse to solid slabs) |
| enclosure's tallest reach into the jig | 4.95 mm, vs the z=5 step — 0.05 mm margin |
| seat | rel z 18.2-18.3 mm, mate **7/7**, unchanged by the collider reduction (0 voxels of material lost) |
| interior blocker | 2 boxes, interior **sealed**, clears the enclosure at the seat |
| plate | 180 x 180 x 10 mm; slid jig misses by **2x** the 5 mm threshold |
| plate size reason | fixture ends at x 0.754 (0.65 + jig 104 mm half-diagonal), so plate centre >= 0.754 + half-side; at 300 mm the side camera sees 51%, at 180 mm it sees 84% |
| front camera | stops at x 0.82, so the plate centre at 0.844 is OUTSIDE it — placement runs on side + wrist |
| C4 accepted distribution | 96% within 0-10 mm of the plate top; aloft states fail `not_far` (real, survives B3) |
| grasps | 512, approach median **5.1 deg** from vertical, 100% within 15 deg |

## Open items

* **Real plate is 5 mm; sim is 10 mm.** Stack two 5 mm plates to match. At 5 mm a slid jig misses
  by exactly 1.0x the threshold — no margin — and the jig can tip up over a 5 mm edge.
* Seated-vs-mat mix in training is a `probs` value, not baked in. Suggested start: mat 15-20%.
* The mat set shares the fixture band, so some jigs land ON the fixture. Realistic for a drop.

---

# v3 CORRECTION (2026-09-05) — back to the authors' FOUR reset types

**v2's Stage-1 failed.** After 1708 iterations: task_0 (seated deployment start) **0.0061**,
task_1 (mat) 0.0090 — both effectively zero — while every already-grasped task trained normally:
task_2 0.72, task_3 0.77, task_4 0.52. Transport and placement were fine; **grasp DISCOVERY was
dead.**

## Cause — two mistakes, both in how v2 was run, not in the assets

**1. A fifth reset type was invented.** v2 made the seated start its own type
(`ObjectPartiallyAssembledEEAnywhere`) and passed an explicit 5-way `reset_types`/`probs`
override. The authors use FOUR (`rl_state_cfg.py` default: AnywhereEEAnywhere,
RestingEEGrasped, AnywhereEEGrasped, PartiallyAssembledEEGrasped, `probs [0.25]*4`), and the
jig-removal v1 line kept four by MERGING the seated states into C1.

**2. That fifth type cannot be seeded.** `reset_end_effector_pregrasp_seeds` — the documented
positive counterpart to the interior blocker (`V2_STABLE_GRASP_PLAN.md` §5 trio) — exists ONLY on
`ObjectAnywhereEEAnywhereEventCfg`. Putting the deployment start on a `PartiallyAssembled` config
placed it on the one EEAnywhere path with no seeding term. Blocker denies the rim pinch, nothing
offers the straddle, so there is no route to a first grasp. This reproduces the plan's own record
of blocker-without-seeds: **task_0 0.003 at iteration 1000.**

## v3 fix — no new code, no new deviation

* **Authors' four types, taken from the CONFIG DEFAULT.** Never pass `reset_types`/`probs`.
* **Seated states MERGED into C1** (`merge_seated_into_c1.py`), and the 5th file DELETED so it
  cannot be loaded as a task. This is v1's pattern.
* **C1 recorded with `seed_prob=0.25`** — a stock parameter of a term already present on this
  branch lineage, and the intended counterpart to the blocker.

Assets, config and bug fixes are unchanged from v2; every deviation listed above still stands and
no new one is added. The 5-way `probs` override is REMOVED.

## Also corrected in v3

* **Near-goal band ±35 mm → ±8 mm.** v1 used ±35 mm against a **50 mm** success gate; reusing it
  against v2's **5 mm** gate left C4 states a median 30.1 mm from success, so task_4 could not act
  as a near-goal bootstrap. ±8 mm / 0–12 mm gives a 10.2 mm median (p10 7.9 mm).
* **`--num_envs 12288`, not 16384.** PhysX caps materials at 65536 — a hard 16-bit limit that no
  `$BIGBUF` setting touches. This scene has 4 materials/env (jig 2 including the blocker,
  enclosure_pcb 1, pedestal 1); 16384 × 4 = 65536 hits it exactly and the run hangs at startup.
