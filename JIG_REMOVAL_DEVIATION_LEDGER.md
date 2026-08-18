# Jig-removal task — deviation ledger

Branch `omnireset/jig-removal`, off `omnireset/jig-v2-stable-grasp` @ `5d78368`.

**Task.** Start from the fully assembled stack — bottom enclosure, PCB seated at 13.60 mm, jig
registered on the corner pillars — grasp the JIG, lift it clear, and set it down at a fixed free
spot on the mat.

**Base choice.** jig-v2, not realpcb-onto-jig: its six commits are exactly the blocker machinery
this task needs (`_add_interior_blocker`, `verify_jig_v2_mass.py`, `eval_expert_success.py`),
while realpcb-onto-jig would have dragged in the pedestal scene body and modifications to
`grasp_sampling_event` (D1 standoff, band/tip parameters) that a 24 mm-thick jig does not need.

House rule: follow the OmniReset authors' implementation; deviate only when physically forced,
and record it here. Every number below is measured.

**Task registrations, config classes and reset-type semantics are UNCHANGED.** The removal task
runs on the stock `OmniReset-UR10eLinearGripper-*` task ids with the authors' four reset types at
0.25 each; everything task-specific is asset choice, CLI overrides, and two authored input files.

---

## B1 / B2 — two BUG FIXES in the authors' collision checker (not deviations)

Both found by bisecting a physically-clean seated jig that the recorder refused, and both fixed
with positive and negative controls.

**B1 — `prim_to_warp_mesh` fed QUAD faces to warp as triangles** (`mdp/utils.py`).
`faceVertexIndices` was passed straight through as a triangle list. Correct for STL-derived
triangle meshes; wrong for our hand-built box colliders (6 faces x 4 indices), which became 8
nonsense triangles — a self-intersecting shard whose signed distances are garbage near the
surface. Fixed by routing through `prim_to_trimesh`, which triangulates via `faceVertexCounts`
exactly as the point-sampling path already did.

**B2 — quaternion convention mismatch** (`mdp/collision_analyzer.py`).
`RigidObjectHasher` stores collider relative quats **WXYZ** (IsaacLab convention); the warp
kernel consumes them as `wp.quat`, which is **XYZW**. An identity `(1,0,0,0)` therefore became a
180-degree rotation about X, querying every obstacle mesh **mirrored through its prim origin**.
Reordered once at construction.

Evidence:

| | before | after |
|---|---|---|
| perfectly seated jig | **-4.25 mm "inside"**, 0/16 accepted | **-0.00 mm**, 16/16 accepted |
| seat lowered 6 mm (real interpenetration) | — | **-1.74 mm**, rejected |
| standalone repro (same meshes, no Isaac) | — | min sign **+0.05 mm**, 0/4096 points below -1 mm |

The un-mirrored worst point lies 4.2-4.4 mm inside the end shelf, matching the measured -4.25
exactly, which closes the loop on the mechanism.

**Blast radius.** Every recorder acceptance in this project ran through this code. Nothing already
recorded is *invalid* (recorded states are real physics), but accept rates were wrong in both
directions near contact — plausibly a contributor to the jig task's C4 misery. Worth a re-look
before trusting old accept-rate numbers.

---

## R1 — jig geometry updated to jig2 (NOT a deviation)

The physical jig was redesigned 2026-08-14: the open slots in the long walls are closed.
`Jig/jig.stl` replaced, so sim matches hardware.

| | old | new |
|---|---|---|
| extents | 164 x 129.03 x 24 mm | identical |
| long-wall top, \|x\| <= 28 | 9.0 mm (notch) | **21.5 mm** |
| clear window | 141x101 clears, 142x102 collides | **identical** |
| PhysX mass | 0.125875 kg | **0.143875 kg** |

`_JIG_BOXES_MM`'s upper tier no longer skips \|x\| < 25 -> jig collider 46 -> **44** boxes.

⚠ The v2/v2b/v2c blockers are now obsolete or wrong (v2c is *shaped* to a notch that no longer
exists). Not rebuilt — a re-decision for the jig-enclosure task, not this one.
⚠ The realpcb pipeline is stale against hardware for the same reason; re-check before deploying.

---

## R2 — `JigBlocked`: a massless collider in the jig window — APPROVED

**Deviation.** The insertive object is the real jig plus ONE massless box filling its window
(x +-72.5, y +-52.5, z +-12 mm). No such thing exists on the real part. Sim-only training
scaffold, same class as jig-v2.

**Why forced.** This task PICKS THE JIG, so jig-v1's failure returns: a jaw descends into the open
window and pinches a single ~14 mm wall. Measured on the v1 expert: median jaw gap **13.8 mm** at
peak lift, **201/202** successes classified as pinch, **zero** straddles.

**Why ONE box suffices here.** With the slots closed the blocker top (+12, the jig's top face)
sits above the wall top (+9.5), so a jaw descending anywhere inboard of the rim lands on the
blocker or the wall. v2's fake grasp — holding the BLOCKER at a 107.7 mm jaw gap — depended on
the blocker standing in open air across the 9 mm notch band, which no longer exists.

**Verified.** `verify_jig_v2_mass.py` with `JIG_V2_VARIANT=jigblocked`: mass, inertia and COM
**bitwise identical** to the plain jig (0.143875107 kg). Recorded C2/C3 jaw gaps: **130.2 / 129.7
mm** = straddles on the 129 mm rim, zero pinches.

**Removal plan.** Stage-1 scaffold; whether it survives the finetune is a separate decision,
measurable with `eval_expert_success.py` (pinch < 40 mm, straddle > 100 mm).

---

## R3 — loose success thresholds — APPROVED

`ParkingSpot` metadata: `position: 0.05` (50 mm) and **no yaw gate**, against the authors' 0.005
and the jig task's yaw gate. Parking is not insertion; which way round the jig is set down does
not matter. `orientation: 0.025` is KEPT so it lands flat.

⚠ Success drives the reward, the ADR warmup gate (0.95) and every training number. A looser
criterion makes this task's numbers not directly comparable with the other tasks'.

---

## R4 — a third scene body: the world-fixed parking marker — APPROVED

**Deviation.** `parking_marker`, an INVISIBLE body added to `RlStateSceneCfg` and
`ResetStatesSceneCfg`, static at **(0.87, 0, 0.004)**, positioned by `init_state` alone and never
moved by any event. No visuals; one 4 mm collider buried 3-7 mm below its origin, inside the mat.

**Why forced — measured both ways.** The two-body alternative (target as an offset in the
FIXTURE's frame) is genuinely simpler, but with FREE fixture yaw the target orbits the fixture, so
the whole circle must stay on the mat. That constrains the fixture to an **82 x 82 mm** window
(x 0.659-0.741, \|y\| <= 0.041) — visibly near-static in the GUI and rejected on review. A
world-fixed marker keeps free yaw AND the wide band (x 0.45-0.62, y +-0.24). Recorded fixture
spread with the marker: x ~165 mm, y ~430 mm, yaw std 75-115 deg.

**Why one collider, not zero.** A zero-shape body is a needless edge case for any
shape-count-derived observation — see the 325-vs-319 critic-width incident on jig-v2.

**Why fixed, not randomised.** The RGB student sees only images + proprioception, never
`receptive_asset_pose`. An invisible target that MOVED per episode could only be approached at the
average of where the expert went; pinned to one spot it is memorised from the visible table.

---

## R5 — success measured against a different entity than `scene.receptive_object` — APPROVED

`terminations.success.params.receptive_asset_cfg.name=parking_marker` (and the same for the
trainer's `TaskCommand` / `ProgressContext`), while `scene.receptive_object` stays the randomised
`EnclosurePcb` fixture. Existing config fields — no authors' code changes — but the authors assume
the two are the same entity.

⚠ **`receptive_asset_cfg` exists ONLY on the PartiallyAssembled success terms.** C1/C2/C3 have no
assembly check and reject the override with `ConfigKeyError: Key 'receptive_asset_cfg' is not in
struct`. Apply it only where it exists.

**Side effect, accepted.** `receptive_asset_pose` and `insertive_asset_in_receptive_asset_frame`
still reference the FIXTURE. Useful for the pick phase; the target needs no observation because it
is fixed.

---

## R6 — the seated start, via the authors' dependent-placement mechanism — APPROVED

**Problem.** Deployment starts with the jig seated on the fixture. The C1 recorder samples the jig
**independently** of the fixture (uniform xy, 12-50 mm drop), so it practically never produces a
seated state — and a drop at the fixture's location spawns *inside* the 36 mm-tall stack.

**Rejected first attempt.** `reset_root_states_uniform`'s `offset_asset_cfg` anchors to the
fixture's **default** pose, not its randomised one, and adds **position only** — the jig would not
rotate with the fixture's yaw. Verified in the source before recording anything.

**What is used instead.** `reset_insertive_object_from_partial_assembly_dataset`, which composes a
relative pose with the fixture's CURRENT pose, quaternion included. Fed an authored
`partial_assemblies.pt` holding exactly the seated relative pose — rel z **0.011563**
(= enc bottom -0.018037 + seat 0.0176 + jig half-height 0.012), yaw 0 and pi (2-fold symmetric
pillar pattern). Recorded with the stock
`OmniReset-UR10eLinearGripper-ObjectPartiallyAssembledEEAnywhere-v0` task.

Data preparation in the authors' file format; no code or config changes.
**Verified:** rel z **11.63 mm**, rel xy **0.9 mm**, fixture yaw std **92.7 deg** across 41 states.

⚠ Two traps, both measured:
* the recorder cannot infer the reset type from this task id -> pass `--reset_type` explicitly;
* `assembly_success_prob` (0.5 by default) requires half the envs to be NEAR the assembled pose,
  and the coin is only re-flipped after a *successful* episode — so envs drawing `True` when the
  state can never be near-goal absorb permanently and the recording freezes (observed: stuck at
  24/32, then 12/32, then 0/16). Set **`assembly_success_prob=0.0`** for the seated recording,
  which pairs with R5: "assembled" then means "at the parking spot", which a seated jig is not.

---

## R7 — fixture spawn x max 0.70 -> 0.62 — APPROVED

Task-specific narrowing; **y (+-0.24) and FREE yaw are unchanged**. Buys the clearance to the
world-fixed spot: fixture edge reaches x 0.719, parked-jig edge starts x 0.766 -> **47 mm gap**;
parked-jig far edge 974 mm against a 1050 mm table edge; reach to the spot 870 mm.

---

## R8 — C4's near-goal poses are authored, in the MARKER frame — APPROVED

`assembly_sampling_event` samples partial assemblies by starting at the assembled pose with
friction ZERO and random forces. On an open mat that just scatters the jig; the useful near-goal
distribution for parking is "held at or above the spot". Authored instead: 256 poses, +-35 mm
lateral, 12-82 mm high, free yaw, in the marker's frame.

**Verified:** recorded C4 sits **10-44 mm** from the marker (median 29.7) against the 50 mm
threshold — genuinely near-goal.

⚠ `compute_pair_dir` names the directory **INSERTIVE__RECEPTIVE**, so with the event retargeted at
the marker the file must live at `Resets/JigBlocked__ParkingSpot/`.

---

## Framing note — an assembly framework used for removal

OmniReset is built around insertive -> receptive ASSEMBLY; this task is the inverse. No code
changes were needed: `ProgressContext` success is generic frame-coincidence, so a "destination"
receptive object expresses removal perfectly well. Recorded as a novel use, not a modification.

---

## Inherited deviations (approved earlier, still in force)

* UR10e + custom linear parallel-jaw gripper instead of UR5e + Robotiq 2F-85 — project premise.
* **Hand-built box colliders** instead of SDF / convex decomposition. Physically forced: SDF jaw
  contacts fling the torque-controlled arm (measured |qd| to 576 rad/s); convex decomposition
  seats the skirt lopsided.
* **Free receptive yaw** (authors: +-15 deg) plus a **yaw gate in success** — jig task.
* Jig-v2's massless interior blocker machinery (this branch's base).

---

## Recorded sample verification (32-state targets, 2026-08-18)

| set | n | jig->marker (mm) | jaw gap (mm) | note |
|---|---|---|---|---|
| PartiallyAssembledEEAnywhere (seated) | 41 | 345 [252,453] | 136.8 open | rel z 11.6, xy 0.9 -> on the pillars |
| AnywhereEEAnywhere (C1) | 39 | 316 [200,444] | 136.8 open | rel z -6.0 -> flat on the mat |
| RestingEEGrasped (C2) | 33 | 323 [197,392] | **130.2** | straddle |
| AnywhereEEGrasped (C3) | 36 | 323 [190,447] | **129.7** | rel z 122 -> aloft |
| PartiallyAssembledEEGrasped (C4) | 32 | **29.7 [10,44]** | 136.8 | at the spot |

Zero states tilted beyond 20 deg in any set. Fixture spread x ~165 mm, y ~430 mm, yaw std
75-115 deg throughout.
