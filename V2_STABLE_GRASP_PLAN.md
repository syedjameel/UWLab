# V2 Stable-Grasp Expert — Plan (jig-enclosure)

**Status: PLAN, no code yet.** Does NOT disturb the current v1 pipeline — the current expert +
80k collection proceed as-is. This is a *from-scratch* v2 (Stage-1 → finetune → 80k) aimed at a
sturdier grasp. Created 2026-07-27.

## 1. Problem
- **task_0 (C1, reach + grasp from scratch) learned a ONE-SIDED rim pinch** (an RL shortcut). It
  completes the task (~0.90 success) but is less stable than the two-sided "straddle" grasp.
- **Tasks 1–3 start already grasped** from the grasp sampler's good (straddle) grasps → ~0.98.
  Those reset states are training scaffolding, not deployment states.
- **Key consequence:** at deployment the robot ALWAYS starts jig-on-mat, gripper open = the
  C1/task_0 scenario. So **the deployed grasp = task_0's learned grasp.** And the RGB student is
  distilled ONLY from successful demos → it imitates whatever the expert grasps. A fragile task_0
  grasp therefore caps deployment AND gets baked into the student.
- **Verified nuance / tradeoff:** RL likely chose one-sided because it's *easier with the slow
  jaw* (0.068 m/s) — a two-sided straddle needs wide, precise jaw placement. So forcing a straddle
  MAY lower task_0 success (stability vs learnability). **Goal = a grasp that is verified sturdier,
  not "two-sided for its own sake."**

## 2. Rejected alternative: "have both skills" (one- and two-sided)
Multi-modal single policy fails in this setup: (a) deployment runs the **deterministic mean**
action → the mean of a bimodal grasp is a mushy in-between that grasps nothing; (b) the student is
a **unimodal MLP regressor** → behavior cloning mode-averages the two grasps into one bad blend.
Only worth it with a real pose-dependent trigger AND a multi-modal student (diffusion / discrete
grasp-selector) — not our current architecture.

## 3. Approach — the grasp-forcing trio (+ safety nets)
1. **Interior collider (negative constraint).** Add a **massless, visually-invisible** collider
   filling the jig's interior, height ≤ rim (24 mm). A jaw then cannot enter the middle → the
   one-sided pinch is PHYSICALLY IMPOSSIBLE from iteration 1. More reliable than reward-shaping,
   which may not escape the one-sided basin RL is already stuck in.
2. **Pre-grasp C1 seeds (positive guidance).** ~20–30% of C1 resets start with the EE at a sampled
   good straddle pose, **jaws OPEN**, jig on the mat. RL learns "this pose + close = success"; the
   far-reach C1 states are then pulled toward it (value bootstrapping).
3. **Honest metric.** Track task_0 success on the realistic **EE-anywhere subset separately** —
   the pre-grasp seeds inflate the aggregate (same trap as the realpcb "task-3 = 90% from
   iteration 1"). The EE-anywhere number is the real one.
4. **Grasp-stability diagnostic.** Measure per grasp: both-jaw contact, rim location, and jig
   slip/rotation relative to the gripper during carry. Use it to (a) quantify v1's one-sided
   instability (is the redo worth it?) and (b) confirm v2's grasp is measurably sturdier.
5. **(Optional) mild stability reward** on top, so among straddle grasps RL picks a firm one, not a
   precarious wide one.

## 4. Sim2real reasoning (why the collider is safe)
- The forced straddle grips the OUTER walls — it never uses the middle. The same motion works on
  the real jig (open middle). The collider is a **training scaffold**, absent in real; the policy
  just learns "don't reach into the middle," which is safe to not-unlearn.
- Keep the collider inside the interior footprint and ≤ rim height, so a straddle approach
  (descending OUTSIDE the frame) never contacts it → no non-transferable approach dynamics.
- Keep it present through RGB collection too (consistent straddle demos for the student); its
  absence at real deployment is fine.

## 5. Gotchas (each bit us before on this jig)
- **Mass:** jig mass is auto-computed from colliders (no MassAPI). Make the interior collider
  MASSLESS (or pin the jig mass), else the dynamics change silently.
- **Seat:** pillars enter the CORNER cone-holes; a center collider should be clear — VERIFY in the
  mate test it doesn't foul seating (7/7).
- **RL still cheats:** blocking the middle removes THIS shortcut; RL may find a new one (top-edge,
  corner). Still verify the resulting grasp visually + with the diagnostic.
- **Ratio:** pre-grasp seeds must stay a MINORITY of C1, or the policy gets lazy (nails
  close-from-pose, under-learns the far-reach that deployment needs).
- **RGB distribution:** pre-grasp seeds ON for RL training, OFF/tiny for the 80k collection (demos
  must reflect the real EE-anywhere start distribution) → likely a seed flag so collection can
  exclude them.
- **Consistency:** re-run the grasp sampler WITH the collider → C2–C4 become straddle too → the
  whole task is consistent.

## 6. Execution order (after v1's 80k is banked)
1. **Grasp-stability diagnostic** on the current v1 expert → quantify the instability gap → decide
   if v2 is worth it.
2. Add the massless interior collider to the jig USD; **verify the mate still seats 7/7**.
3. Re-run **grasp sampling with the collider** → straddle grasps; re-record C2–C4 +
   partial-assembly resets.
4. Generate **pre-grasp C1 seeds (~25%)** from the new grasps; record/assemble C1 (keep a seed flag
   so RGB collection can exclude them).
5. Add the **honest EE-anywhere-subset metric** (+ optional mild stability reward).
6. **Stage-1 → finetune** (`entropy_coef=0.001`, BIGBUF) → verify grasp (diagnostic) → **80k** →
   distill → **compare v2 vs v1 student**.

## 7. Open items (verify, don't assume)
- Confirm the one-sided grasp actually reaches into the middle (so blocking it forces straddle) —
  diagnostic + video.
- Confirm the forced straddle is actually sturdier (not just wider/different).
- Decide the pre-grasp seed fraction empirically.
- Decide whether the mild stability reward is needed.

## 8. Deviations to log (authors-faithful ledger)
Interior collider (sim-only training scaffold), pre-grasp C1 seeds, honest EE-anywhere metric,
(optional) stability reward. All are jig-v2-specific; none touch the shared authors' code paths by
default.

---

# BUILD LOG — step 2 done (2026-07-28, branch `omnireset/jig-v2-stable-grasp`)

**Step 1 (grasp-stability diagnostic on the v1 expert) was SKIPPED by explicit user decision.**
Consequence to remember: there is no v1 baseline measured with the same tool, so when v2 finishes
we can claim "v2 grasps as a straddle" but NOT "v2's grasp is measurably sturdier than v1's". The
v1 expert is still on disk if we ever want it: `logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/
2026-07-27_08-15-21/model_5600.pt` (entropy_coef=0.001, scale_progress 1.0, task_0 0.913,
tasks 1-3 0.980/0.993/0.962).

## What was built
* `scripts_v2/tools/build_jig_enclosure_usds.py`: new `--v2-jig` flag, `_JIG_INTERIOR_BOXES_MM`,
  `_add_interior_blocker()`, `_reveal_colliders()` factored out. Writes
  `Props/Custom/JigV2/jig_v2.usd` (+ `metadata.yaml`). v1's `jig.usd` / enclosure are NOT rebuilt.
* `jigv2` variant registered in **all four** variants dicts (each stage has its own):
  `reset_states_cfg`, `partial_assemblies_cfg`, `rl_state_cfg`, `grasp_sampling_cfg`.
* `scripts_v2/tools/conversions/verify_jig_v2_mass.py` — new keeper diagnostic.

## Measured facts (verify-don't-assume)
| fact | value |
|---|---|
| jig interior void (containment cross-sections of jig.stl) | tiered: z 0-9.5 mm -> x +-70.5, y +-50.5; z 9.5-24 -> x +-72.5, y +-52.5 |
| blocker volume vs jig wall collider | 344,794 vs 125,875 mm^3 (**+274%**) |
| jaw inner-face gap at OPEN (tip.stl + URDF frames) | **136.76 mm** -> 3.88 mm clearance/side on the 129 mm jig |
| jaw pad thickness along the closing axis | **28 mm** (cannot enter any residual slot) |
| blocker vs enclosure at the seat | 0 hard overlaps, closest approach **3.64 mm** (post tower) |
| runtime mass/inertia v2 vs v1 | **bitwise identical** (0.125875130 kg; rel diffs 0.000e+00) |
| mate check | **7/7**, all verdicts match v1, seated depths within 0.1 mm |

## Blocker design (why two tiers)
The **upper** tier (z 9.5-24, x +-72.5, y +-52.5) is flush with the upper inner walls and caps the
whole interior mouth — it alone does the blocking, since anything descending from above hits it.
The **lower** tier (z 0-9.5) is deliberately INSET to x +-68 / y +-48; it only plugs residual
volume, so insetting is free and it raised seat clearance from 1.00 mm to 3.64 mm. The ~2.7 mm
slot the inset opens is unusable (28 mm jaw pad).

## DEVIATION 1 — massless blocker needs `override_mass=False`
`make_insertive_object`/`make_object` gained an `override_mass: bool = True` parameter; when False
`mass_props` is omitted. `jigv2` passes False. **Why:** the blocker prims must carry a `MassAPI`
(near-zero density 1e-9; density 0 is treated as unauthored and ignored), but IsaacLab's
`modify_mass_properties` is `apply_nested` — it walks EVERY nested prim and writes `mass=0.001`
onto each one carrying the API, and an explicit mass overrides density. Measured: jigv2 came out
+2 g (= 2 blocker prims x 1 g), +1.6% mass / +0.6% inertia. Confirmed by the warning appearing for
`/World/Jig_v1` but NOT `/World/Jig_v2`. This is the trap documented in
`omnireset_asset_utils.create_stage`, reached from the child side.
**Not a behavioural deviation for v1 assets:** their roots have no MassAPI, so `mass_props` never
applied anyway (it logs "Could not perform 'modify_mass_properties'"). Default stays True; every
v1 entry is untouched.

## Findings that CHANGE the plan
1. **Grasp re-sampling will not change the grasps.** The sampler reads the VISUAL mesh
   (`_find_mesh_in_prim` returns the first mesh in traversal = `/Jig/visuals/mesh`; verified for
   v2), and it already rejects the X face pair (164 mm > `maximum_aperture` 0.13), so it *only*
   ever emits Y-closing two-sided straddles. **C2-C4 were already straddle in v1.** Re-recording
   is still required (the loader wants `Grasps/JigV2/grasps.pt`) and re-validates each candidate
   in physics, but expect ~v1 statistics, not new grasps. The blocker's whole effect is on task_0.
2. **Datasets auto-namespace.** Paths derive from the USD's PARENT DIRECTORY
   (`object_name_from_usd`), so v2 writes `Grasps/JigV2/` and `Resets/BottomEnclosure__JigV2/`.
   v2 recordings cannot overwrite v1's `Jig` / `BottomEnclosure__Jig` even in a shared dir.

## OPEN CAVEAT — near-miss behaviour differs from v1 (and from the real jig)
In the deliberately-out-of-capture `+12 mm x` config the jig settles at rel z **23.2 mm on v2 vs
16.6 mm on v1**: with the interior open (v1, and the REAL jig) a misaligned jig sinks into the
enclosure cavity; the blocker stops it, so it rests ~6.6 mm higher on the pillars. Both correctly
report success=False, and every config INSIDE the ~8 mm capture basin is identical to v1, so the
training signal is unaffected — but this is a real sim-only artefact in the near-miss regime that
the plan did not anticipate (it only reasoned about the grasp phase and the perfect seat).
If near-miss fidelity turns out to matter, the fix to test is DROPPING the lower tier entirely
(the upper tier alone still blocks every approach from above); that needs re-verification of the
mate + a check of what the upper tier then contacts.

---

# v2 STAGE-1 RESULT + v2b (2026-07-29)

**v2 Stage-1 (`logs/.../2026-07-28_12-33-34`, 16384 envs, entropy_coef 0.006) did not work.**
The full-scale v2 datasets were fine (C1 11214 filtered / C2 10091 / C3 13102 / C4 10392, 6764
grasps -- C2 empty grips even improved 8.8% -> 1.2% vs v1).

| | v1 Stage-1 | v2 Stage-1 |
|---|---|---|
| task_0 first non-zero | iter ~405 | iter ~1046 |
| task_0 @1000-1099 | 0.735 | **0.003** |
| task_0 growth off the same base | 5.6x /100 iters | ~1.7x /100 iters |
| total_fps | 10807 | 6951 (1.56x slower) |

**Diagnosis: grasp DISCOVERY is the only bottleneck.** Reaching is unaffected
(`ee_asset_distance` reward identical to v1) and tasks 1-3 reach 0.92/0.94/0.86, so the straddle
holds once achieved. Aligned by task_0 progress rather than iteration, v2's tasks 1-3 are AHEAD
of v1's (v2 @task_0=0.003: 0.910/0.925/0.850; v1 @task_0=0.001: 0.621/0.699/0.670) -- they are
gated by task_0 feeding ~99.5% failures into the shared network, not by any asset defect.
The blocker removed the shortcut without offering a route to the replacement, which is exactly
what the plan's §5 trio pairs it with. **Two hypotheses were tested and REFUTED:** a shrunken
seating capture basin (`sweep_capture_basin.py` gives v1 and v2 the same map, and v1 trains to
0.97 with it) and "tasks 1-3 plateaued below v1" (they were still climbing; v1 decelerated the
same way -- task_1 took 161 iterations to go 0.92 -> 0.95).

## v2b -- lower tier dropped (`--v2b-jig`, variant `jigv2b`)
The lower tier contributes NOTHING to blocking (the upper tier caps the whole mouth; below
z 9.5 the jig's own walls are solid all round) but costs, measured:
* **fps 1.56x** -- its bottom face is coplanar with the jig underside, so a resting jig presents a
  solid ~136x96 mm patch to the mat where v1 had a thin wall rim;
* **`abnormal_robot` penalty 3.2x** (-0.0013 vs -0.0004);
* the only part that can foul the enclosure: overlaps the corner pillars at x offsets >=5 mm
  (upper tier alone: 0 overlaps at every offset tested, both axes).

Verified: mass/inertia/COM **bitwise identical** to v1; mate **7/7**.

**Seating fidelity vs v1, measured properly** (`sweep_capture_basin.py`, offsets in the ENCLOSURE
frame at yaw 0 -- unlike `visualize_perfect_mate`, whose single `+12 mm x` config applies the
offset in the WORLD frame under a random-but-seeded enclosure yaw, so its number samples one
arbitrary direction and is NOT a fidelity metric):

| region | v2b - v1 settled height |
|---|---|
| within 8 mm radius (the capture basin) | **max 0.00 mm -- exact** |
| on-axis ring 9-12 mm | 0.00-0.27 mm |
| extreme diagonal corners (~15-17 mm) | up to 25.7 mm |
| success map, whole +-12 mm grid | **identical** |

So v2b reproduces v1 EXACTLY wherever seating can succeed. The only divergence is at extreme
diagonal misalignment, where v1 lets the jig collapse ~25 mm into the shell and v2b holds it up.
**This cannot be tuned away:** at those offsets the enclosure's pillars pass up into the jig's
interior -- the very volume the blocker exists to occupy -- so any blocker that stops a jaw
entering the middle also stops the shell entering it. Raising the blocker's floor above z 9.5
would open the wall notch (|x|<28, z 9.5-24) to SIDEWAYS jaw entry and reintroduce the one-sided
pinch. Inherent trade, confined to states 2x outside the capture basin that fail in both.

## Pre-grasp C1 seeds (`reset_end_effector_pregrasp_seeds`)
The positive half of the trio, and the intended fix for the discovery failure. Re-places a
Bernoulli(`seed_prob`) MINORITY of C1 resets at a recorded straddle pose with the jaws OPEN, so
the policy only has to close them. `seed_prob=0.0` (default) is an exact no-op and does not even
open the grasp dataset, so every other object/pipeline is untouched. Seeded env ids are published
on `env.pregrasp_seed_mask`.
* Enable with `env.events.reset_end_effector_pregrasp_seeds.params.seed_prob=0.25`.
* **Record the RGB-collection / honest-metric C1 with `seed_prob=0.0`** -- seeded states do not
  reflect the EE-anywhere start distribution deployment sees, and they inflate task_0.
* Verified on a 414-state smoke: jaws-shut fraction fell 50.2% -> **37.7%**, exactly the
  predicted 0.5 x 0.75 = 37.5% for 25% seeding.
* GOTCHA fixed: the 2F-85 gripper regex includes `.*left.*`, which matches NOTHING on the linear
  gripper (`finger_joint` + `right_finger_joint`) and raises "Not all regular expressions are
  matched" at term-parse time. `linear_gripper_cfg._apply_linear_gripper` now patches this term
  alongside `reset_end_effector_pose_from_grasp_dataset`.

## Not yet done (next steps, unchanged from §6)
Re-run grasp sampling -> C2-C4 + partial assemblies into `Datasets_jig_v2` -> pre-grasp C1 seeds
(~25%, with a flag so RGB collection can exclude them) -> honest EE-anywhere metric ->
Stage-1 -> finetune (`entropy_coef=0.001`, BIGBUF).
**Watch:** the blocker is a large box, so near the enclosure it adds broadphase pairs (14 within
5 mm proximity at the seat vs 50 for the walls). Since the enclosure collider was hand-built
precisely because the GPU contact stack blew up, v2 may need a LARGER BIGBUF than v1 at
4096/16384 envs. Unverified — find out at reset-recording time.
