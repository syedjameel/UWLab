# The four reset substages, defined from the code

Source of truth: `UWLab-cubestack/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/
omnireset/config/ur5e_robotiq_2f85/reset_states_cfg.py`. Paper mapping is §3.2 of the OmniReset
paper; the repo names differ from the paper names and the repo names are what every command takes.

Each substage is a SEPARATE offline recording run producing one `resets_<Name>.pt` bank. At train
time all four are sampled `Uniform`, flat, 0.25 each (`rl_state_cfg.py`, `reset_from_reset_states`).
No curriculum, no ordering between them.

---

## Prerequisite banks (not substages, but required inputs)

| bank | producer | needed by | arm/gripper specific? |
|---|---|---|---|
| `partial_assemblies.pt` | `record_partial_assemblies.py` | C4 | **no** — pure relative pose, reusable |
| `Grasps/<obj>/grasps.pt` | `record_grasps.py` | C2, C3, C4 | **yes** — joint names + geometry |

Dependency order is therefore: partial assemblies -> grasps -> C1 -> C2, and C3/C4 any time after
grasps. C1 must precede C2 because C2 reads C1's bank for the object pose.

---

## C1 — `ObjectAnywhereEEAnywhere`  (paper: Reaching, S^R)

**The full task.** Every headline success rate in the paper is measured from this distribution
alone, and it is the only one that requires no prerequisite bank.

- cube: uniform in a box, `x (0.35,0.60) y (-0.2,0.2) z (0.0,0.3)` relative to the support,
  `bottom_offset` applied so "z=0" means resting on the surface. Roll/pitch jitter +/-0.1 rad,
  **yaw free** over the full circle.
- hand: IK'd to a uniformly sampled pose, `x (0.3,0.7) y (-0.4,0.4) z (0.017,0.5)`,
  pitch `(pi/4, 3pi/4)`, yaw `(pi/2, 3pi/2)`. Gripper open. Not related to the cube at all.
- The z floor is 0.017, not the paper's literal 0.0: this scene's mat surface sits 13 mm higher,
  and the literal value wedges fingertips into the mat.

## C2 — `ObjectRestingEEGrasped`  (paper: Near-Object, S^NO)

- cube: **replayed from the C1 bank** (`MultiResetManager`, `reset_types: [ObjectAnywhereEEAnywhere]`,
  probs 1.0) — i.e. resting somewhere reachable on the table.
- hand: placed at a grasp point drawn from the grasp bank, then perturbed by
  **+/-20 mm** in x/y/z and **+/-pi/16 (11.25 deg)** in roll/pitch/yaw.
- Gripper randomly open OR closed (paper §3.2). This is the "hand is at the object but has not
  necessarily got it" family.
- Hardest family to fill: the branch's own measurement on the leg pair was **0.83% acceptance**
  (40 of 4799).

## C3 — `ObjectAnywhereEEGrasped`  (paper: Stable Grasp, S^G)

- cube: same uniform box as C1 — including `z` up to 0.3, so **in the air**.
- hand: at a grasp point with **zero perturbation** (all `pose_range_b` ranges are exactly 0.0).
  The hand is exactly on the calibrated grasp.
- This is the only family whose cube starts unsupported. It is therefore the only one whose
  validity depends on the grip actually holding — the family the previous campaign found was
  silently dropping the cube one control step after reset.

## C4 — `ObjectPartiallyAssembledEEGrasped`  (paper: Near-Goal, S^NG)

- cube: drawn from `partial_assemblies.pt` with zero extra perturbation — a continuum of
  near-seated poses produced by spawning the cube AT the goal and dislodging it.
- hand: at a grasp point, perturbed by **+/-10 mm** and **+/-pi/32 (5.6 deg)** — half of C2's.
- Slowest bank to fill; the guide warns that under-recording C4 is what produces the
  "holds cube near the box, stuck" policy.

## Registered but never sampled

`ObjectPartiallyAssembledEEAnywhere` exists and is recordable (cube near goal, hand anywhere) but
is absent from the training mixture. Recording it is wasted time unless deliberately ablating.

---

## Pinch requirement — index tip vs thumb tip

The DG-5F metadata already declares exactly this jaw, so the request matches the calibrated
geometry rather than fighting it:

- `Robots/DeltoHand/metadata.yaml` header: *"a virtual jaw of finger 1 against finger 2"*.
- `grasp_center_offset: [0.0637, 0.0465, 0.1500]` is literally
  `midpoint(rl_dg_1_tip, rl_dg_2_tip)` under the closed posture. The 5-tip centroid was
  **rejected** because finger 5 sits at y = -68 mm.
- So finger 1 = thumb side, finger 2 = index. `rl_dg_1_tip` vs `rl_dg_2_tip` IS the pinch pair.

What is NOT yet a pinch: `DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS` (`ur10e_delto/actions.py:113`)
curls **all five** fingers (`*_4` all to ~1.5 rad). That is a whole-hand enclosure. A true
two-tip pinch needs fingers 3/4/5 held clear while 1 and 2 close.

Consequence to respect: fingers 3/4/5 default to the OPEN SPLAY (`_2 = 0.9`), which sticks them
into the workspace. Curling them away is safe for `grasp_center_offset` precisely because that
offset never referenced them -- but it must be checked against `enabled_self_collisions=True`.

---

## Pinch implementation — the exact change

The scripted close is a per-joint servo: `omnireset/mdp/scripted_gripper.py` builds
`_PER_JOINT_POSTURES = {joint: (open, close)}` from `DELTO_HAND_OPEN_JOINT_POS` and
`DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS`. There is no open/close command — every one of the twenty
joints is servoed independently. So a pinch is a new CLOSE posture, nothing more:

| finger | role | close target |
|---|---|---|
| 1 (thumb side) | **pinch** | unchanged from `SCRIPTED_CLOSE` |
| 2 (index)      | **pinch** | unchanged from `SCRIPTED_CLOSE` |
| 3, 4, 5        | retract   | curled clear of the 1-2 jaw line |

Fingers 3/4/5 must NOT simply be left at their open values: the open posture is the splay
(`_2 = 0.9`) which puts them *into* the workspace. They have to be curled toward the palm.

Also narrow the held-check so a five-finger enclosure cannot be scored as a pinch
(`dexlift/mdp/held_check.py:85-86` defaults):

    thumb_contact_names = ("rl_dg_1_tip",)      # was ("rl_dg_1_tip", "rl_dg_5_tip")
    tip_contact_names   = ("rl_dg_2_tip",)      # was ("rl_dg_2_tip","rl_dg_3_tip","rl_dg_4_tip")

Then the predicate is exactly "thumb tip loaded AND index tip loaded".

### Evidence the current banks are NOT a pinch
Read directly from `Datasets_ur10e_delto/OmniReset/.../resets_*.pt` (joint vector is 6 arm + 20
hand, level-major, so dims 21-25 are the `_4` flexion joints of fingers 1-5):

    C1 Reaching      _4 joints  0.31 / 0.77 / 0.77 / 0.82 / 0.72   <- open splay
    C2, C3, C4       _4 joints  1.49 / 1.43 / 1.49 / 1.49 / 1.51   <- ALL FIVE at the closed value

So every grasped bank in the repo is the full enclosure. A pinch requires regenerating the grasp
bank and C2/C3/C4. C1 does not involve the grasp and can be reused.

### Open risk, to settle in sim not on paper
`enabled_self_collisions=True` (`ur10e_delto.py:185`). The curled 3/4/5 posture must not
self-collide, and must not shift the approach axis that `gripper_approach_direction` encodes.
Precedent for getting hand geometry wrong: a stale `grasp_center_offset` cost 6144 attempts -> 0
successes with all 25 hand bodies reading 0.000 N.
