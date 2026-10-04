# The DELTO pinch: what was wrong, what was fixed, what is left

> **The largest finding is not about the gripper.** Three of the four reset families the policy
> trains on were never grasps. Measured over 40 states each, the cube sits a median of 281 mm
> from the hand in `ObjectRestingEEGrasped` (0/40 within grasp range) and 397 mm in
> `ObjectAnywhereEEGrasped` (3/40, worst case 792 mm), against a real grasp distance of ~170 mm.
> A policy resetting into those families starts each episode with the cube in mid-air or on the
> table and the arm somewhere else. No amount of gripper tuning makes that learnable, and it was
> invisible because nothing in the pipeline ever checked.
>
> Verified three ways: forward kinematics; the replay film, in which the C2 cube sits on the mat
> with no hand near it; and this repo's own `grip_probe.py`, which records that C3 "was silently
> dropping the cube one control step after reset". The replay itself was checked first -- the
> scene orders joints `[6 arm, then hand level-major]`, matching the stored 26-vector, and C1
> replays with the cube resting correctly.
>
> `banklab.py --verify-grasped` now makes this an exit code and runs after every family records.

Report (videos, tables): https://mattecache.tailcc6bd2.ts.net/reports/pinch_grasp.html
Bench code: `cube-stack/pinch/`. Repo changes: `UWLab-cubestack` commits `cd8d2a6`..`0d23b31`.

## Why grasp yield was zero

Two causes, neither of them friction, stiffness or effort limits.

**1. The commanded "closed" hand is a fist, not a jaw.** `DeltoHand/metadata.yaml` calls the
hand's grasp geometry "a virtual jaw of finger 1 against finger 2" and sets
`grasp_center_offset` to the midpoint of those two fingertips at
`DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS`. Measured in sim at that posture the two pads finish
**51.03 mm** apart against a 34 mm cube. The declared grasp centre is arithmetically right and
physically the middle of a closing fist. Tucking fingers 3/4/5 with
`DELTO_TUCKED_HELD_FINGER_JOINT_POS` changes the pad separation by **0.00 mm** — it was never
about the other three fingers.

A sweep of 122 880 random postures over the eight joints of fingers 1 and 2 reaches pad
separations down to 1.35 mm, so the hand can oppose those pads. It is simply not asked to.

**2. The controller cannot reach the posture it commands.** Every closing path, recorders
included, goes through `RelativeJointPositionAction` with `scale=0.1`, `clip=(-1,1)`:
`target = measured + clamp(a, -1, 1) * 0.1`. Torque is therefore bounded by `stiffness * 0.1` —
0.0170 N·m on `rj_dg_1_4`, against a 0.06 N·m effort limit that is never reached.

| control | friction | stiff4 | thumb finishes short by |
|---|---|---|---|
| relative term (as shipped) | 0.01 | 1 | **0.844 rad** |
| absolute position target | 0.01 | 1 | 0.060 rad |
| relative term | 3e-4 | 1 | 0.001 rad |
| relative term | 0.01 | 10 | 0.005 rad |

An interaction, not an independent fault — but the combination that stalls is the one that
ships. It is also why `friction` looked guilty: it is a dimensionless coefficient and the
comment in `ur10e_delto.py` calls it "0.01 N·m".

## The pinch

Derived by CEM over all twenty joints (`pinch/pinchpose.py`), using the sim as an FK oracle:

* pad separation 30 mm closed / 50 mm open,
* **every** non-pad hand body ≥ 36 mm from the grasp point — a 34 mm cube has a 29.4 mm
  half-diagonal, and this is the requirement that forces a pinch rather than a fist,
* grasp-point drift between the two postures 0.00 mm, so closing squeezes rather than sweeps.

Drop the clearance requirement and the optimiser returns fist-like postures that score well on
separation and then carry the cube on the curled fingers (thumb 2.22 N, index 0.00 N, fingers
3/4/5 9.97 N). A scoop, which passes any distance-based "is it held" test.

The grasp point moves ~35 mm from metadata's value. `DELTO_PINCH_GRASP_CENTER_OFFSET` carries
the new one, and the sampler cfg sets the posture flag and the grasp centre **together**.

## Squeeze depth

The cube is 34 mm. "True pinch" = held AND both pads loaded by the cube AND the other three
fingers carrying nothing.

| commanded pad separation | held | true pinch | both pads loaded |
|---|---|---|---|
| 18 mm | 19% | 18% | **72%** |
| 20 mm | 51% | 8% | 10% |
| 22 mm | 28% | 20% | 30% |
| 25 mm | 29% | 7% | 16% |
| 28 mm | 89% | 1% | 3% |
| 30 mm | 90% | 0% | 0% |

18-22 mm is the working band. Shallower separations hold more often but only ever load one pad,
which is not a pinch.

## One protocol result worth keeping

Holding the cube immovable while the hand closes, then releasing it, scored **0% true pinch in
every cell**. Letting it float free and weightless during the close scored 25–42%. The pin
stores contact energy and dumps it into the part on release. That is the probe, not the grasp —
and it is exactly what would be read as "the hand ejects the cube".

## Two measurements that were wrong

* **Peak contact force is not grip force.** Max over the episode captures penetration
  transients and read 15–35 N on a hand limited to 0.06 N·m. Replaced by the mean over the hold
  window.
* **Fingertip contact included self-contact.** `enabled_self_collisions=True`, so an unfiltered
  net force counts the hand pressing on itself: tips 3/4/5 read 6.76 N while sitting 36 mm from
  the centre of a cube with a 29.4 mm half-diagonal. Contacts are now filtered to the cube, one
  sensor per fingertip (PhysX sizes the filter list as bodies × envs, so a five-body sensor with
  one cube filter fails to initialise).
* The hold radius was also too generous at 35 mm — a cube that had squirted out and finished
  32 mm from both pads still scored as held. Now 20 mm, with contact as the decisive test.

## Bank defects found along the way

* `partial_assemblies.pt` has 256 rows and **one** unique value: relative position exactly
  (0, 0, 34 mm), identity rotation. The paper's §3.2 wants a continuum from dislodging the part
  with small random forces; the dislodging term is `None` for the cube pair and no generic
  replacement exists. C4 is built from one already-solved state.
* C2's gripper is never opened. The paper specifies "randomly open or closed";
  `record_reset_states.py` forces the close mask all-true for any task matching `EEGrasped`.

## Grasp sampler: how far it got

`UWLAB_DELTO_PINCH=1` + `record_grasps.py --task OmniReset-Delto-GraspSampling-v0
--num_envs 64 env.scene.object=cube34`. Three failures, two fixed:

1. `ValueError: the term 'grasp_sampling' expects ... but received [... grasp_center_offset]`.
   The EventManager validates `cfg.params` against `__call__`'s signature. Fixed by declaring
   the key there (commit `4a21679`).
2. `RuntimeError: placement collision gate rejected all 61440 grasp candidates`. The grasp
   centre had moved to the pinch's but the OPEN posture had not — the metadata splay placed at
   the pinch's grasp point buries fingers 3/4/5 in the cube. Added a `gripper_open_joint_pos`
   override (same commit).
3. Same rejection, and the gate's own diagnostic named the reason: `rl_dg_2_tip` sitting
   **14.44 mm inside the cube**. The three direction vectors in metadata are also fist-derived,
   and `grasp_align_axis` is the CLOSING axis the sampler aligns with the object's width — point
   it wrong and the hand is rotated so a pad enters the part. Derived the pinch's axes by FK
   (`pinch/pinchaxes.py`) and made them overridable (commit `f91e60d`).

   That derivation was validated against the fist first: recomputing the FIST's grasp centre the
   same way reproduces metadata's `[0.0637, 0.0465, 0.1500]` exactly, and its align axis to a
   dot product of −0.982 (same line; metadata's sign convention is `tip1 − tip2`).

4. **Still rejecting all 61440**, deepest intersection improved 14.44 → 12.73 mm, same body.
   The remaining cause is not another cfg key: `_filter_placement_collisions` samples the
   gripper's collision geometry at the articulation's *current* joint state — the spawn default
   splay — not the pre-grasp posture the override supplies. **Posing the hand before the gate
   runs is the next change.** It was not made because it could not be validated in the time
   left.

## Left to do

1. **Get past the placement gate** (item 3 above), then regenerate the grasp bank and check its
   yield before touching C2/C3/C4. Everything writes to
   `Datasets_ur10e_delto/OmniReset_pinchon/`, so nothing existing is at risk.
2. **Raise the true-pinch rate** above ~20%. The 18 mm posture loads both pads 72% of the time
   but keeps the cube only 19% — the loss is between first contact and the end of the shake.
3. **Camera.** Isaac Lab's `Camera` sensor cannot be aimed in this build:
   `set_world_poses_from_view` never moves the prim, `CameraCfg.OffsetCfg` applies translation
   but not rotation, and a direct USD xform write is overwritten by the sensor on its next
   update. All three report a correct pose on readback while rendering the sky. The pinch videos
   work only because that scene's hand happens to sit where the unaimed camera looks. The
   replicator path (`rep.create.camera(position=, look_at=)`) does aim, but renders the table
   scene 180° rolled. Until this is settled, no new scene can be filmed.
4. Fix the ~21 mm first-step pop when a bank state is replayed — the replay scene does not match
   the recording scene.
