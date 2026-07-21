# realpcb thin-object grasp — deviation ledger

Branch: `omnireset/realpcb-thin` (off `omnireset/ur10e-custom-table`).
Goal: grasp a real **140 × 100 × 3 mm** PCB (`realpcb`) with the linear parallel-jaw gripper.
Governing rule: follow the OmniReset authors' implementation; deviate only when physically
forced, and log it here for approval.

## Empirical findings (this sub-task, laptop RTX 3060)

All runs use the author asset pattern (`omnireset_asset_utils.create_stage`: baked friction
material, **no** `MassAPI` → PhysX auto-computes mass; applying `MassAPI` would make it 1 g and
fling it — the documented trap). `realpcb.usd` built via `scripts_v2/tools/build_realpcb_usd.py`.

| Stage | Task | Result |
|---|---|---|
| Grasp sampling | `OmniReset-LinearGripper-GraspSampling-v0` | **251/256 = 98%** |
| Reset (anywhere) | `...ObjectAnywhereEEAnywhere-v0` | 24 recorded, PCB integrates with table |
| Reset (grasped) | `...ObjectRestingEEGrasped-v0` | **12/16 = 75%** (stable, held) |

**The handoff premise ("records ~0 grasps/resets") is disproven** — the thin PCB records at
healthy rates. Why grasp sampling passes despite the 3 mm edge: the object is reset to **z = 0.3 m,
floating** (`grasp_sampling_cfg` `reset_object_position` z=(0.3,0.3)); there is no table to foul,
so a flush side-face pinch (jaws close across the 100 mm axis onto the two 140×3 mm faces; the
140 mm axis is skipped since it exceeds `maximum_aperture = 0.13`) is stable.

**The real defect (measured):** jaw-tip world-Z vs the table surface across the 16 recorded
"successful" grasped resets — median ≈ 0, but **8/16 have the jaw tip buried below the table**,
several by **22–25 mm**. The reset success check (`terminations.py` `check_reset_state_success`)
never guards this: `above_ground` checks the object root and the gripper *base* (`robotiq_base_link`,
~0.14 m above the tip), and `collision_free` checks gripper-vs-object only — neither checks the
fingertips vs. the table. So the dataset is contaminated with grasps the real robot cannot execute
(the real jaw cannot enter the solid table).

Two contributors:
1. **Standoff geometry.** Gripper metadata: `finger_offset = 0.13`, `finger_clearance = 0.01`
   → standoff sweep 0.13–0.14 m; jaw tip at gripper-local Z = 0.144 m. The grasp centre is placed
   at the object centre (1.5 mm above the table for realpcb), so straight-down the tip lands
   ~2.5–12.5 mm below the table. For a 40 mm cube the tip stays ~6–16 mm *above* the table, so this
   only bites thin objects.
2. **Reset-time pose jitter.** `ObjectRestingEEGrasped` applies `pose_range_b` roll/pitch/yaw of
   ±π/16 (±11.25°). Tilting the 0.144 m finger dips one corner deep below the table — this is the
   source of the 22–25 mm burials.

## Deviation D1 (APPROVED — "object-aware standoff") — IMPLEMENTED

Make the top-down sampler keep the jaw **tip at/above the object's bottom face**, so a table-resting
object is gripped at the fingertip instead of the finger being driven past the object bottom.

- **`Robots/LinearGripper/metadata.yaml`**: add `finger_tip_offset: 0.144` (jaw-tip offset from
  `robotiq_base_link` along +Z; already documented there as the grip-zone tip).
- **`mdp/events.py` `grasp_sampling_event`**: read `finger_tip_offset` (default
  `finger_offset + finger_clearance`, so the antipodal 2F-85 path is unaffected).
- **`mdp/events.py` `_sample_topdown_grasps`**: replace the fixed standoff sweep with an
  object-aware one — `ideal = finger_tip_offset − ext[2]/2` puts the tip at the object bottom;
  sweep `[max(finger_offset, ideal), min(finger_tip_offset, ideal + finger_clearance)]`.
  **Reduces exactly to the original 0.13–0.14 sweep for thick objects** (e.g. the cube:
  `ideal = 0.124 < finger_offset`), so only thin objects change.

Physical justification: a bare 3 mm board on a table can only be edge-grasped with the fingertip at
the table surface; the original sweep makes that impossible (all standoffs bury the tip). Faithful
to the authors' top-down design (which is itself the project's linear-gripper addition), and a
no-op for the previously-validated thick objects.

## Deviation D2 = "C" (APPROVED — B + A) — reset augmentation + success floor

### Experiments (sweep_thin_reset_augmentation.py, realpcb, 128 samples/config, tip vs FIXED table top 0.004)

`pose_range_b` is in the gripper BODY frame; local +Z is the (downward) approach, so a POSITIVE z
draw drives the tip DEEPER. Biasing z NEGATIVE (shallower) + cutting roll/pitch is the lever:

| reset config | good-window % | buried >2 mm % | med clr mm |
|---|---|---|---|
| baseline ±2cm / ±11.25° | 10.9 | 38.3 | 1.6 |
| none (0 jitter) | 50.0 | 38.3 | 0.3 |
| **z[-1cm,0] / rp±3° / yaw±11°** | **48.4** | **10.9** | 3.2 |
| z[-2cm,0] / rp±3° | 19.5 (lifts object off table) | 7.0 | 11.0 |
| z[-5mm,0] / rp±1.5° | 47.7 | 28.1 | 1.4 |

Findings: (1) shallower-only z + ±3° roll/pitch cuts deep burial 38%→11% with no loss of good
grasps; yaw is height-neutral so kept full. (2) too-shallow (−2 cm) lifts the object off the table
(invalid resting state). (3) an irreducible ~11% deep-burial floor remains from the recorded grasps'
own ±6° sampler wobble — this is what A must backstop.

### B (IMPLEMENTED) — object-aware reset augmentation for thin insertive objects
`reset_end_effector_from_grasp_dataset.__init__` (events.py): infer object thickness from its
metadata `bottom_offset`; when thin (< 20 mm) clamp `pose_range_b` z to `[-0.01, 0]` (shallower
only) and roll/pitch to ±3°. yaw / x / y unchanged (height-neutral). No-op for thick objects
(cube 40 mm) and for non-thin tasks.

### A (IMPLEMENTED) — fingertip-vs-table floor in the reset success check
`check_reset_state_success` (terminations.py): opt-in params `fingertip_offset` (0.144),
`table_top` (0.004), `fingertip_clearance_tol` (0.001); reject a state if the jaw tip
(robotiq_base_link + offset along +Z) is below `table_top - tol`. Disabled by default (params None)
-> no-op for the 2F-85. Enabled on the UR10e linear-gripper EEGrasped reset tasks via
`_enable_fingertip_floor` (linear_gripper_cfg.py), and **gated to THIN insertive objects** at runtime
(`fingertip_thin_only`, threshold 20 mm) so the 40 mm cube is entirely unaffected. Turns the silent
contamination into an explicit filter.

### Cube safety (verified)
Both B and A are gated on insertive-object thickness < 20 mm (`_infer_thickness` / `_infer_object_thickness`
read the object's metadata `bottom_offset`). cube(pcb)=40 mm -> inactive; realpcb=3 mm -> active.
D1 standoff sweep is byte-identical for the cube (ideal 0.124 < finger_offset). Empirical grasp-sampling
proof: cube 214/256=83.6% (original code) vs 213/256=83.2% (D1) -> unchanged within RNG.

### Verification of C (realpcb, ObjectRestingEEGrasped, laptop)
Before any fix: 75% recorded, but ~50% of "successful" states had the jaw tip buried up to 25 mm.
After C: **84% recorded (27/32), tip_ok=16/16** every batch -> B keeps all tips above the table, A
rejects the residual (0 here, active as a hard floor). Yield improved and the recorded dataset is
realizable by construction.

## Near-goal / realopenbox (added on user request)
The deployed `openbox` fits the 40 mm cube, not a 140x100 PCB, so insertion (C4) needs a
PCB-sized receptacle. NEW `Props/Custom/RealOpenBox/realopenbox.usd` via
`scripts_v2/tools/build_realopenbox_usd.py` (cavity 150x110 mm, walls 4 mm, depth 10 mm; metadata
offsets + success_thresholds auto-written). Wired `realopenbox` receptive variant into
`rl_state_cfg.py`, `reset_states_cfg.py`, `partial_assemblies_cfg.py`. The PCB partially assembles
into it (23 partial poses recorded). Box dims are first-pass defaults -- adjust to the real
receptacle via the build script's `--clearance/--wall/--depth` args.

## Small-scale datasets recorded (./Datasets_realpcb/OmniReset, pair RealOpenBox__RealPcb)
grasps 429 | partial_assemblies 23 | C1 AnywhereEEAnywhere 151 | C2 RestingEEGrasped 164 |
C3 AnywhereEEGrasped 123 (accept ~11%) | C4 PartiallyAssembledEEGrasped 80 (accept ~6%). These are
small VISUALIZATION runs; re-record at scale (~10k each) before training.

## Open QC findings (from visual inspection + .pt analysis; DECISIONS PENDING)
1. **Dual-drive gripper asymmetry.** Recorded grasps are symmetric (<1.2 mm jaw diff), but the
   carried-object sets are asymmetric: C3 median 2.4 mm / max 5 mm / 61% of states >2 mm. The two
   jaws are independent prismatic drives with no linkage, so under asymmetric load they settle
   off-center -- a real single-motor rack-pinion gripper self-centers. Fix = a mimic/coupling
   constraint (deviation; ties to [[gripper-jaw-width-deferred]]).
2. **Object sunk into table.** ~half of C1/C2 resting states have the PCB ~3-5 mm below the table
   surface (the `above_ground` floor is a loose -0.02 m). Consider tightening for thin objects.
3. **C3/C4 low accept + flying** (11% / 6%): a flat plate carried at arbitrary poses on a 3 mm
   pinch is inherently unstable; C4 also limited by fingers hitting the box walls (coll_free).

## Deviation D3 (IMPLEMENTED) — wrist-camera settle anchor (branch-wide, all objects)
Not realpcb-specific: applies to all UR10e linear-gripper state training. After the -90deg joint-1
sim<->real remap, the wrist camera settles ~90deg off (toward -Y / "old +X"), not toward the
operator (+X). ROOT CAUSE (verified, not assumed): the authors' success check IGNORES yaw
(`euler_xy_distance` = roll+pitch only, commands.py:149), so wrist/camera yaw is unconstrained BY
DESIGN. Empirical proof by playing exported actors:
- Authors' 2F-85 expert (model_7600, 2026-06-18, clean main): settle camera SPREAD across 0-180deg,
  concentration R=0.45 -- the authors NEVER settled +X. So there is no author mechanism to restore.
- Our UR10e expert (model_5300): settle camera tight at -97deg (R=0.87), i.e. -Y. wrist_3=-113deg,
  (camera_heading - wrist_3) offset = +24.4deg at the top-down settle -> **wrist_3 = -24deg gives
  camera +X** (measured for the D405 mount; NOT the -90deg "home" the old reverted RGB window
  assumed -- that would park the camera at -66deg).

Fix: `joint_outside_window` DoneTerm on wrist_3, window **[-1.47, +0.62] rad ([-84,+36]deg)** =
+-60deg of the measured -24deg, wired into `Ur10eLinearGripperRelCartesianOSCTrainCfg` (Stage-1) and
`...FinetuneCfg` (Stage-2) via `_enable_wrist_camera_anchor` (linear_gripper_cfg.py). Only successes
are rewarded, so out-of-window episodes are discarded -> the policy learns to keep the camera facing
the operator. Verified wired (Termination Manager shows `wrist_camera_window`). **This IS a deviation
(a net-new termination the authors lacked)** -- justified by our added wrist D405 + vision student.
Retraining-gated: takes effect on the next Stage-1/2 run. Tunable: tighten the +-60deg window or
adjust the -24deg center if a future measurement refines it.

## Files changed (branch omnireset/realpcb-thin, uncommitted)
- NEW: `Props/Custom/RealPcb/{realpcb.usd,metadata.yaml}`, `scripts_v2/tools/build_realpcb_usd.py`,
  `scripts_v2/tools/conversions/{measure_fingertip_vs_table.py,sweep_thin_reset_augmentation.py}`,
  this ledger.
- `mdp/events.py`: D1 (finger_tip_offset + object-aware standoff), B (thin-object reset augmentation
  clamp + `_infer_object_thickness`).
- `mdp/terminations.py`: A (fingertip floor + `_infer_thickness`, thin-gated).
- `Robots/LinearGripper/metadata.yaml`: `finger_tip_offset: 0.144`.
- `config/.../grasp_sampling_cfg.py`, `rl_state_cfg.py`, `reset_states_cfg.py`: `realpcb` variant.
- `config/.../linear_gripper_cfg.py`: `_enable_fingertip_floor` helper.
- `config/.../ur10e_linear_gripper_cfg.py`: enable A on the 3 UR10e EEGrasped reset tasks.
