# UR5e + Tesollo DELTO — the recipe that produced the certified policies

Three stages, each warm-started from the previous one. Every number below is a certification:
128 episodes, four held-out seeds (101/202/303/404 — the training seed is 42), deterministic
inference, fixed *n* decided before the run, and the ADR curriculum **pinned to maximum
difficulty**, i.e. full −9.81 m/s² gravity and every domain-randomisation term at its final value.

All three stages run on the same plant, and it is now the **default** — no environment variable
selects it any more:

| | value | why |
|---|---|---|
| hand colliders | `ur5e_delto_hullfix3.usd` — 25 convexHull + 3 convexDecomposition | a plain convexHull palm swallows the finger roots and the articulation explodes at reset |
| self-collisions | **on** | with them off the policy learns grasps that pass finger 3 through finger 4 and is scored a success |
| hand actuators | reference (effort 30 N·m, velocity 10000 rad/s) | ours cap at 3.0 rad/s against a 6 rad/s commanded closure, so the fingers cannot close at all |
| arm actuators | **our identified UR5e** (150/28 N·m, armature, stiction, viscous) with `randomize_arm_sysid` live | this is the calibrated asset's whole point |
| table leg collider | `SquareTableLeg200mmSdf` — `physics:approximation=sdf`, `sdfResolution=256` | shipping leg since 2026-08-23. The `convexDecomposition` variant is REJECTED: its hulls fill the helical thread grooves, so 56.15% of poses interpenetrated the collider PhysX actually uses. Printed unconditionally as `[dexlift] ASSETS leg=…` and asserted by `run_certify.sh` (bead dr-76w.18) |
| envs / minibatch | 8192 / 73728 | updates per epoch = `mini_epochs × envs × horizon / minibatch` = 20, matching the certified reference |

## Stage 1 — position tracking, 16 primitive objects

Task `DexLift-UR5eDelto-RelJointPos-Lift-v0`, **from scratch**.

```bash
TASK=DexLift-UR5eDelto-RelJointPos-Lift-v0 \
  bash launch_task.sh <gpu> stage1_primitives_lift 8192 73728
```

Certified: **pass@50mm 0.891 · @20mm 0.836 · @10mm 0.523**, median minimum position error 9.7 mm.

Budget ~1500 epochs. The curve is **flat for the first several hundred epochs and that is normal** —
the certified DexSuite reference is itself flat until its own epoch ~1155. Do not diagnose a
blocker from a run below ~epoch 1400.

> Honesty note on the shipped checkpoint: the one that produced the number above was trained from
> scratch under convexHull + self-collisions **off** for 800 epochs and then continued under the
> plant above. A clean-room reproduction should run the whole stage under the default plant; a
> from-scratch run there reached 0.845 and was still climbing when it was measured. The residue of
> the old lineage was checked and is gone — see the geometry gate at the bottom.

## Stage 2 — position tracking, table leg (fine-tune)

Task `DexLift-UR5eDelto-RelJointPos-TableLeg-Lift-v0`, warm from Stage 1.

```bash
TASK=DexLift-UR5eDelto-RelJointPos-TableLeg-Lift-v0 \
WARM=<stage1>/nn/last_..._ep_1500_....pth \
OVERRIDES="env.curriculum.adr.params.init_difficulty=10 env.rewards.success.params.pos_std=0.03" \
  bash launch_task.sh <gpu> stage2_tableleg_lift 8192 73728
```

Certified: **pass@30mm 0.922 · @10mm 0.805**, median 4.8 mm, Wilson 95% [0.862, 0.957] at 30 mm.

Two overrides, both load-bearing:

* `init_difficulty=10` — rl_games restores weights, optimiser and the epoch counter but **not** the
  ADR curriculum, so a warm start otherwise re-climbs from difficulty 0, which is *zero gravity*.
* `pos_std=0.03` sharpens the success kernel. It is the only reward change in this whole campaign
  that ever helped: 0.766 → 0.820 at 1 cm. Note `pos_tol` is derived as `pos_std/2` inside
  `__post_init__`, which runs **before** hydra, so this override moves the reward's width without
  moving the scored gate. Sharpening further (0.015) collapses the run — see the failure list.

Budget ~450 epochs on top of Stage 1. Fine-tuning to the leg is cheap: it went 0.586 → 0.766 at
1 cm within 150 epochs of switching to the decomposed collider.

## Stage 3 — pose tracking (position **and** orientation), table leg (fine-tune)

Task `DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0`, warm from a Stage-1-derived *Reorient*
policy, not from Stage 2 — the orientation objective has to be introduced on the primitives first.

```bash
# 3a — introduce orientation on the primitives, warm from Stage 1
TASK=DexLift-UR5eDelto-RelJointPos-Reorient-v0 TILT=0.3 \
WARM=<stage1>/nn/last_..._ep_1500_....pth \
OVERRIDES="env.curriculum.adr.params.init_difficulty=10" \
  bash launch_task.sh <gpu> stage3a_primitives_reorient 8192 73728

# 3b — move to the table leg
TASK=DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0 TILT=0.3 \
WARM=<stage3a>/nn/last_..._ep_1950_....pth \
OVERRIDES="env.curriculum.adr.params.init_difficulty=10" \
  bash launch_task.sh <gpu> stage3b_tableleg_reorient 8192 73728

# 3c — sharpen
TASK=DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0 TILT=0.3 \
WARM=<stage3b>/nn/last_..._ep_3100_....pth \
OVERRIDES="env.curriculum.adr.params.init_difficulty=10 env.rewards.success.params.pos_std=0.03" \
  bash launch_task.sh <gpu> stage3c_tableleg_reorient_sharp 8192 73728
```

Certified at the best checkpoint: **pass@50mm 0.88 · @30mm 0.70 ± 0.035 · @10mm 0.00**, median
minimum position error 26.4 mm, median minimum **rotation** error 0.060 rad against a 0.25 rad gate.

**`DEXLIFT_POSE_TILT` is not gated on the task being Reorient.** Set it while launching a *Lift*
run and the staging applies there too, silently narrowing the object's reset orientation into an
easier task that still reports under the same task id and the same certified metric name. Stages 1
and 2 were certified with it unset; leave it unset for them. Verified 2026-08-16 — a Lift
construction with `DEXLIFT_POSE_TILT=0.3` in the environment prints the `POSE_TILT staged` banner.

`TILT=0.3` is not a tuning knob, it is part of the task definition and must be quoted with any
number measured under it. It narrows the object's reset roll/pitch/yaw **and** the goal's roll/pitch
to ±0.3 rad together, and clamps the reset drop to 5 cm so the sampled orientation survives to the
first policy step. Without it the demanded rotation is random-to-random (~2.2 rad) and **0 of 128
episodes ever enter the 0.25 rad gate**. Narrowing only the goal does nothing — the demand is the
angle *between* the two.

### What the pose stage costs

Orientation is never the binding constraint: the policy holds it four times tighter than the gate
requires. Position is. Median minimum position error is 26.4 mm with the orientation requirement
against 4.8 mm without — a **4.5× precision penalty** for turning a 200 mm rod, because turning it
means manipulating it and manipulating it displaces it.

## Certifying

```bash
TASK=<gym id> TILT=<0.3 for Reorient, unset for Lift> \
GPU=<n> NUM_ENVS=128 EPISODES=128 POS_TOL=0.03 \
  bash run_certify.sh <name> <checkpoint>
```

`--pos_tol` is *added* to the reported ladder (50/20/10/5 mm), so every rung stays comparable
across runs. The tolerance ladder cannot be re-scored offline for a pose policy: the rule is
`pos_dist < tol` **and** `rot_dist < 0.25` at the *same step*, and the stored per-episode record
keeps only the two minima, which need not coincide.

**Reproducibility is ±2 points at the 10 mm rung and ±3.5 at 30 mm.** The same checkpoint certified
four times scored 0.695 / 0.664 / 0.734 / 0.711 at 30 mm. Treat any difference smaller than that as
no difference. The spread comes from about 3 episodes in 128 that sit on a decision boundary and
flip between bitwise-identical replays.

## Geometry gate — run this before believing any success rate

```bash
python scratchpad/probe_penetration.py --task <gym id> --checkpoint <ckpt> \
    --num_envs 4 --steps 400 --every 5 --tag <name>
```

Measured on both deliverables: non-adjacent finger interpenetration depth p50 0.25–0.34 mm, max
1.06–1.59 mm, under 1 mm on 99.5% of instances — PhysX contact tolerance against 25–42 mm
phalanges. For contrast, an earlier 94.9% policy trained without self-collisions read p50 6.18 mm
and 94.97% over 1 mm: it was fusing finger 3 through finger 4 and being scored a success.

## Things that were tried and did not work

Do not spend a run on these again.

| change | outcome |
|---|---|
| reward re-weighting: position 6.0 / 4.0 / 0.4, orientation 1.0 / 0.0 | every one at or below the stock reward |
| sharpening the kernel further, `pos_std` 0.03 → 0.015 | ADR fell 10 → 0.043 and success to 0.013 in 70 epochs; below 50% success the curriculum's (2p−1) drift is negative and it cannot climb back |
| loosening the rotation gate, 0.25 → 0.5 rad | no effect — the gate was never the constraint |
| staging magnitude, tilt 0.12 vs 0.3 vs 0.6 | staging *at all* is the lever; its magnitude is not |
| longer episodes | 94% already run to timeout, and terminal error is worse than the mid-episode minimum — a control problem, not a time problem |
| adjusting the reset to remove the ~7% of episodes that fail | no handle exists: spawn position, height, orientation, drawn friction and drawn mass of the failing episodes all fall inside the successful range |
| early-terminating those episodes | the ADR predicate samples the *terminal* state, so a truncated failure scores identically; buys ~2% of wall clock and removes recovery time |

## Two traps that cost real time

* **rl_games `--max_iterations` is absolute**, not additional: the epoch counter is restored from
  the checkpoint. A warm start at epoch 3350 with `--max_iterations 6000` gets 2650 more epochs.
* **Judge a reward change no earlier than ~250 epochs after it.** Changing a reward scale
  invalidates the fitted value function. The sharpening above read *worse* at 150 epochs and better
  at 250; it was nearly discarded on the strength of the earlier reading.
