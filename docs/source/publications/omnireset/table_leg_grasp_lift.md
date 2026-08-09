# Table-leg grasp and lift

This environment trains UR10e + Tesollo DELTO to grasp and lift one 200 mm
FurnitureBench square table leg. It does not include the receiver, assembled table,
alignment, insertion, or screw motion.

## Layout

- Asset and MIT provenance: `source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableOneLeg/`
- Environment, curricula, rewards, and termination: `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/dexlift/`
- PPO configuration: `dexlift/agents/rsl_rl_ppo_cfg.py`
- Deterministic evaluator: `scripts/reinforcement_learning/rsl_rl/evaluate.py`

The leg is a 200 mm long-body derivative of the Play2Perfect asset at commit
`ff2fc62873f6c6cd93164123c8358e9ce2d65c9b`. It preserves the source thread and
30 mm cross-section, uses convex-decomposition collision geometry, and has mass
57.35 g at the source leg's measured density. The only other task object is a static
cuboid support table.

## Task contract

`DexLift-UR10eDelto-TableLeg-GraspLift-Curriculum-v0` starts the leg airborne and
the open, palm-down hand 275 mm from its root. Gravity is −9.81 m/s² in training,
evaluation, and playback, so the leg first falls about 139 mm onto the support
table. The policy must then move the arm roughly 200 mm along the calibrated
approach path before finger contact is possible. A held-lift success promotes one
difficulty level and a failure demotes four, making 80% success the curriculum's
interior fixed point. The curriculum widens reset translation and then orientation;
it never disables gravity.

At full range, the airborne root is randomized by ±2 mm in x, ±3 mm in y, 0–1 mm
in z, and ±0.02 rad on each rotation axis. Larger search offsets remain out of
scope, but spawning the leg inside an open or closed hand is explicitly outside
the task contract.

The policy has six absolute, default-centered UR10e joint-position actions and twenty
independent, default-centered DELTO actions. Arm scales are 1.00/0.75/0.75/1.50/1.50/3.20
rad; DELTO scales are 0.30 rad for joints 1–3 and 1.50 rad for distal joint 4. Zero
action therefore holds the separated reset pose. All 25
phalanges retain collision geometry and object-filtered contact sensors. The fixed
object target is `(0.75, 0.10, 0.65)` m in the robot-root frame. Success requires, for
30 consecutive control steps (0.5 s):

- object-root position within 50 mm of the target and at least 80 mm of measured lift
  from the episode's low point;
- contact from an opposable digit and another finger whose contacted phalanges
  lie on opposite sides of the leg;
- at least two fingers flexed by 0.10 rad or more from their reset posture;
- no object contact on the mount, base, or palm;
- leg linear speed and palm-relative speed no greater than 0.15 m/s.

The first contact must also occur after at least 30 control steps, and success is
not evaluated before step 180. Thus an initial overlap, untouched spawn, rigid-hand
push, table-supported leg, same-side finger press, ballistic launch, transient
threshold crossing, or proximity-only motion cannot succeed.

## Train and evaluate

Run one independent seed per GPU; 256 environments fit with all phalange sensors:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task DexLift-UR10eDelto-TableLeg-GraspLift-Curriculum-v0 \
  --num_envs 256 --max_iterations 600 --headless \
  --logger wandb --log_project_name omnireset-table-leg-corrected
```

Playback uses `DexLift-UR10eDelto-TableLeg-GraspLift-Play-v0`. Acceptance evaluation
always uses the full-gravity, full-range base task:

```bash
python scripts/reinforcement_learning/rsl_rl/evaluate.py \
  --task DexLift-UR10eDelto-TableLeg-GraspLift-v0 \
  --checkpoint /path/to/model.pt --episodes 1000 --num_envs 256 \
  --seed 42125 --headless --output /tmp/table-leg-eval.json
```

The evaluator reports checkpoint SHA-256, terminal counts, mean episode length, and a
95% Wilson interval. Checkpoints, W&B data, evaluations, images, and videos are runtime
artifacts and are not committed.

## Checkpoint status

Checkpoints trained before the separated reset were invalidated: their reported
success came from closing around a leg already inside the hand. A replacement is
accepted only after at least 1,000 held-out full-gravity episodes meet the contract
above with at least 80% success.
