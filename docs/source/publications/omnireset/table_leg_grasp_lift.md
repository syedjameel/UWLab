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

`DexLift-UR10eDelto-TableLeg-GraspLift-Curriculum-v0` starts the leg airborne in the
open hand's grasp corridor. Gravity stays at zero for 50 PPO iterations and ramps to
−9.81 m/s² over the next 200; reset position and orientation start widening after 250
iterations and reach the full range after another 300. At full range, the leg starts
255–285 mm above the tabletop with x/y offsets of ±80/120 mm, roll/pitch offsets of
±0.35 rad, and arbitrary yaw. It falls onto the support table when it is not caught.

The policy has six relative UR10e actions and one calibrated DELTO open/close action.
All 25 phalanges retain collision geometry and object-filtered contact sensors.
Success requires, for 12 consecutive control steps:

- leg-root clearance of at least 310 mm above the tabletop;
- contact from at least two distinct fingers;
- leg linear speed no greater than 0.5 m/s.

Thus an untouched spawn, table-supported leg, ballistic launch, or proximity-only
motion cannot succeed.

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
