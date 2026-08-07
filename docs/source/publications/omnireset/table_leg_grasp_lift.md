# Table-leg grasp and lift

This task trains the UR10e + Tesollo DELTO hand to grasp and lift the matched-mass
FurnitureBench square table leg. It intentionally stops before alignment, insertion, or screwing.

## Layout

- Assets and MIT provenance: `source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableOneLeg/`
- Environment and task terms: `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/dexlift/`
- PPO configuration: `dexlift/agents/rsl_rl_ppo_cfg.py`
- Deterministic evaluator: `scripts/reinforcement_learning/rsl_rl/evaluate.py`

The assets are copied from `kushal2000/play2perfect` commit
`ff2fc62873f6c6cd93164123c8358e9ce2d65c9b`. The 22.75 g leg mass, table receiver,
meshes, colliders, license, and checksums are preserved.

## Task contract

Use `DexLift-UR10eDelto-TableLeg-GraspLift-Curriculum-v0` to learn the grasp while
smoothly widening the object reset range, the base `-v0` ID for full-range finetuning
and evaluation, and `-Play-v0` for playback. All 25 Tesollo phalanges have
object-filtered contact sensors.

Success requires all of the following for 12 consecutive control steps:

- leg root at least 75 mm above the table root;
- contact from at least two distinct fingers;
- leg linear speed no greater than 0.5 m/s.

This rejects proximity-only, table-supported, and ballistic lifts.

## Train and evaluate

From the repository root, after setting the normal Isaac Lab package path:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task DexLift-UR10eDelto-TableLeg-GraspLift-Curriculum-v0 \
  --num_envs 2048 --max_iterations 200 --headless \
  --logger wandb --log_project_name omnireset-table-leg
```

Continue the strongest curriculum checkpoint on the base `-v0` task with
`--resume_path /path/to/model.pt`; the accepted run used 50 full-range iterations at
learning rate `2e-5`. Use `torchrun --nproc_per_node=2` plus `--distributed` for two
GPUs, or run independent seeded sweeps. Evaluate with the frozen metric:

```bash
python scripts/reinforcement_learning/rsl_rl/evaluate.py \
  --task DexLift-UR10eDelto-TableLeg-GraspLift-v0 \
  --checkpoint /path/to/model.pt --episodes 1000 --num_envs 256 \
  --seed 42125 --headless --output /tmp/table-leg-eval.json
```

The accepted checkpoint (`b9b6e9e149e4…`) achieved **816/1000 (81.6%)** over seeds
163300 and 164300; its 95% Wilson interval is `[79.08%, 83.88%]`. Failures were 152
timeouts and 32 out-of-bounds terminations, with zero drops. The corresponding
[W&B run](https://wandb.ai/i_domrachev-interactive-robotic-systems-lab-kaist/omnireset-table-leg/runs/v4ism1qt)
contains the training curves. The evaluator also records terminal counts and mean
episode length. Checkpoints, W&B data, JSON, images, and videos remain uncommitted
runtime artifacts.
