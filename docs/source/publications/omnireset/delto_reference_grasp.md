# Reference-style Tesollo grasping

This direct RL environment ports the grasp-and-lift MDP from
`Isaaclab_delto_envs` to UWLab's OmniReset UR10e + Tesollo articulation. It
registers two tasks:

- `UWLab-UR10eDelto-Grasp-Direct-v0`: randomized primitive pretraining;
- `UWLab-UR10eDelto-TableLeg-Grasp-Direct-v0`: the 200 mm FurnitureBench leg.

The implementation is in
`source/uwlab_tasks/uwlab_tasks/direct/delto_grasp/`; both tasks use the
committed robot and object assets, so no asset-generation step is required.

## Task contract

The palm starts in an exact, calibrated palm-down pose above and laterally
separated from the object. The object starts 5--10 cm above a static table and
falls under normal gravity before the policy can grasp it. The target task
spawns only the leg, not the receptive table assembly. All 20 finger joints are
independently position-actuated, every finger-link collider remains enabled,
and Tesollo self-collisions are on.

The 26 actions are incremental joint-position targets: six UR10e joints at
0.05 rad per policy step and 20 Tesollo joints at 0.01 rad per policy step.
Policy observations contain five frames of joint positions, object-only
fingertip forces, fingertip positions, and a 25-point object cloud. A privileged
critic additionally observes the object state and domain parameters.

Success requires the object center above 0.40 m and at least three object-only
fingertip contacts. `Episode/Metrics/held_success_rate` requires that condition
for 30 consecutive policy steps. The success-driven curriculum progressively
adds observation noise and external forces; gravity is never disabled.

## Train

From the repository root, expose the extensions:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl"
```

The source baseline used 4,096 environments. The OmniReset articulation keeps
its 39 rigid bodies and Tesollo self-collisions, so use 512 environments per GPU
for both stages and a matching 18,432-sample PPO minibatch:

```bash
python scripts/reinforcement_learning/rl_games/train.py \
  --task UWLab-UR10eDelto-Grasp-Direct-v0 \
  --num_envs 512 --headless --track \
  --wandb-project-name uwlab-delto-reference-port \
  --wandb-entity YOUR_ENTITY \
  agent.params.config.minibatch_size=18432 \
  agent.params.config.central_value_config.minibatch_size=18432
```

On two GPUs, add `--distributed`; `--num_envs 512` then runs 512 environments
per rank. Fine-tune the target task with the same per-GPU batch:

```bash
python scripts/reinforcement_learning/rl_games/train.py \
  --task UWLab-UR10eDelto-TableLeg-Grasp-Direct-v0 \
  --num_envs 512 --headless --checkpoint /path/to/primitive.pth \
  agent.params.config.minibatch_size=18432 \
  agent.params.config.central_value_config.minibatch_size=18432
```

Checkpoints, W&B files, evaluations, images, and videos are runtime artifacts
and must remain outside Git.
