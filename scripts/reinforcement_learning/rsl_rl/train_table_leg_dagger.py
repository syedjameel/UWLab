# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the table-leg policy from random weights with physical-expert DAgger."""

from __future__ import annotations

import argparse
import os
import random
import sys
import torch
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="DexLift-UR10eDelto-TableLeg-GraspLift-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=63001)
parser.add_argument("--samples_per_step", type=int, default=32)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--dagger_cycles", type=int, default=3)
parser.add_argument("--dagger_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--learning_rate", type=float, default=3.0e-4)
parser.add_argument("--refine_cycles", type=int, default=0)
parser.add_argument("--refine_epochs", type=int, default=30)
parser.add_argument("--refine_learning_rate", type=float, default=3.0e-5)
parser.add_argument("--eval_episodes", type=int, default=1024)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--wandb_project", default="uwlab-delto-tableleg-repro-v2")
parser.add_argument("--wandb_entity", default="i_domrachev-interactive-robotic-systems-lab-kaist")
parser.add_argument("--run_name", default=None)
parser.add_argument("--distributed", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
if args.distributed:
    args.device = f"cuda:{local_rank}"
sys.argv = [sys.argv[0]] + hydra_args
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import combine_frame_transforms, quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.modules import ActorCritic

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.dexlift.table_leg_env_cfg import APPROACH_ARM_JOINT_POS, PALM_BODY


def target_action(env, joint_positions: dict[str, float]) -> torch.Tensor:
    term = env.action_manager.get_term("arm_action")
    ids = [term._joint_names.index(name) for name in joint_positions]
    target = torch.tensor([joint_positions[name] for name in joint_positions], device=env.device)
    return (target - term._offset[0, ids]) / term._scale[0, ids]


class Expert:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.obj = env.scene["object"]
        self.arm_term = env.action_manager.get_term("arm_action")
        self.approach = target_action(env, APPROACH_ARM_JOINT_POS)
        self.palm_id = self.robot.find_bodies(PALM_BODY)[0][0]
        self.jacobian_body_id = self.palm_id - 1 if self.robot.is_fixed_base else self.palm_id
        self.lift_object_pos = torch.zeros((env.num_envs, 3), device=env.device)
        self.lift_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # Observed held offset at the collision-validated grasp.  Using the
        # current palm orientation makes the translational servo robust to the
        # wrist's harmless null-space rotation.
        self.object_offset_p = torch.tensor((0.06681, 0.06487, 0.16058), device=env.device)

    def _arm_action_for_position(self, desired_pos: torch.Tensor) -> torch.Tensor:
        position_error = desired_pos - self.robot.data.body_pos_w[:, self.palm_id]
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.jacobian_body_id, :3, self.arm_term._joint_ids]
        jacobian_t = jacobian.transpose(1, 2)
        damping = 0.05**2 * torch.eye(3, device=self.env.device).unsqueeze(0)
        delta_joint = (
            (jacobian_t @ torch.linalg.solve(jacobian @ jacobian_t + damping, position_error.unsqueeze(-1)))
            .squeeze(-1)
            .clamp(-0.03, 0.03)
        )
        joint_target = self.robot.data.joint_pos[:, self.arm_term._joint_ids] + delta_joint
        return (joint_target - self.arm_term._offset[:, :6]) / self.arm_term._scale[:, :6]

    def __call__(self) -> torch.Tensor:
        step = self.env.episode_length_buf
        reset = step <= 1
        self.lift_initialized[reset] = False
        actions = torch.zeros((self.env.num_envs, 7), device=self.env.device)
        approach_fraction = ((step - 45).float() / 180.0).clamp(0.0, 1.0)
        actions[:, :6] = approach_fraction.unsqueeze(1) * self.approach
        close_fraction = ((step - 450).float() / 120.0).clamp(0.0, 1.0)
        actions[:, 6] = -close_fraction

        lifting = step >= 650
        newly_lifting = lifting & ~self.lift_initialized
        self.lift_object_pos[newly_lifting] = self.obj.data.root_pos_w[newly_lifting]
        self.lift_initialized[newly_lifting] = True
        if lifting.any():
            command = self.env.command_manager.get_command("object_pose")
            target_object_pos, _ = combine_frame_transforms(
                self.robot.data.root_pos_w, self.robot.data.root_quat_w, command[:, :3]
            )
            fraction = ((step - 650).float() / 250.0).clamp(0.0, 1.0).unsqueeze(1)
            desired_object_pos = torch.lerp(self.lift_object_pos, target_object_pos, fraction)
            desired_palm_pos = desired_object_pos - quat_apply(
                self.robot.data.body_quat_w[:, self.palm_id], self.object_offset_p.expand(self.env.num_envs, -1)
            )
            actions[lifting, :6] = self._arm_action_for_position(desired_palm_pos)[lifting]
        return actions


def outcome_counts(env, done_ids: torch.Tensor) -> torch.Tensor:
    names = ("success", "dropped", "object_out_of_bound", "time_out", "other")
    outcomes = torch.full((done_ids.numel(),), 4, dtype=torch.long, device=env.device)
    # Match the environment logger's precedence when termination flags coincide:
    # a completed strict hold remains a success even on the final time-limit step.
    for name in ("time_out", "object_out_of_bound", "dropped", "success"):
        if name in env.termination_manager.active_terms:
            outcomes[env.termination_manager.get_term(name)[done_ids]] = names.index(name)
    return torch.bincount(outcomes, minlength=len(names))


def reduce_counts(counts: torch.Tensor) -> torch.Tensor:
    if world_size > 1:
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
    return counts


def rate_metrics(prefix: str, counts: torch.Tensor) -> dict[str, float]:
    names = ("success", "dropped", "object_out_of_bound", "time_out", "other")
    total = counts.sum().clamp(min=1).item()
    metrics = {f"{prefix}/termination_{name}_rate": counts[i].item() / total for i, name in enumerate(names)}
    metrics[f"{prefix}/success_rate"] = counts[0].item() / total
    metrics[f"{prefix}/completed_episodes"] = total
    return metrics


def main() -> None:
    if args.distributed:
        torch.distributed.init_process_group("nccl")
    seed = args.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = seed
    env = gym.make(args.task, cfg=cfg)
    vec = RslRlVecEnvWrapper(env, clip_actions=None)
    base = env.unwrapped
    obs = vec.get_observations()
    observation_keys = ["policy", "proprio"]
    obs_groups = {"policy": observation_keys, "critic": observation_keys}
    hidden_dims = [512, 256, 128]
    policy = ActorCritic(
        obs,
        obs_groups,
        vec.num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=hidden_dims,
        critic_hidden_dims=hidden_dims,
        activation="elu",
        init_noise_std=0.05,
    ).to(args.device)
    actor = policy.actor
    if world_size > 1:
        actor = torch.nn.parallel.DistributedDataParallel(actor, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    expert = Expert(base)

    run = None
    if rank == 0:
        import wandb

        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.run_name or f"bc_scratch_global{args.num_envs * world_size}_s{args.seed}",
            config={
                "method": "scripted_expert_behavior_cloning",
                "from_scratch": True,
                "seed": args.seed,
                "num_envs_per_rank": args.num_envs,
                "global_num_envs": args.num_envs * world_size,
                "world_size": world_size,
                "samples_per_step_per_rank": args.samples_per_step,
                "epochs": args.epochs,
                "dagger_cycles": args.dagger_cycles,
                "dagger_epochs": args.dagger_epochs,
                "batch_size_per_rank": args.batch_size,
                "learning_rate": args.learning_rate,
                "hidden_dims": hidden_dims,
                "refine_cycles": args.refine_cycles,
                "refine_epochs": args.refine_epochs,
                "refine_learning_rate": args.refine_learning_rate,
            },
        )

    # Collect one complete, physically successful expert rollout.  Uniformly
    # sampling environments at every step gives balanced coverage over time.
    observations: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    completed = 0
    expert_counts = torch.zeros(5, dtype=torch.long, device=args.device)
    while completed < args.num_envs:
        target = expert()
        actor_obs = policy.get_actor_obs(obs)
        sample_count = min(args.samples_per_step, args.num_envs)
        sample_ids = torch.randperm(args.num_envs, device=args.device)[:sample_count]
        observations.append(actor_obs[sample_ids].detach().to("cpu", dtype=torch.float16))
        targets.append(target[sample_ids].detach().to("cpu", dtype=torch.float16))
        obs, _, dones, _ = vec.step(target)
        done_ids = dones.nonzero(as_tuple=True)[0]
        if done_ids.numel():
            expert_counts += outcome_counts(base, done_ids)
            completed += done_ids.numel()

    x = torch.cat(observations).float()
    y = torch.cat(targets).float()
    expert_x = x
    expert_y = y
    del observations, targets
    expert_counts = reduce_counts(expert_counts)
    if rank == 0:
        metrics = rate_metrics("expert", expert_counts)
        metrics["dataset/samples_per_rank"] = x.shape[0]
        run.log(metrics, step=0)
        print(metrics, flush=True)

    # Shuffled offline cloning avoids the phase-forgetting of sequential online
    # updates and still starts from fully random actor weights.
    log_step = 0

    def train_epochs(num_epochs: int, phase: str) -> None:
        nonlocal log_step
        actor.train()
        num_samples = x.shape[0]
        for epoch in range(num_epochs):
            permutation = torch.randperm(num_samples)
            epoch_loss = 0.0
            for batches, start in enumerate(range(0, num_samples, args.batch_size), start=1):
                ids = permutation[start : start + args.batch_size]
                batch_x = x[ids].to(args.device, non_blocking=True)
                batch_y = y[ids].to(args.device, non_blocking=True)
                prediction = actor(batch_x)
                error = prediction - batch_y
                loss = error[:, :6].square().mean() + 2.0 * error[:, 6].square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= batches
            log_step += 1
            if rank == 0:
                run.log({f"{phase}/loss": epoch_loss, "dataset/total_samples_per_rank": num_samples}, step=log_step)
                print(f"{phase} epoch {epoch + 1} loss {epoch_loss:.8f}", flush=True)

    train_epochs(args.epochs, "bc")

    # DAgger labels states induced by the learned policy.  The expert fraction
    # decays to zero, making the last collection a fully autonomous rollout.
    for cycle in range(args.dagger_cycles + args.refine_cycles):
        if cycle == args.dagger_cycles and args.refine_cycles:
            # Match a clean continuation of the scratch lineage: retain the
            # successful expert anchor, discard obsolete assisted-state data,
            # and fit subsequent fully autonomous distributions at low LR.
            x = expert_x
            y = expert_y
            for group in optimizer.param_groups:
                group["lr"] = args.refine_learning_rate
        actor.eval()
        obs, _ = vec.reset()
        cycle_observations: list[torch.Tensor] = []
        cycle_targets: list[torch.Tensor] = []
        cycle_counts = torch.zeros(5, dtype=torch.long, device=args.device)
        cycle_completed = 0
        # Cross the expert-to-actor boundary gradually, then collect one fully
        # autonomous trajectory so the final fit sees its own residual drift.
        beta_schedule = (0.75, 0.5, 0.25, 0.125, 0.0625, 0.0)
        expert_fraction = beta_schedule[min(cycle, len(beta_schedule) - 1)] if cycle < args.dagger_cycles else 0.0
        while cycle_completed < args.num_envs:
            target = expert()
            actor_obs = policy.get_actor_obs(obs)
            with torch.inference_mode():
                prediction = actor(actor_obs)
            sample_count = min(args.samples_per_step, args.num_envs)
            sample_ids = torch.randperm(args.num_envs, device=args.device)[:sample_count]
            cycle_observations.append(actor_obs[sample_ids].detach().to("cpu", dtype=torch.float16))
            cycle_targets.append(target[sample_ids].detach().to("cpu", dtype=torch.float16))
            executed = expert_fraction * target + (1.0 - expert_fraction) * prediction
            obs, _, dones, _ = vec.step(executed)
            done_ids = dones.nonzero(as_tuple=True)[0]
            if done_ids.numel():
                cycle_counts += outcome_counts(base, done_ids)
                cycle_completed += done_ids.numel()
        x = torch.cat((x, torch.cat(cycle_observations).float()))
        y = torch.cat((y, torch.cat(cycle_targets).float()))
        cycle_counts = reduce_counts(cycle_counts)
        if rank == 0:
            cycle_metrics = rate_metrics(f"dagger_{cycle + 1}", cycle_counts)
            cycle_metrics[f"dagger_{cycle + 1}/expert_fraction"] = expert_fraction
            run.log(
                cycle_metrics,
                step=log_step,
            )
            print(
                f"dagger cycle {cycle + 1} expert_fraction {expert_fraction} "
                f"success {cycle_metrics[f'dagger_{cycle + 1}/success_rate']:.6f}",
                flush=True,
            )
        cycle_epochs = args.dagger_epochs if cycle < args.dagger_cycles else args.refine_epochs
        train_epochs(cycle_epochs, f"dagger_{cycle + 1}")

    # Deterministic evaluation uses the policy alone, with no expert blend.
    actor.eval()
    obs, _ = vec.reset()
    eval_completed = 0
    eval_counts = torch.zeros(5, dtype=torch.long, device=args.device)
    # Finish at least one complete episode for every simulator slot. Otherwise
    # a large distributed batch can reach a small requested count using only
    # early successes and silently omit later timeouts.
    eval_target_per_rank = max(args.num_envs, (args.eval_episodes + world_size - 1) // world_size)
    with torch.inference_mode():
        eval_step = 0
        while eval_completed < eval_target_per_rank:
            action = actor(policy.get_actor_obs(obs))
            if rank == 0 and eval_step in (0, 225, 450, 570, 650, 800, 1000, 1200, 1400):
                print(
                    "eval_trace",
                    eval_step,
                    "action_mean",
                    action.mean(dim=0).tolist(),
                    "action_std",
                    action.std(dim=0).tolist(),
                    flush=True,
                )
            obs, _, dones, _ = vec.step(action)
            eval_step += 1
            done_ids = dones.nonzero(as_tuple=True)[0]
            if done_ids.numel():
                eval_counts += outcome_counts(base, done_ids)
                eval_completed += done_ids.numel()
    eval_counts = reduce_counts(eval_counts)
    if rank == 0:
        metrics = rate_metrics("eval", eval_counts)
        run.log(metrics, step=log_step + 1)
        print(metrics, flush=True)
        checkpoint = Path(args.checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iter": 0,
                "infos": {"method": "behavior_cloning", **metrics},
            },
            checkpoint,
        )
        run.save(str(checkpoint), policy="now")
        run.finish()

    vec.close()
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
    simulation_app.close()
