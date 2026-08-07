# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO runner configuration for the dexterous lifting task.

Mirrors the reference ``DexsuiteUR10TessoloPPORunnerCfg``. All three observation groups feed both
the actor and the critic, so the setup is fully privileged and symmetric rather than
asymmetric-critic; that is what the reference does and what the ADR curriculum was tuned against.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DexLiftUR10eDeltoPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    obs_groups = {"policy": ["policy", "proprio", "perception"], "critic": ["policy", "proprio", "perception"]}
    max_iterations = 15000
    save_interval = 250
    experiment_name = "dexlift_ur10e_delto"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TableLegGraspLiftPPORunnerCfg(DexLiftUR10eDeltoPPORunnerCfg):
    """State-based PPO for the fixed-geometry FurnitureBench leg task."""

    obs_groups = {"policy": ["policy", "proprio"], "critic": ["policy", "proprio"]}
    max_iterations = 6000
    save_interval = 25
    experiment_name = "table_leg_grasp_lift_ur10e_delto"
