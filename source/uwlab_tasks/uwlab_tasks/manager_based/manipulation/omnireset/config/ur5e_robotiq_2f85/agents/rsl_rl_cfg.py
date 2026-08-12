# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from uwlab_rl.rsl_rl.rl_cfg import (
    BehaviorCloningCfg,
    OffPolicyAlgorithmCfg,
    RslRlFancyActorCriticCfg,
    RslRlFancyPpoAlgorithmCfg,
)

UR10E_DELTO_EXPERIMENT_NAME = "ur10e_delto_omnireset_agent"


def my_experts_observation_func(env):
    obs = env.unwrapped.obs_buf["expert_obs"]
    return obs


@configclass
class Base_PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 40000
    save_interval = 100
    resume = False
    experiment_name = "ur5e_robotiq_2f85_omnireset_agent"
    policy = RslRlFancyActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="gsde",
        state_dependent_std=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UR10eDelto_PPORunnerCfg(Base_PPORunnerCfg):
    """PPO runner isolated from the linear-gripper checkpoint namespace."""

    experiment_name = UR10E_DELTO_EXPERIMENT_NAME


@configclass
class Base_DAggerRunnerCfg(Base_PPORunnerCfg):
    algorithm = RslRlFancyPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        offline_algorithm_cfg=OffPolicyAlgorithmCfg(
            behavior_cloning_cfg=BehaviorCloningCfg(
                experts_path=[""],
                experts_loader="torch.jit.load",
                experts_observation_group_cfg="uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.rl_state_cfg:ObservationsCfg.PolicyCfg",
                experts_observation_func=my_experts_observation_func,
                experts_action_group_cfg="uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.actions:Ur5eRobotiq2f85RelativeOSCAction",
                cloning_loss_coeff=1.0,
                loss_decay=1.0,
            )
        ),
    )


@configclass
class UR10eLinearGripper_DAggerRunnerCfg(Base_DAggerRunnerCfg):
    """DAgger runner for the UR10e + linear-gripper RGB distillation.

    Identical to ``Base_DAggerRunnerCfg`` except the expert ACTION group is the UR10e
    linear-gripper action (the 2F-85 base points at ``Ur5eRobotiq2f85RelativeOSCAction``).
    The expert OBSERVATION group (the generic state ``ObservationsCfg.PolicyCfg``) is reused;
    ``experts_path`` is supplied at runtime via a Hydra override to the exported policy.
    """

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.offline_algorithm_cfg.behavior_cloning_cfg.experts_action_group_cfg = (
            "uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.actions:"
            "Ur10eLinearGripperRelativeOSCAction"
        )


@configclass
class UR10eDelto_DAggerRunnerCfg(Base_DAggerRunnerCfg):
    """DAgger runner for the UR10e + DELTO hand RGB distillation.

    Same shape as ``UR10eLinearGripper_DAggerRunnerCfg``: only the expert ACTION group changes.
    It cannot be shared with the linear gripper's -- the action group defines the expert's action
    LAYOUT, and the DELTO's hand is 20 independent joint actions against the jaw's single scalar
    (26 total versus 7), so reusing the linear-gripper runner would load the expert against the
    wrong action spec.
    ``experts_path`` is supplied at runtime via a Hydra override.
    """

    experiment_name = UR10E_DELTO_EXPERIMENT_NAME

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.offline_algorithm_cfg.behavior_cloning_cfg.experts_action_group_cfg = (
            "uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.actions:"
            "Ur10eDeltoRelativeOSCAction"
        )
