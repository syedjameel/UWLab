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
class DexLiftUR5eDeltoPPORunnerCfg(DexLiftUR10eDeltoPPORunnerCfg):
    """Same runner for the UR5e+DELTO, under its own experiment name.

    Nothing about the algorithm is arm specific: both robots present 26 actions (6 arm + 20 hand)
    and the same three observation groups. Only the checkpoint directory differs, and it has to --
    the two robots are not interchangeable and a checkpoint from one must never resume the other.
    """

    experiment_name = "dexlift_ur5e_delto"


@configclass
class DexLiftUR5eDeltoRelJointPosPPORunnerCfg(DexLiftUR5eDeltoPPORunnerCfg):
    """VARIANT 1 (DexSuite-style 26-DOF relative joint position) under its own experiment name.

    Nothing about the algorithm is action-space specific, so this changes only where checkpoints
    land -- and it has to. Both UR5e+DELTO action-space variants are 26-dimensional and share all
    three observation groups, so a checkpoint of the other one would LOAD here without complaint
    and mean something different. The directory is the only thing that separates them.
    """

    experiment_name = "dexlift_ur5e_delto_reljointpos"


@configclass
class DexLiftUR5eDeltoOscPPORunnerCfg(DexLiftUR5eDeltoPPORunnerCfg):
    """Same runner again for the UR5e+DELTO's operational-space action variant.

    The two UR5e+DELTO variants present the SAME 26-dimensional action space and the same three
    observation groups, so the algorithm and the network shapes are unchanged. Only the experiment
    name differs, and it has to: six of those twenty-six dimensions mean a Cartesian pose delta
    here and a joint delta in the other variant, so a checkpoint from one loads into the other
    without any shape error and is meaningless. Separate directories are the only thing preventing
    that resume.
    """

    experiment_name = "dexlift_ur5e_delto_osc"


@configclass
class DexLiftUR5eDeltoRelJointPosTableLegPPORunnerCfg(DexLiftUR5eDeltoRelJointPosPPORunnerCfg):
    """Variant 1 on the FurnitureBench table leg. ``experiment_name`` is the ONLY field it sets.

    The reference reuses its primitives runner on the leg bit-identically -- same network, same
    algorithm, same iteration budget -- so every hyperparameter is inherited rather than retuned.
    A separate name is still required, because this repository's runners are per-VARIANT and not
    per-TASK: without it a leg run would resume the primitives Lift checkpoint of the same variant,
    which has the same 26 actions and the same three observation groups and therefore loads with
    no shape error and no meaning.
    """

    experiment_name = "dexlift_ur5e_delto_reljointpos_tableleg"


@configclass
class DexLiftUR5eDeltoOscTableLegPPORunnerCfg(DexLiftUR5eDeltoOscPPORunnerCfg):
    """Variant 2 on the FurnitureBench table leg. Same argument, one directory further apart."""

    experiment_name = "dexlift_ur5e_delto_osc_tableleg"


@configclass
class DexLiftUR5eDeltoRelJointPosTableLegReorientPPORunnerCfg(DexLiftUR5eDeltoRelJointPosPPORunnerCfg):
    """Variant 1 on the table leg, scoring ORIENTATION as well as position.

    The name has to separate it from its Lift sibling for the same reason the leg had to be
    separated from the primitives: Lift and Reorient share the 26 actions and all three observation
    groups, so a Reorient run pointed at the Lift experiment directory resumes that checkpoint
    cleanly and silently trains the wrong task from it.
    """

    experiment_name = "dexlift_ur5e_delto_reljointpos_tableleg_reorient"


@configclass
class DexLiftUR5eDeltoOscTableLegReorientPPORunnerCfg(DexLiftUR5eDeltoOscPPORunnerCfg):
    """Variant 2 on the table leg, scoring orientation as well. Same argument."""

    experiment_name = "dexlift_ur5e_delto_osc_tableleg_reorient"


@configclass
class TableLegGraspLiftPPORunnerCfg(DexLiftUR10eDeltoPPORunnerCfg):
    """State-based PPO for the fixed-geometry FurnitureBench leg task."""

    obs_groups = {"policy": ["policy", "proprio"], "critic": ["policy", "proprio"]}
    max_iterations = 6000
    save_interval = 25
    experiment_name = "table_leg_grasp_lift_ur10e_delto"

    def __post_init__(self):
        # The 200 mm leg is 57.354 g, so the generic dexterous-lift exploration
        # level produces many high-energy launches. Keep enough exploration for
        # contact discovery while making successful grasp refinement dominant.
        self.policy.init_noise_std = 0.15
        # The task observations are already bounded physical quantities.  Keep
        # the actor representation identical between the from-scratch behavior
        # cloning stage and subsequent PPO refinement.
        self.policy.actor_obs_normalization = False
        self.policy.critic_obs_normalization = False
        self.algorithm.entropy_coef = 0.001
        # Fine-tune contact-rich behavior conservatively: the generic 2e-4
        # setting erased valid grasps within a few dozen resumed iterations.
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.schedule = "fixed"
