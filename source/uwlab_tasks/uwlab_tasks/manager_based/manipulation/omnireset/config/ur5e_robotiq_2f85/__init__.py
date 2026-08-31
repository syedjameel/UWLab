# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset states tasks for IsaacLab."""

import gymnasium as gym

from . import agents

# Register the partial assemblies environment
gym.register(
    id="OmniReset-PartialAssemblies-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.partial_assemblies_cfg:PartialAssembliesCfg"},
    disable_env_checker=True,
)

# Register the grasp sampling environment
gym.register(
    id="OmniReset-Robotiq2f85-GraspSampling-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.grasp_sampling_cfg:Robotiq2f85GraspSamplingCfg"},
    disable_env_checker=True,
)

# Linear-gripper variant of grasp sampling (new robot, 2F-85 untouched).
gym.register(
    id="OmniReset-LinearGripper-GraspSampling-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.linear_gripper_cfg:LinearGripperGraspSamplingCfg"},
    disable_env_checker=True,
)

# ---- Linear-gripper RESET STATES variants (mirror the 2F-85 reset tasks) ----
gym.register(
    id="OmniReset-UR5eLinearGripper-ObjectAnywhereEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.linear_gripper_cfg:LinearGripperObjectAnywhereEEAnywhereResetStatesCfg"
    },
)

gym.register(
    id="OmniReset-UR5eLinearGripper-ObjectRestingEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.linear_gripper_cfg:LinearGripperObjectRestingEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eLinearGripper-ObjectAnywhereEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.linear_gripper_cfg:LinearGripperObjectAnywhereEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eLinearGripper-ObjectPartiallyAssembledEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.linear_gripper_cfg:LinearGripperObjectPartiallyAssembledEEAnywhereResetStatesCfg"
        )
    },
)

gym.register(
    id="OmniReset-UR5eLinearGripper-ObjectPartiallyAssembledEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.linear_gripper_cfg:LinearGripperObjectPartiallyAssembledEEGraspedResetStatesCfg"
        )
    },
)

# ---- Linear-gripper RL STATE variants (mirror the 2F-85 RelCartesianOSC-State tasks) ----
gym.register(
    id="OmniReset-UR5eLinearGripper-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.linear_gripper_cfg:Ur5eLinearGripperRelCartesianOSCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR5eLinearGripper-RelCartesianOSC-State-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.linear_gripper_cfg:Ur5eLinearGripperRelCartesianOSCFinetuneCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR5eLinearGripper-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.linear_gripper_cfg:Ur5eLinearGripperRelCartesianOSCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR5eLinearGripper-RelCartesianOSC-State-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.linear_gripper_cfg:Ur5eLinearGripperRelCartesianOSCFinetuneEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# ---- UR10e + linear-gripper RESET STATES variants (same tasks, UR10e arm) ----
# No UR10e grasp-sampling task: grasp sampling is gripper-only (arm-independent), so
# OmniReset-LinearGripper-GraspSampling-v0 serves both arms.
gym.register(
    id="OmniReset-UR10eLinearGripper-ObjectAnywhereEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ur10e_linear_gripper_cfg:Ur10eLinearGripperObjectAnywhereEEAnywhereResetStatesCfg"
        )
    },
)

gym.register(
    id="OmniReset-UR10eLinearGripper-ObjectRestingEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ur10e_linear_gripper_cfg:Ur10eLinearGripperObjectRestingEEGraspedResetStatesCfg"
        )
    },
)

gym.register(
    id="OmniReset-UR10eLinearGripper-ObjectAnywhereEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ur10e_linear_gripper_cfg:Ur10eLinearGripperObjectAnywhereEEGraspedResetStatesCfg"
        )
    },
)

gym.register(
    id="OmniReset-UR10eLinearGripper-ObjectPartiallyAssembledEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ur10e_linear_gripper_cfg:Ur10eLinearGripperObjectPartiallyAssembledEEAnywhereResetStatesCfg"
        )
    },
)

gym.register(
    id="OmniReset-UR10eLinearGripper-ObjectPartiallyAssembledEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ur10e_linear_gripper_cfg:Ur10eLinearGripperObjectPartiallyAssembledEEGraspedResetStatesCfg"
        )
    },
)

# ---- UR10e + linear-gripper RL STATE variants ----
gym.register(
    id="OmniReset-UR10eLinearGripper-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_linear_gripper_cfg:Ur10eLinearGripperRelCartesianOSCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR10eLinearGripper-RelCartesianOSC-State-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_linear_gripper_cfg:Ur10eLinearGripperRelCartesianOSCFinetuneCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR10eLinearGripper-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_linear_gripper_cfg:Ur10eLinearGripperRelCartesianOSCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR10eLinearGripper-RelCartesianOSC-State-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_linear_gripper_cfg:Ur10eLinearGripperRelCartesianOSCFinetuneEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# UR10e SysID env (P8 sim2real: CMA-ES against real UR10e trajectories)
gym.register(
    id="OmniReset-UR10eLinearGripper-Sysid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.ur10e_linear_gripper_cfg:Ur10eLinearGripperSysidEnvCfg"},
)

# ---- UR10e + linear-gripper RGB pipeline (camera align + distillation data collection) ----
# Camera-alignment env (interactive sim2real camera calibration via align_cameras.py --robot ur10e).
gym.register(
    id="OmniReset-UR10eLinearGripper-CameraAlign-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.ur10e_linear_gripper_rgb_cfg:Ur10eLinearGripperCameraAlignEnvCfg"},
)

# RGB data collection (80k expert demos for distillation) + in-distribution RGB play/eval.
gym.register(
    id="OmniReset-UR10eLinearGripper-RelCartesianOSC-RGB-DataCollection-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_linear_gripper_rgb_cfg:Ur10eLinearGripperDataCollectionRGBCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eLinearGripper_DAggerRunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR10eLinearGripper-RelCartesianOSC-RGB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_linear_gripper_rgb_cfg:Ur10eLinearGripperEvalRGBCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eLinearGripper_DAggerRunnerCfg",
    },
)

# =======================================================================================
# UR10e + Tesollo DELTO DG-5F hand. New ids only -- every id above keeps its entry point
# and its cfg class. Same naming convention: OmniReset-{Robot}-{Task}-v0.
# =======================================================================================

# ---- DELTO grasp sampling (hand only) ----
# This id was deleted once, on the argument that grasp sampling only exists to script a CLOSE
# command, which the fully actuated hand no longer has. The env is a scripted rollout, not a
# training path, and the close it needs is a per-joint servo over the SAME twenty independent
# action dimensions the policy gets -- there is no scalar in it. What deleting it actually removed
# was the only producer of ``Grasps/<object>/grasps.pt``, which
# ``reset_end_effector_from_grasp_dataset`` replays into all three ``*EEGrasped`` DELTO reset-state
# envs, whose datasets are three of the four reset types of the TRAINABLE
# ``OmniReset-UR10eDelto-RelCartesianOSC-State-v0``. See DeltoGraspSamplingCfg's docstring.
gym.register(
    id="OmniReset-Delto-GraspSampling-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.delto_cfg:DeltoGraspSamplingCfg"},
    disable_env_checker=True,
)

# ---- UR10e + DELTO RESET STATES variants ----
gym.register(
    id="OmniReset-UR10eDelto-ObjectAnywhereEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.ur10e_delto_cfg:Ur10eDeltoObjectAnywhereEEAnywhereResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR10eDelto-ObjectRestingEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.ur10e_delto_cfg:Ur10eDeltoObjectRestingEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR10eDelto-ObjectAnywhereEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.ur10e_delto_cfg:Ur10eDeltoObjectAnywhereEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR10eDelto-ObjectPartiallyAssembledEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cfg:Ur10eDeltoObjectPartiallyAssembledEEAnywhereResetStatesCfg"
    },
)

gym.register(
    id="OmniReset-UR10eDelto-ObjectPartiallyAssembledEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cfg:Ur10eDeltoObjectPartiallyAssembledEEGraspedResetStatesCfg"
    },
)

# ---- UR10e + DELTO RL STATE variants ----
gym.register(
    id="OmniReset-UR10eDelto-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cfg:Ur10eDeltoRelCartesianOSCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR10eDelto-RelCartesianOSC-State-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cfg:Ur10eDeltoRelCartesianOSCFinetuneCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR10eDelto-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cfg:Ur10eDeltoRelCartesianOSCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR10eDelto-RelCartesianOSC-State-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cfg:Ur10eDeltoRelCartesianOSCFinetuneEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)

# ---- UR10e + DELTO RGB pipeline (camera align + distillation data collection) ----
# Front/side camera poses are the linear gripper's ArUco calibration verbatim and the wrist camera
# is re-derived onto the DELTO palm, so a policy distilled here sees the same rig -- see
# ur10e_delto_rgb_cfg.
gym.register(
    id="OmniReset-UR10eDelto-CameraAlign-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.ur10e_delto_rgb_cfg:Ur10eDeltoCameraAlignEnvCfg"},
)

gym.register(
    id="OmniReset-UR10eDelto-RelCartesianOSC-RGB-DataCollection-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_rgb_cfg:Ur10eDeltoDataCollectionRGBCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_DAggerRunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR10eDelto-RelCartesianOSC-RGB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_rgb_cfg:Ur10eDeltoEvalRGBCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_DAggerRunnerCfg",
    },
)

# =======================================================================================
# UR5e + Tesollo DELTO DG-5F hand (bead UWLab-zvd.3). New ids only -- every id above keeps its
# entry point and its cfg class. Same naming convention as the UR10e+DELTO block:
# OmniReset-{Robot}-{Task}-v0.
#
# NOT registered here: ``OmniReset-Delto-GraspSampling-v0`` and ``OmniReset-PartialAssemblies-v0``.
# Both already exist above and are arm-independent (grasp sampling and partial-assembly placement
# never put an arm in the scene), so they already serve this variant -- see ur5e_delto_cfg.py's
# module docstring.
# =======================================================================================

# ---- UR5e + DELTO RESET STATES variants ----
gym.register(
    id="OmniReset-UR5eDelto-ObjectAnywhereEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.ur5e_delto_cfg:Ur5eDeltoObjectAnywhereEEAnywhereResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eDelto-ObjectRestingEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.ur5e_delto_cfg:Ur5eDeltoObjectRestingEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eDelto-ObjectAnywhereEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.ur5e_delto_cfg:Ur5eDeltoObjectAnywhereEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eDelto-ObjectPartiallyAssembledEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5e_delto_cfg:Ur5eDeltoObjectPartiallyAssembledEEAnywhereResetStatesCfg"
    },
)

gym.register(
    id="OmniReset-UR5eDelto-ObjectPartiallyAssembledEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5e_delto_cfg:Ur5eDeltoObjectPartiallyAssembledEEGraspedResetStatesCfg"
    },
)

# ---- UR5e + DELTO RL STATE variants ----
gym.register(
    id="OmniReset-UR5eDelto-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5e_delto_cfg:Ur5eDeltoRelCartesianOSCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR5eDelto_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR5eDelto-RelCartesianOSC-State-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5e_delto_cfg:Ur5eDeltoRelCartesianOSCFinetuneCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR5eDelto_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR5eDelto-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5e_delto_cfg:Ur5eDeltoRelCartesianOSCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR5eDelto_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR5eDelto-RelCartesianOSC-State-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5e_delto_cfg:Ur5eDeltoRelCartesianOSCFinetuneEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR5eDelto_PPORunnerCfg",
    },
)

# Register reset states environments
gym.register(
    id="OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectAnywhereEEAnywhereResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eRobotiq2f85-ObjectRestingEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectRestingEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectAnywhereEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eRobotiq2f85-ObjectPartiallyAssembledEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectPartiallyAssembledEEAnywhereResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eRobotiq2f85-ObjectPartiallyAssembledEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectPartiallyAssembledEEGraspedResetStatesCfg"},
)

# Register SysID env
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-Sysid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sysid_cfg:SysidEnvCfg"},
)

# Register Camera Alignment env
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-CameraAlign-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.camera_align_cfg:CameraAlignEnvCfg"},
)

# Register RL state environments
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)


# RGB environments for data collection and evaluation
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DataCollection-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85EvalRGBRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

# OOD (out-of-distribution) RGB environments
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-OOD-DataCollection-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCOODCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-OOD-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85EvalRGBRelCartesianOSCOODCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)


# ---- UR10e + DELTO CUBE STACKING (both variants) ----
# Reset-state banks are SHARED by the oriented and position-only variants: recording grades with
# check_reset_state_success, which reads success_thresholds from the receptive object's metadata
# and never consults the command term. Record these five once; train both variants from them.
for _rs_name, _rs_cls in [
    ("ObjectAnywhereEEAnywhere", "CubeStackObjectAnywhereEEAnywhereCfg"),
    ("ObjectRestingEEGrasped", "CubeStackObjectRestingEEGraspedCfg"),
    ("ObjectAnywhereEEGrasped", "CubeStackObjectAnywhereEEGraspedCfg"),
    ("ObjectPartiallyAssembledEEAnywhere", "CubeStackObjectPartiallyAssembledEEAnywhereCfg"),
    ("ObjectPartiallyAssembledEEGrasped", "CubeStackObjectPartiallyAssembledEEGraspedCfg"),
]:
    gym.register(
        id=f"OmniReset-UR10eDelto-CubeStack-{_rs_name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}.ur10e_delto_cubestack_cfg:{_rs_cls}"},
    )
del _rs_name, _rs_cls

# ORIENTED: roll/pitch constrained at 0.05 rad (the docs' cube-stacking tolerance), yaw free.
gym.register(
    id="OmniReset-UR10eDelto-CubeStack-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cubestack_cfg:CubeStackTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR10eDelto-CubeStack-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cubestack_cfg:CubeStackEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)

# POSITION-ONLY: orientation dropped from the success test AND from the dense goal shaping.
gym.register(
    id="OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cubestack_cfg:CubeStackNoOrientTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cubestack_cfg:CubeStackNoOrientEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)

# TWO-FINGER: position-only, with the hand's action narrowed to fingers 1 and 2 -- 6 + 8 = 14
# dimensions instead of 6 + 20 = 26. The stored reset banks are UNCHANGED and stay valid; fingers
# 3/4/5 hold the posture the bank's own recorded joint target puts them in. Checkpoints from the
# 26-dim tasks will not load (actor obs 380 -> 320). See
# ur10e_delto_cubestack_cfg._apply_two_finger_hand.
gym.register(
    id="OmniReset-UR10eDelto-CubeStackTwoFinger-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cubestack_cfg:CubeStackTwoFingerTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR10eDelto-CubeStackTwoFinger-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cubestack_cfg:CubeStackTwoFingerEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)

# GRIP BIAS: identical to the two tasks above except that the hand action carries a constant
# closing offset, so an EEGrasped reset keeps hold of the cube instead of dropping it on the first
# control step. See ur10e_delto_cubestack_cfg._apply_grip_bias for the measurement that motivates it.
# Checkpoints transfer between a task and its Grip twin -- the observation and action dimensions are
# unchanged; only the affine pre-processing of the hand action differs.
gym.register(
    id="OmniReset-UR10eDelto-CubeStackGrip-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cubestack_cfg:CubeStackGripTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR10eDelto-CubeStackGrip-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cubestack_cfg:CubeStackGripEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR10eDelto-CubeStackTwoFingerGrip-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cubestack_cfg:CubeStackTwoFingerGripTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR10eDelto-CubeStackTwoFingerGrip-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cubestack_cfg:CubeStackTwoFingerGripEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)

# GRIP + CARRY: the grip bias, plus a mid-range goal-distance term at std 0.08 m. Between 200 mm and
# 20 mm neither shipped shaping term has any gradient worth acting on -- see
# ur10e_delto_cubestack_cfg._apply_transport_shaping. Checkpoints transfer to and from the Grip
# tasks; only the reward differs.
gym.register(
    id="OmniReset-UR10eDelto-CubeStackGripCarry-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cubestack_cfg:CubeStackGripCarryTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR10eDelto-CubeStackGripCarry-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cubestack_cfg:CubeStackGripCarryEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR10eDelto-CubeStackTwoFingerGripCarry-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cubestack_cfg:CubeStackTwoFingerGripCarryTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-UR10eDelto-CubeStackTwoFingerGripCarry-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cubestack_cfg:CubeStackTwoFingerGripCarryEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-CubeStack-PartialAssemblies-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.ur10e_delto_cubestack_cfg:CubeStackPartialAssembliesCfg"},
)

# A/B probe: C3 with the thumb-side actuator matched to the grasp sampler's. See
# ur10e_delto_cubestack_cfg._apply_grasp_matched_hand_actuator for why the stock actuator cannot
# reproduce the postures the grasp bank records.
gym.register(
    id="OmniReset-UR10eDelto-CubeStack-ObjectAnywhereEEGrasped-ThumbFix-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cubestack_cfg:CubeStackObjectAnywhereEEGraspedThumbFixCfg"
    },
)

# Diagnostic task: near-goal resets only. Separates "insufficient compute" from "cannot learn".
# See CubeStackNearGoalOnlyTrainCfg for why this is not the paper's method and must not be reported
# as a replication of it.
gym.register(
    id="OmniReset-UR10eDelto-CubeStackNearGoal-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_delto_cubestack_cfg:CubeStackNearGoalOnlyTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:UR10eDelto_PPORunnerCfg",
    },
)
