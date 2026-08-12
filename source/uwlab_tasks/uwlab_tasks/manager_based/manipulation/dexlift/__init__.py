# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dexterous lifting and reorientation environments for a Tesollo DELTO on a UR arm.

These subclass the dexsuite base package vendored in IsaacLab and share their articulation -- and
therefore the identified arm dynamics -- with the OmniReset environments. Two robots are
registered: the UR10e+DELTO and the UR5e+DELTO.
"""

import gymnasium as gym

from . import agents

_ENV_CFG = f"{__name__}.dexlift_ur10e_delto_env_cfg"
_UR5E_ENV_CFG = f"{__name__}.dexlift_ur5e_delto_env_cfg"
# The PPO runner is shared: both robots present the same 26-dimensional action space (6 arm + 20
# hand) and the same three observation groups, so nothing in the runner is arm specific. Only
# ``experiment_name`` differs, which is what Ur5e variant below overrides.
_RSL_RL_CFG = f"{agents.__name__}.rsl_rl_ppo_cfg:DexLiftUR10eDeltoPPORunnerCfg"
_UR5E_RSL_RL_CFG = f"{agents.__name__}.rsl_rl_ppo_cfg:DexLiftUR5eDeltoPPORunnerCfg"
_TABLE_LEG_RSL_RL_CFG = f"{agents.__name__}.rsl_rl_ppo_cfg:TableLegGraspLiftPPORunnerCfg"

gym.register(
    id="DexLift-UR10eDelto-Reorient-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_CFG}:DexLiftUR10eDeltoReorientEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
    },
)

gym.register(
    id="DexLift-UR10eDelto-Reorient-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_CFG}:DexLiftUR10eDeltoReorientEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
    },
)

gym.register(
    id="DexLift-UR10eDelto-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_CFG}:DexLiftUR10eDeltoLiftEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
    },
)

gym.register(
    id="DexLift-UR10eDelto-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_CFG}:DexLiftUR10eDeltoLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
    },
)

##
# UR5e + DELTO.
##

gym.register(
    id="DexLift-UR5eDelto-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_UR5E_ENV_CFG}:DexLiftUR5eDeltoLiftEnvCfg",
        "rsl_rl_cfg_entry_point": _UR5E_RSL_RL_CFG,
    },
)

gym.register(
    id="DexLift-UR5eDelto-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_UR5E_ENV_CFG}:DexLiftUR5eDeltoLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _UR5E_RSL_RL_CFG,
    },
)

gym.register(
    id="DexLift-UR5eDelto-Reorient-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_UR5E_ENV_CFG}:DexLiftUR5eDeltoReorientEnvCfg",
        "rsl_rl_cfg_entry_point": _UR5E_RSL_RL_CFG,
    },
)

gym.register(
    id="DexLift-UR5eDelto-Reorient-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_UR5E_ENV_CFG}:DexLiftUR5eDeltoReorientEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _UR5E_RSL_RL_CFG,
    },
)

gym.register(
    id="DexLift-UR10eDelto-TableLeg-GraspLift-v0",
    entry_point=f"{__name__}.table_leg_env:TableLegGraspLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.table_leg_env_cfg:TableLegGraspLiftEnvCfg",
        "rsl_rl_cfg_entry_point": _TABLE_LEG_RSL_RL_CFG,
    },
)

gym.register(
    id="DexLift-UR10eDelto-TableLeg-GraspLift-Curriculum-v0",
    entry_point=f"{__name__}.table_leg_env:TableLegGraspLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.table_leg_env_cfg:TableLegGraspLiftCurriculumEnvCfg",
        "rsl_rl_cfg_entry_point": _TABLE_LEG_RSL_RL_CFG,
    },
)

gym.register(
    id="DexLift-UR10eDelto-TableLeg-GraspLift-Play-v0",
    entry_point=f"{__name__}.table_leg_env:TableLegGraspLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.table_leg_env_cfg:TableLegGraspLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _TABLE_LEG_RSL_RL_CFG,
    },
)
