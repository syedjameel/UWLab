# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dexterous lifting and reorientation environments for the UR10e + Tesollo DELTO.

These subclass the dexsuite base package vendored in IsaacLab and share the UR10e articulation --
and therefore the identified arm dynamics -- with the OmniReset environments.
"""

import gymnasium as gym

from . import agents

_ENV_CFG = f"{__name__}.dexlift_ur10e_delto_env_cfg"
_RSL_RL_CFG = f"{agents.__name__}.rsl_rl_ppo_cfg:DexLiftUR10eDeltoPPORunnerCfg"

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
