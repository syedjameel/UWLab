# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Reference-style dexterous grasping with the OmniReset UR10e + Tesollo."""

import gymnasium as gym

from . import agents


gym.register(
    id="UWLab-UR10eDelto-Grasp-Direct-v0",
    entry_point=f"{__name__}.delto_grasp_env:DeltoGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.delto_grasp_env_cfg:DeltoGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="UWLab-UR10eDelto-TableLeg-Grasp-Direct-v0",
    entry_point=f"{__name__}.delto_grasp_env:DeltoGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.delto_grasp_env_cfg:TableLegGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
