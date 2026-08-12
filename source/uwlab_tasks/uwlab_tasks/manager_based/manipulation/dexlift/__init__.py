# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dexterous lifting and reorientation environments for a Tesollo DELTO on a UR arm.

These subclass the dexsuite base package vendored in IsaacLab and share their articulation -- and
therefore the identified arm dynamics -- with the OmniReset environments. Two robots are
registered: the UR10e+DELTO and the UR5e+DELTO.

The UR5e+DELTO ids also name their ACTION-SPACE VARIANT, because more than one is being compared on
that robot and every one of them is 26-dimensional. See the comment above its registrations.
"""

import gymnasium as gym

from . import agents

_ENV_CFG = f"{__name__}.dexlift_ur10e_delto_env_cfg"
_UR5E_ENV_CFG = f"{__name__}.dexlift_ur5e_delto_env_cfg"
# The UR5e+DELTO's second action-space variant: operational-space arm, same fully actuated hand.
# Same 26 dimensions as the ids above, a different manifold -- hence its own ids and its own
# experiment name. See dexlift_ur5e_delto_osc_env_cfg's module docstring.
_UR5E_OSC_ENV_CFG = f"{__name__}.dexlift_ur5e_delto_osc_env_cfg"
# The PPO runner is shared: both robots present the same 26-dimensional action space (6 arm + 20
# hand) and the same three observation groups, so nothing in the runner is arm specific. Only
# ``experiment_name`` differs, which is what Ur5e variant below overrides.
_RSL_RL_CFG = f"{agents.__name__}.rsl_rl_ppo_cfg:DexLiftUR10eDeltoPPORunnerCfg"
# One runner per UR5e ACTION-SPACE VARIANT, not one per robot: the variants are all 26-dimensional
# over the same observation groups, so only a separate ``experiment_name`` keeps a checkpoint of
# one from silently resuming another. ``DexLiftUR5eDeltoPPORunnerCfg`` is their shared base.
_UR5E_RELJOINTPOS_RSL_RL_CFG = f"{agents.__name__}.rsl_rl_ppo_cfg:DexLiftUR5eDeltoRelJointPosPPORunnerCfg"
_UR5E_OSC_RSL_RL_CFG = f"{agents.__name__}.rsl_rl_ppo_cfg:DexLiftUR5eDeltoOscPPORunnerCfg"
_TABLE_LEG_RSL_RL_CFG = f"{agents.__name__}.rsl_rl_ppo_cfg:TableLegGraspLiftPPORunnerCfg"

# The rl_games PPO configs are verbatim copies of the reference DexSuite one (IsaacLabDexterous @
# 2208576f), which is the setup that measurably reached 88% on the 16-primitive dexsuite lift. That
# is the trainer to reach for when the question is "does OUR robot reproduce the reference"; the
# rsl_rl runners above are a port and differ from it in six hyperparameters (reward scaling,
# critic_coef, entropy_coef, the bounds loss, horizon length, minibatch size). Registering both
# leaves the choice explicit at the command line instead of implicit in the package.
#
# There is ONE FILE PER TASK ID FAMILY, not one shared file, because the reference's single shared
# file is exactly what makes its ``config.name: reorient`` a trap -- see the comment in any of the
# four. The Play id of each family deliberately shares its sibling's file: it must resolve to the
# same ``logs/rl_games/<name>/`` directory to find the checkpoint it is replaying.
_UR5E_RELJOINTPOS_LIFT_RL_GAMES_CFG = f"{agents.__name__}:rl_games_ppo_ur5e_delto_reljointpos_lift_cfg.yaml"
_UR5E_RELJOINTPOS_REORIENT_RL_GAMES_CFG = f"{agents.__name__}:rl_games_ppo_ur5e_delto_reljointpos_reorient_cfg.yaml"
_UR5E_OSC_LIFT_RL_GAMES_CFG = f"{agents.__name__}:rl_games_ppo_ur5e_delto_osc_lift_cfg.yaml"
_UR5E_OSC_REORIENT_RL_GAMES_CFG = f"{agents.__name__}:rl_games_ppo_ur5e_delto_osc_reorient_cfg.yaml"

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
# UR5e + DELTO. One family of ids PER ACTION-SPACE VARIANT.
##
# The variant is named in the id -- ``RelJointPos`` here -- and there is deliberately NO bare
# ``DexLift-UR5eDelto-Lift-v0`` next to it. A bare id would have to resolve to one variant, which
# makes it a silent default: a command line written against it would keep running after a second
# variant landed and would train whichever one the id happened to point at. Unregistered ids fail
# loudly (gym raises ``NameNotFound`` and prints the close matches), so retiring the bare name
# turns that into a typo you see immediately.
#
# VARIANT 1: DexSuite-style, all 26 DOF as relative joint-position commands at 0.1 rad per unit
# action, relative to the MEASURED joint position. See ``dexlift_ur5e_delto_actions``.

gym.register(
    id="DexLift-UR5eDelto-RelJointPos-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_UR5E_ENV_CFG}:DexLiftUR5eDeltoRelJointPosLiftEnvCfg",
        "rl_games_cfg_entry_point": _UR5E_RELJOINTPOS_LIFT_RL_GAMES_CFG,
        "rsl_rl_cfg_entry_point": _UR5E_RELJOINTPOS_RSL_RL_CFG,
    },
)

gym.register(
    id="DexLift-UR5eDelto-RelJointPos-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_UR5E_ENV_CFG}:DexLiftUR5eDeltoRelJointPosLiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": _UR5E_RELJOINTPOS_LIFT_RL_GAMES_CFG,
        "rsl_rl_cfg_entry_point": _UR5E_RELJOINTPOS_RSL_RL_CFG,
    },
)

gym.register(
    id="DexLift-UR5eDelto-RelJointPos-Reorient-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_UR5E_ENV_CFG}:DexLiftUR5eDeltoRelJointPosReorientEnvCfg",
        "rl_games_cfg_entry_point": _UR5E_RELJOINTPOS_REORIENT_RL_GAMES_CFG,
        "rsl_rl_cfg_entry_point": _UR5E_RELJOINTPOS_RSL_RL_CFG,
    },
)

gym.register(
    id="DexLift-UR5eDelto-RelJointPos-Reorient-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_UR5E_ENV_CFG}:DexLiftUR5eDeltoRelJointPosReorientEnvCfg_PLAY",
        "rl_games_cfg_entry_point": _UR5E_RELJOINTPOS_REORIENT_RL_GAMES_CFG,
        "rsl_rl_cfg_entry_point": _UR5E_RELJOINTPOS_RSL_RL_CFG,
    },
)

# VARIANT 2: OmniReset's operational-space arm (6 Cartesian dimensions through the analytical
# Jacobian, torque-only actuator) + the SAME fully actuated 20-DOF hand. Also 26 dimensions, which
# is exactly why it gets its own ids and its own experiment name rather than a flag: a checkpoint
# of either variant loads into the other with no shape error and no meaning. See
# ``dexlift_ur5e_delto_osc_actions`` and ``dexlift_ur5e_delto_osc_env_cfg``.

gym.register(
    id="DexLift-UR5eDelto-OSC-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_UR5E_OSC_ENV_CFG}:DexLiftUR5eDeltoOscLiftEnvCfg",
        "rl_games_cfg_entry_point": _UR5E_OSC_LIFT_RL_GAMES_CFG,
        "rsl_rl_cfg_entry_point": _UR5E_OSC_RSL_RL_CFG,
    },
)

gym.register(
    id="DexLift-UR5eDelto-OSC-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_UR5E_OSC_ENV_CFG}:DexLiftUR5eDeltoOscLiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": _UR5E_OSC_LIFT_RL_GAMES_CFG,
        "rsl_rl_cfg_entry_point": _UR5E_OSC_RSL_RL_CFG,
    },
)

gym.register(
    id="DexLift-UR5eDelto-OSC-Reorient-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_UR5E_OSC_ENV_CFG}:DexLiftUR5eDeltoOscReorientEnvCfg",
        "rl_games_cfg_entry_point": _UR5E_OSC_REORIENT_RL_GAMES_CFG,
        "rsl_rl_cfg_entry_point": _UR5E_OSC_RSL_RL_CFG,
    },
)

gym.register(
    id="DexLift-UR5eDelto-OSC-Reorient-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_UR5E_OSC_ENV_CFG}:DexLiftUR5eDeltoOscReorientEnvCfg_PLAY",
        "rl_games_cfg_entry_point": _UR5E_OSC_REORIENT_RL_GAMES_CFG,
        "rsl_rl_cfg_entry_point": _UR5E_OSC_RSL_RL_CFG,
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
