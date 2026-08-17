# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The dexlift rl_games configs are a copy of a reference; this is what keeps them one.

There are six of them and they are meant to differ from the reference DexSuite config -- and from
each other -- in exactly ONE value: ``params.config.name``. Nothing in YAML enforces that. Copies
drift, and a drifted copy still trains: it just stops being the setup that reached 88%, and the
number it produces is then not comparable to anything.

So this test pins two things:

* every hyperparameter below equals the reference's value, quoted from
  ``IsaacLabDexterous @ 2208576f``:
  ``source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dexsuite/config/ur10_tessolo/
  agents/rl_games_ppo_cfg.yaml``;
* the files are byte-identical to one another apart from the ``name:`` line and the comment
  block above it.

It needs only PyYAML -- no Isaac Sim, no GPU -- so it runs in any environment.
"""

from __future__ import annotations

import pathlib
import pytest
import yaml

AGENTS_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "uwlab_tasks"
    / "manager_based"
    / "manipulation"
    / "dexlift"
    / "agents"
)

# filename -> the ``params.config.name`` it is required to carry. One per task-id family; the Play
# id of each family shares its sibling's file, which is why there are six and not twelve.
EXPECTED_NAMES = {
    "rl_games_ppo_ur5e_delto_reljointpos_lift_cfg.yaml": "dexlift_ur5e_delto_reljointpos_lift",
    "rl_games_ppo_ur5e_delto_reljointpos_reorient_cfg.yaml": "dexlift_ur5e_delto_reljointpos_reorient",
    "rl_games_ppo_ur5e_delto_reljointpos_tableleg_lift_cfg.yaml": "dexlift_ur5e_delto_reljointpos_tableleg_lift",
    "rl_games_ppo_ur5e_delto_osc_lift_cfg.yaml": "dexlift_ur5e_delto_osc_lift",
    "rl_games_ppo_ur5e_delto_osc_reorient_cfg.yaml": "dexlift_ur5e_delto_osc_reorient",
    "rl_games_ppo_ur5e_delto_osc_tableleg_lift_cfg.yaml": "dexlift_ur5e_delto_osc_tableleg_lift",
    "rl_games_ppo_ur5e_delto_reljointpos_tableleg_reorient_cfg.yaml": (
        "dexlift_ur5e_delto_reljointpos_tableleg_reorient"
    ),
    "rl_games_ppo_ur5e_delto_osc_tableleg_reorient_cfg.yaml": "dexlift_ur5e_delto_osc_tableleg_reorient",
}

# Every one of these is quoted from the reference file. The six marked (*) are the ones the rsl_rl
# port silently changed, which is the whole reason these configs exist.
REFERENCE_CONFIG = {
    "reward_shaper": {"scale_value": 0.01},  # (*) absent in rsl_rl -- rewards were 100x larger
    "critic_coef": 4,  # (*) rsl_rl value_loss_coef was 1.0
    "entropy_coef": 0.001,  # (*) rsl_rl was 0.005
    "bounds_loss_coef": 0.0001,  # (*) rsl_rl has no bounds loss at all
    "horizon_length": 36,  # (*) rsl_rl num_steps_per_env was 32
    # (*) rsl_rl expressed this as num_mini_batches=4.
    #
    # 18432, NOT the 36864 that the reference's own ur10_tessolo yaml carries at 6fee5b9f. This is the
    # one place where the checked-in reference SOURCE and the run that produced the certified result
    # disagree, and the run wins: in3kt4m6 (r2a_stock_gate1p0_e2048_s42, the 88 percent primitives run
    # the table-leg policy was warm-started from) has ``minibatch_size: 18432`` in its own serialized
    # W&B config, alongside num_actors 2048 and horizon_length 36. A serialized run config cannot be
    # retro-fitted by a later source edit, so it is the stronger evidence.
    #
    # WHY IT MATTERS, and it is not cosmetic: rl_games does ``mini_epochs`` passes over
    # ``batch = num_actors * horizon_length`` in ``minibatch_size`` chunks, so the OPTIMIZER STEPS PER
    # EPOCH are ``mini_epochs * batch / minibatch_size``. At the reference's 2048 envs that is
    # 5 * 73728 / 18432 = 20 updates/epoch, against 5 * 73728 / 36864 = 10 at the value this constant
    # used to hold. Run idodkymb sat flat for 1200 epochs at 10 updates/epoch while run axn28939 -- same
    # plant, same rewards, but 4096 envs, which restores 20 -- climbed. Matching the reference's
    # ENVIRONMENT COUNT while keeping a minibatch tuned for twice that count silently halves training.
    "minibatch_size": 18432,
    "gamma": 0.99,
    "tau": 0.95,
    # NOT ``1e-3``: YAML 1.1 needs a decimal point and a signed exponent to recognise scientific
    # notation, so the reference's ``learning_rate: 1e-3`` loads as the *string* ``'1e-3'``. That is
    # harmless -- rl_games coerces it in ``algos_torch/a2c_continuous.py:42`` (``self.last_lr =
    # float(self.last_lr)``) before the optimizer or the adaptive scheduler ever see it -- and it is
    # copied as-is rather than "fixed", because the reference's runs are what these numbers cite.
    "learning_rate": "1e-3",
    "lr_schedule": "adaptive",
    "kl_threshold": 0.01,
    "e_clip": 0.2,
    "grad_norm": 1.0,
    "mini_epochs": 5,
    "normalize_input": True,
    "normalize_value": True,
    "normalize_advantage": True,
    "truncate_grads": True,
    "clip_value": True,
    "ppo": True,
    "env_name": "rlgpu",
}

REFERENCE_ENV = {
    "clip_observations": 100.0,
    "clip_actions": 100.0,
    # Verified against the groups our envs actually expose: dexsuite's ``ObservationsCfg`` declares
    # ``policy``, ``proprio`` and ``perception``, and our UR5e+DELTO cfgs inherit it unchanged.
    "obs_groups": {
        "obs": ["policy", "proprio", "perception"],
        "states": ["policy", "proprio", "perception"],
    },
    "concate_obs_groups": True,
}

REFERENCE_NETWORK_MLP = {
    "units": [512, 256, 128],
    "activation": "elu",
    "d2rl": False,
}


def _load(filename: str) -> dict:
    with open(AGENTS_DIR / filename, encoding="utf-8") as f:
        return yaml.full_load(f)


@pytest.mark.parametrize("filename", sorted(EXPECTED_NAMES))
def test_config_matches_reference_hyperparameters(filename: str):
    cfg = _load(filename)["params"]
    for key, expected in REFERENCE_CONFIG.items():
        assert cfg["config"][key] == expected, f"{filename}: config.{key} drifted from the reference"
    for key, expected in REFERENCE_ENV.items():
        assert cfg["env"][key] == expected, f"{filename}: env.{key} drifted from the reference"
    for key, expected in REFERENCE_NETWORK_MLP.items():
        assert cfg["network"]["mlp"][key] == expected, f"{filename}: network.mlp.{key} drifted"
    assert cfg["network"]["separate"] is True
    assert cfg["algo"]["name"] == "a2c_continuous"
    assert cfg["model"]["name"] == "continuous_a2c_logstd"
    # and the value rl_games actually trains with, once it has coerced the string above
    assert float(cfg["config"]["learning_rate"]) == 1e-3


@pytest.mark.parametrize("filename,expected_name", sorted(EXPECTED_NAMES.items()))
def test_each_task_family_has_its_own_name(filename: str, expected_name: str):
    """``name`` is the only thing separating these runs' logs and .pth basenames.

    Every one of these task ids presents the same 26 actions over the same three observation
    groups, so a checkpoint of one loads into another with no shape error and no meaning.
    """
    assert _load(filename)["params"]["config"]["name"] == expected_name


def test_the_files_differ_only_in_the_name_line():
    def strip_name_block(filename: str) -> list[str]:
        lines = (AGENTS_DIR / filename).read_text(encoding="utf-8").splitlines()
        return [ln for ln in lines if not ln.lstrip().startswith("#") and not ln.strip().startswith("name:")]

    baselines = {f: strip_name_block(f) for f in EXPECTED_NAMES}
    first, reference_body = next(iter(baselines.items()))
    for filename, body in baselines.items():
        assert body == reference_body, f"{filename} has drifted from {first} outside its name line"


@pytest.mark.parametrize("num_envs", [4096, 2048])
def test_minibatch_size_divides_the_rollout_batch(num_envs: int):
    """rl_games asserts ``batch_size % minibatch_size == 0`` (a2c_common.py:256).

    ``batch_size = horizon_length * num_actors * num_agents`` and ``num_actors`` is overwritten with
    the env count by the trainer, so this is the arithmetic that decides whether a run boots.
    """
    cfg = _load(next(iter(EXPECTED_NAMES)))["params"]["config"]
    batch_size = cfg["horizon_length"] * num_envs
    assert batch_size % cfg["minibatch_size"] == 0, (
        f"{num_envs} envs x horizon {cfg['horizon_length']} = {batch_size}, which is not a whole"
        f" number of {cfg['minibatch_size']}-sample minibatches"
    )
