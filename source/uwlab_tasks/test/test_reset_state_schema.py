# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit-proves the reset-state PD-target schema (bead UWLab-algw.7).

Needs only torch -- no Isaac Sim, no GPU, no env construction. See
``omnireset/mdp/reset_state_schema.py``'s own module docstring for why the schema logic lives in
its own isaaclab-free module: ``events.py``/``recorders.py`` import ``carb``/``isaaclab``/
``omni``/``warp``/``pxr`` at module scope and cannot be imported outside a running Isaac process,
so this test loads ``reset_state_schema.py`` by file path, same idiom as
``test_held_check_core.py`` uses for ``held_check_core.py``.

THREE THINGS THIS PROVES, matching the bead's TASK 5:
1. A synthetic state dict round-trips the WRITER (``add_joint_targets``) and READER
   (``resolve_joint_targets``) with the target surviving byte-identically.
2. An old-style dict with no target key still loads and falls back to target := q/qdot
   (``resolve_joint_targets``), and ``missing_target_asset_names`` reports it by name
   (the once-per-file warning ``MultiResetManager.__init__`` logs reads off this).
3. The bundled-multi-state case -- a single ``record_pre_reset`` call can carry more than one
   accepted state under a leading batch dim (32 capture events yielding 34 states in the measured
   run this bead is fixing) -- unbundles the new target fields identically to how the existing
   fields (root_pose etc.) already unbundle: indexing by env_ids after ``add_joint_targets`` lands
   on the SAME rows as indexing any other field in the same dict.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "uwlab_tasks/manager_based/manipulation/omnireset/mdp/reset_state_schema.py"
)
_spec = importlib.util.spec_from_file_location("reset_state_schema", _SCHEMA_PATH)
_schema = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_schema)

add_joint_targets = _schema.add_joint_targets
resolve_joint_targets = _schema.resolve_joint_targets
missing_target_asset_names = _schema.missing_target_asset_names
JOINT_POSITION_TARGET_KEY = _schema.JOINT_POSITION_TARGET_KEY
JOINT_VELOCITY_TARGET_KEY = _schema.JOINT_VELOCITY_TARGET_KEY


def test_target_round_trips_byte_identically():
    """WRITER then READER: the stored target survives unchanged, and is preferred over q/qdot."""
    n_envs, n_joints = 4, 26  # 6 arm + 20 DELTO hand, matching the plant this bead measured on
    joint_position = torch.rand(n_envs, n_joints)
    joint_velocity = torch.rand(n_envs, n_joints)
    joint_position_target = torch.rand(n_envs, n_joints)  # deliberately != joint_position
    joint_velocity_target = torch.rand(n_envs, n_joints)

    articulation_state = {"root_pose": torch.rand(n_envs, 7), "root_velocity": torch.rand(n_envs, 6)}
    add_joint_targets(articulation_state, joint_position_target, joint_velocity_target)

    assert JOINT_POSITION_TARGET_KEY in articulation_state
    assert JOINT_VELOCITY_TARGET_KEY in articulation_state
    # byte-identical, not just close: this is a straight dict write/read, no numerics involved.
    assert torch.equal(articulation_state[JOINT_POSITION_TARGET_KEY], joint_position_target)
    assert torch.equal(articulation_state[JOINT_VELOCITY_TARGET_KEY], joint_velocity_target)

    resolved_pos, had_pos, resolved_vel, had_vel = resolve_joint_targets(
        articulation_state, joint_position, joint_velocity
    )
    assert had_pos is True
    assert had_vel is True
    assert torch.equal(resolved_pos, joint_position_target)
    assert torch.equal(resolved_vel, joint_velocity_target)
    # the resolved target must be the STORED one, not silently collapsed to q/qdot.
    assert not torch.equal(resolved_pos, joint_position)
    assert not torch.equal(resolved_vel, joint_velocity)


def test_old_style_dict_falls_back_and_is_named_as_missing():
    """A dict with no target key (a pre-UWLab-algw.7 bank) must still load -- target := q/qdot --
    not raise, and must be identified by name for the once-per-file warning."""
    n_envs, n_joints = 3, 26
    joint_position = torch.rand(n_envs, n_joints)
    joint_velocity = torch.rand(n_envs, n_joints)
    old_style_articulation_state = {
        "root_pose": torch.rand(n_envs, 7),
        "root_velocity": torch.rand(n_envs, 6),
        "joint_position": joint_position,
        "joint_velocity": joint_velocity,
        # no joint_position_target / joint_velocity_target -- this is the pre-fix schema.
    }

    resolved_pos, had_pos, resolved_vel, had_vel = resolve_joint_targets(
        old_style_articulation_state, joint_position, joint_velocity
    )
    assert had_pos is False
    assert had_vel is False
    assert torch.equal(resolved_pos, joint_position)  # today's target := q fallback, unchanged
    assert torch.equal(resolved_vel, joint_velocity)

    loaded_dataset_articulations = {"robot": old_style_articulation_state}
    assert missing_target_asset_names(loaded_dataset_articulations) == ["robot"]

    # a mixed file (one articulation upgraded, one not) is named precisely, not lumped/silenced.
    upgraded_state = dict(old_style_articulation_state)
    add_joint_targets(upgraded_state, joint_position.clone(), joint_velocity.clone())
    mixed = {"robot": old_style_articulation_state, "insertive_object_holder": upgraded_state}
    assert missing_target_asset_names(mixed) == ["robot"]

    # a fully-upgraded file reports nothing missing.
    assert missing_target_asset_names({"robot": upgraded_state}) == []


def test_bundled_multi_state_unbundles_consistently():
    """A single record_pre_reset call can accept MORE THAN ONE env at once (the trap: 32 capture
    events carried 34 states in the last measured run) -- assert indexing the new target fields by
    env_ids lands on the SAME rows as indexing an existing field (root_pose) in the same dict,
    exactly as StableStateRecorder's extract_env_ids_values recursion relies on."""
    n_envs, n_joints = 8, 26
    joint_position_target = torch.arange(n_envs * n_joints, dtype=torch.float32).reshape(n_envs, n_joints)
    joint_velocity_target = -joint_position_target.clone()
    root_pose = torch.arange(n_envs * 7, dtype=torch.float32).reshape(n_envs, 7)

    articulation_state = {"root_pose": root_pose}
    add_joint_targets(articulation_state, joint_position_target, joint_velocity_target)

    def extract_env_ids_values(value, env_ids):
        if isinstance(value, dict):
            return {k: extract_env_ids_values(v, env_ids) for k, v in value.items()}
        return value[env_ids]

    # a single call() bundling 3 accepted envs out of 8, in a NON-contiguous, NOT sorted order --
    # the shape a real multi-env record_pre_reset call actually produces.
    bundled_env_ids = torch.tensor([5, 1, 6])
    sampled = extract_env_ids_values(articulation_state, bundled_env_ids)

    assert sampled["root_pose"].shape == (3, 7)
    assert sampled[JOINT_POSITION_TARGET_KEY].shape == (3, n_joints)
    assert sampled[JOINT_VELOCITY_TARGET_KEY].shape == (3, n_joints)
    # every field indexed by the SAME env_ids must agree on WHICH source rows it carries: row i of
    # every sampled field must trace back to the same original env id.
    for i, env_id in enumerate(bundled_env_ids.tolist()):
        assert torch.equal(sampled["root_pose"][i], root_pose[env_id])
        assert torch.equal(sampled[JOINT_POSITION_TARGET_KEY][i], joint_position_target[env_id])
        assert torch.equal(sampled[JOINT_VELOCITY_TARGET_KEY][i], joint_velocity_target[env_id])


if __name__ == "__main__":
    test_target_round_trips_byte_identically()
    print("PASS: test_target_round_trips_byte_identically")
    test_old_style_dict_falls_back_and_is_named_as_missing()
    print("PASS: test_old_style_dict_falls_back_and_is_named_as_missing")
    test_bundled_multi_state_unbundles_consistently()
    print("PASS: test_bundled_multi_state_unbundles_consistently")
    print("ALL PASS")
