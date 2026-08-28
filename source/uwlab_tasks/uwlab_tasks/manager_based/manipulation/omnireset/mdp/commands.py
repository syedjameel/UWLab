# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

import inspect
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm

from ..assembly_keypoints import Offset
from . import utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import TaskCommandCfg, TaskDependentCommandCfg


# ============================================================================================
# assembled_offset reproducibility guard (bead: asset-manifest / offset-drift audit).
#
# assembled_offset (read from metadata.yaml, just above in __init__) is the geometry that DEFINES
# the task_3 success predicate: it is applied to both the insertive and receptive assets right here
# before xyz_distance / euler_xy_distance are computed against it (see _update_metrics below). The
# SAME two numbers also exist as hardcoded literals in two offline tools that must never silently
# drift from this runtime value:
#   scripts_v2/tools/validate_c4_bank_aro2_3.py:188-190   (_LEG_OFF_POS, _LEG_OFF_QUAT_WXYZ, _RECV_OFF_POS)
#   scripts_v2/tools/gen_ik_c4_reset_bank.py:556-557,609  (leg_off_pos/leg_off_quat_wxyz/recv_off_pos, x2)
# Both are annotated "re-verified directly against ... metadata.yaml ... on 2026-08-22" -- a manual,
# point-in-time check, not an enforced invariant. If metadata.yaml is ever edited (or a different
# leg/fixture variant with different offsets is swapped in) without updating those two files, EVERY
# number those tools produce -- and every bank they generate -- silently becomes wrong. This check
# makes that drift fail loudly at env construction instead.
#
# Scoped by object name (utils.object_name_from_usd) rather than checked unconditionally: TaskCommand
# is a generic insertive/receptive command term used by other asset pairs too, and this specific pair
# of literals is only valid for THIS pair.
_PINNED_OFFSET_LITERALS: dict[str, dict[str, tuple[float, ...]]] = {
    "SquareTableLeg200mmDecomp": {"pos": (-0.106203, 0.0, 0.0), "quat": (0.70710678, 0.0, 0.70710678, 0.0)},
    "SquareTableLeg200mmSdf": {"pos": (-0.106203, 0.0, 0.0), "quat": (0.70710678, 0.0, 0.70710678, 0.0)},
    "SquareTableLeg200mmSdf1024": {"pos": (-0.106203, 0.0, 0.0), "quat": (0.70710678, 0.0, 0.70710678, 0.0)},
    "SquareTableLeg200mmSdf2048": {"pos": (-0.106203, 0.0, 0.0), "quat": (0.70710678, 0.0, 0.70710678, 0.0)},
    "OneLegInsertionFixture": {"pos": (-0.056250, 0.056250, -0.009374), "quat": (1.0, 0.0, 0.0, 0.0)},
}
_PINNED_OFFSET_SOURCE = "scripts_v2/tools/validate_c4_bank_aro2_3.py:188-190 (_LEG_OFF_POS / _LEG_OFF_QUAT_WXYZ / _RECV_OFF_POS)"
_OFFSET_ATOL = 1e-9  # metres (pos), dimensionless quaternion component (quat)


def _assert_offset_matches_pinned_literals(metadata: dict, usd_path: str, role: str) -> None:
    """Fail loudly if metadata.yaml's ``assembled_offset`` disagrees with the hardcoded literals in
    ``validate_c4_bank_aro2_3.py`` for the one leg/fixture asset pair those literals cover.

    No-op for any other asset (this command term is generic; the pinned literals are not).
    """
    object_name = utils.object_name_from_usd(usd_path)
    expected = _PINNED_OFFSET_LITERALS.get(object_name)
    if expected is None:
        return  # not the audited pair -- nothing to compare against

    runtime_offset = metadata.get("assembled_offset") or {}
    runtime_pos = tuple(runtime_offset.get("pos", ()))
    runtime_quat = tuple(runtime_offset.get("quat", ()))

    for field_name, runtime_val, expected_val in (("pos", runtime_pos, expected["pos"]), ("quat", runtime_quat, expected["quat"])):
        mismatched = len(runtime_val) != len(expected_val) or any(
            abs(r - e) > _OFFSET_ATOL for r, e in zip(runtime_val, expected_val)
        )
        if mismatched:
            metadata_path = f"{usd_path.rsplit('/', 1)[0]}/metadata.yaml"
            raise ValueError(
                f"assembled_offset.{field_name} MISMATCH for {object_name!r} ({role} asset): "
                f"runtime value read from {metadata_path} = {runtime_val}, but the hardcoded literal "
                f"in {_PINNED_OFFSET_SOURCE} = {expected_val} (tolerance {_OFFSET_ATOL:g}, exceeded). "
                "EVERY task_3 distance/success number this project computes -- xyz_distance, "
                "euler_xy_distance, position_aligned, orientation_aligned below, and every offline "
                "reset-bank validator that copies these literals -- is INVALID until metadata.yaml "
                "and the literal in validate_c4_bank_aro2_3.py agree. Do not proceed until this is "
                "resolved; update whichever one is stale, do not silently pick one."
            )


class TaskDependentCommand(CommandTerm):
    cfg: TaskDependentCommandCfg

    def __init__(self, cfg: TaskDependentCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        self.reset_terms_when_resample = cfg.reset_terms_when_resample
        self.interval_reset_terms = []
        self.reset_terms = []
        self.ALL_INDICES = torch.arange(self.num_envs, device=self.device)
        for name, term_cfg in self.reset_terms_when_resample.items():
            if not (term_cfg.mode == "reset" or term_cfg.mode == "interval"):
                raise ValueError(f"Term '{name}' in 'reset_terms_when_resample' must have mode 'reset' or 'interval'")
            if inspect.isclass(term_cfg.func):
                term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)
            if term_cfg.mode == "reset":
                self.reset_terms.append(term_cfg)
            elif term_cfg.mode == "interval":
                if term_cfg.interval_range_s != (0, 0):
                    raise ValueError(
                        "task dependent events term with interval mode current only supports range of (0, 0)"
                    )
                self.interval_reset_terms.append(term_cfg)

    def _resample_command(self, env_ids: Sequence[int]):
        for term in self.reset_terms:
            func = term.func
            func(self._env, env_ids, **term.params)
        for term in self.interval_reset_terms:
            func = term.func
            func.reset(env_ids)

    def _update_command(self):
        for term in self.interval_reset_terms:
            func = term.func
            func(self._env, self.ALL_INDICES, **term.params)

    def get_event(self, event_term_name: str):
        """Get the event term by name."""
        return self.reset_terms_when_resample.get(event_term_name).func


class TaskCommand(TaskDependentCommand):
    """Command generator that generates pose commands based on the terrain.

    This command generator samples the position commands from the valid patches of the terrain.
    The heading commands are either set to point towards the target or are sampled uniformly.

    It expects the terrain to have a valid flat patches under the key 'target'.
    """

    cfg: TaskCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TaskCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the terrain asset
        self.insertive_asset: Articulation | RigidObject = env.scene[cfg.insertive_asset_cfg.name]
        self.receptive_asset: Articulation | RigidObject = env.scene[cfg.receptive_asset_cfg.name]
        insertive_meta = utils.read_metadata_from_usd_directory(self.insertive_asset.cfg.spawn.usd_path)
        receptive_meta = utils.read_metadata_from_usd_directory(self.receptive_asset.cfg.spawn.usd_path)
        _assert_offset_matches_pinned_literals(
            insertive_meta, self.insertive_asset.cfg.spawn.usd_path, "insertive"
        )
        _assert_offset_matches_pinned_literals(
            receptive_meta, self.receptive_asset.cfg.spawn.usd_path, "receptive"
        )
        self.insertive_asset_offset = Offset(
            pos=tuple(insertive_meta.get("assembled_offset").get("pos")),
            quat=tuple(insertive_meta.get("assembled_offset").get("quat")),
        )
        self.receptive_asset_offset = Offset(
            pos=tuple(receptive_meta.get("assembled_offset").get("pos")),
            quat=tuple(receptive_meta.get("assembled_offset").get("quat")),
        )
        self.success_position_threshold: float = receptive_meta.get("success_thresholds").get("position")
        self.success_orientation_threshold: float = receptive_meta.get("success_thresholds").get("orientation")

        self.metrics["average_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["average_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_success_rate"] = torch.zeros(self.num_envs, device=self.device)

        self.orientation_aligned = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.position_aligned = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.euler_xy_distance = torch.zeros((self._env.num_envs), device=self._env.device)
        self.xyz_distance = torch.zeros((self._env.num_envs), device=self._env.device)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 3, device=self.device)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs end of episode data
        reset_env = self._env.episode_length_buf == 0
        self.metrics["end_of_episode_rot_align_error"][reset_env] = self.euler_xy_distance[reset_env]
        self.metrics["end_of_episode_pos_align_error"][reset_env] = self.xyz_distance[reset_env]
        last_episode_success = (self.orientation_aligned & self.position_aligned)[reset_env]
        self.metrics["end_of_episode_success_rate"][reset_env] = last_episode_success.float()

        # logs current data
        insertive_asset_alignment_pos_w, insertive_asset_alignment_quat_w = self.insertive_asset_offset.apply(
            self.insertive_asset
        )
        receptive_asset_alignment_pos_w, receptive_asset_alignment_quat_w = self.receptive_asset_offset.apply(
            self.receptive_asset
        )
        insertive_asset_in_receptive_asset_frame_pos, insertive_asset_in_receptive_asset_frame_quat = (
            math_utils.subtract_frame_transforms(
                receptive_asset_alignment_pos_w,
                receptive_asset_alignment_quat_w,
                insertive_asset_alignment_pos_w,
                insertive_asset_alignment_quat_w,
            )
        )
        e_x, e_y, _ = math_utils.euler_xyz_from_quat(insertive_asset_in_receptive_asset_frame_quat)
        self.euler_xy_distance[:] = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
        self.xyz_distance[:] = torch.norm(insertive_asset_in_receptive_asset_frame_pos, dim=1)
        self.position_aligned[:] = self.xyz_distance < self.success_position_threshold
        self.orientation_aligned[:] = self.euler_xy_distance < self.success_orientation_threshold
        self.metrics["average_rot_align_error"][:] = self.euler_xy_distance
        self.metrics["average_pos_align_error"][:] = self.xyz_distance

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

    def _update_command(self):
        super()._update_command()

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
