# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object spawn RESET events for dexlift.

This module exists because ``isaaclab.envs.mdp.events.reset_root_state_uniform`` samples z the
same way it samples x and y: a fixed offset from a default root height, independent of the sampled
orientation. That is fine for a sphere or an axis-aligned box that is never expected to sit near a
support surface, but it CANNOT express "spawn within N mm of the table" for an elongated box whose
lowest point moves by up to its own half-length as it tumbles: a flat leg's lowest corner sits 15 mm
below its origin, a vertical leg's sits 100 mm below. A fixed origin height that is safe for one
orientation buries the box in the table for another.

``reset_object_pose_with_clearance`` below is that same event, with x/y/roll/pitch/yaw sampled
IDENTICALLY to ``reset_root_state_uniform`` (same ``sample_uniform`` calls, same
``quat_from_euler_xyz``/``quat_mul`` composition against the asset's default orientation, same
``env.scene.env_origins`` handling), and z DERIVED from the sampled orientation instead of sampled
independently: z = surface_z + clearance + lowest_offset(orientation), where lowest_offset is how
far the box's lowest corner sits below its own origin at that orientation. This gives EXACT
clearance at any orientation ONLY if ``local_centre_offset`` (below) is right: the leg's bounding-
box centre is NOT at its prim origin (measured 6.2 mm off along local x on the actual USD -- see
that parameter's docstring), and the offset shifts which corner is lowest by a few millimetres
depending on orientation. Get ``local_centre_offset`` wrong (e.g. leave it zero) and the exact-
clearance guarantee silently degrades to an approximate one -- no crash, no penetration, just a
clearance that is a few mm short of what was asked for. See
``dexlift_ur5e_delto_env_cfg.DEXLIFT_SPAWN_CLEARANCE`` for the toggle that switches ``reset_object``
to this term.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["reset_object_pose_with_clearance"]


def reset_object_pose_with_clearance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, tuple[float, float]],
    clearance_range: tuple[float, float],
    half_extents: tuple[float, float, float],
    surface_z: float,
    local_centre_offset: tuple[float, float, float],
):
    """Reset the asset root pose like ``reset_root_state_uniform``, but derive z from clearance.

    x, y, roll, pitch and yaw are sampled exactly as ``reset_root_state_uniform`` samples them
    (any ``pose_range["z"]`` entry is ignored -- z is not sampled here, it is derived). z is set so
    the box's LOWEST corner, at the sampled orientation, sits ``clearance`` metres above
    ``surface_z`` (both in the asset's own ENV frame, i.e. before ``env.scene.env_origins`` is
    added, matching every other height in this task's env-frame constants).

    Args:
        env: The environment.
        env_ids: Environment indices to reset.
        asset_cfg: The rigid/articulated asset to reset (its root state).
        pose_range: Dict with optional ``x``/``y``/``roll``/``pitch``/``yaw`` (min, max) tuples,
            sampled the same way ``reset_root_state_uniform`` samples them. A ``z`` entry, if
            present, is ignored.
        clearance_range: (min, max) metres of clearance above ``surface_z``, sampled uniformly
            per environment.
        half_extents: The object's (x, y, z) half-extents about its own origin, metres. Measured,
            not fit -- see the table-leg bbox note in the caller.
        surface_z: Height of the flat support surface, in the same env-frame convention as
            ``scene.object.init_state.pos``'s z (i.e. ``WORK_SURFACE_Z``-relative, NOT world).
        local_centre_offset: The object's LOCAL bounding-box CENTRE, relative to its own prim
            origin, metres. NOT (0, 0, 0) in general -- ``half_extents`` alone assumes the box is
            centred on its origin, which the table leg is NOT (its centre sits 6.2 mm off-origin
            along local x; see the caller for how that was measured). Passed as a parameter rather
            than measured or hardcoded here, on instruction: a constant measured in one place and
            consumed in another, with nothing tying the two together, is a defect this project has
            hit before. If the object really is centred on its origin, pass (0.0, 0.0, 0.0)
            explicitly.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    # x, y, roll, pitch, yaw -- identical sampling to reset_root_state_uniform. z is deliberately
    # excluded: it is derived below from clearance and the sampled orientation, not sampled here.
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 5), device=asset.device)

    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 2], rand_samples[:, 3], rand_samples[:, 4])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # lowest_offset(q): how far the box's lowest corner sits below its own ORIGIN (not its centre)
    # at orientation q. A corner is centre + s, s in {+-hx, +-hy, +-hz} (the box is NOT centred on
    # its origin -- see local_centre_offset). Its world-frame z offset from the origin is
    # (R(q) @ (centre + s))_z = row2(R).centre + row2(R).s, where row2 is R's third row (R maps
    # local -> world, per isaaclab.utils.math.matrix_from_quat). The most-negative such offset over
    # all eight sign combinations of s is row2.centre - (|R[2,0]|*hx + |R[2,1]|*hy + |R[2,2]|*hz),
    # achieved by picking each corner sign to oppose its row entry's sign. lowest_offset is the
    # positive magnitude of that most-negative offset:
    #   lowest_offset = (|R[2,0]|*hx + |R[2,1]|*hy + |R[2,2]|*hz) - row2(R).centre
    # A box centred on its origin (centre = 0) collapses this to the |row2|.half_extents term alone.
    rotation_matrix = math_utils.matrix_from_quat(orientations)
    world_z_row = rotation_matrix[:, 2, :]  # (N, 3): (R[2,0], R[2,1], R[2,2])
    half_extents_t = torch.tensor(half_extents, device=asset.device)
    centre_t = torch.tensor(local_centre_offset, device=asset.device)
    lowest_offset = (world_z_row.abs() * half_extents_t).sum(dim=-1) - (world_z_row * centre_t).sum(dim=-1)

    clearance = math_utils.sample_uniform(clearance_range[0], clearance_range[1], (len(env_ids),), device=asset.device)
    z_env_frame = surface_z + clearance + lowest_offset

    # Same env-origin handling as reset_root_state_uniform: x/y come from the default root state
    # plus the sampled jitter, z is the derived env-frame height above -- both then shifted into
    # world coordinates by the per-environment origin.
    positions = torch.stack(
        [
            root_states[:, 0] + env.scene.env_origins[env_ids, 0] + rand_samples[:, 0],
            root_states[:, 1] + env.scene.env_origins[env_ids, 1] + rand_samples[:, 1],
            z_env_frame + env.scene.env_origins[env_ids, 2],
        ],
        dim=-1,
    )

    # set into the physics simulation; velocity is zeroed, matching this task's existing
    # reset_object term (velocity_range x/y/z all [0.0, 0.0]).
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros_like(root_states[:, 7:13]), env_ids=env_ids)
