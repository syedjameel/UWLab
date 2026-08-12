# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frame guards: assert at RUNTIME that an asset is where the config says it is.

Written for one measured failure and generalised no further than that. ``ArticulationCfg.init_state``
is TWO things at once -- the pose the spawner writes onto the referencing Xform, and the
``default_root_state`` every reset event samples around -- and on a USD that authors a nonzero
transform between the referenced prim and the ROOT LINK those two disagree silently. Nothing in
Isaac Lab compares them, and the symptom is not an error but a scene assembled the wrong way round.

See ``dexlift_ur5e_delto_env_cfg.BASE_LINK_AUTHORED_YAW`` for the measurement that made this
necessary.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["check_root_matches_default"]


def check_root_matches_default(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pos_atol: float = 1e-3,
    rot_atol: float = 1e-3,
    context: str = "",
) -> None:
    """Raise unless the asset's ROOT LINK sits at its configured ``init_state`` pose.

    ``mode="reset"``, and it must be declared AFTER whichever term writes the root pose -- event
    terms of one mode run in config declaration order, and a subclass's fields come after its
    base's, which is what puts this after dexsuite's ``reset_root``.

    Checked ONCE, on the first reset that reaches it, and then disabled for the life of the
    environment. The invariant is static -- either a term pins the root every reset or nothing
    ever writes it -- so a per-reset check would buy nothing and would cost a GPU->CPU sync on
    every step that resets any environment, which under dexsuite's per-env terminations is every
    step.
    """
    flag = f"_frame_guard_done__{asset_cfg.name}"
    if getattr(env, flag, False):
        return
    setattr(env, flag, True)

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)

    default = asset.data.default_root_state[env_ids]
    # ``default_root_state`` positions are in the ENV frame; ``root_pos_w`` is in the world frame.
    actual_pos = asset.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    actual_quat = asset.data.root_quat_w[env_ids]

    pos_err = torch.linalg.norm(actual_pos - default[:, 0:3], dim=-1).max()
    rot_err = quat_error_magnitude(actual_quat, default[:, 3:7]).max()
    if float(pos_err) <= pos_atol and float(rot_err) <= rot_atol:
        return

    roll, pitch, yaw = (float(v[0]) for v in euler_xyz_from_quat(actual_quat[:1]))
    where = f" [{context}]" if context else ""
    raise RuntimeError(
        f"frame guard{where}: the root link of scene['{asset_cfg.name}'] is NOT at the pose its"
        f" config declares. cfg init_state pos={[round(float(v), 6) for v in default[0, 0:3]]}"
        f" rot={[round(float(v), 6) for v in default[0, 3:7]]}; measured (env frame)"
        f" pos={[round(float(v), 6) for v in actual_pos[0]]}"
        f" rot={[round(float(v), 6) for v in actual_quat[0]]}"
        f" (rpy={roll:.6f}, {pitch:.6f}, {yaw:.6f});"
        f" max position error {float(pos_err):.6f} m, max orientation error {float(rot_err):.6f} rad."
        " ``init_state`` is written onto the REFERENCING PRIM, while the articulation root is a"
        " BODY inside the referenced USD: any transform authored between the two makes the scene"
        " assemble around a frame no config file states. Every workspace number in this task -- the"
        " object spawn, the table, the viewer -- is written in the env frame, while the pose command"
        " is sampled in this root frame, so a mismatch here puts the goal and the object in"
        " different half-spaces. Restore the term that writes the root pose (dexsuite's"
        " ``events.reset_root``) rather than silencing this check."
    )
