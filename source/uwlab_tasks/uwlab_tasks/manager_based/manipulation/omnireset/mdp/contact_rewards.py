# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-gated reward terms, PORTED VERBATIM from ``dexlift/mdp/rewards.py``.

WHY THIS MODULE EXISTS AT ALL. The OmniReset reward graph is a pure function of two rigid-body
poses: ``ee_asset_distance`` reads the palm body ``rl_dg_mount``, ``dense_success_reward`` and
``success_reward`` read ``ProgressContext``'s cube-to-goal error, and the three safety terms read
actions and joint velocities (``config/ur5e_robotiq_2f85/rl_state_cfg.py`` ``RewardsCfg``). Nothing
reads a contact force, a finger joint angle or an aperture, so on a twenty-DOF DELTO hand flexing
every finger changes the total reward by EXACTLY ZERO minus a small action penalty. Measured
consequence on cube stacking: 9.26% genuine success on the NEAR-GOAL family, where the cube starts
already held and the geometric graph is therefore complete, and 0.0% on the other three families,
every one of which requires closing the hand first.

WHY COPIED RATHER THAN IMPORTED. Three reasons, in descending order of severity:

1. ``dexlift.mdp`` already imports ``omnireset.mdp``: ``c1_hand_pose.py:71-72`` and ``c3_rung.py:80``
   and ``episode_mixture.py:193`` and ``partial_assembly.py:127`` all pull
   ``omnireset.mdp.events``/``utils``. The dependency is one-way today. Importing dexlift from an
   omnireset module would close the cycle.
2. ``dexlift/mdp/__init__.py`` star-imports the vendored ``isaaclab_tasks ... dexsuite.mdp`` plus
   roughly fourteen local submodules (held_check, table_leg, c3_rung, gate_proxy, ...). Cube
   stacking would pay all of that at env-construction time for two callables.
3. ``dexlift.mdp.success_reward`` and ``omnireset.mdp.success_reward`` are DIFFERENT functions with
   the same name -- dexlift's own ``__init__`` docstring says it deliberately shadows the upstream
   one. Bringing that namespace anywhere near the OmniReset configs invites a silent shadow of the
   term ``RewardsCfg.success_reward`` actually binds.

The bodies below are byte-for-byte the dexlift originals (``dexlift/mdp/rewards.py`` lines 40-110
and 262-290, ``_sensor_force_magnitudes``/``any_contact``/``contacts``/
``object_upward_velocity_bonus``); only this module docstring is new. They are NOT paraphrased,
because the whole point of porting instead of writing is that this physics is already validated on
this exact hand.

The ``object_hand_s`` branch of ``_sensor_force_magnitudes`` is dormant for the cube-stacking wiring
(``ur10e_delto_cubestack_cfg._apply_fingertip_contact_sensors`` registers the per-tip
``{tip}_object_s`` sensors instead), but it is kept verbatim: it is the cheaper one-view form, it is
already used against an OmniReset scene by ``scripts_v2/tools/validate_c4_bank.py:452-455``, and it
is the fallback if PhysX ever complains about the per-tip views.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _sensor_force_magnitudes(env: ManagerBasedRLEnv, contact_names: list[str] | tuple[str, ...]) -> torch.Tensor:
    """Magnitude of the filtered contact force reported by each named fingertip sensor.

    Each name refers to a :class:`~isaaclab.sensors.ContactSensor` registered in the scene as
    ``f"{name}_object_s"`` and filtered to the object, so ``force_matrix_w`` holds a single
    contact pair per environment.

    Returns:
        Tensor of shape ``(num_envs, len(contact_names))``.
    """
    # The table-leg task uses one object-centric one-to-many sensor instead of
    # one PhysX filtered-contact view per phalanx.  At 4096 environments the
    # latter creates 28 replicated views and dominates both memory and contact
    # patch capacity.  The force is equal-and-opposite, so reading it on the
    # object retains the exact per-body contact magnitudes.
    if "object_hand_s" in env.scene.sensors:
        sensor: ContactSensor = env.scene.sensors["object_hand_s"]
        if not hasattr(env, "_object_hand_contact_indices"):
            env._object_hand_contact_indices = {
                path.rsplit("/", 1)[-1]: index for index, path in enumerate(sensor.cfg.filter_prim_paths_expr)
            }
        try:
            indices = [env._object_hand_contact_indices[name] for name in contact_names]
        except KeyError as exc:
            raise KeyError(f"No object-hand contact filter configured for {exc.args[0]!r}") from exc
        force_w = sensor.data.force_matrix_w[:, 0, indices, :]
        return torch.linalg.vector_norm(force_w, dim=-1)

    magnitudes = []
    for name in contact_names:
        sensor: ContactSensor = env.scene.sensors[f"{name}_object_s"]
        # A sensor may cover one rigid body (the original fingertip setup) or every
        # phalange of a finger (the FurnitureBench leg task).  Reduce all matched
        # body/filter pairs to the strongest force for that logical finger.
        force_w = sensor.data.force_matrix_w.reshape(env.num_envs, -1, 3)
        magnitudes.append(torch.norm(force_w, dim=-1).amax(dim=-1))
    return torch.stack(magnitudes, dim=-1)


def any_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    contact_names: tuple[str, ...],
) -> torch.Tensor:
    """Whether any of the listed fingertips presses on the object harder than ``threshold`` newtons."""
    return _sensor_force_magnitudes(env, contact_names).gt(threshold).any(dim=-1)


def contacts(
    env: ManagerBasedRLEnv,
    threshold: float,
    thumb_contact_name: str | list[str] | tuple[str, ...],
    tip_contact_names: tuple[str, ...],
    unwanted_contact_names: tuple[str, ...] | None = None,
    max_unwanted_contact_force: float = 0.05,
) -> torch.Tensor:
    """The opposition gate shared by every task reward below.

    True where at least one thumb **and** at least one non-thumb fingertip press on the object
    harder than ``threshold`` newtons, i.e. where the object is pinched rather than merely touched.
    The DELTO hand declares two opposable digits, so ``thumb_contact_name`` accepts a sequence.
    """
    if isinstance(thumb_contact_name, str):
        thumb_contact_name = [thumb_contact_name]
    thumb_contact = _sensor_force_magnitudes(env, thumb_contact_name).gt(threshold).any(dim=-1)
    tip_contact = _sensor_force_magnitudes(env, tip_contact_names).gt(threshold).any(dim=-1)
    valid = thumb_contact & tip_contact
    if unwanted_contact_names:
        unwanted_force = _sensor_force_magnitudes(env, unwanted_contact_names).amax(dim=-1)
        valid = valid & (unwanted_force <= max_unwanted_contact_force)
    return valid


def object_upward_velocity_bonus(
    env: ManagerBasedRLEnv,
    std: float,
    thumb_contact_name: str | list[str] | tuple[str, ...],
    tip_contact_names: tuple[str, ...],
    threshold: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    unwanted_contact_names: tuple[str, ...] | None = None,
    max_unwanted_contact_force: float = 0.05,
) -> torch.Tensor:
    """Reward lifting the object while gripping it.

    Signed, so lowering a gripped object is penalized. This is the shaping that actually produces a
    lift; the base task has no explicit lift reward and relies on the gravity curriculum instead.
    """
    object: RigidObject = env.scene[object_cfg.name]
    vel_z = object.data.root_lin_vel_w[:, 2]
    reward = torch.tanh(vel_z / max(std, 1.0e-6))
    return (
        reward
        * contacts(
            env,
            threshold,
            thumb_contact_name,
            tip_contact_names,
            unwanted_contact_names,
            max_unwanted_contact_force,
        ).float()
    )


def fingertip_object_distance_tanh(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward bringing the FINGERTIPS to the object. ``1 - tanh(mean_tip_distance / std)``.

    WHY THIS IS NEEDED ALONGSIDE THE CONTACT TERMS. ``any_contact`` and ``contacts`` pay only once
    contact already exists, so they are a reward for an outcome with nothing leading to it. Measured
    on the four-family run ten iterations after they were switched on: ``any_finger_contact`` logged
    2.1e-4 and ``grasp_contact`` 4e-5, against an ``action_rate`` penalty of -2.8e-2. The signal was
    real -- it was rising, so the sensors read -- but three orders of magnitude below the noise it
    had to be found in, because nothing pulled the fingers towards the cube in the first place.

    ``ee_asset_distance`` does not fill that gap: it reads the pose of ONE body, the palm
    ``rl_dg_mount``, so it is satisfied by parking the palm at the cube with the hand wide open.
    That is exactly the behaviour the recordings show -- the arm descends, the palm arrives
    accurately, and the fingers never close. Distance from the FINGERTIPS is the quantity that
    changes when they do.

    Mean over the five tips rather than the minimum: the minimum is maximised by touching the cube
    with one finger, which is the degenerate behaviour ``any_finger_contact`` is already deliberately
    under-weighted against. The mean falls only when the hand closes around the object.
    """
    robot: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    tips = robot.data.body_pos_w[:, asset_cfg.body_ids]            # (N, n_tips, 3)
    target = obj.data.root_pos_w.unsqueeze(1)                      # (N, 1, 3)
    mean_dist = torch.norm(tips - target, dim=-1).mean(dim=1)
    return 1.0 - torch.tanh(mean_dist / std)
