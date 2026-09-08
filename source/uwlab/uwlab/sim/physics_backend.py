# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-aware physics tuning for the Isaac Lab 3.0 multi-backend split.

Isaac Lab 3.0 decoupled the physics backend: ``SimulationCfg.physics`` is a
``PhysxCfg`` or a ``NewtonCfg`` (Newton = MuJoCo-Warp under the hood). Which one is
live here is chosen by ``UWLAB_PHYSICS_BACKEND`` (see the patched
``isaaclab/sim/simulation_cfg.py``).

**Why this module exists.** ``NewtonCfg`` is a configclass that TOLERATES unknown
attributes. So every one of this repo's ~73 PhysX tuning assignments --
``solver_type``, ``max_position_iteration_count``, ``gpu_collision_stack_size``,
``friction_offset_threshold``, ``bounce_threshold_velocity``, the ``gpu_*``
capacities -- will happily set an attribute that the Newton manager never reads.
No error, no warning, and every solver setting this repo spent a campaign tuning
silently evaporates.

That is precisely the failure mode this codebase keeps getting bitten by (an
undersized collision stack logs "Contacts have been dropped" and continues; a stale
``grasp_center_offset`` cost 6144 attempts -> 0 successes). So the rule here is:
PhysX tuning is applied ONLY on PhysX, and on Newton it is recorded and reported,
never silently dropped.

Newton has no equivalent for most of these. Its knobs are ``num_substeps``,
``solver_cfg`` (e.g. ``MJWarpSolverCfg``) and ``collision_cfg``. Isaac Lab's own
docs are explicit that this is not a free swap:

    "Behaviour on stiff contact stacks can diverge from PhysX; expect to retune
    contact and substep parameters when porting tasks across backends."

A 20-DOF hand pinching a 34 mm cube is a stiff contact stack. Nothing calibrated
against PhysX -- the 24.67-41.48 mm grasp window, the 0.06 N*m distal effort limits,
``grasp_center_offset`` (fitted by PhysX replay) -- carries over unexamined.
"""

from __future__ import annotations

import os

__all__ = [
    "backend",
    "is_newton",
    "is_physx",
    "apply_physx_tuning",
    "set_gravity",
    "dropped_settings",
    "report_dropped",
]

# PhysX-only fields this repo sets. Anything named here is applied on PhysX and
# recorded (not applied) on Newton.
_PHYSX_ONLY = frozenset({
    "solver_type",
    "min_position_iteration_count",
    "max_position_iteration_count",
    "min_velocity_iteration_count",
    "max_velocity_iteration_count",
    "bounce_threshold_velocity",
    "friction_offset_threshold",
    "friction_correlation_distance",
    "enable_stabilization",
    "gpu_max_rigid_contact_count",
    "gpu_max_rigid_patch_count",
    "gpu_found_lost_pairs_capacity",
    "gpu_found_lost_aggregate_pairs_capacity",
    "gpu_total_aggregate_pairs_capacity",
    "gpu_collision_stack_size",
    "gpu_heap_capacity",
    "gpu_temp_buffer_capacity",
    "gpu_max_num_partitions",
    "gpu_max_soft_body_contacts",
    "gpu_max_particle_contacts",
})

_dropped: dict[str, object] = {}


def backend() -> str:
    """``"physx"`` or ``"newton"``, from ``UWLAB_PHYSICS_BACKEND`` (default physx)."""
    return os.environ.get("UWLAB_PHYSICS_BACKEND", "physx").strip().lower() or "physx"


def is_newton() -> bool:
    return backend() == "newton"


def is_physx() -> bool:
    return not is_newton()


def apply_physx_tuning(cfg, **settings) -> None:
    """Apply PhysX-only solver settings, or record them as dropped on Newton.

    Args:
        cfg: the env cfg (anything with ``.sim.physics``).
        **settings: PhysX field name -> value.

    Raises:
        KeyError: if a setting is not a known PhysX-only field. That is deliberate --
            a typo must not be silently written onto a config nobody reads.
    """
    unknown = set(settings) - _PHYSX_ONLY
    if unknown:
        raise KeyError(
            f"not known PhysX-only fields: {sorted(unknown)}. Add them to _PHYSX_ONLY if they"
            " really are PhysX-only, or set them directly if they are backend-neutral."
        )
    if is_newton():
        _dropped.update(settings)
        return
    for key, value in settings.items():
        setattr(cfg.sim.physics, key, value)


def dropped_settings() -> dict[str, object]:
    """PhysX settings that were requested but not applied, because Newton is live."""
    return dict(_dropped)


def report_dropped(logger=print) -> None:
    """Print what was dropped. Call once after cfg construction, before training."""
    if not _dropped:
        return
    logger(
        f"[uwlab.physics_backend] backend={backend()!r}: {len(_dropped)} PhysX-only solver"
        " settings were NOT applied. Newton has no equivalent; these must be re-derived"
        " against the Newton solver, not assumed:"
    )
    for key in sorted(_dropped):
        logger(f"    {key} = {_dropped[key]!r}")


def set_gravity(vec) -> None:
    """Set world gravity on whichever backend is live.

    UWLab's OmniReset grasp sampler toggles gravity mid-episode (off while the grasp
    is established, on at t=10 s, then a mass-scaled shake), and did it through the
    PhysX tensor API: ``SimulationContext.instance().physics_sim_view.set_gravity(...)``.
    On Newton ``physics_sim_view`` is a list, so that call raises AttributeError.

    Newton keeps gravity on the Model (``NewtonManager._model.set_gravity``), so this
    routes to whichever one is live.

    Args:
        vec: (x, y, z) in m/s^2.
    """
    vec = tuple(float(v) for v in vec)
    if is_newton():
        from isaaclab_newton.physics.newton_manager import NewtonManager

        model = getattr(NewtonManager, "_model", None)
        if model is None:
            raise RuntimeError("Newton model not initialised; cannot set gravity yet.")
        model.set_gravity(vec)
        return

    import carb

    from isaaclab.sim import SimulationContext

    SimulationContext.instance().physics_sim_view.set_gravity(carb.Float3(*vec))
