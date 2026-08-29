# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac-touching half of the C3(S_t) spawn-pose-tolerance addon (bead dr-sj6.22).

``V2_C3_DESIGN.md`` sec 5 / ``V2_ACCEPTANCE_CRITERIA.md`` sec 4: S_t's acceptance criterion is
``held_with_probe`` AND "the leg still within a tolerance of its own spawn pose" -- the S_t
analogue of ``scripts_v2/tools/generate_reset_states_policy.py``'s ``_SeatingGateAddon`` /
``SeatedHeldWithProbe``, deliberately NOT that class: S_t is a horizontal peg with no mating frame,
and the bore-seating gate would reject ~100% of valid S_t states (same design doc, same section --
the exact trap the retracted v1 ``stays_seated`` proposal fell into for S2'). Do not compose
``_SeatingGateAddon`` with S_t.

STRUCTURED THE SAME WAY AS ``_SeatingGateAddon``: :class:`_SpawnPoseToleranceAddon` is a plain
composed object (not a mixin), constructed once per env with a ``resolve()``+assert for the
``SceneEntityCfg`` it needs -- this codebase's own documented "``SceneEntityCfg.resolve()`` failure
swallowed inside a deferred 'at play' callback" trap (see ``_SeatingGateAddon``'s own docstring) --
and :class:`SpawnToleranceHeldWithProbe` ANDs its ``.check()`` onto ``held_with_probe``'s own
decision, the same shape ``SeatedHeldWithProbe`` uses.

WHY NOT ``env.command_manager``'s ``"object_pose"`` COMMAND. ``V2_C3_DESIGN.md``'s S_t maps onto
``GoalAtSpawnPoseCommand`` (``partial_assembly.py:258``), which pins the GOAL to the object's OWN
spawn pose -- so on that specific episode-mixture branch, reading the live goal command and reading
the object's own recorded spawn pose are the SAME quantity, and ``success.py``'s
``goal_pose_error``/``within_success_tolerance`` (``success.py:225``) could in principle be reused
directly. This addon does not take that path: it does not assume that wiring is active for whatever
run constructs it (a unit-style generation run, a future ``--reset_type`` that never sets up the
episode mixture, or a smoke with a different goal command entirely would silently make the goal
command mean something else, or not exist at all) -- it captures the object's OWN pose directly,
off the object itself, the SAME idiom
``generate_reset_states_policy.py``'s own ``_C2RewindBank`` already uses for its
``spawn_obj_pos`` bookkeeping. This keeps the addon correct regardless of what command machinery a
given run happens to have wired, and avoids a further instance of this project's own recurring
defect (F27, ``V2_POSE_FINDINGS.md``): a value established under one config (the mixture's
spawn-pinned branch) silently consumed under a different one.

WHY CAPTURE ON THE FIRST ``check()`` AFTER RESET, NOT IN ``reset()`` ITSELF. A termination term's
``reset()`` runs from inside ``ManagerBasedRLEnv._reset_idx``, whose ordering relative to the
reset-mode event that WRITES the object's fresh spawn pose is not something this addon wants to
depend on without a live Isaac run to verify it against (unlike ``held_check.py``'s own extensively
measured ordering notes for ITS caching, which this module has no equivalent verification for).
``held_with_probe.reset`` (``held_check.py:137-146``) itself never reads scene state in ``reset()``
for exactly this class of reason. So :meth:`_SpawnPoseToleranceAddon.reset` only marks the env as
needing a fresh capture; the capture itself happens lazily, inside :meth:`_SpawnPoseToleranceAddon.check`,
on the first call after that reset -- which always runs from a normal ``env.step()``, where
``obj.data.root_pos_w`` is unambiguously fresh (this is the SAME point in the control loop
``held_with_probe.__call__`` itself reads scene state from). The env's distance from spawn on that
very first captured step is trivially zero, which is the correct degenerate case.

TOLERANCES ARE REQUIRED, NO DEFAULT (bead dr-sj6.24 -- OPEN in ``V2_ACCEPTANCE_CRITERIA.md`` sec
4). See :mod:`spawn_tolerance_core`'s own docstring for why: they are meant to be DERIVED from the
R4 validation run's own measured grasp-induced displacement distribution, and
:attr:`_SpawnPoseToleranceAddon.last_pos_dist_m`/:attr:`~_SpawnPoseToleranceAddon.last_rot_dist_rad`
below (surfaced through :meth:`SpawnToleranceHeldWithProbe.gate_breakdown`) exist specifically to
produce that distribution -- that measurement is the whole point of running R4 on S_t. Inventing a
plausible-looking number here instead is exactly the failure R7 exists to prevent, and this
campaign has already shipped one invented constant (``RESET_SPEC_V2.md`` sec 6 item 0, the withdrawn
``stays_seated`` 6.02%->43.19% pair).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .held_check import held_with_probe
from .spawn_tolerance_core import SpawnToleranceConfig, pose_distance, within_spawn_tolerance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg

__all__ = ["SpawnToleranceHeldWithProbe", "_SpawnPoseToleranceAddon"]


class _SpawnPoseToleranceAddon:
    """The "is the leg still within tolerance of its OWN spawn pose" AND-term (bead dr-sj6.22),
    factored out as a plain composed object exactly like ``_SeatingGateAddon`` -- see this module's
    own docstring for why S_t needs this instead of that class.

    FAILS LOUDLY AT CONSTRUCTION, not inside a deferred callback: ``object_cfg.resolve()`` is
    called here, synchronously, with an assert right after it -- the SAME defensive idiom
    ``held_with_probe``/``_SeatingGateAddon`` already use.

    TOLERANCES HAVE NO DEFAULT. ``pos_tol_m``/``rot_tol_rad`` are threaded straight into
    :class:`~.spawn_tolerance_core.SpawnToleranceConfig`, whose own ``__post_init__`` raises if
    ``pos_tol_m`` is missing or non-positive, or if ``rot_tol_rad`` is given but non-positive -- see
    that class's own docstring. This class does not add a second check on top; it relies on that
    one, so the validation rule has exactly one implementation (this project's own F27 discipline).
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg,
        pos_tol_m: float,
        rot_tol_rad: float | None = None,
    ) -> None:
        # NB: pos_tol_m/rot_tol_rad have no default at THIS signature either, deliberately -- but a
        # caller passing an explicit ``None`` (e.g. an unset argparse flag threaded straight
        # through) would not trip a bare missing-argument TypeError, so the REAL validation lives
        # in SpawnToleranceConfig.__post_init__, invoked unconditionally right here.
        self.cfg = SpawnToleranceConfig(pos_tol_m=pos_tol_m, rot_tol_rad=rot_tol_rad)

        self.object_cfg = object_cfg
        self.object_cfg.resolve(env.scene)
        assert self.object_cfg.name in env.scene.rigid_objects, (
            f"_SpawnPoseToleranceAddon: {self.object_cfg.name!r} did not resolve to a rigid object "
            "in the scene. Refusing to construct a gate that would silently have nothing to "
            "measure against."
        )

        n = env.num_envs
        device = env.device
        self._spawn_pos_w = torch.zeros(n, 3, device=device)
        self._spawn_quat_w = torch.zeros(n, 4, device=device)
        self._spawn_quat_w[:, 0] = 1.0  # identity until the first captured check() overwrites it
        # False here means "needs a fresh spawn-pose capture on the next check()" -- see this
        # module's own docstring, "WHY CAPTURE ON THE FIRST check() AFTER RESET".
        self._captured = torch.zeros(n, dtype=torch.bool, device=device)

        self.last_pos_dist_m = torch.zeros(n, device=device)
        self.last_rot_dist_rad = torch.zeros(n, device=device)
        self.last_within_tolerance = torch.zeros(n, dtype=torch.bool, device=device)

        rot_tol_str = "disabled" if self.cfg.rot_tol_rad is None else f"{math.degrees(self.cfg.rot_tol_rad):.2f}deg"
        print(
            f"[c3-st-spawn-tolerance-gate] ENABLED pos_tol={self.cfg.pos_tol_m * 1000.0:.2f}mm "
            f"rot_tol={rot_tol_str}  object={self.object_cfg.name}",
            flush=True,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Mark these envs as needing a fresh spawn-pose capture. Does NOT read scene state here --
        see this module's own docstring, "WHY CAPTURE ON THE FIRST check() AFTER RESET"."""
        if env_ids is None:
            env_ids = slice(None)
        self._captured[env_ids] = False

    def check(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        obj = env.scene[self.object_cfg.name]
        live_pos_w = obj.data.root_pos_w
        live_quat_w = obj.data.root_quat_w

        needs_capture = ~self._captured
        if needs_capture.any():
            self._spawn_pos_w[needs_capture] = live_pos_w[needs_capture]
            self._spawn_quat_w[needs_capture] = live_quat_w[needs_capture]
            self._captured[needs_capture] = True

        pos_dist_m, rot_dist_rad = pose_distance(self._spawn_pos_w, self._spawn_quat_w, live_pos_w, live_quat_w)
        self.last_pos_dist_m = pos_dist_m
        self.last_rot_dist_rad = rot_dist_rad

        within = within_spawn_tolerance(pos_dist_m, rot_dist_rad, self.cfg)
        self.last_within_tolerance = within
        return within


class SpawnToleranceHeldWithProbe(held_with_probe):
    """S_t's acceptance criterion (bead dr-sj6.22): ``held_with_probe`` AND
    ``_SpawnPoseToleranceAddon.check()``. The S_t analogue of
    ``generate_reset_states_policy.py``'s ``SeatedHeldWithProbe`` -- see this module's own
    docstring for why S_t composes THIS addon and never the bore-mating seating gate.

    CONFIGURATION COMES FROM ``env.cfg.c3_st_spawn_tolerance_config``, NOT ``cfg.params`` -- the
    SAME reason and the SAME mechanism ``SeatedHeldWithProbe`` uses ``env.cfg.c4_seating_gate_config``
    for (see that class's own docstring, "CONFIGURATION COMES FROM"): IsaacLab's manager
    construction validates ``__call__``'s signature against ``cfg.params``' keys, and
    ``TerminationManager.compute()`` re-passes ``cfg.params`` as ``**kwargs`` on EVERY step, not
    only at construction -- so a non-empty ``cfg.params`` would force ``__call__`` to declare (and
    keep in sync with) every key forever. Reading config off ``env.cfg`` instead sidesteps that,
    the same idiom ``MixtureResetObject`` already uses for its own probabilities.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        st_cfg = getattr(env.cfg, "c3_st_spawn_tolerance_config", None)
        assert st_cfg is not None, (
            "SpawnToleranceHeldWithProbe constructed but env.cfg.c3_st_spawn_tolerance_config is "
            "missing -- the caller must set env_cfg.c3_st_spawn_tolerance_config BEFORE gym.make() "
            "whenever this class is wired in as terminations.success, as a dict with an EXPLICIT "
            "'pos_tol_m' key (and optionally 'rot_tol_rad') -- see this class's own docstring, "
            "'CONFIGURATION COMES FROM', and _SpawnPoseToleranceAddon's own docstring for why "
            "there is no default to silently fall back to."
        )
        self._spawn_tolerance = _SpawnPoseToleranceAddon(
            env, self.object_cfg, st_cfg.get("pos_tol_m"), st_cfg.get("rot_tol_rad")
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        self._spawn_tolerance.reset(env_ids)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        # SAME SIGNATURE AS THE BASE CLASS, deliberately -- see SeatedHeldWithProbe's own docstring
        # on why (cfg.params must stay empty for the same reason there).
        held = super().__call__(env)
        within = self._spawn_tolerance.check(env)
        return held & within

    def gate_breakdown(self, env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
        # Base class's own breakdown, cached by super().__call__() above -- .copy() so appending
        # here never mutates the base instance's own cached dict (same discipline
        # SeatedHeldWithProbe.gate_breakdown documents for the identical pattern). "spawn_tolerance"
        # plus the raw displacement are appended so a caller can both filter on pass/fail AND
        # collect the raw (pos_dist, rot_dist) distribution R4 needs to derive the tolerances
        # themselves (bead dr-sj6.24) -- this is the "record the per-state displacement into the
        # output" requirement this class exists to satisfy.
        bd = super().gate_breakdown(env).copy()
        bd["spawn_tolerance"] = self._spawn_tolerance.last_within_tolerance.clone()
        bd["spawn_pos_dist_m"] = self._spawn_tolerance.last_pos_dist_m.clone()
        bd["spawn_rot_dist_rad"] = self._spawn_tolerance.last_rot_dist_rad.clone()
        return bd
