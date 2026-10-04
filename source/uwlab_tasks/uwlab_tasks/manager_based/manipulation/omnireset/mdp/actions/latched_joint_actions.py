# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A joint action that accumulates onto a LATCHED target instead of onto the live measurement.

WHY THIS EXISTS -- measured, not argued.

``RelativeJointPositionAction`` recomputes, inside ``apply_actions``::

    target = measured_joint_pos + scale * action + offset

and ``apply_actions`` runs on EVERY PHYSICS SUBSTEP (12 per control step here, dt = 1/120 with
decimation 12). A constant "bias" offset is therefore not a set point at all -- it is a RAMP of
``offset`` radians per substep, because each substep re-adds it to a measurement that has already
moved. Held against an object, the fingers drive continuously further closed until the object is
squeezed out from between them.

Measured on the C3 (``ObjectAnywhereEEGrasped``) bank, null action, 128 envs, with a 0.08 rad bias:

===========  ==========  ===============  =============  ============
step         in_hand     finger drift     tip force      cube-to-tip
===========  ==========  ===============  =============  ============
5            0.461       +0.132 rad       0.91 N         0.083 m
20           0.078       +0.365 rad       0.16 N         0.205 m
120          0.000       +0.453 rad       0.00 N         0.212 m
===========  ==========  ===============  =============  ============

The fingers travel 0.45 rad past the posture the reset wrote, from a nominal 0.08 rad set point, and
end as a closed fist holding nothing. Grip force peaks at film-level (~0.9-1.2 N, the reset films
measure ~1.2 N) and then vanishes, because the cube is gone -- this is EJECTION, not slipping.

Holding an ABSOLUTE target latched at reset, plus a fixed 0.05 rad squeeze, instead gives::

    step   5: in_hand 0.867   finger drift +0.012 rad   tip force 1.72 N
    step  20: in_hand 0.391   finger drift +0.028 rad   tip force 0.85 N
    step 120: in_hand 0.117   finger drift +0.044 rad   tip force 0.23 N

Same hand, same cube, same banks: the fingers settle instead of ramping, and early holding goes from
0.461 to 0.867. This term makes that the action semantics rather than a probe-only hack.

SEMANTICS. The term keeps its own target buffer:

* ``reset``  -- latch the target from the joint position target the reset wrote (the bank's recorded
  squeeze), falling back to the measured position when no target exists, then add ``grip_bias`` ONCE.
  A one-shot squeeze is the whole point: adding it per step is precisely the ramp being fixed.
* ``process_actions`` -- ``target += scale * action`` (clipped), once per control step.
* ``apply_actions`` -- re-issue that same latched target on every substep. No measurement feedback,
  so nothing accumulates.

Action 0 therefore means HOLD THE CURRENT GRASP, which is what every ``*EEGrasped`` reset family
needs and what neither the relative term (ramps shut) nor a plain absolute term (commands a fixed
posture unrelated to what the reset placed in the hand) provides. The policy keeps full authority:
each step it may add up to ``scale * clip`` radians in either direction, so opening remains available.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class LatchedJointPositionAction(ActionTerm):
    """Accumulate policy deltas onto a target latched at reset. See the module docstring."""

    cfg: LatchedJointPositionActionCfg
    _asset: Articulation

    def __init__(self, cfg: LatchedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_dof = len(self._joint_ids)

        def _per_joint(value: float | dict[str, float], what: str) -> torch.Tensor:
            if isinstance(value, dict):
                missing = [n for n in self._joint_names if n not in value]
                if missing:
                    raise ValueError(f"{type(self).__name__}: {what} missing entries for {missing}")
                per_joint = [float(value[n]) for n in self._joint_names]
                return torch.tensor(per_joint, device=self.device, dtype=torch.float32)
            return torch.full((self._num_dof,), float(value), device=self.device, dtype=torch.float32)

        self._scale = _per_joint(self.cfg.scale, "scale")
        self._grip_bias = _per_joint(self.cfg.grip_bias, "grip_bias")
        self._clip = float(self.cfg.clip_per_step)

        # HELD joints: driven to a latched target like the others, but NOT exposed to the policy, so
        # adding them does not change ``action_dim`` and existing checkpoints still load. This is how
        # the undriven fingers 3/4/5 get a posture: the two-finger variant leaves them out of the
        # action term entirely, so they hold whatever the bank recorded and splay open near the table.
        self._hold_ids: list[int] = []
        self._hold_names: list[str] = []
        self._hold_target: torch.Tensor | None = None
        if self.cfg.hold_joint_names:
            self._hold_ids, self._hold_names = self._asset.find_joints(self.cfg.hold_joint_names)
            n_hold = len(self._hold_ids)
            bias = self.cfg.hold_bias
            if isinstance(bias, dict):
                missing = [n for n in self._hold_names if n not in bias]
                if missing:
                    raise ValueError(f"{type(self).__name__}: hold_bias missing entries for {missing}")
                self._hold_bias = torch.tensor(
                    [float(bias[n]) for n in self._hold_names], device=self.device, dtype=torch.float32
                )
            else:
                self._hold_bias = torch.full((n_hold,), float(bias), device=self.device, dtype=torch.float32)
            self._hold_target = torch.zeros(self.num_envs, n_hold, device=self.device)

        self._raw_actions = torch.zeros(self.num_envs, self._num_dof, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._target = torch.zeros_like(self._raw_actions)

        limits = self._asset.data.joint_pos_limits
        limits = getattr(limits, "torch", limits)
        self._lower = limits[:, self._joint_ids, 0].clone()
        self._upper = limits[:, self._joint_ids, 1].clone()

    @property
    def action_dim(self) -> int:
        return self._num_dof

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Latch the grasp the reset produced, plus a one-shot squeeze."""
        idx = slice(None) if env_ids is None else env_ids
        self._raw_actions[idx] = 0.0
        # Prefer the target the reset wrote (MultiResetManager restores the bank's recorded
        # joint_position_target); fall back to the measurement when no target is meaningful.
        target = getattr(self._asset.data, "joint_pos_target", None)
        target = getattr(target, "torch", target)
        if target is None:
            target = getattr(self._asset.data.joint_pos, "torch", self._asset.data.joint_pos)
        latched = target[:, self._joint_ids]
        if env_ids is None:
            self._target[:] = latched + self._grip_bias
        else:
            self._target[env_ids] = latched[env_ids] + self._grip_bias
        self._target[idx] = torch.clamp(self._target[idx], self._lower[idx], self._upper[idx])

        if self._hold_target is not None:
            held = target[:, self._hold_ids]
            if env_ids is None:
                self._hold_target[:] = held + self._hold_bias
            else:
                self._hold_target[env_ids] = held[env_ids] + self._hold_bias
            lo = self._asset.data.joint_pos_limits
            lo = getattr(lo, "torch", lo)
            self._hold_target[idx] = torch.clamp(
                self._hold_target[idx], lo[idx][:, self._hold_ids, 0], lo[idx][:, self._hold_ids, 1]
            )

    def process_actions(self, actions: torch.Tensor):
        """Once per control step: advance the latched target by the policy's delta."""
        self._raw_actions[:] = actions
        delta = actions * self._scale
        if self._clip > 0.0:
            delta = torch.clamp(delta, -self._clip, self._clip)
        self._processed_actions[:] = delta
        self._target[:] = torch.clamp(self._target + delta, self._lower, self._upper)

    def apply_actions(self):
        """Every substep: re-issue the SAME target. No measurement feedback, so nothing ramps."""
        self._asset.set_joint_position_target(self._target, joint_ids=self._joint_ids)
        if self._hold_target is not None:
            self._asset.set_joint_position_target(self._hold_target, joint_ids=self._hold_ids)


@configclass
class LatchedJointPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`LatchedJointPositionAction`."""

    class_type: type[ActionTerm] = LatchedJointPositionAction

    joint_names: list[str] = []
    """Joint names or regexes the term drives."""

    scale: float | dict[str, float] = 0.1
    """Radians of target change per unit action, per control step."""

    grip_bias: float | dict[str, float] = 0.0
    """One-shot squeeze added to the latched target at reset. NOT applied per step -- that is the
    ramp this term exists to remove. 0.05 rad measured 1.72 N at the fingertips on the pinch."""

    clip_per_step: float = 1.0
    """Maximum radians the target may move in one control step, after scaling. 0 disables."""

    hold_joint_names: list[str] = []
    """Joints held at a latched target but NOT exposed to the policy.

    These do not count toward ``action_dim``, so adding them leaves the observation width and every
    existing checkpoint intact. The use is the two-finger pinch, where fingers 3/4/5 are driven by
    nothing and splay open beside the cube; giving them a curl keeps them clear of the table without
    putting twelve joints the policy cannot use into its action space.
    """

    hold_bias: float | dict[str, float] = 0.0
    """One-shot curl added to each held joint's latched target at reset, in radians."""
