# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-episode MIXTURE of the staged goal and a VERTICAL near-bore goal (epic UWLab-nnlv).

WHY THIS EXISTS. DexReset needs two intermediate reset rungs between C4 (leg seated in the bore)
and C3 (leg held anywhere, median 83 deg from vertical):

    S1  -- leg tip 0-10 mm INSIDE the bore mouth, vertical, grasped
    S2' -- leg tip 20-120 mm ABOVE the mouth, vertical, grasped

Measured 2026-08-26/27, the shipped ep_3600 Reorient policy cannot produce either: from a
partial-assembly spawn it flips the leg horizontal and carries it away (0 accepted even at
lateral <= 60 mm / tilt <= 60 deg), and once the leg is in the bore neither an axial pull nor an
unscrew extracts it. The policy can GRASP but cannot be TOLD WHERE TO HOLD, because its
goal-conditioning was only ever trained on the NARROWED task ``_apply_pose_tilt_stage`` defines:
``DEXLIFT_POSE_TILT=0.3`` clamps the goal's roll and pitch to +-17.2 deg, so a vertical
(tip-down, pitch ~ -pi/2) goal is outside everything it has seen.

Opening that clamp at INFERENCE already gets partway there -- the untrained-for-vertical policy
reaches 14.5-20.8 deg from vertical against a 43 deg MINIMUM in the whole shipped C3 bank -- at a
1-3 percent yield. This module closes the gap by TRAINING for it.

WHY A MIXTURE AND NOT A REPLACEMENT. Finetuning 100 percent on a changed goal distribution has
destroyed this exact policy before: a goal-at-spawn finetune lost 55 percent of its skill in 50
epochs and certified 0.0000 at 30 mm by epoch 1550, with the damage FASTEST AT THE START, so
"stop early" is not an escape hatch. The surviving remedy on record is a MIXED distribution that
keeps the original task rewarded. This term therefore draws, per episode and per environment:

    with probability ``vertical_prob``    the VERTICAL goal defined by :attr:`vertical_ranges`
    otherwise                             the inherited staged goal, byte-for-byte unchanged

WHY IT IS SAFE FROM THE SILENT-CLAMP TRAP. ``_apply_pose_tilt_stage`` writes ``cfg.ranges``, and a
hydra override of ``commands.object_pose.ranges.pitch`` is therefore silently overwritten by it --
that defect cost a whole invalid experiment. The vertical band lives in a SEPARATE field
(:attr:`vertical_ranges`) which the staging function does not touch and cannot reach, so the two
distributions cannot clobber one another whatever order the config runs in.

DEFAULT IS INERT. ``vertical_prob = 0.0`` reproduces :class:`TaskStateVisPoseCommand` exactly --
the same sampling, the same command buffer, the same metrics -- so wiring this class into a config
that does not ask for the mixture changes nothing.
"""

from __future__ import annotations

import dataclasses
import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_unique, subtract_frame_transforms

from .task_state_vis import TaskStateVisPoseCommand, TaskStateVisPoseCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class MixedGoalPoseCommand(TaskStateVisPoseCommand):
    """:class:`TaskStateVisPoseCommand` that draws a fraction of goals from a second distribution.

    Everything the policy observes keeps its meaning: the command is still a 7-vector
    ``(x, y, z, qw, qx, qy, qz)`` in the robot root frame, still resampled on the inherited
    schedule, and the success predicate, markers and rewards are untouched. The ONLY change is
    where a resampled goal is drawn from.
    """

    cfg: MixedGoalPoseCommandCfg

    def __init__(self, cfg: MixedGoalPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if not 0.0 <= cfg.vertical_prob <= 1.0:
            raise ValueError(f"vertical_prob must be in [0, 1]; got {cfg.vertical_prob}")
        # NEVER 1.0 IN TRAINING, and the guard is here rather than in a comment because the failure
        # is silent and expensive: a 100-percent finetune on a changed goal distribution removes the
        # original task from the objective, and the reward CURVE RISES while the certified score
        # goes to zero. A pure-vertical evaluation is legitimate, which is why it is a warning at
        # construction and not a hard error.
        if cfg.vertical_prob >= 1.0:
            print(
                "[dexlift] WARNING: goal mixture vertical_prob = 1.0. NO episode keeps the original"
                " goal distribution, so this run cannot preserve the parent policy's skill. This is"
                " the configuration that previously cost 55 percent of the skill in 50 epochs.",
                flush=True,
            )

        # Per-env record of which distribution the LIVE goal came from. Exposed as a metric so a
        # run can be verified against its configured fraction instead of trusted -- a cfg field
        # that failed to apply and one that applied look identical in a config dump.
        self.goal_is_vertical = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["goal_vertical_frac"] = torch.zeros(self.num_envs, device=self.device)

        rng = cfg.vertical_ranges
        print(
            f"[dexlift] GOAL MIXTURE staged: {cfg.vertical_prob:.3f} of resampled goals drawn from"
            f" the VERTICAL band roll {tuple(rng.roll)} pitch {tuple(rng.pitch)} yaw"
            f" {tuple(rng.yaw)} rad, pos_z {tuple(rng.pos_z)} m (pitch -pi/2 ="
            f" {-math.pi / 2:.4f} is leg tip-down); the remaining {1.0 - cfg.vertical_prob:.3f}"
            f" keep the inherited staged goal (roll {tuple(cfg.ranges.roll)}, pitch"
            f" {tuple(cfg.ranges.pitch)}).",
            flush=True,
        )

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["goal_vertical_frac"] = self.goal_is_vertical.float()

    def _resample_command(self, env_ids: Sequence[int]):
        # Draw the inherited distribution for EVERY env first, then overwrite the selected subset.
        # Doing it in this order means the non-vertical envs are bit-for-bit what the unmixed term
        # would have produced, and it keeps the RNG consumption of the base term intact.
        super()._resample_command(env_ids)
        if self.cfg.vertical_prob <= 0.0:
            return

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        pick = torch.rand(ids.numel(), device=self.device) < self.cfg.vertical_prob
        self.goal_is_vertical[ids] = pick
        sel = ids[pick]
        if sel.numel() == 0:
            return

        rng = self.cfg.vertical_ranges

        # -- X/Y ANCHORING (bead UWLab-nnlv.4). MEASURED CONSEQUENCE OF GETTING THIS WRONG: with
        # x/y inherited from the staged draw, the finetune taught the policy to hold the leg
        # vertical SOMEWHERE rather than vertical AT A COMMANDED PLACE, and reset-state generation
        # -- which accepts a state only if the tip is within 5 mm (S1) or 20 mm (S2') of the bore
        # axis -- produced ZERO states from 148 attempts. The banked poses under an ungated run
        # decomposed to lateral median 141 mm and tilt median 89.9 deg: the policy withdraws by
        # flipping the leg horizontal and carrying it away, because nothing in its objective ever
        # rewarded staying over the spot it started from.
        #
        # The original reasoning for inheriting x/y is preserved verbatim below because it is still
        # RIGHT about what it addresses, and only wrong about the requirement:
        #     "the bore's lateral position is randomized per episode by the fixture reset, so
        #      pinning the goal's x/y here would teach a fixed point in space instead of a
        #      commanded one."
        # A FIXED point in space would indeed be useless. Anchoring to the OBJECT'S OWN CURRENT
        # POSITION is not a fixed point -- it is "straight up from wherever the leg happens to be",
        # which moves with the object every episode and therefore stays a commanded relation. The
        # dexlift training scene has NO FIXTURE at all, so anchoring to the bore itself is not
        # available here; the object is the correct proxy, and it is an exact one for the case that
        # matters: at generation the leg spawns IN THE BORE, so straight-up-from-the-leg IS above
        # the bore.
        if self.cfg.vertical_anchor_xy == "object":
            # self.robot and self.object are the BASE CLASS's own handles (ObjectUniformPoseCommand
            # binds asset_name -> robot, object_name -> object). Reusing them rather than looking the
            # assets up again guarantees this reads the same two bodies the base class's own
            # error metric compares, so the goal and the thing it is a goal FOR cannot diverge.
            #
            # pose_command_b is in the ROBOT ROOT frame (the base class converts it to world with
            # combine_frame_transforms against robot root), so the object's world position must be
            # brought into that same frame -- the inverse transform. Read live, not cached:
            # _resample_command runs after the reset events have placed the object, so for a
            # just-reset env this IS the spawn position.
            obj_pos_b, _ = subtract_frame_transforms(
                self.robot.data.root_pos_w[sel],
                self.robot.data.root_quat_w[sel],
                self.object.data.root_pos_w[sel],
            )
            self.pose_command_b[sel, 0] = obj_pos_b[:, 0]
            self.pose_command_b[sel, 1] = obj_pos_b[:, 1]
        elif self.cfg.vertical_anchor_xy != "inherit":
            raise ValueError(
                f"vertical_anchor_xy must be 'inherit' or 'object'; got {self.cfg.vertical_anchor_xy!r}"
            )

        r = torch.empty(sel.numel(), device=self.device)
        self.pose_command_b[sel, 2] = r.uniform_(*rng.pos_z)
        euler = torch.zeros(sel.numel(), 3, device=self.device)
        euler[:, 0].uniform_(*rng.roll)
        euler[:, 1].uniform_(*rng.pitch)
        euler[:, 2].uniform_(*rng.yaw)
        quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        self.pose_command_b[sel, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat


@configclass
class MixedGoalPoseCommandCfg(TaskStateVisPoseCommandCfg):
    """Config for :class:`MixedGoalPoseCommand`. Inert at the default ``vertical_prob = 0.0``."""

    class_type: type = MixedGoalPoseCommand

    vertical_anchor_xy: str = "inherit"
    """Where the vertical goal's x and y come from: ``"inherit"`` or ``"object"``.

    ``"inherit"`` takes them from the staged draw. It is the ORIGINAL behaviour and stays the
    default so no existing config changes meaning silently -- but it is also the setting that made
    reset-state generation impossible (bead UWLab-nnlv.4): the policy learned to hold the leg
    vertical SOMEWHERE rather than at a commanded place, and generation, which accepts a state only
    if the tip is within 5 mm (S1) or 20 mm (S2') of the bore axis, banked ZERO states from 148
    attempts. The ungated poses decomposed to lateral median 141 mm, tilt median 89.9 deg.

    ``"object"`` pins x/y to the object's OWN live position -- "hold it vertical straight up from
    where it lies". This is not a fixed point in space: it moves with the object every episode, so
    it remains a commanded relation rather than a memorised location. At generation the leg spawns
    INSIDE THE BORE, so straight-up-from-the-leg is exactly above the bore, which is the relation
    the S1 and S2' rungs need and the one nothing in the objective previously rewarded.
    """

    vertical_prob: float = 0.0
    """Fraction of RESAMPLED goals drawn from :attr:`vertical_ranges` instead of ``ranges``.

    0.0 reproduces the unmixed term exactly. 1.0 is permitted for evaluation and warned about at
    construction -- see the class docstring for what a 100-percent finetune did to this policy.
    """

    @configclass
    class VerticalRanges:
        """The second distribution. Euler angles in radians, height in metres, ROBOT ROOT frame.

        Only height and orientation are specified: x and y stay on the inherited draw, because the
        fixture's lateral position is itself randomized per episode and the skill being taught is
        a COMMANDED orientation at a COMMANDED height, not a fixed point in space.

        WHAT THE BAND ACTUALLY COVERS, because reading the three angle ranges as a cone is wrong.
        At ``pitch = -pi/2`` the XYZ-Euler parameterisation is GIMBAL-DEGENERATE: ``Rz(yaw) @
        Ry(-pi/2) @ Rx(roll)`` collapses so that roll and yaw drive THE SAME degree of freedom,
        rotation about the (now vertical) leg axis. So with ``pitch`` swept about ``-pi/2`` and
        ``roll`` swept about 0, the band is not a cone around vertical -- it is a FAN (tilt off
        vertical in the pitch plane only) crossed with an AXIAL SPIN of the same angular width.
        That is adequate here because the CENTRE of the band is exactly tip-down and the spread is
        padding against a razor-thin target, not a coverage requirement; and the axial spin it does
        cover (+-0.35 rad = +-20 deg at the default tilt) contains the +-15 deg of fixture yaw
        (``partial_assembly.RECEPTIVE_POSE_RANGE``) that a generation-time spawn-pinned goal can
        present. Tilt in the plane ORTHOGONAL to pitch is not sampled. Do not widen ``yaw`` hoping
        to buy orthogonal tilt -- at this pitch it buys more spin.
        """

        roll: tuple[float, float] = MISSING
        pitch: tuple[float, float] = MISSING
        yaw: tuple[float, float] = MISSING
        pos_z: tuple[float, float] = MISSING

    vertical_ranges: VerticalRanges = MISSING
    """Bounds of the vertical goal band. Ignored entirely while ``vertical_prob`` is 0.0."""


def upgrade_pose_command_to_mixed_goal(
    command_cfg: TaskStateVisPoseCommandCfg,
    *,
    vertical_prob: float,
    roll: tuple[float, float],
    pitch: tuple[float, float],
    yaw: tuple[float, float],
    pos_z: tuple[float, float],
    anchor_xy: str = "inherit",
) -> MixedGoalPoseCommandCfg:
    """Rebuild an already-upgraded task-state-vis command as a :class:`MixedGoalPoseCommandCfg`.

    Every field is COPIED rather than restated, exactly as
    :func:`~.task_state_vis.upgrade_pose_command_to_task_state_vis` does, so the sampling ranges
    (including whatever ``_apply_pose_tilt_stage`` staged into them), the marker geometry, the
    success tolerances and ``position_only`` all survive untouched. ``class_type`` is excluded for
    the obvious reason -- it names the term being replaced.
    """
    fields = {
        field.name: getattr(command_cfg, field.name)
        for field in dataclasses.fields(command_cfg)
        if field.name != "class_type"
    }
    upgraded = MixedGoalPoseCommandCfg(**fields)
    upgraded.vertical_prob = vertical_prob
    upgraded.vertical_ranges = MixedGoalPoseCommandCfg.VerticalRanges(
        roll=roll, pitch=pitch, yaw=yaw, pos_z=pos_z
    )
    # Validated HERE as well as at construction: this runs at CONFIG time, so a typo fails before
    # Isaac boots rather than after, and the message can name the caller's spelling.
    if anchor_xy not in ("inherit", "object"):
        raise ValueError(
            f"anchor_xy must be 'inherit' or 'object'; got {anchor_xy!r}"
        )
    upgraded.vertical_anchor_xy = anchor_xy
    return upgraded
