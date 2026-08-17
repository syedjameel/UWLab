# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR5e + Tesollo DELTO DG-5F variants of the OmniReset reset-state and RL-state tasks.

Structural twin of ``ur10e_delto_cfg.py`` (bead UWLab-zvd.2), with the same ``_apply_delto`` swap:
each task config subclasses its 2F-85 base, calls ``super().__post_init__()`` first, then swaps in
the DELTO robot + action and fixes everything that assumed a parallel jaw. Nothing else changes --
objects, events, rewards, sim settings and the object ``variants`` are all inherited, and the
2F-85, linear-gripper and UR10e+DELTO paths are untouched.

THE ONE STRUCTURAL SIMPLIFICATION RELATIVE TO ``ur10e_delto_cfg.py``, AND WHY: that file overrides
the arm's sysid delay range (``_set_arm_max_delay`` + ``delay_range = (0, 1)``) in its Finetune
config, because swapping to the UR10e also swaps the ARM identity away from the 2F-85 base's UR5e.
Here the arm does NOT change -- ``ur5e_delto.IMPLICIT_UR5E_DELTO`` / ``EXPLICIT_UR5E_DELTO`` are
the SAME calibrated UR5e the 2F-85 and linear-gripper variants use, just with a different hand
bolted on (``actions.py``'s ``_UR5E_DELTO_CALIBRATION_DIR`` points at the graft's own metadata,
which is a byte-identical copy of the cloud UR5e's arm-identification block -- see that file). The
inherited ``Ur5eRobotiq2f85RelCartesianOSC*Cfg`` base classes already carry the correct UR5e delay
range (``delay_range = (0, 1)``, ``rl_state_cfg.py:309/353``) for free, so no delay override is
needed or written here. This is a genuine simplification, not an omission: reintroducing the UR10e
file's override machinery on an arm that never changed would be restating a number that already
carries over correctly by inheritance.

Everything else DELTO-specific -- gripper joints, full-actuation guard, end-effector body
(``rl_dg_mount``), the roll/pitch approach-axis correction, the abnormal-robot scoping -- is
entirely inside ``_apply_delto`` (``delto_cfg.py``), which is written generically over
``(cfg, robot, action)`` and already independent of which arm the DELTO is mounted on. Nothing in
that function is duplicated or reimplemented here.

Registered gym ids (mirroring the UR10e+DELTO ones):
* ``OmniReset-UR5eDelto-ObjectAnywhereEEAnywhere-v0``
* ``OmniReset-UR5eDelto-ObjectRestingEEGrasped-v0``
* ``OmniReset-UR5eDelto-ObjectAnywhereEEGrasped-v0``
* ``OmniReset-UR5eDelto-ObjectPartiallyAssembledEEAnywhere-v0``
* ``OmniReset-UR5eDelto-ObjectPartiallyAssembledEEGrasped-v0``
* ``OmniReset-UR5eDelto-RelCartesianOSC-State-v0`` (+ Finetune / Play / Finetune-Play)

NOT registered here, because they already exist and are arm-independent (grasp sampling and
partial-assembly placement never put an arm in the scene):
* ``OmniReset-Delto-GraspSampling-v0`` (``delto_cfg.DeltoGraspSamplingCfg``) -- serves every DELTO
  arm variant, same as it already serves the UR10e one.
* ``OmniReset-PartialAssemblies-v0`` (``partial_assemblies_cfg.PartialAssembliesCfg``) -- its own
  docstring says "without robot".

NOT written here, out of zvd.1-3's scope: an RGB pipeline (camera-align / data-collection / play)
for this variant, mirroring ``ur10e_delto_rgb_cfg.py``. The UR10e+DELTO one exists; a UR5e+DELTO one
was not requested and would need its own camera-rig derivation (front/side ArUco calibration, wrist
camera on ``rl_dg_mount``) that nobody has done for this arm yet.
"""

from __future__ import annotations

from isaaclab.utils import configclass

import uwlab_assets.robots.ur5e_delto as ur5e_delto

from .actions import Ur5eDeltoRelativeOSCAction, Ur5eDeltoRelativeOSCEvalAction
from .delto_cfg import _apply_delto, _apply_delto_collision_stack_size, _apply_delto_dataset_dir
from .reset_states_cfg import (
    ObjectAnywhereEEAnywhereResetStatesCfg,
    ObjectAnywhereEEGraspedResetStatesCfg,
    ObjectPartiallyAssembledEEAnywhereResetStatesCfg,
    ObjectPartiallyAssembledEEGraspedResetStatesCfg,
    ObjectRestingEEGraspedResetStatesCfg,
)
from .rl_state_cfg import (
    Ur5eRobotiq2f85RelCartesianOSCEvalCfg,
    Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg,
    Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg,
    Ur5eRobotiq2f85RelCartesianOSCTrainCfg,
)

# Dataset root for every DELTO reset-state/grasp/partial-assembly dataset this UR5e+DELTO family
# reads or writes -- see _apply_delto_dataset_dir (delto_cfg.py) for why every dataset_dir-bearing
# event term needs this, not just one. Mirrors ur10e_delto_rgb_cfg.py's _DELTO_RESET_DIR naming
# convention, but is DELIBERATELY its own constant, not a shared/reused one: a recorded reset state
# is a full articulation snapshot and bakes in arm-specific kinematics (joint count, link lengths,
# IK solution), so a UR5e+DELTO state and a UR10e+DELTO state are NOT interchangeable, unlike
# grasps.pt, which is hand-only. Reusing the UR10e root here would silently feed UR10e-shaped
# articulation snapshots into a UR5e-DOF-shaped scene -- the same class of bug this constant exists
# to prevent, just one level up.
_UR5E_DELTO_RESET_DIR = "./Datasets_ur5e_delto/OmniReset"

# ---------------------------------------------------------------------------------------
# Reset states (full UR5e + DELTO hand)
# ---------------------------------------------------------------------------------------
@configclass
class Ur5eDeltoObjectAnywhereEEAnywhereResetStatesCfg(ObjectAnywhereEEAnywhereResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur5e_delto.IMPLICIT_UR5E_DELTO, Ur5eDeltoRelativeOSCAction())
        _apply_delto_collision_stack_size(self)
        _apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR)


@configclass
class Ur5eDeltoObjectRestingEEGraspedResetStatesCfg(ObjectRestingEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur5e_delto.IMPLICIT_UR5E_DELTO, Ur5eDeltoRelativeOSCAction())
        _apply_delto_collision_stack_size(self)
        _apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR)


@configclass
class Ur5eDeltoObjectAnywhereEEGraspedResetStatesCfg(ObjectAnywhereEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur5e_delto.IMPLICIT_UR5E_DELTO, Ur5eDeltoRelativeOSCAction())
        _apply_delto_collision_stack_size(self)
        _apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR)


@configclass
class Ur5eDeltoObjectPartiallyAssembledEEAnywhereResetStatesCfg(ObjectPartiallyAssembledEEAnywhereResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur5e_delto.IMPLICIT_UR5E_DELTO, Ur5eDeltoRelativeOSCAction())
        _apply_delto_collision_stack_size(self)
        _apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR)


@configclass
class Ur5eDeltoObjectPartiallyAssembledEEGraspedResetStatesCfg(ObjectPartiallyAssembledEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur5e_delto.IMPLICIT_UR5E_DELTO, Ur5eDeltoRelativeOSCAction())
        _apply_delto_collision_stack_size(self)
        _apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR)


# ---------------------------------------------------------------------------------------
# RL state (training / finetune / eval)
# ---------------------------------------------------------------------------------------
@configclass
class Ur5eDeltoRelCartesianOSCTrainCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    """Stage 1 training: implicit actuator, no curriculum."""

    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur5e_delto.IMPLICIT_UR5E_DELTO, Ur5eDeltoRelativeOSCAction())
        _apply_delto_collision_stack_size(self)
        _apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR)


@configclass
class Ur5eDeltoRelCartesianOSCFinetuneCfg(Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg):
    """Stage 2 finetune: explicit actuator + curriculum (base sets EXPLICIT 2F-85; we override last).

    No arm-delay override, unlike ``Ur10eDeltoRelCartesianOSCFinetuneCfg`` -- see the module
    docstring. The ADR still ramps the arm dynamics toward the identified UR5e; the payload the
    sysid fit saw was the 2F-85 (or nothing), not this 1.7735 kg hand, same caveat every other
    end-effector swap on this arm family carries and the reason the OSC rotational damping ratio is
    re-derived rather than inherited (``actions._UR5E_DELTO_ROT_DAMPING_RATIO``).
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur5e_delto.EXPLICIT_UR5E_DELTO, Ur5eDeltoRelativeOSCAction())
        _apply_delto_collision_stack_size(self)
        _apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR)


@configclass
class Ur5eDeltoRelCartesianOSCEvalCfg(Ur5eRobotiq2f85RelCartesianOSCEvalCfg):
    """Eval after Stage 1: implicit actuator, soft gains."""

    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur5e_delto.IMPLICIT_UR5E_DELTO, Ur5eDeltoRelativeOSCAction())
        _apply_delto_collision_stack_size(self)
        _apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR)


@configclass
class Ur5eDeltoRelCartesianOSCFinetuneEvalCfg(Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg):
    """Eval after Stage 2: explicit actuator, stiff eval gains.

    Uses ``Ur5eDeltoRelativeOSCEvalAction``, which carries the known kd*dt/I stability caveat (see
    ``actions.py``) -- inherited here unchanged, same as the UR10e+DELTO variant's equivalent.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur5e_delto.EXPLICIT_UR5E_DELTO, Ur5eDeltoRelativeOSCEvalAction())
        _apply_delto_collision_stack_size(self)
        _apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR)
