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

import os
import pathlib

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
    ThresholdCurriculumCfg,
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
# GENERATION PLANT ALIGNMENT (bead UWLab-snuv follow-on)
#
# ``ur5e_delto.IMPLICIT_UR5E_DELTO`` / ``EXPLICIT_UR5E_DELTO`` -- what every class below spawns
# via ``_apply_delto`` -- carry the ASSET DEFAULT hand collider set (23 sdf + 5 convexHull) and
# the IDENTIFIED hand actuator (0.06-0.17 N.m / 3.0 rad/s). Every reset bank this project reads
# under ``_UR5E_DELTO_RESET_DIR``, and the certified checkpoint (``ep_3600_rew_38.38917``), were
# instead produced by dexlift's certified generation route, which unconditionally spawns
# ``ur5e_delto_hullfix3.usd`` and the reference hand-actuator block (30 N.m / 10000 rad/s) -- see
# ``uwlab_assets.robots.ur5e_delto.UR5E_DELTO_HULLFIX3_USD_NAME`` and
# ``.REFERENCE_HAND_ACTUATOR_KWARGS`` for the full derivation and why the asset default cannot be
# used as-is (a plain convexHull hand explodes the articulation at reset; the identified actuator
# caps below the closing speed a policy commands and cannot close the hand at all).
#
# Every OmniReset UR5eDelto env below needs the SAME plant its training data was generated on, so
# ``_apply_ur5e_delto_generation_plant`` is called once per class, right after ``_apply_delto``
# sets ``cfg.scene.robot``. It is NOT folded into ``_apply_delto`` itself (delto_cfg.py): that
# helper is shared with the UR10eDelto family -- a different, stronger-armed robot with its own
# plant -- and rewriting it here would silently change that family's resolved robot too.
# --------------------------------------------------------------------------------------------

_LEGACY_PLANT_ENV_VAR = "OMNIRESET_UR5EDELTO_LEGACY_PLANT"
"""Set to ``"1"`` to fall back to the asset-default identified-hand plant (sdf colliders, 0.06-
0.17 N.m / 3.0 rad/s hand actuator) instead of the generation plant every held reset bank and the
certified checkpoint were produced under. The way back, for anyone who needs to reproduce a
pre-alignment run; default is the generation plant."""


def _apply_ur5e_delto_generation_plant(cfg) -> None:
    """Point ``cfg.scene.robot`` at the hullfix3 colliders and the reference hand actuator.

    Call AFTER ``_apply_delto`` (which sets ``cfg.scene.robot``) and BEFORE
    ``_assert_ur5e_delto_generation_plant`` (which verifies the result).

    Both source values are IMPORTED from ``uwlab_assets.robots.ur5e_delto``, never retyped here:
    see that module for the full derivation and for why dexlift's own generation route consumes
    the exact same two constants.

    Mutates only ``cfg.scene.robot``'s OWN ``.spawn``/``.actuators`` attributes, and rebinds
    ``.actuators`` to a NEW dict before writing into it, so nothing here reaches back into the
    shared ``IMPLICIT_UR5E_DELTO``/``EXPLICIT_UR5E_DELTO`` module-level objects or their shared
    ``DELTO_HAND_ACTUATOR`` instance -- same defensive rebind-then-write pattern
    ``Ur5eDeltoMixinCfg.__post_init__`` (dexlift) and ``DeltoGraspSamplingCfg.__post_init__``'s
    sampler-local hand-stiffness block (delto_cfg.py) already use for this exact reason.
    """
    robot_cfg = cfg.scene.robot
    if os.environ.get(_LEGACY_PLANT_ENV_VAR) == "1":
        print(
            f"[ur5e_delto] LEGACY plant ({_LEGACY_PLANT_ENV_VAR}=1): asset-default hand colliders"
            " + identified hand actuator (0.06-0.17 N.m / 3.0 rad/s) -- NOT what any held reset"
            " bank or the certified checkpoint was generated under.",
            flush=True,
        )
        return

    hullfix3_path = pathlib.Path(robot_cfg.spawn.usd_path).with_name(ur5e_delto.UR5E_DELTO_HULLFIX3_USD_NAME)
    if not hullfix3_path.is_file():
        raise FileNotFoundError(
            f"{hullfix3_path} does not exist. Generate it with scratchpad/reauthor_hullfix.py"
            " (dexlift), which verifies the saved layer by re-reading it. The OmniReset UR5eDelto"
            f" generation plant requires this asset; set {_LEGACY_PLANT_ENV_VAR}=1 to fall back to"
            " the asset-default identified-hand plant instead."
        )
    robot_cfg.spawn = robot_cfg.spawn.replace(usd_path=str(hullfix3_path))

    robot_cfg.actuators = dict(robot_cfg.actuators)
    robot_cfg.actuators["hand"] = robot_cfg.actuators["hand"].replace(**ur5e_delto.REFERENCE_HAND_ACTUATOR_KWARGS)

    print(
        "[ur5e_delto] generation plant: hullfix3 colliders (25 convexHull + 3 convexDecomposition,"
        f" zero sdf) + reference hand actuator (30 N.m / 10000 rad/s) usd={hullfix3_path.name}",
        flush=True,
    )


def _assert_ur5e_delto_generation_plant(cfg) -> None:
    """Fail loudly at CONSTRUCTION if the resolved robot is not on the intended plant.

    Call last in every UR5eDelto task config's ``__post_init__``, after every other mutation of
    ``cfg.scene.robot`` in that chain -- so a future edit that reintroduces the asset-default
    plant, reorders these calls, or adds a call site that forgets
    ``_apply_ur5e_delto_generation_plant`` cannot reach training silently.

    Raising HERE, at construction, is deliberate rather than incidental: in this codebase an
    exception raised while an event term is lazily instantiated is SWALLOWED, ``term_cfg.func``
    is left as the raw class, and the failure resurfaces at the first reset as a misleading
    "unexpected keyword argument" error with no trace back to the actual cause. A guard that dies
    quietly is worse than none.
    """
    if os.environ.get(_LEGACY_PLANT_ENV_VAR) == "1":
        return

    robot_cfg = cfg.scene.robot
    usd_path = robot_cfg.spawn.usd_path
    if not usd_path.endswith(ur5e_delto.UR5E_DELTO_HULLFIX3_USD_NAME):
        raise RuntimeError(
            f"{type(cfg).__name__}: robot spawn usd_path is {usd_path!r}, expected it to end with"
            f" {ur5e_delto.UR5E_DELTO_HULLFIX3_USD_NAME!r}. The UR5eDelto generation plant was not"
            " applied (or was overwritten after being applied) -- every held reset bank and the"
            f" certified checkpoint were generated on hullfix3. Set {_LEGACY_PLANT_ENV_VAR}=1 if"
            " the asset-default identified-hand plant is genuinely intended here."
        )
    hand = robot_cfg.actuators["hand"]
    for key, expected in ur5e_delto.REFERENCE_HAND_ACTUATOR_KWARGS.items():
        actual = getattr(hand, key)
        if actual != expected:
            raise RuntimeError(
                f"{type(cfg).__name__}: hand actuator {key!r} is {actual!r}, expected {expected!r}"
                " (the reference block every held reset bank and the certified checkpoint were"
                f" generated under). Set {_LEGACY_PLANT_ENV_VAR}=1 if the identified hand actuator"
                " is genuinely intended here."
            )


# ---------------------------------------------------------------------------------------
# Reset states (full UR5e + DELTO hand)
# ---------------------------------------------------------------------------------------
@configclass
class Ur5eDeltoObjectAnywhereEEAnywhereResetStatesCfg(ObjectAnywhereEEAnywhereResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur5e_delto.IMPLICIT_UR5E_DELTO, Ur5eDeltoRelativeOSCAction())
        _apply_ur5e_delto_generation_plant(self)
        _apply_delto_collision_stack_size(self)
        _apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR)
        _assert_ur5e_delto_generation_plant(self)


@configclass
class Ur5eDeltoObjectRestingEEGraspedResetStatesCfg(ObjectRestingEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur5e_delto.IMPLICIT_UR5E_DELTO, Ur5eDeltoRelativeOSCAction())
        _apply_ur5e_delto_generation_plant(self)
        _apply_delto_collision_stack_size(self)
        _apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR)
        _assert_ur5e_delto_generation_plant(self)


@configclass
class Ur5eDeltoObjectAnywhereEEGraspedResetStatesCfg(ObjectAnywhereEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur5e_delto.IMPLICIT_UR5E_DELTO, Ur5eDeltoRelativeOSCAction())
        _apply_ur5e_delto_generation_plant(self)
        _apply_delto_collision_stack_size(self)
        _apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR)
        _assert_ur5e_delto_generation_plant(self)


@configclass
class Ur5eDeltoObjectPartiallyAssembledEEAnywhereResetStatesCfg(ObjectPartiallyAssembledEEAnywhereResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur5e_delto.IMPLICIT_UR5E_DELTO, Ur5eDeltoRelativeOSCAction())
        _apply_ur5e_delto_generation_plant(self)
        _apply_delto_collision_stack_size(self)
        _apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR)
        _assert_ur5e_delto_generation_plant(self)


@configclass
class Ur5eDeltoObjectPartiallyAssembledEEGraspedResetStatesCfg(ObjectPartiallyAssembledEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur5e_delto.IMPLICIT_UR5E_DELTO, Ur5eDeltoRelativeOSCAction())
        _apply_ur5e_delto_generation_plant(self)
        _apply_delto_collision_stack_size(self)
        _apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR)
        _assert_ur5e_delto_generation_plant(self)


# ---------------------------------------------------------------------------------------
# RL state (training / finetune / eval)
# ---------------------------------------------------------------------------------------
@configclass
class Ur5eDeltoRelCartesianOSCTrainCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    """Stage 1 training: implicit actuator.

    ``curriculum`` is ``ThresholdCurriculumCfg`` (rl_state_cfg.py), not the inherited base's
    ``NoCurriculumsCfg`` -- this is the fix for the 465-iteration / 0.0000-success run on
    ``OmniReset-UR5eDelto-RelCartesianOSC-State-v0``: the shipped success gate (0.0025 m /
    0.025 rad) is unreachable from a cold start, and this term ramps it in from a loose starting
    rung instead. The wired term is OFF BY DEFAULT (``enabled=False`` in ``ThresholdCurriculumCfg``),
    so this class, default-constructed, is byte-equivalent in behaviour to before this change --
    see ``ThresholdCurriculumCfg``'s docstring for the exact Hydra override string that turns it on
    and picks a starting rung.
    """

    curriculum: ThresholdCurriculumCfg = ThresholdCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur5e_delto.IMPLICIT_UR5E_DELTO, Ur5eDeltoRelativeOSCAction())
        _apply_ur5e_delto_generation_plant(self)
        _apply_delto_collision_stack_size(self)
        _apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR)
        _assert_ur5e_delto_generation_plant(self)


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
        _apply_ur5e_delto_generation_plant(self)
        _apply_delto_collision_stack_size(self)
        _apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR)
        _assert_ur5e_delto_generation_plant(self)


@configclass
class Ur5eDeltoRelCartesianOSCEvalCfg(Ur5eRobotiq2f85RelCartesianOSCEvalCfg):
    """Eval after Stage 1: implicit actuator, soft gains."""

    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur5e_delto.IMPLICIT_UR5E_DELTO, Ur5eDeltoRelativeOSCAction())
        _apply_ur5e_delto_generation_plant(self)
        _apply_delto_collision_stack_size(self)
        _apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR)
        _assert_ur5e_delto_generation_plant(self)


@configclass
class Ur5eDeltoRelCartesianOSCFinetuneEvalCfg(Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg):
    """Eval after Stage 2: explicit actuator, stiff eval gains.

    Uses ``Ur5eDeltoRelativeOSCEvalAction``, which carries the known kd*dt/I stability caveat (see
    ``actions.py``) -- inherited here unchanged, same as the UR10e+DELTO variant's equivalent.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_delto(self, ur5e_delto.EXPLICIT_UR5E_DELTO, Ur5eDeltoRelativeOSCEvalAction())
        _apply_ur5e_delto_generation_plant(self)
        _apply_delto_collision_stack_size(self)
        _apply_delto_dataset_dir(self, _UR5E_DELTO_RESET_DIR)
        _assert_ur5e_delto_generation_plant(self)
