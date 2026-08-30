# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e + Tesollo DELTO DG-5F on the paper's CUBE STACKING task, in two variants.

This is a thin layer over ``ur10e_delto_cfg.py``. Every class here subclasses the corresponding
UR10eDelto class, so the arm, the hand actuator, the OSC gains, ``_apply_delto`` and the collision
stack size are all inherited unchanged. Only two things are added:

1. a DELTO-specific ``dataset_dir`` (see DATASET DIR below), and
2. for the position-only variant, the orientation clause is removed from the success criterion and
   from the dense goal-distance shaping.

THE CUBE HAD TO BE RESIZED -- 34 mm, NOT THE PAPER'S 40 mm
------------------------------------------------------------
``InsertiveCube``/``ReceptiveCube`` were already registered in all four variant registries and
their USDs ship in the cloud asset repo. Measured directly off the USD meshes with pxr
(2026-08-31): both cubes are exactly **40.0 mm** on every axis, and neither authors a root
``MassAPI``.

40 mm is inside this hand's validated grasp window of 24.67 .. 41.48 mm
(``Props/Custom/DeltoBlock/metadata.yaml:6-7``) and under ``maximum_aperture`` 0.059 m, so it
looked usable. **It is not.** Measured here with identical grasp-sampler settings, only the object
differing:

    deltoblock  34 mm 0.030 kg    64 attempts ->  2 successes  (3.1%)
    the cube    40 mm 0.049 kg   320 attempts ->  0 successes  (0.0%)
    cube34      34 mm 0.030 kg  1024 attempts -> 51 successes  (5.0%)

At 40 mm the hand does not merely grip poorly, it never holds the object: ``UWLAB_GRASP_DEBUG``
read ``not_far=0`` and ``above_ground=0`` in every environment of every batch from the instant
gravity engaged at t=10 s, while every pre-gravity condition sat at 64/64. The window is a
geometric reachability bound, not a holding guarantee, and 40 mm sits 1.5 mm below its top.

So this task uses the paper's cube ASSET at ``scale=0.85`` -> 34.0 mm, mass 0.03 kg: geometrically
and inertially identical to ``deltoblock``, the object this hand's grip-force budget was actually
sized against. Variants ``cube34`` in all four registries, backed by local assets
``Props/Custom/{InsertiveCube34,ReceptiveCube34}``. They need their own directories, rather than a
spawn-scale override alone, because ``read_metadata_from_usd_directory`` keys metadata by USD
DIRECTORY and the assembled/bottom offsets must track the spawn scale (+-0.017 m, not +-0.02 m).

This is a stated, forced deviation. The paper's cube is sized for a Robotiq 2F-85 parallel jaw;
ours for a Tesollo DG-5F. The TASK -- stack one cube on another -- is unchanged.

For context, the DELTO grasp path is a known-weak link even on its own reference object:
``delto_cfg.py:600-632`` records its measured yield as "2/12, 1/12, 1/12 -- roughly 11 percent
per-episode" and calls that "well below the 60 percent acceptance bar".

The receptive cube is spawned ``kinematic_enabled=True`` by ``make_receptive_object``, i.e. the
bottom cube is immovable. This is "place a cube on a fixed cube", not a topple-able tower -- the
authors' own framing, inherited, not a simplification introduced here.

DATASET DIR -- the seam ``ur10e_delto_cfg.py`` is missing
----------------------------------------------------------
``compute_pair_dir`` (``mdp/utils.py:384-404``) keys every dataset by OBJECT PAIR ONLY -- there is
no arm or gripper discriminator in either the ``Resets/{pair}/`` or the ``Grasps/{obj}/`` layout.
``ur5e_delto_cfg.py`` opts out via ``_apply_delto_dataset_dir``; **``ur10e_delto_cfg.py`` never
calls it**, so every UR10eDelto task silently defaults to the shared cloud ``Datasets/OmniReset``.
For this pair that directory already contains a complete set of banks -- recorded with a UR5e +
Robotiq 2F-85. Loading a 12-joint Robotiq bank into this 26-DOF articulation dies at reset with a
shape mismatch. Hence the seam call in every class below.

ORIENTATION, AND WHAT "ORIENTATION-FREE" ACTUALLY MEANS HERE
-------------------------------------------------------------
``ProgressContext.__call__`` (``mdp/rewards.py``) is the single source of truth:

    e_x, e_y, _ = euler_xyz_from_quat(...)
    euler_xy_distance = |wrap_to_pi(e_x)| + |wrap_to_pi(e_y)|     # YAW IS DROPPED, deliberately
    orientation_aligned = euler_xy_distance < success_orientation_threshold
    position_aligned    = xyz_distance      < success_position_threshold
    success = orientation_aligned & position_aligned

Yaw is already discarded upstream, so the ORIENTED variant constrains roll and pitch only -- i.e.
"a chosen face must end up facing up, but the cube may be spun freely about the vertical". That is
exactly the paper's "with a desired orientation" (§4.1) as this codebase implements it.

The paper's reward (Eq. 5) is
    r_dist = lambda_dist * 1/2 * [ exp(-|x_err|/s) + exp(-|th_err|/s) ]
and the orientation-free variant is the clean deletion of the second summand together with the
``1/2``. ``dense_success_reward(use_orientation=False)`` implements precisely that. Making
``std_angle`` huge instead is NOT equivalent: it leaves a constant ``+0.5`` and halves the position
gradient.

Why a finite 7.0 rad rather than ``math.inf`` for the disabled threshold: ``euler_xy_distance`` is
a sum of two ``wrap_to_pi`` magnitudes, so it is bounded by ``2*pi ~= 6.2832``. 7.0 is therefore
unconditionally satisfied, exactly like infinity, while remaining a finite number that logging,
metric aggregation and JSON serialisation handle without special cases.

Registered gym ids
------------------
* ``OmniReset-CubeStack-PartialAssemblies-v0``                 near-goal offset bank
* ``OmniReset-UR10eDelto-CubeStack-Object{...}-v0``            x5, reset-state recording
* ``OmniReset-UR10eDelto-CubeStack-RelCartesianOSC-State-v0``  oriented, + ``-Play``
* ``OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-v0``  position-only, + ``-Play``

All of them must be run with ``env.scene.insertive_object=cube34 env.scene.receptive_object=cube34``
(``env.scene.object=cube34`` for grasp sampling). The variant is not pinned in these configs,
matching how the published recipe drives every other object pair.

The reset-state banks are SHARED between the two variants: bank recording grades with
``check_reset_state_success``, which reads ``success_thresholds`` from the receptive object's
metadata directly and never consults the command term. Record once, train twice.
"""

from __future__ import annotations

from isaaclab.utils import configclass

import uwlab_assets.robots.ur10e_delto as ur10e_delto

from .actions import Ur10eDeltoRelativeOSCAction, Ur10eDeltoRelativeOSCEvalAction
from .delto_cfg import _apply_delto, _apply_delto_collision_stack_size, _apply_delto_dataset_dir
from .partial_assemblies_cfg import PartialAssembliesCfg
from .ur10e_delto_cfg import (
    Ur10eDeltoObjectAnywhereEEAnywhereResetStatesCfg,
    Ur10eDeltoObjectAnywhereEEGraspedResetStatesCfg,
    Ur10eDeltoObjectPartiallyAssembledEEAnywhereResetStatesCfg,
    Ur10eDeltoObjectPartiallyAssembledEEGraspedResetStatesCfg,
    Ur10eDeltoObjectRestingEEGraspedResetStatesCfg,
    Ur10eDeltoRelCartesianOSCEvalCfg,
    Ur10eDeltoRelCartesianOSCTrainCfg,
)

# Every DELTO cube artifact lives here: Grasps/InsertiveCube34/grasps.pt and
# Resets/InsertiveCube34__ReceptiveCube34/resets_*.pt. Kept out of the shared cloud tree so a DELTO
# bank can never be confused with the shipped 2F-85 one for the same pair.
CUBE_DATASET_DIR = "./Datasets_ur10e_delto/OmniReset"

# Loosened from the shipped 0.025 rad. `new_task.rst:92` names 0.025 as the TIGHT-FIT value (screw
# insertion) and explicitly recommends `position: 0.005, orientation: 0.05` for cube stacking. The
# shipped ReceptiveCube metadata carries the tight number; this override is the documented one.
ORIENTED_ORIENTATION_THRESHOLD = 0.05

# euler_xy_distance = |wrap_to_pi(e_x)| + |wrap_to_pi(e_y)| <= 2*pi ~= 6.2832, so 7.0 is
# unconditionally satisfied. See the module docstring for why this is not `math.inf`.
ORIENTATION_FREE_THRESHOLD = 7.0


def _apply_cube_dataset_dir(cfg) -> None:
    """Point every ``dataset_dir``-bearing event term at the DELTO-specific cube tree."""
    _apply_delto_dataset_dir(cfg, CUBE_DATASET_DIR)


def _apply_oriented(cfg) -> None:
    """Paper-faithful variant: keep the orientation clause, at the docs' cube-stacking tolerance."""
    cfg.commands.task_command.success_orientation_threshold = ORIENTED_ORIENTATION_THRESHOLD


def _apply_orientation_free(cfg) -> None:
    """Position-only variant: drop orientation from BOTH the success test and the dense shaping.

    Two edits, and they must go together. Relaxing only the threshold would leave
    ``dense_success_reward`` still paying for roll/pitch alignment, so the policy would keep
    optimising an objective the task no longer scores.
    """
    cfg.commands.task_command.success_orientation_threshold = ORIENTATION_FREE_THRESHOLD
    cfg.rewards.dense_success_reward.params["use_orientation"] = False


# ---------------------------------------------------------------------------------------
# Reset states -- shared by both variants (recording grades from metadata, not the command term)
# ---------------------------------------------------------------------------------------
@configclass
class CubeStackObjectAnywhereEEAnywhereCfg(Ur10eDeltoObjectAnywhereEEAnywhereResetStatesCfg):
    """C1 "Reaching". No prerequisite bank. Must be recorded BEFORE C2, which samples from it."""

    def __post_init__(self):
        super().__post_init__()
        _apply_cube_dataset_dir(self)


@configclass
class CubeStackObjectRestingEEGraspedCfg(Ur10eDeltoObjectRestingEEGraspedResetStatesCfg):
    """C2 "Near-Object". Consumes the C1 bank AND the grasp bank."""

    def __post_init__(self):
        super().__post_init__()
        _apply_cube_dataset_dir(self)


@configclass
class CubeStackObjectAnywhereEEGraspedCfg(Ur10eDeltoObjectAnywhereEEGraspedResetStatesCfg):
    """C3 "Stable Grasp". Consumes the grasp bank only; object pose sampled uniformly in the air."""

    def __post_init__(self):
        super().__post_init__()
        _apply_cube_dataset_dir(self)


@configclass
class CubeStackObjectPartiallyAssembledEEAnywhereCfg(Ur10eDeltoObjectPartiallyAssembledEEAnywhereResetStatesCfg):
    """Registered and recordable, but NEVER sampled at train time (``rl_state_cfg`` uses four)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_cube_dataset_dir(self)


@configclass
class CubeStackObjectPartiallyAssembledEEGraspedCfg(Ur10eDeltoObjectPartiallyAssembledEEGraspedResetStatesCfg):
    """C4 "Near-Goal". Consumes partial assemblies AND the grasp bank. The slowest to record, and
    the one the guide warns about: under-recording C4 is what produces "holds the object near the
    goal, stuck"."""

    def __post_init__(self):
        super().__post_init__()
        _apply_cube_dataset_dir(self)


# ---------------------------------------------------------------------------------------
# RL state -- ORIENTED (paper-faithful: roll/pitch constrained, yaw free)
# ---------------------------------------------------------------------------------------
@configclass
class CubeStackTrainCfg(Ur10eDeltoRelCartesianOSCTrainCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_cube_dataset_dir(self)
        _apply_oriented(self)


@configclass
class CubeStackEvalCfg(Ur10eDeltoRelCartesianOSCEvalCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_cube_dataset_dir(self)
        _apply_oriented(self)


# ---------------------------------------------------------------------------------------
# RL state -- POSITION-ONLY
# ---------------------------------------------------------------------------------------
@configclass
class CubeStackNoOrientTrainCfg(Ur10eDeltoRelCartesianOSCTrainCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_cube_dataset_dir(self)
        _apply_orientation_free(self)


@configclass
class CubeStackNoOrientEvalCfg(Ur10eDeltoRelCartesianOSCEvalCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_cube_dataset_dir(self)
        _apply_orientation_free(self)


# ---------------------------------------------------------------------------------------
# Partial assemblies -- the "near-goal offsets" bank (paper §3.2 pre-computation step B)
# ---------------------------------------------------------------------------------------
@configclass
class CubeStackPartialAssembliesCfg(PartialAssembliesCfg):
    """Partial assemblies for a STACK, with the thread-insertion sampler removed.

    The paper's mechanism (§3.2) is simply: spawn the target at the goal, then apply small random
    forces to dislodge it, yielding a continuum of near-goal configurations. That is what the base
    ``PartialAssembliesCfg`` does.

    ``axial_depth_sampling`` is a LEG-SPECIFIC ADDITION layered on top of it by this branch, not
    part of the paper's method. It is unconditionally parameterised with the table-leg thread
    constants (``LEG200MM_ONELEGFIXTURE_*``, ``partial_assemblies_cfg.py:55-196``, including a
    21-row thread-yaw table) and models a peg descending into a bore: mouth plane, bore radius,
    engaged length, thread yaw coupling, radial clearance.

    A cube stacked on a cube has no mouth, no bore and no thread. The term's two raise-loudly
    sanity checks (``events.py:1835-1848``, ``:1865-1884``) happen to PASS numerically for this
    pair -- the mating quat is identity so the insertion axis really is local Z, and
    ``mouth_local_z_m`` 0.015625 differs from cube34's ``seat_local_z`` 0.017 so the zero-depth
    guard does not trip -- which is precisely the danger: it would run and silently produce
    offsets drawn from a bore geometry that does not exist. Setting it to ``None`` drops the term.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.axial_depth_sampling = None
