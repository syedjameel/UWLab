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

Every class below PINS the cube34 pair via ``_apply_cube_objects``, so
``env.scene.insertive_object=cube34 env.scene.receptive_object=cube34`` is no longer required on the
command line -- it is merely redundant, and still overrides if passed. Grasp sampling takes
``env.scene.object=cube34`` separately.

Pinning is not cosmetic. ``RlStateSceneCfg`` defaults these two fields to
``Props/Custom/Peg/peg.usd`` and ``Props/Custom/PegHole/peg_hole.usd``, so a run that omitted the
flags was peg-in-hole -- different geometry, different ``assembled_offset``, and a success threshold
read from PegHole's metadata -- while every log line, checkpoint directory and TensorBoard run still
said CubeStack. The ``assembled_offset`` assertion in ``commands.py`` covers the table-leg/fixture
pair only and would not have caught it.

The reset-state banks are SHARED between the two variants: bank recording grades with
``check_reset_state_success``, which reads ``success_thresholds`` from the receptive object's
metadata directly and never consults the command term. Record once, train twice.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import uwlab_assets.robots.ur10e_delto as ur10e_delto

from ... import mdp as task_mdp
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

# Shaping length scale for the added short-range goal-distance term. See
# ``_apply_precision_shaping`` for the measurement that fixes it at 0.02 m.
PRECISION_SHAPING_STD = 0.02
PRECISION_SHAPING_WEIGHT = 0.1


def _apply_precision_shaping(cfg, std: float = PRECISION_SHAPING_STD) -> None:
    """Add a SHORT-RANGE goal-distance term, because the shipped one cannot see the last centimetre.

    THE MEASUREMENT THAT MOTIVATES THIS. Two independent seeds of the near-goal-only task (42 and 7)
    both drove end-of-episode success from ~1.5% to ~20% within 60 iterations and then held flat for
    240 more, with median end-of-episode position error stuck near 16 mm against a 5 mm tolerance.
    Neither actuation nor observability explains that floor: RelCartesianOSC's ``scale_xyz`` is
    0.02 m and the controller realises ~4.7 mm of end-effector motion per unit action per 0.1 s
    step with no clip or quantisation anywhere (``input_clip`` and ``clip_actions`` are both None),
    while ``insertive_asset_in_receptive_asset_frame`` carries the exact relative pose and no
    ObsTerm in the package declares a ``noise=`` term.

    The reward does explain it. ``dense_success_reward`` is ``exp(-d/std)`` with ``std = 1.0`` m --
    a shaping length scale 200x the 0.005 m tolerance -- so it spends 98% of its dynamic range
    getting the cube from 1 m to 2 cm and 1% on the band where the task is decided:

        closing 16 mm -> 5 mm      0.1 * (exp(-0.005) - exp(-0.016))  = +1.09e-4 per step
        one step of action_rate    -1e-3 * 28 (measured)              = -2.80e-3 per step

    The penalty paid every step is 26x the entire reward for closing the last 11 mm, and after GAE
    (gamma 0.99, lam 0.95, effective horizon 16.8 steps) the advantage of permanently closing it is
    1.8e-3 against 1.68 for crossing the binary gate. The policy stopped where the reward stopped
    paying.

    WHY ADD A TERM RATHER THAN RETUNE THE EXISTING ONE. Simply setting ``std = 0.02`` would fix the
    endgame and destroy the approach: at the 94 mm mean distance these episodes actually spend most
    of their time at, ``exp(-0.094/0.02)`` is 0.009, i.e. no transport gradient at all. That is
    survivable for the near-goal family, which starts ~13 mm out, but not for the full four-family
    mixture where the cube starts anywhere on the table. Keeping the 1.0 m term for transport and
    adding a 0.02 m term for the endgame gives both, and leaves the shipped term untouched for every
    other task in the package.

    At ``std = 0.02`` the same 16 mm -> 5 mm step is worth ``0.1 * (exp(-0.25) - exp(-0.80))`` =
    +3.3e-2 per step -- 30x the action-rate penalty rather than 1/26th of it.
    """
    params = {"std": std, "std_angle": std}
    use_orientation = cfg.rewards.dense_success_reward.params.get("use_orientation")
    if use_orientation is not None:
        params["use_orientation"] = use_orientation
    cfg.rewards.dense_success_reward_fine = RewTerm(
        func=task_mdp.dense_success_reward, weight=PRECISION_SHAPING_WEIGHT, params=params
    )


def _apply_cube_objects(cfg) -> None:
    """Pin the scene to the cube34 pair, so the task cannot silently become peg-in-hole.

    ``RlStateSceneCfg`` defaults ``insertive_object`` to ``Props/Custom/Peg/peg.usd`` and
    ``receptive_object`` to ``Props/Custom/PegHole/peg_hole.usd``. Until now every CubeStack class
    inherited those defaults and relied on the caller passing
    ``env.scene.insertive_object=cube34 env.scene.receptive_object=cube34`` on the command line.
    Forget the flags and the run is peg-in-hole: different geometry, different
    ``assembled_offset``, and different ``success_thresholds`` read from PegHole's metadata -- while
    every log line, checkpoint directory and TensorBoard run still says CubeStack. There is an
    assertion guarding assembled offsets (``commands.py``) but it covers the table-leg/fixture pair
    only, so nothing would have caught it.

    Pinning here makes the flags redundant rather than forbidden: ``variants`` still resolves, so
    ``env.scene.insertive_object=<other>`` on the command line continues to override this.
    """
    v = cfg.variants
    cfg.scene.insertive_object = v["scene.insertive_object"]["cube34"]
    cfg.scene.receptive_object = v["scene.receptive_object"]["cube34"]


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
        _apply_cube_objects(self)
        _apply_cube_dataset_dir(self)
        _apply_grasp_matched_hand_actuator(self)
        _apply_hand_matched_mass_randomization(self)


@configclass
class CubeStackObjectRestingEEGraspedCfg(Ur10eDeltoObjectRestingEEGraspedResetStatesCfg):
    """C2 "Near-Object". Consumes the C1 bank AND the grasp bank."""

    def __post_init__(self):
        super().__post_init__()
        _apply_cube_objects(self)
        _apply_cube_dataset_dir(self)
        _apply_grasp_matched_hand_actuator(self)
        _apply_hand_matched_mass_randomization(self)


@configclass
class CubeStackObjectAnywhereEEGraspedCfg(Ur10eDeltoObjectAnywhereEEGraspedResetStatesCfg):
    """C3 "Stable Grasp". Consumes the grasp bank only; object pose sampled uniformly in the air."""

    def __post_init__(self):
        super().__post_init__()
        _apply_cube_objects(self)
        _apply_cube_dataset_dir(self)
        _apply_grasp_matched_hand_actuator(self)
        _apply_hand_matched_mass_randomization(self)


@configclass
class CubeStackObjectPartiallyAssembledEEAnywhereCfg(Ur10eDeltoObjectPartiallyAssembledEEAnywhereResetStatesCfg):
    """Registered and recordable, but NEVER sampled at train time (``rl_state_cfg`` uses four)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_cube_objects(self)
        _apply_cube_dataset_dir(self)
        _apply_grasp_matched_hand_actuator(self)
        _apply_hand_matched_mass_randomization(self)


@configclass
class CubeStackObjectPartiallyAssembledEEGraspedCfg(Ur10eDeltoObjectPartiallyAssembledEEGraspedResetStatesCfg):
    """C4 "Near-Goal". Consumes partial assemblies AND the grasp bank. The slowest to record, and
    the one the guide warns about: under-recording C4 is what produces "holds the object near the
    goal, stuck"."""

    def __post_init__(self):
        super().__post_init__()
        _apply_cube_objects(self)
        _apply_cube_dataset_dir(self)
        _apply_grasp_matched_hand_actuator(self)
        _apply_hand_matched_mass_randomization(self)


# ---------------------------------------------------------------------------------------
# RL state -- ORIENTED (paper-faithful: roll/pitch constrained, yaw free)
# ---------------------------------------------------------------------------------------
@configclass
class CubeStackTrainCfg(Ur10eDeltoRelCartesianOSCTrainCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_cube_objects(self)
        _apply_cube_dataset_dir(self)
        _apply_grasp_matched_hand_actuator(self)
        _apply_hand_matched_mass_randomization(self)
        _apply_oriented(self)
        _apply_precision_shaping(self)


@configclass
class CubeStackEvalCfg(Ur10eDeltoRelCartesianOSCEvalCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_cube_objects(self)
        _apply_cube_dataset_dir(self)
        _apply_grasp_matched_hand_actuator(self)
        _apply_hand_matched_mass_randomization(self)
        _apply_oriented(self)
        _apply_precision_shaping(self)


# ---------------------------------------------------------------------------------------
# RL state -- POSITION-ONLY
# ---------------------------------------------------------------------------------------
@configclass
class CubeStackNoOrientTrainCfg(Ur10eDeltoRelCartesianOSCTrainCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_cube_objects(self)
        _apply_cube_dataset_dir(self)
        _apply_grasp_matched_hand_actuator(self)
        _apply_hand_matched_mass_randomization(self)
        _apply_orientation_free(self)
        _apply_precision_shaping(self)


@configclass
class CubeStackNoOrientEvalCfg(Ur10eDeltoRelCartesianOSCEvalCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_cube_objects(self)
        _apply_cube_dataset_dir(self)
        _apply_grasp_matched_hand_actuator(self)
        _apply_hand_matched_mass_randomization(self)
        _apply_orientation_free(self)
        _apply_precision_shaping(self)


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


# ---------------------------------------------------------------------------------------
# The thumb-side stiffness seam
# ---------------------------------------------------------------------------------------
# THE INCONSISTENCY THIS ADDRESSES. `DeltoGraspSamplingCfg` rebinds two joints before it samples
# (delto_cfg.py:623-632): `rj_dg_1_4` 0.1698 -> 1.698 and `rj_dg_5_4` 0.1694 -> 1.694, with
# `velocity_limit_sim` 3.0 -> 1.0. Those are the distal joints of the THUMB-SIDE pair the held-check
# opposes against fingers 2/3/4, and delto_cfg records exactly why the stock values do not work:
#
#   "the relative action caps PD error at 0.1 rad per control step regardless of distance to target,
#    so realised torque is bounded by stiffness * 0.1. At 0.1698 that is 0.0170 N.m -- matching the
#    measured applied torque to four decimals -- against a 0.01 N.m friction floor ... The joint sat
#    frozen at ~1.05 rad, velocity exactly 0.000, in EVERY velocity/deadband/stiffness combination
#    swept."
#
# and what the 10x rebind bought:
#
#   "the thumb gap collapsing 1.267 -> 0.006 rad and holding through gravity AND shake, 7 bodies in
#    contact, no limit cycle."
#
# That rebind is scoped to the sampler alone -- deliberately, per its own comment. The consequence
# is that a grasp is SAMPLED AND SHAKE-VALIDATED by a hand that can close its thumb, and then
# REPLAYED, during reset-state recording and during RL, by a hand that cannot. The grasp bank
# describes postures the training articulation is not equipped to reach.
#
# NOT MONOTONIC, so do not raise it casually: delto_cfg records that 20x is WORSE than 10x
# (0/180 successes, fewer bodies in contact, lower peak force), and that 0.30 -- the value its
# already-fixed siblings use -- leaves rj_dg_1_4 0.364 rad short of target with 0/180 successes.
# 10x is the measured value, and it is reused verbatim here rather than re-derived.
_THUMB_SIDE_STIFFNESS = {"rj_dg_1_4": 1.698, "rj_dg_5_4": 1.694}


def _apply_grasp_matched_hand_actuator(cfg) -> None:
    """Give the training/reset hand the actuator its grasp bank was validated against.

    Rebinds through the actuators dict rather than mutating the shared ``DELTO_HAND_ACTUATOR``
    instance, exactly as the sampler does, so `IMPLICIT_UR10E_DELTO` and `EXPLICIT_UR10E_DELTO`
    stay untouched for every other task on this robot.
    """
    hand = cfg.scene.robot.actuators["hand"]
    stiffness = dict(hand.stiffness)
    stiffness.update(_THUMB_SIDE_STIFFNESS)
    cfg.scene.robot.actuators = {
        **cfg.scene.robot.actuators,
        # velocity_limit_sim 1.0 travels exactly the 0.1 rad a control step commands, instead of
        # the 0.3 rad that 3.0 allows -- the 3x overshoot delto_cfg names as a standing cause of a
        # permanent limit cycle. Matched to the sampler for the same reason the stiffness is.
        "hand": hand.replace(stiffness=stiffness, velocity_limit_sim={r"rj_dg_[1-5]_[1-4]": 1.0}),
    }


@configclass
class CubeStackObjectAnywhereEEGraspedThumbFixCfg(CubeStackObjectAnywhereEEGraspedCfg):
    """Retained as the A/B probe's identity; now identical to its base.

    This class existed to measure the grasp-matched actuator against the stock one before adopting
    it. That A/B ran: 512 envs, 420 s per arm, same object and grasp bank, only the two thumb-side
    stiffnesses differing -- **stock 23 states, grasp-matched 31**. At those counts Poisson noise is
    +-4.8 and +-5.6, so the difference is 8 +- 7.4, about 1.1 sigma: NOT significant, and not
    evidence that the stiffness explains the 5% -> ~1% sampling-to-reset gap.

    The actuator was adopted regardless, on the consistency argument rather than the speed one (see
    ``_apply_grasp_matched_hand_actuator``), so every CubeStack class now applies it and this
    subclass adds nothing. Kept registered so the A/B's task id still resolves and the experiment
    stays reproducible.
    """


# ---------------------------------------------------------------------------------------
# Mass randomization, narrowed to what this hand can actually hold
# ---------------------------------------------------------------------------------------
# The shared `randomize_insertive_object_mass` draws mass_abs ~ U(0.02, 0.2) kg at STARTUP, so each
# environment keeps one fixed mass for the whole run. Its own comment states the assumption:
# "we assume insertive object is somewhere between 20g and 200g" -- reasonable for the Robotiq 2F-85
# the task family was built around.
#
# It is not reasonable for the DG-5F. Measured in this repo, the hand holds its VALIDATED 34 mm
# 0.030 kg reference object about 5% of the time, and fails to hold a 40 mm 0.049 kg cube at all
# (0/320). Under U(0.02, 0.2), only (0.05-0.02)/(0.2-0.02) = 17% of environments would draw a mass
# at or below 50 g; the other ~83% would spend the entire run holding a cube heavier than anything
# this hand has been shown to hold. The policy would be optimising against mostly unsolvable
# episodes, and the failure would look like "RL did not converge" rather than "the task was
# impossible in five environments out of six".
#
# NARROWED TO U(0.02, 0.06): centred on the validated 0.03 kg, spanning 2/3x to 2x of it, so domain
# randomization still does its job (the policy cannot assume one exact mass) without spending most
# of its samples on a physically unavailable grasp.
#
# WHAT THIS IS NOT: it is not a claim that 0.06 kg is the hand's limit. The fingertip force cap is
# ~2.2 N (effort_limit_sim 0.06 N.m over a 25.5 mm distal lever), which in a frictionless-free-body
# sense would suggest a far higher bound -- but the measured hold rate at 0.030 kg is already only
# 5%, so the analytic bound plainly is not what governs here. The range is set from the measurement,
# not from the force budget, and the honest statement is that the upper end is untested.
CUBE_MASS_RANGE_KG = (0.02, 0.06)


def _apply_hand_matched_mass_randomization(cfg) -> None:
    """Narrow insertive-object mass DR to the range this hand is measured to handle."""
    term = getattr(cfg.events, "randomize_insertive_object_mass", None)
    if term is not None:
        term.params["mass_distribution_params"] = CUBE_MASS_RANGE_KG


# ---------------------------------------------------------------------------------------
# DIAGNOSTIC: near-goal-only training, to separate "not enough compute" from "something is broken"
# ---------------------------------------------------------------------------------------
# After 402 iterations on the paper's four-family mixture, the policy scored 9.05% on near-goal
# against an untrained 10.13% (n~820 each, p=0.454) and 0.0% on the other three families, while its
# hand action magnitude rose 11.5x. It moved a long way and learned nothing measurable.
#
# Two explanations fit that equally well from the outside:
#   (a) the compute budget is ~2 orders below the paper's and the task is simply unlearnable here;
#   (b) something in this port prevents learning at all, and (a) is masking it.
#
# They are distinguishable. Train on the EASIEST family alone -- near-goal, where the cube starts
# about 13 mm from a 5 mm tolerance and needs only a small local correction -- and give it 4x the
# near-goal experience per iteration by removing the other three families from the mixture. If the
# policy learns THAT, the pipeline is sound and (a) holds. If it cannot learn even that, (b) is live
# and the negative result means something quite different.
#
# This is deliberately NOT the paper's method: OmniReset's whole claim rests on the flat four-family
# mixture producing an emergent backwards curriculum. Removing three families removes the mechanism
# under test. It is a diagnostic on the implementation, and it is labelled as one everywhere it is
# reported.
#
# entropy_coef is also lowered, on evidence rather than taste: gSDE's noise std climbed monotonically
# 0.75 -> 2.65 across the run while the task reward terms stayed flat, which is the entropy bonus
# dominating a weak advantage signal. At std 2.65 against an action clip of +-1 the sampled actions
# are largely saturated. That is set on the runner cfg at launch, not here.
@configclass
class CubeStackNearGoalOnlyTrainCfg(CubeStackNoOrientTrainCfg):
    """Position-only cube stacking, trained from the near-goal family alone. Diagnostic, not a port."""

    def __post_init__(self):
        super().__post_init__()
        rst = getattr(self.events, "reset_from_reset_states", None)
        if rst is not None:
            rst.params["reset_types"] = ["ObjectPartiallyAssembledEEGrasped"]
            rst.params["probs"] = [1.0]
