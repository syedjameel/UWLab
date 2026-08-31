# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e + Tesollo DELTO DG-5F on the paper's CUBE STACKING task, in two variants.

This is a thin layer over ``ur10e_delto_cfg.py``. Every class here subclasses the corresponding
UR10eDelto class, so the arm, the hand actuator, the OSC gains, ``_apply_delto`` and the collision
stack size are all inherited unchanged. Only three things are added:

1. a DELTO-specific ``dataset_dir`` (see DATASET DIR below),
2. for the position-only variant, the orientation clause is removed from the success criterion and
   from the dense goal-distance shaping, and
3. for the RL-state variants only, fingertip contact sensing and the grasp/lift reward terms that
   read it -- see THE GRASP SEAM below and ``_apply_fingertip_contact_sensors`` /
   ``_apply_grasp_shaping``. The shared ``RewardsCfg`` and the shared robot cfg are NOT touched;
   these are ``__post_init__`` overlays exactly like ``_apply_precision_shaping``, so every other
   task in the package is byte-for-byte unchanged.

THE GRASP SEAM -- the reward graph had no arc across the grasp
----------------------------------------------------------------
The shipped OmniReset reward is a pure function of two rigid-body poses (see the block comment
above ``_apply_fingertip_contact_sensors`` for the full audit and the numbers). Closing this hand
was worth 0.0 and cost a small action penalty, so the reward was maximised by parking the OPEN hand
on the cube and holding still -- 9.26% success on the one family that starts already holding the
cube, 0.0% on the three that do not. The fix ports dexlift's already-validated contact terms for
this exact hand rather than inventing new physics.

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
* ``OmniReset-UR10eDelto-CubeStackTwoFinger-RelCartesianOSC-State-v0``  position-only, hand acts
  in 8 dims instead of 20 (total 14, not 26), + ``-Play``

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
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.ur10e_delto as ur10e_delto
from uwlab.envs.mdp.full_actuation import assert_action_cfg_fully_actuates

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


# ---------------------------------------------------------------------------------------
# The grasp seam -- fingertip contact sensing, and the three reward terms that read it
# ---------------------------------------------------------------------------------------
# WHAT IS BROKEN. The shipped ``RewardsCfg`` (rl_state_cfg.py:561-603) is a pure function of two
# rigid-body poses. ``ee_asset_distance`` reads the palm body ``rl_dg_mount`` and ``dense_success_
# reward``/``dense_success_reward_fine``/``success_reward`` read ``ProgressContext``'s cube-to-goal
# error; ``progress_context`` itself returns zeros. Not one term reads a contact force, a finger
# joint angle, an aperture or an object height. On this hand that means flexing all twenty fingers
# moves the total reward by EXACTLY ZERO -- ``ee_asset_distance`` depends only on the pose of
# ``rl_dg_mount``, which the fingers do not move, and ``dense_success_reward`` only on the cube,
# which an ungrasped cube does not move either -- minus the action penalty of having moved them.
# The reward is maximised by parking the OPEN hand on the cube and holding still.
#
# MEASURED CONSEQUENCE. 9.26% genuine success on the NEAR-GOAL family, the one family where the
# cube starts already held and the geometric graph is therefore complete, and exactly 0.0% on the
# other three, every one of which requires closing the hand first. That is not a compute shortfall;
# it is a reward graph with no arc across the grasp.
#
# BODY AND PRIM NAMES, read from ``dexlift/dexlift_ur10e_delto_env_cfg.py:59`` and ``:68-72`` and restated here
# rather than imported. Importing that module runs the vendored dexsuite package and ``omnireset.mdp``
# -- the cycle ``omnireset/mdp/contact_rewards.py`` refuses to close -- for five string constants.
# They describe the same five fingertips of the same articulation: both tasks spawn
# ``IMPLICIT_UR10E_DELTO``. The prim layout was confirmed against the live USD stage with pxr, not
# assumed: ``scripts_v2/tools/smoke_test_ik_c4_holding.py:59`` records
# "{ENV_REGEX_NS}/Robot/gripper/rl_dg_mount and .../gripper/rl_dg_{1..5}_tip all exist exactly as"
# expected, and ``render_reset_states_viewport.py:206-234`` wires this same sensor set against this
# same OmniReset task.
DELTO_HAND_PRIM = "gripper"
DELTO_THUMB_TIP_NAMES = ("rl_dg_1_tip", "rl_dg_5_tip")
"""The DG-5F has TWO opposable digits, so the opposition gate takes a sequence of thumbs."""
DELTO_TIP_NAMES = ("rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip")
DELTO_ALL_TIP_NAMES = DELTO_THUMB_TIP_NAMES + DELTO_TIP_NAMES

# Force above which a fingertip counts as loading the cube. 0.2 N is dexlift's value for this exact
# hand (its ``good_finger_contact``/``any_finger_contact``/``object_upward_motion`` all use it,
# dexlift_ur10e_delto_env_cfg.py:196-232), and it is ~9% of the fingertip force ceiling this hand
# can actually produce -- ``effort_limit_sim`` 0.06 N.m over the 25.5 mm distal lever is ~2.2 N
# (delto_cfg.py:65 records the same backstop) -- i.e. comfortably above sensor noise and far below a
# real pinch. Not re-derived here.
GRASP_CONTACT_THRESHOLD_N = 0.2

ANY_CONTACT_WEIGHT = 0.01
GRASP_CONTACT_WEIGHT = 0.05
LIFT_WEIGHT = 0.05
# Velocity scale of the lift kernel. NOT dexlift's 0.2: see ``_apply_grasp_shaping``.
LIFT_STD_MPS = 0.05
FINGERTIP_DISTANCE_WEIGHT = 0.05
# 0.06 m, NOT the 0.03 this started at -- corrected from the measured working range.
#
# `1 - tanh(d/std)` is flat beyond about 3*std, so at std 0.03 the term carried no gradient past
# ~90 mm. Decoding the logged reward back to a distance on the four-family run showed exactly that:
# mean fingertip-to-cube distance fell 106 mm -> 82 mm over the first nine iterations and then
# stopped and drifted back to 85 mm, which is precisely where the term goes flat. The palm term
# read 228 mm over the same window, so the hand genuinely spends its time out at that range and
# the pull has to reach it.
#
#     std 0.03, 84 -> 83 mm    0.05 * sech^2(2.80)/0.03 * 0.001  = +2.5e-5 per step per mm
#     std 0.06, 84 -> 83 mm    0.05 * sech^2(1.40)/0.06 * 0.001  = +1.8e-4 per step per mm
#
# and across the span that matters, 84 mm -> 20 mm, std 0.06 pays 0.028 per step against the
# -2.8e-3 action_rate penalty, while still holding a usable 7.8e-3 over the last 20 -> 10 mm. This
# is the same failure the goal-distance term had at std 1.0 against a 5 mm tolerance, mirrored: a
# length scale chosen without reference to the distances the policy actually occupies.
FINGERTIP_DISTANCE_STD_M = 0.06


def _apply_fingertip_contact_sensors(cfg) -> None:
    """Give the cube-stacking scene the five per-fingertip contact sensors its rewards will read.

    MUST be called AFTER ``_apply_cube_objects``, which is what binds ``scene.insertive_object`` to
    the cube34 entry; the reporter API has to be flipped on whatever object the scene ends up with,
    not on the peg default it starts with.

    RL-STATE CLASSES ONLY, not the five reset-state recording classes. Bank recording grades with
    ``check_reset_state_success``, which is geometric and never reads a force, so those runs would
    pay the contact-reporting cost for a signal nothing consumes -- and they are already the slowest
    step of the pipeline.

    ``.replace()``, NOT in-place mutation, on the robot spawn -- this one is load-bearing.
    ``_apply_delto`` binds the robot as ``IMPLICIT_UR10E_DELTO.replace(prim_path=...)``, and
    ``configclass``'s ``replace`` is ``dataclasses.replace``, i.e. a SHALLOW reconstruction: the new
    ``ArticulationCfg``'s ``spawn`` IS the module-level ``IMPLICIT_UR10E_DELTO.spawn`` object at the
    moment this runs (the per-instance deepcopy ``_custom_post_init`` performs happens only after
    the whole ``__post_init__`` chain returns). Writing ``activate_contact_sensors`` through that
    handle would turn contact reporting on for every UR10e+DELTO cfg constructed later in the same
    process -- dexlift states the same hazard in its own words at
    ``dexlift_ur10e_delto_env_cfg.py:313-316``. Rebinding instead is exactly what
    ``_apply_grasp_matched_hand_actuator`` does with ``.actuators`` rather than mutating the shared
    ``DELTO_HAND_ACTUATOR``.

    The insertive object is written the same way for a weaker reason: ``cfg.variants`` is already a
    per-instance deepcopy (``configclass``'s ``_return_f`` default factory), so in place would be
    safe TODAY. It is spelled identically so the two lines cannot drift, and so that a future
    ``_apply_cube_objects`` that sourced the object from the module-level ``variants`` dict directly
    -- which is how the sibling ``render_reset_states_viewport.py`` reaches it -- could not silently
    reintroduce the leak.

    THE REPORTER API IS NOT OPTIONAL AND NOT SYMMETRIC-BY-DEFAULT. ``ur10e_delto.py:167`` ships the
    robot with ``activate_contact_sensors=False`` and ``make_insertive_object`` never sets it, so
    without both writes below every sensor registered here raises "could not find any bodies with
    contact reporter API" at ``gym.make``. That failure mode was confirmed empirically against this
    task family (``render_reset_states_viewport.py:226-234``).

    SENSOR NAMING IS LOAD-BEARING. ``contact_rewards._sensor_force_magnitudes`` looks each finger up
    as ``env.scene.sensors[f"{name}_object_s"]``. The suffix is the contract, not decoration.

    COST, AND WHY IT IS EXPECTED TO FIT. Five filtered contact views per environment at the 2048
    envs these runs use. This does NOT add collision pairs -- the fingertip/cube pairs are already
    generated and already consume stack whether or not anyone reads them; what is new is that PhysX
    also writes the filtered results out, which lands in ``gpu_max_rigid_patch_count`` /
    ``gpu_max_rigid_contact_count`` (both 2**23, rl_state_cfg.py:891-892), not in the collision
    stack. And the collision stack ceiling was already sized against a profile that INCLUDES these:
    ``_apply_delto_collision_stack_size``'s own docstring describes the plant it pinned 3.75 GiB for
    as "23 collider bodies, self-collisions ON, 5 fingertip sensors". Independently, dexlift's
    UR10e+DELTO task runs this identical five-view set on this identical hand today.
    ``gpu_collision_stack_size`` is deliberately NOT raised here: 4026531840 is already the maximum
    legal value (the USD attribute is an unsigned int and exactly 4 GiB dies at ``gym.make``), so
    there is no headroom to spend even if this were the pool under pressure.
    IF PhysX nonetheless logs "collisionStackSize buffer overflow detected ... Contacts have been
    dropped" -- which does not crash, it silently makes a touching finger indistinguishable from a
    missing one, and would therefore silently disable every gate below -- the fix is the
    OBJECT-CENTRIC form: one ``object_hand_s`` sensor on ``{ENV_REGEX_NS}/InsertiveObject`` filtered
    to the five tips, which ``_sensor_force_magnitudes`` already prefers when present and which
    ``scripts_v2/tools/validate_c4_bank.py:451-454`` already runs against an OmniReset DELTO scene.
    One view instead of five, same forces (equal and opposite). Flagged rather than adopted because
    the per-tip form is the one confirmed working on THIS task id.
    """
    cfg.scene.robot = cfg.scene.robot.replace(spawn=cfg.scene.robot.spawn.replace(activate_contact_sensors=True))
    cfg.scene.insertive_object = cfg.scene.insertive_object.replace(
        spawn=cfg.scene.insertive_object.spawn.replace(activate_contact_sensors=True)
    )
    for tip in DELTO_ALL_TIP_NAMES:
        setattr(
            cfg.scene,
            f"{tip}_object_s",
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{DELTO_HAND_PRIM}/{tip}",
                filter_prim_paths_expr=["{ENV_REGEX_NS}/InsertiveObject"],
            ),
        )


def _apply_grasp_shaping(cfg) -> None:
    """Pay for closing the hand and for lifting what it closed on. Nothing else in this task does.

    Added dynamically on ``cfg.rewards``, the same mechanism ``_apply_precision_shaping`` uses and
    for the same reason: the new terms land after ``progress_context`` in ``vars()``, so the
    ProgressContext-reading terms still see a populated context. (These three read no context at
    all, so ordering is free for them; the mechanism is shared for consistency, not necessity.)

    THE YARDSTICK, and it is the same one ``_apply_precision_shaping`` argued against. The standing
    per-step cost of moving at all, measured on this policy, is

        action_rate    -1e-3 * 28 (measured)                        = -2.80e-3 per step

    and the failure that motivated the fine goal term was a task signal of +1.09e-4 per step sitting
    26x BELOW it -- invisible. Every weight below is sized as a multiple of that same -2.80e-3, in
    the same ``weight * f`` units (``step_dt`` = 0.1 s multiplies every term identically, so it
    cancels out of every ratio here).

    (a) ``any_finger_contact`` -- ``any_contact``, f in {0,1}, true when ANY of the five tips loads
        the cube above 0.2 N.

            0.01 * 1  = +1.0e-2 per step  =  3.6x the action-rate penalty

        The cheap approach signal. Deliberately the SMALLEST of the three: it is satisfied by one
        finger resting on the cube, which is precisely the degenerate "park the open hand on it"
        behaviour the audit found. Big enough to be visible (anything under 2.8e-3 is not), small
        enough that (b) strictly dominates it.

    (b) ``grasp_contact`` -- ``contacts``, f in {0,1}, true only when at least one of the two
        opposable digits AND at least one of fingers 2/3/4 both exceed 0.2 N, i.e. the cube is
        PINCHED rather than touched.

            0.05 * 1  = +5.0e-2 per step  = 17.9x the action-rate penalty
                                          =  5.0x term (a), so a pinch beats a touch by 4.0e-2/step

        Cross-checked against the precedent's own GAE arithmetic (gamma 0.99, lam 0.95, effective
        horizon 16.8 steps, ``_apply_precision_shaping``): the advantage of switching to a
        permanently-pinching policy is 16.8 * 5.0e-2 = 0.84, against the 1.68 that docstring
        computes for crossing the binary success gate. Holding the cube is worth HALF of solving the
        task, per unit horizon -- a strict subgoal, never a substitute. And the two are not
        alternatives in the first place: ``success_reward`` pays 1.0 every step the cube is aligned
        and does not require a release, so a policy holding the cube at the goal collects (b) AND
        the success term together.

    (c) ``object_lift`` -- ``object_upward_velocity_bonus``, f = tanh(vz / std) gated on the same
        opposition contact, signed.

        std is 0.05 m/s, NOT dexlift's 0.2. dexlift set 0.2 against ``RelativeJointPositionAction``
        at 0.1 rad per step on the arm; this task drives ``RelCartesianOSC``, whose realised motion
        was measured at ~4.7 mm of end-effector travel per unit action per 0.1 s step
        (``_apply_precision_shaping``) -- about 0.047 m/s at full throttle, which the held cube
        tracks. At std 0.2 a full-throttle lift scores tanh(0.235) = 0.23 and three quarters of the
        kernel's range sits on velocities this action space cannot reach. At std 0.05 it scores
        tanh(0.94) = 0.735:

            0.05 * 0.735 = +3.7e-2 per step  = 13.1x the action-rate penalty, full-throttle lift
            0.05 * 0.44  = +2.2e-2 per step  =  7.9x, at half throttle (0.024 m/s)

        Signed and odd in vz, so a symmetric up-then-down round trip integrates to ~0: it pays for
        getting the cube off the table and hands the payment back on the placement descent, rather
        than biasing the policy to hold the cube high. The stack seat sits ~34 mm ABOVE the spawn
        height, so the net over a completed stack is positive.

        HONEST BOUND on the obvious exploit: because tanh saturates, a fast-up/slow-down oscillation
        does farm this rather than integrating to zero. It is bounded at 0.05 * 1.0 = 5.0e-2 per
        step, i.e. a logged 0.05 against ``success_reward``'s 1.0, and the existing ``joint_vel``
        (-1e-2) and ``abnormal_robot`` (-100) terms already price arm velocity.

    WHAT THIS LOOKS LIKE IN TENSORBOARD. The RewardManager multiplies by ``step_dt`` = 0.1 s
    (decimation 12, sim.dt 1/120) and IsaacLab divides the episode sum by ``max_episode_length_s`` =
    16.0, so for a full 160-step episode the logged value is exactly ``weight * mean(f)``. A policy
    pinching for 60% of the episode logs 0.030 for (b), against ``action_rate``'s -0.028,
    ``dense_success_reward``'s ~0.090 and ``dense_success_reward_fine``'s <=0.100. Visible, and about
    a third of the transport term -- not a new dominant objective.

    NO OBSERVATION CHANGES. Deliberately: the actor is 380 wide (76 features x 5 history) and the
    critic 375, and ``runner.load()`` is ``strict=True``, so a single extra ObsTerm would make every
    existing checkpoint fail to load and break the curriculum runs that resume from them. The policy
    reads contact only through the return, which is what an RL reward is for.
    """
    gate = {
        "threshold": GRASP_CONTACT_THRESHOLD_N,
        "thumb_contact_name": DELTO_THUMB_TIP_NAMES,
        "tip_contact_names": DELTO_TIP_NAMES,
    }
    # THE DENSE TERM THAT LEADS TO CONTACT. The three contact-gated terms below pay only once
    # contact already exists, so on their own they are a reward for an outcome with nothing leading
    # to it. Measured on the four-family run ten iterations after they were switched on:
    # any_finger_contact logged 2.1e-4 and grasp_contact 4e-5, against an action_rate penalty of
    # -2.8e-2 -- real and rising, so the sensors read, but three orders of magnitude below the noise
    # they had to be discovered in.
    #
    # ee_asset_distance does not lead there either: it reads ONE body, the palm rl_dg_mount, so it
    # is fully satisfied by parking the palm at the cube with the hand wide open -- which is exactly
    # what the recordings show the policy doing.
    #
    #     0.05 * (1 - tanh(0.010/0.03))  -  0.05 * (1 - tanh(0.030/0.03))  =  +2.2e-2 per step
    #
    # for closing the fingertips from 30 mm to 10 mm, i.e. ~8x the action_rate penalty, on the same
    # yardstick every other weight here is sized against.
    cfg.rewards.fingertip_object_distance = RewTerm(
        func=task_mdp.fingertip_object_distance_tanh,
        weight=FINGERTIP_DISTANCE_WEIGHT,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(DELTO_ALL_TIP_NAMES)),
            "object_cfg": SceneEntityCfg("insertive_object"),
            "std": FINGERTIP_DISTANCE_STD_M,
        },
    )
    cfg.rewards.any_finger_contact = RewTerm(
        func=task_mdp.any_contact,
        weight=ANY_CONTACT_WEIGHT,
        params={"threshold": GRASP_CONTACT_THRESHOLD_N, "contact_names": DELTO_ALL_TIP_NAMES},
    )
    cfg.rewards.grasp_contact = RewTerm(func=task_mdp.contacts, weight=GRASP_CONTACT_WEIGHT, params=dict(gate))
    cfg.rewards.object_lift = RewTerm(
        func=task_mdp.object_upward_velocity_bonus,
        weight=LIFT_WEIGHT,
        # The ported default is SceneEntityCfg("object"); OmniReset's manipulated body is
        # "insertive_object" and there is no entity named "object" in this scene at all.
        params={**gate, "std": LIFT_STD_MPS, "object_cfg": SceneEntityCfg("insertive_object")},
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
        _apply_fingertip_contact_sensors(self)
        _apply_grasp_shaping(self)


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
        _apply_fingertip_contact_sensors(self)
        _apply_grasp_shaping(self)


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
        _apply_fingertip_contact_sensors(self)
        _apply_grasp_shaping(self)


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
        _apply_fingertip_contact_sensors(self)
        _apply_grasp_shaping(self)


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


# ---------------------------------------------------------------------------------------
# TWO-FINGER VARIANT: the hand acts in 8 dimensions, not 20
# ---------------------------------------------------------------------------------------
# WHAT IT DOES. Replaces the 20-dimensional hand action with ``DELTO_TWO_FINGER_ACTIONS``
# (``ur10e_delto/actions.py:198-204``), which drives fingers 1 and 2 only. Total action dimension
# falls 6 + 20 = 26 -> 6 + 8 = 14. For reference the paper's own robot acts in 7 (6 Cartesian + a
# binary gripper), so 26 was always a large departure from the setting its hyperparameters were
# chosen in.
#
# WHY {1, 2} AND NOT {1, 5}. Both are stated in this repo and they are not in conflict, but the
# distinction decides whether an opposed grip is reachable at all. Fingers 1 and 5 are the two
# THUMB-SIDE digits -- ``validate_c4_bank.py:87`` ("a thumb-side tip (rl_dg_1 or rl_dg_5)"),
# ``_THUMB_SIDE_STIFFNESS`` above, and ``DELTO_THUMB_TIP_NAMES`` all say so -- and the opposition
# gate opposes EITHER of them against fingers 2/3/4. So {1, 5} is two digits on the SAME side of
# the hand and cannot pinch anything between them; {1, 2} takes one from each side and is the
# hand's designed jaw. ``DeltoHand/metadata.yaml:8`` states it outright ("a virtual jaw of finger 1
# against finger 2") and ``grasp_center_offset`` is ``midpoint(rl_dg_1_tip, rl_dg_2_tip)``, with
# the same file recording that the five-tip centroid was REJECTED as the grasp centre because
# finger 5 sits at y = -68 mm. {1, 2} is therefore the one pair that leaves the calibrated grasp
# centre -- and with it the whole sampled grasp bank -- meaningful.
#
# WHY THE STORED BANKS STAY VALID. The obvious form of this change is to hold fingers 3/4/5 at
# ``DELTO_TUCKED_HELD_FINGER_JOINT_POS`` (``actions.py:179-192``). That would invalidate every
# bank: the banks record all 26 joint positions from grasps sampled with all five fingers, so
# forcing 3/4/5 to a different posture at reset moves them against the cube and disturbs the grasp
# the bank was graded on. Regenerating the banks is hours of GPU.
#
# So the tuck is NOT applied. Fingers 3/4/5 are simply left out of the action term, and hold
# whatever posture the reset put them in. Verified against the sources rather than assumed:
#
#   * ``RelativeJointPositionAction.apply_actions`` calls
#     ``set_joint_position_target(..., joint_ids=self._joint_ids)`` -- it writes ONLY the joints
#     the term resolved. ``Articulation.set_joint_position_target`` assigns into
#     ``_data.joint_pos_target[env_ids, joint_ids]``, a persistent buffer, so entries no term
#     writes are left untouched.
#   * ``Articulation.reset()`` does NOT clear ``joint_pos_target``, so a stale target from the
#     previous episode WOULD persist if nothing rewrote it. Both reset paths this task uses do
#     rewrite it, for the full joint set:
#       - ``MultiResetManager._reset_to`` (``omnireset/mdp/events.py:2506``) calls
#         ``set_joint_position_target(joint_position_target, env_ids=env_ids)`` with no
#         ``joint_ids``, i.e. all 26 joints from the bank's own recorded target, and then asserts
#         at ``:2549-2560`` that what was applied equals what was intended.
#       - ``reset_end_effector_from_grasp_dataset`` (``events.py:1500-1508``) reconciles the
#         targets explicitly over ``self.gripper_joint_ids`` -- all 20 hand joints -- and its own
#         comment records why (a stale gripper target is a measured ~11 rad median gap that reaches
#         PhysX on the first physics step).
#
#   Net: after reset, fingers 3/4/5 carry the bank's own recorded target and nothing moves it for
#   the rest of the episode. That is strictly closer to the recorded grasp than the tuck would be.
#
# WHAT IS NARROWED, AND WHAT DELIBERATELY IS NOT. ``override_gripper_joints`` (called once for
# every DELTO variant at ``delto_cfg.py:262``) points three things at the 20-joint regex. Only ONE
# of them may be narrowed here:
#   * ``check_gripper_joints`` -- the startup half of the full-actuation guard. It resolves its
#     ``joint_names`` on the articulation and then requires each resolved joint to own an action
#     dimension (``events.py:1137-1138`` -> ``full_actuation.assert_joints_independently_actuated``).
#     Left at 20 it would fail the process at startup, correctly. Narrowed to the two-finger regex
#     it enforces exactly the same rule over exactly the joints this variant drives -- the guard
#     forbids joints SHARING an action dimension, and ``required_joints`` is a caller argument
#     precisely so a variant can say which joints it is making that promise about.
#   * ``reset_end_effector_pose_from_grasp_dataset``'s ``gripper_cfg`` -- NOT narrowed. This is the
#     selection the grasp replay writes fingers 3/4/5 through; narrowing it is exactly how the
#     "banks stay valid" argument above would be broken.
#   * ``randomize_gripper_actuator_parameters`` -- NOT narrowed. Gain DR on a joint no action
#     drives is inert, and leaving it whole keeps the DR distribution identical to the runs this
#     variant is compared against.
# The narrowing is written into THIS variant's own event term, so ``delto_cfg.py`` and every other
# DELTO task keep the 20-joint requirement unchanged.
#
# THE CONSTRUCTION-TIME GUARD IS RE-ASSERTED, not skipped. ``delto_cfg.py:268`` runs
# ``assert_action_cfg_fully_actuates`` over all 20 joints inside ``_apply_delto``, which happens
# during ``super().__post_init__()`` -- i.e. BEFORE the swap below, against the action term that is
# still there at that moment. Passing it therefore proves nothing about the config that ships. The
# helper re-runs the guard after the swap, at the narrowed requirement, so the two-finger term is
# checked in its own right rather than inheriting a check of something it replaced.
#
# CONTACT REWARDS MUST BE NARROWED WITH IT. ``fingertip_object_distance_tanh`` is
# ``1 - tanh(mean_tip_distance / std)``, a MEAN over the tips it is handed. Three tips that can no
# longer be commanded would contribute a distance the policy cannot reduce, putting a floor under
# that mean and flattening the term's gradient -- the same "length scale chosen without reference
# to the distances the policy occupies" failure ``FINGERTIP_DISTANCE_STD_M`` documents, arrived at
# from the other direction. The opposition gate is narrowed for a stronger reason: its thumb side
# would otherwise be satisfiable by finger 5, which no action can move, and its non-thumb side by
# fingers 3/4.
#
# The sensors for the undriven tips are dropped too. Setting a scene entry to ``None`` removes it
# (``InteractiveScene._add_entities_from_cfg`` skips ``asset_cfg is None``); this is the same idiom
# ``CubeStackPartialAssembliesCfg`` uses to drop ``axial_depth_sampling``. Three fewer filtered
# contact views per environment at 2048 envs, reading forces nothing consumes.
#
# COST. Existing checkpoints will NOT load. ``prev_actions`` is an ObsTerm, so the actor's feature
# width falls 76 -> 64 (26 prev_actions -> 14, ``joint_pos`` stays 26 because the articulation
# still HAS 20 hand joints, 4 pose terms x 6 unchanged) and the concatenated actor observation
# 380 -> 320 at history 5. The critic falls 375 -> 363 at history 1. ``runner.load()`` is
# ``strict=True``. That is expected and is the price of the smaller action space, not a bug.
DELTO_TWO_FINGER_THUMB_TIP_NAMES = ("rl_dg_1_tip",)
"""Finger 5 is the OTHER thumb-side digit and is undriven here, so the gate's thumb side is one tip."""
DELTO_TWO_FINGER_TIP_NAMES = ("rl_dg_2_tip",)
DELTO_TWO_FINGER_ALL_TIP_NAMES = DELTO_TWO_FINGER_THUMB_TIP_NAMES + DELTO_TWO_FINGER_TIP_NAMES
# Derived by difference rather than restated, so a change to either list above cannot leave a
# sensor registered for a finger the action term no longer drives.
DELTO_UNDRIVEN_TIP_NAMES = tuple(tip for tip in DELTO_ALL_TIP_NAMES if tip not in DELTO_TWO_FINGER_ALL_TIP_NAMES)


def _apply_two_finger_hand(cfg) -> None:
    """Drive fingers 1 and 2 only; leave 3/4/5 holding whatever posture the reset gave them.

    MUST be called LAST -- after ``_apply_fingertip_contact_sensors`` and ``_apply_grasp_shaping``,
    which is what ``super().__post_init__()`` on the NoOrient base does. It narrows the terms those
    two register rather than pre-empting them, so the base classes stay byte-for-byte untouched and
    the three live NoOrient training runs are unaffected.

    See the block comment above for why the pair is {1, 2}, why the stored reset banks survive this
    unchanged, and which of ``override_gripper_joints``'s three selections may be narrowed.
    """
    # ``.replace()`` rather than assigning the module-level instance: the same shared-config hazard
    # ``_apply_fingertip_contact_sensors`` documents at length for ``scene.robot.spawn``. Nothing
    # mutates an action term today, so this is insurance, not a fix.
    cfg.actions.gripper = ur10e_delto.DELTO_TWO_FINGER_ACTIONS.replace()

    # The startup guard, re-scoped to the joints this variant actually promises to actuate
    # independently. Term absent on a variant that does not carry it -> nothing to narrow.
    check = getattr(cfg.events, "check_gripper_joints", None)
    if check is not None:
        check.params["joint_names"] = [ur10e_delto.DELTO_TWO_FINGER_JOINT_REGEX]

    # ...and the construction-time half, which delto_cfg.py:268 ran against the action term this
    # line just replaced. Raises from this frame, so it cannot be swallowed by a PLAY callback.
    assert_action_cfg_fully_actuates(cfg.actions, ur10e_delto.DELTO_TWO_FINGER_JOINT_NAMES, context=type(cfg).__name__)

    for tip in DELTO_UNDRIVEN_TIP_NAMES:
        setattr(cfg.scene, f"{tip}_object_s", None)

    gate = {
        "threshold": GRASP_CONTACT_THRESHOLD_N,
        "thumb_contact_name": DELTO_TWO_FINGER_THUMB_TIP_NAMES,
        "tip_contact_names": DELTO_TWO_FINGER_TIP_NAMES,
    }
    cfg.rewards.fingertip_object_distance.params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=list(DELTO_TWO_FINGER_ALL_TIP_NAMES)
    )
    cfg.rewards.any_finger_contact.params["contact_names"] = DELTO_TWO_FINGER_ALL_TIP_NAMES
    # ``update``, not replacement: ``object_lift`` carries ``std`` and ``object_cfg`` alongside the
    # gate and both must survive.
    cfg.rewards.grasp_contact.params.update(gate)
    cfg.rewards.object_lift.params.update(gate)


@configclass
class CubeStackTwoFingerTrainCfg(CubeStackNoOrientTrainCfg):
    """Position-only cube stacking with an 8-dimensional hand. Total action dimension 14."""

    def __post_init__(self):
        super().__post_init__()
        _apply_two_finger_hand(self)


@configclass
class CubeStackTwoFingerEvalCfg(CubeStackNoOrientEvalCfg):
    """Play/eval twin of :class:`CubeStackTwoFingerTrainCfg`, on the eval OSC gains."""

    def __post_init__(self):
        super().__post_init__()
        _apply_two_finger_hand(self)


# =========================================================================================
# GRIP BIAS -- the fix for "every EEGrasped reset drops the cube on the first control step"
# =========================================================================================
# THE DEFECT, MEASURED. `grip_probe.py` resets into each family and applies the null action for
# 6 s. ObjectAnywhereEEGrasped starts the cube at 102 mm and ends at 34 mm: it slides out of the
# hand and onto the table, with every fingertip contact force under 0.01 N against a 0.29 N cube
# weight. ObjectRestingEEGrasped and ObjectPartiallyAssembledEEGrasped look stable only because
# their cube is already resting on the table or on the base cube -- there too the nearest fingertip
# is 35-40 mm from the cube centre and carries no load. None of the three "EEGrasped" families is
# actually holding anything by the time the policy gets to act.
#
# WHY. The hand action term is `RelativeJointPositionAction` with `use_zero_offset=True`:
#
#     joint_position_target = measured_joint_pos + scale * action        (+ offset, which is 0)
#
# `MultiResetManager._reset_to` restores the bank's recorded `joint_position_target` -- the squeeze
# the recorder had commanded when it validated the grasp -- but that target survives only until the
# first `apply_actions()`, which unconditionally overwrites it. At action = 0 the new target equals
# the measurement, the PD error is zero, and so is the grip force. The recorded grasp is discarded
# one control step after every single reset.
#
# The consequence is visible in the training logs and explains them exactly:
# `Metrics/task_1_success_rate` (ObjectAnywhereEEGrasped) sits at 0.0000 after 200 iterations, while
# the near-goal family trains to ~0.20 -- because near-goal never needed a grasp. Its cube is
# resting on the base cube and the task is a nudge.
#
# THE FIX. A constant offset on the flexion joints restores exactly the condition the recorder
# captured: a PD set point held a fixed angle ahead of the measurement. Against a finger blocked by
# the cube that is a constant force (stiffness x bias); against a free finger it is a slow close.
# It is a BIAS, NOT A CONSTRAINT -- with scale 0.1 and clip (-1, 1) the policy still commands
# [-0.1 + bias, +0.1 + bias] per step, so opening the hand stays available at every bias below
# 0.1 rad. Nothing about the reset banks changes; they become valid instead of hollow.
#
# WHY THE BANK CANNOT SUPPLY THE SQUEEZE ITSELF. Decoding the banks directly: the stored hand
# posture IS the validated closed posture (mean |q - closed| = 0.049 rad on C3 and 0.065 on C4,
# against |q - open| = 0.281 / 0.293), so the geometry the recorder captured is right -- the fingers
# really are around the cube. But the stored `joint_position_target` sits a mean of 0.0036 rad from
# the stored `joint_position`. Applied effort on a PD joint is stiffness x (target - q), so 3.6 mrad
# on the 0.30 N*m/rad distal joints is about 0.04 N at the fingertip against a 0.29 N cube. The
# recorder ended its scripted close with the fingers SETTLED at the closed posture, so what it
# captured is a geometric closure with essentially no force in it. Restoring that target perfectly
# would still not hold the cube; there is no grip in the bank to restore.
#
# WHICH JOINTS, AND HOW MUCH -- measured, not argued. The `--hold_hand` control in `grip_probe.py`
# drives all twenty hand joints at +0.1 rad/step and settles the fingertip forces at
# 0.48 / 1.10 / 0.94 / 0.50 / 2.56 N, holding the C3 cube for the whole episode (z 91.3 -> 95.0 mm,
# it even lifts slightly, and the cube is pulled 6 mm further into the grasp centre). So closure is
# entirely within the hand's authority; only the command was missing. A bias restricted to the
# `_2`/`_4` flexion pair -- the joints that differ between the stored open and closed postures --
# got the C3 drop only from 83 mm down to 40 mm at 0.08 rad. The `_3` joints are IDENTICAL in both
# stored postures yet carry the largest stiffness in the hand (4.0 N*m/rad), and the hold_hand
# forces show they supply most of the grip. So: every hand joint.
GRIP_BIAS_RAD = 0.05


def _grip_bias_joint_names(action_cfg) -> list[str]:
    """Every joint the installed hand action term drives. See the block comment above."""
    return list(action_cfg.scale) if isinstance(action_cfg.scale, dict) else []


def _apply_grip_bias(cfg, bias: float = GRIP_BIAS_RAD) -> None:
    """Hold the grasp the reset bank recorded, instead of dropping it on the first step.

    MUST be called LAST, after any function that replaces ``cfg.actions.gripper`` -- notably
    ``_apply_two_finger_hand``. It reads the joint set off whatever action term is installed, so it
    composes with the two-finger variant without knowing about it.
    """
    g = cfg.actions.gripper
    offset = {n: bias for n in _grip_bias_joint_names(g)}
    if not offset:
        raise ValueError(
            f"{type(cfg).__name__}: gripper action term has no dict `scale`, so the grip-bias joint "
            "set cannot be resolved by name. Give the action term a per-joint scale dict."
        )
    cfg.actions.gripper = g.replace(use_zero_offset=False, offset=offset)


@configclass
class CubeStackGripTrainCfg(CubeStackNoOrientTrainCfg):
    """Five-finger position-only cube stacking whose EEGrasped resets keep hold of the cube."""

    def __post_init__(self):
        super().__post_init__()
        _apply_grip_bias(self)


@configclass
class CubeStackGripEvalCfg(CubeStackNoOrientEvalCfg):
    """Play/eval twin of :class:`CubeStackGripTrainCfg`."""

    def __post_init__(self):
        super().__post_init__()
        _apply_grip_bias(self)


@configclass
class CubeStackTwoFingerGripTrainCfg(CubeStackTwoFingerTrainCfg):
    """The two-finger hand AND the grip bias. Action dimension 14; resets that actually hold."""

    def __post_init__(self):
        super().__post_init__()
        _apply_grip_bias(self)


@configclass
class CubeStackTwoFingerGripEvalCfg(CubeStackTwoFingerEvalCfg):
    """Play/eval twin of :class:`CubeStackTwoFingerGripTrainCfg`."""

    def __post_init__(self):
        super().__post_init__()
        _apply_grip_bias(self)
