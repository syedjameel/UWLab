# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Tesollo DELTO DG-5F per-gripper seam.

Structural twin of ``linear_gripper_cfg.py``: the 2F-85 task configs are subclassed and only the
robot and the action are swapped; objects, events, rewards, sim settings and the object
``variants`` are inherited unchanged, and neither the 2F-85 nor the linear-gripper path is
touched. This module holds the parts that are the same for every DELTO variant; the arm-specific
task classes live in ``ur10e_delto_cfg.py``.

THREE THINGS DIFFER FROM THE PARALLEL JAW, and each is one call in :func:`_apply_delto`:

1. **Gripper joints.** 20 revolute ``rj_dg_{1..5}_{1..4}`` instead of two prismatic jaws, and the
   OPEN configuration is a 20-entry POSTURE rather than a single number. Declared through
   ``gripper_seam.override_gripper_joints`` (the same seam the linear gripper uses), with the
   posture read from ``Robots/DeltoHand/metadata.yaml`` -- the HAND's file, not the full robot's.
   That is where the hand's grasp fields live by design: the grasp sampler's ``scene.robot`` is
   the hand-only articulation, so that file is what the recorded grasps were sampled against, and
   reading it here is what keeps the pre-grasp seeds opening the hand into the same configuration.
   ``Ur10eDelto/metadata.yaml`` carries only ``gripper_offset`` and the arm identification.

2. **End-effector body name.** The linear gripper renames its links to the 2F-85 contract
   (``robotiq_base_link``), so every ``body_names=`` reference in the base configs is inherited
   for free. The DELTO does not: its palm link is ``rl_dg_mount``. Eight call sites bind that
   name -- reset IK configs, the reset-state success check, and the ``ee_asset_distance``
   reward -- and all of them are repointed by :func:`_apply_delto_ee_body`. Geometrically nothing
   moves: the graft mounts ``rl_dg_mount`` directly on the wrist_3 flange with zero standoff.

   THE SWEEP IS NOT EXHAUSTIVE, AND CANNOT BE. There is a NINTH end-effector call site of a
   different kind: ``mdp/terminations.py``'s ``check_reset_state_success`` resolves the gripper's
   ``gripper_approach_direction`` from ``env.scene["robot"].cfg.spawn.usd_path`` -- the FULL
   robot's metadata directory, not the hand's. That is an end-effector METADATA SOURCE, selected by
   ``SceneEntityCfg.name`` (which asset), whereas :data:`_EE_BODY_TERMS` rewrites
   ``SceneEntityCfg.body_names`` (which body). No amount of body repointing reaches it. It killed
   every DELTO reset-state env at construction until the key was duplicated into
   ``Ur10eDelto/metadata.yaml``.
   ``mdp/rewards.py``'s ``ee_asset_distance_tanh`` reads ``gripper_offset`` the same way and works
   only because that key happens to live on the robot side of the split. When adding a task variant,
   check both kinds.

3. **Approach axis, which needs a ROLL as well as a pitch shift.** The 2F-85 approaches along its
   +X and the linear gripper along +Z (hence its hard-coded ``pitch += pi/2``). Both are basis
   vectors, so a pitch shift alone re-centres them. The DELTO's approach is
   ``gripper_approach_direction``, which is a general direction -- forward of the palm as well as
   out along the fingers, and off the palm's x-z plane. Pitch rotates about Y and cannot move a
   vector out of, or into, that plane, so with roll pinned at 0 the y-component of the approach
   axis is a hard FLOOR on how close to vertical the hand can be placed: 12.64 deg at the A7 axis,
   at every pitch and every yaw, silently. Both corrections are therefore DERIVED from the axis and
   applied together -- see :func:`_delto_roll_offset` and :func:`_delto_pitch_shift`. Both reduce
   to the shipped behaviour when ``a_y`` is 0, i.e. for either parallel jaw.

EVERY NUMBER THIS MODULE USES COMES FROM METADATA BY KEY, NEVER COPIED INTO CODE. That is not
style: the DELTO grasp block (``maximum_aperture``, ``grasp_align_axis``,
``gripper_approach_direction``, ``finger_offset``, ``finger_clearance``, ``gripper_offset``) is
tied to the OPEN posture, which has already moved once (the first one failed a self-collision
check at 0.15 mm minimum non-adjacent clearance and a grip-force check at 40 N on a 30-52 g part;
A7 replaced it, commit 6f4ae54). The few figures quoted in the docstrings below are illustrations
of the values current on 2026-08-06, not inputs -- nothing recomputes from them, and the next
posture change flows through with no edit to this module.

NOT here: grip force. It is set entirely by the hand actuator in ``ur10e_delto.py`` (per-joint
stiffness from the USD, zeta ~0.7 damping, a ~2.2 N fingertip effort backstop and a 3.0 rad/s
closing speed) together with whatever posture the policy commands over the twenty independent
hand actions. A task-config override on top of those would be a flat number replacing per-joint
ones, i.e. looser, not safer.

Registered gym id from this module:
* ``OmniReset-Delto-GraspSampling-v0`` (:class:`DeltoGraspSamplingCfg`)

The arm-carrying DELTO task classes live in ``ur10e_delto_cfg.py``.
"""

from __future__ import annotations

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.ur10e_delto as ur10e_delto
from uwlab.envs.mdp.full_actuation import assert_action_cfg_fully_actuates
from uwlab_assets.robots.ur10e_delto.actions import DELTO_HAND_JOINT_NAMES

from ...mdp.utils import read_metadata_from_usd_directory
from .grasp_sampling_cfg import Robotiq2f85GraspSamplingCfg
from .gripper_seam import override_gripper_joints

# The hand's 20 actuated joints, as one regex -- the same expression the actuator group and the
# hand action use, so all three select the same set by construction.
_DELTO_HAND_JOINTS = [r"rj_dg_[1-5]_[1-4]"]

# The DELTO's palm link -- its equivalent of ``robotiq_base_link``. This is the body the metadata's
# ``gripper_offset`` is expressed in, the body the reset IK drives, and the body the wrist camera
# hangs off.
DELTO_EE_BODY = "rl_dg_mount"

# Every base-config term that binds the end-effector BODY by name, as
# (config section, term name, parameter holding the binding). ``ee_body_name`` holds a bare string;
# the rest hold a SceneEntityCfg whose ``body_names`` is rewritten.
_EE_BODY_TERMS = (
    ("events", "reset_end_effector_pose", "robot_ik_cfg"),
    ("events", "reset_end_effector_pose_from_grasp_dataset", "robot_ik_cfg"),
    ("rewards", "ee_asset_distance", "root_asset_cfg"),
)


def _hand_metadata() -> dict:
    """``Robots/DeltoHand/metadata.yaml`` -- the hand's own grasp fields.

    Deliberately NOT the full robot's file. ``read_metadata_from_usd_directory`` keys off a USD's
    directory, and the DELTO's grasp geometry (open posture, approach direction, aperture, closing
    axis) is authored next to the HAND-ONLY USD because that is the articulation the grasp sampler
    spawns. The full robot's file holds ``gripper_offset`` and the arm identification and has none
    of these keys, so reading it here would raise -- which is the correct failure, but only after
    someone wonders why. Cached by ``read_metadata_from_usd_directory``.
    """
    return read_metadata_from_usd_directory(ur10e_delto.DELTO_HAND.spawn.usd_path)


def _delto_roll_offset(metadata) -> float:
    """Roll that brings the hand's approach axis into the plane the pitch sweep can reach.

    The EEAnywhere reset builds the end-effector's WORLD orientation as
    ``quat_from_euler_xyz(roll, pitch, yaw)`` (``events.py:795``), which is
    ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` -- verified numerically against the IsaacLab
    implementation rather than read off the name. The approach axis in world is then
    ``d = Rz(yaw) @ Ry(pitch) @ Rx(roll) @ a``.

    Write ``v = Rx(roll) @ a``. ``Ry(pitch)`` cannot change ``v_y`` and ``Rz(yaw)`` only spins the
    result about world Z, so as pitch sweeps, ``d`` traces a CONE about the body Y axis of
    half-angle ``90 deg - asin(v_y)``. It passes through straight-down only if ``v_y == 0``. With
    roll pinned at 0 -- which is what the 2F-85 base config does, correctly for a gripper whose
    approach axis is a basis vector -- ``a_y`` is therefore a hard FLOOR on how close to vertical
    the hand can ever be placed, at every pitch and every yaw. Nothing reports it; the resets are
    simply drawn from a tilted distribution.

    Zeroing ``v_y = a_y*cos(roll) - a_z*sin(roll)`` gives

        roll = atan2(a_y, a_z)

    after which ``v = (a_x, 0, hypot(a_y, a_z))``, exactly in the sweep plane.

    Note this is NOT ``asin(a_y)``, the angle the axis leans out of the plane -- that is the size
    of the error, not the rotation that removes it. At the A7 approach direction the two differ by
    1.5 deg (14.184 vs 12.639).

    Returns 0.0 for both shipped parallel jaws (``a_y == 0``), so the formula is a strict
    generalisation and no existing variant would move if it were applied to them.
    """
    _, a_y, a_z = (float(v) for v in metadata["gripper_approach_direction"])
    return math.atan2(a_y, a_z)


def _delto_pitch_shift(metadata) -> float:
    """Pitch offset that points the hand's approach axis DOWN over the inherited pitch range.

    The 2F-85's range ``(pi/4, 3pi/4)`` is centred on ``pi/2``, which is where ITS approach axis
    (+X of the palm) points straight down. A gripper whose approach axis is ``a`` points down when
    ``Ry(pitch) @ a`` is most negative in z, i.e. at ``pitch = pi - atan2(a_x, a_z)``, so the range
    slides by ``pi/2 - atan2(a_x, a_z)``.

    ONE WRINKLE, AND IT COUPLES THIS TO :func:`_delto_roll_offset`: the shift has to be computed on
    the ROLL-CORRECTED axis, not the raw one. After the roll the axis is
    ``(a_x, 0, hypot(a_y, a_z))``, whose z-component is larger than ``a_z``, so

        shift = pi/2 - atan2(a_x, hypot(a_y, a_z))

    Applying the roll without re-deriving the shift leaves the sweep mis-centred by the difference
    (0.7 deg at the A7 axis) -- small, but it is free to get right and silent to get wrong. For a
    parallel jaw ``a_y`` is 0 and ``hypot(a_y, a_z) == a_z``, so this still reproduces both shipped
    grippers exactly: 0 for the 2F-85 (``a = +X``) and ``+pi/2`` for the linear gripper
    (``a = +Z``, the hard-coded shift in ``_apply_linear_gripper``).

    At the A7 metadata (2026-08-06) the pair comes out roll ``+0.24756`` rad (+14.184 deg), shift
    ``+1.10389`` rad, pitch ``(1.8893, 3.4601)``. Measured over 200k samples of the resulting
    range, the approach axis reaches straight down to 0.0000 deg, against a 12.6392 deg floor
    before the fix. Those figures move with ``gripper_approach_direction``; the formulas do not.
    """
    a_x, a_y, a_z = (float(v) for v in metadata["gripper_approach_direction"])
    return math.pi / 2.0 - math.atan2(a_x, math.hypot(a_y, a_z))


def _exclude_hand_from_abnormal(cfg) -> None:
    """Restrict the authors' ``abnormal_robot`` check (|joint_vel| > 2x limit) to the ARM joints.

    Same scoping as the linear gripper's ``_exclude_gripper_from_abnormal``, for the same reason:
    the check exists to catch runaway ARM dynamics, and a force-limited end effector cannot run
    away. The DELTO adds 20 joints to a check that fires an episode-ending -100 penalty if ANY
    selected joint trips, so leaving them in means twenty extra chances for a benign drive
    overshoot to be read as a robot fault.

    THIS IS LOAD-BEARING. DO NOT DELETE IT AS DEAD CODE. An earlier revision of this docstring said
    the opposite -- that the scoping was a no-op because ``_DELTO_HAND_ACTUATOR`` set
    ``velocity_limit_sim = 10000``, putting 2x it out of reach. The current 3.0 rad/s limit makes
    the hand-joint threshold 2 x 3.0 = 6.0 rad/s -- BELOW the USD's independently-enforced
    ``physxJoint:maxJointVelocity`` of 7.31 rad/s, and therefore reachable. The comment silently
    became false when a number moved under it, which is exactly how the reader most likely to
    delete this line would be misled.

    Reachable is not the same as measured, but the margin that used to make this comfortable is
    gone. The old argument was that the binary close commanded at most 0.1411 rad per joint, so a
    normal closure could not come near 6 rad/s. The hand is now twenty independent relative joint
    actions and a policy can drive any of them at the actuator's full 3.0 rad/s indefinitely, i.e.
    at half the trip threshold by intent rather than by accident. The exposure is therefore larger
    than it was: on top of reset teleports (``write_joint_state_to_sim`` writes a posture with no
    velocity budget) and hard contact during PPO exploration, ordinary hand exploration now sits
    much closer to the limit. Nobody has measured whether it trips in practice -- and the
    linear gripper is the precedent for why that matters: its equivalent check started firing on
    ~9% of episodes as soon as its jaws were speed-capped, freezing the ADR curriculum below its
    0.95 gate. Scoping the check to the arm removes that failure mode before it can appear.

    CALLED FROM :func:`_apply_delto`, i.e. for every DELTO variant. It used to be called only from
    the two finetune configs, which left the Stage-1 training env -- the one this commit newly
    exposed -- without it.
    """
    arm = SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"])
    if getattr(cfg, "rewards", None) is not None and getattr(cfg.rewards, "abnormal_robot", None) is not None:
        cfg.rewards.abnormal_robot.params = {"asset_cfg": arm}
    if getattr(cfg, "terminations", None) is not None and getattr(cfg.terminations, "abnormal_robot", None) is not None:
        cfg.terminations.abnormal_robot.params = {"asset_cfg": arm}


def _apply_delto_ee_body(cfg) -> None:
    """Repoint every end-effector body binding from ``robotiq_base_link`` to the DELTO palm."""
    for section_name, term_name, param in _EE_BODY_TERMS:
        section = getattr(cfg, section_name, None)
        term = getattr(section, term_name, None) if section is not None else None
        if term is None or not hasattr(term, "params") or param not in term.params:
            continue
        term.params[param] = term.params[param].replace(body_names=DELTO_EE_BODY)
    # The reset-state success check takes the body as a bare name, not a SceneEntityCfg.
    success = getattr(getattr(cfg, "terminations", None), "success", None)
    if success is not None and "ee_body_name" in getattr(success, "params", {}):
        success.params["ee_body_name"] = DELTO_EE_BODY
        # The finite soft-pad contact has harmless residual rolling that is above the parallel
        # jaw's very tight recorder threshold but far below an ejection.  These overrides are
        # DELTO-local; the generic defaults and both existing grippers remain byte-for-byte.
        success.params["max_articulation_velocity_sum"] = 10.0
        success.params["max_rigid_linear_velocity_sum"] = 0.15
        success.params["max_rigid_angular_velocity_sum"] = 3.0
        success.params["consecutive_stability_steps"] = 3


def _apply_delto(cfg, robot, action) -> None:
    """Swap the 2F-85 robot/action for the UR10e + DELTO and fix everything that is 2-jaw.

    Call this AFTER ``super().__post_init__()``: the finetune configs set ``scene.robot`` inside
    their own ``__post_init__`` (to the EXPLICIT 2F-85), so the override has to come last to win.
    """
    cfg.scene.robot = robot.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.actions = action

    metadata = _hand_metadata()

    # (1) Point grasp replay and gripper-gain randomization at the hand's 20 joints. The base
    # 2F-85 regex (".*left.*") matches nothing here and would raise during term parsing.
    # ``require_independent_actuation`` also arms the startup half of the full-actuation guard,
    # which re-runs the check below against the joints the action terms RESOLVED to.
    override_gripper_joints(cfg, _DELTO_HAND_JOINTS, require_independent_actuation=True)

    # (1a) THE FULL-ACTUATION GUARD, at construction time. Every DELTO variant reaches this line,
    # because every DELTO variant swaps its action here -- that is what makes the guard structural
    # rather than a convention each new task config has to remember. It raises from this frame, so
    # it cannot be swallowed the way a SceneEntityCfg failure inside a PLAY callback is.
    assert_action_cfg_fully_actuates(cfg.actions, DELTO_HAND_JOINT_NAMES, context=type(cfg).__name__)

    # (1b) Scope the authors' abnormal_robot check to the ARM, for EVERY DELTO variant rather than
    # only the finetune ones. The check fires an episode-ending -100 reward and a termination when
    # any selected joint exceeds twice its velocity limit; the hand's limit is 3.0 rad/s, so its
    # threshold is 6.0 rad/s, BELOW the USD's independently enforced 7.31 rad/s cap and therefore
    # reachable -- and reachable by ordinary exploration now that the hand is twenty independent
    # relative commands rather than one binary posture selection. Applying it only to the finetune
    # configs left the Stage-1 TRAINING env, the one env this exposure is new in, unprotected.
    _exclude_hand_from_abnormal(cfg)

    # (2) The DELTO's palm link is rl_dg_mount, not robotiq_base_link.
    _apply_delto_ee_body(cfg)

    # (3) EEAnywhere orientation range: a fixed ROLL that puts the approach axis in the plane the
    # pitch sweep can reach, plus the PITCH shift that re-centres that sweep on straight-down. The
    # two are one correction and are derived together from the hand's approach axis -- see
    # _delto_roll_offset and _delto_pitch_shift. Both collapse to the shipped behaviour for a
    # parallel jaw (roll 0, and the same shift as before), so this is a generalisation of
    # _apply_linear_gripper's hard-coded +pi/2, not a divergence from it.
    #
    # Roll is set as a DEGENERATE range (r, r), not sampled: it is a property of the hand's
    # geometry, not something to randomize. Orientation variety comes from pitch and yaw, exactly
    # as it does for the parallel jaws.
    #
    # The dict is copied before writing so the shared 2F-85 pose_range_b is never mutated.
    ee = getattr(cfg.events, "reset_end_effector_pose", None)
    if ee is not None and "pose_range_b" in ee.params and "pitch" in ee.params["pose_range_b"]:
        shift = _delto_pitch_shift(metadata)
        roll = _delto_roll_offset(metadata)
        pr = dict(ee.params["pose_range_b"])
        lo, hi = pr["pitch"]
        pr["pitch"] = (lo + shift, hi + shift)
        pr["roll"] = (roll, roll)
        ee.params["pose_range_b"] = pr
        # The off-axis approach requires a much larger wrist rotation than the parallel-jaw
        # start pose. Ten relaxed IK teleports leave about 1.6 deg of error; 25 reduces the same
        # geometric residual below the C7 one-degree gate. This is deliberately DELTO-local so
        # the established grippers keep their reset distributions byte-for-byte.
        ee.params["ik_iterations"] = 25


def _apply_delto_collision_stack_size(cfg) -> None:
    """Widen the PhysX GPU collision-stack pool for the DELTO insertion task's plant.

    Lives here, not in ``ur10e_delto_cfg.py`` or ``ur5e_delto_cfg.py``, because the issue this
    fixes is a property of the DELTO HAND's contact profile (23 collider bodies, self-collisions
    ON, 5 fingertip sensors), not of either arm -- exactly the same reason ``_apply_delto`` itself
    is written generically over ``(cfg, robot, action)`` and shared by both arm files rather than
    duplicated. Call explicitly from a DELTO RL-state (insertion) task's ``__post_init__``, after
    ``_apply_delto``, the same way ``_set_arm_max_delay`` (arm-specific, so it correctly stays in
    ``ur10e_delto_cfg.py``) is called ad hoc from the one config that needs it.

    ``Ur5eRobotiq2f85RlStateCfg.__post_init__`` (rl_state_cfg.py:748) sets
    ``sim.physx.gpu_collision_stack_size = 2**31`` (2 GiB) for every gripper that base serves --
    DELTO included, on EITHER arm, via ``_apply_delto`` -- but that value was only ever proven
    sufficient for dexlift's plant and task, never for insertion's. Both certified dexlift runs
    (2026-08-16) use 4026531840 (3.75 GiB, env.yaml:48) instead: the MAXIMUM LEGAL value, since the
    USD attribute ``physxScene:gpuCollisionStackSize`` is an unsigned int and exactly 4 GiB
    (4294967296) dies at ``gym.make`` with a type mismatch. Different task family, so 3.75 GiB is
    not evidence 2 GiB (or 3.75 GiB) is enough here -- classic verified-here-applied-there (bead
    UWLab-z9j.11; first applied only to the UR10e+DELTO family and missed the UR5e+DELTO one this
    campaign actually trains, which is why this now lives in the shared, arm-independent file).

    An undersized stack does not crash: PhysX logs "collisionStackSize buffer overflow detected ...
    Contacts have been dropped" and the run keeps going. Reward and success on this task are gated
    on PER-FINGERTIP CONTACT FORCE, so a dropped contact is indistinguishable from a finger that
    never touched -- invisible in training metrics, the run just learns worse for no visible
    reason. It already bit once at 4096 envs with a constant that had worked at 2048.

    CORRECTION (independent audit, 2026-08-20): the "gated on per-fingertip contact force" claim
    above is FALSE for the OmniReset UR5eDelto/UR10eDelto RL-state task IDs this function is
    called from -- that reward/termination graph is entirely geometric (pose and joint state
    only; ``activate_contact_sensors=False`` on the robot, zero ``ContactSensorCfg`` in the
    scene). It correctly describes :class:`DeltoGraspSamplingCfg` below, which DOES check
    per-fingertip contact force via ``check_grasp_success``. A dropped contact here still risks
    silently corrupting whatever grasp/reset-state dataset the sampler produced upstream (see
    that class's own collision-stack-size comment for the chain this feeds), so the same ceiling
    value is still the right call for the RL-state task -- just not for the reason stated above.

    Set to the SAME maximum-legal value the certified dexlift runs use. The pool is FIXED-SIZE, not
    per-env, so there is no VRAM saving from a smaller value that would justify risking a silent
    contact drop, and the DELTO hand's contact profile is strictly heavier than whatever the 2f85
    precedent rl_state_cfg's 2 GiB was presumably sized against -- go straight to the ceiling rather
    than guess a number in between.

    NOT edited in rl_state_cfg.py: that base is shared by every gripper variant (Robotiq included);
    widening it there would silently change every OTHER gripper's resolved value too, which is not
    this bead's call. ``reset_states_cfg.py``'s own shared ``2**31`` default (line 675) is likewise
    untouched -- every DELTO class (reset-state, RL-state, and grasp sampling) opts in explicitly by
    calling this function instead.
    """
    cfg.sim.physx.gpu_collision_stack_size = 4026531840


def _apply_delto_dataset_dir(cfg, dataset_dir: str) -> None:
    """Repoint EVERY dataset_dir-bearing event term on ``cfg.events`` to ``dataset_dir``.

    Datasets (reset states, grasps, partial-assembly poses) are keyed by OBJECT PAIR ONLY --
    ``compute_pair_dir`` (mdp/utils.py:391-400) is ``"__".join(sorted(object_name_from_usd(p) for p
    in usd_paths))``, no gripper or arm discriminator anywhere, in either the ``Resets/{pair}/`` or
    the ``Grasps/{obj}/`` layout. Every gripper/arm variant that doesn't opt out of the shared
    default therefore reads and writes the SAME path. A UR5e+DELTO env left on that default reads a
    12-joint Robotiq dataset into a 26-DOF DELTO articulation and dies at reset with a shape
    mismatch -- proven live, not hypothetical (bead UWLab-z9j.11 follow-on).

    Iterates rather than hand-wires one term by name, DELIBERATELY: multiple terms across the five
    reset-state classes and the RL-state classes carry ``dataset_dir`` independently --
    ``reset_insertive_object_pose_from_reset_states``, ``reset_end_effector_pose_from_grasp_dataset``,
    ``reset_insertive_object_pose_from_partial_assembly_dataset``, ``reset_from_reset_states`` -- and
    which of them are even present varies by class (``ObjectAnywhereEEAnywhereEventCfg`` has none;
    ``ObjectRestingEEGraspedEventCfg`` and ``ObjectPartiallyAssembledEEGraspedEventCfg`` each carry
    two). A per-term hand-wiring is exactly the shape of bug this function exists to rule out: it
    silently stops covering a term the day someone adds a sixth reset-state variant, where iterating
    `cfg.events`'s own attributes for a ``"dataset_dir"`` params key does not.

    Repoints ALL of them uniformly to the SAME ``dataset_dir``, including the grasp-dataset term --
    even though ``grasps.pt`` is hand-only and, unlike a full articulation reset-state snapshot,
    would in principle be shareable across arms mounting the same hand. Uniform namespacing is
    simpler to reason about and audit than a per-term exception, and the storage cost of one more
    copy of a grasp dataset is not a real constraint.
    """
    for term_name, term_cfg in vars(cfg.events).items():
        params = getattr(term_cfg, "params", None)
        if isinstance(params, dict) and "dataset_dir" in params:
            params["dataset_dir"] = dataset_dir


# ---------------------------------------------------------------------------------------
# Grasp sampling (hand-only, like ROBOTIQ_2F85 / LINEAR_GRIPPER)
# ---------------------------------------------------------------------------------------
@configclass
class DeltoGraspSamplingCfg(Robotiq2f85GraspSamplingCfg):
    """Grasp sampling with the DELTO hand alone -- no arm, teleported onto candidate grasps.

    THIS ENV WAS DELETED ONCE, ON THE ARGUMENT THAT GRASP SAMPLING ONLY EXISTS TO SCRIPT A CLOSE
    COMMAND, WHICH THE FULLY ACTUATED HAND NO LONGER HAS, AND THAT NOTHING DOWNSTREAM NEEDED THE
    DATASET. The second half is false, and it is worth spelling out the chain because it is long
    enough to look absent:

        ``Grasps/<object>/grasps.pt``
          -> ``events.reset_end_effector_from_grasp_dataset`` (ACTIVE in reset_states_cfg for all
             three ``*EEGrasped`` variants; ``_apply_delto`` repoints it at the DELTO palm and the
             gripper seam points its ``gripper_cfg`` at these twenty joints)
          -> ``OmniReset-UR10eDelto-Object{Resting,Anywhere,PartiallyAssembled}EEGrasped-v0``
          -> ``Resets/<pair>/resets_*.pt``
          -> ``rl_state_cfg.TrainEventCfg.reset_from_reset_states`` (three of its four reset types)
          -> ``OmniReset-UR10eDelto-RelCartesianOSC-State-v0``, which is a TRAINING env.

    ``_load_and_precompute_grasps`` hard-raises when the loaded file names none of the resolved
    gripper joints, and the dataset path is keyed by OBJECT only, so the DELTO cannot borrow the
    2F-85's file: without this env, that chain has no producer and the branch runs off an
    unreproducible artifact.

    The first half of the argument was the real question, and the answer is that the CLOSE
    COMMAND WAS NEVER A PROPERTY OF THE ENVIRONMENT -- it is a property of the scripted rollout
    that drives it. The action here is the same twenty independent dimensions the policy gets
    (:data:`DeltoFullHandAction`), and ``record_grasps.py`` closes the hand by servoing each of
    those twenty toward a measured posture, by name. Nothing in this config, and nothing a policy
    could ever see, is a scalar. The full-actuation guard is applied to it like any other DELTO
    variant.

    The sampler reads its gripper geometry from the metadata of the GRIPPER asset's directory, so
    this env is driven by ``Robots/DeltoHand/metadata.yaml`` -- the same file ``_apply_delto`` reads
    for the full robot, which is what keeps the sampled grasps and the replayed ones consistent.
    """

    def __post_init__(self):
        # Swap the hand-only robot and its fully actuated action before the base configures sim.
        #
        # SAMPLER-LOCAL SEGMENT-4 STIFFNESS, and this is what makes the sampler produce grasps at
        # all. ur10e_delto.py:90-94 already records that the distal ``_4`` joints at their authored
        # 0.0957 N.m/rad "cannot overcome the reference actuator friction during this small
        # commanded stroke", and raises them to 0.30 -- but ONLY for rj_dg_2_4/3_4/4_4. rj_dg_1_4
        # (0.1698) and rj_dg_5_4 (0.1694) were left out, and those two are the distal joints of the
        # THUMB-SIDE pair the held-check opposes against fingers 2/3/4. With them stuck there is no
        # opposing jaw, so every candidate is a one-sided pinch and nothing is ever held.
        #
        # MEASURED, and exact: the relative action caps PD error at 0.1 rad per control step
        # regardless of distance to target, so realised torque is bounded by stiffness * 0.1. At
        # 0.1698 that is 0.0170 N.m -- matching the measured applied torque to four decimals --
        # against a 0.01 N.m friction floor and a 0.06 N.m effort limit that is never reached. The
        # joint sat frozen at ~1.05 rad, velocity exactly 0.000, in EVERY velocity/deadband/stiffness
        # combination swept. Joint limits are not the cause (target 1.495991 rad is inside the
        # authored 0-1.5708 range); friction is uniform at 0.01 across all hand joints.
        #
        # WHY NOT SIMPLY 0.30 LIKE ITS SIBLINGS: tested, and it is a measurable half-step, not a fix.
        # rj_dg_5_4 closes fully at 0.30, but rj_dg_1_4 only reaches 0.364 rad short of target by
        # episode end and 0/180 results. The strokes are unequal -- rj_dg_1_4 travels 1.337 rad
        # against rj_dg_5_4's 0.944, 42 percent further -- so the value chosen for the shorter-stroke
        # siblings was never sized for the thumb.
        #
        # EVIDENCE FOR 10x, and its limits: 10x reproduces across three seeds (2/12, 1/12, 1/12 --
        # roughly 11 percent per-episode) with the thumb gap collapsing 1.267 -> 0.006 rad and
        # holding through gravity AND shake, 7 bodies in contact, no limit cycle. 20x is WORSE
        # (0/180, fewer bodies, lower peak force), so this is NOT monotonic and the value must not be
        # raised casually. 11 percent is well below the 60 percent acceptance bar; an asymmetric
        # value (the thumb wanting more than finger 5) is the untested next step, not a settled one.
        # Scoped to THIS sampler cfg via an actuators-dict rebind: the shared DELTO_HAND_ACTUATOR
        # instance that IMPLICIT_UR10E_DELTO and EXPLICIT_UR10E_DELTO use is deliberately untouched.
        # The segment-4 stiffness and velocity_limit_sim overrides this comment describes are
        # applied AFTER super().__post_init__() below, together with the deadband opt-out, so all
        # three parts of the working configuration sit in one block rather than being split either
        # side of the base call. Search for _sampler_hand_stiffness.
        self.scene.robot = ur10e_delto.DELTO_HAND.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions = ur10e_delto.DeltoFullHandAction()
        super().__post_init__()
        assert_action_cfg_fully_actuates(self.actions, DELTO_HAND_JOINT_NAMES, context=type(self).__name__)
        # The sampler poses the gripper by this BODY; the hand's root link is rl_dg_mount.
        self.events.grasp_sampling.params["gripper_cfg"] = self.events.grasp_sampling.params["gripper_cfg"].replace(
            body_names=DELTO_EE_BODY
        )
        self.events.grasp_sampling.params["num_orientations"] = 16
        # Topdown sweep knobs (roll/pitch tilt, lateral offset fraction, height offset fraction).
        # DeltoHand/metadata.yaml:48-51 zeroes all three -- a single fixed candidate pose, measured
        # (grasp_center_offset, same file) by direct PhysX replay against the 34 mm DeltoBlock cube,
        # where that fixed spot is clear of the part. Against a 200 mm bar ("leg200mm") the same
        # fixed spot lands fingers 2/3 inside the part on EVERY candidate, unconditionally -- a
        # structural 100% placement-gate rejection, not a threshold problem (see events.py
        # _filter_placement_collisions; that gate and _PLACEMENT_TOLERANCE are untouched by this).
        #
        # metadata.yaml is keyed by the GRIPPER's USD directory and is shared by every object this
        # sampler touches -- deltoblock included -- so it cannot carry a per-object tuning without
        # breaking deltoblock. Mirrors override_mass in grasp_sampling_cfg.make_object: a per-
        # object-variant knob that lives in config, not in the shared metadata file. Defaulted here
        # to TODAY's metadata value via the same _hand_metadata() read _delto_pitch_shift and
        # _delto_roll_offset use (single source of truth, not a copied literal), so deltoblock -- and
        # every other object that doesn't override this dict -- is bit-for-bit unchanged.
        #
        # The keys must be SET here, not merely reachable via the metadata fallback in events.py:
        # update_class_from_dict raises KeyError on any params key absent from the default cfg
        # (isaaclab/utils/dict.py:95,164), so a Hydra override can only ever touch a key that
        # already exists in this dict. With the keys present, tune leg200mm from the command line,
        # e.g.:
        #   ... env.scene.object=leg200mm \
        #       env.events.grasp_sampling.params.grasp_topdown_roll_pitch_deg=15.0 \
        #       env.events.grasp_sampling.params.grasp_topdown_position_fraction=0.3 \
        #       env.events.grasp_sampling.params.grasp_topdown_height_fraction=0.2 \
        #       env.events.grasp_sampling.params.grasp_topdown_yaw_deg=45.0
        # No numeric values are picked here for leg200mm -- that needs empirical iteration against
        # the bar's real geometry under Isaac, which is intentionally not done in this change.
        hand_metadata = _hand_metadata()
        self.events.grasp_sampling.params["grasp_topdown_roll_pitch_deg"] = float(
            hand_metadata.get("grasp_topdown_roll_pitch_deg", 6.0)
        )
        self.events.grasp_sampling.params["grasp_topdown_position_fraction"] = float(
            hand_metadata.get("grasp_topdown_position_fraction", 0.3)
        )
        self.events.grasp_sampling.params["grasp_topdown_height_fraction"] = float(
            hand_metadata.get("grasp_topdown_height_fraction", 0.2)
        )
        # Yaw sweep about the approach axis (decoupled from the roll/pitch wobble above -- see
        # events.py's _sample_topdown_grasps). Defaulted to 0.0 here for the same reason as the
        # three knobs above: DeltoHand/metadata.yaml doesn't carry this key, so hand_metadata.get
        # falls back to 0.0 and deltoblock (and everything else that doesn't override this dict) is
        # bit-for-bit unchanged. No numeric value is picked here for leg200mm -- that needs
        # empirical iteration against the bar's real geometry under Isaac, same as the other three.
        self.events.grasp_sampling.params["grasp_topdown_yaw_deg"] = float(
            hand_metadata.get("grasp_topdown_yaw_deg", 0.0)
        )
        # The scripted close has a 1.34 rad maximum stroke and the force-limited hand needs about
        # 10 s to close and settle. The inherited 4 s episode enables gravity after 1 s, while the
        # pads are still more than a radian from their target, and rejects valid candidates as
        # drops. Keep this timing DELTO-local: parallel jaws retain their 4 s sampler. The direct
        # C2 jitter matrix at 1200 simulation steps establishes the value.
        self.episode_length_s = 18.0
        physics = self.events.global_physics_control_event.params
        physics["gravity_on_interval"] = (10.0, float("inf"))
        physics["force_torque_on_interval"] = (11.0, 13.0)
        # Mass-scaled shake, DELTO only -- restores commit a589120 (2026-06-21, "grasp quality (WIP):
        # mass-scaled sampling shake"), deleted from the tree along with the rest of that WIP commit.
        # See events.py::global_physics_control_event for the mechanism and the full arithmetic this
        # restores (flat 0.01 N is ~3% of a 30 g object's weight linearly -- negligible -- but ~1730
        # rad/s^2 of angular acceleration against that same object's inertia -- the mechanism behind
        # every violent sampling tumble this campaign has measured). 5.0 m/s^2 is the value that
        # commit chose ("a firm, mass-independent tug"); kept, not re-derived, because we are
        # restoring a validated mechanism, not designing a new one. THIS IS A CHANGE TO A VALIDATION
        # TEST and must not be tuned to make grasps pass -- see the fuller argument on
        # force_torque_scale_by_mass in events.py for why mass-scaling is a MORE meaningful bar, not
        # a laxer one. Unlike a589120, which set this on the SHARED base (grasp_sampling_cfg.py,
        # every gripper), it is applied here, DELTO-only, so 2F-85 and the linear gripper keep
        # today's flat-Newton behaviour bit-for-bit.
        physics["force_torque_magnitude"] = 5.0
        physics["force_torque_scale_by_mass"] = True
        # Finite soft-pad contacts retain small harmless rolling motion after the shake.  Keep the
        # generic parallel-jaw thresholds unchanged; DELTO accepts motion below 15 cm/s summed
        # linear and 3 rad/s summed angular velocity, still far below an ejection trajectory.
        success = self.terminations.success.params
        success["max_object_linear_velocity_sum"] = 0.15
        success["max_object_angular_velocity_sum"] = 3.0
        success["max_gripper_joint_velocity_sum"] = 10.0
        success["consecutive_stability_steps"] = 3
        # Widen the collision-stack pool here too: this is the ROOT of the dependency tree, not a
        # leaf. The sampler validates each candidate by SHAKING the held object and re-reading the
        # same per-fingertip contact channel a dropped-contact overflow corrupts elsewhere -- so a
        # silent drop here does not merely corrupt one reset state, it CERTIFIES a grasp that never
        # actually held. That grasp is then replayed by differential IK into every downstream
        # reset-state and RL-state dataset built on it (see this class's own docstring for the
        # chain: Grasps/<object>/grasps.pt -> reset_end_effector_from_grasp_dataset ->
        # Resets/<pair>/resets_*.pt -> reset_from_reset_states -> the RL-state TRAINING env). Same
        # value, same reasoning, as every other DELTO class -- see _apply_delto_collision_stack_size.
        _apply_delto_collision_stack_size(self)

        # Segment-4 stiffness + velocity_limit_sim, a SAMPLER-ONLY hand-actuator copy (bead
        # UWLab-z9j.11 follow-on). Same defensive pattern as _set_arm_max_delay
        # (ur10e_delto_cfg.py) and _apply_delto_collision_stack_size above: rebind .actuators to a
        # NEW dict holding a .replace()'d ImplicitActuatorCfg, never writing through the shared
        # _DELTO_HAND_ACTUATOR / DELTO_HAND_ACTUATOR object IMPLICIT_UR10E_DELTO, EXPLICIT_UR10E_
        # DELTO and DELTO_HAND itself all point at. The certified training plant is untouched by
        # construction here, not by convention -- confirm by grep, not by inspection, if in doubt.
        #
        # RESULT THIS LANDS (first grasps of the campaign): check_grasp_success fired 2/12, after
        # 1620+ prior episodes at 0. The thumb joint (rj_dg_1_4), frozen at ~1.05 rad in every
        # earlier configuration, closed cleanly: 1.267 rad at t=0.5 s decaying monotonically to
        # 0.009 by t=11.0 s and holding flat at 0.006 through the entire shake. Seven bodies in
        # contact, peak 0.75 N, no limit cycle. REPRODUCIBILITY RESEEDS WERE RUNNING IN PARALLEL
        # when this landed -- if x10 does not replicate, these values may not survive; read this as
        # a record of the evidence that produced them, not a settled, re-derivable-from-first-
        # -principles answer.
        #
        # ROOT CAUSE: ur10e_delto.py:90-94 already documents that distal _4 joints at their authored
        # 0.0957 N*m/rad "cannot overcome the reference actuator friction during this small
        # commanded stroke", and raises rj_dg_2_4/3_4/4_4 to 0.30 -- but OMITS rj_dg_1_4/5_4 (the
        # thumb-side pair), which stayed at the authored 0.1698/0.1694. This completes that partial
        # fix; it does not invent a new value out of nothing.
        #
        # TORQUE ARITHMETIC AT STOCK (rj_dg_1_4), matching the measured applied torque to four
        # decimals: peak commanded torque = stiffness * max step-error = 0.1698 N*m/rad * 0.1 rad
        # (DELTO_HAND_ACTION_SCALE, one action unit) = 0.0170 N*m. Against the 0.01 N*m friction
        # floor for this joint class, that is only 0.0070 N*m of margin -- and nowhere near the
        # 0.06 N*m effort_limit_sim ceiling for the same class, which is never reached at stock.
        # That is why it was frozen: barely enough torque to move at all, not zero.
        #
        # WHY THE OMISSION WAS WORSE THAN A FLAT OVERSIGHT: the two thumb-side strokes are NOT
        # equal. rj_dg_1_4 travels 1.337 rad open-to-closed; rj_dg_5_4 travels 0.944 rad -- rj_dg_1_4
        # needs MORE torque than its sibling, not the same. Measured directly: at exactly 0.30
        # (matching the already-fixed siblings), rj_dg_5_4 closed fully but rj_dg_1_4 was still
        # 0.364 rad short at episode end, and 0/180 grasps succeeded. The value must sit above 0.30.
        #
        # WHY x10 (~1.70) AND NOT MORE: at x10, peak commanded torque rises to 1.698 * 0.1 = 0.1698
        # N*m, which EXCEEDS the 0.06 N*m effort_limit_sim ceiling -- so the torque actually
        # delivered for a large position error is capped at 0.06 N*m, not literally 10x the stock
        # torque; x10 is what makes the joint able to use its full existing 0.06 N*m effort budget
        # rather than a fraction of it. A stiffness of x20 (~3.40) was ALSO tested and is WORSE --
        # 0/180, fewer bodies in contact, lower peak force -- so this relationship is NOT monotonic
        # and the value should not be raised casually on the assumption that more stiffness is
        # strictly better; x20's own failure mode was not substep-traced the way x10's success was.
        #
        # VELOCITY_LIMIT_SIM 1.0 (down from the shared actuator's 3.0), same class of reasoning as
        # the deadband fix earlier in this file: at 3.0 rad/s the actuator can travel
        # velocity_limit_sim * control_dt = 3.0 * 0.1 s = 0.3 rad within one control step, against
        # the SAME 0.1 rad commanded delta -- a 3x overshoot that was a standing cause of the
        # permanent limit cycle the deadband alone had to absorb. At 1.0 rad/s, one control step's
        # maximum travel (0.1 rad) matches the commanded delta exactly, so there is nothing left to
        # oscillate about even before the deadband intervenes.
        _sampler_hand_stiffness = dict(ur10e_delto.DELTO_HAND_ACTUATOR.stiffness)
        _sampler_hand_stiffness["rj_dg_1_4"] = 1.698  # 10x the authored 0.1698
        _sampler_hand_stiffness["rj_dg_5_4"] = 1.694  # 10x the authored 0.1694
        self.scene.robot.actuators = {
            **self.scene.robot.actuators,
            "hand": self.scene.robot.actuators["hand"].replace(
                stiffness=_sampler_hand_stiffness,
                velocity_limit_sim={r"rj_dg_[1-5]_[1-4]": 1.0},
            ),
        }
        # Deadband OFF for the sampler: at velocity_limit_sim 1.0 and this stiffness, the standing
        # limit cycle the deadband exists to break was not observed in the result above -- see
        # scripted_gripper.py's own comment on scripted_gripper_deadband_rad for the mechanism and
        # why this must not become the shared default.
        self.scripted_gripper_deadband_rad = 0.0
