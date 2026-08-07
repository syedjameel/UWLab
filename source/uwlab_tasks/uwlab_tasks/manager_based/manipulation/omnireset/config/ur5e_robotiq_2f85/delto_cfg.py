# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Tesollo DELTO DG-5F per-gripper seam, plus its grasp-sampling task.

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
tied to the open/closed posture pair, and that pair has already moved once (the first one failed a
self-collision check at 0.15 mm minimum non-adjacent clearance and a grip-force check at 40 N on a
30-52 g part; A7 replaced it, commit 6f4ae54). The few figures quoted in the docstrings below are
illustrations of the values current on 2026-08-06, not inputs -- nothing recomputes from them, and
the next posture change flows through with no edit to this module.

NOT here: grip force. It is set entirely by the hand actuator in ``ur10e_delto.py`` (per-joint
stiffness from the USD, zeta ~0.7 damping, a ~2.2 N fingertip effort backstop and a 3.0 rad/s
closing speed) together with the binary action's closed posture. A task-config override on top of
those would be a flat number replacing per-joint ones, i.e. looser, not safer.

Registered gym id from this module:
* ``OmniReset-Delto-GraspSampling-v0``
"""

from __future__ import annotations

import math

from isaaclab.utils import configclass

import uwlab_assets.robots.ur10e_delto as ur10e_delto

from ...mdp.utils import read_metadata_from_usd_directory
from .grasp_sampling_cfg import Robotiq2f85GraspSamplingCfg
from .gripper_seam import override_gripper_joints

# The hand's 20 actuated joints, as one regex -- the same expression the actuator group and the
# binary action use, so all three select the same set by construction.
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
    override_gripper_joints(cfg, _DELTO_HAND_JOINTS)

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


# ---------------------------------------------------------------------------------------
# Grasp sampling (hand-only, like ROBOTIQ_2F85 / LINEAR_GRIPPER)
# ---------------------------------------------------------------------------------------
@configclass
class DeltoGraspSamplingCfg(Robotiq2f85GraspSamplingCfg):
    """Grasp sampling with the DELTO hand alone -- no arm, teleported onto candidate grasps.

    The sampler reads its gripper geometry from the metadata of the GRIPPER asset's directory, so
    this env is driven by ``Robots/DeltoHand/metadata.yaml`` -- the same file ``_apply_delto`` reads
    for the full robot, which is what keeps the sampled grasps and the replayed ones consistent.
    """

    def __post_init__(self):
        # Swap the hand-only robot and its binary action before the base configures sim.
        self.scene.robot = ur10e_delto.DELTO_HAND.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions = ur10e_delto.DeltoBinaryGripperAction()
        super().__post_init__()
        # The sampler poses the gripper by this BODY; the hand's root link is rl_dg_mount.
        self.events.grasp_sampling.params["gripper_cfg"] = self.events.grasp_sampling.params["gripper_cfg"].replace(
            body_names=DELTO_EE_BODY
        )
        self.events.grasp_sampling.params["num_orientations"] = 16
        # The distal-only level posture has a 1.34 rad maximum stroke and the force-limited hand
        # needs about 10 s to close and settle. The inherited 4 s episode enables gravity after 1 s,
        # while the pads are still more than a radian from their target, and rejects valid
        # candidates as drops. Keep this timing DELTO-local: parallel jaws retain their 4 s
        # sampler. The direct C2 jitter matrix at 1200 simulation steps establishes the value.
        self.episode_length_s = 18.0
        physics = self.events.global_physics_control_event.params
        physics["gravity_on_interval"] = (10.0, float("inf"))
        physics["force_torque_on_interval"] = (11.0, 13.0)
        # Finite soft-pad contacts retain small harmless rolling motion after the shake.  Keep the
        # generic parallel-jaw thresholds unchanged; DELTO accepts motion below 15 cm/s summed
        # linear and 3 rad/s summed angular velocity, still far below an ejection trajectory.
        success = self.terminations.success.params
        success["max_object_linear_velocity_sum"] = 0.15
        success["max_object_angular_velocity_sum"] = 3.0
        success["max_gripper_joint_velocity_sum"] = 10.0
        success["consecutive_stability_steps"] = 3
