# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e + DELTO RGB configs: camera alignment + RGB data collection / play.

Structural twin of ``ur10e_linear_gripper_rgb_cfg.py``, with ``_apply_delto`` in place of
``_apply_linear_gripper``. The scene, the three ``TiledCamera``s, the observations, the
terminations and the object/table/HDRI randomization are all inherited unchanged.

THE POINT OF THIS MODULE IS CAMERA PARITY. A vision policy trained on the linear gripper and one
trained on the DELTO have to differ in the hand they see and in NOTHING ELSE -- same viewpoints,
same intrinsics, same rig. So:

* **Front and side cameras are reused verbatim.** They are parented to the robot ROOT, not to the
  end effector, so the 2026-07-16 ArUco calibration of the real rig transfers with no derivation
  at all. This module imports ``_UR10E_CAMERA_POSES`` rather than restating the numbers, which is
  what makes "byte-identical" true by construction instead of by careful copying.
* **The curtain planes are reused verbatim** for the same reason -- same physical rig, same
  fabric, measured once.
* **The wrist camera is the only thing that moves, and it is DERIVED.** It hangs off the gripper
  base link, and the DELTO has no ``robotiq_base_link``. Its stored extrinsic is LINK-RELATIVE, so
  re-parenting it without re-deriving would leave it at a different world pose while every number
  still looked calibrated. See :data:`_DELTO_CAMERA_POSES` below.

Registered gym ids (mirroring the UR10e linear-gripper ones):
* ``OmniReset-UR10eDelto-CameraAlign-v0``
* ``OmniReset-UR10eDelto-RelCartesianOSC-RGB-DataCollection-v0``
* ``OmniReset-UR10eDelto-RelCartesianOSC-RGB-Play-v0``
"""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import uwlab_assets.robots.ur10e_delto as ur10e_delto

from ... import mdp as task_mdp
from .actions import Ur10eDeltoRelativeOSCEvalAction, Ur10eDeltoSysidOSCAction
from .camera_align_cfg import CameraAlignEnvCfg
from .data_collection_rgb_cfg import (
    Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg,
    Ur5eRobotiq2f85EvalRGBRelCartesianOSCCfg,
)
from .delto_cfg import _apply_delto
from .ur10e_linear_gripper_rgb_cfg import _UR10E_CAMERA_POSES, _apply_curtain_poses

# Reset states are robot-specific: the DELTO's grasps and reset states are recorded with the hand,
# not the jaw, so they cannot share the linear gripper's ``Datasets_ur10e``. CLI-overridable.
_DELTO_RESET_DIR = "./Datasets_ur10e_delto/OmniReset"

# The graft nests the hand under ``/Robot/gripper/rl_dg_mount`` exactly as it nests the jaw under
# ``/Robot/gripper/robotiq_base_link``. Body-name lookups go through ``_apply_delto``; the camera
# is a PRIM PATH and has to be spelled out.
_WRIST_CAM_PRIM = "{ENV_REGEX_NS}/Robot/gripper/rl_dg_mount/rgb_wrist_camera"
_WRIST_CAM_TEMPLATE = "/World/envs/env_{}/Robot/gripper/rl_dg_mount/rgb_wrist_camera"


def _fix_wrist_camera_path(cfg) -> None:
    """Repoint the wrist camera (scene prim + any DR event templates) at the DELTO palm link."""
    if getattr(cfg.scene, "wrist_camera", None) is not None:
        cfg.scene.wrist_camera.prim_path = _WRIST_CAM_PRIM
    for term in ("randomize_wrist_camera", "randomize_wrist_camera_focal_length"):
        ev = getattr(cfg.events, term, None) if getattr(cfg, "events", None) is not None else None
        if ev is not None and "camera_path_template" in ev.params:
            ev.params["camera_path_template"] = _WRIST_CAM_TEMPLATE


# ---------------------------------------------------------------------------------------
# Wrist camera extrinsic, RE-DERIVED for the DELTO palm link.
#
# The calibrated wrist pose in ``_UR10E_CAMERA_POSES`` is expressed relative to
# ``robotiq_base_link``. To land the camera at the IDENTICAL WORLD pose on a robot whose gripper
# base link is ``rl_dg_mount`` instead, the offset transforms as
#
#     E_mount_cam = inv(X_delto) @ X_linear @ E_robotiq_cam
#
# where ``X = inv(T_wrist3) @ T_gripper_base`` is that robot's wrist_3 -> gripper-base transform.
# Both grafts attach the gripper base to ``wrist_3_link`` with a FixedJoint, so X is constant and
# the relation holds at EVERY arm pose, not just the one the USDs are authored at. The OffsetCfg
# ``convention`` cancels out: a convention is a fixed rotation applied on the RIGHT, and this is a
# left-multiplication.
#
# THE MOUNT IS A TRANSFORM, NOT A SCALAR, AND IT IS COMPOSED AS ONE BELOW. It is tempting to
# shortcut this: ``graft_gripper_on_ur10e.py`` takes a scalar ``--standoff`` along wrist_3 +Z, both
# grafts were run at 0.049 m, so "subtract the standoffs" gives the right answer today. That
# shortcut is only valid while BOTH mounts are pure +Z translations, which is an accident of the
# current grafts and not a property anything enforces -- and it is exactly the assumption that
# would fail silently. The graft's own history says the residuals are real: ``rl_dg_mount`` came
# out 52 um and 0.036 deg off the ``/DeltoHand`` subtree root during grafting (fixed by placing the
# BASE LINK on target rather than the subtree root), which was enough to have the solver yank the
# hand at step 0. A pose correction here of that size lands the camera visibly wrong with every
# number still reading as calibrated.
#
# So the constants below are the MEASURED wrist_3 -> gripper-base poses read back out of the two
# shipped USDs, not the graft's input flags, and they are composed with full rigid-transform
# arithmetic. Re-measure and update them whenever either robot is re-grafted:
#
#     .../tmp/derive_delto_wrist_cam.py     (reads both USDs, prints the composed offset)
#
# Measured 2026-08-05: both are exactly (0, 0, 0.049) with identity rotation to 1e-19, so the
# composed correction is the identity and the calibrated wrist pose transfers verbatim -- the two
# cameras land at the same wrist_3-frame pose to 2.2e-16 m. That the answer is "no change" is the
# RESULT of the derivation, not a shortcut around it.
# ---------------------------------------------------------------------------------------
# (pos, quat wxyz) of the gripper base link in the wrist_3_link frame.
_LINEAR_GRAFT_MOUNT = ((0.0, 0.0, 0.049), (1.0, 0.0, 0.0, 0.0))
_DELTO_GRAFT_MOUNT = ((0.0, 0.0, 0.049), (1.0, 0.0, 0.0, 0.0))


def _quat_mul(a, b):
    """Hamilton product of two (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_rotate(q, v):
    """Rotate the 3-vector ``v`` by the unit quaternion ``q`` (w, x, y, z)."""
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * (q_vec x v); v' = v + w*t + q_vec x t
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _pose_inverse(pose):
    """Inverse of a (pos, quat) rigid transform.

    The quaternion is normalized first: a conjugate is only an inverse for a UNIT quaternion, and
    the constants above are hand-transcribed from a measurement, so they are unit only to the
    digits written down. (The shipped calibrated wrist quaternion, for one, has norm 1 - 1.3e-9.)
    """
    pos, quat = pose
    norm = math.sqrt(sum(c * c for c in quat))
    w, x, y, z = (c / norm for c in quat)
    inv_quat = (w, -x, -y, -z)
    inv_pos = _quat_rotate(inv_quat, (-pos[0], -pos[1], -pos[2]))
    return inv_pos, inv_quat


def _pose_compose(outer, inner):
    """``outer @ inner`` for two (pos, quat) rigid transforms."""
    (op, oq), (ip, iq) = outer, inner
    rotated = _quat_rotate(oq, ip)
    return (op[0] + rotated[0], op[1] + rotated[1], op[2] + rotated[2]), _quat_mul(oq, iq)


# inv(X_delto) @ X_linear -- the change of gripper-base frame the camera offset has to absorb.
_MOUNT_CORRECTION = _pose_compose(_pose_inverse(_DELTO_GRAFT_MOUNT), _LINEAR_GRAFT_MOUNT)

_LINEAR_WRIST_CAM = _UR10E_CAMERA_POSES["wrist_camera"]
_DELTO_WRIST_CAM_POS, _DELTO_WRIST_CAM_ROT = _pose_compose(
    _MOUNT_CORRECTION, (_LINEAR_WRIST_CAM["pos"], _LINEAR_WRIST_CAM["rot"])
)

_DELTO_CAMERA_POSES = {
    **_UR10E_CAMERA_POSES,
    "wrist_camera": {
        **_LINEAR_WRIST_CAM,
        "pos": _DELTO_WRIST_CAM_POS,
        "rot": _DELTO_WRIST_CAM_ROT,
    },
}


def _apply_camera_poses(cfg) -> None:
    """Write ``_DELTO_CAMERA_POSES`` onto the scene cameras AND the per-episode camera-pose /
    focal randomization event bases, so one edit updates the whole pipeline. Shared by the
    CameraAlign env (no DR events -> those parts no-op) and the DataCollection/Play envs.

    Same logic as the linear gripper's ``_apply_camera_poses``; it is repeated rather than
    imported only because that one reads its own module-level pose dict.
    """
    events = getattr(cfg, "events", None)

    def _event_term(term_name):
        ev = getattr(events, term_name, None) if events is not None else None
        return ev if ev is not None and hasattr(ev, "params") else None

    for name, p in _DELTO_CAMERA_POSES.items():
        cam = getattr(cfg.scene, name, None)
        if cam is not None:
            cam.offset.pos = p["pos"]
            cam.offset.rot = p["rot"]
            if getattr(cam, "spawn", None) is not None and hasattr(cam.spawn, "focal_length"):
                cam.spawn.focal_length = p["focal"]
                if "clip" in p:
                    cam.spawn.clipping_range = p["clip"]
        rc = _event_term(f"randomize_{name}")
        if rc is not None and "base_position" in rc.params:
            rc.params["base_position"] = p["pos"]
            rc.params["base_rotation"] = p["rot"]
        fc = _event_term(f"randomize_{name}_focal_length")
        if fc is not None and "focal_length_range" in fc.params:
            lo, hi = fc.params["focal_length_range"]
            hw = (hi - lo) / 2.0
            fc.params["focal_length_range"] = (p["focal"] - hw, p["focal"] + hw)


def _apply_wrist_camera_tracking(cfg) -> None:
    """Make the wrist camera's RENDERED pose track the DELTO palm link.

    Identical mechanism to the linear gripper's: in this Isaac build a link-mounted camera renders
    from a frozen spawn-time pose until its USD transform op is re-authored once, after which
    Fabric composes the local offset with the live physics link every frame. Installs the
    reset-time offset write plus the offset-jitter replacement for the authors' direct-prim-write
    wrist camera DR.
    """
    p = _DELTO_CAMERA_POSES["wrist_camera"]
    old = getattr(cfg.events, "randomize_wrist_camera", None)
    if old is not None and hasattr(old, "params") and "position_deltas" in old.params:
        cfg.events.randomize_wrist_camera = EventTerm(
            func=task_mdp.sample_link_camera_offset_jitter,
            mode="reset",
            params={
                "camera_name": "wrist_camera",
                "base_position": p["pos"],
                "base_rotation": p["rot"],
                "position_deltas": old.params["position_deltas"],
                "euler_deltas": old.params["euler_deltas"],
            },
        )
    # un-pin + track (runs after the jitter sampler: declaration order = execution order)
    cfg.events.track_wrist_camera = EventTerm(
        func=task_mdp.track_link_mounted_camera,
        mode="reset",
        params={
            "camera_path_template": _WRIST_CAM_TEMPLATE,
            "base_position": p["pos"],
            "base_rotation": p["rot"],
            "camera_name": "wrist_camera",
        },
    )


# ---------------------------------------------------------------------------------------
# Gripper appearance DR, repointed from the two jaw pads to the DELTO's own meshes.
#
# The 2F-85 pair of terms is "the housing right in front of the camera" plus "the pads that touch
# the object", and the DELTO mapping keeps that split: the palm assembly (mount + base + palm
# shells, the large near-field object in the wrist view) and the five fingertip pads.
#
# The 20 PHALANX links are deliberately NOT included. The term draws an INDEPENDENT material per
# matched prim, so adding them would paint one hand in 28 unrelated random colours -- not the
# thing the 2F-85 term does, and not a distribution any real hand is in. If per-phalanx variation
# turns out to matter, add ``gripper/rl_dg_(1|2|3|4|5)_(1|2|3|4)/visuals/.*`` here and expect the
# material count per env to grow accordingly.
#
# Patterns verified against ``Robots/Ur10eDelto/ur10e_delto.usd``: 3 and 5 mesh prims respectively,
# and no collision mesh is matched (colliders live below ``collisions`` while these expressions
# explicitly select ``visuals``). The graft
# flattens the hand, so these visuals are already de-instanced and bindable.
# ---------------------------------------------------------------------------------------
_DELTO_DR_MESHES = {
    "randomize_wrist_mount_appearance": ["gripper/rl_dg_(mount|base|palm)/visuals/.*/Scene/mesh"],
    "randomize_inner_finger_appearance": ["gripper/rl_dg_(1|2|3|4|5)_tip/visuals/.*/Scene/mesh"],
}


def _apply_delto_dr_meshes(cfg) -> None:
    for term, meshes in _DELTO_DR_MESHES.items():
        ev = getattr(cfg.events, term, None)
        if ev is not None and hasattr(ev, "params") and "mesh_names" in ev.params:
            ev.params["mesh_names"] = meshes


# ---------------------------------------------------------------------------------------
# Camera alignment (interactive sim2real camera calibration; used by align_cameras.py)
# ---------------------------------------------------------------------------------------
@configclass
class Ur10eDeltoCameraAlignEnvCfg(CameraAlignEnvCfg):
    """DELTO camera-alignment env: same minimal RGB scene as the 2F-85 one, UR10e + DELTO robot.

    No grasp/reset events (just RGB obs + a positioning action), so it swaps robot + action
    directly instead of going through ``_apply_delto``. The end-effector body rename does not
    matter here for the same reason -- nothing in this env binds it.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot = ur10e_delto.EXPLICIT_UR10E_DELTO.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Our rig's nominal is 0 (table asset frame == robot base frame).
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)
        self.actions = Ur10eDeltoSysidOSCAction()
        _fix_wrist_camera_path(self)
        _apply_camera_poses(self)
        _apply_curtain_poses(self)
        _apply_wrist_camera_tracking(self)


# ---------------------------------------------------------------------------------------
# RGB data collection / play
# ---------------------------------------------------------------------------------------
def _apply_delto_rgb(cfg) -> None:
    """Swap the 2F-85 robot/action for the UR10e + DELTO and fix the RGB-specific bits."""
    # Robot (IMPLICIT, matching the 2F-85 RGB pattern; measured arm delay is 0) + eval action.
    _apply_delto(cfg, ur10e_delto.IMPLICIT_UR10E_DELTO, Ur10eDeltoRelativeOSCEvalAction())
    # No TASK-CONFIG speed cap, unlike the linear-gripper RGB path which calls
    # _apply_real_gripper_speed here. The DELTO's cap lives in the actuator instead
    # (velocity_limit_sim 3.0 rad/s, A8), so it is already in force on this env; adding a second one
    # here would shadow it with a coarser number. See the note at the top of ur10e_delto_cfg.py.
    # Pin the measured arm motor delay (0) -- deployment dynamics.
    if getattr(cfg.events, "randomize_arm_sysid", None) is not None:
        cfg.events.randomize_arm_sysid.params["delay_range"] = (0, 0)
    # Reset from the DELTO datasets, not the cloud 2F-85 default (CLI-overridable).
    if getattr(cfg.events, "reset_from_reset_states", None) is not None:
        cfg.events.reset_from_reset_states.params["dataset_dir"] = _DELTO_RESET_DIR
    _apply_delto_dr_meshes(cfg)
    # Wrist camera prim path -> the DELTO palm link (+ its DR event templates).
    _fix_wrist_camera_path(cfg)
    # Calibrated camera poses/focals -- front/side verbatim, wrist re-derived.
    _apply_camera_poses(cfg)
    # Curtain planes at the measured real-rig fabric positions (same rig, same numbers).
    _apply_curtain_poses(cfg)
    # Wrist camera: per-step USD tracking + offset-jitter DR (renders frozen otherwise).
    _apply_wrist_camera_tracking(cfg)
    # Env spacing: our real-rig scene spans ~1.8 m in x, so the inherited 1.5 m puts the +x
    # neighbour's back curtain in front of this env's cameras. Same fix as the linear-gripper RGB
    # env; the scene geometry is identical and the hand does not change it.
    cfg.scene.env_spacing = 3.0


@configclass
class Ur10eDeltoDataCollectionRGBCfg(Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg):
    """RGB data collection (4-path resets): UR10e arm + DELTO hand."""

    def __post_init__(self):
        super().__post_init__()
        _apply_delto_rgb(self)


@configclass
class Ur10eDeltoEvalRGBCfg(Ur5eRobotiq2f85EvalRGBRelCartesianOSCCfg):
    """RGB play / in-distribution eval (1-path resets): UR10e arm + DELTO hand."""

    def __post_init__(self):
        super().__post_init__()
        _apply_delto_rgb(self)
