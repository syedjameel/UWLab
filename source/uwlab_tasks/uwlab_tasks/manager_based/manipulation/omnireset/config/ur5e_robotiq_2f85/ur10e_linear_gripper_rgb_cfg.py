# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e + linear-gripper RGB configs: camera alignment + RGB data collection / play.

Mirrors the 2F-85 RGB stack (``camera_align_cfg.py`` + ``data_collection_rgb_cfg.py``) with
the UR10e arm + custom linear gripper, using the same subclass-and-swap pattern as
``ur10e_linear_gripper_cfg.py``. The 2F-85 arms share the ``robotiq_base_link`` /
``wrist_3_link`` link contract, so the RGB scene, three ``TiledCamera``s (front/side/wrist
parented to ``robotiq_base_link``), observations, terminations, and the
object/table/HDRI + camera-pose/focal randomization all carry over unchanged (the curtain
planes are repositioned to our measured rig -- see ``_UR10E_CURTAIN_POSES``).

What is UR10e / linear-gripper specific here:
* robot + action swapped via ``_apply_linear_gripper`` (RGB collection/play) or directly
  (camera align, which has no grasp/reset events);
* the two 2F-85 gripper-appearance randomization terms are REPOINTED at our gripper's
  meshes (2026-07-17): the graft de-instances the gripper visuals (instance proxies cannot
  take per-env materials) and the wrist-mount term retextures the whole base_visual mesh
  (housing + camera-mount bracket + D405 body) while the finger term covers both jaw tips;
* resets read the UR10e datasets (``./Datasets_ur10e/OmniReset``, CLI-overridable);
* motor delay pinned to the measured 0 (matches Finetune-Play deployment dynamics).

Camera pos/rot/focal are the 2026-07-16 ArUco calibration of the real rig (see the
``_UR10E_CAMERA_POSES`` comment); verify/refine by eye with ``align_cameras.py`` (run
against the CameraAlign env below) before the real 80k collection. See
``UR10E_SIM2REAL_PROCEDURE.md`` §9.

Registered gym ids (mirroring the 2F-85 ones):
* ``OmniReset-UR10eLinearGripper-CameraAlign-v0``
* ``OmniReset-UR10eLinearGripper-RelCartesianOSC-RGB-DataCollection-v0``
* ``OmniReset-UR10eLinearGripper-RelCartesianOSC-RGB-Play-v0``
"""

from __future__ import annotations

import uwlab_assets.robots.ur10e_linear_gripper as ur10e_linear_gripper

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from ... import mdp as task_mdp
from .actions import Ur10eLinearGripperRelativeOSCEvalAction, Ur10eLinearGripperSysidOSCAction
from .camera_align_cfg import CameraAlignEnvCfg
from .data_collection_rgb_cfg import (
    Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg,
    Ur5eRobotiq2f85EvalRGBRelCartesianOSCCfg,
)
from .linear_gripper_cfg import _apply_linear_gripper
from .ur10e_linear_gripper_cfg import _apply_real_gripper_speed

# Reset states are robot-specific; the RGB collection resets from the UR10e datasets.
_UR10E_RESET_DIR = "./Datasets_ur10e/OmniReset"

# Our graft nests the gripper under ``/Robot/gripper/robotiq_base_link`` (the 2F-85 cloud
# asset has ``robotiq_base_link`` directly under ``/Robot``). Body-name obs lookups are
# unaffected, but the wrist camera is a PRIM PATH and must point at the nested link.
_WRIST_CAM_PRIM = "{ENV_REGEX_NS}/Robot/gripper/robotiq_base_link/rgb_wrist_camera"
_WRIST_CAM_TEMPLATE = "/World/envs/env_{}/Robot/gripper/robotiq_base_link/rgb_wrist_camera"


def _fix_wrist_camera_path(cfg) -> None:
    """Repoint the wrist camera (scene prim + any DR event templates) at the nested link."""
    if getattr(cfg.scene, "wrist_camera", None) is not None:
        cfg.scene.wrist_camera.prim_path = _WRIST_CAM_PRIM
    for term in ("randomize_wrist_camera", "randomize_wrist_camera_focal_length"):
        ev = getattr(cfg.events, term, None) if getattr(cfg, "events", None) is not None else None
        if ev is not None and "camera_path_template" in ev.params:
            ev.params["camera_path_template"] = _WRIST_CAM_TEMPLATE


# ---------------------------------------------------------------------------------------
# Calibrated camera poses (pos, rot=quat wxyz, focal_length).
#
# Both the CameraAlign env (what you calibrate WITH) and the DataCollection/Play envs (what
# collects) use these, so one edit updates the whole pipeline -- the cfgs read this dict at
# construction. Real D405s: front 409122273078, side 323622272232, wrist 409122272284
# (positional order in the diffusion_policy deploy).
#
# Calibrated 2026-07-16 (ArUco, anchor = TCP touch-off [0.455, 0, 0]; one D405 at a time on
# the laptop; arm held at pendant joints [69.58, -98.08, 138.53, -130.43, -89.95, -20.42]).
# front/side are base-frame from 2_get_isaacsim_extrinsics.py directly (robot at origin);
# wrist is link-relative: inv(T_base_robotiq_base_link) @ T_base_cam, link pose read from
# sim at the capture joints (get_link_pose_ur10e.py, sim q1 = pendant q1 - 90 deg).
# FOCAL: the intrinsics-derived value (fx * 20.955 / 640 ~= 12.8) does NOT reproduce the
# real FOV -- the renderer's effective FOV is ~11% wider than the USD focal/aperture math
# (and than TiledCamera.data.intrinsic_matrices claims). Focals here are therefore fitted
# EMPIRICALLY with sweep_camera_align.py (edge-map score against the real capture; the
# front sweep showed a clean peak: 12.82 -> 0.42, 14.32 -> 0.48, 14.82 -> 0.43).
# Refine further by eye with align_cameras.py; per-camera calib archived in diffusion_policy
# scripts/sim2real/perception/calibrations/{front,side,wrist}_camera_calib.json.
_UR10E_CAMERA_POSES = {
    # ArUco warm start + sweep_camera_align refinement (2026-07-16: focal 12.82->14.32,
    # yaw -0.5 deg, roll +1.0 deg, x +15 mm; blend residual ~0 -- see table_swap_snaps/sweep_front)
    "front_camera": dict(
        pos=(1.0143132, -0.2340576, 0.2753867),
        rot=(0.6296674, 0.5090740, 0.3741183, 0.4521041),
        focal=14.32,
    ),
    # sweep 2026-07-16 (focal 12.84->13.34, pitch -1.0, yaw -0.5, roll -0.5 deg, x -15 mm)
    # + align_cameras hand-tune same day (final: +x/+y/+z few mm, small rot nudge,
    # focal 13.34->13.64)
    "side_camera": dict(
        pos=(1.0052121, 0.3149001, 0.2685370),
        rot=(0.30918181, 0.26638421, 0.58000093, 0.70501417),
        focal=13.64,
    ),
    # LINK-relative offset (robotiq_base_link), straight from the ArUco calibration --
    # kept over the sweep result (user eyeball 2026-07-16): with the jaws properly OPEN
    # the ArUco values already blend cleanly (table_swap_snaps/18_openjaws); the sweep's
    # +15 mm x was fitting the OSC-hold settle artifact, not a real offset error.
    "wrist_camera": dict(
        pos=(0.0098791, -0.1087037, 0.0389222),
        rot=(0.0018127, -0.0175937, 0.91723, -0.3979652),
        focal=12.74,
        # the gripper visual mesh models the D405 body; the calibrated optical center sits
        # ~8 mm INSIDE it (exits 15 mm along the view axis) -> raise the near clip past the
        # modeled glass. 5 cm also matches the real D405's minimum range.
        clip=(0.05, 1.0e5),
    ),
}


def _apply_camera_poses(cfg) -> None:
    """Write ``_UR10E_CAMERA_POSES`` onto the scene cameras AND the per-episode camera-pose /
    focal randomization event bases, so a single edit updates the whole pipeline (this is the
    manual two-place edit the 2F-85 sim2real doc describes, automated). Shared by the
    CameraAlign env (no DR events -> those parts no-op) and the DataCollection/Play envs."""
    events = getattr(cfg, "events", None)

    def _event_term(term_name):
        # Guarded: CameraAlign has no DR events (events MISSING/None) -> returns None.
        ev = getattr(events, term_name, None) if events is not None else None
        return ev if ev is not None and hasattr(ev, "params") else None

    for name, p in _UR10E_CAMERA_POSES.items():
        cam = getattr(cfg.scene, name, None)
        if cam is not None:
            cam.offset.pos = p["pos"]
            cam.offset.rot = p["rot"]
            if getattr(cam, "spawn", None) is not None and hasattr(cam.spawn, "focal_length"):
                cam.spawn.focal_length = p["focal"]
                if "clip" in p:
                    cam.spawn.clipping_range = p["clip"]
        # keep the reset-time camera-pose jitter centered on the calibrated pose
        rc = _event_term(f"randomize_{name}")
        if rc is not None and "base_position" in rc.params:
            rc.params["base_position"] = p["pos"]
            rc.params["base_rotation"] = p["rot"]
        # recenter the focal-length jitter on the calibrated focal (keep the original width)
        fc = _event_term(f"randomize_{name}_focal_length")
        if fc is not None and "focal_length_range" in fc.params:
            lo, hi = fc.params["focal_length_range"]
            hw = (hi - lo) / 2.0
            fc.params["focal_length_range"] = (p["focal"] - hw, p["focal"] + hw)


# ---------------------------------------------------------------------------------------
# Curtain placement -- measured on the real rig 2026-07-16 (tape, table edge -> fabric):
# viewer at the green mat facing the robot (= the front camera's side), so viewer-left = -y.
# left 14 cm past y=-0.35, back 25 cm past x=-0.35, right 19 cm past y=+0.35. Rotations
# stay the authors'; planes are sized up so the longer UR10e table's background stays
# covered (visual-only, collision off -- oversizing is harmless). The fabric reaches the
# FLOOR (z=-0.676, see table_dims.yaml) and rises well above the arm: planes span
# z -0.68..+1.75. Default color = the real fabric's grey (collection DR retextures per
# episode anyway; this is what CameraAlign and non-DR frames show).
# ---------------------------------------------------------------------------------------
_UR10E_CURTAIN_POSES = {
    "curtain_left": dict(pos=(0.3, -0.49, 0.535), size=(0.01, 1.8, 2.43)),
    "curtain_back": dict(pos=(-0.60, 0.025, 0.535), size=(0.01, 1.25, 2.43)),
    "curtain_right": dict(pos=(0.3, 0.54, 0.535), size=(0.01, 1.8, 2.43)),
}
_UR10E_CURTAIN_COLOR = (0.32, 0.32, 0.33)  # real grey fabric (photo-matched, slightly cool)


def _apply_curtain_poses(cfg) -> None:
    """Place the background curtain planes where the real rig's fabric hangs."""
    for name, p in _UR10E_CURTAIN_POSES.items():
        curtain = getattr(cfg.scene, name, None)
        if curtain is not None:
            curtain.init_state.pos = p["pos"]
            curtain.spawn.size = p["size"]
            if getattr(curtain.spawn, "visual_material", None) is not None:
                curtain.spawn.visual_material.diffuse_color = _UR10E_CURTAIN_COLOR


def _apply_wrist_camera_tracking(cfg) -> None:
    """Make the wrist camera's RENDERED pose track the gripper link.

    In this Isaac build a link-mounted camera renders from a frozen spawn-time pose: the
    sensor view pins its world matrix in Fabric at spawn. One re-author of the camera's
    USD transform op un-pins it -- Fabric then composes the local offset with the live
    physics link every frame (see ``task_mdp.track_link_mounted_camera``). This installs:

    * a reset-time write of the link->camera offset (sufficient for tracking);
    * a replacement for the authors' per-episode wrist camera-pose DR: the direct prim
      write (``randomize_tiled_cameras``) is superseded by jittering the LINK->CAMERA
      offset the tracker authors (``sample_link_camera_offset_jitter``, same delta ranges).
    """
    p = _UR10E_CAMERA_POSES["wrist_camera"]
    # per-episode pose DR -> offset jitter (ranges inherited from the authors' term)
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
# Camera alignment (interactive sim2real camera calibration; used by align_cameras.py)
# ---------------------------------------------------------------------------------------
@configclass
class Ur10eLinearGripperCameraAlignEnvCfg(CameraAlignEnvCfg):
    """UR10e camera-alignment env: same minimal RGB scene as the 2F-85 one, UR10e robot.

    No grasp/reset events (just RGB obs + a positioning action), so it swaps robot + action
    directly like ``Ur10eLinearGripperSysidEnvCfg`` instead of ``_apply_linear_gripper``.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot = ur10e_linear_gripper.EXPLICIT_UR10E_LINEAR_GRIPPER.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        # Our rig's nominal is 0 (table asset frame == robot base frame; -0.039 was the
        # UR5e rig's measured placement -- the base class now uses 0 too).
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)
        self.actions = Ur10eLinearGripperSysidOSCAction()
        _fix_wrist_camera_path(self)
        _apply_camera_poses(self)
        _apply_curtain_poses(self)
        _apply_wrist_camera_tracking(self)


# ---------------------------------------------------------------------------------------
# RGB data collection / play
# ---------------------------------------------------------------------------------------
def _apply_ur10e_rgb(cfg) -> None:
    """Swap the 2F-85 robot/action for the UR10e linear gripper and fix RGB-specific bits."""
    # robot (IMPLICIT, matching the 2F-85 RGB pattern; measured delay is 0) + eval action.
    _apply_linear_gripper(
        cfg, ur10e_linear_gripper.IMPLICIT_UR10E_LINEAR_GRIPPER, Ur10eLinearGripperRelativeOSCEvalAction()
    )
    # Real jaw speed (1.0 s full stroke): the demos must teach the distilled policy the
    # real grip-wait timing, so collection runs at deployment gripper dynamics.
    _apply_real_gripper_speed(cfg)
    # Pin the measured motor delay (0) -- deployment dynamics (mirrors Finetune-Play).
    if getattr(cfg.events, "randomize_arm_sysid", None) is not None:
        cfg.events.randomize_arm_sysid.params["delay_range"] = (0, 0)
    # Reset from the UR10e datasets, not the cloud 2F-85 default (CLI-overridable).
    if getattr(cfg.events, "reset_from_reset_states", None) is not None:
        cfg.events.reset_from_reset_states.params["dataset_dir"] = _UR10E_RESET_DIR
    # Repoint the 2F-85 per-mesh gripper-appearance DR at OUR gripper meshes (2026-07-17;
    # previously dropped). The graft de-instances the gripper visuals so these are bindable.
    # The wrist-mount term now covers the whole base_visual mesh = housing + camera-mount
    # bracket + D405 body (the authors' D415_to_Robotiq_Mount equivalent); all other params
    # stay the authors' verbatim.
    # REGEX patterns (the term matches via rep.functional.get.prims): naming-agnostic to the
    # URDF converter's internal node names, which differ between conversion vintages
    # (".../visuals/base/node_STL_BINARY_/mesh" on one machine bit us on another). The
    # visuals subtrees only become matchable once the graft has DE-INSTANCED them --
    # re-run graft_gripper_on_ur10e.py after pulling, expect "de-instanced 3 prim(s)".
    _GRIPPER_DR_MESHES = {
        "randomize_wrist_mount_appearance": ["gripper/robotiq_base_link/visuals/.*"],
        "randomize_inner_finger_appearance": [
            "gripper/left_inner_finger/visuals/.*",
            "gripper/right_inner_finger/visuals/.*",
        ],
    }
    for term, meshes in _GRIPPER_DR_MESHES.items():
        ev = getattr(cfg.events, term, None)
        if ev is not None and hasattr(ev, "params") and "mesh_names" in ev.params:
            ev.params["mesh_names"] = meshes
    # Wrist camera prim path -> our nested gripper link (+ its DR event templates).
    _fix_wrist_camera_path(cfg)
    # Calibrated camera poses/focals.
    _apply_camera_poses(cfg)
    # Curtain planes at the measured real-rig fabric positions.
    _apply_curtain_poses(cfg)
    # Wrist camera: per-step USD tracking + offset-jitter DR (renders frozen otherwise).
    _apply_wrist_camera_tracking(cfg)
    # Env spacing: our real-rig scene is LONGER than the authors' (back curtain x=-0.60,
    # front camera x=+1.01 -> ~1.8 m x-span vs their ~1.25 m). At the inherited 1.5 m
    # spacing, the +x neighbor's back curtain stands ~10 cm IN FRONT of this env's
    # front/side cameras -- entire episodes stare at a close "wall" (and camera jitter at
    # the curtain edge peeks into the neighbor's cell: the "robot seen from behind"
    # frames). Verified in the 2026-07-17 4090 smoke (59/102 episodes flagged); 2-env
    # laptop smokes never showed it because 2 envs get placed along y. 3.0 m clears the
    # span with margin.
    cfg.scene.env_spacing = 3.0
    # NOTE (2026-07-17): a wrist_3 +-60 deg cable-constraint (joint_outside_window
    # termination + filter_reset_states --wrist3-window) was tried and REVERTED: the
    # discards cut collection throughput ~2-3x (100 demos took ~36 min -> 80k would take
    # weeks). The user instead routes the real cabling to tolerate full +-180 deg wrist
    # rotation. The tools remain available if the constraint ever returns.


@configclass
class Ur10eLinearGripperDataCollectionRGBCfg(Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg):
    """RGB data collection (4-path resets): UR10e arm + linear gripper."""

    def __post_init__(self):
        super().__post_init__()
        _apply_ur10e_rgb(self)


@configclass
class Ur10eLinearGripperEvalRGBCfg(Ur5eRobotiq2f85EvalRGBRelCartesianOSCCfg):
    """RGB play / in-distribution eval (1-path resets): UR10e arm + linear gripper."""

    def __post_init__(self):
        super().__post_init__()
        _apply_ur10e_rgb(self)
