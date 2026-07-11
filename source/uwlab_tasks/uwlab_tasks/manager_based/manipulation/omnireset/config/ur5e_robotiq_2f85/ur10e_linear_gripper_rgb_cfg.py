# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e + linear-gripper RGB configs: camera alignment, generic RGB data collection / play,
and the box-assembly Stage-A/B/C RGB (vision-distillation) variants.

Mirrors the 2F-85 RGB stack (``camera_align_cfg.py`` + ``data_collection_rgb_cfg.py``) with
the UR10e arm + custom linear gripper, using the same subclass-and-swap pattern as
``ur10e_linear_gripper_cfg.py``. The 2F-85 arms share the ``robotiq_base_link`` /
``wrist_3_link`` link contract, so the RGB scene, three ``TiledCamera``s (front/side/wrist
parented to ``robotiq_base_link``), observations, terminations, and the
object/table/curtain/HDRI + camera-pose/focal randomization all carry over unchanged.

What is UR10e / linear-gripper specific here:
* robot + action swapped via ``_apply_linear_gripper`` (RGB collection/play) or directly
  (camera align, which has no grasp/reset events);
* the two 2F-85 gripper-appearance randomization terms are dropped -- their mesh paths
  (``robotiq_base_link/visuals/D415_to_Robotiq_Mount``, ``*_inner_finger/visuals/mesh_1``)
  do not exist on our linear gripper (its visuals are instanced prototypes with different
  names). The gripper still renders; only its per-mesh appearance DR is disabled for now;
* resets read the UR10e datasets (``./Datasets_ur10e/OmniReset``, CLI-overridable);
* motor delay pinned to the measured 0 (matches Finetune-Play deployment dynamics).

Camera pos/rot/focal are inherited from the 2F-85 configs as PLACEHOLDERS -- replace them
with the calibrated values from ``align_cameras.py`` (run against the CameraAlign env below)
before the real 80k collection. See ``UR10E_SIM2REAL_PROCEDURE.md`` §9.

Registered gym ids (mirroring the 2F-85 ones):
* ``OmniReset-UR10eLinearGripper-CameraAlign-v0``
* ``OmniReset-UR10eLinearGripper-RelCartesianOSC-RGB-DataCollection-v0``
* ``OmniReset-UR10eLinearGripper-RelCartesianOSC-RGB-Play-v0``

Box-assembly stage variants (bottom of the file) build on the same generic RGB pipeline via the
extracted ``_paper_stage_{box_center,object_in_box,cover_close}`` helpers, with two deltas:

1. **Dynamics mirror FinetuneEval (not the stage-1 train action).** The experts distilled there are
   the STAGE-2 FINETUNED policies, which converged under the stiff eval action + fixed sysid/OSC gains
   (see ``Ur10eLinearGripperRelCartesianOSCFinetuneEvalCfg``): EXPLICIT UR10e-linear actuator +
   ``Ur10eLinearGripperRelativeOSCEvalAction``, the inherited fixed ``randomize_arm_sysid`` /
   ``randomize_osc_gains`` (scale_progress=1) KEPT, and the motor delay pinned to (0, 0).
2. **Wrist camera REBOUND with a retuned mount** (``_rebind_wrist_camera``): the upstream D415 offset
   is calibrated in the 2F-85 base frame and looks into empty space on the graft; the retuned pose was
   render-verified to keep both jaw tips + the object/target seam in frame.

The expert STATE obs group is injected by ``collect_demos.py`` at collection time; the RGB student
sees front + side + wrist images + proprioception only. Their reset loader is repointed at the
UR10e-linear reset datasets (``_repoint_ur10e_resets``).

Stage gym ids: ``OmniReset-Ur10eLinearGripper-{BoxCenterPaper,ObjectInBoxPaper,CoverCloseRimPaper}-RGB-{DataCollection,Play}-v0``.

NOTE (P6): front/side camera poses are UR5e-workspace-tuned (~0.85 m reach); the UR10e reaches ~1.3 m.
"""

from __future__ import annotations

import uwlab_assets.robots.ur10e_linear_gripper as ur10e_linear_gripper

from isaaclab.utils import configclass

from .actions import Ur10eLinearGripperRelativeOSCEvalAction, Ur10eLinearGripperSysidOSCAction
from .camera_align_cfg import CameraAlignEnvCfg
from .data_collection_rgb_cfg import (
    DataCollectionRGBEventCfg,
    RGBEventCfg,
    Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg,
    Ur5eRobotiq2f85EvalRGBRelCartesianOSCCfg,
    Ur5eRobotiq2f85RGBRelCartesianOSCEvalCfg,
)
from .linear_gripper_cfg import _apply_linear_gripper
from .rl_state_cfg import _paper_stage_box_center, _paper_stage_cover_close, _paper_stage_object_in_box
from .ur10e_linear_gripper_cfg import _repoint_ur10e_resets

# Reset states are robot-specific; the RGB collection resets from the UR10e datasets.
_UR10E_RESET_DIR = "./Datasets_ur10e/OmniReset"

# 2F-85-specific per-mesh gripper-appearance DR terms that reference meshes absent on the
# linear gripper (its base/finger visuals are instanced prototypes with different names).
_ROBOTIQ_APPEARANCE_TERMS = ("randomize_wrist_mount_appearance", "randomize_inner_finger_appearance")

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
# ⚠ THESE ARE PLACEHOLDERS (inherited from the 2F-85 rig). Replace each with the value
# printed by ``align_cameras.py --robot ur10e --camera <front|side|wrist>_camera`` against a
# real D405 image, then rebuild nothing -- the cfgs read this dict at construction. Both the
# CameraAlign env (what you calibrate WITH) and the DataCollection/Play envs (what collects)
# use these, so one edit updates the whole pipeline. Real D405s: front 409122273078,
# side 323622272232, wrist 409122272284 (positional order in the diffusion_policy deploy).
# ---------------------------------------------------------------------------------------
_UR10E_CAMERA_POSES = {
    "front_camera": dict(
        pos=(1.0770121, -0.1679045, 0.4486344),
        rot=(0.70564552, 0.46613815, 0.25072644, 0.47107948),
        focal=13.20,
    ),
    "side_camera": dict(
        pos=(0.8323904, 0.5877843, 0.2805111),
        rot=(0.29008842, 0.22122445, 0.51336143, 0.77676798),
        focal=20.10,
    ),
    "wrist_camera": dict(
        pos=(0.0182505, -0.00408447, -0.0689107),
        rot=(0.34254336, -0.61819255, -0.6160212, 0.347879),
        focal=24.55,
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
        # Keep the average real-world placement the base set for the UR5e (workspace kept).
        self.scene.robot.init_state.pos = (0.0, -0.039, 0.0)
        self.actions = Ur10eLinearGripperSysidOSCAction()
        _fix_wrist_camera_path(self)
        _apply_camera_poses(self)


# ---------------------------------------------------------------------------------------
# RGB data collection / play
# ---------------------------------------------------------------------------------------
def _apply_ur10e_rgb(cfg) -> None:
    """Swap the 2F-85 robot/action for the UR10e linear gripper and fix RGB-specific bits."""
    # robot (IMPLICIT, matching the 2F-85 RGB pattern; measured delay is 0) + eval action.
    _apply_linear_gripper(
        cfg, ur10e_linear_gripper.IMPLICIT_UR10E_LINEAR_GRIPPER, Ur10eLinearGripperRelativeOSCEvalAction()
    )
    # Pin the measured motor delay (0) -- deployment dynamics (mirrors Finetune-Play).
    if getattr(cfg.events, "randomize_arm_sysid", None) is not None:
        cfg.events.randomize_arm_sysid.params["delay_range"] = (0, 0)
    # Reset from the UR10e datasets, not the cloud 2F-85 default (CLI-overridable).
    if getattr(cfg.events, "reset_from_reset_states", None) is not None:
        cfg.events.reset_from_reset_states.params["dataset_dir"] = _UR10E_RESET_DIR
    # Drop the 2F-85-specific per-mesh gripper-appearance DR (meshes absent on our gripper).
    for term in _ROBOTIQ_APPEARANCE_TERMS:
        if getattr(cfg.events, term, None) is not None:
            setattr(cfg.events, term, None)
    # Wrist camera prim path -> our nested gripper link (+ its DR event templates).
    _fix_wrist_camera_path(cfg)
    # Calibrated (currently placeholder) camera poses/focals.
    _apply_camera_poses(cfg)


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


# ---------------------------------------------------------------------------------------
# Box-assembly PAPER stage RGB variants (vision distillation of the stage-2 finetuned experts).
# See the module docstring: FinetuneEval dynamics + retuned wrist mount.
# ---------------------------------------------------------------------------------------
# Retuned wrist mount for the linear gripper (the upstream numbers are D415-bracket-calibrated in the
# 2F-85 base frame and look into empty space here). Linear-gripper base frame: +Z -> fingertips
# (9.4 cm ahead), jaws travel +/-X, +Y up at the rest pose. This pose sits 5.5 cm above / 2 cm ahead
# of the base looking down the approach axis with a ~25 deg pitch — both jaw tips, the grasped object
# and the workspace below stay in frame (visually verified on renders, 2026-07-08).
_WRIST_POS = (0.0, 0.055, 0.02)
_WRIST_ROT = (0.2164, -0.9763, 0.0, 0.0)


def _rebind_wrist_camera(cfg) -> None:
    """Rebind the inherited wrist camera onto the linear gripper's base link instead of dropping it.

    OmniReset is a 3-camera setup, so we keep ``scene.wrist_camera``, the ``wrist_rgb`` obs in both the
    policy + data_collection groups, the ``wrist_camera`` entry in the corrupted-camera termination, and
    the ``randomize_wrist_camera``(+focal) events -- repointing the prim path / path templates at the
    grafted gripper base and installing the retuned mount pose (``_WRIST_POS``/``_WRIST_ROT``) both on
    the sensor cfg and as the DR event's base pose. Still NULL the 2F-85 wrist-mount + inner-finger
    appearance DR: those target mesh prims (robotiq_base_link/visuals/D415_to_Robotiq_Mount,
    left/right_inner_finger/visuals/mesh_1) are absent on the linear gripper -> "No prims found
    matching"."""
    _fix_wrist_camera_path(cfg)
    cfg.scene.wrist_camera.offset.pos = _WRIST_POS
    cfg.scene.wrist_camera.offset.rot = _WRIST_ROT
    cfg.events.randomize_wrist_camera.params["base_position"] = _WRIST_POS
    cfg.events.randomize_wrist_camera.params["base_rotation"] = _WRIST_ROT
    for term in _ROBOTIQ_APPEARANCE_TERMS:
        setattr(cfg.events, term, None)


def _to_ur10e_linear(cfg) -> None:
    """Swap robot+action to the UR10e linear gripper with FinetuneEval dynamics, and rebind the wrist
    camera. Call BEFORE the ``_paper_stage_*`` helper + ``_repoint_ur10e_resets`` (which need the pair).

    Mirrors ``Ur10eLinearGripperRelCartesianOSCFinetuneEvalCfg``: EXPLICIT actuator + eval action, and
    the inherited fixed sysid pinned to the measured residual motor delay (0, 0). The RGB events
    (``RGBEventCfg`` / ``DataCollectionRGBEventCfg``) subclass ``FinetuneEvalEventCfg``, so the fixed
    ``randomize_arm_sysid`` / ``randomize_osc_gains`` (scale_progress=1) are already present and kept."""
    _apply_linear_gripper(
        cfg, ur10e_linear_gripper.EXPLICIT_UR10E_LINEAR_GRIPPER, Ur10eLinearGripperRelativeOSCEvalAction()
    )
    # Pin the motor delay to the measured residual (0 steps @ this env's rate), matching the FinetuneEval
    # cfg (the inherited FinetuneEvalEventCfg draws (0, 1)). UR10e-only override.
    cfg.events.randomize_arm_sysid.params["delay_range"] = (0, 0)
    _rebind_wrist_camera(cfg)


# ---------------------------------------------------------------------------------------
# Stage A (BoxCenterPaper): box -> table-center target
# ---------------------------------------------------------------------------------------
@configclass
class Ur10eLinearGripperBoxCenterPaperRGBEvalCfg(Ur5eRobotiq2f85RGBRelCartesianOSCEvalCfg):
    """Stage-A RGB Play/Eval (UR10e + linear gripper, front+side+wrist cameras)."""

    events: RGBEventCfg = RGBEventCfg()

    def __post_init__(self):
        super().__post_init__()  # generic RGB scene/obs/terminations/render + FinetuneEval fixed sysid
        _to_ur10e_linear(self)  # EXPLICIT robot + eval action + delay pin + wrist rebind
        _paper_stage_box_center(self)  # box -> table-center pair + canonical-handoff yaw gate
        _repoint_ur10e_resets(self)


@configclass
class Ur10eLinearGripperBoxCenterPaperRGBDataCollectionCfg(Ur10eLinearGripperBoxCenterPaperRGBEvalCfg):
    """Stage-A RGB data-collection (all 4 reset types, 0.25 each)."""

    events: DataCollectionRGBEventCfg = DataCollectionRGBEventCfg()


# ---------------------------------------------------------------------------------------
# Stage B (ObjectInBoxPaper): object -> box cavity
# ---------------------------------------------------------------------------------------
@configclass
class Ur10eLinearGripperObjectInBoxPaperRGBEvalCfg(Ur5eRobotiq2f85RGBRelCartesianOSCEvalCfg):
    """Stage-B RGB Play/Eval (UR10e + linear gripper, front+side+wrist cameras)."""

    events: RGBEventCfg = RGBEventCfg()

    def __post_init__(self):
        super().__post_init__()  # generic RGB scene/obs/terminations/render + FinetuneEval fixed sysid
        _to_ur10e_linear(self)  # EXPLICIT robot + eval action + delay pin + wrist rebind
        _paper_stage_object_in_box(self)  # object -> box cavity pair (default success, no yaw gate)
        _repoint_ur10e_resets(self)


@configclass
class Ur10eLinearGripperObjectInBoxPaperRGBDataCollectionCfg(Ur10eLinearGripperObjectInBoxPaperRGBEvalCfg):
    """Stage-B RGB data-collection (all 4 reset types)."""

    events: DataCollectionRGBEventCfg = DataCollectionRGBEventCfg()


# ---------------------------------------------------------------------------------------
# Stage C (CoverCloseRimPaper): caprim edge-rim cover -> box (object inside)
# ---------------------------------------------------------------------------------------
@configclass
class Ur10eLinearGripperCoverCloseRimPaperRGBEvalCfg(Ur5eRobotiq2f85RGBRelCartesianOSCEvalCfg):
    """Stage-C RGB Play/Eval (UR10e + linear gripper, front+side+wrist cameras). Built from the generic
    RGB eval + the Stage-C caprim pair/augment (object restored inside the box)."""

    events: RGBEventCfg = RGBEventCfg()

    def __post_init__(self):
        super().__post_init__()  # generic RGB scene/obs/terminations/render + FinetuneEval fixed sysid
        _to_ur10e_linear(self)  # EXPLICIT robot + eval action + delay pin + wrist rebind
        _paper_stage_cover_close(self)  # caprim -> box pair + object-inside-box augment
        _repoint_ur10e_resets(self)


@configclass
class Ur10eLinearGripperCoverCloseRimPaperRGBDataCollectionCfg(Ur10eLinearGripperCoverCloseRimPaperRGBEvalCfg):
    """Stage-C RGB data-collection (all 4 reset types)."""

    events: DataCollectionRGBEventCfg = DataCollectionRGBEventCfg()
