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
"""

from __future__ import annotations

import uwlab_assets.robots.ur10e_linear_gripper as ur10e_linear_gripper

from isaaclab.utils import configclass

from .actions import Ur10eLinearGripperRelativeOSCEvalAction, Ur10eLinearGripperSysidOSCAction
from .camera_align_cfg import CameraAlignEnvCfg
from .data_collection_rgb_cfg import (
    Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg,
    Ur5eRobotiq2f85EvalRGBRelCartesianOSCCfg,
)
from .linear_gripper_cfg import _apply_linear_gripper

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
