# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e + linear-gripper RGB (vision-distillation) task variants for box-assembly stages A/B/C.

Built directly on the generic RGB pipeline (``Ur5eRobotiq2f85RGBRelCartesianOSCEvalCfg``: front/side/
wrist TiledCameras + curtains, RGB obs groups, RGB terminations/render, and the fixed sysid + OSC-gain
events it inherits from ``FinetuneEvalEventCfg``), applied to the Stage-A/B/C box-assembly scenes via
the extracted ``_paper_stage_{box_center,object_in_box,cover_close}`` helpers. Three changes for the
adopted setup:

1. **Dynamics mirror FinetuneEval (not the stage-1 train action).** The experts distilled here are now
   the STAGE-2 FINETUNED policies, which converged under the stiff eval action + fixed sysid/OSC gains
   (see ``Ur10eLinearGripperRelCartesianOSCFinetuneEvalCfg``). The RGB env therefore reproduces that
   dynamics exactly: EXPLICIT UR10e-linear actuator + ``Ur10eLinearGripperRelativeOSCEvalAction``
   (Kp 1000/50, scale 0.01/0.01/0.002/0.02/0.02/0.2), the inherited fixed ``randomize_arm_sysid`` /
   ``randomize_osc_gains`` (scale_progress=1) KEPT, and the motor delay pinned to (0, 0). (Earlier RGB
   cfgs NULLED the sysid DR and forced the stage-1 TRAIN action because they distilled stage-1
   experts; that assumption no longer holds.)
2. **Robot swapped** to the UR10e + linear gripper via ``_apply_linear_gripper``. Stage pair/success/
   scene come from the extracted ``_paper_stage_*`` helpers, then the reset loader is repointed at the
   UR10e-linear reset datasets (``_repoint_ur10e_resets``; the 2F-85 loader defaults to the CLOUD
   12-joint 2F-85 reset states -> a [*,12] vs [*,8] shape mismatch on our 8-joint gripper).
3. **Wrist camera REBOUND, not dropped.** OmniReset is a 3-camera setup (front+side+wrist); we keep the
   wrist camera but repoint its prim + randomization templates onto the linear gripper's base link (the
   graft nests the gripper under ``/gripper``: .../Robot/gripper/robotiq_base_link/...). The 2F-85
   wrist-mount + inner-finger appearance DR is nulled (those mesh prims are absent on the linear
   gripper).

The expert STATE obs group is injected by ``collect_demos.py`` at collection time. The RGB student sees
front + side + wrist images + proprioception only.

NOTE (P6): front/side camera poses are UR5e-workspace-tuned (~0.85 m reach); the UR10e reaches ~1.3 m.
The rebound wrist-camera offset is the upstream 2F-85 number, flagged for a later visual retune.
"""

from __future__ import annotations

from isaaclab.utils import configclass

import uwlab_assets.robots.ur10e_linear_gripper as ur10e_linear_gripper

from .actions import Ur10eLinearGripperRelativeOSCEvalAction
from .data_collection_rgb_cfg import (
    DataCollectionRGBEventCfg,
    RGBEventCfg,
    Ur5eRobotiq2f85RGBRelCartesianOSCEvalCfg,
)
from .linear_gripper_cfg import _apply_linear_gripper
from .rl_state_cfg import _paper_stage_box_center, _paper_stage_cover_close, _paper_stage_object_in_box
from .ur10e_linear_gripper_cfg import _repoint_ur10e_resets

# The graft references the gripper under /ur10e/gripper (see conversions/graft_gripper_on_ur10e.py), so
# in the scene the base link is .../Robot/gripper/robotiq_base_link (the 2F-85 mounts it directly at
# .../Robot/robotiq_base_link). The randomize_tiled_cameras/focal events address the same prim via the
# flattened /World/envs/env_{} path template.
_WRIST_PRIM = "{ENV_REGEX_NS}/Robot/gripper/robotiq_base_link/rgb_wrist_camera"
_WRIST_TEMPLATE = "/World/envs/env_{}/Robot/gripper/robotiq_base_link/rgb_wrist_camera"
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
    cfg.scene.wrist_camera.prim_path = _WRIST_PRIM
    cfg.scene.wrist_camera.offset.pos = _WRIST_POS
    cfg.scene.wrist_camera.offset.rot = _WRIST_ROT
    cfg.events.randomize_wrist_camera.params["camera_path_template"] = _WRIST_TEMPLATE
    cfg.events.randomize_wrist_camera.params["base_position"] = _WRIST_POS
    cfg.events.randomize_wrist_camera.params["base_rotation"] = _WRIST_ROT
    cfg.events.randomize_wrist_camera_focal_length.params["camera_path_template"] = _WRIST_TEMPLATE
    cfg.events.randomize_wrist_mount_appearance = None
    cfg.events.randomize_inner_finger_appearance = None


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
