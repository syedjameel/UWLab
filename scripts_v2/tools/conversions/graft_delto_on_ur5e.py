# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Graft the Tesollo DELTO DG-5F hand onto the calibrated UR5e arm.

This is the UR5e sibling of ``graft_gripper_on_ur10e.py`` and the DELTO sibling of
``graft_gripper_on_ur5e.py``. It takes the arm half from the latter and the hand half from the
former:

  * ARM half (from ``graft_gripper_on_ur5e.py``, unchanged): the input is the CLOUD
    ``ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd``, resolved/downloaded through
    ``resolve_cloud_path``. That asset -- not a bare UR5e URDF conversion -- is what carries the
    UR5e sysid and FK calibration, so it is stripped of its 2F-85 rather than replaced. The 9
    2F-85 bodies + 10 2F-85 joints go (the 2F-85's own mount FixedJoint is nested under
    ``robotiq_base_link``, so it goes with the body); the 6 arm links/joints and the articulation
    root ``root_joint`` are left untouched.
  * HAND half (from ``graft_gripper_on_ur10e.py``): the hand's base link is ``rl_dg_mount``, which
    bolts directly to the UR flange, so the mount standoff resolves to 0 rather than the linear
    gripper's 49 mm adapter plate. The hand's own free root joint is stripped, its nested
    ``ArticulationRootAPI`` is removed so it joins the arm's single articulation, its massless
    ``rl_dg_*_tip`` frame bodies are seeded with a small mass, and the subtree is re-placed so that
    ``rl_dg_mount`` -- not the referenced subtree's root prim -- lands exactly on the flange.

The referenced hand is nested at ``{ROOT}/gripper``. That name is NOT cosmetic: dexlift's
``HAND_PRIM`` constant is literally ``"gripper"`` and the fingertip ``ContactSensorCfg`` prim paths
address it. Renaming it breaks those sensors SILENTLY -- they resolve to nothing and simply never
fire -- so the prim name is fixed here rather than exposed as a flag.

THE ASSET IS THE USD **AND** ITS ``metadata.yaml``. ``read_metadata_from_usd_directory`` keys off
the USD's own directory and hard-``open()``s the file, so a USD written without one is a
``FileNotFoundError`` at env construction, not a default. This script therefore emits the sibling
``metadata.yaml`` as part of the graft, COMPOSED from the two source files rather than copied from
either -- see :func:`write_metadata`. The wrong-constant trap it exists to close is that the arm's
own metadata (the only UR5e one) carries the 2F-85's grasp block expressed in ``robotiq_base_link``,
a prim this script DELETES; copying that file wholesale gives the five-finger hand a two-jaw TCP and
a ``maximum_aperture``.

WHAT IS DELIBERATELY *NOT* CARRIED OVER FROM THE TWO SOURCE SCRIPTS

  * The dual-drive / mimic-strip block (``graft_gripper_on_ur5e.py`` step 3b and its UR10e
    ``--dual-drive-joint`` / ``--strip-mimic-joint`` flags). That block exists because the linear
    gripper's PRISMATIC PhysX mimic goes inert once embedded in the arm articulation while still
    capturing the follower joint's control. The DELTO has 20 independently actuated revolute joints
    and no mimic at all, so re-activating a "follower" drive and deleting "inert" mimic properties
    would be operating on machinery that does not exist here.
  * The ``--gripper-mass`` rescale. The linear gripper is rescaled from its ~1.1 kg URDF total to
    the 0.575 kg measured assembly. The DELTO's masses are AUTHORED by ``prepare_delto_hand.py``
    from the vendor model and are verified below against that authored total; there is no
    independent measurement here to rescale toward.
  * The UR10e graft's wrist-limit clamp and its ``base``/``base_link``/``flange``/``tool0``
    zero-mass hygiene fixes. Both are properties of the locally URDF-converted UR10e. The
    calibrated UR5e cloud asset has no such massless frame links (they were merged away), and its
    joint limits are part of the calibration this script exists to preserve.

Pure-USD edit (``pxr`` only -- no Isaac app, no GPU). It self-verifies the written asset and exits
non-zero on any mismatch. Run::

    python scripts_v2/tools/conversions/graft_delto_on_ur5e.py
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys

import yaml
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR, resolve_cloud_path

# The calibrated UR5e+2F-85 arm USD is a CLOUD asset. resolve_cloud_path downloads it once to
# ~/.cache/uwlab/assets/... and returns the local path (so this works on a fresh cache, not only
# where the 2F-85 tasks were already run). Same URL the 2F-85 robot cfg spawns from.
_ARM_USD_URL = (
    f"{UWLAB_CLOUD_ASSETS_DIR}/Robots/UniversalRobots/Ur5e2f85RobotiqGripperCalibrated/"
    "ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd"
)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ROOT = "/ur5e_robotiq_gripper_d415_mount"
F85_BODIES = [
    "robotiq_base_link", "left_outer_knuckle", "left_outer_finger", "left_inner_finger",
    "left_inner_knuckle", "right_outer_knuckle", "right_outer_finger", "right_inner_finger",
    "right_inner_knuckle",
]
F85_JOINTS = [
    "finger_joint", "right_outer_knuckle_joint", "right_inner_finger_joint",
    "right_inner_knuckle_joint", "right_inner_finger_knuckle_joint",
    "left_inner_finger_knuckle_joint", "left_inner_finger_joint", "left_inner_knuckle_joint",
    "left_outer_finger_joint", "right_outer_finger_joint",
]

# The prim the hand is referenced under. Fixed, not a flag -- see the module docstring.
HAND_PRIM = "gripper"

# The 6 UR5e revolute DOFs that must survive the strip untouched.
_ARM_JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
_FINGERS = (1, 2, 3, 4, 5)
# The 20 actuated DELTO revolutes: four phalanx joints on each of five fingers.
_DELTO_JOINT_RE = re.compile(r"^rj_dg_[1-5]_[1-4]$")
_DELTO_JOINTS = tuple(f"rj_dg_{finger}_{phalanx}" for finger in _FINGERS for phalanx in (1, 2, 3, 4))
_DELTO_TIP_LINKS = tuple(f"rl_dg_{finger}_tip" for finger in _FINGERS)
# Every rigid body of the prepared hand, in the order prepare_delto_hand.py authors them.
_DELTO_LINKS = ("rl_dg_mount", "rl_dg_base", "rl_dg_palm") + tuple(
    f"rl_dg_{finger}_{part}" for finger in _FINGERS for part in ("1", "2", "3", "4", "tip")
)

# The AUTHORED total explicit mass of the prepared DELTO hand (prepare_delto_hand.py's
# _EXPECTED_TOTAL_MASS). The DG-5F spec sheet says 1.4 kg; the vendor USD's per-link inertials sum
# to 1.7735 kg. That discrepancy is KNOWN and RECORDED -- it is not silently rescaled away here,
# because the number the dynamics actually see must stay traceable to the asset it came from.
_EXPECTED_HAND_MASS = 1.7735
_MASS_TOL = 1e-4

# --- metadata.yaml composition -------------------------------------------------------------------
# The ARM identification, taken from the arm USD's OWN metadata.yaml so it cannot drift from the
# asset it identifies. Never re-typed here.
_ARM_METADATA_KEYS = ("sysid", "calibrated_joints", "link_inertials")
# The end-effector keys the FULL-ROBOT file must carry, taken from the reference DELTO robot file.
# gripper_offset is the TCP in the rl_dg_mount frame -- a property of the HAND and its mount, not of
# the arm, and this graft bolts rl_dg_mount to the flange with the same zero standoff and identity
# rotation the UR10e graft uses, so the value transfers unchanged. gripper_approach_direction is
# duplicated from the hand's file for terminations.py, which resolves it from the ROBOT's directory.
_DELTO_ROBOT_KEYS = ("gripper_offset", "gripper_approach_direction")
# 2F-85 grasp constants that must NEVER reach the output. They are expressed in robotiq_base_link --
# a prim this script deletes -- and describe a two-jaw gripper: a maximum_aperture and a single
# finger_open_joint_angle are meaningless for a five-finger hand, whose equivalents are a 59 mm
# opposition limit and a 20-entry posture in Robots/DeltoHand/metadata.yaml.
_F85_METADATA_KEYS = (
    "finger_offset", "finger_clearance", "maximum_aperture", "grasp_align_axis",
    "orientation_sample_axis", "finger_open_joint_angle",
)
_METADATA_HEADER = """\
# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# UR5e + Tesollo DELTO DG-5F metadata (FULL ROBOT).
#
# GENERATED by scripts_v2/tools/conversions/graft_delto_on_ur5e.py alongside ur5e_delto.usd. Do not
# hand-edit: re-run the graft. read_metadata_from_usd_directory keys off the USD's own directory and
# hard-open()s this file, so the USD is not a usable asset without it.
#
# FIELD SPLIT -- read this before adding anything here. The GRASP-SAMPLING fields
# (maximum_aperture, grasp_align_axis, grasp_center_offset, grasp_center_clearance,
# orientation_sample_axis, grasp_sample_mode, grasp_topdown_*, finger_open_joint_angles,
# finger_offset, finger_clearance) live in Robots/DeltoHand/metadata.yaml, because the grasp env's
# scene.robot is the HAND-ONLY articulation and events.py reads that directory. This file carries
# gripper_offset -- which rl_state_cfg.py binds to a body of the FULL robot -- plus the arm
# identification. Same split as Ur10eDelto/metadata.yaml.
#
# THE 2F-85 GRASP BLOCK IS DELIBERATELY ABSENT. The only other UR5e metadata.yaml is the calibrated
# cloud arm's, and it carries gripper_offset pos [0.1345, 0, 0] quat [0.5, 0.5, 0.5, 0.5],
# finger_offset 0.1345, finger_clearance 0.06, maximum_aperture 0.09, gripper_approach_direction
# [1, 0, 0], grasp_align_axis [1, 0, 0], finger_open_joint_angle 0.0 -- two-jaw constants expressed
# in the robotiq_base_link frame, a prim the graft DELETES. Copying that file wholesale onto this
# robot silently gives a five-finger hand a 2F-85 TCP and a parallel-jaw aperture. The graft
# asserts none of those keys reached this file.
#
# 1. sysid / calibrated_joints / link_inertials are the ARM, copied by the graft from the arm USD's
#    own metadata.yaml (the calibrated cloud Ur5e2f85RobotiqGripperCalibrated asset). There is no
#    inheritance, so the same numbers are repeated in every robot directory mounting this arm; never
#    re-derive them here, re-run the graft. kinematics.py's analytical OSC reads link_inertials.
#    CAVEAT, and it is the UR5e equivalent of the one Ur10eDelto/metadata.yaml records: that
#    identification ran with the 2F-85 + D415 wrist assembly mounted. The DELTO hand is 1.7735 kg
#    with a different COM, so the payload the sysid saw is not the payload this robot carries. The
#    graft preserves the arm's FK calibration exactly (the 6 links/joints and root_joint are never
#    touched); the DYNAMIC fit is applied to a different end effector. Unavoidable for any new end
#    effector, but it is a known, recorded approximation -- not a validated number.
#
# 2. HAND MASS is 1.7735 kg, the sum of the 28 explicit per-body masses of the prepared hand, and is
#    verified by the graft rather than rescaled. That is 0.3735 kg over the 1.4 kg Tesollo DG-5F
#    specification -- a known model/spec discrepancy. Do not silently rescale it.
#
# 3. The wrist camera has no bracket geometry on this robot. The D415 mount (adapter,
#    D415_to_Robotiq_Mount, d415_and_cable) lived under robotiq_base_link and went with the 2F-85
#    strip; the DELTO graft adds no replacement. Anything siting a wrist camera must re-site it on
#    rl_dg_mount.

# TCP: the grasp centre of the virtual two-jaw, in the rl_dg_mount frame. rl_state_cfg.py binds this
# via SceneEntityCfg("robot", body_names=...) -- for this robot that body is rl_dg_mount, NOT
# robotiq_base_link, and downstream env cfgs must be repointed accordingly. This is the
# thumb/finger-2 visible-tip pinch frame, not a centroid over all five fingers. quat is (w, x, y, z),
# with +Z along approach and +X along the closing axis. Must match DeltoHand/metadata.yaml's
# grasp_center_offset (the graft asserts it) and moves whenever the OPEN/CLOSED posture pair moves.
#
# gripper_approach_direction is DUPLICATED from Robots/DeltoHand/metadata.yaml, the one exception to
# the field split above. terminations.py's check_reset_state_success reads it from the ROBOT's USD
# directory with metadata.get() and no default; without it tuple(None) raises inside
# TerminationManager._prepare_terms and every reset-state env id dies at construction.
"""


def _csv(value: str) -> list[str]:
    """Parse a comma-separated flag value into a list, treating "" / "none" as empty."""
    if not value or value.strip().lower() == "none":
        return []
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _read_yaml(path: str, what: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"{what} metadata.yaml not found: {path}")
    with open(path) as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"{what} metadata.yaml is not a mapping: {path}")
    return data


def write_metadata(output_usd: str, arm_usd: str, hand_usd: str, reference_robot_metadata: str) -> str:
    """Emit the output asset's sibling ``metadata.yaml``, COMPOSED from its two real sources.

    A USD without this file is not a usable asset: ``read_metadata_from_usd_directory`` keys off the
    USD's own directory and hard-``open()``s ``metadata.yaml``, so the miss is a ``FileNotFoundError``
    at env construction, and the robot would in any case have no sysid, no calibrated joints, no
    link inertials for the analytical OSC and no TCP.

    It is composed, not copied, because the two candidate files are each half wrong for this robot:

      * the ARM's own file (the calibrated cloud UR5e+2F-85 asset) is the only UR5e metadata that
        exists, and it is the one a copy would reach for -- but its whole grasp block is 2F-85
        geometry expressed in ``robotiq_base_link``, the prim this graft deletes. Only
        :data:`_ARM_METADATA_KEYS` is taken from it.
      * the DELTO reference robot file (``Ur10eDelto/metadata.yaml``) has the right end-effector
        keys -- ``gripper_offset`` is the TCP in the ``rl_dg_mount`` frame, and this graft bolts
        ``rl_dg_mount`` to the flange with the same identity rotation and zero standoff, so the
        value is the same hand on the same mount -- but its arm half is a UR10e.

    The hand's own file is read only to CROSS-CHECK the transplanted end-effector keys against the
    articulation the grasp datasets were sampled on, so the two DELTO files cannot drift apart
    silently. Returns the written path.
    """
    arm_meta = _read_yaml(os.path.join(os.path.dirname(arm_usd), "metadata.yaml"), "arm")
    hand_meta = _read_yaml(os.path.join(os.path.dirname(hand_usd), "metadata.yaml"), "hand")
    ref_meta = _read_yaml(reference_robot_metadata, "reference DELTO robot")

    metadata: dict = {}
    for key in _DELTO_ROBOT_KEYS:
        if key not in ref_meta:
            raise SystemExit(f"reference DELTO robot metadata has no '{key}': {reference_robot_metadata}")
        metadata[key] = ref_meta[key]
    for key in _ARM_METADATA_KEYS:
        if key not in arm_meta:
            raise SystemExit(f"arm metadata has no '{key}': {os.path.dirname(arm_usd)}/metadata.yaml")
        metadata[key] = arm_meta[key]

    # Cross-check the transplanted end-effector keys against the HAND's own file. Both DELTO robot
    # files and the hand file describe one grasp frame; a mismatch means the posture moved in one
    # place only, and every reset/reward/observation that reads gripper_offset would be off.
    tcp = list(metadata["gripper_offset"]["pos"])
    hand_tcp = list(hand_meta.get("grasp_center_offset", []))
    if hand_tcp and [round(v, 9) for v in tcp] != [round(v, 9) for v in hand_tcp]:
        raise SystemExit(f"gripper_offset.pos {tcp} does not match the hand's grasp_center_offset "
                         f"{hand_tcp} -- the OPEN/CLOSED posture moved in one file only")
    approach = [round(float(v), 9) for v in metadata["gripper_approach_direction"]]
    hand_approach = [round(float(v), 9) for v in hand_meta.get("gripper_approach_direction", [])]
    if hand_approach and approach != hand_approach:
        raise SystemExit(f"gripper_approach_direction {approach} does not match the hand's "
                         f"{hand_approach} -- terminations.py and events.py would disagree")

    # The trap this function exists to close: none of the 2F-85's two-jaw constants may survive, and
    # the TCP must not be the 2F-85's robotiq_base_link one.
    leaked = sorted(k for k in _F85_METADATA_KEYS if k in metadata)
    if leaked:
        raise SystemExit(f"2F-85 grasp constants leaked into the DELTO robot metadata: {leaked}")
    if arm_meta.get("gripper_offset") == metadata["gripper_offset"]:
        raise SystemExit("gripper_offset equals the 2F-85 arm asset's -- that TCP is expressed in "
                         "robotiq_base_link, a prim this graft deletes")

    path = os.path.join(os.path.dirname(output_usd), "metadata.yaml")
    with open(path, "w") as handle:
        handle.write(_METADATA_HEADER)
        yaml.safe_dump(metadata, handle, sort_keys=False, default_flow_style=None, width=100)
    print(f"Wrote {path}")
    print(f"  arm identification {list(_ARM_METADATA_KEYS)} <- {os.path.dirname(arm_usd)}/metadata.yaml")
    print(f"  end effector {list(_DELTO_ROBOT_KEYS)} <- {reference_robot_metadata}")
    print(f"  2F-85 grasp block withheld: {list(_F85_METADATA_KEYS)}")
    return path


def verify(output: str, base_link: str, standoff: float, root_joint_names: list[str]) -> list[str]:
    """Re-open the written asset and check every property the consumers depend on."""
    failures: list[str] = []
    stage = Usd.Stage.Open(output)
    if not stage:
        return [f"cannot re-open the written USD: {output}"]
    # Instance proxies must be expanded or instanced subtrees read as empty -- the exact blind spot
    # that hides missing tip geometry (see prepare_delto_hand.py).
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    prims = list(Usd.PrimRange.Stage(stage, predicate))
    gpath = f"{ROOT}/{HAND_PRIM}"

    # 1) the hand must be nested under the exact prim name dexlift's contact sensors address.
    if not stage.GetPrimAtPath(gpath):
        failures.append(f"hand prim {gpath} missing -- dexlift's HAND_PRIM/ContactSensorCfg paths "
                        f"would resolve to nothing and the fingertip sensors would never fire")

    # 2) 26 actuated joints: the 6 UR5e revolutes + the DELTO's 20 rj_dg_[1-5]_[1-4].
    revolutes = sorted(p.GetName() for p in prims if p.GetTypeName() == "PhysicsRevoluteJoint")
    expected_revolutes = sorted(_ARM_JOINTS + _DELTO_JOINTS)
    if revolutes != expected_revolutes:
        missing = sorted(set(expected_revolutes) - set(revolutes))
        extra = sorted(set(revolutes) - set(expected_revolutes))
        failures.append(f"expected {len(expected_revolutes)} revolute joints (6 arm + 20 hand), "
                        f"found {len(revolutes)}; missing={missing} unexpected={extra}")
    hand_revolutes = [n for n in revolutes if _DELTO_JOINT_RE.match(n)]
    if len(hand_revolutes) != len(_DELTO_JOINTS):
        failures.append(f"{len(hand_revolutes)} joints match rj_dg_[1-5]_[1-4], expected {len(_DELTO_JOINTS)}")

    # 3) exactly ONE articulation root, still the arm's own root_joint. A surviving nested root on
    #    the hand would split the robot into two articulations that the solver couples only through
    #    the mount joint -- the arm would appear to work while the hand went limp.
    roots = sorted(p.GetPath().pathString for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI))
    expected_root = f"{ROOT}/root_joint"
    if roots != [expected_root]:
        failures.append(f"expected exactly one ArticulationRootAPI on {expected_root}, found {roots}")
    nested = [r for r in roots if r.startswith(gpath + "/") or r == gpath]
    if nested:
        failures.append(f"nested ArticulationRootAPI survived inside the hand subtree: {nested}")

    # 4) the hand's AUTHORED explicit mass, unrescaled. See _EXPECTED_HAND_MASS.
    hand_bodies = {
        p.GetPath().pathString: p
        for p in prims
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetPath().pathString.startswith(gpath + "/")
    }
    hand_mass = sum(UsdPhysics.MassAPI(p).GetMassAttr().Get() or 0.0 for p in hand_bodies.values())
    if abs(hand_mass - _EXPECTED_HAND_MASS) > _MASS_TOL:
        failures.append(f"hand explicit mass {hand_mass:.6f} kg, expected the AUTHORED "
                        f"{_EXPECTED_HAND_MASS} kg (+/-{_MASS_TOL}) -- do NOT 'fix' this by "
                        f"rescaling to the 1.4 kg spec-sheet figure; the discrepancy is recorded")

    # 5) all five fingertip frame bodies present under the hand prim (the contact-sensor targets).
    for name in _DELTO_TIP_LINKS:
        tip = stage.GetPrimAtPath(f"{gpath}/{name}")
        if not tip or not tip.IsValid():
            failures.append(f"missing fingertip body {gpath}/{name}")

    # 6) the mount FixedJoint really connects the arm wrist to the hand base.
    mount_joint = stage.GetPrimAtPath(f"{gpath}/{base_link}/MountJoint")
    if not mount_joint or not mount_joint.IsValid():
        failures.append(f"mount FixedJoint not found at {gpath}/{base_link}/MountJoint")
    else:
        body0 = [t.pathString for t in mount_joint.GetRelationship("physics:body0").GetTargets()]
        body1 = [t.pathString for t in mount_joint.GetRelationship("physics:body1").GetTargets()]
        if body0 != [f"{ROOT}/wrist_3_link"]:
            failures.append(f"MountJoint body0 {body0}, expected [{ROOT}/wrist_3_link]")
        if body1 != [f"{gpath}/{base_link}"]:
            failures.append(f"MountJoint body1 {body1}, expected [{gpath}/{base_link}]")

    # 7) no 2F-85 residue. A leftover jaw body is an invisible extra collider bolted to the wrist.
    residue = []
    for prim in prims:
        name = prim.GetName()
        if name in F85_BODIES and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            residue.append(prim.GetPath().pathString)
        elif name in F85_JOINTS and "Joint" in str(prim.GetTypeName()):
            residue.append(prim.GetPath().pathString)
    if residue:
        failures.append(f"2F-85 residue survived the strip: {sorted(residue)}")

    # 8) NO SECOND ANCHOR INSIDE THE HAND. The hand's own root joint must be gone, and more
    #    generally no joint in the hand subtree may be anchored to WORLD (an unset/empty body
    #    relationship). PhysX reads a world-anchored joint inside an articulation as a second root:
    #    the robot is unusable, and nothing else here would notice -- such a joint is Fixed, not
    #    revolute (so check 2 passes), and carries no ArticulationRootAPI (so check 3 passes).
    #    The strip itself CANNOT be trusted to report this: the joint arrives through the
    #    {ROOT}/gripper reference, where UsdStage.RemovePrim has no spec to remove in the root layer
    #    and silently returns False. This check is the one that actually fires.
    for name in root_joint_names:
        for parent in (gpath, f"{gpath}/joints"):
            leftover = stage.GetPrimAtPath(f"{parent}/{name}")
            if leftover and leftover.IsValid():
                failures.append(f"the hand's own root joint survived at {parent}/{name} -- it would "
                                f"anchor the hand to world inside the arm's articulation")
    world_anchored = []
    for prim in prims:
        path = prim.GetPath().pathString
        if not path.startswith(gpath + "/") or "Joint" not in str(prim.GetTypeName()):
            continue
        for rel_name in ("physics:body0", "physics:body1"):
            rel = prim.GetRelationship(rel_name)
            if not rel or not rel.GetTargets():
                world_anchored.append(f"{path}[{rel_name}]")
    if world_anchored:
        failures.append(f"world-anchored joint(s) inside the hand subtree (a second articulation "
                        f"anchor): {sorted(world_anchored)}")

    # 9) GEOMETRY. The mount FixedJoint asserts the hand base sits exactly at (0, 0, standoff) in
    #    wrist_3 with identity rotation. If the subtree is actually placed anywhere else, the solver
    #    yanks the hand at the first step and every frame derived from this transform (the wrist
    #    camera extrinsic) inherits the error. This is the failure mode the placement code calls
    #    catastrophic, and it is the whole risk surface of the graft, so it is checked here against
    #    the WRITTEN file rather than trusted from the code that authored it.
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    wrist_prim = stage.GetPrimAtPath(f"{ROOT}/wrist_3_link")
    base_prim = stage.GetPrimAtPath(f"{gpath}/{base_link}")
    if wrist_prim and wrist_prim.IsValid() and base_prim and base_prim.IsValid():
        expected = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, 0.0, standoff)) * xc.GetLocalToWorldTransform(wrist_prim)
        actual = xc.GetLocalToWorldTransform(base_prim)
        d_pos = (actual.ExtractTranslation() - expected.ExtractTranslation()).GetLength()
        # Rotational error as the half-angle of the relative quaternion expected^-1 * actual.
        rel_rot = (expected.GetInverse() * actual).ExtractRotationQuat().GetNormalized()
        d_ang = 2.0 * math.acos(min(1.0, abs(rel_rot.GetReal())))
        if d_pos > 1e-6 or d_ang > 1e-6:
            failures.append(f"{base_link} is {d_pos * 1e6:.3f} um / {math.degrees(d_ang):.6f} deg off the "
                            f"pose the MountJoint asserts ((0, 0, {standoff}) in wrist_3, identity "
                            f"rotation) -- the solver would yank the hand at the first step")
        print(f"  verify: {base_link} placement error {d_pos * 1e6:.4f} um / {math.degrees(d_ang):.8f} deg")
    else:
        failures.append(f"cannot check placement: wrist_3_link or {gpath}/{base_link} missing")

    # 10) the sibling metadata.yaml. The USD alone is not the asset (see write_metadata): a missing
    #     file is a FileNotFoundError at env construction, and a file carrying the 2F-85 grasp block
    #     silently gives this five-finger hand a two-jaw TCP.
    meta_path = os.path.join(os.path.dirname(output), "metadata.yaml")
    if not os.path.exists(meta_path):
        failures.append(f"no metadata.yaml beside the USD ({meta_path}) -- "
                        f"read_metadata_from_usd_directory hard-open()s it at env construction")
    else:
        with open(meta_path) as handle:
            meta = yaml.safe_load(handle) or {}
        for key in _DELTO_ROBOT_KEYS + _ARM_METADATA_KEYS:
            if key not in meta:
                failures.append(f"metadata.yaml is missing '{key}'")
        leaked = sorted(k for k in _F85_METADATA_KEYS if k in meta)
        if leaked:
            failures.append(f"metadata.yaml carries 2F-85 two-jaw constants {leaked} -- they are "
                            f"expressed in robotiq_base_link, which this graft deletes")

    print(f"  verify: {len(hand_bodies)} hand bodies, {len(revolutes)} revolute joints "
          f"({len(hand_revolutes)} hand), hand mass {hand_mass:.6f} kg, articulation root {roots}")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description="Graft the Tesollo DELTO DG-5F hand onto the UR5e arm.")
    ap.add_argument("--arm-usd", default=None,
                    help="Calibrated UR5e+2F-85 USD (input). Default: resolve/download from the cloud.")
    ap.add_argument("--gripper-usd",
                    default=os.path.join(_REPO, "source/uwlab_assets/uwlab_assets/local/Robots/DeltoHand/delto_hand.usd"),
                    help="Prepared DELTO hand USD, the output of prepare_delto_hand.py.")
    ap.add_argument("--output",
                    default=os.path.join(_REPO, "source/uwlab_assets/uwlab_assets/local/Robots/Ur5eDelto/ur5e_delto.usd"))
    ap.add_argument("--reference-robot-metadata",
                    default=os.path.join(_REPO, "source/uwlab_assets/uwlab_assets/local/Robots/Ur10eDelto/metadata.yaml"),
                    help="Existing DELTO FULL-ROBOT metadata.yaml to take the end-effector keys "
                         "from (gripper_offset, gripper_approach_direction). Same hand on the same "
                         "rl_dg_mount at the same zero standoff, so those values transfer; the arm "
                         "half of that file is a UR10e and is NOT read. Cross-checked against "
                         "Robots/DeltoHand/metadata.yaml.")
    ap.add_argument(
        "--standoff",
        type=float,
        default=None,
        help="Mount offset along wrist_3 +Z (m). Default: 0 for rl_dg_mount (the DELTO bolts "
        "directly to the flange, whose frame is coincident with wrist_3_link), 0.049 for other "
        "grippers using the adapter plate. An explicit value always overrides the inference.",
    )
    ap.add_argument("--gripper-root-prim", default="",
                    help="Prim path to reference out of --gripper-usd (e.g. /DeltoHand). Empty (the "
                         "default) references the file's defaultPrim, which prepare_delto_hand.py "
                         "sets; a hand authored without a defaultPrim needs this.")
    ap.add_argument("--gripper-base-link", default="rl_dg_mount",
                    help="Link inside the hand that bolts to the wrist: the nested "
                         "ArticulationRootAPI is stripped from it and the mount FixedJoint "
                         "attaches to it.")
    ap.add_argument("--strip-gripper-root-joint", default="root_joint,rootJoint",
                    help="Comma-separated names of joints-to-world inside the hand to delete. The "
                         "DELTO source is fixed to world by a free 6-DOF root joint that would "
                         "fight the arm's articulation; the two spellings seen in the vendor and "
                         "prepared assets are both tried. Whichever prepared-hand script removes it "
                         "upstream, this stays a safety net -- and one that must actually work, so "
                         "it is enforced in verify() rather than trusted from a log line. Pass '' "
                         "to skip.")
    ap.add_argument("--gripper-frame-links", default=",".join(_DELTO_TIP_LINKS),
                    help="Comma-separated hand links that may be massless, shapeless frames (the "
                         "DELTO's rl_dg_*_tip bodies). Any of them with no authored mass is given "
                         "--gripper-frame-mass and a small isotropic inertia; links that already "
                         "carry an authored mass are LEFT ALONE, so the hand total stays the "
                         "authored one that verify() checks.")
    ap.add_argument("--gripper-frame-mass", type=float, default=0.01,
                    help="Mass (kg) seeded onto massless --gripper-frame-links. Same 0.01 kg "
                         "hygiene value used for the UR10e's massless frame links.")
    ap.add_argument("--deinstance-links", default=",".join(_DELTO_LINKS),
                    help="Comma-separated hand links whose 'visuals' and 'collisions' prims are "
                         "de-instanced. Instance proxies are read-only: visuals must be writable "
                         "for per-env appearance DR and collision meshes for PhysX collider "
                         "visualization. De-instancing changes no geometry and no physics.")
    args = ap.parse_args()

    if args.standoff is None:
        args.standoff = 0.0 if args.gripper_base_link == "rl_dg_mount" else 0.049

    gripper_frame_links = _csv(args.gripper_frame_links)
    deinstance_links = _csv(args.deinstance_links)
    root_joint_names = _csv(args.strip_gripper_root_joint)

    # Resolve the arm USD: explicit --arm-usd as given, else download the cloud asset to the cache.
    arm_usd = resolve_cloud_path(args.arm_usd) if args.arm_usd else resolve_cloud_path(_ARM_USD_URL)
    if not os.path.exists(arm_usd):
        raise SystemExit(f"arm USD not found: {arm_usd}\nDownload the calibrated USD from the cloud first.")
    if not os.path.exists(args.gripper_usd):
        raise SystemExit(f"hand USD not found: {args.gripper_usd}\nThis is the PREPARED hand asset "
                         f"(explicit per-body masses, de-instanceable meshes), not the vendor USD.")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    stage = Usd.Stage.Open(arm_usd)

    # 1) strip the 2F-85. The 2F-85's own mount FixedJoint is nested under robotiq_base_link, so it
    #    is removed along with that body. The 6 arm links/joints and root_joint are untouched, which
    #    is what preserves the UR5e sysid/FK calibration.
    stripped = 0
    for name in F85_BODIES + F85_JOINTS:
        p = f"{ROOT}/{name}"
        if stage.GetPrimAtPath(p):
            stage.RemovePrim(p)
            stripped += 1
    print(f"  stripped {stripped} 2F-85 prim(s) ({len(F85_BODIES)} bodies + {len(F85_JOINTS)} joints expected)")

    # 2) reference the hand under {ROOT}/gripper and place it at the flange. The prim name is fixed
    #    ('gripper') because dexlift's HAND_PRIM and its fingertip ContactSensorCfg paths address it.
    gpath = f"{ROOT}/{HAND_PRIM}"
    gprim = stage.DefinePrim(gpath, "Xform")
    if args.gripper_root_prim:
        gprim.GetReferences().AddReference(os.path.abspath(args.gripper_usd), Sdf.Path(args.gripper_root_prim))
    else:
        gprim.GetReferences().AddReference(os.path.abspath(args.gripper_usd))  # references its defaultPrim

    # world pose of the hand = wrist_3 world * mount(translate +standoff along wrist_3 +Z).
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    wrist = stage.GetPrimAtPath(f"{ROOT}/wrist_3_link")
    if not wrist or not wrist.IsValid():
        raise SystemExit(f"wrist_3_link not found under {ROOT}; is this the calibrated UR5e arm USD?")
    T_wrist_w = xc.GetLocalToWorldTransform(wrist)
    T_mount = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, 0.0, args.standoff))
    T_grip_w = T_mount * T_wrist_w  # pre-multiply: mount is in wrist frame
    # gripper prim local = (root world)^-1 * gripper world
    T_root_w = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(ROOT))
    grip_xform = UsdGeom.Xformable(gprim).MakeMatrixXform()
    grip_xform.Set(T_grip_w * T_root_w.GetInverse())

    # 2a) the target pose belongs to the hand's BASE LINK, not to the referenced subtree's root
    #     prim. Those coincide in the linear gripper (robotiq_base_link sits exactly on
    #     /linear_gripper) but not in general: a hand extracted out of a full-robot USD keeps
    #     whatever residual pose it had inside that robot -- the DELTO's rl_dg_mount was measured
    #     52 um and 0.036 deg off /DeltoHand in the variant this correction was written for. Left
    #     uncorrected, the base link misses the flange by that much while the mount FixedJoint's
    #     anchor still asserts exactly (0, 0, standoff), so the solver yanks the hand at the first
    #     step, and every frame derived from this transform (the wrist camera extrinsic) inherits
    #     the error. Re-place the subtree so the base link lands exactly on target. This is an
    #     identity correction when the base link already sits at the subtree root.
    base = stage.GetPrimAtPath(f"{gpath}/{args.gripper_base_link}")
    if not base or not base.IsValid():
        raise SystemExit(f"hand base link not found: {gpath}/{args.gripper_base_link} "
                         f"(pass --gripper-base-link)")
    xc.Clear()
    T_base_in_grip = xc.GetLocalToWorldTransform(base) * xc.GetLocalToWorldTransform(gprim).GetInverse()
    residual_t = T_base_in_grip.ExtractTranslation()
    grip_xform.Set(T_base_in_grip.GetInverse() * T_grip_w * T_root_w.GetInverse())
    xc.Clear()
    print(f"  base-link residual corrected: {tuple(round(float(v) * 1e6, 1) for v in residual_t)} um")

    # 3) the referenced hand carries its own ArticulationRootAPI (prepare_delto_hand.py applies it
    #    to rl_dg_mount so the standalone asset is spawnable); remove it so the hand joins the arm's
    #    single articulation and the root stays {ROOT}/root_joint. Search the WHOLE referenced
    #    subtree rather than a fixed prim -- which link carries it is an asset-authoring detail.
    #    A hand may also carry its own joint to world; that has to go for the same reason.
    for prim in Usd.PrimRange(gprim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            print(f"  removed nested ArticulationRootAPI from {prim.GetPath()}")
    #    STRIPPING IT IS NOT stage.RemovePrim(). That joint arrives through the {ROOT}/gripper
    #    REFERENCE, so it has no spec in the arm's root layer; UsdStage.RemovePrim only removes
    #    specs in the current edit target, returns False, and -- with the return value discarded --
    #    the operation reads as a success while the joint sails into the export. Deactivate instead
    #    (deactivation is a composed opinion and does compose across a reference), and delete the
    #    surviving spec from the FLATTENED layer, the same way the sibling graft strips the mimic
    #    properties post-Flatten. verify() then asserts the joint is actually gone -- the log line
    #    is not the evidence.
    root_joint_paths: list[str] = []
    for name in root_joint_names:
        for parent in (gpath, f"{gpath}/joints"):
            rj = stage.GetPrimAtPath(f"{parent}/{name}")
            if rj and rj.IsValid():
                rj.SetActive(False)
                root_joint_paths.append(f"{parent}/{name}")
                print(f"  deactivated the hand's own root joint {parent}/{name} (spec deleted post-flatten)")
    if root_joint_names and not root_joint_paths:
        print(f"  no hand root joint found under {gpath} for {root_joint_names} (nothing to strip)")

    # 3a) seed any massless, shapeless frame body (the DELTO's fingertip links) with a small mass +
    #     inertia. PhysX cannot derive either for a body with no collider and no density, and a
    #     zero-mass dynamic body in an articulation is ill-conditioned. Links that already carry an
    #     authored mass are left exactly as authored -- this must not perturb the hand total that
    #     verify() checks against _EXPECTED_HAND_MASS.
    seeded, already_massed = 0, 0
    for name in gripper_frame_links:
        prim = stage.GetPrimAtPath(f"{gpath}/{name}")
        if not prim or not prim.IsValid():
            raise SystemExit(f"hand frame link not found: {gpath}/{name}")
        api = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) else UsdPhysics.MassAPI.Apply(prim)
        if api.GetMassAttr().Get():
            already_massed += 1
            continue
        api.CreateMassAttr().Set(args.gripper_frame_mass)
        # Point-like body: a 5 mm-radius solid sphere of this mass, well below any real link.
        i = 0.4 * args.gripper_frame_mass * 0.005**2
        api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(i, i, i))
        # Author the COM as well: unset, it resolves to UsdPhysics' (-inf, -inf, -inf) "derive it
        # from geometry" sentinel, and these bodies have no geometry.
        api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        seeded += 1
    if gripper_frame_links:
        print(f"  hand frame links: seeded {seeded} massless body(ies) with {args.gripper_frame_mass} kg, "
              f"left {already_massed} authored mass(es) untouched")
    if seeded:
        # The safety net and the mass assertion are mutually exclusive by construction: anything
        # seeded here is mass the AUTHORED total does not contain, so verify() below will fail at
        # _EXPECTED_HAND_MASS. Say so at the seeding site, where the operator can act on it, instead
        # of letting the check 100 lines down look like a rescale problem.
        print(f"  WARNING: {seeded} seeded body(ies) add {seeded * args.gripper_frame_mass:.4f} kg on top of "
              f"the authored total -- the hand USD changed under this script. Re-baseline "
              f"_EXPECTED_HAND_MASS against the new authored sum (do not rescale the hand).")

    # 4) mount FixedJoint wrist_3 -> the hand base link, authored with the same attribute set as the
    #    2F-85's own mount joint but with our mount transform: identity rotation (hand approach +Z
    #    aligned to wrist_3 +Z) and a +standoff offset along wrist_3 +Z.
    fj = UsdPhysics.FixedJoint.Define(stage, f"{gpath}/{args.gripper_base_link}/MountJoint")
    fp = fj.GetPrim()
    fj.CreateBody0Rel().SetTargets([Sdf.Path(f"{ROOT}/wrist_3_link")])
    fj.CreateBody1Rel().SetTargets([Sdf.Path(f"{gpath}/{args.gripper_base_link}")])
    fp.CreateAttribute("physics:localPos0", Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(0.0, 0.0, args.standoff))
    fp.CreateAttribute("physics:localRot0", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(1, 0, 0, 0))
    fp.CreateAttribute("physics:localPos1", Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fp.CreateAttribute("physics:localRot1", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(1, 0, 0, 0))
    fp.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    fp.CreateAttribute("physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool).Set(False)
    fp.CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    fp.CreateAttribute("physics:breakForce", Sdf.ValueTypeNames.Float).Set(float("inf"))
    fp.CreateAttribute("physics:breakTorque", Sdf.ValueTypeNames.Float).Set(float("inf"))

    # 5) flatten (inlines the hand + its meshes) and export a self-contained USD.
    flat = stage.Flatten()

    # 5-) delete the deactivated root-joint SPEC from the flattened layer (see step 3). Once
    #     flattened the referenced opinions are local specs, so this is the point where the joint
    #     can actually be deleted rather than merely overridden. Flatten may already have pruned a
    #     deactivated prim, in which case there is nothing left to do -- both outcomes are correct,
    #     and verify() is what decides.
    for path in root_joint_paths:
        spec = flat.GetPrimAtPath(path)
        if spec is None:
            print(f"  root joint {path}: pruned by flatten (deactivated)")
            continue
        del spec.nameParent.nameChildren[spec.name]
        print(f"  root joint {path}: spec deleted from the flattened layer")

    # 5a) de-instance the hand's visual AND collision prims. Instance proxies are read-only: visuals
    #     must be writable for per-env appearance DR, and collision meshes must be writable for
    #     PhysX collider visualization. This changes no collision geometry and no physics; it only
    #     makes each small mesh independently authorable.
    deinstanced_visuals = 0
    deinstanced_collisions = 0
    for link in deinstance_links:
        for subtree, kind in (("visuals", "visual"), ("collisions", "collision")):
            spec = flat.GetPrimAtPath(f"{gpath}/{link}/{subtree}")
            if spec is not None and spec.instanceable:
                spec.instanceable = False
                if kind == "visual":
                    deinstanced_visuals += 1
                else:
                    deinstanced_collisions += 1
    print(f"  hand meshes: de-instanced {deinstanced_visuals} visual and "
          f"{deinstanced_collisions} collision prim(s)")
    if deinstance_links and deinstanced_visuals == 0:
        print("  WARNING: no hand visuals were de-instanced -- if they are also not plain prims, "
              "the RGB collection's gripper-appearance DR will fail with 'No prims found "
              "matching ... visuals/.*'.")

    flat.Export(args.output)
    print(f"Wrote {args.output}")
    print(f"  standoff along wrist_3 +Z = {args.standoff} m (identity rotation; approach +Z = wrist_3 +Z)")

    # 6) the sibling metadata.yaml. Without it the USD is not a usable asset -- see write_metadata.
    write_metadata(args.output, arm_usd, args.gripper_usd, args.reference_robot_metadata)

    failures = verify(args.output, args.gripper_base_link, args.standoff, root_joint_names)
    if failures:
        print(f"\nFAILED -- {len(failures)} problem(s) with the written asset:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        # pure pxr, no kit: a plain non-zero exit is not swallowed the way it is under Isaac.
        sys.exit(1)
    print("\nOK -- all checks passed.")


if __name__ == "__main__":
    main()
