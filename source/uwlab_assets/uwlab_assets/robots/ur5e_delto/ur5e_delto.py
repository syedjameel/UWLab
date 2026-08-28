# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Articulation configurations for the UR5e with a Tesollo DELTO DG-5F hand.

The USD is built by ``scripts_v2/tools/conversions/graft_delto_on_ur5e.py``: the calibrated cloud
UR5e keeps its six links, its six joints and its ``root_joint`` untouched, the 2F-85 + D415 wrist
assembly is stripped, and the prepared 28-body hand is referenced under ``{ROOT}/gripper`` at the
flange. ``Ur5eDelto/metadata.yaml`` next to it carries the arm identification, copied byte-equal
from the genuine calibrated UR5e metadata.

WHERE EACH NUMBER COMES FROM, because this robot is a graft of two identified halves and mixing
them up is the failure mode this file exists to prevent:

* ARM side -- default posture, effort and velocity limits, actuator structure -- comes from
  :mod:`uwlab_assets.robots.ur5e_robotiq_gripper.ur5e_robotiq_2f85_gripper`, i.e. the UR5e. It does
  NOT come from the UR10e+DELTO configuration, whose arm is a much stronger machine (330/330/150/
  56/56/56 N*m against this arm's 150/150/150/28/28/28).
* HAND side -- the open posture and the per-joint actuator -- is imported from the UR10e+DELTO
  module. Those are properties of the HAND, not of whatever arm it is bolted to, and the open
  posture in particular is checked BY NAME against ``finger_open_joint_angles`` in the hand's own
  ``metadata.yaml`` and against the graft's assertions, so it is aliased here rather than restated.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR

# Hand-side constants. Imported from the module rather than the package so this file does not pull
# in ``ur10e_delto.actions`` (and with it ``isaaclab.envs``) just to read two hand properties.
from uwlab_assets.robots.ur10e_delto.ur10e_delto import DELTO_HAND_ACTUATOR, DELTO_HAND_DEFAULT_JOINT_POS

# Arm-side constants: the UR5e's own, from the calibrated 2F-85 configuration.
from uwlab_assets.robots.ur5e_robotiq_gripper.ur5e_robotiq_2f85_gripper import (
    ROBOTIQ_2F85_DEFAULT_JOINT_POS,
    UR5E_DEFAULT_JOINT_POS,
    UR5E_EFFORT_LIMITS,
    UR5E_VELOCITY_LIMITS,
)

# UR5E_DEFAULT_JOINT_POS bundles the arm posture WITH the 2F-85's six joints, which the graft
# deletes -- an init_state key that matches no joint is an error at spawn. Drop exactly those six
# rather than restating the six arm values, so a change to the arm's home posture reaches this
# robot too. (Same construction as ``_UR10E_ARM_DEFAULT_JOINT_POS`` in the UR10e+DELTO module.)
_UR5E_ARM_DEFAULT_JOINT_POS = {
    name: value for name, value in UR5E_DEFAULT_JOINT_POS.items() if name not in ROBOTIQ_2F85_DEFAULT_JOINT_POS
}

UR5E_DELTO_DEFAULT_JOINT_POS = {
    **_UR5E_ARM_DEFAULT_JOINT_POS,
    **DELTO_HAND_DEFAULT_JOINT_POS,
}

UR5E_DELTO_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_LOCAL_ASSETS_DIR}/Robots/Ur5eDelto/ur5e_delto.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # The arm is driven by torque or by position targets, never by gravity compensation, and
            # every OmniReset/dexsuite UR configuration in this tree spawns it weightless.
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            # NOTE, deliberately NOT inherited from the UR10e+DELTO config: that one adds
            # linear_damping/angular_damping 500 to suppress wrist ringing from the soft 20-joint
            # hand. It is a package value pinned to the D2/D3 datasets and the D4 training run of
            # that robot, and 500 is enormous -- it is a property of those artifacts, not of the
            # hand. This robot has no such artifacts yet, so it starts from the UR5e's own rigid
            # properties. If the wrist rings once this is run in Isaac, that damping pair is the
            # first thing to try, and the value should be re-derived rather than copied.
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Self-collisions ON, as on the calibrated UR5e and as the hand asset itself asks for:
            # it comes from the reference's ``..._self_collision.usd`` variant, which authors
            # ``physxArticulation:enabledSelfCollisions = True``.
            enabled_self_collisions=True,
            solver_position_iteration_count=36,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0), rot=(1, 0, 0, 0), joint_pos=UR5E_DELTO_DEFAULT_JOINT_POS
    ),
    soft_joint_pos_limit_factor=1,
)

EXPLICIT_UR5E_DELTO = UR5E_DELTO_ARTICULATION.copy()  # type: ignore
EXPLICIT_UR5E_DELTO.actuators = {
    "arm": DelayedPDActuatorCfg(
        joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
        # Zero gains: the OmniReset environments drive this arm through an operational-space
        # controller that supplies joint torques directly. A task that writes joint POSITION
        # targets instead (dexsuite/dexlift) must set real PD gains on this actuator itself.
        stiffness=0.0,
        damping=0.0,
        effort_limit=UR5E_EFFORT_LIMITS,
        effort_limit_sim=UR5E_EFFORT_LIMITS,
        velocity_limit=UR5E_VELOCITY_LIMITS,
        velocity_limit_sim=UR5E_VELOCITY_LIMITS,
        min_delay=0,
        max_delay=1,
    ),
    # The hand actuator OBJECT, shared rather than copied: it is identified per hand joint and is
    # the same hardware on both arms. See DELTO_HAND_ACTUATOR's own comments for every gain.
    "hand": DELTO_HAND_ACTUATOR,
}

IMPLICIT_UR5E_DELTO = UR5E_DELTO_ARTICULATION.copy()  # type: ignore
IMPLICIT_UR5E_DELTO.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
        stiffness=0.0,
        damping=0.0,
        effort_limit_sim=UR5E_EFFORT_LIMITS,
        velocity_limit_sim=UR5E_VELOCITY_LIMITS,
    ),
    "hand": DELTO_HAND_ACTUATOR,
}

# --------------------------------------------------------------------------------------------
# GENERATION PLANT: the hand collider set and hand actuator dexlift's certified route spawns
# instead of the two objects above, and every UR5eDelto reset bank and checkpoint this project
# holds was produced under. Defined HERE -- not in ``dexlift_ur5e_delto_env_cfg.py`` and not
# retyped into any OmniReset config -- so both task families build the same generation plant
# from one source and cannot silently diverge (a retyped constant is how this project has
# repeatedly ended up with two values that agree today and disagree later).
# --------------------------------------------------------------------------------------------

UR5E_DELTO_HULLFIX3_USD_NAME = "ur5e_delto_hullfix3.usd"
"""Filename of the hand-collider USD living next to ``ur5e_delto.usd``: 3 convexDecomposition
bodies (``rl_dg_mount``, ``rl_dg_base``, ``rl_dg_palm``) + 25 convexHull (20 phalanges + 5
tips), zero sdf, plus ``UsdPhysics.FilteredPairsAPI`` on ``rl_dg_base``/``rl_dg_palm`` against
all 25 finger bodies.

``ur5e_delto.usd`` above carries 23 sdf + 5 convexHull instead. A blanket sdf-to-convexHull
rewrite EXPLODES the articulation at reset: the palm shell is CONCAVE and a convex hull of it
swallows the finger roots, which PhysX then depenetrates violently under self-collision (8045
deg of hand deviation measured with zero actions applied, against a +-28.6 deg reset jitter
band). ``convexDecomposition`` on the three shells is what preserves the concavity. The
filtered pairs are reference-faithful, not a cheat: ``rl_dg_base``/``rl_dg_palm`` are rigidly
welded to ``rl_dg_mount``, the phalanx joints' own parent, so PhysX already filters the
kinematically identical mount-vs-phalanx pair; ``rl_dg_mount`` vs finger and finger vs finger
both stay LIVE.

Resolve as ``pathlib.Path(<ur5e_delto.usd path>).with_name(UR5E_DELTO_HULLFIX3_USD_NAME)`` --
never hard-code the full path -- so every consumer tracks ``UR5E_DELTO_ARTICULATION``'s own
``usd_path`` automatically. Generate it with ``scratchpad/reauthor_hullfix.py`` (dexlift), which
verifies the saved layer by re-reading it.
"""

REFERENCE_HAND_ACTUATOR_KWARGS = {
    "effort_limit_sim": {r"rj_dg_[1-5]_[1-4]": 30.0},
    "velocity_limit_sim": {r"rj_dg_[1-5]_[1-4]": 10000.0},
    "stiffness": {r"rj_dg_[1-5]_[1-4]": 3.0},
    "damping": {r"rj_dg_[1-5]_[1-4]": 0.1},
}
"""The reference implementation's (IsaacLabDexterous) flat hand-actuator gains, applied via
``actuator.replace(**REFERENCE_HAND_ACTUATOR_KWARGS)`` -- NEVER by editing
:data:`~uwlab_assets.robots.ur10e_delto.ur10e_delto.DELTO_HAND_ACTUATOR` in place, which is
shared with every other consumer of this hand (``EXPLICIT``/``IMPLICIT_UR10E_DELTO``,
``DELTO_HAND``, the grasp recorder, and both actuator dicts above).

~176x-500x looser than the identified hand's 0.06-0.17 N.m effort_limit_sim / 3.0 rad/s
velocity_limit_sim. At action scale 0.1 and 60 Hz a policy commands roughly 6 rad/s of finger
closure; the identified hand's 3.0 rad/s ceiling means it CANNOT track that command at all, so
data generated under the identified actuator produces a hand that never really closes (measured
across dexlift's ablation: 2.69% acceptance under the identified actuator vs 46.71% under this
block). EVERY UR5eDelto reset bank and the certified checkpoint this project holds were produced
with this block active.
"""
