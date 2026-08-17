# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR5e + DELTO Lift variant that spawns a single real FurnitureBench table leg.

Ported from the reference DexSuite task that measurably reached 92.87% on this object
(``IsaacLabDexterous @ dexsuite/config/ur10_tessolo/dexsuite_ur10_tessolo_tableleg_env_cfg.py``).
It replaces the base task's 16-primitive ``MultiAssetSpawnerCfg`` with one ``UsdFileCfg`` and
narrows two randomization ranges that were tuned for procedurally scalable primitives. NOTHING
else changes: rewards, actions, observations, terminations, commands, the ADR curriculum, the sim
settings, the robot and the episode length are all inherited from
:mod:`.dexlift_ur5e_delto_env_cfg` / :mod:`.dexlift_ur5e_delto_osc_env_cfg`, so a leg run is an
apples-to-apples delta against the primitives run of the same action-space variant.

The 26-dimensional contract (6 arm + 20 hand, all twenty finger joints independent) is therefore
inherited untouched, and the inherited ``check_hand_fully_actuated`` startup term re-verifies it.

BOTH ACTION-SPACE VARIANTS ARE COVERED HERE, and each gets its own event class rather than sharing
one. The bodies are identical; the BASE is not. Variant 2 overrides ``events`` with
:class:`Ur5eDeltoOscEventCfg`, which carries the OSC joint-order guard, so a single event class
subclassing only the variant-1 base would silently drop that guard from variant 2.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.dexsuite import dexsuite_env_cfg as dexsuite

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR

from . import mdp
from .dexlift_ur5e_delto_env_cfg import Ur5eDeltoEventCfg, Ur5eDeltoRelJointPosMixinCfg
from .dexlift_ur5e_delto_osc_env_cfg import Ur5eDeltoOscEventCfg, Ur5eDeltoOscMixinCfg

TABLE_LEG_USD_PATH = (
    f"{UWLAB_LOCAL_ASSETS_DIR}/Props/FurnitureBench/SquareTableLeg200mmDecomp/square_table_leg4_200mm.usd"
)
print(
    "[dexlift] leg collider: convexDecomposition (re-authored). The SHIPPED asset has no"
    " physics:approximation, so PhysX silently falls back to convexHull and fills the thread"
    " relief; this variant authors the approximation the converter's own config.yaml requested."
    " NOTE the reference's USD is md5-identical to the shipped one, so the certified 92.87%"
    " run also used the hull -- a number measured here is NOT strictly comparable to it.",
    flush=True,
)
"""The certified table-leg USD, copied byte-identical out of the reference repository.

THE SHIPPED ASSET DOES NOT ACTUALLY GET THE COLLIDER ITS OWN CONFIG DESCRIBES, and the description
below records the request rather than the result. Every table-leg run in this repository logs::

    PhysicsUSD: Parse collision - triangle mesh collision (approximation None/MeshSimplification)
    cannot be a part of a dynamic body, falling back to convexHull approximation

because the mesh prim inside ``Props/instanceable_meshes.usd`` carries only ``PhysicsCollisionAPI``:
``UsdPhysics.MeshCollisionAPI`` is absent, so ``physics:approximation`` does not exist and PhysX
substitutes a hull. This module therefore spawns the RE-AUTHORED variant, built by
``scratchpad/reauthor_leg_decomp.py`` (PhysX tuning parameters could not be applied in this USD
build, so it decomposes at PhysX defaults). It is not a toggle: on the hull the pose task COLLAPSES
to 0.063 at the 5 cm tolerance where the decomposed leg reaches 0.88, from an identical warm start
and an identical reward -- the hull fills the thread relief, which is the feature a finger seats in
to hold the rod against gravity torque. Position-only is far more forgiving (0.929 on the hull), so
the defect stayed invisible until orientation was demanded.

VERIFY BEHAVIOURALLY, not by reading this comment: the PhysX fallback line quoted above must be
ABSENT from the run log.

``UWLAB_LOCAL_ASSETS_DIR`` is this repository's analogue of the reference's
``ISAACLAB_ASSETS_DATA_DIR``. A literal port of the reference's path would raise at spawn: UWLab's
``isaaclab_assets`` is editable-installed from ``_isaaclab/IsaacLab``, whose ``data/`` directory
holds only a ``.gitkeep`` and has no ``props/`` tree at all.

The BYTES are copied rather than the geometry re-converted. Measured on the reference asset: extent
0.200 x 0.030 x 0.030 m (long axis local X), volume 157.305 cm^3, convexDecomposition collision
(hull_vertex_limit 64, max_convex_hulls 32, voxel_resolution 500000 -- see ``config.yaml`` next to
the USD), recentred so the prim origin is the volume centroid. Copying pins the exact geometry
behind the reference's 92.87%; re-running the converter needs an Isaac runtime and would not be
bit-reproducible. The neighbouring ``Props/instanceable_meshes.usd`` is part of the asset, not a
build artifact: the top-level USD references it by relative path.
"""

TABLE_LEG_MASS_KG = 0.12
"""Authored mass of the leg, in kg. The reference's value, and its argument transfers verbatim.

Written explicitly rather than left implicit in the USD so it is a real dataclass field and is
therefore reachable from the CLI as ``env.scene.object.spawn.mass_props.mass=<value>`` --
IsaacLab's ``update_class_from_dict`` rejects any hydra key not already present in the tree.

157.3 cm^3 at 0.12 kg is ~763 kg/m^3, the hardwood range. (The converter's own 0.02275 kg implied
~145 kg/m^3, styrofoam.) The operative reason is the REWARD GATE, not the density: this package's
``rewards.position_tracking`` -- weight 2.0, the dominant shaping term -- is gated on 1.0 N of
fingertip contact force, inherited from the primitives task whose objects weigh 40-400 g. At
22.75 g the object weighs 0.223 N and the gate would demand ~4.5x its own weight before paying
anything, which on a free light object ejects it rather than holds it. With ``object_scale_mass``
below sampling [0.5, 1.5], 0.12 kg gives 60-180 g, inside the envelope the gate was calibrated
against.

ONE CORRECTION to the reference's own docstring, which says both ``position_tracking`` and
``success`` are gated. On the LIFT task they are not: ``dexlift.mdp.rewards.success_reward``
returns ``(1 - tanh(pos_dist / pos_std))**2`` and RETURNS before reaching the contact gate when
``rot_std`` is None, which is exactly what the Lift subclass configures. The live 1.0 N gate on
Lift is carried by ``position_tracking`` alone. The mass is still the right lever.

In UWLab the 1.0 N is a FUNCTION DEFAULT (``threshold: float = 1.0`` on ``success_reward`` and
``position_command_error_tanh`` in ``mdp/rewards.py``) rather than an explicit param --
``DeltoHandRewardsCfg`` writes ``threshold`` only on its four 0.2 N terms. So unlike the
reference's UR10-Tessolo mixin, ``env.rewards.position_tracking.params.threshold=`` is NOT
CLI-overridable here. Making it so would be a reward change, and this port makes none.
"""


@configclass
class Ur5eDeltoTableLegSceneCfg(dexsuite.SceneCfg):
    """The dexsuite scene with its ``object`` -- and only its ``object`` -- swapped for the leg.

    The robot, table, plane and lights are inherited untouched; ``Ur5eDeltoMixinCfg.__post_init__``
    then replaces the table and shifts the workspace exactly as it does for the primitives task.

    THE ENTITY NAME AND PRIM PATH ARE LOAD-BEARING. The five fingertip contact sensors installed by
    ``Ur5eDeltoMixinCfg`` hardcode ``filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"]``, so
    renaming either would zero every contact-gated reward without an error.

    ``semantic_tags`` is deliberately OMITTED, unlike the reference, which sets it only because ITS
    base ``SceneCfg`` does; UWLab's base does not. Matching our own base is what keeps this a clean
    one-field delta. Behaviourally inert either way -- semantic tags feed only semantic
    segmentation, which no observation in these ids reads.

    No ``scale=`` and no ``physics_material=``, matching the reference: object friction is
    overwritten anyway by the inherited ``object_physics_material`` startup event.
    """

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_LEG_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=TABLE_LEG_MASS_KG),
        ),
        # The base's own spawn position, kept for structural parity and then OVERWRITTEN by
        # ``Ur5eDeltoMixinCfg.__post_init__`` to (WORKSPACE_X, 0.0, 0.35 + WORKSPACE_Z_SHIFT) =
        # (0.55, 0.0, 0.095). That overwrite is what puts the leg on the UR5e work surface instead
        # of dexsuite's, and it is correct: leave it alone.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35)),
    )


@configclass
class Ur5eDeltoTableLegEventCfg(Ur5eDeltoEventCfg):
    """VARIANT 1's events, with the two primitive-tuned randomization ranges corrected for a real,
    dimensionally-fixed 200 mm part. Everything else -- the physics-material randomization, the
    actuator/friction narrowing, the sysid term, the resets, the frame guards -- is inherited.
    """

    # DISABLED, not pinned to a degenerate scale_range=(1.0, 1.0) no-op, for two reasons.
    # Dimensional: this is a manufactured 200 mm part, and the base's (0.75, 1.5) would train on
    # legs between 15 cm and 30 cm. Structural: ``randomize_rigid_body_scale`` runs in "prestartup"
    # and writes a per-env ``xformOp:scale``, which its own docstring says requires
    # ``replicate_physics=False``; a pinned-but-still-running term would keep that requirement
    # alive and conflict with the ``replicate_physics=True`` the mixins below set. The
    # EventManager skips ``None`` terms outright, so this removes the conflict at zero behavioural
    # cost -- every env would have drawn 1.0 anyway.
    randomize_object_scale = None

    # Narrowed from the base's [0.2, 2.0] SCALE range (24-240 g against a 0.12 kg leg) to
    # [0.5, 1.5] -> 60-180 g: still +-50% mass domain randomization, and still more than an order
    # of magnitude above the ~1 g floor where a prop left at its default mass gets flung by contact
    # and produces false-positive grasps.
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": [0.5, 1.5],
            "operation": "scale",
        },
    )


@configclass
class Ur5eDeltoOscTableLegEventCfg(Ur5eDeltoOscEventCfg):
    """VARIANT 2's events, with the same two corrections.

    Identical body to :class:`Ur5eDeltoTableLegEventCfg`, different base -- see the module
    docstring. The base is what carries ``check_osc_arm_joints``.
    """

    randomize_object_scale = None

    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": [0.5, 1.5],
            "operation": "scale",
        },
    )


@configclass
class Ur5eDeltoTableLegRelJointPosMixinCfg(Ur5eDeltoRelJointPosMixinCfg):
    """VARIANT 1 (26-DOF relative joint position) on the table leg.

    ``scene`` is declared HERE and not on ``Ur5eDeltoMixinCfg``, which has no such field -- it
    mutates ``self.scene.*`` inside ``__post_init__`` instead. Declared on this class it outranks
    ``DexsuiteReorientEnvCfg.scene`` in the MRO, which is what makes the swap take.

    ``replicate_physics=True`` against the base's ``False``, for the reference's stated reason,
    re-verified on this scene: the base sets it False because ``MultiAssetSpawnerCfg`` puts a
    different primitive in every env and sets the ``/isaaclab/spawn/multi_assets`` carb flag the
    cloner warns on. Here no multi-asset spawner remains, ``randomize_object_scale`` (the other
    per-env prestartup USD writer) is disabled above, and this scene's table is a single
    ``CuboidCfg``, so no per-env heterogeneity is left for the cloner to mishandle. This is the
    only scene-level change.

    NO ``__post_init__``. ``Ur5eDeltoMixinCfg``'s runs unchanged and supplies the robot, the table,
    the workspace shift, the sensors, the guards and the task-state-visualization binding; since
    this port replaces neither ``scene.table`` nor ``curriculum``, that binding does not need
    re-running. Adding one here would also be the one way to reach the shared module-level
    ``IMPLICIT_UR5E_DELTO`` / ``EXPLICIT_UR5E_DELTO`` articulations, which every other environment
    in the process spawns too.
    """

    scene: Ur5eDeltoTableLegSceneCfg = Ur5eDeltoTableLegSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)
    events: Ur5eDeltoTableLegEventCfg = Ur5eDeltoTableLegEventCfg()


@configclass
class Ur5eDeltoTableLegOscMixinCfg(Ur5eDeltoOscMixinCfg):
    """VARIANT 2 (operational-space arm, same fully actuated hand) on the table leg.

    Same scene and the same ``replicate_physics`` argument as variant 1; the events come from the
    OSC base so the joint-order guard survives.
    """

    scene: Ur5eDeltoTableLegSceneCfg = Ur5eDeltoTableLegSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)
    events: Ur5eDeltoOscTableLegEventCfg = Ur5eDeltoOscTableLegEventCfg()


# REORIENT ON THE LEG IS A DELIBERATE EXTENSION, NOT A PORT. The reference registers no Reorient
# table-leg task, so nothing below has a certified number behind it and none of it may be cited as
# reference-faithful. It exists because the deliverable is a commanded POSE, not a commanded
# position: the leg must arrive at both.
#
# WHY THE OBJECT MAKES THIS PLAUSIBLE AT A TIGHT POSITION TOLERANCE, while a naive port would not.
# ``convert_table_leg.py`` recentres the merged mesh on its VOLUME CENTROID, so the USD prim origin
# -- which is what ``root_pos_w`` reports, since it aliases ``root_link_pos_w`` and never the COM --
# sits at the centroid. Without that recentring the raw origin is ~0.0625 m from the centroid, and
# ORIENTATION randomization alone would sweep the tracked point by up to ~0.125 m: more than twice a
# 5 cm success radius and twelve times a 1 cm one. On Lift that was already worth fixing; on Reorient
# it is the difference between a solvable task and one whose position error is dominated by an
# artifact of where the prim origin happens to sit.
#
# WHAT IS NOT ESTABLISHED, and must be measured before any failure here is read as a training
# problem: the goal command samples roll and pitch over +-pi with yaw pinned to 0, and this object is
# a 200 x 30 x 30 mm rod. Whether an in-hand reorientation of a rod to an arbitrary roll/pitch is
# reachable AT ALL for this hand is an open physical question, not a hyperparameter.


@configclass
class DexLiftUR5eDeltoRelJointPosTableLegLiftEnvCfg(Ur5eDeltoTableLegRelJointPosMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexLiftUR5eDeltoRelJointPosTableLegLiftEnvCfg_PLAY(
    Ur5eDeltoTableLegRelJointPosMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY
):
    pass


@configclass
class DexLiftUR5eDeltoOscTableLegLiftEnvCfg(Ur5eDeltoTableLegOscMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexLiftUR5eDeltoOscTableLegLiftEnvCfg_PLAY(Ur5eDeltoTableLegOscMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass


@configclass
class DexLiftUR5eDeltoRelJointPosTableLegReorientEnvCfg(
    Ur5eDeltoTableLegRelJointPosMixinCfg, dexsuite.DexsuiteReorientEnvCfg
):
    pass


@configclass
class DexLiftUR5eDeltoRelJointPosTableLegReorientEnvCfg_PLAY(
    Ur5eDeltoTableLegRelJointPosMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY
):
    pass


@configclass
class DexLiftUR5eDeltoOscTableLegReorientEnvCfg(Ur5eDeltoTableLegOscMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexLiftUR5eDeltoOscTableLegReorientEnvCfg_PLAY(
    Ur5eDeltoTableLegOscMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY
):
    pass
