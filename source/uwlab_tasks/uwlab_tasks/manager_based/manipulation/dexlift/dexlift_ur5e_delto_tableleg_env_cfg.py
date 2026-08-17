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


def _apply_partial_assembly_and_goal_toggles(env_cfg) -> None:
    """PARTIALLY-ASSEMBLED SPAWN, and separately GOAL-AT-SPAWN, opt-in via the environment (bead
    UWLab-qiao.2/.6/.9). Two INDEPENDENT toggles, because C3's box measurement showed the height
    floor a policy converges to is set by the GOAL command's range, not the spawn distribution --
    ``DEXLIFT_SPAWN_CLEARANCE`` alone (leg spawns 1-40 cm above the table) did not move the accepted
    height distribution at all; it tracked ``commands.object_pose.ranges.pos_z`` exactly. Pinning
    the goal to the spawn pose (this bead's original Y3, ``GoalAtSpawnPoseCommand``) removes the
    incentive to lift at all, forcing an in-place grasp instead -- ``contacts()`` still gates
    ``success_reward`` and ``rewards.position_tracking`` (``dexlift/mdp/rewards.py:113-165``), so
    the grasp still has to be real.

    A MODULE-LEVEL FUNCTION, not inline in one class's ``__post_init__``, and called from BOTH the
    TRAIN and the PLAY Reorient-table-leg classes (bead UWLab-qiao.9/J) -- exactly the pattern
    ``_apply_pose_tilt_stage`` above already uses for ``DEXLIFT_POSE_TILT``/``DEXLIFT_DROP_Z``, and
    the ONLY reason it works for those and briefly did not for this: those are called from
    ``Ur5eDeltoMixinCfg.__post_init__`` in ``dexlift_ur5e_delto_env_cfg.py``, the SHARED base every
    task family's MRO passes through (train AND ``_PLAY``, Lift AND Reorient, RelJointPos AND OSC)
    -- so a single call site there reaches every id. This toggle pair is Reorient-table-leg-specific
    and has no such shared base to hang off; ``DexLiftUR5eDeltoRelJointPosTableLegReorientEnvCfg``
    and its ``_PLAY`` sibling are NOT in an inheritance relationship with each other (both are built
    fresh from ``Ur5eDeltoTableLegRelJointPosMixinCfg`` + a *different* dexsuite base --
    ``DexsuiteReorientEnvCfg`` vs ``DexsuiteReorientEnvCfg_PLAY``), so a block written directly into
    one class's ``__post_init__`` is invisible to the other's MRO -- confirmed on the box: the
    sixth ``[verify]`` line reported ``reset_root_state_uniform`` under a Play-id run with
    ``DEXLIFT_PARTIAL_ASSEMBLY=1`` exported, run killed 20s in before any GPU time was spent on it.

    MECHANISM CHOSEN OVER THE ALTERNATIVES, and why. A shared MIXIN class (both Reorient classes
    inheriting one small ``Ur5eDeltoTableLegReorientPartialAssemblyMixinCfg`` ahead of
    ``Ur5eDeltoTableLegRelJointPosMixinCfg`` in their bases) would also work, but adds a THIRD base
    to reason about the MRO of, for a toggle that is pure ``__post_init__`` behaviour with no fields
    of its own -- a plain function call is the same fix with no linearization to get right. Making
    ``_PLAY`` inherit from the TRAIN class directly was explicitly ruled out: ``dexsuite``'s
    ``*_PLAY`` bases are near-certain to differ in env count, episode length or randomization
    (mirroring every other ``_PLAY`` sibling in this file, e.g. ``DexLiftUR5eDeltoRelJointPos
    TableLegLiftEnvCfg_PLAY`` is its own sibling of the Lift train class, not a subclass of it), and
    routing Play's construction through the Train class's MRO risks silently picking up whatever
    those differences are.

    LIFT IS STILL EXCLUDED BY CONSTRUCTION: this function is called from exactly two ``__post_init__``
    methods, both on Reorient classes, in this same module. Nothing calls it from either
    ``DexLiftUR5eDeltoRelJointPosTableLegLiftEnvCfg`` or its ``_PLAY`` sibling -- on Lift, ``rot_std``
    is forced ``None``, which makes ``success_reward`` return before its contact gate, so goal-at-
    spawn there would pay an idle policy. The OSC Reorient variants
    (``DexLiftUR5eDeltoOscTableLegReorientEnvCfg{,_PLAY}``) are ALSO not wired to this function --
    out of scope for this bead (target task was RelJointPos only) and left that way deliberately, not
    silently: see the audit table in this bead's chat log for the full toggle-by-toggle reachability
    accounting.

    DEFAULT PATH IS BYTE-IDENTICAL WHEN BOTH ENV VARS ARE UNSET: nothing below runs unless one of the
    two is "1", matching the ``DEXLIFT_REF_RESET`` / ``DEXLIFT_SPAWN_CLEARANCE`` idiom this mirrors.
    """
    partial_assembly = os.environ.get("DEXLIFT_PARTIAL_ASSEMBLY") == "1"
    goal_at_spawn = partial_assembly or os.environ.get("DEXLIFT_GOAL_AT_SPAWN") == "1"
    spawn_clearance = os.environ.get("DEXLIFT_SPAWN_CLEARANCE") == "1"

    if partial_assembly:
        # -- Y1: the fixture. Never present in this scene before this toggle.
        env_cfg.scene.receptive_object = mdp.make_dexlift_receptive_object_cfg()

        # -- Y2: ONE event places the fixture, then composes+places the leg against it, in that
        # order, inside one Python call -- see partial_assembly.py's docstring for why two
        # separate EventTerms would race. REPLACES (does not add to) whatever ``reset_object``
        # currently is -- including a ``DEXLIFT_SPAWN_CLEARANCE=1`` assignment from earlier in
        # this same ``__post_init__`` chain, if that was also set: the fixture-composed pose is
        # the intended source of truth for this pairing and wins over a free-scatter clearance
        # spawn, not merged with it.
        env_cfg.events.reset_object = EventTerm(
            func=mdp.SpawnPartialAssembly,
            mode="reset",
            params={
                "dataset_dir": mdp.DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR,
                "insertive_object_cfg": SceneEntityCfg("object"),
                "receptive_object_cfg": SceneEntityCfg("receptive_object"),
                "fixture_pose_range": mdp.RECEPTIVE_POSE_RANGE,
                # No extra jitter on top of the stored partial-assembly relative pose -- the
                # leg spawns exactly where a recorded partial-assembly sample puts it.
                "pose_range_b": {},
            },
        )

    if goal_at_spawn:
        # -- Y3: the goal is the object's own spawn pose, not a fresh uniform draw. MUST be a
        # command SUBCLASS -- see partial_assembly.py's docstring, "Y3" section: an event term
        # cannot do this, because CommandManager.reset() always resamples afterward, in the same
        # reset call, regardless of resampling_time_range. Independent of ``partial_assembly``:
        # this block is reached whenever EITHER toggle asks for it.
        env_cfg.commands.object_pose = mdp.upgrade_to_goal_at_spawn(env_cfg.commands.object_pose)

        # -- G3: name every toggle that is live and where the goal is coming from, so a
        # generation log states the configuration rather than leaving it to be inferred. A
        # silently-unset toggle here is a plausible wrong number, not an obvious one.
        reset_object_source = (
            "SpawnPartialAssembly (partial_assemblies.pt)" if partial_assembly
            else "reset_object_pose_with_clearance (DEXLIFT_SPAWN_CLEARANCE=1)" if spawn_clearance
            else "reset_root_state_uniform (dexsuite default pose_range)"
        )
        print(
            f"[dexlift] DEXLIFT_PARTIAL_ASSEMBLY={int(partial_assembly)}"
            f" DEXLIFT_GOAL_AT_SPAWN={int(goal_at_spawn)}"
            f" DEXLIFT_SPAWN_CLEARANCE={int(spawn_clearance)}:"
            f" receptive_object {'ADDED' if partial_assembly else 'absent'}"
            + (
                f" at x={mdp.RECEPTIVE_POSE_RANGE['x']} y={mdp.RECEPTIVE_POSE_RANGE['y']}"
                f" z={mdp.RECEPTIVE_POSE_RANGE['z'][0]}"
                if partial_assembly else ""
            )
            + f"; reset_object -> {reset_object_source};"
            " goal SOURCE = object spawn pose (pinned, not uniform-sampled)", flush=True,
        )


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
    # -- PARTIALLY-ASSEMBLED SPAWN / GOAL-AT-SPAWN toggles: see
    # ``_apply_partial_assembly_and_goal_toggles``'s docstring above for the full argument (why two
    # independent toggles, why a shared function rather than inline code, why Lift is excluded).
    # Called from BOTH this class and its ``_PLAY`` sibling below -- they are NOT in an inheritance
    # relationship with each other, so the call has to be written twice, once per class, or Play
    # never sees it (bead UWLab-qiao.9/J -- this is precisely the bug being fixed here).
    def __post_init__(self):
        super().__post_init__()
        _apply_partial_assembly_and_goal_toggles(self)


@configclass
class DexLiftUR5eDeltoRelJointPosTableLegReorientEnvCfg_PLAY(
    Ur5eDeltoTableLegRelJointPosMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY
):
    # -- Same toggles as the train class above, same reason, same function -- see its docstring.
    # This class does NOT inherit from ``DexLiftUR5eDeltoRelJointPosTableLegReorientEnvCfg``, so the
    # call above does not reach here on its own; it has to be repeated.
    def __post_init__(self):
        super().__post_init__()
        _apply_partial_assembly_and_goal_toggles(self)


@configclass
class DexLiftUR5eDeltoOscTableLegReorientEnvCfg(Ur5eDeltoTableLegOscMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexLiftUR5eDeltoOscTableLegReorientEnvCfg_PLAY(
    Ur5eDeltoTableLegOscMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY
):
    pass
