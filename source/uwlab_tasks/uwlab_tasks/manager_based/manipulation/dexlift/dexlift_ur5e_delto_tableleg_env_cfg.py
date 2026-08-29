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
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.dexsuite import dexsuite_env_cfg as dexsuite

from uwlab_assets import UWLAB_LOCAL_ASSETS_DIR, assert_omnireset_leg_literals_agree

from . import mdp

# Imported DIRECTLY, departing from ``_apply_c1_hand_pose_stage``'s precedent of re-typing its
# core module's env parsing inline (see that function's own note explaining the duplication). The
# departure is deliberate and is the safer half of the trade: this file would otherwise carry a
# SECOND copy of the DEXRESET_C3_* parsing and bounds, and the unit test
# (``test/test_c3_rung_stage.py``) would then prove a copy that no run actually executes -- the
# exact "a constant established in one place and consumed in another, with nothing checking they
# agree" failure this campaign has recorded repeatedly (V2_POSE_FINDINGS.md F27's family). Importing
# the Isaac-free core costs nothing here: this module already imports isaaclab at module scope.
from .mdp import c3_rung_core
from .dexlift_ur10e_delto_env_cfg import TIP_NAMES, THUMB_TIP_NAMES
from .dexlift_ur5e_delto_env_cfg import Ur5eDeltoEventCfg, Ur5eDeltoRelJointPosMixinCfg
from .dexlift_ur5e_delto_osc_env_cfg import Ur5eDeltoOscEventCfg, Ur5eDeltoOscMixinCfg

# Fatal, at import time, not a warning: this is the dexlift-side half of the five-literal guard --
# see assert_omnireset_leg_literals_agree's own docstring (uwlab_assets/__init__.py) for the full
# defect this catches. Runs BEFORE TABLE_LEG_USD_PATH below is even read, so a stale literal here
# or in any of the four OmniReset registries is caught before this module finishes importing.
assert_omnireset_leg_literals_agree()

TABLE_LEG_USD_PATH = os.environ.get(
    "DEXLIFT_TABLE_LEG_USD_PATH_OVERRIDE",
    f"{UWLAB_LOCAL_ASSETS_DIR}/Props/FurnitureBench/SquareTableLeg200mmSdf/square_table_leg4_200mm.usd",
)
# SquareTableLeg200mmSdf IS NOW THE SHIPPING LEG (changed 2026-08-23). It is the same merged
# geometry as the old SquareTableLeg200mmDecomp -- 31855 points, identical bounds, verified
# vertex-for-vertex -- differing ONLY in the collision approximation: physics:approximation=sdf
# at sdfResolution=256 instead of convexDecomposition. The decomposition variant is REJECTED:
# its hulls fill the helical thread grooves, so 56.15% of poses interpenetrated the collider
# PhysX actually uses (median -0.068 mm) and depenetration ejected the leg within five steps
# even with the robot frozen. The SDF variant holds flat at ~16.56 mm under the same test.
#
# This constant, and the four OmniReset registries (rl_state_cfg, reset_states_cfg,
# grasp_sampling_cfg, partial_assemblies_cfg), MUST agree. They are five separate literals
# naming one asset, and they disagreed until 2026-08-23: generation read this file while
# training read rl_state_cfg, so "verified on SDF, trained on Decomp" was silently possible.
# If you repoint one, repoint all five.
#
# DEXLIFT_TABLE_LEG_USD_PATH_OVERRIDE remains a diagnostic escape hatch for pointing a one-off
# run at a different variant (e.g. the 1024/2048 SDF builds) without touching shared config.
# It is dexlift-only -- OmniReset has NO equivalent, and a raw USD path on the CLI does not
# resolve there because that override mechanism only looks up keys in the variants registry.
# So an override set here does NOT propagate to training. Do not rely on it to change the
# shipping asset; change the five literals.
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


def _apply_partial_assembly_and_goal_toggles(env_cfg) -> bool:
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

    RETURNS whether either legacy toggle fired (bead UWLab-g3z4). Every existing caller of this
    function ignored its return value, so adding one is backward compatible; the new per-episode
    ``episode_mixture`` (see that module's docstring) reads it to skip installing the probabilistic
    mixture whenever a caller has explicitly asked for the deterministic, whole-run legacy path --
    several tools outside training (reset-state generation, certification, rendering) depend on being
    able to force 100% of envs into partial-assembly/goal-at-spawn, which a mixture cannot give them.

    A THIRD, INDEPENDENT GOAL VARIANT (bead UWLab-xp05.3, "Arm 3"): ``DEXLIFT_GOAL_BELOW_SPAWN_MM``
    (a SIGNED float, millimetres) installs ``mdp.GoalBelowSpawnPoseCommand`` instead of
    ``GoalAtSpawnPoseCommand`` -- goal = the leg's spawn pose displaced ``delta`` along the BORE's
    own axis, not pinned exactly at spawn. The SIGN picks the direction along that one axis (bead
    UWLab-nnlv.3, which made this signed; it was unsigned and capped at +25mm before):

      * ``> 0`` -- goal displaced DEEPER INTO the bore. Upper bound +25mm, the bore's own engaged
        span, asserted (with its reasoning) in ``GoalBelowSpawnPoseCommand.__init__``; unchanged by
        the signed rewrite. This is the S1-shaped direction: it opposes the withdrawal the parent
        policy performs by default.
      * ``< 0`` -- goal displaced the OPPOSITE way along that SAME axis, i.e. OUT of the mouth,
        ABOVE it. Lower bound -200mm. This exists for the S2' rung (bead UWLab-nnlv), whose target
        band is 20-120mm above the mouth and which had no usable shaping knob at all while the
        value was unsigned -- the only one in the tree could push the goal deeper, mildly OPPOSING
        that band.
      * ``== 0`` (the default) -- NO shaping command is installed at all; ``goal_at_spawn``'s plain
        ``GoalAtSpawnPoseCommand`` is used, exactly as before this variant existed.

    See ``partial_assembly.py``'s "Y5" module comment for
    the shaping-device argument. Requires ``DEXLIFT_PARTIAL_ASSEMBLY=1`` (the fixture must already be
    in the scene for the command to read the bore's own axis off it -- asserted below, not just
    documented). Same env-var-not-Hydra-field reachability reasoning as every other whole-run toggle
    in this file (see ``_apply_c4_seating_training``'s "REACHABILITY" section): this value is read
    directly from ``os.environ`` here, inside ``__post_init__``, so there is no override-ordering
    window for a later ``env.foo=...`` to lose a race against.
    """
    partial_assembly = os.environ.get("DEXLIFT_PARTIAL_ASSEMBLY") == "1"
    _goal_below_spawn_mm_raw = os.environ.get("DEXLIFT_GOAL_BELOW_SPAWN_MM", "0") or "0"
    goal_below_spawn_mm = float(_goal_below_spawn_mm_raw)
    # -- SIGNED LOWER BOUND (bead UWLab-nnlv.3; was `>= 0.0`, which aborted env construction on any
    # negative value). ONLY the lower bound is checked here. The +25mm ceiling is deliberately NOT
    # duplicated at this level: it lives in GoalBelowSpawnPoseCommand.__init__ together with the
    # physical reasoning that justifies it, and moving/copying it here would change WHERE and WITH
    # WHAT MESSAGE an out-of-range POSITIVE value fails today -- which must stay byte-identical for
    # the tooling that already depends on it (regenerate_four_banks_post_finetune.sh, the C4 work).
    # -200mm is a POLICY bound, not a measured physical constant: the rung this sign was added for
    # (S2', bead UWLab-nnlv) tops out 120mm above the mouth, so 200mm is headroom around that band,
    # chosen to catch unit slips and typos rather than to describe any feature of the hardware.
    assert goal_below_spawn_mm >= -200.0, (
        f"DEXLIFT_GOAL_BELOW_SPAWN_MM={_goal_below_spawn_mm_raw!r} is below the signed lower bound"
        " of -200 (mm). This value displaces the COMMANDED goal from the leg's spawn pose along the"
        " bore's own axis: POSITIVE means DEEPER INTO the bore (ceiling +25mm, the bore's engaged"
        " span, asserted in GoalBelowSpawnPoseCommand.__init__), NEGATIVE means the opposite way"
        " along that same axis -- OUT of the mouth, ABOVE it (floor -200mm, headroom around the S2'"
        " rung's 20-120mm band), and 0 means no shaping command is installed at all."
    )
    # Signed: a NEGATIVE delta must install the command just as a positive one does (bead
    # UWLab-nnlv.3). `> 0.0` here silently meant "negative = feature off", which is precisely how
    # this change could half-apply. 0.0 alone still means "not installed", unchanged.
    goal_below_spawn = goal_below_spawn_mm != 0.0
    goal_at_spawn = partial_assembly or os.environ.get("DEXLIFT_GOAL_AT_SPAWN") == "1" or goal_below_spawn
    spawn_clearance = os.environ.get("DEXLIFT_SPAWN_CLEARANCE") == "1"

    if goal_below_spawn:
        assert partial_assembly, (
            "DEXLIFT_GOAL_BELOW_SPAWN_MM requires DEXLIFT_PARTIAL_ASSEMBLY=1 -- GoalBelowSpawnPoseCommand"
            " reads the bore's own 'deep' axis off scene.receptive_object (the fixture), which only"
            f" DEXLIFT_PARTIAL_ASSEMBLY=1 adds. Got DEXLIFT_GOAL_BELOW_SPAWN_MM={goal_below_spawn_mm}"
            f" DEXLIFT_PARTIAL_ASSEMBLY={os.environ.get('DEXLIFT_PARTIAL_ASSEMBLY')!r}."
        )

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
        # -- DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR (env var, distinct from the module CONSTANT of the
        # same base name it overrides) -- the same override mechanism added to _apply_episode_mixture
        # for the SAME reason: SpawnPartialAssembly.__init__ downloads partial_assemblies.pt for this
        # pair from mdp.DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR the moment it is constructed, and that
        # default (Hugging Face) path 404s for this exact pair -- confirmed 2026-08-20, see mdp/
        # episode_mixture.py's "THE MIXTURE IS OPT-IN" section. Every consumer of THIS legacy toggle
        # (generate_reset_states_policy.py's --reset_type ObjectPartiallyAssembledEEGrasped / C4,
        # cert scripts that might one day certify under it) hits the identical 404 without this.
        _partial_assembly_dataset_dir = os.environ.get(
            "DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR", mdp.DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR
        )
        env_cfg.events.reset_object = EventTerm(
            func=mdp.SpawnPartialAssembly,
            mode="reset",
            params={
                "dataset_dir": _partial_assembly_dataset_dir,
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
        # this block is reached whenever ANY of the three toggles asks for it.
        if goal_below_spawn:
            # -- Y5 (bead UWLab-xp05.3; signed by UWLab-nnlv.3): goal = spawn pose displaced along
            # the bore's own axis -- DEEPER into it for a positive delta, OUT of the mouth (above
            # it) for a negative one -- not pinned exactly at spawn. TABLE_LEG_USD_PATH (this
            # module's own constant, already in scope) is passed explicitly rather than duplicated
            # inside partial_assembly.py -- see GoalBelowSpawnPoseCommandCfg's docstring for why.
            env_cfg.commands.object_pose = mdp.upgrade_to_goal_below_spawn(
                env_cfg.commands.object_pose,
                leg_usd_path=TABLE_LEG_USD_PATH,
                delta_m=goal_below_spawn_mm / 1000.0,
            )

            # -- Y6 MID-EPISODE RESAMPLE GUARD (critic3, CONFIRMED via isaaclab source trace, bead
            # UWLab-xp05.3): CommandManager.compute() re-resamples via _resample_command whenever
            # time_left <= 0 (isaaclab command_manager.py:160-166), INDEPENDENT of episode reset.
            # The Reorient _PLAY classes' own resampling_time_range=(2.0,3.0) against
            # episode_length_s=4.0 (dexsuite_env_cfg.py, confirmed by reading it directly) means a
            # SECOND resample fires mid-episode (~t=2-3s), rebasing the goal onto wherever the leg
            # has ALREADY withdrawn to by then -- silently absorbing exactly the withdrawal this
            # arm exists to penalise, for the back half of every episode. The train class is
            # unaffected (resample 10s > episode 4s); this only bites the Play/generation path,
            # which is the one this toggle is actually used under. Force resampling_time_range
            # strictly past episode_length_s so exactly one resample happens, at reset.
            #
            # READ episode_length_s OFF THE CFG HERE, not hardcoded 4.0 -- but this is still a
            # SNAPSHOT taken at __post_init__ time: generate_reset_states_policy.py's own
            # --episode_length_s override mutates env_cfg.episode_length_s AFTER parse_env_cfg
            # (i.e. after this __post_init__ has already run), which would go stale against a
            # number fixed only here. GoalBelowSpawnPoseCommand.__init__ (partial_assembly.py)
            # independently RE-ASSERTS the same relation at manager-construction time (inside
            # gym.make, strictly after any such override), so a stale value set here still fails
            # loudly rather than silently mis-training/mis-generating -- but the generator script
            # should still re-derive this line's output if it changes episode_length_s afterward.
            _episode_length_s = float(env_cfg.episode_length_s)
            _min_resample_s = _episode_length_s + 1.0
            env_cfg.commands.object_pose.resampling_time_range = (_min_resample_s, _min_resample_s + 1.0)
            assert env_cfg.commands.object_pose.resampling_time_range[0] > _episode_length_s, (
                f"resampling_time_range {env_cfg.commands.object_pose.resampling_time_range} does "
                f"not clear episode_length_s={_episode_length_s}s"
            )
        else:
            env_cfg.commands.object_pose = mdp.upgrade_to_goal_at_spawn(env_cfg.commands.object_pose)

        # -- G3: name every toggle that is live and where the goal is coming from, so a
        # generation log states the configuration rather than leaving it to be inferred. A
        # silently-unset toggle here is a plausible wrong number, not an obvious one.
        reset_object_source = (
            f"SpawnPartialAssembly (dataset_dir={_partial_assembly_dataset_dir})" if partial_assembly
            else "reset_object_pose_with_clearance (DEXLIFT_SPAWN_CLEARANCE=1)" if spawn_clearance
            else "reset_root_state_uniform (dexsuite default pose_range)"
        )
        goal_source = (
            f"object spawn pose displaced {abs(goal_below_spawn_mm):.2f}mm"
            f" {'deeper into the bore' if goal_below_spawn_mm > 0.0 else 'out of the mouth (above it)'}"
            " along the bore's own axis"
            " (SHAPING DEVICE, not a target -- judge by banked depth, see GoalBelowSpawnPoseCommand's"
            " own docstring)"
            if goal_below_spawn
            else "object spawn pose (pinned, not uniform-sampled)"
        )
        # -- Y6 continued: echo the RESOLVED class name (not just "a toggle fired") and the
        # resampling numbers, so a future reader of the log can see directly that this specific
        # construction wired the toggle and got exactly one resample per episode -- see
        # generate_reset_states_policy.py's own post-parse_env_cfg guard for the complementary
        # check that fails loudly when the wrong --task means NONE of this ever runs at all.
        resample_banner = (
            f" resampling_time_range={tuple(env_cfg.commands.object_pose.resampling_time_range)}"
            f" vs episode_length_s={float(env_cfg.episode_length_s)}s (exactly one resample/episode)"
            if goal_below_spawn else ""
        )
        print(
            f"[dexlift] {type(env_cfg).__name__}:"
            f" DEXLIFT_PARTIAL_ASSEMBLY={int(partial_assembly)}"
            f" DEXLIFT_GOAL_AT_SPAWN={int(goal_at_spawn)}"
            f" DEXLIFT_GOAL_BELOW_SPAWN_MM={goal_below_spawn_mm}"
            f" DEXLIFT_SPAWN_CLEARANCE={int(spawn_clearance)}:"
            f" receptive_object {'ADDED' if partial_assembly else 'absent'}"
            + (
                f" at x={mdp.RECEPTIVE_POSE_RANGE['x']} y={mdp.RECEPTIVE_POSE_RANGE['y']}"
                f" z={mdp.RECEPTIVE_POSE_RANGE['z'][0]}"
                if partial_assembly else ""
            )
            + f"; reset_object -> {reset_object_source};"
            f" goal SOURCE = {goal_source};{resample_banner}", flush=True,
        )

    return partial_assembly or goal_at_spawn


def _apply_full_gravity(env_cfg) -> None:
    """Pin gravity at full magnitude for the REORIENT table-leg FINETUNE (epic UWLab-g3z4), instead
    of letting the ADR curriculum ramp it in from exactly zero.

    THE BUG THIS REPLACES. ``dexsuite``'s ``EventCfg.variable_gravity`` ships at
    ``((0,0,0),(0,0,0))`` (see ``dexsuite_env_cfg.py``'s own comment: gravity starts at zero
    "which removes the need for a special Lift reward") and ``curriculum.gravity_adr``
    (``adr_curriculum.py``) interpolates it toward full gravity via ``initial_final_interpolate_fn``
    as the ADR ``adr`` difficulty term rises. THAT SAME FUNCTION RETURNS ``NO_CHANGE`` BELOW
    DIFFICULTY FRACTION 0.1 (``dexsuite/mdp/curriculums.py``), so a finetune run that starts --
    and stays -- at or near the curriculum floor never leaves zero gravity: the leg floats and is
    never asked to fall, be caught, or be placed against a real weight. THIS IS EXACTLY THE FAILURE
    MODE ``dexlift_ur5e_delto_env_cfg.py``'s own gravity comment (search ``gravity: LEFT ALONE``)
    documents choosing to accept for the BASE Lift/primitives task, on measured evidence that a long
    enough run eventually escapes the floor. That argument does not transfer here: this finetune is
    deliberately run short and pinned near the curriculum floor, so "eventually" does not arrive.

    TWO EDITS ARE REQUIRED, AND EITHER ALONE IS WRONG -- the exact trap
    ``table_leg_env_cfg.TableLegCurriculumCfg``/``TableLegGraspLiftEnvCfg`` (the older UR10e table-leg
    task) already avoids, mirrored here:

    * Pinning ``events.variable_gravity.params["gravity_distribution_params"]`` to full gravity
      WITHOUT nulling ``curriculum.gravity_adr`` does nothing durable: the curriculum term re-derives
      and overwrites that same address every time it runs (every reset), snapping it back toward
      whatever ``initial_final_interpolate_fn`` computes from the current ADR difficulty -- back to
      ~zero at the floor this finetune sits near.
    * Nulling ``curriculum.gravity_adr`` WITHOUT also pinning ``events.variable_gravity`` freezes
      gravity permanently at dexsuite's UNTOUCHED default, which is exactly zero -- the curriculum
      term is the only thing that ever writes a non-zero value in the first place.

    Scoped to the Reorient table-leg classes only (called from both the TRAIN class and its
    ``_PLAY`` sibling, same reason ``_apply_partial_assembly_and_goal_toggles`` is): the base Lift
    and primitives tasks keep the reference ADR ramp, per the comment cited above.
    """
    full_gravity = ((0.0, 0.0, -9.81), (0.0, 0.0, -9.81))
    env_cfg.events.variable_gravity.params["gravity_distribution_params"] = full_gravity
    env_cfg.curriculum.gravity_adr = None

    # Fail loudly on a half-applied edit rather than silently training in vacuum -- see the
    # docstring's "TWO EDITS ARE REQUIRED" section for what each half alone would do wrong.
    written = env_cfg.events.variable_gravity.params["gravity_distribution_params"]
    assert tuple(tuple(v) for v in written) == full_gravity, (
        f"events.variable_gravity was not pinned to full gravity; got {written}"
    )
    assert env_cfg.curriculum.gravity_adr is None, (
        "curriculum.gravity_adr must be nulled, or it overwrites variable_gravity back toward zero"
        " on every reset"
    )
    print(f"[dexlift] gravity PINNED at {full_gravity} (curriculum.gravity_adr disabled)", flush=True)


def _apply_c4_seating_training(env_cfg) -> None:
    """C4 SEATING-AWARE TRAINING VARIANT (DELIVERABLE 2, team-lead ask). THREE INDEPENDENT opt-in
    toggles -- ``DEXLIFT_C4_SEATING_REWARD``, ``DEXLIFT_C4_GROSS_UNSEATING_TERM``, and
    ``DEXLIFT_C4_AXIAL_REWARD`` (follow-up, added after the first seating retrain's own measurement
    showed WHICH axis it left unsolved -- see ``mdp.axial_displacement_error_tanh``'s docstring) --
    layered on top of the existing ``DEXLIFT_PARTIAL_ASSEMBLY``/``DEXLIFT_GOAL_AT_SPAWN`` path. All
    three default OFF; this function returns immediately, touching nothing, when none is set.

    === THE PROBLEM THIS ANSWERS ===
    ``generate_reset_states_policy.py``'s ``held_with_probe`` gate has NO spatial term (see that
    module's docstring): it only asks "is the object held", never "is it still where a C4 state
    needs it". Probing the checkpoint this finetune is meant to improve on (25%-partial-assembly
    mixture, 30.03% acceptance) found 0/100 accepted states inside a seated depth band and 60% with
    the tip already back at or above the bore mouth -- the policy grasps well and withdraws the leg
    while doing it, because NOTHING in the current reward asks it not to. ``success_reward`` under
    ``DEXLIFT_GOAL_AT_SPAWN`` is technically already "reward matching the object's current pose to
    its own spawn pose" -- but at ``pos_std=0.1`` (10cm) / ``rot_std=0.5`` (~29deg), calibrated for
    an arbitrary full-workspace repose goal, the measured 14mm / 12.32deg median drift barely dents
    it. This is Reorient (``rot_std`` is set), so ``success_reward`` takes its MULTIPLICATIVE form,
    not Lift's squared position-only one: ``(1-tanh(0.014/0.1)) * (1-tanh(0.2150/0.5)) ~= 0.861 *
    0.595 ~= 0.51`` of max, weight 10 -> ~5.1 reward. The existing objective is satisfied by "roughly
    still in the neighbourhood", not "still seated in a 25mm bore".

    === THE FIX: A TIGHT-TOLERANCE, CONTACT-GATED "STILL SEATED" BONUS ===
    ``DEXLIFT_C4_SEATING_REWARD=1`` adds ONE new reward term, ``rewards.c4_seating_hold`` --
    REUSING ``mdp.success_reward`` (not reimplemented) against the SAME ``object_pose`` command,
    just at mm/deg tolerances matched to the bore instead of the workspace:
    ``pos_std=DEXLIFT_C4_SEATING_POS_STD_M`` (default 0.02 m = 20mm) and
    ``rot_std=DEXLIFT_C4_SEATING_ROT_STD_RAD`` (default 0.22 rad ~= 12.6deg).

    THESE DEFAULTS ARE CHOSEN FOR GRADIENT, NOT JUST FOR SCALE (team-lead correction after review --
    the first pass, pos_std=0.006/rot_std=0.15, was TOO TIGHT). ``d/dx[1-tanh(x/s)] =
    -sech^2(x/s)/s`` is maximized, for a FIXED operating-point error ``x``, near ``s = x/0.77`` --
    at ``x=14mm`` that is ``s~=18mm``. At the old ``s=6mm`` the policy sits at ``x/s=2.33``, deep in
    the saturated tail (``sech^2~=0.040``); at the new ``s=20mm`` it sits at ``x/s=0.7``
    (``sech^2~=0.635``), a ~16x steeper LOCAL gradient at the exact point the policy currently lives
    -- the training reward does not need to be as tight as DELIVERABLE 1's acceptance band (that
    gate enforces strictness separately, at generation time); it needs to be LEARNABLE. The level
    still discriminates properly: at these defaults, ``c4_seating_hold`` pays ``(1-tanh(0.014/0.02))
    * (1-tanh(0.2150/0.22)) ~= 0.396 * 0.248 ~= 0.098`` of max at the CURRENT 14mm/12.32deg
    operating point (weight 15 -> ~1.47 reward -- still well under ``success``'s ~5.1 there, as it
    should be: this is the term whose GRADIENT is supposed to pull the policy away from that point,
    not one that already dominates at it), rising to ~0.85 at a clean 3mm hold and falling back to
    ~0.09-0.36 as position error alone approaches the 15-30mm range ``terminations.gross_unseating``
    treats as unrecoverable (see below). Compare the FIRST-PASS defaults at the same 14mm/12.32deg
    point: ``(1-tanh(0.014/0.006)) * (1-tanh(0.2150/0.15)) ~= 0.0186 * 0.108 ~= 0.0020`` of max,
    weight 15 -> ~0.030 reward -- ~49x smaller AND in the flat part of the tail, i.e. barely
    distinguishable from the reward at 30mm+. That was the real defect the old defaults had: not
    merely "outweighed by success", but nearly gradient-dead exactly where training starts.

    This is the discriminating signal ``success_reward`` structurally cannot provide at ITS OWN
    calibration, added ALONGSIDE it (weight ``DEXLIFT_C4_SEATING_WEIGHT``, default 15.0 --
    deliberately the largest single term in this reward set while this flag is on, since "held
    without disturbing the seat" IS the task a specialist run exists for) rather than by retuning
    ``success``/``position_tracking``/``orientation_tracking`` in place, which stay exactly as
    calibrated for every OTHER episode kind and consumer of this env family (e.g. ``curriculum.adr``
    reads ``rewards.success.params``).

    ``mdp.success_reward`` is ALREADY contact-gated once ``rot_std`` is set (true here, Reorient
    only -- see that function's own docstring): ``(1-tanh(pos_err/pos_std)) *
    (1-tanh(rot_err/rot_std)) * contacts(...)``. THE DO-NOTHING HACK: a policy that never touches
    the leg scores EXACTLY 0 from this term, identically to how it already scores 0 from
    ``success``/``position_tracking``/``orientation_tracking``/``good_finger_contact`` today --
    ``contacts()`` is a hard multiplicative gate, not an additive bonus/penalty pair, so there is no
    "collect the low-disturbance reward without gripping" path. Contrast the OBVIOUS hack the brief
    names -- an UNCONDITIONAL penalty on ``|pos - spawn|`` -- which is minimized (0 cost) by NEVER
    TOUCHING the leg at all (an untouched, already-seated leg does not move on its own): that
    failure mode is why this is a positive, contact-gated BONUS layered on top of the existing
    contact-gated terms, not a penalty bolted on beside them.

    === THE SPECIALIST CALL ===
    This is intended to be trained with ``DEXLIFT_PARTIAL_ASSEMBLY=1`` (100% partial-assembly
    spawn + goal-at-spawn, the LEGACY deterministic toggle -- see
    ``_apply_partial_assembly_and_goal_toggles``), NOT ``DEXLIFT_EPISODE_MIXTURE=1``. Argued, not
    assumed: a C4-GENERATOR policy has no need for base-task (arbitrary-goal repose) competence --
    a separately certified checkpoint (ep3600) already drives C1/C2/C3 generation and remains the
    repose policy. The measured collapse this project guards against with
    ``classic_goal_prob > 0`` (55% of the skill gone in 50 epochs, 89% by 300, pass@30mm -> 0.0000)
    is specifically the loss of TRANSPORT competence when the objective stops containing it -- for
    a specialist that will never be asked to transport, that is not a cost, it is the point. The
    CURRENT 25%-mixture finetune is a cautionary data point FOR this call, not against it: it spent
    most of its gradient on the classic/low-goal fraction, certified 20 points below its parent on
    the very task it was still training for, and STILL only reached 30% partial-assembly
    acceptance with zero seated states -- a diluted objective bought neither goal. A further
    consideration structurally rules the mixture path out anyway:
    ``assert_episode_mixture_is_sane`` REQUIRES ``classic_goal_prob > 0``, so
    ``DEXLIFT_EPISODE_MIXTURE`` cannot express "100% partial-assembly" even if asked to -- the
    legacy toggle is the only path capable of a pure specialist run. Warm-starting from the
    existing 30%-acceptance finetune (already a real, if unseating, grasp policy) rather than the
    ep3600 base makes a further narrow specialist pass a REFINEMENT of standing grasp competence,
    not learning it from zero, which is what makes a short, narrow, single-objective run low-risk
    here in a way it was not for the original goal-at-spawn collapse (that regression trained a
    GENERAL policy to convergence under a narrowed objective; this is a short finetune of an
    ALREADY-NARROW specialist that never needs to generalize back).

    === THE EARLY-TERMINATION QUESTION ===
    ``DEXLIFT_C4_GROSS_UNSEATING_TERM=1`` adds ``terminations.gross_unseating``
    (``mdp.gross_seating_loss`` -- see that function's own docstring) plus the SAME
    ``is_terminated_term`` penalty pattern this env already uses for ``abnormal_robot``
    (``rewards.early_termination``, weight -1): ``rewards.c4_gross_unseating_penalty``, weight
    ``DEXLIFT_C4_GROSS_PENALTY_WEIGHT`` (default -2.0). Recommended ON alongside the reward, WITH
    A CAVEAT this project has hit before: an early termination cannot move a score whose predicate
    samples the TERMINAL state -- if some downstream evaluation ever snapshots "the object's pose
    at episode end" as its metric, adding this termination makes that snapshot MORE likely to catch
    a mid-failure state (episodes that would have drifted back toward center given more time now
    end at their worst point instead), which would look like a regression that is really just a
    sampling artifact of when the snapshot was taken. This is NOT what this termination is used
    for: ``generate_reset_states_policy.py --c4_seating_gate`` (DELIVERABLE 1) filters on ONE
    reset-worthy state per accepted episode, chosen by ``held_with_probe``'s own probe logic, not a
    terminal-state snapshot -- so THAT metric is not exposed to this trap. What this termination
    DOES do, legitimately: end an already-lost episode sooner (faster credit assignment, more
    env-steps/sec of USEFUL rollout, mirroring exactly why ``abnormal_robot`` already exists as a
    termination rather than only ever being left to time out). Do not, in future work, repurpose
    ``gross_unseating``'s firing RATE as a training-progress metric -- use held-out generation runs
    for that, for the reason above.

    === REACHABILITY ===
    Every new numeric knob here (``DEXLIFT_C4_SEATING_POS_STD_M``, ``_ROT_STD_RAD``, ``_WEIGHT``,
    ``DEXLIFT_C4_GROSS_POS_THRESHOLD_M``, ``_ROT_THRESHOLD_RAD``, ``_PENALTY_WEIGHT``) is read from
    ``os.environ`` HERE, inside ``__post_init__``, exactly like every other whole-run toggle in this
    file (``DEXLIFT_SPAWN_CLEARANCE``, ``DEXLIFT_POSE_TILT``, ``DEXLIFT_PARTIAL_ASSEMBLY`` itself)
    -- NOT as Hydra-overridable dataclass fields, deliberately: that trap
    (``_apply_episode_mixture``'s own docstring) is specific to fields captured into
    ``EventTermCfg.params`` inside ``__post_init__``, which runs BEFORE Hydra CLI overrides land on
    ``env_cfg`` -- a dataclass-field default snapshotted there is frozen forever regardless of a
    later ``env.foo=...`` override. An env var has no such window: it is resolved from the process
    environment at ``export FOO=1`` time, before Python even starts, so reading it here or reading
    it anywhere else in the process produces the identical value -- there is nothing for a later
    override to race against.

    CALLED FROM BOTH the TRAIN class and its ``_PLAY`` sibling below, AFTER
    ``_apply_partial_assembly_and_goal_toggles`` -- same reason every other toggle pair in this file
    is called from both (they are not in an inheritance relationship; see that function's own
    docstring), and the ordering matters: this function's precondition assert reads
    ``os.environ`` directly rather than that call's return value (which collapses ``partial_assembly``
    and ``goal_at_spawn`` into one bool), but it must still run AFTER that call so
    ``env_cfg.commands.object_pose`` has already been upgraded to ``GoalAtSpawnPoseCommand`` by the
    time anything downstream reads it.
    """
    seating_reward_requested = os.environ.get("DEXLIFT_C4_SEATING_REWARD") == "1"
    gross_term_requested = os.environ.get("DEXLIFT_C4_GROSS_UNSEATING_TERM") == "1"
    # AXIAL DISPLACEMENT term (follow-up, team-lead diagnosis 2026-08-21): independent opt-in, same
    # as the two above -- see mdp.axial_displacement_error_tanh's own docstring for the full
    # argument (c4_seating_hold's isotropic pos_dist barely discriminates axial error once radial
    # error is already small; this term gives the FAILING axis its own tolerance and weight).
    axial_reward_requested = os.environ.get("DEXLIFT_C4_AXIAL_REWARD") == "1"
    if not (seating_reward_requested or gross_term_requested or axial_reward_requested):
        return

    goal_at_spawn = os.environ.get("DEXLIFT_GOAL_AT_SPAWN") == "1" or os.environ.get("DEXLIFT_PARTIAL_ASSEMBLY") == "1"
    assert goal_at_spawn, (
        "DEXLIFT_C4_SEATING_REWARD/DEXLIFT_C4_GROSS_UNSEATING_TERM/DEXLIFT_C4_AXIAL_REWARD require"
        " DEXLIFT_GOAL_AT_SPAWN=1 (implied by DEXLIFT_PARTIAL_ASSEMBLY=1) -- without it,"
        " commands.object_pose is a uniformly sampled goal, not the leg's own spawn/seated pose, and"
        " this reward/termination would be rewarding/penalizing proximity to an arbitrary point"
        " instead of 'stayed seated'."
        f" Got DEXLIFT_GOAL_AT_SPAWN={os.environ.get('DEXLIFT_GOAL_AT_SPAWN')!r}"
        f" DEXLIFT_PARTIAL_ASSEMBLY={os.environ.get('DEXLIFT_PARTIAL_ASSEMBLY')!r}."
    )

    if seating_reward_requested:
        # Defaults 0.02 / 0.22 (NOT 0.006 / 0.15 -- see this function's own docstring, "THESE
        # DEFAULTS ARE CHOSEN FOR GRADIENT" section, for the tanh-slope argument this correction is
        # based on). Widened in the DEFAULT, not only in launch-command guidance: an env var that
        # must be remembered to avoid a bad value is a trap -- a forgotten export would silently
        # give the unlearnable, saturated-tail version back.
        pos_std = float(os.environ.get("DEXLIFT_C4_SEATING_POS_STD_M", "0.02"))
        rot_std = float(os.environ.get("DEXLIFT_C4_SEATING_ROT_STD_RAD", "0.22"))
        weight = float(os.environ.get("DEXLIFT_C4_SEATING_WEIGHT", "15.0"))
        # Set as a plain instance attribute on the already-constructed RewardsCfg -- same pattern
        # generate_reset_states_policy.py already uses for env_cfg.terminations.success: managers
        # discover terms via self.cfg.__dict__.items(), not dataclass field introspection, so a
        # dynamically added attribute is picked up identically to a declared field.
        env_cfg.rewards.c4_seating_hold = RewTerm(
            func=mdp.success_reward,
            weight=weight,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "align_asset_cfg": SceneEntityCfg("object"),
                "command_name": "object_pose",
                "pos_std": pos_std,
                "rot_std": rot_std,
                "thumb_contact_name": THUMB_TIP_NAMES,
                "tip_contact_names": TIP_NAMES,
                "threshold": 1.0,  # matches rewards.success's own gate strength, not the looser 0.2N shaping terms
            },
        )
        print(
            f"[dexlift] C4 SEATING REWARD wired: rewards.c4_seating_hold weight={weight} "
            f"pos_std={pos_std}m rot_std={rot_std}rad (mdp.success_reward reused, contact-gated at 1.0N)",
            flush=True,
        )

    if gross_term_requested:
        pos_threshold = float(os.environ.get("DEXLIFT_C4_GROSS_POS_THRESHOLD_M", "0.03"))
        rot_threshold = float(os.environ.get("DEXLIFT_C4_GROSS_ROT_THRESHOLD_RAD", "0.5"))
        penalty_weight = float(os.environ.get("DEXLIFT_C4_GROSS_PENALTY_WEIGHT", "-2.0"))
        env_cfg.terminations.gross_unseating = DoneTerm(
            func=mdp.gross_seating_loss,
            params={
                "command_name": "object_pose",
                "asset_cfg": SceneEntityCfg("robot"),
                "align_asset_cfg": SceneEntityCfg("object"),
                "pos_threshold": pos_threshold,
                "rot_threshold": rot_threshold,
            },
        )
        # Same is_terminated_term pattern already used for rewards.early_termination/abnormal_robot.
        env_cfg.rewards.c4_gross_unseating_penalty = RewTerm(
            func=mdp.is_terminated_term, weight=penalty_weight, params={"term_keys": "gross_unseating"}
        )
        print(
            f"[dexlift] C4 GROSS-UNSEATING TERMINATION wired: terminations.gross_unseating "
            f"pos_threshold={pos_threshold}m rot_threshold={rot_threshold}rad, "
            f"rewards.c4_gross_unseating_penalty weight={penalty_weight} "
            "(SHAPING signal only -- see this function's own docstring on the terminal-state-sampling trap)",
            flush=True,
        )

    if axial_reward_requested:
        # STRONGER precondition than the shared goal_at_spawn assert above: this term reads
        # env.scene["receptive_object"]'s LIVE orientation every step (mdp.axial_displacement_error_
        # tanh's own "AXIS SOURCE" section), which only exists when DEXLIFT_PARTIAL_ASSEMBLY=1 added
        # it (_apply_partial_assembly_and_goal_toggles, called before this function). goal_at_spawn
        # alone (DEXLIFT_GOAL_AT_SPAWN=1 without DEXLIFT_PARTIAL_ASSEMBLY=1) satisfies the shared
        # assert above but would NOT add the fixture -- checking the ACTUAL constructed scene here,
        # not just the env var, so a misconfigured launch fails loudly at cfg-construction time
        # rather than with a scene-entity KeyError deep inside the first training step.
        assert hasattr(env_cfg.scene, "receptive_object"), (
            "DEXLIFT_C4_AXIAL_REWARD=1 requires scene.receptive_object (the fixture) to already be"
            " present -- only DEXLIFT_PARTIAL_ASSEMBLY=1 adds it; DEXLIFT_GOAL_AT_SPAWN=1 alone does"
            " not. mdp.axial_displacement_error_tanh reads the fixture's live orientation every step"
            " to project displacement onto its insertion axis, and has nothing to read without it."
        )
        axial_std = float(os.environ.get("DEXLIFT_C4_AXIAL_STD_M", "0.005"))
        axial_weight = float(os.environ.get("DEXLIFT_C4_AXIAL_WEIGHT", "20.0"))
        env_cfg.rewards.c4_axial_displacement_hold = RewTerm(
            func=mdp.axial_displacement_error_tanh,
            weight=axial_weight,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "align_asset_cfg": SceneEntityCfg("object"),
                "receptive_object_cfg": SceneEntityCfg("receptive_object"),
                "command_name": "object_pose",
                "std": axial_std,
                "thumb_contact_name": THUMB_TIP_NAMES,
                "tip_contact_names": TIP_NAMES,
                "threshold": 1.0,  # matches c4_seating_hold's own gate strength
            },
        )
        print(
            f"[dexlift] C4 AXIAL DISPLACEMENT REWARD wired: rewards.c4_axial_displacement_hold "
            f"weight={axial_weight} std={axial_std}m (mdp.axial_displacement_error_tanh, "
            "contact-gated at 1.0N, DISPLACEMENT from spawn along the fixture's live insertion axis"
            " -- NOT the generator's absolute depth-from-mouth; see that function's own docstring)",
            flush=True,
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


def _apply_episode_mixture(env_cfg) -> None:
    """Wire the per-episode mixture MECHANISM (epic UWLab-g3z4) over {classic goal, low goal,
    partial-assembly grasp-only} into ``env_cfg`` -- see ``mdp/episode_mixture.py``'s module
    docstring for the full design argument (shared per-env kind buffer, the fixture-parking fix, why
    the classic fraction may never reach zero, why this does not replace the legacy whole-run
    toggles).

    DELIBERATELY DOES NOT TOUCH THE FRACTIONS, and does not validate them either. An earlier revision
    read ``env_cfg.classic_goal_prob`` etc. HERE and both baked them as literal floats into
    ``EventTerm.params`` and called ``assert_episode_mixture_is_sane`` on them -- at ``__post_init__``
    time, i.e. before Hydra CLI overrides ever reach ``env_cfg``. That made any
    ``env.classic_goal_prob=...`` sweep a silent no-op: the baked params dict kept the pre-override
    defaults forever, and the assert had already passed on those same defaults before a bad override
    could exist to reject. ``mdp.MixtureResetObject.__init__`` now reads
    ``env.cfg.classic_goal_prob`` / ``env.cfg.low_goal_prob`` / ``env.cfg.partial_assembly_prob``
    itself, at MANAGER-CONSTRUCTION time (inside ``gym.make``, strictly after
    ``env_cfg.from_dict(...)`` has applied any override) and calls
    ``assert_episode_mixture_is_sane`` there -- see that class's docstring and the ``mdp/
    episode_mixture.py`` module docstring's "THE MIXTURE PROBABILITIES ARE READ AT TERM-CONSTRUCTION
    TIME" section for the full argument. Nothing here can validate what the fractions will actually
    be, because at ``__post_init__`` time they are not yet known.

    SKIPPED WHEN A LEGACY TOGGLE FIRED. Callers must check ``_apply_partial_assembly_and_goal_toggles``'s
    return value themselves and not call this function at all in that case -- see that function's
    docstring for why (reset-state generation / certification tooling needs the deterministic 100%
    path, which a probabilistic mixture cannot give it).

    OPT-IN, NOT DEFAULT-ON: callers must ALSO check ``DEXLIFT_EPISODE_MIXTURE == "1"`` themselves and
    not call this function at all when it is unset -- see ``mdp/episode_mixture.py``'s module
    docstring, "THE MIXTURE IS OPT-IN" section, for the regression this closes. An earlier revision
    called this function unconditionally whenever no legacy toggle fired, which silently wired the
    mixture into EVERY construction of these classes -- including ordinary
    ``generate_reset_states_policy.py`` runs that set neither legacy env var because they never
    wanted partial-assembly at all, and crashed them (``MixtureResetObject.__init__() got an
    unexpected keyword argument 'dataset_dir'`` -- a swallowed HTTPError 404 one level up, see the
    same docstring section). The legacy-toggle check alone cannot express "I don't want the mixture
    either" -- both are OFF for an ordinary run -- so a second, independent, explicit opt-in is
    required.
    """
    # The fixture is needed unconditionally: even a CLI override that raises partial_assembly_prob
    # above its 0.25 default must find scene.receptive_object already present.
    env_cfg.scene.receptive_object = mdp.make_dexlift_receptive_object_cfg()

    base_pose_range = dict(env_cfg.events.reset_object.params["pose_range"])
    base_velocity_range = dict(
        env_cfg.events.reset_object.params.get("velocity_range", {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]})
    )
    # -- DEXLIFT_EPISODE_MIXTURE_DATASET_DIR overrides the default (Hugging Face) dataset root,
    # readable from a plain env var rather than only a Hydra dotlist override -- train.py's
    # `env.events.reset_object.params.dataset_dir=...` still works too and, applied later
    # (post-construction, via Hydra), wins over this if both are set. The env var exists because
    # NOT every consumer of this cfg goes through hydra_task_config: scripts_v2/tools/certification/
    # certify_pose.py calls parse_env_cfg directly and has no override mechanism of its own, so a
    # plain `export DEXLIFT_EPISODE_MIXTURE_DATASET_DIR=...` (as cert_g3z4_finetune.sh's run_certify.sh
    # delegate does) is the only way to point it at a local copy. Needed because the class default
    # (mdp.DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR) 404s for this pair today -- see mdp/episode_mixture.py's
    # "THE MIXTURE IS OPT-IN" section for how that was found.
    _dataset_dir = os.environ.get("DEXLIFT_EPISODE_MIXTURE_DATASET_DIR", mdp.DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR)
    env_cfg.events.reset_object = EventTerm(
        func=mdp.MixtureResetObject,
        mode="reset",
        params={
            "dataset_dir": _dataset_dir,
            "insertive_object_cfg": SceneEntityCfg("object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "fixture_pose_range": mdp.RECEPTIVE_POSE_RANGE,
            # -- CLASSIC / LOW GOAL spawn: whatever reset_object already carried (narrowed x
            # included) -- see MixtureResetObject.__call__, only the PARTIAL ASSEMBLY fraction spawns
            # differently. NOTE: the mixture fractions themselves are NOT here -- see this function's
            # docstring; MixtureResetObject reads them off env.cfg at its own __init__ instead.
            "pose_range": base_pose_range,
            "velocity_range": base_velocity_range,
            # No extra jitter on top of the stored partial-assembly relative pose, matching
            # SpawnPartialAssembly's own default.
            "pose_range_b": {},
        },
    )

    env_cfg.commands.object_pose = mdp.upgrade_to_episode_mixture(env_cfg.commands.object_pose)

    print(
        "[dexlift] episode mixture MECHANISM wired (fractions validated later, post-override, in"
        f" MixtureResetObject.__init__); low goal pos_z={mdp.LOW_GOAL_POS_Z_RANGE} m, classic goal"
        f" pos_z={tuple(env_cfg.commands.object_pose.ranges.pos_z)} m; partial-assembly dataset_dir="
        f"{_dataset_dir}", flush=True,
    )


def _apply_c3_rung_stage(env_cfg, legacy_toggle_active: bool) -> bool:
    """C3 RUNG stage -- **C3 = 50% S1 + 50% S_t** (``RESET_SPEC_V2.md`` sec 1 C3, bead ``dr-ai1.4``).

    **OFF unless ``DEXRESET_C3_RUNG=1``.** Returns whether it fired. With the variable unset the
    default path is byte-identical to what it was before this function existed -- nothing below runs,
    no term is replaced, no banner is printed. Same opt-in idiom as ``DEXRESET_C1_HAND`` /
    ``DEXLIFT_PARTIAL_ASSEMBLY`` / ``DEXLIFT_EPISODE_MIXTURE``.

    See ``mdp/c3_rung_core.py``'s module docstring for the design argument in full (what S1 and S_t
    each are, why S_t's peg is HORIZONTAL and needs no spawn change, why this is a whole-run stage
    rather than episode-mixture fractions, and the F49 frame rule) and ``mdp/c3_rung.py`` for the
    two terms this installs. In one paragraph:

    * **S1** -- partial-assembly spawn (leg pre-inserted, hence tip-down), goal displaced a shallow
      ``DEXRESET_C3_S1_GOAL_DELTA_MM`` deeper along the bore's own axis, orientation unchanged.
    * **S_t** -- the ORDINARY table spawn, **unchanged**, goal pinned at the leg's own pose with
      ZERO delta in position and orientation.

    REFUSES RATHER THAN LOSES A RACE. Both the legacy whole-run toggles and the episode mixture
    replace ``events.reset_object`` and ``commands.object_pose`` -- the same two slots this stage
    needs. Whoever ran last would silently win, and the run would train or generate under a staging
    its own launch script does not describe. That is Trap 3 in ``RESET_SPEC_V2.md`` sec 1a ("an env
    toggle can silently override a hydra override ... more than one v1 conclusion turned out to
    concern a variable that never took effect"), and it is the single most repeated defect in this
    campaign's record. So a conflicting combination raises here, at config time, before Isaac starts,
    naming both variables -- it is not resolved by precedence and not warned about and continued.

    THE Y6 MID-EPISODE RESAMPLE GUARD IS NOT OPTIONAL HERE. ``CommandManager.compute()`` resamples
    whenever ``time_left <= 0``, independent of episode reset, and the ``_PLAY``/generation classes'
    ``resampling_time_range=(2.0, 3.0)`` against ``episode_length_s=4.0`` fires a second resample
    mid-episode. For a goal derived from the leg's own live pose that rebases the target onto
    wherever the leg has been carried to -- which for S_t means rewarding the policy for holding the
    leg ANYWHERE, i.e. the rung's entire content. Forced past ``episode_length_s`` below, and
    RE-ASSERTED at manager-construction time in ``C3RungGoalPoseCommand.__init__`` because
    ``generate_reset_states_policy.py``'s ``--episode_length_s`` override lands after this runs and
    would go stale against a number fixed only here. Same defect, same fix, same two-place structure
    as the ``goal_below_spawn`` branch of ``_apply_partial_assembly_and_goal_toggles``.
    """
    staging = c3_rung_core.parse_c3_rung_env(os.environ)
    if staging is None:
        return False

    if legacy_toggle_active:
        raise ValueError(
            "DEXRESET_C3_RUNG=1 conflicts with a legacy whole-run toggle"
            f" (DEXLIFT_PARTIAL_ASSEMBLY={os.environ.get('DEXLIFT_PARTIAL_ASSEMBLY')!r},"
            f" DEXLIFT_GOAL_AT_SPAWN={os.environ.get('DEXLIFT_GOAL_AT_SPAWN')!r},"
            f" DEXLIFT_GOAL_BELOW_SPAWN_MM={os.environ.get('DEXLIFT_GOAL_BELOW_SPAWN_MM')!r}). Both"
            " replace events.reset_object and commands.object_pose, so one would silently overwrite"
            " the other and the run would not be staged as its launcher describes. Unset one."
            " NOTE the legacy pair is not redundant with this stage: DEXLIFT_PARTIAL_ASSEMBLY=1 plus"
            " DEXLIFT_GOAL_BELOW_SPAWN_MM=5 is a pure-S1 run and DEXLIFT_GOAL_AT_SPAWN=1 alone is a"
            " pure-S_t run -- what they cannot do, and what this stage exists for, is draw BETWEEN"
            " the two halves within one run."
        )
    if os.environ.get("DEXLIFT_EPISODE_MIXTURE") == "1":
        raise ValueError(
            "DEXRESET_C3_RUNG=1 conflicts with DEXLIFT_EPISODE_MIXTURE=1. Both replace"
            " events.reset_object and commands.object_pose. They are also not interchangeable:"
            " assert_episode_mixture_is_sane REQUIRES classic_goal_prob > 0 (guarding a measured"
            " collapse -- 55% of the skill gone in 50 epochs, 89% by 300, pass@30mm 0.0000 -- when"
            " the objective stops containing the transport task), so the mixture structurally cannot"
            " express C3 = 50% S1 + 50% S_t, which leaves the classic fraction at zero. That guard is"
            " not weakened; this stage is the separate, deterministic path instead. Unset one."
        )

    # The fixture is required by the S1 half (the leg is composed against it) and is what the S_t
    # half parks out of the way, so it is added unconditionally -- exactly as
    # _apply_episode_mixture and the legacy partial-assembly toggle both do.
    env_cfg.scene.receptive_object = mdp.make_dexlift_receptive_object_cfg()

    # -- S_t's spawn is WHATEVER reset_object already carried (narrowed x, staged drop height and
    # tilt included), captured here and passed straight through. This is the "S_t needs NO spawn
    # change" requirement expressed in code: nothing in this function narrows, recentres or
    # reorients it, and DEXRESET_ST_SPAWN_TIPDOWN is not read anywhere in this stage.
    base_pose_range = dict(env_cfg.events.reset_object.params["pose_range"])
    base_velocity_range = dict(
        env_cfg.events.reset_object.params.get("velocity_range", {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]})
    )
    # Same override, same reason, as _apply_episode_mixture's: the class default (Hugging Face) path
    # 404s for this pair, and not every consumer of this cfg goes through hydra_task_config, so a
    # plain env var is the only way some of them can point at a local copy.
    _dataset_dir = os.environ.get(
        "DEXLIFT_EPISODE_MIXTURE_DATASET_DIR",
        os.environ.get("DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR", mdp.DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR),
    )
    env_cfg.events.reset_object = EventTerm(
        func=mdp.C3RungResetObject,
        mode="reset",
        params={
            "dataset_dir": _dataset_dir,
            "insertive_object_cfg": SceneEntityCfg("object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "fixture_pose_range": mdp.RECEPTIVE_POSE_RANGE,
            "pose_range": base_pose_range,
            "velocity_range": base_velocity_range,
            "s1_fraction": staging.s1_fraction,
            # No extra jitter on top of the stored partial-assembly relative pose, matching
            # SpawnPartialAssembly's own default.
            "pose_range_b": {},
        },
    )

    env_cfg.commands.object_pose = mdp.upgrade_to_c3_rung(
        env_cfg.commands.object_pose, s1_goal_delta_m=staging.s1_goal_delta_m
    )

    # -- Y6, see this function's docstring. Read episode_length_s off the cfg rather than hardcoding
    # it; C3RungGoalPoseCommand.__init__ re-asserts the same relation after any later override.
    _episode_length_s = float(env_cfg.episode_length_s)
    _min_resample_s = _episode_length_s + 1.0
    env_cfg.commands.object_pose.resampling_time_range = (_min_resample_s, _min_resample_s + 1.0)
    assert env_cfg.commands.object_pose.resampling_time_range[0] > _episode_length_s, (
        f"resampling_time_range {env_cfg.commands.object_pose.resampling_time_range} does not clear"
        f" episode_length_s={_episode_length_s}s"
    )

    # R5: the run must STATE its staging. The banner text is built (and asserted on) in
    # c3_rung_core, so what a log shows and what a test checks cannot drift apart.
    print(c3_rung_core.c3_rung_banner(staging), flush=True)
    print(
        f"[dexreset] C3 RUNG wiring: reset_object -> C3RungResetObject (dataset_dir={_dataset_dir}),"
        " commands.object_pose -> C3RungGoalPoseCommand, resampling_time_range="
        f"{tuple(env_cfg.commands.object_pose.resampling_time_range)} vs"
        f" episode_length_s={_episode_length_s}s (exactly one resample per episode -- a second one"
        " would rebase the goal onto the carried leg and destroy S_t).",
        flush=True,
    )
    return True


@configclass
class DexLiftUR5eDeltoRelJointPosTableLegReorientEnvCfg(
    Ur5eDeltoTableLegRelJointPosMixinCfg, dexsuite.DexsuiteReorientEnvCfg
):
    # -- Per-episode mixture fractions (epic UWLab-g3z4). Literal dataclass fields, not env-var
    # switches like the two above: ``update_class_from_dict`` (IsaacLab's hydra override applier)
    # refuses any key not already present in the dataclass, so these have to exist here with
    # defaults before ``env.classic_goal_prob=...`` etc. can ever be set from the CLI.
    #
    # THESE DEFAULTS ARE NOT VALIDATED HERE, AND NOTHING IN ``__post_init__`` READS THEM EITHER --
    # deliberately. Hydra applies CLI overrides to this cfg object AFTER ``__post_init__`` has already
    # run (``env_cfg.from_dict(...)`` in ``isaaclab_tasks.utils.hydra``), so anything that captured
    # ``self.classic_goal_prob`` etc. here would capture a pre-override snapshot forever -- exactly
    # the bug this structure now avoids. The values actually drawn from are read straight off
    # ``env.cfg`` by ``mdp.MixtureResetObject.__init__`` at manager-construction time (inside
    # ``gym.make``, after overrides land), and validated by ``assert_episode_mixture_is_sane`` THERE.
    # Defaults must still sum to 1.0 with ``classic_goal_prob > 0`` -- that assert enforces it on
    # every construction, override or not; see its docstring for the measured collapse it guards
    # against.
    classic_goal_prob: float = 0.50
    low_goal_prob: float = 0.25
    partial_assembly_prob: float = 0.25
    # SIGNED metres (bead UWLab-nnlv.5). 0.0 = the original goal-AT-spawn behaviour, where a
    # partial-assembly episode has tracking satisfied at t=0 and therefore carries NO GRADIENT.
    # Negative displaces the goal back OUT of the bore mouth -- the S2 rung target -- so those
    # episodes teach something. Bounds [-0.200, +0.025] enforced in MixtureGoalPoseCommand.
    partial_goal_delta_m: float = 0.0
    # -- TRANSPORT GOAL branch (bead dr-ai1.13, V2_POSE_FINDINGS.md F43): tip-down +- tilt, x/y
    # anchored to the object's own spawn -- see mdp.c3_transport_core's module docstring and
    # mdp.episode_mixture.MixtureGoalPoseCommand._resample_transport. 0.0 keeps this branch OFF by
    # default, same idiom as partial_goal_delta_m above -- an existing run's mixture is unchanged
    # unless a caller explicitly raises this. Validated (with the other three fractions) by
    # assert_episode_mixture_is_sane inside MixtureResetObject.__init__.
    transport_goal_prob: float = 0.0
    # Half-width, radians, of the roll/pitch band around the tip-down nominal (yaw pinned to 0).
    # Same default (0.35 rad = 20 deg) _apply_goal_vertical_mixture shipped for the analogous band
    # -- mdp.c3_transport_core.DEFAULT_TRANSPORT_GOAL_TILT_RAD, inlined here rather than imported
    # (the *_core modules are deliberately not exposed through the ``mdp`` namespace -- same choice
    # ``_apply_c1_hand_pose_stage`` made for ``c1_hand_pose_core``). Ignored while
    # transport_goal_prob == 0.0.
    transport_goal_tilt: float = 0.35
    # Root-frame z band (metres) the transport goal's height is drawn from. Default (0.13, 0.27) is
    # the same band _apply_goal_vertical_mixture shipped for DEXLIFT_GOAL_VERTICAL_Z -- tip 24-164 mm
    # above the work surface (mdp.c3_transport_core.DEFAULT_TRANSPORT_GOAL_Z_RANGE_M). Ignored while
    # transport_goal_prob == 0.0.
    transport_goal_z: tuple[float, float] = (0.13, 0.27)

    # -- PARTIALLY-ASSEMBLED SPAWN / GOAL-AT-SPAWN toggles: see
    # ``_apply_partial_assembly_and_goal_toggles``'s docstring above for the full argument (why two
    # independent toggles, why a shared function rather than inline code, why Lift is excluded).
    # Called from BOTH this class and its ``_PLAY`` sibling below -- they are NOT in an inheritance
    # relationship with each other, so the call has to be written twice, once per class, or Play
    # never sees it (bead UWLab-qiao.9/J -- this is precisely the bug being fixed here). GRAVITY
    # (``_apply_full_gravity``) is unconditional -- always intended for this task family regardless of
    # caller. The EPISODE MIXTURE MECHANISM (``_apply_episode_mixture``) is OPT-IN, gated behind
    # DEXLIFT_EPISODE_MIXTURE=1 -- see _apply_episode_mixture's own docstring for the regression that
    # made this a required second gate, independent of the legacy-toggle check: an ordinary
    # generate_reset_states_policy.py run sets neither legacy env var (it never wanted
    # partial-assembly) NOR this new one (it never wanted the mixture either) -- both must be checked,
    # since "no legacy toggle fired" alone cannot distinguish "wants the mixture" from "wants neither".
    def __post_init__(self):
        super().__post_init__()
        _apply_full_gravity(self)
        legacy_toggle_active = _apply_partial_assembly_and_goal_toggles(self)
        # DELIVERABLE 2 (C4 seating-aware training variant): see _apply_c4_seating_training's own
        # docstring. Called after the toggles above so commands.object_pose is already
        # GoalAtSpawnPoseCommand by the time this function's precondition assert runs.
        _apply_c4_seating_training(self)
        # C3 RUNG stage (bead dr-ai1.4): 50% S1 + 50% S_t, off unless DEXRESET_C3_RUNG=1. Called
        # BEFORE the episode-mixture branch below and given legacy_toggle_active, because it
        # REFUSES (raises) rather than silently losing or winning a race for events.reset_object /
        # commands.object_pose -- see _apply_c3_rung_stage's docstring. Written out in both Reorient
        # classes for the same reason every other line in this block is: they are NOT in an
        # inheritance relationship with each other (bead UWLab-qiao.9/J).
        _apply_c3_rung_stage(self, legacy_toggle_active)
        episode_mixture_requested = os.environ.get("DEXLIFT_EPISODE_MIXTURE") == "1"
        if episode_mixture_requested and not legacy_toggle_active:
            _apply_episode_mixture(self)
        elif episode_mixture_requested and legacy_toggle_active:
            print(
                "[dexlift] DEXLIFT_EPISODE_MIXTURE=1 requested but a legacy toggle"
                " (DEXLIFT_PARTIAL_ASSEMBLY/DEXLIFT_GOAL_AT_SPAWN) fired first -- the legacy,"
                " deterministic whole-run path wins; the mixture is NOT installed on top of it.",
                flush=True,
            )


@configclass
class DexLiftUR5eDeltoRelJointPosTableLegReorientEnvCfg_PLAY(
    Ur5eDeltoTableLegRelJointPosMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY
):
    # -- Same fields and toggles as the train class above, same reasons -- see its docstring. This
    # class does NOT inherit from ``DexLiftUR5eDeltoRelJointPosTableLegReorientEnvCfg``, so nothing
    # above reaches here on its own; both the fields and the __post_init__ calls have to be repeated.
    classic_goal_prob: float = 0.50
    low_goal_prob: float = 0.25
    partial_assembly_prob: float = 0.25
    # SIGNED metres (bead UWLab-nnlv.5). 0.0 = the original goal-AT-spawn behaviour, where a
    # partial-assembly episode has tracking satisfied at t=0 and therefore carries NO GRADIENT.
    # Negative displaces the goal back OUT of the bore mouth -- the S2 rung target -- so those
    # episodes teach something. Bounds [-0.200, +0.025] enforced in MixtureGoalPoseCommand.
    partial_goal_delta_m: float = 0.0
    # -- TRANSPORT GOAL branch (bead dr-ai1.13, V2_POSE_FINDINGS.md F43): tip-down +- tilt, x/y
    # anchored to the object's own spawn -- see mdp.c3_transport_core's module docstring and
    # mdp.episode_mixture.MixtureGoalPoseCommand._resample_transport. 0.0 keeps this branch OFF by
    # default, same idiom as partial_goal_delta_m above -- an existing run's mixture is unchanged
    # unless a caller explicitly raises this. Validated (with the other three fractions) by
    # assert_episode_mixture_is_sane inside MixtureResetObject.__init__.
    transport_goal_prob: float = 0.0
    # Half-width, radians, of the roll/pitch band around the tip-down nominal (yaw pinned to 0).
    # Same default (0.35 rad = 20 deg) _apply_goal_vertical_mixture shipped for the analogous band
    # -- mdp.c3_transport_core.DEFAULT_TRANSPORT_GOAL_TILT_RAD, inlined here rather than imported
    # (the *_core modules are deliberately not exposed through the ``mdp`` namespace -- same choice
    # ``_apply_c1_hand_pose_stage`` made for ``c1_hand_pose_core``). Ignored while
    # transport_goal_prob == 0.0.
    transport_goal_tilt: float = 0.35
    # Root-frame z band (metres) the transport goal's height is drawn from. Default (0.13, 0.27) is
    # the same band _apply_goal_vertical_mixture shipped for DEXLIFT_GOAL_VERTICAL_Z -- tip 24-164 mm
    # above the work surface (mdp.c3_transport_core.DEFAULT_TRANSPORT_GOAL_Z_RANGE_M). Ignored while
    # transport_goal_prob == 0.0.
    transport_goal_z: tuple[float, float] = (0.13, 0.27)

    def __post_init__(self):
        super().__post_init__()
        _apply_full_gravity(self)
        legacy_toggle_active = _apply_partial_assembly_and_goal_toggles(self)
        # DELIVERABLE 2 (C4 seating-aware training variant): see _apply_c4_seating_training's own
        # docstring. Called after the toggles above so commands.object_pose is already
        # GoalAtSpawnPoseCommand by the time this function's precondition assert runs.
        _apply_c4_seating_training(self)
        # C3 RUNG stage (bead dr-ai1.4): 50% S1 + 50% S_t, off unless DEXRESET_C3_RUNG=1. Called
        # BEFORE the episode-mixture branch below and given legacy_toggle_active, because it
        # REFUSES (raises) rather than silently losing or winning a race for events.reset_object /
        # commands.object_pose -- see _apply_c3_rung_stage's docstring. Written out in both Reorient
        # classes for the same reason every other line in this block is: they are NOT in an
        # inheritance relationship with each other (bead UWLab-qiao.9/J).
        _apply_c3_rung_stage(self, legacy_toggle_active)
        episode_mixture_requested = os.environ.get("DEXLIFT_EPISODE_MIXTURE") == "1"
        if episode_mixture_requested and not legacy_toggle_active:
            _apply_episode_mixture(self)
        elif episode_mixture_requested and legacy_toggle_active:
            print(
                "[dexlift] DEXLIFT_EPISODE_MIXTURE=1 requested but a legacy toggle"
                " (DEXLIFT_PARTIAL_ASSEMBLY/DEXLIFT_GOAL_AT_SPAWN) fired first -- the legacy,"
                " deterministic whole-run path wins; the mixture is NOT installed on top of it.",
                flush=True,
            )


@configclass
class DexLiftUR5eDeltoOscTableLegReorientEnvCfg(Ur5eDeltoTableLegOscMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexLiftUR5eDeltoOscTableLegReorientEnvCfg_PLAY(
    Ur5eDeltoTableLegOscMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY
):
    pass
