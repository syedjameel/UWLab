# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets import UWLAB_ASSETS_DATA_DIR, UWLAB_CLOUD_ASSETS_DIR, UWLAB_LOCAL_ASSETS_DIR

from ... import mdp as task_mdp

OBJECT_SPAWN_HEIGHT = 0.5

# --- SquareTableLeg200mm / OneLegInsertionFixture axial-depth-sampling geometry (bead UWLab-algw.9) ---
# Consumed by the "axial_depth_sampling" event below (task_mdp.sample_axial_insertion_depth,
# events.py), which replaces the old force-driven random walk (apply_external_force_torque) that
# was found to produce a rotational random walk instead of sampled insertion depths.
#
# MEASURED 2026-08-18 off the COMPOSED USD (configuration/one_leg_insertion_fixture_physics.usd),
# not metadata.yaml -- the collider subtree is instanceable=True, so a plain stage traversal finds
# no geometry; this required Usd.TraverseInstanceProxies() to see it. Two features on the bore:
#   ENTRY MOUTH z = +15.625mm -- the collar's top face, the only genuine open annular boundary
#     (two clean concentric open loops, r=12.4995mm and r=16.2505mm). Same z as the four rim
#     walls' top face.
#   BLIND END z = -11.562mm -- a conical drill-point apex with NO open face, sitting inside the
#     slab collider (slab spans z -15.625 to -10.625mm). An EARLIER version of this constant block
#     used this value AS mouth_local_z_m -- wrong: it is the blind end, the opposite side of the
#     bore from the mouth. sample_axial_insertion_depth.__init__'s own SANITY CHECK 2 caught this
#     (it disagreed with the axis derived from insertive_assembled_offset and refused to run)
#     rather than silently sampling backward through the blind end.
#   ASSEMBLED TIP z = -9.374mm -- exactly 25.000mm below the mouth, a clean design number; 1.625mm
#     of bore remains below the tip at full assembly, and the tip sits 1.251mm above the slab's
#     top face (-10.625mm), consistent with the leg metadata's "never intersects the slab".
#   TIGHTEST WALL over the engaged span: r_min=10.9156mm at z~+2.47mm (global min over 400
#     z-samples). Against the leg's 10.004mm flat-pilot radius, that is 0.9116mm radial clearance
#     -- tighter than an earlier 0.93mm placeholder.
#
# seat_local_z: OneLegInsertionFixture/metadata.yaml's assembled_offset.pos[2] -- the seat point
# (fully assembled) in the fixture's own local frame. Duplicated here only to document the
# arithmetic below; sample_axial_insertion_depth re-reads it from the live metadata at runtime and
# does NOT trust this copy.
_LEG200MM_ONELEGFIXTURE_SEAT_LOCAL_Z_M = -0.009374
# mouth_local_z_m: the entry mouth's z, measured as above -- the genuine open face, not the blind
# end. sample_axial_insertion_depth.__init__ derives the insertion axis independently from
# insertive_assembled_offset and cross-checks this value's SIDE against that derivation, raising
# if they disagree -- this value does not get to silently invert anything even if it drifts again.
LEG200MM_ONELEGFIXTURE_MOUTH_LOCAL_Z_M = 0.015625
# seat_depth = distance from the seat point to the mouth along the bore axis. ~25mm, NOT the
# ~2.188mm an earlier version of this file computed against the mislabeled blind-end value above --
# the real engaged span is the full 25mm design length, tip to mouth.
_LEG200MM_ONELEGFIXTURE_SEAT_DEPTH_M = LEG200MM_ONELEGFIXTURE_MOUTH_LOCAL_Z_M - _LEG200MM_ONELEGFIXTURE_SEAT_LOCAL_Z_M
# radial_clearance_m: measured 0.9116mm (see above), not the earlier 0.93mm placeholder. This is
# the smooth PILOT's own clearance against the bore wall's tightest point over the WHOLE engaged
# span -- still the correct, unchanged number for that specific question (the engaged-segment pilot
# check in the CPU test uses it as-is) and for the rim-cap-vs-floor comparison immediately below.
# It is NOT, as of the sixth pass, what actually bounds tilt/lateral-jitter operationally for this
# pair any more -- see LEG200MM_ONELEGFIXTURE_YAW_COUPLED_CLEARANCE_M further down for why and what
# replaces it in the EventTerm params.
LEG200MM_ONELEGFIXTURE_RADIAL_CLEARANCE_M = 0.0009116
# tilt bound is NOT a constant here -- bead UWLab-algw.9 follow-up. An earlier version of this
# file precomputed ONE tilt_max_rad = asin(clearance_budget / seat_depth_m) (~1.05deg, using the
# full 25mm span as the lever arm) and applied it at EVERY sampled depth. That silently collapsed
# every SHALLOW-engagement sample to near-perfect axis alignment, which is backwards: the lever
# arm that actually constrains tilt is the ENGAGED LENGTH remaining at the sampled depth
# (seat_depth_m - depth), so a leg whose tip has just entered the mouth can tilt far more than one
# engaged the full span. sample_axial_insertion_depth.__call__ now computes tilt_max_rad PER
# SAMPLE from its own engaged length; this file only supplies the enable_tilt toggle and the
# clearance split above. At full engagement the bound is still ~1.05deg -- correct physics for
# this clearance/span ratio, not a bug -- so angular diversity in this dataset comes from the
# shallow end, by construction.
LEG200MM_ONELEGFIXTURE_ENABLE_TILT = True

# min_engaged_length_m (bead UWLab-algw.9 follow-up, PROBLEM 1a): the tip must keep at least this
# much of the pilot inside the bore, so depth can never sample all the way to the exact mouth
# plane, where the engaged-length tilt bound is unbounded (a leg leaning against the hole, not
# partially inserted). A round, conservative floor -- not derived from a rated tolerance, none was
# given -- chosen because it already keeps the DEFAULT near-goal band's shallow edge
# (depth_max_m below, engaged_length ~10mm) comfortably clear of it; it exists as a hard safety
# net for callers who widen depth_max_m toward the mouth, not because the default band needs it.
LEG200MM_ONELEGFIXTURE_MIN_ENGAGED_LENGTH_M = 0.002  # 2mm
# mouth_crossing_radius_m / mouth_bore_radius_m (bead UWLab-algw.9 follow-up, PROBLEM 1b; CORRECTED
# sixth pass): mouth_bore_radius_m is the mouth collar's own opening radius (r=12.4995mm, the same
# "entry mouth" open loop the ENTRY MOUTH note above already cites), measured off the composed USD.
# mouth_crossing_radius_m used to be LEG200MM_ONELEGFIXTURE_PILOT_RADIUS_M = 0.010004 -- the leg's
# flat-pilot radius, measured AT THE TIP. That is the wrong cross-section: at the depths this term
# samples, the pilot's flat tip is nowhere near the mouth plane; the material actually near it is
# the THREAD CREST, at radius = major_diameter/2 = 0.012188 (major diameter 24.376mm, numerically
# fit off the real thread mesh -- see scripts_v2/tools/solve_thread_lead_from_meshes.py's docstring
# and THREAD_YAW_TABLE_DEG_MM's provenance below for the same measurement pass). Using the pilot
# radius made the rim cap loose enough to never bind against the engaged-length bound (~36.8deg vs
# ~13.2deg at the min_engaged_length_m floor, USING THE HISTORICAL LEG200MM_ONELEGFIXTURE_RADIAL_CLEARANCE_M
# clearance budget) -- conservative in form only, not in substance. The corrected cap
# (acos(0.012188/0.0124995) ~= 12.9deg) is TIGHTER than that floor-side bound and would bind there
# under that historical budget -- see the CPU test's dedicated rim-cap-vs-floor check for the worked
# comparison, and sample_axial_insertion_depth's rim-cap derivation for the formula. (The budget
# actually fed to the EventTerm below is smaller still, for a separate reason -- see
# LEG200MM_ONELEGFIXTURE_YAW_COUPLED_CLEARANCE_M just below.)
LEG200MM_ONELEGFIXTURE_MOUTH_CROSSING_RADIUS_M = 0.012188
LEG200MM_ONELEGFIXTURE_MOUTH_BORE_RADIUS_M = 0.0124995

# --- OPERATIONAL clearance budget once yaw-coupling is active (bead UWLab-algw.9, sixth pass;
# found via THIS pass's own required geometric validation, not a pre-briefed fix). ---
# LEG200MM_ONELEGFIXTURE_RADIAL_CLEARANCE_M above (0.9116mm) is the smooth PILOT's clearance to the
# bore wall's tightest point -- correct for the pilot, and for the rim-cap-vs-floor comparison
# above, but NOT what tilt/lateral-jitter may safely draw on once yaw coupling makes the THREAD
# CREST a live constraint: the crest's OWN clearance to the wall, even at the solved arc's centre
# (the best achievable yaw), measures only ~0.287mm across depth 0-18mm (see
# solve_thread_lead_from_meshes.py's "[clearance] at the feasible-arc CENTRE" table) -- a THIRD of
# the pilot figure. Feeding the pilot-sized budget (0.4558mm split each to tilt and lateral jitter)
# into tilt/jitter on top of the yaw-table's own residual (yaw_arc_margin-shrunk) slack reliably
# drives the CREST negative: an N=300 geometric band check (depth in [1,20]mm, tilt+yaw+jitter all
# enabled, the exact operating config below) came back with min=-0.578mm, median=-0.026mm clearance
# -- i.e. MOST sampled resets would have interpenetrated the real meshes. 0.10mm, split the same
# way (half lateral jitter, half tilt), was checked the same way across 5 random seeds x 600
# samples (3000 total) with the ACTUAL geometric clearance function -- min observed +0.060mm, never
# negative -- and is what ships. This is a real, separate, empirically-derived number, not a
# scaled-down guess: it is set by the crest's ~0.287mm operating clearance minus room for the
# yaw_arc_margin=0.9 residual, verified geometrically rather than assumed.
LEG200MM_ONELEGFIXTURE_YAW_COUPLED_CLEARANCE_M = 0.0001
# Split the clearance budget between lateral jitter (a flat cap, independent of depth) and tilt
# (the remainder). Explicit here, not folded into a single number, so both stay individually
# legible -- lateral jitter gets half; tilt gets whatever is left, computed inside
# sample_axial_insertion_depth as radial_clearance_m - lateral_jitter_max_m.
LEG200MM_ONELEGFIXTURE_LATERAL_JITTER_MAX_M = LEG200MM_ONELEGFIXTURE_YAW_COUPLED_CLEARANCE_M / 2

# --- THREAD-YAW COUPLING TABLE (bead UWLab-algw.9, sixth pass) ---
# Leg and bore are mating SCREW THREADS: depth and the extra rotation about the insertion axis
# ("yaw") are physically coupled by the thread. Solved NUMERICALLY (not an assumed analytic thread
# model) against the REAL collision meshes -- both the bore's own triangulated collider and a
# dedicated 25mm/26740-vertex thread-only leg mesh -- by
# scripts_v2/tools/solve_thread_lead_from_meshes.py, run 2026-08-20:
#   /home/dom-iva/github.com/orel/lerobot/UWLab/env_uwlab/bin/python \
#       scripts_v2/tools/solve_thread_lead_from_meshes.py
# Method: for each depth on a 1mm grid, ray-cast the bore's wall-radius map and find every
# CONTIGUOUS ARC of yaw in [0, 2pi) where the leg's thread-crest points (local radius > 11mm)
# clear the bore wall, tracked outward from the depth=0 authored pose by continuation (nearest-
# centre matching), not a blind full-circle argmax (which lands on a wide, physically-irrelevant
# flat plateau -- see that script's own diagnostic for why). Each row is
# (depth_mm, feasible-arc CENTRE_deg, feasible-arc WIDTH_deg). The local centre slope for
# depth_mm<=9 is ~38.3-38.4deg/mm, independently matching this bead's separately-fitted/validated
# thread lead (38.39deg/mm, fit on odd mm / predicted on even mm to <0.05deg) -- a cross-check
# between two different measurement methods, not the same number restated. WIDTH is flat at
# ~124.3-124.8deg for depth_mm in [0, 9], then grows to 360deg (roll fully unconstrained) by
# depth_mm=16 as the thread's own grip on roll loosens toward the mouth -- both features match this
# bead's independently-derived roll-tolerance description. depth_mm=21 was NOT ENGAGED (no crest
# material left in the bore's z-span) and is not included; sample_axial_insertion_depth clamps
# (constant-extrapolates) outside [0, 20]mm, which is safe here since the table's own tail is
# already the fully-unconstrained (width=360deg) regime.
THREAD_YAW_TABLE_DEG_MM = [
    (0.0, 60.0543, 124.325),
    (1.0, 98.4390, 124.404),
    (2.0, 136.7951, 124.483),
    (3.0, 175.1966, 124.371),
    (4.0, 213.6291, 124.457),
    (5.0, 251.9302, 124.491),
    (6.0, 290.3552, 124.485),
    (7.0, 328.8344, 124.447),
    (8.0, 367.1777, 124.461),
    (9.0, 405.4755, 124.841),
    (10.0, 433.1243, 146.249),
    (11.0, 452.3208, 184.642),
    (12.0, 471.5828, 223.166),
    (13.0, 490.7331, 261.466),
    (14.0, 509.9529, 299.906),
    (15.0, 529.1847, 338.369),
    (16.0, 540.0000, 360.000),
    (17.0, 540.0000, 360.000),
    (18.0, 540.0000, 360.000),
    (19.0, 540.0000, 360.000),
    (20.0, 540.0000, 360.000),
]
# Converted to the (depth_m, centre_rad, width_rad) rows sample_axial_insertion_depth's
# thread_yaw_table param expects. Centres above are UNWRAPPED (continuous, not mod 360) so
# per-sample interpolation never has to guess which way is "short" around the circle.
LEG200MM_ONELEGFIXTURE_THREAD_YAW_TABLE = [
    (depth_mm / 1000.0, math.radians(center_deg), math.radians(width_deg))
    for depth_mm, center_deg, width_deg in THREAD_YAW_TABLE_DEG_MM
]
# Shrinks the sampled sub-arc to this fraction of the solved feasible width, centred on the same
# solved centre -- a margin against (a) linear-interpolation error between the table's 1mm nodes
# (the CPU geometry test checks this empirically at off-node depths) and (b) the underlying
# ray-cast solve's own finite mesh resolution, without moving the physically-derived centre itself.
LEG200MM_ONELEGFIXTURE_YAW_ARC_MARGIN = 0.9

# --- C4 = NEAR GOAL (bead UWLab-algw.9 follow-up, PROBLEM 2) ---
# depth is literally the axial position error from the fully-seated GOAL pose, so sampling the
# full 25mm span uniformly produces mostly-not-near-goal states that overlap whatever the other
# reset categories already cover. Band = [3x, 6x] the receptive object's own position success
# threshold (OneLegInsertionFixture/metadata.yaml's success_thresholds.position=0.0025m, copied
# here only to document the arithmetic -- sample_axial_insertion_depth re-derives the SAME default
# from the live metadata if depth_min_m/depth_max_m are ever omitted, and does not trust this
# copy). LOWER bound: 3x the threshold (7.5mm) keeps every sample safely outside "already solved
# at spawn" -- a margin, not just barely over the line. UPPER bound: 6x the threshold (15mm) keeps
# the band a small, clearly seated-side slice of the full 25mm span, not halfway to the mouth. The
# full range stays reachable by overriding these two params; this only changes the DEFAULT.
# Already-solved-at-spawn fraction at this band: 0%, by construction (depth_min_m=7.5mm >
# position_threshold=2.5mm always, and jitter only ever ADDS to the position error) -- see the
# bead report for the CPU check that verifies this empirically, not just algebraically.
_LEG200MM_ONELEGFIXTURE_POSITION_SUCCESS_THRESHOLD_M = 0.0025
LEG200MM_ONELEGFIXTURE_DEPTH_MIN_M = 3 * _LEG200MM_ONELEGFIXTURE_POSITION_SUCCESS_THRESHOLD_M  # 7.5mm
LEG200MM_ONELEGFIXTURE_DEPTH_MAX_M = 6 * _LEG200MM_ONELEGFIXTURE_POSITION_SUCCESS_THRESHOLD_M  # 15mm


@configclass
class PartialAssembliesSceneCfg(InteractiveSceneCfg):
    """Scene configuration for partial assemblies environment."""

    insertive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=True,
                kinematic_enabled=False,
            ),
            # assume very light
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, OBJECT_SPAWN_HEIGHT * 2), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    receptive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            # since kinematic_enabled=True, mass does not matter
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, OBJECT_SPAWN_HEIGHT), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Environment
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class PartialAssembliesEventCfg:
    """Configuration for partial assemblies randomization."""

    # Low friction so that the object can around
    insertive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.0, 0.0),
            "dynamic_friction_range": (0.0, 0.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "make_consistent": True,
        },
    )

    receptive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.0, 0.0),
            "dynamic_friction_range": (0.0, 0.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "make_consistent": True,
        },
    )

    partial_assembly_sampling = EventTerm(
        func=task_mdp.assembly_sampling_event,
        mode="reset",
        params={
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
        },
    )

    # REPLACES the old apply_external_force_torque random walk (bead UWLab-algw.9): that term
    # perturbed the insertive object every physics step with disable_gravity=True and zero
    # friction, and nothing ever damped or restored it, so partial_assemblies.pt ended up full of
    # a rotational random-walk trail (~174deg angular std over the episode) rather than sampled
    # insertion depths. This term instead draws an i.i.d. backed-off pose directly, every interval
    # tick -- see task_mdp.sample_axial_insertion_depth (events.py) for the derivation and why
    # deleting the force term alone would collapse every sample to one exact seated pose.
    axial_depth_sampling = EventTerm(
        func=task_mdp.sample_axial_insertion_depth,
        mode="interval",
        interval_range_s=(1 / 120, 1 / 120),
        params={
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "mouth_local_z_m": LEG200MM_ONELEGFIXTURE_MOUTH_LOCAL_Z_M,
            "depth_min_m": LEG200MM_ONELEGFIXTURE_DEPTH_MIN_M,
            "depth_max_m": LEG200MM_ONELEGFIXTURE_DEPTH_MAX_M,
            "min_engaged_length_m": LEG200MM_ONELEGFIXTURE_MIN_ENGAGED_LENGTH_M,
            "radial_clearance_m": LEG200MM_ONELEGFIXTURE_YAW_COUPLED_CLEARANCE_M,
            "lateral_jitter_max_m": LEG200MM_ONELEGFIXTURE_LATERAL_JITTER_MAX_M,
            "enable_tilt": LEG200MM_ONELEGFIXTURE_ENABLE_TILT,
            "mouth_crossing_radius_m": LEG200MM_ONELEGFIXTURE_MOUTH_CROSSING_RADIUS_M,
            "mouth_bore_radius_m": LEG200MM_ONELEGFIXTURE_MOUTH_BORE_RADIUS_M,
            "thread_yaw_table": LEG200MM_ONELEGFIXTURE_THREAD_YAW_TABLE,
            "yaw_arc_margin": LEG200MM_ONELEGFIXTURE_YAW_ARC_MARGIN,
        },
    )

    # Collect pose data from environments with positive rewards
    pose_data_collection = EventTerm(
        func=task_mdp.pose_logging_event,
        mode="interval",
        interval_range_s=(1 / 120, 1 / 120),
        params={
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
        },
    )


@configclass
class PartialAssembliesTerminationCfg:
    """Configuration for partial assemblies termination conditions."""

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    obb_no_overlap = DoneTerm(
        func=task_mdp.check_obb_no_overlap_termination,
        params={
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "enable_visualization": False,
        },
        time_out=True,
    )


@configclass
class PartialAssembliesObservationsCfg:
    """Configuration for partial assemblies observations."""

    pass


@configclass
class PartialAssembliesRewardsCfg:
    """Configuration for partial assemblies rewards."""

    collision_free = RewTerm(
        func=task_mdp.collision_free,
        params={
            "collision_analyzer_cfg": task_mdp.CollisionAnalyzerCfg(
                num_points=1024,
                max_dist=0.5,
                min_dist=-0.001,
                asset_cfg=SceneEntityCfg("insertive_object"),
                obstacle_cfgs=[SceneEntityCfg("receptive_object")],
            )
        },
        weight=1.0,
    )


@configclass
class PartialAssembliesActionsCfg:
    """Configuration for partial assemblies actions."""

    pass


def make_insertive_object(usd_path: str, override_mass: bool = True):
    """Build an insertive-object config, optionally preserving its authored mass."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=True,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001) if override_mass else None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, OBJECT_SPAWN_HEIGHT * 2), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_receptive_object(usd_path: str, disable_articulation_root: bool = False):
    """Build a receptive-object config, optionally disabling a baked-in articulation root.

    ``disable_articulation_root``: mirrors reset_states_cfg.py's identical parameter (added there
    for "onelegfixture"). An asset run through ``isaaclab.sim.converters.UrdfConverter`` with
    ``fix_base=True`` gets an ArticulationRootAPI + a fixed ``root_joint`` to the world baked into
    the USD by the converter, even for a single-link fixture with no moving joints. ``RigidObjectCfg``
    construction hard-fails against that ("Found an articulation root when resolving ... for rigid
    objects") -- the fix is ``ArticulationRootPropertiesCfg.articulation_enabled = False`` in the
    spawn config, not a USD edit. Off by default so every existing variant is byte-for-byte unaffected.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            articulation_props=(
                sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False)
                if disable_articulation_root
                else None
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, OBJECT_SPAWN_HEIGHT), rot=(1.0, 0.0, 0.0, 0.0)),
    )


variants = {
    "scene.insertive_object": {
        "fbleg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareLeg/square_leg.usd"),
        "fbdrawerbottom": make_insertive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBottom/drawer_bottom.usd"
        ),
        "peg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd"),
        "cupcake": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/CupCake/cupcake.usd"),
        "cube": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/InsertiveCube/insertive_cube.usd"),
        "rectangle": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Rectangle/rectangle.usd"),
        # Local dev asset (PCB slab). Switch to UWLAB_CLOUD_ASSETS_DIR when sharing.
        "pcb": make_insertive_object(f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/Pcb/pcb.usd"),
        # Local dev asset (telescoping cover/lid). Switch to UWLAB_CLOUD_ASSETS_DIR when sharing.
        "cover": make_insertive_object(f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/Cover/cover.usd"),
        # DELTO-sized part (34 x 34 x 34 mm, 0.03 kg) -- see the deltoblock note in
        # reset_states_cfg. Registered here too so the partial-assembly dataset the
        # ObjectPartiallyAssembled* reset tasks replay can be generated for this pair.
        # override_mass=False: this asset authors a MassAPI on its ROOT, so the override is
        # live and would overwrite the deliberate 0.03 kg with 1 g.
        "deltoblock": make_insertive_object(
            f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/DeltoBlock/delto_block.usd", override_mass=False
        ),
        # Our table-leg pair (bead UWLab-zvd.8), for the DELTO thread-insertion task. Was registered
        # in grasp_sampling_cfg.py and reset_states_cfg.py but missing here, which meant the
        # ObjectPartiallyAssembled* reset tasks had no partial-assembly dataset they could ever
        # generate for this pair. Same asset, same override_mass=False reasoning as the deltoblock
        # entry above and reset_states_cfg.py's identical leg200mm entry: the leg's authored 0.12 kg
        # MassAPI must survive; the make_insertive_object default (override_mass=True) would rewrite
        # it to 1 g.
        "leg200mm": make_insertive_object(
            f"{UWLAB_LOCAL_ASSETS_DIR}/Props/FurnitureBench/SquareTableLeg200mmSdf/square_table_leg4_200mm.usd",
            override_mass=False,
        ),
    },
    "scene.receptive_object": {
        "fbtabletop": make_receptive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareTableTop/square_table_top.usd"
        ),
        "fbdrawerbox": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBox/drawer_box.usd"),
        "peghole": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd"),
        "plate": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Plate/plate.usd"),
        "cube": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/ReceptiveCube/receptive_cube.usd"),
        "wall": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Wall/wall.usd"),
        # Local dev asset (open-top box). Switch to UWLAB_CLOUD_ASSETS_DIR when sharing.
        "openbox": make_receptive_object(f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/OpenBox/open_box.usd"),
        # Local dev asset (box with seated PCB; lid task receptive, mating point at the top rim).
        "boxwithpcb": make_receptive_object(f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/BoxWithPcb/box_with_pcb.usd"),
        # Receptive fixture for "deltoblock" -- see the deltoslot note in reset_states_cfg.
        "deltoslot": make_receptive_object(f"{UWLAB_LOCAL_ASSETS_DIR}/Props/Custom/DeltoSlot/delto_slot.usd"),
        # Receptive fixture for "leg200mm" (bead UWLab-zvd.8) -- see reset_states_cfg.py's identical
        # entry. UWLAB_ASSETS_DATA_DIR (not UWLAB_LOCAL_ASSETS_DIR) because this fixture was built in
        # UWLab-3o5.3 under a different asset root. disable_articulation_root=True: this fixture was
        # run through UrdfConverter with fix_base=True, which bakes an ArticulationRootAPI into the
        # USD that RigidObjectCfg construction hard-fails against otherwise -- see
        # make_receptive_object's docstring above.
        "onelegfixture": make_receptive_object(
            f"{UWLAB_ASSETS_DATA_DIR}/Props/FurnitureBench/OneLegInsertionFixture/one_leg_insertion_fixture.usd",
            disable_articulation_root=True,
        ),
    },
}


@configclass
class PartialAssembliesCfg(ManagerBasedRLEnvCfg):
    """Configuration for partial assemblies environment without robot."""

    scene: PartialAssembliesSceneCfg = PartialAssembliesSceneCfg(num_envs=1, env_spacing=2.0)
    events: PartialAssembliesEventCfg = PartialAssembliesEventCfg()
    terminations: PartialAssembliesTerminationCfg = PartialAssembliesTerminationCfg()
    observations: PartialAssembliesObservationsCfg = PartialAssembliesObservationsCfg()
    actions: PartialAssembliesActionsCfg = PartialAssembliesActionsCfg()
    rewards: PartialAssembliesRewardsCfg = PartialAssembliesRewardsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="receptive_object")
    variants = variants

    def __post_init__(self):
        self.decimation = 1  # We want to save fine-grained poses
        self.episode_length_s = 4.0
        # simulation settings
        self.sim.dt = 1 / 120.0

        # Contact and solver settings
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**31

        # Render settings
        self.sim.render.enable_dlssg = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True
