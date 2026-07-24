# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Augment an OmniReset env cfg so every box-assembly scene spawns ALL four entities.

OmniReset's base scene spawns only the pair (``insertive_object`` + ``receptive_object``).
The box-assembly task requires every scene to additionally contain the remaining box
parts and the assembly-target marker (spawn rule 1a), with the non-manipulated
("context") objects already placed in their preceding-stage goal poses (rule 1b), and an
``allow_overlap`` toggle controlling whether they may pile up (rule 1c).

``augment_box_assembly(env_cfg, allow_overlap)`` is called *after* the pair is selected
(Hydra at generation time, or ``__post_init__`` at training time). Because IsaacLab's
recorder captures every scene rigid object and ``MultiResetManager`` replays them by
name, adding the context entities here makes them flow automatically into the recorded
reset datasets and back out at training time -- no dataset-format changes needed.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from ... import mdp as task_mdp
from ...mdp.box_assembly import asset_usd, semantic_of_usd

# Compact work zone (env-origin frame). The whole task lives here: work-object reset ranges are
# clamped to it, context objects scatter within it, and box_assembly_goal's workspace AABB wraps it.
_ZONE_X = (0.36, 0.58)
_ZONE_Y = (-0.10, 0.30)
# Scatter footprint (inside the zone, with footprint margin so objects rest within the workspace).
_SCATTER_RANGE = {"x": (0.39, 0.56), "y": (-0.06, 0.26), "z": (0.0, 0.004), "yaw": (-3.14159, 3.14159)}
_OBJECT_MASS = 0.045  # MuJoCo part mass

# workspace AABB (env-origin frame) -- must match box_assembly_goal; out-of-workspace spawns rejected
_WS_XY = ((0.28, 0.66), (-0.20, 0.40))
_WS_Z = (-0.05, 0.45)
# easy-mode correct rest orientation (roll, pitch): box cavity-up & object flat = identity;
# cover rests opening-UP like a bowl = -90 deg about Y (flipped from the closed/goal pose,
# whose opening faces down). The robot flips it during the cover-close stage.
_BASE_EULER = {"box": (0.0, 0.0), "object": (0.0, 0.0), "cover": (0.0, -math.pi / 2.0)}
# Easy-mode GRASPED rest orientation (roll, pitch) = the grasp-SAMPLING orientation, so the recorded
# top-down grasp stays top-down and the object lies FLAT/horizontal in the gripper (perpendicular to
# the fingers) instead of at a random tilt. box/object flat = identity; cover is sampled opening-DOWN
# (closed part up) = +90 deg about Y -- NOTE this is the opposite of _BASE_EULER's free-rest cover
# (opening-up), because the grasp is taken on the closed part from above.
_GRASP_BASE_EULER = {"box": (0.0, 0.0), "object": (0.0, 0.0), "cover": (0.0, math.pi / 2.0)}


def _clamp_reset_ranges(env_cfg):
    """Clamp the work-object (insertive/receptive) reset pose ranges to the compact work zone, so
    every spawn starts inside the (now smaller) workspace. Dataset-driven reset types (Resting,
    PartiallyAssembled) inherit the tighter region via the data they read."""
    for ev_name in ("reset_insertive_object_pose", "reset_receptive_object_pose"):
        ev = getattr(env_cfg.events, ev_name, None)
        if ev is None or "pose_range" not in ev.params:
            continue
        pr = dict(ev.params["pose_range"])
        cx = pr.get("x", _ZONE_X)
        cy = pr.get("y", _ZONE_Y)
        pr["x"] = (max(cx[0], _ZONE_X[0]), min(cx[1], _ZONE_X[1]))
        pr["y"] = (max(cy[0], _ZONE_Y[0]), min(cy[1], _ZONE_Y[1]))
        ev.params["pose_range"] = pr


def _make_context_object(usd_path: str, prim: str, kinematic: bool) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=kinematic,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=(0.5 if kinematic else _OBJECT_MASS)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def augment_box_assembly(
    env_cfg,
    allow_overlap: bool = False,
    min_separation: float = 0.10,
    scene_only: bool = False,
    gate_insertive: bool = True,
    reset_type: str = "",
):
    """Add the missing box-assembly entities + their reset placement to ``env_cfg``.

    Returns the (mutated) env_cfg. No-op if the pair is not a box-assembly pair.

    ``scene_only`` (training): only declare the context scene entities. Their poses are
    restored from the recorded reset dataset by ``MultiResetManager`` replay, so the
    placement/scatter events and the spawn-validity collision gate are skipped (they
    would re-randomise the restored state).
    """
    ins_usd = env_cfg.scene.insertive_object.spawn.usd_path
    rec_usd = env_cfg.scene.receptive_object.spawn.usd_path
    ins_sem = semantic_of_usd(ins_usd)
    rec_sem = semantic_of_usd(rec_usd)
    if ins_sem is None or rec_sem is None:
        return env_cfg  # not a box-assembly pair; leave untouched

    # which scene entity currently holds the box (its tray is the assembly base)
    box_entity = "insertive_object" if ins_sem == "box" else ("receptive_object" if rec_sem == "box" else None)

    present = {ins_sem, rec_sem}
    missing = [s for s in ("box", "object", "cover", "target") if s not in present]

    # 1) spawn the missing entities as context rigid objects
    ctx_name = {}
    for sem in missing:
        name = f"ctx_{sem}"
        prim = "Ctx" + sem.capitalize()
        setattr(env_cfg.scene, name, _make_context_object(asset_usd(sem), prim, kinematic=(sem == "target")))
        ctx_name[sem] = name

    if scene_only:
        return env_cfg  # training: entities restored from dataset by MultiResetManager

    # tighten work-object spawn to the compact work zone (matches the smaller workspace AABB)
    _clamp_reset_ranges(env_cfg)

    # 2) decide placement: prior-stage goals are pre-placed; future-stage objects scatter freely
    goal_placed = []  # (ctx_name) placed at its assembled goal relative to the box
    free = []  # (ctx_name) scattered on the table
    for sem in missing:
        name = ctx_name[sem]
        prior_in_box = sem == "object" and ins_sem == "cover"  # pair C: object already in box
        prior_target = sem == "target"  # B/C: box already sits in the target
        if (prior_in_box or prior_target) and box_entity is not None:
            ev = EventTerm(
                func=task_mdp.assembly_sampling_event,
                mode="reset",
                params={
                    "receptive_object_cfg": SceneEntityCfg(box_entity),
                    "insertive_object_cfg": SceneEntityCfg(name),
                },
            )
            setattr(env_cfg.events, f"place_{name}", ev)
            goal_placed.append(name)
        else:
            free.append(name)

    # 3) scatter the free context objects (no-contact unless allow_overlap)
    if free:
        avoid = [SceneEntityCfg("insertive_object"), SceneEntityCfg("receptive_object")]
        avoid += [SceneEntityCfg(n) for n in goal_placed]
        # easy mode: spawn scattered objects in their correct rest orientation (cover lid-down).
        base_euler = {} if allow_overlap else {n: _BASE_EULER[n.replace("ctx_", "")] for n in free}
        env_cfg.events.scatter_context = EventTerm(
            func=task_mdp.scatter_objects_no_contact,
            mode="reset",
            params={
                "object_cfgs": [SceneEntityCfg(n) for n in free],
                "avoid_cfgs": avoid,
                "pose_range": _SCATTER_RANGE,
                "min_separation": min_separation,
                "allow_overlap": allow_overlap,
                "use_bottom_offset": True,
                "base_euler": base_euler,
            },
        )

    # 4) gate spawn validity on the GOAL-PLACED context objects (stability + deviation): they rest
    #    at their goal and must stay there. Free-scattered objects are NOT added to the deviation
    #    gate (they may settle/topple a few mm, which would spuriously fail strict reset types).
    success = getattr(env_cfg.terminations, "success", None)
    if success is not None and "object_cfgs" in success.params:
        gated = [n for n in goal_placed if n != ctx_name.get("target")]
        success.params["object_cfgs"] = list(success.params["object_cfgs"]) + [SceneEntityCfg(n) for n in gated]
        # Also gate the FREE-scattered ctx objects against falling through / off the table: when the
        # work zone is crowded (e.g. stage-A near-goal: box at target + 2 free ctx) the no-contact
        # scatter can place them overlapping and physics ejects them to the floor (z=-0.868), and they
        # were exported as "valid" while invisible under the table. Add them to the below-ground +
        # workspace gate (deviation tolerances 0.025-0.05 absorb the few-mm settle).
        free_movable = [n for n in free if n != ctx_name.get("target")]
        success.params["object_cfgs"] = success.params["object_cfgs"] + [SceneEntityCfg(n) for n in free_movable]

    # 5) no-contact gate (overlap OFF): add accurate SDF collision analyzers between every
    #    non-intended object pair, so contacting spawns are rejected (spec 1b/1c). Intended
    #    contacts are exempt: box-target always (the box-may-overlap-target exception), and for
    #    pair C also object-box / object-target (object already inside the box). For the
    #    PartiallyAssembled reset type the insertive is intentionally at the goal, so gate_insertive
    #    is False and pairs involving the insertive are skipped.
    if not allow_overlap and success is not None and "collision_analyzer_cfgs" in success.params:
        # semantic -> scene-entity name (all four are present after step 1)
        sem_entity = {
            "box": box_entity,
            "target": "receptive_object" if rec_sem == "target" else ctx_name.get("target"),
            "object": "insertive_object" if ins_sem == "object"
            else ("receptive_object" if rec_sem == "object" else ctx_name.get("object")),
            "cover": "insertive_object" if ins_sem == "cover"
            else ("receptive_object" if rec_sem == "cover" else ctx_name.get("cover")),
        }
        intended = {frozenset(("box", "target"))}
        if ins_sem == "cover":  # pair C: object already seated in the box (and thus the target)
            intended |= {frozenset(("object", "box")), frozenset(("object", "target"))}
        sems = ["box", "object", "cover", "target"]
        analyzers = list(success.params["collision_analyzer_cfgs"])
        for i in range(len(sems)):
            for j in range(i + 1, len(sems)):
                a, b = sems[i], sems[j]
                if frozenset((a, b)) in intended:
                    continue
                if not gate_insertive and (sem_entity[a] == "insertive_object" or sem_entity[b] == "insertive_object"):
                    continue
                ea, eb = sem_entity[a], sem_entity[b]
                if ea is None or eb is None:
                    continue
                analyzers.append(
                    task_mdp.CollisionAnalyzerCfg(
                        num_points=512,
                        max_dist=0.5,
                        min_dist=0.0,
                        asset_cfg=SceneEntityCfg(ea),
                        obstacle_cfgs=[SceneEntityCfg(eb)],
                    )
                )
        success.params["collision_analyzer_cfgs"] = analyzers

    # 6) workspace filter (note 3): reject spawns with any object outside the workspace AABB.
    if success is not None:
        success.params["workspace_xy"] = _WS_XY
        success.params["workspace_z"] = _WS_Z

    # 7) airborne retention (notes 1/2): for the "object anywhere incl. in the air" distribution,
    #    don't require full settling -- validate after a few physics steps (short episode) by
    #    collision + workspace + below-ground, so airborne samples survive instead of being discarded.
    if "AnywhereEEAnywhere" in reset_type:
        if success is not None:
            success.params["require_stability"] = False
        env_cfg.episode_length_s = 0.1  # ~1 control step (decimation 12 @ 1/120 s)

    # 8) easy-mode orientation (note 4): box cavity-up, cover lid-down. Only the FREE insertive
    #    (object lying on the table/in the air, AnywhereEEAnywhere) is constrained; grasped types
    #    keep random orientations so the (sparse) validated grasps stay reachable. The scattered
    #    context objects already get their base orientation via base_euler above.
    if not allow_overlap and "AnywhereEEAnywhere" in reset_type:
        ev_ins = getattr(env_cfg.events, "reset_insertive_object_pose", None)
        if ev_ins is not None and "pose_range" in ev_ins.params:
            roll, pitch = _BASE_EULER[ins_sem]
            pr = dict(ev_ins.params["pose_range"])
            pr["roll"] = (roll, roll)
            pr["pitch"] = (pitch, pitch)
            ev_ins.params["pose_range"] = pr

    # 8b) easy-mode HORIZONTAL grasp (stable-grasp): place the grasped object FLAT at its sampling
    #     orientation so the recorded top-down grasp stays top-down and the object is perpendicular to
    #     the fingers, instead of grasped at a random tilt. Only the stable-grasp type
    #     (ObjectAnywhereEEGrasped) has a free reset_insertive_object_pose; near-object/near-goal get
    #     their orientation from reset-state/partial-assembly datasets. HARD mode keeps random tilt.
    if not allow_overlap and "AnywhereEEGrasped" in reset_type:
        ev_ins = getattr(env_cfg.events, "reset_insertive_object_pose", None)
        if ev_ins is not None and "pose_range" in ev_ins.params:
            roll, pitch = _GRASP_BASE_EULER[ins_sem]
            pr = dict(ev_ins.params["pose_range"])
            pr["roll"] = (roll, roll)
            pr["pitch"] = (pitch, pitch)
            if ins_sem == "cover":
                pr["yaw"] = (0.0, 0.0)  # avoid euler gimbal at pitch=+90deg; keep the lid flat
            ev_ins.params["pose_range"] = pr

    return env_cfg
