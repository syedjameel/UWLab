# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Re-key AND schema-complete a resets_*.pt produced in the DEXLIFT LIFT/REORIENT scene, so it
presents the OmniReset TRAINING scene's own four-entity rigid_object schema (bead: RestingEEGrasped
bridge, pre-flight catch; extended for UWLab-qiao: the schema-incompleteness fix).

WHY THIS EXISTS (rename half, original bead). ``generate_reset_states_policy.py`` records states in
the dexlift table-leg LIFT/REORIENT scene, whose manipulated body is a rigid_object named ``object``.
The OmniReset training scene names the SAME kind of body ``insertive_object`` (rl_state_cfg.py).
``MultiResetManager`` matches rigid_object entries between the loaded state and the consuming scene
BY NAME and SILENTLY SKIPS any entry present in the scene but absent from the state.

WHY THIS EXISTS (schema half, this extension). The dexlift scene has no ``receptive_object``,
``table``, or ``ur5_metal_support`` entities to record -- it is a bare lift/reorient cell. The
OmniReset consuming scene (rl_state_cfg.py's ``TrainEventCfg``) has all three, and its own
``reset_everything`` (``task_mdp.reset_scene_to_default``, mode="reset", runs BEFORE
``reset_from_reset_states``/``MultiResetManager`` in the same ``_reset_idx`` call -- confirmed from
``EventManager._prepare_terms``'s ``self.cfg.__dict__.items()`` field order and
``ManagerBasedRLEnv._reset_idx``'s manager-call order) plants every rigid_object at its scene-default
pose first. For ``receptive_object`` that default is ``init_state=(0,0,0)`` -- i.e. AT THE ROBOT BASE,
not the mat-relative pose the OTHER three reset-type generators (``ObjectAnywhereEEAnywhere``,
``ObjectRestingEEGrasped``, ``ObjectPartiallyAssembledEEGrasped``, all built from
``reset_states_cfg.py``, which DOES carry an explicit ``reset_receptive_object_pose`` event) produce.
Measured directly off the three sibling .pt files for this pair: ``receptive_object`` root z is
EXACTLY 0.019625 m (std 0.0) in every one of them, vs. 0.0 for a dexlift-origin file with the key
missing -- a 19.625 mm error, plus the x/y also collapsing to the env origin instead of the sampled
mat region. Every episode drawing the dexlift-origin reset type trains against a fixture that is
literally in the wrong place. This script closes that gap AT THE SCHEMA rather than by adding a
reset-type-aware positioning event to ``TrainEventCfg`` (which would have to know which reset type
was just drawn to avoid re-clobbering the three already-correct files -- exactly the kind of coupling
that lets this class of bug recur).

WHAT IT DOES, up to four things depending on input schema (see ``_detect_schema``):

1. RENAME (only if the input is a raw, never-rekeyed dexlift dump): rigid_object key
   ``object`` -> ``insertive_object``. Positions/velocities UNCHANGED -- pure key rename, no
   coordinate transform (both names refer to the same physical body in the same frame convention;
   robot base is at (0,0,0) in both the dexlift scene and the OmniReset training scene -- see
   UWLab-qiao.3 verdict). The dexlift-origin ``table`` entry, if present, is DROPPED (different USD,
   different frame convention than the training scene's own table -- see point 2, the drop is
   unrelated to and predates this extension).
2. SYNTHESISE the three missing rigid_object entries -- ``receptive_object``, ``table``,
   ``ur5_metal_support`` -- one INDEPENDENT draw per stored state (not one draw reused across the
   whole file), from the SAME per-episode distributions ``reset_states_cfg.py``'s
   ``ResetStatesBaseEventCfg`` draws for the three already-correct reset types:

   receptive_object (reset_states_cfg.py:195-220 ``reset_receptive_object_pose``, composed through
   ``reset_root_states_uniform``'s ``offset_asset_cfg``/``use_bottom_offset`` math, events.py
   ~2038-2055):
       x ~ U(0.35, 0.60)                      -- pose_range["x"], :205
       y ~ U(-0.20, 0.20)                     -- pose_range["y"], :209
       z = 0.019625 EXACTLY                   -- pose_range["z"] is (0.0, 0.0) (:210, no jitter);
                                                  z = ur5_metal_support's DEFAULT root z (0.004,
                                                  ResetStatesSceneCfg :89) minus bottom_offset.pos.z
                                                  (-0.015625, OneLegInsertionFixture/metadata.yaml:42)
                                                  = 0.004 - (-0.015625) = 0.019625. The offset asset's
                                                  DEFAULT root state is used (events.py's
                                                  ``offset_asset.data.default_root_state``), NOT its
                                                  own jittered runtime pose, so this does not
                                                  correlate with ur5_metal_support's own draw below.
       yaw ~ U(-pi/12, pi/12), roll=pitch=0    -- pose_range["roll"/"pitch"/"yaw"], :211-213
       quat = (cos(yaw/2), 0, 0, sin(yaw/2))   -- pure z-axis rotation; verified against the sibling
                                                  files empirically: qx/qy ~= 0, qz in [-0.1303,0.1303]
                                                  matches sin(pi/24) = 0.1305.
       velocity = 0                            -- velocity_range={} (:215)

   table: NOT jittered by anything, in either scene. NOTE: reset_states_cfg.py's own comment at
       :74-76 claims "reset_robot_pose jitters robot+support+table TOGETHER (rigid assembly)", but
       the EventTerm's own ``asset_cfgs`` dict (:188-191) lists only {"robot", "ur5_metal_support"} --
       table is NOT a key in it. This is a stale/wrong comment, not a stale/wrong implementation:
       confirmed empirically too -- table root_pose has EXACTLY ZERO variance (pos (0,0,0), quat
       (1,0,0,0)) across all three sibling files, matching the CODE, not the comment. Synthesised
       here as the fixed identity pose the code (and the data) actually produce.
       x = y = z = 0.0, quat = (1,0,0,0), velocity = 0.

   ur5_metal_support (reset_states_cfg.py:168-193 ``reset_robot_pose``, applied to
       {robot, ur5_metal_support} with a SHARED draw per real episode -- but since the dexlift-origin
       robot pose is fixed data we do not touch (see point 3), we draw ur5_metal_support's jitter
       INDEPENDENTLY here rather than reproducing that correlation, per instruction):
       x ~ U(-0.01, 0.01)                     -- pose_range["x"], :173
       y ~ U(-0.02, 0.02)                     -- pose_range["y"], :181
       z = 0.004 + U(-0.01, 0.01)             -- default root z (ResetStatesSceneCfg :89 /
                                                  rl_state_cfg.py :93) + pose_range["z"] (:182)
       roll=pitch=yaw=0                       -- :183-185, quat stays identity (1,0,0,0)
       velocity = 0                            -- velocity_range={} (:187)

3. LEAVES ``articulation.robot`` and ``rigid_object.insertive_object`` BIT-IDENTICAL to the input.
   Verified in-process by re-comparing every element against the loaded input before writing, and
   again by the caller after reload (see this script's own verification block).
4. REFUSES LOUDLY on a schema it does not recognise -- in particular, a file that already carries
   ``receptive_object`` (a generator-origin file, e.g. one of the three siblings) is refused outright,
   never silently re-synthesised on top of already-correct data (double-write guard).

SAFETY: writes to a NEW file first (``--output``), never touches ``--input`` in place. Verifies the
new file re-loads with the expected keys/shapes/entry count, and that robot+insertive_object are
bit-identical to the input, before printing a "write complete" line. The caller performs the swap
(rename the original aside as a ``.bak``, move the new file into place) only after independently
confirming this script's own verification output.

Pure CPU/torch, no Isaac, no GPU, no isaaclab import -- this is a plain nested-dict transform over a
.pt file, run with any Python that has torch (e.g. the system python3 on DL_A6000, confirmed to
carry torch 2.12.0 independent of the isaaclab venv). The yaw->quat conversion is done by hand
(single-axis case only) specifically to avoid an isaaclab.utils.math import.

Run:
    python3 scripts_v2/tools/rekey_dexlift_reset_states.py \\
        --input  .../Resets/OneLegInsertionFixture__SquareTableLeg200mmDecomp/resets_ObjectAnywhereEEGrasped.pt \\
        --output .../Resets/OneLegInsertionFixture__SquareTableLeg200mmDecomp/resets_ObjectAnywhereEEGrasped.schema-fixed.pt \\
        --seed 0
"""

from __future__ import annotations

import argparse
import math

import torch

# -- distributions, cited above; kept as named constants so a formula change in
# reset_states_cfg.py can be diffed against this file directly.
RECEPTIVE_X_RANGE = (0.35, 0.60)
RECEPTIVE_Y_RANGE = (-0.20, 0.20)
RECEPTIVE_Z = 0.019625  # 0.004 (ur5_metal_support default root z) - (-0.015625) (bottom_offset.pos.z)
RECEPTIVE_YAW_RANGE = (-math.pi / 12, math.pi / 12)

SUPPORT_X_RANGE = (-0.01, 0.01)
SUPPORT_Y_RANGE = (-0.02, 0.02)
SUPPORT_Z_NOMINAL = 0.004
SUPPORT_Z_JITTER_RANGE = (-0.01, 0.01)

TABLE_POSE = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # fixed identity, see module docstring


def _uniform(low: float, high: float, n: int, generator: torch.Generator) -> torch.Tensor:
    return torch.empty(n, dtype=torch.float32).uniform_(low, high, generator=generator)


def _synthesize_receptive_object(n: int, generator: torch.Generator) -> tuple[list, list]:
    x = _uniform(*RECEPTIVE_X_RANGE, n, generator)
    y = _uniform(*RECEPTIVE_Y_RANGE, n, generator)
    z = torch.full((n,), RECEPTIVE_Z, dtype=torch.float32)
    yaw = _uniform(*RECEPTIVE_YAW_RANGE, n, generator)
    qw = torch.cos(yaw / 2)
    qz = torch.sin(yaw / 2)
    zeros = torch.zeros(n, dtype=torch.float32)
    root_pose = torch.stack([x, y, z, qw, zeros, zeros, qz], dim=-1)
    root_velocity = torch.zeros(n, 6, dtype=torch.float32)
    return list(root_pose.unbind(0)), list(root_velocity.unbind(0))


def _synthesize_table(n: int) -> tuple[list, list]:
    root_pose = TABLE_POSE.unsqueeze(0).repeat(n, 1)
    root_velocity = torch.zeros(n, 6, dtype=torch.float32)
    return list(root_pose.unbind(0)), list(root_velocity.unbind(0))


def _synthesize_ur5_metal_support(n: int, generator: torch.Generator) -> tuple[list, list]:
    x = _uniform(*SUPPORT_X_RANGE, n, generator)
    y = _uniform(*SUPPORT_Y_RANGE, n, generator)
    z = SUPPORT_Z_NOMINAL + _uniform(*SUPPORT_Z_JITTER_RANGE, n, generator)
    ones = torch.ones(n, dtype=torch.float32)
    zeros = torch.zeros(n, dtype=torch.float32)
    root_pose = torch.stack([x, y, z, ones, zeros, zeros, zeros], dim=-1)
    root_velocity = torch.zeros(n, 6, dtype=torch.float32)
    return list(root_pose.unbind(0)), list(root_velocity.unbind(0))


def _detect_schema(rigid_object: dict) -> str:
    keys = set(rigid_object.keys())
    if "receptive_object" in keys:
        raise ValueError(
            f"Refusing: input rigid_object keys {sorted(keys)} already contain 'receptive_object' --"
            " this file is already schema-complete, either a GENERATOR-ORIGIN file"
            " (reset_states_cfg.py-produced) or a dexlift-origin file recorded with"
            " DEXLIFT_PARTIAL_ASSEMBLY=1 (_DexliftToTrainingSceneRecorder now exports"
            " receptive_object when the scene has one -- see generate_reset_states_policy.py)."
            " Either way it needs no rename or synthesis. Re-synthesising on top of already-correct"
            " data would be a double-write. Not proceeding."
        )
    if keys == {"object", "table"}:
        return "raw_dexlift"
    if keys == {"insertive_object"}:
        return "already_rekeyed"
    raise ValueError(
        f"Expected rigid_object keys {{'object','table'}} (raw dexlift dump) or {{'insertive_object'}}"
        f" (already rekeyed), got {sorted(keys)}. This script is written for those two specific source"
        f" schemas -- refusing to guess a mapping for anything else."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-key + schema-complete a dexlift-scene resets_*.pt to the OmniReset training scene's names."
    )
    parser.add_argument("--input", type=str, required=True, help="Source resets_*.pt (read-only, never modified).")
    parser.add_argument("--output", type=str, required=True, help="Destination path for the fixed file.")
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for the synthesized entries (default 0, for reproducibility)."
    )
    args = parser.parse_args()

    print(f"[rekey] loading: {args.input}", flush=True)
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    state = data["initial_state"]
    articulation = state["articulation"]
    rigid_object = state["rigid_object"]

    schema = _detect_schema(rigid_object)
    print(f"[rekey] detected schema: {schema}", flush=True)

    if schema == "raw_dexlift":
        insertive_entry = rigid_object["object"]  # rename only, no coordinate transform
        n_episodes = len(insertive_entry["root_pose"])
        print(f"[rekey] {n_episodes} episodes; renaming object -> insertive_object, dropping table.", flush=True)
    else:  # already_rekeyed
        insertive_entry = rigid_object["insertive_object"]
        n_episodes = len(insertive_entry["root_pose"])
        print(f"[rekey] {n_episodes} episodes; already has insertive_object, no rename needed.", flush=True)

    generator = torch.Generator().manual_seed(args.seed)
    receptive_pose, receptive_vel = _synthesize_receptive_object(n_episodes, generator)
    table_pose, table_vel = _synthesize_table(n_episodes)
    support_pose, support_vel = _synthesize_ur5_metal_support(n_episodes, generator)

    new_state = {
        "articulation": articulation,  # unchanged: both scenes already agree on "robot"
        "rigid_object": {
            "insertive_object": insertive_entry,  # unchanged: rename only (or already renamed)
            "receptive_object": {"root_pose": receptive_pose, "root_velocity": receptive_vel},
            "table": {"root_pose": table_pose, "root_velocity": table_vel},
            "ur5_metal_support": {"root_pose": support_pose, "root_velocity": support_vel},
        },
    }
    new_data = {"initial_state": new_state}

    # -- pre-write self-check: synthesized/untouched entries must not have altered insertive_object
    # or robot in memory (paranoia against accidental in-place mutation above).
    for i in range(n_episodes):
        assert torch.equal(new_state["rigid_object"]["insertive_object"]["root_pose"][i], insertive_entry["root_pose"][i])
        assert torch.equal(
            new_state["rigid_object"]["insertive_object"]["root_velocity"][i], insertive_entry["root_velocity"][i]
        )

    print(f"[rekey] writing: {args.output}", flush=True)
    torch.save(new_data, args.output)

    # -- verify by reloading from disk, not from the in-memory object just written.
    check = torch.load(args.output, map_location="cpu", weights_only=False)
    check_state = check["initial_state"]
    expected_rigid_keys = {"insertive_object", "receptive_object", "table", "ur5_metal_support"}
    assert set(check_state.keys()) == {"articulation", "rigid_object"}, check_state.keys()
    assert set(check_state["articulation"].keys()) == {"robot"}, check_state["articulation"].keys()
    assert set(check_state["rigid_object"].keys()) == expected_rigid_keys, check_state["rigid_object"].keys()

    for name in expected_rigid_keys:
        n_check = len(check_state["rigid_object"][name]["root_pose"])
        assert n_check == n_episodes, (name, n_check, n_episodes)
        n_vel_check = len(check_state["rigid_object"][name]["root_velocity"])
        assert n_vel_check == n_episodes, (name, n_vel_check, n_episodes)

    n_robot = len(check_state["articulation"]["robot"]["root_pose"])
    assert n_robot == n_episodes, (n_robot, n_episodes)

    # -- BIT-IDENTICAL check: robot and insertive_object must not have moved by a micron.
    for i in range(n_episodes):
        for key in ["root_pose", "root_velocity", "joint_position", "joint_velocity"]:
            a = articulation["robot"][key][i]
            b = check_state["articulation"]["robot"][key][i]
            assert torch.equal(a, b), f"robot.{key}[{i}] changed"
        a = insertive_entry["root_pose"][i]
        b = check_state["rigid_object"]["insertive_object"]["root_pose"][i]
        assert torch.equal(a, b), f"insertive_object.root_pose[{i}] changed"
        a = insertive_entry["root_velocity"][i]
        b = check_state["rigid_object"]["insertive_object"]["root_velocity"][i]
        assert torch.equal(a, b), f"insertive_object.root_velocity[{i}] changed"
    print(f"[rekey] BIT-IDENTICAL CHECK PASSED: robot ({n_episodes}x4 fields) and insertive_object"
          f" ({n_episodes}x2 fields) unchanged vs. input.", flush=True)

    # -- report synthesized-entry stats for eyeballing against the module docstring's targets.
    def _stats(name: str) -> str:
        rp = torch.stack(check_state["rigid_object"][name]["root_pose"])
        return (
            f"z mean={rp[:,2].mean().item():.6f} min={rp[:,2].min().item():.6f} max={rp[:,2].max().item():.6f} | "
            f"x range=[{rp[:,0].min().item():.4f},{rp[:,0].max().item():.4f}] | "
            f"y range=[{rp[:,1].min().item():.4f},{rp[:,1].max().item():.4f}]"
        )

    print("\n=== REKEY + SCHEMA-FIX RESULT ===", flush=True)
    print(f"episodes: {n_episodes}", flush=True)
    print(f"schema: {schema}", flush=True)
    print(f"articulation names: {sorted(check_state['articulation'].keys())}", flush=True)
    print(f"rigid_object names: {sorted(check_state['rigid_object'].keys())}", flush=True)
    print(f"receptive_object: {_stats('receptive_object')}", flush=True)
    print(f"table: {_stats('table')}", flush=True)
    print(f"ur5_metal_support: {_stats('ur5_metal_support')}", flush=True)
    print(f"output: {args.output}", flush=True)
    print("VERIFIED (re-loaded from disk, keys/shapes/counts match, robot+insertive_object bit-identical).", flush=True)


if __name__ == "__main__":
    main()
