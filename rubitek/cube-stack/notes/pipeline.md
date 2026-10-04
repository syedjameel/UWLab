# OmniReset pipeline + the cube pair — as it actually is

## HEADLINE: cube/cube is already fully specified
Not half-wired. Registered in all four variant registries, both roles. Cloud assets verified live
by HTTP (2026-08-31):

    Props/Custom/InsertiveCube/insertive_cube.usd   200  11900 B
    Props/Custom/InsertiveCube/metadata.yaml        200    141 B
    Props/Custom/ReceptiveCube/receptive_cube.usd   200  10193 B
    Props/Custom/ReceptiveCube/metadata.yaml        200    198 B

    # InsertiveCube/metadata.yaml
    assembled_offset: {pos: [0,0,-0.02], quat: [1,0,0,0]}
    bottom_offset:    {pos: [0,0,-0.02], quat: [1,0,0,0]}
    # ReceptiveCube/metadata.yaml
    assembled_offset: {pos: [0,0, 0.02], quat: [1,0,0,0]}
    bottom_offset:    {pos: [0,0,-0.02], quat: [1,0,0,0]}
    success_thresholds: {position: 0.005, orientation: 0.025}

**40 mm cubes.** `UWLAB_CLOUD_ASSETS_DIR` is not an env var — it is hardcoded at
`uwlab_assets/__init__.py:25` to the HuggingFace `UW-Lab/uwlab-assets` repo and fetched+cached
by `resolve_cloud_path` (`mdp/utils.py:339-365`).

### 40 mm vs the DG-5F grasp window — THE key compatibility fact
Validated window 24.67 .. 41.48 mm (`DeltoBlock/metadata.yaml:7`); `maximum_aperture` 59 mm.
**40 mm fits**, though near the top of the window. So the paper's own cube is graspable by this
hand with no resize. If grasp yield turns out poor, the fallback is a scaled cube (deltoblock is
34 mm and sits mid-window) — but that departs from the paper's asset.

Shipped `orientation: 0.025` rad is the TIGHT value; `new_task.rst:92` explicitly suggests
`position: 0.005, orientation: 0.05` for cube stacking. Loosening it is sanctioned by the docs.

## Pre-generated cloud datasets exist — but only for the WRONG robot
    Resets/InsertiveCube__ReceptiveCube/resets_ObjectAnywhereEEAnywhere.pt           38.2 MB
    Resets/InsertiveCube__ReceptiveCube/resets_ObjectRestingEEGrasped.pt             37.6 MB
    Resets/InsertiveCube__ReceptiveCube/resets_ObjectAnywhereEEGrasped.pt            37.7 MB
    Resets/InsertiveCube__ReceptiveCube/resets_ObjectPartiallyAssembledEEGrasped.pt  41.4 MB
    Resets/InsertiveCube__ReceptiveCube/partial_assemblies.pt                        18.7 KB
    Grasps/InsertiveCube/grasps.pt                                                    5.0 MB
Recorded with UR5e + Robotiq 2F-85. Unusable on a 26-DOF UR10e+DELTO articulation.

**What we must regenerate vs reuse:**
| artifact | status |
|---|---|
| `partial_assemblies.pt` | **REUSE** — pure relative pose, arm/gripper independent (`events.py:1555-1568`) |
| `Grasps/InsertiveCube/grasps.pt` | **RE-RECORD** — 2F-85 joint names; hard-fails at `events.py:1374` |
| all four `resets_*.pt` | **RE-RECORD** — articulation shape mismatch at reset, "proven live" |

## The four training banks (a fifth exists but is never sampled)
`rl_state_cfg.py:273-288` TrainEventCfg samples exactly:
  ObjectAnywhereEEAnywhere (C1, Reaching)      probs 0.25   deps: none
  ObjectRestingEEGrasped   (C2, Near-Object)   probs 0.25   deps: C1 bank + grasp bank
  ObjectAnywhereEEGrasped  (C3, Grasped)       probs 0.25   deps: grasp bank
  ObjectPartiallyAssembledEEGrasped (C4, Near-Goal) 0.25    deps: partial assemblies + grasp bank
`ObjectPartiallyAssembledEEAnywhere` is registered and recordable but NEVER sampled at train time.
Eval configs use `["ObjectAnywhereEEAnywhere"], probs [1.0]` — i.e. **the headline number is
measured from the full-task reaching distribution**, matching the paper.
Order matters: C1 must be recorded before C2. C4 is the slowest.
Guide warns under-recording C4 is what produces "holds cube near box, stuck".

## Pipeline order
1. `record_partial_assemblies.py --task OmniReset-PartialAssemblies-v0` (~30 s for cube) [we reuse cloud]
2. `record_grasps.py --task OmniReset-Delto-GraspSampling-v0 env.scene.object=cube` (~1 min for 2F-85)
3. `record_reset_states.py` x4 with `OmniReset-UR10eDelto-Object*-v0`, `--num_envs 4096
   --num_reset_states 10000 --dataset_dir <dir>`  (1 min .. multiple hours)
   QC: `visualize_reset_states.py`, `conversions/qc_reset_states_ur10e.py`, `filter_reset_states.py`
4. `rsl_rl/train.py --task OmniReset-UR10eDelto-RelCartesianOSC-State-v0 --num_envs 16384`
   with `env.events.reset_from_reset_states.params.dataset_dir=<dir>`
5. (sim2real only — SKIP) Finetune, camera align, RGB collect, distillation

## TRAP: dataset_dir is not robot-discriminated
`compute_pair_dir` (`mdp/utils.py:384-404`) = `"__".join(sorted(parent-dir names))`. No arm or
gripper in the key. `_apply_delto_dataset_dir` (`delto_cfg.py:363-393`) exists to repoint it, and
**`ur5e_delto_cfg.py` calls it but `ur10e_delto_cfg.py` NEVER DOES** — every UR10eDelto task still
defaults to the cloud `Datasets/OmniReset`. => pass `--dataset_dir` / `env.events.*.params.dataset_dir=`
on EVERY UR10eDelto command, or add the seam call. **I will add the seam call.**

## TRAP: axial_depth_sampling is hardwired to the table leg
`PartialAssembliesEventCfg.axial_depth_sampling` (`partial_assemblies_cfg.py:318-337`) is
unconditionally parameterised with `LEG200MM_ONELEGFIXTURE_*` thread constants (`:55-196`, incl. a
21-row thread-yaw table). `sample_axial_insertion_depth.__init__` has two raise-loudly checks
(`events.py:1835-1848`, `:1865-1884`) comparing those literals against the live metadata.
For cube (`seat_local_z=+0.02`) this is meaningless and likely a hard raise.
=> Use the cloud `partial_assemblies.pt`, or override the params from the CLI.

## Orientation: exactly where it lives
`ProgressContext.__call__` (`rewards.py:117-159`) is the single source of truth:
    :142-143  e_x, e_y, _ = euler_xyz_from_quat(...);  euler_xy_distance = |wrap(e_x)| + |wrap(e_y)|
              ** YAW IS DELIBERATELY DROPPED ** (comment :141 "yaw could be different")
    :144      xyz_distance = ||rel_pos||
    :145      position_aligned    = xyz_distance      < success_position_threshold
    :146      orientation_aligned = euler_xy_distance < success_orientation_threshold
    :147      success = orientation_aligned & position_aligned

So the paper's "desired orientation" is a **roll/pitch constraint only** — the cube may be rotated
freely about Z. "Red face up" is exactly an euler-xy constraint. Consistent.

Reward terms (`rl_state_cfg.py:561-603`):
| term | rewards.py | weight | uses orientation |
|---|---|---|---|
| `ee_asset_distance_tanh` | :52-83 | 0.1 | **NO** (computes pose error at :72-79, discards the angle-axis half, uses only ‖pos‖ at :81) |
| `ProgressContext` | :117 | 0.1 | context only, returns zeros |
| `dense_success_reward` | :162-177 | 0.1 | **YES** — mean(exp(-angle/std_angle), exp(-xyz/std)), 50/50 |
| `success_reward` | :180-184 | 1.0 | **YES** — orientation_aligned & position_aligned |

Maps 1:1 onto the paper: `ee_asset_distance_tanh` = r_reach (λ=0.1), `dense_success_reward` = r_dist
(λ=0.1, the ½[exp+exp] blend), `success_reward` = r_success (λ=1.0).

### Orientation-free variant = three edits (or one)
Cleanest single point: make `success_orientation_threshold` huge. It flows `commands.py:181` ->
`rewards.py:126` and is the ONLY consumer of orientation in reward and termination.
Full explicit form:
  1. `rewards.py:184` drop `orientation_aligned &`
  2. `rewards.py:176-177` drop the `angle_diff` row from the stack
  3. `rewards.py:146` force `orientation_aligned[:] = True`
`commands.py:233` / `terminations.py:490` recompute independently and would keep logging a now
unused orientation metric — harmless, and actually useful as a free diagnostic.
Observations (`rl_state_cfg.py:403-453`) still carry full 6-DoF poses; leaving them is the
paper-faithful choice (the actor always saw orientation).

## Other facts worth keeping
- `make_receptive_object` sets `kinematic_enabled=True` — the bottom cube is IMMOVABLE. So this is
  "place cube on a fixed cube", not a physics-toppling stack. Lowers difficulty; matches the paper.
- Training `TerminationsCfg` (`rl_state_cfg.py:607-612`) is **timeout + abnormal_robot only** — no
  success termination. Authors' design; keep it.
- `assembly_keypoints.py` on this branch is 52 lines, only a generic `Offset` dataclass. Mating
  frames live entirely in `metadata.yaml`. There is no per-object keypoint table.
- `_PINNED_OFFSET_LITERALS` (`commands.py:49-99`) returns early for anything that is not
  SquareTableLeg200mm / OneLegInsertionFixture -> a no-op for cube.
- `threshold_curriculum` (`rl_state_cfg.py:657-719`) can ramp both thresholds; `enabled: False`.
- Agent cfg: `agents/rsl_rl_cfg.py:60` `UR10eDelto_PPORunnerCfg`.
