# Plan — cube stacking on UR10e + Tesollo DG-5F

## Deliverables
Two trained state-based RL policies, with video evidence, plus an HTML report.
- **V-B  position-only** (PRIMARY, per the user): success = position threshold alone.
- **V-A  oriented** (paper-faithful): success = position AND euler-xy orientation.
  Note the codebase already drops YAW (`rewards.py:141-143`), so "oriented" means roll/pitch only —
  which is exactly a "this face must end up up" constraint. Consistent with the paper's figures.

## Fixed design decisions (with the evidence for each)
1. **Use the paper's own 40 mm cubes.** `InsertiveCube`/`ReceptiveCube`, already registered in all
   four variant registries. 40 mm sits inside the DG-5F validated grasp window (24.67-41.48 mm) and
   well under `maximum_aperture` 59 mm. No asset authoring needed.
2. **Robot = UR10e + DG-5F** (`OmniReset-UR10eDelto-*`). The UR10e arm is unchanged from Ivan's
   validated linear-gripper stack; the sysid block is a byte-identical copy.
3. **Reuse the cloud `partial_assemblies.pt`** (arm/gripper independent). Regenerate the grasp bank
   and all four reset banks for the DELTO articulation — the cloud ones are 2F-85 and will
   shape-mismatch at reset.
4. **Skip the sim2real dynamics curriculum entirely.** The paper's own sim-only numbers are trained
   without it (§5, §A.3.9). This is a sim deliverable; the curriculum only costs compute.
5. **Loosen V-A's orientation threshold 0.025 -> 0.05 rad.** Sanctioned explicitly by
   `new_task.rst:92` for cube stacking; the shipped 0.025 is the tight-fit value meant for screw
   insertion.
6. **Add the `_apply_delto_dataset_dir` seam to `ur10e_delto_cfg.py`.** It is missing there (the
   UR5e variant has it), so every UR10eDelto task silently defaults to the cloud 2F-85 dataset dir.

## The one open experiment: reduced hand actuation
Hypothesis: for a 40 mm cube we only need the hand's designed two-finger jaw, so actuating fingers
1+2 (8 joints) and curling 3/4/5 out of the way cuts the action space **26 -> 14** and should
materially speed up RL — which directly buys budget.

Why {1,2} and not any other subset: the grasp geometry is defined as "a virtual jaw of finger 1
against finger 2" (`DeltoHand/metadata.yaml:8`) and `grasp_center_offset` is
`midpoint(rl_dg_1_tip, rl_dg_2_tip)` (`:55-59`). The 5-tip centroid was explicitly rejected. So
{1,2} is the only subset that leaves the calibrated grasp centre meaningful.

**This is a hypothesis, not a decision.** It will be settled by evidence, cheaply and locally:
run the grasp sampler for the cube under BOTH configurations and compare grasp yield and the
`UWLAB_GRASP_DEBUG=1` rejection breakdown. Full-20 is the fallback and stays implemented.
Risk to watch: `enabled_self_collisions=True`, so the curled posture must not self-collide, and
must not change the approach axis.

## Stages
| # | stage | where | evidence produced |
|---|---|---|---|
| 0 | local Isaac Sim 5.1 env | local | import + empty-sim launch |
| 1 | cube pair constructs on UR10eDelto | local | smoke test PASS |
| 2 | config work: dataset seam, V-B reward variant, reduced-hand variant | local | diff + guard passes |
| 3 | grasp bank for cube (both hand configs) | local | **contact-sheet render of real grasps**; yield numbers |
| 4 | four reset banks C1-C4 | local | **reset-state viewport renders**; bank QC verdicts |
| 5 | short training smoke, ~200 iters | local | reward curve moves; no NaN |
| 6 | full training, both variants | vast 4x3090 | success-rate curves |
| 7 | eval + rollout video from the reaching distribution | local (RTX + driver 580) | **the money shot: successful stacks on video** |
| 8 | HTML report | local | tabbed, media-led |

Rendering is local-only by necessity: the guide requires RTX-class + driver branch 580
(this box is 4070 Ti Super / 580.159.04); A100/H100 cannot render at all, and driver 595
segfaults `rtx.scenedb` on Isaac 5.1.

## Measurement discipline
The headline number is success rate from the **reaching** distribution
(`ObjectAnywhereEEAnywhere`, probs 1.0) — the same distribution the paper reports, not the
easy near-goal one. Near-goal / grasped / near-object rates are reported alongside as the
decomposition, exactly as the paper's Fig. 8 panels do.
Reference points: paper's state-RL sim numbers are Peg 94.97%, Leg 93.94%, Drawer 89.60%.
