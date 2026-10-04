# DexReset v2 (Ivan's branch tip) — status, and why I am NOT building on it

## The decisive finding
**Most of the v2 certification layer is STAGED and has never been executed.** These all carry an
explicit "STAGED / PREPARED, NOT RUN" marker at the top of the file:
`cert_g3z4_finetune.sh:3`, `cert_nnlv_vertgoal.sh:6`, `recertify_v1_baselines.sh:2`,
`regenerate_four_banks_post_finetune.sh:2`, `launch_dexreset_s1_s2_bank_gen.sh:2`,
`r0_control.sh:4`, `ramp_stage.sh:3`.
The v2 four-bank regeneration and both finetune gates had produced **no numbers** as of the branch
tip. Every concrete figure on the branch comes from the earlier v1 campaign or from
generation-time acceptance measurements.

=> The branch is a validated **design** with an unvalidated **campaign**.

## Consequence for this work
Build on:
  * the **v1 OmniReset pipeline** — `record_grasps.py` -> `record_reset_states.py` x4 ->
    `rsl_rl/train.py`. This is what the PUBLISHED cube recipe uses
    (`rl_training.rst:638-791`), and cube is one of the two fastest-converging tasks in the paper.
  * Ivan's **DELTO integration** — `ur10e_delto.py`, `delto_cfg.py`, `ur10e_delto_cfg.py`,
    `actions.py`. This part IS validated, for the `deltoblock`/`deltoslot` pair.
Do NOT depend on the v2 rung/gate chain, `validate_c4_bank*.py`, or `run_certify.sh`.

## Numbers the branch DID establish (v1 campaign / generation-time), useful as calibration
- S1/S2' bore-spawn breakthrough (commit 7bd01c5), same instrument, ungated rollouts:
  | | parent ep_3600 | anchored ep_4500 | bore-mixture ep_4300 |
  |---|---|---|---|
  | lateral median | 141.1 mm | 139.3 mm | **1.60 mm** |
  | tilt median | 89.9 deg | 91.0 deg | **7.31 deg** |
  | states/600 steps | 11 | 11 | **62** |
  Bore radial clearance is 0.91 mm, so 1.60 mm lateral median = the leg is IN the bore.
- Reset-type acceptance on the leg pair: `ObjectAnywhereEEAnywhere` **57.21%**,
  `ObjectRestingEEGrasped` **0.83%** (40/4799). C2 is by far the hardest to fill.
- Contact grading beat displacement grading on n=298: **0.9597 vs 0.5369**, disagreement one-sided.
- DexLift stage ladder: Stage1 pass@50 0.891 / @20 0.836 / @10 0.523; Stage2 @30 0.922 / @10 0.805;
  Stage3 @50 0.88 / @30 0.70+-0.035 / @10 0.00.
- **Adding orientation cost a 4.5x precision penalty on the leg task: 26.4 mm vs 4.8 mm median.**
  This is the single best in-repo evidence that the position-only cube variant should be markedly
  easier than the oriented one — i.e. the user's instinct to prioritise it is well founded.

## Traps inherited from the branch that apply to me
1. **`run_certify.sh`'s exit code is not trustworthy** — its last command is a `grep`, and a run
   that died with a traceback still MATCHES the pattern, so grep exits 0
   (`cert_nnlv_vertgoal.sh:232-247`). Any wrapper must check positive evidence: output JSON exists
   and parses, checkpoint hash matches, episode count is right, and a result marker is present.
   I will apply the same rule to my own harnesses.
2. **`eval_reset_type_success_rates.py` hard-asserts `pos_thr < 0.01 and rot_thr < 0.10`** before
   any rollout step. My position-only variant uses `rot_thr = 7.0`, so **that tool cannot grade
   V-B**. Grade V-B through the standard `play.py` / Play-task path instead, or pass the oriented
   threshold purely for that tool's benefit. Noted, not yet actioned.
3. `--dataset_dir` is required with no default in several v2 tools, deliberately.
4. Bank recording sticky-OR: `certify_pose.py` counts an episode successful if the predicate held
   at ANY step, even if the object is dropped afterwards. If I reuse that grading style I must say
   so explicitly in the report rather than let it read as "held at the end".
