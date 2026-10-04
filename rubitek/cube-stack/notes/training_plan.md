# Training configuration — verified against the paper

## What the repo already implements (agents/rsl_rl_cfg.py:26-56, Base_PPORunnerCfg)
    num_steps_per_env  32          max_iterations 40000 (a CEILING, not a target)
    actor/critic       [512,256,128,64] elu, obs normalization on both
    noise_std_type     gsde        <- the paper's §3.3 decision 4, already here
    lr 1e-4 adaptive   clip 0.2    entropy 0.006   epochs 5   minibatches 4
    gamma 0.99  lam 0.95  desired_kl 0.01  max_grad_norm 1.0
`UR10eDelto_PPORunnerCfg` only changes `experiment_name` (-> `ur10e_delto_omnireset_agent`).

## Env timing (rl_state_cfg.py:876-879)
    sim.dt 1/120   decimation 12   episode_length_s 16.0
=> **10 Hz control**, 160 control steps per episode. Matches the paper's stated 10 Hz policy /
120 Hz sim exactly. (The paper's 500 Hz is the torque loop, which OSC runs internally.)

## Batch size
num_envs x 32. At 4096 envs that is 131,072 transitions per update — squarely in the regime the
paper's §4.3 ablation says matters (it sweeps 2048..65536 envs and finds small counts solve
near-goal but fail the full task from reaching states).

## Iteration budget
The paper's own figures run to **2000-2500 updates** (Fig. 8 to 2000, Fig. 9 to 2500). So the
target is ~2000 iterations, NOT the config's 40000 ceiling — `--max_iterations 2000` must be passed
explicitly or the run never stops.

## Revised cost estimate
Earlier I budgeted ~24 h/variant on 4x3090 = $14 each. With 2000 iterations at 4096 envs the real
figure is far lower: at a few seconds per iteration a run is single-digit hours on ONE good GPU.
Plan: single-GPU runs first (1x RTX 4090 @ $0.335/h, or 1x 3090 @ $0.148/h), measure s/iter on the
real task, and only scale to `--distributed` if the measured rate says 2000 iterations would
otherwise take longer than ~8 h. Re-estimate from measurement rather than committing up front.

## What I am NOT doing, and why
- **No sim2real dynamics curriculum.** The paper's own simulation-only numbers are trained without
  it (§5, §A.3.9); it exists to close the reality gap and would only spend compute here.
- **No Stage-2 finetune** for the same reason.
- **No distillation to a vision student.** Out of scope for a state-policy deliverable.

## Seeds
The paper selects among seeds by an OFFLINE PROXY: evaluate each under injected action noise and
take the best, because seeds differ in BEHAVIOUR (some learn drop-and-regrasp strategies that are
brittle), not just in noise. Seed count and noise magnitude are not stated. Plan: 2 seeds per
variant if budget allows after the first run lands, ranked by success under action noise — and say
plainly in the report if only one seed was affordable.
