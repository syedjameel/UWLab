# OmniReset paper spec — extracted for the cube-stacking replication
Source: arXiv:2603.15789 v3 (ICLR 2026). Full text cached at
`~/.cache/tmp/claude-1000/.../scratchpad/omnireset.txt`.

## Cube Stacking, as the paper describes it
Two clauses total (§4.1): *"The Cube Stacking task requires stacking one cube on top of
another with a desired orientation."* Plus a Fig.4 thumbnail. **No** success thresholds, **no**
cube dimensions, **no** mass, **no** compute, **no** success rate reported. It is not in the
sim2real set and appears in no learning curve, ablation or robustness figure.
=> The repo, not the paper, is the source of truth for cube task parameters.
The "red face up" reading is a figure-only inference; the text neither states nor contradicts it.

## Reward (§3.3 Eq.1, §A.1) — task-agnostic, shared across ALL tasks, no per-task tuning
r = r_success + r_dist + r_reach + r_smooth + r_term

  r_smooth = -λ_act‖a‖² - λ_rate‖a_t - a_{t-1}‖² - λ_vel‖q̇‖²
  r_term   = -λ_abnormal · 1[abnormal state]
  r_reach  = λ_reach · (1 - tanh(‖p_ee - p_obj‖ / σ))
  r_dist   = λ_dist · ½ · [ exp(-‖x_err‖/σ) + exp(-‖θ_err‖/σ) ]      <-- goal frame
  r_success= λ_success · 1[position AND orientation thresholds satisfied]

  λ_reach=0.1  λ_dist=0.1  λ_success=1.0
  λ_act=1e-4   λ_rate=1e-3 λ_vel=1e-2   λ_abnormal=100
  σ: NOT STATED numerically. Thresholds: NOT STATED.

**Orientation is cleanly separable.** The no-orientation variant is exactly:
  1. drop `exp(-‖θ_err‖/σ)` from r_dist (and the ½, to keep scale)
  2. drop the orientation clause from the Eq.6 success indicator
Nothing else in the reward touches orientation.

Paper's own claim: *"performance is largely insensitive to moderate changes in these weights,
and the primary driver of successful learning is the coverage of the initial state distribution
rather than reward shaping."*

## Reset generation (§3.2) — the actual method
User supplies only: target object s_tar, goal set G, workspace W.

Pre-compute A: **grasp bank** — 1000 feasible grasp points on s_tar, IsaacLab grasp sampler,
  **broad** sampling range (Fig.3a; narrow/moderate ablated and are worse, §4.3 Fig.9).
Pre-compute B: **near-goal offsets ("partial assemblies")** — spawn s_tar AT the goal, apply
  small random forces to dislodge it. Yields a continuum of partial-assembly configurations.

Four reset families:
  1. S^R  Reaching     — object on tabletop, gripper at random poses through W. No prereq.
                         This is the FULL-TASK distribution; all headline success rates are
                         measured from it.
  2. S^NO Near-Object  — object across tabletop, EE at a grasp point + small random offset,
                         gripper randomly open OR closed. Needs grasp bank.
  3. S^G  Stable Grasp — object randomly IN THE AIR through W, gripper at a feasible grasp point.
                         Needs grasp bank.
  4. S^NG Near-Goal    — object at a pre-computed offset from goal, gripper in contact as in NO.
                         Needs offset bank + grasp bank.

Sampled OFFLINE, invalid proposals rejected by collision check + a few sim steps to stabilize.
At train time: **Uniform(D^R ∪ D^NO ∪ D^G ∪ D^NG)** — flat, unweighted, NO curriculum.
Cached on-GPU. The backwards curriculum ("solve near-goal first, then the whole task") is
EMERGENT, not imposed.

Explicit non-properties: no graph/ordering between states; no demonstrated dynamics between
them. RL may simply ignore reset states it does not need (drawer policy never grasps).

## Algorithmic decisions (§3.3)
- **num_envs is the single most important knob.** Ablation sweeps 2048..65536 (§4.3 Fig.8).
  Small num_envs still solves near-goal but FAILS to scale to the full task from reaching states.
  PPO batch size scales with num_envs.
- Asymmetric actor-critic: actor sees history of **5 previous timesteps** {robot state, all
  object poses, previous actions}. Critic sees the same PLUS privileged params (masses etc.)
  and **no history**. Giving the actor more led to less stable training.
- **gSDE** exploration noise (state-dependent, temporally correlated). Called crucial for
  heterogeneous multi-stage tasks.

## Control (§A.3.4-A.3.6)
- Torque-level **operational space control**, K_d = 2√K_p, pose error clipped.
- policy 10 Hz | torques 500 Hz | sim dt 1/120
- Action: **relative Cartesian** 6-D EE deltas + binary gripper. Absolute Cartesian or large
  scales "fail to learn".
- Action scale (train): (0.02,0.02,0.02, 0.02,0.02,0.2) for (x,y,z,rx,ry,rz).
  The 10x larger rz is deliberate, "to encourage exploration of twisting motions".
- Action scale (sim2real): (0.01,0.01,0.002, 0.02,0.02,0.2), reached by a success-conditioned
  curriculum. Hard clipping instead collapses training.
- Torque clipping applied ONLY at real deployment, never in sim training.

## Compute actually reported (§A.3.9)
  Peg Insertion     ~8 h   on 4x L40S   (from scratch)
  Leg Twisting      ~16 h  on 4x L40S
  Drawer Assembly   ~24 h  on 4x L40S
  Cube Stacking     NOT STATED
Distillation: 80k trajs ~24 GPU-h on a 3090; student training ~350k iters ~2 days on 1x H200.
Sim2real dynamics curriculum is NOT needed for sim-only results (their sim numbers are
trained without it) -- so we can skip it entirely for a sim-only deliverable.

## Seed selection (§A.3.10)
Seeds produce *behaviourally different* policies, not just noisier ones. Selection is by an
OFFLINE PROXY: evaluate each seed **under injected action noise**, pick the highest. Correlates
with real-world performance. Seed count and noise magnitude: NOT STATED.

## NOT STATED anywhere (we must choose these ourselves)
PPO lr/clip/GAE-λ/γ/entropy/epochs/minibatches/horizon; num_envs for headline runs; actor-critic
MLP sizes; episode length; total steps; σ in r_reach and r_dist; all success thresholds;
cube size and mass.

## Reference sim success rates (Table 1, state RL, their robot)
  Peg 94.97%   Leg 93.94%   Drawer 89.60%
