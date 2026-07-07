# What the SysID Numbers Mean — and Everything That Happens After

An intuition-level companion to `UR10E_SIM2REAL_PROCEDURE.md` (which has the exact
commands). This explains **what the 25 identified parameters physically are, what they do
once we have them, and why each remaining step of the pipeline exists** — so the procedure
makes sense, not just runs.

---

## 1. The problem the parameters solve

Our simulated UR10e and the real UR10e are told to do the same thing by the same
controller — yet they move differently. The links (masses, lengths) match, so what's left?

**The joints themselves.** A real UR joint is not an ideal hinge: it is a small, fast motor
behind a ~100:1 harmonic-drive gearbox, plus grease, seals, bearings, and a firmware
control loop. None of that exists in the URDF. The five parameter types are precisely the
physics of "a real geared joint", added on top of the ideal model:

### armature (×6 joints) — "the hidden flywheel"

The motor rotor spins ~100× faster than the joint. Reflected through the gearbox, its tiny
inertia appears **gear-ratio² ≈ 10 000× bigger** at the joint. So every joint drags an
invisible flywheel that has nothing to do with the link masses. Effect: the joint is
*harder to accelerate and harder to stop* than the CAD model predicts. In PhysX this adds
to the diagonal of the joint-space mass matrix. Bigger arm → bigger motors → the UR10e's
base joints need much larger armature than the UR5e's (this is one of the values that
slammed into the search bound on the first fit).

### static friction (×6) — "the sticky floor"

The torque needed to *start* moving (stiction). Below it, commanded torque produces zero
motion — a dead zone. Effects you can see: the arm ignores small corrections, sticks
momentarily every time a joint reverses direction, and needs a torque "kick" to break
loose. In the fit plot this is why sim wrist_1 sat perfectly still (flat red line) early in
the chirp while the command was ramping — correct stiction reproduces the real dead zone.

### dynamic_ratio (×6) — "easier to keep moving than to start"

Kinetic friction = `dynamic_ratio × static friction`, always ≤ 1. Once a joint breaks
loose, friction drops. This asymmetry produces the characteristic stick-slip feel of geared
joints at low speed.

### viscous friction (×6) — "moving through honey"

Torque proportional to velocity — gear mesh and grease churn. This is the arm's built-in
**damping**: it's what makes oscillations die out. It is exactly what our first fit was
starving for on the shoulder-lift joint: the bound capped viscous at 20, the sim kept
ringing for seconds after the real joint had settled, and the lift's RMSE (2.64°) was the
one failure. Real big joints churn a lot of grease.

### motor delay (×1, in 500 Hz steps) — "the reaction time"

Commands don't act instantly: communication + firmware + current-loop add a few
milliseconds. At 500 Hz, "delay = 4" means the torque you command now takes effect 8 ms
later. Effect: phase lag; with stiff gains, enough lag turns correction into overshoot.
The sim implements it by buffering actuator commands (`DelayedPDActuator`).

### How CMA-ES finds them

We shook the real arm with a chirp (0.1→3 Hz frequency sweep) and recorded exactly what it
did. Then, in simulation, **512 copies of the robot replay the identical commands, each
with a different guess of the 25 numbers**. The guesses whose motion best matches the real
recording "reproduce", the population's spread adapts, repeat ×200 generations. That's it —
evolutionary curve-fitting where the "curve" is a full physics simulation. A good result
means: *the simulated robot now moves like your specific robot, flaws included* —
the target is < 2° RMSE per joint over the whole 8 s chirp.

---

## 2. What we do with the numbers — the road after the fit

### Step 1 — the numbers go into `metadata.yaml` (and nothing changes yet)

The `sysid:` block in `Ur10eLinearGripper/metadata.yaml` is pure data. Stage-1 training
never reads it. It exists for exactly one consumer: the finetune's domain randomization.

### Step 2 — why finetuning works this way (the ADR curriculum)

Here's the counterintuitive part, straight from the paper: **training with realistic
friction from the start fails.** A policy that has never moved can't explore when every
joint is sticky — it never discovers the task. So the pipeline deliberately trains Stage 1
on an *idealized* robot (zero friction, zero armature, zero delay) where exploration is
easy, and only then confronts the policy with reality, gradually:

- The finetune curriculum **ramps the sim dynamics from ideal toward YOUR identified
  values** (randomized ×0.8–1.2 around them, delay 0–2 steps), *conditioned on the policy
  still succeeding* — like adding weight to the bar only after each clean lift.
- Simultaneously it **raises the OSC gains** (a sticky robot needs firmer control — soft
  gains that worked frictionless would stall in the dead zones) and **shrinks the action
  scale** (slower, smaller motions; especially vertical z, so the gripper stops ramming
  things during contact — smooth motion transfers, flailing doesn't).

Why randomize *around* the identified values instead of using them exactly? Because the
fit is good but not perfect, friction changes with temperature and pose, and a policy
trained on a ±20% cloud of "robots like yours" is robust to landing anywhere inside it.
That's the whole sim2real bet: **make reality just another sample from the training
distribution.**

### Step 3 — sim hardening first (two small things, done before finetune)

- **Gripper mass 1.1 → 0.575 kg**: the sim wrist currently swings a phantom extra ~0.5 kg
  (URDF overestimate vs the weighed reality). Wrist inertia shapes everything the policy
  feels through its hand.
- **Wrist limits ±360° → ±180°**: the sim lets wrists wind up in ways the real safety
  system would abort. Cheaper to make such strategies unlearnable than to discover them at
  the robot.

Both alter dynamics/data → do them once, re-record resets, then finetune.

### Step 4 — pick the right brain before polishing it

Different training seeds learn genuinely different *strategies* for the same task (one
inserts directly; another drops and re-grasps — clever in sim, fragile on hardware). The
proxy test: inject action noise and measure which seed's success degrades least. Noise
robustness correlates with real-world transfer. Polish (finetune) only the winner.

### Step 5 — validate where it hurts: in contact

Our hard-won P6 lesson, now a rule: any stiff-gain configuration (the finetune-eval OSC
runs rot Kp 50) must be checked **with the jaws closed on an object**, not in free space.
The failure mode is invisible until contact: the mass-less PD + stiff jaws + object forms
a feedback loop that limit-cycles the wrist at its velocity limit. Free-space probes pass;
the grasp chatters. Five minutes of test, hours of debugging saved.

### Step 6 — the deployment gap: the expert is blind without the simulator

The Stage-2 policy still *observes object poses* — an oracle only the simulator provides.
The real world has no pose oracle, so the expert **cannot run on hardware**. The paper's
answer is student–teacher distillation:

1. The finetuned expert (with oracle eyes) rolls out ~80 000 successful episodes in sim
   while three virtual cameras record — under aggressive **visual** randomization
   (hundreds of lighting conditions, textures, camera jitter) so no single "look" is
   memorized.
2. A camera-only student (ResNet-18 + small MLP, 5-frame history, 10 Hz) is trained to
   imitate the expert's action *distribution* (KL matching) plus an auxiliary "where is
   the object" reconstruction loss that forces the visual encoder to actually localize.
3. Deployed zero-shot: the student has effectively already seen thousands of worlds;
   the real lab is just one more. Expect the student to score noticeably below the expert
   in sim (~50–60%) yet transfer well (paper: 85% real success on peg insertion).

For our rig this stage additionally needs (custom-hardware items, no upstream reference):
a **real-world driver for the linear gripper** (the stack only knows the Robotiq URCap), a
**wrist-camera mount** designed for our gripper, camera calibration (ArUco + interactive
sim-overlay alignment), and a UR10e variant of the RGB data-collection config.

### The end state, one breath

At 10 Hz, a ResNet looks through three cameras and outputs a small end-effector nudge; at
500 Hz, the same OSC we validated byte-for-byte against sim turns nudges into joint
torques on a robot whose simulated twin — armature, stiction, grease, reaction time and
all — is where the policy grew up. Everything in between (chirp, CMA-ES, metadata, ADR,
distillation) exists to make those two loops agree.

---

## Where each thing lives

| thing | place |
|---|---|
| Identified params (after §1 of this doc) | `Ur10eLinearGripper/metadata.yaml` → `sysid:` |
| Who consumes them | `randomize_arm_from_sysid(_fixed)` events (Finetune tasks only) |
| The curriculum that ramps them | `rl_state_cfg.py` finetune events/curriculum (`adr_sysid`) |
| Exact commands for every step | `UR10E_SIM2REAL_PROCEDURE.md` §5–§9 |
| The theory | OmniReset paper (`2603.15789v3.pdf`) A.1, A.3; PACE (arXiv:2509.06342) for sysid |
