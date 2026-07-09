# UR10e + Linear Gripper — Sim2Real Procedure (P8)

The complete sim2real record for the UR10e + custom linear-gripper OmniReset port: what was
done (with the exact commands as executed), what is running, and every remaining step in
order. Companion to `UR10E_PIPELINE_README.md` (the sim-only pipeline: USD rebuild, dataset
generation, training, play).

Sources of truth: the official sim2real doc
(https://uw-lab.github.io/UWLab/main/source/publications/omnireset/sim2real.html) and the
OmniReset paper (`2603.15789v3.pdf` at repo root, esp. Appendix A.3).

- **Robot:** UR10e, IP `192.168.0.100` (ssh `root@`, serial 20185000717), PolyScope 5.25
- **Sim repo:** `~/work/repos/UWLab`, branch `omnireset/ur10e-linear-gripper` (fork `syedjameel/UWLab`)
- **Real-robot repo:** `~/work/repos/diffusion_policy` — our UR10e changes live on branch
  **`ur10e-linear-gripper` of the fork `syedjameel/diffusion_policy`** (based on the
  `omnireset` branch of WEIRDLabUW/diffusion_policy). Colleagues:
  `git clone -b ur10e-linear-gripper https://github.com/syedjameel/diffusion_policy.git`

## Status at a glance

| step | state |
|---|---|
| 1. Kinematic calibration | **RESOLVED: proceed with NOMINAL** (factory calibration lost — see §1) |
| 2. Real-side UR10e port (kinematics module, payload, collect script) | **DONE** (§2) |
| 3. Chirp data collection on the real UR10e | **DONE** → `~/sysid_data_ur10e_real.pt` (§3) |
| 4. CMA-ES sysid fit | **DONE** — 3 rounds; round 3 accepted (§4) |
| 5. Fit verification (plot, <2°/joint) | **DONE** — pan 0.32 / lift 1.93 / elbow 1.31 / w1 1.11 / w2 0.34 / w3 0.27° (§5) |
| 6. Sysid params → metadata.yaml | **DONE** — `Ur10eLinearGripper/metadata.yaml` sysid block = real UR10e values (§6) |
| 7. Sim hardening before finetune | **DONE** — gripper mass 0.575 kg + wrist ±180° limits both in the graft; **re-record resets before finetune** (14–27% of the old states violate the new wrist limits) (§7) |
| 8. Stage-2 finetune (ADR) + eval-gain validation in contact | **RUNNING** — launched 2026-07-06 after dataset QC + salvage (§7b); eval gains validated in contact; watch `Curriculum/adr_sysid/scale_progress` → 1.0 (§8.1) |
| 9. Real deployment path (RGB distillation, cameras, gripper driver) | **IN PROGRESS** — sim RGB cfgs built; real-side code done incl. the 90° rig-orientation fix (§10.2a); lean `robodiff_real` env built; calibration chain dry-run-validated on the real front D405 (2026-07-09). Remaining: final-mount camera calibration (§10.3) → expert export → 80k collection → student training → deploy (§10) |

---

## 1. Kinematic calibration — outcome: NOMINAL

The official flow extracts the factory calibration and regenerates the URDF. We ran it:

```bash
sudo apt install ros-humble-ur-calibration
source /opt/ros/humble/setup.bash
ros2 launch ur_calibration calibration_correction.launch.py \
  robot_ip:=192.168.0.100 target_filename:=$HOME/ur10e_calibration.yaml
```

**Finding (2026-07-03): the robot returned ALL-ZERO deltas / pure nominal DH.** Confirmed
from controller files (`~/urcontrol_from_ur10e/` on the laptop): no `calibration.conf`
anywhere, active `urcontrol.conf` byte-identical to the stock template, firmware re-imaged
2026-06-15 → **the per-robot factory kinematic calibration was wiped by the reflash**.
(`robot_calibration_summary.txt` is MOTOR calibration — torque constants — not kinematics.)

**Decision: proceed with nominal kinematics.**
- Sim already uses nominal (FK cross-checked to <0.5 mm vs Isaac in P2) — nothing to change.
- The robot's own controller also runs nominal now, so robot-reported TCP agrees with our
  model; the residual is physical manufacturing tolerance (typically a few mm at the EE).
- Risk (paper A.3.1): uncalibrated kinematics degrade insertion precision. If real insertion
  struggles later, the fix is factory recalibration via UR/distributor (needs their fixture;
  the `calibration_trajectory_*.ct` files on the controller are for that procedure). After
  any future calibration: replace `local/Robots/UR10e/ur10e.urdf`, rebuild USDs, re-extract
  metadata, **re-record resets + retrain Stage 1** (geometry-dependent), verify FK <0.01 mm
  via `collect_fk_pairs.py` (sim) + `diffusion_policy scripts/sim2real/test_fk_comparison.py`.

Bonus from `urcontrol.conf [Link]`: UR's own dynamic model — `mass=[7.369, 13.051, 3.989,
2.100, 1.980, 0.615]` (URDF wrist_3 is 0.202 — 3x lighter!), CoM + inertia per link,
gravity 9.82. CAUTION: UR frames ≠ URDF link frames; don't transplant without conversion.

---

## 2. Real-side UR10e port (diffusion_policy fork)

```bash
cd ~/work/repos && git clone -b ur10e-linear-gripper https://github.com/syedjameel/diffusion_policy.git
```

Changes (commit `3ffb2e6` on the fork's `ur10e-linear-gripper` branch, on top of upstream `omnireset`):

1. **NEW `diffusion_policy/real_world/ur10e_kinematics.py`** — data-swapped copy of
   `ur5e_kinematics.py`:
   - `CALIBRATED_JOINTS` = UR10e **nominal** transforms (== `local/Robots/UR10e/metadata.yaml`).
     Validated offline: FK matches the independent metadata reference to **0.0000 mm /
     0.00000°** over 50 random poses; Jacobian finite-difference-consistent (2e-7).
   - `LINK_INERTIAS` = UR10e URDF values.
   - `PAYLOAD_MASS = 0.575` kg, `PAYLOAD_COG = [0, 0, 0.050]` — **real measured values**
     (also configured on the pendant).
2. **`scripts/sim2real/collect_sysid_data.py`**: added `--robot {ur5e,ur10e}` via a
   `ROBOT_SPECS` dict — selects the kinematics module and `torque_max`
   (UR10e: `[330, 330, 150, 56, 56, 56]`). UR5e path unchanged.

Minimal env for sysid collection (the full `conda_environment_real.yaml` is only needed
later for cameras/deployment; its solve is very slow):

```bash
conda create -n ur_sysid python=3.10 -y && conda activate ur_sysid
pip install numpy click pynput ur-rtde==1.6.2
pip install torch --index-url https://download.pytorch.org/whl/cpu
# run scripts with PYTHONPATH=$PWD from the diffusion_policy root (no pip install -e .)
```

Robot-side setup used (from `diffusion_policy/README_ur5e.md` Part 1): pendant → Manual →
Settings → System: **Remote Control** + Constrained Freedrive ON; Network static
(robot `192.168.0.100`, PC same subnet); **External Control URCap** installed, Host IP = PC,
port 30004. The Robotiq URCap step is SKIPPED — our gripper is custom (it is not actuated
during sysid; it just has to be **mounted** so the wrist payload is real).

---

## 3. Chirp data collection — DONE

Pre-position the arm near `0,-90,90,-90,-90,0` (freedrive) so the script's initial `moveJ`
is a small correction, clear a ~30 cm bubble around the EE home
(`[0.69, 0.17, 0.68]` m in base frame), e-stop in hand. Motion envelope: ±10 cm x/y,
±15 cm z, ±14–29° wrist, ramped in/out; the same OSC as sim runs on the robot via
`ur_rtde.directTorque` at 500 Hz. `q` aborts safely.

```bash
cd ~/work/repos/diffusion_policy && conda activate ur_sysid
PYTHONPATH=$PWD python scripts/sim2real/collect_sysid_data.py --robot ur10e \
  --robot_ip 192.168.0.100 --output ~/sysid_data_ur10e_real.pt \
  --duration 8 --f0 0.1 --f1 3.0
```

Result (2026-07-03): 4000 steps @ 500 Hz saved. Validation: all six joints excited
(6–30° excursions), zero NaNs, embedded OSC params match the sim sysid action exactly
(Kp 1000/50, damping ratio 1, UR10e torque_max). Tracking error peaked ~275 mm / 242 N·m
around 1.55 Hz — large lag is expected and IS the excitation, not a fault.

File format (consumed by the sim-side fit): `joint_positions (T,6)`, `joint_torques`,
`tcp_forces`, `initial_joint_pos`, `dt`, `osc_params`, `waypoint_step_indices/_target_pos/_target_quat`.

---

## 4. CMA-ES sysid fit — DONE (round 3 accepted)

Fits 25 params (armature×6, static_friction×6, dynamic_ratio×6, viscous_friction×6,
motor_delay×1) by closed-loop replay of the chirp in the UR10e Sysid env
(`OmniReset-UR10eLinearGripper-Sysid-v0`, same OSC as RL, 500 Hz, DelayedPD arm).

```bash
# A100 (env_uwlab); needs commit c2747a1+ for --robot
scp ~/sysid_data_ur10e_real.pt haka01:~/
./uwlab.sh -p scripts_v2/tools/sim2real/sysid_ur5e_osc.py --headless --robot ur10e \
  --num_envs 512 --real_data ~/sysid_data_ur10e_real.pt --max_iter 200
#   -> logs/sysid/<timestamp>/checkpoint_XXXX.pt every 5 iters + final_results.pt
```

Wiring was smoke-tested on the laptop first (8 envs / 2 iters → clean `final_results.pt`).
NOTE: this script is plain argparse — hydra-style `env.*` overrides are NOT accepted (and
the Sysid env doesn't set the giant PhysX buffers, so no trims are needed anywhere).

**Fit history (what it took to converge):**

1. **Round 1** (default bounds): pan/lift/elbow SLAMMED the UR5e-sized ceilings (armature 10,
   friction 20, viscous 20) — sim lift kept ringing after the real joint damped; lift 2.64°.
2. **Round 2** (`--armature_max 40 --friction_max 60 --viscous_friction_max 80 --delay_max 8`):
   all ≤ 2° but lift exactly 2.00° and wrist_1 armature pinned at the LOWER bound (~0) — the
   phantom sim gripper mass (URDF 1.1 kg vs real 0.575 kg).
3. **Round 3** (same bounds, after the graft's `--gripper-mass 0.575` fix): accepted —
   pan 0.32 / lift 1.93 / elbow 1.31 / w1 1.11 / w2 0.34 / w3 0.27°, no bound saturation,
   params stable vs round 2. Run: `logs/sysid/20260705_120940`.

**Delay correction (2026-07-06 audit, commit 510248a):** the round-1..3 "identified delay"
was never actually simulated — both `sysid_ur5e_osc.py` and `plot_sysid_fit.py` reset the
env AFTER applying the delay, and `Articulation.reset()` re-randomizes the DelayedPD
buffers. Every CMA-ES candidate was scored at a *random* delay (the reported `delay=4` is
optimizer drift around the initial mean), and every verification plot replayed at a random
delay in {0..5} — that was the known "±0.2° run-to-run variance". The 24 armature/friction
params are unaffected (they persist through reset and validated <2°/joint at any drawn
delay). Both scripts now reset-then-apply, and `plot_sysid_fit.py --delay N` sweeps the
delay over a frozen fit. **Measured**: RMSE rises monotonically with delay — total 1.019°
at delay 0 vs 1.485° at delay 8 (w1 is the sensitive joint: 0.89° → 2.37°). The residual
delay paired with the round-3 params is **0 steps @ 500 Hz (< 2 ms)**; the true fit quality
is pan 0.30 / lift 1.87 / elbow 1.29 / w1 0.89 / w2 0.34 / w3 0.26° (total 1.02°), better
than the accepted numbers, which were measured at an accidental delay≈2. Config outcome:
Finetune DR keeps the paper's delay {0,1,2} @ 120 Hz; Finetune-Play pins delay 0.

Lesson: the printed `RMSE: X°` (= sqrt of the pooled CMA-ES score) is NOT a per-joint RMSE and
can exceed all of them — judge fits by the per-joint titles in `sysid_fit_error.png`. Also
check every parameter against BOTH bound ends; saturation = wrong bounds or wrong model, not a
bad optimizer. And when a replay's variance gets attributed to "random draw" noise, check the
draw isn't replacing the very parameter you think you set.

---

## 5. DONE — verify the fit (laptop, after copying checkpoints back)

```bash
# copy from A100:  rsync -av haka01:UWLab/logs/sysid/ logs/sysid/
conda activate leisaac
./uwlab.sh -p scripts_v2/tools/sim2real/plot_sysid_fit.py --headless --robot ur10e \
  --checkpoint logs/sysid/<timestamp>/checkpoint_0200.pt \
  --real_data ~/sysid_data_ur10e_real.pt
```

**Accept: < 2° RMSE per joint** (paper reference: ~7° without sysid, ~1° total with).
If a wrist joint fits poorly, suspect the sim gripper-mass mismatch (§7) — the real gripper
is 0.575 kg but the sim carries ~1.1 kg — and redo the fit after aligning the mass.

---

## 6. DONE — integrate the identified params

Paste the best params (from `final_results.pt`: `best_armature`, `best_friction` =
static_friction, `best_dynamic_ratio`, `best_viscous_friction`) into the `sysid:` block of
`source/uwlab_assets/uwlab_assets/local/Robots/Ur10eLinearGripper/metadata.yaml`,
**replacing the UR5e placeholder** (and update its warning comment). `best_delay` (physics
steps @500 Hz) documents the motor delay — the Finetune DelayedPD uses a delay range; note
the identified value in the metadata comment.

Consumers: `randomize_arm_from_sysid(_fixed)` events in the Finetune/Finetune-Play tasks
only — Stage-1 train/eval never read sysid. Commit to the UWLab fork per the usual
conventions.

---

## 7. DONE — sim hardening before finetune

1. **Align the sim gripper mass to reality.** URDF/USD gripper totals ~1.1 kg vs the real
   0.575 kg (~2x). Gravity is off in sim, but wrist INERTIA shapes the dynamics the policy
   feels and the fit in §4 absorbs the error into armature. Scale the grafted gripper link
   masses (graft script step or spawn `mass_props`) to total 0.575 kg, then re-verify the
   1500 N/m dual-drive jaw stiffness still tracks (`test_fullrobot_mimic.py --dual-drive
   --arm-wiggle`) and ideally redo §4-5 for a cleaner fit.
2. **Wrist joint limits ±360° → ±180° in sim** (paper A.3.1; NOT in the released assets —
   verified). Prevents the policy exploiting extreme wrist rotations that trigger real
   safety stops. **DONE**: the graft now sets ±180 by default (`--wrist-limit`, 0 keeps the
   URDF's ±360); build+step smoke passed. Measured impact on the existing A100 datasets:
   13.6% (Reaching), 26.6% (Grasped), 23.6% (Near Object), 15.6% (Near Goal) of states have
   |wrist| > 180° → **the four reset datasets MUST be re-recorded** after regenerating the
   USD on the A100 (loading a violating state clamps joints mid-teleport). The sysid fit is
   NOT affected (chirp wrists stay within ±110°, limits don't change dynamics away from
   limits).

Both are in the graft now ⇒ regenerate the USD on the A100, re-record the four reset
datasets once (§Pipeline README Step C), then finetune.

---

## 7b. Dataset QC & salvage tools (added 2026-07-06)

Two CPU-only tools (torch+numpy+yaml, no Isaac — run them on the A100 next to the
recorder) gate every reset dataset before it feeds a training run:

### `qc_reset_states_ur10e.py` — the gate

```bash
python scripts_v2/tools/conversions/qc_reset_states_ur10e.py --dataset_dir ./Datasets_ur10e/OmniReset
# per reset type, one line + FAIL details; end verdict [QC_RESULT] [PASS]/[FAIL]
```

What each column means and what is / is not a failure:

| check | meaning | gate |
|---|---|---|
| `wrist beyond180` | states with any \|wrist\| > 180.1° — **impossible** to reach dynamically on the ±180 USD; they exist because the reset events WRITE IK joint positions directly and nothing re-checks limits. Loading one clamps the wrist mid-teleport (wrong EE pose). | FAIL if > 0 → filter them out |
| `at180` | states with a wrist exactly AT ±180 (float32 π reads a hair above float64 π — not a violation). These are the old "long way" IK solutions saturating at the new boundary. The gripper is 180°-symmetric, so a wrist parked at ±180 is grip-equivalent; states load fine. On the re-recorded Grasped set this is ~99.8% of states — expected, benign. | reported only |
| `topdown≤45/30°` | gripper +Z tilt from straight-down (FK on the recorded joints) | ≥85% @45° for Anywhere/Resting grasped types |
| `fingertip<0` | fingertip point below the support surface (inherited EEAnywhere sampler artifact; for Resting grips it's the tip-point approximation near the tabletop) | reported only |
| `grip q` (min/median/max of `finger_joint`) | **grip semantics are per-type**: AnywhereEEGrasped holds the pcb mid-air at the canonical width (~0.0487); RestingEEGrasped mostly grips the on-table pcb across its ~2 mm THICKNESS (→ ~0.067–0.068 — do NOT read that as closed-on-air); PartiallyAssembledEEGrasped mixes width and exposed-edge thickness grips. `0.0000` = the OPEN default = the grasp event never engaged. | Anywhere: median ≈ 0.0487; Near Goal: median ≥ 0.03 |
| `jaw asym` | \|finger − right_finger\| (dual-drive symmetry) | p99 ≤ 1.5 mm (Anywhere type) |
| `open-jaw states` (Near Goal only) | fraction with `finger_joint < 0.02` — see the salvage note below | FAIL if median grip < 0.03 |

### `filter_reset_states.py` — the salvage (no re-recording)

```bash
# drop beyond-limit states (states recorded within ±180 load identically on the new USD)
python scripts_v2/tools/conversions/filter_reset_states.py --in-place \
  --input .../resets_ObjectAnywhereEEAnywhere.pt --drop-wrist-beyond
# drop never-engaged open-jaw "grasped" states
python scripts_v2/tools/conversions/filter_reset_states.py --in-place \
  --input .../resets_ObjectPartiallyAssembledEEGrasped.pt --min-grip 0.03
```

`--in-place` keeps a `.bak` next to the file; without it a `.filtered.pt` is written.
Re-run the QC afterwards — it must PASS before training.

### What the 2026-07-06 re-record QC actually found (for the record)

* **Reaching**: 12/10611 states beyond ±180 (worst 207°) — the direct-write corner case
  above, filtered out.
* **Grasped / Near Object**: clean; ~99.8% / ~20% at-limit (benign saturation).
* **Near Goal**: **66% open-jaw** — `check_reset_state_success` has NO jaws-on-object
  condition, so whenever the in-box grasp IK fails (much more often under the ±180 limits,
  which removed the long-way wrist solutions those grasps used), the event leaves the
  gripper open and the stable hover is accepted anyway. Those states are effectively
  `ObjectPartiallyAssembledEE**Anywhere**` — a type deliberately NOT in the training mix.
  Filtered 2500 → ~850 genuine grips and launched with that. This is also WHY Near Goal
  records so slowly: real in-box grips are the hardest states to stabilize; hovers pad the
  accept count. **Known gap / future fix**: add a jaws-on-object success condition to the
  recorder so C4 recording time only buys real grips, then re-record a full-size set.

---

## 8. RUNNING — Stage-2 finetune (ADR) + eval validation

The full A100 sequence (after `git pull fork omnireset/ur10e-linear-gripper` — needs
**b861f06 or later**: the 2026-07-06 audit fixed a DelayedPD buffer sizing that would
crash the finetune hours in when the ADR curriculum reaches delay 2, made the gripper-gain
DR cover BOTH dual-drive jaws, and added the dataset QC tool):

```bash
OBJ="env.scene.insertive_object=pcb env.scene.receptive_object=openbox"

# 8a. regenerate the USD (picks up gripper mass 0.575 + wrist +/-180). The graft's INPUTS
#     (UR10e/ur10e.usd, LinearGripper/linear_gripper.usd) are unchanged -- do NOT re-run the
#     Part-1 conversion steps unless this is a fresh checkout.
python scripts_v2/tools/conversions/graft_gripper_on_ur10e.py
#     expect BOTH: "gripper mass: 1.100 -> 0.575" and "wrist joint limits -> +/-180 deg"

# 8b. re-record the four reset datasets (Pipeline README Step C, C1->C4 in order) --
#     REQUIRED: 14-27% of the old states violate the new wrist limits.

# 8b'. QC the fresh datasets (CPU-only, runs anywhere; expects wrist>180 == 0 everywhere):
python scripts_v2/tools/conversions/qc_reset_states_ur10e.py \
  --dataset_dir ./Datasets_ur10e/OmniReset
#     expect: [QC_RESULT] [PASS]. On the OLD datasets this correctly fails with the
#     13.6/26.6/23.6/15.6% wrist-violation counts.

# 8c. (optional, paper-recommended) train 2 more Stage-1 seeds, then pick by noise robustness:
./uwlab.sh -p scripts_v2/tools/sim2real/eval_robustness.py --headless \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-Play-v0 \
  --checkpoints <ckpt_seed1.pt> <ckpt_seed2.pt> <ckpt_seed3.pt> \
  --action_noise 2.0 --eval_steps 1000 --num_envs 4096 $OBJ

# 8d. finetune (A100, 1 GPU; peg-class task per the paper ~= 8 h):
./uwlab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-Finetune-v0 \
  --num_envs 4096 --headless --logger tensorboard $OBJ \
  --resume_path logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/2026-07-02_21-39-37/model_1100.pt \
  env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e/OmniReset

# 8e. play the finetuned policy (laptop, GUI; note the Finetune-Play task id):
./uwlab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-Finetune-Play-v0 \
  --num_envs 4 $OBJ $TRIMS \
  env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e/OmniReset \
  --load_run <finetune run folder> --checkpoint model_<iter>.pt
```

**Resume caveat:** the Stage-1 checkpoint was trained on the OLD model (1.1 kg gripper,
±360° wrists) and old resets. The curriculum's job is exactly to adapt to shifted dynamics,
but watch the tensorboard success curve in the first hour — it should recover to Stage-1
levels before the friction ramp starts. If it stays low, the fallback is a fresh Stage-1 run
on the new USD/datasets (Pipeline README Step D) and finetuning from that instead.
(Observed 2026-07-06: recovery to >92% on all four tasks within 4 iterations.)

### 8.1 What `scale_progress` is — and when the finetune is DONE

Stage 1 trained on an *idealized* robot: zero joint friction, zero armature, zero motor
delay, soft OSC gains, large action steps. The real UR10e is none of those things. The
finetune's single job is to walk the policy from that ideal robot to the measured one
**without ever breaking it** — and `scale_progress` (call it `p`) is the position on that
walk, from 0 (ideal) to 1 (the identified robot).

**One knob, four channels.** Two curriculum terms share the same controller and climb
together (tensorboard: `Curriculum/adr_sysid/scale_progress`,
`Curriculum/action_scale/scale_progress`, each with its `mean_success_rate`):

| channel | at p = 0 | at p = 1 (per env, re-drawn every reset) |
|---|---|---|
| joint friction + armature | 0 (ideal) | the §6 sysid values × U(0.8, 1.2) per joint |
| motor delay | 0 | drawn from {0, 1, 2} physics steps @ 120 Hz (ceiling = round(p·2)) |
| OSC gains | train Kp 200/3 | eval Kp 1000/50 × U(0.8, 1.2), damping_ratio → 1 |
| action scale | (0.02…, 0.2) | (0.01, 0.01, **0.002**, 0.02, 0.02, 0.2) — z cut 10× so contact = pressing gently |

Intuition for the coupling: a sticky, delayed robot needs FIRM control (soft gains stall in
the friction dead zones), and firm control needs SMALL commanded steps (or contact turns
into ramming). So dynamics hardness, controller stiffness, and step size must rise
*together* — that is why one scalar drives all of it.

**The controller is bang-bang on success rate**, updated every 200 env steps (≈ every
handful of training iterations, independent of num_envs):

* mean success > 0.95 → `p += 0.01`
* mean success < 0.90 → `p -= 0.01`
* in between → hold
* warmup latch: `p` stays 0 until success first reaches 0.95 (the resume recovery), then
  the latch never re-engages.

Like adding weight to the bar only after a clean lift. Consequences worth knowing:

* **Success hovering in the 0.90–0.95 band during the ramp is the mechanism working**, not
  a regression — the controller deliberately surfs that band. Only `p` ratcheting
  down repeatedly / success pinned below 0.90 means the run hit a wall.
* Timeline: 100 increments × ~200 gated env steps ⇒ **≥ ~800 training iterations if
  success never dips; realistically several hours**. Expect dips and partial retreats.
* Milestone at **p ≈ 0.75**: the delay ceiling first reaches 2 — the exact point the
  pre-audit code crashed (`ValueError: max time lag > history length`, fixed 74910d0).
  Sailing past it confirms the fix in vivo.

**"Done" = `p` pinned at 1.0 with success holding ~0.95.** At that point every episode
runs at the full measured dynamics, stiff eval gains, and eval action scale — i.e. the
exact distribution the `Finetune-Play` eval task freezes (`randomize_*_fixed` at p = 1,
delay pinned to the measured 0). Practical checkpoint rule: let it *sit* at p = 1 for a
few hundred more iterations so most recent gradient steps come from the terminal
distribution, then take the latest checkpoint. If `p` plateaus below 1.0 oscillating,
note WHERE — the value tells you which dynamics level breaks the policy — and consider
longer training before reaching for config changes.

The finetune curriculum ramps dynamics toward the §6 sysid values, raises OSC gains, and
shrinks the action scale (paper A.3.6/A.3.9). **The Finetune-Play stiff eval gains (rot
Kp 50) were validated IN CONTACT on 2026-07-06** (P6-pattern probe, jaws closed on the pcb,
10 reset draws with the full fixed-sysid randomization: rot Kp drawn 40–57, delay pinned 0):
worst wrist |dq| 0.0017 rad/s, worst held-pcb angular velocity 0.35 rad/s — the P6 failure
signature was ~3.1/3.5 rad/s. The identified joint friction is what damps the massless PD
here; P6's instability was on the frictionless Stage-1 setup. Note this validates the
curriculum ENDPOINT — intermediate (partial-gain, partial-friction) points are guarded by
the ADR back-off, and the endpoint is what deploys. Probe:
`scripts_v2/tools/conversions/probe_eval_gains_in_contact.py` (untracked, re-creatable).

Real-robot teleop sanity check of the OSC (after adapting for the custom gripper):
`diffusion_policy demo_real_robot.py -o <dir> --robot_ip 192.168.0.100`
(`--osc_kp_pos/--osc_kp_rot` if the arm stalls; needs the full robodiff_real env + Mello).

---

## 9. NEXT — real deployment path (PLANNED 2026-07-06, start here tomorrow)

The state-based expert **cannot run on the real robot** (it observes object poses). The
OmniReset deployment path is student-teacher distillation to RGB, zero-shot.

**Decisions made (2026-07-06):**
- **Deployment machine: the RTX 4090 PC** — cameras (3× D405 on USB3), gripper serial,
  ResNet inference, AND the 500 Hz RTDE loop all on that one box. This is the clean path
  to the <40 ms end-to-end budget (a LAN hop between camera PC and control PC would eat
  5–15 ms + jitter). To install there: `diffusion_policy` fork (ur10e-linear-gripper
  branch, `robodiff_real` env) + robot subnet access (192.168.0.x).
- **Cameras: 3× RealSense D405.** Wrist = ideal (D405 sweet spot is 7–50 cm). Front/side
  at ~0.8–1.1 m are OUTSIDE the design range — RGB-only is what we use, so acceptable, but
  eyeball sharpness during calibration before committing.
- **RGB sim scene strategy: do NOT model our room.** The authors' RGB cfg is an abstract
  STAGE — three flat "curtain" planes (left/back/right) around the table — with per-episode
  texture/color randomization of every visible surface (curtains, table, objects, fingers,
  wrist mount) + camera pose/focal jitter. We keep that stage in sim and make the REAL
  workspace roughly match its geometry (backdrop panels around the table); randomization
  covers appearance. The real lab becomes "just another sample".

**Real hardware config (corrected 2026-07-08/09)** — UR10e `192.168.0.100`, RTDE 500 Hz;
gripper serial `/dev/ttyACM0` @ 115200; 3× RealSense **D405** serials (TRUE roles, physically
verified — the lerobot json labels were swapped): **front `409122273078`, side `323622272232`,
wrist `409122272284`** (640×480@30). Images are **resized, not cropped** (640×480 → 224×224;
crop boxes were a lerobot-ACT thing, not diffusion_policy). **Payload = 0.575 kg confirmed**
(weighed; the lerobot `0.3` was wrong) — `ur10e_kinematics.PAYLOAD_MASS` drives the real OSC's
`setPayload` gravity comp. Real home pose (pendant deg): `67.94 -93.33 146.23 -142.91 -90.04
-22.95` (= sim `-22.06 ...` per §10.2a).

**diffusion_policy real-side — ✅ code done (fork commits b0b0808 + 42a6e15 + 0961090;
untested on hardware until deployment)**:
- arm swapped `ur5e_kinematics → ur10e_kinematics` in `rtde_interpolation_controller.py` /
  `real_env.py` / `eval_real_robot.py` (identical API, drop-in); `torque_max` → UR10e
  `330/330/150/56/56/56`; real home pose = `real_env` default init joints.
- new `real_world/linear_gripper.py` (hardened serial Open/Close on `/dev/ttyACM0`:
  transition-only writes, serial exceptions swallowed so a USB hiccup can't drop the 500 Hz
  torque loop, no activate/encoder) replaces `RobotiqGripper` in the controller; plumbed
  through `real_env`. Real jaw travel measured **1.1–1.2 s** (see §10 gaps).
- D405 serials set in `eval_real_robot`/`demo_real_robot` + `camera_configs=None` (D405
  rejects the 415/435/455 advanced-mode presets; verified `None` flows through cleanly).
- **90° rig-orientation fix (§10.2a, commit 0961090)**: `real_to_sim_joints` (`q1 − 90°`)
  applied at the RTDE read boundary (init read, OSC-loop read, ring-buffer `ActualQ`) so
  FK / OSC / policy obs are all sim-frame; `moveJ` init + shutdown `servoJ` stay raw
  real-frame; `ActualQd` offset-free; torques per-joint invariant. `eval_real_robot`
  recomputes `end_effector_pose` from `arm_joint_pos` via FK, so ALL policy obs flow through
  the shifted joints.

Still TODO (hardware): physical camera calibration of the final mounts (§10.3; the chain
itself was dry-run-validated 2026-07-09); first real teleop/OSC sanity run. Student training
is **`dataset_dir`-only** (`config/task/sim2real_image.yaml`).

**Work items, in order** (A can start immediately; the finetune does not block A–E):

1. **A — Linear-gripper driver** (small). Source: `RC10_control/rc10_api/gripper.py`
   (serial `Open\n`/`Close\n`, sign convention `state<=0 -> close` — identical to
   `real_env.py`'s `action[6]<0 -> close`; drop-in match). Plan: new
   `diffusion_policy/real_world/linear_gripper.py` hardened for the 500 Hz process
   (transition-only writes, serial-exception safety so a USB hiccup can't kill the torque
   loop, commanded-Open on activation since there is no encoder, `--gripper_device` arg);
   `rtde_interpolation_controller.py` gets a `gripper` selector (`robotiq`|`linear`|`none`)
   replacing `RobotiqGripper.activate()/move()` on the linear path; plumb through
   `real_env.py` + teleop/eval scripts. NO gripper feedback needed: the stack observes
   `last_gripper_action` (commanded), never encoder values — true for the 2F-85 too.
   Validation: measure real open↔close travel time vs sim (~0.1–0.2 s); if much slower,
   tune the sim jaw velocity limit to match before the NEXT training round.
2. **B — Wrist camera mount** (hardware). D405 on the linear gripper. Requirements: rigid;
   fingers + grip zone in view at 7–30 cm; cable strain relief for wrist rotation; rough
   viewpoint like the sim wrist cam (mounted on `robotiq_base_link`, offset ~(0.018,
   −0.004, −0.069), looking at the grip zone) — exact placement NOT critical (calibration +
   pose randomization absorb it). Sim side afterwards: add a simple proxy box for the mount
   to the graft (front/side cameras see it; its texture gets randomized like the authors').
3. **C — Camera rig + calibration** (front/side can start before the mount exists). Mount
   rigidly ~where the sim cfg puts them (front ~1.1 m out, side ~0.8 m lateral). Per
   camera: ArUco coarse extrinsic (6x6_50 ID 12, 150 mm; `0/1/2_camera_*.py`) →
   `align_cameras.py` interactive overlay refine (press `p` → pos/rot/focal). Precision
   target is only ~cm/degree: the per-episode camera randomization (±2–3 cm pose, focal
   ranges) absorbs the residual. Deliverable: three (pos, rot, focal) tuples.
4. **D — UR10e RGB configs — ✅ BUILT (sim side, 2026-07-08)**. New
   `config/ur5e_robotiq_2f85/ur10e_linear_gripper_rgb_cfg.py` holds three UR10e cfgs
   (subclass-and-swap via `_apply_linear_gripper`): a **CameraAlign** env (for §C
   calibration), and **RGB DataCollection** + **RGB Play** envs. Plus a
   `UR10eLinearGripper_DAggerRunnerCfg` (agents), 3 gym registrations, and
   `align_cameras.py --robot {ur5e,ur10e}`. Task ids:
   `OmniReset-UR10eLinearGripper-CameraAlign-v0`,
   `-RelCartesianOSC-RGB-DataCollection-v0`, `-RGB-Play-v0`.
   Details baked in: IMPLICIT UR10e robot + eval OSC action; delay pinned 0; resets from
   `./Datasets_ur10e/OmniReset`; the two 2F-85 gripper-appearance DR terms dropped (their
   meshes are absent on our instanced gripper visuals); **wrist camera re-pathed to
   `/Robot/gripper/robotiq_base_link/rgb_wrist_camera`** (our graft nests the gripper — the
   original `/Robot/robotiq_base_link` path errored). Smoke-tested on the laptop
   (`smoke_test_rgb_ur10e.py`): both envs build, all 3 cameras render, obs shapes exact —
   `policy` group `(3,224,224)` float + `data_collection` group `(224,224,3)` uint8, matching
   the diffusion_policy `shape_meta`. Camera pos/rot/focal are still the authors'
   **placeholders** (front frames the UR10e low; wrist renders black) — replace with §C
   calibrated values before the 80k. Collection command (A100/4090, after export + calib):
   `collect_demos.py --task ...-RGB-DataCollection-v0 --dataset_file <x>.zarr --num_envs 32
   --num_demos 80000 --enable_cameras --headless $OBJ
   agent...behavior_cloning_cfg.experts_path=[<run>/exported/policy.pt]` (zarr files merge
   across runs). **Cross-repo (diffusion_policy) work is documented in the plan
   `~/.claude/plans/vivid-giggling-tower.md` Part 2** (arm import-swap, linear-gripper serial
   driver, D405 camera stack, student training = dataset_dir only).
5. **E — Physical stage prep** (paper A.4): backdrop panels ~where the sim curtains sit;
   **command-strip the openbox to the table** (sim treats it as static); compliant mat;
   consistent lighting; verify real table/mount geometry vs the sim scene (work surface
   ~level with z=0, robot base on its plate ~1.3 cm above).
6. **F — Distillation** (A100, AFTER the finetune converges): optionally 1–2 more finetune
   seeds + `eval_robustness.py` selection (noise robustness predicts real transfer); then
   80k expert episodes under the RGB randomization (~24 GPU-h), ResNet-18 + MLP student,
   5-frame stack @ 10 Hz, KL-matching + pose-reconstruction aux loss, ~350k iters
   (~2 days). Expect student sim success ≈ 50–60% of the expert — normal; real transfer is
   better than that number suggests (paper: peg 85% real).
7. **G — Real eval extras** (paper A.4): stuck-detection auto-recovery (no joint motion
   >2 s → open gripper 1 s, not counted as failure).

---

## 10. RGB distillation & deployment — step-by-step (command-by-command)

Follows the official OmniReset **sim2real** (camera calibration) and **distillation**
(export → collect → train → eval → deploy) docs, adapted for **UR10e + linear gripper +
3× D405**. Both the sim configs (UWLab, §9 item D) and the real stack (diffusion_policy, §9
"diffusion_policy real-side") are already built. `$OBJ = env.scene.insertive_object=pcb
env.scene.receptive_object=openbox` throughout.

**The whole pipeline in one line:** the trained expert lives in sim and secretly reads object
poses, so it can never run on the real robot — instead: *measure where your real cameras are
(10.3 `0/1/2`), make the sim cameras match by eye (`align_cameras`), film the all-knowing
expert through those matched cameras 80k times (10.4), distill it into a student that needs
only pixels (10.5), rehearse in sim (10.6), perform live (10.7).* Each step below carries a
"what it does" line so you know what the command is doing and what output to expect.

**Three conda envs (per the docs):**
- **SIM** — `env_uwlab` on the A100/4090 (`leisaac` on the laptop). Runs the UWLab sim
  scripts: export, `collect_demos`, `align_cameras`, `eval_distilled_policy`.
- **ROBODIFF** — the training env (diffusion_policy `conda_environment.yaml`).
- **ROBODIFF_REAL** — the real-robot env (diffusion_policy `conda_environment_real.yaml`).
  Runs the calibration capture (`0/1/2_camera_*.py`) and `eval_real_robot`.

### 10.0 — One-time setup (distillation doc Step 1)
```bash
# both repos as siblings; pull the forks
cd ~/work/repos/UWLab            && git pull fork omnireset/ur10e-linear-gripper
cd ~/work/repos/diffusion_policy && git pull fork ur10e-linear-gripper
# install diffusion_policy into the SIM env (collect_demos zarr writing + shared utils)
cd ~/work/repos/diffusion_policy && conda activate env_uwlab && python -m pip install -e . \
  && python -m pip install dill hydra-core omegaconf zarr einops "diffusers<0.37" wandb accelerate
# create the training env (once)
mamba env create -f conda_environment.yaml        # -> robodiff (training)
```

**`robodiff_real` — DON'T use the full `conda_environment_real.yaml` here.** That yaml is a
heavy, old-pinned env (Py3.9 / CUDA 11.6 / PyTorch 1.12 / pytorch3d / MuJoCo / robosuite / r3m /
dm-control) whose solve hangs (no mamba on the laptop, only conda), and it needs `sudo apt`
system libs (librealsense, OSMesa, spnav). ~Half of it is sim-benchmark reproduction we never
run on the real rig. Build a **lean env** instead, in two tiers:

```bash
# tier 1 -- CALIBRATION capture only (0/1/2_camera_*.py + perception/): reliable, ~2 min
conda create -n robodiff_real python=3.9 -y
conda run -n robodiff_real python -m pip install \
  "numpy<2" scipy matplotlib "opencv-contrib-python-headless<4.10" pyrealsense2 open3d
# NOTE: numpy<2 is REQUIRED -- the opencv-contrib 4.9 wheel is compiled against numpy 1.x and
# crashes under numpy 2.0 ("_ARRAY_API not found"). pip pulls numpy 2 by default, so pin it.
# (pyrealsense2 wheel bundles librealsense w/ D405 support; opencv-contrib gives cv2.aruco --
#  code uses the new ArucoDetector API + solvePnP, so 4.9 is fine; don't need <4.7.)
# Verified: numpy 1.26.4 / cv2 4.9.0 / pyrealsense2 2.56.5 / open3d 0.19.0; perception/ imports OK.

# tier 1.5 -- TELEOP (demo_real_robot + real_env + controller; verified on the laptop
# 2026-07-09, full import chain OK). CPU torch suffices (only the image resize uses it):
conda run -n robodiff_real python -m pip install \
  av click numba numcodecs pynput "ur-rtde==1.6.2" pyserial threadpoolctl zarr "atomics==1.0.2"
conda run -n robodiff_real python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# tier 2 -- add DEPLOYMENT (eval_real_robot) when you actually deploy, on the 4090 PC
# (CUDA torch there for ResNet inference):
conda run -n robodiff_real python -m pip install \
  torch torchvision dill hydra-core omegaconf einops \
  "diffusers<0.37" tqdm imageio imageio-ffmpeg scikit-video scikit-image termcolor robomimic
# run diffusion_policy scripts from the repo root (PYTHONPATH=$PWD) -- no `pip install -e .` needed.
```
Skipped vs the yaml (all sim-benchmark, unused on the real rig): free-mujoco-py, mujoco,
dm-control, robosuite, pybullet-svl, pytorchvideo, r3m, spnav, pytorch3d, ray, wandb.

Prereq: the §8 finetune is converged (p pinned 1.0, success ~0.95, a checkpoint chosen).

### 10.1 — Export the finetuned expert to TorchScript (distillation doc Step 2)
**What it does:** the finetune checkpoint (`model_<iter>.pt`) is a training-framework object
(needs rsl_rl classes + config to load). `play.py` loads it once, runs it (your visual
confirmation the checkpoint is good), and **JIT-traces** it — records the raw tensor ops into
a standalone `exported/policy.pt`: the network frozen as a pure `obs -> action` function,
loadable anywhere with `torch.jit.load`. That is the exact form `collect_demos` replays via
`experts_path`.
```bash
conda activate env_uwlab && cd ~/work/repos/UWLab
./uwlab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-Finetune-Play-v0 \
  --num_envs 4 --checkpoint <path/to/finetune/model_<iter>.pt> --headless $OBJ \
  env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e/OmniReset
# -> <checkpoint_dir>/exported/policy.pt   (+ policy.onnx)  -- path reused in 10.4
```

### ⚠ 10.2a — RIG ORIENTATION (read first — differs from the authors' rig by 90°)
**Verified against the sim reset datasets:** the sim robot spawns at the origin with identity
rotation and the trained sim workspace/table is toward **sim +X** (objects ~`[0.42, 0.1]`,
table `x=0.4`); FK of reset joints matches the sim only with `R_180Z`, so sim frame = REP-103.
Our real rig has the workspace/marker toward **pendant −Y** (`[0, -0.463]` on the pendant);
the authors' rig had it at sim +X (their `aruco_offset [0.24,0,0]` is **sim-frame**). Same
joints ⇒ our real EE maps to sim `[0, +0.463]` — **90° away from the sim table**. Two
conventions fix this everywhere (both implemented):

1. **Joint mapping (deployment):** `q1_sim = q1_real − 90°`
   (`ur10e_kinematics.real_to_sim_joints`, applied at the RTDE read boundary in
   `rtde_interpolation_controller` — obs/FK/OSC are sim-frame; `moveJ`/`servoJ` stay real).
   Verified: FK of the shifted real home lands at `[0.463, 0, 0.22]` — over the sim table.
   Real home `67.94 -93.33 146.23 -142.91 -90.04 -22.95` ⇒ **sim** home
   `-22.06 -93.33 146.23 -142.91 -90.04 -22.95`.
2. **Marker convention (calibration):** marker **+X points from the robot base toward the
   marker/workspace** (physically pendant −Y = sim +X); `aruco_offset = [0.463, 0, 0]`
   (sim frame). Then `0/1/2` output the camera pose **directly in the sim frame** — the
   authors' design; **no rotation conversion belongs in `2_get`** (verified; its docstring
   states the contract).

### 10.2 — Physical rig (hardware)
Mount the 3× D405 (front `409122273078`, side `323622272232`, wrist `409122272284`); build the
backdrop curtains (front ≈1.1 m out, side ≈0.8 m lateral); command-strip the openbox to the
table; drive the arm to home `67.94 -93.33 146.23 -142.91 -90.04 -22.95` deg. Print + place the
**ArUco marker** (dictionary `6x6_50`, ID `12`, size `150 mm` — the `marker_6x6_150mm_id12.pdf`
linked from the sim2real doc) flat on the table 0.463 m from the base toward the workspace,
**oriented per §10.2a** (+X away from the base, toward the marker). Confirm:
`rs-enumerate-devices | grep -A1 D405`.

### 10.3 — Camera calibration (sim2real doc) — ONE camera at a time (unplug the others)
For each of `front` / `side` / `wrist`:

**What each script does:**
- **`0_camera_calibrate.py`** — "where is the camera, in robot coordinates?" Photographs the
  ArUco marker; since the marker's printed size (150 mm) and the lens intrinsics (read live
  from the D405) are known, `solvePnP` computes where the camera must be standing (and how
  tilted) for the marker to appear that size/shape in the image — like judging your distance
  from a door because you know how big doors are. That gives camera-relative-to-MARKER;
  adding `aruco_offset` (marker-in-base) makes it camera-relative-to-BASE. 10 rounds,
  averaged. Expect: the labeled triads (BIG=base, MID=marker at +X 0.463, SMALL=camera,
  tilted like the real mount) in a true-scale point cloud + printed view-dir/tilt per camera.
- **`1_camera_get_rgb.py`** — takes THE reference photo (`real_rgb.png`): the frozen record of
  "exactly what this camera sees from here". Robot must be AT the known pose; don't touch the
  camera afterwards.
- **`2_get_isaacsim_extrinsics.py`** — pure math on the json: flips the OpenCV camera axes
  (z-forward) to Isaac/OpenGL (z-backward) and prints pos + quat(wxyz) — the ~cm-accurate
  WARM START for align_cameras, not the final answer.

**(a) capture + coarse extrinsics** — diffusion_policy, ROBODIFF_REAL:
```bash
conda activate robodiff_real && cd ~/work/repos/diffusion_policy
python scripts/sim2real/0_camera_calibrate.py        # ArUco -> intrinsics + extrinsics (sim frame)
python scripts/sim2real/1_camera_get_rgb.py          # -> perception/calibrations/real_rgb.png
cp scripts/sim2real/perception/calibrations/real_rgb.png ~/real_<cam>.png  # it's overwritten per camera!
python scripts/sim2real/2_get_isaacsim_extrinsics.py # prints sim-frame warm-start pos + quat(wxyz)
```
Sanity-check the `0_` output before moving on: the printed camera pos should sit in the **+X
quadrant** (front cam ≈ `[0.7, 0, 0.2]`), view direction toward −X and tilted down like the
physical mount; an all-NaN json = the marker was missed in some rounds (glare/blur) — rerun.
Record the arm's joint angles (deg) at the capture pose (pendant). Don't touch the camera
after the `1_` capture. **Wrist:** put the arm in freedrive and position it so the wrist
camera sees the marker.

**(b) interactive alignment** — UWLab, SIM, `--robot ur10e`:
**What it does:** the pixel-match. Builds the CameraAlign env, poses the sim UR10e at
`--joint_angles` (sim frame — the sim arm must strike the SAME pose as the real arm in the
photo), renders the virtual camera, and overlays that render on your real photo in a
matplotlib window. You nudge the virtual camera (pos/rot/focal) until the rendered robot lies
ON TOP of the photographed robot — matching the whole articulated silhouette is dozens of
constraints, far stronger than one flat marker, and it recovers the focal length ArUco can't.
ArUco got ~cm; your eyes get the last mm/degrees. Expect: overlay starts NEAR aligned (the
warm start); if it starts rotated ~90°/mirrored, a frame convention is wrong — stop.
```bash
conda activate env_uwlab && cd ~/work/repos/UWLab
./uwlab.sh -p scripts_v2/tools/sim2real/align_cameras.py --enable_cameras --headless \
  --robot ur10e --camera front_camera --real_image /path/to/real_front.png \
  --joint_angles <j1_pendant - 90> <j2> <j3> <j4> <j5> <j6>
# ⚠ SIM joints (§10.2a): subtract 90° from the PENDANT q1. Captured at home:
#   --joint_angles -22.06 -93.33 146.23 -142.91 -90.04 -22.95
# nudge the sim camera onto the real image; press 'p' to print calibrated pos, rot, focal
```

**After all three cameras:** paste the three `(pos, rot, focal)` into **`_UR10E_CAMERA_POSES`**
in `config/ur5e_robotiq_2f85/ur10e_linear_gripper_rgb_cfg.py`. The 2F-85 doc has you edit the
`TiledCameraCfg` entries **and** the `randomize_*_camera` `base_position`/`base_rotation` by
hand — our hook applies **both** (scene cameras + the DR event bases + recentered focal jitter)
from that one dict, for the CameraAlign, DataCollection, and Play envs at once. No rebuild.

✅ **D405 status (resolved 2026-07-09; the old `_fovy`/preset caveat was a false alarm):**
`_fovy = 65` is UNUSED metadata (`2_get` outputs only pos+quat; the focal comes from
`align_cameras`), and the calibration `perception/realsense.py` loads NO advanced-mode preset
(color+depth 640×480 — D405-native). The real D405 bug was viz-only: `depth_to_points`
hardcoded 1 mm depth units, but the D405 uses ~0.1 mm → the debug cloud rendered ~10× too big
(fixed: the device's `get_depth_scale()` is queried; commit `93d0b98`). The full `0/1/2` chain
was **validated end-to-end on the real front D405 on 2026-07-09** (dry run): output landed in
the sim frame at the expected +X quadrant, view 40° down, 6.7° off the cam→marker line.
The debug window now shows labeled triads — BIG = robot base @ origin, MID = marker
@ `[0.463,0,0]`, SMALL = camera (rotated to its calibrated pose) — and prints each camera's
view direction + tilt; use them as the per-camera sanity check.

### 10.4 — Collect the 80k RGB demos (distillation doc Step 3) — SIM, needs `--enable_cameras`
**What it does:** films the expert working. Builds the RGB DataCollection env — the
"stage-set" task: 3 calibrated cameras rendering, curtain/table/object textures re-randomized
every episode, camera poses jittered around YOUR calibrated values, lighting/object DR. The
STATE expert (secretly reading object poses — allowed in sim) drives; every step the env
records what the cameras see + robot state + the expert's action. Successful episodes append
to the zarr; failures are discarded. You are building "here's what the world looks like →
here's what the expert did", 80k times. Sanity: open a few frames — robot/objects framed like
your real photos, textures varying wildly.
```bash
conda activate env_uwlab && cd ~/work/repos/UWLab
./uwlab.sh -p scripts_v2/tools/collect_demos.py \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-RGB-DataCollection-v0 \
  --dataset_file datasets/ur10e_pcb/rgb0.zarr --num_envs 32 --num_demos 80000 \
  --enable_cameras --headless $OBJ \
  agent.algorithm.offline_algorithm_cfg.behavior_cloning_cfg.experts_path='["<ckpt_dir>/exported/policy.pt"]'
# only SUCCESSFUL episodes are saved; ~24 GPU-h for 80k (10k ~2 h for a sim-only smoke test).
# Zarr files MERGE across runs -- split collection by rerunning into rgb1.zarr, rgb2.zarr, ...
# in the same datasets/ur10e_pcb/ dir.
```

### 10.5 — Train the RGB student (distillation doc Step 4) — ROBODIFF
**What it does:** supervised imitation, no simulator — just the zarr. A ResNet-18 encodes the
3 camera views (5-frame stacks); an MLP head predicts the expert's action. The `_kl_` variant
also matches the expert's action DISTRIBUTION (mean/std saved at collection), and an
auxiliary loss makes the encoder predict object poses — forcing the vision backbone to learn
"where things are" instead of texture shortcuts. Because collection randomized every
appearance, the student can't overfit to any one look — the real lab becomes just another
texture draw. Output: the deployable student `.ckpt` (needs only images + robot state).
```bash
conda activate robodiff && cd ~/work/repos/diffusion_policy
python train.py --config-name train_mlp_sim2real_image_with_aux_loss_workspace.yaml \
  --config-dir diffusion_policy/config \
  task.dataset.dataset_dir=<abs path to ~/work/repos/UWLab/datasets/ur10e_pcb>
# dataset_dir = the folder holding rgb0.zarr / rgb1.zarr / ... -- the ONLY UR10e change
# (config is shape_meta-driven: front/side/wrist_rgb @224x224, 6-D EE/joint/last-action,
# 7-D action). ~350k iters (~2 days on an H200); reasonable by ~1 day. -> a .ckpt.
# KL-distill variant (matches the paper's KL-matching, uses the saved expert_action_mean/std):
#   --config-name train_mlp_sim2real_image_with_aux_loss_kl_workspace.yaml
```

### 10.6 — Evaluate the student in sim (distillation doc Step 5) — SIM
**What it does:** the exam before the real exam. Loads the STUDENT into the RGB Play env
(in-distribution resets, same cameras) and lets it drive on its own predictions — the first
time its small errors get to compound. Reports success rate + saves videos. ~50–60% of the
expert is NORMAL (paper: students score modestly in sim yet transfer better in reality —
real peg 85%). Near-zero = something structural (paths, obs mismatch) — debug, don't deploy.
```bash
conda activate env_uwlab && cd ~/work/repos/UWLab
./uwlab.sh -p scripts_v2/tools/eval_distilled_policy.py \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-RGB-Play-v0 \
  --checkpoint <student>.ckpt --num_envs 32 --num_trajectories 100 \
  --headless --enable_cameras --save_video $OBJ
# expect student sim success ~50-60% of the expert (paper); real transfer is better than that.
# (No UR10e -RGB-OOD-Play-v0 yet -- the OOD variant was deferred; add it like the 2F-85 if wanted.)
```

### 10.7 — Deploy on the real UR10e (distillation doc Step 6) — ROBODIFF_REAL, on the 4090 PC
**What it does:** the whole built stack comes alive. Three D405s stream → resize to 224²
(same 4:3→1:1 squish as sim); RTDE reads joints at 500 Hz → `real_to_sim_joints` shifts q1 so
the robot reports itself in the sim's language (§10.2a) → FK computes the EE-pose obs; the
student consumes images+state at 10 Hz and outputs the same actions it produced in sim; the
OSC turns them into joint torques (`directTorque`, `setPayload 0.575` gravity comp); jaw
commands go down `/dev/ttyACM0`. Between episodes it homes the arm; a stuck-detector opens
the gripper if the arm freezes >2 s. **First-run protocol:** hand on the e-stop; verify the
startup moveJ goes to YOUR home pose and the first policy motions head TOWARD the workspace —
any 90°-sideways tendency means a frame bug survived: kill it immediately.
```bash
# 1. cameras at the calibrated poses (10.3); 2. copy <student>.ckpt to the 4090 PC; 3.:
conda activate robodiff_real && cd ~/work/repos/diffusion_policy
python eval_real_robot.py --input <student>.ckpt --output ./demo --robot_ip 192.168.0.100 -j
# uses the built stack: ur10e_kinematics, LinearGripper on /dev/ttyACM0, D405 serials
# front/side/wrist, torque_max 330/330/150/56/56/56, setPayload 0.575 kg, home pose above.
```

### ⚠ Known sim↔real gaps to watch
- **Gripper actuation is ~6–10× slower on the real robot: 1.1–1.2 s measured to open/close vs
  the near-instant sim jaw** (~11–12 control steps at 10 Hz). The expert (and student) learned
  grasp timing on the fast sim jaw, so at deployment the arm may move on before the real grasp
  completes. Mitigations, cheapest first: the deploy-side stuck-detection/`'g'` open macro
  already helps; if grasps fail on the slow close, slow the **sim** jaw to ~1.1 s (lower the
  gripper `maxJointVelocity`/drive in the graft) and **re-finetune** before the next collection
  — only if deployment shows it matters (it costs a training round).
- **Payload 0.575 kg confirmed** (weighed; `ur10e_kinematics.PAYLOAD_MASS`), used by the real
  OSC `setPayload` gravity comp. (The lerobot `0.3` was wrong.)
- **Images are resized, not cropped** — `real_env` resizes 640×480 → 224×224, matching the sim
  (320×240 → 224×224); same 4:3→1:1 squish on both sides, so no crop boxes.

---

## Quick reference — file locations

| what | where |
|---|---|
| Real chirp data | `~/sysid_data_ur10e_real.pt` (laptop) + copy on A100 |
| Sysid fits | `logs/sysid/<timestamp>/` (A100 → rsync to laptop) |
| UR10e real-side kinematics | `diffusion_policy/real_world/ur10e_kinematics.py` on `syedjameel/diffusion_policy` branch `ur10e-linear-gripper` |
| Controller file dump | `~/urcontrol_from_ur10e/` (laptop) |
| Extracted (nominal) calibration | `~/ur10e_calibration.yaml` |
| Sysid target metadata | `source/uwlab_assets/.../local/Robots/Ur10eLinearGripper/metadata.yaml` (`sysid:` block = REAL UR10e values since 5cb15a7) |
| Sim pipeline manual | `UR10E_PIPELINE_README.md` |
| Paper / official doc | `2603.15789v3.pdf` / uw-lab.github.io → publications → omnireset → sim2real |
