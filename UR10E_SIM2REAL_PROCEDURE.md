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
| 8. Stage-2 finetune (ADR) + eval-gain validation in contact | todo (§8) |
| 9. Real deployment path (RGB distillation, cameras, gripper driver) | todo — big items (§9) |

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

## 4. CMA-ES sysid fit — RUNNING (A100)

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
   params stable vs round 2. Run: `logs/sysid/20260705_120940`. Delay: 4 steps @ 500 Hz (8 ms).

Lesson: the printed `RMSE: X°` (= sqrt of the pooled CMA-ES score) is NOT a per-joint RMSE and
can exceed all of them — judge fits by the per-joint titles in `sysid_fit_error.png`. Also
check every parameter against BOTH bound ends; saturation = wrong bounds or wrong model, not a
bad optimizer.

---

## 5. NEXT — verify the fit (laptop, after copying checkpoints back)

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

## 6. NEXT — integrate the identified params

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

## 7. NEXT — sim hardening before finetune (both optional but recommended)

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

## 8. NEXT — Stage-2 finetune (ADR) + eval validation

The full A100 sequence (after `git pull fork omnireset/ur10e-linear-gripper`):

```bash
OBJ="env.scene.insertive_object=pcb env.scene.receptive_object=openbox"

# 8a. regenerate the USD (picks up gripper mass 0.575 + wrist +/-180). The graft's INPUTS
#     (UR10e/ur10e.usd, LinearGripper/linear_gripper.usd) are unchanged -- do NOT re-run the
#     Part-1 conversion steps unless this is a fresh checkout.
python scripts_v2/tools/conversions/graft_gripper_on_ur10e.py
#     expect BOTH: "gripper mass: 1.100 -> 0.575" and "wrist joint limits -> +/-180 deg"

# 8b. re-record the four reset datasets (Pipeline README Step C, C1->C4 in order) --
#     REQUIRED: 14-27% of the old states violate the new wrist limits.

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

The finetune curriculum ramps dynamics toward the §6 sysid values, raises OSC gains, and
shrinks the action scale (paper A.3.6/A.3.9). **Before trusting the Finetune-Play task:
validate its stiff eval OSC gains (rot Kp 50) IN CONTACT** — jaws closed on the pcb, watch
wrist joint velocity. This is the P6 lesson: free-space probes pass while contact
limit-cycles (see the comment block in `config/ur5e_robotiq_2f85/actions.py`).

Real-robot teleop sanity check of the OSC (after adapting for the custom gripper):
`diffusion_policy demo_real_robot.py -o <dir> --robot_ip 192.168.0.100`
(`--osc_kp_pos/--osc_kp_rot` if the arm stalls; needs the full robodiff_real env + Mello).

---

## 9. NEXT — real deployment path (the big remaining items)

The state-based expert **cannot run on the real robot** (it observes object poses). The
OmniReset deployment path is student-teacher distillation to RGB, zero-shot:

1. **Custom gripper real-world driver.** The stack drives a 2F-85 via its URCap
   (`real_env.py` / `rtde_interpolation_controller.py`); our linear gripper needs its own
   open/close interface wired in. Engineering item, no upstream reference.
2. **Wrist camera mount.** The repo's `2f85_d415_mount.stl` is 2F-85-specific — design one
   for the linear gripper (paper: wrist D415 + front D435 + side D455, all USB 3.2;
   40 ms end-to-end latency budget, 60 ms degrades badly).
3. **Camera calibration** (per-camera): diffusion_policy `0_camera_calibrate.py`,
   `1_camera_get_rgb.py`, `2_get_isaacsim_extrinsics.py` (ArUco 6x6_50 ID 12, 150 mm,
   marker offset to base default [0.24, 0, 0]) → UWLab
   `scripts_v2/tools/sim2real/align_cameras.py --camera front_camera --real_image ...
   --joint_angles ...` (press `p` for pos/rot/focal) → paste into
   `config/ur5e_robotiq_2f85/data_collection_rgb_cfg.py` (a UR10e variant of that cfg
   still needs to be created, same subclass-and-swap pattern).
4. **RGB data collection + distillation** (A100/H200): 80k expert trajectories under the
   Table-2 randomizations (~24 GPU-h to collect), ResNet-18 + MLP student, 5-frame stack,
   KL-matching + pose-reconstruction aux loss, end-to-end encoder, ~350k iters (~2 days).
   Expect RGB sim success ≈ 50–60% of the expert's — real transfer is better than that
   number suggests (paper Table 1: peg 85% real).
5. **Real eval practicalities** (paper A.4): fix the receptive object (openbox) to the
   table with command strips — sim treats it as static; compliant silicone mat; consistent
   stage lighting; stuck-detection auto-recovery (no joint motion >2 s → open gripper 1 s).

---

## Quick reference — file locations

| what | where |
|---|---|
| Real chirp data | `~/sysid_data_ur10e_real.pt` (laptop) + copy on A100 |
| Sysid fits | `logs/sysid/<timestamp>/` (A100 → rsync to laptop) |
| UR10e real-side kinematics | `diffusion_policy/real_world/ur10e_kinematics.py` on `syedjameel/diffusion_policy` branch `ur10e-linear-gripper` |
| Controller file dump | `~/urcontrol_from_ur10e/` (laptop) |
| Extracted (nominal) calibration | `~/ur10e_calibration.yaml` |
| Sysid target metadata | `source/uwlab_assets/.../local/Robots/Ur10eLinearGripper/metadata.yaml` (`sysid:` block = UR5e PLACEHOLDER until §6) |
| Sim pipeline manual | `UR10E_PIPELINE_README.md` |
| Paper / official doc | `2603.15789v3.pdf` / uw-lab.github.io → publications → omnireset → sim2real |
