# UR10e + Linear Gripper — OmniReset Pipeline (Laptop + A100)

End-to-end guide for the **UR10e + custom linear parallel-jaw gripper** port of the OmniReset
pcb/openbox task: rebuild the USDs, validate + visualize reset states on the laptop, run the
full dataset-generation + training pipeline on the A100, and play the trained policy back on
the laptop.

- **Branch:** `omnireset/ur10e-linear-gripper` (fork `syedjameel/UWLab`)
- **Task ids:** `OmniReset-UR10eLinearGripper-*` (the UR5e `OmniReset-UR5eLinearGripper-*` and
  2F-85 tasks are untouched and coexist)
- **Environment (both machines):** `conda activate leisaac`, run everything via `./uwlab.sh -p <script>`

Every USD used below is **gitignored and regenerated** — the committed artifacts are the
conversion/graft scripts plus the URDF + meshes + `metadata.yaml`. A fresh checkout (A100)
must run Part 1 first.

---

## Contents

1. [Rebuild the USDs (laptop and A100, identical)](#1-rebuild-the-usds)
2. [Laptop: small validation recordings](#2-laptop-small-validation-recordings)
3. [Laptop: visualize the four reset types](#3-laptop-visualize-the-four-reset-types)
4. [A100: full pipeline (datasets → training)](#4-a100-full-pipeline)
5. [Laptop: play the trained policy](#5-laptop-play-the-trained-policy)
6. [Gotchas & debugging](#6-gotchas--debugging)

---

## 1. Rebuild the USDs

Three USDs, built in order (arm → gripper → graft). Needed once per machine / fresh checkout.

```bash
conda activate leisaac
cd <repo root>

# 1a. UR10e arm: URDF -> USD (URDF + meshes are committed, self-contained, no internet)
./uwlab.sh -p scripts_v2/tools/conversions/convert_gripper_urdf.py --fix-base \
  --input  source/uwlab_assets/uwlab_assets/local/Robots/UR10e/ur10e.urdf \
  --output source/uwlab_assets/uwlab_assets/local/Robots/UR10e/ur10e.usd

# 1b. Standalone linear gripper (grasp sampling uses this; arm-independent)
./uwlab.sh -p scripts_v2/tools/conversions/convert_gripper_urdf.py \
  --input  source/uwlab_assets/uwlab_assets/local/Robots/LinearGripper/gripper.urdf \
  --output source/uwlab_assets/uwlab_assets/local/Robots/LinearGripper/linear_gripper.usd
./uwlab.sh -p scripts_v2/tools/conversions/add_gripper_mimic.py \
  --usd source/uwlab_assets/uwlab_assets/local/Robots/LinearGripper/linear_gripper.usd \
  --dual-drive
# add_gripper_mimic.py edits the USD IN PLACE: --dual-drive (both jaws driven, no mimic --
# what the pipeline uses) + defaults bake finger friction 100, maxJointVelocity 130 and the
# JawBox colliders -- all required for grasping to work.

# 1c. Graft the gripper onto the UR10e arm (pure python, no Isaac needed)
python scripts_v2/tools/conversions/graft_gripper_on_ur10e.py
#   -> source/.../local/Robots/Ur10eLinearGripper/ur10e_linear_gripper.usd
#   The graft also: dual-drives the follower jaw, strips the inert mimic, and gives the
#   URDF importer's zero-mass frame links (base/base_link/flange/tool0) 0.01 kg.
#   Mount standoff along wrist_3 +Z defaults to 0.049 m (--standoff to retune; eyeball in
#   the GUI during Part 3 — it is inherited from the UR5e and not yet visually confirmed).
```

Sanity check (optional, ~2 min): build + step one env.

```bash
./uwlab.sh -p scripts_v2/tools/conversions/smoke_test_linear_tasks.py --device cpu \
  --task OmniReset-UR10eLinearGripper-ObjectAnywhereEEAnywhere-v0 --step
# expect: [SMOKE_RESULT] [PASS] ... OK (built + stepped)
```

---

## 2. Laptop: small validation recordings

Small-scale versions of the real pipeline (16 envs, 20 states each) to validate the flow
before burning A100 time. **Two rules on the laptop:**

- **PhysX buffer trims are mandatory** on the 6 GB RTX 3060 — without them the sim OOMs the
  instant contact happens (`PhysX Internal CUDA error code 2`). The A100 does NOT need them.
- **Output to `./Datasets_ur10e/`, never `./Datasets/`** — reset files are keyed by object
  pair only (`Resets/OpenBox__Pcb/resets_<type>.pt`), so recording into `./Datasets/` would
  silently **overwrite your UR5e reset datasets**.

Define these once per shell:

```bash
TRIMS="env.sim.physx.gpu_collision_stack_size=67108864 \
  env.sim.physx.gpu_max_rigid_contact_count=2097152 \
  env.sim.physx.gpu_max_rigid_patch_count=2097152 \
  env.sim.physx.gpu_total_aggregate_pairs_capacity=2097152 \
  env.sim.physx.gpu_found_lost_aggregate_pairs_capacity=2097152"
OBJ="env.scene.insertive_object=pcb env.scene.receptive_object=openbox"
```

Inputs reused from the UR5e work (all **arm-independent**, already in `./Datasets/OmniReset`):
`Grasps/Pcb/grasps.pt` (gripper-only) and `Resets/OpenBox__Pcb/partial_assemblies.pt`
(object poses only). Only the reset states are robot-specific and must be re-recorded.

```bash
# 2a. Reaching — self-contained, no dataset inputs
./uwlab.sh -p scripts_v2/tools/record_reset_states.py \
  --task OmniReset-UR10eLinearGripper-ObjectAnywhereEEAnywhere-v0 \
  --num_envs 16 --num_reset_states 20 --headless \
  --dataset_dir ./Datasets_ur10e/OmniReset $OBJ $TRIMS

# 2b. Grasped — needs grasps (input from ./Datasets/OmniReset)
./uwlab.sh -p scripts_v2/tools/record_reset_states.py \
  --task OmniReset-UR10eLinearGripper-ObjectAnywhereEEGrasped-v0 \
  --num_envs 16 --num_reset_states 20 --headless \
  --dataset_dir ./Datasets_ur10e/OmniReset $OBJ \
  env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets/OmniReset \
  $TRIMS

# 2c. Near Object — needs grasps AND the UR10e's own 2a output (note the two different dirs)
./uwlab.sh -p scripts_v2/tools/record_reset_states.py \
  --task OmniReset-UR10eLinearGripper-ObjectRestingEEGrasped-v0 \
  --num_envs 16 --num_reset_states 20 --headless \
  --dataset_dir ./Datasets_ur10e/OmniReset $OBJ \
  env.events.reset_insertive_object_pose_from_reset_states.params.dataset_dir=./Datasets_ur10e/OmniReset \
  env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets/OmniReset \
  $TRIMS

# 2d. Near Goal — needs partial assemblies + grasps
./uwlab.sh -p scripts_v2/tools/record_reset_states.py \
  --task OmniReset-UR10eLinearGripper-ObjectPartiallyAssembledEEGrasped-v0 \
  --num_envs 16 --num_reset_states 20 --headless \
  --dataset_dir ./Datasets_ur10e/OmniReset $OBJ \
  env.events.reset_insertive_object_pose_from_partial_assembly_dataset.params.dataset_dir=./Datasets/OmniReset \
  env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets/OmniReset \
  $TRIMS
```

Healthy laptop rates (validated 2026-07-02): 2a ≈ 4.7/s, 2b ≈ 3.0/s, 2c ≈ 1.8/s. If a
grasped type grinds at minutes-per-state, see [§6 debugging](#6-gotchas--debugging).

---

## 3. Laptop: visualize the four reset types

Opens the Isaac GUI (no `--headless`) on the Play env and fires a reset from the chosen
dataset every `--reset_interval` seconds. Startup takes ~1 min before the window appears.

```bash
# common tail for all four:
VIZ="--num_envs 4 --dataset_dir ./Datasets_ur10e/OmniReset --reset_interval 2.0 $OBJ $TRIMS"

# 1/4 Reaching
./uwlab.sh -p scripts_v2/tools/visualize_reset_states.py \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-Play-v0 \
  --reset_type ObjectAnywhereEEAnywhere $VIZ

# 2/4 Near Object
./uwlab.sh -p scripts_v2/tools/visualize_reset_states.py \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-Play-v0 \
  --reset_type ObjectRestingEEGrasped $VIZ

# 3/4 Grasped
./uwlab.sh -p scripts_v2/tools/visualize_reset_states.py \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-Play-v0 \
  --reset_type ObjectAnywhereEEGrasped $VIZ

# 4/4 Near Goal
./uwlab.sh -p scripts_v2/tools/visualize_reset_states.py \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-Play-v0 \
  --reset_type ObjectPartiallyAssembledEEGrasped $VIZ

# Or omit --reset_type to cycle all four types randomly in one session.
```

**Checklist while watching:**

- Jaws symmetric and actually ON the pcb in the grasped types (not hovering, not spun off).
- Wrist quiet after each reset — no visible chatter/vibration (that was the rot-Kp-6 bug,
  fixed; if you see it, you are on a stale checkout).
- Approaches predominantly top-down.
- **Gripper mount**: eyeball the standoff on the UR10e wrist (still the UR5e's 0.049 m). If
  the gripper base sits inside / too far from the flange, regenerate with
  `python scripts_v2/tools/conversions/graft_gripper_on_ur10e.py --standoff <m>` and re-record.

Laptop caveat: Isaac sometimes deadlocks at window close on this machine — `Ctrl+C`/`kill`
the process; the session's visuals were already valid.

---

## 4. A100: full pipeline

Numbers below are the repo's reference pcb/openbox pipeline (see
`OMNIRESET_PCB_TASK_DEEP_DIVE.md` §4-5), with the task ids swapped to the UR10e and the
outputs kept in `./Datasets_ur10e/` (same overwrite hazard as the laptop if the A100 copy has
UR5e datasets). **No PhysX buffer trims on the A100.**

```bash
conda activate leisaac
cd <repo root>
git fetch fork && git checkout omnireset/ur10e-linear-gripper
# Part 1 (USD rebuild) first if this is a fresh checkout.
OBJ="env.scene.insertive_object=pcb env.scene.receptive_object=openbox"
```

### Step A — partial assemblies (~30 s) — ONLY if not already present

Object-only (arm-independent). Skip if `Datasets/OmniReset/Resets/OpenBox__Pcb/partial_assemblies.pt`
already exists from the UR5e work; otherwise:

```bash
./uwlab.sh -p scripts_v2/tools/record_partial_assemblies.py \
  --task OmniReset-PartialAssemblies-v0 --num_envs 10 --num_trajectories 10 --headless $OBJ
#   -> Datasets/OmniReset/Resets/OpenBox__Pcb/partial_assemblies.pt
```

### Step B — grasps (~minutes) — ONLY if not already present

Gripper-only (arm-independent): the UR5e linear-gripper grasps are valid for the UR10e.
Skip if `Datasets/OmniReset/Grasps/Pcb/grasps.pt` (linear-gripper version) exists; otherwise:

```bash
./uwlab.sh -p scripts_v2/tools/record_grasps.py \
  --task OmniReset-LinearGripper-GraspSampling-v0 \
  --num_envs 8192 --num_grasps 1000 --headless \
  env.scene.object=pcb
#   note: env.scene.object (not insertive_object) for grasp sampling
#   -> Datasets/OmniReset/Grasps/Pcb/grasps.pt
```

### Step C — the four UR10e reset datasets (10 000 states each)

Order matters: C1 before C2 (C2 consumes C1's output).

```bash
# C1/4 Reaching (no dependencies)
./uwlab.sh -p scripts_v2/tools/record_reset_states.py \
  --task OmniReset-UR10eLinearGripper-ObjectAnywhereEEAnywhere-v0 \
  --num_envs 4096 --num_reset_states 10000 --headless \
  --dataset_dir ./Datasets_ur10e/OmniReset $OBJ

# C2/4 Near Object (needs grasps + C1)
./uwlab.sh -p scripts_v2/tools/record_reset_states.py \
  --task OmniReset-UR10eLinearGripper-ObjectRestingEEGrasped-v0 \
  --num_envs 4096 --num_reset_states 10000 --headless \
  --dataset_dir ./Datasets_ur10e/OmniReset $OBJ \
  env.events.reset_insertive_object_pose_from_reset_states.params.dataset_dir=./Datasets_ur10e/OmniReset \
  env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets/OmniReset

# C3/4 Grasped (needs grasps)
./uwlab.sh -p scripts_v2/tools/record_reset_states.py \
  --task OmniReset-UR10eLinearGripper-ObjectAnywhereEEGrasped-v0 \
  --num_envs 4096 --num_reset_states 10000 --headless \
  --dataset_dir ./Datasets_ur10e/OmniReset $OBJ \
  env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets/OmniReset

# C4/4 Near Goal (needs partial assemblies + grasps)
./uwlab.sh -p scripts_v2/tools/record_reset_states.py \
  --task OmniReset-UR10eLinearGripper-ObjectPartiallyAssembledEEGrasped-v0 \
  --num_envs 4096 --num_reset_states 10000 --headless \
  --dataset_dir ./Datasets_ur10e/OmniReset $OBJ \
  env.events.reset_insertive_object_pose_from_partial_assembly_dataset.params.dataset_dir=./Datasets/OmniReset \
  env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets/OmniReset
```

### Step D — train (single A100 80 GB)

Stage-1 RL (implicit actuator, soft OSC gains). Run directly — no `torch.distributed`,
no `--distributed` for one GPU. Checkpoints land under `logs/rsl_rl/<experiment>/<run>/`.

```bash
./uwlab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-v0 \
  --num_envs 16384 --headless --logger tensorboard $OBJ \
  env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e/OmniReset
```

Monitor with `tensorboard --logdir logs/rsl_rl` (success rate + return curves).

> Stage-2 finetune (`...-RelCartesianOSC-State-Finetune-v0`) reads the **sysid** block from
> `Ur10eLinearGripper/metadata.yaml` — since 2026-07-06 those are the REAL identified UR10e
> values (chirp + CMA-ES; see `UR10E_SIM2REAL_PROCEDURE.md` §4–§6 for provenance and §8 for
> the exact finetune command with `--resume_path`).

---

## 5. Laptop: play the trained policy

Copy the run folder from the A100 (keep the same relative path so auto-discovery works):

```bash
# from the laptop
rsync -av a100:<repo>/logs/rsl_rl/<experiment>/<run>/ logs/rsl_rl/<experiment>/<run>/
# and the datasets (the Play env resets from them):
rsync -av a100:<repo>/Datasets_ur10e/ Datasets_ur10e/
```

Play with the eval task (GUI, few envs, trims):

```bash
./uwlab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-Play-v0 \
  --num_envs 4 $OBJ $TRIMS \
  env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e/OmniReset
# picks the latest run/checkpoint under logs/rsl_rl automatically;
# pin one explicitly with:  --load_run <run folder name> --checkpoint model_<iter>.pt
```

---

## 6. Gotchas & debugging

- **Never record resets into `./Datasets/OmniReset`** — `Resets/<pair>/` paths carry no robot
  name; you would overwrite the UR5e datasets. Keep the UR10e in `./Datasets_ur10e/`.
- **Laptop = trims, A100 = no trims.** The trims cap the 2 GB PhysX collision stack the task
  configs request; on the A100 the full buffers are wanted.
- **One Isaac env per process.** Creating a second env after closing the first corrupts the
  GPU PhysX context (`computeArticulationData CUDA error 2`). Loop over tasks with fresh
  processes.
- **Low accept-rate on grasped reset types?** Run the recorder with `UWLAB_GRASP_DEBUG=1` —
  `check_reset_state_success` then prints per-condition pass counts + per-asset velocity
  detail at every episode timeout. That diagnostic is how the rot-Kp-6 contact instability
  was found (symptom: `stable=0` everywhere, wrist_3 velocity pinned at ±π, held pcb
  chattering ~3.5 rad/s).
- **OSC gain changes must be validated IN CONTACT** (jaws closed on an object), not just with
  free-space jitter/authority probes. Rot Kp stays 3 (see the comment in
  `config/ur5e_robotiq_2f85/actions.py`).
- **Isaac GUI deadlock at shutdown** (laptop): known; `Ctrl+C` the process after closing.
- **Grasps and partial assemblies are arm-independent** — never re-record them for an arm swap.
- **Quality-check any recorded reset file** (structure: `initial_state/articulation/robot/...`
  lists of tensors; `torch.load(..., map_location="cpu")`): FK the arm joints and check
  top-down %, `finger_joint` should be ~0.0487 (a real grip on the pcb) in grasped types,
  jaw asymmetry < 1 mm.
- **Mount standoff 0.049 m is inherited from the UR5e** and not yet visually confirmed on the
  UR10e — eyeball in Part 3, retune via the graft's `--standoff`, then re-record.
