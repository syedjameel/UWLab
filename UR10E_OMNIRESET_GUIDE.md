# UR10e + Linear Gripper — OmniReset Sim2Real: Complete Guide

One document, scratch → deployment, for the **UR10e + custom linear parallel-jaw gripper + 3× RealSense D405** port of the OmniReset pcb/openbox insertion task. Every command in order, plus the traps that matter. Supersedes `UR10E_PIPELINE_README.md` and `UR10E_SIM2REAL_PROCEDURE.md`.

**Result achieved with this pipeline:** ~90% success on the real robot (beats the paper's ~85% peg transfer).

---

## 0. What you're building

The pipeline distills a privileged **state** RL expert (trained in sim, reads object poses) into a **vision** student (3 camera images + robot state) that deploys on the real arm:

```
USDs → sysid → reset datasets → Stage-1 RL → Stage-2 finetune (expert)
     → camera calibration → 80k RGB demos → vision student → sim eval → REAL DEPLOY
```

### Repos & branches (push only to the forks)
| repo | path | branch | fork |
|---|---|---|---|
| sim (UWLab) | `~/work/repos/UWLab` | `omnireset/ur10e-custom-table` | `syedjameel/UWLab` |
| real (diffusion_policy) | `~/work/repos/diffusion_policy` | `ur10e-linear-gripper` | `syedjameel/diffusion_policy` |

### Machines & conda envs
| machine | role | conda env(s) |
|---|---|---|
| **RTX GPU box** (4090 / L40S / laptop 3060) | anything with `--enable_cameras` (calibration render, RGB collect, sim eval), and the deploy | `env_uwlab` (=`leisaac` on laptop), `robodiff_real` |
| **compute box** (A100 / H100) | non-render: reset recording, RL training, student training | `env_uwlab`, `robodiff` |
| **laptop 3060** | small validation, viz | `leisaac` |

> ⚠ **A100/H100 CANNOT render** — no graphics engine; any `--enable_cameras` run segfaults 1 s into startup (`carb.glinterop`/`gpu.foundation`). Non-render jobs (reset recording, RL/student training) run fine there. Rendering needs an **RTX-class** GPU **on driver branch 580** (595 segfaults `rtx.scenedb` on Isaac 5.1 / kit 107.3.3 — `sudo apt install nvidia-driver-580`).

### Hardware constants (real rig)
- Robot: UR10e, IP `192.168.0.100`, PolyScope 5.25, payload **0.575 kg**, COG `[0, 0, 0.050]`.
- D405 serials: front `409122273078`, side `323622272232`, wrist `409122272284`.
- Home pose (pendant deg): `69.58 -98.08 138.53 -130.43 -89.95 -20.42` = TCP `[0, -500, 100] mm` (100 mm above the mat).
- ArUco marker: dict `6x6_50`, ID `12`, size `150 mm`, laid flat on the green mat, **+X pointing from the base toward the workspace** (pendant −Y = sim +X).

> ⚠ **Rig is rotated 90° from the authors'.** The workspace is toward pendant **−Y** (sim +X). Two conventions handle it, both already implemented: (1) deploy applies `q1_sim = q1_real − 90°` at the RTDE read boundary; (2) calibration uses `aruco_offset = [0.455, 0, 0]` (sim frame). Nothing else to do — just don't "fix" it.

---

## 1. One-time environment setup

```bash
mkdir -p ~/work/repos && cd ~/work/repos
git clone -b omnireset/ur10e-custom-table https://github.com/syedjameel/UWLab.git
git clone -b ur10e-linear-gripper        https://github.com/syedjameel/diffusion_policy.git
```

### 1a. `env_uwlab` (Isaac sim env) — the compute/RTX boxes
Isaac Sim 5.1 + UWLab. On the laptop it is called `leisaac`. Fresh-server bring-up:
```bash
cd ~/work/repos/UWLab
./uwlab.sh --conda env_uwlab            # python 3.11 env from environment.yml
conda activate env_uwlab && pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
conda install -y -c conda-forge libglu    # else "HydraEngine rtx failed"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # add to ~/.bashrc
which gcc g++ make cmake || conda install -y -c conda-forge cmake make ninja c-compiler cxx-compiler
./uwlab.sh --install                     # UWLab extensions + rsl_rl (UW-Lab fork 3.1.2)
pip install torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128   # restore if dropped
./uwlab.sh -p scripts/tutorials/00_sim/create_empty.py   # verify (first load is slow: shader warm-up)
```
**Also install diffusion_policy into this env** (needed for `eval_distilled_policy.py`):
```bash
cd ~/work/repos/diffusion_policy && conda activate env_uwlab \
  && pip install -e . \
  && pip install dill hydra-core omegaconf zarr einops "diffusers<0.37" wandb accelerate robomimic
```

### 1b. `robodiff` (student training env) — the compute box
```bash
cd ~/work/repos/diffusion_policy && mamba env create -f conda_environment.yaml
# pinned 2022-era env (py3.9, protobuf 3.20). NEVER `pip install -U` inside it (breaks protobuf/wandb).
# if you must: pin, e.g. pip install "wandb==0.13.3" "protobuf==3.19.6"
```

### 1c. `robodiff_real` (calibration + deploy env) — the RTX/deploy box
```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf libspnav-dev spacenavd
cd ~/work/repos/diffusion_policy
mamba env create -f conda_environment_real.yaml
conda activate robodiff_real && pip install -e .
# the yaml pins opencv-*-headless; the deploy script needs a GUI window, so swap it:
pip uninstall -y opencv-contrib-python-headless opencv-python-headless
pip install "opencv-contrib-python<4.10"
```

---

## 2. Rebuild the USDs (every fresh checkout / after asset changes)

All USDs are gitignored and regenerated from committed URDFs/meshes/`table_dims.yaml`. Run on **every machine** that builds envs, in this order:

```bash
conda activate env_uwlab && cd ~/work/repos/UWLab

# 2a. UR10e arm (URDF committed, self-contained)
./uwlab.sh -p scripts_v2/tools/conversions/convert_gripper_urdf.py --fix-base \
  --input  source/uwlab_assets/uwlab_assets/local/Robots/UR10e/ur10e.urdf \
  --output source/uwlab_assets/uwlab_assets/local/Robots/UR10e/ur10e.usd

# 2b. Standalone linear gripper (grasp sampling uses this)
./uwlab.sh -p scripts_v2/tools/conversions/convert_gripper_urdf.py \
  --input  source/uwlab_assets/uwlab_assets/local/Robots/LinearGripper/gripper.urdf \
  --output source/uwlab_assets/uwlab_assets/local/Robots/LinearGripper/linear_gripper.usd
./uwlab.sh -p scripts_v2/tools/conversions/add_gripper_mimic.py \
  --usd source/uwlab_assets/uwlab_assets/local/Robots/LinearGripper/linear_gripper.usd --dual-drive

# 2c. Graft the gripper onto the arm (pure python). Bakes: mass 0.575 kg, wrist ±180°,
#     de-instanced gripper visuals (for the RGB gripper-appearance DR).
python scripts_v2/tools/conversions/graft_gripper_on_ur10e.py
#     MUST print: "gripper mass: 1.100 -> 0.575", "wrist joint limits -> +/-180 deg",
#                 "gripper visuals: de-instanced 3 prim(s)"

# 2d. Custom lab table + mount plate (from measured dims). Pillars OFF for the real rig.
grep "enabled:" source/uwlab_assets/uwlab_assets/local/Props/Mounts/CustomLabTable/table_dims.yaml  # want false
python scripts_v2/tools/conversions/make_custom_table_usd.py
```

Smoke check (optional, ~2 min): `./uwlab.sh -p scripts_v2/tools/conversions/smoke_test_linear_tasks.py --device cpu --task OmniReset-UR10eLinearGripper-ObjectAnywhereEEAnywhere-v0 --step` → `[SMOKE_RESULT] [PASS]`.

---

## 3. Kinematic calibration — use NOMINAL

The UR10e's factory kinematic calibration was wiped by a firmware reflash, so the robot's own controller runs nominal DH — which matches the sim (FK cross-checks <0.5 mm). **Proceed with nominal; nothing to do.** If real insertion precision struggles later, factory-recalibrate via UR (needs their fixture), then replace `local/Robots/UR10e/ur10e.urdf`, rebuild USDs, re-extract metadata, and **re-record resets + retrain** (geometry-dependent).

---

## 4. System identification (real dynamics → sim)

Fits joint friction/armature/delay so the sim arm behaves like the real one. Only Stage-2 finetune reads these; Stage-1 ignores them.

### 4a. Collect the chirp (real robot, `robodiff_real` or a minimal env)
Pre-position near `0 -90 90 -90 -90 0` in freedrive, clear a 30 cm bubble around the EE, e-stop in hand.
```bash
cd ~/work/repos/diffusion_policy && conda activate robodiff_real
PYTHONPATH=$PWD python scripts/sim2real/collect_sysid_data.py --robot ur10e \
  --robot_ip 192.168.0.100 --output ~/sysid_data_ur10e_real.pt --duration 8 --f0 0.1 --f1 3.0
# -> 4000 steps @ 500 Hz. Same OSC as sim runs on the robot via ur_rtde.directTorque.
```

### 4b. Fit (compute box, `env_uwlab`)
```bash
scp ~/sysid_data_ur10e_real.pt <compute-box>:~/
./uwlab.sh -p scripts_v2/tools/sim2real/sysid_ur5e_osc.py --headless --robot ur10e \
  --num_envs 512 --real_data ~/sysid_data_ur10e_real.pt --max_iter 200 \
  --armature_max 40 --friction_max 60 --viscous_friction_max 80 --delay_max 8
# -> logs/sysid/<timestamp>/final_results.pt  (plain argparse; NO hydra env.* overrides)
```

### 4c. Verify (laptop) & integrate
```bash
rsync -av <compute-box>:UWLab/logs/sysid/ logs/sysid/
./uwlab.sh -p scripts_v2/tools/sim2real/plot_sysid_fit.py --headless --robot ur10e \
  --checkpoint logs/sysid/<timestamp>/checkpoint_0200.pt --real_data ~/sysid_data_ur10e_real.pt
```
**Accept: <2° RMSE per joint** (read the per-joint titles in `sysid_fit_error.png`, NOT the pooled "RMSE"). Then paste `best_armature / best_friction / best_dynamic_ratio / best_viscous_friction` into the `sysid:` block of `local/Robots/Ur10eLinearGripper/metadata.yaml`. The identified **motor delay is ~0**; the finetune DR draws delay from **(0, 1)** @ 120 Hz.

---

## 5. Record the four reset datasets + QC (compute box)

The RL curriculum and demo collection seed episodes from four reset datasets, sampled **25% each** (`probs: [0.25]*4`). Record into `./Datasets_ur10e/` (NOT `./Datasets/` — reset files are keyed by object pair only and would clobber the UR5e sets).

```bash
conda activate env_uwlab && cd ~/work/repos/UWLab
OBJ="env.scene.insertive_object=pcb env.scene.receptive_object=openbox"

# 5a. Inputs (arm-independent; skip if already present under ./Datasets/OmniReset):
./uwlab.sh -p scripts_v2/tools/record_partial_assemblies.py \
  --task OmniReset-PartialAssemblies-v0 --num_envs 10 --num_trajectories 10 --headless $OBJ
./uwlab.sh -p scripts_v2/tools/record_grasps.py \
  --task OmniReset-LinearGripper-GraspSampling-v0 --num_envs 8192 --num_grasps 1000 --headless \
  env.scene.object=pcb

# 5b. The four reset types (C1 before C2; ~10k states each). Aim for ~10k on ALL FOUR,
#     including PartiallyAssembled — under-recording it weakens task COMPLETION (see Appendix C).
# C1 Reaching
./uwlab.sh -p scripts_v2/tools/record_reset_states.py \
  --task OmniReset-UR10eLinearGripper-ObjectAnywhereEEAnywhere-v0 \
  --num_envs 4096 --num_reset_states 10000 --headless \
  --dataset_dir ./Datasets_ur10e/OmniReset $OBJ
# C2 Near Object (needs grasps + C1)
./uwlab.sh -p scripts_v2/tools/record_reset_states.py \
  --task OmniReset-UR10eLinearGripper-ObjectRestingEEGrasped-v0 \
  --num_envs 4096 --num_reset_states 10000 --headless \
  --dataset_dir ./Datasets_ur10e/OmniReset $OBJ \
  env.events.reset_insertive_object_pose_from_reset_states.params.dataset_dir=./Datasets_ur10e/OmniReset \
  env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets/OmniReset
# C3 Grasped (needs grasps)
./uwlab.sh -p scripts_v2/tools/record_reset_states.py \
  --task OmniReset-UR10eLinearGripper-ObjectAnywhereEEGrasped-v0 \
  --num_envs 4096 --num_reset_states 10000 --headless \
  --dataset_dir ./Datasets_ur10e/OmniReset $OBJ \
  env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets/OmniReset
# C4 Near Goal (needs partial assemblies + grasps; slowest — real in-box grips are hard)
./uwlab.sh -p scripts_v2/tools/record_reset_states.py \
  --task OmniReset-UR10eLinearGripper-ObjectPartiallyAssembledEEGrasped-v0 \
  --num_envs 4096 --num_reset_states 10000 --headless \
  --dataset_dir ./Datasets_ur10e/OmniReset $OBJ \
  env.events.reset_insertive_object_pose_from_partial_assembly_dataset.params.dataset_dir=./Datasets/OmniReset \
  env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets/OmniReset

# 5c. QC gate — must PASS before training:
python scripts_v2/tools/conversions/qc_reset_states_ur10e.py --dataset_dir ./Datasets_ur10e/OmniReset
# expect [QC_RESULT] [PASS]. If Near Goal has many open-jaw states, salvage:
#   python scripts_v2/tools/conversions/filter_reset_states.py --in-place \
#     --input ./Datasets_ur10e/OmniReset/Resets/OpenBox__Pcb/resets_ObjectPartiallyAssembledEEGrasped.pt --min-grip 0.03
```

> Laptop only: prepend the PhysX trims (`env.sim.physx.gpu_collision_stack_size=67108864 ...`) — the 6 GB GPU OOMs on contact without them. **Never on the A100/H100.**

---

## 6. Train the state expert (compute box)

### 6a. Stage-1 RL (from scratch)
```bash
./uwlab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-v0 \
  --num_envs 16384 --headless --logger tensorboard $OBJ \
  env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e/OmniReset
# --seed <int> for extra seeds; monitor success/return with `tensorboard --logdir logs/rsl_rl`
```

### 6b. Stage-2 finetune (ADR, adapts the ideal Stage-1 policy to the sysid'd robot)
```bash
./uwlab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-Finetune-v0 \
  --num_envs 4096 --headless --logger tensorboard $OBJ \
  --resume_path logs/rsl_rl/<experiment>/<stage1_run>/model_<iter>.pt \
  env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e/OmniReset
# DONE when Curriculum/adr_sysid/scale_progress -> 1.0 at success ~0.95 (~8 h, peg-class).
```

### 6c. Play the expert (laptop, GUI — sanity check)
```bash
./uwlab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-Finetune-Play-v0 \
  --num_envs 4 $OBJ $TRIMS \
  env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e/OmniReset \
  --load_run <finetune run> --checkpoint model_<iter>.pt
```

---

## 7. Camera calibration (RTX box, `robodiff_real` + `env_uwlab`)

Do this once the cameras are in their final rigid mounts. **One camera at a time** for the ArUco capture. Anchor = a **TCP touch-off** on the marker center: jog the tool tip onto the marker, read pendant base-frame xyz → this is `aruco_offset` (2026-07-16: `[0.455, 0, 0]`).

### 7a. Capture + coarse extrinsics (per camera; `robodiff_real`)
```bash
conda activate robodiff_real && cd ~/work/repos/diffusion_policy/scripts/sim2real
python 0_camera_calibrate.py          # ArUco -> pose in sim frame (10 rounds averaged)
python 1_camera_get_rgb.py            # -> perception/calibrations/real_rgb.png (arm at capture pose)
python 2_get_isaacsim_extrinsics.py   # prints sim-frame pos + quat (warm start)
cp perception/calibrations/most_recent_calib.json perception/calibrations/<cam>_camera_calib.json
cp perception/calibrations/real_rgb.png           perception/calibrations/real_rgb_<cam>.png
```
Wrist: hold the arm at a pose where the wrist camera sees the whole marker (capture pose 2026-07-16: pendant `69.58 -98.08 138.53 -130.43 -89.95 -20.42`). Its `2_` output is base-frame; convert to the LINK-relative offset with `get_link_pose_ur10e.py`.

### 7b. Automated sweep refinement (`env_uwlab`) — do BEFORE the interactive pass
```bash
conda activate env_uwlab && cd ~/work/repos/UWLab
JOINTS="-20.42 -98.08 138.53 -130.43 -89.95 -20.42"   # sim = pendant q1 - 90
CAL=~/work/repos/diffusion_policy/scripts/sim2real/perception/calibrations
./uwlab.sh -p scripts_v2/tools/conversions/sweep_camera_align.py --camera front_camera \
  --real_image $CAL/real_rgb_front.png --joint_angles $JOINTS --out table_swap_snaps/sweep_front
./uwlab.sh -p scripts_v2/tools/conversions/sweep_camera_align.py --camera side_camera \
  --real_image $CAL/real_rgb_side.png  --joint_angles $JOINTS --out table_swap_snaps/sweep_side
# wrist needs --mask (marker exists only in the real image) + --reset_each (OSC hold drifts):
./uwlab.sh -p scripts_v2/tools/conversions/sweep_camera_align.py --camera wrist_camera \
  --real_image $CAL/real_rgb_wrist.png --joint_angles $JOINTS \
  --mask 100 150 560 480 --reset_each --out table_swap_snaps/sweep_wrist
```
⚠ **Do NOT derive the focal from intrinsics** — the renderer's FOV runs ~11% wider than the USD focal math. The sweep fits it empirically (front **14.32**, side **13.34**, wrist **12.74**).

### 7c. Interactive fine-tune (`env_uwlab`, needs a display)
```bash
./uwlab.sh -p scripts_v2/tools/sim2real/align_cameras.py --enable_cameras --headless \
  --robot ur10e --camera front_camera --real_image $CAL/real_rgb_front.png --joint_angles $JOINTS
# keys: w/x a/d up/down move | i/k j/l u/o rotate (camera axes) | left/right focal | 1/2 blend | p print | q
# repeat for side/wrist. Gripper holds OPEN by default (do NOT pass --gripper_pos -1).
```
Paste the printed `pos/rot/focal` from each camera into `_UR10E_CAMERA_POSES` in `config/ur5e_robotiq_2f85/ur10e_linear_gripper_rgb_cfg.py` — one dict drives the scene cameras, DR event bases, and focal jitter for the CameraAlign / DataCollection / Play envs.

---

## 8. Collect the 80k RGB demos (RTX box, `env_uwlab`)

```bash
conda activate env_uwlab && cd ~/work/repos/UWLab
OBJ="env.scene.insertive_object=pcb env.scene.receptive_object=openbox"

# 8a. Regenerate the local USDs (pillar-free table + de-instanced graft) — see §2c/2d.

# 8b. Export the finetuned expert to TorchScript (also your checkpoint sanity check):
./uwlab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-Finetune-Play-v0 \
  --num_envs 4 --checkpoint <path/to/finetune/model_<iter>.pt> --headless $OBJ \
  env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e/OmniReset
# -> <checkpoint_dir>/exported/policy.pt

# 8c. 100-demo smoke (~2 min once assets cached; first run downloads ~957 textures + ~920 HDRIs):
./uwlab.sh -p scripts_v2/tools/collect_demos.py \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-RGB-DataCollection-v0 \
  --dataset_file datasets/ur10e_pcb/rgb_smoke.zarr --num_envs 32 --num_demos 100 \
  --enable_cameras --headless $OBJ \
  agent.algorithm.offline_algorithm_cfg.behavior_cloning_cfg.experts_path='["<ckpt_dir>/exported/policy.pt"]'

# 8d. QC gate — eyeball before the long run (plain python, no Isaac):
python scripts_v2/tools/visualize_rgb_demos.py --dataset datasets/ur10e_pcb/rgb_smoke.zarr --out demo_viz --episodes 8
# CHECK: wrist tracks the gripper (not black); mats+curtains+objects+GRIPPER retexture across
# episodes; framing matches real; demos finish the assembly; no wall/solid-color frames.
# NOISE flags at std 80-90 with clean MP4s = false positives (busy textures) -> ignore.

# 8e. The 80k (run under tmux; ~11 h at 32 envs on a 4090, ~2 demos/s):
./uwlab.sh -p scripts_v2/tools/collect_demos.py \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-RGB-DataCollection-v0 \
  --dataset_file datasets/ur10e_pcb/rgb0.zarr --num_envs 32 --num_demos 80000 \
  --enable_cameras --headless $OBJ \
  agent.algorithm.offline_algorithm_cfg.behavior_cloning_cfg.experts_path='["<ckpt_dir>/exported/policy.pt"]'
# Only successful episodes saved. A crash loses in-flight episodes; a restart begins a FRESH
# dataset -> collect extra runs into rgb1.zarr, rgb2.zarr, ... (same dir merges at training).
```

---

## 9. Train the vision student (compute box, `robodiff`)

Pure PyTorch — no Isaac, no USDs. Copy the zarr(s) to a clean dir on the training box:
```bash
mkdir -p ~/UWLab_datasets/ur10e_pcb
rsync -a --info=progress2 <rtx-box>:~/work/repos/UWLab/datasets/ur10e_pcb/rgb0.zarr ~/UWLab_datasets/ur10e_pcb/

conda activate robodiff && cd ~/work/repos/diffusion_policy
wandb login    # or append logging.mode=offline
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name train_mlp_sim2real_image_with_aux_loss_workspace.yaml \
  --config-dir diffusion_policy/config \
  task.dataset.dataset_dir=/home/<user>/UWLab_datasets/ur10e_pcb
# ResNet-18 encoder + MLP head + aux object-pose loss. 1 iteration = 1 batch (bs 64);
# ~39k batches/epoch for 80k demos -> the authors' 350k iterations ≈ 9 epochs.
# It does NOT auto-stop: grab checkpoints/step_0350000.ckpt (every 10k steps), then kill.
```

---

## 10. Evaluate the student in sim (RTX box, `env_uwlab`)

```bash
conda activate env_uwlab && cd ~/work/repos/UWLab
./uwlab.sh -p scripts_v2/tools/eval_distilled_policy.py \
  --task OmniReset-UR10eLinearGripper-RelCartesianOSC-RGB-Play-v0 \
  --checkpoint <student>/step_0350000.ckpt \
  --num_envs 1 --num_trajectories 20 --headless --enable_cameras --save_video $OBJ
# -> policy_cameras.mp4 (front|side|wrist). ~50-60% of the expert is NORMAL and fine to
# proceed (paper: students score modestly in sim yet transfer better). Near-zero / broken
# motion = structural bug (obs mismatch, path) -> debug, don't deploy.
```

---

## 11. Deploy on the real robot (deploy PC, `robodiff_real`)

```bash
conda activate robodiff_real && cd ~/work/repos/diffusion_policy && git pull fork ur10e-linear-gripper
scp <train-box>:.../checkpoints/step_0350000.ckpt checkpoints/student_350k.ckpt
python eval_real_robot.py -i checkpoints/student_350k.ckpt -o ./demo_eval --robot_ip 192.168.0.100 -j -f 10
# keys (in the OpenCV window): C = hand to policy | S = take back | R = reset | Q = quit
```
**Control rates:** policy **10 Hz** (`-f 10`, must match training), OSC torque loop **500 Hz**, cameras **30 fps**. Do not change `-f`.

### Pre-flight (every run)
1. ⭐ **Cameras rigidly at their calibration positions** (they were unplugged during §7 — confirm none shifted; a moved camera = out-of-distribution images).
2. Green mat / openbox / 40 mm pcb placed like collection. **Anchor the openbox** (tape/clamp) — sim treats it as immovable (kinematic); a sliding real box makes the policy drag it (Appendix C).
3. `ls /dev/ttyACM*` shows the gripper; press `g` and confirm the jaws OPEN before `C`.
4. Payload/TCP set on the pendant, protective stop cleared, **hand on the e-stop**.

### What happens on launch
The instant `RealEnv` starts, the arm does a **stiff `moveJ` to home** (`[69.58,-98.08,138.53,-130.43,-89.95,-20.42]°`, ~1 rad/s, position-controlled — NOT compliant) — before any keypress. Since the code home now matches the rig home, this is a near-zero move, but **clear the path and hold the e-stop from launch.** Everything after `C` (policy) and the `R` reset use the compliant bounded-force OSC.

### Why it's safe
Compliant OSC with a **bounded task-space error clamp (0.05 m / 0.3 rad)** → a bad action yields ≤50 N/axis, torque hard-capped at `[330,330,150,56,56,56]` Nm, soft push on contact. Action scale, gains, payload all match the sim eval action. Startup `moveJ` is the only stiff motion.

---

## Appendix A — Traps that each cost a debugging round

1. **A100/H100 can't render** → collection/eval/deploy-render on RTX only.
2. **NVIDIA driver 595 segfaults the RTX renderer** on Isaac 5.1 → use the **580** branch. Set CPU governor to `performance`.
3. **First DR run downloads ~957 textures + ~920 HDRIs** (~20 GB, one-time) to `~/.cache/uwlab/assets`; it looks idle (watch `find ~/.cache/uwlab/assets -type f | wc -l`). A transient download failure dies at the first reset with a misleading `ManagerTermBase.reset() missing 'self'` — just rerun (downloads resume, now with 3× retry + a clear error banner).
4. **Loading the finetune checkpoint's critic needs the PILLARED table** (172 obs dims vs 160 pillar-free). For the §8b export, `sed -i 's/enabled: false/enabled: true/'` `table_dims.yaml` + regenerate, then flip back to `false` + regenerate for collection. The exported actor-only policy.pt doesn't care. Forgetting the flip-back = pillars in the demos.
5. **RGB env spacing must be 3.0 m** (baked in) — the ~1.8 m scene overlaps neighbors at the authors' 1.5 m (front/side cameras stare at a neighbor's curtain / see impossible "robot from behind" frames). 2-env smokes hide it; ≥4-env grids show it.
6. **Gripper-appearance DR needs the graft's de-instancing** (`de-instanced 3 prim(s)`); mesh patterns are naming-agnostic regexes because converter node names differ per machine.
7. **Camera hardware-reset on open**: `SingleRealsense` now resets each D405 before opening (a device left bad by a prior crash `pipeline.start()`s OK but times out in `wait_for_frames`). Kill any process holding the cameras before launching.
8. **Gripper binary-action sign:** positive/zero = OPEN, negative = CLOSE. Never pass `--gripper_pos -1` to `align_cameras` (marches the jaws shut, fakes misalignment).
9. **Never record resets into `./Datasets/`** — clobbers the UR5e sets. Use `./Datasets_ur10e/`.
10. **One Isaac env per process**; laptop needs PhysX trims, A100/H100 must not.

## Appendix B — Known sim2real gaps & tuning knobs

- **Openbox is kinematic (immovable) in sim** (`receptive_object.kinematic_enabled=True`). The policy learned to push against a fixed receptacle → **physically anchor the real openbox**, or it gets dragged. (Making it dynamic + retraining is the expensive alternative.)
- **Task-completion weakness ("holds cube near box, stuck")** traces to under-recording `ObjectPartiallyAssembledEEGrasped` (the only near-completion reset type). Record it to full ~10k; a partial mitigation is bumping its `probs` share above 0.25.
- **Gripper 1 s stroke** is modeled in the finetune/RGB envs (`_apply_real_gripper_speed`); deploy has an open-gripper macro (`g`) + a 2 s stuck-detector as backstops.
- **Payload 0.575 kg** (weighed) drives the deploy OSC gravity comp (`setPayload`).
- **Images resized, not cropped**: 640×480 → 224×224 (same 4:3→1:1 squish on both sides).

## Appendix C — Task-id quick reference

| purpose | task id |
|---|---|
| reset recording | `OmniReset-UR10eLinearGripper-Object{AnywhereEEAnywhere,RestingEEGrasped,AnywhereEEGrasped,PartiallyAssembledEEGrasped}-v0` |
| Stage-1 RL | `OmniReset-UR10eLinearGripper-RelCartesianOSC-State-v0` |
| Stage-2 finetune | `...-RelCartesianOSC-State-Finetune-v0` |
| play expert | `...-RelCartesianOSC-State-Finetune-Play-v0` |
| camera align | `OmniReset-UR10eLinearGripper-CameraAlign-v0` |
| RGB collect | `...-RelCartesianOSC-RGB-DataCollection-v0` |
| RGB play / student eval | `...-RelCartesianOSC-RGB-Play-v0` |

**Commit conventions:** no Claude co-author; push only to the forks (`syedjameel/UWLab`, `syedjameel/diffusion_policy`), never upstream.
