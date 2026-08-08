# UR10e + Tesollo DELTO environments

This branch adds the Tesollo DG-5F hand to the existing UR10e OmniReset stack and a separate
dexterous lifting task. The required USD assets are committed; no model-generation step is needed.

## Assets and code

- Mounted robot: `source/uwlab_assets/uwlab_assets/local/Robots/Ur10eDelto/ur10e_delto.usd`
- Standalone hand: `source/uwlab_assets/uwlab_assets/local/Robots/DeltoHand/delto_hand.usd`
- OmniReset parts: `Props/Custom/DeltoBlock/delto_block.usd` and `DeltoSlot/delto_slot.usd`
- Robot config: `source/uwlab_assets/uwlab_assets/robots/ur10e_delto/`
- OmniReset config: `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/ur10e_delto_cfg.py`
- RGB config: the adjacent `ur10e_delto_rgb_cfg.py`
- Lift/reorient config: `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/dexlift/`

The hand is mounted directly on the UR flange (zero standoff). It has 20 actuated finger joints;
all 25 finger-link colliders are enabled. OmniReset uses a binary, calibrated grasp posture while
generic DexLift exposes all 26 arm-and-hand joints as relative position actions. The table-leg
task instead uses absolute, default-centered targets so zero action holds its validated pregrasp.

## Task IDs

OmniReset provides grasp sampling, the five reset-state variants, state train/finetune/play,
camera alignment, RGB collection, and RGB play under the prefix `OmniReset-UR10eDelto-`. Use
`deltoblock` and `deltoslot` for the integrated part pair.

DexLift registers:

- `DexLift-UR10eDelto-Lift-v0`
- `DexLift-UR10eDelto-Lift-Play-v0`
- `DexLift-UR10eDelto-Reorient-v0`
- `DexLift-UR10eDelto-Reorient-Play-v0`
- `DexLift-UR10eDelto-TableLeg-GraspLift-v0`
- `DexLift-UR10eDelto-TableLeg-GraspLift-Curriculum-v0`
- `DexLift-UR10eDelto-TableLeg-GraspLift-Play-v0`

The FurnitureBench table-leg task has a separate [grasp-and-lift guide](table_leg_grasp_lift.md).

## Run

Set the package path once from the repository root:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/source/uwlab_assets:$PWD/source/uwlab_tasks:$PWD/source/uwlab:$PWD/source/uwlab_rl"
```

Generate an OmniReset grasp dataset:

```bash
python scripts_v2/tools/record_grasps.py \
  --task OmniReset-Delto-GraspSampling-v0 --num_envs 256 --num_grasps 10000 --headless \
  env.scene.object=deltoblock
```

Build or train the OmniReset state task with the standard commands from
[`UR10E_OMNIRESET_GUIDE.md`](../../../../UR10E_OMNIRESET_GUIDE.md), replacing the task prefix with
`OmniReset-UR10eDelto-` and selecting:

```text
env.scene.insertive_object=deltoblock env.scene.receptive_object=deltoslot
```

Train DexLift:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task DexLift-UR10eDelto-Lift-v0 --num_envs 4096 --headless
```

Play a checkpoint:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task DexLift-UR10eDelto-Lift-Play-v0 --num_envs 1 --checkpoint /path/to/model.pt
```

OmniReset reset and grasp datasets are runtime artifacts and are intentionally not committed.
