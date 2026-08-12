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
all 25 finger-link colliders are enabled. **Every environment a policy trains in exposes all 20
hand joints as independent actions.** OmniReset pairs a 6-D Cartesian OSC arm with 20 relative
hand joint actions (26 total); generic DexLift uses 26 relative position actions over the whole
robot; the table-leg task uses 6 absolute, default-centered arm targets plus the same 20 relative
hand actions, so zero arm action holds its validated pregrasp.

A one-scalar closure (a binary open/closed command, or one fraction interpolating between two
calibrated postures) is **banned** on those paths and the ban is enforced, not documented:
`uwlab.envs.mdp.full_actuation` raises at config construction and again as a `startup` event if any
of the 20 joints is not its own action dimension. The reason is not tidiness — no single close
fraction puts this hand's opposing pads in front of its phalanges, so the two-jaw pinch is
unreachable at every value of such a scalar.

One fixed closed posture does still exist, as **data**: `DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS` in
`ur10e_delto/actions.py`, a measured posture that holds the 0.03 kg `DeltoBlock` through the
sampler's shake test. It is not an action and no policy can reach it; the offline dataset recorders
servo the 20 independent actions toward it, per joint and by name, so that grasp and reset datasets
remain reproducible.

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

The reference-style direct port registers
`UWLab-UR10eDelto-Grasp-Direct-v0` and
`UWLab-UR10eDelto-TableLeg-Grasp-Direct-v0`. See the concise
[direct-task guide](delto_reference_grasp.md) for its MDP and RL-Games commands.

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

OmniReset reset and grasp datasets are runtime artifacts and are intentionally not committed — so
the two commands above are the only thing that makes them exist. Both recorders build their gripper
command from the environment's own action terms (`omnireset.mdp.scripted_gripper`), writing all 20
hand dimensions; neither writes a closure into `actions[:, -1]`, which on this hand is `rj_dg_5_4`
alone and, at -1, commands extension rather than closure.
