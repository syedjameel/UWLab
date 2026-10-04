# DG-5F hand: what the integration actually constrains
All paths relative to the worktree `~/Work/Rubitek/UWLab-cubestack` (branch `cube-stack/delto`
off `origin/dexreset/v2-resets`).

## Finger topology — the single most useful fact
`rj_dg_1_*` AND `rj_dg_5_*` are BOTH thumb-side. The DG-5F opposes {1,5} against {2,3,4}.
  - `generate_reset_states_policy.py:1603` `thumb_contact_names = ("rl_dg_1_tip", "rl_dg_5_tip")`
  - held-check requires >=1 thumb-side tip AND >=1 of 2/3/4 (`:1249-1250`)
  - open posture: fingers 2-5 all have `_2 = 0.9`, finger 1 has `_2 = -0.3736` -> folded the other way
**But the grasp geometry is a two-finger jaw, 1 vs 2:**
  - `DeltoHand/metadata.yaml:8` "a virtual jaw of finger 1 against finger 2"
  - `grasp_center_offset = midpoint(rl_dg_1_tip, rl_dg_2_tip)` (`metadata.yaml:55-59,64`)
  - the 5-tip centroid was explicitly REJECTED because finger 5 sits at y = -68 mm (`:58`)

## Validated grasp window
**24.67 .. 41.48 mm** (`Props/Custom/DeltoBlock/metadata.yaml:6-7`) — full-body grasp window of the
close posture. Separately `maximum_aperture = 0.059` (`DeltoHand/metadata.yaml:34`); a face pair
wider than that is skipped outright by `_sample_topdown_grasps` (`events.py:512-513`).
=> **A cube for this hand must have edge ~25-41 mm.** `deltoblock` is 34 mm and sits mid-window.
The paper's `InsertiveCube` is sized for a Robotiq 2F-85 and its edge length is unverified.

## The full-actuation guard — `source/uwlab/uwlab/envs/mdp/full_actuation.py`
Rule (`:178-230`): every name in `required_joints` must be driven by a term whose
`action_dim == len(joint_names)`. It NEVER counts to 20 itself. What pins it at 20 is the
ARGUMENT `DELTO_HAND_JOINT_NAMES` passed at `delto_cfg.py:268`.
Fires at config construction AND as a startup event (`events.py:1138-1139`).
Note: a joint simply DROPPED from the action also fails (`:204` "NO action term names it") —
subsetting must be an explicit edit, not an omission.

## Finger reduction: verdict = VIABLE, and the right subset is {1,2}
The guard is a 4-line edit. The real cost is elsewhere. Edits required, in lockstep:
  1. `delto_cfg.py:93`  `_DELTO_HAND_JOINTS` -> `[r"rj_dg_[1-2]_[1-4]"]`
  2. `delto_cfg.py:268` narrow the `DELTO_HAND_JOINT_NAMES` argument to match
  3. `actions.py:58,71,76` narrow `joint_names`, `scale`, `clip` TOGETHER
     (a matched joint missing from `scale` silently falls back to scale 1.0 = 10x stroke)
  4. `delto_grasp_env_cfg.py:297,312-317` `action_space: int = 26` and its assert
Action dim goes **26 -> 14** (6 OSC + 8 hand). Paper's own robot is 7 (6 + binary gripper).

Does NOT break: the actuator regex, observations (`joint_pos` is over the whole articulation,
`rl_state_cfg.py:412`), `scripted_gripper._per_joint_command` (name-keyed), grasp replay
(`events.py:1370-1383` only raises if the dataset names NONE of the resolved joints).

**The real cost, and why {1,2} is the only sane subset:**
Frozen fingers hold `DELTO_HAND_DEFAULT_JOINT_POS`, which is the OPEN SPLAY — fingers 3/4/5 at
`_2 = 0.9` stick into the workspace. Changing that posture normally invalidates
`grasp_center_offset`, `gripper_approach_direction`, `grasp_align_axis`, `maximum_aperture` and
every existing grasp/reset dataset (`DeltoHand/metadata.yaml:29-30,58-63`).
**But we are building a NEW object pair from scratch — we have no cube datasets to invalidate.**
And because the grasp geometry is defined by tips 1 and 2 ONLY (never the 5-tip centroid), curling
fingers 3/4/5 away leaves `grasp_center_offset` meaningful. That is exactly the subset the agent
report flags as "the one subset choice that leaves this number meaningful".
Residual risk to verify empirically in Isaac: self-collision (`enabled_self_collisions=True`,
`ur10e_delto.py:185`) at the curled posture, and that the curl does not change the approach axis.
Precedent for getting this wrong: a stale `grasp_center_offset` cost **6144 attempts -> 0 successes**
with all 25 hand bodies reading 0.000 N for entire 18 s episodes (`metadata.yaml:46-53`).

## Landmines found (independent of our change)
- `Ur10eDelto/metadata.yaml:43` still carries the STALE pre-2026-08-17 `gripper_offset.pos`
  `[0.07445, 0.05800, 0.14970]`; `DeltoHand/metadata.yaml:64` has the corrected
  `[0.0637, 0.0465, 0.1500]`. The file's own comment says they must match. They do not.
  This key feeds `rewards.ee_asset_distance_tanh` and the RL-state TCP binding. **Fix before training.**
- `delto_cfg.py:198` and `ur10e_delto_cfg.py:81` cite a stale 7.31 rad/s velocity limit; the real
  run-time cap is `velocity_limit_sim = 3.0` (`ur10e_delto.py:150`).
- `_apply_delto_collision_stack_size` sets `gpu_collision_stack_size = 4026531840` (3.75 GiB, max
  legal; exactly 4 GiB dies at `gym.make`). Undersized does NOT crash — PhysX logs
  "Contacts have been dropped" and silently continues. **Must be called by any new DELTO class.**
- `compute_pair_dir` (`mdp/utils.py:391-400`) keys datasets by OBJECT PAIR ONLY — no gripper or arm
  discriminator. `_apply_delto_dataset_dir` (`delto_cfg.py:363-393`) exists to repoint them.
  A new pair must go through it or it will read a Robotiq dataset into a 26-DOF articulation.
- `terminations.py:274-276` reads `gripper_approach_direction` from the FULL ROBOT's usd dir with
  no default; `rewards.py ee_asset_distance_tanh` reads `gripper_offset` the same way. Both keys
  must exist in `Ur10eDelto/metadata.yaml`, not only in `DeltoHand/metadata.yaml`.

## What a NEW object needs (from `new_task.rst` + the report)
1. USD + adjacent `metadata.yaml` with `assembled_offset`, `bottom_offset`, and — if receptive —
   `success_thresholds` (a receptive object REQUIRES it; `commands.TaskCommand` reads
   position/orientation with no default).
2. A `variants` entry in FOUR files: `partial_assemblies_cfg.py`, `grasp_sampling_cfg.py`,
   `reset_states_cfg.py`, `rl_state_cfg.py`.
3. `override_mass=False` if the USD authors a root MassAPI (default forces 0.001 kg — at 1 g the
   shake test passes grasps the real part would drop).
4. Gripped extents inside 24.67-41.48 mm; gripped face width <= 59 mm.
5. Object modelled UPRIGHT (topdown sampler assumes Z vertical, `events.py:503-505`).

## Grasp sampler facts
`OmniReset-Delto-GraspSampling-v0` -> `DeltoGraspSamplingCfg` (`delto_cfg.py:399-637`).
Hand only, no arm (`DELTO_HAND`, fixed root). `num_orientations = 16`. `episode_length_s = 18.0`.
Gravity on at t=10 s; mass-scaled shake at t=11-13 s, `force_torque_magnitude = 5.0`.
Writes `<dataset_dir>/Grasps/<obj>/grasps.pt`, succeeded-only.
Four Hydra-tunable knobs exist BECAUSE `update_class_from_dict` raises KeyError on any params key
absent from the default cfg: `grasp_topdown_{roll_pitch_deg,position_fraction,height_fraction,yaw_deg}`.
`UWLAB_GRASP_DEBUG=1` prints the per-condition rejection breakdown.

## Rendering tooling that already exists
`scripts_v2/tools/delto/render_delto_grasps_raw.py` (725 lines) — grasps or reset-states -> PNG
contact sheet. Uses TWO FREE UNPARENTED cameras with world poses computed BEFORE Isaac launches:
`TiledCamera.set_world_poses_from_view` is a **silent no-op** in this build, and a camera parented
under `Robot/rl_dg_mount` **crashes natively with no Python traceback** at first `env.reset()`.
Farthest-point sampling over (position, orientation) for selection.
