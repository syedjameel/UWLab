# Port to Isaac Sim 6.0.1 + Isaac Lab 3.0.0-beta2 (driver 610)

## Why
Isaac Sim 5.1 segfaults in `librtx.scenedb.plugin.so` at kit bootstrap on NVIDIA driver
610.57.04. Reproduced with renderer flags off, NVIDIA-only Vulkan ICD, and the RTX extensions
stripped from a custom `.kit`; the kit kernel itself boots fine (`omni.app.empty.kit` is clean),
so it is specifically the RTX renderer.

NVIDIA confirm this and give the fix (IsaacSim discussion #689, maintainer PeterL-NV):
*"upgrading to Isaac Sim 6.0 alongside the release/3.0.0-beta2 branch of Isaac Lab should resolve
your driver compatibility issue."* Isaac Sim 5.1 + Isaac Lab 2.3.2 is capped around driver 591.86.
Downgrading the driver is NOT viable here: 580-era modules do not build against kernel 7.2.2
(the same catch-22 reported in IsaacSim issue #651).

Isaac Sim 5.1's own compatibility check declares only `minimum = "535.161"` with an empty
`unsupported` list, so nothing in the shipped metadata predicts this. It is untested
forward-incompatibility, not a declared limit.

## Environment
    /home/mattephi/Work/Rubitek/uwlab-env/.venv-is6   python 3.12, isaacsim 6.0.1.0, torch 2.11+cu130
    /home/mattephi/Work/Rubitek/_isaaclab3            IsaacLab worktree, detached at release/3.0.0-beta2

Install Isaac Sim with **pip, not uv** — uv deadlocks on the 5.9 GB `isaacsim-extscache-kit`
wheel (zero open sockets, zero I/O, 0% CPU). Point `TMPDIR` at disk: pip stages every wheel
in `/tmp`, which is a 16 GB tmpfs here, and several multi-GB wheels will exhaust it.

## Fixes, in the order they were hit

1. **`isaacsim.core.utils` not on the ext path.** Still ships, under `extsDeprecated`. Register
   that folder and enable the ext — see `uwlab-env/compat/uwlab_is6_bridge.py`.
2. **`AdditiveUniformNoiseCfg` / `AdditiveGaussianNoiseCfg` removed** (they were deprecated
   aliases). Renamed to `UniformNoiseCfg` / `GaussianNoiseCfg` across 11 UWLab files.
3. **`pybullet` missing** — a declared UWLab dependency, skipped by `--no-deps`.
4. **`pytorch3d` unavailable.** No wheel for torch 2.11 / cu130 / cp312 and a source build needs
   nvcc, which this box lacks. Replaced with a pure-torch shim in `uwlab-env/compat/pytorch3d/`
   covering the only three symbols used: `Meshes`, `sample_points_from_meshes`,
   `sample_farthest_points`. Self-tested: sampled points stay inside the source mesh, and FPS
   gives ~6x the minimum pairwise spacing of a random subset (0.230 vs 0.038 on a unit cube).
   **This is a reimplementation, not upstream pytorch3d.**
5. **`SimulationCfg.physx` -> `.physics`** (a `PhysxCfg`). All 13 fields UWLab sets carry over
   unchanged; renamed 73 sites. Upstream leaves `physics=None` and lets `SimulationContext`
   substitute `PhysxCfg()` at init (`simulation_context.py:166-169`), but env cfgs tune PhysX in
   `__post_init__`, which runs first — so the worktree makes the default eager instead.
6. **`WrenchComposer` type check evaluated the property.** `hasattr(self._asset.data, ...)` is a
   type test (its own comment says so) but reads through to the PhysX backend, which raises a
   plain `Exception` during the first initialize callback — and `hasattr` only swallows
   `AttributeError`. Changed to `hasattr(type(self._asset.data), ...)`.
7. **THE BIG ONE — a 3.75 GiB PhysX collision stack that no longer fits.**
   `_apply_delto_collision_stack_size` pins `gpu_collision_stack_size = 4026531840`. Under
   Isaac Sim 6 on this 16 GB card that allocation FAILS:
       PxgCudaDeviceMemoryAllocator failed to allocate memory 4026531840 bytes
   after which every PhysX GPU kernel fails and the views return nothing. It surfaces as
   `Failed to get DOF velocities from backend` — an error that looks like an API break and is
   not one. Now `UWLAB_GPU_COLLISION_STACK_SIZE`, default 2 GiB.
   **Undersizing does not crash either** — PhysX logs "Contacts have been dropped" and continues
   with degraded contact reporting, which a grasping task must not have. Grep every run for it;
   the current runs show zero.
8. **Asset data is a warp `ProxyArray`, not a Tensor.** Isaac Lab's own `@torch.jit.script` math
   helpers reject it. The bridge rebinds all 33 scripted functions in `isaaclab.utils.math` with
   coercing wrappers; every UWLab call site in the OmniReset path uses the module form, so this
   reaches all of them. Separately `ProxyArray.device` is a *warp* Device that
   `torch.Tensor.to()` rejects — fixed at the one site that hit it
   (`assembly_keypoints.py:apply`) rather than by patching the property, since Isaac Lab
   internals rely on it.

## Status
Verified constructing, resetting and stepping 10 steps at `num_envs=4`, action dim (4, 26):

    OmniReset-Delto-GraspSampling-v0                                OK
    OmniReset-UR10eDelto-CubeStack-ObjectAnywhereEEAnywhere-v0      OK   (C1 Reaching)
    OmniReset-UR10eDelto-CubeStack-ObjectRestingEEGrasped-v0        OK   (C2 Near-Object)
    OmniReset-UR10eDelto-CubeStack-ObjectAnywhereEEGrasped-v0       OK   (C3 Stable Grasp)
    OmniReset-UR10eDelto-CubeStack-ObjectPartiallyAssembledEEGrasped-v0  OK (C4 Near-Goal)

RTX rendering verified separately: a real RGB frame with geometry and shadows.

## Not yet verified
Training, the RL-state task, and whether physics BEHAVIOUR matches Isaac Sim 5.1. A port that
constructs is not a port that reproduces — contact-rich grasping is exactly where PhysX version
differences show up, so the grasp-yield numbers must be re-measured, not carried over.

## OPEN PROBLEM — the hand does not close under Isaac Sim 6

The port CONSTRUCTS but does not yet REPRODUCE. Two independent measurements agree:

1. **Grasp sampler yield is zero.** `record_grasps.py` on `cube34`, 256 envs, `UWLAB_GRASP_DEBUG=1`,
   every batch identical:

       not_abnormal=256/256  stable=256  not_far=0  above_ground=0  coll_free=256 -> success=0

   `not_far=0` and `above_ground=0` together mean the cube leaves the hand and falls, in every
   environment. Baseline on Isaac Sim 5.1 for this same object was 51/1024 = 5.0%.

2. **The fingers do not reach their commanded posture.** Servoing the hand to
   `DELTO_HAND_SCRIPTED_CLOSE_JOINT_POS` leaves `rj_dg_1_4` **1.31 rad** short, and the fingertips
   settle 44-48 mm from a cube placed at the jaw centre. A 34 mm cube needs a tip within ~17-25 mm
   to be touched at all, so the hand closes on air.

Ruled out so far:
- `ActuatorBaseCfg` semantics (`effort_limit_sim`, `velocity_limit_sim`, `friction`) are
  byte-identical between IsaacLab 2.3.2 and 3.0.0-beta2.
- Not the collision-stack allocation: zero "failed to allocate" and zero "Contacts have been
  dropped" in these runs.

The suspicion is the hand actuator's very small limits interacting with a PhysX/solver change:
the distal `_4` joints are capped at **0.06 N*m** effort against **0.01 N*m** joint friction, so
very little changes the reachable posture. Next step is to measure applied joint torque vs
commanded target on `rj_dg_1_4` under both simulators, not to re-tune gains blindly.

Until this is closed, no grasp bank, no C2/C3/C4 reset banks and no pinch variant can be
generated: all three depend on the hand actually holding the cube.
