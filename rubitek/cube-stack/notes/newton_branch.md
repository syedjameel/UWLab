# Newton backend branch — status

Branch: `cube-stack/delto-newton`, off `cube-stack/delto-isaacsim6` (the PhysX port).
Select the backend with `UWLAB_PHYSICS_BACKEND=physx|newton` (default `physx`).

This is an EXPLORATION branch, not a recommendation. Isaac Lab's own docs call the
Newton integration experimental and say support is not expected until official release.

## What works

    num_envs=4, 10 steps, action dim 26

    task                                  physx  newton
    Delto-GraspSampling                   OK     OK
    C1 ObjectAnywhereEEAnywhere           OK     OK
    C2 ObjectRestingEEGrasped             OK     OK
    C3 ObjectAnywhereEEGrasped            OK     OK
    C4 ObjectPartiallyAssembledEEGrasped  OK     OK

Confirmed genuinely Newton, not a silent fallback: `isaaclab_newton.physics.newton_manager`
is the live manager and there are zero `physx_manager` lines in the run.

## The two breaks, now fixed

    Delto-GraspSampling   AttributeError: 'list' object has no attribute 'set_gravity'
    C4 Near-Goal          TypeError: write_root_link_pose_to_sim_index() takes 1 positional arg

Neither was a physics problem.

**Gravity.** The sampler toggles gravity mid-episode (off while the grasp is established,
on at t=10 s, mass-scaled shake at t=11-13 s) through the PhysX tensor API. On Newton
`physics_sim_view` is a list. `uwlab.sim.physics_backend.set_gravity` now routes to
`NewtonManager._model.set_gravity` or the PhysX view, whichever is live.

**C4.** An UPSTREAM Isaac Lab bug, not a UWLab one: the Newton `RigidObject`'s deprecated
`write_root_state_to_sim` / `write_root_link_state_to_sim` wrappers call
`write_root_{link,com}_{pose,velocity}_to_sim_index` positionally, but every one of those is
keyword-only. Six sites fixed on the `uwlab/isaacsim6-port` branch of the Isaac Lab worktree.

## The trap this branch exists to prevent

`NewtonCfg` is a configclass that **tolerates unknown attributes**. Every one of this repo's
~73 PhysX tuning assignments — `solver_type`, `max_position_iteration_count`,
`gpu_collision_stack_size`, `friction_offset_threshold`, `bounce_threshold_velocity`, the
`gpu_*` capacities — will set an attribute that the Newton manager never reads. No error, no
warning. A whole campaign of solver tuning evaporates and the run still starts.

`source/uwlab/uwlab/sim/physics_backend.py` exists so that cannot happen quietly:
`apply_physx_tuning()` applies on PhysX, and on Newton records the setting and reports it,
with a `KeyError` on any field name that is not a known PhysX-only key (so a typo cannot be
written onto a config nobody reads).

**Only `gpu_collision_stack_size` is routed through it so far**, as the demonstration and
because it is the most consequential. The other ~72 sites are still direct assignments and
would silently vanish on Newton. Wiring them is mechanical follow-up work and MUST be done
before any Newton number is believed.

## CUDA graph capture — fixed

Newton was falling back to EAGER execution:

    libcudart not available; cannot use relaxed graph capture
    Newton deferred CUDA graph capture failed; using eager execution

Also upstream. `newton_manager.py` probes only `libcudart.so.12` and the unversioned
`libcudart.so`, but Isaac Sim 6 / torch cu130 ship CUDA **13**, installed as
`site-packages/nvidia/cu13/lib/libcudart.so.13` with no unversioned symlink and that
directory not on the loader path. Both probes fail, capture is disabled, and it surfaces
only as a warning while costing a large part of Newton's speed advantage.

Patched to try `.13` first, then the older names, then locate the pip-installed runtime via
`sysconfig`. The run now logs:

    Newton CUDA graph captured (deferred relaxed mode, RTX-compatible)

No extra CUDA install was needed — the runtime was already present as a pip package
(`nvidia-cuda-runtime 13.0.96`), just unfindable by the name Newton looked for.

## What is NOT established

Nothing about physics fidelity. Isaac Lab's docs: *"Behaviour on stiff contact stacks can
diverge from PhysX; expect to retune contact and substep parameters when porting tasks across
backends."* A 20-DOF hand pinching a 34 mm cube is a stiff contact stack, and every calibrated
number in this repo was fitted in PhysX — the 24.67-41.48 mm grasp window, the 0.06 N*m distal
effort limits, `grasp_center_offset` (fitted by PhysX replay), the 5.0% grasp yield. None of it
transfers unexamined.

Also note the PhysX branch currently has an OPEN regression (the hand does not close, grasp
yield 0 vs 5.0% on Isaac Sim 5.1). Until that is understood there is no working baseline to
compare Newton against, so a Newton grasp number measured today would be uninterpretable.
