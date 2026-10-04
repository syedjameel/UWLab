"""Why is a ManagerTermBase event term still a CLASS at reset time?

IsaacLab defers instantiation of non-'prestartup' class event terms: manager_base.py registers
_resolve_terms_callback on the timeline PLAY event, and _resolve_common_term_cfg only instantiates
inline when sim.is_playing() is already true. If neither path fires, term_cfg.func stays a class and
EventManager.apply() calls it as func(env, env_ids, **params) -- which lands in __init__ and raises
"unexpected keyword argument 'dataset_dir'". This prints which path was taken.
"""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--num_envs", type=int, default=4)
p.add_argument("--collision_stack_mb", type=int, default=1024)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import inspect  # noqa: E402
import gymnasium as gym  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs)
v = cfg.variants
cfg.scene.insertive_object = v["scene.insertive_object"]["cube34"]
cfg.scene.receptive_object = v["scene.receptive_object"]["cube34"]
cfg.sim.physx.gpu_collision_stack_size = a.collision_stack_mb * 1024 * 1024

env = gym.make(a.task, cfg=cfg)
u = env.unwrapped
em = u.event_manager

print("\n=== sim / manager state ===")
print("sim.is_playing()               :", u.sim.is_playing())
print("_is_scene_entities_resolved    :", getattr(em, "_is_scene_entities_resolved", "?"))
print("has _resolve_terms_handle      :", hasattr(em, "_resolve_terms_handle"))

print("\n=== event terms by mode: class (BAD, uninstantiated) vs instance (OK) ===")
for mode, names in em._mode_term_names.items():
    for name, tc in zip(names, em._mode_term_cfgs[mode]):
        kind = "CLASS  <-- not instantiated" if inspect.isclass(tc.func) else "instance"
        if inspect.isclass(tc.func) or "Manager" in type(tc.func).__name__:
            print(f"  [{mode:10}] {name:38} {kind}  ({getattr(tc.func,'__name__',type(tc.func).__name__)})")

print("\n_mode_class_term_cfgs (deferred queue):")
for mode, lst in getattr(em, "_mode_class_term_cfgs", {}).items():
    if lst:
        print(f"  {mode}: {[getattr(t.func,'__name__',type(t.func).__name__) for t in lst]}")

import inspect as _i
tc = None
for mode, names in em._mode_term_names.items():
    for name, t in zip(names, em._mode_term_cfgs[mode]):
        if name == "reset_from_reset_states":
            tc = t
print("\nBEFORE reset(): func is", "CLASS" if _i.isclass(tc.func) else "instance")
try:
    env.reset()
    print("AFTER  reset(): func is", "CLASS" if _i.isclass(tc.func) else "instance")
    print("reset() OK")
except Exception as e:
    print("AFTER  reset(): func is", "CLASS" if _i.isclass(tc.func) else "instance")
    print("reset() RAISED:", type(e).__name__, e)

env.close()
app.close()
