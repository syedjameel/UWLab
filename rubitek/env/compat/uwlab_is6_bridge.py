"""Isaac Sim 6 / Isaac Lab 3.0 port bridge for UWLab.

Two things Isaac Lab 3.0 changed that UWLab code predates:

1. Asset data (``robot.data.root_link_quat_w`` and friends) is now a warp-backed
   ``ProxyArray`` rather than a ``torch.Tensor``. Isaac Lab's own math helpers are
   ``@torch.jit.script``-compiled and reject anything that is not a real Tensor
   ("Expected a value of type 'Tensor' ... but instead found type 'ProxyArray'").
   Every UWLab call site in the OmniReset path uses the module form
   (``math_utils.foo(...)``), so rebinding the module attributes reaches all of them.

2. The deprecated ``isaacsim.core.utils`` extension is no longer on the default
   extension search path. ``install_deprecated_ext_path`` re-registers it.

Import and call :func:`apply` AFTER the Isaac Lab AppLauncher has started and BEFORE
importing ``uwlab``/``uwlab_tasks``.
"""

from __future__ import annotations

def deprecated_ext_dir() -> str:
    """Locate ``isaacsim/extsDeprecated`` in the CURRENTLY INSTALLED Isaac Sim.

    This was a hardcoded path to the dev box's venv, which made the bridge a silent no-op anywhere
    else: ``add_path`` on a non-existent directory raises nothing, the extensions never enable, and
    the first ``import uwlab`` dies with "No module named 'isaacsim.core.utils'" -- which is how a
    rented box failed every record/train job while the same code worked locally.
    ``UWLAB_IS6_DEPRECATED_EXT_DIR`` overrides it for odd layouts.
    """
    import os

    override = os.environ.get("UWLAB_IS6_DEPRECATED_EXT_DIR")
    if override:
        return override
    import isaacsim

    return os.path.join(os.path.dirname(os.path.abspath(isaacsim.__file__)), "extsDeprecated")


def install_deprecated_ext_path() -> None:
    import os

    import omni.kit.app

    path = deprecated_ext_dir()
    if not os.path.isdir(path):
        raise RuntimeError(
            f"[uwlab_is6_bridge] deprecated extension dir not found at {path!r}; set"
            " UWLAB_IS6_DEPRECATED_EXT_DIR to the isaacsim/extsDeprecated of the active install"
        )
    mgr = omni.kit.app.get_app().get_extension_manager()
    mgr.add_path(path)
    for ext in (
        "isaacsim.core.deprecation_manager",
        "isaacsim.core.utils",
        "isaacsim.core.api",
        "isaacsim.core.prims",
    ):
        mgr.set_extension_enabled_immediate(ext, True)


def patch_math_utils() -> int:
    """Wrap Isaac Lab's jit-scripted math helpers so they accept ProxyArray.

    Returns the number of functions wrapped.
    """
    import torch

    import isaaclab.utils.math as M
    from isaaclab.utils.warp.proxy_array import ProxyArray

    def coerce(x):
        return x.torch if isinstance(x, ProxyArray) else x

    def wrap(fn):
        def wrapper(*args, **kwargs):
            return fn(*(coerce(a) for a in args), **{k: coerce(v) for k, v in kwargs.items()})

        wrapper.__name__ = getattr(fn, "name", "wrapped")
        wrapper.__doc__ = getattr(fn, "__doc__", None)
        wrapper.__wrapped_jit__ = fn
        return wrapper

    n = 0
    for name in dir(M):
        fn = getattr(M, name)
        if isinstance(fn, torch.jit.ScriptFunction):
            setattr(M, name, wrap(fn))
            n += 1
    return n


def apply() -> None:
    install_deprecated_ext_path()
    n = patch_math_utils()
    print(f"[uwlab_is6_bridge] deprecated ext path registered; wrapped {n} jit math fns")
