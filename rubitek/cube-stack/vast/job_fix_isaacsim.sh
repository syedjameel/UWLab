#!/usr/bin/env bash
# REPAIR the provisioning of a box whose Isaac Sim install failed with uv's default index strategy:
#   isaacsim-core 6.0.1.0 depends on mujoco-usd-converter==0.2.0, published on pypi.org but not on
#   pypi.nvidia.com -> "No solution found when resolving dependencies".
# Re-runs the whole Python side of provisioning in the right order and leaves /workspace/ISAACSIM_OK.
set -uo pipefail
WS=/workspace
PY="$WS/venv/bin/python"
export PATH="$HOME/.local/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
say(){ echo "FIX $* $(date -Is)"; }

say "isaacsim 6.0.1.0 (unsafe-best-match)"
uv pip install --python "$PY" --index-strategy unsafe-best-match \
  "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com || { say "ISAACSIM_FAILED"; exit 1; }

say "IsaacLab 3 editable installs"
for d in isaaclab isaaclab_physx isaaclab_newton isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_ppisp; do
  [ -d "$WS/_isaaclab3/source/$d" ] && uv pip install --python "$PY" --index-strategy unsafe-best-match \
    -e "$WS/_isaaclab3/source/$d" --extra-index-url https://pypi.nvidia.com
done

say "UWLab source packages"
for d in "$WS"/UWLab/source/*/; do
  [ -f "$d/setup.py" ] && uv pip install --python "$PY" -e "$d"
done

say "rsl_rl UW-Lab fork (gSDE) + import-time deps"
uv pip install --python "$PY" --no-deps --force-reinstall \
  "rsl-rl-lib @ git+https://github.com/UW-Lab/rsl_rl.git@main"
uv pip install --python "$PY" GitPython tensordict onnx tensorboard

say "re-pin torch 2.11.0+cu130"
uv pip install --python "$PY" torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

"$PY" - <<'PYEOF'
import inspect, torch, isaacsim
import rsl_rl.modules.actor_critic as m
print("isaacsim import OK | torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "| gsde:", "gsde" in inspect.getsource(m))
PYEOF
rc=$?
[ $rc -eq 0 ] && touch "$WS/ISAACSIM_OK"
say "DONE rc=$rc"
