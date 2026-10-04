#!/usr/bin/env bash
# Final reset-bank build for the three grasp-dependent families.
#
# TASK_SUFFIX selects the actuator arm:
#   ""          stock hand actuator (as Ivan's configs ship)
#   "-ThumbFix" hand actuator matched to the one the grasp bank was validated against
# Set from the A/B in results/thumb_stiffness.md -- do not guess it.
#
# Only C3 has a -ThumbFix task registered (it is the A/B probe). If the fix wins, C2 and C4 need
# their own ThumbFix classes before this script can apply it to them; until then they run stock and
# that is stated in the log rather than silently assumed.
set -uo pipefail
WS=/workspace
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
cd "$WS/UWLab" || exit 1

PY=$WS/venv/bin/python
DS=./Datasets_ur10e_delto/OmniReset
PAIR=InsertiveCube34__ReceptiveCube34
VARIANTS="env.scene.insertive_object=cube34 env.scene.receptive_object=cube34"
ENVS=${ENVS:-512}
STATES=${STATES:-300}
SUFFIX=${TASK_SUFFIX:-}

echo "=== final banks: ENVS=$ENVS STATES=$STATES TASK_SUFFIX='${SUFFIX:-<stock>}' ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for spec in "C3 ObjectAnywhereEEGrasped" "C2 ObjectRestingEEGrasped"; do
  tag=${spec%% *}; rt=${spec##* }
  task="OmniReset-UR10eDelto-CubeStack-${rt}${SUFFIX}-v0"
  # Fall back to stock if this family has no ThumbFix variant registered, and SAY SO.
  if [ -n "$SUFFIX" ] && ! "$PY" - <<PYEOF >/dev/null 2>&1
import gymnasium, uwlab_tasks
assert "$task" in gymnasium.registry
PYEOF
  then
    echo "[note] $task is not registered; falling back to the stock task for $tag"
    task="OmniReset-UR10eDelto-CubeStack-${rt}-v0"
  fi

  out="$DS/Resets/$PAIR/resets_${rt}.pt"
  t0=$(date +%s)
  echo; echo "=== [$(date -Is)] $tag $rt  task=$task ==="
  rm -f "$out"
  "$PY" scripts_v2/tools/record_reset_states.py \
    --task "$task" --reset_type "$rt" \
    --num_envs "$ENVS" --num_reset_states "$STATES" --headless \
    --dataset_dir "$DS" $VARIANTS 2>&1 | tr '\r' '\n' | grep -vE '^\[[0-9]|Warning|warning' | tail -300

  if [ -f "$out" ] && [ "$(stat -c%Y "$out")" -ge "$t0" ]; then
    n=$("$PY" - "$out" <<'PYEOF'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
init = d.get("initial_state", d)
n = "?"
for g in ("articulation", "rigid_object"):
    if isinstance(init, dict) and g in init:
        for _, e in init[g].items():
            for _, v in e.items():
                n = len(v) if isinstance(v, list) else int(v.shape[0]); break
            break
        break
print(n)
PYEOF
)
    echo "[PASS] $tag $rt: $n states ($(stat -c%s "$out") bytes)"
  else
    echo "[FAIL] $tag $rt: no fresh artefact"
  fi
done

echo; echo "=== final banks done $(date -Is) ==="
ls -la "$DS/Resets/$PAIR/"
