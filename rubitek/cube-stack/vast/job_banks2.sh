#!/usr/bin/env bash
# Record the three grasp-dependent reset banks on the rented A6000.
#
# WHY THESE THREE AND NOT ALL FIVE. C1 (reaching) and the partial assemblies were recorded locally
# and shipped with the repo -- C1 accepts at a high rate and took ~2 minutes. The three below all
# require the hand to actually HOLD the cube, and that is where acceptance collapses: measured
# locally at roughly 1 state per 25 s with 1024 envs. This branch's own record puts the analogous
# leg-pair number at 0.83% (40/4799), so the low rate is inherent to the pipeline, not a bug here.
#
# Acceptance-bound work scales linearly with parallel envs, which is the entire reason this box was
# rented: 46 GB against the dev card's 16 GB.
set -uo pipefail
WS=/workspace
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
cd "$WS/UWLab" || exit 1

PY=$WS/venv/bin/python
DS=./Datasets_ur10e_delto/OmniReset
PAIR=InsertiveCube34__ReceptiveCube34
VARIANTS="env.scene.insertive_object=cube34 env.scene.receptive_object=cube34"
ENVS=${ENVS:-2048}
STATES=${STATES:-600}
# The full 3.75 GiB _apply_delto_collision_stack_size pins is affordable here -- this card has the
# headroom the dev card did not. Left at the config default deliberately: an undersized stack does
# not crash, PhysX just logs "Contacts have been dropped" and silently continues, so the bank would
# be quietly wrong rather than absent.

echo "=== banks on $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "ENVS=$ENVS STATES=$STATES"

for spec in "C4 ObjectPartiallyAssembledEEGrasped" "C3 ObjectAnywhereEEGrasped"; do
  tag=${spec%% *}; rt=${spec##* }
  out="$DS/Resets/$PAIR/resets_${rt}.pt"
  t0=$(date +%s)
  echo; echo "=== [$(date -Is)] $tag $rt ==="
  rm -f "$out"
  "$PY" scripts_v2/tools/record_reset_states.py \
    --task "OmniReset-UR10eDelto-CubeStack-${rt}-v0" --reset_type "$rt" \
    --num_envs "$ENVS" --num_reset_states "$STATES" --headless \
    --dataset_dir "$DS" $VARIANTS 2>&1 | tr '\r' '\n' | grep -vE '^\[[0-9]|Warning|warning' | tail -400
  # NOT `| tail -40` on the raw stream: a pipe to tail buffers everything until the process exits,
  # so a multi-hour recording shows no progress at all while it runs. Converting CR to LF first
  # makes tqdm's in-place progress line land in the log as ordinary lines.

  # Positive evidence only: fresh file, non-trivial size, and no silently dropped contacts.
  if [ -f "$out" ] && [ "$(stat -c%Y "$out")" -ge "$t0" ] && [ "$(stat -c%s "$out")" -ge 100000 ]; then
    echo "[PASS] $tag $rt: $(stat -c%s "$out") bytes"
  else
    echo "[FAIL] $tag $rt: no fresh artefact of usable size"
  fi
done

echo; echo "=== bank job done $(date -Is) ==="
ls -la "$DS/Resets/$PAIR/"
