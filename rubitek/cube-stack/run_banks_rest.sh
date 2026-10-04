#!/usr/bin/env bash
# The three banks that still need recording. C1 already exists at 4000 states and partial
# assemblies at ~256 offsets, so neither is re-run.
set -uo pipefail
RUN=/home/mattephi/Work/Rubitek/cube-stack/run.sh
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
LOGS=/home/mattephi/Work/Rubitek/cube-stack/logs
DS=./Datasets_ur10e_delto/OmniReset
PAIR=InsertiveCube34__ReceptiveCube34
VARIANTS="env.scene.insertive_object=cube34 env.scene.receptive_object=cube34"
STACK=2147483648
NUM_ENVS=${NUM_ENVS:-1024}
NUM_STATES=${NUM_STATES:-2000}
cd "$REPO" || exit 1

free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "${free_mb:-0}" -lt 8000 ] && { echo "[ABORT] only ${free_mb} MiB VRAM free"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; exit 1; }
echo "VRAM free: ${free_mb} MiB | NUM_ENVS=$NUM_ENVS NUM_STATES=$NUM_STATES"

for spec in "C3 ObjectAnywhereEEGrasped" "C2 ObjectRestingEEGrasped" "C4 ObjectPartiallyAssembledEEGrasped"; do
  tag=${spec%% *}; rt=${spec##* }
  out="$DS/Resets/$PAIR/resets_${rt}.pt"
  t0=$(date +%s)
  echo; echo "=== [$(date +%H:%M:%S)] $tag $rt ==="
  rm -f "$out"                       # never let a stale artefact be read as this run's output
  "$RUN" scripts_v2/tools/record_reset_states.py \
    --task "OmniReset-UR10eDelto-CubeStack-${rt}-v0" --reset_type "$rt" \
    --num_envs "$NUM_ENVS" --num_reset_states "$NUM_STATES" --headless \
    --dataset_dir "$DS" $VARIANTS \
    env.sim.physx.gpu_collision_stack_size=$STACK > "$LOGS/banks_${tag}_${rt}.log" 2>&1
  if [ -f "$out" ] && [ "$(stat -c%Y "$out")" -ge "$t0" ] && [ "$(stat -c%s "$out")" -ge 100000 ]; then
    drops=$(grep -ic 'contacts have been dropped' "$LOGS/banks_${tag}_${rt}.log"); drops=${drops:-0}
    if [ "$drops" -gt 0 ]; then echo "[FAIL] $tag $rt: $drops dropped-contact lines"; exit 1; fi
    echo "[PASS] $tag $rt: $(stat -c%s "$out") bytes, 0 contact drops"
  else
    echo "[FAIL] $tag $rt: no fresh artefact"; tail -20 "$LOGS/banks_${tag}_${rt}.log" | tr '\r' '\n' | tail -6; exit 1
  fi
done
echo; echo "=== REMAINING BANKS BUILT ==="; ls -la "$DS/Resets/$PAIR/"
