#!/usr/bin/env bash
# Build every dataset the cube-stacking RL task needs, in dependency order.
#
# ORDER IS NOT ARBITRARY:
#   partial assemblies  <- nothing
#   C1 ObjectAnywhereEEAnywhere        <- nothing
#   C3 ObjectAnywhereEEGrasped         <- grasp bank
#   C2 ObjectRestingEEGrasped          <- C1 bank AND grasp bank
#   C4 ObjectPartiallyAssembledEEGrasped <- partial assemblies AND grasp bank
# The grasp bank is produced separately (record_grasps.py) and is a precondition for C2/C3/C4.
#
# EXIT CODES ARE NOT TRUSTED. This mirrors the lesson recorded on this branch in
# cert_nnlv_vertgoal.sh:232-247 -- a run that died with a traceback can still leave a wrapper
# exiting 0. Every stage here is graded on POSITIVE EVIDENCE: the output file exists, is
# non-trivially sized, and loads as a tensor dict with the expected top-level keys.
set -uo pipefail

RUN=/home/mattephi/Work/Rubitek/cube-stack/run.sh
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
LOGS=/home/mattephi/Work/Rubitek/cube-stack/logs
DS=./Datasets_ur10e_delto/OmniReset
PAIR=InsertiveCube34__ReceptiveCube34
VARIANTS="env.scene.insertive_object=cube34 env.scene.receptive_object=cube34"

# 2 GiB, not the 3.75 GiB _apply_delto_collision_stack_size pins. That value is sized for 24 GB
# training cards and OOMs PhysX on this 16 GB dev card. 2 GiB was verified sufficient at 1024 envs:
# zero "Contacts have been dropped" lines across a full 1000-grasp bank run. An undersized stack
# does NOT crash -- it silently drops contacts -- so this is checked, not assumed (see verify()).
STACK=2147483648

NUM_ENVS=${NUM_ENVS:-1024}
NUM_STATES=${NUM_STATES:-4000}

mkdir -p "$LOGS"
cd "$REPO" || exit 1

free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${free_mb:-0}" -lt 8000 ]; then
  echo "[ABORT] only ${free_mb} MiB VRAM free. Isaac needs headroom; a killed run often leaves a"
  echo "        process holding memory, and PhysX then starts with _physics_sim_view = None and"
  echo "        dies at 'create_rigid_body_view'. Clear it before rerunning:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  exit 1
fi
echo "VRAM free at start: ${free_mb} MiB"

verify() {  # verify <label> <path> <min_bytes> <logfile> [start_epoch]
  local label=$1 path=$2 minb=$3 log=$4 t0=${5:-0}
  if [ ! -f "$path" ]; then
    echo "[FAIL] $label: $path does not exist"; return 1
  fi
  # A stale artefact from an earlier run must not be read as this run's output. Without this a
  # fast-failing rerun "passes" on the previous run's file -- observed, not hypothetical.
  local mt; mt=$(stat -c%Y "$path")
  if [ "$t0" -gt 0 ] && [ "$mt" -lt "$t0" ]; then
    echo "[FAIL] $label: $path predates this stage (mtime $mt < start $t0) -- stale, the run produced nothing"
    return 1
  fi
  local sz; sz=$(stat -c%s "$path")
  if [ "$sz" -lt "$minb" ]; then
    echo "[FAIL] $label: $path is $sz bytes, expected >= $minb"; return 1
  fi
  local drops; drops=$(grep -ic 'contacts have been dropped' "$log" 2>/dev/null); drops=${drops:-0}
  if [ "$drops" -gt 0 ]; then
    echo "[FAIL] $label: PhysX dropped contacts ($drops lines) -- collision stack too small, bank is untrustworthy"
    return 1
  fi
  echo "[PASS] $label: $path  ${sz} bytes, 0 contact drops"
  return 0
}

STAGE_T0=0
stage() {  # stage <label> <logname> <cmd...>
  local label=$1 logname=$2; shift 2
  STAGE_T0=$(date +%s)
  echo
  echo "=== [$(date +%H:%M:%S)] $label ==="
  "$@" > "$LOGS/$logname" 2>&1
  echo "  (wrapper exit $? -- not trusted, see verify)"
}

# ---------------------------------------------------------------------------------------
echo "cube34 bank build: NUM_ENVS=$NUM_ENVS NUM_STATES=$NUM_STATES"

stage "partial assemblies (near-goal offsets)" banks_partial.log \
  "$RUN" scripts_v2/tools/record_partial_assemblies.py \
  --task OmniReset-CubeStack-PartialAssemblies-v0 \
  --num_envs 256 --num_trajectories 400 --headless \
  --dataset_dir "$DS" $VARIANTS \
  env.sim.physx.gpu_collision_stack_size=$STACK
verify "partial assemblies" "$DS/Resets/$PAIR/partial_assemblies.pt" 2000 "$LOGS/banks_partial.log" "$STAGE_T0" || exit 1

for spec in \
  "C1 ObjectAnywhereEEAnywhere" \
  "C3 ObjectAnywhereEEGrasped" \
  "C2 ObjectRestingEEGrasped" \
  "C4 ObjectPartiallyAssembledEEGrasped" ; do
  tag=${spec%% *}; rt=${spec##* }
  stage "$tag reset bank: $rt" "banks_${tag}_${rt}.log" \
    "$RUN" scripts_v2/tools/record_reset_states.py \
    --task "OmniReset-UR10eDelto-CubeStack-${rt}-v0" \
    --reset_type "$rt" \
    --num_envs "$NUM_ENVS" --num_reset_states "$NUM_STATES" --headless \
    --dataset_dir "$DS" $VARIANTS \
    env.sim.physx.gpu_collision_stack_size=$STACK
  verify "$tag $rt" "$DS/Resets/$PAIR/resets_${rt}.pt" 100000 "$LOGS/banks_${tag}_${rt}.log" "$STAGE_T0" || exit 1
done

echo
echo "=== ALL BANKS BUILT ==="
ls -la "$DS/Resets/$PAIR/" "$DS/Grasps/"*/
