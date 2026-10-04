#!/usr/bin/env bash
# Validate the Play config, then record C2 locally while the rented box does C4 and C3.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
DS=./Datasets_ur10e_delto/OmniReset
PAIR=InsertiveCube34__ReceptiveCube34
cd "$REPO" || exit 1

echo "=== [$(date +%H:%M:%S)] Play-config smoke, cube34 ==="
"$CS/run.sh" "$CS/smoke_cubestack.py" \
  --task OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-Play-v0 \
  --num_envs 8 --steps 20 --collision_stack_mb 1024 --headless > "$CS/logs/smoke_play3.log" 2>&1
grep -E '\[(PASS|FAIL)\]|SMOKE_RESULT|LIVE thresholds|action space|robot joints|dense_success' "$CS/logs/smoke_play3.log"

echo
echo "=== [$(date +%H:%M:%S)] C2 ObjectRestingEEGrasped, local, 600 states ==="
out="$DS/Resets/$PAIR/resets_ObjectRestingEEGrasped.pt"
rm -f "$out"; t0=$(date +%s)
"$CS/run.sh" scripts_v2/tools/record_reset_states.py \
  --task OmniReset-UR10eDelto-CubeStack-ObjectRestingEEGrasped-v0 \
  --reset_type ObjectRestingEEGrasped \
  --num_envs 512 --num_reset_states 600 --headless \
  --dataset_dir "$DS" env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 \
  env.sim.physx.gpu_collision_stack_size=2147483648 \
  > "$CS/logs/banks_C2_local.log" 2>&1

if [ -f "$out" ] && [ "$(stat -c%Y "$out")" -ge "$t0" ]; then
  /home/mattephi/Work/Rubitek/uwlab-env/.venv/bin/python "$CS/count_bank.py" "$out"
else
  echo "[FAIL] C2: no fresh artefact"; tail -12 "$CS/logs/banks_C2_local.log" | tr '\r' '\n' | tail -5
fi
echo "=== local chain done $(date +%H:%M:%S) ==="
