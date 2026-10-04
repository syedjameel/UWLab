#!/usr/bin/env bash
# C4 (near-goal) locally at 512 envs. 512 beats 2048 measurably on this workload: the A/B arms hit
# 23-31 states per 420 s at 512, while the rented box managed ~1.2 states/min at 2048. Acceptance is
# per-episode-per-env, but per-step cost grows with env count, and here the second effect dominates.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
DS=./Datasets_ur10e_delto/OmniReset
out="$DS/Resets/InsertiveCube34__ReceptiveCube34/resets_ObjectPartiallyAssembledEEGrasped.pt"
cd "$REPO" || exit 1
rm -f "$out"; t0=$(date +%s)
echo "=== [$(date +%H:%M:%S)] C4 local, 512 envs, 300 states, grasp-matched actuator ==="
"$CS/run.sh" scripts_v2/tools/record_reset_states.py \
  --task OmniReset-UR10eDelto-CubeStack-ObjectPartiallyAssembledEEGrasped-v0 \
  --reset_type ObjectPartiallyAssembledEEGrasped \
  --num_envs 512 --num_reset_states 300 --headless \
  --dataset_dir "$DS" env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 \
  env.sim.physx.gpu_collision_stack_size=1073741824 \
  > "$CS/logs/banks_C4_local.log" 2>&1
if [ -f "$out" ] && [ "$(stat -c%Y "$out")" -ge "$t0" ]; then
  /home/mattephi/Work/Rubitek/uwlab-env/.venv/bin/python "$CS/count_bank.py" "$out"
else
  echo "[FAIL] C4"; tail -14 "$CS/logs/banks_C4_local.log" | tr '\r' '\n' | tail -6
fi
echo "=== C4 local done $(date +%H:%M:%S) ==="
