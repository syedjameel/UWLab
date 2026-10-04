#!/usr/bin/env bash
# A/B: does matching the training hand's actuator to the grasp sampler's raise C3 acceptance?
# Same envs, same seed, same wall-clock budget, only the actuator differing.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
DS=./Datasets_ur10e_delto/OmniReset
cd "$REPO" || exit 1
BUDGET=${BUDGET:-420}   # seconds per arm

run_arm () {  # run_arm <label> <task>
  local label=$1 task=$2
  echo; echo "=== [$(date +%H:%M:%S)] arm: $label ==="
  timeout "$BUDGET" "$CS/run.sh" scripts_v2/tools/record_reset_states.py \
    --task "$task" --reset_type ObjectAnywhereEEGrasped \
    --num_envs 512 --num_reset_states 100000 --headless \
    --dataset_dir /tmp/ab_$label env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 \
    env.sim.physx.gpu_collision_stack_size=1073741824 \
    > "$CS/logs/ab_$label.log" 2>&1
  # tqdm reports "<n>/100000" accepted; take the last one it printed inside the budget.
  local n
  n=$(tr '\r' '\n' < "$CS/logs/ab_$label.log" | grep -oE '[0-9]+/100000' | tail -1 | cut -d/ -f1)
  echo "$label: ${n:-0} states accepted in ${BUDGET}s at 512 envs"
}

rm -rf /tmp/ab_stock /tmp/ab_thumbfix
run_arm stock    OmniReset-UR10eDelto-CubeStack-ObjectAnywhereEEGrasped-v0
run_arm thumbfix OmniReset-UR10eDelto-CubeStack-ObjectAnywhereEEGrasped-ThumbFix-v0
echo; echo "=== A/B done $(date +%H:%M:%S) ==="
