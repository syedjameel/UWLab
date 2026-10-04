#!/usr/bin/env bash
# C2/C3/C4 reset banks for the hold-optimised pinch, WITH the acceptance breakdown on.
#
# The previous attempt ran 30 minutes at 0/40 with no diagnostics, so there was nothing to say
# about why. check_reset_state_success ANDs six conditions; UWLAB_GRASP_DEBUG prints how many
# environments pass each one, which turns "it produced nothing" into a specific blocked term.
set -uo pipefail
cd /home/mattephi/Work/Rubitek/cube-stack/pinch
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
DS=./Datasets_ur10e_delto/OmniReset_hold
PAIR=InsertiveCube34__ReceptiveCube34
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
export UWLAB_DELTO_PINCH=1
export UWLAB_CUBE_DATASET_DIR="$DS"
export UWLAB_GRASP_DEBUG=1
export UWLAB_GRAVITY_COMP=1
log(){ echo "=== $(date -Is) $*" | tee -a out/stage19.log; }

log "grasps: $(ls -la "$REPO/${DS#./}/Grasps/InsertiveCube34/grasps.pt" | awk '{print $5}') bytes"
# C3 first: it consumes the grasp bank directly and is the cheapest of the three, so it is the
# fastest way to find out whether the new posture clears the acceptance gate at all.
for T in ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped ObjectRestingEEGrasped; do
  rm -f "$REPO/${DS#./}/Resets/$PAIR/resets_$T.pt"
  log "$T"
  timeout 9000 ./run6.sh "$REPO/scripts_v2/tools/record_reset_states.py" \
      --task "OmniReset-UR10eDelto-CubeStack-$T-v0" --reset_type "$T" \
      --num_envs 128 --num_reset_states 40 --headless --dataset_dir "$DS" \
      env.scene.insertive_object=cube34 env.scene.receptive_object=cube34 \
      > "out/resets19_$T.log" 2>&1
  log "  rc=$? bytes=$(ls -la "$REPO/${DS#./}/Resets/$PAIR/resets_$T.pt" 2>/dev/null | awk '{print $5}')"
  grep -aoE "reset dbg[^\\n]{0,200}" "out/resets19_$T.log" | tail -3 | tee -a out/stage19.log
done
echo "STAGE19_DONE $(date -Is)" | tee -a out/stage19.log
