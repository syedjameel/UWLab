#!/usr/bin/env bash
# Grasp bank for the hold-optimised posture.
#
# Short probe first. The closing axis written by applypose.py follows pinchaxes' own convention
# (tip2 - tip1), which is the OPPOSITE of metadata's (tip1 - tip2). For a symmetric cube that
# should not matter, but the placement gate answers it in one minute instead of an argument: a
# wrong-signed axis rotates a pad into a cube face and the gate rejects everything, which is
# exactly the signature seen when the pinch was first wired in (61440/61440 rejected).
set -uo pipefail
cd /home/mattephi/Work/Rubitek/cube-stack/pinch
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
DS=Datasets_ur10e_delto/OmniReset_hold
export UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
export UWLAB_DELTO_PINCH=1
log(){ echo "=== $(date -Is) $*" | tee -a out/stage16.log; }

rm -rf "$REPO/$DS"; mkdir -p "$REPO/$DS"
log "probe: 40 grasps, checking the placement gate"
./run6.sh "$REPO/scripts_v2/tools/record_grasps.py" \
    --task OmniReset-Delto-GraspSampling-v0 --num_envs 64 --num_grasps 40 --headless \
    --dataset_dir "$REPO/$DS" env.scene.object=cube34 > out/grasps_hold_probe.log 2>&1
log "probe rc=$? | $(grep -E 'placement collision gate rejected' out/grasps_hold_probe.log | tail -1)"
grep -E "BATCH_END|gate rejected" out/grasps_hold_probe.log | tail -4 | tee -a out/stage16.log
echo "PROBE_DONE $(date -Is)" | tee -a out/stage16.log
