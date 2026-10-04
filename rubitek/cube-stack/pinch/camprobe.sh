#!/usr/bin/env bash
# Probe camera placements against the REAL scene datum: world z=0 is the table work surface
# (+0.004 in asset terms) and the ground plane sits at z=-0.676. A camera at z~0 is therefore
# level with the tabletop, not with the floor -- which is how the first framing ended up grazing
# the surface and showing the table edge-on with the floor beyond it.
set -uo pipefail
cd /home/mattephi/Work/Rubitek/cube-stack/pinch
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
BANK="$REPO/Datasets_ur10e_delto/OmniReset_hold/Resets/InsertiveCube34__ReceptiveCube34/resets_ObjectAnywhereEEGrasped.pt"
probe () {
  local tag="$1" eye="$2" at="$3"
  UWLAB_VIEW_EYE="$eye" UWLAB_VIEW_AT="$at" UWLAB_DELTO_PINCH=0 \
  timeout 900 ./run6.sh "$REPO/scripts_v2/tools/render_reset_states_viewport.py" \
    --reset-states "$BANK" \
    --task OmniReset-UR10eDelto-CubeStack-ObjectAnywhereEEGrasped-v0 \
    --dataset-dir ./Datasets_ur10e_delto/OmniReset_hold --insertive-variant cube34 \
    --receptive-variant cube34 --out "out/films/cam_$tag" --count 2 --settle-steps 5 --repeats 1 \
    > "out/cam_$tag.log" 2>&1
  echo "cam_$tag rc=$? eye=$eye at=$at"
}
probe A "2.00,-2.00,1.20" "0.45,0.00,0.05"
probe B "1.70,0.00,0.95"  "0.45,0.00,0.05"
probe C "0.95,-1.35,0.65" "0.50,0.00,0.10"
echo CAMPROBE_DONE
