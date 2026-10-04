#!/usr/bin/env bash
# Film all four reset families through the REPO'S OWN renderer, in the recorder's own scene.
#
# Not banklab.py. That was a hand-built duplicate of a scene the repo already had, and it drifted
# from the real one -- the arm ended up under the table and every distance measured in it was a
# distance to the wrong place. This script only drives the repo tool.
set -uo pipefail
cd /home/mattephi/Work/Rubitek/cube-stack/pinch
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
OUT=/home/mattephi/Work/Rubitek/cube-stack/pinch/out/films
mkdir -p "$OUT"
# One fixed camera per run: set_camera_view is a no-op in this build, so cfg.viewer at env
# creation is the only control that works. Framed wide enough to hold arm, table and both cubes.
export UWLAB_VIEW_EYE="${UWLAB_VIEW_EYE:-0.60,-1.05,0.02}"
export UWLAB_VIEW_AT="${UWLAB_VIEW_AT:-0.58,0.00,-0.07}"
export UWLAB_DELTO_PINCH=0
DS="${1:-./Datasets_ur10e_delto/OmniReset}"
TAG="${2:-shipped}"
log(){ echo "=== $(date -Is) $*" | tee -a out/stage18.log; }

run () {  # run <family> <extra args...>
  local T="$1"; shift
  local BANK="$REPO/${DS#./}/Resets/InsertiveCube34__ReceptiveCube34/resets_$T.pt"
  [ -f "$BANK" ] || { log "$TAG/$T SKIP (no bank at $BANK)"; return; }
  log "$TAG/$T"
  timeout 1500 ./run6.sh "$REPO/scripts_v2/tools/render_reset_states_viewport.py" \
      --reset-states "$BANK" \
      --task "OmniReset-UR10eDelto-CubeStack-$T-v0" \
      --dataset-dir "$DS" --insertive-variant cube34 --receptive-variant cube34 \
      --out "$OUT/${TAG}_$T" --count 6 --settle-steps 20 --repeats 1 \
      --video "$OUT/${TAG}_$T.mp4" --video-freeze 14 --video-settle-steps 60 --video-every 3 "$@" \
      > "out/film_${TAG}_$T.log" 2>&1
  log "  rc=$? $(grep -a 'WROTE' "out/film_${TAG}_$T.log" | tail -1)"
  grep -aE "^  state " "out/film_${TAG}_$T.log" | tee -a out/stage18.log
}

run ObjectAnywhereEEAnywhere --allow-no-dataset-dir
run ObjectRestingEEGrasped
run ObjectAnywhereEEGrasped
run ObjectPartiallyAssembledEEGrasped
echo "STAGE18_DONE $TAG $(date -Is)" | tee -a out/stage18.log
