#!/usr/bin/env bash
# Film reset families through the repo renderer. film.sh <dataset_dir> <tag> <task prefix>
set -uo pipefail
cd /home/mattephi/Work/Rubitek/cube-stack/linear
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
R=/home/mattephi/Work/Rubitek/cube-stack/pinch/run6.sh
OUT=/home/mattephi/Work/Rubitek/cube-stack/linear/out/films
DS="$1"; TAG="$2"; PFX="$3"
export UWLAB_VIEW_EYE="${UWLAB_VIEW_EYE:-1.55,-1.35,0.95}" UWLAB_VIEW_AT="${UWLAB_VIEW_AT:-0.45,0.00,0.10}"
log(){ echo "=== $(date -Is) $*" | tee -a out/film.log; }
run () {
  local T="$1"; shift
  [ -n "${UWLAB_FILM_ONLY:-}" ] && [ "$T" != "$UWLAB_FILM_ONLY" ] && return
  local BANK="$REPO/${DS#./}/Resets/InsertiveCube34__ReceptiveCube34/resets_$T.pt"
  [ -f "$BANK" ] || { log "$TAG/$T SKIP (no bank)"; return; }
  log "$TAG/$T"
  timeout 1500 $R "$REPO/scripts_v2/tools/render_reset_states_viewport.py" --reset-states "$BANK" \
      --task "$PFX-$T-v0" --dataset-dir "$DS" --insertive-variant cube34 --receptive-variant cube34 \
      --out "$OUT/${TAG}_$T" --count 6 --settle-steps 20 --repeats 1 \
      --video "$OUT/${TAG}_$T.mp4" --video-freeze 14 --video-settle-steps 60 --video-every 3 "$@" \
      > "out/film_${TAG}_$T.log" 2>&1
  log "  rc=$? $(grep -a 'WROTE' "out/film_${TAG}_$T.log" | tail -1)"
  grep -aE "^  state |Traceback|Error" "out/film_${TAG}_$T.log" | tail -8 | tee -a out/film.log
}
run ObjectAnywhereEEAnywhere --allow-no-dataset-dir
run ObjectRestingEEGrasped
run ObjectAnywhereEEGrasped
run ObjectPartiallyAssembledEEGrasped
echo "FILM_DONE $TAG $(date -Is)" | tee -a out/film.log
