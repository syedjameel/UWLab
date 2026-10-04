#!/usr/bin/env bash
# Stage 12: film the reset banks. BEFORE = the five-finger banks that ship; AFTER = the pinch
# banks stage 11 just recorded. Same camera, same settle time, so the two are comparable.
set -uo pipefail
cd "$(dirname "$0")"; P=$PWD
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
PAIR=InsertiveCube34__ReceptiveCube34
log(){ echo "=== $(date -Is) $*" | tee -a out/stage12.log; }

film () {  # film <tag> <dataset-root>
  local tag="$1" root="$2"
  for T in ObjectAnywhereEEAnywhere ObjectRestingEEGrasped ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped; do
    local bank="$root/Resets/$PAIR/resets_$T.pt"
    [ -f "$bank" ] || { log "SKIP $tag $T (no bank)"; continue; }
    log "$tag $T"
    ./run6.sh "$P/banklab.py" --bank "$bank" --states 8 --settle-s 1.5 --focal 22 \
        --video "$P/out/bank_${tag}_${T}.mp4" --stills "$P/out/stills_${tag}_${T}" \
        --out "$P/out/bank_${tag}_${T}.json" --label "$tag $T" \
        > "out/bank_${tag}_${T}.log" 2>&1
    echo "    rc=$? | $(grep -E '^(CAMERA|FRAME CHECK|SUMMARY)' "out/bank_${tag}_${T}.log" | tr '\n' ' | ')" | tee -a out/stage12.log
  done
}

film after "$REPO/Datasets_ur10e_delto/OmniReset_pinchon"
film before "$REPO/Datasets_ur10e_delto/OmniReset"
echo "STAGE12_DONE $(date -Is)" | tee -a out/stage12.log
