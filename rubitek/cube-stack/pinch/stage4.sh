#!/usr/bin/env bash
# Stage 4: film the four EXISTING reset banks. This is the "before" half of the deliverable --
# what the current banks look like when their states are replayed under real physics.
set -uo pipefail
cd "$(dirname "$0")"; P=$PWD
DS=${DS:-/home/mattephi/Work/Rubitek/UWLab-cubestack/Datasets_ur10e_delto/OmniReset/Resets/InsertiveCube34__ReceptiveCube34}
TAG=${TAG:-before}
log () { echo "=== $(date -Is) $*" | tee -a out/stage4.log; }

for T in ObjectAnywhereEEAnywhere ObjectRestingEEGrasped ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped; do
  [ -f "$DS/resets_$T.pt" ] || { log "SKIP $T (no bank)"; continue; }
  log "$TAG $T"
  ./run6.sh "$P/banklab.py" --bank "$DS/resets_$T.pt" \
      --states 10 --settle-s 1.5 \
      --video "$P/out/bank_${TAG}_${T}.mp4" --stills "$P/out/stills_${TAG}_${T}" \
      --out "$P/out/bank_${TAG}_${T}.json" --label "$TAG $T" \
      > "out/bank_${TAG}_${T}.log" 2>&1
  echo "    rc=$? $(grep -E '^(SUMMARY|WROTE)' "out/bank_${TAG}_${T}.log" | tr '\n' ' | ')" | tee -a out/stage4.log
done
echo "STAGE4_DONE $(date -Is)" | tee -a out/stage4.log
