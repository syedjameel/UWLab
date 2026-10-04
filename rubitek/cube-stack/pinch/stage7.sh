#!/usr/bin/env bash
# Stage 7: re-film the four existing banks with a camera whose aim is verified, not assumed.
set -uo pipefail
cd "$(dirname "$0")"; P=$PWD
DS=/home/mattephi/Work/Rubitek/UWLab-cubestack/Datasets_ur10e_delto/OmniReset/Resets/InsertiveCube34__ReceptiveCube34
log () { echo "=== $(date -Is) $*" | tee -a out/stage7.log; }
for T in ObjectAnywhereEEAnywhere ObjectRestingEEGrasped ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped; do
  log "$T"
  ./run6.sh "$P/banklab.py" --bank "$DS/resets_$T.pt" --states 10 --settle-s 1.5 \
      --cam-eye "${CAM_EYE:-0.95,-0.80,0.55}" --cam-at "${CAM_AT:-0.47,0.0,0.06}" --focal "${FOCAL:-24}" \
      --video "$P/out/bank_before_${T}.mp4" --stills "$P/out/stills_before_${T}" \
      --out "$P/out/bank_before_${T}.json" --label "before $T" > "out/bank_before_${T}.log" 2>&1
  echo "    rc=$? $(grep -E '^(CAMERA|SUMMARY)' "out/bank_before_${T}.log" | tr '\n' ' | ')" | tee -a out/stage7.log
done
echo "STAGE7_DONE $(date -Is)" | tee -a out/stage7.log
