#!/usr/bin/env bash
# Stage 6: (a) film the four EXISTING reset banks -- the "before" deliverable; (b) find a
# squeeze depth at which BOTH pads load reliably, since 30 mm on a 34 mm cube (2 mm of
# interference per side) loads only one pad about half the time.
set -uo pipefail
cd "$(dirname "$0")"; P=$PWD
log () { echo "=== $(date -Is) $*" | tee -a out/stage6.log; }
run () { local l="$1"; shift; log "$l"; ./run6.sh "$@" > "out/$l" 2>&1
  echo "    rc=$? $(grep -E '^(SUMMARY|RESULT|WROTE)' "out/$l" | tail -3 | tr '\n' ' | ')" | tee -a out/stage6.log; }

DS=/home/mattephi/Work/Rubitek/UWLab-cubestack/Datasets_ur10e_delto/OmniReset/Resets/InsertiveCube34__ReceptiveCube34
for T in ObjectAnywhereEEAnywhere ObjectRestingEEGrasped ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped; do
  [ -f "$DS/resets_$T.pt" ] || continue
  run "bank_before_${T}.log" "$P/banklab.py" --bank "$DS/resets_$T.pt" --states 10 --settle-s 1.5 \
      --video "$P/out/bank_before_${T}.mp4" --stills "$P/out/stills_before_${T}" \
      --out "$P/out/bank_before_${T}.json" --label "before $T"
done

for MM in 22 25 28; do
  run "pinchpose_${MM}.log" "$P/pinchpose.py" --envs 2048 --iters 70 --restarts 3 \
      --close-mm "$MM" --open-mm 50 --clear-mm 36 --out "$P/out/pinchpose_${MM}.json"
  run "sweep_${MM}.log" "$P/pinchlab.py" --grid pinch --postures "$P/out/pinchpose_${MM}.json" \
      --out "$P/out/sweep_${MM}.json" --label "pinch grid, ${MM}mm squeeze"
done
echo "STAGE6_DONE $(date -Is)" | tee -a out/stage6.log
