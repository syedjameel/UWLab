#!/usr/bin/env bash
# Stage 5: measure the true open pinch with contacts FILTERED to the cube, then film it.
set -uo pipefail
cd "$(dirname "$0")"; P=$PWD
POSE=${POSE:-$P/out/pinchpose.json}
log () { echo "=== $(date -Is) $*" | tee -a out/stage5.log; }
run () { local l="$1"; shift; log "$l"
  ./run6.sh "$@" > "out/$l" 2>&1
  echo "    rc=$? $(grep -E '^(WROTE|CALIBRATED)' "out/$l" | tail -4 | tr '\n' ' | ')" | tee -a out/stage5.log; }

run diag_v3.log "$P/pinchlab.py" --grid ctrl --postures "$POSE" --only '{"ctrl":"direct"}' \
    --out "$P/out/diag_v3.json" --label "filtered contacts, 3 postures"

run sweep_v3.log "$P/pinchlab.py" --grid main --postures "$POSE" \
    --out "$P/out/sweep_v3.json" --label "main grid, true pinch, filtered contacts"

for N in repo_full opt_pinch; do
  run "hero3_${N}.log" "$P/pinchlab.py" --grid demo --only "{\"posture\":\"$N\"}" \
      --postures "$POSE" --video "$P/out/hero3_${N}.mp4" --stills "$P/out/stills3_${N}" \
      --video-envs 1 --cam-span 0.22 --focal 26 --fps 30 \
      --out "$P/out/hero3_${N}.json" --label "hero3 $N"
done
echo "STAGE5_DONE $(date -Is)" | tee -a out/stage5.log
