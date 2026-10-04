#!/usr/bin/env bash
# Stage 8: settle the squeeze depth with repeats, then film the winner.
set -uo pipefail
cd "$(dirname "$0")"; P=$PWD
log(){ echo "=== $(date -Is) $*" | tee -a out/stage8.log; }
run(){ local l="$1"; shift; log "$l"; ./run6.sh "$@" > "out/$l" 2>&1
  echo "    rc=$? $(grep -E '^(RESULT|WROTE)' "out/$l"|tail -3|tr '\n' ' | ')" | tee -a out/stage8.log; }

for MM in 18 20; do
  run "pinchpose_${MM}.log" "$P/pinchpose.py" --envs 2048 --iters 70 --restarts 3 \
      --close-mm "$MM" --open-mm 50 --clear-mm 36 --out "$P/out/pinchpose_${MM}.json"
done
for MM in 18 20 22; do
  run "final_${MM}.log" "$P/pinchlab.py" --grid final --postures "$P/out/pinchpose_${MM}.json" \
      --reps 3 --out "$P/out/final_${MM}.json" --label "final, ${MM}mm squeeze"
done
echo "STAGE8_DONE $(date -Is)" | tee -a out/stage8.log
