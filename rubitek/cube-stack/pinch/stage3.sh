#!/usr/bin/env bash
# Stage 3: re-derive the posture with fingers 3/4/5 actually clear of the jaw, then re-measure
# and re-film. The previous "working" pinch was a scoop: the cube rested on the curled fingers
# (thumb 2.22 N, index 0.00 N, tips 3/4/5 9.97 N).
set -uo pipefail
cd "$(dirname "$0")"; P=$PWD
log () { echo "=== $(date -Is) $*" | tee -a out/stage3.log; }
run () { local l="$1"; shift; log "$l"
  ./run6.sh "$@" > "out/$l" 2>&1
  echo "    rc=$? $(grep -E '^(RESULT|WROTE) ' "out/$l" | tail -3 | tr '\n' ' | ')" | tee -a out/stage3.log; }

run jawopt_v2.log  "$P/jawopt.py" --envs 1024 --iters 60 --tuck-iters 60 \
    --close-mm 30.5 --open-mm 52 --tuck-clear-mm 45 --out "$P/out/jawopt_v2.json"

run sweep_v2.log "$P/pinchlab.py" --grid pinch --postures "$P/out/jawopt_v2.json" \
    --out "$P/out/sweep_v2.json" --label "pinch grid, tuck-cleared jaw"

for N in repo_full repo_pinch opt_pinch; do
  run "hero2_${N}.log" "$P/pinchlab.py" --grid demo --only "{\"posture\":\"$N\"}" \
      --postures "$P/out/jawopt_v2.json" \
      --video "$P/out/hero2_${N}.mp4" --stills "$P/out/stills2_${N}" \
      --video-envs 1 --cam-span 0.20 --focal 26 --fps 30 \
      --out "$P/out/hero2_${N}.json" --label "hero2 $N"
done
echo "STAGE3_DONE $(date -Is)" | tee -a out/stage3.log
