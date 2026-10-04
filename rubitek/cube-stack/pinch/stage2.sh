#!/usr/bin/env bash
# Stage 2: re-measure with steady grip force, then render the hero clips one environment at a
# time so each panel is framed on its own jaw.
set -uo pipefail
cd "$(dirname "$0")"
P=$PWD
run () { local log="$1"; shift
  echo "=== $(date -Is) $log" | tee -a out/stage2.log
  ./run6.sh "$P/pinchlab.py" "$@" > "out/$log" 2>&1
  echo "    rc=$? $(grep -E '^WROTE .*(json|mp4)' "out/$log" | tail -2 | tr '\n' ' ')" | tee -a out/stage2.log
}

run sweep_pinch26.log --grid pinch --postures "$P/out/jawopt.json"   --out "$P/out/sweep_pinch26.json" --label "pinch grid, 26mm jaw"
run sweep_pinch30.log --grid pinch --postures "$P/out/jawopt30.json" --out "$P/out/sweep_pinch30.json" --label "pinch grid, 30.5mm jaw"
run sweep_main26.log  --grid main  --postures "$P/out/jawopt.json"   --out "$P/out/sweep_main26.json"  --label "main grid, 26mm jaw"

# Hero clips: one environment each, slow close, free cube, repo-default friction/stiffness.
for P_NAME in repo_full repo_pinch opt_pinch; do
  run "hero_${P_NAME}.log" --grid demo --only "{\"posture\":\"$P_NAME\"}" \
      --postures "$P/out/jawopt30.json" \
      --video "$P/out/hero_${P_NAME}.mp4" --stills "$P/out/stills_${P_NAME}" \
      --video-envs 1 --cam-span 0.20 --focal 26 --fps 30 \
      --out "$P/out/hero_${P_NAME}.json" --label "hero $P_NAME"
done

echo "STAGE2_DONE $(date -Is)" | tee -a out/stage2.log
