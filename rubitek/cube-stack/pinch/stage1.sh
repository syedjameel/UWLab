#!/usr/bin/env bash
# Stage 1: the sweeps. Serial by necessity -- 16 GB of VRAM fits one Isaac Sim, so the
# parallelism lives inside each run (one config per environment), not across runs.
set -uo pipefail
cd "$(dirname "$0")"
P=$PWD

run () {  # run <log> <args...>
  local log="$1"; shift
  echo "=== $(date -Is) $log" | tee -a out/stage1.log
  ./run6.sh "$P/pinchlab.py" "$@" > "out/$log" 2>&1
  echo "    rc=$? $(grep -c . "out/$log") lines; $(grep -E '^WROTE .*json' "out/$log" | tail -1)" | tee -a out/stage1.log
}

# A: the full matrix at the 26 mm squeeze. Answers control-mode and posture together.
run sweep_main26.log --grid main --postures "$P/out/jawopt.json" \
    --out "$P/out/sweep_main26.json" --label "main grid, 26mm jaw"

# B/C: schedule x friction x stiffness, at both squeeze depths. 26 mm held the cube but at
# 14 N of tip force, well above the 5 N crush bound metadata declares, so the gentler jaw is
# not a nicety -- a grasp that only survives by crushing the part is not a grasp we can ship.
run sweep_pinch26.log --grid pinch --postures "$P/out/jawopt.json" \
    --out "$P/out/sweep_pinch26.json" --label "pinch grid, 26mm jaw"

run sweep_pinch30.log --grid pinch --postures "$P/out/jawopt30.json" \
    --out "$P/out/sweep_pinch30.json" --label "pinch grid, 30.5mm jaw"

echo "STAGE1_DONE $(date -Is)" | tee -a out/stage1.log
