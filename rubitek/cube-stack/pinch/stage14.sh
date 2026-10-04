#!/usr/bin/env bash
# NOTE: the posture matters as much as friction. Run against pinchpose_seg.json -- the posture
# tuned to clear the sampler's placement gate -- every mu gave 0% held with 0.00 N of pad force,
# i.e. the pads were not touching at all. Clearing the gate and gripping the cube pull in
# opposite directions. This sweep uses pinchpose_22.json, the posture that measured 20% true
# pinch, so the variable under test really is friction.
#
# Stage 14: does surface friction explain the pinch's marginal hold? Arithmetic says the pinch
# sits at a safety factor of 0.98 at the scene's mu=0.2. Measure it instead of arguing.
set -uo pipefail
cd "$(dirname "$0")"; P=$PWD
for MU in 0.2 0.4 0.7 1.0; do
  echo "=== $(date -Is) mu=$MU" | tee -a out/stage14.log
  ./run6.sh "$P/pinchlab.py" --grid final --postures "$P/out/pinchpose_22.json" \
      --friction-mu "$MU" --reps 2 --out "$P/out/mu_$MU.json" --label "mu=$MU" \
      > "out/mu_$MU.log" 2>&1
  echo "    rc=$? | $(grep -E '^WROTE .*json' "out/mu_$MU.log" | tail -1)" | tee -a out/stage14.log
done
echo "STAGE14_DONE $(date -Is)" | tee -a out/stage14.log
