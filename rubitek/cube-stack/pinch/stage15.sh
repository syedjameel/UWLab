#!/usr/bin/env bash
# Close-up of each posture doing the same close/hold/shake, one env per shot so the camera can
# actually frame the jaw. The 3-env demo render put all three hands at distance and showed nothing.
set -uo pipefail
cd /home/mattephi/Work/Rubitek/cube-stack/pinch
P=/home/mattephi/Work/Rubitek/cube-stack/pinch/out/pinchhold_smoke.json
for post in repo_full repo_pinch opt_pinch; do
  echo "=== $post"
  ./run6.sh /home/mattephi/Work/Rubitek/cube-stack/pinch/pinchlab.py \
    --grid demo --postures "$P" \
    --only "{\"posture\":\"$post\"}" \
    --video "/home/mattephi/Work/Rubitek/cube-stack/pinch/out/cu_$post.mp4" \
    --video-envs 1 --focal 24 --cam-span 0.27 \
    --out "/home/mattephi/Work/Rubitek/cube-stack/pinch/out/cu_$post.json" \
    > "out/cu_$post.log" 2>&1
  echo "  exit $? -> $(ls -la out/cu_$post.mp4 2>/dev/null | awk '{print $5}') bytes"
  grep -E "CALIBRATED|WROTE" "out/cu_$post.log" | tail -3
done
echo "ALL DONE"
