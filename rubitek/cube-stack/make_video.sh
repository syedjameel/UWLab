#!/usr/bin/env bash
# Record rollouts and cut the genuine successes into the report's media directory.
#
# EPISODE COUNT IS SET FROM THE MEASURED RATE, NOT PICKED. The deterministic evaluation of
# model_300 on the near-goal family put GENUINE success (episodes longer than the 10-step
# pre-solved boundary) at 9.26% over 821 episodes. At that rate 24 episodes carry a 9.9% chance of
# recording none at all, while 60 drops that to 0.3% and expects ~5.6 -- enough that the four
# longest can be selected rather than whatever turned up.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv/bin/python

CK=${CK:?set CK to a checkpoint path}
TASK=${TASK:-OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-Play-v0}
RT=${RT:-ObjectPartiallyAssembledEEGrasped}
N=${N:-60}
PREFIX=${PREFIX:-neargoal}
EXTRA=${EXTRA:---hero}

OUT=$CS/results/video_${PREFIX}
rm -rf "$OUT"
echo "=== rendering $N episodes / $RT / $(basename "$CK") ==="
"$CS/run.sh" "$CS/render_rollout.py" \
  --task "$TASK" --checkpoint "$CK" --num_envs 1 --episodes "$N" --reset_type "$RT" \
  --collision_stack_mb 1024 --out "$OUT" > "$CS/logs/render_${PREFIX}.log" 2>&1
grep -E 'RENDER_RESULT' "$CS/logs/render_${PREFIX}.log" || {
  echo "RENDER FAILED:"; tail -8 "$CS/logs/render_${PREFIX}.log" | tr '\r' '\n' | tail -6; exit 1; }

# cut_successes exits 0 both when it writes clips and when it finds nothing worth cutting, so the
# caller cannot tell those apart from the status alone. Grade on the artefact instead.
"$PY" "$CS/cut_successes.py" --dir "$OUT" --prefix "$PREFIX" --max-clips 4 $EXTRA || exit 1
N_CLIPS=$(find "$CS/report/media" -name "${PREFIX}_success_*.mp4" -newermt '-1 hour' | wc -l)
"$PY" "$CS/collect_status.py" || exit 1
if [ "$N_CLIPS" -eq 0 ]; then
  echo "[MAKE_VIDEO] NO genuine success clip was written for $PREFIX"
else
  echo "[MAKE_VIDEO] $N_CLIPS genuine success clip(s) written for $PREFIX"
fi
