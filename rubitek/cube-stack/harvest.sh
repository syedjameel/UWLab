#!/usr/bin/env bash
# Stop training at a chosen point, then evaluate and render from the best checkpoint.
#
# WHY NOT EVALUATE CONCURRENTLY. Tried it: training at 2048 envs leaves ~500 MiB free on the 16 GB
# dev card, and an eval process alongside it does not fit. Evaluation and rendering therefore have
# to come after training stops, which makes the stop point a real decision rather than an
# afterthought.
#
# WHY STOP AT ALL. Measured 47.4 s/iteration, so the configured 2000 iterations is 26 h. Checkpoints
# land every 100 iterations and each is usable. Stopping at a planned point leaves time to produce
# the actual deliverables -- success numbers and video -- instead of holding a GPU on a run that
# cannot finish.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
RUNS=$REPO/logs/rsl_rl/ur10e_delto_omnireset_agent
SSHKEY=/home/mattephi/Work/IRiS/ip/meaningful-latents/.secrets/id_ed25519_ml_vast
HOSTARGS="-i $SSHKEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p 40624"
HOST=root@38.29.145.10

TARGET_ITER=${TARGET_ITER:-400}
DEADLINE=${DEADLINE:-21600}   # 6 h wall-clock ceiling regardless of iteration count

t0=$(date +%s)
echo "=== harvest armed $(date +%H:%M:%S): stop at iter ${TARGET_ITER} or ${DEADLINE}s ==="
while true; do
  it=$(grep -oE 'Learning iteration [0-9]+' "$CS/logs/train_noorient.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
  it=${it:-0}
  el=$(( $(date +%s) - t0 ))
  echo "[$(date +%H:%M:%S)] local iter=${it} el=${el}s"
  [ "$it" -ge "$TARGET_ITER" ] 2>/dev/null && { echo "=== reached iter ${it} -- stopping ==="; break; }
  [ "$el" -ge "$DEADLINE" ] && { echo "=== deadline at iter ${it} -- stopping ==="; break; }
  pgrep -f '[t]rain.py' >/dev/null || { echo "=== training already exited at iter ${it} ==="; break; }
  sleep 120
done

echo "=== stopping both training runs $(date +%H:%M:%S) ==="
pkill -9 -f '[t]rain.py' 2>/dev/null || true
timeout 25 ssh $HOSTARGS $HOST "pkill -9 -f '[t]rain.py'" 2>/dev/null || true
sleep 12

echo "=== pulling the remote (oriented) checkpoints ==="
mkdir -p "$REPO/logs/rsl_rl_remote"
rsync -az -e "ssh $HOSTARGS" "$HOST:/workspace/UWLab/logs/rsl_rl/" "$REPO/logs/rsl_rl_remote/" 2>/dev/null || echo "  (no remote checkpoints yet)"

echo
echo "=== checkpoints available ==="
find "$RUNS" "$REPO/logs/rsl_rl_remote" -name 'model_*.pt' -printf '%T@ %s %p\n' 2>/dev/null | sort -rn | head -8

CK=$(find "$RUNS" -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
[ -z "${CK:-}" ] && { echo "NO local checkpoint -- nothing to evaluate"; exit 1; }
# model_0 MUST come from the SAME run directory as the trained checkpoint. A bare
# `find $RUNS -name model_0.pt | head -1` picked up the ABORTED first run (which died on the gsde
# ValueError before training anything) -- a different initialisation, which would have made the
# baseline and the trained policy incomparable while looking perfectly fine in the output.
CK0=$(dirname "$CK")/model_0.pt
[ -f "$CK0" ] || CK0=""

# CONTROLLED COMPARISON: did the policy actually learn anything?
#
# The training reward curve cannot answer this. It falls over the run, but the fall is the
# action_rate penalty tracking gSDE's exploration noise (std 0.75 -> 2.50), while the task terms stay
# flat. Evaluation does not use that noise at all -- rsl_rl's get_inference_policy returns
# act_inference, which is the actor's MEAN action -- so training return and evaluated behaviour are
# measuring different things.
#
# Comparing the untrained model_0 against the final checkpoint under identical deterministic
# evaluation is the only honest way to separate "learned a little" from "learned nothing".
if [ -n "${CK0:-}" ] && [ "$CK0" != "$CK" ]; then
  echo
  echo "=== BASELINE: untrained model_0, near-goal family ==="
  "$CS/run.sh" "$CS/eval_policy.py" \
    --task OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-Play-v0 \
    --checkpoint "$CK0" --num_envs 128 --episodes 800 \
    --reset_type ObjectPartiallyAssembledEEGrasped \
    --collision_stack_mb 1024 --headless \
    --out "$CS/results/eval_baseline_model0.json" > "$CS/logs/eval_baseline.log" 2>&1
  grep -E 'EVAL_RESULT' "$CS/logs/eval_baseline.log" || { echo "  baseline eval FAILED:"; tail -5 "$CS/logs/eval_baseline.log" | tr '\r' '\n' | tail -3; }
fi
echo
echo "=== evaluating $CK, decomposed by reset family (the paper's Fig. 8 breakdown) ==="
# Ordered end-of-task -> start-of-task, which is how the paper orders its panels and how competence
# is expected to propagate: near-goal first, full task last. A partially trained policy can be
# genuinely competent at the near-goal end while still scoring zero on the full task, and reporting
# only the latter would hide a real result -- while reporting only the former would overstate one.
# Episode budget is spent where precision is needed. The near-goal family carries the one
# statistical CLAIM (trained vs untrained), so it gets 800 -- the n required to resolve a ~5.7% vs
# ~9.2% difference at p<0.05. The other three are DESCRIPTIVE: at n=300 a true 0% is bounded above
# by ~1% at 95% confidence, which is all that is needed to say "no successes". Spending 800 on each
# would add ~25 minutes for precision nothing depends on.
for RT in ObjectPartiallyAssembledEEGrasped ObjectAnywhereEEGrasped ObjectRestingEEGrasped ObjectAnywhereEEAnywhere; do
  case "$RT" in
    ObjectPartiallyAssembledEEGrasped) N=800 ;;
    *) N=300 ;;
  esac
  echo "--- $RT (n=$N) ---"
  "$CS/run.sh" "$CS/eval_policy.py" \
    --task OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-Play-v0 \
    --checkpoint "$CK" --num_envs 128 --episodes "$N" --reset_type "$RT" \
    --collision_stack_mb 1024 --headless \
    --out "$CS/results/eval_noorient_${RT}.json" > "$CS/logs/eval_final_${RT}.log" 2>&1
  grep -E 'EVAL_RESULT' "$CS/logs/eval_final_${RT}.log" || { echo "  FAILED:"; tail -6 "$CS/logs/eval_final_${RT}.log" | tr '\r' '\n' | tail -4; }
done

# ---------------------------------------------------------------------------------------
# The ORIENTED variant, evaluated against its own criterion.
#
# This is the comparison the user's instinct was about: position-only was prioritised on the
# expectation that orientation is markedly harder. The branch already measured that cost on the
# table-leg task -- adding orientation moved median precision from 4.8 mm to 26.4 mm, a 4.5x
# penalty -- so the expectation is well founded, but expectation is not measurement.
#
# Note the criterion differs: this variant scores success only when euler_xy < 0.05 rad AS WELL AS
# position < 5 mm. It is a strictly harder test on the same physics, not a different task.
CKO=$(find "$REPO/logs/rsl_rl_remote" -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "${CKO:-}" ]; then
  echo
  echo "=== ORIENTED variant: $(basename "$CKO") ==="
  for RT in ObjectPartiallyAssembledEEGrasped ObjectAnywhereEEAnywhere; do
    echo "--- oriented / $RT (n=300) ---"
    "$CS/run.sh" "$CS/eval_policy.py" \
      --task OmniReset-UR10eDelto-CubeStack-RelCartesianOSC-State-Play-v0 \
      --checkpoint "$CKO" --num_envs 128 --episodes 300 --reset_type "$RT" \
      --collision_stack_mb 1024 --headless \
      --out "$CS/results/eval_oriented_${RT}.json" > "$CS/logs/eval_oriented_${RT}.log" 2>&1
    grep -E 'EVAL_RESULT|thresholds' "$CS/logs/eval_oriented_${RT}.log" \
      || { echo "  FAILED:"; tail -5 "$CS/logs/eval_oriented_${RT}.log" | tr '\r' '\n' | tail -3; }
  done
else
  echo "=== no oriented checkpoint pulled -- skipping that comparison ==="
fi

echo
echo "=== rendering video: near-goal AND full task ==="
# Two clips, both labelled. Near-goal is where a compute-limited policy is most likely to show real
# behaviour; the full task is the honest headline. Showing only the first would overstate, showing
# only the second would very likely be ten identical failures.
# Episode counts are set from the measured success rate, not picked. At ~9% on near-goal, 8
# episodes carry a 46% chance of recording ZERO successes -- a video with nothing in it. 40 episodes
# drops that to 2.1% and expects ~3.7. The full-task family gets fewer because the honest
# expectation there is no successes at all, and the clip exists to show what the policy does, not to
# fish for one.
render_and_cut () {  # render_and_cut <reset_family> <episodes>
  local RT=$1 N=$2
  echo "--- rendering $N episodes from $RT ---"
  "$CS/run.sh" "$CS/render_rollout.py" \
    --task OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-Play-v0 \
    --checkpoint "$CK" --num_envs 1 --episodes "$N" --reset_type "$RT" \
    --collision_stack_mb 1024 --out "$CS/results/video_${RT}" \
    > "$CS/logs/render_${RT}.log" 2>&1
  grep -E 'RENDER_RESULT' "$CS/logs/render_${RT}.log" \
    || { echo "  RENDER FAILED:"; tail -6 "$CS/logs/render_${RT}.log" | tr '\r' '\n' | tail -4; return; }
  # Cut the successful episodes out, selected from the recorded predicate rather than by eye.
  /home/mattephi/Work/Rubitek/uwlab-env/.venv/bin/python "$CS/cut_successes.py" \
    --dir "$CS/results/video_${RT}" --fallback-clips 2 2>&1 | tail -12
}

render_and_cut ObjectPartiallyAssembledEEGrasped 40
render_and_cut ObjectAnywhereEEAnywhere 16
grep -E 'RENDER_RESULT|SUCCESS|touched' "$CS/logs/render_noorient.log" | head -20 || tail -8 "$CS/logs/render_noorient.log" | tr '\r' '\n' | tail -5
ls -la "$CS/results/video_noorient/" 2>/dev/null

echo "=== harvest done $(date +%H:%M:%S) ==="
