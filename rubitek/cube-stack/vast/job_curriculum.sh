#!/usr/bin/env bash
# EXPLICIT BACKWARDS CURRICULUM: four stages, each resuming from the last.
#
# WHY THIS EXISTS AND HOW IT DIFFERS FROM THE PAPER. OmniReset's claim is that a FLAT Uniform(D)
# mixture over the four reset families produces a backwards curriculum by itself -- competence
# appears near the goal first and propagates towards the start, with no schedule. That claim is
# being tested separately by the four-family run on the other box, unchanged.
#
# This run does not test that claim; it spends a small compute budget as well as it can be spent.
# Measured cost is ~73 s per iteration at 2048 envs on the rented A6000 (897 env steps/s; the local
# 4070 Ti SUPER does the same work at 1167). Physics is ~99% of it -- collection 72.7 s against
# learning 0.29 s -- and it scales linearly with env count, so raising envs buys proportionally
# fewer gradient updates per hour rather than free throughput. Wall-clock per update is effectively
# fixed and the only real lever is what the samples are spent on. The paper trains for 2000-2500
# updates while collecting far more per update; near-goal-only reached its ceiling here in 60.
# Staging spends each stage's samples on the family currently at the frontier instead of on three
# families the policy cannot yet reach.
#
# STAGE STRUCTURE. Each stage adds the next family backwards along the task and resumes the previous
# stage's weights, so the four stages are one continuous policy:
#
#   A  near-goal only                                    cube held ~13 mm from the stack pose
#   B  + stable grasp                                    cube held anywhere in the workspace
#   C  + near-object                                     cube on the table, hand already at it
#   D  + reaching  (= the paper's flat mixture)          cube on the table, hand anywhere
#
# Stage A is skipped when START_CK names an existing near-goal checkpoint, because that stage is
# already trained and measured: model_300 scores 9.26% genuine success (episodes that ran past the
# 10-step pre-solved boundary) against an untrained 1.43%, n=821 and 840.
set -uo pipefail
WS=/workspace
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
cd "$WS/UWLab" || exit 1

UV="$HOME/.local/bin/uv"
PY="$WS/venv/bin/python"
if ! "$PY" -c "import inspect, rsl_rl.modules.actor_critic as m, sys; sys.exit(0 if 'gsde' in inspect.getsource(m) else 1)"; then
  echo "=== repairing rsl_rl: installing the UW-Lab gSDE fork ==="
  "$UV" pip install --python "$PY" --no-deps --force-reinstall \
    "rsl-rl-lib @ git+https://github.com/UW-Lab/rsl_rl.git@main" || exit 1
  "$UV" pip install --python "$PY" torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128 || exit 1
fi

ENVS=${ENVS:-2048}
SEED=${SEED:-42}
ENTROPY=${ENTROPY:-0.0005}
TASK=OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-v0
RUNS=logs/rsl_rl/ur10e_delto_omnireset_agent
# Stage A is skipped only when an EXPLICITLY SEEDED near-goal checkpoint is present.
#
# This used to default to "the newest model_*.pt anywhere under logs/rsl_rl on this box", which is
# wrong in the exact case it was meant to serve: a box that has just been repurposed still carries
# the previous run's directory, and the newest file in it is that run's model_0 -- an UNTRAINED
# network. The curriculum then skipped stage A and started stage B from random weights, which is
# strictly worse than training stage A, and it reported "resuming from ..." while doing it.
#
# /workspace/seed_neargoal/model_300.pt is uploaded deliberately before this job is queued. If it is
# absent, stage A trains from scratch, which is the correct fallback.
START_CK=${START_CK:-/workspace/seed_neargoal/model_300.pt}

C4=ObjectPartiallyAssembledEEGrasped
C3=ObjectAnywhereEEGrasped
C2=ObjectRestingEEGrasped
C1=ObjectAnywhereEEAnywhere

newest_ck () { find "$RUNS" -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-; }

stage () {  # stage <name> <iters> <types-list> <probs-list> <resume-ck>
  local name=$1 iters=$2 types=$3 probs=$4 ck=$5
  echo
  echo "=== STAGE $name : $iters iters, families $types, probs $probs ==="
  echo "=== resuming from ${ck:-<scratch>} ==="
  local resume=()
  [ -n "$ck" ] && [ -f "$ck" ] && resume=(--resume_path "$ck")
  "$PY" scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$TASK" --num_envs "$ENVS" --max_iterations "$iters" --seed "$SEED" --headless \
    "${resume[@]}" \
    --run_name "curr_$name" \
    env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e_delto/OmniReset \
    "env.events.reset_from_reset_states.params.reset_types=$types" \
    "env.events.reset_from_reset_states.params.probs=$probs" \
    agent.algorithm.entropy_coef="$ENTROPY" \
    agent.save_interval=50 2>&1 | tr '\r' '\n' | grep -vE '^\[[0-9]'
  echo "=== stage $name finished, newest checkpoint: $(newest_ck) ==="
}

CK=""
if [ -f "$START_CK" ]; then
  echo "=== stage A supplied as $START_CK (near-goal, already trained and measured) ==="
  CK=$START_CK
else
  stage A "${ITERS_A:-300}" "[$C4]" "[1.0]" ""
  CK=$(newest_ck)
fi

# Budget moved to the far end for the same reason as the two-finger run: stage D is the only stage
# containing Reaching, and Reaching is the unsolved part. Stage A is seeded, not trained, so these
# counts are ADDITIONAL iterations on top of the seed checkpoint's 300.
stage B "${ITERS_B:-450}" "[$C4,$C3]"       "[0.5,0.5]"           "$CK"; CK=$(newest_ck)
stage C "${ITERS_C:-600}" "[$C4,$C3,$C2]"   "[0.34,0.33,0.33]"    "$CK"; CK=$(newest_ck)
stage D "${ITERS_D:-1400}" "[$C4,$C3,$C2,$C1]" "[0.25,0.25,0.25,0.25]" "$CK"

echo "=== curriculum done ==="
find "$RUNS" -name 'model_*.pt' -newermt '-1 day' -printf '%s %p\n' | sort -rn | head -8
