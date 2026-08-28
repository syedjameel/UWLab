#!/usr/bin/env bash
# DexReset S1/S2' step 1 (bead UWLab-82wg, epic UWLab-nnlv): finetune the DexLift table-leg REORIENT
# policy so it can be COMMANDED to hold the leg VERTICAL (tip-down) at bore height, warm-started from
# the certified Stage-3 ep_3600 checkpoint, under a MIXED goal distribution.
#
# Usage:  launch_dexreset_vertical_goal_finetune.sh <gpu-index> [vertical_prob]
#   e.g.  launch_dexreset_vertical_goal_finetune.sh 0 0.50
#
# Derived from launch_dexlift_tableleg_reorient_finetune.sh (epic UWLab-g3z4). WHAT DIFFERS, and why
# each difference is deliberate:
#
#   * DEXLIFT_EPISODE_MIXTURE is NOT set. That is the g3z4 gravity/low-goal/partial-assembly mixture,
#     a different experiment, and its finetune certified 20 POINTS WORSE than its own parent at 30 mm
#     (0.4766 vs 0.6797). Nothing here needs it, and enabling it would confound this run with that
#     one. Consequence: no partial_assemblies.pt is required, so that script's dataset guards are gone.
#   * DEXLIFT_GOAL_VERTICAL_PROB is set. See _apply_goal_vertical_mixture in
#     dexlift_ur5e_delto_env_cfg.py and mdp/goal_mixture.py for the full argument; in one line, the
#     policy can grasp but has never been shown a goal outside +-0.3 rad of upright, so it cannot be
#     told to hold the leg tip-down, which is exactly the state S1/S2' need.
#   * Box defaults are DL_H100's, not the rented-box layout the g3z4 script defaults to.
#
# =====================================================================================
# THE MIXTURE IS THE POINT -- DO NOT SET vertical_prob TO 1.0
# =====================================================================================
# A 100-percent finetune on a changed goal distribution has already destroyed this exact policy once:
# 55 percent of its skill gone in 50 epochs, certified 0.0000 at 30 mm by epoch 1550, with the damage
# FASTEST AT THE START (so "stop early" is not an escape hatch). The remaining fraction of episodes
# keeps drawing the ORIGINAL staged goal, so the parent task stays rewarded for the whole finetune.
# 0.50 is the default: enough exposure to learn the new goal region, half the batch still anchoring
# the old one.
#
# =====================================================================================
# --max_iterations IS ABSOLUTE, NOT ADDITIONAL
# =====================================================================================
# rl_games restores epoch_num from the checkpoint and trains until max_epochs. Warm-starting at 3600
# and asking for 5600 is +2000 epochs of finetuning, not 5600 epochs of training.
MAX_ITERATIONS=${MAX_ITERATIONS:-5600}   # ABSOLUTE, NOT additional: 3600 (warm) + 2000 finetune

# =====================================================================================
# OPTIMIZER UPDATE RATE MATCHES THE CERTIFIED LINEAGE
# =====================================================================================
# updates_per_epoch = mini_epochs * num_envs * horizon_length / minibatch_size, with mini_epochs=5 and
# horizon_length=36 fixed by the task's own agent yaml. The certified lineage does 20/epoch. Asserted
# at runtime below rather than trusted from this comment.
NUM_ENVS=4096
MINIBATCH=36864

WANDB_ENTITY="i_domrachev-interactive-robotic-systems-lab-kaist"
WANDB_PROJECT="uwlab-dexinsertion"

set -uo pipefail

GPU=${1:?"usage: $0 <gpu-index> [vertical_prob]"}
VERTICAL_PROB=${2:-0.50}

# -- Reject a shared GPU. Never launch on a GPU that already has memory resident on it: two Isaac
# processes on one device have been measured to return silently invalid physics at rc=0.
USED=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
[ "${USED:-999999}" -lt 2000 ] || { echo "REFUSING: GPU $GPU has ${USED} MiB already in use"; exit 1; }

EXPECT=$(( 5 * NUM_ENVS * 36 / MINIBATCH ))
[ "$EXPECT" -eq 20 ] || {
  echo "REFUSING: NUM_ENVS=$NUM_ENVS MINIBATCH=$MINIBATCH gives $EXPECT updates/epoch, not the certified 20"
  exit 1
}
[ $(( NUM_ENVS * 36 % MINIBATCH )) -eq 0 ] || {
  echo "REFUSING: minibatch $MINIBATCH does not evenly divide the $NUM_ENVS*36 batch"
  exit 1
}

# -- vertical_prob is validated HERE as well as in the term, because a typo'd value that parses (0.5
# vs 5.0) would otherwise only surface after a 15-minute Isaac boot.
# A GLOB IS NOT A NUMERIC VALIDATOR, and the difference is a silent wrong number rather than an
# error. `case ... in 0.*)` matches ANY string beginning with "0.", so "0.5abc" passes; `printf
# '%.3f'` then writes "invalid number" to stderr and STILL PRINTS 0.500, and with `set -uo pipefail`
# (no -e) its failure exit code stops nothing. The run would proceed at a fraction nobody typed.
# Match the whole token against a numeric pattern instead, and check printf's status as well.
[[ "$VERTICAL_PROB" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || {
  echo "REFUSING: vertical_prob must be a number in [0, 1]; got '$VERTICAL_PROB'"
  exit 1
}
VERTICAL_PROB_FMT=$(printf '%.3f' "$VERTICAL_PROB" 2>/dev/null) || {
  echo "REFUSING: vertical_prob '$VERTICAL_PROB' is not formattable as a number"
  exit 1
}
[ "$VERTICAL_PROB_FMT" = "1.000" ] && {
  echo "REFUSING: vertical_prob 1.0 removes the original task from the objective entirely."
  echo "  That configuration cost this policy 55 percent of its skill in 50 epochs. Use < 1.0."
  exit 1
}
[ "$VERTICAL_PROB_FMT" = "0.000" ] && {
  echo "REFUSING: vertical_prob 0.0 is the unmixed task -- this launch would train nothing new."
  exit 1
}

export CUDA_VISIBLE_DEVICES="$GPU"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHONUNBUFFERED=1

# -- The four plant vars, together, plus the task-definition tilt. A silently-unset plant var yields a
# PLAUSIBLE WRONG NUMBER rather than an error: the identified hand caps at 3.0 rad/s against a ~6
# rad/s commanded closure, so its fingers cannot track a closing command at all.
export DEXLIFT_REF_RESET=1
export DEXLIFT_REF_ACTUATORS=1
export DEXLIFT_REF_HAND_ACT=1
export DEXLIFT_REF_ARM_ACT=0
export DEXLIFT_POSE_TILT=0.3

# -- THE NEW TERM. Off unless set; see _apply_goal_vertical_mixture for the band and its derivation.
# NOTE it does NOT go through commands.object_pose.ranges: DEXLIFT_POSE_TILT overwrites those, and a
# hydra override of ranges.pitch is silently clamped by it -- a defect that already cost one whole
# invalid experiment. The vertical band is a separate field the staging cannot reach.
export DEXLIFT_GOAL_VERTICAL_PROB="$VERTICAL_PROB_FMT"
export DEXLIFT_GOAL_VERTICAL_TILT=${DEXLIFT_GOAL_VERTICAL_TILT:-0.35}
export DEXLIFT_GOAL_VERTICAL_Z=${DEXLIFT_GOAL_VERTICAL_Z:-0.13,0.27}

# -- DL_H100 SPECIFICS. /tmp/uwlab and /tmp/isaaclab are owned by another uid on this box, and the
# second is NOT covered by UWLAB_TMP_ROOT (IsaacLab's logger uses tempfile.gettempdir()), so both are
# needed or the run dies with PermissionError before the scene builds.
export UWLAB_TMP_ROOT=${UWLAB_TMP_ROOT:-$HOME/tmp}
export TMPDIR=${TMPDIR:-$HOME/tmp}
mkdir -p "$UWLAB_TMP_ROOT" "$TMPDIR"

# -- api.wandb.ai is blocked at this host's network edge (HTTP/2 403 from Google Cloud Armor on every
# endpoint, with a valid key in ~/.netrc making no difference). Log offline and sync from a machine
# that can reach it. Set explicitly or rsl_rl/rl_games land the run in a generic default project.
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_ENTITY
export WANDB_PROJECT

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/source/uwlab:$REPO_ROOT/source/uwlab_tasks:$REPO_ROOT/source/uwlab_assets:$REPO_ROOT/source/uwlab_rl"

TASK="DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0"
WANDB_NAME="dexreset-vertical-goal-mix${VERTICAL_PROB_FMT}-warm3600-gpu${GPU}"

# lr_schedule is ADAPTIVE (agent yaml), so `learning_rate` is driven by the KL controller and
# overriding it does not stick -- kl_threshold is the knob that actually bounds how far the
# policy moves per update, which is what a warm-started run risking its parent skill needs.
KL_THRESHOLD=${KL_THRESHOLD:-0.01}
LAUNCH_PYTHON_BIN="${LAUNCH_PYTHON_BIN:-$HOME/venv_uwlab/bin/python}"
[ -x "$LAUNCH_PYTHON_BIN" ] || { echo "REFUSING: LAUNCH_PYTHON_BIN '$LAUNCH_PYTHON_BIN' not executable"; exit 1; }

# -- Checkpoint identity verified off the BYTES on this box, not from a filename. This project has
# two files whose names differ only in an epoch number and whose training rewards rank them the
# OPPOSITE way round from their certified scores, so a name is not an identity.
CKPT=${CKPT:-$HOME/ckpt/last_dexlift_ur5e_delto_reljointpos_tableleg_reorient_ep_3600_rew_38.38917.pth}
# Overridable ONLY to continue from a checkpoint this experiment itself produced (bead
# UWLab-nnlv). The DEFAULT stays the parent, so an unqualified launch still refuses anything but
# ep3600 -- a continuation has to name the bytes it means to continue from, not merely point at
# a path. Two files here differ only in an epoch number and rank OPPOSITE ways by reward and by
# certified score, so a filename is not an identity.
CKPT_SHA256_EXPECT=${CKPT_SHA256_EXPECT:-9534e102a64fc06a3588c5102fc3421e69ef1b18fe30e7ffb40f7df63b5d76af}
[ -f "$CKPT" ] || { echo "REFUSING: warm-start checkpoint not found at $CKPT"; exit 1; }
CKPT_SHA256_GOT="$(sha256sum "$CKPT" | awk '{print $1}')"
[ "$CKPT_SHA256_GOT" = "$CKPT_SHA256_EXPECT" ] || {
  echo "REFUSING: checkpoint sha256 mismatch."
  echo "  expected: $CKPT_SHA256_EXPECT"
  echo "  got:      $CKPT_SHA256_GOT"
  exit 1
}
echo "checkpoint sha256 verified: $CKPT_SHA256_GOT"

LOG="$REPO_ROOT/logs/launch_dexreset_vertical_goal_finetune_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
echo "task=$TASK gpu=$GPU num_envs=$NUM_ENVS minibatch=$MINIBATCH updates/epoch=$EXPECT max_iterations=$MAX_ITERATIONS (warm 3600 + 2000)" | tee -a "$LOG"
echo "vertical_prob=$VERTICAL_PROB_FMT tilt=$DEXLIFT_GOAL_VERTICAL_TILT z=$DEXLIFT_GOAL_VERTICAL_Z" | tee -a "$LOG"
echo "checkpoint=$CKPT" | tee -a "$LOG"
echo "wandb: mode=$WANDB_MODE entity=$WANDB_ENTITY project=$WANDB_PROJECT name=$WANDB_NAME" | tee -a "$LOG"

# -- setsid + nohup + timeout -s KILL, redirected straight to a FILE. Never pipe an Isaac run through
# grep: grep buffers, and a killed process loses everything in that buffer. Isaac also ignores
# SIGTERM, so the master timeout must be -s KILL.
setsid nohup timeout -s KILL 172800 "$LAUNCH_PYTHON_BIN" -u \
  scripts/reinforcement_learning/rl_games/train.py \
  --task "$TASK" \
  --checkpoint "$CKPT" \
  --num_envs "$NUM_ENVS" \
  --seed 42 \
  --max_iterations "$MAX_ITERATIONS" \
  --headless \
  --track true \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-project-name "$WANDB_PROJECT" \
  --wandb-name "$WANDB_NAME" \
  agent.params.config.minibatch_size="$MINIBATCH" \
  env.curriculum.adr.params.init_difficulty=10 \
  agent.params.config.kl_threshold="$KL_THRESHOLD" \
  >> "$LOG" 2>&1 < /dev/null &

PID=$!
echo "LAUNCHED pid $PID, log $LOG"

kill_run() {
  # Kill the process GROUP: launched under setsid, so $PID is the `timeout` leader and Isaac is its
  # child. Killing only the wrapper leaves Isaac holding VRAM with ppid 1.
  kill -9 -"$PID" 2>/dev/null || kill -9 "$PID" 2>/dev/null
  sleep 5
}

DEADLINE=$(( SECONDS + 1200 )); STATUS=TIMEOUT
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  sleep 15
  grep -aqE "OutOfMemoryError|out of memory|Traceback|Error executing job|PhysX ABORT error" "$LOG" && { STATUS=CRASHED; break; }
  grep -aqE "epoch: [0-9]+/" "$LOG" && { STATUS=TRAINING; break; }
  kill -0 "$PID" 2>/dev/null || { STATUS=EXITED; break; }
done

echo "LAUNCH_STATUS=$STATUS"
if [ "$STATUS" != "TRAINING" ]; then
  echo "-- last errors in log --"
  grep -anE "out of memory|OutOfMemoryError|Traceback|Error executing job|PhysX ABORT error|ValueError|KeyError|AssertionError|TypeError" "$LOG" | tail -8
  echo "-- tail of log --"
  tail -30 "$LOG"
  kill_run
  exit 1
fi

# -- Confirm each health marker explicitly BY ITS NUMBER, never by a label or an "OK" line. A check
# that matches the label rather than the value is the failure mode that has cost this project the
# most time. Note the tilt marker matches the WHOLE formatted number: an earlier version of this
# check used a character class that excluded the decimal point and turned "+-0.3000 rad" into "+-0",
# manufacturing its own mismatch and killing a correctly staged run.
FAIL=0
for marker in \
  '\[dexlift\] gravity PINNED at \(\(0.0, 0.0, -9.81\)' \
  "\[dexlift\] POSE_TILT staged: .*limited to \+-0\.3000 rad" \
  "\[dexlift\] GOAL VERTICAL MIXTURE staged: ${VERTICAL_PROB_FMT} of goals drawn tip-down" \
  "\[dexlift\] GOAL VERTICAL ANCHOR_XY: ${DEXLIFT_GOAL_VERTICAL_ANCHOR_XY:-inherit}" \
  '\[dexlift\] reference HAND actuators \(30 N\.m / 10000 rad/s' \
  ; do
  if grep -aqE "$marker" "$LOG"; then
    echo "HEALTH_OK: $marker"
  else
    echo "HEALTH_MISSING: $marker"
    FAIL=1
  fi
done

if [ "$FAIL" -ne 0 ]; then
  echo "HEALTH CHECK FAILED -- killing and leaving the log for diagnosis: $LOG"
  kill_run
  exit 1
fi

echo "ALL_HEALTH_CHECKS_PASSED"
grep -aoE "epoch: [0-9]+/[0-9]+" "$LOG" | tail -1
grep -aoE "fps total: [0-9]+" "$LOG" | tail -1
nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader
echo "Training is running in the background, pid $PID, log $LOG. This script does not wait for it."
