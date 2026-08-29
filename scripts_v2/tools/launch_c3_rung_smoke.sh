#!/usr/bin/env bash
# scripts_v2/tools/launch_c3_rung_smoke.sh -- GPU smoke launcher for the C3 RUNG stage's
# Isaac-touching half (bead dr-ai1.4, commit 922c3d3). See smoke_c3_rung_isaac.py's module
# docstring for WHY this exists (the per-env S1/S_t tensor draw and dispatch, c3_rung.py:195-231,
# has never been imported by a Python interpreter) and analyze_c3_rung_smoke.py's for what gets
# asserted from the output.
#
# Usage:
#   launch_c3_rung_smoke.sh <gpu-index> [num_envs] [rounds]
#     e.g. launch_c3_rung_smoke.sh 1
#          launch_c3_rung_smoke.sh 1 256 4
#          MODE=settle SETTLE_MARGIN=60 launch_c3_rung_smoke.sh 1 256 2   # phase 2, bead dr-ai1.20
#
# MODE=reset (default, bead dr-ai1.4) or MODE=settle (bead dr-ai1.20, exercises the deferred S_t
# goal re-pin -- needs stepping, so budget more wall time and prefer fewer --rounds).
#
# GPU INDEX IS REQUIRED, NOT DEFAULTED, ON PURPOSE -- per this box's own compute policy: run
# `nvidia-smi` immediately before firing this, pick a device with low Volatile GPU-Util AND small
# Memory-Used (< ~10% capacity), and pass THAT index. Never launch on a GPU already running another
# user's process. This script does not pick one for you.
#
# THIS SCRIPT ONLY WRITES/PREPARES -- launching it is a separate, later action (do not run this
# from an agent session without explicit authorization; see the task this was written under).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."
cd "$REPO_ROOT" || exit 1

GPU="${1:?Usage: launch_c3_rung_smoke.sh <gpu-index> [num_envs] [rounds] -- pick the gpu-index from a FRESH nvidia-smi, do not reuse a stale one}"
NUM_ENVS="${2:-256}"
ROUNDS="${3:-4}"
S1_FRACTION="${S1_FRACTION:-0.5}"
POSE_TILT="${POSE_TILT:-0.3}"
MODE="${MODE:-reset}"
SETTLE_MARGIN="${SETTLE_MARGIN:-60}"
if [ "$MODE" != "reset" ] && [ "$MODE" != "settle" ]; then
  echo "REFUSING: MODE must be 'reset' or 'settle', got '$MODE'"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl"

# -- MANDATORY: /tmp/uwlab and /tmp/isaaclab/logs are owned by another user on this shared box.
# Without BOTH of these exported, the job dies before its first iteration with a PermissionError
# that looks exactly like a job that never launched (TMPDIR is NOT covered by UWLAB_TMP_ROOT --
# IsaacLab's own logger calls tempfile.gettempdir() directly). Measured, not assumed: this is the
# same failure mode launch_dexreset_s1_s2_bank_gen.sh and every measurement script in this
# directory guards against.
export UWLAB_TMP_ROOT="${UWLAB_TMP_ROOT:-$HOME/tmp_uwlab}"
export TMPDIR="${TMPDIR:-$HOME/tmp_local}"
mkdir -p "$UWLAB_TMP_ROOT" "$TMPDIR"

# -- THE STAGE THIS SMOKE EXERCISES. DEXRESET_ST_SPAWN_TIPDOWN is deliberately NOT exported here --
# it is surplus for C3 (F51) and smoke_c3_rung_isaac.py REFUSES outright if it is ever set to "1"
# in its environment, so an accidental inherited value from the caller's shell fails loudly instead
# of silently changing what is measured.
export DEXRESET_C3_RUNG=1
unset DEXRESET_ST_SPAWN_TIPDOWN || true

TASK="DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-Play-v0"
PYTHON_BIN="${PYTHON_BIN:-/home/dom_iva/venv_uwlab/bin/python}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/logs"
mkdir -p "$OUT_DIR"
NPZ="$OUT_DIR/c3_rung_smoke_${MODE}_${TS}.npz"
LOG="$OUT_DIR/c3_rung_smoke_${MODE}_${TS}.log"
ANALYSIS_LOG="$OUT_DIR/c3_rung_smoke_${MODE}_${TS}_analysis.log"

echo "GPU=$GPU MODE=$MODE NUM_ENVS=$NUM_ENVS ROUNDS=$ROUNDS S1_FRACTION=$S1_FRACTION POSE_TILT=$POSE_TILT"
[ "$MODE" = "settle" ] && echo "SETTLE_MARGIN=$SETTLE_MARGIN"
echo "npz -> $NPZ"
echo "log -> $LOG"

# -- ~92 s of Isaac startup, measured (not assumed), before the first env.reset() prints anything.
# MODE=reset: num_envs*rounds resets, no stepping -- startup-dominated, 900 s is ample headroom.
# MODE=settle: additionally steps (SETTLE_STEPS + SETTLE_MARGIN) times per round with zero actions
# -- budget more; 1800 s covers a slow boot plus a generous settle window at default num_envs/rounds.
SMOKE_TIMEOUT=900
EXTRA_ARGS=()
if [ "$MODE" = "settle" ]; then
  SMOKE_TIMEOUT=1800
  EXTRA_ARGS+=(--mode settle --settle_margin "$SETTLE_MARGIN")
fi

timeout -s KILL "$SMOKE_TIMEOUT" "$PYTHON_BIN" -u scripts_v2/tools/smoke_c3_rung_isaac.py \
  --task "$TASK" \
  --num_envs "$NUM_ENVS" --rounds "$ROUNDS" \
  --s1_fraction "$S1_FRACTION" --pose_tilt "$POSE_TILT" \
  "${EXTRA_ARGS[@]}" \
  --out "$NPZ" --headless \
  > "$LOG" 2>&1
SMOKE_EXIT=$?
echo "EXIT_CODE=$SMOKE_EXIT" >> "$LOG"
echo "smoke exit code: $SMOKE_EXIT (log: $LOG)"

if [ "$SMOKE_EXIT" -ne 0 ]; then
  echo "REFUSING to analyze: the Isaac smoke itself failed (exit $SMOKE_EXIT). See $LOG."
  exit "$SMOKE_EXIT"
fi

# -- Do not trust $SMOKE_EXIT alone. 2026-08-29, DL_H100: smoke_c3_rung_isaac.py's own refusal
# guard raised SystemExit (a real, intentional refusal, message printed to the log) but the process
# that reached this line still reported exit 0 -- something downstream of the raised exception,
# most likely Isaac/Kit's own teardown, overrode the interpreter's exit code during shutdown. That
# is now worked around at the source (smoke_c3_rung_isaac.py's _refuse() calls os._exit(1)
# directly, bypassing whatever swallowed it), but this check stays as a second, independent signal
# so a similar swallow anywhere else in the chain cannot repeat the same failure: the smoke's only
# real job is to produce this npz, so if it is not there, nothing downstream can be trusted
# regardless of what $SMOKE_EXIT claims.
if [ ! -s "$NPZ" ]; then
  echo "REFUSING to analyze: no npz was produced at $NPZ (smoke reported exit $SMOKE_EXIT, but the"
  echo "file is missing or empty). Do not trust a 0 exit code alone -- see $LOG."
  exit 1
fi

# -- Analysis is Isaac-free (numpy + stdlib only, no AppLauncher, no GPU) -- it does NOT need the
# Isaac venv to run. It DOES need an interpreter with numpy on it, though, and checked directly on
# the box: the system `python3` (/usr/bin/python3, 3.12.3) has no numpy installed at all --
# ModuleNotFoundError before a single line of analysis runs. venv_uwlab is the only interpreter
# confirmed to have numpy here, so default to reusing PYTHON_BIN (it just never launches Isaac from
# it -- analyze_c3_rung_smoke.py has no AppLauncher import). Override ANALYSIS_PYTHON_BIN if a
# lighter numpy-only venv shows up later; don't switch back to bare `python3` without re-checking
# what's actually installed on whatever box runs this.
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-$PYTHON_BIN}"
"$ANALYSIS_PYTHON_BIN" scripts_v2/tools/analyze_c3_rung_smoke.py \
  --npz "$NPZ" --log "$LOG" \
  --expected_s1_fraction "$S1_FRACTION" --expected_pose_tilt "$POSE_TILT" \
  | tee "$ANALYSIS_LOG"
ANALYSIS_EXIT="${PIPESTATUS[0]}"
echo "analysis exit code: $ANALYSIS_EXIT (log: $ANALYSIS_LOG)"
exit "$ANALYSIS_EXIT"
