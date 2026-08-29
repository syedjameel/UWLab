#!/usr/bin/env bash
# Exercise the v2 launcher scripts' REFUSING guards. Runs in ~1 s, needs no GPU and no Isaac.
#
#   bash scripts_v2/tools/test_launcher_scripts.sh
#
# === WHY THIS EXISTS, AND WHY `bash -n` IS NOT ENOUGH ===
#
# `ramp_stage.sh` once contained:
#
#     CKPT=${CKPT:?set CKPT to the previous stage's checkpoint (R1 starts from ep_3600)}
#
# The apostrophe inside ${VAR:?message} opens a single-quoted string, and bash swallows everything
# up to the NEXT apostrophe anywhere in the file. The count happened to balance, so **`bash -n`
# PASSED** -- on a file whose middle had been eaten. The `check()` function definition and every
# REFUSING guard after line 69 had become string content. A run pointed at a nonexistent REPO_DIR
# went straight past the guard that exists to stop it and died 130 lines later on
# `check: command not found`. It was caught only because a later edit added one more apostrophe and
# flipped the parity into a visible syntax error.
#
# `bash -n` answers "is this file parseable". It cannot answer "does this file still contain the
# code I wrote", and those came apart. That is the V2_POSE_FINDINGS.md F27 defect class in a
# TEST: a check that is individually valid and wrong against the thing it was standing in for.
#
# BE PRECISE ABOUT WHAT `bash -n` DOES HERE, because "it never catches this" would be wrong and
# equally misleading. Whether it catches the bug depends on the PARITY of apostrophes elsewhere in
# the same file: with an even count the swallow closes again and the file parses (silently broken);
# with an odd count it is a hard syntax error. Reintroducing the bug into the CURRENT file does make
# `bash -n` fail -- and it passed on the version that shipped. The detector's verdict moves with
# text that has nothing to do with the defect, which is exactly why it cannot be relied on and why
# the lint below tests for the construct itself.
#
# So this suite asserts on BEHAVIOUR -- each guard must actually print its refusal and exit
# nonzero. A swallowed guard fails here even when the file parses.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PASS=0
FAIL=0

# Every case runs with a REPO_DIR that exists (this repo) so the script gets past its own cd, and
# a PYTHON_BIN that is a real executable, so the guard under test is the one that fires.
REPO=$(cd "$HERE/../.." && pwd)
FAKE_PY=$(command -v bash)

expect_refusal() {  # expect_refusal <label> <expected substring> <script> [env assignments...]
  local label="$1" want="$2" script="$3"; shift 3
  local out rc
  out=$(env "$@" bash "$script" 2>&1); rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "FAIL [$label]: expected a nonzero exit, got 0"
    FAIL=$((FAIL + 1)); return
  fi
  case "$out" in
    *"$want"*) echo "PASS [$label]"; PASS=$((PASS + 1)) ;;
    *) echo "FAIL [$label]: expected output containing '$want'"
       echo "$out" | head -5 | sed 's/^/        /'
       FAIL=$((FAIL + 1)) ;;
  esac
}

echo "== ramp_stage.sh =="
# The guard the swallowed-quote bug disabled. This case is the regression test for it.
expect_refusal "ramp: missing REPO_DIR" "REFUSING: REPO_DIR=" "$HERE/ramp_stage.sh" \
  STAGE=R1 CKPT=/etc/hostname REPO_DIR=/nonexistent-repo-dir
expect_refusal "ramp: unknown STAGE" "REFUSING: unknown STAGE" "$HERE/ramp_stage.sh" \
  STAGE=R9 CKPT=/etc/hostname
expect_refusal "ramp: STAGE unset" "set STAGE" "$HERE/ramp_stage.sh" \
  CKPT=/etc/hostname
expect_refusal "ramp: CKPT unset" "set CKPT" "$HERE/ramp_stage.sh" \
  STAGE=R1
expect_refusal "ramp: missing checkpoint file" "REFUSING: checkpoint" "$HERE/ramp_stage.sh" \
  STAGE=R1 CKPT=/nonexistent-ckpt.pth REPO_DIR="$REPO" PYTHON_BIN="$FAKE_PY"
# dr-tlx.9: a minibatch that does not divide num_envs*horizon silently reshapes the batch.
expect_refusal "ramp: minibatch does not divide" "REFUSING: minibatch" "$HERE/ramp_stage.sh" \
  STAGE=R1 CKPT=/etc/hostname REPO_DIR="$REPO" PYTHON_BIN="$FAKE_PY" MINIBATCH=70000
# O14 / sec 6.2: one 80 GB card holds 16384 envs, not 32768.
expect_refusal "ramp: 1 GPU with 2-GPU env count" "REFUSING: NPROC=" "$HERE/ramp_stage.sh" \
  STAGE=R1 CKPT=/etc/hostname REPO_DIR="$REPO" PYTHON_BIN="$FAKE_PY" NPROC=1 NUM_ENVS=32768
# ~/ckpt holds the only copy of the certified baselines (RESET_SPEC_V2.md sec 6a).
expect_refusal "ramp: refuses to write into ~/ckpt" "REFUSING: OUT_DIR is inside" "$HERE/ramp_stage.sh" \
  STAGE=R1 CKPT=/etc/hostname REPO_DIR="$REPO" PYTHON_BIN="$FAKE_PY" OUT_DIR="$HOME/ckpt/scratch"

echo "== r0_control.sh =="
# r0_control.sh takes its stage as a POSITIONAL argument, which expect_refusal (env-only) cannot
# express -- so this one case is written out longhand rather than bent to fit the helper.
out=$(bash "$HERE/r0_control.sh" bogus 2>&1); rc=$?
if [ "$rc" -ne 0 ] && [ "${out#*REFUSING: stage must be}" != "$out" ]; then
  echo "PASS [r0: bad positional stage]"; PASS=$((PASS + 1))
else
  echo "FAIL [r0: bad positional stage]"; FAIL=$((FAIL + 1))
fi
expect_refusal "r0: missing REPO_DIR" "REFUSING: REPO_DIR=" "$HERE/r0_control.sh" \
  REPO_DIR=/nonexistent-repo-dir

echo "== wandb_sync_loop.sh =="
expect_refusal "wandb: missing dir argument" "usage: wandb_sync_loop.sh" "$HERE/wandb_sync_loop.sh"

echo "== r0_control.sh: record_timing emits valid JSON (closes O19) =="
# The number sec 6.3's cut ordering depends on is written by this function after a ~20-minute
# certification. A malformed heredoc here would be discovered only at the END of that run, which is
# the most expensive moment to find it. So the function is extracted and exercised standalone.
TMPJ=$(mktemp); TMPS=$(mktemp)
{
  echo 'set -uo pipefail'
  echo 'CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}'
  sed -n '/^record_timing()/,/^}/p' "$HERE/r0_control.sh"
  echo "record_timing $TMPJ certification 1187 num_envs 256 episodes 512"
} > "$TMPS"
if bash "$TMPS" >/dev/null 2>&1 && python3 -c "
import json,sys
d=json.load(open('$TMPJ'))
assert d['wall_clock_s']==1187, d
assert abs(d['wall_clock_min']-19.78)<0.01, d
assert d['rollout_only_s_approx']==1095, d      # total minus the measured 92 s Isaac startup
assert d['num_envs']==256 and d['episodes']==512, d
" 2>/dev/null; then
  echo "PASS [r0: record_timing emits valid JSON with correct fields]"; PASS=$((PASS + 1))
else
  echo "FAIL [r0: record_timing produced invalid or wrong JSON]"
  cat "$TMPJ" 2>/dev/null | head -20 | sed 's/^/        /'
  FAIL=$((FAIL + 1))
fi
rm -f "$TMPJ" "$TMPS"

echo "== lint: no apostrophe inside \${VAR:?...} or \${VAR:-...} =="
# The specific construct that caused the swallowed-quote bug, checked directly rather than hoped
# about. An earlier draft of this block counted REFUSING lines with grep and CLAIMED to be reading
# the parsed token stream -- it was not, and a grep counts strings inside a swallowed literal just
# as happily as real code, so it would have passed on the very file it was meant to catch. Replaced
# with a lint for the actual hazard.
#
# The behavioural cases above are what prove the guards RUN; this proves the hazard that disabled
# them cannot silently return.
for script in ramp_stage.sh r0_control.sh wandb_sync_loop.sh reconcile_remote_checkout.sh launch_c3_rung_smoke.sh test_launcher_scripts.sh; do
  # Exclude comment lines: this file documents the bad form on purpose, in the header above.
  hits=$(grep -n '\${[A-Za-z_][A-Za-z_0-9]*:[?-][^}]*'"'" "$HERE/$script" | grep -v '^[0-9]*: *#' || true)
  if [ -z "$hits" ]; then
    echo "PASS [$script: no quote-fragile parameter expansion]"; PASS=$((PASS + 1))
  else
    echo "FAIL [$script: apostrophe inside \${VAR:?...}/\${VAR:-...} -- bash will swallow from here]"
    echo "$hits" | sed 's/^/        /'
    FAIL=$((FAIL + 1))
  fi
done

echo
echo "passed=$PASS failed=$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
echo "all launcher guards fire"
