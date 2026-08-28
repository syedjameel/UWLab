#!/usr/bin/env bash
# Certification for epic UWLab-nnlv's vertical-goal MIXTURE finetune -- PART 2 OF THE GATE, the
# "did it FORGET?" half. Part 1 ("did it LEARN?") is scripts_v2/tools/measure_vertical_hold.py and
# answers a different question on a different task; neither half alone decides anything.
#
# STAGED, NOT RUN. Two arms are still training (vertical_prob 0.50 and 0.25, warm-started from
# ep3600). Do not launch this until they have written checkpoints.
#
# DO NOT JUDGE THE FINETUNE BY TRAINING REWARD. This project has a documented case where the WORSE
# policy had the HIGHER training reward (42.8 against 38.4) and certified 24 points LOWER. Here the
# trap is sharper still: BOTH arms are expected to show a LOWER reward than the parent's 38.39
# simply because a fraction of their episodes carry a goal the parent never had to reach, and the
# lower-mixture arm is expected to sit higher than the higher-mixture arm for the same reason. That
# ordering is an artefact of the goal distribution, not evidence about skill. This certification --
# run on the CLASSIC-ONLY goal distribution, identically for both arms and the control -- is the
# only number that says whether the original skill survived.
#
# WHAT THIS IS GUARDING AGAINST, concretely. A 100-percent goal-at-spawn finetune of THIS SAME
# parent policy lost 55 percent of its skill in 50 epochs and certified 0.0000 at 30 mm by epoch
# 1550, with the damage FASTEST AT THE START. The mixture exists to prevent exactly that. So the
# question this script answers is not "is the finetune better" but "is the parent's skill still
# there" -- and the failure it is looking for is one that a rising training-reward curve hides.
#
# =====================================================================================
# PROTOCOL -- identical to cert_g3z4_finetune.sh, deliberately, so the numbers are comparable
# =====================================================================================
#   - 128 episodes total, seeds 101/202/303/404 (certify_pose.py's default list), 32 per seed.
#   - Deterministic policy: certify_pose.py hard-codes is_deterministic=True. No flag, nothing to
#     forget to pass.
#   - ADR pinned to max, passed EXPLICITLY through run_certify.sh rather than left to a default.
#     Every CERTIFY_RESULT line carries adr_difficulty_frac; if it reads anything but full
#     difficulty the number is not comparable to the stored 0.6953 and must not be quoted against it.
#   - The BASE Reorient task under the CLASSIC-ONLY goal distribution -- the same task id training
#     runs on, but with every goal-distribution toggle unset (see the defensive unsets below).
#   - The parent ep3600 is re-certified IN THE SAME BATCH as an in-batch control. Stored figure:
#     pass@30mm 0.6953. Measured replay reproducibility on an IDENTICAL checkpoint in this project
#     is about +-2 points at 10 mm and +-3.5 points at 30 mm. If the control does not land within
#     that band in THIS batch, the batch is not comparable to the stored certification and the
#     arms' numbers must be read only against the control run in this same batch.
#
# READING THE RESULT. Both arms are certified, and the choice between them is not "highest number
# wins": Part 1 (vertical hold) and Part 2 (retained skill) both have to be satisfied, and they pull
# in opposite directions by construction -- more vertical goals buys more of the new skill and risks
# more of the old. An arm that certifies within the control's noise band AND passes the vertical
# gate is the one to keep. An arm that passes the vertical gate while certifying clearly below the
# control has paid for the new rung with the task the whole pipeline depends on.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# -- FILL IN AFTER TRAINING PRODUCES CHECKPOINTS. No defaults, deliberately: a script that quietly
# certified nothing, or certified the wrong .pth, would be worse than one that refuses to run.
# Point these at the .pth files under logs/rl_games/dexlift_ur5e_delto_reljointpos_tableleg_reorient/
# <run timestamp>/nn/ that launch_dexreset_vertical_goal_finetune.sh's own log names.
ARM_A_CKPT=${ARM_A_CKPT:?"set ARM_A_CKPT=<path to the vertical_prob 0.50 .pth> before running"}
ARM_B_CKPT=${ARM_B_CKPT:?"set ARM_B_CKPT=<path to the vertical_prob 0.25 .pth> before running"}

# -- Control checkpoint: same top-of-script-variable pattern as the sibling cert scripts, for the
# identical portability reason -- do not assume any particular home layout on the box this runs on.
# Override CONTROL_CKPT=<path> if it differs.
CONTROL_CKPT=${CONTROL_CKPT:?"set CONTROL_CKPT=<path to the ep3600 parent .pth> before running"}
CONTROL_SHA256_EXPECT="9534e102a64fc06a3588c5102fc3421e69ef1b18fe30e7ffb40f7df63b5d76af"

GPU=${GPU:-0}
USED=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
[ "${USED:-999999}" -lt 2000 ] || { echo "REFUSING: GPU $GPU has ${USED} MiB already in use"; exit 1; }
# NOT a tidiness check. Two Isaac processes sharing one GPU have been observed to return frozen-arm
# and falling-part garbage at exit code 0 -- corrupt physics with no error anywhere. A certification
# run on a busy GPU can produce a clean-looking number that means nothing.

for _ck in "$CONTROL_CKPT" "$ARM_A_CKPT" "$ARM_B_CKPT"; do
  [ -f "$_ck" ] || { echo "REFUSING: checkpoint not found at $_ck"; exit 1; }
done
CONTROL_SHA="$(sha256sum "$CONTROL_CKPT" | awk '{print $1}')"
ARM_A_SHA="$(sha256sum "$ARM_A_CKPT" | awk '{print $1}')"
ARM_B_SHA="$(sha256sum "$ARM_B_CKPT" | awk '{print $1}')"

[ "$CONTROL_SHA" = "$CONTROL_SHA256_EXPECT" ] || {
  echo "REFUSING: control checkpoint sha256 mismatch -- this is the anchor the whole comparison"
  echo "hangs on, so a mismatch invalidates the batch rather than merely warning."
  echo "  expected: $CONTROL_SHA256_EXPECT"
  echo "  got:      $CONTROL_SHA"
  exit 1
}

# ALL THREE CHECKPOINTS ARE HASHED AND COMPARED, not just the control. Both arms write a file named
# last_..._ep_NNNN_rew_X.pth into two DIFFERENT timestamped directories under the SAME logs root, so
# the two paths differ only by a timestamp component and a reward suffix -- they are easy to confuse
# by eye and impossible to tell apart from the JSON afterwards. Two specific mistakes this catches:
#   * the same run directory pasted for both arms -- A and B then certify ONE policy, agree to
#     within replay noise, and "both mixtures retained the skill equally" gets concluded from a
#     single checkpoint;
#   * the CONTROL path pasted into an arm -- that arm certifies ~0.6953, the control's own sha guard
#     above still passes because it only ever looked at CONTROL_CKPT, and the finetune is certified
#     as having lost nothing. That is the exact conclusion this script exists to test, produced by
#     the parent policy wearing the arm's name.
[ "$ARM_A_SHA" != "$ARM_B_SHA" ] || {
  echo "REFUSING: arm A and arm B are the SAME file (sha256 $ARM_A_SHA)."
  echo "  A: $ARM_A_CKPT"
  echo "  B: $ARM_B_CKPT"
  exit 1
}
for _pair in "A:$ARM_A_SHA:$ARM_A_CKPT" "B:$ARM_B_SHA:$ARM_B_CKPT"; do
  _name=${_pair%%:*}; _rest=${_pair#*:}; _sha=${_rest%%:*}; _path=${_rest#*:}
  [ "$_sha" != "$CONTROL_SHA256_EXPECT" ] || {
    echo "REFUSING: arm $_name is the PARENT checkpoint (sha256 matches the ep3600 control)."
    echo "  $_path"
    echo "  Certifying the parent under an arm's name would report zero forgetting by construction."
    exit 1
  }
done

echo "control checkpoint sha256 verified: $CONTROL_SHA"
echo "arm A (vertical_prob 0.50): $ARM_A_CKPT"
echo "  sha256 $ARM_A_SHA"
echo "arm B (vertical_prob 0.25): $ARM_B_CKPT"
echo "  sha256 $ARM_B_SHA"
echo "no known-good hash exists for either arm -- they are what this script exists to evaluate;"
echo "the checks above only establish that the three files are three DIFFERENT policies."

DRIVER_LOG="$REPO_ROOT/logs/cert_nnlv_vertgoal_$(date +%Y%m%d_%H%M%S).driver.log"
mkdir -p "$(dirname "$DRIVER_LOG")"
: > "$DRIVER_LOG"

# DEFENSIVE UNSETS -- every one of these changes WHAT TASK IS SCORED, and every one is an opt-in env
# var that survives in an interactive shell. Leaking any of them in from having sourced a launcher
# would inflate the numbers while the output stayed entirely clean-looking.
#
# The first four are inherited verbatim from cert_g3z4_finetune.sh's list and its reasoning: a
# partial-assembly episode pins the goal to the object's OWN SPAWN POSE, so it is trivially
# "successful" at every tolerance the instant the leg is grasped, with no transport required.
unset DEXLIFT_EPISODE_MIXTURE
unset DEXLIFT_GOAL_AT_SPAWN
unset DEXLIFT_PARTIAL_ASSEMBLY
unset DEXLIFT_SPAWN_CLEARANCE
# These three are NEW IN THIS EPIC and are the ones a run of this script is most likely to leak,
# because the operator will have just been running the launcher that sets them. With
# DEXLIFT_GOAL_VERTICAL_PROB set, a fraction of scored episodes would carry a vertical goal --
# scoring each arm partly on the task it was just trained on, which is the precise way to make a
# forgetting check unable to detect forgetting. Unsetting is EXACT, and the reason is stronger than
# "the module treats prob 0.0 as inert" -- that is true but is NOT the path protecting this run.
# _apply_goal_vertical_mixture reads the variable and `if raw is None: return` BEFORE touching any
# config, so with it unset the command term is never swapped: it stays TaskStateVisPoseCommandCfg
# and MixedGoalPoseCommandCfg is never constructed at all. No default anywhere activates the
# mixture, and certify_pose.py builds the cfg through parse_env_cfg rather than Hydra, so there is
# no override path either.
unset DEXLIFT_GOAL_VERTICAL_PROB
unset DEXLIFT_GOAL_VERTICAL_TILT
unset DEXLIFT_GOAL_VERTICAL_Z
# GOAL_BELOW_SPAWN_MM is a generation-time shaping device (bead UWLab-ck2b) and has no business in a
# certification. Unlike the vars above this one is belt-and-braces rather than load-bearing: it
# RAISES without DEXLIFT_PARTIAL_ASSEMBLY=1 (dexlift_ur5e_delto_tableleg_env_cfg.py:377), which the
# unset above guarantees, so a leak would crash rather than silently rescore. Unset anyway, because
# a guard that depends on another guard staying correct is not one I want to reason about later.
unset DEXLIFT_GOAL_BELOW_SPAWN_MM
# DROP_Z IS THE ONE REAL GAP IN THIS LIST, and it is the only variable here that can silently move
# the number without crashing. _apply_pose_tilt_stage reads it (dexlift_ur5e_delto_env_cfg.py:932)
# and rewrites the OBJECT SPAWN drop range: pose_range["z"] = [0.0, drop], default 0.05. That path
# is GUARANTEED to run in this script because TILT=0.3 is exported below, and run_certify.sh only
# ever SETS DEXLIFT_DROP_Z from DROP_Z -- unlike the collider vars, it never unsets it. run_certify's
# own header calls this out as a footgun and then does not defend against it.
#   A LARGE leak (0.4) depresses all three runs together, the control misses its +-3.5 point band,
#   and the batch is correctly thrown out. A SMALL leak (0.02, 0.1) moves all three by LESS than the
#   noise band, so nothing trips, and the whole batch scores a task that neither the arms nor the
#   parent's stored 0.6953 were measured on. The quiet one is the dangerous one.
# Evidence the stored control had it unset: certify_pose.py dumps every DEXLIFT_* it sees into
# plant.dexlift_env, and the stored cert JSON carries no DEXLIFT_DROP_Z key.
unset DEXLIFT_DROP_Z
unset DROP_Z

# BOX LAYOUT. run_certify.sh used to hard-code three paths -- the repo at $HOME/UWLab_ur5edelto, the
# interpreter at $HOME/UWLab/env_uwlab/bin/python, the entry point at $HOME/certify_pose.py -- and ALL
# THREE ARE ABSENT on the box this epic's training actually runs on (verified by listing them, not
# assumed). It is now parameterised with those exact values as defaults, so nothing else changes, and
# this driver supplies its own. Left unfixed, the three runs below would each have died on `cd ... ||
# exit 1`, the status would have been swallowed by the pipe into tee, and this script would have
# printed CERT_NNLV_ALL_DONE having certified nothing whatsoever.
REPO_DIR_FOR_CERT=${REPO_DIR_FOR_CERT:-$REPO_ROOT}
PYTHON_BIN_FOR_CERT=${PYTHON_BIN_FOR_CERT:?"set PYTHON_BIN_FOR_CERT=<path to the python that has isaaclab installed on this box>"}
CERTIFY_PY_FOR_CERT=${CERTIFY_PY_FOR_CERT:-$REPO_ROOT/scripts_v2/tools/certification/certify_pose.py}
OUT_DIR_FOR_CERT=${OUT_DIR_FOR_CERT:-$REPO_ROOT/logs/cert_nnlv}
[ -x "$PYTHON_BIN_FOR_CERT" ] || { echo "REFUSING: PYTHON_BIN_FOR_CERT=$PYTHON_BIN_FOR_CERT is not executable"; exit 1; }
[ -f "$CERTIFY_PY_FOR_CERT" ] || { echo "REFUSING: CERTIFY_PY_FOR_CERT=$CERTIFY_PY_FOR_CERT not found"; exit 1; }
mkdir -p "$OUT_DIR_FOR_CERT"

FAILED_RUNS=()

run() {  # tag  checkpoint  expected_sha256
  local tag="$1" ckpt="$2" want_sha="$3" rc=0
  local out_json="$OUT_DIR_FOR_CERT/cert_$tag.json"
  local out_log="$OUT_DIR_FOR_CERT/cert_$tag.log"
  echo "=== $tag  $(date -u +%H:%M:%SZ)" | tee -a "$DRIVER_LOG"
  echo "CKPT=$ckpt" | tee -a "$DRIVER_LOG"

  # DELETE THE PREVIOUS RESULT BEFORE RUNNING, and this is the single most important line in the
  # function. run_certify.sh writes to a FIXED path per tag, and certify_pose.py writes the JSON
  # ONLY on success while the .log is truncated by `>` on every attempt. So a re-run whose python
  # dies leaves a FRESH log sitting next to a STALE JSON from the previous batch -- and every
  # existence/parse check downstream passes on it. The control is the file most likely to be reused
  # unchanged across re-runs, which means the leg that decides "is this batch comparable" is the leg
  # most likely to be silently answering for a different batch.
  rm -f "$out_json" "$out_log"

  (
    export TASK=DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0
    export COLLIDERS=hullfix3 SELFCOLL=on HAND=ref ARM=ours
    export ADR_DIFFICULTY=max
    export REPO_DIR="$REPO_DIR_FOR_CERT" PYTHON_BIN="$PYTHON_BIN_FOR_CERT"
    export CERTIFY_PY="$CERTIFY_PY_FOR_CERT" OUT_DIR="$OUT_DIR_FOR_CERT"
    # TILT=0.3 is the plant/task staging the arms trained under, and is kept because certification
    # must measure the policy under the plant it actually learned in.
    # WHAT IT ACTUALLY WRITES, because an earlier revision of this comment said "the classic goal
    # band only" and that is FALSE in a way that matters: _apply_pose_tilt_stage writes THREE
    # things -- the OBJECT SPAWN roll/pitch/yaw range, the goal band's roll/pitch, AND the spawn
    # DROP HEIGHT (pose_range["z"] = [0.0, DEXLIFT_DROP_Z or 0.05]). The spawn narrowing is the
    # whole point of the staging, not a side effect. That third write is exactly why DROP_Z is in
    # the unset list above, and the old wording was the sentence that would have talked a reader
    # out of putting it there.
    # What IS true: this never touched the vertical band. That lived in MixedGoalPoseCommandCfg's
    # own `vertical_ranges` field, which _apply_pose_tilt_stage does not write, precisely so the
    # two could not clobber one another.
    export GPU NUM_ENVS=128 EPISODES=128 POS_TOL=0.03 TILT=0.3
    bash "$REPO_ROOT/scripts_v2/tools/certification/run_certify.sh" "$tag" "$ckpt"
  ) 2>&1 | tee -a "$DRIVER_LOG"
  # `set -o pipefail` makes this the subshell's status rather than tee's. WITHOUT capturing it here
  # the status is simply discarded: no -e is set (deliberately -- one failed arm should not abandon
  # the others), so the script would sail on and print its completion marker regardless. The sibling
  # drivers print a "rc above" line that reports no rc at all; this one records the actual outcome.
  rc=${PIPESTATUS[0]}
  echo "--- $tag rc=$rc ---" | tee -a "$DRIVER_LOG"

  # THE EXIT CODE HERE IS NEARLY MEANINGLESS, so it is checked but never trusted alone.
  # run_certify.sh's LAST command is `grep -aE "RESULT|Traceback|Error|inference tensor" "$LOG"`,
  # and that grep's status IS the script's status. A run that died with a traceback MATCHES the
  # pattern, so grep exits 0, so run_certify.sh exits 0. "Crashed at seed 3" and "completed 4 seeds"
  # are indistinguishable by exit code -- the only way to get non-zero out of it is a log with no
  # match whatsoever. This is on top of the general rule here that an Isaac process has been seen
  # printing EXIT_CODE=0 over a fatal traceback.
  #
  # So the run is verified by POSITIVE EVIDENCE, four checks, each closing a case the others miss:
  #   1. the JSON exists and parses            -- it is written only on success, and rm'd above, so
  #                                               its presence means THIS attempt got to the end;
  #   2. its checkpoint_sha256 equals the .pth we passed -- catches a stale file that somehow
  #                                               survived, and catches a mixed-up path;
  #   3. episodes == 128                       -- catches a short run that ended early but still
  #                                               serialised;
  #   4. a CERTIFY_RESULT line for this tag is in the log -- the script's own completion marker.
  if [ "$rc" -ne 0 ]; then
    FAILED_RUNS+=("$tag (rc=$rc)")
    return
  fi
  if [ ! -s "$out_json" ]; then
    FAILED_RUNS+=("$tag (no result JSON at $out_json -- the run did not reach the end)")
    return
  fi
  local verdict
  verdict=$(python3 - "$out_json" "$want_sha" <<'PYEOF'
import json, sys
path, want = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
except Exception as exc:
    print(f"result JSON does not parse: {exc}")
    raise SystemExit(0)
got = d.get("checkpoint_sha256")
if got != want:
    print(f"result JSON is for a DIFFERENT checkpoint: recorded {got}, passed {want}")
    raise SystemExit(0)
# `episodes` lives under "summary", NOT at the top level. An earlier revision read
# d.get("episodes") and got None for every run, so all three legs of a batch whose
# CERTIFY_RESULT lines were perfectly good got marked FAILED. The check was right to exist and
# wrong about where the field is; a guard that cannot pass on healthy input is as useless as one
# that cannot fail on broken input, so it now reads the real path and says what it saw.
eps = (d.get("summary") or {}).get("episodes")
if eps != 128:
    print(f"result JSON's summary.episodes is {eps!r}, protocol requires 128")
    raise SystemExit(0)
print("OK")
PYEOF
)
  if [ "$verdict" != "OK" ]; then
    FAILED_RUNS+=("$tag ($verdict)")
    return
  fi
  if ! grep -aq "CERTIFY_RESULT" "$out_log"; then
    FAILED_RUNS+=("$tag (no CERTIFY_RESULT marker in $out_log)")
    return
  fi
  echo "    verified: $tag produced a complete result for sha256 $want_sha" | tee -a "$DRIVER_LOG"
}

# CONTROL FIRST. If it does not reproduce ~0.6953 (+-3.5 points) at 30 mm, the batch is not
# comparable and neither arm's number should be read against the stored figure.
run nnlv_control_ep3600 "$CONTROL_CKPT" "$CONTROL_SHA"
run nnlv_armA_vprob050  "$ARM_A_CKPT"   "$ARM_A_SHA"
run nnlv_armB_vprob025  "$ARM_B_CKPT"   "$ARM_B_SHA"

# THE COMPLETION MARKER IS EARNED, NOT PRINTED UNCONDITIONALLY. A driver that announces ALL_DONE
# after a run that never produced a number is the exact shape of a clean-looking wrong result this
# whole gate exists to avoid.
if [ "${#FAILED_RUNS[@]}" -ne 0 ]; then
  echo "CERT_NNLV_INCOMPLETE -- ${#FAILED_RUNS[@]} of 3 runs did not produce a usable result:" | tee -a "$DRIVER_LOG"
  for f in "${FAILED_RUNS[@]}"; do echo "  FAILED: $f" | tee -a "$DRIVER_LOG"; done
  echo "Do NOT read the surviving runs as a certification: the in-batch control is what makes these" \
       "numbers comparable, and a batch missing any of its three legs is not the protocol." | tee -a "$DRIVER_LOG"
  echo "Driver log: $DRIVER_LOG" | tee -a "$DRIVER_LOG"
  exit 1
fi

echo "CERT_NNLV_ALL_DONE" | tee -a "$DRIVER_LOG"
echo "Results (one cert_<tag>.json and cert_<tag>.log per run) are under $OUT_DIR_FOR_CERT --" \
     "read pass@0.03 (30 mm) from cert_nnlv_control_ep3600.json, cert_nnlv_armA_vprob050.json and" \
     "cert_nnlv_armB_vprob025.json, ALL FROM THIS SAME BATCH, per the protocol above." \
     "Driver log (all three runs' stdout/stderr): $DRIVER_LOG"

# HOW TO RUN THIS DETACHED: this driver does not background or timeout-wrap itself -- run_certify.sh
# already wraps its own python invocation in `timeout -s KILL 7200`, and the three `run` calls here
# are sequential and bounded (128 episodes each) by that. Same convention as cert_ft.sh and
# cert_g3z4_finetune.sh: left to the operator to invoke under
# `setsid nohup bash cert_nnlv_vertgoal.sh &> driver.log &` if a detached run is wanted. A second,
# differently-configured timeout wrapping the same underlying call is not added here.
