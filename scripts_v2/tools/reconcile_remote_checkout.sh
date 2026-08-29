#!/usr/bin/env bash
# Bring the box's UWLab_v2 checkout up to origin/dexreset/v2-resets before tonight's queue runs
# against it. WRITTEN, NOT RUN -- the throughput sweep is still using the box; running this while
# that sweep is live would make its later stages execute different code than its earlier ones and
# silently invalidate the comparison. team-lead runs this at handback, not before.
#
# Remote login shell is fish -- never a bare ssh one-liner. Invoke via h100.sh, which pipes this
# file into a remote `bash -s`:
#
#   tools/h100.sh run scripts_v2/tools/reconcile_remote_checkout.sh
#
# WHY THIS IS NOT A PLAIN `git reset --hard`:
#   1. The working tree carries UNCOMMITTED changes. A stale (three-hours-old) check found them
#      byte-identical to a known commit for the files they touch -- this script RE-VERIFIES that
#      itself, against both the commit the tree currently sits on and the commit it is about to
#      move to, and REFUSES the reset if anything no longer matches either.
#   2. The tree holds UNTRACKED measurement outputs (measure_v2_pose_distribution.py,
#      measure_v2_bank_tilt.py, their _out.npz files, and whatever has accumulated since) that a
#      hard reset does not touch -- but "does not touch" is asserted here by enumerating them
#      before the reset, not assumed.
#   3. Every .usd asset under the tree is gitignored and was hand-copied in, from two SEPARATE
#      gitignored roots (uwlab_assets/local/Props for the leg variants, uwlab_assets/data/Props for
#      the fixture). A reset --hard does not touch gitignored paths either, but a checkout that
#      syncs clean and then cannot launch sim is worse than not syncing, so this is checked
#      afterwards by directory, not filename (both leg variants ship a same-named .usd).
#
# ORDER, refusing at each step rather than continuing past it:
#   0. require an explicit confirmation env var (this step exists only to stop an accidental
#      `h100.sh run` of this file from doing something irreversible)
#   1. refuse if any python/Isaac process has this tree as its cwd or in its command line
#   2. fetch; report exactly which commits are incoming and which files they touch
#   3. for every locally-modified TRACKED file: hash the working copy, the blob at the incoming
#      commit, and the blob at the current commit; classify IDENTICAL-TO-INCOMING,
#      IDENTICAL-TO-CURRENT, or DIVERGENT. Refuse the whole reset if anything is DIVERGENT.
#   4. enumerate untracked files that reset --hard will preserve, with sizes; count gitignored ones
#   5. only then: git reset --hard to the incoming commit
#   6. afterwards: confirm HEAD, confirm the three known gitignored assets are present, confirm
#      tonight's seven tools exist, print one final result line
#
# Safe to run twice: if the tree is already at the incoming commit, the commit/file report is
# empty, the divergence loop finds nothing left to disagree about (a prior run either fixed or
# refused on anything divergent), and `git reset --hard` to the commit already checked out is a
# no-op. No branch of this script special-cases "already up to date" -- it is just what the general
# path does when there is nothing to do.
set -uo pipefail

REPO_DIR=${REPO_DIR:-$HOME/github.com/orel/UWLab_v2}
REMOTE_NAME=${REMOTE_NAME:-origin}
BRANCH=${BRANCH:-dexreset/v2-resets}
RECONCILE_CONFIRM=${RECONCILE_CONFIRM:-}

echo "[reconcile] repo=$REPO_DIR remote=$REMOTE_NAME branch=$BRANCH"

# ---------------------------------------------------------------------------------------------
# STEP 0: explicit opt-in. This is a hard reset of a shared box; nothing below should run just
# because this file happened to get piped over.
# ---------------------------------------------------------------------------------------------
[ "$RECONCILE_CONFIRM" = "yes" ] || {
  echo "REFUSING: set RECONCILE_CONFIRM=yes to actually run this. Nothing has been touched."
  exit 1
}

cd "$REPO_DIR" || { echo "REFUSING: REPO_DIR=$REPO_DIR does not exist"; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "REFUSING: REPO_DIR=$REPO_DIR is not a git repository"; exit 1
}

CURRENT_SHA=$(git rev-parse HEAD)
echo "[reconcile] current HEAD: $CURRENT_SHA ($(git log -1 --format=%s -- . 2>/dev/null))"

# ---------------------------------------------------------------------------------------------
# STEP 1: refuse under a live run. Checked by comm (python/isaac/kit) AND either cwd or cmdline
# referencing this tree, so a python process that merely has this directory on PYTHONPATH but is
# running elsewhere is not a false positive, while one actually working inside it is caught even
# if its cwd is somewhere else (its cmdline will still name paths under REPO_DIR).
# ---------------------------------------------------------------------------------------------
echo "[reconcile] checking for live python/Isaac processes against $REPO_DIR ..."
busy_report=""
for procdir in /proc/[0-9]*; do
  pid=${procdir#/proc/}
  comm=$(cat "$procdir/comm" 2>/dev/null) || continue
  case "$comm" in
    *python*|*isaac*|*kit) ;;
    *) continue ;;
  esac
  cwd=$(readlink -f "$procdir/cwd" 2>/dev/null) || cwd="?"
  cmd=$(tr "\0" " " < "$procdir/cmdline" 2>/dev/null) || cmd=""
  hit=0
  case "$cwd" in "$REPO_DIR"|"$REPO_DIR"/*) hit=1 ;; esac
  case "$cmd" in *"$REPO_DIR"*) hit=1 ;; esac
  if [ "$hit" = 1 ]; then
    busy_report="${busy_report}  pid=$pid comm=$comm cwd=$cwd cmd=$cmd
"
  fi
done
if [ -n "$busy_report" ]; then
  echo "REFUSING: process(es) still running against $REPO_DIR -- reconcile must not run under a live job:"
  printf "%s" "$busy_report"
  exit 1
fi
echo "[reconcile] no python/Isaac process has this tree as its cwd or on its command line"

# ---------------------------------------------------------------------------------------------
# STEP 2: fetch, then report what is incoming.
# ---------------------------------------------------------------------------------------------
echo "[reconcile] fetching $REMOTE_NAME ..."
git fetch "$REMOTE_NAME" "$BRANCH" || {
  echo "REFUSING: git fetch $REMOTE_NAME $BRANCH failed"; exit 1
}
INCOMING_REF="$REMOTE_NAME/$BRANCH"
INCOMING_SHA=$(git rev-parse "$INCOMING_REF" 2>/dev/null) || {
  echo "REFUSING: cannot resolve $INCOMING_REF after fetch"; exit 1
}

echo "[reconcile] incoming: $INCOMING_SHA ($(git log -1 --format=%s "$INCOMING_SHA"))"
echo "[reconcile] commits incoming ($CURRENT_SHA..$INCOMING_SHA):"
LOG=$(git log --oneline "$CURRENT_SHA..$INCOMING_SHA" 2>/dev/null)
if [ -n "$LOG" ]; then
  echo "$LOG" | sed "s/^/  /"
else
  echo "  (none -- already at $INCOMING_SHA)"
fi
echo "[reconcile] files touched by incoming commits:"
DIFFSTAT=$(git diff --name-status "$CURRENT_SHA" "$INCOMING_SHA" 2>/dev/null)
if [ -n "$DIFFSTAT" ]; then
  echo "$DIFFSTAT" | sed "s/^/  /"
else
  echo "  (none)"
fi

# ---------------------------------------------------------------------------------------------
# STEP 3: divergence check on locally-modified TRACKED files. blob_hash distinguishes "absent at
# this ref" (prints nothing, returns 1) from "present and genuinely empty" (a real, computed
# sha256sum of zero bytes) -- collapsing those two would let an absent-at-ref file look identical
# to an empty file by coincidence, which is exactly the kind of wrong-but-plausible number this
# campaign has spent the night hunting.
# ---------------------------------------------------------------------------------------------
blob_hash() {  # blob_hash <ref> <path>
  git cat-file -e "$1:$2" 2>/dev/null || return 1
  git show "$1:$2" 2>/dev/null | sha256sum | awk "{print \$1}"
}

echo
echo "[reconcile] checking git status for rename/copy entries before the NUL-safe scan below ..."
if git status --porcelain=v1 | cut -c1-2 | grep -q "^[RC]"; then
  echo "REFUSING: git status shows a rename or copy entry -- this scripts NUL-parsing below does not"
  echo "handle the two-path record renames produce; reconcile that file by hand first."
  exit 1
fi

echo "[reconcile] checking locally-modified tracked files against current HEAD and incoming ..."
DIVERGENT=0
CHECKED=0
while IFS= read -r -d "" entry; do
  status=${entry:0:2}
  path=${entry:3}
  case "$status" in
    "??"|"!!") continue ;;  # untracked / ignored -- step 4
  esac
  [ -n "$path" ] || continue
  CHECKED=$((CHECKED + 1))
  if [ ! -e "$path" ]; then
    echo "  DIVERGENT (locally deleted, present in history): $path"
    DIVERGENT=$((DIVERGENT + 1))
    continue
  fi
  working_hash=$(sha256sum "$path" | awk "{print \$1}")
  incoming_hash=$(blob_hash "$INCOMING_SHA" "$path") || incoming_hash=""
  current_hash=$(blob_hash "$CURRENT_SHA" "$path") || current_hash=""
  if [ -n "$incoming_hash" ] && [ "$working_hash" = "$incoming_hash" ]; then
    echo "  IDENTICAL-TO-INCOMING: $path"
  elif [ -n "$current_hash" ] && [ "$working_hash" = "$current_hash" ]; then
    echo "  IDENTICAL-TO-CURRENT: $path"
  else
    echo "  DIVERGENT: $path (working=$working_hash incoming=${incoming_hash:-<absent-at-incoming>} current=${current_hash:-<absent-at-current>})"
    DIVERGENT=$((DIVERGENT + 1))
  fi
done < <(git status --porcelain=v1 -z)
echo "[reconcile] $CHECKED locally-modified tracked file(s) checked, $DIVERGENT divergent"

if [ "$DIVERGENT" -gt 0 ]; then
  echo "REFUSING: $DIVERGENT locally-modified tracked file(s) match neither current HEAD nor"
  echo "incoming $BRANCH -- unaccounted-for work. A hard reset would discard it silently. Resolve by"
  echo "hand (commit it, stash it, or confirm it should be dropped) and re-run."
  exit 1
fi
echo "[reconcile] no divergent tracked files -- every local modification is a no-op relative to current or incoming"

# ---------------------------------------------------------------------------------------------
# STEP 4: enumerate what reset --hard will leave untouched. Untracked (non-ignored) files listed
# individually with size, since these are the measurement outputs that matter; gitignored files
# (the hand-copied .usd assets, mostly) just counted -- there can be thousands of mesh files and
# the specific ones that matter are checked by name after the reset, in step 6.
# ---------------------------------------------------------------------------------------------
echo
echo "[reconcile] untracked (non-ignored) files that reset --hard will preserve:"
UNTRACKED_COUNT=0
while IFS= read -r -d "" path; do
  [ -n "$path" ] || continue
  size=$(stat -c%s "$path" 2>/dev/null || echo "?")
  echo "  $path  (${size} bytes)"
  UNTRACKED_COUNT=$((UNTRACKED_COUNT + 1))
done < <(git status --porcelain=v1 -z | awk -v RS="\0" '/^\?\? /{printf "%s\0", substr($0,4)}')
echo "[reconcile] $UNTRACKED_COUNT untracked file(s) will be preserved"

IGNORED_COUNT=$(git status --porcelain=v1 --ignored -z | awk -v RS="\0" '/^!! /{c++} END{print c+0}')
echo "[reconcile] $IGNORED_COUNT gitignored path(s) present under $REPO_DIR (also preserved; the ones that matter are checked by name in step 6)"

# ---------------------------------------------------------------------------------------------
# STEP 5: the reset. Everything above either refused or confirmed nothing unaccounted-for exists.
# ---------------------------------------------------------------------------------------------
echo
echo "[reconcile] all checks passed. Resetting $REPO_DIR to $INCOMING_REF ($INCOMING_SHA) ..."
git reset --hard "$INCOMING_SHA" || { echo "REFUSING: git reset --hard failed"; exit 1; }

# ---------------------------------------------------------------------------------------------
# STEP 6: confirm the reset produced what it should have.
# ---------------------------------------------------------------------------------------------
NEW_HEAD=$(git rev-parse HEAD)
if [ "$NEW_HEAD" != "$INCOMING_SHA" ]; then
  echo "FAIL: HEAD is $NEW_HEAD after reset, expected $INCOMING_SHA"
  exit 1
fi
echo "[reconcile] HEAD confirmed at $NEW_HEAD"

echo
echo "[reconcile] checking gitignored assets survived the reset (by parent directory, not filename --"
echo "both leg variants ship a file of the same name) ..."
ASSET_FAIL=0
check_asset() {  # check_asset <label> <path>
  if [ -f "$2" ]; then
    echo "  OK      $1: $2"
  else
    echo "  MISSING $1: $2"
    ASSET_FAIL=$((ASSET_FAIL + 1))
  fi
}
check_asset "leg (Decomp)" "$REPO_DIR/source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmDecomp/square_table_leg4_200mm.usd"
check_asset "leg (Sdf)"    "$REPO_DIR/source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf/square_table_leg4_200mm.usd"
check_asset "fixture"      "$REPO_DIR/source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/one_leg_insertion_fixture.usd"

if [ "$ASSET_FAIL" -gt 0 ]; then
  echo "[reconcile] WARNING: $ASSET_FAIL gitignored asset(s) missing after reset -- sim will not launch"
  echo "until these are restored from wherever they were hand-copied from originally. This is a"
  echo "report, not a refusal: the reset already happened and this script cannot undo it."
else
  echo "[reconcile] all 3 known gitignored assets present"
fi

echo
echo "[reconcile] checking tonight's tools exist at the new HEAD ..."
TOOL_FAIL=0
check_tool() {  # check_tool <path relative to REPO_DIR>
  if [ -f "$REPO_DIR/$1" ]; then
    echo "  OK      $1"
  else
    echo "  MISSING $1"
    TOOL_FAIL=$((TOOL_FAIL + 1))
  fi
}
check_tool scripts_v2/tools/smoke_c3_rung_isaac.py
check_tool scripts_v2/tools/analyze_c3_rung_smoke.py
check_tool scripts_v2/tools/launch_c3_rung_smoke.sh
check_tool scripts_v2/tools/r0_control.sh
check_tool scripts_v2/tools/measure_gate_proxy.py
check_tool scripts_v2/tools/analyze_c3_st_measure_only.py
check_tool scripts_v2/tools/certification/recertify_v1_baselines.sh

if [ "$TOOL_FAIL" -gt 0 ]; then
  echo "REFUSING (post-check): $TOOL_FAIL expected tool(s) missing at the new HEAD -- the checkout"
  echo "reset succeeded but does not contain what tonight needs. Do not launch anything from it."
  exit 1
fi
echo "[reconcile] all 7 expected tools present"

ASSET_RESULT="OK"
[ "$ASSET_FAIL" -eq 0 ] || ASSET_RESULT="MISSING($ASSET_FAIL)"
echo
echo "RECONCILE_RESULT head=$NEW_HEAD assets=$ASSET_RESULT tools=OK"
