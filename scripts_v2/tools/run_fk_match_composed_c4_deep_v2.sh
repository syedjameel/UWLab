#!/usr/bin/env zsh
# scripts_v2/tools/run_fk_match_composed_c4_deep_v2.sh
#
# Runner for fk_match_composed_c4.py (already reviewed, never run -- needs a real Isaac boot:
# AppLauncher + SimulationContext + Articulation, not just isaaclab.utils.math CPU-only). This
# script does NOT itself require Isaac to be readable/editable, but every python invocation it
# makes does -- run it only on an Isaac-capable box. Written to be ready-to-execute, not executed
# here.
#
# WHAT THIS RUN WILL TELL YOU: whether the C2/C3 arm-config libraries (10,000 + 10,501 real,
# FK-verified robot configurations) contain any density near the palm poses a deep-band leg
# insertion + a validated grasp actually demand -- a coverage/density go/no-go, reported as a
# residual-distribution + match-count-matrix + pose-coverage summary in
# <OUT_DIR>/fk_match_summary.json. This is the answer to "is the composed-state route worth a
# generation campaign at all".
#
# WHAT THIS RUN WILL NOT TELL YOU: it does not write, settle, or validate a single reset state (see
# the script's own module docstring, "THIS SCRIPT MEASURES REACHABILITY ONLY"). Even a clean go
# here is only the green light for the SEPARATE generator,
# scripts_v2/tools/gen_composed_c4_reset_bank.py (this same delivery), which does the actual
# matching + state construction + writing + physics-settle question -- do not skip straight from
# this script's json to a reset bank.
#
# THE SELF-CHECK GATE. fk_match_composed_c4.py hard-asserts a composition round-trip
# (combine_frame_transforms(subtract_frame_transforms(...)) reconstructs one real FK'd palm pose to
# <0.01mm/<0.01deg) BEFORE printing anything past that point, and prints the literal line
#   [fk_match] SELF-CHECK PASSED -- composition math verified against real FK. Proceeding.
# on success. This runner does not trust the python process's own exit code alone (a hang, an OOM
# kill, or any failure mode that doesn't route through that specific assert could still exit
# nonzero for an unrelated reason, or -- more dangerously -- exit zero without ever reaching the
# assert if something upstream silently short-circuited). It greps the log for that EXACT string
# after the process ends and FAILS LOUDLY, unconditionally, if it is absent -- regardless of the
# reported exit code. Treat every number in fk_match_summary.json as meaningless until this
# runner itself prints "RUNNER: SELF-CHECK CONFIRMED PRESENT".
#
# Isaac Sim ignores SIGTERM (kit's own shutdown path needs the app event loop, which a plain
# SIGTERM does not drive) -- use `timeout -s KILL`, never a bare `timeout`, or a hung/stuck process
# survives the timeout and keeps holding the GPU.
set -x

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR}/../.."
cd "$REPO_ROOT" || exit 1

export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl"

# PYTHON_BIN is host-specific -- set it to this box's Isaac-enabled interpreter before running (on
# DL_H100, per the layout as given, that is /home/dom_iva/venv_uwlab/bin/python -- documented here
# as an EXAMPLE, not hardcoded as a default, since the whole point of this variable is that it
# differs per box).
: "${PYTHON_BIN:?Set PYTHON_BIN to this host's Isaac-enabled python before running.}"

# --- Inputs. NO absolute, machine-specific path is hardcoded anywhere below -- every one of these
# is either a REPO-ROOT-RELATIVE default (portable: resolves correctly on any checkout of this repo,
# this machine's or another's, because it is built from $REPO_ROOT, not a baked-in absolute path) or
# a REQUIRED variable with no default at all where the real location is genuinely unknown ahead of
# time. A hardcoded absolute path that resolves on one box and not another is exactly how this
# project has lost time before -- see PARTIAL_ASSEMBLY_PATH below, which refuses to guess. ---

# grasps.pt for SquareTableLeg200mmDecomp -- repo-root-relative, so this resolves correctly on any
# checkout, this machine's or another's. Override via GRASPS_PATH if a specific box's copy differs.
GRASPS_PATH="${GRASPS_PATH:-$REPO_ROOT/Datasets_ur5e_delto/OmniReset/Grasps/SquareTableLeg200mmDecomp/grasps.pt}"

# THE DEEP, WIDE-BAND ([12,25]mm) partial-assembly pose set -- v2, per team-lead direction, NOT v1
# (v1 is partial_assemblies_deep_v1.pt, [18,25]mm, sha b233dae5...; keep the two distinct rather than
# defaulting to either). NO DEFAULT: as of this writing the deep spawn sets are still being staged on
# the target box and their directory there is not yet known -- guessing a path here would be exactly
# the "hardcoded path that resolves on one box and not another" failure mode. Set
# PARTIAL_ASSEMBLY_PATH explicitly before running.
: "${PARTIAL_ASSEMBLY_PATH:?Set PARTIAL_ASSEMBLY_PATH to wherever partial_assemblies_deep_v2.pt (sha 49ea8852...) has been staged on this box -- no default, location TBC.}"

# C3 (ObjectAnywhereEEGrasped, "Stable Grasp") -- primary arm-config source. Repo-root-relative
# default, matching the DL_H100 layout as given: <tree>/Datasets_ur5e_delto/OmniReset/Resets/
# OneLegInsertionFixture__SquareTableLeg200mmDecomp/resets_ObjectAnywhereEEGrasped.pt. IMPORTANT: on
# THIS analysis machine, that same relative path points at an OLDER-GENERATION bank missing
# joint_position_target/joint_velocity_target (gen_composed_c4_reset_bank.py's _load_c_bank guard
# will refuse it, loudly, naming exactly this) -- the H100's copy at the same relative path is the
# GOOD 2026-08-21 consolidated set (sha256-verified on both ends of that transfer). Do not "fix" a
# failure here by pointing back at this repo's local copy; if this ever needs to run somewhere the
# relative path is wrong, override C3_BANK_PATH explicitly rather than editing this default.
C3_BANK_PATH="${C3_BANK_PATH:-$REPO_ROOT/Datasets_ur5e_delto/OmniReset/Resets/OneLegInsertionFixture__SquareTableLeg200mmDecomp/resets_ObjectAnywhereEEGrasped.pt}"

# C2 (ObjectRestingEEGrasped) -- secondary arm-config source. Same repo-root-relative default and
# same stale-locally/good-on-H100 caveat as C3 above.
C2_BANK_PATH="${C2_BANK_PATH:-$REPO_ROOT/Datasets_ur5e_delto/OmniReset/Resets/OneLegInsertionFixture__SquareTableLeg200mmDecomp/resets_ObjectRestingEEGrasped.pt}"

# MODE: "sanity" (default) runs the script's own recommended first pass -- --max-arm-configs 512,
# a single chunk at the default chunk-size -- before scaling up. "full" removes the cap and uses
# every C3+C2 state. Per the script's own docstring: "START SMALL... as a sanity pass before
# scaling to the full C3(+C2) bank size... If chunk construction is slow or VRAM-heavy at 512,
# lower this before raising --max-arm-configs."
MODE="${1:-sanity}"
if [[ "$MODE" == "sanity" ]]; then
  MAX_ARM_CONFIGS_ARGS=(--max-arm-configs 512)
  OUT_DIR="$REPO_ROOT/local_ckpts/fk_match_composed_c4_deep_v2_sanity"
elif [[ "$MODE" == "full" ]]; then
  MAX_ARM_CONFIGS_ARGS=()
  OUT_DIR="$REPO_ROOT/local_ckpts/fk_match_composed_c4_deep_v2_full"
else
  echo "Usage: $0 [sanity|full]  (default: sanity)" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"

LOG="$OUT_DIR/fk_match_composed_c4_deep_v2_${MODE}.log"

timeout -s KILL 1800 "$PYTHON_BIN" -u scripts_v2/tools/fk_match_composed_c4.py \
  --grasps-path "$GRASPS_PATH" \
  --partial-assembly-path "$PARTIAL_ASSEMBLY_PATH" \
  --c3-bank-path "$C3_BANK_PATH" \
  --c2-bank-path "$C2_BANK_PATH" \
  "${MAX_ARM_CONFIGS_ARGS[@]}" \
  --max-grasps 300 \
  --chunk-size 512 \
  --pos-thresholds-mm "5,10,20,50" \
  --rot-thresholds-deg "5,15,30" \
  --seed 0 \
  --out-dir "$OUT_DIR" \
  --headless \
  > "$LOG" 2>&1
PY_EXIT=$?
echo "PYTHON_EXIT_CODE=$PY_EXIT" >> "$LOG"

# --- THE GATE. Absence of this exact line is fatal, independent of $PY_EXIT. ---
if grep -qF "[fk_match] SELF-CHECK PASSED -- composition math verified against real FK. Proceeding." "$LOG"; then
  echo "RUNNER: SELF-CHECK CONFIRMED PRESENT -- numbers in $LOG / $OUT_DIR/fk_match_summary.json may be trusted."
  echo "RUNNER: python exit code was $PY_EXIT."
  exit 0
else
  echo "RUNNER: SELF-CHECK LINE NOT FOUND IN LOG -- REFUSING TO TRUST THIS RUN." >&2
  echo "RUNNER: this is fatal regardless of the python exit code ($PY_EXIT). Do not read $OUT_DIR/fk_match_summary.json." >&2
  echo "RUNNER: check $LOG for a Python traceback, an OOM/SIGKILL, or (worst case) a silent early exit." >&2
  exit 1
fi
