#!/usr/bin/env bash
# STAGED, NOT RUN. Final phase of the g3z4 campaign: regenerate all four reset banks
# (C1/C3/C2/C4, 10,000 states each) from the RETRAINED, CERTIFIED finetune checkpoint. This is
# the only large piece of the campaign with no script behind it yet -- staged now so it is off
# the critical path once the finetune actually certifies. Do not run until then.
#
# Usage:
#   regenerate_four_banks_post_finetune.sh <finetune_ckpt.pth> <finetune_agent_yaml> [gpu]
#
# REUSES, DOES NOT REINVENT, four already-hardened pieces (per instruction):
#   - generate_reset_states_policy.py itself, including --c2_rewind (locked offsets 0.03/0.05/
#     0.10/0.20s, resting filter, RewindC2) -- this script's own chunk loop mirrors
#     run_c2_rewind_paperscale_reorient.sh's proven chunking/seeding/isolation pattern rather than
#     copying its body; read that script alongside this one, it is the worked single-arm example.
#   - merge_c2_chunks.py (commit 3bb7c40) for BOTH the C2 offset merges (--c2_reset_type) and the
#     plain C1/C3/C4 chunk merges (--filename) -- same tool, two modes, both used below.
#   - rekey_dexlift_reset_states.py -- MANDATORY on every merged bank before it is usable; a
#     one-key (insertive_object only) file trains with the fixture 0.35-0.60m laterally displaced
#     and nothing warns you.
#
# =====================================================================================
# DESIGN DECISION: C2 RIDES ON C3'S OWN CHUNKS, NOT A SEPARATE PASS
# =====================================================================================
# run_c2_rewind_paperscale_reorient.sh generates C2 as its OWN ~4200-accepted-episode run,
# isolated from C3, because at the time it was staged C3 was ALREADY mid-flight WITHOUT
# --c2_rewind attached -- a catch-up job forced by that gap, not the ideal shape. --c2_rewind
# rewinds EVERY accepted ObjectAnywhereEEGrasped episode for free, as a side effect of the SAME
# rollout that produces C3's own accept-time bank -- there is no reason to pay for a second Isaac
# pass over the identical reset_type when generating fresh. So below, C3's chunk loop carries
# --c2_rewind --c2_reset_type ObjectAnywhereEENear from the start, and C2's merge step runs
# against the SAME chunk directories C3's merge step does. If C2's real yield (measured from this
# run's own "[c2][progress] total_emitted=..." lines, watched live) falls short of 10,000 once
# C3's 10,000-accepted target is reached, that is a real finding to report, not a bug in this
# design -- C2_EXTRA_CHUNKS below exists for exactly that case (see C3 section).
#
# =====================================================================================
# FIVE THINGS EVERY ARM DOES (each learned the hard way today, per instruction)
# =====================================================================================
#   1. CHUNKED at CHUNK_TARGET_SIZE (default 1800, inside the measured-safe 1700-2000 band)
#      accepted states per process -- the generator rewrites its entire accumulated bank on every
#      accepted state, so cost is QUADRATIC in that process's own n (measured on C3: windowed
#      rate 0.97->0.645->0.395->0.327 accepted/s as n grew). One 10,000-state run projects to
#      ~13h; five/six chunks of ~1800 to ~3h. Chunks are NEVER consolidated mid-campaign to save
#      an Isaac boot -- that is exactly the trap this system avoids.
#   2. DISTINCT SEED PER CHUNK via make_seeded_agent_yaml.py, verified from each chunk's OWN log
#      at BOTH ends -- "[generator] agent_cfg params.seed: N" (the yaml contained it) AND rl_games'
#      own unprompted "self.seed = N" (rl_games actually consumed it). A filename or a directory
#      being different is not evidence of seed diversity; only the log is.
#   3. ISOLATED OUTPUT DIR PER CHUNK under one run-scoped scratch base, checked against the
#      production root before anything launches -- this generator TRUNCATES AND REWRITES its
#      output in place several times a minute, so a misaimed run destroys a finished bank, not
#      just wastes GPU time. The production Resets/ directory is backed up ONCE, unconditionally,
#      before any chunk of any arm runs, independent of whether the isolation guard holds.
#   4. PRODUCTION CONFIGURATION EXACTLY: all four DEXLIFT_REF_* vars together, DEXLIFT_POSE_TILT=
#      0.3, --episode_length_s 8.0, --num_envs 128, an EXPLICIT --agent_yaml (never this tool's
#      own derived-path fallback). Measured today: missing --episode_length_s costs 3.6x
#      acceptance, missing DEXLIFT_POSE_TILT another 2.5x on top of that -- both silent, neither
#      an error.
#   5. VALIDATE EACH FINISHED (rekeyed) BANK BY LOADING IT, not by trusting the pipeline that
#      produced it: four rigid_object keys (insertive_object, receptive_object, table,
#      ur5_metal_support), joint_position_target AND joint_velocity_target present on the
#      articulation (bead UWLab-algw.7 -- their absence means target := q at replay, i.e. zero
#      commanded squeeze; the recorder fix for this landed in recorders.py today, so a bank
#      generated on the current tree should always have them, and a fresh run without them is a
#      regression to catch, not a compatibility case to shrug off), per-field lengths equal to the
#      state count, receptive_object z exactly 0.019625 with std 0.
#
# =====================================================================================
# THE C4 ACCEPTANCE QUESTION -- WRITTEN DOWN, NOT GUESSED AT
# =====================================================================================
# C4 has never been generated successfully by either historical route; this finetune (trained
# partly on partial-assembly grasp-only episodes via DEXLIFT_EPISODE_MIXTURE's
# partial_assembly_prob fraction) is what is supposed to unblock it. GENERATION ITSELF does NOT
# use that training-time mixture -- it uses the older, deterministic, WHOLE-RUN legacy toggle
# (DEXLIFT_PARTIAL_ASSEMBLY=1, _apply_partial_assembly_and_goal_toggles), which forces every env
# into SpawnPartialAssembly + GoalAtSpawnPoseCommand, because generation wants 100% of attempts
# to be grasp-only recording opportunities, not a 25% training fraction -- an efficiency choice,
# not a correctness one; the two mechanisms are structurally independent (this file's own
# __post_init__ short-circuits the new mixture whenever the legacy toggle fires) and were
# confirmed to compose correctly earlier in this same epic.
#
# THE GENERIC CHECKLIST ABOVE (item 5) DOES NOT PROVE C4-GENUINENESS, AND I VERIFIED THIS RATHER
# THAN ASSUMED IT. rekey_dexlift_reset_states.py SYNTHESISES a receptive_object entry, drawn from
# the SAME per-episode distribution generation would have used, for any bank that did not already
# have one (C1/C3's raw output is {"object","table"} -- no fixture in that scene at all) --
# including writing z = 0.019625 exactly, the identical constant a REAL C4 fixture would show.
# So "receptive_object z == 0.019625, std 0" -- item 5's own check -- passes on EVERY rekeyed
# bank, C1/C3 included, and cannot by itself distinguish a genuine C4 recording from a C1/C3 bank
# wearing synthesised C4 clothing. This is exactly last shape of "banks whose CONTENTS did not
# match their FILENAME" the instruction warned about, found by reading rekey's own synthesis code
# rather than trusting its output schema at face value.
#
# WHAT ACTUALLY DISTINGUISHES THEM, and what validate_c4_arm() below checks, on the RAW
# (pre-rekey, pre-merge, per-chunk) accept-time bank -- BEFORE synthesis can paper over anything:
#   (a) receptive_object must be NATIVELY PRESENT in the raw chunk's own rigid_object dict.
#       _detect_schema refuses to even run on a file that already has it (calls it
#       schema-complete), which is itself the tell: a raw C1/C3 file is structurally INCAPABLE of
#       having a native receptive_object (that scene never instantiates the entity -- see
#       dexlift_ur5e_delto_tableleg_env_cfg.py's partial_assembly-gated
#       `env_cfg.scene.receptive_object = ...` line), so its presence in the RAW file is real
#       evidence the recording scene genuinely had a fixture, not something synthesis could have
#       added at this stage.
#   (b) per-episode LEG-TO-FIXTURE PROXIMITY at capture: insertive_object and (the native)
#       receptive_object root_pose XY distance, computed per state. A genuine partial-assembly
#       spawn seats the leg AT the fixture (RECEPTIVE_POSE_RANGE's own x/y span is under 0.3m x
#       0.4m and the leg starts already engaged, not transported there); synthesised filler on a
#       C1/C3 bank draws the fake fixture pose INDEPENDENTLY of the real (uniformly-anywhere) leg
#       pose, so the two would show no correlation and a much wider, unstructured spread. No
#       pre-measured threshold exists for this pair's C4 output (it has never been generated
#       before) -- report the distribution (median/max) rather than a hardcoded pass/fail; a human
#       reads it against the fixture's own footprint before trusting the bank.
#   (c) each chunk's own log must show the real, swallow-proof confirmation line this epic's
#       earlier bugfix preserved: "[dexlift] DEXLIFT_PARTIAL_ASSEMBLY=1 ... reset_object ->
#       SpawnPartialAssembly (dataset_dir=...)" -- proving the toggle was READ AS REQUESTED by the
#       CONSTRUCTED cfg, not merely passed on a command line that something downstream silently
#       ignored (the exact "verify the constructed thing, not the request" seam this campaign has
#       been bitten by twice already: a plant checked from env vars instead of the articulation, a
#       bank checked by filename instead of contents).
# All three together are the C4 acceptance criteria; (a) and (c) are hard refusals below, (b) is
# reported for human judgement since there is no prior number to compare it to.
#
# =====================================================================================
# THE PARTIAL-ASSEMBLY DATASET (C4 ONLY): POINT AT THE VERIFIED 2048-POSE FILE, NEVER THE DEFAULT
# =====================================================================================
# SpawnPartialAssembly (the legacy toggle's spawn term, used ONLY by the C4 arm) downloads
# partial_assemblies.pt for this pair from mdp.DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR the moment it
# is constructed -- that default (Hugging Face) path 404s for this pair, confirmed today, and even
# if it resolved, THIS machine's only two local copies under that name are the OLD 525-pose
# random-walk file (0 of 525 poses actually seated in the bore -- confirmed by reprojection against
# the fixture's own metadata.yaml). PARTIAL_ASSEMBLY_DATASET_DIR below MUST point at a verified
# 2048-pose file (sha256 eb7c71545f9d482e4e1b85ed76a509b47810372bfd1ef869f473ebe7b2d4fdd0 is the
# one already confirmed on this machine, at local_ckpts/partial_assemblies_2048/) -- overridable,
# but never silently falls back to the broken default; the count-and-refuse guard below is the
# SAME pattern launch_dexlift_tableleg_reorient_finetune.sh already uses, applied here to the
# generation-time (DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR) path instead of the training-time
# (DEXLIFT_EPISODE_MIXTURE_DATASET_DIR) one -- two different env vars, same underlying file, same
# guard logic, because they are two structurally independent code paths (see the design-decision
# section above).
#
# =====================================================================================
# THESE BANKS ARE ADDRESSED BY PAIR DIRECTORY -- ANY CONSUMER MUST SET THE VARIANT PAIR EXPLICITLY
# =====================================================================================
# Every bank this script produces is written under Resets/OneLegInsertionFixture__
# SquareTableLeg200mmDecomp/ -- the leg200mm/onelegfixture pair, computed the same way
# MultiResetManager itself computes it (compute_pair_dir off the scene's actual insertive/
# receptive USD paths), NOT a fixed or assumed location. A consumer that does not ALSO resolve to
# this exact pair will not find these banks at all.
#
# CONCRETE, MEASURED TRAP: OmniReset-UR5eDelto-RelCartesianOSC-State-v0 -- the actual eventual RL
# task this campaign's banks exist to feed -- DEFAULTS TO A DIFFERENT PAIR (Peg__PegHole), not
# leg200mm/onelegfixture, unless its scene objects are explicitly overridden to the variant this
# campaign targets. Confirmed directly while verifying the C4 arm's MultiResetManager guard today:
# a construction pointed at that task's DEFAULT pair looked for Resets/Peg__PegHole/... and, left
# to fetch that pair's assets from the cloud, has previously presented as a SILENT ~25-MINUTE HANG
# right after "Time taken for scene creation" with no further log line -- not a missing-file error,
# not a crash, just silence. A launch of the eventual RL run that forgets to set the variant pair
# will not find a single bank this campaign generates, and the failure will look like a hang, not
# a "file not found."
#
# THIS SCRIPT DOES NOT WIRE THAT -- pointing the eventual RL launch at the right variant pair is a
# separate job, out of scope here. This note exists so whoever writes that launch command reads it
# BEFORE discovering the hang, not after.
set -uo pipefail

CKPT=${1:?"usage: $0 <finetune_ckpt.pth> <finetune_agent_yaml> [gpu]"}
AGENT_YAML=${2:?"usage: $0 <finetune_ckpt.pth> <finetune_agent_yaml> [gpu]"}
GPU="${3:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Isaac-enabled python interpreter path on this box before running.}"

[ -f "$CKPT" ] || { echo "REFUSING: checkpoint not found at $CKPT"; exit 1; }
[ -f "$AGENT_YAML" ] || { echo "REFUSING: agent yaml not found at $AGENT_YAML"; exit 1; }

# -- GPU must be idle, same rule every launcher in this repo now uses.
USED=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
[ "${USED:-999999}" -lt 2000 ] || { echo "REFUSING: GPU $GPU has ${USED} MiB already in use"; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/source/uwlab:$REPO_ROOT/source/uwlab_tasks:$REPO_ROOT/source/uwlab_assets:$REPO_ROOT/source/uwlab_rl"

TASK="DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-Play-v0"
RECEPTIVE_USD_PATH="$REPO_ROOT/source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/one_leg_insertion_fixture.usd"

TARGET_PER_BANK="${TARGET_PER_BANK:-10000}"
CHUNK_TARGET_SIZE="${CHUNK_TARGET_SIZE:-1800}"   # accepted states/chunk; measured-safe band 1700-2000

PRODUCTION_DATASET_DIR="./Datasets_ur5e_delto/OmniReset"
RUN_BASE="${FOURBANK_DATASET_DIR:-./Datasets_ur5e_delto/OmniReset_FOURBANK_$(date +%Y%m%d_%H%M%S)}"
if [ "$(realpath -m "$RUN_BASE")" = "$(realpath -m "$PRODUCTION_DATASET_DIR")" ]; then
  echo "REFUSING: RUN_BASE ($RUN_BASE) resolves to the production root ($PRODUCTION_DATASET_DIR)." >&2
  exit 1
fi
mkdir -p "$RUN_BASE"
echo "[isolation] every arm/chunk writes under: $RUN_BASE  (production root untouched until the printed copy-in)"

# -- BACK UP THE PRODUCTION BANK ONCE, BEFORE ANY CHUNK OF ANY ARM RUNS.
PRODUCTION_RESETS_DIR="$PRODUCTION_DATASET_DIR/Resets"
if [ -d "$PRODUCTION_RESETS_DIR" ]; then
  BACKUP_DIR="${PRODUCTION_RESETS_DIR%/}_backup_$(date +%Y%m%d_%H%M%S)"
  cp -a "$PRODUCTION_RESETS_DIR" "$BACKUP_DIR" || { echo "BACKUP FAILED -- refusing to run without one" >&2; exit 1; }
  echo "[backup] $PRODUCTION_RESETS_DIR -> $BACKUP_DIR"
else
  echo "[backup] no existing production Resets/ dir found at $PRODUCTION_RESETS_DIR -- nothing to back up yet."
fi

# -- The PARTIAL-ASSEMBLY dataset, C4 only. See the header's dedicated section.
PARTIAL_ASSEMBLY_DATASET_DIR="${PARTIAL_ASSEMBLY_DATASET_DIR:-/root/ckpt/Datasets_ur5e_delto/OmniReset}"
_PA_PAIR_FILE="$PARTIAL_ASSEMBLY_DATASET_DIR/Resets/OneLegInsertionFixture__SquareTableLeg200mmDecomp/partial_assemblies.pt"

verify_partial_assembly_dataset() {
  [ -f "$_PA_PAIR_FILE" ] || {
    echo "REFUSING: partial-assembly dataset not found at $_PA_PAIR_FILE (needed for the C4 arm)."
    exit 1
  }
  local n
  n=$("$PYTHON_BIN" -c "
import torch
d = torch.load('$_PA_PAIR_FILE', map_location='cpu', weights_only=True)
print(d['relative_position'].shape[0])
" 2>/dev/null) || { echo "REFUSING: could not load $_PA_PAIR_FILE to count its poses."; exit 1; }
  echo "[c4] partial-assembly dataset pose count: $n ($_PA_PAIR_FILE)"
  if [ "$n" = "525" ]; then
    echo "REFUSING: this is the KNOWN-BROKEN 525-pose file (0 of 525 poses seated in the bore)." >&2
    exit 1
  fi
  if [ "$n" != "2048" ]; then
    echo "REFUSING: expected 2048 poses, got $n -- verify this file's own geometry before overriding." >&2
    exit 1
  fi
  echo "[c4] partial-assembly dataset verified: 2048 poses."
}

# =====================================================================================
# REUSABLE CHUNK LOOP -- one arm's worth of chunked generation. Mirrors
# run_c2_rewind_paperscale_reorient.sh's proven pattern (seeded yaml, isolated dir, per-chunk
# timeout, producer+consumer seed verification), generalised over reset_type/target/extra CLI
# args instead of hardcoding C2's own values. Writes to $RUN_BASE/<arm>/chunk_<i>/, deterministic
# and globbable -- nothing here needs to hand a chunk-dir list back to the caller through bash.
# =====================================================================================
run_chunked_arm() {
  local arm="$1" reset_type="$2" target_total="$3"; shift 3
  local extra_args=("$@")
  local n_chunks=$(( (target_total + CHUNK_TARGET_SIZE - 1) / CHUNK_TARGET_SIZE ))
  local per_chunk_target=$(( (target_total + n_chunks - 1) / n_chunks ))
  local base_seed="${FOURBANK_BASE_SEED:-42}"
  echo "[$arm] target=$target_total chunks=$n_chunks per_chunk_target=$per_chunk_target"

  for i in $(seq 1 "$n_chunks"); do
    local chunk_dir="$RUN_BASE/$arm/chunk_$i"
    local chunk_log="$REPO_ROOT/fourbank_${arm}_chunk${i}.log"
    local chunk_seed=$(( base_seed + i - 1 ))
    local chunk_agent_yaml="$RUN_BASE/$arm/agent_seed_${chunk_seed}.yaml"
    mkdir -p "$RUN_BASE/$arm"

    "$PYTHON_BIN" scripts_v2/tools/make_seeded_agent_yaml.py \
      --source_yaml "$AGENT_YAML" --seed "$chunk_seed" --output_yaml "$chunk_agent_yaml" \
      || { echo "[$arm chunk $i/$n_chunks] FATAL: could not write a seeded agent.yaml -- aborting" \
                "this chunk before spending GPU time on a seed we cannot confirm." >&2; return 1; }

    echo "[$arm chunk $i/$n_chunks] target=$per_chunk_target seed=$chunk_seed dir=$chunk_dir log=$chunk_log"
    timeout -s KILL "${FOURBANK_CHUNK_TIMEOUT_S:-43200}" "$PYTHON_BIN" -u scripts_v2/tools/generate_reset_states_policy.py \
      --task "$TASK" \
      --reset_type "$reset_type" \
      --receptive_usd_path "$RECEPTIVE_USD_PATH" \
      --checkpoint "$CKPT" \
      --agent_yaml "$chunk_agent_yaml" \
      --dataset_dir "$chunk_dir" \
      --episode_length_s 8.0 \
      --num_envs 128 \
      --num_reset_states "$per_chunk_target" \
      --headless \
      "${extra_args[@]}" \
      > "$chunk_log" 2>&1
    local chunk_exit=$?
    echo "EXIT_CODE=$chunk_exit" >> "$chunk_log"
    if [ "$chunk_exit" -ne 0 ]; then
      echo "[$arm chunk $i/$n_chunks] WARNING: exited $chunk_exit (see $chunk_log) -- continuing to" \
           "the next chunk; the merge step tolerates a chunk with no usable output." >&2
    fi

    # -- seed verified at BOTH producer (yaml contained it) and consumer (rl_games consumed it).
    local yaml_ok=0 rlgames_ok=0
    grep -q "agent_cfg params.seed: $chunk_seed\$" "$chunk_log" && yaml_ok=1
    grep -q "self.seed = $chunk_seed\$" "$chunk_log" && rlgames_ok=1
    if [ "$yaml_ok" -eq 1 ] && [ "$rlgames_ok" -eq 1 ]; then
      echo "[$arm chunk $i/$n_chunks] seed verified at both ends: $chunk_seed"
    else
      echo "[$arm chunk $i/$n_chunks] *** SEED NOT FULLY CONFIRMED (expected $chunk_seed):" \
           "yaml=$yaml_ok rl_games=$rlgames_ok -- see $chunk_log ***" >&2
    fi

    # -- PLANT confirmed from the CONSTRUCTED articulation's ACTUAL printed value, not merely that
    # a line mentioning it exists (train.py's own PLANT-VERIFY convention; matching the "match the
    # number, not the label" correction already applied elsewhere in this epic).
    # generate_reset_states_policy.py prints: "[verify] hand effort_limit_sim (distinct values):
    # [30.0]  (expect [30.0] for reference, ...)" when the reference hand actuators are live.
    grep -q "hand effort_limit_sim (distinct values): \[30.0\]" "$chunk_log" \
      || echo "[$arm chunk $i/$n_chunks] WARNING: constructed hand actuators do not read [30.0] in" \
              "this chunk's [verify] line -- inspect $chunk_log before trusting it." >&2
  done
}

# =====================================================================================
# C1 -- ObjectAnywhereEEAnywhere
# =====================================================================================
run_chunked_arm c1 ObjectAnywhereEEAnywhere "$TARGET_PER_BANK"

# =====================================================================================
# C3 -- ObjectAnywhereEEGrasped, WITH C2 (via --c2_rewind) riding on the same chunks. See the
# design-decision section above for why this is one pass, not two.
# =====================================================================================
run_chunked_arm c3 ObjectAnywhereEEGrasped "$TARGET_PER_BANK" \
  --c2_rewind --c2_reset_type ObjectAnywhereEENear

# -- C2_EXTRA_CHUNKS: if C2's real yield (watch each c3 chunk's "[c2][progress] total_emitted=..."
# lines) falls short of 10,000 once C3's own target is met, set this to run additional c3-shaped
# chunks purely to keep feeding C2 -- default 0 (do not guess a number ahead of the real yield,
# which this pair has never been measured on with the retrained checkpoint).
C2_EXTRA_CHUNKS="${C2_EXTRA_CHUNKS:-0}"
if [ "$C2_EXTRA_CHUNKS" -gt 0 ]; then
  echo "[c3/c2] running $C2_EXTRA_CHUNKS extra chunk(s) to top up C2's yield -- check total_emitted first."
  for j in $(seq 1 "$C2_EXTRA_CHUNKS"); do
    EXTRA_CHUNK_DIR="$RUN_BASE/c3/chunk_extra_$j"
    EXTRA_SEED=$(( ${FOURBANK_BASE_SEED:-42} + 1000 + j ))
    EXTRA_YAML="$RUN_BASE/c3/agent_seed_${EXTRA_SEED}.yaml"
    "$PYTHON_BIN" scripts_v2/tools/make_seeded_agent_yaml.py \
      --source_yaml "$AGENT_YAML" --seed "$EXTRA_SEED" --output_yaml "$EXTRA_YAML"
    timeout -s KILL "${FOURBANK_CHUNK_TIMEOUT_S:-43200}" "$PYTHON_BIN" -u scripts_v2/tools/generate_reset_states_policy.py \
      --task "$TASK" --reset_type ObjectAnywhereEEGrasped --receptive_usd_path "$RECEPTIVE_USD_PATH" \
      --checkpoint "$CKPT" --agent_yaml "$EXTRA_YAML" --dataset_dir "$EXTRA_CHUNK_DIR" \
      --episode_length_s 8.0 --num_envs 128 --num_reset_states "$CHUNK_TARGET_SIZE" --headless \
      --c2_rewind --c2_reset_type ObjectAnywhereEENear \
      > "$REPO_ROOT/fourbank_c3_chunk_extra${j}.log" 2>&1
  done
fi

# =====================================================================================
# C4 -- ObjectPartiallyAssembledEEGrasped. Legacy toggle, scoped to a subshell so
# DEXLIFT_PARTIAL_ASSEMBLY never leaks into C1/C3's env.
# =====================================================================================
verify_partial_assembly_dataset
(
  export DEXLIFT_PARTIAL_ASSEMBLY=1
  export DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR="$PARTIAL_ASSEMBLY_DATASET_DIR"
  run_chunked_arm c4 ObjectPartiallyAssembledEEGrasped "$TARGET_PER_BANK"
)

# =====================================================================================
# MERGE. C1/C3/C4 via --filename (plain chunk merge); C2 via --c2_reset_type (offset-file merge,
# reading the SAME c3/chunk_* directories C3's own merge does).
# =====================================================================================
merge_plain() {  # arm reset_type
  local arm="$1" reset_type="$2"
  local chunk_dirs=("$RUN_BASE/$arm"/chunk_*)
  local merged_dir="$RUN_BASE/$arm/merged"
  local merge_log="$REPO_ROOT/fourbank_${arm}_merge.log"
  echo "[$arm] merging $(echo "${chunk_dirs[@]}" | wc -w) chunk dir(s) -> $merged_dir"
  "$PYTHON_BIN" scripts_v2/tools/merge_c2_chunks.py \
    --chunk_dirs "${chunk_dirs[@]}" \
    --output_dir "$merged_dir" \
    --filename "resets_${reset_type}.pt" \
    > "$merge_log" 2>&1
  local merge_exit=$?
  cat "$merge_log"
  [ "$merge_exit" -eq 0 ] || { echo "[$arm] MERGE FAILED (exit $merge_exit), see $merge_log" >&2; return 1; }
}

merge_plain c1 ObjectAnywhereEEAnywhere
merge_plain c3 ObjectAnywhereEEGrasped

C2_CHUNK_DIRS=("$RUN_BASE/c3"/chunk_* "$RUN_BASE/c3"/chunk_extra_*)
C2_MERGED_DIR="$RUN_BASE/c2/merged"
C2_MERGE_LOG="$REPO_ROOT/fourbank_c2_merge.log"
"$PYTHON_BIN" scripts_v2/tools/merge_c2_chunks.py \
  --chunk_dirs "${C2_CHUNK_DIRS[@]}" \
  --output_dir "$C2_MERGED_DIR" \
  --c2_reset_type ObjectAnywhereEENear \
  > "$C2_MERGE_LOG" 2>&1
cat "$C2_MERGE_LOG"

merge_plain c4 ObjectPartiallyAssembledEEGrasped

# =====================================================================================
# C4-SPECIFIC ACCEPTANCE CHECK, ON THE RAW CHUNKS -- see the header's dedicated section for why
# this must run on the raw (pre-rekey) output, not the merged/rekeyed bank.
# =====================================================================================
"$PYTHON_BIN" - <<PYEOF
import glob, torch

chunk_files = sorted(glob.glob("$RUN_BASE/c4/chunk_*/Resets/*/resets_ObjectPartiallyAssembledEEGrasped.pt"))
print(f"[c4-check] found {len(chunk_files)} raw chunk file(s)")
if not chunk_files:
    print("[c4-check] REFUSING: no raw C4 chunk output found -- nothing to validate.")
    raise SystemExit(1)

total_states = 0
all_dist = []
for f in chunk_files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    rigid = d["initial_state"]["rigid_object"]
    keys = set(rigid.keys())
    if "receptive_object" not in keys:
        print(f"[c4-check] REFUSING: {f} has rigid_object keys {sorted(keys)} -- no NATIVE"
              " receptive_object. This chunk's scene never had a fixture; its contents do not"
              " match the C4 filename regardless of --reset_type.")
        raise SystemExit(1)
    leg_pos = torch.stack([p[:3] for p in rigid.get("object", rigid.get("insertive_object"))["root_pose"]])
    fix_pos = torch.stack([p[:3] for p in rigid["receptive_object"]["root_pose"]])
    dist_xy = torch.linalg.vector_norm(leg_pos[:, :2] - fix_pos[:, :2], dim=-1)
    all_dist.append(dist_xy)
    total_states += leg_pos.shape[0]
    print(f"[c4-check] {f}: n={leg_pos.shape[0]} leg-fixture XY dist median={dist_xy.median():.4f}m"
          f" max={dist_xy.max():.4f}m")

all_dist_t = torch.cat(all_dist)
print(f"[c4-check] TOTAL n={total_states} leg-fixture XY dist: median={all_dist_t.median():.4f}m"
      f" max={all_dist_t.max():.4f}m mean={all_dist_t.mean():.4f}m")
print("[c4-check] native receptive_object confirmed on every raw chunk. Read the distance"
      " distribution above by hand -- a genuine partial-assembly bank should cluster tightly"
      " (well under the fixture's own RECEPTIVE_POSE_RANGE footprint); no pre-measured threshold"
      " exists for this checkpoint, this is the first time C4 has been generated.")
PYEOF

for i in $(seq 1 3); do
  CLOG=$(ls "$REPO_ROOT"/fourbank_c4_chunk${i}.log 2>/dev/null)
  [ -n "$CLOG" ] && grep -q "DEXLIFT_PARTIAL_ASSEMBLY=1.*SpawnPartialAssembly" "$CLOG" \
    && echo "[c4-check] chunk $i confirms SpawnPartialAssembly construction: $(grep -o 'reset_object -> SpawnPartialAssembly[^;]*' "$CLOG" | head -1)" \
    || echo "[c4-check] WARNING: chunk $i log missing the SpawnPartialAssembly construction line -- verify by hand." >&2
done

# =====================================================================================
# REKEY -- C1/C3 ONLY. NOT C4, AND THIS IS DELIBERATE, FOUND BY READING rekey's OWN REFUSAL
# CONDITION RATHER THAN ASSUMED. generate_reset_states_policy.py wires
# _DexliftToTrainingSceneRecorder as env_cfg.recorders' class_type UNCONDITIONALLY (all four reset
# types), and that recorder already renames object->insertive_object, drops table, and KEEPS a
# real receptive_object entry inline, live, during generation -- so C4's raw/merged output is
# ALREADY {"insertive_object","receptive_object"}, not the {"object","table"} rekey_dexlift_
# reset_states.py's _detect_schema expects as its "raw_dexlift" input. That function explicitly
# REFUSES (raises ValueError) the moment receptive_object is already present, calling such a file
# "already schema-complete" -- running rekey on C4's merged bank would error, not help.
# table/ur5_metal_support stay ABSENT from C4's bank as a result. THIS IS NOT SAFE BY THE "THEY
# ARE KINEMATIC" ARGUMENT ALONE -- checked, not assumed, because this exact project has a scar on
# that inference: MultiResetManager's own coverage guard (_assert_reset_file_covers_scene,
# omnireset/mdp/events.py) used to exempt any kinematic body and was WRONG to, because
# kinematic_enabled says nothing about whether the default pose is authored-correct or a
# placeholder -- make_receptive_object hardcodes it True for all ten receptive variants, INCLUDING
# receptive_object itself, whose own init_state is a bare (0,0,0) placeholder (the omission that
# was catastrophic). The guard was rewritten to require an EXPLICIT, per-config
# ``assumed_static_assets`` claim instead, cross-checked against kinematic_enabled but never
# inferred from it alone. VERIFIED against the actual consuming config, not inferred from the
# mechanism being "real": rl_state_cfg.py's RL_STATE_ASSUMED_STATIC_ASSETS = ["table",
# "ur5_metal_support"] (its own comment there gives each an authored, non-placeholder init_state
# as the reason) is passed as assumed_static_assets at every MultiResetManager EventTerm in that
# file, and receptive_object is deliberately NOT in that list, on record, for the same reason
# stated above. Construction-time check (OmniReset-UR5eDelto-RelCartesianOSC-State-v0, a two-key
# insertive_object+receptive_object bank standing in for a merged C4 output): loads through
# MultiResetManager without raising.
#
# THIS CLEARANCE IS SCOPED TO THAT DECLARATION, NOT UNIVERSAL -- the same seam (a fact verified
# under one config, consumed under a different one, nothing gating the two) that produced most of
# today's other defects. A C4 bank from this script is valid for a MultiResetManager EventTerm
# whose OWN assumed_static_assets declares table and ur5_metal_support static (rl_state_cfg.py's
# TrainEventCfg/TrainEvalEventCfg/etc. all do, today). If it is ever pointed at a DIFFERENT
# consumer whose assumed_static_assets omits either name, that construction will raise --
# loudly, at env construction, before any training step, which is the correct failure mode; it is
# not something this script can guard against generically, since which config it does what
# is a fact about the CONSUMER, not about the bank file itself.
#
# C1/C3 have no receptive_object at all straight out of the recorder (that scene never has the
# entity outside the C4-only legacy toggle), so their merged output IS rekey's "already_rekeyed"
# input case and needs the full four-key synthesis -- receptive_object is NOT assumed-static
# anywhere, so a C1/C3 bank omitting it would raise regardless of which config consumes it.
for arm_type in "c1:ObjectAnywhereEEAnywhere" "c3:ObjectAnywhereEEGrasped"; do
  arm="${arm_type%%:*}"; reset_type="${arm_type##*:}"
  MERGED_FILE=$(find "$RUN_BASE/$arm/merged" -name "resets_${reset_type}.pt" | head -1)
  [ -n "$MERGED_FILE" ] || { echo "[$arm] REFUSING to rekey: no merged file found under $RUN_BASE/$arm/merged" >&2; continue; }
  REKEYED_FILE="$RUN_BASE/$arm/rekeyed_resets_${reset_type}.pt"
  "$PYTHON_BIN" scripts_v2/tools/rekey_dexlift_reset_states.py \
    --input "$MERGED_FILE" --output "$REKEYED_FILE" --seed 0 \
    > "$REPO_ROOT/fourbank_${arm}_rekey.log" 2>&1
  cat "$REPO_ROOT/fourbank_${arm}_rekey.log"
done
# C2's four offset files are NOT rekeyed either -- they carry only the insertive object's
# rewound pose (same schema C2's own paper-scale script already produces and the training scene
# already consumes unmodified); rekeying is a dexlift-scene -> training-scene bridge C2's output
# never needed in the first place. Confirm this assumption against merge_c2_chunks.py's own
# output schema before copy-in if that has changed since this script was staged.

# =====================================================================================
# VALIDATE every finished bank by loading it -- C1/C3 (rekeyed, four keys expected) AND C4
# (merged, straight from the recorder, two keys expected -- see the rekey section immediately
# above for why C4 is not, and should not be, rekeyed).
# =====================================================================================
"$PYTHON_BIN" - <<PYEOF
import glob, torch

FOUR_KEY_EXPECTED = {"insertive_object", "receptive_object", "table", "ur5_metal_support"}
C4_EXPECTED = {"insertive_object", "receptive_object"}

def check_bank(f, expected_keys, label):
    print(f"=== {f}  [{label}] ===")
    d = torch.load(f, map_location="cpu", weights_only=False)
    state = d["initial_state"]
    rigid = state["rigid_object"]
    robot = state["articulation"]["robot"]

    keys = set(rigid.keys())
    ok = keys == expected_keys
    print(f"  rigid_object keys: {sorted(keys)}  expected {sorted(expected_keys)}  {'OK' if ok else 'MISMATCH'}")

    n = len(robot["root_pose"])
    print(f"  state count (robot root_pose): {n}")
    for name, entry in rigid.items():
        ln = len(entry["root_pose"])
        print(f"  {name}: root_pose len={ln}  {'OK' if ln == n else 'LENGTH MISMATCH'}")

    has_pt = "joint_position_target" in robot
    has_vt = "joint_velocity_target" in robot
    print(f"  joint_position_target present: {has_pt}  joint_velocity_target present: {has_vt}")
    if has_pt:
        print(f"    len={len(robot['joint_position_target'])}  {'OK' if len(robot['joint_position_target']) == n else 'LENGTH MISMATCH'}")
    if not (has_pt and has_vt):
        print("  *** WARNING: missing PD target field(s) -- bead UWLab-algw.7's recorder fix should"
              " make this present on any bank generated from the current tree; investigate before"
              " trusting this bank for training. ***")

    if "receptive_object" in rigid:
        rz = torch.stack([p[2] for p in rigid["receptive_object"]["root_pose"]])
        print(f"  receptive_object z: median={rz.median():.6f} std={rz.std():.8f}"
              f"  {'OK (== 0.019625)' if abs(rz.median() - 0.019625) < 1e-6 and rz.std() < 1e-9 else 'UNEXPECTED'}")

for f in sorted(glob.glob("$RUN_BASE/c1/rekeyed_resets_*.pt")) + sorted(glob.glob("$RUN_BASE/c3/rekeyed_resets_*.pt")):
    check_bank(f, FOUR_KEY_EXPECTED, "rekeyed, four keys expected")

for f in sorted(glob.glob("$RUN_BASE/c4/merged/Resets/*/resets_ObjectPartiallyAssembledEEGrasped.pt")):
    check_bank(f, C4_EXPECTED, "NOT rekeyed -- recorder already schema-complete, see rekey section")
PYEOF

# =====================================================================================
# COPY-IN IS A SEPARATE, HUMAN-GATED STEP -- print the command, never run it.
# =====================================================================================
PAIR_DIR="OneLegInsertionFixture__SquareTableLeg200mmDecomp"
echo
echo "############################################################################"
echo "# All four arms generated and merged; C1/C3 rekeyed (four keys); C4 left as the recorder"
echo "# produced it -- NOT rekeyed, see the rekey section's comment for why that is correct, not"
echo "# an omission. NOTHING has been copied into $PRODUCTION_DATASET_DIR yet. Review every"
echo "# validation block above BY EYE, especially the C4 leg-fixture distance"
echo "# distribution (no prior threshold exists to auto-gate it), THEN run:"
echo "#"
echo "#   cp \"$RUN_BASE/c1/rekeyed_resets_ObjectAnywhereEEAnywhere.pt\" \\"
echo "#      \"$PRODUCTION_DATASET_DIR/Resets/$PAIR_DIR/resets_ObjectAnywhereEEAnywhere.pt\""
echo "#   cp \"$RUN_BASE/c3/rekeyed_resets_ObjectAnywhereEEGrasped.pt\" \\"
echo "#      \"$PRODUCTION_DATASET_DIR/Resets/$PAIR_DIR/resets_ObjectAnywhereEEGrasped.pt\""
echo "#   cp \"\$(find \"$RUN_BASE/c4/merged\" -name resets_ObjectPartiallyAssembledEEGrasped.pt)\" \\"
echo "#      \"$PRODUCTION_DATASET_DIR/Resets/$PAIR_DIR/resets_ObjectPartiallyAssembledEEGrasped.pt\""
echo "#   cp \"$C2_MERGED_DIR\"/Resets/*/resets_ObjectAnywhereEENear_off*.pt \\"
echo "#      \"$PRODUCTION_DATASET_DIR/Resets/$PAIR_DIR/\""
echo "#"
echo "# The production directory was already backed up once, unconditionally, before this run"
echo "# started (see the [backup] line near the top of this script's output/log)."
echo "############################################################################"
