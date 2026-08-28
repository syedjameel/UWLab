#!/usr/bin/env bash
# STAGED, NOT YET RUN (bead UWLab-weyl). Paper-scale C2-via-rewind generation on the Reorient
# table-leg checkpoint, carrying every production-matching setting this project measured today --
# see the 4-message investigation this script closes out (plant already right; episode_length_s
# and DEXLIFT_POSE_TILT were both wrong and both mattered, tilt more than length: 6.63% -> 23.62%
# -> 58.17% acceptance as each was fixed, against production's own ~62%). DO NOT RUN while the box
# is doing other heavy work -- as of staging, C3 is mid-flight on this same box with hours left.
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/source/uwlab:$REPO_ROOT/source/uwlab_tasks:$REPO_ROOT/source/uwlab_assets:$REPO_ROOT/source/uwlab_rl"

# -- REFERENCE PLANT (all four, together -- a partial export silently builds the IDENTIFIED hand
# against a REFERENCE-trained checkpoint; measured history: 46.71% -> 2.69% acceptance). Verify
# this run's own [verify] lines read hand effort_limit_sim=[30.0] and velocity_limit_sim=[10000.0]
# before trusting anything else it prints.
export DEXLIFT_REF_RESET=1
export DEXLIFT_REF_ACTUATORS=1
export DEXLIFT_REF_HAND_ACT=1
export DEXLIFT_REF_ARM_ACT=0

# -- POSE TILT (task-definition, not a tuning knob -- see _apply_pose_tilt_stage's own docstring).
# The single biggest lever measured today: with plant+episode_length already correct but this
# UNSET, acceptance was 23.62%; with it set to the SAME value production uses, 58.17%. Also clamps
# reset_object's drop height from [0.0, 0.4] to [0.0, DEXLIFT_DROP_Z] (default 0.05, left at
# default here -- unset in production's own command, do not add it).
export DEXLIFT_POSE_TILT=0.3

TASK="DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-Play-v0"
RESET_TYPE="ObjectAnywhereEEGrasped"
RECEPTIVE_USD_PATH="$REPO_ROOT/source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/one_leg_insertion_fixture.usd"

# -- CHECKPOINT: the certified Stage-3 Reorient checkpoint, per the box's own layout (matches this
# bead's original task brief: /root/ckpt/<name> with --agent_yaml alongside it). FAIL FAST rather
# than silently falling back to this tool's own <ckpt's parent's parent>/params/agent.yaml
# derivation, which assumes a runs/<ts>/nn/<file>.pth layout this flat ckpt/ directory does not
# have -- getting this wrong silently generates under the WRONG checkpoint's hyperparameters
# (normalize_input's RunningMeanStd shape et al), not an error.
CKPT="${C2_PAPERSCALE_CKPT:-/root/ckpt/last_dexlift_ur5e_delto_reljointpos_tableleg_reorient_ep_3600_rew_38.38917.pth}"
AGENT_YAML="${C2_PAPERSCALE_AGENT_YAML:?Set C2_PAPERSCALE_AGENT_YAML to the boxs agent.yaml beside CKPT before running -- refusing to guess a derived path.}"

# -- YIELD-BASED TARGET, BIASED UP (team-lead instruction: the accept-time bank is a throwaway
# under the isolation below, so undershooting costs a second Isaac boot while overshooting only
# costs marginal GPU time -- bias up, not down). Measured states-per-ACCEPTED-episode at the
# corrected config: 464 C2 states / 153 accepted = 3.03. Raw parity target with C3's own
# 10,000-accepted bank would be 10000 / 3.03 ~= 3300 accepted episodes; padded to 4200 (+27%).
# --num_reset_states counts ACCEPTED episodes (this tool's loop stops on
# n_accepted_at_reset >= this value), not attempts and not C2 states -- do not confuse the three.
# This is an ESTIMATE off an n=153 sample -- MONITOR the actual C2 yield via each chunk's own
# periodic "[c2][progress] total_emitted=..." lines (piggybacks on the existing --progress_
# every_episodes/--progress_every_seconds heartbeat) and stop a chunk once its share is there,
# rather than waiting for --num_reset_states to be reached -- see STOPPING A CHUNK EARLY below.
NUM_RESET_STATES_TOTAL="${C2_PAPERSCALE_NUM_RESET_STATES:-4200}"

# -- CHUNKED, NOT ONE LONG PROCESS (team-lead instruction, same trap that derailed C3): the
# generator rewrites its ENTIRE accept-time bank on every accepted state, so per-state cost is
# linear in n and TOTAL cost is quadratic -- measured on C3, windowed rate fell
# 0.97->0.645->0.395->0.327 accepted/s as n grew, fitting rate ~ 1034/(466+n). This run pays that
# SAME cost on its throwaway accept-time bank. Splitting into N_CHUNKS pieces, each its own Isaac
# process writing to its own isolated directory, avoids the quadratic tail and -- independently --
# bounds a crash's blast radius: this tool's C2 files are written ONCE, by write(), at the very
# end, so one long process losing at hour 2 loses everything, while one chunk losing loses only
# that chunk's share. merge_c2_chunks.py (this same commit) recombines the chunks afterward --
# see its own docstring for the schema (LISTS of per-state tensors, concatenate not stack) and the
# hard count-check it runs (merged count must equal the exact sum of the chunks').
N_CHUNKS="${C2_PAPERSCALE_N_CHUNKS:-3}"
PER_CHUNK_TARGET=$(( (NUM_RESET_STATES_TOTAL + N_CHUNKS - 1) / N_CHUNKS ))  # ceiling division

# -- DISTINCT SEED PER CHUNK, NOT JUST A DISTINCT DIRECTORY (team-lead, traced empirically:
# GenRunner's own same-seed comparison of two chunks at n~3000 each). Isolated output directories
# stop chunks from CLOBBERING each other but do nothing for DIVERSITY: agent.yaml's params.seed is
# read by rl_games' torch_runner.load_config and pins torch/numpy's GLOBAL RNG before the env is
# ever constructed, identically in every process launched with the same yaml. Same checkpoint +
# same task + same seed means the only thing differentiating chunks would be uncontrolled
# GPU/PhysX non-determinism -- real, but not something to design a dataset's diversity around, and
# states can stay highly CORRELATED (same spawn/goal draws, slowly diverging) even where they are
# not byte-identical. make_seeded_agent_yaml.py (this same commit) writes ONE field, params.seed,
# and verifies its own write by reloading; BASE_SEED+i-1 gives chunk 1 the source yaml's original
# seed and chunks 2..N distinct ones on top of it.
BASE_SEED="${C2_PAPERSCALE_BASE_SEED:-42}"

# -- OUTPUT ISOLATION (fixes a real defect caught in review of the first staged version of this
# script): --reset_type ObjectAnywhereEEGrasped resolves to the SAME resets_ObjectAnywhereEEGrasped
# .pt file the currently-running C3 job is writing, if a chunk's dataset_dir is left at this
# tool's own default (the shared production root). A chunk run there would truncate 10,000
# hard-won C3 states down to whatever that chunk reached. FIX: EVERY chunk's entire output -- both
# its throwaway accept-time bank and its real C2 files -- goes into its OWN isolated subdirectory
# under one timestamped run base, never the shared root, by DEFAULT; C2_PAPERSCALE_DATASET_DIR can
# still override the base, so a hard guard below refuses to run if the resolved base ever matches
# the production root, however it got set -- belt and braces, not just a different default (and
# since every chunk nests under this same guarded base, no chunk can collide with the shared root
# even if the ceiling-division/loop below had a bug).
PRODUCTION_DATASET_DIR="./Datasets_ur5e_delto/OmniReset"  # this tool's own default; what C3 uses
SCRATCH_RUN_BASE="./Datasets_ur5e_delto/OmniReset_C2_SCRATCH_$(date +%Y%m%d_%H%M%S)"
RUN_BASE="${C2_PAPERSCALE_DATASET_DIR:-$SCRATCH_RUN_BASE}"
if [ "$(realpath -m "$RUN_BASE")" = "$(realpath -m "$PRODUCTION_DATASET_DIR")" ]; then
  echo "REFUSING TO RUN: the resolved run base ($RUN_BASE) matches the shared production root" \
       "($PRODUCTION_DATASET_DIR), which C3 is currently writing to. Leave" \
       "C2_PAPERSCALE_DATASET_DIR unset (isolated scratch default) or point it somewhere else." >&2
  exit 1
fi
echo "[isolation] all $N_CHUNKS chunks write under: $RUN_BASE  (production root untouched: $PRODUCTION_DATASET_DIR)"

PYTHON_BIN="${C2_PAPERSCALE_PYTHON_BIN:?Set C2_PAPERSCALE_PYTHON_BIN to the boxs Isaac-enabled python before running.}"

# -- BACK UP THE PRODUCTION BANK ONCE, BEFORE ANY CHUNK RUNS (belt, independent of whether the
# isolation above works as designed -- hard requirement, learned today: this generator truncates
# and rewrites its output in place several times a minute; a crash mid-write zeroes a bank with no
# recovery). This backs up the REAL production Resets/ directory regardless of what any chunk's
# dataset_dir resolves to -- it is not a substitute for the isolation guard above, it is what
# protects C3's bank if that guard is ever bypassed or this script is copied and re-parameterized
# carelessly later.
PRODUCTION_RESETS_DIR="$PRODUCTION_DATASET_DIR/Resets"
if [ -d "$PRODUCTION_RESETS_DIR" ]; then
  BACKUP_DIR="${PRODUCTION_RESETS_DIR%/}_backup_$(date +%Y%m%d_%H%M%S)"
  cp -a "$PRODUCTION_RESETS_DIR" "$BACKUP_DIR" || { echo "BACKUP FAILED -- refusing to run without one" >&2; exit 1; }
  echo "[backup] $PRODUCTION_RESETS_DIR -> $BACKUP_DIR"
fi

# -- PER-CHUNK TIMEOUT, NOT ONE MONOLITHIC ONE (hard requirement, learned today -- a tight timeout
# nearly destroyed C3). Each chunk gets its OWN 12h SAFETY NET for a genuinely hung process
# (Isaac's app.close() hangs routinely; a missing completion line means hung, not crashed -- kill
# -s KILL, not -TERM, which Kit ignores), not a work budget: measured yield (states-per-attempt
# 1.764 on Reorient) suggests one chunk needs a small fraction of that. Raise
# C2_PAPERSCALE_CHUNK_TIMEOUT_S if a chunk is legitimately still progressing (check its own
# [progress]/[c2][progress] lines) when it fires. One chunk hitting its net does not block the
# others -- the loop below moves on regardless, and merge_c2_chunks.py tolerates a chunk that
# produced nothing (WARNING, not FATAL) as long as at least one chunk has usable output.
CHUNK_TIMEOUT_S="${C2_PAPERSCALE_CHUNK_TIMEOUT_S:-43200}"

# STOPPING A CHUNK EARLY, DELIBERATELY, ONCE ITS SHARE IS THERE: PER_CHUNK_TARGET is padded
# (ceiling of an already-padded total) precisely so it is NOT the real stopping signal for that
# chunk -- watch a chunk's own log for "[c2][progress] total_emitted=..." and, once its share
# looks sufficient, stop just that chunk's process yourself (SIGKILL) and let the loop move on to
# the next one. This is SAFE at any point -- every chunk's output is isolated (see above), so
# there is nothing production to corrupt by stopping a chunk early or by its 12h net firing. Run
# this script itself under `nohup bash run_c2_rewind_paperscale_reorient.sh > script.out 2>&1 &`
# (or tmux/screen) so it survives a dropped connection, then `tail -f` whichever chunk's log you
# are watching.
CHUNK_DIRS=()
ANY_SEED_MISMATCH=0
for i in $(seq 1 "$N_CHUNKS"); do
  CHUNK_DIR="$RUN_BASE/chunk_$i"
  CHUNK_DIRS+=("$CHUNK_DIR")
  CHUNK_LOG="$REPO_ROOT/c2_rewind_paperscale_reorient_chunk${i}.log"
  CHUNK_SEED=$(( BASE_SEED + i - 1 ))
  CHUNK_AGENT_YAML="$RUN_BASE/agent_seed_${CHUNK_SEED}.yaml"

  # Write this chunk's seeded yaml BEFORE launching it. FATAL (not a warning) if this fails --
  # unlike a chunk's own generation failing, a seed we cannot verify was even written correctly
  # means this chunk would silently duplicate another one's RNG state, defeating the entire point
  # of chunking for diversity.
  "$PYTHON_BIN" scripts_v2/tools/make_seeded_agent_yaml.py \
    --source_yaml "$AGENT_YAML" --seed "$CHUNK_SEED" --output_yaml "$CHUNK_AGENT_YAML" \
    || { echo "[chunk $i/$N_CHUNKS] FATAL: could not write a seeded agent.yaml -- aborting" \
              "before spending any GPU time on a chunk whose seed we cannot confirm." >&2; exit 1; }

  echo "[chunk $i/$N_CHUNKS] target=$PER_CHUNK_TARGET accepted  seed=$CHUNK_SEED  dataset_dir=$CHUNK_DIR  log=$CHUNK_LOG"
  timeout -s KILL "$CHUNK_TIMEOUT_S" "$PYTHON_BIN" -u scripts_v2/tools/generate_reset_states_policy.py \
    --task "$TASK" \
    --reset_type "$RESET_TYPE" \
    --receptive_usd_path "$RECEPTIVE_USD_PATH" \
    --checkpoint "$CKPT" \
    --agent_yaml "$CHUNK_AGENT_YAML" \
    --dataset_dir "$CHUNK_DIR" \
    --episode_length_s 8.0 \
    --num_envs 128 \
    --num_reset_states "$PER_CHUNK_TARGET" \
    --headless \
    --c2_rewind \
    --c2_reset_type ObjectAnywhereEENear \
    > "$CHUNK_LOG" 2>&1
  CHUNK_EXIT=$?
  echo "EXIT_CODE=$CHUNK_EXIT" >> "$CHUNK_LOG"
  if [ "$CHUNK_EXIT" -ne 0 ]; then
    echo "[chunk $i/$N_CHUNKS] WARNING: exited $CHUNK_EXIT (see $CHUNK_LOG) -- continuing to the" \
         "next chunk regardless; merge_c2_chunks.py below tolerates a chunk with no usable output." >&2
  fi

  # -- VERIFY THE SEED THAT ACTUALLY LOADED, FROM THIS CHUNK'S OWN LOG, AT BOTH THE PRODUCER AND
  # THE CONSUMER (team-lead correction: these two lines prove DIFFERENT things, checking only one
  # is the same "verify at the consumer, not the producer" seam that already bit this campaign
  # twice today -- a plant verified from env vars instead of the constructed articulation, a bank
  # verified by filename instead of contents).
  #   - "[generator] agent_cfg params.seed: N" (this bead's own print, fires early) proves the
  #     YAML CONTAINED the intended seed -- i.e. make_seeded_agent_yaml.py's write was read back
  #     correctly. Producer side.
  #   - rl_games' own "self.seed = N" (unprompted, confirmed present in a real log at this bead's
  #     own smoke-test stage) proves RL_GAMES ACTUALLY CONSUMED params.seed -- if rl_games ever
  #     silently ignored that key (wrong nesting, a schema change), the first line could read N
  #     while the process still ran at whatever it defaulted to, and only this second line would
  #     catch it. Consumer side.
  # Both must confirm; report which one(s) failed distinctly rather than collapsing to one bit.
  YAML_OK=0
  RLGAMES_OK=0
  grep -q "agent_cfg params.seed: $CHUNK_SEED\$" "$CHUNK_LOG" && YAML_OK=1
  grep -q "self.seed = $CHUNK_SEED\$" "$CHUNK_LOG" && RLGAMES_OK=1
  if [ "$YAML_OK" -eq 1 ] && [ "$RLGAMES_OK" -eq 1 ]; then
    echo "[chunk $i/$N_CHUNKS] seed verified at both producer and consumer: $CHUNK_SEED (see $CHUNK_LOG)"
  else
    ANY_SEED_MISMATCH=1
    echo "[chunk $i/$N_CHUNKS] *** SEED NOT FULLY CONFIRMED (expected $CHUNK_SEED): yaml-contained-it=$YAML_OK" \
         "rl_games-consumed-it=$RLGAMES_OK -- see $CHUNK_LOG (chunk may have crashed before printing," \
         "or one side silently disagreed with the other). This chunk's contribution to the" \
         "diversity check below is suspect. ***" >&2
  fi

  # -- EMPIRICAL DIVERSITY CHECK, AS SOON AS TWO CHUNKS EXIST, BEFORE SPENDING GPU ON A THIRD
  # (team-lead requirement). With distinct per-chunk seeds (above) this check's role has flipped:
  # it started as a GUARD against accidentally-identical chunks (traced root cause: rl_games pins
  # torch/numpy's global RNG from agent.yaml's params.seed, identically, before per-chunk seeding
  # existed here), and is now a cheap CONFIRMATION that the seeding actually took effect -- it
  # should come back near zero. --report_only writes nothing -- pure diagnostic, safe mid-batch.
  # This does NOT auto-abort remaining chunks -- deciding what a nonzero result means (seed
  # verification failed above? still just correlated draws?) is a judgment call for whoever is
  # watching this run, not something to silently automate. The WARNING line (if any) is the
  # loudest this script gets; read it.
  if [ "$i" -eq 2 ]; then
    DUP_CHECK_LOG="$REPO_ROOT/c2_rewind_paperscale_reorient_dupcheck_chunk1v2.log"
    echo "[dup-check] chunk 1 vs chunk 2 -- checking BEFORE chunk 3 starts (see $DUP_CHECK_LOG)"
    "$PYTHON_BIN" scripts_v2/tools/merge_c2_chunks.py \
      --chunk_dirs "${CHUNK_DIRS[0]}" "${CHUNK_DIRS[1]}" \
      --c2_reset_type ObjectAnywhereEENear \
      --report_only \
      > "$DUP_CHECK_LOG" 2>&1
    cat "$DUP_CHECK_LOG"
    if grep -q "WARNING: .*near/exact match" "$DUP_CHECK_LOG"; then
      echo "[dup-check] *** chunk 1 and chunk 2 look like clones (see WARNING above) -- a third" \
           "identical chunk would be wasted GPU. STOP AND INVESTIGATE before chunk 3 launches" \
           "if you are watching this run live; it will proceed automatically otherwise. ***" >&2
    fi
  fi
done

if [ "$ANY_SEED_MISMATCH" -eq 1 ]; then
  echo "[chunks] SUMMARY: at least one chunk's expected seed was NOT confirmed in its own log --" \
       "see the *** SEED NOT CONFIRMED *** lines above before trusting the duplicate-check result" \
       "or this bank's diversity." >&2
fi

# -- MERGE THE CHUNKS (team-lead instruction: merge the per-offset files across chunks BEFORE the
# manual copy-in, not per-chunk copy-ins). See merge_c2_chunks.py's own docstring for the schema
# this depends on and the hard count-check it runs. Runs with the SAME python as generation (it
# needs torch, which that interpreter already has -- it does not need Isaac).
MERGED_DIR="$RUN_BASE/merged"
MERGE_LOG="$REPO_ROOT/c2_rewind_paperscale_reorient_merge.log"
"$PYTHON_BIN" scripts_v2/tools/merge_c2_chunks.py \
  --chunk_dirs "${CHUNK_DIRS[@]}" \
  --output_dir "$MERGED_DIR" \
  --c2_reset_type ObjectAnywhereEENear \
  > "$MERGE_LOG" 2>&1
MERGE_EXIT=$?
echo "EXIT_CODE=$MERGE_EXIT" >> "$MERGE_LOG"
cat "$MERGE_LOG"
if [ "$MERGE_EXIT" -ne 0 ]; then
  echo "[merge] FAILED (exit $MERGE_EXIT, see $MERGE_LOG) -- per-chunk output is still intact" \
       "under $RUN_BASE/chunk_*/ if you need to merge by hand or investigate." >&2
  exit "$MERGE_EXIT"
fi

# -- DELIBERATE, MANUAL, VALIDATED COPY-IN -- NOT AUTOMATED (team-lead instruction: copy ONLY the
# C2 files into the real Resets/ directory afterwards, deliberately, once validated -- not as part
# of this same run). This script stops here and does NOT touch $PRODUCTION_DATASET_DIR. To promote
# the MERGED result once you have checked it (schema, report() numbers per chunk, resting-filter
# rejection counts, and the merge's own printed per-offset counts -- all already in the chunk logs
# and "$MERGE_LOG"):
#
#   1. Inspect: grep -A2 "C2-VIA-REWIND RESULT" c2_rewind_paperscale_reorient_chunk*.log
#      and "$MERGE_LOG"'s own per-offset "N + N + N = TOTAL" lines -- confirm TOTAL matches what
#      you expect and the per-chunk counts add up (the merge script asserts this itself, but read
#      it here too).
#   2. Copy ONLY the four MERGED C2 offset files (the wildcard below already excludes every
#      chunk's throwaway accept-time bank -- merge_c2_chunks.py never touches those files):
#        cp "$MERGED_DIR"/Resets/*/resets_ObjectAnywhereEENear_off*.pt \
#           "$PRODUCTION_DATASET_DIR"/Resets/<the same pair directory>/
#   3. Verify the copy (file sizes, or re-run this bead's own torch.load schema probe) before
#      relying on it for training.
echo "[next steps] Merged output at: $MERGED_DIR  (per-chunk output still at $RUN_BASE/chunk_*/)"
echo "[next steps] Validate, THEN manually copy resets_ObjectAnywhereEENear_off*.pt (only, from" \
     "the MERGED dir) into $PRODUCTION_DATASET_DIR/Resets/<pair dir> -- see this script's own" \
     "trailing comment block."
