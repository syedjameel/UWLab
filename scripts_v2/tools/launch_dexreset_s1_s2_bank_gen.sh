#!/usr/bin/env bash
# STAGED, NOT RUN (epic UWLab-nnlv). Generate the two NEW DexReset reset rungs, S1 and S2', with
# generate_reset_states_policy.py driven by a POLICY checkpoint -- the vertical-goal finetune's
# output, which does not exist yet. Do not run this until that finetune certifies.
#
# Usage:
#   launch_dexreset_s1_s2_bank_gen.sh <policy_ckpt.pth> [S1|S2P|both]
#
# Environment (all optional except the two marked REQUIRED):
#   PYTHON_BIN                       REQUIRED. Isaac-enabled interpreter on this box.
#   PARTIAL_ASSEMBLY_DATASET_DIR     REQUIRED. Root holding Resets/<pair>/partial_assemblies.pt.
#   AGENT_YAML                       defaults to <ckpt>/../../params/agent.yaml, and is always
#                                    passed EXPLICITLY (see the production-config section).
#   GPU, TARGET_PER_RUNG, CHUNK_TARGET_SIZE, CHUNK_TIMEOUT_S, BASE_SEED, RUN_BASE
#   S1_GOAL_BELOW_SPAWN_MM, S2P_GOAL_BELOW_SPAWN_MM  (see the shaping section)
#
# WHAT THESE TWO RUNGS ARE, in the geometry --c4_seating_gate actually measures. Depth is SIGNED
# and measured in mm BELOW THE BORE MOUTH (generate_reset_states_policy.py's _MatingFrameGeometry
# .decompose: depth_m = engaged_span_m - z_t), the engaged span mouth->seat is 25 mm, and
# _SeatingGateAddon asserts only depth_min < depth_max -- so a NEGATIVE depth is a legal band and
# means the tip is ABOVE the mouth.
#
#   S1   tip just inside the bore   depth   0 .. 10 mm   lateral <=  5 mm   tilt <= 15 deg
#   S2'  tip above the bore         depth -120 .. -20 mm lateral <= 20 mm   tilt <= 25 deg
#
# =====================================================================================
# BOTH RUNGS ARE GENERATED UNDER --reset_type ObjectPartiallyAssembledEEGrasped. THE RUNG NAME IS
# A DOWNSTREAM RENAME.
# =====================================================================================
# generate_reset_states_policy.py's --reset_type carries choices=_CANONICAL_RESET_TYPES -- exactly
# ObjectAnywhereEEAnywhere / ObjectRestingEEGrasped / ObjectAnywhereEEGrasped /
# ObjectPartiallyAssembledEEGrasped. ObjectAtBoreMouthEEGrasped and ObjectAboveBoreEEGrasped are
# NOT accepted and argparse rejects them before Isaac ever boots. That is not an obstacle, it is
# the correct coupling: --c4_seating_gate asserts --reset_type == ObjectPartiallyAssembledEEGrasped
# (nothing else has a receptive_object to measure against), and that in turn is _require()d to
# agree with BOTH DEXLIFT_PARTIAL_ASSEMBLY=1 AND events.reset_object.func actually having been
# built as SpawnPartialAssembly. So the generator runs as C4 and each merged bank is RENAMED to its
# rung's own name at the end -- a filename change on an already-written file, with no reinterpretation
# of its contents.
#
# A CONSUMER MUST BE TOLD THESE NAMES. MultiResetManager matches banks by the reset_types entries
# its own cfg declares; a bank called resets_ObjectAtBoreMouthEEGrasped.pt is invisible to a config
# that never names it. Wiring that is a separate job, out of scope here.
#
# =====================================================================================
# --c4_seating_gate IS OPT-IN AND MUST BE PASSED. A RUN THAT FORGETS IT LOOKS SUCCESSFUL.
# =====================================================================================
# Without the flag the acceptance decision is held_with_probe alone, which has NO spatial term at
# all: it asks "is the object held", never "is it still where this rung needs it". Measured on a
# 25%-partial-assembly finetune, n=100 accepted: 0/100 inside any seated depth band, median lateral
# miss 21.55 mm, median tilt 12.32 deg, with carry-away poses out at lateral 146-395 mm and tilt
# 69-90 deg. Every one of those is "accepted", banked, and indistinguishable from a good state by
# exit code, log banner, or state count. The flag is passed unconditionally below and its
# constructed band is verified out of each chunk's own log afterwards.
#
# =====================================================================================
# CHUNKING IS MANDATORY -- SINGLE-PROCESS COST IS QUADRATIC IN BANK SIZE
# =====================================================================================
# The generator rewrites its ENTIRE accumulated bank file on every accepted state. Measured on C3:
# windowed accept rate 0.97 -> 0.645 -> 0.395 -> 0.327 states/s as n grew inside one process. One
# 10,000-state run projects to ~13 h; chunks of ~1800 to ~3 h. Each chunk writes to its own
# --dataset_dir and the chunks are merged afterwards; chunks are never consolidated mid-run to save
# an Isaac boot.
set -uo pipefail

CKPT=${1:?"usage: $0 <policy_ckpt.pth> [S1|S2P|both]"}
RUNGS=${2:-both}

# -- Whole-token match, not a prefix/glob. `case ... in S1*)` would accept "S1nonsense" and run
# something nobody typed; a wrong rung here is 3+ GPU-hours spent on the wrong depth band.
[[ "$RUNGS" =~ ^(S1|S2P|both)$ ]] || {
  echo "REFUSING: rung selector must be exactly S1, S2P, or both; got '$RUNGS'"; exit 1;
}
[ -f "$CKPT" ] && [ -r "$CKPT" ] || {
  echo "REFUSING: policy checkpoint '$CKPT' is not a readable file."; exit 1;
}

PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Isaac-enabled python interpreter path on this box before running.}"
[ -x "$PYTHON_BIN" ] || { echo "REFUSING: PYTHON_BIN '$PYTHON_BIN' is not executable"; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# -- EXPLICIT agent yaml, never the generator's own derived-path fallback. Derived here so the
# caller need not repeat it, but REQUIRED to exist and always passed on the command line: the
# fallback silently reconstructs the same path and a missing file surfaces as an open() traceback
# 15 minutes into an Isaac boot instead of now.
AGENT_YAML="${AGENT_YAML:-$(dirname "$(dirname "$CKPT")")/params/agent.yaml}"
[ -f "$AGENT_YAML" ] || {
  echo "REFUSING: agent yaml not found at '$AGENT_YAML' -- set AGENT_YAML explicitly."; exit 1;
}

GPU="${GPU:-0}"
# -- Reject a shared GPU. Two Isaac processes on one device have been measured to return silently
# invalid physics at rc=0 -- frozen arms and falling parts, no error anywhere.
USED=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
[ "${USED:-999999}" -lt 2000 ] || { echo "REFUSING: GPU $GPU has ${USED} MiB already in use"; exit 1; }

# -- The vertical-goal MIXTURE is a TRAINING-time device and is inert here, so an exported value is
# a sign the caller has confused the two phases rather than a harmless extra. Generation forces
# DEXLIFT_PARTIAL_ASSEMBLY=1, which makes goal_at_spawn unconditionally true
# (_apply_partial_assembly_and_goal_toggles) and REPLACES commands.object_pose with the
# goal-pinned-at-spawn command -- whatever band _apply_goal_vertical_mixture staged earlier in the
# same __post_init__ chain is discarded, without an error. Refuse rather than run a job whose
# operator believes a knob is doing something it is not.
for _v in DEXLIFT_GOAL_VERTICAL_PROB DEXLIFT_GOAL_VERTICAL_TILT DEXLIFT_GOAL_VERTICAL_Z; do
  if [ -n "${!_v:-}" ]; then
    echo "REFUSING: $_v is exported. The vertical-goal mixture is a TRAINING-time device; at"
    echo "  generation DEXLIFT_PARTIAL_ASSEMBLY=1 pins the goal to the leg's own spawn pose and the"
    echo "  mixture is silently discarded. Unset it and re-run."
    exit 1
  fi
done

# =====================================================================================
# DEXLIFT_GOAL_BELOW_SPAWN_MM IS A SHAPING DEVICE, NOT A TARGET
# =====================================================================================
# It displaces the COMMANDED goal from the leg's spawn pose along the bore's own axis --
# `goal_pos_w = object_pos_w + delta_m * axis_world` (partial_assembly.py's _resample_command), so a
# POSITIVE value commands DEEPER INTO the bore and a NEGATIVE one commands the opposite way along
# that same axis, OUT of the mouth and above it. It does not set, promise, or bound the depth
# anything lands at. JUDGE A RUN BY THE BANKED DEPTH DISTRIBUTION -- the accepted counts and the
# gate band printed per chunk below -- NEVER by command-tracking success, which is measured against
# the shaped goal and not against the rung's band.
#
# WHAT THIS TREE ACCEPTS, read out of the code rather than assumed (bead UWLab-nnlv.3 made this
# SIGNED; re-read these four lines before trusting the range, they have moved once already):
#   dexlift_ur5e_delto_tableleg_env_cfg.py:392  assert goal_below_spawn_mm >= -200.0
#   dexlift_ur5e_delto_tableleg_env_cfg.py:403  goal_below_spawn = goal_below_spawn_mm != 0.0
#   mdp/partial_assembly.py:480                 assert delta_m >= -_ABOVE_MOUTH_LIMIT_M (0.200 m)
#   mdp/partial_assembly.py:502                 assert delta_m <= _ENGAGED_SPAN_M   (0.025 m)
# Accepted range is therefore -200.0 .. +25.0 mm, and 0.0 alone means no shaping command is
# installed at all. THE TWO BOUNDS ARE NOT THE SAME KIND OF NUMBER: +25 mm is the bore's own engaged
# span, a measured physical feature; -200 mm is a policy bound, headroom around S2's band chosen to
# catch a unit slip. Only the positive one describes the hardware.
#
#   S1  gets +5 mm. Spawn sits at depth 10.0-17.5 mm and the policy withdraws from there, so for a
#       0-10 mm band the shaping opposes the withdrawal. 5 mm is inside the 3-5 mm planned range
#       partial_assembly.py:510 warns above.
#
#   S2' gets -60 mm, and the sign is the point: the goal now ASKS FOR the withdrawal instead of
#       opposing it, and asks for it ALONG THE BORE AXIS. That axis is what the value is really
#       shaping against. The measured failure from a partial-assembly spawn is not too little
#       withdrawal, it is a CARRY-AWAY -- the policy flips the leg horizontal and leaves, at lateral
#       146-395 mm and tilt 69-90 deg, none of which survives S2's 20 mm / 25 deg acceptance. A
#       goal displaced up the bore axis is a request for the one withdrawal that stays inside those
#       two limits.
#
#       WHY -60 AND NOT THE BAND'S MIDPOINT. The command displaces from the SPAWN pose, so the
#       commanded tip depth is spawn_depth - 60 = 10.0-17.5 - 60 = -42.5 .. -50.0 mm. Two properties
#       of that placement, neither of which the midpoint gives you:
#         * The whole 7.5 mm spawn spread lands in one 7.5 mm window well inside S2's 100 mm band,
#           so no part of the spawn distribution starts an episode aimed at a band edge.
#         * The margins are DELIBERATELY ASYMMETRIC -- ~26 mm to the near edge (-20), ~74 mm to the
#           far edge (-120) -- and biased toward the near edge on purpose. Every measurement of this
#           policy says it errs by withdrawing TOO MUCH, so the long runway belongs on the overshoot
#           side, where over-withdrawal still banks, while the short margin guards the failure that
#           actually costs a state: falling back toward the mouth, out of S2' and into S1's
#           territory. A midpoint (-70) would spend margin on the direction the policy does not err
#           in.
#         * -60 also stays clear of partial_assembly.py:531's warning at -120 mm. A warning that
#           fires on every intended use is one nobody reads -- that file's own argument at :525-530.
#
#       WHAT TO EXPECT NOW, AND WHAT IS STILL UNMEASURED. S2' no longer depends on the policy
#       failing to track a goal that opposes it; the accepted states should now come from the policy
#       TRACKING a goal that is already inside the band, which is the same mechanism S1 relies on.
#       So the two rungs' yields should be the same order of magnitude, and the gate's lateral/tilt
#       terms -- not its depth term -- become the binding constraint for S2', since depth is what the
#       command is now steering and the carry-away is what it is steering away from. THAT IS A
#       PREDICTION, NOT A MEASUREMENT: no bank has ever been generated at either of these two depth
#       bands, on this checkpoint or any other. Read the per-chunk accepted counts before believing
#       any of this paragraph, and treat a large S1/S2' asymmetry as a finding rather than an error.
S1_GOAL_BELOW_SPAWN_MM="${S1_GOAL_BELOW_SPAWN_MM:-5}"
S2P_GOAL_BELOW_SPAWN_MM="${S2P_GOAL_BELOW_SPAWN_MM:--60}"

# -- Validated HERE against the same range the code enforces, so an out-of-range value costs a line
# of output instead of a 15-minute Isaac boot followed by an assert. Whole-token numeric match: a
# glob or a `case` prefix would let "5abc" through, printf would write "invalid number" to stderr
# and STILL print 5.000, and with `set -uo pipefail` (no -e) nothing would stop the run.
_check_shaping_mm() {  # name value
  local name="$1" value="$2"
  [[ "$value" =~ ^-?([0-9]+|[0-9]*\.[0-9]+)$ ]] || {
    echo "REFUSING: $name must be a number in mm (sign allowed); got '$value'."
    exit 1
  }
  # -- Range re-derived from the asserts themselves, not from a note about them. The floor and the
  # ceiling are cited separately because they fail in DIFFERENT places with different messages, and
  # a reader chasing one should not be sent to the other.
  awk -v v="$value" 'BEGIN { exit !(v >= -200.0) }' || {
    echo "REFUSING: $name=$value is below the signed floor of -200 mm, asserted at"
    echo "  dexlift_ur5e_delto_tableleg_env_cfg.py:392 and again as delta_m >= -_ABOVE_MOUTH_LIMIT_M"
    echo "  at mdp/partial_assembly.py:480. That floor is a POLICY bound (headroom around S2's"
    echo "  20-120mm band, sized to catch a unit slip), not a physical feature of the bore."
    exit 1
  }
  awk -v v="$value" 'BEGIN { exit !(v <= 25.0) }' || {
    echo "REFUSING: $name=$value exceeds the bore's engaged span (+25 mm), asserted as"
    echo "  delta_m <= _ENGAGED_SPAN_M at mdp/partial_assembly.py:502. Unlike the -200 mm floor this"
    echo "  one IS the hardware: there is no bore left to command into past the seat."
    exit 1
  }
}
_check_shaping_mm S1_GOAL_BELOW_SPAWN_MM "$S1_GOAL_BELOW_SPAWN_MM"
_check_shaping_mm S2P_GOAL_BELOW_SPAWN_MM "$S2P_GOAL_BELOW_SPAWN_MM"

export CUDA_VISIBLE_DEVICES="$GPU"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/source/uwlab:$REPO_ROOT/source/uwlab_tasks:$REPO_ROOT/source/uwlab_assets:$REPO_ROOT/source/uwlab_rl"

# -- BOTH tmp roots. IsaacLab's own logger goes through tempfile.gettempdir(), which UWLAB_TMP_ROOT
# does not cover, so a box whose /tmp/isaaclab is owned by another uid dies with PermissionError
# before the scene builds unless TMPDIR is set too.
export UWLAB_TMP_ROOT=${UWLAB_TMP_ROOT:-$HOME/tmp}
export TMPDIR=${TMPDIR:-$HOME/tmp}
mkdir -p "$UWLAB_TMP_ROOT" "$TMPDIR"

# -- PRODUCTION PLANT + START-POSE DISTRIBUTION. All four DEXLIFT_REF_* together plus the
# task-definition tilt. Every one of them is SILENT when missing and yields a plausible wrong
# number, not an error: the identified hand caps at 3.0 rad/s against a ~6 rad/s commanded closure,
# so its fingers cannot track a closing command at all, and the reset distribution silently becomes
# dexsuite's instead of this task's. Measured cost of omission: missing --episode_length_s 3.6x
# acceptance, missing DEXLIFT_POSE_TILT a further 2.5x on top of it. Verified from each chunk's own
# log below, never trusted from these lines.
export DEXLIFT_REF_RESET=1
export DEXLIFT_REF_ACTUATORS=1
export DEXLIFT_REF_HAND_ACT=1
export DEXLIFT_REF_ARM_ACT=0
export DEXLIFT_POSE_TILT=0.3

# -- The Reorient _PLAY task, NOT this generator's own --task default. The generator's default
# (...-TableLeg-Lift-Play-v0) never calls _apply_partial_assembly_and_goal_toggles, so
# DEXLIFT_PARTIAL_ASSEMBLY and DEXLIFT_GOAL_BELOW_SPAWN_MM would sit exported and unconsumed and
# the run would read as "the rung does not work" rather than "the toggle was never installed".
TASK="DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-Play-v0"
RECEPTIVE_USD_PATH="$REPO_ROOT/source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/one_leg_insertion_fixture.usd"
[ -f "$RECEPTIVE_USD_PATH" ] || { echo "REFUSING: fixture USD not found at $RECEPTIVE_USD_PATH"; exit 1; }

TARGET_PER_RUNG="${TARGET_PER_RUNG:-10000}"        # matches the existing production banks' size
CHUNK_TARGET_SIZE="${CHUNK_TARGET_SIZE:-1800}"     # accepted states/chunk; measured-safe band 1700-2000
CHUNK_TIMEOUT_S="${CHUNK_TIMEOUT_S:-43200}"
BASE_SEED="${BASE_SEED:-42}"
NUM_ENVS=128

STAMP="$(date +%Y%m%d_%H%M%S)"
PRODUCTION_DATASET_DIR="./Datasets_ur5e_delto/OmniReset"
RUN_BASE="${RUN_BASE:-./Datasets_ur5e_delto/OmniReset_DEXRESET_S1S2_$STAMP}"
# -- The generator TRUNCATES AND REWRITES its output in place several times a minute. A run aimed
# at the production root destroys finished banks rather than wasting GPU time.
if [ "$(realpath -m "$RUN_BASE")" = "$(realpath -m "$PRODUCTION_DATASET_DIR")" ]; then
  echo "REFUSING: RUN_BASE ($RUN_BASE) resolves to the production root ($PRODUCTION_DATASET_DIR)." >&2
  exit 1
fi
mkdir -p "$RUN_BASE"
LOG_DIR="$REPO_ROOT/logs/dexreset_s1_s2_bankgen_$STAMP"
mkdir -p "$LOG_DIR"

# -- SpawnPartialAssembly downloads partial_assemblies.pt for this pair the moment it is
# constructed, from a default Hugging Face path that 404s for this pair. The failure is a swallowed
# lazy-term exception, not a crash. The known-bad local file under that name is the 525-pose
# random-walk one, of which 0 of 525 poses are actually seated in the bore -- it loads fine and
# produces a bank of states that were never partially assembled.
PARTIAL_ASSEMBLY_DATASET_DIR="${PARTIAL_ASSEMBLY_DATASET_DIR:?Set PARTIAL_ASSEMBLY_DATASET_DIR to the root holding Resets/<pair>/partial_assemblies.pt (the verified 2048-pose file). There is deliberately no default: the built-in one 404s for this pair.}"
# -- THE ASSET PAIR IS DERIVED FROM THE DATASET, NOT ASSUMED. This was hard-coded to
# OneLegInsertionFixture__SquareTableLeg200mmDecomp and refused on a perfectly good dataset root,
# because the reset banks for this work live under ...__SquareTableLeg200mmSdf: the shipped leg
# collider is the whole-part SDF, and the decomposed one is the collider that EJECTS the leg (a
# previous training run was contaminated by exactly that mix-up). The grasps directory in the same
# root really is named ...Decomp, so the two names legitimately coexist under one dataset and a
# hard-coded guess picks wrong roughly half the time.
#
# So: glob for the pair directory that actually holds partial_assemblies.pt, and REFUSE on anything
# other than exactly one hit. Zero means the root is wrong; two or more means the operator has to
# say which pair is intended (PAIR=...), because silently taking the first would choose the leg's
# collider -- and therefore the physics -- by directory sort order.
if [ -n "${PAIR:-}" ]; then
  _PA_PAIR_FILE="$PARTIAL_ASSEMBLY_DATASET_DIR/Resets/$PAIR/partial_assemblies.pt"
  [ -f "$_PA_PAIR_FILE" ] || { echo "REFUSING: PAIR=$PAIR given but $_PA_PAIR_FILE does not exist"; exit 1; }
else
  _PA_HITS=()
  while IFS= read -r _h; do _PA_HITS+=("$_h"); done < <(
    find "$PARTIAL_ASSEMBLY_DATASET_DIR/Resets" -mindepth 2 -maxdepth 2 -name partial_assemblies.pt 2>/dev/null | sort
  )
  case "${#_PA_HITS[@]}" in
    0) echo "REFUSING: no Resets/<pair>/partial_assemblies.pt under $PARTIAL_ASSEMBLY_DATASET_DIR"; exit 1 ;;
    1) _PA_PAIR_FILE="${_PA_HITS[0]}" ;;
    *) echo "REFUSING: $PARTIAL_ASSEMBLY_DATASET_DIR holds ${#_PA_HITS[@]} partial-assembly banks:"
       printf '  %s\n' "${_PA_HITS[@]}"
       echo "  Set PAIR=<pair directory name> -- this choice selects the leg's COLLIDER and so the"
       echo "  physics of every generated state; it must not be made by sort order."
       exit 1 ;;
  esac
fi
PAIR="$(basename "$(dirname "$_PA_PAIR_FILE")")"
echo "[partial-assembly] pair: $PAIR"
_PA_N=$("$PYTHON_BIN" -c "
import torch
d = torch.load('$_PA_PAIR_FILE', map_location='cpu', weights_only=True)
print(d['relative_position'].shape[0])
" 2>/dev/null) || { echo "REFUSING: could not load $_PA_PAIR_FILE to count its poses."; exit 1; }
echo "[partial-assembly] pose count: $_PA_N ($_PA_PAIR_FILE)"
if [ "$_PA_N" = "525" ]; then
  echo "REFUSING: this is the KNOWN-BROKEN 525-pose file (0 of 525 poses seated in the bore)." >&2
  exit 1
fi
[ "$_PA_N" = "2048" ] || {
  echo "REFUSING: expected the verified 2048-pose file, got $_PA_N poses -- confirm this file's own" >&2
  echo "  geometry by reprojection against the fixture's metadata.yaml before overriding." >&2
  exit 1
}

echo "================================================================================"
echo "checkpoint          : $CKPT"
echo "agent yaml          : $AGENT_YAML"
echo "task                : $TASK"
echo "rungs               : $RUNGS"
echo "gpu                 : $GPU   num_envs=$NUM_ENVS"
echo "target/rung         : $TARGET_PER_RUNG   chunk=$CHUNK_TARGET_SIZE"
echo "run base            : $RUN_BASE"
echo "logs                : $LOG_DIR"
echo "shaping (mm deeper) : S1=$S1_GOAL_BELOW_SPAWN_MM  S2P=$S2P_GOAL_BELOW_SPAWN_MM"
echo "================================================================================"

# =====================================================================================
# ONE RUNG'S CHUNKED GENERATION.
#   $1 rung id        $2 output bank name        $3 depth_min_mm   $4 depth_max_mm
#   $5 lateral_max_mm $6 tilt_max_deg            $7 goal_below_spawn_mm
# =====================================================================================
run_rung() {
  local rung="$1" bank_name="$2" dmin="$3" dmax="$4" lat="$5" tilt="$6" shaping="$7"
  local n_chunks=$(( (TARGET_PER_RUNG + CHUNK_TARGET_SIZE - 1) / CHUNK_TARGET_SIZE ))
  local per_chunk=$(( (TARGET_PER_RUNG + n_chunks - 1) / n_chunks ))
  local rung_dir="$RUN_BASE/$rung"
  mkdir -p "$rung_dir"

  # -- The gate band as _SeatingGateAddon will PRINT it, so the health check below matches the
  # NUMBER the constructed gate holds, not a label that merely mentions the flag. Formatting
  # mirrors that print's own "%.2f", and every "." is escaped: an unescaped one is a regex wildcard,
  # which would let a DIFFERENT band satisfy this check -- the precise shape of "matched the label,
  # not the value" that this project keeps paying for.
  local gate_marker
  gate_marker="$(printf '\\[c4-seating-gate\\] ENABLED depth=\\[%.2f,%.2f\\]mm lateral<=%.2fmm tilt<=%.2fdeg' \
    "$dmin" "$dmax" "$lat" "$tilt" | sed 's/\./\\./g')"

  echo
  echo "### $rung -> resets_${bank_name}.pt   depth=[${dmin},${dmax}]mm lateral<=${lat}mm tilt<=${tilt}deg"
  echo "### chunks=$n_chunks per_chunk_target=$per_chunk shaping=${shaping}mm"

  local i chunk_dir chunk_log chunk_seed chunk_yaml chunk_exit accepted healthy
  local -a accepted_counts=()

  for i in $(seq 1 "$n_chunks"); do
    chunk_dir="$rung_dir/chunk_$i"
    chunk_log="$LOG_DIR/${rung}_chunk${i}.log"
    chunk_seed=$(( BASE_SEED + i - 1 ))
    chunk_yaml="$rung_dir/agent_seed_${chunk_seed}.yaml"

    # -- torch_runner.load_config seeds torch and numpy GLOBALLY from the yaml, before the env is
    # built. Identical seeds across chunks with an identical command line means identical env
    # sampling no matter how many distinct output DIRECTORIES the chunks write to.
    "$PYTHON_BIN" scripts_v2/tools/make_seeded_agent_yaml.py \
      --source_yaml "$AGENT_YAML" --seed "$chunk_seed" --output_yaml "$chunk_yaml" \
      || { echo "[$rung chunk $i/$n_chunks] FATAL: could not write a seeded agent.yaml -- aborting this" \
                "rung before spending GPU time on a seed we cannot confirm." >&2; return 1; }

    echo "[$rung chunk $i/$n_chunks] seed=$chunk_seed dir=$chunk_dir log=$chunk_log"
    (
      export DEXLIFT_PARTIAL_ASSEMBLY=1
      export DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR="$PARTIAL_ASSEMBLY_DATASET_DIR"
      export DEXLIFT_GOAL_BELOW_SPAWN_MM="$shaping"
      timeout -s KILL "$CHUNK_TIMEOUT_S" "$PYTHON_BIN" -u scripts_v2/tools/generate_reset_states_policy.py \
        --task "$TASK" \
        --reset_type ObjectPartiallyAssembledEEGrasped \
        --receptive_usd_path "$RECEPTIVE_USD_PATH" \
        --checkpoint "$CKPT" \
        --agent_yaml "$chunk_yaml" \
        --dataset_dir "$chunk_dir" \
        --episode_length_s 8.0 \
        --num_envs "$NUM_ENVS" \
        --num_reset_states "$per_chunk" \
        --c4_seating_gate \
        --c4_depth_min_mm "$dmin" \
        --c4_depth_max_mm "$dmax" \
        --c4_lateral_max_mm "$lat" \
        --c4_tilt_max_deg "$tilt" \
        --headless
    ) > "$chunk_log" 2>&1
    chunk_exit=$?
    echo "EXIT_CODE=$chunk_exit" >> "$chunk_log"

    # -- EXIT_CODE IS NOT EVIDENCE. An Isaac process prints EXIT_CODE=0 over a fatal traceback often
    # enough that a positive marker is the only thing worth believing. The generator's own post-run
    # assertion re-LOADS the bank it just wrote and compares its length to the in-memory accepted
    # counter, so this one line proves the states reached disk, not merely that a loop finished.
    healthy=1
    grep -aq "POST-RUN BANK CHECK PASSED (accept-time bank)" "$chunk_log" || healthy=0
    grep -aq "Traceback" "$chunk_log" && healthy=0
    [ "$chunk_exit" -eq 0 ] || healthy=0

    if [ "$healthy" -ne 1 ]; then
      echo "[$rung chunk $i/$n_chunks] UNHEALTHY (exit=$chunk_exit, positive marker and/or Traceback)" \
           "-- continuing to the next chunk; the merge tolerates a chunk with no usable output." >&2
      grep -anE "Traceback|Error executing job|OutOfMemoryError|out of memory|PhysX ABORT error|AssertionError|ValueError|RuntimeError" \
        "$chunk_log" | tail -8 >&2
      continue
    fi

    # -- CONFIG CHECKS ARE HARD, AND ONLY ON A HEALTHY CHUNK. A crashed chunk's log is legitimately
    # truncated and would false-positive, teaching a reader to ignore these. On a chunk that ran to
    # completion, a missing marker is not transient: the plant, the reset distribution, the pose
    # tilt, the partial-assembly toggle and the gate band are DETERMINISTIC AND GLOBAL -- they are
    # wrong in every chunk of both rungs or in none. Under warning-only semantics such a run does
    # not fail, it COMPLETES: 20,000 states in the wrong regime under a success banner. Aborting
    # costs one Isaac boot.
    local marker
    for marker in \
      'hand effort_limit_sim \(distinct values\): \[30\.0\]' \
      'events\.reset_robot_joints\.position_range = \[-0\.5, 0\.5\]' \
      '\+-0\.3000 rad' \
      'DEXLIFT_PARTIAL_ASSEMBLY=1.*reset_object -> SpawnPartialAssembly' \
      "$gate_marker" \
      ; do
      grep -aqE -- "$marker" "$chunk_log" || {
        echo "[$rung chunk $i/$n_chunks] FATAL: the CONSTRUCTED configuration does not show" \
             "/$marker/ in this chunk's log. Global, not transient -- every chunk of both rungs is" \
             "affected. See $chunk_log. Aborting rather than banking a wrong-regime rung." >&2
        return 1
      }
    done

    accepted=$(sed -n 's/.*POST-RUN BANK CHECK PASSED (accept-time bank).*contains \([0-9]*\) states.*/\1/p' \
      "$chunk_log" | tail -1)
    accepted_counts+=("chunk${i}=${accepted:-0}")
    echo "[$rung chunk $i/$n_chunks] OK  accepted=${accepted:-0}/$per_chunk  (gate band confirmed from the log)"
  done

  echo "[$rung] per-chunk accepted: ${accepted_counts[*]:-none}"

  # -- MERGE. Same tool the four-bank regeneration uses, --filename mode: the chunks carry the
  # generator's own C4 filename, and the rung rename happens after this, on the merged file.
  local merged_dir="$rung_dir/merged" merge_log="$LOG_DIR/${rung}_merge.log"
  "$PYTHON_BIN" scripts_v2/tools/merge_c2_chunks.py \
    --chunk_dirs "$rung_dir"/chunk_* \
    --output_dir "$merged_dir" \
    --filename "resets_ObjectPartiallyAssembledEEGrasped.pt" \
    > "$merge_log" 2>&1
  local merge_exit=$?
  cat "$merge_log"
  [ "$merge_exit" -eq 0 ] || { echo "[$rung] MERGE FAILED (exit $merge_exit), see $merge_log" >&2; return 1; }

  local merged_file
  merged_file=$(find "$merged_dir" -name "resets_ObjectPartiallyAssembledEEGrasped.pt" | head -1)
  [ -n "$merged_file" ] || { echo "[$rung] REFUSING: no merged file under $merged_dir" >&2; return 1; }

  # =====================================================================================
  # REKEY. MANDATORY WHENEVER THE BANK IS ONE-KEY -- a {insertive_object}-only file trains with the
  # fixture displaced 0.35-0.60 m laterally, silently, with no error anywhere.
  #
  # THE DECISION IS MADE FROM THE FILE'S OWN KEYS, NOT FROM THE PIPELINE THAT PRODUCED IT. Under
  # DEXLIFT_PARTIAL_ASSEMBLY=1 the scene HAS a receptive_object and
  # _DexliftToTrainingSceneRecorder forwards it live, so these two rungs' banks reach disk already
  # carrying it -- and rekey_dexlift_reset_states.py:173-182 REFUSES such a file outright as
  # "already schema-complete", to avoid double-writing synthesised poses over real ones. Running
  # rekey unconditionally here would therefore ERROR on every healthy run of this script. The
  # branch below runs it exactly when it applies and PROVES the alternative rather than assuming
  # it: a bank that skips rekey must be shown to carry a native receptive_object.
  # =====================================================================================
  local rekey_out="$rung_dir/final_resets_${bank_name}.pt"
  local schema
  schema=$("$PYTHON_BIN" -c "
import torch
d = torch.load('$merged_file', map_location='cpu', weights_only=False)
print(','.join(sorted(d['initial_state']['rigid_object'].keys())))
") || { echo "[$rung] REFUSING: could not read merged bank keys from $merged_file" >&2; return 1; }
  echo "[$rung] merged rigid_object keys: $schema"

  case "$schema" in
    insertive_object)
      echo "[$rung] one-key bank -> rekey REQUIRED (synthesises receptive_object/table/ur5_metal_support)."
      "$PYTHON_BIN" scripts_v2/tools/rekey_dexlift_reset_states.py \
        --input "$merged_file" --output "$rekey_out" --seed 0 \
        > "$LOG_DIR/${rung}_rekey.log" 2>&1
      local rekey_exit=$?
      cat "$LOG_DIR/${rung}_rekey.log"
      [ "$rekey_exit" -eq 0 ] || { echo "[$rung] REKEY FAILED (exit $rekey_exit)" >&2; return 1; }
      ;;
    insertive_object,receptive_object)
      echo "[$rung] bank already carries a NATIVE receptive_object (the partial-assembly scene had a"
      echo "        real fixture) -- rekey_dexlift_reset_states.py:173-182 refuses this schema by"
      echo "        design, so it is skipped and the merged file is used as-is."
      cp "$merged_file" "$rekey_out" || return 1
      ;;
    *)
      echo "[$rung] REFUSING: unrecognised rigid_object schema '$schema'. rekey handles" >&2
      echo "  {object,table} and {insertive_object}; the partial-assembly recorder produces" >&2
      echo "  {insertive_object,receptive_object}. Anything else needs a human." >&2
      return 1
      ;;
  esac

  # -- FINAL COUNT + the one distinguishing fact for these rungs: the fixture must be NATIVE, not
  # synthesised, or the bank's contents do not match its name however the file is called.
  "$PYTHON_BIN" - <<PYEOF
import torch
d = torch.load("$rekey_out", map_location="cpu", weights_only=False)
rigid = d["initial_state"]["rigid_object"]
robot = d["initial_state"]["articulation"]["robot"]
n = len(robot["root_pose"])
print(f"[$rung] FINAL BANK $rekey_out")
print(f"[$rung]   states: {n}")
print(f"[$rung]   rigid_object keys: {sorted(rigid.keys())}")
for name, entry in rigid.items():
    ln = len(entry["root_pose"])
    print(f"[$rung]   {name}: root_pose len={ln}  {'OK' if ln == n else 'LENGTH MISMATCH'}")
for field in ("joint_position_target", "joint_velocity_target"):
    print(f"[$rung]   robot.{field} present: {field in robot}")
if not ("joint_position_target" in robot and "joint_velocity_target" in robot):
    print("[$rung]   *** WARNING: a missing PD target field means target := q at replay, i.e. ZERO"
          " commanded squeeze. Investigate before training on this bank. ***")
PYEOF

  echo "[$rung] DONE -> $rekey_out"
}

FAILED=0
if [ "$RUNGS" = "S1" ] || [ "$RUNGS" = "both" ]; then
  run_rung s1 ObjectAtBoreMouthEEGrasped 0.0 10.0 5.0 15.0 "$S1_GOAL_BELOW_SPAWN_MM" \
    || { echo "[s1] RUNG FAILED -- see $LOG_DIR/s1_*.log" >&2; FAILED=1; }
fi
if [ "$RUNGS" = "S2P" ] || [ "$RUNGS" = "both" ]; then
  # Negative bounds are passed literally: argparse's negative-number matcher accepts "-120.0" as a
  # value (this parser declares no option strings that look like negative numbers), and
  # _SeatingGateAddon asserts only depth_min < depth_max, so the band stays legal above the mouth.
  run_rung s2p ObjectAboveBoreEEGrasped -120.0 -20.0 20.0 25.0 "$S2P_GOAL_BELOW_SPAWN_MM" \
    || { echo "[s2p] RUNG FAILED -- see $LOG_DIR/s2p_*.log" >&2; FAILED=1; }
fi

echo
echo "############################################################################"
echo "# NOTHING has been copied into $PRODUCTION_DATASET_DIR. Read the per-chunk accepted counts"
echo "# and the FINAL BANK block(s) above by eye first -- there is no prior yield measurement for"
echo "# either of these depth bands, so no threshold can gate them automatically. Then:"
echo "#"
echo "#   cp \"$RUN_BASE/s1/final_resets_ObjectAtBoreMouthEEGrasped.pt\" \\"
echo "#      \"$PRODUCTION_DATASET_DIR/Resets/$PAIR/resets_ObjectAtBoreMouthEEGrasped.pt\""
echo "#   cp \"$RUN_BASE/s2p/final_resets_ObjectAboveBoreEEGrasped.pt\" \\"
echo "#      \"$PRODUCTION_DATASET_DIR/Resets/$PAIR/resets_ObjectAboveBoreEEGrasped.pt\""
echo "#"
echo "# A consumer must NAME these two reset types in its own reset_types entries; MultiResetManager"
echo "# matches banks by name and a bank nothing names is simply never drawn."
echo "############################################################################"

exit "$FAILED"
