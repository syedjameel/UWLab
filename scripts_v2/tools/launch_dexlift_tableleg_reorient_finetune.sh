#!/usr/bin/env bash
# Epic UWLab-g3z4 finetune launch: dexlift table-leg REORIENT, warm-started from the certified
# Stage-3 checkpoint, under the three new features (full gravity, low-goal band, partial-assembly
# grasp-only mixture) landed in commit 4818256. STAGED, NOT RUN -- do not execute this until a
# training box has passed its gate. One command, assembled here instead of under time pressure.
#
# Usage:  launch_dexlift_tableleg_reorient_finetune.sh <gpu-index>
#   e.g.  launch_dexlift_tableleg_reorient_finetune.sh 0
#
# =====================================================================================
# WARM START, NOT FROM SCRATCH, AND WHY
# =====================================================================================
# Pinning gravity to full (dexlift_ur5e_delto_tableleg_env_cfg._apply_full_gravity) removes the
# shaping the ADR gravity ramp gave the reference recipe -- dexsuite_env_cfg.py says outright that
# the ramp "removes the need for a special Lift reward". With gravity pinned, difficulty 0 is now
# STRICTLY HARDER than the task the reference solved from scratch. Warm-starting from a checkpoint
# that already solved the task at full gravity and full ADR difficulty (see below) sidesteps that
# instead of re-fighting it.
#
# CHECKPOINT: ep_3600, rew 38.38917, from the RL-Games run at
#   logs/rl_games/dexlift_ur5e_delto_reljointpos_tableleg_reorient/2026-08-16_09-33-54/
# THIS REPO'S local_ckpts/ ONLY VENDORS A LOCAL COPY FOR DEVELOPMENT/CERTIFICATION -- it is NOT
# assumed to exist on the training box. CKPT (below, near the launch command) is a top-of-script
# variable defaulting to that BOX's own known layout (checkpoint alone in /root/ckpt/, no
# local_ckpts/ tree, no params/ subdir beside it), overridable with CKPT=<path> if the box differs.
# Its own certification record (local_ckpts/certification_2026-08-17/cert_t30_pose_ep3600.json,
# read from THIS repo's copy, not the box's) shows it was evaluated at adr_difficulty pinned_at=10
# (max) with gravity already at full (-9.81) -- i.e. this is a policy that has already solved the
# full-difficulty, full-gravity task once; finetuning it under the new mixture is a smaller ask than
# training that combination from difficulty 0.
#
# sha256 of the checkpoint MUST read 9534e102a6...: verified below, at launch time, off the actual
# bytes wherever CKPT points on THIS box, not assumed from a filename or a path. A silently-wrong,
# truncated, or MISPLACED (wrong directory layout) transfer of this checkpoint is exactly the
# "plausible wrong number" class of bug this repo keeps finding.
#
# =====================================================================================
# --max_iterations IS ABSOLUTE, NOT ADDITIONAL
# =====================================================================================
# RL-Games' Runner, given load_checkpoint=True, restores epoch_num from the checkpoint and then
# trains until agent_cfg["params"]["config"]["max_epochs"] (== --max_iterations) is reached --
# NOT for that many MORE epochs. Warm-starting at ep3600 and asking for --max_iterations 5600 below
# is deliberately +2000 epochs of additional finetuning, not 5600 epochs of new training. Chosen as
# a moderate, bounded finetune budget: enough for the new mixture (and the DR terms other than
# gravity, which still ramp -- see below) to show measurable adaptation without committing to an
# open-ended run on a paid box. If a different budget is wanted, change MAX_ITERATIONS below AND
# this comment -- the number must always be read as "warm-start epoch + intended additional epochs".
MAX_ITERATIONS=5600   # = 3600 (warm start) + 2000 (intended additional finetune epochs)
#
# =====================================================================================
# init_difficulty PINNED TO MAX -- RL-GAMES DOES NOT RESTORE THE ADR CURRICULUM
# =====================================================================================
# Unlike the epoch counter, RL-Games' checkpoint does NOT restore env-side curriculum state
# (curriculum.adr's own difficulty counter lives in the ENV, not the agent). A warm start would
# otherwise silently re-climb the ADR "adr" term from difficulty 0 (adr_curriculum.py:
# DifficultyScheduler(init_difficulty=0, min_difficulty=0, max_difficulty=10)), which -- even though
# gravity itself is now permanently pinned full and OUT of that system (curriculum.gravity_adr is
# nulled by _apply_full_gravity) -- still governs every OTHER DR term the ADR schedule ramps
# (observation noise, action-space noise, etc: see CurriculumCfg in adr_curriculum.py). Re-climbing
# those from 0 on a policy the checkpoint already solved at difficulty 10 would look like the
# finetune regressing for the first tens-to-hundreds of epochs, when it is really just the curriculum
# re-narrowing. Pinned via a Hydra override on the env cfg (max_difficulty is 10 -- adr_curriculum.py):
#   env.curriculum.adr.params.init_difficulty=10
#
# =====================================================================================
# THE FOUR PLANT VARS, EXPORTED TOGETHER, PLUS THE TASK-DEFINITION TILT
# =====================================================================================
# DEXLIFT_REF_RESET=1 DEXLIFT_REF_ACTUATORS=1 DEXLIFT_REF_HAND_ACT=1 DEXLIFT_REF_ARM_ACT=0 is the
# EXACT plant the certified checkpoint was generated under (dexlift_ur5e_delto_env_cfg.ref_actuators):
# reference hand actuators (30.0 N.m effort / 10000.0 rad/s velocity) + our identified arm plant +
# dexsuite's own start-pose distribution. Missing any one of these is a SILENTLY plausible wrong
# number, not an obvious failure: in particular, if DEXLIFT_REF_HAND_ACT is unset, the hand falls
# back to the "identified" actuator block, capped at 3.0 rad/s against a ~6 rad/s commanded closure
# -- the fingers physically cannot close, and nothing in the log says so unless the constructed
# plant is checked (see the PLANT-VERIFY health check below). DEXLIFT_POSE_TILT=0.3 is a TASK
# DEFINITION change (narrows the object's reset AND goal orientation together), not a plant knob --
# it is exported alongside the plant vars but is a different kind of setting; see
# dexlift_ur5e_delto_env_cfg.py's ``_apply_pose_tilt_stage``.
#
# =====================================================================================
# THE EPISODE MIXTURE IS OPT-IN, AND ITS PARTIAL-ASSEMBLY DATASET IS NOT AT THE HF DEFAULT
# =====================================================================================
# DEXLIFT_EPISODE_MIXTURE=1 is exported below because this launch IS the run the mixture (gravity/
# low-goal/partial-assembly, epic UWLab-g3z4) was built for -- the mixture defaults OFF for every
# OTHER construction of these classes (mdp/episode_mixture.py's "THE MIXTURE IS OPT-IN" section:
# an ordinary tool run that forgot to opt in previously got the mixture installed anyway and crashed
# on a missing dataset -- this launch must not repeat that mistake in the other direction by
# forgetting to opt IN and silently training the old, no-mixture task under the new name).
#
# CONFIRMED TODAY: the class default partial_assembly_prob=0.25 makes MixtureResetObject construct a
# composer that downloads partial_assemblies.pt for this exact pair, and that file 404s at the
# default (Hugging Face) DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR -- verified against the HF repo's own
# directory listing, which has no entry for this pair at all. DATASET_DIR below points at a locally-
# vendored copy instead; it MUST be transferred to the box (same class of requirement as CKPT).
#
# =====================================================================================
# OPTIMIZER UPDATE RATE MATCHES THE CERTIFIED LINEAGE
# =====================================================================================
# updates_per_epoch = mini_epochs * num_envs * horizon_length / minibatch_size. The reference recipe
# (and this task's own rl_games_ppo_ur5e_delto_reljointpos_tableleg_reorient_cfg.yaml) fixes
# mini_epochs=5, horizon_length=36. The reference does 20 updates/epoch. Solving for minibatch at
# NUM_ENVS=4096:  5 * 4096 * 36 / minibatch = 20  =>  minibatch = 5*4096*36/20 = 36864.
# Verified by the runtime assert below (do not trust this comment alone).
NUM_ENVS=4096
MINIBATCH=36864
#
# =====================================================================================
# WANDB IS MANDATORY (user requirement), TRACK/ENTITY/PROJECT EXPLICIT
# =====================================================================================
# --track defaults to True in scripts/reinforcement_learning/rl_games/train.py already, but wired
# explicitly below so a future edit of that default cannot silently untrack this run. train.py itself
# raises ValueError before booting the sim if --track is true and --wandb-entity is empty -- that is
# the safety net if this script's own value is ever blanked. PROJECT is uwlab-dexinsertion, the
# script's own default and the umbrella prior dexlift work has been logged under, so this run is
# findable NEXT TO that work rather than starting a new, harder-to-find project. NAME encodes task +
# the three new features + warm-start provenance so it is legible in a run list without opening it.
WANDB_ENTITY="i_domrachev-interactive-robotic-systems-lab-kaist"
WANDB_PROJECT="uwlab-dexinsertion"
WANDB_NAME="dexlift-tableleg-reorient-fullgravity-lowgoal-partialassembly-warm3600"
#
# =====================================================================================
# THE FIRST HEALTH CHECK -- if any of these four is missing from the log in the first ~15 minutes,
# KILL AND DIAGNOSE. Do not let it burn hours on a paid box past this point undiagnosed:
#   1. "[dexlift] gravity PINNED at ((0.0, 0.0, -9.81)"                 -- gravity really is full+fixed
#   2. "[dexlift] episode mixture (validated post-override): classic_goal_prob=0.500" -- the mixture
#      wired and its ACTUAL (post-Hydra-override) fractions read back correctly. 0.500 here is the
#      class default -- this launch does not override the three probabilities, so this MUST read the
#      defaults (0.500/0.250/0.250); a different number here means an override leaked in unexpectedly.
#   3. "[PLANT-VERIFY] hand actuator (constructed): effort_limit_sim=[30.0] velocity_limit_sim=
#      [10000.0]" -- the ACTUAL NUMBERS on the CONSTRUCTED hand, not merely train.py's own "OK" line
#      that follows it. A check that matches the label rather than the number is exactly the failure
#      mode that has cost this project the most time: an "OK" line only proves SOME comparison
#      passed, not that the compared values were the ones intended. See PLANT-VERIFY note above for
#      why the number matters (identified-plant hand caps at 3.0 rad/s against ~6 rad/s commanded).
#   4. an "epoch: N/M" line (rl_games' own per-iteration marker) -- confirms training actually reached
#      real optimizer updates, not just environment construction.
# The health-check loop below polls the LOG FILE for exactly these four markers.
# =====================================================================================

set -uo pipefail

GPU=${1:?"usage: $0 <gpu-index>"}

# -- Reject a shared GPU. Never launch on a GPU another user already has memory resident on.
USED=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
[ "${USED:-999999}" -lt 2000 ] || { echo "REFUSING: GPU $GPU has ${USED} MiB already in use"; exit 1; }

# -- Sanity-check the optimizer update rate BEFORE launching anything, not just in a comment.
EXPECT=$(( 5 * NUM_ENVS * 36 / MINIBATCH ))
[ "$EXPECT" -eq 20 ] || {
  echo "REFUSING: NUM_ENVS=$NUM_ENVS MINIBATCH=$MINIBATCH gives $EXPECT updates/epoch, not the certified 20"
  exit 1
}
[ $(( NUM_ENVS * 36 % MINIBATCH )) -eq 0 ] || {
  echo "REFUSING: minibatch $MINIBATCH does not evenly divide the $NUM_ENVS*36 batch"
  exit 1
}

export CUDA_VISIBLE_DEVICES="$GPU"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHONUNBUFFERED=1

# -- The four plant vars, together, plus the task-definition tilt. See the header block above.
export DEXLIFT_REF_RESET=1
export DEXLIFT_REF_ACTUATORS=1
export DEXLIFT_REF_HAND_ACT=1
export DEXLIFT_REF_ARM_ACT=0
export DEXLIFT_POSE_TILT=0.3

# -- THE EPISODE MIXTURE IS OPT-IN (mdp/episode_mixture.py's own docstring, "THE MIXTURE IS OPT-IN"
# section): DexLiftUR5eDeltoRelJointPosTableLegReorientEnvCfg builds byte-identical to the pre-mixture
# task unless this is exported. This IS the run the mixture was built for, so it must be set --
# forgetting it would silently train the OLD (no-mixture) task under the new name.
export DEXLIFT_EPISODE_MIXTURE=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/source/uwlab:$REPO_ROOT/source/uwlab_tasks:$REPO_ROOT/source/uwlab_assets:$REPO_ROOT/source/uwlab_rl"

TASK="DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0"

# -- partial_assembly_prob defaults to 0.25 > 0 (class default), so MixtureResetObject WILL construct
# the partial-assembly composer, which downloads partial_assemblies.pt for this exact pair the moment
# it is constructed. CONFIRMED TODAY: that file does not exist at the default (Hugging Face)
# DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR for this pair -- HTTP 404, verified against the HF repo's own
# directory listing, which has no OneLegInsertionFixture__SquareTableLeg200mmDecomp entry at all. Left
# at the default, this launch would reach the exact swallow-then-TypeError crash
# (`MixtureResetObject.__init__() got an unexpected keyword argument 'dataset_dir'`) diagnosed in the
# reset-generation regression this same epic caused -- just later, inside RslRlVecEnvWrapper/
# RlGamesVecEnvWrapper's own reset(), instead of a tool's. DATASET_DIR below overrides it to a
# LOCALLY-VENDORED copy that must be transferred to the box (same category of requirement as CKPT
# above) -- not this repo's default local_ckpts/-relative dev convention, since that is not assumed to
# exist on the box either. If this specific pair's dataset is ever published to the HF default, this
# override becomes unnecessary but remains harmless (still points at a valid copy).
DATASET_DIR=${DATASET_DIR:-/root/ckpt/Datasets_ur5e_delto/OmniReset}
DATASET_PAIR_FILE="$DATASET_DIR/Resets/OneLegInsertionFixture__SquareTableLeg200mmDecomp/partial_assemblies.pt"
[ -f "$DATASET_PAIR_FILE" ] || {
  echo "REFUSING: partial-assembly dataset not found at $DATASET_PAIR_FILE"
  echo "  (DEXLIFT_EPISODE_MIXTURE=1 with the default partial_assembly_prob=0.25 needs this file;"
  echo "   the default Hugging Face path 404s for this pair -- see the comment above DATASET_DIR)."
  exit 1
}
# -- A FILE EXISTING AT THIS PATH IS NOT ENOUGH -- CONFIRMED BLOCKER, 2026-08-20. The 525-pose file
# that shipped locally on this machine (local_ckpts/reset_states_2026-08-17/partial_assemblies.pt,
# byte-identical to a second copy under Datasets_render/) was produced by the OLD force-driven
# random-walk generator (superseded by bead UWLab-algw.9's axial_depth_sampling -- see
# partial_assemblies_cfg.py's header) and is PHYSICALLY INVALID for this pair: reprojected against
# OneLegInsertionFixture/metadata.yaml's own assembled_offset (the seat point), ZERO of its 525
# poses have z within the [seat, mouth] engaged span at all -- every one is a leg hovering near the
# hole, not seated in it. The correct replacement is a 2048-pose file from the rebuilt axial-depth
# sampler (lateral miss median ~0.035mm, tilt median ~0.143deg, depth spread evenly across the
# operational band -- verified in a live Isaac run). COUNT THE POSES BEFORE TRUSTING THIS FILE, not
# just its path or size: a file that merely exists here is exactly what let the broken one through
# unnoticed before.
# PATH RESOLUTION: the original hardcoded a dev-workstation layout
# ($HOME/UWLab/env_uwlab/bin/python) that does not exist on a training box.
LAUNCH_PYTHON_BIN="${LAUNCH_PYTHON_BIN:-/root/venv/bin/python}"
[ -x "$LAUNCH_PYTHON_BIN" ] || { echo "REFUSING: LAUNCH_PYTHON_BIN '$LAUNCH_PYTHON_BIN' not executable (set LAUNCH_PYTHON_BIN to override)"; exit 1; }

_DATASET_POSE_COUNT=$("$LAUNCH_PYTHON_BIN" -c "
import torch
d = torch.load('$DATASET_PAIR_FILE', map_location='cpu', weights_only=True)
print(d['relative_position'].shape[0])
" 2>/dev/null) || {
  echo "REFUSING: could not load $DATASET_PAIR_FILE to count its poses -- do not trust an unreadable dataset file"
  exit 1
}
echo "partial-assembly dataset pose count: $_DATASET_POSE_COUNT ($DATASET_PAIR_FILE)"
if [ "$_DATASET_POSE_COUNT" = "525" ]; then
  echo "REFUSING: this is the KNOWN-BROKEN 525-pose file (old random-walk generator, zero poses"
  echo "  actually seated in the bore -- see the comment above). Point DATASET_DIR at the rebuilt"
  echo "  2048-pose axial-depth-sampler file instead; do not override this check."
  exit 1
fi
if [ "$_DATASET_POSE_COUNT" != "2048" ]; then
  echo "REFUSING: expected the 2048-pose axial-depth-sampler file, got $_DATASET_POSE_COUNT poses."
  echo "  An unrecognised count is not automatically trusted just because it isn't 525 -- verify this"
  echo "  file's own geometry (lateral miss / tip height / tilt against the seat point) before"
  echo "  overriding this check by hand."
  exit 1
fi
echo "partial-assembly dataset verified: 2048 poses, not the known-broken 525-pose file."
# Read by _apply_episode_mixture (dexlift_ur5e_delto_tableleg_env_cfg.py) at cfg-construction time --
# a plain env var, not only a Hydra override, so the identical mechanism also works for
# scripts_v2/tools/certification/certify_pose.py (cert_g3z4_finetune.sh's own delegate), which has no
# Hydra override path of its own.
export DEXLIFT_EPISODE_MIXTURE_DATASET_DIR="$DATASET_DIR"

# -- CHECKPOINT PATH IS A TOP-OF-SCRIPT VARIABLE, NOT DERIVED. Default is the training box's own
# layout (checkpoint alone in /root/ckpt/, no local_ckpts/ tree, no params/ subdir beside it) --
# NOT the repo-relative local_ckpts/... path this repo happens to vendor a copy at locally. Do not
# assume local_ckpts/ exists on the box: this is exactly the layout mismatch that already broke the
# reset generator today with FileNotFoundError on /root/params/agent.yaml (something derived an
# agent-yaml path from the checkpoint's parent-of-parent, which does not exist in this layout).
# Nothing here does that: the agent config for this launch comes entirely from the registered
# rl_games_cfg_entry_point (the yaml under source/uwlab_tasks/.../dexlift/agents/), never from a
# file beside the checkpoint -- if a future edit of this script ever needs a standalone agent yaml,
# it must be its own explicit CKPT_AGENT_YAML-style variable, never derived from $CKPT's directory.
# Override CKPT=/some/other/path ./launch_....sh <gpu> if the box's layout differs from the default.
CKPT=${CKPT:-/root/ckpt/last_dexlift_ur5e_delto_reljointpos_tableleg_reorient_ep_3600_rew_38.38917.pth}
CKPT_SHA256_EXPECT="9534e102a64fc06a3588c5102fc3421e69ef1b18fe30e7ffb40f7df63b5d76af"

# -- Verify the checkpoint bytes on THIS box, not just that a file with the right name exists.
[ -f "$CKPT" ] || { echo "REFUSING: warm-start checkpoint not found at $CKPT (did the transfer to this box include local_ckpts/?)"; exit 1; }
CKPT_SHA256_GOT="$(sha256sum "$CKPT" | awk '{print $1}')"
[ "$CKPT_SHA256_GOT" = "$CKPT_SHA256_EXPECT" ] || {
  echo "REFUSING: checkpoint sha256 mismatch."
  echo "  expected: $CKPT_SHA256_EXPECT"
  echo "  got:      $CKPT_SHA256_GOT"
  exit 1
}
echo "checkpoint sha256 verified: $CKPT_SHA256_GOT"

LOG="$REPO_ROOT/logs/launch_dexlift_tableleg_reorient_finetune_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
echo "task=$TASK gpu=$GPU num_envs=$NUM_ENVS minibatch=$MINIBATCH updates/epoch=$EXPECT max_iterations=$MAX_ITERATIONS (warm 3600 + 2000)" | tee -a "$LOG"
echo "checkpoint=$CKPT" | tee -a "$LOG"
echo "partial-assembly dataset_dir=$DATASET_DIR" | tee -a "$LOG"
echo "wandb: entity=$WANDB_ENTITY project=$WANDB_PROJECT name=$WANDB_NAME" | tee -a "$LOG"

# -- setsid + nohup + timeout -s KILL, redirected straight to a FILE. Never pipe an Isaac run
# through grep or anything else live: grep buffers, and a killed process loses everything sitting in
# that buffer. 172800s = 48h of generous headroom on the master timeout; the loop below judges
# LAUNCH health separately and much sooner (see DEADLINE).
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
  >> "$LOG" 2>&1 < /dev/null &

PID=$!
echo "LAUNCHED pid $PID, log $LOG"

# -- Kill the whole process GROUP, not just the wrapper. Launched under setsid, so $PID is the
# `timeout` leader and the Isaac python is its child in that new session; `kill -9 $PID` alone reaps
# only the wrapper and leaves Isaac holding VRAM with ppid 1, which then makes the NEXT launch refuse
# a GPU it thinks is still busy.
kill_run() {
  kill -9 -"$PID" 2>/dev/null || kill -9 "$PID" 2>/dev/null
  sleep 5
}

# -- FIRST HEALTH CHECK, polled against the LOG FILE (never a live pipe). 15 minutes is generous for
# a fresh 4096-env dexterous scene to construct (contact-rich, first-run shader/kernel compilation)
# while still being "the first few minutes", not "the first hour".
DEADLINE=$(( SECONDS + 900 )); STATUS=TIMEOUT
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

# -- Confirm all four health-check markers explicitly, by name, not just "did SOMETHING resembling
# training start". A run can reach an epoch line while any one of the other three is silently wrong.
FAIL=0
for marker in \
  '\[dexlift\] gravity PINNED at \(\(0.0, 0.0, -9.81\)' \
  '\[dexlift\] episode mixture \(validated post-override\): classic_goal_prob=0.500' \
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
echo "Training is running in the background, pid $PID, log $LOG. This script does not wait for it to finish."
