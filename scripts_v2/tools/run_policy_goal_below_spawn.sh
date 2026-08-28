#!/usr/bin/env bash
# scripts_v2/tools/run_policy_goal_below_spawn.sh (bead UWLab-xp05.3, "Arm 3" of epic UWLab-xp05).
# ALREADY RUN (2026-08-22, delta=3mm and delta=5mm, both n=256 -- see the bead for results): this
# is no longer a staged/pending script -- do not read the line below as current authorization
# status, and do not gate a re-run on GPU/critic3 clearance that has already happened. Arm 3 is
# CLOSED AS REFUTED (bead UWLab-xp05.3): banked depth median ~2.4mm at both deltas, at or below the
# 2.67mm ungated baseline, with acceptance and rejection profile indistinguishable between the two
# deltas -- see the bead's close reason for the full verdict before re-running this for any reason
# other than reproducing that result. One command, assembled here instead of under time pressure,
# mirroring launch_dexlift_tableleg_reorient_finetune.sh's own sha256-pinning discipline.
#
# Usage: run_policy_goal_below_spawn.sh <delta_mm> [gpu-index]
#   e.g. run_policy_goal_below_spawn.sh 3
#        run_policy_goal_below_spawn.sh 5 1
#
# =====================================================================================
# THE IDEA, in one line
# =====================================================================================
# Pin the commanded goal `delta_mm` DEEPER into the bore than the leg spawns (not at spawn), so
# withdrawing the leg during the grasp actively costs reward instead of being free -- see
# dexlift/mdp/partial_assembly.py's GoalBelowSpawnPoseCommand for the mechanism and its docstring's
# explicit warning: judge this by the BANKED DEPTH DISTRIBUTION the run produces, never by whether
# the policy reaches the commanded goal. Zero-shot: this is the EXISTING ep7100 checkpoint, no
# retrain, driven at generation time by DEXLIFT_GOAL_BELOW_SPAWN_MM (env var, not a Hydra field --
# generate_reset_states_policy.py has no Hydra override path at all; see
# dexlift_ur5e_delto_tableleg_env_cfg.py's _apply_partial_assembly_and_goal_toggles docstring for
# the reachability argument this mirrors).
#
# =====================================================================================
# TASK CHOICE, NOT A DEFAULT LEFT ALONE
# =====================================================================================
# --task below deliberately overrides generate_reset_states_policy.py's own default
# (...-TableLeg-Lift-Play-v0). DEXLIFT_PARTIAL_ASSEMBLY/DEXLIFT_GOAL_BELOW_SPAWN_MM are wired into
# exactly the RelJointPos-Reorient{,_PLAY} __post_init__ methods (dexlift_ur5e_delto_tableleg_env_cfg.py's
# _apply_partial_assembly_and_goal_toggles docstring: "LIFT IS STILL EXCLUDED BY CONSTRUCTION"), and
# NOT into the OSC-Reorient sibling either. Exporting either env var against the wrong task has ZERO
# effect on the constructed env_cfg -- generate_reset_states_policy.py itself now REFUSES to run in
# that case (a construction-time guard added for exactly this failure mode, bead UWLab-xp05.3,
# critic3 review: a silent no-op here would read as "the shaping idea does not work" instead of "the
# toggle was never installed", a false negative on the user's own hypothesis wearing the costume of
# data).
#
# =====================================================================================
# CHECKPOINT: ep7100, NOT ep3600
# =====================================================================================
# ep7100 is the ONLY policy ever trained with partial-assembly episodes -- it took C4 acceptance
# from 12.5% to ~30%. ep3600 never saw a fixture and is the wrong plant for this test. ep7100's
# LOWER training reward (24.15 vs ep3600's 38.39) is NOT a quality signal: training reward is read
# at whatever curriculum difficulty a run reached and is not comparable across runs -- this project
# has a measured case where the worse policy carried the higher training reward and certified 24
# points lower (team-lead correction, 2026-08-22). sha256 verified below, at launch time, off the
# actual bytes on THIS box -- same discipline launch_dexlift_tableleg_reorient_finetune.sh already
# applies to ep3600.
#
# =====================================================================================
# AGENT YAML -- no params/ dir beside either checkpoint on DL_H100 (confirmed by a filesystem-wide
# search, 2026-08-22: the only agent.yaml files on the box belong to unrelated rsl_rl/omnireset_agent
# runs). ep7100 came from a rented A6000 run (2026-08-20_17-47-11) that no longer exists and
# predates the local rescue backup, so there is no original copy to recover.
# =====================================================================================
# Using the IN-REPO yaml instead, EARNED not assumed: git log on
# dexlift/agents/rl_games_ppo_ur5e_delto_reljointpos_tableleg_reorient_cfg.yaml shows exactly ONE
# commit ever touched it (4465605, 2026-08-17 15:45:48 +0900) -- well before ep7100 finished
# training -- and zero uncommitted changes since. The DL_H100 tree's copy is byte-identical to this
# repo's (sha256 55c87f6c... both sides, verified 2026-08-22). So the repo copy IS the training-time
# config for ep7100's window. NOT YET LOAD-TESTED against ep7100 -- that needs an actual Isaac boot
# (restore the checkpoint into the runner, step a couple of iterations) and is held pending GPU
# release; a network-architecture mismatch would fail LOUDLY there with a shape error (rl_games
# restores the state dict against the built model), never silently -- see team-lead's message,
# "Your recon is right... Use the in-repo yaml instead, but EARN it".
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."
cd "$REPO_ROOT" || exit 1

DELTA_MM="${1:?Usage: run_policy_goal_below_spawn.sh <delta_mm> [gpu-index]}"
GPU="${2:-1}"

export UWLAB_TMP_ROOT="${UWLAB_TMP_ROOT:-/home/dom_iva/tmp}"
export TMPDIR="${TMPDIR:-/home/dom_iva/tmp}"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl"

# REFERENCE-PLANT VARS -- same actuator/reset-plant reasoning as run_policy_partial_assembly.sh and
# its siblings (bead UWLab-qiao.1 follow-on: missing these turned a 46.71% acceptance run into a
# 2.69% one). Orthogonal to which goal-command variant is installed.
export DEXLIFT_REF_RESET=1
export DEXLIFT_REF_ACTUATORS=1
export DEXLIFT_REF_HAND_ACT=1
export DEXLIFT_REF_ARM_ACT=0

# THE TOGGLES THIS SCRIPT EXISTS TO EXPORT. DEXLIFT_PARTIAL_ASSEMBLY=1 is REQUIRED by
# DEXLIFT_GOAL_BELOW_SPAWN_MM (GoalBelowSpawnPoseCommand reads the bore's own "deep" axis off
# scene.receptive_object, which only DEXLIFT_PARTIAL_ASSEMBLY=1 adds -- asserted loudly in
# _apply_partial_assembly_and_goal_toggles if this pairing is ever broken).
export DEXLIFT_PARTIAL_ASSEMBLY=1
export DEXLIFT_GOAL_BELOW_SPAWN_MM="$DELTA_MM"

# -- DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR, found the hard way (bead UWLab-xp05.3 load-test,
# 2026-08-22): DEXLIFT_PARTIAL_ASSEMBLY=1 makes SpawnPartialAssembly download partial_assemblies.pt
# for this exact pair the moment it's constructed, and mdp.DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR's
# default (Hugging Face) 404s for this pair -- already documented in partial_assembly.py's own
# module docstring and in launch_dexlift_tableleg_reorient_finetune.sh's header (same failure
# shape there: "MixtureResetObject.__init__() got an unexpected keyword argument 'dataset_dir'" --
# a swallowed HTTPError one level up, surfacing as a TypeError at env.reset() instead of at
# construction). Overridable; default matches where this pair's file actually lives on DL_H100.
export DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR="${DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR:-$PWD/Datasets_ur5e_delto/OmniReset}"
_PARTIAL_ASSEMBLY_PAIR_FILE="$DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR/Resets/OneLegInsertionFixture__SquareTableLeg200mmDecomp/partial_assemblies.pt"
[ -f "$_PARTIAL_ASSEMBLY_PAIR_FILE" ] || {
  echo "REFUSING: partial-assembly dataset not found at $_PARTIAL_ASSEMBLY_PAIR_FILE"
  echo "  (DEXLIFT_PARTIAL_ASSEMBLY=1 needs this file; set DEXLIFT_PARTIAL_ASSEMBLY_DATASET_DIR if"
  echo "  this box's layout differs, or the default HF path will 404 instead)."
  exit 1
}
echo "partial-assembly dataset verified present: $_PARTIAL_ASSEMBLY_PAIR_FILE"

TASK="DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-Play-v0"
RESET_TYPE="ObjectPartiallyAssembledEEGrasped"
# Byte-identical to dexlift/mdp/partial_assembly.py's own DEXLIFT_ONELEGFIXTURE_USD_PATH constant --
# generate_reset_states_policy.py's dexlift scene has no fixture entity to read this off of (module
# docstring), so it must be supplied here and must match what make_dexlift_receptive_object_cfg()
# actually spawns, or compute_pair_dir() names the wrong output directory.
RECEPTIVE_USD_PATH="$PWD/source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/one_leg_insertion_fixture.usd"

CKPT="${CKPT:-/home/dom_iva/ckpt/last_dexlift_ur5e_delto_reljointpos_tableleg_reorient_ep_7100_rew_24.150904.pth}"
CKPT_SHA256_EXPECT="d6a1b676ba32b9821495c241cf7c3dd4e36f675d6b874ead17fc7aaf893890f2"

AGENT_YAML="${AGENT_YAML:-$PWD/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/dexlift/agents/rl_games_ppo_ur5e_delto_reljointpos_tableleg_reorient_cfg.yaml}"
AGENT_YAML_SHA256_EXPECT="55c87f6c7b8004114583c50d8bfa1404b5e35522e8ae6262e5b1e1d75a9d7490"

# PYTHON_BIN is host-specific; default matches DL_H100's known venv, overridable if the box differs.
PYTHON_BIN="${PYTHON_BIN:-/home/dom_iva/venv_uwlab/bin/python}"

# -- Verify the checkpoint bytes on THIS box, not just that a file with the right name exists.
[ -f "$CKPT" ] || { echo "REFUSING: checkpoint not found at $CKPT"; exit 1; }
CKPT_SHA256_GOT="$(sha256sum "$CKPT" | awk '{print $1}')"
[ "$CKPT_SHA256_GOT" = "$CKPT_SHA256_EXPECT" ] || {
  echo "REFUSING: checkpoint sha256 mismatch."
  echo "  expected: $CKPT_SHA256_EXPECT"
  echo "  got:      $CKPT_SHA256_GOT"
  exit 1
}
echo "checkpoint sha256 verified: $CKPT_SHA256_GOT (ep7100, partial-assembly-trained)"

# -- Verify the agent yaml bytes too -- this repo copy is being used as a STAND-IN for ep7100's own
# (unrecoverable) params/agent.yaml on the argument that it has not changed since before ep7100
# trained (see the header comment above); re-verify that argument here, at launch time, rather than
# trusting it stays true.
[ -f "$AGENT_YAML" ] || { echo "REFUSING: agent yaml not found at $AGENT_YAML"; exit 1; }
AGENT_YAML_SHA256_GOT="$(sha256sum "$AGENT_YAML" | awk '{print $1}')"
[ "$AGENT_YAML_SHA256_GOT" = "$AGENT_YAML_SHA256_EXPECT" ] || {
  echo "REFUSING: agent yaml sha256 mismatch -- the repo copy changed since this was verified"
  echo "  unchanged across ep7100's training window (2026-08-22). Re-verify with git log before"
  echo "  trusting it as a stand-in for ep7100's own missing params/agent.yaml."
  echo "  expected: $AGENT_YAML_SHA256_EXPECT"
  echo "  got:      $AGENT_YAML_SHA256_GOT"
  exit 1
}
echo "agent yaml sha256 verified: $AGENT_YAML_SHA256_GOT (unchanged since 2026-08-17, before ep7100's training window)"

# Per-delta scratch dataset dir -- deliberately NOT the default (or the canonical C4 bank's) dir, so
# this zero-shot sweep never collides with or overwrites a bank other work depends on.
DATASET_DIR="./Datasets_ur5e_delto/OmniReset_xp05_3_zeroshot_delta${DELTA_MM}mm"

LOG="$REPO_ROOT/logs/run_policy_goal_below_spawn_delta${DELTA_MM}mm_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"

# No --c4_seating_gate: the point of this run is to bank states and then read the depth
# distribution with scripts_v2/tools/validate_c4_bank.py (harness's tool), not to pre-filter with
# the strict accept/reject gate.
#
# --num_reset_states 256, not 100: harness's baseline (shipped C4 bank, depth median 2.67mm,
# held(contact) 0.953 at 2s) was read at n=256, and team-lead's instruction is to match that n for
# a like-for-like comparison -- a held fraction read at a smaller n has too wide an interval to
# distinguish the arms. 3600s budget (up from the smoke-test's 480s): a real accept-to-256 run is
# not bounded by the same few-hundred-step smoke window.
timeout -s KILL 3600 "$PYTHON_BIN" -u scripts_v2/tools/generate_reset_states_policy.py \
  --task "$TASK" \
  --reset_type "$RESET_TYPE" \
  --receptive_usd_path "$RECEPTIVE_USD_PATH" \
  --checkpoint "$CKPT" \
  --agent_yaml "$AGENT_YAML" \
  --dataset_dir "$DATASET_DIR" \
  --num_envs 128 --num_reset_states 256 --headless \
  > "$LOG" 2>&1
echo "EXIT_CODE=$?" >> "$LOG"
echo "log: $LOG"
