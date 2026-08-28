#!/usr/bin/env zsh
# scripts_v2/tools/run_policy_partial_assembly.sh (bead UWLab-algw.9). NEW, not vendored: none of
# the four run_policy_*.sh siblings (run_policy_bounded.sh, run_policy_refplant.sh,
# run_policy_reset_states.sh, run_policy_smoke.sh) ever passed --reset_type or
# --receptive_usd_path -- both `required=True` in generate_reset_states_policy.py's argparse, so
# those four cannot even be the launcher that produced a real ObjectPartiallyAssembledEEGrasped
# file -- and none of them exported DEXLIFT_PARTIAL_ASSEMBLY. This IS "whatever wrapper regenerates
# the C4 bank" that the new --reset_type guard added to generate_reset_states_policy.py (this same
# bead) needs to be able to catch: the two guards that already existed there both PASSED under
# whatever launcher produced the bad Resting-shipped-as-Partial bank, because neither ever looked
# at --reset_type.
#
# TASK CHOICE, NOT A DEFAULT LEFT ALONE. --task below deliberately overrides
# generate_reset_states_policy.py's own default (...-TableLeg-Lift-Play-v0). DEXLIFT_PARTIAL_ASSEMBLY
# is wired into exactly two __post_init__ methods, both on Reorient classes
# (dexlift_ur5e_delto_tableleg_env_cfg.py's _apply_partial_assembly_and_goal_toggles docstring:
# "LIFT IS STILL EXCLUDED BY CONSTRUCTION" -- on Lift, rot_std is forced None, so goal-at-spawn
# would pay an idle policy against Lift's own reward gate). Exporting the toggle against the Lift
# task would have ZERO effect on the constructed env_cfg -- events.reset_object.func would stay
# whatever Lift's default is, not SpawnPartialAssembly -- and the new --reset_type guard would then
# correctly REFUSE to run at all. Using the Reorient-Play task id is what makes the toggle (and the
# guard) meaningful, not optional polish.
#
# CHECKPOINT CHOICE: local_ckpts/dexlift_ur5e_delto_reljointpos_tableleg_reorient/.../last_..._
# ep_3600_rew_38.38917.pth -- the REORIENT-trained checkpoint already on this machine, matching the
# Reorient task above (the run_policy_*.sh siblings all point at a LIFT checkpoint, which would be
# the wrong plant for this task). NOT independently verified by this change whether this
# checkpoint's observation/action space behaves sensibly when rolled out against a leg spawned
# mid-insertion (partial assembly) rather than the freshly-randomized poses it was likely trained
# against -- there is no Isaac access in this change to check. Flagged in the bead report; verify
# on the rented box before trusting any reset states this produces.
set -x

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR}/../.."
cd "$REPO_ROOT" || exit 1

export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl"

# REFERENCE-PLANT VARS -- same actuator/reset-plant reasoning as the other run_policy_*.sh
# launchers (bead UWLab-qiao.1 follow-on: missing these turned a 46.71% acceptance run into a
# 2.69% one). The robot's physical plant (hand effort/velocity limits) is orthogonal to Lift vs
# Reorient -- both tasks share the same robot -- so this block is unchanged from the siblings.
export DEXLIFT_REF_RESET=1
export DEXLIFT_REF_ACTUATORS=1
export DEXLIFT_REF_HAND_ACT=1
export DEXLIFT_REF_ARM_ACT=0

# THE TOGGLE THIS LAUNCHER EXISTS TO EXPORT (bead UWLab-algw.9). Unconditional, not the
# default-off/caller-overrides pattern DEXLIFT_SPAWN_CLEARANCE uses elsewhere in this file's
# siblings -- generating the ObjectPartiallyAssembledEEGrasped bank IS this script's entire
# purpose, so there is no "off" mode worth defaulting to.
export DEXLIFT_PARTIAL_ASSEMBLY=1

TASK="DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-Play-v0"
RESET_TYPE="ObjectPartiallyAssembledEEGrasped"
# Byte-identical to dexlift/mdp/partial_assembly.py's own DEXLIFT_ONELEGFIXTURE_USD_PATH constant
# (UWLAB_ASSETS_DATA_DIR/Props/FurnitureBench/OneLegInsertionFixture/one_leg_insertion_fixture.usd)
# -- generate_reset_states_policy.py's dexlift scene has no fixture entity to read this off of
# (module docstring), so it must be supplied here and must match what make_dexlift_receptive_
# object_cfg() actually spawns, or compute_pair_dir() names the wrong output directory.
RECEPTIVE_USD_PATH="$PWD/source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/one_leg_insertion_fixture.usd"
CKPT="$PWD/local_ckpts/dexlift_ur5e_delto_reljointpos_tableleg_reorient/2026-08-16_09-33-54/nn/last_dexlift_ur5e_delto_reljointpos_tableleg_reorient_ep_3600_rew_38.38917.pth"

# PYTHON_BIN is host-specific (wherever this host's Isaac-enabled interpreter lives) -- the
# run_policy_*.sh siblings hardcode a DL_A6000 path (/home/dom_iva/UWLab/env_uwlab/bin/python);
# this script deliberately does NOT, because the bead report this was written for says GPU
# verification happens "later on a rented box" -- a host this change has no path conventions for
# yet. Set PYTHON_BIN in the caller's environment before running.
: "${PYTHON_BIN:?Set PYTHON_BIN to this host's Isaac-enabled python before running.}"

LOG="$REPO_ROOT/policy_reset_states_partial_assembly.log"
timeout -s KILL 1800 "$PYTHON_BIN" -u scripts_v2/tools/generate_reset_states_policy.py \
  --task "$TASK" \
  --reset_type "$RESET_TYPE" \
  --receptive_usd_path "$RECEPTIVE_USD_PATH" \
  --checkpoint "$CKPT" \
  --num_envs 128 --num_reset_states 300 --headless \
  > "$LOG" 2>&1
echo "EXIT_CODE=$?" >> "$LOG"
