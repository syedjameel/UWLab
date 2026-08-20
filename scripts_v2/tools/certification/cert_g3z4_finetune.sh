#!/usr/bin/env bash
# Certification for epic UWLab-g3z4's finetune (full gravity, low-goal band, partial-assembly
# grasp-only mixture -- commit 4818256). STAGED, NOT RUN: this is the gate on whether the finetune
# is kept at all, and must not be launched until launch_dexlift_tableleg_reorient_finetune.sh (in
# the parent tools/ directory) has actually produced a checkpoint.
#
# DO NOT JUDGE THE FINETUNE BY TRAINING REWARD. This project has a documented case where the WORSE
# policy had the HIGHER training reward (42.8 against 38.4) and certified 24 points LOWER. The
# number this script produces is the only one that decides whether the finetune is kept.
#
# =====================================================================================
# PROTOCOL (team lead, epic UWLab-g3z4)
# =====================================================================================
#   - 128 episodes total, seeds 101/202/303/404 -- certify_pose.py's own default --seeds list, 32
#     episodes per seed (EPISODES is split evenly across seeds, rounding up if it does not divide).
#   - Deterministic policy: certify_pose.py hard-codes agent.get_action(obs, is_deterministic=True)
#     -- there is no flag for this, nothing to pass, it cannot be accidentally left stochastic.
#   - ADR pinned to max: ADR_DIFFICULTY=max, exported below and forwarded by run_certify.sh to
#     certify_pose.py's own --adr_difficulty flag -- passed explicitly now, not merely inherited from
#     certify_pose.py's own default (:119). An earlier revision of this comment claimed it was passed
#     explicitly while the code only relied on the default; that was fixed by actually wiring
#     ADR_DIFFICULTY through run_certify.sh's python invocation (see run_certify.sh), not by editing
#     the comment to match the weaker behaviour -- a silent dependency on a default is exactly what
#     breaks when someone later changes that default. Still double-checkable in the output regardless:
#     every CERTIFY_RESULT line carries adr_difficulty_frac, and the stored control cert records
#     adr_difficulty 'max'/pinned_at 10. If it ever reads anything but full difficulty, the number is
#     not comparable to the stored 0.6953 and must not be quoted against it.
#   - The BASE Reorient task, i.e. the SAME task id training runs on
#     (DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0) UNDER THE CLASSIC-ONLY GOAL DISTRIBUTION --
#     NOT under the episode mixture the finetune trained under, and this is DELIBERATE, not an
#     oversight. CORRECTED FROM AN EARLIER REVISION OF THIS SCRIPT, which reasoned the opposite way
#     (certify under "the same mixture it trained under") before the mixture became opt-in
#     (DEXLIFT_EPISODE_MIXTURE=1, see mdp/episode_mixture.py's "THE MIXTURE IS OPT-IN" section) --
#     that reasoning does not survive contact with what pass@30mm actually measures: a
#     PARTIAL-ASSEMBLY episode pins the goal to the object's OWN SPAWN POSE, so it is trivially
#     "successful" at every tolerance the instant the leg is grasped, with no transport required.
#     Certifying with DEXLIFT_EPISODE_MIXTURE=1 would let 25% of scored episodes (the class default
#     partial_assembly_prob) be near-automatic passes that say nothing about whether the policy can
#     TRANSPORT the leg to a real goal -- inflating pass@30mm regardless of what actually changed.
#     run_certify.sh (the delegate below) exports neither the legacy toggles NOR
#     DEXLIFT_EPISODE_MIXTURE, so the constructed task is the plain, classic-goal-only Reorient task
#     for BOTH the control and the finetune -- the one "base Reorient task" and "pass@30mm" both
#     refer to, and the one the STORED 0.6953 control figure was itself measured under.
#
# =====================================================================================
# THE PARENT (ep3600) IS RE-CERTIFIED IN THE SAME BATCH, AS A CONTROL -- NOT COMPARED AGAINST ITS
# STORED FIGURE ALONE
# =====================================================================================
# Stored: pass@30mm 0.6953 (local_ckpts/certification_2026-08-17/cert_t30_pose_ep3600.json).
# Measured reproducibility on an IDENTICAL checkpoint in this project, across bitwise-identical
# replays, is roughly +-2 points at 10mm and +-3.5 points at 30mm (cert_ft.sh's own finding) --
# treat any difference smaller than that as noise, not signal. If the control does not land within
# that band of 0.6953 in THIS batch, the batch itself is not comparable to the stored certification
# and the finetune's number must not be read against 0.6953 at all -- only against the control run
# in the SAME batch.
#
# =====================================================================================
# WHY THIS DELEGATES TO run_certify.sh RATHER THAN REIMPLEMENTING ANYTHING
# =====================================================================================
# run_certify.sh already asserts the constructed plant (COLLIDERS/SELFCOLL/HAND/ARM) against a
# [dexlift] PLANT log line before trusting the run's numbers -- the same class of check
# launch_dexlift_tableleg_reorient_finetune.sh's health-check loop performs for training. This
# script sets the SAME four plant axes and TILT the finetune trained under (HAND=ref -> reference
# hand actuators, ARM=ours -> our identified arm plant, TILT=0.3), so certification measures the
# policy under the plant it actually learned in.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# -- FILL IN AFTER TRAINING PRODUCES A CHECKPOINT. No default, deliberately: a script that quietly
# certified nothing, or certified the wrong .pth, would be a worse outcome than refusing to run.
# Point this at the .pth under logs/rl_games/dexlift_ur5e_delto_reljointpos_tableleg_reorient/<run
# timestamp>/nn/ that launch_dexlift_tableleg_reorient_finetune.sh's own log names.
FINETUNE_CKPT=${FINETUNE_CKPT:?"set FINETUNE_CKPT=<path to the trained .pth> before running this script"}

# -- Control checkpoint: same top-of-script-variable, box-layout-default pattern as the launch
# script, for the identical portability reason (do not assume local_ckpts/ exists on the box this
# runs on -- override CONTROL_CKPT=<path> if it differs).
CONTROL_CKPT=${CONTROL_CKPT:-/root/ckpt/last_dexlift_ur5e_delto_reljointpos_tableleg_reorient_ep_3600_rew_38.38917.pth}
CONTROL_SHA256_EXPECT="9534e102a64fc06a3588c5102fc3421e69ef1b18fe30e7ffb40f7df63b5d76af"

GPU=${GPU:-0}
USED=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
[ "${USED:-999999}" -lt 2000 ] || { echo "REFUSING: GPU $GPU has ${USED} MiB already in use"; exit 1; }

[ -f "$CONTROL_CKPT" ] || { echo "REFUSING: control checkpoint not found at $CONTROL_CKPT"; exit 1; }
[ -f "$FINETUNE_CKPT" ] || { echo "REFUSING: FINETUNE_CKPT not found at $FINETUNE_CKPT"; exit 1; }
GOT_SHA256="$(sha256sum "$CONTROL_CKPT" | awk '{print $1}')"
[ "$GOT_SHA256" = "$CONTROL_SHA256_EXPECT" ] || {
  echo "REFUSING: control checkpoint sha256 mismatch."
  echo "  expected: $CONTROL_SHA256_EXPECT"
  echo "  got:      $GOT_SHA256"
  exit 1
}
echo "control checkpoint sha256 verified: $GOT_SHA256"
echo "finetune checkpoint: $FINETUNE_CKPT (no known-good hash to check this one against -- it is what this script exists to evaluate)"

DRIVER_LOG="$REPO_ROOT/logs/cert_g3z4_finetune_$(date +%Y%m%d_%H%M%S).driver.log"
mkdir -p "$(dirname "$DRIVER_LOG")"
: > "$DRIVER_LOG"

# DEFENSIVE UNSETS -- these four change WHAT TASK IS SCORED, and all four are opt-in env vars that
# survive in an interactive shell. If DEXLIFT_EPISODE_MIXTURE=1 leaked in from having sourced the
# finetune launch script, ~25% of scored episodes would be partial-assembly episodes whose goal is
# pinned to their own spawn -- near-automatic passes -- and the finetune would certify as an
# improvement BECAUSE it was scored on an easier task. The number would look entirely clean.
# GOAL_AT_SPAWN is the same defect without the mixture wrapper; PARTIAL_ASSEMBLY changes the spawn;
# SPAWN_CLEARANCE (dexlift_ur5e_delto_env_cfg.py:1543) swaps reset_object for a clearance-aware spawn
# in the SAME shared __post_init__ path -- cert_ft.sh's own precedent names it explicitly as one of
# the two things "run_certify.sh does NOT export ... which is exactly the point".
unset DEXLIFT_EPISODE_MIXTURE
unset DEXLIFT_GOAL_AT_SPAWN
unset DEXLIFT_PARTIAL_ASSEMBLY
unset DEXLIFT_SPAWN_CLEARANCE

run() {  # tag  checkpoint
  local tag="$1" ckpt="$2"
  echo "=== $tag  $(date -u +%H:%M:%SZ)" | tee -a "$DRIVER_LOG"
  echo "CKPT=$ckpt" | tee -a "$DRIVER_LOG"
  (
    export TASK=DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0
    export COLLIDERS=hullfix3 SELFCOLL=on HAND=ref ARM=ours
    # ADR pinned to max, EXPLICITLY -- forwarded by run_certify.sh to certify_pose.py's own
    # --adr_difficulty. Matches certify_pose.py's own default (:119) and the stored control cert's
    # domain.adr_difficulty ('max'/pinned_at 10); made explicit here so a future change to that
    # default cannot silently decouple this certification from the stored 0.6953 protocol.
    export ADR_DIFFICULTY=max
    # DEXLIFT_LEG_DECOMP IS DELIBERATELY ABSENT AND ITS ABSENCE IS NOT A PLANT DIFFERENCE.
    # It appears in the plant.dexlift_env dict of EVERY historical stored cert, including the
    # control's own 0.6953, so a reader diffing this script against those JSONs will notice it
    # missing here. A tree-wide grep finds it SET by older cert scripts and recorded in their
    # JSONs, and READ NOWHERE -- there is no os.environ.get/os.environ[] site for it anywhere.
    # The decomposed-leg USD is now unconditional (dexlift_ur5e_delto_tableleg_env_cfg.py:39-77),
    # so the variable is vestigial and the simulated plant is identical with or without it.
    # Documented rather than re-added: setting a variable nothing reads is theatre.
    export GPU NUM_ENVS=128 EPISODES=128 POS_TOL=0.03 TILT=0.3
    bash "$REPO_ROOT/scripts_v2/tools/certification/run_certify.sh" "$tag" "$ckpt"
  ) 2>&1 | tee -a "$DRIVER_LOG"
  echo "--- $tag rc above ---" | tee -a "$DRIVER_LOG"
}

# CONTROL FIRST. If it does not reproduce ~0.6953 (+-3.5 points) at 30mm, the batch is not
# comparable and the finetune number below must not be trusted against the stored figure.
run g3z4_control_ep3600 "$CONTROL_CKPT"
run g3z4_finetune "$FINETUNE_CKPT"

echo "CERT_G3Z4_ALL_DONE" | tee -a "$DRIVER_LOG"
echo "run_certify.sh writes its own \$HOME/cert_<tag>.json and \$HOME/cert_<tag>.log per invocation" \
     "(its own path convention, not overridable from here) -- read pass@0.03 (30mm) from" \
     "\$HOME/cert_g3z4_control_ep3600.json and \$HOME/cert_g3z4_finetune.json, IN THIS SAME BATCH," \
     "per the protocol above. Driver log (both runs' stdout/stderr): $DRIVER_LOG"

# NOTE ON HOW TO RUN THIS DETACHED: this driver does not background or timeout-wrap itself --
# run_certify.sh already wraps its own python invocation in `timeout -s KILL 7200`, and this
# script's two `run` calls are sequential and bounded (128 episodes each) by that. cert_ft.sh, the
# existing precedent this driver mirrors, is invoked the same way: NOT self-backgrounding, left to
# the operator to run under `setsid nohup bash cert_g3z4_finetune.sh &> driver.log &` if a detached
# invocation is wanted. Not duplicated here to avoid a second, differently-configured timeout
# wrapping the same underlying call.
