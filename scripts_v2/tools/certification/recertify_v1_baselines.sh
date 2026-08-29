#!/usr/bin/env bash
# PREPARED, NOT RUN. Bead dr-ai1.12 / dr-76w.18.
#
# Re-certifies the two v1 baselines with the leg asset PINNED and ASSERTED, so the result cannot
# repeat the ambiguity that made dr-ai1.12 necessary. Queue behind the C3 smoke and the bore-mixture
# certification; the box is busy until ~07:45 +03:00.
#
# ------------------------------------------------------------------------------------------------
# WHY THIS EXISTS
#
# The original certifications recorded WHICH HAND asset they loaded (plant.robot_usd) but never the
# LEG. The only leg statement in their logs was the "[dexlift] leg collider: convexDecomposition
# (re-authored)" banner, which was stale prose that named the wrong collider on every run and
# asserted, inside its own text, a conclusion about a number ("the certified 92.87% run also used
# the hull"). F35 then quoted that sentence back out of a log as evidence. The banner was quoting
# its own author.
#
# The investigation reached MEDIUM-HIGH confidence that both baselines ran on the re-authored
# convexDecomposition leg, from two independent lines (no leg override in the exhaustive DEXLIFT_*
# capture; no PhysX hull-fallback warning in complete logs whose structurally identical siblings do
# carry it). This script exists to convert that into a MEASURED answer, and to produce the
# hull-vs-decomposition delta that decides whether the v1 numbers are comparable to v2 at all.
#
# ------------------------------------------------------------------------------------------------
# WHAT MAKES THIS RUN UNAMBIGUOUS WHERE THE ORIGINAL WAS NOT
#
#   1. DEXLIFT_TABLE_LEG_USD_PATH_OVERRIDE pins the leg EXPLICITLY. It is the only live selector,
#      and being a DEXLIFT_* variable it is captured verbatim into the cert JSON's plant.dexlift_env
#      by certify_pose.py -- so the artifact records the choice rather than implying it.
#   2. LEG makes run_certify.sh ASSERT the resolved leg against the new "[dexlift] ASSETS" banner
#      and exit 1 on mismatch, INCLUDING when the banner is absent. A wrong or missing leg can no
#      longer produce a number.
#   3. The banner names the parent DIRECTORY, not the filename: both variants contain a file called
#      square_table_leg4_200mm.usd, so the filename discriminates nothing.
#
# ------------------------------------------------------------------------------------------------
# PRECONDITIONS -- verify on the box before running, none are checked here
#
#   * ~/ckpt holds both checkpoints. DO NOT OVERWRITE ~/ckpt; it is the only local copy.
#       lift   ep_1950  sha256 857dabb3...  last_..._tableleg_lift_ep_1950_rew_22.796772.pth
#       repose ep_3600  sha256 9534e102...  last_..._tableleg_reorient_ep_3600_rew_38.38917.pth
#     Confirm with sha256sum before trusting any result -- the sha is what ties a number to a
#     checkpoint, and both are recorded in RESET_SPEC_V2.md section 6a.
#   * BOTH leg variant directories exist. Every .usd here is gitignored, so their presence is NOT
#     implied by a clean checkout and MUST be checked on the box with ls, not assumed:
#       $ASSETS/Props/FurnitureBench/SquareTableLeg200mmDecomp/square_table_leg4_200mm.usd
#       $ASSETS/Props/FurnitureBench/SquareTableLeg200mmSdf/square_table_leg4_200mm.usd
#     If Decomp is absent, scripts/reauthor_leg_decomp.py regenerates it from SquareTableLeg200mm.
#   * run_certify.sh is the version carrying the LEG assertion (bead dr-76w.18). If the assertion is
#     missing, this script's whole point is missing with it -- grep for ASSET MISMATCH first.
#
# ------------------------------------------------------------------------------------------------
# WHAT TO CONCLUDE FROM THE OUTPUT
#
#   * Decomp numbers matching 92.19 / 88.28 within Wilson CI  -> the investigation's finding is
#     confirmed, F35 is refuted, and section 6a's baselines stand as measured on a decomposition.
#   * Sdf numbers differing materially from them               -> v1 and v2 are NOT comparable
#     across the 2026-08-23 asset switch and every v2-vs-v1 claim needs the delta stated.
#   * Either run failing the ASSET assertion                   -> stop; the leg is not what the
#     command asked for, and no number from it may be quoted.
#
# The Sdf arm is the one that actually settles the comparability question, so do not skip it to save
# GPU time: without it, a confirmed Decomp baseline still says nothing about the leg v2 runs on.

set -uo pipefail

REPO_DIR=${REPO_DIR:?set REPO_DIR to the UWLab checkout on the box}
ASSETS="$REPO_DIR/source/uwlab_assets/uwlab_assets/local"
CKPT_DIR=${CKPT_DIR:-$HOME/ckpt}
OUT_DIR=${OUT_DIR:-$HOME/recert_dr_ai1_12}

LIFT_CKPT="$CKPT_DIR/last_dexlift_ur5e_delto_reljointpos_tableleg_lift_ep_1950_rew_22.796772.pth"
POSE_CKPT="$CKPT_DIR/last_dexlift_ur5e_delto_reljointpos_tableleg_reorient_ep_3600_rew_38.38917.pth"

# Plant axes: EXACTLY the originals, read back from the two cert JSONs' plant.dexlift_env rather
# than retyped from a recipe -- REF_ARM_ACT=0 (identified arm) and REF_HAND_ACT=1 (reference hand),
# which is what run_certify.sh spells HAND=ref ARM=ours. DEXLIFT_HULLFIX and DEXLIFT_LEG_DECOMP
# appeared in those JSONs too and are deliberately NOT set here: both are dead (bead dr-76w.18) and
# reproducing them would only re-record two variables nothing reads.
export COLLIDERS=hullfix3 SELFCOLL=on HAND=ref ARM=ours
export DEXLIFT_REF_RESET=1 DEXLIFT_REF_ACTUATORS=1
export NUM_ENVS=${NUM_ENVS:-128} EPISODES=${EPISODES:-128}
mkdir -p "$OUT_DIR"

recert() {  # $1 tag  $2 checkpoint  $3 leg variant dir  $4 extra env ("" or "DEXLIFT_POSE_TILT=0.3")
  local tag=$1 ckpt=$2 variant=$3 extra=${4:-}
  local usd="$ASSETS/Props/FurnitureBench/$variant/square_table_leg4_200mm.usd"

  [ -f "$ckpt" ] || { echo "REFUSING $tag: checkpoint missing: $ckpt"; return 1; }
  [ -f "$usd" ]  || { echo "REFUSING $tag: leg usd missing: $usd (gitignored -- check the box)"; return 1; }

  echo "=== $tag : $variant ==="
  ( export DEXLIFT_TABLE_LEG_USD_PATH_OVERRIDE="$usd"   # pins it
    export LEG="$variant"                               # asserts it, fail-closed
    [ -n "$extra" ] && export "${extra?}"
    export OUT_DIR
    bash "$REPO_DIR/scripts_v2/tools/certification/run_certify.sh" "$tag" "$ckpt"
  ) 2>&1 | tee "$OUT_DIR/$tag.console"
}

# TILT: the repose baseline recorded DEXLIFT_POSE_TILT=0.3 in its plant and the lift one did NOT.
# That asymmetry is real and must be reproduced -- a repose number measured unstaged is not
# comparable to 88.28 (RESET_SPEC_V2.md section 1a trap 2: a tilt also clamps the drop height).
recert recert_lift_ep1950_decomp "$LIFT_CKPT" SquareTableLeg200mmDecomp ""
recert recert_lift_ep1950_sdf    "$LIFT_CKPT" SquareTableLeg200mmSdf    ""
recert recert_pose_ep3600_decomp "$POSE_CKPT" SquareTableLeg200mmDecomp "DEXLIFT_POSE_TILT=0.3"
recert recert_pose_ep3600_sdf    "$POSE_CKPT" SquareTableLeg200mmSdf    "DEXLIFT_POSE_TILT=0.3"

cat <<'SUMMARY'

############################################################################
# Compare against RESET_SPEC_V2.md section 6a:
#     lift   ep_1950  pass@30mm = 0.9219
#     repose ep_3600  pass@50mm = 0.8828   (and pass@30mm = 0.6953)
#
# For each run, read THREE things out of the log before believing the number:
#     grep "assets: "          -- the resolved leg, asserted
#     grep "plant: "           -- the four plant axes, asserted
#     grep "CERTIFY_RESULT"    -- the rates, with the Wilson interval
#
# Quote no number whose run did not print BOTH assertion lines. That is the
# whole point of this script.
############################################################################
SUMMARY
