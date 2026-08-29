#!/usr/bin/env bash
# Does the goal-at-spawn finetune still do the ORIGINAL reorient task?
#
# Stage A of chain2 measured the finetuned ep5150 producing reset states with a much better
# height distribution (frac in 0.019-0.050 band 0.137 -> 0.283) but at a third of the
# acceptance rate (58.14% -> 19.46%). Two explanations fit that equally well:
#   1. grasping near the table is genuinely harder, so hold rate falls with goal height
#   2. the finetune degraded the policy's holding ability generally
# Reset-state generation cannot separate them -- it only ever runs the finetuned task.
# Certifying under the BASE reorient task can: if the base skill is intact, (1); if it
# collapsed, (2), and the parent checkpoint is the better generator.
#
# run_certify.sh does NOT export DEXLIFT_GOAL_AT_SPAWN or DEXLIFT_SPAWN_CLEARANCE, which is
# exactly the point -- this scores ep5150 on the task ep3600 was certified on.
#
# ep3600 is re-certified IN THE SAME BATCH rather than compared against its stored 0.6953.
# Measured reproducibility on an identical checkpoint in this project is +-2 points at 10 mm
# and +-3.5 at 30 mm across bitwise-identical replays, so a cross-batch difference smaller
# than that is not a difference. An in-batch control is the only way to read the gap.
set -uo pipefail
R=/home/dom_iva/UWLab_ur5edelto/logs/rl_games/dexlift_ur5e_delto_reljointpos_tableleg_reorient
LOG=/home/dom_iva/cert_ft_driver.log
echo "CERT_FT_STARTED $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG"

# GPU 1 was idle at 17 MiB when this was written; GPU 0 and GPU 3 both carry ~36 GB tenants.
# Pinned explicitly rather than taking run_certify.sh's GPU=3 default, which would land on the
# fully-loaded card.
run() {  # $1 tag  $2 checkpoint
  echo "=== $1  $(date -u +%H:%M:%SZ)" >> "$LOG"
  echo "CKPT=$2" >> "$LOG"
  (
    export TASK=DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0
    export COLLIDERS=hullfix3 SELFCOLL=on HAND=ref ARM=ours
    export GPU=1 NUM_ENVS=128 EPISODES=128 POS_TOL=0.03 TILT=0.3
    # DEXLIFT_LEG_DECOMP removed (bead dr-76w.18): read by no python anywhere, so it never
    # selected a leg. The leg comes from TABLE_LEG_USD_PATH (shipping: SquareTableLeg200mmSdf)
    # and is overridable only by DEXLIFT_TABLE_LEG_USD_PATH_OVERRIDE. run_certify.sh now
    # asserts the resolved leg against LEG (default SquareTableLeg200mmSdf) and exits 1 on
    # mismatch, so the choice is checked rather than merely requested.
    bash /home/dom_iva/run_certify.sh "$1" "$2"
  ) >> "$LOG" 2>&1
  echo "--- $1 rc above ---" >> "$LOG"
}

# Control first. If the control itself does not reproduce ~0.69 at 30 mm, the batch is not
# comparable to the stored certification and the ep5150 number must not be read against it.
run ftctl_pose_ep3600 "$R/2026-08-16_09-33-54/nn/last_dexlift_ur5e_delto_reljointpos_tableleg_reorient_ep_3600_rew_38.38917.pth"
run ftnew_pose_ep5150 /home/dom_iva/ckpt_ft_ep5150.pth

echo "CERT_FT_ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
