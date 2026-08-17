#!/usr/bin/env bash
# The bar moved to 3 cm. 30 mm was NOT in the scored ladder and CANNOT be recovered offline: the
# pose rule is (pos_dist < tol) AND (rot_dist < 0.25) AT THE SAME STEP, and the stored per-episode
# record keeps only the two minima separately, which may occur at different steps. So re-score.
# --pos_tol is ADDED to the ladder, so every rung stays comparable with the earlier runs.
set -uo pipefail
export DEXLIFT_LEG_DECOMP=1
R=/home/dom_iva/UWLab_ur5edelto/logs/rl_games

# decisive two first: the best pose checkpoint and the position-only deliverable
run() {  # name task ckpt tilt
  # A word produced by EXPANSION is not an assignment prefix -- the shell decides that
  # syntactically, before expanding. `${4:+TILT=$4} GPU=3 cmd` therefore ran TILT=0.3 as the
  # COMMAND. Export inside a subshell instead, so the optional variable is genuinely optional.
  echo "=== $1"
  echo "CKPT=$3"
  (
    export TASK="$2" COLLIDERS=hullfix3 SELFCOLL=on HAND=ref ARM=ours
    export GPU=3 NUM_ENVS=128 EPISODES=128 POS_TOL=0.03
    [ -n "${4:-}" ] && export TILT="$4"
    bash /home/dom_iva/run_certify.sh "$1" "$3"
  )
}

R1=$R/dexlift_ur5e_delto_reljointpos_tableleg_reorient
RL=$R/dexlift_ur5e_delto_reljointpos_tableleg_lift
run t30_pose_ep3350 DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0 \
  $R1/2026-08-16_09-33-54/nn/last_dexlift_ur5e_delto_reljointpos_tableleg_reorient_ep_3350_rew_37.49787.pth 0.3
run t30_lift_ep1950 DexLift-UR5eDelto-RelJointPos-TableLeg-Lift-v0 \
  $RL/2026-08-16_08-24-59/nn/last_dexlift_ur5e_delto_reljointpos_tableleg_lift_ep_1950_rew_22.796772.pth
run t30_pose_ep3200 DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0 \
  $R1/2026-08-16_05-55-34/nn/last_dexlift_ur5e_delto_reljointpos_tableleg_reorient_ep_3200_rew_41.823456.pth 0.3
run t30_pose_ep3550pull DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0 \
  $R1/2026-08-16_10-36-54/nn/last_dexlift_ur5e_delto_reljointpos_tableleg_reorient_ep_3550_rew_43.362007.pth 0.3
run t30_pose_ep3600 DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0 \
  $R1/2026-08-16_09-33-54/nn/last_dexlift_ur5e_delto_reljointpos_tableleg_reorient_ep_3600_rew_38.38917.pth 0.3
echo CERT30_ALL_DONE
