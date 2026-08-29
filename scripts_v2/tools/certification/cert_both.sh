#!/usr/bin/env bash
# Certify the two live pose arms at their latest checkpoint, sequentially, on GPU3.
# Both arms are warm from the 0.8906 pose checkpoint (ep3350) and are ~200 epochs past
# their warm start, i.e. past the reward-change transient that made an earlier
# certification read WORSE than the policy actually was.
set -uo pipefail
# DEXLIFT_LEG_DECOMP removed (bead dr-76w.18): read by no python anywhere, so it never
# selected a leg. The leg comes from TABLE_LEG_USD_PATH (shipping: SquareTableLeg200mmSdf)
# and is overridable only by DEXLIFT_TABLE_LEG_USD_PATH_OVERRIDE. run_certify.sh now
# asserts the resolved leg against LEG (default SquareTableLeg200mmSdf) and exits 1 on
# mismatch, so the choice is checked rather than merely requested.

R=/home/dom_iva/UWLab_ur5edelto/logs/rl_games/dexlift_ur5e_delto_reljointpos_tableleg_reorient

# 10-36-54 is legreorient_sharp_pull (pid 3616874, started 10:36:46)
# 09-33-54 is legreorient_sharp03   (pid 3583141, started 09:33:46)
PULL=$(ls -t $R/2026-08-16_10-36-54/nn/last_*.pth | head -1)
SH03=$(ls -t $R/2026-08-16_09-33-54/nn/last_*.pth | head -1)

for pair in "legreorient_sharp_pull_latest:$PULL" "legreorient_sharp03_latest:$SH03"; do
  NAME=${pair%%:*}; CKPT=${pair#*:}
  echo "=== $NAME"
  echo "CKPT=$CKPT"
  TASK=DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0 \
  COLLIDERS=hullfix3 SELFCOLL=on HAND=ref ARM=ours \
  TILT=0.3 GPU=3 NUM_ENVS=128 EPISODES=128 POS_TOL=0.01 \
    bash /home/dom_iva/run_certify.sh "$NAME" "$CKPT"
done
echo ALL_CERTS_DONE
