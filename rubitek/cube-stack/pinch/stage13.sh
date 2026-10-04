#!/usr/bin/env bash
# Stage 13: is the SHIPPED C2 bank actually a grasped state? Then regenerate the pinch families.
set -uo pipefail
cd "$(dirname "$0")"; P=$PWD
SRC=/home/mattephi/Work/Rubitek/UWLab-cubestack/Datasets_ur10e_delto/OmniReset/Resets/InsertiveCube34__ReceptiveCube34
log(){ echo "=== $(date -Is) $*" | tee -a out/stage13.log; }

for T in ObjectRestingEEGrasped ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped; do
  log "dump $T"
  ./run6.sh "$P/banklab.py" --bank "$SRC/resets_$T.pt" --states 40 --settle-s 0.02 \
      --dump-relative "$P/out/rel40_$T.json" > "out/rel40_$T.log" 2>&1
  echo "    rc=$?" | tee -a out/stage13.log
done
log "restarting reset generation"
./stage11b.sh >> out/stage11b_driver.log 2>&1
echo "STAGE13_DONE $(date -Is)" | tee -a out/stage13.log
