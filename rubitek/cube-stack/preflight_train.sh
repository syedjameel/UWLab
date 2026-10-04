#!/usr/bin/env bash
# Verify everything the cube-stacking training env needs, BEFORE spending hours on a run.
#
# This exists because of D6: a missing reset bank does not produce a "file not found" at the point
# of use. MultiResetManager is constructed inside an Omniverse timeline callback that prints and
# continues, so the term is left as a class and the failure surfaces later as
# "TypeError: MultiResetManager.__init__() got an unexpected keyword argument 'dataset_dir'",
# which names neither the file nor the directory. Checking the four files up front is far cheaper
# than decoding that at hour three of a training run.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
REPO=${REPO:-/home/mattephi/Work/Rubitek/UWLab-cubestack}
DS=$REPO/Datasets_ur10e_delto/OmniReset
PAIR=InsertiveCube34__ReceptiveCube34
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv/bin/python
MIN_STATES=${MIN_STATES:-100}

# The four families TrainEventCfg samples at 0.25 each (rl_state_cfg.py). The fifth registered
# family, ObjectPartiallyAssembledEEAnywhere, is never sampled and is deliberately not required.
REQUIRED=(ObjectAnywhereEEAnywhere ObjectRestingEEGrasped ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped)

echo "=== training pre-flight $(date +%H:%M:%S) ==="
echo "pair dir: $DS/Resets/$PAIR"
fail=0
for rt in "${REQUIRED[@]}"; do
  f="$DS/Resets/$PAIR/resets_${rt}.pt"
  if [ ! -f "$f" ]; then echo "  [MISSING] $rt"; fail=1; continue; fi
  line=$("$PY" "$CS/count_bank.py" "$f")
  n=$(echo "$line" | awk '{print $(NF-1)}')
  # PRESENCE IS NOT ENOUGH. Killed runs leave stub banks of a few dozen states that satisfy a
  # file-exists check and would train against almost no coverage. MIN_STATES is the floor below
  # which a bank is a leftover, not a dataset.
  if ! [ "$n" -ge "$MIN_STATES" ] 2>/dev/null; then
    echo "  [TOO SMALL] $line  (need >= $MIN_STATES)"; fail=1
  else
    echo "  [ok]        $line"
  fi
done
for extra in "$DS/Resets/$PAIR/partial_assemblies.pt" "$DS/Grasps/InsertiveCube34/grasps.pt"; do
  [ -f "$extra" ] && "$PY" "$CS/count_bank.py" "$extra" | sed 's/^/  /' || { echo "  [MISSING] $extra"; fail=1; }
done

echo
if [ "$fail" -eq 0 ]; then
  echo "[PREFLIGHT] PASS -- all four sampled families present with >= $MIN_STATES states"
else
  echo "[PREFLIGHT] FAIL -- fix before training. A missing bank dies with the misleading"
  echo "            TypeError from D6; a stub bank trains silently on no coverage, which is worse."
fi
exit $fail
