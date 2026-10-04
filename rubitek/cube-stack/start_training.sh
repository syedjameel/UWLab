#!/usr/bin/env bash
# Bring both boxes' banks together, then train BOTH variants at once -- one per machine.
#
# The two variants share every bank (recording grades from asset metadata, never from the command
# term), so there is no reason to run them sequentially. Local takes the position-only variant
# because it is the primary deliverable and the dev card is the faster of the two; the rented A6000
# takes the oriented one.
#
# Banks live on whichever box recorded them: C1 + C4 locally, C3 + C2 remotely. Both boxes need all
# four, so this syncs in both directions before launching anything.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv/bin/python
SSHKEY=/home/mattephi/Work/IRiS/ip/meaningful-latents/.secrets/id_ed25519_ml_vast
HOSTARGS="-i $SSHKEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p 40624"
HOST=root@38.29.145.10
PAIR=InsertiveCube34__ReceptiveCube34
REM=/workspace/UWLab/Datasets_ur10e_delto/OmniReset/Resets/$PAIR
LOC=$REPO/Datasets_ur10e_delto/OmniReset/Resets/$PAIR

# Code first. Config changes made after the last push -- the narrowed mass range, the
# grasp-matched actuator -- must reach the remote box or the oriented variant trains against a
# different task than the local one, and the two results stop being comparable.
echo "=== [$(date +%H:%M:%S)] pushing current code to the remote box ==="
"$CS/vast/launch.sh" push >/dev/null 2>&1 && echo "  code pushed" || { echo "  CODE PUSH FAILED"; exit 1; }

echo "=== [$(date +%H:%M:%S)] pulling remote-recorded banks (C3, C2) ==="
for rt in ObjectAnywhereEEGrasped ObjectRestingEEGrasped; do
  rsync -az -e "ssh $HOSTARGS" "$HOST:$REM/resets_${rt}.pt" "$LOC/" && echo "  pulled $rt"
done

echo
echo "=== [$(date +%H:%M:%S)] pushing locally-recorded banks (C1, C4, partial) ==="
for f in resets_ObjectAnywhereEEAnywhere.pt resets_ObjectPartiallyAssembledEEGrasped.pt partial_assemblies.pt; do
  rsync -az -e "ssh $HOSTARGS" "$LOC/$f" "$HOST:$REM/" && echo "  pushed $f"
done

echo
echo "=== bank inventory, both sides ==="
echo "-- local --"; $PY "$CS/count_bank.py" "$LOC"/*.pt
echo "-- remote --"; ssh $HOSTARGS $HOST "ls -la $REM/ | tail -6"

echo
echo "=== preflight ==="
MIN_STATES=${MIN_STATES:-100} "$CS/preflight_train.sh" || {
  echo "ABORT: banks below floor. Not spending training runs on stub coverage."; exit 1; }

ITERS=${ITERS:-2000}
ENVS=${ENVS:-2048}

echo
echo "=== [$(date +%H:%M:%S)] launching REMOTE: oriented variant ==="
sed -e "s/^ITERS=.*/ITERS=\${ITERS:-$ITERS}/" -e "s/^ENVS=.*/ENVS=\${ENVS:-$ENVS}/" \
  "$CS/vast/job_train_oriented.sh" > /tmp/job_train_oriented.sh
chmod +x /tmp/job_train_oriented.sh
"$CS/vast/launch.sh" job /tmp/job_train_oriented.sh

echo
echo "=== [$(date +%H:%M:%S)] launching LOCAL: position-only variant ==="
nohup "$CS/train_local.sh" noorient "$ITERS" "$ENVS" > "$CS/logs/train_noorient_driver.log" 2>&1 &
echo "local training pid $!"
echo "=== both variants training in parallel ==="
