#!/usr/bin/env bash
# Rent a GPU, provision it, and push the repo + datasets to it.
#
#   ./launch.sh create <offer_id> [disk_gb]   rent and provision
#   ./launch.sh push                          rsync the worktree + datasets to the instance
#   ./launch.sh job <local_job.sh>            queue a job (runs under container init)
#   ./launch.sh log <name>                    tail a job log
#   ./launch.sh status                        instance + provisioning state
#   ./launch.sh destroy                       DESTROY the instance (billing stops)
#
# Cost discipline: this rents real hardware on the Sinew team account, which had $65.76 of credit
# and no other funding. Nothing here starts an instance implicitly, and `status` is cheap — run it
# rather than guessing whether something is still billing.
set -uo pipefail

VAST=/home/mattephi/Work/IRiS/ip/meaningful-latents/scripts/vast
SSHKEY=/home/mattephi/Work/IRiS/ip/meaningful-latents/.secrets/id_ed25519_ml_vast
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
HERE=/home/mattephi/Work/Rubitek/cube-stack
IDFILE=$HERE/vast/instance_id${BOX:+_$BOX}

ssh_args() {  # prints "-p PORT user@host"
  local id; id=$(cat "$IDFILE")
  # The instance was created with --direct, so the usable endpoint is the DIRECT one
  # (public_ipaddr : direct_port_start), not the ssh_host/ssh_port proxy pair -- the proxy
  # refuses the key even after `attach ssh` succeeds.
  "$VAST" show instance "$id" --raw 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('-p', d.get('direct_port_start'), 'root@'+str(d.get('public_ipaddr')))
"
}

case "${1:-}" in
create)
  offer=${2:?usage: launch.sh create <offer_id> [disk_gb]}
  disk=${3:-80}
  echo "renting offer $offer with ${disk}GB disk ..."
  "$VAST" create instance "$offer" \
    --image nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 \
    --disk "$disk" --ssh --direct \
    --label cubestack \
    --onstart "$HERE/vast/onstart.sh" --raw | tee "$HERE/vast/create.json"
  python3 -c "
import json,sys
d=json.load(open('$HERE/vast/create.json'))
nid=d.get('new_contract')
assert nid, d
open('$IDFILE','w').write(str(nid))
print('instance id', nid)
"
  echo "provisioning takes ~10-15 min. Watch with: $0 status"
  ;;

push)
  read -r -a A <<< "$(ssh_args)"
  echo "rsyncing worktree ..."
  # --delete is scoped to CODE only. Datasets_ur10e_delto is excluded because the remote box
  # WRITES banks into it while jobs run: an unguarded --delete removed a C4 file that was still
  # being recorded, and overwrote finished remote banks with shorter local ones. Datasets move in
  # one direction only, via `pull`, and never as a side effect of pushing code.
  rsync -az --delete -e "ssh -i $SSHKEY -o StrictHostKeyChecking=accept-new -p ${A[1]}" \
    --exclude '.git' --exclude '_isaaclab' --exclude 'logs' --exclude 'Datasets_ur10e_delto' \
    "$REPO/" "${A[2]}:/workspace/UWLab/" || { echo "FAILED: code rsync"; exit 1; }
  echo "rsyncing datasets (additive, never --delete) ..."
  rsync -az --ignore-existing -e "ssh -i $SSHKEY -o StrictHostKeyChecking=accept-new -p ${A[1]}" \
    "$REPO/Datasets_ur10e_delto" "${A[2]}:/workspace/UWLab/" || { echo "FAILED: dataset rsync"; exit 1; }
  ssh -i "$SSHKEY" -o StrictHostKeyChecking=accept-new -p "${A[1]}" "${A[2]}" \
    'touch /workspace/UWLab/.repo_ready && ls /workspace/UWLab'
  ;;

pull)
  # Bring remote-produced banks home. Newer-only, so a longer local bank is never replaced by a
  # shorter remote one (or vice versa) just because of transfer order.
  read -r -a A <<< "$(ssh_args)"
  rsync -az --update -e "ssh -i $SSHKEY -o StrictHostKeyChecking=accept-new -p ${A[1]}" \
    "${A[2]}:/workspace/UWLab/Datasets_ur10e_delto/" "$REPO/Datasets_ur10e_delto/"
  find "$REPO/Datasets_ur10e_delto" -name '*.pt' -printf '%s %p\n' | sort -rn
  ;;

job)
  jobfile=${2:?usage: launch.sh job <local_job.sh>}
  name=$(basename "$jobfile" .sh)
  read -r -a A <<< "$(ssh_args)"
  # REFUSE TO DOUBLE-QUEUE. Two things queued the same job on b5 -- an armed autostart and a
  # seeding script -- and the autostart won the race, starting the run BEFORE the seed checkpoint
  # was uploaded. The job then trained its first stage from scratch instead of resuming, and said so
  # only in a log line nobody was reading. It cost 33 iterations before it was noticed.
  if [ "${FORCE:-0}" != "1" ]; then
    busy=$(ssh -i "$SSHKEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p "${A[1]}" \
      "${A[2]}" "echo \$(pgrep -cf '[t]rain.py') \$(ls /workspace/jobs/*.sh 2>/dev/null | wc -l)" 2>/dev/null)
    if [ -n "${busy:-}" ] && [ "$(echo "$busy" | awk '{print $1+$2}')" -gt 0 ] 2>/dev/null; then
      echo "REFUSING to queue $name: this box already has a running or queued job ($busy)."
      echo "  Use '$0 stop' first, or FORCE=1 to queue anyway."
      exit 1
    fi
  fi

  # The success line used to print unconditionally. A job that never reached the box then looked
  # exactly like one that did, and the box sat idle and billing until someone read a log.
  if ! scp -i "$SSHKEY" -o StrictHostKeyChecking=accept-new -P "${A[1]}" \
      "$jobfile" "${A[2]}:/workspace/jobs/$name.sh"; then
    echo "FAILED to queue $name: scp did not complete"
    exit 1
  fi
  echo "queued $name — it will start within 15s, owned by init, and survive ssh close"
  ;;

log)
  name=${2:?usage: launch.sh log <name>}
  read -r -a A <<< "$(ssh_args)"
  ssh -i "$SSHKEY" -o StrictHostKeyChecking=accept-new -p "${A[1]}" "${A[2]}" \
    "tail -n 60 /workspace/logs/$name.log"
  ;;

status)
  if [ -f "$IDFILE" ]; then
    "$VAST" show instance "$(cat "$IDFILE")" --raw 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
for k in ['id','actual_status','cur_state','gpu_name','num_gpus','dph_total','ssh_host','ssh_port','duration']:
    if k in d: print(f'{k:16} {d[k]}')
"
  fi
  echo "--- all instances (anything listed here is BILLING) ---"
  "$VAST" show instances
  ;;

stop)
  # Stop whatever is running WITHOUT destroying the box. Kills the job wrapper first: these job
  # scripts run several training stages in sequence, so killing only train.py makes the wrapper
  # advance to the next stage -- and if the checkpoints were cleared at the same time, that stage
  # starts from scratch while reporting that it resumed. Patterns are bracketed because the ssh
  # command line itself contains them and an unbracketed pkill -f kills the ssh session.
  read -r -a A <<< "$(ssh_args)"
  ssh -i "$SSHKEY" -o StrictHostKeyChecking=accept-new -p "${A[1]}" "${A[2]}" \
    "pkill -9 -f '[j]ob_' ; pkill -9 -f '[t]rain.py' ; sleep 4 ; \
     rm -f /workspace/jobs/*.sh /workspace/done/*.sh ; \
     echo 'remaining train.py:' \$(pgrep -cf '[t]rain.py')"
  ;;

destroy)
  id=$(cat "$IDFILE")
  echo "destroying instance $id ..."
  "$VAST" destroy instance "$id" && rm -f "$IDFILE"
  "$VAST" show instances
  ;;

*) sed -n '2,12p' "$0" ;;
esac
