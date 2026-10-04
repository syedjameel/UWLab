#!/usr/bin/env bash
# Rent / provision / drive ONE Isaac Sim 6 box on the SINEW account (Rubitek / UWLab work).
#
#   ./launch_is6.sh create <offer_id> [disk_gb]   rent and provision (STARTS BILLING)
#   ./launch_is6.sh push                          rsync worktree + _isaaclab3 + compat + datasets
#   ./launch_is6.sh job <local_job.sh>            queue a job (runs under container init)
#   ./launch_is6.sh log <name>                    tail a job log
#   ./launch_is6.sh pull <remote> <local>         rsync results back
#   ./launch_is6.sh status                        instance + provisioning state
#   ./launch_is6.sh destroy                       DESTROY the instance (BILLING STOPS)
#
# Cost discipline: sinew credit is shared with other Rubitek work and already carries unrelated
# RTX 5090s. NEVER destroy an instance this script did not create -- only the id in $IDFILE.
set -uo pipefail
KEYFILE=${KEYFILE:-$HOME/.config/vastai/keys/sinew}
VASTBIN=${VASTBIN:-$HOME/.local/bin/vastai}
SSHKEY=${SSHKEY:-$HOME/.ssh/id_ed25519}
HERE=/home/mattephi/Work/Rubitek/cube-stack
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
ISAACLAB=/home/mattephi/Work/Rubitek/_isaaclab3
COMPAT=/home/mattephi/Work/Rubitek/uwlab-env/compat
IDFILE=$HERE/vast/instance_id_is6
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv-is6/bin/python
vast(){ "$VASTBIN" "$@" --api-key "$(cat "$KEYFILE")"; }
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o BatchMode=yes"

endpoint(){  # prints "PORT root@IP"
  local id; id=$(cat "$IDFILE" 2>/dev/null) || return 1
  # direct_port_start can be -1 even on a --direct instance; the real mapping is in ports['22/tcp'].
  # Falls back to the ssh proxy host/port when no direct mapping exists.
  vast show instance "$id" --raw 2>/dev/null | "$PY" -c "
import sys, json
d = json.load(sys.stdin)
maps = (d.get('ports') or {}).get('22/tcp') or []
port = maps[0].get('HostPort') if maps else None
if port and d.get('public_ipaddr'):
    print(port, 'root@' + str(d['public_ipaddr']))
else:
    print(d.get('ssh_port'), 'root@' + str(d.get('ssh_host')))"
}

case "${1:-}" in
create)
  offer=${2:?usage: launch_is6.sh create <offer_id> [disk_gb]}
  disk=${3:-200}
  [ -f "$IDFILE" ] && { echo "REFUSING: $IDFILE exists (instance $(cat "$IDFILE")). destroy it first."; exit 1; }
  tmp=$(mktemp /tmp/onstart_is6.XXXX.sh)
  sed "s|__SSH_PUBKEY__|$(cat "$SSHKEY.pub")|" "$HERE/vast/onstart_is6.sh" > "$tmp"
  vast create instance "$offer" \
    --image nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 \
    --disk "$disk" --ssh --direct \
    --onstart "$tmp" --raw | tee "$HERE/vast/create_is6.json"
  "$PY" -c "
import json
d = json.load(open('$HERE/vast/create_is6.json'))
open('$IDFILE', 'w').write(str(d['new_contract']))
print('instance', d['new_contract'])"
  rm -f "$tmp"
  echo "created. provisioning takes ~20-30 min; watch with: $0 status"
  ;;
push)
  read -r P H <<<"$(endpoint)"
  echo "pushing to $H:$P"
  rsync -az --delete --exclude .git --exclude logs --exclude out --exclude 'Datasets_*' \
    -e "ssh -i $SSHKEY $SSH_OPTS -p $P" "$REPO/" "$H:/workspace/UWLab/" || exit 1
  rsync -az --exclude .git -e "ssh -i $SSHKEY $SSH_OPTS -p $P" "$ISAACLAB/" "$H:/workspace/_isaaclab3/" || exit 1
  rsync -az -e "ssh -i $SSHKEY $SSH_OPTS -p $P" "$COMPAT/uwlab_is6_bridge.py" "$H:/workspace/UWLab/" || exit 1
  rsync -az -e "ssh -i $SSHKEY $SSH_OPTS -p $P" "$REPO/Datasets_ur10e_delto" "$H:/workspace/UWLab/" || exit 1
  # Checkpoints to resume from: the code rsync excludes logs/, so the staged checkpoint goes here.
  # job_train_curriculum.sh skips stage A when /workspace/ck/model_*.pt exists.
  if ls "$HERE"/train/ck_push/model_*.pt >/dev/null 2>&1; then
    ssh -i "$SSHKEY" $SSH_OPTS -p "$P" "$H" "mkdir -p /workspace/ck"
    rsync -az -e "ssh -i $SSHKEY $SSH_OPTS -p $P" "$HERE"/train/ck_push/model_*.pt "$H:/workspace/ck/" || exit 1
  fi
  ssh -i "$SSHKEY" $SSH_OPTS -p "$P" "$H" "touch /workspace/UWLab/.repo_ready /workspace/_isaaclab3/.repo_ready"
  echo "push done"
  ;;
job)
  f=${2:?usage: launch_is6.sh job <local_job.sh>}
  read -r P H <<<"$(endpoint)"
  scp -i "$SSHKEY" $SSH_OPTS -P "$P" "$f" "$H:/workspace/jobs/$(basename "$f")" && echo "queued $(basename "$f")"
  ;;
log)
  name=${2:?usage: launch_is6.sh log <name>}
  read -r P H <<<"$(endpoint)"
  ssh -i "$SSHKEY" $SSH_OPTS -p "$P" "$H" "tail -n 60 /workspace/logs/$name.log"
  ;;
pull)
  remote=${2:?usage: launch_is6.sh pull <remote_path> <local_path>}; local_p=${3:?}
  read -r P H <<<"$(endpoint)"
  rsync -az -e "ssh -i $SSHKEY $SSH_OPTS -p $P" "$H:$remote" "$local_p"
  ;;
status)
  id=$(cat "$IDFILE" 2>/dev/null) || { echo "no instance"; exit 0; }
  vast show instance "$id" --raw | "$PY" -c "
import sys, json
d = json.load(sys.stdin)
print('id', d.get('id'), d.get('gpu_name'), d.get('actual_status'), '\$/h', round(d.get('dph_total', 0), 3),
      'up_h', round((d.get('duration') or 0) / 3600, 2), 'ssh', str(d.get('public_ipaddr')) + ':' + str(d.get('direct_port_start')))"
  read -r P H <<<"$(endpoint)"
  ssh -i "$SSHKEY" $SSH_OPTS -p "$P" "$H" \
    "ls /workspace/READY /workspace/PROVISION_FAILED 2>/dev/null; tail -3 /workspace/provision.log 2>/dev/null" 2>/dev/null || echo "(ssh not up yet)"
  ;;
destroy)
  id=$(cat "$IDFILE" 2>/dev/null) || { echo "no instance"; exit 0; }
  vast destroy instance "$id" && rm -f "$IDFILE" && echo "destroyed $id (billing stopped)"
  ;;
*) sed -n 2,12p "$0" ;;
esac
