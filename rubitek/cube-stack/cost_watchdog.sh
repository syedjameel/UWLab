#!/usr/bin/env bash
# Keep every rented box either WORKING or GONE, and stop everything before the credit runs out.
#
# WHY. A vast.ai box bills by the second whether or not anything is running on it, and the failure
# modes here all end the same way: a job crashes and the box sits idle, an autostart script dies
# mid-wait, a provisioning step fails on an old driver. That has already happened twice today -- a
# box idled through a broken autostart, and another idled after its job hit an OOM. Neither showed
# up as an error anywhere; they showed up as spend.
#
# So this checks two things every few minutes:
#
#   IDLE   box reachable, no train.py running, no job queued. First occurrence tries to requeue the
#          box's own last job. If it is still idle after IDLE_KILL_MIN it is DESTROYED, because an
#          idle box is pure loss and there is no state on it worth keeping -- checkpoints are pulled
#          by probe_runs.py continuously and the code is a push away.
#
#   CREDIT below CREDIT_WARN, notify. Below CREDIT_STOP, destroy everything. The stop threshold is
#          not zero: vast bills continuously and a run that outlives the credit gets the instance
#          reclaimed mid-write, which is a worse way to lose the same money.
#
# Destroying is irreversible, so every destroy is logged with its reason and pushed to the phone.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
KEY=/home/mattephi/Work/IRiS/ip/meaningful-latents/.secrets/id_ed25519_ml_vast
VAST=/home/mattephi/Work/IRiS/ip/meaningful-latents/scripts/vast
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes"

POLL=${POLL:-240}
IDLE_KILL_MIN=${IDLE_KILL_MIN:-45}      # minutes idle before a box is destroyed
CREDIT_WARN=${CREDIT_WARN:-15}
CREDIT_STOP=${CREDIT_STOP:-6}

notify () { printf '%s\n' "$*"; }

boxes () { ls "$CS"/vast/instance_id_* 2>/dev/null | sed 's/.*instance_id_//'; }

endpoint () {  # endpoint <box> -> "PORT IP" or empty
  local id
  id=$(cat "$CS/vast/instance_id_$1" 2>/dev/null) || return 0
  [ -z "$id" ] && return 0
  "$VAST" show instance "$id" --raw 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    raise SystemExit
p = d.get('direct_port_start')
if p and int(p) > 0 and d.get('public_ipaddr'):
    print(p, d['public_ipaddr'])
"
}

credit () {
  "$VAST" show user --raw 2>/dev/null | python3 -c "
import sys, json
try:
    print('%.2f' % json.load(sys.stdin).get('credit', 0))
except Exception:
    print('')
"
}

declare -A IDLE_SINCE
declare -A REQUEUED

echo "=== cost watchdog armed $(date +%H:%M) ==="
echo "    idle box destroyed after ${IDLE_KILL_MIN} min; credit warn \$${CREDIT_WARN}, stop \$${CREDIT_STOP}"

while true; do
  now=$(date +%s)

  # ---- credit ---------------------------------------------------------------------------
  c=$(credit)
  if [ -n "$c" ]; then
    if awk -v c="$c" -v s="$CREDIT_STOP" 'BEGIN{exit !(c < s)}'; then
      notify "CREDIT \$$c BELOW \$$CREDIT_STOP -- destroying every instance now"
      for b in $(boxes); do
        id=$(cat "$CS/vast/instance_id_$b")
        echo y | "$VAST" destroy instance "$id" >/dev/null 2>&1
        rm -f "$CS/vast/instance_id_$b"
        notify "  destroyed $b ($id) on credit floor"
      done
      "$VAST" show instances
      notify "=== watchdog stopped: credit exhausted ==="
      exit 0
    fi
    if awk -v c="$c" -v w="$CREDIT_WARN" 'BEGIN{exit !(c < w)}'; then
      notify "CREDIT LOW: \$$c (warn at \$$CREDIT_WARN, auto-stop at \$$CREDIT_STOP)"
    fi
  fi

  # ---- per-box idleness -----------------------------------------------------------------
  for b in $(boxes); do
    read -r PORT IP <<< "$(endpoint "$b")"
    [ -z "${PORT:-}" ] && continue
    state=$(timeout 40 ssh -i "$KEY" $SSH_OPTS -p "$PORT" "root@$IP" \
      "echo \$(pgrep -cf '[t]rain.py') \$(ls /workspace/jobs/*.sh 2>/dev/null | wc -l)" 2>/dev/null) || continue
    procs=$(awk '{print $1}' <<<"$state"); queued=$(awk '{print $2}' <<<"$state")
    procs=${procs:-0}; queued=${queued:-0}

    if [ "$procs" -gt 0 ] 2>/dev/null || [ "$queued" -gt 0 ] 2>/dev/null; then
      unset 'IDLE_SINCE[$b]'
      unset 'REQUEUED[$b]'
      continue
    fi

    : "${IDLE_SINCE[$b]:=$now}"
    idle_min=$(( (now - IDLE_SINCE[$b]) / 60 ))

    # One requeue attempt: most idleness is a crashed job, and the box's own last job is the right
    # thing to run again. `done/` holds it because the job loop moves it there before running.
    if [ -z "${REQUEUED[$b]:-}" ] && [ "$idle_min" -ge 5 ]; then
      REQUEUED[$b]=1
      last=$(timeout 30 ssh -i "$KEY" $SSH_OPTS -p "$PORT" "root@$IP" \
        "ls -t /workspace/done/*.sh 2>/dev/null | head -1" 2>/dev/null)
      if [ -n "${last:-}" ]; then
        notify "$b idle ${idle_min} min -- requeuing $(basename "$last")"
        timeout 30 ssh -i "$KEY" $SSH_OPTS -p "$PORT" "root@$IP" \
          "cp '$last' /workspace/jobs/" 2>/dev/null && continue
      else
        notify "$b idle ${idle_min} min and has no previous job to requeue"
      fi
    fi

    if [ "$idle_min" -ge "$IDLE_KILL_MIN" ]; then
      id=$(cat "$CS/vast/instance_id_$b")
      notify "$b IDLE ${idle_min} min after a requeue attempt -- DESTROYING $id"
      echo y | "$VAST" destroy instance "$id" >/dev/null 2>&1
      rm -f "$CS/vast/instance_id_$b"
      unset 'IDLE_SINCE[$b]'
    elif [ "$idle_min" -ge 10 ]; then
      notify "$b idle ${idle_min} min (destroy at ${IDLE_KILL_MIN})"
    fi
  done

  sleep "$POLL"
done
