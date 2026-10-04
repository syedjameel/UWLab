#!/usr/bin/env bash
# Print one line per poll: where every grip box is, and what Stable-Grasp is scoring.
#
# task_1_success_rate IS the falsifiable prediction of the grip fix. Under the old action term that
# family scored exactly 0.0000 for 200+ iterations on two independent boxes, because the cube was
# out of the hand and on the table within two seconds of every reset. If it stays at zero now, the
# bias is not the answer and something else is wrong; if it moves, the fix is doing what the
# null-action probe says it should.
set -uo pipefail
K=/home/mattephi/Work/IRiS/ip/meaningful-latents/.secrets/id_ed25519_ml_vast
SSHO="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o BatchMode=yes"
POLL=${POLL:-900}
BOXES=/home/mattephi/Work/Rubitek/cube-stack/boxes.txt

best=0
while true; do
  line=""
  while read -r b p i; do
    [ -z "$b" ] && continue
    o=$(timeout 40 ssh -n -i "$K" $SSHO -p "$p" "root@$i" '
      L=$(ls -t /workspace/logs/job_*.log 2>/dev/null | head -1)
      [ -z "$L" ] && { echo "nojob|na|na|$(pgrep -cf "[t]rain.py")"; exit 0; }
      # Strip ANSI escapes BEFORE pulling digits. The trainer prints the iteration inside a bold
      # sequence, so a naive `tr -dc 0-9/` keeps the 1 from ESC[1m and the 0 from ESC[0m and turns
      # "26/120" into "126/1200" -- which reads as a run four times longer than it is.
      # Take the last COMPLETE iteration line. The trainer writes these continuously, so tailing
      # the raw last line catches a half-flushed one -- which parsed to "2" and read as a box that
      # had restarted, twice sending me to ssh in and check a box that was training normally.
      # Requiring the full <n>/<n> shape makes a torn read fall back to the previous good line.
      T=$(tr "\r" "\n" < "$L" | sed -e "s/\x1b\[[0-9;]*m//g" \
          | grep -aoE "Learning iteration [0-9]+/[0-9]+" | tail -1 | sed "s/.*iteration //")
      C3=$(tr "\r" "\n" < "$L" | grep -a "task_1_success_rate" | tail -1 | awk "{print \$NF}")
      C4=$(tr "\r" "\n" < "$L" | grep -a "task_0_success_rate" | tail -1 | awk "{print \$NF}")
      echo "${T:-?}|${C3:-na}|${C4:-na}|$(pgrep -cf "[t]rain.py")"' 2>/dev/null)
    if [ -z "$o" ]; then line="$line  $b:unreachable"; continue; fi
    it=${o%%|*}; rest=${o#*|}
    c3=${rest%%|*}; rest=${rest#*|}
    c4=${rest%%|*}; procs=${rest##*|}
    line="$line  $b $it C3=$c3 C4=$c4"
    [ "${procs:-0}" = "0" ] && line="$line[IDLE]"
    case "$c3" in ''|na|nojob) ;; *) awk -v a="$c3" -v b="$best" 'BEGIN{exit !(a>b)}' && best=$c3 ;; esac
  done < "$BOXES"
  echo "$(date +%H:%M)$line"
  if awk -v b="$best" 'BEGIN{exit !(b>0.01)}'; then
    echo "*** STABLE-GRASP OFF ZERO: best task_1_success_rate = $best ***"
  fi
  sleep "$POLL"
done
