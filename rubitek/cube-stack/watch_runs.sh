#!/usr/bin/env bash
# Emit a line only when something changed that would change what I do next.
#
# The baselines to beat are already measured, so a poll that reprinted the current rate every few
# minutes would be almost entirely noise:
#   near-goal, flat reward   ~0.20 end-of-episode success, reached by iteration 60 on two seeds
#   four-family mixture      no measurable learning under the old entropy_coef
# This emits on a new best (by a margin wider than the metric's own iteration-to-iteration noise),
# on a run stopping, and on a probe that stops reporting a box at all. Silence therefore means
# "running, nothing has moved", which is a real and readable state rather than an absence of news.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv/bin/python

declare -A best seen_iter missing stalls
while true; do
  out=$(timeout 200 "$PY" "$CS/probe_runs.py" 2>/dev/null)
  for box in b1 b3 b4 b5; do
    line=$(grep "^$box " <<<"$out" || true)
    if [ -z "$line" ]; then
      missing[$box]=$(( ${missing[$box]:-0} + 1 ))
      # Three consecutive misses, not one: a single ssh timeout is routine on these boxes.
      [ "${missing[$box]}" = "3" ] && echo "$box UNREACHABLE for 3 consecutive probes -- check it"
      continue
    fi
    missing[$box]=0
    it=$(grep -oE 'iter=[0-9]+' <<<"$line" | cut -d= -f2)
    r25=$(grep -oE 'recent25=[0-9.]+' <<<"$line" | cut -d= -f2)
    bst=$(grep -oE 'best=[0-9.]+' <<<"$line" | cut -d= -f2)
    it=${it:-0}; r25=${r25:-0}; bst=${bst:-0}

    # Stall detection: the iteration counter must advance between probes.
    if [ "${seen_iter[$box]:-}" = "$it" ] && [ "$it" != "0" ]; then
      stalls[$box]=$(( ${stalls[$box]:-0} + 1 ))
      [ "${stalls[$box]:-0}" = "2" ] && echo "$box STALLED at iteration $it across two probes"
    else
      stalls[$box]=0
    fi
    seen_iter[$box]=$it

    # recent25 rather than the last point: this metric swings several points iteration to iteration,
    # so a threshold on the raw value would fire on noise.
    if awk -v r="$r25" -v b="${best[$box]:-0}" 'BEGIN{exit !(r > b + 0.04)}'; then
      echo "$box iter $it: 25-iteration mean success ${r25} (best point ${bst}), was ${best[$box]:-0}"
      best[$box]=$r25
    fi
  done
  sleep 400
done
