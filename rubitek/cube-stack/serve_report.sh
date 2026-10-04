#!/usr/bin/env bash
# Serve the report over the tailnet only. See serve_report.py for why it is not `http.server`.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
PORT=${PORT:-8899}
IP=$(tailscale ip -4 2>/dev/null | head -1)
[ -z "$IP" ] && { echo "tailscale is not up -- no address to bind to"; exit 1; }

pkill -f "[s]erve_report.py" 2>/dev/null
pkill -f "[h]ttp.server $PORT" 2>/dev/null
sleep 1

nohup python3 "$CS/serve_report.py" "$PORT" > /tmp/report_server.log 2>&1 &
sleep 1
echo "  http://$IP:$PORT/"
echo "  http://$(hostname):$PORT/    (tailnet MagicDNS)"
curl -sS -o /dev/null -w "  self-check: HTTP %{http_code}\n" "http://$IP:$PORT/" || echo "  self-check FAILED"
