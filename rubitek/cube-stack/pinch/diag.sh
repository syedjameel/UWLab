#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"; P=$PWD
./run6.sh "$P/pinchlab.py" --grid ctrl --postures "$P/out/jawopt30.json" \
  --only '{"ctrl":"direct"}' --out "$P/out/diag.json" --label diag > out/diag.log 2>&1
echo "rc=$?"
