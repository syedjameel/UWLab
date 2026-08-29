#!/usr/bin/env bash
# Periodic offline->cloud wandb sync for the v2 campaign (RESET_SPEC_V2.md sec 5: "offline logging +
# periodic sync because the H100 link is flaky"; V2_REPOSE_RECIPE.md sec 8).
#
# STAGED, NOT LAUNCHED.
#
# Run alongside a ramp stage, e.g.:
#   nohup bash scripts_v2/tools/wandb_sync_loop.sh ~/dexreset_v2/ramp_R1/wandb 1800 > ~/wandb_sync.log 2>&1 &
#
# WHY A LOOP AND NOT `WANDB_MODE=online`. The link drops. An online run that loses the link can
# block or lose steps at the moment it matters, and the moment it matters is a 40-hour job whose
# abort rules are read off these very series. Offline writes to local disk unconditionally; this
# loop ships whatever has accumulated and does not care whether any individual attempt fails.
#
# A FAILED SYNC MUST NEVER KILL TRAINING, which is why this is a separate process rather than a
# hook inside train.py, and why every failure here is logged and swallowed.
set -uo pipefail

DIR=${1:?usage: wandb_sync_loop.sh <wandb dir> [interval seconds]}
INTERVAL=${2:-1800}

# Fallback checked directly on DL_H100, not assumed: no `wandb` on PATH in a non-interactive shell,
# and $HOME/UWLab/env_uwlab (the old fallback) does not exist there at all -- venv_uwlab is the real
# venv (confirmed: wandb 0.28.2 installed).
WANDB_BIN=${WANDB_BIN:-$(command -v wandb || echo "$HOME/venv_uwlab/bin/wandb")}

[ -x "$WANDB_BIN" ] || { echo "REFUSING: wandb binary not found (set WANDB_BIN)"; exit 1; }
[ -d "$DIR" ] || { echo "REFUSING: $DIR does not exist -- start the training job first"; exit 1; }
case "$INTERVAL" in ''|*[!0-9]*) echo "REFUSING: interval must be an integer number of seconds"; exit 1 ;; esac

echo "[wandb-sync] dir=$DIR interval=${INTERVAL}s bin=$WANDB_BIN"
echo "[wandb-sync] NO WEIGHTS are uploaded: nothing calls wandb.watch and no .pth is logged as an"
echo "[wandb-sync] artifact. This ships the offline run directories rl_games already wrote."

while true; do
  # --sync-all ships every offline-run directory that has not been shipped yet, and is idempotent:
  # an already-synced run is skipped, so re-running on a schedule cannot duplicate anything.
  if "$WANDB_BIN" sync --sync-all "$DIR" 2>&1 | sed 's/^/[wandb-sync] /'; then
    echo "[wandb-sync] $(date -Is) ok"
  else
    # Swallowed on purpose. The link being down is the expected case this script exists for; the
    # next pass ships whatever accumulated. Logged so a permanently broken credential is still
    # visible rather than silently retried for 40 hours.
    echo "[wandb-sync] $(date -Is) sync FAILED (link down or credentials) -- will retry in ${INTERVAL}s"
  fi
  sleep "$INTERVAL"
done
