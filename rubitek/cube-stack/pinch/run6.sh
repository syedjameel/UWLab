#!/usr/bin/env bash
# Runner for the Isaac Sim 6.0.1 / IsaacLab 3.0-beta2 env.
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export TMPDIR=/home/mattephi/Work/Rubitek/uwlab-env/tmp
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
exec /home/mattephi/Work/Rubitek/uwlab-env/.venv-is6/bin/python "$@"
