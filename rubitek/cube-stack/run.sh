#!/usr/bin/env bash
# Runner for the local Isaac Sim env. Usage: ./run.sh <script.py> [args...]
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
# Without this, Python block-buffers stdout when it is a file rather than a tty, so a script's own
# prints never reach the log while it runs -- which reads exactly like a hang.
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/mattephi/Work/Rubitek/UWLab-cubestack/source/uwlab_assets:/home/mattephi/Work/Rubitek/UWLab-cubestack/source/uwlab_tasks:/home/mattephi/Work/Rubitek/UWLab-cubestack/source/uwlab:/home/mattephi/Work/Rubitek/UWLab-cubestack/source/uwlab_rl${PYTHONPATH:+:$PYTHONPATH}"
cd /home/mattephi/Work/Rubitek/UWLab-cubestack
exec /home/mattephi/Work/Rubitek/uwlab-env/.venv/bin/python "$@"
