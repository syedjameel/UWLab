#!/usr/bin/env bash
# IsaacLab 3.0.0-beta2 editable install into the Isaac Sim 6 venv.
# Bypasses ./isaaclab.sh because that script hardcodes apt-get for system deps
# (this is Arch) and needs sudo for packages we already have.
V6=/home/mattephi/Work/Rubitek/uwlab-env/.venv-is6
IL3=/home/mattephi/Work/Rubitek/_isaaclab3
export OMNI_KIT_ACCEPT_EULA=YES
export TMPDIR=/home/mattephi/Work/Rubitek/uwlab-env/tmp
export PIP_CACHE_DIR=/home/mattephi/Work/Rubitek/uwlab-env/pipcache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

PKGS="isaaclab isaaclab_ppisp isaaclab_assets isaaclab_physx isaaclab_rl isaaclab_tasks"

for p in $PKGS; do
  echo "=== INSTALL $p ==="
  "$V6/bin/python" -m pip install --no-build-isolation -e "$IL3/source/$p" \
      --extra-index-url https://pypi.nvidia.com --retries 6 --timeout 120 --progress-bar off
  echo "RC $p = $?"
done
echo "IL3_INSTALL_DONE"
