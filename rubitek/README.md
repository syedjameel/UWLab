# Rubitek cube-stack work (UR10e + DELTO, OmniReset)

- `cube-stack/` — experiment driver: training/eval/render/harvest scripts, vast.ai job scripts, run configs and metrics (`linear/`, `pinch/`, `pinch2f/`, `train/`; images, videos, logs and checkpoints are not committed), `notes/`, `report/`. Checkpoints/results archive: easystore `Private/Checkpoints/Rubitek/cube-stack/results`.
- `isaaclab-patches/` — the Isaac Sim 6 port of IsaacLab used by this branch (apply with `git am`, see its README).
- `env/` — local environment setup (`setup_local_env.sh` builds the venvs; `ilx.sh`, `compat/`).
Scripts reference an SSH key path on the original workstation (`.secrets/id_ed25519_ml_vast`); the key itself is not in the repo.
