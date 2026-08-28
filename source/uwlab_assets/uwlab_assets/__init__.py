# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing asset and sensor configurations."""

import logging
import os
import re
import time
import toml
import urllib.request
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Conveniences to other module directories via relative paths
UWLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""
UWLAB_ASSETS_DATA_DIR = os.path.join(UWLAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""
UWLAB_ASSETS_METADATA = toml.load(os.path.join(UWLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

UWLAB_CLOUD_ASSETS_DIR = "https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main"

UWLAB_LOCAL_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "local")
"""Path to in-repo local (dev) assets, resolved relative to this package so it is portable
across machines. Mirrors the layout under ``UWLAB_CLOUD_ASSETS_DIR`` (e.g. ``Props/Custom/...``)."""


def _extract_relative_path(url: str) -> str:
    """Strip the HuggingFace resolve-URL prefix, returning the repo-relative path.

    Example:
        ``https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Props/Custom/Peg/peg.usd``
        -> ``Props/Custom/Peg/peg.usd``
    """
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    try:
        idx = parts.index("resolve")
        return "/".join(parts[idx + 2 :])
    except ValueError:
        return parsed.path.strip("/")


def _urlretrieve_quiet(url: str, dest: str) -> None:
    """Download *url* to *dest* silently."""
    req = urllib.request.urlopen(url)
    chunk_size = 1 << 16  # 64 KiB
    with open(dest, "wb") as f:
        while True:
            chunk = req.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    req.close()


def resolve_cloud_path(path: str) -> str:
    """Resolve a cloud asset path to a local file, downloading if needed.

    * Local paths (including already-cached files) are returned immediately.
    * HTTPS URLs are downloaded once to ``~/.cache/uwlab/assets/<relative>``
      and the local cached path is returned on subsequent calls.
    * Downloads are atomic (write to a temp file, then ``os.rename``).
    """
    if not path.startswith(("http://", "https://")):
        return path

    rel = _extract_relative_path(path)
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "uwlab", "assets")
    local = os.path.join(cache_dir, rel)

    if os.path.isfile(local):
        return local

    os.makedirs(os.path.dirname(local), exist_ok=True)
    tmp = f"{local}.tmp.{os.getpid()}"
    # Retry transient network failures (SSL resets, timeouts): a single hiccup used to
    # abort whole multi-GB batch downloads at file N of ~1000 (seen 2026-07-16: an
    # ssl.SSLError at texture 606/957 killed a 45-minute download and surfaced later as
    # a cryptic manager TypeError -- see UR10E_SIM2REAL_PROCEDURE.md section 10.4).
    attempts = 4
    for attempt in range(attempts):
        try:
            logger.info(f"Downloading {rel} ...")
            _urlretrieve_quiet(path, tmp)
            os.rename(tmp, local)
            break
        except Exception as exc:
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == attempts - 1:
                raise
            delay = 2.0 * 2**attempt  # 2, 4, 8 s
            logger.warning(f"Download failed ({exc!r}), retry {attempt + 1}/{attempts - 1} in {delay:.0f}s: {rel}")
            time.sleep(delay)

    return local


# ============================================================================================
# OmniReset-leg five-literal reproducibility guard.
#
# The SquareTableLeg200mm* leg asset is named by FIVE separate hardcoded literals, in five
# different files, because nothing imports a shared constant for it:
#   rl_state_cfg.py, reset_states_cfg.py, grasp_sampling_cfg.py, partial_assemblies_cfg.py
#   (the four OmniReset registries) and dexlift_ur5e_delto_tableleg_env_cfg.py's TABLE_LEG_USD_PATH.
# They DISAGREED until 2026-08-23 (some pointed at SquareTableLeg200mmDecomp, which is REJECTED as
# an OmniReset collider -- 56.15% pose interpenetration -- while others already pointed at the
# correct SquareTableLeg200mmSdf). While they disagreed, generation could read one asset and
# training read another with nothing in the logs to distinguish "verified on SDF, trained on
# Decomp" from a normal run -- the exact defect class this function exists to make impossible again.
#
# This does not import the five modules (several drag in the full Isaac stack at import time, and a
# given process may only ever import the subset its task family needs); it re-reads their SOURCE
# TEXT directly by path, which is why it lives here in uwlab_assets -- the one lightweight package
# every one of the five already depends on for UWLAB_LOCAL_ASSETS_DIR.
# ============================================================================================

_LEG_LITERAL_SOURCES = [
    (
        "source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/rl_state_cfg.py",
        804,
    ),
    (
        "source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/reset_states_cfg.py",
        617,
    ),
    (
        "source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/grasp_sampling_cfg.py",
        206,
    ),
    (
        "source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/partial_assemblies_cfg.py",
        482,
    ),
    (
        "source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/dexlift/dexlift_ur5e_delto_tableleg_env_cfg.py",
        48,
    ),
]
_LEG_LITERAL_PATTERN = re.compile(r"Props/FurnitureBench/(SquareTableLeg200mm\w*)/square_table_leg4_200mm\.usd")

# Repo root: this file is source/uwlab_assets/uwlab_assets/__init__.py, so two levels above
# UWLAB_ASSETS_EXT_DIR (source/uwlab_assets) is the repo root (uwlab/).
_REPO_ROOT = os.path.abspath(os.path.join(UWLAB_ASSETS_EXT_DIR, "..", ".."))


def describe_leg_literals(repo_root: str = _REPO_ROOT) -> list[tuple[str, int, str]]:
    """Read-only: returns [(rel_path, line, variant_name_or_error), ...] for all five sources,
    without raising. Used by both the assertion below and external tooling (e.g. the asset
    manifest generator) that wants to report the values without also enforcing them.
    """
    found: list[tuple[str, int, str]] = []
    for rel_path, line in _LEG_LITERAL_SOURCES:
        full_path = os.path.join(repo_root, rel_path)
        try:
            with open(full_path, encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            found.append((rel_path, line, f"UNREADABLE ({exc})"))
            continue
        match = _LEG_LITERAL_PATTERN.search(text)
        found.append((rel_path, line, match.group(1) if match else "NO MATCH FOUND"))
    return found


def assert_omnireset_leg_literals_agree(repo_root: str = _REPO_ROOT) -> None:
    """Fail loudly if the five hardcoded SquareTableLeg200mm* literals do not all name the same
    leg variant. Fatal (raises), never a warning -- see module comment above for why.
    """
    found = describe_leg_literals(repo_root)
    variants = {v for _, _, v in found}
    if len(variants) != 1:
        detail = "\n".join(f"  {rel}:{line} -> {variant}" for rel, line, variant in found)
        raise RuntimeError(
            "OmniReset leg literal MISMATCH: the five files that name the SquareTableLeg200mm* leg "
            f"asset do not all agree (tolerance: none -- these are file paths, not floats):\n{detail}\n"
            "They disagreed until 2026-08-23 (some pointed at the REJECTED SquareTableLeg200mmDecomp "
            "collider -- 56.15% pose interpenetration as an OmniReset collider -- while others already "
            "pointed at SquareTableLeg200mmSdf), which silently let generation read one asset while "
            "training read another. EVERY reset bank, checkpoint, and success number produced while "
            "these five disagree is INVALID -- do not proceed until all five name the same leg "
            "variant. If you repoint one, repoint all five."
        )


# Configure the module-level variables
__version__ = UWLAB_ASSETS_METADATA["package"]["version"]
