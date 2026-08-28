#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Generate a manifest of every runtime asset the pinned commit needs that git does NOT carry.

Why this exists: the pinned commit is reproducible *code* but not a reproducible *artifact*.
``.gitignore`` blocks ``**/*.usd(a|c|z)`` (comment: "No USD files allowed in the repo") except an
explicit 11-file allow-list, so a fresh clone of this repo is missing every USD asset outside that
allow-list. Those assets fall into three classes, in increasing order of "does git know this file
exists at all":

  CLASS 1 -- gitignored USD geometry (``git status --ignored`` shows it, but it is never committed).
  CLASS 2 -- ``metadata.yaml`` siblings of class-1 assets. NOT gitignored (only ``*.usd*`` is), but
             several of these directories are themselves untracked, so a fresh clone is missing them
             too. These files carry ``assembled_offset`` -- the geometry that DEFINES the task_3
             success predicate (``omnireset/mdp/commands.py``) -- so getting this class wrong is the
             single most consequential failure mode of the three.
  CLASS 3 -- reset-bank ``.pt`` files, loaded at runtime from ``UWLAB_CLOUD_ASSETS_DIR`` (or a
             ``dataset_dir`` override) via ``compute_pair_dir(insertive_usd_path, receptive_usd_path)``.
             These are generated artifacts, not fixed repo-relative assets, so this script documents
             the SELECTION MECHANISM rather than enumerating a canonical file set -- see the
             ``describe_class3`` docstring for why a fixed list would be misleading.

Usage:
    # Hash the real bytes on a host that actually has the assets checked out (a fresh clone does
    # NOT have them -- that is the whole point of this manifest). Source-tree greps (for selector
    # verification) always run against --repo-root; asset hashing runs against --assets-root, which
    # defaults to --repo-root but can point at a different checkout (e.g. the remote host that has
    # the 42 assets patched in).
    python build_asset_manifest.py --repo-root /path/to/uwlab --assets-root /path/to/populated/uwlab

Every claim this script prints about *which env var or code path selects an asset* is verified by
grepping --repo-root's actual source at run time (see ``verify``); a selector this script cannot
substantiate against the current source is printed as UNKNOWN rather than invented.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass, field


@dataclass
class Asset:
    path: str  # repo-relative path
    klass: int  # 1 or 2
    origin: str  # "UW Lab" | "FurnitureBench (third-party)" | "DELTO (third-party)" | ...
    selector: str  # human-readable description of what picks this file at runtime
    verify: list[tuple[str, str]] = field(default_factory=list)  # (repo-relative file, regex)
    note: str = ""


# ============================================================================================
# CLASS 1 -- gitignored USD geometry. Every path below was confirmed present via
# `git status --porcelain --ignored=matching -- source` against the populated remote checkout
# (~/github.com/orel/UWLab_v2 on DL_H100) on 2026-08-29.
# ============================================================================================

_FB = "FurnitureBench (third-party, per path -- licence not verified by this script)"
_DELTO = "DELTO / UW Lab graft (see uwlab_assets/robots/ur5e_delto/ur5e_delto.py for provenance)"

CLASS1: list[Asset] = [
    # -- OneLegInsertionFixture (receptive object) --------------------------------------------
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/one_leg_insertion_fixture.usd",
        1, _FB,
        "Hardcoded literal: UWLAB_ASSETS_DATA_DIR + '/Props/FurnitureBench/OneLegInsertionFixture/"
        "one_leg_insertion_fixture.usd'. No env-var override exists for this asset.",
        [("source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/reset_states_cfg.py",
          r"OneLegInsertionFixture/one_leg_insertion_fixture\.usd")],
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/configuration/one_leg_insertion_fixture_base.usd",
        1, _FB, "UsdConverter-authored companion layer of one_leg_insertion_fixture.usd (referenced by internal USD "
        "sublayer/reference composition, not a separate Python selector).",
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/configuration/one_leg_insertion_fixture_physics.usd",
        1, _FB, "UsdConverter-authored companion layer (physics/collision) of one_leg_insertion_fixture.usd.",
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/configuration/one_leg_insertion_fixture_robot.usd",
        1, _FB, "UsdConverter-authored companion layer (robot/none) of one_leg_insertion_fixture.usd.",
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/configuration/one_leg_insertion_fixture_sensor.usd",
        1, _FB, "UsdConverter-authored companion layer (sensor) of one_leg_insertion_fixture.usd.",
    ),
    # -- SquareTableLeg200mm variants (insertive object) ---------------------------------------
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf/square_table_leg4_200mm.usd",
        1, _FB,
        "DEXLIFT_TABLE_LEG_USD_PATH_OVERRIDE env var (dexlift task family), else this literal is the "
        "hardcoded default in dexlift_ur5e_delto_tableleg_env_cfg.py AND is hardcoded directly (no env "
        "override) in the 4 OmniReset registries: rl_state_cfg.py, reset_states_cfg.py, "
        "grasp_sampling_cfg.py, partial_assemblies_cfg.py. THE SHIPPING LEG since 2026-08-23 -- "
        "SquareTableLeg200mmDecomp below was rejected (56.15% of poses interpenetrated its collider).",
        [("source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/dexlift/dexlift_ur5e_delto_tableleg_env_cfg.py",
          r"DEXLIFT_TABLE_LEG_USD_PATH_OVERRIDE"),
         ("source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/rl_state_cfg.py",
          r"SquareTableLeg200mmSdf/square_table_leg4_200mm\.usd")],
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf/Props/instanceable_meshes.usd",
        1, _FB, "Referenced from square_table_leg4_200mm.usd's own USD composition (sibling Props/ prim), "
        "not a separate Python selector; travels with the file above.",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmDecomp/square_table_leg4_200mm.usd",
        1, _FB,
        "UNKNOWN via any currently-wired selector in source/. Byte-identical top-level USD to the Sdf "
        "variant (same sha256, see manifest); only the Props/instanceable_meshes.usd collision layer "
        "differs (convexDecomposition vs SDF). REJECTED as the shipping leg 2026-08-23 (see the Sdf "
        "entry above) -- kept on disk but not selected by any config found by this script's greps.",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmDecomp/Props/instanceable_meshes.usd",
        1, _FB, "Travels with square_table_leg4_200mm.usd (Decomp) above; same UNKNOWN-selector status.",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf1024/square_table_leg4_200mm.usd",
        1, _FB, "UNKNOWN via any currently-wired selector in source/. Appears to be an SDF-resolution "
        "sweep variant (1024) referenced only in conversion-tool comments, not a live config.",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf1024/Props/instanceable_meshes.usd",
        1, _FB, "Travels with square_table_leg4_200mm.usd (Sdf1024) above; same UNKNOWN-selector status.",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf2048/square_table_leg4_200mm.usd",
        1, _FB, "UNKNOWN via any currently-wired selector in source/. SDF-resolution sweep variant (2048), "
        "same status as Sdf1024.",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf2048/Props/instanceable_meshes.usd",
        1, _FB, "Travels with square_table_leg4_200mm.usd (Sdf2048) above; same UNKNOWN-selector status.",
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/SquareTableLeg200mmThreadSdfHybrid/square_table_leg4_200mm_thread_sdf_hybrid.usd",
        1, _FB, "UNKNOWN via any currently-wired selector in source/. Produced by "
        "scripts_v2/tools/conversions/build_leg_thread_sdf_hybrid_usd.py; not referenced by any live "
        "task config found by this script's greps -- looks like a conversion-tool output artifact, not "
        "a currently-selected training asset.",
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/SquareTableLeg200mmThreadSdfHybrid/configuration/square_table_leg4_200mm_thread_sdf_hybrid_base.usd",
        1, _FB, "UsdConverter companion layer of the ThreadSdfHybrid leg above.",
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/SquareTableLeg200mmThreadSdfHybrid/configuration/square_table_leg4_200mm_thread_sdf_hybrid_physics.usd",
        1, _FB, "UsdConverter companion layer (physics/collision) of the ThreadSdfHybrid leg above.",
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/SquareTableLeg200mmThreadSdfHybrid/configuration/square_table_leg4_200mm_thread_sdf_hybrid_robot.usd",
        1, _FB, "UsdConverter companion layer of the ThreadSdfHybrid leg above.",
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/SquareTableLeg200mmThreadSdfHybrid/configuration/square_table_leg4_200mm_thread_sdf_hybrid_sensor.usd",
        1, _FB, "UsdConverter companion layer of the ThreadSdfHybrid leg above.",
    ),
    # -- UR5e+DELTO hand collider variants -------------------------------------------------------
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Robots/Ur5eDelto/ur5e_delto_hullfix3.usd",
        1, _DELTO,
        "Default for every OmniReset UR5eDelto config, applied by "
        "_apply_ur5e_delto_generation_plant / _assert_ur5e_delto_generation_plant "
        "(ur5e_delto_cfg.py) UNLESS OMNIRESET_UR5EDELTO_LEGACY_PLANT=1 (falls back to the tracked, "
        "in-git ur5e_delto.usd). ALSO the dexlift task family's unconditional hand-collider set "
        "(dexlift_ur5e_delto_env_cfg.py, no toggle). 'every held reset bank and the certified "
        "checkpoint were generated on hullfix3' per that file's own assertion message.",
        [("source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/ur5e_delto_cfg.py",
          r"OMNIRESET_UR5EDELTO_LEGACY_PLANT"),
         ("source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/dexlift/dexlift_ur5e_delto_env_cfg.py",
          r"UR5E_DELTO_HULLFIX3_USD_NAME")],
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Robots/Ur5eDelto/ur5e_delto_hullfix.usd",
        1, _DELTO, "NOT selectable by any current code path: dexlift_ur5e_delto_env_cfg.py's own comment "
        "says this variant was 'deleted rather than kept as options ... leaving them selectable only "
        "widens the space of silently-wrong plants.' File remains on disk (referenced only by the "
        "reauthor_hullfix.py generator and by scripts_v2/tools/*/closure_test.py, a standalone probe).",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Robots/Ur5eDelto/ur5e_delto_hullfix2.usd",
        1, _DELTO, "NOT selectable by any current code path -- same deliberate deletion-from-options as "
        "hullfix.usd above (dexlift_ur5e_delto_env_cfg.py's comment names hullfix2 explicitly).",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Robots/Ur5eDelto/ur5e_delto_convexhull.usd",
        1, _DELTO, "UNKNOWN via any currently-wired selector in source/ -- this script's greps found it "
        "only in a reauthor_colliders.py DST path pointing at a DIFFERENT sibling repo "
        "(UWLab_ur5edelto, out of scope for this manifest) and in asset_authoring/randenv.py's debug "
        "print of a DEXLIFT_CONVEXHULL env var that no production config reads.",
    ),
]

# ============================================================================================
# CLASS 2 -- metadata.yaml. THE consequential class: assembled_offset here defines the task_3
# success predicate (commands.py:127-147). All 6 metadata.yaml files below were confirmed
# byte-identical (same sha256) on the remote checkout -- see the "MUST stay byte-identical to the
# Decomp/Sdf sibling copy" comments inside the files themselves.
# ============================================================================================

CLASS2: list[Asset] = [
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/metadata.yaml",
        2, _FB, "Read at runtime by utils.read_metadata_from_usd_directory(receptive_asset usd_path) "
        "in TaskCommand.__init__ (commands.py) and _MatingFrameGeometry.__init__ "
        "(generate_reset_states_policy.py). assembled_offset: pos=[-0.056250, 0.056250, -0.009374], "
        "quat=[1, 0, 0, 0] (WXYZ).",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmDecomp/metadata.yaml",
        2, _FB, "Read at runtime for the insertive (leg) asset. assembled_offset: "
        "pos=[-0.106203, 0, 0], quat=[0.70710678, 0, 0.70710678, 0] (WXYZ). Byte-identical to the "
        "Sdf/Sdf1024/Sdf2048/ThreadSdfHybrid copies below (by design, per an in-file comment).",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf/metadata.yaml",
        2, _FB, "Sibling copy of the Decomp metadata.yaml above -- THIS is the one actually read at "
        "runtime for the shipping (Sdf) leg. Byte-identical to it.",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf1024/metadata.yaml",
        2, _FB, "Sibling copy, byte-identical to the Decomp metadata.yaml.",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf2048/metadata.yaml",
        2, _FB, "Sibling copy, byte-identical to the Decomp metadata.yaml.",
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/SquareTableLeg200mmThreadSdfHybrid/metadata.yaml",
        2, _FB, "Sibling copy, byte-identical to the Decomp metadata.yaml.",
    ),
]


def sha256_and_size(path: str) -> tuple[str, int] | None:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def verify_selector(repo_root: str, verify: list[tuple[str, str]]) -> bool:
    """Grep repo_root's actual source for each (file, pattern) pair. True iff ALL match."""
    if not verify:
        return True  # nothing claimed to verify (e.g. deliberate UNKNOWN entries)
    ok = True
    for rel_file, pattern in verify:
        full = os.path.join(repo_root, rel_file)
        try:
            with open(full, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            ok = False
            continue
        if not re.search(pattern, text):
            ok = False
    return ok


def describe_class3() -> str:
    """CLASS 3 has no fixed file list -- unlike class 1/2, reset-bank .pt files are GENERATED
    training artifacts keyed by (insertive_usd, receptive_usd) via compute_pair_dir(), not assets
    pinned to a commit. Enumerating one experiment directory as "the" manifest would misrepresent
    which bank actually backed any given result -- dozens of per-experiment Datasets_*/Resets/
    directories exist on the training host, each valid only for the run that produced it.
    """
    return (
        "CLASS 3 -- reset bank .pt files (mechanism documented, no fixed file list)\n"
        "\n"
        "Loaded by omnireset/mdp/events.py's MultiResetManager and "
        "reset_insertive_object_from_partial_assembly_dataset via "
        "utils.safe_retrieve_file_path(f'{dataset_dir}/Resets/{pair}/{name}.pt'), where "
        "pair = utils.compute_pair_dir(insertive_usd_path, receptive_usd_path) (alphabetically "
        "sorted per-asset directory names joined with '__', e.g. "
        "'OneLegInsertionFixture__SquareTableLeg200mmDecomp').\n"
        "\n"
        "SELECTOR: dataset_dir is an EventTerm param. Its default across the OmniReset registries "
        "(rl_state_cfg.py, reset_states_cfg.py, data_collection_rgb_cfg.py) is "
        "f'{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset' (UWLAB_CLOUD_ASSETS_DIR = "
        "https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main). Several UR5eDelto and "
        "UR10eDelto configs override it per-variant via _apply_delto_dataset_dir / "
        "cfg.events.*.params['dataset_dir'] to a local directory instead. Downloads land in "
        "~/.cache/uwlab/assets/Datasets/OmniReset/Resets/<pair>/ (resolve_cloud_path's cache "
        "convention).\n"
        "\n"
        "AUDITED 2026-08-29 on DL_H100 (~/github.com/orel/UWLab_v2's live cache): the canonical "
        "cloud-dir cache directory for the certified pair, "
        "~/.cache/uwlab/assets/Datasets/OmniReset/Resets/OneLegInsertionFixture__SquareTableLeg200mmDecomp/, "
        "is EMPTY -- no reset bank has been fetched from UWLAB_CLOUD_ASSETS_DIR for this pair on "
        "this host. Actual training runs on this host instead point dataset_dir at one of several "
        "dozen ad hoc ~/Datasets_*/Resets/ directories (per-experiment, per-chunk). This script does "
        "NOT guess which one backed any specific certified checkpoint -- that requires the actual "
        "training launch command's dataset_dir value, not something inferable from the repo alone. "
        "Treat this as an open finding, not a manifest gap this tool can close.\n"
    )


def render_manifest(repo_root: str, assets_root: str) -> str:
    lines: list[str] = []
    lines.append("# Asset Manifest\n")
    lines.append(
        "Generated by `scripts_v2/tools/build_asset_manifest.py` against the pinned commit of "
        "`uwlab` (branch `dexreset/v2-resets`). Every asset below is required at runtime and is "
        "NOT reproducible from a fresh `git clone` -- see that script's module docstring for why.\n"
    )
    lines.append(f"Asset bytes hashed from: `{assets_root}`\n")
    lines.append("Selector claims verified by grepping: `{}`\n".format(repo_root))

    total_bytes = 0
    total_count = 0
    missing: list[str] = []
    unverified: list[str] = []

    for klass, assets, title in ((1, CLASS1, "CLASS 1 -- gitignored USD geometry"),
                                  (2, CLASS2, "CLASS 2 -- asset metadata.yaml (defines assembled_offset)")):
        lines.append(f"\n## {title}\n")
        lines.append("| path | bytes | sha256 | origin | selector |")
        lines.append("|---|---|---|---|---|")
        for a in assets:
            full = os.path.join(assets_root, a.path)
            result = sha256_and_size(full)
            if result is None:
                missing.append(a.path)
                sha, size = "MISSING", "MISSING"
            else:
                sha, size = result
                total_bytes += size
                total_count += 1
            if not verify_selector(repo_root, a.verify):
                unverified.append(a.path)
                sel = f"[UNVERIFIED CLAIM] {a.selector}"
            else:
                sel = a.selector
            lines.append(f"| `{a.path}` | {size} | `{sha}` | {a.origin} | {sel} |")

    lines.append("\n## " + describe_class3().splitlines()[0] + "\n")
    lines.append("```\n" + "\n".join(describe_class3().splitlines()[2:]) + "\n```\n")

    lines.append("\n## Summary\n")
    lines.append(f"- Total assets hashed (class 1 + 2): {total_count}")
    lines.append(f"- Total bytes: {total_bytes}")
    if missing:
        lines.append(f"- MISSING from `{assets_root}` (not hashed): {len(missing)} -- {missing}")
    if unverified:
        lines.append(f"- Selector claims that did NOT verify against current source: {len(unverified)} -- {unverified}")
    lines.append(
        "- Third-party origin note: FurnitureBench- and DELTO-derived assets are flagged by path in "
        "the tables above. This script does not assert a licence for them -- none has been read as "
        "part of generating this manifest."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                     help="uwlab checkout to grep for selector verification (default: this script's own repo).")
    ap.add_argument("--assets-root", default=None,
                     help="Checkout to hash asset bytes from (default: same as --repo-root).")
    ap.add_argument("-o", "--output", default=None, help="Write manifest to this file instead of stdout.")
    args = ap.parse_args()

    assets_root = args.assets_root or args.repo_root
    manifest = render_manifest(args.repo_root, assets_root)

    if args.output:
        with open(args.output, "w") as f:
            f.write(manifest)
    else:
        sys.stdout.write(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
