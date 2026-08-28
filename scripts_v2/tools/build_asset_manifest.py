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

Every class-1/2 asset also carries a STATUS, recorded PER (asset, consumer) PAIR rather than per
asset -- see V2_POSE_FINDINGS.md F23 (DexReset repo, not this one). The same bytes can be correct
for one task family and wrong for another: SquareTableLeg200mmDecomp is the leg the CERTIFIED
dexlift checkpoint trained on, and is a REJECTED OmniReset collider (56.15% pose interpenetration)
in the SAME checkout. A manifest that recorded status per asset instead of per (asset, consumer)
would hide exactly that trap.

  LOAD-BEARING              -- reproduction of a specific, named result is impossible without it.
  REJECTED                  -- measured to fail; kept on disk as negative evidence; never a default.
  SUPERSEDED / EXPERIMENTAL -- superseded by a later choice, or never wired to a live selector;
                               kept for the record, not offered as an option.

Also see F24 (DexReset V2_POSE_FINDINGS.md): the OmniReset leg is named by five separate hardcoded
literals across five files, which disagreed until 2026-08-23. ``describe_leg_literals_section``
below reports their CURRENT values live (via ``uwlab_assets.describe_leg_literals``, the same
function the runtime assertion uses) rather than hardcoding them a second time.

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
    origin: str  # provenance note -- see LICENCE_NOTE; never a licence assertion (F25)
    status: str  # STATUS_* below, per (asset, consumer) -- see this file's module docstring
    selector: str  # human-readable description of what picks this file at runtime
    verify: list[tuple[str, str]] = field(default_factory=list)  # (repo-relative file, regex)
    note: str = ""


# Status vocabulary -- exactly these three, per V2_POSE_FINDINGS.md F23 (DexReset repo).
STATUS_LOAD_BEARING = "LOAD-BEARING"
STATUS_REJECTED = "REJECTED"
STATUS_SUPERSEDED = "SUPERSEDED / EXPERIMENTAL"

# ============================================================================================
# CLASS 1 -- gitignored USD geometry. Every path below was confirmed present via
# `git status --porcelain --ignored=matching -- source` against the populated remote checkout
# (~/github.com/orel/UWLab_v2 on DL_H100) on 2026-08-29.
# ============================================================================================

# Licence: do NOT assert one (F25). The leg/fixture sit under Props/FurnitureBench/ (pointing at
# the FurnitureBench benchmark as geometric source); the hand USDs are Tesollo DELTO vendor
# geometry. Neither licence has been verified -- do not infer one from a directory name or from
# this repo's own declared licence.
LICENCE_NOTE = "UNVERIFIED -- third-party, see F25"
_FB = f"FurnitureBench-derived path. {LICENCE_NOTE}."
_DELTO = f"DELTO vendor / UW Lab graft (see uwlab_assets/robots/ur5e_delto/ur5e_delto.py). {LICENCE_NOTE}."

CLASS1: list[Asset] = [
    # -- OneLegInsertionFixture (receptive object; single consumer -- OmniReset) ----------------
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/one_leg_insertion_fixture.usd",
        1, _FB, STATUS_LOAD_BEARING,
        "Hardcoded literal: UWLAB_ASSETS_DATA_DIR + '/Props/FurnitureBench/OneLegInsertionFixture/"
        "one_leg_insertion_fixture.usd'. No env-var override exists for this asset.",
        [("source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/reset_states_cfg.py",
          r"OneLegInsertionFixture/one_leg_insertion_fixture\.usd")],
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/configuration/one_leg_insertion_fixture_base.usd",
        1, _FB, STATUS_LOAD_BEARING,
        "UsdConverter-authored companion layer of one_leg_insertion_fixture.usd (referenced by internal "
        "USD sublayer/reference composition, not a separate Python selector).",
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/configuration/one_leg_insertion_fixture_physics.usd",
        1, _FB, STATUS_LOAD_BEARING,
        "UsdConverter-authored companion layer (physics/collision) of one_leg_insertion_fixture.usd.",
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/configuration/one_leg_insertion_fixture_robot.usd",
        1, _FB, STATUS_LOAD_BEARING,
        "UsdConverter-authored companion layer (robot/none) of one_leg_insertion_fixture.usd.",
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/configuration/one_leg_insertion_fixture_sensor.usd",
        1, _FB, STATUS_LOAD_BEARING,
        "UsdConverter-authored companion layer (sensor) of one_leg_insertion_fixture.usd.",
    ),
    # -- SquareTableLeg200mmSdf (insertive object; LOAD-BEARING for OmniReset) -------------------
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf/square_table_leg4_200mm.usd",
        1, _FB,
        f"{STATUS_LOAD_BEARING} for OmniReset (whole-part SDF at resolution 256; zero-action with the "
        f"robot frozen it holds at 16.564 mm -- F23). Also the current HEAD default for dexlift since "
        f"2026-08-23 -- see the Decomp entry below for why that is NOT the same claim as "
        f"'what the certified dexlift checkpoint used.'",
        "Hardcoded directly (no env override) in the 4 OmniReset registries: rl_state_cfg.py, "
        "reset_states_cfg.py, grasp_sampling_cfg.py, partial_assemblies_cfg.py. Also the "
        "dexlift_ur5e_delto_tableleg_env_cfg.py default (overridable there via "
        "DEXLIFT_TABLE_LEG_USD_PATH_OVERRIDE, dexlift-only, diagnostic escape hatch -- see F24 "
        "section below for why an override there does not propagate to training). See F24 for the "
        "current value of all five literals, read live from source.",
        [("source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/dexlift/dexlift_ur5e_delto_tableleg_env_cfg.py",
          r"DEXLIFT_TABLE_LEG_USD_PATH_OVERRIDE"),
         ("source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/rl_state_cfg.py",
          r"SquareTableLeg200mmSdf/square_table_leg4_200mm\.usd")],
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf/Props/instanceable_meshes.usd",
        1, _FB, STATUS_LOAD_BEARING,
        "Referenced from square_table_leg4_200mm.usd's own USD composition (sibling Props/ prim), "
        "not a separate Python selector; travels with the file above.",
    ),
    # -- SquareTableLeg200mmDecomp -- LOAD-BEARING for dexlift, REJECTED for OmniReset -----------
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmDecomp/square_table_leg4_200mm.usd",
        1, _FB,
        f"{STATUS_LOAD_BEARING} for dexlift / {STATUS_REJECTED} for OmniReset -- SAME asset, opposite "
        f"verdicts by consumer (F23, the exact trap this manifest exists to make impossible). "
        f"dexlift: the CERTIFIED checkpoint's own dumped env.yaml names this leg -- it predates the "
        f"2026-08-23 fix that repointed dexlift's own default to Sdf (see the Sdf entry above), so "
        f"reproducing THAT checkpoint specifically requires Decomp, not today's HEAD default. "
        f"OmniReset: REJECTED as a collider -- 56.15% of poses interpenetrate the collider PhysX "
        f"actually uses (median -0.068 mm); depenetration ejects the leg within five steps even "
        f"frozen, depth collapsing 13.7 -> 2.7 mm (F23).",
        "No LIVE selector points here in HEAD (dexlift's own default moved to Sdf 2026-08-23; see "
        "F24). DEXLIFT_LEG_DECOMP=1 IS exported by every certification launcher for dexlift certs "
        "(cert_ft.sh:35, cert30.sh:7, cert_both.sh:7) and recorded in every historical cert JSON's "
        "plant.dexlift_env -- that part is real, verified provenance, not invented. But it is a "
        "VESTIGIAL export: cert_g3z4_finetune.sh:126-134's own comment states, and this script's "
        "grep across every .py file confirms, there is no os.environ.get/os.environ[] site for "
        "DEXLIFT_LEG_DECOMP anywhere in source/ -- the decomposed-leg USD selection it once gated is "
        "now unconditional (see the Sdf entry's selector). Setting the var still happens; reading it "
        "does not. Byte-identical top-level USD to the Sdf variant (same sha256, see table) -- only "
        "the Props/instanceable_meshes.usd collision layer differs (convexDecomposition vs SDF).",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmDecomp/Props/instanceable_meshes.usd",
        1, _FB, f"{STATUS_LOAD_BEARING} for dexlift (certified checkpoint) / {STATUS_REJECTED} for OmniReset",
        "Travels with square_table_leg4_200mm.usd (Decomp) above; same status.",
    ),
    # -- SquareTableLeg200mmSdf1024 / Sdf2048 -- SUPERSEDED/EXPERIMENTAL -------------------------
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf1024/square_table_leg4_200mm.usd",
        1, _FB, STATUS_SUPERSEDED,
        "Not wired to any live selector (this script's greps confirm). Resolution is NOT a reason to "
        "prefer any of the three SDF resolutions: 256 / 1024 / 2048 settle at 16.564 / 16.594 / "
        "16.633 mm, a 0.07 mm spread (F23). Kept for the record; do not offer as an option.",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf1024/Props/instanceable_meshes.usd",
        1, _FB, STATUS_SUPERSEDED, "Travels with square_table_leg4_200mm.usd (Sdf1024) above.",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf2048/square_table_leg4_200mm.usd",
        1, _FB, STATUS_SUPERSEDED,
        "Not wired to any live selector. Resolution sweep variant (2048) -- see the Sdf1024 entry's "
        "note; same 0.07 mm-spread finding applies.",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf2048/Props/instanceable_meshes.usd",
        1, _FB, STATUS_SUPERSEDED, "Travels with square_table_leg4_200mm.usd (Sdf2048) above.",
    ),
    # -- SquareTableLeg200mmThreadSdfHybrid -- REJECTED, measured to fail ------------------------
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/SquareTableLeg200mmThreadSdfHybrid/square_table_leg4_200mm_thread_sdf_hybrid.usd",
        1, _FB, STATUS_REJECTED,
        "Not wired to any live selector (produced by "
        "scripts_v2/tools/conversions/build_leg_thread_sdf_hybrid_usd.py; conversion-tool output, not "
        "a currently-selected training asset). MEASURED TO FAIL (F23): body hulls overshoot the "
        "body/thread seam by ~4 mm through hull inflation; lateral reaches 26 mm against 0.91 mm "
        "radial clearance and the leg exits the bore sideways. Keep as negative evidence, never a "
        "default.",
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/SquareTableLeg200mmThreadSdfHybrid/configuration/square_table_leg4_200mm_thread_sdf_hybrid_base.usd",
        1, _FB, STATUS_REJECTED, "UsdConverter companion layer of the ThreadSdfHybrid leg above.",
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/SquareTableLeg200mmThreadSdfHybrid/configuration/square_table_leg4_200mm_thread_sdf_hybrid_physics.usd",
        1, _FB, STATUS_REJECTED,
        "UsdConverter companion layer (physics/collision) of the ThreadSdfHybrid leg above.",
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/SquareTableLeg200mmThreadSdfHybrid/configuration/square_table_leg4_200mm_thread_sdf_hybrid_robot.usd",
        1, _FB, STATUS_REJECTED, "UsdConverter companion layer of the ThreadSdfHybrid leg above.",
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/SquareTableLeg200mmThreadSdfHybrid/configuration/square_table_leg4_200mm_thread_sdf_hybrid_sensor.usd",
        1, _FB, STATUS_REJECTED, "UsdConverter companion layer of the ThreadSdfHybrid leg above.",
    ),
    # -- UR5e+DELTO hand collider variants ---------------------------------------------------------
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Robots/Ur5eDelto/ur5e_delto_hullfix3.usd",
        1, _DELTO,
        f"{STATUS_LOAD_BEARING} -- the plant both certified policies were trained and certified in (F23).",
        "Default for every OmniReset UR5eDelto config, applied by "
        "_apply_ur5e_delto_generation_plant / _assert_ur5e_delto_generation_plant "
        "(ur5e_delto_cfg.py) UNLESS OMNIRESET_UR5EDELTO_LEGACY_PLANT=1 (falls back to the tracked, "
        "in-git ur5e_delto.usd). ALSO the dexlift task family's unconditional hand-collider set "
        "(dexlift_ur5e_delto_env_cfg.py, no toggle). DEXLIFT_HULLFIX=1/2/3 IS exported by the "
        "certification launchers (run_certify.sh:26-28, cert_ft.sh, launch_task.sh:41-43) as if it "
        "selected a collider variant -- that export is real, but VESTIGIAL: this script's grep across "
        "every .py file in the tree finds zero os.environ.get/os.environ[] read sites for "
        "DEXLIFT_HULLFIX anywhere (same defect class, and same certification-script author, as "
        "DEXLIFT_LEG_DECOMP -- see the Decomp entry above; unlike that one, no comment in-tree "
        "documents this one as dead, so this manifest is the first record of it). The dexlift plant "
        "is genuinely unconditional hullfix3, confirmed by direct read of "
        "dexlift_ur5e_delto_env_cfg.py:1444-1450 and its own comment 'colliders and self-collisions "
        "are no longer switchable.' Confirmed by the generator's own startup log too: smoke_s1_bmix.log "
        "and smoke_s2band.log (both on DL_H100, the only 2 of 6 v1 generator runs whose "
        "[c4-seating-gate] banner fires -- see F24 log-provenance note below) print "
        "leg_usd=.../SquareTableLeg200mmSdf/square_table_leg4_200mm.usd for both. 'every "
        "held reset bank and the certified checkpoint were generated on hullfix3' per that file's own "
        "assertion message.",
        [("source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/ur5e_delto_cfg.py",
          r"OMNIRESET_UR5EDELTO_LEGACY_PLANT"),
         ("source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/dexlift/dexlift_ur5e_delto_env_cfg.py",
          r"UR5E_DELTO_HULLFIX3_USD_NAME")],
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Robots/Ur5eDelto/ur5e_delto_hullfix.usd",
        1, _DELTO, STATUS_SUPERSEDED,
        "NOT selectable by any current code path: dexlift_ur5e_delto_env_cfg.py's own comment says "
        "this variant was 'deleted rather than kept as options ... leaving them selectable only "
        "widens the space of silently-wrong plants.' Earlier step toward hullfix3 (F23). File "
        "remains on disk (referenced only by the reauthor_hullfix.py generator and by "
        "scripts_v2/tools/*/closure_test.py, a standalone probe).",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Robots/Ur5eDelto/ur5e_delto_hullfix2.usd",
        1, _DELTO, STATUS_SUPERSEDED,
        "NOT selectable by any current code path -- same deliberate deletion-from-options as "
        "hullfix.usd above (dexlift_ur5e_delto_env_cfg.py's comment names hullfix2 explicitly). "
        "Earlier step toward hullfix3 (F23).",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Robots/Ur5eDelto/ur5e_delto_convexhull.usd",
        1, _DELTO,
        f"{STATUS_SUPERSEDED} -- THE CHEAT ASSET (F23). This is the plant in which a 0.9424 run fused "
        f"fingers 3 and 4 through each other. Labelled explicitly here so no headline number is ever "
        f"produced in it.",
        "UNKNOWN via any currently-wired selector in source/ -- this script's greps found it only in "
        "a reauthor_colliders.py DST path pointing at a DIFFERENT sibling repo (UWLab_ur5edelto, out "
        "of scope for this manifest) and in asset_authoring/randenv.py's debug print of a "
        "DEXLIFT_CONVEXHULL env var that no production config reads.",
    ),
]

# ============================================================================================
# CLASS 2 -- metadata.yaml. THE consequential class: assembled_offset here defines the task_3
# success predicate (commands.py:127-147). All 6 metadata.yaml files below were confirmed
# byte-identical (same sha256) on the remote checkout -- see the "MUST stay byte-identical to the
# Decomp/Sdf sibling copy" comments inside the files themselves. Status mirrors the parent USD's
# status per consumer (Decomp's metadata.yaml is read whenever Decomp itself is -- see CLASS1).
# ============================================================================================

CLASS2: list[Asset] = [
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/OneLegInsertionFixture/metadata.yaml",
        2, _FB, STATUS_LOAD_BEARING,
        "Read at runtime by utils.read_metadata_from_usd_directory(receptive_asset usd_path) "
        "in TaskCommand.__init__ (commands.py) and _MatingFrameGeometry.__init__ "
        "(generate_reset_states_policy.py). assembled_offset: pos=[-0.056250, 0.056250, -0.009374], "
        "quat=[1, 0, 0, 0] (WXYZ).",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmDecomp/metadata.yaml",
        2, _FB, f"{STATUS_LOAD_BEARING} for dexlift (certified checkpoint) / {STATUS_REJECTED} for OmniReset",
        "Read at runtime for the insertive (leg) asset, mirroring the Decomp USD's own status. "
        "assembled_offset: pos=[-0.106203, 0, 0], quat=[0.70710678, 0, 0.70710678, 0] (WXYZ). "
        "Byte-identical to the Sdf/Sdf1024/Sdf2048/ThreadSdfHybrid copies below (by design, per an "
        "in-file comment).",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf/metadata.yaml",
        2, _FB, STATUS_LOAD_BEARING,
        "Sibling copy of the Decomp metadata.yaml above -- THIS is the one actually read at runtime "
        "for OmniReset's shipping (Sdf) leg. Byte-identical to it.",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf1024/metadata.yaml",
        2, _FB, STATUS_SUPERSEDED, "Sibling copy, byte-identical to the Decomp metadata.yaml.",
    ),
    Asset(
        "source/uwlab_assets/uwlab_assets/local/Props/FurnitureBench/SquareTableLeg200mmSdf2048/metadata.yaml",
        2, _FB, STATUS_SUPERSEDED, "Sibling copy, byte-identical to the Decomp metadata.yaml.",
    ),
    Asset(
        "source/uwlab_assets/data/Props/FurnitureBench/SquareTableLeg200mmThreadSdfHybrid/metadata.yaml",
        2, _FB, STATUS_REJECTED, "Sibling copy, byte-identical to the Decomp metadata.yaml.",
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
    which bank actually backed any given result.

    Two DIFFERENT findings live here, for two DIFFERENT consumers -- conflating them was an error
    in an earlier draft of this manifest, corrected below:

      DexLift (lift_ep1950, repose_ep3600 -- the certified checkpoints): NOT APPLICABLE. DexLift
      does not load a reset bank at all. Its as-trained env.yaml (saved beside each checkpoint) has
      zero hits for dataset_dir/Resets/reset_state/reset_type/.pt/OneLegInsertionFixture, and the
      scene itself contains no OneLegInsertionFixture -- a table and the leg only. The reset
      distribution is PARAMETRIC, printed at startup by DEXLIFT_REF_RESET=1: "dexsuite start
      distribution restored (base +-0.5 rad, elbow +-0.2, wrist_3 +-0.5 rad [NOT dexsuite's +-3.0],
      object x [0.0, 0.2] far half [NOT dexsuite's +-0.2])". That printed line IS the complete reset
      spec for these two checkpoints. An empty
      ~/.cache/uwlab/assets/Datasets/OmniReset/Resets/OneLegInsertionFixture__SquareTableLeg200mmDecomp/
      is therefore the CORRECT state of the world for them, not a gap -- nothing was lost, there is
      no dependency to find, and "unrecoverable" would send a reader looking for a file that never
      existed. Keep the narrower observation this cache emptiness DOES support: the
      UWLAB_CLOUD_ASSETS_DIR default (below) was never the delivery path for anything real on this
      host -- a pipeline fact, not a missing-file fact.

      OmniReset (does load banks -- events.py:2301-2320, MultiResetManager and
      reset_insertive_object_from_partial_assembly_dataset): provenance EXISTS per run but is a
      POINTER TO A MUTABLE DIRECTORY, not a content record -- see the two defects below.
    """
    return (
        "CLASS 3 -- reset bank .pt files (mechanism documented, no fixed file list)\n"
        "\n"
        "### DexLift (lift_ep1950, repose_ep3600): NOT APPLICABLE\n"
        "\n"
        "DexLift does not load a reset bank. Confirmed by both the certified checkpoints' own "
        "as-trained env.yaml (zero hits for dataset_dir / Resets / reset_state / reset_type / .pt / "
        "OneLegInsertionFixture) and by the scene itself (table + leg only, no fixture). The reset "
        "distribution is parametric, entirely specified by the DEXLIFT_REF_RESET=1 startup banner: "
        "'dexsuite start distribution restored (base +-0.5 rad, elbow +-0.2, wrist_3 +-0.5 rad [NOT "
        "dexsuite's +-3.0], object x [0.0, 0.2] far half [NOT dexsuite's +-0.2])'. An empty "
        "~/.cache/uwlab/assets/Datasets/OmniReset/Resets/OneLegInsertionFixture__SquareTableLeg200mmDecomp/ "
        "on DL_H100 is therefore CORRECT for these checkpoints, not a gap.\n"
        "\n"
        "Narrower observation the empty cache DOES support: UWLAB_CLOUD_ASSETS_DIR's default "
        "(f'{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset', UWLAB_CLOUD_ASSETS_DIR = "
        "https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main) was never the delivery "
        "path for anything real on this host -- a pipeline fact (the default is unused; real inputs "
        "were ad hoc), not a missing-file fact.\n"
        "\n"
        "### OmniReset: provenance exists per run, but points at a MUTABLE directory\n"
        "\n"
        "Loaded by omnireset/mdp/events.py's MultiResetManager and "
        "reset_insertive_object_from_partial_assembly_dataset via "
        "utils.safe_retrieve_file_path(f'{dataset_dir}/Resets/{pair}/{name}.pt'), where "
        "pair = utils.compute_pair_dir(insertive_usd_path, receptive_usd_path). dataset_dir is an "
        "EventTerm param; several UR5eDelto/UR10eDelto configs override the cloud default per-variant "
        "via _apply_delto_dataset_dir to a local directory, and a tracking board records one per run "
        "(e.g. armU: dataset_dir=./Datasets_ur5e_delto/OmniReset). That record is real, but is much "
        "weaker evidence than it looks, for two reasons:\n"
        "\n"
        "1. It is a RELATIVE path, resolved against a working directory the config does not record. "
        "That it resolves under a given repo root today is an inference from where the tree currently "
        "sits, not a stored fact.\n"
        "2. The path is not the artifact. Confirmed on DL_H100 (read-only) inside "
        "~/github.com/orel/UWLab_ur5edelto/Datasets_ur5e_delto/OmniReset/: the live Resets/ directory "
        "holds 6 pair-dirs (Decomp, Sdf, Sdf1024, Sdf2048, ThreadSdfHybrid, "
        "Decomp_validate_scratch/Sdf_validate_scratch siblings), AND there are two separate backup "
        "snapshots of the same tree -- Resets_backup_c2rewind_20260825_045513/ and "
        "Resets_backup_c2rewind_20260825_061316/ -- each also containing a Decomp pair-dir. Banks are "
        "regenerated and backed up IN PLACE, so the same dataset_dir string maps to different "
        "contents at different times. Knowing the path does not tell you which states a run consumed.\n"
        "\n"
        "CONCLUSION: for OmniReset, provenance exists per run but is a pointer to a mutable "
        "directory, not a content record. The v2 fix -- log the per-reset-type state count and a "
        "hash of the bank tensor at load time, and absolute-ise the dataset_dir field -- is already "
        "its own bead; this manifest recommends it rather than implementing it.\n"
    )


def describe_leg_literals_section(repo_root: str) -> str:
    """F24: the OmniReset leg is named by five separate hardcoded literals. Report their CURRENT
    values live, via uwlab_assets.describe_leg_literals -- the exact same function
    assert_omnireset_leg_literals_agree() (the runtime check) uses -- so this section can never
    silently drift from what the runtime assertion actually checks.
    """
    sys.path.insert(0, os.path.join(repo_root, "source", "uwlab_assets"))
    try:
        import uwlab_assets  # noqa: E402
    except ImportError as exc:
        return (
            "F24 -- five literals name the OmniReset leg (COULD NOT VERIFY LIVE)\n\n"
            f"```\nFailed to import uwlab_assets from {repo_root} ({exc}). Falling back to the "
            "static record: rl_state_cfg.py:804, reset_states_cfg.py:617, grasp_sampling_cfg.py:206, "
            "partial_assemblies_cfg.py:482, dexlift_ur5e_delto_tableleg_env_cfg.py:48. Re-run this "
            "generator in an environment where uwlab_assets imports to get live values.\n```\n"
        )

    found = uwlab_assets.describe_leg_literals(repo_root)
    variants = {v for _, _, v in found}
    agree = len(variants) == 1

    lines = ["F24 -- five literals name the OmniReset leg (live-checked)\n"]
    lines.append("```")
    for rel_path, line, variant in found:
        lines.append(f"{rel_path}:{line} -> {variant}")
    lines.append("```\n")
    if agree:
        lines.append(f"**Result: AGREE.** All five literals currently name `{variants.pop()}`.")
    else:
        lines.append(f"**Result: MISMATCH.** {len(variants)} distinct values found -- {sorted(variants)}. "
                      "Every reset bank, checkpoint, and success number produced while these disagree is "
                      "INVALID. See uwlab_assets.assert_omnireset_leg_literals_agree, which raises fatally "
                      "on this exact condition at both TaskCommand.__init__ (OmniReset) and "
                      "dexlift_ur5e_delto_tableleg_env_cfg.py import time (dexlift).")
    lines.append(
        "\nThey disagreed until 2026-08-23: while they did, generation could read one leg variant and "
        "training another, silently, with nothing in the logs to distinguish that from a normal run "
        "(same defect class as F8)."
    )
    lines.append(
        "\n**Log-derived confirmation, and its limit.** The generator prints "
        "'[c4-seating-gate] ENABLED ... leg_usd=... fixture_usd=...' at startup, but ONLY when the "
        "seating gate is enabled -- this is NOT a property of runs in general. Of the 6 v1 generator "
        "runs on DL_H100, exactly 2 carry it and are independently confirmed by their own log: "
        "smoke_s1_bmix.log:191 and smoke_s2band.log:190, both "
        "leg_usd=.../SquareTableLeg200mmSdf/square_table_leg4_200mm.usd, "
        "fixture_usd=.../OneLegInsertionFixture/one_leg_insertion_fixture.usd. The other 4 -- "
        "smoke_s2probe.log, smoke_boremix4300.log, tmp/c2fwd5.log, and the vertical-goal run -- print "
        "no banner and carry NO in-log asset record at all; their asset selection is inferred from "
        "source only (F24 table above), not independently confirmed by their own run. RECOMMENDATION "
        "(already its own bead, not implemented here): print the resolved leg_usd/fixture_usd "
        "unconditionally at startup, outside the gate banner, so every future run is self-describing "
        "regardless of which gates are on."
    )
    return "\n".join(lines) + "\n"


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
    rejected: list[str] = []

    for klass, assets, title in ((1, CLASS1, "CLASS 1 -- gitignored USD geometry"),
                                  (2, CLASS2, "CLASS 2 -- asset metadata.yaml (defines assembled_offset)")):
        lines.append(f"\n## {title}\n")
        lines.append("| path | status | bytes | sha256 | origin | selector |")
        lines.append("|---|---|---|---|---|---|")
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
            if STATUS_REJECTED in a.status:
                rejected.append(a.path)
            lines.append(f"| `{a.path}` | {a.status} | {size} | `{sha}` | {a.origin} | {sel} |")

    lines.append("\n## " + describe_class3().splitlines()[0] + "\n")
    lines.append("```\n" + "\n".join(describe_class3().splitlines()[2:]) + "\n```\n")

    lines.append("\n## " + describe_leg_literals_section(repo_root))

    lines.append("\n## Summary\n")
    lines.append(f"- Total assets hashed (class 1 + 2): {total_count}")
    lines.append(f"- Total bytes: {total_bytes}")
    lines.append(f"- REJECTED for at least one consumer: {len(rejected)} -- kept as negative evidence, never a default: {rejected}")
    if missing:
        lines.append(f"- MISSING from `{assets_root}` (not hashed): {len(missing)} -- {missing}")
    if unverified:
        lines.append(f"- Selector claims that did NOT verify against current source: {len(unverified)} -- {unverified}")
    lines.append(f"- Licence: {LICENCE_NOTE}. Every FurnitureBench- and DELTO-derived asset above carries this "
                  "same note -- do not infer a licence from a directory name or from this repo's own "
                  "declared licence; that decision belongs to the user.")
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
