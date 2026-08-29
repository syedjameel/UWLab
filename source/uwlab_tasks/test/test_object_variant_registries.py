# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guards the invariant that let leg200mm go missing from three separate registries, one at a time,
across three different debugging sessions before anyone enumerated all four at once (bead
UWLab-z9j.11 follow-on).

THE DEFECT CLASS: object identity for this task family is declared independently in FOUR places --
``grasp_sampling_cfg.variants["scene.object"]``, ``reset_states_cfg.variants["scene.insertive_object"
/"scene.receptive_object"]``, ``partial_assemblies_cfg.variants[...]``, and
``rl_state_cfg.variants[...]``. Nothing ties them together. An object can be added to one registry
and silently omitted from the others; the only way anyone found out, historically, was hitting an
"unknown variant" or "shape mismatch" error at construction time, for ONE task at a time.

THE RULE THIS TEST ENFORCES, deliberately NOT "every object in every registry" (some objects
legitimately belong to one path only -- e.g. a grasp-sampling-only candidate never promoted to a
reset-state/RL-state task):

  RULE A: ``scene.insertive_object`` key sets must be IDENTICAL across ``reset_states_cfg.py``,
  ``partial_assemblies_cfg.py`` and ``rl_state_cfg.py``. These three are, by this codebase's own
  design (see the comments on every leg200mm/deltoblock entry in each), different stages of the
  SAME downstream pipeline for the SAME object universe: reset-state generation, partial-assembly
  generation, and RL training. An insertive object usable in one of these three but not another is,
  in every case found so far, a registration gap rather than an intentional restriction.

  RULE B: ``scene.receptive_object`` key sets must be IDENTICAL across the same three files.

  RULE C: every key in the union of insertive-object sets from Rule A must ALSO appear in
  ``grasp_sampling_cfg.variants["scene.object"]`` -- an object cannot be reset-state/RL-state
  usable without a recorded grasps.pt, which only grasp sampling produces. This direction is
  one-way: grasp_sampling_cfg.py MAY carry extra objects (grasp-sampling-only candidates) that the
  other three don't need yet, and that is not a violation.

  ``grasp_sampling_cfg.py`` itself is excluded from Rules A/B: it has no receptive_object concept at
  all (grasp sampling poses a gripper against one free object, no receptive fixture), so it is
  structurally not part of the "same three stages" group Rules A/B describe.

WHY THIS IS AST-BASED, NOT AN IMPORT-AND-INSPECT TEST, even though a plain import was the first
idea: importing any of these four cfg modules pulls in ``uwlab_tasks.manager_based.manipulation.
omnireset.mdp`` -> ``isaaclab.envs`` -> ``isaaclab.managers.action_manager``, which does
``import omni.kit.app`` at module scope -- verified directly, not assumed (see git history / the
report this test shipped with). That import only resolves inside a running Isaac Sim/Kit process.
``test_held_check_core.py`` documents the identical constraint for a different module and works
around it by loading a single dependency-free file directly; there is no equivalent free lunch here
because the object we need (the ``variants`` dict) lives inside modules that always drag in
``isaaclab.envs`` via their own imports (``EventTermCfg``, ``RigidObjectCfg``, etc.), so "import just
this one file" is not available. Reading the dict as a syntax tree, and pulling out its string
keys, needs nothing beyond the standard library, so this test runs with plain python: no Isaac, no
GPU, no environment construction, no dependency on isaaclab being importable at all.

FRAGILITY THIS TRADES FOR THAT: this only sees keys written as a literal ``variants = {...}`` dict
of literal string keys, which is what all four files use today. If a future refactor generates
these dicts programmatically (a loop, a comprehension, a helper function), this test goes BLIND to
that file rather than erroring -- update it to match if that ever happens; do not add a
programmatic branch it silently reports zero keys for.
"""

from __future__ import annotations

import ast
from pathlib import Path

_CONFIG_DIR = (
    Path(__file__).resolve().parents[1]
    / "uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85"
)

_GRASP_SAMPLING_PATH = _CONFIG_DIR / "grasp_sampling_cfg.py"
_RESET_STATES_PATH = _CONFIG_DIR / "reset_states_cfg.py"
_PARTIAL_ASSEMBLIES_PATH = _CONFIG_DIR / "partial_assemblies_cfg.py"
_RL_STATE_PATH = _CONFIG_DIR / "rl_state_cfg.py"


def _extract_variants(path: Path) -> dict[str, list[str]]:
    """Parse a config file's module-level ``variants = {...}`` and return {outer_key: [inner keys]}.

    Static (AST) extraction, on purpose -- see the module docstring for why this cannot import the
    file instead. Only matches a literal dict assigned directly to a module-level name ``variants``;
    a ``ClassName.variants = variants`` class-attribute assignment (present in every one of these
    files, referencing the very dict this function already found) is deliberately NOT matched here
    (its value is an ``ast.Name``, not an ``ast.Dict``), so it is skipped rather than double-counted
    or crashed on.
    """
    tree = ast.parse(path.read_text())
    result: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        is_module_level_variants_dict = (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "variants"
            and isinstance(node.value, ast.Dict)
        )
        if not is_module_level_variants_dict:
            continue
        for outer_key_node, inner_value in zip(node.value.keys, node.value.values):
            assert isinstance(outer_key_node, ast.Constant), (
                f"{path.name}: variants dict has a non-string-literal outer key ({ast.dump(outer_key_node)}) "
                "-- this test only understands literal keys, see the module docstring's fragility note."
            )
            assert isinstance(inner_value, ast.Dict), (
                f"{path.name}: variants[{outer_key_node.value!r}] is not a literal dict "
                f"({type(inner_value).__name__}) -- this test only understands literal keys, see the "
                "module docstring's fragility note."
            )
            keys = []
            for inner_key_node in inner_value.keys:
                assert isinstance(inner_key_node, ast.Constant), (
                    f"{path.name}: variants[{outer_key_node.value!r}] has a non-string-literal key "
                    f"({ast.dump(inner_key_node)})."
                )
                keys.append(inner_key_node.value)
            result[outer_key_node.value] = keys
        return result  # exactly one module-level `variants = {...}` per file, by construction
    raise AssertionError(f"{path.name}: no module-level literal `variants = {{...}}` dict found.")


def _missing_report(label_a: str, keys_a: set[str], label_b: str, keys_b: set[str]) -> str:
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    lines = []
    if only_a:
        lines.append(f"  in {label_a} but not {label_b}: {only_a}")
    if only_b:
        lines.append(f"  in {label_b} but not {label_a}: {only_b}")
    return "\n".join(lines)


def test_insertive_object_keys_consistent_across_paired_registries():
    """RULE A: reset_states_cfg / partial_assemblies_cfg / rl_state_cfg must agree on every
    insertive object. This is the exact rule that would have caught both gaps this bead found:
    leg200mm missing from partial_assemblies_cfg.py, and separately from rl_state_cfg.py -- the
    latter blocking the actual RL training task, not a side path.
    """
    files = {
        "reset_states_cfg.py": _extract_variants(_RESET_STATES_PATH)["scene.insertive_object"],
        "partial_assemblies_cfg.py": _extract_variants(_PARTIAL_ASSEMBLIES_PATH)["scene.insertive_object"],
        "rl_state_cfg.py": _extract_variants(_RL_STATE_PATH)["scene.insertive_object"],
    }
    key_sets = {name: set(keys) for name, keys in files.items()}
    reference_name, reference_keys = next(iter(key_sets.items()))
    for name, keys in key_sets.items():
        if keys == reference_keys:
            continue
        report = _missing_report(reference_name, reference_keys, name, keys)
        raise AssertionError(
            f"scene.insertive_object key sets differ between {reference_name} and {name}:\n{report}\n"
            "Every insertive object must be registered in all three of reset_states_cfg.py, "
            "partial_assemblies_cfg.py and rl_state_cfg.py -- see this file's module docstring "
            "(RULE A) for why. Add the missing entry, mirroring how the other objects are wired in "
            "the file that has it."
        )


def test_receptive_object_keys_consistent_across_paired_registries():
    """RULE B: the receptive-object mirror of the above."""
    files = {
        "reset_states_cfg.py": _extract_variants(_RESET_STATES_PATH)["scene.receptive_object"],
        "partial_assemblies_cfg.py": _extract_variants(_PARTIAL_ASSEMBLIES_PATH)["scene.receptive_object"],
        "rl_state_cfg.py": _extract_variants(_RL_STATE_PATH)["scene.receptive_object"],
    }
    key_sets = {name: set(keys) for name, keys in files.items()}
    reference_name, reference_keys = next(iter(key_sets.items()))
    for name, keys in key_sets.items():
        if keys == reference_keys:
            continue
        report = _missing_report(reference_name, reference_keys, name, keys)
        raise AssertionError(
            f"scene.receptive_object key sets differ between {reference_name} and {name}:\n{report}\n"
            "Every receptive object must be registered in all three of reset_states_cfg.py, "
            "partial_assemblies_cfg.py and rl_state_cfg.py -- see this file's module docstring "
            "(RULE B) for why. Add the missing entry, mirroring how the other objects are wired in "
            "the file that has it."
        )


def test_grasp_sampling_covers_every_insertive_object():
    """RULE C: every insertive object used downstream must have a grasp-sampling entry -- one-way,
    grasp_sampling_cfg.py may carry extra objects the other three don't need yet.
    """
    insertive_union: set[str] = set()
    for path in (_RESET_STATES_PATH, _PARTIAL_ASSEMBLIES_PATH, _RL_STATE_PATH):
        insertive_union |= set(_extract_variants(path)["scene.insertive_object"])

    grasp_sampling_keys = set(_extract_variants(_GRASP_SAMPLING_PATH)["scene.object"])

    missing = sorted(insertive_union - grasp_sampling_keys)
    assert not missing, (
        f"Object(s) {missing} are registered as scene.insertive_object in reset_states_cfg.py / "
        "partial_assemblies_cfg.py / rl_state_cfg.py but have no grasp_sampling_cfg.py "
        "scene.object entry -- they cannot have a recorded grasps.pt, so reset_end_effector_from_"
        "grasp_dataset (and everything downstream of it) has no dataset to read. Add the missing "
        "grasp_sampling_cfg.py entry -- see this file's module docstring (RULE C)."
    )


if __name__ == "__main__":
    # This file had NO runner. It is pytest-collectible, but pytest is not installed in either
    # interpreter used to run this suite locally, so `python3 test_object_variant_registries.py`
    # printed nothing and exited 0 -- indistinguishable from a pass. Three real assertions were
    # simply never executed here. Same report-every-failure shape as the other v2 suites (bead
    # dr-76w.23): one failing registry must not hide the state of the other three.
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
            except Exception as _exc:  # noqa: BLE001 -- a runner, it must report every failure
                _failures += 1
                print(f"[object_variants] {_name} FAILED: {type(_exc).__name__}: {_exc}", flush=True)
            else:
                print(f"[object_variants] {_name} OK", flush=True)
    if _failures:
        print(f"[object_variants] {_failures} test(s) FAILED", flush=True)
        raise SystemExit(1)
    print("[object_variants] all tests passed", flush=True)
