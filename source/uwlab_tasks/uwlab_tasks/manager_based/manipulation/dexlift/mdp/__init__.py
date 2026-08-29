# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the dexterous lifting task.

Everything the vendored dexsuite package already provides is re-exported here unchanged, so a
config module only ever imports from this one place. The local :mod:`.rewards` import comes last
and deliberately shadows the upstream ``contacts``, ``success_reward``,
``position_command_error_tanh`` and ``orientation_command_error_tanh``, whose upstream versions are
hardcoded to the Kuka-Allegro fingertip sensor names.
"""

from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp import *  # noqa: F401, F403

# NOTE: there is no local ``actions`` module. It held ContinuousSynergyJointPositionAction/Cfg --
# one scalar interpolating all twenty DELTO joints between two calibrated postures. It had no call
# sites left, but with no ``__all__`` anywhere in this package the star-import above put it in the
# ``mdp`` namespace that every dexlift env config already uses, so reinstating the banned
# one-scalar closure was a single line, ``mdp.ContinuousSynergyJointPositionActionCfg(...)``, in
# any of them. Deleting the class is what makes the ban structural instead of a convention.
# NOTE: there is no local ``terminations`` module either. It held ONE function,
# ``abnormal_robot_state_scoped`` -- the upstream velocity-limit cut with ``asset_cfg.joint_ids``
# honoured, so it could be pointed at the arm alone. It was deleted with its only call site: the
# test is ``|joint_vel| > 2 * data.joint_vel_limits``, and ``joint_vel_limits`` is the very number
# Isaac Lab writes into PhysX as each joint's maximum velocity
# (``articulation.py:1773`` -> ``write_joint_velocity_limit_to_sim`` -> ``set_dof_max_velocities``),
# so no scoping of it can fire. See ``_drop_unreachable_abnormal_robot_cut`` in
# ``dexlift_ur5e_delto_env_cfg``.
from .frame_guards import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403

# The policy-driven generator's HELD check (bead UWLab-dwx.2): settled & opposed-contact & co-move
# & probe-displacement-tracks. held_check_core.py (the pure-tensor decision function, unit-tested
# without Isaac) is imported directly by callers that want it standalone; only the stateful
# ManagerTermBase wrapper is re-exported here, same convention as every other mdp submodule.
from .held_check import *  # noqa: F401, F403

# The one thresholded success test of the task family -- the predicate the ADR curriculum promotes
# on -- extracted so the curriculum and the evaluation harness call one rule instead of two copies.
# It shadows nothing upstream: every name it exports is new, and it subclasses ``DifficultyScheduler``
# rather than replacing it.
from .success import *  # noqa: F401, F403
from .table_leg import *  # noqa: F401, F403

# The clearance-aware low-spawn reset (bead UWLab-qiao.1): a table-leg spawn whose z is DERIVED
# from clearance-above-surface and the sampled orientation, instead of sampled independently like
# ``reset_root_state_uniform``'s z. See ``dexlift_ur5e_delto_env_cfg.DEXLIFT_SPAWN_CLEARANCE``.
from .spawn import *  # noqa: F401, F403

# The goal / red-green task-state markers. This module SUBCLASSES the dexsuite pose command rather
# than replacing it -- upstream already owns all three visualizers -- so the star-import above must
# stay first: ``TaskStateVisPoseCommandCfg`` is a strict extension of the
# ``ObjectUniformPoseCommandCfg`` it re-exports, not a shadow of it.
from .task_state_vis import *  # noqa: F401, F403

# Partially-assembled table-leg spawn (bead UWLab-qiao.2/.6), gated behind
# ``DEXLIFT_PARTIAL_ASSEMBLY`` in the table-leg REORIENT config only. Imported after
# ``task_state_vis`` for the same reason that module is imported after the dexsuite star-import:
# ``GoalAtSpawnPoseCommandCfg`` extends ``TaskStateVisPoseCommandCfg``, not the raw dexsuite command.
from .partial_assembly import *  # noqa: F401, F403

# VERTICAL-goal MIXTURE for the DexReset S1/S2 finetune (epic UWLab-nnlv). Imported after
# ``task_state_vis``: ``MixedGoalPoseCommandCfg`` extends ``TaskStateVisPoseCommandCfg``.
from .goal_mixture import *  # noqa: F401, F403


# Per-episode MIXTURE of {classic goal, low goal, partial-assembly grasp-only} for the table-leg
# REORIENT finetune (epic UWLab-g3z4). Imported last: reuses both
# ``omnireset.mdp.events.reset_insertive_object_from_partial_assembly_dataset`` (same delegate
# ``partial_assembly.SpawnPartialAssembly`` uses) and ``task_state_vis.TaskStateVisPoseCommand``.
from .episode_mixture import *  # noqa: F401, F403

# RE-INTRODUCES a ``terminations`` submodule (see that module's own docstring for why the previous
# one was deleted, and why this is not a reversion of that): the C4 seating-aware training variant
# (DELIVERABLE 2, ``DEXLIFT_C4_GROSS_UNSEATING_TERM``, wired in
# ``dexlift_ur5e_delto_tableleg_env_cfg.py``) needs one predicate the base package has no
# equivalent of. Imported last, after ``episode_mixture``: shadows nothing.
from .terminations import *  # noqa: F401, F403

# C1's Cartesian palm-pose IK reset (RESET_SPEC_V2.md sec 1, V2_POSE_FINDINGS.md F10). Off by
# default; wired only by ``dexlift_ur5e_delto_env_cfg._apply_c1_hand_pose_stage`` when
# ``DEXRESET_C1_HAND=1``. Shadows nothing -- ``reset_end_effector_c1_hand_pose`` is a new name.
# ``c1_hand_pose_core`` (the Isaac-free half, unit-tested standalone) is deliberately NOT
# star-imported here; callers that want it use it directly, same convention as
# ``held_check_core``/``held_check`` next to it.
from .c1_hand_pose import *  # noqa: F401, F403

# C3 RUNG stage -- 50% S1 + 50% S_t (RESET_SPEC_V2.md sec 1 C3, bead dr-ai1.4). Off by default;
# wired only by ``dexlift_ur5e_delto_tableleg_env_cfg._apply_c3_rung_stage`` when
# ``DEXRESET_C3_RUNG=1``. Imported after ``episode_mixture`` and ``partial_assembly``: it reuses
# ``PARKED_FIXTURE_POSE_RANGE`` from the former and ``live_bore_deep_axis`` from the latter, and
# ``C3RungGoalPoseCommandCfg`` extends ``TaskStateVisPoseCommandCfg``. Shadows nothing -- every
# exported name is new. ``c3_rung_core`` (the Isaac-free half, unit-tested standalone) is
# deliberately NOT star-imported here, same convention as ``c1_hand_pose_core`` above.
from .c3_rung import *  # noqa: F401, F403

# TRAINING-TIME GATE PROXY -- the three probe-free gates of the generator's held chain, logged per
# episode and per mixture branch (V2_REPOSE_RECIPE.md sec 4, bead dr-tlx.2). Off by default; wired
# only by ``dexlift_ur5e_delto_env_cfg._attach_gate_proxy_metric`` when ``DEXRESET_GATE_PROXY=1``.
# Imported after ``episode_mixture`` (it reads that module's ``EPISODE_KIND_NAMES`` and
# ``EPISODE_KIND_BUFFER_ATTR`` rather than restating the kind integers) and after ``held_check``
# (it calls the same ``passive_gates`` ``held_decision`` calls, so the proxy and the generator
# cannot drift apart). ``gate_proxy_core`` (the Isaac-free half, unit-tested standalone) is
# deliberately NOT star-imported here, same convention as ``held_check_core``/``c1_hand_pose_core``.
from .gate_proxy import *  # noqa: F401, F403
