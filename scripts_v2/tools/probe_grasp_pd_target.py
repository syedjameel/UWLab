# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the PD-target gap at the exact instant a reset state is accepted (bead UWLab-algw.1).

THE QUESTION. A saved StableState carries root_pose, root_velocity, joint_position, joint_velocity
-- NOT joint_position_target. ``replay_reset_states_policy.py`` writes ``target := q`` on restore
(``robot.set_joint_position_target(q)``), i.e. the commanded squeeze is set to exactly zero the
moment a state is written back, however hard the hand was actually squeezing when the state was
recorded. H1: if the PD target the generator's rollout was actually holding at accept-time is
materially different from ``q``, that zeroing is the dominant loss mechanism behind the 103.8mm
median in-palm drift measured on replay, and the fix is to record and restore the target.

THIS SCRIPT MEASURES, IT DOES NOT FIX. No dataset is written (``DatasetExportMode.EXPORT_NONE``),
no bank is regenerated, no checkpoint is retrained.

TWO PHASES.

PHASE 1 (the rollout, reusing generate_reset_states_policy.py's own machinery: same checkpoint-load
idiom, same ``held_with_probe`` termination wired as ``terminations.success``, same re-arming probe
bias injection -- see that script's module docstring for why the plant and the probe are built the
way they are; nothing about the rollout mechanics is changed here). The only addition is a custom
recorder (``_PDTargetProbeRecorder``, subclassing ``recorders.StableStateRecorder`` so the ordinary
state-capture logic is reused rather than reimplemented) that, inside ``record_pre_reset``, reads
``robot.data.joint_pos_target`` / ``joint_pos`` / ``applied_torque`` / ``computed_torque`` for every
env whose ``success`` termination fired THIS step, and stashes a copy of the full scene state (for
Phase 2) alongside the target that produced it.

WHY record_pre_reset IS THE RIGHT HOOK, NOT A CONVENIENT ONE. Read directly off
``ManagerBasedRLEnv.step`` (isaaclab/envs/manager_based_rl_env.py:172-221): actions are applied
(action_manager.apply_action, once per decimation substep) and physics stepped BEFORE terminations
are computed, and ``recorder_manager.record_pre_reset(reset_env_ids)`` runs immediately after that
-- strictly before ``self._reset_idx(reset_env_ids)``, which is what would overwrite
``joint_pos_target`` with whatever the reset events set. So ``robot.data.joint_pos_target`` read
inside ``record_pre_reset`` is EXACTLY the target the actuator was driving toward on the physics
step that made ``held_with_probe`` return True -- the same simulation instant the recorder captures
``joint_position`` from for the state that ships. There is no window where a later event could have
moved the target between "the state that gets saved" and "the target this script reports".

CROSS-CHECK, VERIFIED BY READING SOURCE, NOT RE-DERIVED BY THIS SCRIPT.
``isaaclab.envs.mdp.actions.joint_actions.RelativeJointPositionAction.apply_actions`` is::
    current_actions = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
    self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)
i.e. ``joint_pos_target`` IS the value the ``arm``/``gripper`` action terms wrote via
``set_joint_position_target`` (dexlift_ur5e_delto_actions.py's two RelativeJointPositionAction
terms) -- not some other quantity a later manager could have silently substituted. The one caveat:
``joint_pos`` at read time is POST the last physics substep of the step that produced this target,
so it differs from the ``joint_pos`` the action term computed the target FROM (pre-substep) by
however far the joint moved during that one substep (sim.dt = 1/120s) -- an expected, small effect
of measuring after the fact, not a bug in either number.

PHASE 2 (the persistence question, added after a finding that generation runs at 60 Hz -- dexlift,
decimation 2 -- while the OmniReset consumer runs at 10 Hz -- decimation 12, same sim.dt=1/120s, so
one consumer control period is 12 physics substeps / 100ms). If restoring the recorded TARGET
(instead of zeroing it, as today's replay does) is to be sufficient ON ITS OWN, the PD error it
implies has to survive un-refreshed for a full 100ms before the consumer's own policy gets to issue
its first new action. Phase 2 tests exactly that, on a FOUR-ARM sweep over the SAME accepted states
from Phase 1 (never re-drawn between arms -- that is the whole point, bead UWLab-algw.11):

    target   in {recorded target, q_recorded}   -- q_recorded is exactly what today's replay does
    velocity in {recorded velocity, zeroed}

For each of the four (target, velocity) combinations: write back the FULL scene state plus that
arm's target/velocity choice into a batch of live envs, then advance physics for
``--hold_substeps`` (default 12 = 100ms) WITHOUT ever calling ``action_manager.apply_action()``
again -- so ``joint_pos_target`` is never recomputed by a new policy action, exactly mirroring
"restore once, then nothing else touches it until the next control decision".
``Articulation.write_data_to_sim()`` itself calls ``_apply_actuator_model()`` (articulation.py:258)
-- so calling it once per substep (mirroring the env's own per-substep call, only omitting
``apply_action()``) still re-computes actuator torque against the FROZEN target and the EVOLVING
measured position every substep, for both the hand's ``ImplicitActuator`` (PhysX-internal PD,
continuously driven once the target is latched) and the arm's explicit actuator (Python-side
``compute()``, which without this per-substep ``write_data_to_sim()`` call would freeze at whatever
torque was last computed and NOT track the restored target at all -- the loop deliberately avoids
that trap). The object-in-palm metric (``in_hand()``/``EE_BODY``, reused verbatim from
``replay_reset_states_policy.py``, including its 20mm ``HOLD_MM`` "still held" threshold) is tracked
over the same window per arm, so the target-gap story and the physical slip story are read side by
side, and a still-held fraction is reported per arm.

WHY THE CONTRAST IS THE MEASUREMENT (bead UWLab-algw.11), not any one arm's number in isolation: if
a useful fraction of states survive the hold at ``target := q_recorded`` (arms C/D), a state can be
made self-consistent with replay's own convention just by settling it at generation time and NO
schema change is needed. If essentially nothing survives at ``target := q_recorded`` while the
recorded-target arms (A/B) hold fine, that is positive proof the hand cannot hold at zero commanded
squeeze and the schema change (recording and restoring the target) becomes mandatory. Nothing here
is tuned to make any arm look better; the four are run identically over the identical states.

WHAT A NULL RESULT LOOKS LIKE (H1 FALSE), stated up front so it is not invented after the fact: arm
D (``target := q``, ``velocity := 0`` -- byte-for-byte today's replay convention) shows a held
fraction close to arm A's (the fully-recorded restore), or in absolute terms "usefully high" (most
states still under the 20mm ``HOLD_MM`` threshold at 100ms) even with the commanded squeeze zeroed.
Under that outcome the 103.8mm median drift measured on the existing replay is NOT explained by the
missing target -- something else (material re-randomisation, the mid-episode-teleport transient, a
different bug) is the dominant loss mechanism, and adding the target to the schema would not fix the
observed problem. THE POSITIVE RESULT (H1 TRUE) is the opposite shape: a LARGE, clean gap between the
recorded-target arms (A/B) and the q-target arms (C/D) -- C/D's held fraction near zero while A/B's
stays high. A small or noisy gap between the four (given PhysX-on-GPU is unseeded -- see
``replay_reset_states_policy.py``'s own caveat, inherited here unchanged, not a new limitation of
this script) should be read as inconclusive, not as a positive result rounded up.

PHASE 3 (optional, skip with ``--skip_duration_sweep``): once the four arms have run, whichever one
had the highest held-fraction at the end of the Phase-2 window is re-run ONCE for
``max(--duration_sweep_substeps)`` substeps (default checkpoints 12/60/120 = 100ms/500ms/1s), and
the shorter checkpoints are read out of that single longer trajectory rather than re-run -- so states
that survive 100ms can be checked against whether they also survive 1s, for the cost of one extra
hold instead of three.

WHAT PHASE 2/3 CANNOT SEE, stated so the numbers are not over-read: they measure whether a given
(target, velocity) write survives un-refreshed for the window with NOTHING else driving it. Neither
simulates what the actual OmniReset consumer's own trained policy would do once its first control
decision lands -- that decision could re-tighten the grip, relax it further, or something else the
checkpoint was never asked about. This answers "is restoring {this arm} alone enough to bridge that
first gap", not "does the object end up in the hand once the real controller resumes".

Run (one Isaac process; never via uwlab.sh):
    PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl" \\
        timeout -s KILL 600 <python> -u scripts_v2/tools/probe_grasp_pd_target.py \\
        --checkpoint <path>.pth --num_envs 32 --min_states 32 --max_steps 3000

Smoke test with no GPU / no Isaac process at all:
    <python> scripts_v2/tools/probe_grasp_pd_target.py --dry-run [--checkpoint <path>.pth]
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure the PD-target gap at accepted-state time.")
parser.add_argument(
    "--task", type=str, default="DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-Play-v0",
    help="Must match the checkpoint's own plant (RelJointPos, table-leg, 60Hz). The checkpoint this"
    " bead points at is a Reorient checkpoint -- see local_ckpts/dexlift_ur5e_delto_reljointpos_"
    "tableleg_reorient/ -- so this defaults to the Reorient-Play id, matching"
    " replay_reset_states_policy.py's own default.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to the .pth checkpoint. Required unless --dry-run.")
parser.add_argument("--agent_yaml", type=str, default=None, help="Defaults to <checkpoint>/../../params/agent.yaml.")
parser.add_argument("--num_envs", type=int, default=32, help="Also the Phase-2 write-back batch size.")
parser.add_argument("--min_states", type=int, default=32, help="Phase 1 stops once this many accepted states are captured.")
parser.add_argument("--max_steps", type=int, default=3000, help="Hard cap on Phase-1 control steps regardless of yield.")
parser.add_argument("--hold_substeps", type=int, default=12, help="Phase-2 window, physics substeps (12 = 100ms consumer control period at sim.dt=1/120s).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--skip_duration_sweep", action="store_true",
    help="Skip the optional Phase 3 (does the best-held-at-100ms arm also survive longer?).",
)
parser.add_argument(
    "--duration_sweep_substeps", type=str, default="12,60,120",
    help="Comma-separated physics-substep checkpoints for the optional Phase 3 duration sweep"
    " (12=100ms, 60=500ms, 120=1s at sim.dt=1/120s). The sweep runs ONE hold for max(checkpoints)"
    " and reads the others out of that single trajectory, not three separate runs.",
)
parser.add_argument(
    "--dry-run", action="store_true",
    help="Validate argument handling only. Builds nothing (no AppLauncher, no torch/gym/isaaclab-task"
    " import) so this is smoke-testable with no GPU and no Isaac process.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

if args_cli.dry_run:
    # -- SELF-CHECK ONLY. Everything below this block (AppLauncher construction, torch/gym/isaaclab-
    # task imports, os.environ plant vars) is skipped on purpose: a --dry-run smoke test has to be
    # runnable with no GPU and no rented box, or it cannot be run before one is provisioned.
    print("[probe] DRY RUN -- argument handling self-check only, no GPU/Isaac process constructed.", flush=True)
    print(f"[probe]   task            = {args_cli.task}", flush=True)
    print(f"[probe]   checkpoint      = {args_cli.checkpoint!r}", flush=True)
    if args_cli.checkpoint:
        derived_agent_yaml = args_cli.agent_yaml or os.path.join(
            os.path.dirname(os.path.dirname(args_cli.checkpoint)), "params", "agent.yaml"
        )
    else:
        derived_agent_yaml = args_cli.agent_yaml or "<derived from --checkpoint, none given>"
    print(f"[probe]   agent_yaml      = {derived_agent_yaml}", flush=True)
    print(f"[probe]   num_envs        = {args_cli.num_envs}  (also the Phase-2 batch size)", flush=True)
    print(f"[probe]   min_states      = {args_cli.min_states}", flush=True)
    print(f"[probe]   max_steps       = {args_cli.max_steps}", flush=True)
    print(f"[probe]   hold_substeps   = {args_cli.hold_substeps}  (100ms consumer control period at sim.dt=1/120s if 12)", flush=True)
    print(f"[probe]   seed            = {args_cli.seed}", flush=True)
    print(f"[probe]   skip_duration_sweep    = {args_cli.skip_duration_sweep}", flush=True)
    try:
        parsed_checkpoints = sorted(int(x) for x in args_cli.duration_sweep_substeps.split(","))
        print(f"[probe]   duration_sweep_substeps = {args_cli.duration_sweep_substeps!r} -> parsed {parsed_checkpoints}", flush=True)
    except ValueError as exc:
        raise SystemExit(f"[probe] --duration_sweep_substeps {args_cli.duration_sweep_substeps!r} did not parse as a comma-separated int list: {exc}")
    print(
        "[probe]   four-arm sweep  = target in {recorded, q_recorded} x velocity in {recorded, zero},"
        " same accepted states across all four (no re-draw)",
        flush=True,
    )
    print(
        "[probe]   plant env vars  = DEXLIFT_REF_RESET=1 DEXLIFT_REF_ACTUATORS=1 DEXLIFT_REF_HAND_ACT=1"
        " DEXLIFT_REF_ARM_ACT=0 DEXLIFT_POSE_TILT=0.3 (hardcoded below the dry-run gate, not CLI-overridable)",
        flush=True,
    )
    print("[probe]   expected hand joints (20): rj_dg_[1-5]_[1-4]", flush=True)
    print(
        "[probe]   expected arm joints  (6): shoulder_pan_joint, shoulder_lift_joint, elbow_joint,"
        " wrist_1_joint, wrist_2_joint, wrist_3_joint",
        flush=True,
    )
    print("[probe]   expect hand effort_limit/velocity_limit to read 30.0 N*m / 10000.0 rad/s once built.", flush=True)
    if not args_cli.checkpoint:
        print("[probe]   NOTE: --checkpoint not given -- fine for --dry-run, required otherwise.", flush=True)
    print("DRY_RUN_OK", flush=True)
    raise SystemExit(0)

if not args_cli.checkpoint:
    raise SystemExit("[probe] --checkpoint is required outside --dry-run")

# THE PLANT VARS MUST BE SET BEFORE parse_env_cfg -- they are read at cfg construction. Mirrors
# replay_reset_states_policy.py exactly: reference reset distribution, reference actuators overall,
# reference (30 N*m / 10000 rad/s) hand block specifically (the block the checkpoint trained under),
# identified arm block (DEXLIFT_REF_ARM_ACT=0, matching the OmniReset training scene this bank feeds),
# and the +-0.3 rad pose-tilt narrowing. Not CLI-overridable: this script exists to answer one
# question under one fixed, previously-agreed plant, not to sweep the plant.
os.environ["DEXLIFT_REF_RESET"] = "1"
os.environ["DEXLIFT_REF_ACTUATORS"] = "1"
os.environ["DEXLIFT_REF_HAND_ACT"] = "1"
os.environ["DEXLIFT_REF_ARM_ACT"] = "0"
os.environ["DEXLIFT_POSE_TILT"] = "0.3"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import uwlab_tasks  # noqa: F401,E402
import uwlab_tasks.manager_based.manipulation.dexlift.mdp as dexlift_mdp  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.managers.recorder_manager import DatasetExportMode  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from uwlab_tasks.manager_based.manipulation.dexlift.mdp.held_check_core import PROBE_ARM_ACTION_BIAS  # noqa: E402
from uwlab_tasks.manager_based.manipulation.dexlift.mdp.rewards import _sensor_force_magnitudes  # noqa: E402
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.recorders import recorders as _recorders_mod  # noqa: E402
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.recorders.recorders_cfg import (  # noqa: E402
    StableStateRecorderManagerCfg,
)

# Same five fingertip sensor names held_with_probe uses (thumb + tip, in that gate's own order),
# concatenated so one call returns all five per env -- see held_check.py's defaults.
ALL_TIP_NAMES = ("rl_dg_1_tip", "rl_dg_5_tip", "rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip")
EE_BODY = "rl_dg_mount"  # same palm body replay_reset_states_policy.py's in_hand() uses
HOLD_MM = 20.0  # same "still held" threshold replay_reset_states_policy.py uses on in-palm drift


def unwrap(o):
    if isinstance(o, tuple):
        o = o[0]
    if isinstance(o, dict):
        o = o["obs"]
    return o


RECORDS: list[dict] = []
"""Module-level sink, one entry per ACCEPTED env, filled by _PDTargetProbeRecorder.record_pre_reset.
Module-level (not an attribute read back off the term instance) because RecorderManager owns and
constructs the term; this is the simplest way to get the captures back out into main() afterward."""


class _PDTargetProbeRecorder(_recorders_mod.StableStateRecorder):
    """Reuses ``StableStateRecorder``'s own state-capture (not reimplemented -- see its
    ``record_pre_reset``: ``self._env.scene.get_state(is_relative=True)``, indexed by env_ids) and
    additionally reads the PD-target gap etc. for every env whose ``success`` term fired THIS step.

    See the module docstring for why ``record_pre_reset`` is the correct (not merely convenient)
    hook: it runs after the step's last physics substep and before ``_reset_idx`` overwrites
    anything, so ``robot.data.joint_pos_target`` here is exactly what drove that substep.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        robot = env.scene["robot"]
        self.hand_ids, self.hand_names = robot.find_joints([r"rj_dg_[1-5]_[1-4]"], preserve_order=True)
        self.arm_ids, self.arm_names = robot.find_joints(
            [r"shoulder.*", r"elbow.*", r"wrist.*"], preserve_order=True
        )
        assert len(self.hand_ids) == 20, f"expected 20 DELTO hand joints, got {len(self.hand_ids)}: {self.hand_names}"
        assert len(self.arm_ids) == 6, f"expected 6 UR5e arm joints, got {len(self.arm_ids)}: {self.arm_names}"
        print(f"[probe] hand joints ({len(self.hand_ids)}): {self.hand_names}", flush=True)
        print(f"[probe] arm joints  ({len(self.arm_ids)}): {self.arm_names}", flush=True)

    def record_pre_reset(self, env_ids):
        # -- reuse the production capture verbatim; `full_state` is env_ids-indexed (shape
        # (len(env_ids), ...) per leaf tensor), same as what ships in a real reset-state file.
        key, full_state = super().record_pre_reset(env_ids)

        env = self._env
        robot = env.scene["robot"]
        obj = env.scene["object"]
        success = env.termination_manager.get_term("success")
        ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        succ_local = success[ids_t]  # (len(env_ids),) bool -- LOCAL indexing, matches full_state's own order
        succ_ids = ids_t[succ_local]  # GLOBAL env indices, for reading LIVE robot.data directly
        if succ_ids.numel() == 0:
            return key, full_state

        jp = robot.data.joint_pos[succ_ids]
        jpt = robot.data.joint_pos_target[succ_ids]
        gap = (jpt - jp).abs()

        applied_torque = getattr(robot.data, "applied_torque", None)
        computed_torque = getattr(robot.data, "computed_torque", None)
        applied = applied_torque[succ_ids].abs() if applied_torque is not None else None
        computed = computed_torque[succ_ids].abs() if computed_torque is not None else None

        tip_forces = _sensor_force_magnitudes(env, ALL_TIP_NAMES)[succ_ids]  # (k, 5)

        p_palm, q_palm = robot.data.body_pos_w[succ_ids, self._ee_id(robot)], robot.data.body_quat_w[
            succ_ids, self._ee_id(robot)
        ]
        obj_in_palm, _ = math_utils.subtract_frame_transforms(p_palm, q_palm, obj.data.root_pos_w[succ_ids], obj.data.root_quat_w[succ_ids])

        def sub_nested(d):
            if isinstance(d, dict):
                return {k: sub_nested(v) for k, v in d.items()}
            return d[succ_local].cpu()

        RECORDS.append({
            "hand_gap": gap[:, self.hand_ids].cpu(),
            "arm_gap": gap[:, self.arm_ids].cpu(),
            "tip_forces": tip_forces.cpu(),
            "target": jpt.cpu(),  # full 26-d, for Phase-2 write-back
            "obj_in_palm": obj_in_palm.cpu(),  # (k, 3), Phase-2 baseline
            "full_state": sub_nested(full_state),  # for Phase-2 write-back
            **({"hand_applied_torque": applied[:, self.hand_ids].cpu(), "arm_applied_torque": applied[:, self.arm_ids].cpu()} if applied is not None else {}),
            **({"hand_computed_torque": computed[:, self.hand_ids].cpu(), "arm_computed_torque": computed[:, self.arm_ids].cpu()} if computed is not None else {}),
        })
        return key, full_state

    def _ee_id(self, robot) -> int:
        if not hasattr(self, "_ee_id_cache"):
            ids, _ = robot.find_bodies([EE_BODY])
            self._ee_id_cache = ids[0]
        return self._ee_id_cache


def cat(records: list[dict], key: str) -> torch.Tensor | None:
    if not records or key not in records[0]:
        return None
    return torch.cat([r[key] for r in records], dim=0)


def flatten_records(records: list[dict]) -> list[dict]:
    """Each entry in ``RECORDS`` comes from ONE ``record_pre_reset(env_ids)`` call, which can accept
    MORE THAN ONE env in the same env.step() (simultaneous successes) -- every per-record tensor
    then carries a leading dim ``k >= 1`` (see ``_PDTargetProbeRecorder``: ``gap[:, self.hand_ids]``
    etc are batched over ``succ_ids``, not squeezed to one state). Phase 1's own reporting already
    handles that correctly via ``cat()``'s ``torch.cat(dim=0)``. Phase 2's write-back needs the
    OPPOSITE shape -- one dict per state, no leading dim, so ``torch.stack`` across a batch of
    records produces a clean ``(batch, ...)`` tensor -- so this splits every bundled record into
    ``k`` single-state dicts before Phase 2 ever touches them.
    """
    flat: list[dict] = []
    for r in records:
        k = r["target"].shape[0]
        for j in range(k):
            def sub(d, j=j):
                if isinstance(d, dict):
                    return {key: sub(v, j) for key, v in d.items()}
                return d[j]
            flat.append({key: sub(val) for key, val in r.items()})
    return flat


def report_gap(name: str, t: torch.Tensor) -> None:
    flat = t.reshape(-1)
    per_env_max = t.max(dim=1).values
    print(
        f"[result] {name}  n_states={t.shape[0]}  n_joints={t.shape[1]}  "
        f"per-joint(rad): median={torch.median(flat).item():.4f} p90={torch.quantile(flat, 0.9).item():.4f} "
        f"max={flat.max().item():.4f}  "
        f"per-STATE max(rad): median={per_env_max.median().item():.4f} "
        f"p90={torch.quantile(per_env_max, 0.9).item():.4f} max={per_env_max.max().item():.4f}",
        flush=True,
    )


def main() -> None:
    agent_yaml = args_cli.agent_yaml
    if agent_yaml is None:
        agent_yaml = os.path.join(os.path.dirname(os.path.dirname(args_cli.checkpoint)), "params", "agent.yaml")
    with open(agent_yaml) as f:
        agent_cfg = yaml.safe_load(f)
    print(f"[probe] agent yaml: {agent_yaml}", flush=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # Small-env dev scene; the training default (3.75 GiB) fails to allocate at low env counts on a
    # 16 GB card.
    env_cfg.sim.physx.gpu_collision_stack_size = 256 * 1024 * 1024
    env_cfg.seed = args_cli.seed

    # -- THE HELD-CHECK, wired exactly as generate_reset_states_policy.py does (see that script's
    # comment on why this must be a dynamically-added instance attribute, not a rebuilt cfg).
    env_cfg.terminations.success = DoneTerm(func=dexlift_mdp.held_with_probe, time_out=True, params={})

    # -- recorder plumbing: ONLY the probe recorder above, EXPORT_NONE (no dataset ever touches
    # disk -- this run measures, it does not generate).
    env_cfg.recorders = StableStateRecorderManagerCfg()
    env_cfg.recorders.record_pre_reset_states.class_type = _PDTargetProbeRecorder
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    obj = env.scene["object"]
    print(f"[probe] robot DOF: {robot.num_joints}, action dim: {env.action_manager.total_action_dim}", flush=True)

    # -- VERIFY THE PLANT THAT WAS BUILT, never the env var requested (a silently ineffective
    # DEXLIFT_REF_HAND_ACT is exactly the failure mode this whole family of scripts guards against).
    hand_ids_all, _ = robot.find_joints([r"rj_dg_[1-5]_[1-4]"], preserve_order=False)
    eff = robot.data.joint_effort_limits[0, hand_ids_all]
    vel = robot.data.joint_velocity_limits[0, hand_ids_all]
    print(f"[verify] hand effort_limit  min/max: {eff.min():.3f}/{eff.max():.3f} N*m  (expect 30.0/30.0)", flush=True)
    print(f"[verify] hand velocity_limit min/max: {vel.min():.1f}/{vel.max():.1f} rad/s  (expect 10000.0/10000.0)",
          flush=True)
    if not (float(eff.min()) > 1.0 and float(vel.min()) > 100.0):
        raise SystemExit(
            "[probe] REFUSING: hand actuator limits do not read as the reference block "
            f"(effort {eff.min():.3f}-{eff.max():.3f}, velocity {vel.min():.1f}-{vel.max():.1f}). "
            "DEXLIFT_REF_HAND_ACT did not take effect as requested -- this would build the "
            "identified hand and every number downstream would be a plausible lie."
        )

    # -- (1) POLICY ROLLOUT, play.py's idiom verbatim (same as generate_reset_states_policy.py).
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_act = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    wrapped_env = RlGamesVecEnvWrapper(env, args_cli.device, clip_obs, clip_act, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: wrapped_env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    agent_cfg["params"]["config"]["num_actors"] = env.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(args_cli.checkpoint)
    player.reset()
    player.has_batch_dimension = True
    print("[probe] POLICY_LOADED", flush=True)

    obs = unwrap(wrapped_env.reset())
    success_term = env.termination_manager.get_term_cfg("success").func  # the held_with_probe instance

    n_attempts = 0
    for step in range(args_cli.max_steps):
        with torch.inference_mode():
            act = player.get_action(obs, is_deterministic=True)
            # -- same re-arming, event-triggered probe bias as generate_reset_states_policy.py.
            in_probe = success_term.probe_active
            if in_probe.any():
                act[in_probe, 0:6] = act[in_probe, 0:6] + PROBE_ARM_ACTION_BIAS
            ret = wrapped_env.step(act)
            obs = unwrap(ret[0])
            dones = ret[2].flatten().to(torch.bool)

        if dones.any():
            n_attempts += int(dones.sum())

        if step % 100 == 0:
            print(f"[progress] step={step}  attempts={n_attempts}  captured={len(RECORDS)}", flush=True)

        if len(RECORDS) >= args_cli.min_states:
            print(f"[probe] reached {len(RECORDS)} captured states at step {step}, stopping Phase 1.", flush=True)
            break
        if env.sim.is_stopped():
            break

    print("\n=== PHASE 1 RESULT (rollout) ===", flush=True)
    print(f"attempts (episodes ended): {n_attempts}", flush=True)
    print(f"captured (held_with_probe True at reset, PD state read): {len(RECORDS)}", flush=True)

    if not RECORDS:
        print("[probe] NO STATES CAPTURED -- cannot report a distribution. See attempts above.", flush=True)
        env.close()
        return

    hand_gap = cat(RECORDS, "hand_gap")
    arm_gap = cat(RECORDS, "arm_gap")
    print(f"\n[result] === PD TARGET GAP |joint_pos_target - joint_pos| over {hand_gap.shape[0]} accepted states ===",
          flush=True)
    report_gap("HAND (20 DELTO joints)", hand_gap)
    report_gap("ARM  (6 UR5e joints)", arm_gap)

    for label, key in (
        ("HAND applied |torque| (N*m)", "hand_applied_torque"),
        ("ARM  applied |torque| (N*m)", "arm_applied_torque"),
        ("HAND computed |torque| (N*m)", "hand_computed_torque"),
        ("ARM  computed |torque| (N*m)", "arm_computed_torque"),
    ):
        t = cat(RECORDS, key)
        if t is None:
            print(f"[result] {label}: field not present on this ArticulationData build, skipped.", flush=True)
            continue
        flat = t.reshape(-1)
        print(
            f"[result] {label}  median={torch.median(flat).item():.4f}  "
            f"p90={torch.quantile(flat, 0.9).item():.4f}  max={flat.max().item():.4f}",
            flush=True,
        )

    tip_forces = cat(RECORDS, "tip_forces")
    print(f"\n[result] === fingertip contact normal force (N), 5 tips, {tip_forces.shape[0]} accepted states ===",
          flush=True)
    for i, name in enumerate(ALL_TIP_NAMES):
        col = tip_forces[:, i]
        print(f"[result]   {name:12s}  median={col.median().item():.3f}  p90={torch.quantile(col, 0.9).item():.3f}  "
              f"max={col.max().item():.3f}  min={col.min().item():.3f}", flush=True)

    # ================================================================================================
    # PHASE 2: four-arm sweep -- does target/velocity, restored and held with no new action, survive
    # one 100ms consumer control period? SAME accepted states across all four arms (bead UWLab-algw.11).
    # See module docstring for exactly what this does and does not simulate.
    # ================================================================================================
    # -- UNBUNDLE: RECORDS holds one entry per record_pre_reset() call, which can bundle >1
    # simultaneous accept (see flatten_records' docstring). Phase 2 needs one dict per STATE.
    FLAT_RECORDS = flatten_records(RECORDS)
    print(f"[probe] Phase 1 produced {len(RECORDS)} capture event(s) containing {len(FLAT_RECORDS)} total"
          f" accepted state(s) (>1 event bundled multiple simultaneous accepts if these differ).",
          flush=True)

    ee_id, _ = robot.find_bodies([EE_BODY])
    ee_id = ee_id[0]
    # hand_ids/arm_ids live on the recorder term instance, which RecorderManager owns and does not
    # expose here -- re-derive once, identically, so Phase 2/3 need no coupling to the
    # (already-torn-down-by-now) recorder object.
    hand_ids, _ = robot.find_joints([r"rj_dg_[1-5]_[1-4]"], preserve_order=True)
    arm_ids, _ = robot.find_joints([r"shoulder.*", r"elbow.*", r"wrist.*"], preserve_order=True)

    def in_hand() -> torch.Tensor:
        p, _ = math_utils.subtract_frame_transforms(
            robot.data.body_pos_w[:, ee_id], robot.data.body_quat_w[:, ee_id],
            obj.data.root_pos_w, obj.data.root_quat_w,
        )
        return p

    def run_hold_arm(records: list[dict], target_mode: str, vel_mode: str, hold_substeps: int) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Write back ``records`` (never re-drawn between arms) under one (target_mode, vel_mode)
        arm, hold for ``hold_substeps`` physics substeps with no new action, return per-substep
        (hand_gap, arm_gap, in_palm_drift_mm) trajectories -- each a list of length
        ``hold_substeps + 1`` (index 0 = immediately after write, before any physics advances).

        target_mode: "recorded" -> the PD target this script measured in Phase 1.
                     "q"        -> today's replay_reset_states_policy.py convention (target := q).
        vel_mode:    "recorded" -> the recorded joint velocity.
                     "zero"     -> today's replay_reset_states_policy.py convention (velocity := 0).
        """
        per_hand: list[list[torch.Tensor]] = [[] for _ in range(hold_substeps + 1)]
        per_arm: list[list[torch.Tensor]] = [[] for _ in range(hold_substeps + 1)]
        per_drift: list[list[torch.Tensor]] = [[] for _ in range(hold_substeps + 1)]

        batch_size = args_cli.num_envs
        n_batches = (len(records) + batch_size - 1) // batch_size
        # Phase 1's rollout ran entirely inside `with torch.inference_mode():` (matches
        # generate_reset_states_policy.py's own convention), so every buffer on env.scene["robot"]
        # /["object"].data that Phase 1 touched is now an INFERENCE TENSOR for the rest of the
        # process's life -- PyTorch requires in-place writes to those buffers (write_root_pose_to_sim
        # etc. do `self._data.root_link_pose_w[env_ids] = ...`) to also happen inside inference_mode,
        # or it raises "Inplace update to inference tensor outside InferenceMode is not allowed."
        # First run hit exactly that on the very first write of Phase 2 -- wrapping the whole
        # write-back + hold loop here is the fix, not a workaround.
        with torch.inference_mode():
            for b in range(n_batches):
                batch = records[b * batch_size: (b + 1) * batch_size]
                bsz = len(batch)
                ids = torch.arange(bsz, device=env.device)
                origins = env.scene.env_origins[ids]

                # `batch` entries are each ONE state (flatten_records already unbundled any record
                # that captured >1 simultaneous accept), so every per-record tensor here has NO
                # leading dim -- stack across the batch directly, never index into a record's tensor.
                robot_root_pose = torch.stack([r["full_state"]["articulation"]["robot"]["root_pose"] for r in batch]).to(env.device)
                robot_root_pose[:, :3] += origins
                robot_root_vel = torch.stack([r["full_state"]["articulation"]["robot"]["root_velocity"] for r in batch]).to(env.device)
                q = torch.stack([r["full_state"]["articulation"]["robot"]["joint_position"] for r in batch]).to(env.device)
                qd = torch.stack([r["full_state"]["articulation"]["robot"]["joint_velocity"] for r in batch]).to(env.device)
                recorded_target = torch.stack([r["target"] for r in batch]).to(env.device)

                obj_root_pose = torch.stack([r["full_state"]["rigid_object"]["object"]["root_pose"] for r in batch]).to(env.device)
                obj_root_pose[:, :3] += origins
                obj_root_vel = torch.stack([r["full_state"]["rigid_object"]["object"]["root_velocity"] for r in batch]).to(env.device)

                target = recorded_target if target_mode == "recorded" else q.clone()
                vel_target = qd.clone() if vel_mode == "recorded" else torch.zeros_like(qd)

                robot.write_root_pose_to_sim(robot_root_pose, env_ids=ids)
                robot.write_root_velocity_to_sim(robot_root_vel, env_ids=ids)
                robot.write_joint_state_to_sim(q, qd, env_ids=ids)
                robot.set_joint_position_target(target, env_ids=ids)
                robot.set_joint_velocity_target(vel_target, env_ids=ids)
                obj.write_root_pose_to_sim(obj_root_pose, env_ids=ids)
                obj.write_root_velocity_to_sim(obj_root_vel, env_ids=ids)
                env.scene.write_data_to_sim()
                env.sim.forward()

                baseline_in_hand = in_hand()[ids].clone()
                gap0 = (robot.data.joint_pos_target[ids] - robot.data.joint_pos[ids]).abs()
                per_hand[0].append(gap0[:, hand_ids].cpu())
                per_arm[0].append(gap0[:, arm_ids].cpu())
                per_drift[0].append(torch.zeros(bsz))

                for sub in range(hold_substeps):
                    # Mirrors ManagerBasedRLEnv.step's own per-substep loop, MINUS
                    # action_manager.apply_action() -- see module docstring's Phase 2 section for why
                    # write_data_to_sim() alone (which calls Articulation._apply_actuator_model()
                    # internally) is what keeps this physically honest for BOTH actuator types
                    # without ever recomputing joint_pos_target from a new action.
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(dt=env.physics_dt)

                    gap = (robot.data.joint_pos_target[ids] - robot.data.joint_pos[ids]).abs()
                    per_hand[sub + 1].append(gap[:, hand_ids].cpu())
                    per_arm[sub + 1].append(gap[:, arm_ids].cpu())
                    drift_mm = torch.linalg.norm(in_hand()[ids] - baseline_in_hand, dim=1) * 1e3
                    per_drift[sub + 1].append(drift_mm.cpu())

        hand_traj = [torch.cat(s, dim=0) for s in per_hand]
        arm_traj = [torch.cat(s, dim=0) for s in per_arm]
        drift_traj = [torch.cat(s, dim=0) for s in per_drift]
        return hand_traj, arm_traj, drift_traj

    def print_hold_table(
        label: str, target_mode: str, vel_mode: str,
        hand_traj: list[torch.Tensor], arm_traj: list[torch.Tensor], drift_traj: list[torch.Tensor],
        substeps_to_print=None,
    ) -> float:
        n = drift_traj[0].shape[0]
        held_final = drift_traj[-1] < HOLD_MM
        print(f"\n[result] === ARM {label}  (n_states={n}) ===", flush=True)
        print("[result] substep    t(ms)  HAND gap median/p90 (rad)   ARM gap median/p90 (rad)   in-palm drift median/p90 (mm)",
              flush=True)
        for s in (substeps_to_print if substeps_to_print is not None else range(len(hand_traj))):
            hg = hand_traj[s].reshape(-1)
            ag = arm_traj[s].reshape(-1)
            dr = drift_traj[s]
            t_ms = s / 120 * 1000
            hg_med, hg_p90 = torch.median(hg).item(), torch.quantile(hg, 0.9).item()
            ag_med, ag_p90 = torch.median(ag).item(), torch.quantile(ag, 0.9).item()
            dr_med, dr_p90 = torch.median(dr).item(), torch.quantile(dr, 0.9).item()
            print(
                f"[result]   {s:3d}    {t_ms:7.1f}   {hg_med:.4f} / {hg_p90:.4f}"
                f"              {ag_med:.4f} / {ag_p90:.4f}"
                f"              {dr_med:.2f} / {dr_p90:.2f}",
                flush=True,
            )
            # -- MACHINE-READABLE, one line per (arm, substep), fixed field names (key=value, space-
            # separated) so this can be re-analysed with `grep '^\[data\] '` + a simple parser without
            # re-running a ~15 minute boot. The human-readable [result] table above is for reading
            # now; this line is for reading later.
            print(
                f"[data] arm={label} target_mode={target_mode} vel_mode={vel_mode} substep={s} t_ms={t_ms:.4f} "
                f"hand_gap_median_rad={hg_med:.6f} hand_gap_p90_rad={hg_p90:.6f} "
                f"arm_gap_median_rad={ag_med:.6f} arm_gap_p90_rad={ag_p90:.6f} "
                f"drift_mm_median={dr_med:.6f} drift_mm_p90={dr_p90:.6f} n_states={n}",
                flush=True,
            )
        held_frac = float(held_final.float().mean())
        print(f"[result]   STILL HELD at end (in-palm drift < {HOLD_MM:.0f}mm): {int(held_final.sum())}/{n} = {held_frac:.3f}",
              flush=True)
        print(
            f"[data_summary] arm={label} target_mode={target_mode} vel_mode={vel_mode} "
            f"hold_substeps={len(hand_traj) - 1} n_states={n} n_held={int(held_final.sum())} held_frac={held_frac:.6f} "
            f"hold_mm_threshold={HOLD_MM:.1f}",
            flush=True,
        )
        return held_frac

    print(f"\n=== PHASE 2: four-arm sweep, {args_cli.hold_substeps} physics substeps"
          f" = {args_cli.hold_substeps / 120 * 1000:.1f}ms, SAME {len(FLAT_RECORDS)} accepted states across all arms"
          f" (never re-drawn) ===", flush=True)

    ARMS = [
        ("A_target-recorded_vel-recorded", "recorded", "recorded"),
        ("B_target-recorded_vel-zero", "recorded", "zero"),
        ("C_target-q_vel-recorded", "q", "recorded"),
        ("D_target-q_vel-zero_(todays_replay_convention)", "q", "zero"),
    ]
    arm_held_frac: dict[str, float] = {}
    for label, target_mode, vel_mode in ARMS:
        hand_traj, arm_traj, drift_traj = run_hold_arm(FLAT_RECORDS, target_mode, vel_mode, args_cli.hold_substeps)
        arm_held_frac[label] = print_hold_table(label, target_mode, vel_mode, hand_traj, arm_traj, drift_traj)

    print("\n[result] === FOUR-ARM SUMMARY (held fraction @ end of window) ===", flush=True)
    for label, _, _ in ARMS:
        print(f"[result]   {label:50s}  held={arm_held_frac[label]:.3f}", flush=True)

    # ================================================================================================
    # PHASE 3 (optional): does whichever arm held best at the Phase-2 window also hold for longer?
    # Re-uses the SAME accepted states again (no re-draw), runs ONE hold for max(checkpoints), and
    # reads the shorter checkpoints out of that single trajectory -- not three separate runs.
    # ================================================================================================
    if not args_cli.skip_duration_sweep:
        try:
            checkpoints = sorted(int(x) for x in args_cli.duration_sweep_substeps.split(","))
        except ValueError as exc:
            raise SystemExit(f"[probe] --duration_sweep_substeps did not parse: {exc}")
        best_label, best_target_mode, best_vel_mode = max(
            ((label, target_mode, vel_mode) for label, target_mode, vel_mode in ARMS),
            key=lambda item: arm_held_frac[item[0]],
        )
        duration_steps = max(checkpoints)
        print(f"\n=== PHASE 3 (optional): duration sweep on best-held arm '{best_label}'"
              f" (held={arm_held_frac[best_label]:.3f} @ {args_cli.hold_substeps} substeps),"
              f" one hold of {duration_steps} substeps, checkpoints {checkpoints} ===", flush=True)
        hand_traj, arm_traj, drift_traj = run_hold_arm(FLAT_RECORDS, best_target_mode, best_vel_mode, duration_steps)
        print_hold_table(f"{best_label}_duration_sweep", best_target_mode, best_vel_mode, hand_traj, arm_traj, drift_traj,
                          substeps_to_print=[s for s in checkpoints if s <= duration_steps])
    else:
        print("\n[probe] --skip_duration_sweep set: Phase 3 skipped.", flush=True)

    print("\nPROBE_OK", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
