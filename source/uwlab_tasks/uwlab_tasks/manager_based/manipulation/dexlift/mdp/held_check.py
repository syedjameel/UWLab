# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stateful IsaacLab termination term wrapping :mod:`held_check_core` (bead UWLab-dwx.2, probe
re-arming fixed under dwx.6/dwx.7's STEP 1 diagnosis).

Reads dexlift's own scene entities (the fingertip ContactSensors, the DELTO palm body, the object)
each step and calls :func:`held_check_core.held_decision`. Nothing decision-relevant lives here --
see that module for the actual gates and their unit tests, which need no Isaac process at all.

THE PROBE ITSELF IS NOT COMMANDED HERE. A termination term only reads env state; it has no write
access to the action pipeline. The generator script adds the small arm-joint bias WHENEVER
:attr:`probe_active` is True for a given env (not a fixed global window any more, see below) --
this term only MEASURES what happened (palm and object displacement over that window) and decides
whether the object tracked it.

RE-ARMING PROBE, NOT A ONE-SHOT FIXED WINDOW. The original design anchored the probe to a single
absolute window, SETTLE_STEPS..SETTLE_STEPS+PROBE_STEPS (steps 60-70), once per episode. Measured
directly (scripts_v2/tools/diagnose_held_timing.py, 198 episodes of the certified Stage-2 lift
checkpoint, dwx.6/dwx.7 STEP 1): the probe-independent gates (settled & opposed_contact & co_move)
were true at SOME point in 26.3% of episodes but at natural episode-end in 0.0% of them, and of the
episodes that ever reached that state, the FIRST time was after step 70 in 100% of cases (min 75,
median 153, out of a 240-step episode) -- i.e. the fixed window measured the wrong moment on every
single episode that would otherwise have passed. A checkpoint graded by a STICKY-OR over the whole
episode cannot be honestly measured by a probe that only ever looks at one early instant.

The fix: the probe now ARMS itself whenever (settled & opposed_contact & co_move) is newly true and
no probe is currently in flight, wherever in the episode that happens, captures a start snapshot,
runs for PROBE_STEPS, and DECIDES. If it fails (object didn't track), the probe is free to re-arm
again on the NEXT qualifying moment -- unlike the old ``_probe_ready`` latch, which never re-checked
after its one shot. This makes ``held_with_probe``'s own return value, wired as a terminating
DoneTerm, an honest implementation of "record state when held becomes true" (bead dwx.6/dwx.7 STEP
1's option (a)): RecorderManager's "success" predicate MUST be delivered as a termination term
(P1), and any termination term unconditionally ends the episode the step it returns True
(termination_manager.py:182) -- so option (a) was never something to newly build on top of this
plumbing, it falls out of it, PROVIDED the probe can actually reach the moment the policy is doing
well. The re-arming is what makes that true instead of accidentally false.

Wire this as ``terminations.success`` on the dexlift env variant the generator constructs
(IsaacLab's RecorderManager.record_pre_reset reads a term literally named "success" -- see the
P1 recorder-seam notes on bead UWLab-dwx.2). AND it with whatever other safety gates the host env
already declares (abnormal_robot / object_out_of_bound) rather than trusting this term alone to
notice a NaN or a flung object -- see the generator script for how that AND is applied.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .held_check_core import PROBE_STEPS, SETTLE_STEPS, held_decision
from .rewards import _sensor_force_magnitudes  # reused, not reimplemented -- see rewards.py:40-76

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg


class held_with_probe(ManagerTermBase):
    """The dwx.2 HELD check: settled & opposed-contact & co-move & probe-displacement-tracks.

    Params (all optional, defaults match the bead's adopted design):
        robot_cfg: SceneEntityCfg naming the palm body (``body_ids`` must resolve to exactly one
            body -- the DELTO's ``rl_dg_mount``).
        object_cfg: SceneEntityCfg for the manipulated object (default ``SceneEntityCfg("object")``).
        thumb_contact_names / tip_contact_names: fingertip sensor names, as
            ``mdp.rewards.contacts`` takes them (default dexlift's own THUMB_TIP_NAMES/TIP_NAMES).
        force_threshold: N, per-fingertip normal-force gate (default 0.2, matches
            ``good_finger_contact`` / probe_grasp.py's own calibration).
        comove_speed_thresh, comove_vz_thresh, probe_min_disp, probe_track_tol: see
            ``held_check_core.held_decision``.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self.robot_cfg: SceneEntityCfg = p.get("robot_cfg", SceneEntityCfg("robot", body_names="rl_dg_mount"))
        self.object_cfg: SceneEntityCfg = p.get("object_cfg", SceneEntityCfg("object"))
        self.thumb_contact_names = p.get("thumb_contact_names", ("rl_dg_1_tip", "rl_dg_5_tip"))
        self.tip_contact_names = p.get("tip_contact_names", ("rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip"))
        self.force_threshold = p.get("force_threshold", 0.2)
        self.comove_speed_thresh = p.get("comove_speed_thresh", 0.05)
        self.comove_vz_thresh = p.get("comove_vz_thresh", 0.1)
        self.probe_min_disp = p.get("probe_min_disp", 0.003)
        self.probe_track_tol = p.get("probe_track_tol", 0.003)
        self.probe_track_tol_frac = p.get("probe_track_tol_frac", 0.3)
        self.settle_steps = p.get("settle_steps", SETTLE_STEPS)
        self.probe_steps = p.get("probe_steps", PROBE_STEPS)

        self.robot_cfg.resolve(env.scene)
        self.object_cfg.resolve(env.scene)
        assert len(self.robot_cfg.body_ids) == 1, (
            f"held_with_probe.robot_cfg must resolve to exactly one body, got {self.robot_cfg.body_ids}"
        )
        self._palm_id = self.robot_cfg.body_ids[0]

        n = env.num_envs
        device = env.device
        # Per-env probe bookkeeping, RE-ARMING (see module docstring):
        #   probe_active   -- a probe window is currently in flight for this env.
        #   _probe_ready   -- the MOST RECENT probe has completed and its displacement is valid.
        #                      Read by held_decision() this step only; a FAILED probe re-arms on
        #                      the next qualifying moment (see __call__), a SUCCESSFUL one ends the
        #                      episode via the DoneTerm before another can start.
        #   _probe_start_step -- episode step the current probe opened, to know when PROBE_STEPS
        #                      have elapsed.
        self.probe_active = torch.zeros(n, dtype=torch.bool, device=device)
        self._probe_ready = torch.zeros(n, dtype=torch.bool, device=device)
        self._probe_start_step = torch.zeros(n, dtype=torch.long, device=device)
        self._obj_pos_at_probe_start = torch.zeros(n, 3, device=device)
        self._palm_pos_at_probe_start = torch.zeros(n, 3, device=device)
        self._obj_disp_probe = torch.zeros(n, 3, device=device)
        self._gripper_disp_probe = torch.zeros(n, 3, device=device)
        self._last_breakdown: dict[str, torch.Tensor] = {}
        # Cumulative diagnostics for the generator's end-of-run summary -- not used in any
        # decision, plain python ints incremented in __call__.
        self.n_probes_armed = 0
        self.n_probes_finalized = 0
        self.n_probes_tracked = 0

    def _tracks(self, obj_disp: torch.Tensor, gripper_disp: torch.Tensor) -> torch.Tensor:
        """Same relative-tolerance formula ``held_decision`` uses internally, exposed here so the
        breakdown cache and the debug print never compute a DIFFERENT answer than the actual
        decision does. See ``held_check_core.held_decision``'s docstring for why the tolerance is
        relative to the realised jog size, not a fixed absolute number."""
        gripper_norm = torch.linalg.norm(gripper_disp, dim=-1)
        mismatch = torch.linalg.norm(obj_disp - gripper_disp, dim=-1)
        tol = torch.clamp(self.probe_track_tol_frac * gripper_norm, min=self.probe_track_tol)
        return mismatch < tol

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.probe_active[env_ids] = False
        self._probe_ready[env_ids] = False
        self._probe_start_step[env_ids] = 0
        self._obj_pos_at_probe_start[env_ids] = 0.0
        self._palm_pos_at_probe_start[env_ids] = 0.0
        self._obj_disp_probe[env_ids] = 0.0
        self._gripper_disp_probe[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        robot = env.scene[self.robot_cfg.name]
        obj = env.scene[self.object_cfg.name]

        obj_pos = obj.data.root_pos_w
        palm_pos = robot.data.body_pos_w[:, self._palm_id, :]
        steps = env.episode_length_buf

        # -- the three non-probe gates, computed fresh every step (cheap, and lets `held` track
        # contact loss immediately rather than latching a stale True).
        thumb_force = _sensor_force_magnitudes(env, self.thumb_contact_names)
        tip_force = _sensor_force_magnitudes(env, self.tip_contact_names)
        thumb_loaded = thumb_force.gt(self.force_threshold).any(dim=-1)
        tip_loaded = tip_force.gt(self.force_threshold).any(dim=-1)

        relative_speed = torch.linalg.vector_norm(
            obj.data.root_lin_vel_w - robot.data.body_lin_vel_w[:, self._palm_id, :], dim=-1
        )
        obj_vz = obj.data.root_lin_vel_w[:, 2]

        settled = steps > self.settle_steps
        pre_probe_ok = (
            settled
            & thumb_loaded
            & tip_loaded
            & (relative_speed < self.comove_speed_thresh)
            & (obj_vz.abs() < self.comove_vz_thresh)
        )

        # -- RE-ARMING probe bookkeeping (see module docstring for why this replaced a fixed
        # single window). ARM: no probe currently active, and the three non-probe gates just
        # became satisfied -- capture a start snapshot, wherever in the episode this happens.
        # FINALIZE: PROBE_STEPS after arming, compute displacement and clear `probe_active` so a
        # FAILED attempt can arm again on the next qualifying moment.
        arm = pre_probe_ok & ~self.probe_active
        if arm.any():
            self._obj_pos_at_probe_start[arm] = obj_pos[arm]
            self._palm_pos_at_probe_start[arm] = palm_pos[arm]
            self._probe_start_step[arm] = steps[arm]
            self.probe_active[arm] = True
            self._probe_ready[arm] = False  # this step's decision must not see a stale prior probe
            self.n_probes_armed += int(arm.sum())

        finalize = self.probe_active & (steps >= (self._probe_start_step + self.probe_steps))
        if finalize.any():
            self._obj_disp_probe[finalize] = obj_pos[finalize] - self._obj_pos_at_probe_start[finalize]
            self._gripper_disp_probe[finalize] = palm_pos[finalize] - self._palm_pos_at_probe_start[finalize]
            self._probe_ready[finalize] = True
            self.probe_active[finalize] = False
            self.n_probes_finalized += int(finalize.sum())
            tracked_now = self._tracks(self._obj_disp_probe, self._gripper_disp_probe)
            self.n_probes_tracked += int((finalize & tracked_now).sum())
            if os.environ.get("HELD_CHECK_DEBUG_PROBE"):
                for i in torch.nonzero(finalize).flatten().tolist():
                    g = self._gripper_disp_probe[i].norm().item()
                    o = self._obj_disp_probe[i].norm().item()
                    print(f"[held_check DEBUG] env {i} probe finalize: |gripper_disp|={g*1000:.2f}mm "
                          f"|obj_disp|={o*1000:.2f}mm tracked={bool(tracked_now[i])}", flush=True)

        held = held_decision(
            steps_since_reset=steps,
            thumb_loaded=thumb_loaded,
            tip_loaded=tip_loaded,
            relative_speed=relative_speed,
            obj_vz=obj_vz,
            probe_ready=self._probe_ready,
            obj_disp_probe=self._obj_disp_probe,
            gripper_disp_probe=self._gripper_disp_probe,
            settle_steps=self.settle_steps,
            comove_speed_thresh=self.comove_speed_thresh,
            comove_vz_thresh=self.comove_vz_thresh,
            probe_min_disp=self.probe_min_disp,
            probe_track_tol=self.probe_track_tol,
            probe_track_tol_frac=self.probe_track_tol_frac,
        )

        # -- CACHE the per-gate breakdown computed THIS step, for gate_breakdown() to read later.
        # Load-bearing ordering point: ManagerBasedRLEnv auto-resets done envs INSIDE the same
        # env.step() call that computed `held` (episode_length_buf is zeroed and this term's own
        # .reset() runs before step() returns to the caller). A caller that reads live env state
        # AFTER step() returns -- as the generator script originally did -- sees POST-reset state
        # for every env that just finished, misattributing every rejection to "not yet settled"
        # regardless of what actually happened. Caching here, at the moment __call__ actually runs
        # (before any reset), is what RecorderManager.record_pre_reset itself relies on when it
        # reads get_term("success") -- termination buffers are overwritten fresh each compute(), not
        # cleared by reset, so a cache written here survives exactly as long as that buffer does.
        gripper_moved = torch.linalg.norm(self._gripper_disp_probe, dim=-1) > self.probe_min_disp
        tracks = self._tracks(self._obj_disp_probe, self._gripper_disp_probe)
        self._last_breakdown = {
            "settled": steps > self.settle_steps,
            "opposed_contact": thumb_loaded & tip_loaded,
            "co_move": (relative_speed < self.comove_speed_thresh) & (obj_vz.abs() < self.comove_vz_thresh),
            "probe_ready": self._probe_ready.clone(),
            "probe_gripper_moved": gripper_moved,
            "probe_tracks": tracks,
        }

        # -- AND with the host env's own generic safety gates, mirroring OmniReset's
        # check_reset_state_success (collision/stability/no-NaN folded INTO the success bit itself,
        # not left to coincidentally not have fired the same step). check_reset_state_success
        # itself does not transplant onto dexlift's scene (it is parameterized over
        # insertive_object/receptive_object/collision_analyzer_cfgs that do not exist here); the
        # nearest structurally-correct equivalent already declared on this env's own
        # TerminationsCfg is `abnormal_robot` (|joint vel| > 2x limit) and `object_out_of_bound`.
        # Read their CURRENT-STEP values (TerminationManager evaluates dataclass fields in
        # declaration order and this term is declared last, so both are already computed) rather
        # than trust neither fired just because held's own gates are already strict.
        tm = env.termination_manager
        for name in ("abnormal_robot", "object_out_of_bound"):
            if name in tm.active_terms:
                held = held & ~tm.get_term(name)

        return held

    # -- diagnostics for the generator's per-term rejection breakdown (bead requirement #3). Reads
    # the cache __call__ wrote THIS step -- NOT a live recompute -- because by the time the
    # generator script calls this (after env.step() has returned), done envs have already been
    # auto-reset and live env state would silently describe next episode's first step instead of
    # the one that just ended. See the comment on `self._last_breakdown` in __call__.
    def gate_breakdown(self, env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
        return self._last_breakdown
