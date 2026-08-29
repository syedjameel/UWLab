# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-tensor core of the policy-driven generator's HELD check (bead UWLab-dwx.2).

Deliberately has NO isaaclab / omni import anywhere in this file. Every quantity the decision
needs is passed in as a plain tensor, so :func:`held_decision` can be, and is, unit-tested with
synthetic data and no Isaac process -- see ``test_held_check_core.py`` in this directory.
:mod:`held_check` (a sibling module) is the thin IsaacLab ``ManagerTermBase`` wrapper that reads
these tensors off ``env.scene`` each step and calls this function; nothing decision-relevant lives
there that isn't here.

WHY A FOURTH GATE. Three gates -- settled, opposed contact, co-moving velocity -- are ALL satisfied
by the adversarial case this check exists to reject: an object already sitting at (or near) its
target pose, with the hand resting nearby but not actually gripping it. Settled is trivially true
(nothing is moving). Co-move is trivially true (relative velocity ~0 when both are at rest). Contact
can ALSO be true: an idle hand posed near an object can have a thumb pad and a fingertip pad both
resting against the surface with light residual normal force above a naive threshold, without any
real opposed squeeze holding it. Only the fourth gate -- does the object's displacement TRACK the
gripper's own commanded displacement during a probe -- distinguishes "resting against" from "held
by". See :func:`held_decision`'s docstring for the exact adversarial construction this catches.
"""

from __future__ import annotations

import torch

# Shared between held_check.py (the stateful term) and the generator script, so the two cannot
# drift apart on when the probe happens. Torch-only file, so both a plain unit test and the
# Isaac-side term import the SAME constants rather than each restating them.
SETTLE_STEPS = 60
"""Episode steps before anything is trusted -- kills the ballistic post-reset-drop false-positive
bucket (same idea as probe_grasp.py's SETTLE_STEPS)."""

PROBE_STEPS = 10
"""Duration (env steps) of the probe jog, starting the step after SETTLE_STEPS."""

PROBE_ARM_ACTION_BIAS = 0.15
"""Constant relative-joint-position bias the generator adds to all six arm action dims (on top of
whatever the policy commands) during the probe window, in units of the arm action term's own scale
(|action| <= 1 post-clip, see dexlift_ur5e_delto_actions.py's UR5E_ARM_ACTION_CLIP).

MEASURED, not guessed: the first value tried here was 0.5 (half the clip ceiling), on the theory
that it needed to be "large enough to produce a measurable palm displacement without saturating the
clip". Logged with HELD_CHECK_DEBUG_PROBE=1 against the certified Stage-2 lift checkpoint, that bias
over PROBE_STEPS=10 steps produced 11-130 mm of actual palm displacement -- one to two orders of
magnitude past "jog the gripper a few mm" (dwx.2's own bead notes), because RelativeJointPositionAction
integrates a SUSTAINED per-step command over multiple steps, not a single delta. 0.15 was chosen to
bring that back toward a few-mm-to-low-tens-of-mm regime; PROBE_TRACK_TOL_FRAC below is the primary
robustness fix regardless (a relative criterion tolerates whatever the jog size turns out to be in
practice), but an excessively large jog is still worth avoiding on its own terms: it risks yanking a
marginal-but-real grasp apart with the probe itself, which is not what "measure, don't disturb" means."""


COMOVE_SPEED_THRESH = 0.05
"""m/s ceiling on ``|v_obj - v_palm|`` for the co-move gate.

NAMED, not new. This is the same 0.05 that was already the default of :func:`held_decision` and
:func:`passive_gates`; nothing about the value changes, and nothing that consumed the default
before behaves differently. It is given a name so that a CONSUMER OUTSIDE THE DECISION -- the
training-time gate proxy's banner (``mdp/gate_proxy.py``) -- can print the threshold it is actually
using instead of a literal that goes stale (V2_POSE_FINDINGS.md F42). Do not retune it here: it is
read by both the generator's held predicate and the proxy that is supposed to predict it, and a
change needs both re-validated."""

COMOVE_VZ_THRESH = 0.1
"""m/s ceiling on ``|obj_vz|`` for the co-move gate. Named for the same reason and under the same
caution as :data:`COMOVE_SPEED_THRESH`; the value is unchanged."""


def passive_gates(
    steps_since_reset: torch.Tensor,
    thumb_loaded: torch.Tensor,
    tip_loaded: torch.Tensor,
    relative_speed: torch.Tensor,
    obj_vz: torch.Tensor,
    settle_steps: int = SETTLE_STEPS,
    comove_speed_thresh: float = COMOVE_SPEED_THRESH,
    comove_vz_thresh: float = COMOVE_VZ_THRESH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three gates of :func:`held_decision` that need NO probe: ``(settled, opposed_contact,
    co_move)``, each ``(N,)`` bool.

    WHY THIS IS A SEPARATE FUNCTION AND NOT A COPY. These three are the only gates in the chain
    that can be evaluated during TRAINING. The fourth gate needs a probe, and the probe is an
    action bias the generator's rollout loop injects on top of the policy
    (:data:`PROBE_ARM_ACTION_BIAS`) -- injecting it during training would perturb the thing being
    trained. So a training-time predictor of the generator's gate-chain pass rate can only be built
    from these three (``V2_REPOSE_RECIPE.md`` sec 4.1), and it must agree with the generator
    EXACTLY or it predicts nothing.

    WHAT CHANGED 2026-08-29, AND WHAT IT DOES TO THE PREDICTOR. ``co_move`` no longer gates
    :func:`held_decision` (see its docstring for the physical argument). It is still computed and
    still returned here, on purpose. But the composition of any predictor built on these three has
    therefore changed: **the chain now requires only two of them** -- ``settled`` and
    ``opposed_contact`` -- plus a probe gate that no training-time proxy can evaluate at all.

    So a proxy that ANDs all three is no longer predicting what the chain requires: it is strictly
    PESSIMISTIC, because it still charges episodes for a gate acceptance no longer consults, and
    ``co_move`` was the dominant rejector at 46.8%. A proxy reading ``settled & opposed_contact``
    predicts the passive half of the current chain; ``co_move``'s own per-gate fraction remains
    worth logging precisely because nobody yet knows whether it was doing work.

    This function's signature and return are deliberately UNCHANGED, so that consumers keep getting
    the third boolean and can decide for themselves what to do with it. Nothing here silently
    redefines what a caller receives.

    "Agree exactly" is why this is a factored-out function rather than the same five lines written
    again in the logging term. ``V2_POSE_FINDINGS.md`` F27 is a recurring defect of this project --
    a constant or a definition established in one place and restated in another, each individually
    valid, wrong only against each other, with nothing checking. Importing the thresholds and
    re-writing the comparisons would still leave two implementations. There is now one:
    :func:`held_decision` calls this, and so does the training-time logger. A change to either
    threshold or to either comparison reaches both by construction.

    Args:
        steps_since_reset: ``(N,)``, e.g. ``env.episode_length_buf``.
        thumb_loaded: ``(N,)`` bool, >=1 thumb-side fingertip above the force threshold.
        tip_loaded: ``(N,)`` bool, >=1 non-thumb fingertip above the force threshold.
        relative_speed: ``(N,)`` ``|v_obj - v_palm|`` in m/s.
        obj_vz: ``(N,)`` object world-frame z linear velocity in m/s.
        settle_steps: episode-length threshold before anything is trusted.
        comove_speed_thresh: m/s ceiling on ``relative_speed``.
        comove_vz_thresh: m/s ceiling on ``|obj_vz|``.

    Returns:
        ``(settled, opposed_contact, co_move)``, in the priority order the generator's rejection
        histogram reports them (``V2_POSE_FINDINGS.md`` F28/F30). ``co_move`` is the dominant
        rejector on both band-gated v1 rungs -- 46.8% and 53.2% of all attempts.
    """
    settled = steps_since_reset > settle_steps
    opposed_contact = thumb_loaded & tip_loaded
    co_move = (relative_speed < comove_speed_thresh) & (obj_vz.abs() < comove_vz_thresh)
    return settled, opposed_contact, co_move


def held_decision(
    steps_since_reset: torch.Tensor,
    thumb_loaded: torch.Tensor,
    tip_loaded: torch.Tensor,
    relative_speed: torch.Tensor,
    obj_vz: torch.Tensor,
    probe_ready: torch.Tensor,
    obj_disp_probe: torch.Tensor,
    gripper_disp_probe: torch.Tensor,
    settle_steps: int = 60,
    comove_speed_thresh: float = COMOVE_SPEED_THRESH,
    comove_vz_thresh: float = COMOVE_VZ_THRESH,
    probe_min_disp: float = 0.003,
    probe_track_tol: float = 0.003,
    probe_track_tol_frac: float = 0.3,
) -> torch.Tensor:
    """AND of four gates. Returns a ``(num_envs,)`` bool tensor.

    Args:
        steps_since_reset: ``(N,)`` int/float, e.g. ``env.episode_length_buf``.
        thumb_loaded: ``(N,)`` bool, >=1 thumb-side fingertip above the force threshold.
        tip_loaded: ``(N,)`` bool, >=1 non-thumb fingertip above the force threshold. Combined with
            ``thumb_loaded`` this is dexlift's own opposition gate (``mdp.rewards.contacts``) --
            reused, not reimplemented; see ``held_check.py``.
        relative_speed: ``(N,)`` ``|v_obj - v_palm|`` (m/s), the ``held_lift_stability`` term's own
            quantity (``table_leg.py:460-496``), reused the same way.
        obj_vz: ``(N,)`` object world-frame z linear velocity (m/s).
        probe_ready: ``(N,)`` bool, whether a probe window has completed for this env since the
            last reset (False before the first probe finishes -- held is undecided, not true).
        obj_disp_probe: ``(N, 3)`` object position delta measured over the most recent probe
            window.
        gripper_disp_probe: ``(N, 3)`` COMMANDED (or measured) gripper displacement over that same
            window -- the perturbation the generator's rollout loop injected on top of the policy's
            own actions. See ``held_check.py`` / the generator script for how this is produced.
        settle_steps: episode-length threshold before anything is trusted (kills the ballistic
            post-reset-drop false-positive bucket, same idea as ``probe_grasp.py``'s SETTLE_STEPS).
        comove_speed_thresh: m/s, ceiling on ``relative_speed`` for the co-move gate.
        comove_vz_thresh: m/s, ceiling on ``|obj_vz|`` for the co-move gate.
        probe_min_disp: m, the gripper must actually have moved at least this much during the probe
            for the probe to be considered meaningful -- guards against a degenerate near-zero jog
            making ``obj_disp ~= gripper_disp`` trivially true because both are ~zero.
        probe_track_tol: m, ABSOLUTE floor on the allowed ``|obj_disp_probe - gripper_disp_probe|``
            mismatch, for small jogs.
        probe_track_tol_frac: fraction of ``|gripper_disp_probe|`` ALSO allowed as mismatch,
            whichever of the two is larger. Real jog magnitude varies with how hard the arm has to
            work against whatever it's carrying (measured 11-130mm for a jog "intended" to be a few
            mm -- RelativeJointPositionAction integrates a sustained command over multiple steps,
            not a single delta), so a fixed absolute tolerance calibrated for a small jog rejects
            genuinely-tracking objects whenever the realised jog is larger: a 1-2mm gap on a 40mm
            jog is normal actuation/compliance slack, not evidence of a loose grip, but the SAME
            1-2mm gap would be everything on a 3mm jog. The floor (``probe_track_tol``) is what
            still catches a non-tracking object on however small a jog actually occurs.

    Returns:
        ``(N,)`` bool: held := settled & opposed_contact & (probe_ready & gripper_moved & tracks).

    WHY ``co_move`` IS NOT IN THIS AND (user decision, 2026-08-29). It is still computed here and
    still reported through ``held_check.gate_breakdown``; it no longer votes.

    * ``co_move`` demanded the object move RIGIDLY with the hand. A leg seated in a bore is
      CONSTRAINED BY THE FIXTURE, so palm motion necessarily produces relative velocity -- the gate
      therefore rejects genuinely-held states for doing the very thing the task asks. It was the
      dominant rejector: 46.8% of all attempts on S1 (``V2_POSE_FINDINGS.md`` F28/F30).
    * It was also never the gate that catches the adversarial case below. This function's own
      brief says gates 1-3 are ALL satisfied by that case -- object at target, hand resting nearby
      with incidental contact, nothing moving. Only the probe separates "resting against" from
      "held by", so ``tracks`` is what rejects it, not ``co_move``.

      SCOPE OF THAT CLAIM, stated exactly. ``test_adversarial_case_rejected`` shows that GIVEN a
      finalized probe, ``tracks`` alone rejects the adversarial case. It passes ``probe_ready=True``
      in directly, so it never exercises the ARMING path in ``held_check.py`` and cannot observe
      whether something else gates acceptance upstream of this function. It did not, in fact:
      ``co_move`` was dropped from this AND and left in ``held_check``'s ``pre_probe_ok``, so
      acceptance still required a co_move-true instant until that was fixed. The upstream half is
      pinned separately by
      ``test_held_check_core.py::test_probe_arming_does_not_reintroduce_a_dropped_gate``, which
      compares this function's returned expression against that arming condition -- the two must
      not disagree about what is required.

    WHY ``gripper_moved`` STAYED, when the same change was first scoped to drop it too. It is not
    an acceptance gate. It is a VALIDITY CHECK ON THE MEASUREMENT: it does not judge the grasp, it
    judges whether the probe happened. Dropping it would not loosen a criterion, it would make
    ``tracks`` report on a measurement that was never taken -- and it would do so in the very change
    that makes ``tracks`` the ONLY remaining gate. An env whose probe failed to jog would then be
    accepted and would look identical to a clean accept in every log this project has.
    ``test_degenerate_zero_jog_does_not_trivially_pass`` is the test that catches this; it failed
    under the drop-both variant and was NOT weakened to accommodate it.

    IF TABLE-LEG MOTION BECOMES A PROBLEM, RAISE THE FORCE THRESHOLD IN ``opposed_contact``.
    Do NOT reinstate ``co_move``. This is the user's own stated fallback and it is recorded here so
    that nobody restores the gate by reflex when a run looks noisy.

    THE ADVERSARIAL CASE THIS MUST REJECT (Option-B, per bead notes): construct an env where the
    object is already at/near its target pose, the hand is posed nearby with light incidental
    contact (thumb_loaded=True, tip_loaded=True from residual touch, not a real pinch), nothing is
    moving (steps_since_reset large, relative_speed~0, obj_vz~0) -- i.e. gates 1-3 all pass -- but
    the object is NOT actually attached to the hand. When the generator's probe jogs the gripper by
    ``gripper_disp_probe``, an untouched-but-nearby object does not move with it:
    ``obj_disp_probe`` stays near zero while ``gripper_disp_probe`` does not, so
    ``|obj_disp_probe - gripper_disp_probe| >= probe_track_tol`` and the probe gate -- and
    therefore the whole AND -- is False. See
    ``test_held_check_core.py::test_adversarial_case_rejected``. That test is now load-bearing in a
    way it was not before: with ``co_move`` dropped, ``tracks`` is the ONLY gate standing between
    this function and the case it exists to reject.
    """
    settled, opposed_contact, co_move = passive_gates(
        steps_since_reset=steps_since_reset,
        thumb_loaded=thumb_loaded,
        tip_loaded=tip_loaded,
        relative_speed=relative_speed,
        obj_vz=obj_vz,
        settle_steps=settle_steps,
        comove_speed_thresh=comove_speed_thresh,
        comove_vz_thresh=comove_vz_thresh,
    )

    gripper_disp_norm = torch.linalg.norm(gripper_disp_probe, dim=-1)
    # co_move is STILL COMPUTED, DELIBERATELY NOT GATING (user decision 2026-08-29). It remains
    # available to held_check.gate_breakdown and to the reach-count histogram, because logging it is
    # free and reversible and is the only way anyone learns whether it was doing work. What was
    # removed is its vote, not its computation.
    #
    # gripper_moved STAYS IN THE CHAIN, and the distinction is the whole reason: it is not an
    # acceptance gate, it is a VALIDITY CHECK ON THE MEASUREMENT. It does not judge the grasp; it
    # judges whether the probe happened at all. Remove it and `tracks` reports on a measurement that
    # was never taken -- a 0.5 mm jog against a 3 mm tolerance floor makes `tracks` trivially true.
    gripper_moved = gripper_disp_norm > probe_min_disp
    mismatch = torch.linalg.norm(obj_disp_probe - gripper_disp_probe, dim=-1)
    tol = torch.clamp(probe_track_tol_frac * gripper_disp_norm, min=probe_track_tol)
    tracks = mismatch < tol

    return settled & opposed_contact & (probe_ready & gripper_moved & tracks)
