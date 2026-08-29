# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python core for the C3 RUNG stage -- ``RESET_SPEC_V2.md`` sec 1 C3, bead ``dr-ai1.4``:

    **C3 = 50% S1 + 50% S_t.**

Needs only ``math`` (no torch, no Isaac Sim, no GPU, no env construction) -- same split, and same
reason, as ``c3_transport_core.py`` / ``c1_hand_pose_core.py`` next to this file: the
ISAAC-TOUCHING half (the actual reset event and goal command terms) lives in ``c3_rung.py`` and in
``dexlift_ur5e_delto_tableleg_env_cfg.py``, both of which import ``isaaclab`` at module scope and
therefore need a running Isaac Sim process just to import. This module has none of that dependency,
so ``source/uwlab_tasks/test/test_c3_rung_stage.py`` can load it with plain ``python3``.

WHAT THE TWO HALVES ARE (frozen spec sec 1 C3, plus the user's 2026-08-29 clarification recorded as
``V2_POSE_FINDINGS.md`` F51):

* **S1 -- "about to insert".** The leg is held in the CORRECT TIP-DOWN orientation, reasonably close
  to the target, in a shallow depth band about the mating frame. Generated from the PARTIAL-ASSEMBLY
  spawn (the leg starts pre-inserted in the bore, which is tip-down by construction -- F43 measured
  that branch's goal tilt at 0.00-0.28 deg from tip-down) with the goal displaced a shallow
  POSITIVE ``delta`` DEEPER along the bore's own axis. Nothing here reorients the leg: the goal
  quaternion IS the spawn quaternion (:func:`s1_goal_orientation`), only the position moves.

* **S_t -- "just picked it up".** The leg lies HORIZONTAL on the table and the target is placed
  EXACTLY where the leg already is -- ZERO delta, position AND orientation. The policy acquires and
  holds; it does not transport and it does not reorient.

  **S_t'S PEG IS HORIZONTAL. THIS IS NOT AN INFERENCE AND MUST NOT BE "CORRECTED" BACK.** The frozen
  spec was silent on S_t's orientation; an earlier session inferred TIP-DOWN, wrote that inference
  into ``V2_C3_DESIGN.md`` sec 4 as a blocker, and commissioned a spawn-tumble experiment to remove
  a constraint the spec never contained. The user's clarification (F51) settled it, and the measured
  baseline settles it independently: production staging, settled after physics, n=2048 -- **99.02%
  of legs settle lying flat with the tip within 20 mm of the table** (F50's baseline column, read
  against the corrected definition in F51). That IS the S_t precondition and it is already satisfied
  by the DEFAULT spawn. **S_t therefore requires NO spawn change**, and
  ``DEXRESET_ST_SPAWN_TIPDOWN`` (commit ``dffe5de``, the artifact of that cancelled experiment) is
  surplus here: this module neither reads it nor is wired to it.
  :data:`ST_NOMINAL_TILT_RAD` encodes the horizontal orientation as a number so the frame
  conversions below cannot silently assume otherwise -- see "FRAMES" .

WHY THIS IS A WHOLE-RUN STAGE AND NOT A SET OF EPISODE-MIXTURE FRACTIONS. Both halves' GOAL
mechanisms already exist and are reused, not reimplemented (see ``c3_rung.py``); what does not exist
is a single run that draws BETWEEN them. The episode mixture cannot express this composition:
``assert_episode_mixture_is_sane`` REQUIRES ``classic_goal_prob > 0`` (guarding a measured collapse
-- 55% of the skill gone in 50 epochs, 89% by 300, pass@30mm -> 0.0000 -- when the objective stops
containing the transport task), and C3 = 50% S1 + 50% S_t leaves the classic fraction at zero. That
guard is not weakened here. ``dexlift_ur5e_delto_tableleg_env_cfg.py`` already records the same
conclusion for the same reason -- "``DEXLIFT_EPISODE_MIXTURE`` cannot express '100%
partial-assembly' even if asked to -- the legacy toggle is the only path capable of a pure
specialist run" -- and ``_apply_partial_assembly_and_goal_toggles``'s own docstring names the
consumers that need it: "several tools outside training (reset-state generation, certification,
rendering) depend on being able to force 100% of envs into partial-assembly/goal-at-spawn, which a
mixture cannot give them." C3 rung GENERATION is exactly such a tool. So this follows the repo's own
established answer -- a deterministic whole-run stage behind its own env var -- rather than
loosening a measured training guard for a rollout that carries no gradient anyway.

FRAMES -- ``RESET_SPEC_V2.md`` sec 1a, and F49, which this module obeys by construction rather than
by comment. Every Z band in the spec is TIP frame measured from the tabletop unless it says
``root-Z`` in so many words, and the conversion is
``root_z - tip_z = ROOT_ABOVE_TIP_M * cos(tilt)`` -- **a VECTOR projection, not a bare subtraction**
(F49: the offset ``assembled_offset.pos = [-0.106203, 0, 0]`` is a displacement in the LEG'S OWN
local frame, so near horizontal its world-z component is approximately zero AND CHANGES SIGN across
90 deg). This module never restates that arithmetic: :func:`goal_tip_z_from_root_z` delegates to
``c3_transport_core.tip_z_from_root_z``, whose ``tilt_rad`` is a REQUIRED KEYWORD-ONLY argument
precisely so no caller can reproduce F49 by omission. The two rung halves sit at the two extremes of
that cosine and that is the whole point of naming them here:

* :data:`S1_NOMINAL_TILT_RAD` ``= 0.0`` -- tip-down, ``cos = 1``, the full 106.203 mm applies.
* :data:`ST_NOMINAL_TILT_RAD` ``= pi/2`` -- horizontal, ``cos = 0``, **NO offset applies at all**;
  the leg's root and tip sit at the same height, which is why a settled S_t leg's tip is ~15 mm off
  the table (its own half-thickness) and not 106 mm below its root.

A bare ``root_z - 0.106203`` applied to S_t would be wrong by the full 106 mm. That is the exact
failure F49 recorded, and :func:`goal_tip_z_from_root_z` is what makes it unrepresentable here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ``c3_transport_core`` is a sibling module in this same package and, like this one, imports nothing
# from Isaac. The dual import exists so BOTH modules can also be loaded BY FILE PATH, outside any
# package, by ``source/uwlab_tasks/test/test_c3_rung_stage.py`` -- a relative import has no package
# to resolve against under ``importlib.util.spec_from_file_location``. The test registers
# ``c3_transport_core`` in ``sys.modules`` before loading this file, so the fallback binds the same
# module object the package import would have.
try:  # normal package import (training, generation, anything that goes through ``mdp/__init__``)
    from . import c3_transport_core
except ImportError:  # pragma: no cover -- exercised only by the file-path load in the unit test
    import c3_transport_core  # type: ignore[no-redef]

# -- The two halves of C3. Values, not an Enum, for the same reason ``episode_mixture``'s
# ``EPISODE_KIND_*`` codes are: the runtime buffer is a plain ``torch.long`` tensor compared with
# ``==`` against these.
C3_KIND_S1 = 0
"""S1 -- near-goal held, leg TIP-DOWN, shallow depth band about the mating frame. "About to insert.\""""

C3_KIND_ST = 1
"""S_t -- near-table grasp, leg HORIZONTAL on the table, goal exactly at the leg's own pose (zero
delta). "Just picked it up." See the module docstring for why HORIZONTAL is the requirement and not
an inference (F51)."""

DEFAULT_S1_FRACTION = 0.5
"""The spec's 50/50. ``RESET_SPEC_V2.md`` sec 1: "Composition: C3 = 50% S1 + 50% S_t." The S_t
fraction is always ``1 - s1_fraction`` -- there is no third branch to desync against, which is why
this stage takes ONE fraction rather than the episode mixture's sum-to-1.0 tuple (that mixture's
own sum check exists because a CLI override of one of four fields silently desyncs the rest)."""

DEFAULT_S1_GOAL_DELTA_MM = 5.0
"""Millimetres the S1 goal is displaced DEEPER along the bore's own axis from the leg's spawn pose.

+5.0 mm is the value S1 bank generation already runs at today -- the
``S1_GOAL_BELOW_SPAWN_MM:-5`` default
in ``scripts_v2/tools/launch_dexreset_s1_s2_bank_gen.sh``, fed to the legacy
``DEXLIFT_GOAL_BELOW_SPAWN_MM`` path -- carried over rather than re-picked so the S1 half of this
rung is the S1 the campaign has already been generating, not a new one.

IT IS A SHAPING DEVICE, NOT A TARGET. See ``GoalBelowSpawnPoseCommand``'s own docstring: what makes
a state S1 is the ACCEPTANCE band applied downstream by ``generate_reset_states_policy.py``'s
``_SeatingGateAddon`` (tip frame), not this displacement."""

S1_GOAL_DELTA_BOUNDS_MM = (-200.0, 25.0)
"""Signed bounds on the S1 goal displacement, in millimetres, IDENTICAL to the pair enforced by
``GoalBelowSpawnPoseCommand.__init__`` / ``_apply_partial_assembly_and_goal_toggles`` (+25 mm is the
bore's own engaged span; -200 mm is a policy bound giving headroom around the S2' rung's 20-120 mm
band). Restated here only so this stage refuses a bad value BEFORE Isaac starts; the runtime command
re-checks it anyway, so the two cannot disagree about what is legal."""

S1_NOMINAL_TILT_RAD = 0.0
"""S1's leg is TIP-DOWN, so the 106.203 mm root-above-tip offset projects FULLY onto world z
(``cos(0) = 1``). F43 measured the partial-assembly branch -- S1's spawn -- at 0.00-0.28 deg from
tip-down, and F49's table confirms ``root - tip = 0.1062`` on every row of it."""

ST_NOMINAL_TILT_RAD = math.pi / 2
"""S_t's leg is HORIZONTAL, so the offset lies in the horizontal plane and projects to ZERO
(``cos(pi/2) = 0``): root and tip sit at the same height. F50's baseline column, n=2048 settled:
99.02% of legs settle within 5 deg of 90 (lying flat). This constant is what stops a future reader
subtracting 106.203 mm from an S_t height -- the precise error F49 recorded."""


DEFAULT_ST_SETTLE_SPEED_MPS = 0.05
"""Ceiling on the object's ABSOLUTE linear speed (m/s, world frame) for S_t's leg to count as
settled, i.e. for its goal to be re-pinned (bead ``dr-ai1.18``, ``V2_C3_DESIGN.md`` sec 7).

PROVENANCE, and read this before changing it -- the number is sourced, the FILE it was attributed
to was not. ``generate_reset_states_policy.py``'s ``--c2_max_resting_speed`` (default **0.05**)
is "a hard filter, enforced not just documented: a rewound candidate whose object linear velocity
magnitude exceeds this (m/s) at the REWOUND step is rejected at emit time ... measured medians at
the locked offsets are 0.000-0.049 m/s, so this discards only the tail that was not actually
resting." ``--c4_rewind_max_speed`` (also 0.05) is the same quantity applied at candidate-selection
time. **That is the established absolute-object-linear-speed resting convention in this tree, and
this constant deliberately matches it**, so the env side and the generation side call the same
states resting.

WHAT THIS IS *NOT*: ``held_check.py``'s ``settled``. That gate is
``steps_since_reset > SETTLE_STEPS`` -- a STEP COUNT with no speed term at all (``held_check.py``,
``settled = steps > self.settle_steps``; ``held_check_core.SETTLE_STEPS == 60``). The ``0.05`` that
does appear in ``held_check`` is ``comove_speed_thresh``, a ceiling on the RELATIVE speed
``|v_obj - v_palm|`` for the co-move gate -- a leg being carried steadily by the hand passes it, so
it does not mean "at rest on the table" and cannot be used here. Both halves of ``held_check``'s
settle logic ARE reused: the step count as :data:`ST_SETTLE_MIN_STEPS_SOURCE` below, and this
absolute-speed ceiling alongside it. Neither is re-derived, and no third settle test is written."""

DEFAULT_ST_SETTLE_ANG_SPEED_RAD_S = 0.05
"""Ceiling on the object's ABSOLUTE angular speed (rad/s, world frame) for S_t's leg to count as
settled (team-lead decision 2026-08-29, bead ``dr-ai1.18``).

PROVENANCE: F50/F51's own settled definition, the measurement that produced the 99.02%-settle-flat
figure this whole rung rests on -- "settled (lin < 0.01 m/s, ang < 0.05 rad/s)". The **angular**
half of that pair is taken here verbatim.

WHY AN ANGULAR TERM IS NOT REDUNDANT WITH THE LINEAR ONE. This bead exists because a pre-settle pin
captures the wrong ORIENTATION. A leg pivoting on a corner, or spinning about a vertical axis, can
have a near-zero LINEAR speed while its orientation is still changing -- so a linear-only predicate
can pin a wrong orientation at a moment of zero linear speed, which is the failure mode being
removed, arriving by a narrower path. The step floor covers most of that window, but "most" is not
the standard for the one quantity the bead is about.

DELIBERATE DIVERGENCE -- DO NOT "FIX" ONE OF THESE TO MATCH THE OTHER. The two thresholds come from
DIFFERENT sources on purpose:

* ANGULAR, here: **0.05 rad/s**, from F50/F51's settled pair.
* LINEAR, :data:`DEFAULT_ST_SETTLE_SPEED_MPS`: **0.05 m/s**, from ``--c2_max_resting_speed`` --
  NOT F50/F51's tighter 0.01 m/s.

The linear bound intentionally follows the GENERATION side rather than the measurement, because
cross-layer agreement on what "resting" means is worth more than internal agreement with F50/F51:
the seam between the env side and the generation side is where this campaign keeps getting hurt, and
a state this stage calls resting must be a state ``generate_reset_states_policy.py`` also calls
resting. There is no such cross-layer consumer for the angular bound, so it follows the measurement.
Numerically the two happen to both read 0.05 in their own units, which is a coincidence of unit
choice and not a shared constant -- they are not linked and changing one must not change the other.
"""

ST_SETTLE_MIN_STEPS_SOURCE = "held_check_core.SETTLE_STEPS"
"""Where the minimum-step floor comes from -- a POINTER, not a copy of the number.

This module never restates ``60``. :func:`st_should_repin` takes ``min_steps`` as a REQUIRED
keyword-only argument with no default (the same API shape as ``c3_transport_core.tip_z_from_root_z``'s
``tilt_rad``, and for the same F49b reason: "the fix worth copying is the API shape, not the
arithmetic"), and ``c3_rung.py`` passes ``held_check_core.SETTLE_STEPS`` into it. So the constant
lives in exactly one place in the tree and a future change to it reaches this stage automatically.

WHY A STEP FLOOR IS NEEDED AT ALL, alongside the speed ceiling: a dropped leg's linear speed passes
through ZERO at the apex of every bounce. Speed alone would re-pin S_t's goal mid-bounce, at an
airborne pose in whatever orientation the leg was tumbling through -- reintroducing the exact defect
the deferred re-pin exists to remove. The two conditions together mean "enough time has passed AND
the leg is actually at rest"."""


def validate_s1_fraction(s1_fraction: float) -> None:
    """Fail loudly on a C3 split outside ``[0, 1]``.

    Deliberately does NOT require exactly 0.5: the spec's composition is 50/50 (:data:`DEFAULT_S1_FRACTION`)
    and that is the default, but a measurement pass that wants a pure-S1 or pure-S_t run to
    characterise one half in isolation is a legitimate use of this same stage and is not something to
    forbid. What IS forbidden is a value that cannot be a probability."""
    if not 0.0 <= s1_fraction <= 1.0:
        raise ValueError(f"C3 s1_fraction must be a probability in [0, 1]; got {s1_fraction}")


def validate_s1_goal_delta_mm(delta_mm: float) -> None:
    """Fail loudly on an S1 goal displacement outside :data:`S1_GOAL_DELTA_BOUNDS_MM`."""
    lo, hi = S1_GOAL_DELTA_BOUNDS_MM
    if not lo <= delta_mm <= hi:
        raise ValueError(
            f"C3 S1 goal delta must be within [{lo}, {hi}] mm -- the same signed bounds"
            " GoalBelowSpawnPoseCommand enforces (+25 mm is the bore's engaged span, -200 mm is"
            f" headroom around the S2' band). Got {delta_mm} mm."
        )


def validate_st_settle_speed(speed_mps: float) -> None:
    """Fail loudly on a non-positive settle-speed ceiling. Zero would mean "re-pin only at exactly
    zero speed", which floating-point physics never reports, so the re-pin would silently never
    fire and S_t would quietly revert to its pre-settle behaviour -- a regression that looks like
    nothing at all in a log."""
    if not speed_mps > 0.0:
        raise ValueError(
            f"DEXRESET_C3_ST_SETTLE_SPEED must be > 0 m/s; got {speed_mps}. Zero would mean the"
            " re-pin never fires and S_t silently keeps its spawn-pinned goal."
        )


def validate_st_settle_ang_speed(ang_speed_rad_s: float) -> None:
    """Fail loudly on a non-positive angular settle ceiling -- same reasoning as
    :func:`validate_st_settle_speed`: zero would mean the re-pin never fires and S_t would silently
    revert to its spawn-pinned goal, a regression invisible in a log."""
    if not ang_speed_rad_s > 0.0:
        raise ValueError(
            f"DEXRESET_C3_ST_SETTLE_ANG_SPEED must be > 0 rad/s; got {ang_speed_rad_s}. Zero would"
            " mean the re-pin never fires and S_t silently keeps its spawn-pinned goal."
        )


def validate_st_settle_min_steps(min_steps: int) -> None:
    """Fail loudly on a negative step floor."""
    if min_steps < 0:
        raise ValueError(f"DEXRESET_C3_ST_SETTLE_STEPS must be >= 0; got {min_steps}")


def st_should_repin(
    *,
    already_repinned: bool,
    steps_since_reset: int,
    object_lin_speed_mps: float,
    object_ang_speed_rad_s: float,
    settle_speed_mps: float,
    settle_ang_speed_rad_s: float,
    min_steps: int,
) -> bool:
    """Should S_t's goal be re-pinned at the object's CURRENT pose on this step?

    ``RESET_SPEC_V2.md`` sec 1 C3 / ``V2_C3_DESIGN.md`` sec 7 / bead ``dr-ai1.18``. True on the
    FIRST step at which the leg has both waited out ``min_steps`` and come to rest, and never again
    that episode -- ``already_repinned`` is the latch, and the caller clears it at reset.

    WHY THIS EXISTS AT ALL. S_t's defining property is that the target sits exactly where the leg
    is. Pinning the goal during ``CommandManager.reset()`` pins it where the leg WAS, in mid-air.
    The position error that causes is bounded by the drop-height clamp (~50 mm), but **the
    ORIENTATION error is not bounded at all**: spawn orientation is randomized and settling is
    precisely the process that carries a randomly-oriented airborne leg to a flat resting one
    (F50/F51: 99.02% settle lying flat). A spawn-pinned goal can therefore sit up to ~90 deg from
    the leg's resting orientation, which does not merely add error -- it INVERTS the rung, turning
    S_t into a commanded REORIENTATION, the one thing its definition says it must never ask for.

    NOT A SPAWN CHANGE, and it does not contradict F51. The leg spawns and settles exactly as it
    does today and the 99.02% figure still describes it; only the MOMENT THE GOAL IS READ moves.
    ``DEXRESET_ST_SPAWN_TIPDOWN`` stays off and stays surplus.

    BOTH CONDITIONS ARE LOAD-BEARING -- see :data:`ST_SETTLE_MIN_STEPS_SOURCE` for why the step
    floor cannot be dropped (a bouncing leg's linear speed passes through zero at every apex) and
    :data:`DEFAULT_ST_SETTLE_SPEED_MPS` for where the speed ceiling comes from and why
    ``held_check``'s own ``settled`` (a step count) and ``comove_speed_thresh`` (a RELATIVE speed)
    are each only half of what is needed here.

    ``min_steps`` is REQUIRED and keyword-only so this module holds no copy of the number -- the
    caller passes ``held_check_core.SETTLE_STEPS``. Every argument is keyword-only: the two float
    arguments are a speed and a speed ceiling, and a positional swap between them would produce a
    predicate that is wrong in a way no type checker would catch.
    """
    if already_repinned:
        return False
    return (
        steps_since_reset > min_steps
        and object_lin_speed_mps <= settle_speed_mps
        and object_ang_speed_rad_s <= settle_ang_speed_rad_s
    )


@dataclass(frozen=True)
class C3RungStaging:
    """The parsed, validated C3 stage configuration. Frozen: nothing downstream may mutate it."""

    s1_fraction: float
    s1_goal_delta_m: float
    st_settle_speed_mps: float = DEFAULT_ST_SETTLE_SPEED_MPS
    st_settle_ang_speed_rad_s: float = DEFAULT_ST_SETTLE_ANG_SPEED_RAD_S
    st_settle_min_steps: int | None = None
    """``None`` means "use :data:`ST_SETTLE_MIN_STEPS_SOURCE`" -- resolved by ``c3_rung.py``, which
    can import it. Kept as ``None`` rather than eagerly resolved so this module holds no copy of the
    number; see :data:`ST_SETTLE_MIN_STEPS_SOURCE`."""

    @property
    def st_fraction(self) -> float:
        """Always the complement -- see :data:`DEFAULT_S1_FRACTION`."""
        return 1.0 - self.s1_fraction


def parse_c3_rung_env(environ) -> C3RungStaging | None:
    """Parse the C3 rung stage out of a mapping of environment variables.

    Returns ``None`` -- the stage is OFF, and nothing about any existing run changes -- unless
    ``DEXRESET_C3_RUNG`` is exactly ``"1"``. Same opt-in idiom, and same strictly-``"1"`` test, as
    ``DEXRESET_C1_HAND`` / ``DEXLIFT_PARTIAL_ASSEMBLY`` / ``DEXLIFT_EPISODE_MIXTURE``.

    Variables:

    * ``DEXRESET_C3_RUNG`` -- ``"1"`` turns the stage on. Anything else (unset, ``"0"``, ``"true"``)
      leaves it off. **Default: OFF.**
    * ``DEXRESET_C3_S1_FRACTION`` -- share of envs drawing S1; the rest draw S_t. Default
      :data:`DEFAULT_S1_FRACTION` (0.5), the spec's 50/50.
    * ``DEXRESET_C3_S1_GOAL_DELTA_MM`` -- signed millimetres the S1 goal is displaced deeper along
      the bore axis. Default :data:`DEFAULT_S1_GOAL_DELTA_MM` (+5.0), the value S1 bank generation
      already uses.
    * ``DEXRESET_C3_ST_SETTLE_SPEED`` -- m/s ceiling on the object's absolute linear speed for S_t's
      deferred goal re-pin. Default :data:`DEFAULT_ST_SETTLE_SPEED_MPS` (0.05).
    * ``DEXRESET_C3_ST_SETTLE_ANG_SPEED`` -- rad/s ceiling on the object's absolute ANGULAR speed
      for the same re-pin. Default :data:`DEFAULT_ST_SETTLE_ANG_SPEED_RAD_S` (0.05). Sourced from
      F50/F51, unlike the linear bound -- see that constant for why the divergence is deliberate.
    * ``DEXRESET_C3_ST_SETTLE_STEPS`` -- minimum steps since reset before the re-pin may fire.
      Unset (the default) means :data:`ST_SETTLE_MIN_STEPS_SOURCE`, resolved by the caller.

    Read from ``os.environ`` at ``__post_init__`` time by the caller, NOT declared as Hydra
    dataclass fields -- the same reachability argument every other whole-run toggle in
    ``dexlift_ur5e_delto_tableleg_env_cfg.py`` makes: an env var read directly inside
    ``__post_init__`` has no override-ordering window for a later ``env.foo=...`` to lose a race
    against. (The episode mixture's fractions are dataclass fields precisely because they are meant
    to be swept from the CLI; these are not.)
    """
    if environ.get("DEXRESET_C3_RUNG") != "1":
        return None

    fraction_raw = environ.get("DEXRESET_C3_S1_FRACTION", str(DEFAULT_S1_FRACTION))
    try:
        s1_fraction = float(fraction_raw)
    except ValueError as exc:
        raise ValueError(
            f"DEXRESET_C3_S1_FRACTION must be a single probability, e.g. '0.5'; got {fraction_raw!r}"
        ) from exc
    validate_s1_fraction(s1_fraction)

    delta_raw = environ.get("DEXRESET_C3_S1_GOAL_DELTA_MM", str(DEFAULT_S1_GOAL_DELTA_MM))
    try:
        delta_mm = float(delta_raw)
    except ValueError as exc:
        raise ValueError(
            f"DEXRESET_C3_S1_GOAL_DELTA_MM must be a single signed millimetres value, e.g. '5.0';"
            f" got {delta_raw!r}"
        ) from exc
    validate_s1_goal_delta_mm(delta_mm)

    speed_raw = environ.get("DEXRESET_C3_ST_SETTLE_SPEED", str(DEFAULT_ST_SETTLE_SPEED_MPS))
    try:
        settle_speed = float(speed_raw)
    except ValueError as exc:
        raise ValueError(
            f"DEXRESET_C3_ST_SETTLE_SPEED must be a single metres-per-second value, e.g. '0.05';"
            f" got {speed_raw!r}"
        ) from exc
    validate_st_settle_speed(settle_speed)

    ang_raw = environ.get("DEXRESET_C3_ST_SETTLE_ANG_SPEED", str(DEFAULT_ST_SETTLE_ANG_SPEED_RAD_S))
    try:
        settle_ang_speed = float(ang_raw)
    except ValueError as exc:
        raise ValueError(
            f"DEXRESET_C3_ST_SETTLE_ANG_SPEED must be a single radians-per-second value, e.g."
            f" '0.05'; got {ang_raw!r}"
        ) from exc
    validate_st_settle_ang_speed(settle_ang_speed)

    steps_raw = environ.get("DEXRESET_C3_ST_SETTLE_STEPS")
    if steps_raw is None:
        settle_min_steps = None
    else:
        try:
            settle_min_steps = int(steps_raw)
        except ValueError as exc:
            raise ValueError(
                f"DEXRESET_C3_ST_SETTLE_STEPS must be a non-negative integer, e.g. '60'; got {steps_raw!r}"
            ) from exc
        validate_st_settle_min_steps(settle_min_steps)

    return C3RungStaging(
        s1_fraction=s1_fraction,
        s1_goal_delta_m=delta_mm / 1000.0,
        st_settle_speed_mps=settle_speed,
        st_settle_ang_speed_rad_s=settle_ang_speed,
        st_settle_min_steps=settle_min_steps,
    )


def c3_kind_for_draw(draw: float, s1_fraction: float) -> int:
    """Map ONE uniform draw in ``[0, 1)`` to :data:`C3_KIND_S1` or :data:`C3_KIND_ST`.

    ``draw < s1_fraction`` is S1; everything else is S_t. Half-open on purpose and in this direction
    specifically: with ``s1_fraction == 0.0`` nothing is ever less than 0.0, so a zero fraction
    really does produce zero S1 envs (the same convention, and the same reasoning,
    ``MixtureResetObject.__call__``'s ``draw < self.partial_assembly_prob`` band already uses).

    The runtime term does this on a whole tensor at once rather than by calling this per env -- the
    comparison there is the same expression. This scalar form exists so the split is provable
    without a GPU; :func:`c3_kind_counts` is what the test actually measures the ratio with.
    """
    validate_s1_fraction(s1_fraction)
    return C3_KIND_S1 if draw < s1_fraction else C3_KIND_ST


def c3_kind_counts(draws, s1_fraction: float) -> dict[int, int]:
    """Count how many of ``draws`` land in each half. Returns both keys always, zeros included, so a
    caller cannot mistake "no S_t envs this reset" for "S_t is not configured"."""
    counts = {C3_KIND_S1: 0, C3_KIND_ST: 0}
    for draw in draws:
        counts[c3_kind_for_draw(draw, s1_fraction)] += 1
    return counts


def s1_goal_position(spawn_pos, bore_deep_axis_world, delta_m: float):
    """S1's goal POSITION: the leg's own spawn position displaced ``delta_m`` along the bore's own
    "deep" axis (already rotated into world by the FIXTURE's live orientation, and unit-normalised,
    by ``partial_assembly.live_bore_deep_axis`` -- the runtime caller passes what that returns).

    ONE expression for both signs, exactly as ``GoalBelowSpawnPoseCommand`` and
    ``MixtureGoalPoseCommand._resample_goal_at_spawn`` already do it: ``axis_world`` points INTO the
    bore, so a positive ``delta_m`` goes deeper and a negative one goes back out of the mouth.
    Nothing downstream branches on the sign.

    The axis is read off the FIXTURE, never off the leg -- the question is "which way is further
    INTO the bore", not "which way is this leg pointing". Reused rather than re-derived so this and
    the legacy shaping command cannot disagree about which way "deeper" is.
    """
    return tuple(p + delta_m * a for p, a in zip(spawn_pos, bore_deep_axis_world))


def s1_goal_orientation(spawn_quat):
    """S1's goal ORIENTATION: the leg's spawn orientation, UNCHANGED.

    S1's spawn is the partial-assembly pose -- the leg already seated in the bore, hence TIP-DOWN by
    construction (F43 measured that branch at 0.00-0.28 deg from tip-down). Returning it unchanged
    is what makes S1 "held in the correct tip-down orientation": the commanded goal asks for the
    orientation the leg is already in, so the rung is a DEPTH task about the mating frame and never a
    reorientation task. Identity by design, written as a named function rather than left implicit so
    the intent is greppable and so the test can state it.
    """
    return tuple(spawn_quat)


def st_goal_pose(spawn_pos, spawn_quat):
    """S_t's goal: the leg's OWN current pose, position AND orientation, ZERO delta.

    "The target is placed exactly where the leg is" (``RESET_SPEC_V2.md`` sec 1 C3). The policy
    acquires and holds -- it does not transport (position unchanged) and it does not reorient
    (orientation unchanged, and per F51 that orientation is HORIZONTAL, which is fine and is the
    point). Identity in both arguments, named for the same reason :func:`s1_goal_orientation` is.

    This is byte-for-byte what ``GoalAtSpawnPoseCommand`` / ``MixtureGoalPoseCommand._resample_goal_at_spawn``
    already compute at ``delta == 0``; the runtime half reuses that code rather than reimplementing
    it, and this function exists so the ZERO-delta requirement is testable without Isaac.
    """
    return tuple(spawn_pos), tuple(spawn_quat)


def nominal_tilt_rad(kind: int) -> float:
    """The nominal axis-tilt-from-tip-down of the leg in each half of C3, in radians.

    :data:`S1_NOMINAL_TILT_RAD` (0, tip-down) or :data:`ST_NOMINAL_TILT_RAD` (pi/2, HORIZONTAL). The
    ONLY consumer is :func:`goal_tip_z_from_root_z` -- see this module's "FRAMES" docstring section
    for why hard-coding either one into a bare subtraction is F49.
    """
    if kind == C3_KIND_S1:
        return S1_NOMINAL_TILT_RAD
    if kind == C3_KIND_ST:
        return ST_NOMINAL_TILT_RAD
    raise ValueError(f"unknown C3 kind {kind!r}; expected C3_KIND_S1 ({C3_KIND_S1}) or C3_KIND_ST ({C3_KIND_ST})")


def goal_tip_z_from_root_z(root_z_m: float, kind: int) -> float:
    """Convert a ROOT-frame height to the TIP-frame height it implies FOR THE NAMED HALF of C3.

    Delegates to ``c3_transport_core.tip_z_from_root_z`` -- the single implementation of
    ``root_z - tip_z = ROOT_ABOVE_TIP_M * cos(tilt_rad)``, whose ``tilt_rad`` is required and
    keyword-only exactly so it cannot be called without naming the pose it converts for (F49b: "the
    fix worth copying is the API shape, not the arithmetic"). This wrapper's whole job is to supply
    the RIGHT tilt per half rather than let a caller guess:

    * S1 (tip-down) subtracts the full 106.203 mm.
    * S_t (horizontal) subtracts NOTHING -- the offset lies in the horizontal plane.

    The arithmetic is not restated here and this module does not carry its own copy of the constant.
    """
    return c3_transport_core.tip_z_from_root_z(root_z_m, tilt_rad=nominal_tilt_rad(kind))


def c3_rung_banner(staging: C3RungStaging) -> str:
    """The exact banner text printed when the C3 stage is wired in (``RESET_SPEC_V2.md`` R5: a run
    must STATE its staging, not leave it to be inferred -- trap 3, "read the staged value back out
    of the run log").

    Returned as a string rather than printed here so a test can assert on it byte-for-byte, same
    technique as ``c3_transport_core.transport_goal_banner``.
    """
    settle_steps_text = (
        ST_SETTLE_MIN_STEPS_SOURCE if staging.st_settle_min_steps is None else str(staging.st_settle_min_steps)
    )
    return (
        f"[dexreset] C3 RUNG staged (bead dr-ai1.4, RESET_SPEC_V2.md sec 1 C3):"
        f" {staging.s1_fraction:.3f} of envs draw S1 and {staging.st_fraction:.3f} draw S_t."
        " S1 = partial-assembly spawn (leg pre-inserted, TIP-DOWN) with the goal displaced"
        f" {staging.s1_goal_delta_m * 1000.0:+.2f} mm along the bore's own deep axis from the leg's"
        " own spawn pose, orientation UNCHANGED (a depth task, never a reorientation task)."
        " S_t = the ORDINARY table spawn, unchanged -- the leg settles HORIZONTAL on the table"
        " (measured baseline, n=2048 settled: 99.02% lie flat with the tip within 20 mm of the"
        " table) -- with the goal pinned at the leg's OWN pose, ZERO delta in position AND"
        " orientation, so the policy acquires and holds without transporting or reorienting."
        " S_t's goal is RE-PINNED ONCE at the SETTLED pose -- the first step with"
        f" steps_since_reset > {settle_steps_text}"
        f" AND object linear speed <= {staging.st_settle_speed_mps:.3f} m/s"
        f" AND object angular speed <= {staging.st_settle_ang_speed_rad_s:.3f} rad/s -- NOT at the spawn"
        " pose, which is mid-air and in a randomized orientation up to ~90 deg from where the leg"
        " comes to rest (bead dr-ai1.18, V2_C3_DESIGN.md sec 7). This moves only WHEN the goal is"
        " read; the spawn is unchanged."
        " S_t makes NO spawn change and DEXRESET_ST_SPAWN_TIPDOWN is not read here (F51)."
        " Root-to-tip conversion differs between the halves and is never a bare subtraction:"
        f" a root z of 0.200 m means tip z {goal_tip_z_from_root_z(0.200, C3_KIND_S1):.4f} m for S1"
        f" (tip-down, full 106.203 mm) but {goal_tip_z_from_root_z(0.200, C3_KIND_ST):.4f} m for S_t"
        " (horizontal, the offset projects to zero) -- F49."
    )
