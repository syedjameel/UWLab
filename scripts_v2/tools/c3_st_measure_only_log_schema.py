# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Single source of truth for the C3(S_t) measure-only per-episode JSONL log schema (bead
dr-sj6.24, team-lead instruction 2026-08-29).

BOTH the producer (``generate_reset_states_policy.py``'s ``--c3_st_tolerance_measure_only`` path)
and the consumer (``analyze_c3_st_measure_only.py``) import the field-name constants and
:func:`validate_log_record` from HERE, rather than each restating its own literal list of field
names. This is the same fix this campaign has now applied repeatedly for the identical underlying
defect -- two sides of one contract, each individually valid, with nothing checking they agree --
as ``SETTLE_STEPS`` (imported, not restated, by every consumer of the settle floor),
``goal_is_final`` (one public accessor replacing two independent re-derivations of the same
settle predicate), and ``passive_gates`` (``held_check.py``'s shared gate implementation). Team-lead,
on approving this file: "the field names must come from ONE place... a second literal list is not"
a fix.

Isaac-free, stdlib only -- no ``isaaclab``/``torch`` import, so ``analyze_c3_st_measure_only.py``
(which must stay importable with plain ``python3`` + ``numpy``, no Isaac Sim) can import this
without pulling in anything heavier than what it already needs.
"""

from __future__ import annotations

FIELD_SUCCESS = "success"
FIELD_POS_DIST_M = "pos_dist_m"
FIELD_ROT_DIST_RAD = "rot_dist_rad"
FIELD_AXIS_TILT_RAD = "axis_tilt_rad"

REQUIRED_FIELDS: tuple[str, ...] = (FIELD_SUCCESS, FIELD_POS_DIST_M, FIELD_ROT_DIST_RAD, FIELD_AXIS_TILT_RAD)
"""Every field a valid log record must carry. Order matches the JSON.dumps() call
generate_reset_states_policy.py's own producer block writes them in -- not load-bearing (JSON
objects are unordered), kept only for readability."""

__all__ = [
    "FIELD_AXIS_TILT_RAD",
    "FIELD_POS_DIST_M",
    "FIELD_ROT_DIST_RAD",
    "FIELD_SUCCESS",
    "REQUIRED_FIELDS",
    "LogSchemaError",
    "validate_log_record",
]


class LogSchemaError(Exception):
    """Raised by :func:`validate_log_record` when a JSONL log line is missing a required field or
    carries the wrong type for one. Never caught and silently downgraded (e.g. by defaulting a
    missing field to 0.0) anywhere in this codebase -- a producer that silently emits four fields
    where five are expected must not read as a distribution with one metric quietly missing; it
    must refuse, the same way an addon that records nothing and one that records zeros must not
    look identical in a summary line."""


def validate_log_record(record: dict, *, lineno: int | None = None) -> None:
    """Raise :class:`LogSchemaError` unless ``record`` has ALL of :data:`REQUIRED_FIELDS`, each of
    the correct type (bool for ``success``, a real number for the three displacement fields --
    ``bool`` is deliberately rejected for the displacement fields even though ``bool`` is a
    ``int`` subclass in Python, since a displacement can never legitimately be exactly ``True``/
    ``False``). Extra, unrecognised keys are NOT an error -- this schema is a strict SUBSET of
    whatever a producer's own per-episode line might carry, not an exhaustive one.
    """
    where = f" (line {lineno})" if lineno is not None else ""
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise LogSchemaError(
            f"log record{where} is missing required field(s) {missing} -- expected ALL of "
            f"{list(REQUIRED_FIELDS)} (c3_st_measure_only_log_schema.REQUIRED_FIELDS). Got keys "
            f"{sorted(record.keys())}. Refusing to read this as a distribution with a metric "
            "silently missing."
        )
    success_val = record[FIELD_SUCCESS]
    if not isinstance(success_val, bool):
        raise LogSchemaError(
            f"log record{where}: {FIELD_SUCCESS!r} must be a bool, got "
            f"{type(success_val).__name__} ({success_val!r})."
        )
    for field in (FIELD_POS_DIST_M, FIELD_ROT_DIST_RAD, FIELD_AXIS_TILT_RAD):
        val = record[field]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise LogSchemaError(
                f"log record{where}: {field!r} must be a real number, got {type(val).__name__} ({val!r})."
            )
