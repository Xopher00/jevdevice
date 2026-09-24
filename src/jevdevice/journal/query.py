"""Normalized read-only view over journal rows: typed records + iteration/
filter/join helpers, so eval/analysis tools never touch raw row dicts.

One adapter function per row type turns a raw (old-format) row into its
record. A second adapter for typesymbolic's new row shape can be added
later without touching any caller of the public helpers below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from . import decision_log
from .decision_log import DecisionJournal

DECISION = decision_log.DECISION
OUTCOME = decision_log.OUTCOME
CALIBRATION = decision_log.CALIBRATION


@dataclass(frozen=True)
class DecisionRecord:
    call_id: str | None
    ts: str | None
    engine: str | None
    model_revision: str | None
    phase: str | None
    goal: str | None
    goal_id: str | None
    shadow_of: str | None
    generated: str | None
    error: str | None
    elapsed_ms: float | None
    usage: dict | None
    truncation: dict | None
    state: dict | None
    questions: dict | None
    answers: dict | None
    raw: dict = field(repr=False)


@dataclass(frozen=True)
class OutcomeRecord:
    call_id: str | None
    ts: str | None
    goal: str | None
    goal_id: str | None
    device: str | None
    executed_command: str | None
    executed: list | None
    verification: str | None
    status: str | None
    recovery_command: str | None
    graph_edge: dict | None
    decision: str | None
    reasons: list | None
    exit_code: int | None
    satisfied: float | None
    kind: str | None
    tier: int | None
    recipe_id: str | None
    raw: dict = field(repr=False)


@dataclass(frozen=True)
class CalibrationRecord:
    ts: str | None
    event: str | None
    engine: str | None
    provenance: dict | None
    result: dict | None
    raw: dict = field(repr=False)


def _decision_from_old_row(row: dict) -> DecisionRecord:
    """Adapter: current (pre-typesymbolic) decision row -> DecisionRecord."""
    return DecisionRecord(
        call_id=row.get("call_id"), ts=row.get("ts"), engine=row.get("engine"),
        model_revision=row.get("model_revision"), phase=row.get("phase"),
        goal=row.get("goal"), goal_id=row.get("goal_id"), shadow_of=row.get("shadow_of"),
        generated=row.get("generated"), error=row.get("error"),
        elapsed_ms=row.get("elapsed_ms"), usage=row.get("usage"),
        truncation=row.get("truncation"), state=row.get("state"),
        questions=row.get("questions"), answers=row.get("answers"), raw=row,
    )


def _outcome_from_old_row(row: dict) -> OutcomeRecord:
    """Adapter: current (pre-typesymbolic) outcome row -> OutcomeRecord."""
    return OutcomeRecord(
        call_id=row.get("call_id"), ts=row.get("ts"), goal=row.get("goal"),
        goal_id=row.get("goal_id"), device=row.get("device"),
        executed_command=row.get("executed_command"), executed=row.get("executed"),
        verification=row.get("verification"), status=row.get("status"),
        recovery_command=row.get("recovery_command"), graph_edge=row.get("graph_edge"),
        decision=row.get("decision"), reasons=row.get("reasons"),
        exit_code=row.get("exit_code"), satisfied=row.get("satisfied"),
        kind=row.get("kind"), tier=row.get("tier"), recipe_id=row.get("recipe_id"),
        raw=row,
    )


def _calibration_from_old_row(row: dict) -> CalibrationRecord:
    """Adapter: current (pre-typesymbolic) calibration row -> CalibrationRecord."""
    return CalibrationRecord(
        ts=row.get("ts"), event=row.get("event"), engine=row.get("engine"),
        provenance=row.get("provenance"), result=row.get("result"), raw=row,
    )


_ADAPTERS = {
    DECISION: _decision_from_old_row,
    OUTCOME: _outcome_from_old_row,
    CALIBRATION: _calibration_from_old_row,
}


def adapt(row: dict) -> DecisionRecord | OutcomeRecord | CalibrationRecord | None:
    """Turn one already-blob-resolved raw row into its typed record, or None
    for an unrecognized/untyped row."""
    adapter = _ADAPTERS.get(row.get("type"))
    return adapter(row) if adapter else None


class JournalReader:
    """Typed, filterable view over a DecisionJournal's rows. Blob refs are
    resolved by DecisionJournal.replay before adaptation (never duplicated
    here)."""

    def __init__(self, journal: DecisionJournal | None = None) -> None:
        self.journal = journal or decision_log.get_journal()

    def rows(self, *, day: str | None = None):
        """Every row, adapted, in write order. Rows of an unknown type are
        skipped."""
        for row in self.journal.replay(day=day):
            record = adapt(row)
            if record is not None:
                yield record

    def decisions(self, *, day: str | None = None):
        return (r for r in self.rows(day=day) if isinstance(r, DecisionRecord))

    def outcomes(self, *, day: str | None = None):
        return (r for r in self.rows(day=day) if isinstance(r, OutcomeRecord))

    def calibrations(self, *, day: str | None = None):
        return (r for r in self.rows(day=day) if isinstance(r, CalibrationRecord))

    def by_call_id(self, call_id: str, *, day: str | None = None) -> list:
        """All records (any type) sharing a call_id, in write order."""
        return [r for r in self.rows(day=day) if getattr(r, "call_id", None) == call_id]

    def by_phase(self, phase: str, *, day: str | None = None) -> list:
        return [r for r in self.rows(day=day) if getattr(r, "phase", None) == phase]

    def by_goal_id(self, goal_id: str, *, day: str | None = None) -> list:
        return [r for r in self.rows(day=day) if getattr(r, "goal_id", None) == goal_id]

    def in_time_range(
        self, start: str | datetime | None = None, end: str | datetime | None = None,
        *, day: str | None = None,
    ) -> list:
        """Records whose `ts` falls in [start, end] (either bound optional).
        ISO-8601 strings compare correctly as-is; datetimes are normalized."""
        lo = start.isoformat(timespec="milliseconds") if isinstance(start, datetime) else start
        hi = end.isoformat(timespec="milliseconds") if isinstance(end, datetime) else end
        out = []
        for record in self.rows(day=day):
            ts = getattr(record, "ts", None)
            if ts is None:
                continue
            if lo is not None and ts < lo:
                continue
            if hi is not None and ts > hi:
                continue
            out.append(record)
        return out

    def join_outcomes(self, *, day: str | None = None) -> dict[str, list[OutcomeRecord]]:
        """call_id -> its outcome records, for pairing against decisions()."""
        joined: dict[str, list[OutcomeRecord]] = {}
        for outcome in self.outcomes(day=day):
            if outcome.call_id:
                joined.setdefault(outcome.call_id, []).append(outcome)
        return joined

    def decision_outcome_pairs(
        self, *, day: str | None = None,
    ) -> list[tuple[DecisionRecord, list[OutcomeRecord]]]:
        """Every decision paired with its (possibly empty) outcome list,
        joined by call_id."""
        outcomes_by_call = self.join_outcomes(day=day)
        return [
            (decision, outcomes_by_call.get(decision.call_id, []))
            for decision in self.decisions(day=day)
        ]
