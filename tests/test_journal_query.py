"""journal/query.py: typed record adapters, blob resolution, and the
iteration/filter/join helpers eval readers rely on."""

from __future__ import annotations

import os

import pytest

from jevdevice.journal.decision_log import BLOB_MIN_BYTES, DecisionJournal
from jevdevice.journal.query import (
    CalibrationRecord,
    DecisionRecord,
    JournalReader,
    OutcomeRecord,
    adapt,
)

os.environ.setdefault("TYPESAFE_AI_API", "placeholder-for-import")
os.environ.setdefault("ANDROID_SERIAL", "placeholder-for-import")


@pytest.fixture
def journal(tmp_path):
    return DecisionJournal(tmp_path, clock=lambda: __import__("datetime").datetime(2026, 9, 20, 12, 0, 0))


def _decision_kwargs(**overrides):
    kwargs = {
        "call_id": "c1", "engine": "jev", "model_revision": "jev-1.13.0", "phase": "gate",
        "state": {"goal": "press 7"}, "questions": {"q": {"type": "noul"}},
        "answers": {"safe": {"type": "noul", "noul": 0.9}}, "truncation": {},
        "usage": {"input_tokens": 5}, "error": None, "goal": "press 7", "goal_id": "gid1",
        "elapsed_ms": 12.5, "shadow_of": None, "generated": None,
    }
    kwargs.update(overrides)
    return kwargs


def _outcome_kwargs(**overrides):
    kwargs = {
        "call_id": "c1", "executed_command": "tap 7", "verification": "verified", "status": "ok",
        "recovery_command": None, "graph_edge": {"from_node": "a", "to_node": "b"},
        "device": "cli:1", "decision": "approve", "reasons": ["fit"], "exit_code": 0,
        "satisfied": 1.0, "goal": "press 7", "goal_id": "gid1", "executed": ["tap 7"],
        "kind": "ui", "tier": 1, "recipe_id": "r1",
    }
    kwargs.update(overrides)
    return kwargs


def test_adapt_decision_row_all_fields(journal):
    journal.record_decision(**_decision_kwargs())
    row = next(journal.replay())
    record = adapt(row)
    assert isinstance(record, DecisionRecord)
    assert record.call_id == "c1"
    assert record.engine == "jev"
    assert record.model_revision == "jev-1.13.0"
    assert record.phase == "gate"
    assert record.goal == "press 7"
    assert record.goal_id == "gid1"
    assert record.shadow_of is None
    assert record.generated is None
    assert record.error is None
    assert record.elapsed_ms == 12.5
    assert record.usage == {"input_tokens": 5}
    assert record.truncation == {}
    assert record.state == {"goal": "press 7"}
    assert record.questions == {"q": {"type": "noul"}}
    assert record.answers == {"safe": {"type": "noul", "noul": 0.9}}
    assert record.raw["type"] == "decision"


def test_adapt_outcome_row_all_fields(journal):
    journal.record_outcome(**_outcome_kwargs())
    row = next(journal.replay())
    record = adapt(row)
    assert isinstance(record, OutcomeRecord)
    assert record.call_id == "c1"
    assert record.executed_command == "tap 7"
    assert record.executed == ["tap 7"]
    assert record.verification == "verified"
    assert record.status == "ok"
    assert record.recovery_command is None
    assert record.graph_edge == {"from_node": "a", "to_node": "b"}
    assert record.decision == "approve"
    assert record.reasons == ["fit"]
    assert record.exit_code == 0
    assert record.satisfied == 1.0
    assert record.kind == "ui"
    assert record.tier == 1
    assert record.recipe_id == "r1"
    assert record.device == "cli:1"
    assert record.goal_id == "gid1"


def test_adapt_calibration_row_all_fields(journal):
    journal.record_calibration(
        event="proposal", engine="jev", provenance={"window": 50}, result={"proposal_values": {}},
    )
    row = next(journal.replay())
    record = adapt(row)
    assert isinstance(record, CalibrationRecord)
    assert record.event == "proposal"
    assert record.engine == "jev"
    assert record.provenance == {"window": 50}
    assert record.result == {"proposal_values": {}}


def test_adapt_unknown_type_returns_none():
    assert adapt({"type": "mystery"}) is None
    assert adapt({}) is None


def test_reader_rows_skip_unknown(journal, tmp_path):
    journal.record_decision(**_decision_kwargs())
    (tmp_path / journal._day_file().name).write_text(
        journal._day_file().read_text() + '{"type":"mystery"}\n', encoding="utf-8",
    )
    reader = JournalReader(journal)
    kinds = [type(r).__name__ for r in reader.rows()]
    assert kinds == ["DecisionRecord"]


def test_reader_decisions_outcomes_calibrations_split(journal):
    journal.record_decision(**_decision_kwargs())
    journal.record_outcome(**_outcome_kwargs())
    journal.record_calibration(event="no-proposal", engine="jev", provenance={}, result=None)
    reader = JournalReader(journal)
    assert len(list(reader.decisions())) == 1
    assert len(list(reader.outcomes())) == 1
    assert len(list(reader.calibrations())) == 1


def test_by_call_id_joins_decision_and_outcome(journal):
    journal.record_decision(**_decision_kwargs(call_id="shared"))
    journal.record_outcome(**_outcome_kwargs(call_id="shared"))
    journal.record_decision(**_decision_kwargs(call_id="other"))
    reader = JournalReader(journal)
    matched = reader.by_call_id("shared")
    assert len(matched) == 2
    assert {type(r).__name__ for r in matched} == {"DecisionRecord", "OutcomeRecord"}


def test_by_phase(journal):
    journal.record_decision(**_decision_kwargs(call_id="a", phase="gate"))
    journal.record_decision(**_decision_kwargs(call_id="b", phase="kind"))
    reader = JournalReader(journal)
    matched = reader.by_phase("gate")
    assert [r.call_id for r in matched] == ["a"]


def test_by_goal_id(journal):
    journal.record_decision(**_decision_kwargs(call_id="a", goal_id="g1"))
    journal.record_decision(**_decision_kwargs(call_id="b", goal_id="g2"))
    journal.record_outcome(**_outcome_kwargs(call_id="a", goal_id="g1"))
    reader = JournalReader(journal)
    matched = reader.by_goal_id("g1")
    assert {r.call_id for r in matched} == {"a"}


def test_in_time_range(journal):
    journal.record_decision(**_decision_kwargs(call_id="a"))
    reader = JournalReader(journal)
    row = next(reader.rows())
    ts = row.ts
    assert len(reader.in_time_range(start=ts, end=ts)) == 1
    assert reader.in_time_range(start="2099-01-01T00:00:00.000") == []
    assert len(reader.in_time_range(end="2099-01-01T00:00:00.000")) == 1


def test_join_outcomes_groups_by_call_id(journal):
    journal.record_outcome(**_outcome_kwargs(call_id="a"))
    journal.record_outcome(**_outcome_kwargs(call_id="a", executed_command="tap 8"))
    journal.record_outcome(**_outcome_kwargs(call_id="b"))
    reader = JournalReader(journal)
    joined = reader.join_outcomes()
    assert len(joined["a"]) == 2
    assert len(joined["b"]) == 1


def test_decision_outcome_pairs(journal):
    journal.record_decision(**_decision_kwargs(call_id="a"))
    journal.record_decision(**_decision_kwargs(call_id="b"))
    journal.record_outcome(**_outcome_kwargs(call_id="a"))
    reader = JournalReader(journal)
    pairs = {d.call_id: outcomes for d, outcomes in reader.decision_outcome_pairs()}
    assert len(pairs["a"]) == 1
    assert pairs["b"] == []


def test_blob_ref_resolved_transparently(journal):
    big_state = {"goal": "x" * (BLOB_MIN_BYTES + 100)}
    journal.record_decision(**_decision_kwargs(state=big_state))
    raw_line = journal._day_file().read_text()
    assert '"blob"' in raw_line
    reader = JournalReader(journal)
    record = next(reader.decisions())
    assert record.state == big_state


def test_reader_defaults_to_module_journal(monkeypatch, tmp_path):
    monkeypatch.setenv("JEV_JOURNAL_DIR", str(tmp_path))
    from jevdevice.journal import decision_log

    monkeypatch.setattr(decision_log, "_default_journal", None)
    reader = JournalReader()
    assert reader.journal.directory == tmp_path
