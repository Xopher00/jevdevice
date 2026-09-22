"""Decision journal: replay semantics, wire-shape freeze proof, call-site
labels, outcome-row helpers, truncation telemetry, rotation + writer cost."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from jevdevice.jev import JevClient, JevError, Noul
from jevdevice.journal import decision_log
from jevdevice.journal.decision_log import (
    BLOB_MIN_BYTES,
    DecisionJournal,
    goal_id_for,
    goal_scope,
)
from jevdevice.judge.gate import CommandVariant, GateResult

# mcp_server bootstrap() needs both env vars at import; placeholders are enough --
# the outcome helpers tested here never touch a device or the network.
os.environ.setdefault("TYPESAFE_AI_API", "placeholder-for-import")
os.environ.setdefault("ANDROID_SERIAL", "placeholder-for-import")


class RecordingJournal:
    """Test double with the DecisionJournal record_* shape."""

    def __init__(self) -> None:
        self.decisions: list[dict] = []
        self.outcomes: list[dict] = []

    def record_decision(self, **row) -> None:
        self.decisions.append(row)

    def record_outcome(self, **row) -> None:
        self.outcomes.append(row)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeHTTP:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.bodies: list[dict] = []

    async def post(self, url, headers=None, json=None):
        self.bodies.append(json)
        return _FakeResponse(self.payload)


def _client_with(http_payload: dict, journal) -> tuple[JevClient, _FakeHTTP]:
    client = JevClient("test-key", journal=journal)
    fake = _FakeHTTP(http_payload)
    client._client = fake
    return client, fake


# --- rows write and replay ---------------------------------------------------

def _decision_row(journal: DecisionJournal, big: str) -> None:
    journal.record_decision(
        call_id="c1", engine="jev", model_revision="jev-1.13.0", phase="gate",
        state={"goal": "press 7", "screen": big}, questions={"q": {"type": "noul"}},
        answers={"q": {"type": "noul", "noul": 0.9}}, goal_id=goal_id_for("press 7"),
        goal="press 7", usage={"input_tokens": 10, "output_tokens": 2},
    )


def test_decision_and_outcome_rows_replay_with_blobs_resolved(tmp_path: Path) -> None:
    big = "x" * (BLOB_MIN_BYTES * 2)  # forces the content-addressed store
    journal = DecisionJournal(tmp_path)
    _decision_row(journal, big)
    journal.record_outcome(
        call_id="c1", executed_command="input tap 5 700", verification="verified",
        device="serial-1", graph_edge={"from_node": "app.a", "to_node": "app.b"},
    )

    # A fresh instance over the same directory proves on-disk (not in-memory) replay.
    rows = list(DecisionJournal(tmp_path).replay())
    assert len(rows) == 2
    decision, outcome = rows
    assert decision["type"] == "decision"
    assert decision["state"]["screen"] == big, "blob ref must resolve to the original value"
    assert decision["answers"]["q"]["noul"] == 0.9
    assert outcome["type"] == "outcome"
    assert outcome["executed_command"] == "input tap 5 700"
    assert outcome["graph_edge"] == {"from_node": "app.a", "to_node": "app.b"}
    # the join key
    assert outcome["call_id"] == decision["call_id"] == "c1"


def test_small_values_stay_inline_and_big_values_get_sha256_refs(tmp_path: Path) -> None:
    journal = DecisionJournal(tmp_path)
    small = {"goal": "press 7"}
    big = "y" * (BLOB_MIN_BYTES * 2)
    journal.record_decision(call_id="c2", engine="jev", model_revision="m",
                            state={"meta": small, "dump": big}, phase="verify")
    # Raw row on disk: the big value is a sha256 ref, the small one stays inline.
    raw_line = next(iter(sorted(tmp_path.glob("journal-*.jsonl")))).read_text().strip().splitlines()[0]
    raw = json.loads(raw_line)
    assert raw["state"]["meta"] == small
    stored = raw["state"]["dump"]
    assert set(stored) == {"blob"}
    assert stored["blob"]["bytes"] == len(big.encode())
    # Replay resolves it back to the exact original bytes.
    assert DecisionJournal(tmp_path).blobs.get(stored["blob"]) == big.encode()
    resolved = next(iter(DecisionJournal(tmp_path).replay()))
    assert resolved["state"]["dump"] == big


def test_day_rotation_writes_one_file_per_day(tmp_path: Path) -> None:
    def day21() -> datetime:
        return datetime(2026, 9, 21, 12, 0, 0)  # noqa: DTZ001 -- naive clock, same family as the writer

    def day22() -> datetime:
        return datetime(2026, 9, 22, 8, 0, 0)  # noqa: DTZ001 -- naive clock

    DecisionJournal(tmp_path, clock=day21).record_decision(call_id="a", engine="jev", model_revision="m")
    DecisionJournal(tmp_path, clock=day22).record_decision(call_id="b", engine="jev", model_revision="m")
    files = sorted(p.name for p in tmp_path.glob("journal-*.jsonl"))
    assert files == ["journal-20260921.jsonl", "journal-20260922.jsonl"]


def test_max_age_prunes_old_files_keeps_recent(tmp_path: Path) -> None:
    (tmp_path / "journal-20260901.jsonl").write_text("{}\n")
    (tmp_path / "journal-20260918.jsonl").write_text("{}\n")
    DecisionJournal(tmp_path, max_age_days=5, clock=lambda: datetime(2026, 9, 20))  # noqa: DTZ001 -- naive clock
    assert not (tmp_path / "journal-20260901.jsonl").exists()
    assert (tmp_path / "journal-20260918.jsonl").exists()


def test_disabled_journal_writes_nothing(tmp_path: Path) -> None:
    journal = DecisionJournal(tmp_path, enabled=False)
    journal.record_decision(call_id="x", engine="jev", model_revision="m")
    journal.record_outcome(call_id="x", verification="verified")
    assert list(tmp_path.glob("journal-*.jsonl")) == []
    assert list(journal.replay()) == []


def test_goal_scope_sets_stable_id_and_text() -> None:
    assert decision_log.current_goal() == (None, None)
    with goal_scope("open the calculator"):
        assert decision_log.current_goal() == (goal_id_for("open the calculator"), "open the calculator")
        with goal_scope("press 7"):  # nested scopes restore correctly
            pass
        assert decision_log.current_goal() == (goal_id_for("open the calculator"), "open the calculator")
    assert decision_log.current_goal() == (None, None)


# --- ask() emits decision rows, wire body unchanged -------------------------

_PAYLOAD = {"answers": {"q1": {"type": "noul", "noul": 0.9}}, "usage": {"input_tokens": 10, "output_tokens": 2}}


async def test_ask_emits_a_replayable_decision_row() -> None:
    recorder = RecordingJournal()
    client, _ = _client_with(_PAYLOAD, recorder)
    answers = await client.ask({"goal": "press 7"}, {"q1": Noul(instructions="does it fit?")},
                               phase="gate", call_id="cid-1", truncation={"elements_after": 3})
    assert answers["q1"].noul == 0.9
    row = recorder.decisions[0]
    assert row["call_id"] == "cid-1"
    assert row["phase"] == "gate"
    assert row["engine"] == "jev"
    assert row["model_revision"] == client._model
    assert row["state"] == {"goal": "press 7"}
    assert row["questions"] == {"q1": {"type": "noul", "instructions": "does it fit?"}}
    assert row["answers"] == {"q1": {"type": "noul", "noul": 0.9}}  # full distribution as sent back
    assert row["truncation"] == {"elements_after": 3}
    assert row["usage"] == {"input_tokens": 10, "output_tokens": 2}
    assert row["error"] is None


async def test_ask_wire_body_stays_exactly_the_frozen_shape() -> None:
    """The wire proof: journaling metadata must never reach the engine."""
    recorder = RecordingJournal()
    client, fake = _client_with(_PAYLOAD, recorder)
    with goal_scope("open the calculator"):
        await client.ask({"goal": "press 7"}, {"q1": Noul(instructions="i")},
                         phase="gate", goal_id="explicit-id", call_id="cid", truncation={"a": 1})
    body = fake.bodies[0]
    assert set(body) == {"model", "state", "questions"}, f"wire shape changed: {set(body)}"
    assert body["state"] == {"goal": "press 7"}
    assert "phase" not in body["questions"]["q1"]


async def test_ask_without_scope_or_phase_logs_unlabeled_and_no_goal(tmp_path: Path) -> None:
    client, _ = _client_with(_PAYLOAD, DecisionJournal(tmp_path))
    await client.ask({"goal": "g"}, {"q1": Noul(instructions="i")})
    row = next(iter(DecisionJournal(tmp_path).replay()))
    assert row["phase"] == "unlabeled"
    assert row["goal_id"] is None and row["goal"] is None
    assert isinstance(row["call_id"], str) and uuid.UUID(row["call_id"])  # generated, valid


async def test_ask_failure_still_journals_the_request_then_reraises() -> None:
    recorder = RecordingJournal()
    client = JevClient("test-key", journal=recorder)

    class _FailingHTTP:
        async def post(self, url, headers=None, json=None):
            raise httpx.ConnectError("boom")

    client._client = _FailingHTTP()
    with pytest.raises(JevError):
        await client.ask({"goal": "g"}, {"q1": Noul(instructions="i")}, phase="verify")
    row = recorder.decisions[0]
    assert row["answers"] is None
    assert "ConnectError" in row["error"]
    assert row["phase"] == "verify"


async def test_call_id_is_generated_once_per_ask_when_not_supplied() -> None:
    recorder = RecordingJournal()
    client, _ = _client_with(_PAYLOAD, recorder)
    await client.ask({"g": 1}, {"q1": Noul(instructions="i")})
    await client.ask({"g": 2}, {"q1": Noul(instructions="i")})
    first, second = (row["call_id"] for row in recorder.decisions)
    assert first != second


# --- every ask() call site carries a phase label -----------------------------

def test_every_ask_call_site_carries_a_phase_label() -> None:
    """Static guard over src/jevdevice/*.py (calibrate/ CLIs excluded): any
    jev.ask( call site added later must pass phase=... or this fails, so one
    day of usage can't accumulate 'unlabeled' rows unnoticed."""
    package_dir = Path(__file__).resolve().parent.parent / "src" / "jevdevice"
    offenders: list[str] = []
    for path in sorted(package_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"await \w+\.ask\(", text):
            depth, i = 1, match.end()
            while i < len(text) and depth:
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                i += 1
            if "phase=" not in text[match.end():i]:
                line = text[:match.start()].count("\n") + 1
                offenders.append(f"{path.name}:{line}")
    assert not offenders, f"ask() call sites missing phase= label: {offenders}"


# --- outcome-row helpers -----------------------------------------------------

def test_pending_action_carries_the_gate_call_id() -> None:
    from jevdevice import mcp_server
    from jevdevice.judge.gate import GateVerdict, Pending

    gate_result = GateResult(verdict=GateVerdict.NEEDS_APPROVAL, reason="jev_uncertain",
                             confidence=0.5, call_id="cid-gate")
    pending = Pending(command=CommandVariant("input tap 5 700", "tap per the goal"),
                      chosen_label="tap '7'", gate_result=gate_result)
    mcp_server._store_pending("press 7", "tap", "element", 0.9, pending)
    stored = next(iter(mcp_server._PENDING.values()))
    assert stored.call_id == "cid-gate"
    from jevdevice.execution.dispatch import call_id_of
    assert call_id_of(pending) == "cid-gate"
    assert mcp_server.PendingAction(goal="g", kind="tap", resume_arg=None, confidence=0.0,
                                    pending=pending).call_id is None  # backward compatible default


def test_verification_mapping_from_response_status() -> None:
    from jevdevice.journal.outcomes import verification_from_response
    assert verification_from_response({"status": "ok"}) == "verified"
    assert verification_from_response({"status": "escalated"}) == "escalated"
    assert verification_from_response({"status": "unverified", "exit_code": 1}) == "failed"
    assert verification_from_response({"status": "unverified", "exit_code": 0}) == "none"
    assert verification_from_response({"status": "unverified"}) == "none"


def test_emit_outcome_writes_a_row_joined_to_the_goal_scope(monkeypatch) -> None:
    from jevdevice import mcp_server

    recorder = RecordingJournal()
    monkeypatch.setattr(decision_log, "_default_journal", recorder)
    from jevdevice.journal.outcomes import verification_from_response
    with goal_scope("press 7"):
        response = {"status": "ok", "satisfied": 0.97}
        mcp_server._emit_outcome(call_id="cid-gate", executed_command="input tap 5 700",
                                 verification=verification_from_response(response),
                                 response=response, decision="approve")
    row = recorder.outcomes[0]
    assert row["verification"] == "verified"
    assert row["executed_command"] == "input tap 5 700"
    assert row["goal"] == "press 7" and row["goal_id"] == goal_id_for("press 7")
    assert row["satisfied"] == 0.97
    assert row["device"] == mcp_server.transport.serial


# --- truncation telemetry ----------------------------------------------------

def test_describe_screen_telemetry_records_what_was_cut() -> None:
    from jevdevice.actions.elements import describe_screen
    nodes = "".join(
        f'<node text="item{i}" resource-id="" content-desc="" clickable="true" bounds="[0,{i}][100,{i + 10}]"/>'
        for i in range(10)
    )
    dump_xml = f"<hierarchy>{nodes}</hierarchy>"
    telemetry: dict = {}
    labels = describe_screen(dump_xml, limit=3, telemetry=telemetry)
    assert len(labels) == 3
    assert telemetry["elements_before"] == 10
    assert telemetry["elements_after"] == 3
    assert telemetry["bytes_before"] == len(dump_xml.encode())
    assert 0 < telemetry["bytes_after"] < telemetry["bytes_before"]


def test_describe_screen_without_telemetry_dict_is_unchanged() -> None:
    from jevdevice.actions.elements import describe_screen
    dump_xml = '<hierarchy><node text="Send" resource-id="" content-desc="" clickable="true" bounds="[0,0][100,50]"/></hierarchy>'
    assert describe_screen(dump_xml) == ["text='Send'"]  # existing callers unaffected


# --- writer cost --------------------------------------------------------------

def test_writer_adds_under_a_millisecond_per_row(tmp_path: Path) -> None:
    """The hot-loop budget: synchronous append must stay cheap. The bound is
    generous so a cold filesystem doesn't flake the suite."""
    journal = DecisionJournal(tmp_path)
    row_kwargs = {
        "engine": "jev", "model_revision": "jev-1.13.0", "phase": "verify",
        "state": {"goal": "g"}, "questions": {"q": {"type": "noul"}},
        "answers": {"q": {"type": "noul", "noul": 0.9}},
    }
    n = 300
    start = time.perf_counter()
    for i in range(n):
        journal.record_decision(call_id=str(i), **row_kwargs)
    per_row_ms = (time.perf_counter() - start) * 1000 / n
    assert per_row_ms < 1.0, f"journal write cost {per_row_ms:.3f} ms/row exceeds the 1 ms budget"
