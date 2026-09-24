"""Regression spec, ported onto core (typesymbolic) transport + journal for
2b2: JevEngine over httpx2.MockTransport instead of a fake httpx.AsyncClient,
decision_log._default_journal monkeypatched instead of a DecisionJournal
instance, row field names updated to core's (answers/scope/asked/state/extra/
error/elapsed_ms). The first 6 tests below are the ported ask()/journal spec
(da4e4a3's tests/test_decision_log.py via aa1d9ee's verbatim copy); the last 4
still exercise decision_log.NONE/VERIFIED-era outcome helpers that aren't
migrated yet and are left for agent 2c, unmodified."""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

import httpx2
import pytest
from typesymbolic.journal import Journal
from typesymbolic.judge import JevEngine, JudgeError

from jevdevice.jev import Noul, ask
from jevdevice.journal import decision_log
from jevdevice.journal.decision_log import goal_id_for, goal_scope
from jevdevice.judge.gate import CommandVariant, GateResult

# mcp_server bootstrap() needs both env vars at import; placeholders are enough --
# the outcome helpers tested here never touch a device or the network.
os.environ.setdefault("TYPESAFE_AI_API", "placeholder-for-import")
os.environ.setdefault("ANDROID_SERIAL", "placeholder-for-import")


class RecordingJournal:
    """Test double with the core Journal's record_* shape; record_outcome
    flattens `extra` into the row like the real Journal does."""

    def __init__(self) -> None:
        self.decisions: list[dict] = []
        self.outcomes: list[dict] = []
        self.verdicts: list[dict] = []

    def record_decision(self, **row) -> None:
        self.decisions.append(row)

    def record_outcome(self, *, key=None, outcome=None, extra=None, **row) -> None:
        row = {**row, "outcome": outcome, "key": key if key is not None else getattr(outcome, "key", None)}
        self.outcomes.append({**row, **(extra or {})})

    def record_verdict(self, **row) -> None:
        self.verdicts.append(row)


def _ok_response(**extra) -> httpx2.Response:
    return httpx2.Response(200, json={
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 10, "output_tokens": 2},
        "answers": {"q1": {"type": "noul", "noul": 0.9}},
        **extra,
    })


def _engine(handler) -> JevEngine:
    return JevEngine(api_key="test-key", transport=httpx2.MockTransport(handler))


# --- ask() emits decision rows, wire body unchanged -------------------------

async def test_ask_emits_a_replayable_decision_row(monkeypatch) -> None:
    recorder = RecordingJournal()
    monkeypatch.setattr(decision_log, "_default_journal", recorder)
    engine = _engine(lambda request: _ok_response())
    call_id, answers = await ask(engine, {"goal": "press 7"}, {"q1": Noul(instructions="does it fit?")},
                                 phase="gate", truncation={"elements_after": 3})
    assert answers["q1"].noul == 0.9
    row = recorder.decisions[0]
    assert row["call_id"] == call_id
    assert row["phase"] == "gate"
    assert row["engine"] == "jev"
    assert row["model_revision"] == "jev-1.13.0"
    assert row["state"] == {"goal": "press 7"}
    assert row["asked"] == {"q1": {"type": "noul", "instructions": "does it fit?"}}
    assert row["answers"]["q1"].noul == 0.9
    assert row["extra"]["truncation"] == {"elements_after": 3}
    assert row["extra"]["usage"] == {"input_tokens": 10, "output_tokens": 2}
    assert row.get("error") is None


async def test_ask_wire_body_stays_exactly_the_frozen_shape(monkeypatch) -> None:
    """The wire proof: journaling metadata must never reach the engine."""
    recorder = RecordingJournal()
    monkeypatch.setattr(decision_log, "_default_journal", recorder)
    bodies: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return _ok_response()

    engine = _engine(handler)
    with goal_scope("open the calculator"):
        await ask(engine, {"goal": "press 7"}, {"q1": Noul(instructions="i")}, phase="gate", truncation={"a": 1})
    body = bodies[0]
    assert set(body) == {"model", "state", "questions"}, f"wire shape changed: {set(body)}"
    assert body["state"] == {"goal": "press 7"}
    assert "phase" not in body["questions"]["q1"]


async def test_ask_without_scope_or_phase_logs_unlabeled_and_no_goal(tmp_path: Path, monkeypatch) -> None:
    journal = Journal(root=tmp_path, background_writes=False)
    monkeypatch.setattr(decision_log, "_default_journal", journal)
    engine = _engine(lambda request: _ok_response())
    call_id, _ = await ask(engine, {"goal": "g"}, {"q1": Noul(instructions="i")})
    row = next(iter(journal.replay()))
    assert row["phase"] is None
    assert row["scope"] == {}  # no goal_scope() active, no shadow -- nothing to carry
    assert row["call_id"] == call_id and uuid.UUID(call_id)  # generated, valid


async def test_ask_failure_still_journals_the_request_then_reraises(monkeypatch) -> None:
    recorder = RecordingJournal()
    monkeypatch.setattr(decision_log, "_default_journal", recorder)
    engine = _engine(lambda request: httpx2.Response(400, json={"error": {"message": "boom"}}))
    with pytest.raises(JudgeError):
        await ask(engine, {"goal": "g"}, {"q1": Noul(instructions="i")}, phase="verify")
    row = recorder.decisions[0]
    assert row["answers"] == {}
    assert row["error"]
    assert row["phase"] == "verify"


async def test_call_id_is_generated_once_per_ask_when_not_supplied(monkeypatch) -> None:
    recorder = RecordingJournal()
    monkeypatch.setattr(decision_log, "_default_journal", recorder)
    engine = _engine(lambda request: _ok_response())
    first, _ = await ask(engine, {"g": 1}, {"q1": Noul(instructions="i")})
    second, _ = await ask(engine, {"g": 2}, {"q1": Noul(instructions="i")})
    assert first != second
    row_ids = [row["call_id"] for row in recorder.decisions]
    assert row_ids == [first, second]  # the journaled rows carry the ids ask() returned


async def test_primary_ask_then_device_verdict_labels_the_pooled_unit(tmp_path: Path, monkeypatch) -> None:
    """End to end: a compiled-question primary ask carries a QuestionRef
    (jev.py's `_refs_for`), and a device-verified outcome on its answer key
    (journal/outcomes.record_action, the "tests = pick key" verdict) joins
    into the live LabelIndex."""
    from jevdevice import question_sets
    from jevdevice.journal import outcomes

    journal = Journal(root=tmp_path, background_writes=False)
    monkeypatch.setattr(decision_log, "_default_journal", journal)
    engine = _engine(lambda request: _ok_response())
    question = question_sets.load().noul("tap.safe")  # ".safe" suffix -> the pooled "gate" unit
    call_id, _ = await ask(engine, {"goal": "g"}, {"q1": question}, phase="gate")

    outcomes.record_action(call_id=call_id, key="q1", response={"status": "ok"})

    assert journal.labeled_pairs("gate", "noul_p", engine="jev", any_revision=True) == [(0.9, True)]


async def test_dispatch_run_kind_labels_the_pick_not_the_kind_or_gate(tmp_path: Path, monkeypatch) -> None:
    """Production-path guard: dispatch.run_kind's final
    record_action must key a verified device outcome by the pick's OWN
    answer key/call_id (execution/dispatch.py's PICK_KEY_FOR + pick_call_id_of),
    never the kind name and never the gate's call_id -- a proposal here
    carries a gate call_id that is deliberately a DIFFERENT, unlabeled row."""
    from types import SimpleNamespace

    from typesymbolic.gate import GateVerdict

    from jevdevice import question_sets
    from jevdevice.execution import dispatch
    from jevdevice.judge.gate import CommandVariant, GateResult

    journal = Journal(root=tmp_path, background_writes=False)
    monkeypatch.setattr(decision_log, "_default_journal", journal)

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={
            "model": "jev-1.13.0", "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {"pick": {"type": "choice", "choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9}}},
        })

    engine = _engine(handler)
    question = question_sets.load().choice("swipe.pick", criteria={"a": None})
    pick_call_id, _ = await ask(engine, {"goal": "g"}, {"pick": question}, phase="fill")

    proposal = SimpleNamespace(
        ready=CommandVariant(command="input swipe 1 2 3 4 300", rationale="r"), pending=None,
        reasons=(), confidence=0.9, pick_call_id=pick_call_id,
        gate_result=GateResult(GateVerdict.ACT, "read_only", 1.0, "cid-gate-unrelated"),
    )
    async def _propose(jev, device, goal):
        return proposal

    monkeypatch.setitem(dispatch.KIND_TABLE, "swipe", dispatch.KindHandler(
        propose=_propose,
        execute=lambda jev, device, goal, proposal, command, **kw: _exit_zero(),
        resume_arg=lambda proposal: None,
    ))

    await dispatch.run_kind(engine, object(), "swipe", "go back")

    assert journal.labeled_pairs("swipe.pick", "confidence", engine="jev", any_revision=True) == [(0.9, True)]
    assert journal.labeled_pairs("gate", "noul_p", engine="jev", any_revision=True) == []


async def _exit_zero() -> int:
    return 0


# --- every ask() call site carries a phase label -----------------------------

def test_every_ask_call_site_carries_a_phase_label() -> None:
    """Static guard over src/jevdevice/*.py (calibrate/ CLIs excluded): any
    ask( call site added later must pass phase=... or this fails, so one
    day of usage can't accumulate an unlabeled row unnoticed."""
    package_dir = Path(__file__).resolve().parent.parent / "src" / "jevdevice"
    offenders: list[str] = []
    for path in sorted(package_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"await ask\(", text):
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


# --- outcome/verdict-row helpers, migrated onto core record_outcome/record_verdict --

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
    from jevdevice.journal.outcomes import verdict_from_response
    assert verdict_from_response({"status": "ok"}).status == "verified"
    assert verdict_from_response({"status": "escalated"}).status == "escalated"
    assert verdict_from_response({"status": "unverified", "exit_code": 1}).status == "failed"
    assert verdict_from_response({"status": "unverified", "exit_code": 0}).status == "unconfirmed"
    assert verdict_from_response({"status": "unverified"}).status == "unconfirmed"


def test_emit_outcome_writes_a_row_joined_to_the_goal_scope(monkeypatch) -> None:
    from jevdevice import mcp_server

    recorder = RecordingJournal()
    monkeypatch.setattr(decision_log, "_default_journal", recorder)
    with goal_scope("press 7"):
        response = {"status": "ok", "satisfied": 0.97}
        mcp_server._emit_outcome(call_id="cid-gate", executed_command="input tap 5 700",
                                 response=response, decision="approve")
    row = recorder.outcomes[0]
    assert recorder.verdicts[0]["verdict"].status == "verified"
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
