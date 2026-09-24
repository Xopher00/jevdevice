"""Shadow mode: while the primary engine answers live traffic, a second
engine observes the exact same (state, questions) and journals its answers --
and nothing else. Ported off the removed `JevClient` onto the current
`jev.ask()` + `judge/shadow.py` wrapper: a primary `JevEngine` (over
`httpx2.MockTransport`, as test_journal_behaviour.py does) with a `.shadow`/
`.shadow_tasks` pair hung on it by plain attribute assignment (attach()'s own
contract), and `jev.ask()` itself schedules/drains -- no client class remains.
Both the primary and its shadow now write through the same
`decision_log.get_journal()` singleton (there is no more per-client journal),
so every test monkeypatches it once and reads both rows off it.

Old -> new mapping: `JevClient("test-key", journal=...)` -> `JevEngine(api_key=
"test-key", transport=httpx2.MockTransport(handler))`, journal monkeypatched
onto `decision_log._default_journal` (like every other ported ask() test);
`client.ask(state, questions, phase=...)` -> `await ask(engine, state,
questions, phase=...)`; `client.shadow`/`client.shadow_tasks` -> unchanged
attribute names, set directly instead of via a JevClient constructor;
`client.aclose()` -> `shadow_mode.drain(engine)`; the old flat `row["shadow_of"]`
-> `row["scope"]["shadow_of"]` (shadow_of now rides `scope`, per jev.ask()).
Every original assertion is kept."""

from __future__ import annotations

import asyncio

import httpx2
import pytest
from typesymbolic.journal import Journal
from typesymbolic.judge import JevEngine

from jevdevice.jev import Noul, ask
from jevdevice.journal import decision_log
from jevdevice.judge import shadow as shadow_mode
from jevdevice.laya_backend import MODEL, LayaClient


class RecordingJournal:
    def __init__(self) -> None:
        self.decisions: list[dict] = []
        self.outcomes: list[dict] = []
        self.verdicts: list[dict] = []

    def record_decision(self, **row) -> None:
        self.decisions.append(row)

    def record_outcome(self, **row) -> None:
        self.outcomes.append(row)

    def record_verdict(self, **row) -> None:
        self.verdicts.append(row)


class FakeRouter:
    """Records the moment a shadow ask actually starts (deterministic
    ordering proof for after-vs-concurrent scheduling)."""

    def __init__(self, payload: dict | None = None, error: Exception | None = None, events: list | None = None) -> None:
        self.payload = payload or {
            "answers": {"q1": {"type": "noul", "noul": 0.42, "confidence": 0.42}},
            "usage": {"input_tokens": 7, "output_tokens": 0},
            "routing": {"model": MODEL, "repo": "convaiinnovations/laya", "reason": "explicit"},
        }
        self.error = error
        self.calls: list[dict] = []
        self.events = events if events is not None else []

    def predict(self, state, questions, model=None, **_kw) -> dict:
        self.calls.append({"state": state, "questions": questions, "model": model})
        self.events.append("shadow-start")
        if self.error is not None:
            raise self.error
        return self.payload


_PRIMARY_PAYLOAD = {
    "model": "jev-1.13.0",
    "answers": {"q1": {"type": "noul", "noul": 0.9}},
    "usage": {"input_tokens": 10, "output_tokens": 2},
}


def _primary(events: list | None = None, delay: float = 0.0) -> JevEngine:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if events is not None:
            events.append("primary-post")
        return httpx2.Response(200, json=_PRIMARY_PAYLOAD)

    async def delayed_handler(request: httpx2.Request) -> httpx2.Response:
        if delay:
            await asyncio.sleep(delay)
        return handler(request)

    transport = httpx2.MockTransport(delayed_handler if delay else handler)
    return JevEngine(api_key="test-key", transport=transport)


def _shadowed(primary: JevEngine, router: FakeRouter) -> None:
    primary.shadow = LayaClient(router=router)
    primary.shadow_tasks = set()


def _drain(engine) -> None:
    """Let scheduled shadow tasks run to completion inside the test."""
    return asyncio.gather(*engine.shadow_tasks)


# --- attach -------------------------------------------------------------------

async def test_attach_gives_a_jev_primary_a_laya_shadow(monkeypatch) -> None:
    monkeypatch.delenv(shadow_mode.SHADOW_ENV, raising=False)
    engine = _primary()
    shadow_mode.attach(engine)
    assert engine.shadow is not None
    assert engine.shadow.name == "laya"
    first = engine.shadow
    shadow_mode.attach(engine)  # idempotent
    assert engine.shadow is first


async def test_attach_skips_laya_primaries_and_the_off_switch(monkeypatch) -> None:
    monkeypatch.delenv(shadow_mode.SHADOW_ENV, raising=False)

    class _LayaPrimary:
        name = "laya"
        shadow = None

    laya_primary = _LayaPrimary()
    shadow_mode.attach(laya_primary)
    assert laya_primary.shadow is None  # the shadow observes JEV traffic only

    monkeypatch.setenv(shadow_mode.SHADOW_ENV, "0")
    engine = _primary()
    shadow_mode.attach(engine)
    assert getattr(engine, "shadow", None) is None


def test_unknown_shadow_mode_fails_fast(monkeypatch) -> None:
    monkeypatch.setenv(shadow_mode.SHADOW_MODE_ENV, "side_by_side")
    with pytest.raises(SystemExit):
        shadow_mode.mode()


# --- scheduling + linkage -----------------------------------------------------

async def test_after_mode_shadows_once_the_answer_and_row_are_final(monkeypatch) -> None:
    monkeypatch.delenv(shadow_mode.SHADOW_MODE_ENV, raising=False)
    events: list[str] = []
    journal = RecordingJournal()
    monkeypatch.setattr(decision_log, "_default_journal", journal)
    engine = _primary(events=events)
    _shadowed(engine, FakeRouter(events=events))

    call_id, answers = await ask(engine, {"goal": "press 7"}, {"q1": Noul(instructions="i")}, phase="verify")
    assert answers["q1"].noul == 0.9  # the primary answer, untouched
    assert events == ["primary-post"]  # nothing shadow-side ran during the ask
    await _drain(engine)
    assert events == ["primary-post", "shadow-start"]

    primary, shadowed = journal.decisions
    assert not (primary.get("scope") or {}).get("shadow_of") and primary["engine"] == "jev"
    assert shadowed["scope"]["shadow_of"] == call_id  # the linkage
    assert shadowed["engine"] == "laya" and shadowed["phase"] == "verify"
    assert isinstance(primary["elapsed_ms"], float) and isinstance(shadowed["elapsed_ms"], float)


async def test_concurrent_mode_starts_the_shadow_before_the_answer_lands(monkeypatch) -> None:
    events: list[str] = []
    journal = RecordingJournal()
    monkeypatch.setattr(decision_log, "_default_journal", journal)
    engine = _primary(events=events, delay=0.02)  # primary yields first
    _shadowed(engine, FakeRouter(events=events))

    monkeypatch.setenv(shadow_mode.SHADOW_MODE_ENV, "concurrent")
    call_id, _ = await ask(engine, {"goal": "g"}, {"q1": Noul(instructions="i")}, phase="recall")
    await _drain(engine)
    assert events == ["shadow-start", "primary-post"]  # the shadow beat the answer
    shadowed = next(r for r in journal.decisions if (r.get("scope") or {}).get("shadow_of"))
    assert shadowed["scope"]["shadow_of"] == call_id


async def test_shadow_failure_never_breaks_the_primary(monkeypatch) -> None:
    monkeypatch.delenv(shadow_mode.SHADOW_MODE_ENV, raising=False)
    journal = RecordingJournal()
    monkeypatch.setattr(decision_log, "_default_journal", journal)
    engine = _primary()
    _shadowed(engine, FakeRouter(error=RuntimeError("laya blew up")))

    call_id, answers = await ask(engine, {"goal": "g"}, {"q1": Noul(instructions="i")}, phase="gate")
    await _drain(engine)
    assert answers["q1"].noul == 0.9  # the primary path is whole
    shadowed = journal.decisions[1]
    assert shadowed["answers"] == {} and "laya blew up" in shadowed["error"]
    assert shadowed["scope"]["shadow_of"] == call_id  # even the failure pairs up


async def test_shadow_answers_go_nowhere(monkeypatch) -> None:
    journal = RecordingJournal()
    monkeypatch.setattr(decision_log, "_default_journal", journal)
    engine = _primary()
    _shadowed(engine, FakeRouter())

    _, answers = await ask(engine, {"goal": "g"}, {"q1": Noul(instructions="i")}, phase="recall")
    await _drain(engine)
    # The returned answers are the PRIMARY's, byte for byte; the shadow's
    # (0.42) exist only in its own journal row, for the report to read.
    assert answers["q1"].noul == 0.9
    assert journal.decisions[1]["answers"]["q1"].noul == 0.42
    assert len(journal.decisions) == 2  # primary + shadow, nothing more


async def test_drain_waits_for_pending_shadow_tasks(monkeypatch) -> None:
    monkeypatch.delenv(shadow_mode.SHADOW_MODE_ENV, raising=False)
    journal = RecordingJournal()
    monkeypatch.setattr(decision_log, "_default_journal", journal)
    engine = _primary()
    _shadowed(engine, FakeRouter())

    await ask(engine, {"goal": "g"}, {"q1": Noul(instructions="i")}, phase="recall")
    assert engine.shadow_tasks  # scheduled, not yet run
    assert len(journal.decisions) == 1  # only the primary row so far
    await shadow_mode.drain(engine)
    assert len(journal.decisions) == 2
    assert journal.decisions[1]["scope"]["shadow_of"]  # drained, not dropped
    assert not [t for t in engine.shadow_tasks if not t.done()]


# --- drift guard: a shadow row can never become a calibration label ---------

async def test_shadow_decision_adds_no_label_pair_but_the_primarys_does(tmp_path, monkeypatch) -> None:
    """Replaces the deleted select_window-era test_p45_window_excludes_shadow_rows:
    that mechanism is gone with continuous.py, but the invariant -- a shadow
    row can never contribute a calibration label -- still needs a guard,
    proved here against the live LabelIndex instead. A verdict on the
    shadow's own call_id must add no (value, verified) pair for its unit,
    while the same verdict shape on the primary's call_id does -- because
    only the primary ask() carries a QuestionRef (jev.py's `_refs_for`,
    `shadow_of is None`)."""
    from typesymbolic.domain import Verdict

    from jevdevice import question_sets

    monkeypatch.delenv(shadow_mode.SHADOW_MODE_ENV, raising=False)
    journal = Journal(root=tmp_path, background_writes=False)
    monkeypatch.setattr(decision_log, "_default_journal", journal)
    engine = _primary()
    _shadowed(engine, FakeRouter())

    question = question_sets.load().noul("recall.any")  # answer type matches the fake payloads
    primary_call_id, _ = await ask(engine, {"goal": "g"}, {"q1": question}, phase="fill")
    await _drain(engine)

    rows = [r for r in journal.replay() if r.get("type") == "decision"]
    shadow_call_id = next(r["call_id"] for r in rows if r["call_id"] != primary_call_id)

    journal.record_verdict(call_id=shadow_call_id, verdict=Verdict(status="verified", tests=("q1",)))
    assert journal.labeled_pairs("recall.any", "noul_p", engine="laya", any_revision=True) == []

    journal.record_verdict(call_id=primary_call_id, verdict=Verdict(status="verified", tests=("q1",)))
    assert len(journal.labeled_pairs("recall.any", "noul_p", engine="jev", any_revision=True)) == 1
